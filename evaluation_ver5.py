import os, csv, json
from pathlib import Path
import numpy as np
from scipy import linalg # FID 계산용

import tensorflow as tf
import keras

import torch
import lpips # LPIPS 평가용

import only_dcp_ver5 as od
from only_dcp_ver5 import DiffusionModel  

# -------------------- 사용자 설정 --------------------
CONFIG = {
    "hazy_dir": "/home/jang/DDIM_python/RESIDE-b/test_imgs_8/hazy",
    "ref_dir":  "/home/jang/DDIM_python/RESIDE-b/test_imgs_8/clear",
    "weights": "/home/jang/DDIM_python/paper/OnlyDcp/checkpoints/best.online.weights.h5",
    "out_csv":  "./eval/eval_results.csv",
    "out_dir":  "./eval/preds",     
    "image_size": 128,
    "diffusion_steps": 50,          
    "batch_size": 8,
    "adapt_batches": 8,            
    "widths": [64, 128, 256, 256],
    "block_depth": 2,
}
# ----------------------------------------------------------------------
exts = (".png", ".jpg", ".jpeg")

def _gpu_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

def _read_image(path):
    path = tf.cast(path, tf.string)
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return tf.image.convert_image_dtype(img, tf.float32)  # [0,1]

def _load_pair(h_path, r_path, image_size):
    hazy  = tf.image.resize(_read_image(h_path),  [image_size, image_size])
    clear = tf.image.resize(_read_image(r_path), [image_size, image_size])
    return hazy, clear

def _make_dataset(hazy_dir, ref_dir, image_size, batch_size):
    hazy_files = sorted(str(p) for p in Path(hazy_dir).rglob("*") if Path(p).suffix.lower() in exts)
    ref_files  = sorted(str(p) for p in Path(ref_dir).rglob("*")  if Path(p).suffix.lower() in exts)
    assert len(hazy_files) == len(ref_files), f"파일 수 불일치: hazy({len(hazy_files)}), GT({len(ref_files)})"
    
    hazy_files_tf = tf.constant(hazy_files, dtype=tf.string)
    ref_files_tf  = tf.constant(ref_files, dtype=tf.string)
    
    ds = tf.data.Dataset.from_tensor_slices((hazy_files_tf, ref_files_tf))
    ds = ds.map(lambda h, r: _load_pair(h, r, image_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, hazy_files, ref_files

# --- FID 계산용 수학 함수 ---
def calculate_fid(real_features, fake_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


def run_eval():
    cfg = CONFIG.copy()
    _gpu_memory_growth()

    if cfg["out_dir"]:
        os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["out_csv"]) or ".", exist_ok=True)

    ds_adapt, _, _ = _make_dataset(cfg["hazy_dir"], cfg["ref_dir"], cfg["image_size"], cfg["batch_size"])
    ds_eval,  hazy_list, ref_list = _make_dataset(cfg["hazy_dir"], cfg["ref_dir"], cfg["image_size"], cfg["batch_size"])
    
    # ------------------ 평가 모델 로드 (FID & LPIPS) ------------------
    print("[INFO] Loading Evaluator Models (InceptionV3 for FID, AlexNet for LPIPS)...")
    # 1. FID용 InceptionV3 모델 (TF 기반)
    inception_model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    
    # 2. LPIPS용 모델 (PyTorch 기반)
    # VRAM OOM을 막기 위해 연산은 CPU에서 수행하도록 강제합니다.
    lpips_fn = lpips.LPIPS(net='alex', verbose=False).eval().to('cpu')
    # ------------------------------------------------------------------

    # 모델 로드
    model = DiffusionModel(image_size=cfg["image_size"], widths=cfg["widths"], block_depth=cfg["block_depth"])

    dummy_noisy7 = tf.zeros((1, cfg["image_size"], cfg["image_size"], 7), dtype=tf.float32)
    dummy_t      = tf.zeros((1, 1, 1, 1), dtype=tf.float32)
    _ = model([dummy_noisy7, dummy_t], training=False)
    _ = model.ema_network([dummy_noisy7, dummy_t], training=False)
        
    weights_path = cfg["weights"]
    USE_EMA = False

    if weights_path.endswith(".ema.weights.h5"):
        model.ema_network.load_weights(weights_path)
        USE_EMA = True
    elif weights_path.endswith(".online.weights.h5") or "diffusion_model.weights.h5" in weights_path:
        model.load_weights(weights_path)
    else:
        try:
            model.load_weights(weights_path)
        except Exception:
            model.ema_network.load_weights(weights_path)
            USE_EMA = True

    model.hazy_norm.adapt(ds_adapt.map(lambda x, y: x).take(cfg["adapt_batches"]))
    if not hasattr(model, "clear_norm") or not isinstance(model.clear_norm, tf.keras.layers.Normalization):
        model.clear_norm = keras.layers.Normalization(axis=-1, dtype="float32")
    model.clear_norm.adapt(ds_adapt.map(lambda x, y: y).take(cfg["adapt_batches"]))
    
    if not USE_EMA:
        model.ema_network.set_weights(model.network.get_weights())
        print("[INFO] Copied ONLINE -> EMA for inference")

    # 평가 루프 변수 준비
    results = []
    psnr_all, ssim_all, lpips_all = [], [], []
    real_features_all, fake_features_all = [], [] # FID용
    idx = 0

    print("[INFO] Starting Evaluation Loop...")
    for hazy_b, gt_b in ds_eval:
        b = int(hazy_b.shape[0])
        
        # 이미지 생성 (0~1 float32)
        preds = model.generate(num_images=b,
                               diffusion_steps=cfg["diffusion_steps"],
                               hazy_img=hazy_b)
        
        # 1. PSNR & SSIM 계산
        psnr = tf.image.psnr(gt_b, preds, max_val=1.0).numpy()
        ssim = tf.image.ssim(gt_b, preds, max_val=1.0).numpy()
        psnr_all.extend(psnr.tolist())
        ssim_all.extend(ssim.tolist())

        # 2. LPIPS 계산 (PyTorch 변환 및 CPU 연산)
        # lpips 패키지는 (B, C, H, W) 형태와 [-1, 1] 범위를 요구합니다.
        gt_pt = torch.tensor(gt_b.numpy(), dtype=torch.float32).permute(0, 3, 1, 2) * 2.0 - 1.0
        preds_pt = torch.tensor(preds.numpy(), dtype=torch.float32).permute(0, 3, 1, 2) * 2.0 - 1.0
        
        with torch.no_grad():
            # CPU에서 계산하여 TF와의 VRAM 충돌 방지
            lpips_vals = lpips_fn(gt_pt.to('cpu'), preds_pt.to('cpu')).squeeze().numpy()
        
        # 배치가 1일 경우 스칼라가 되므로 리스트로 변환
        if lpips_vals.ndim == 0:
            lpips_vals = [float(lpips_vals)]
        else:
            lpips_vals = lpips_vals.tolist()
            
        lpips_all.extend(lpips_vals)

        # 3. FID를 위한 Feature 추출 (InceptionV3 연산)
        # InceptionV3는 299x299 크기와 [-1, 1] 범위를 요구합니다.
        gt_299 = tf.image.resize(gt_b, (299, 299))
        preds_299 = tf.image.resize(preds, (299, 299))
        
        gt_inc = (gt_299 * 2.0) - 1.0
        preds_inc = (preds_299 * 2.0) - 1.0
        
        real_feat = inception_model(gt_inc, training=False)
        fake_feat = inception_model(preds_inc, training=False)
        
        real_features_all.append(real_feat.numpy())
        fake_features_all.append(fake_feat.numpy())

        # 파일 저장 및 로깅
        for i in range(b):
            results.append({
                "index": idx,
                "hazy_path": hazy_list[idx],
                "gt_path":   ref_list[idx],
                "psnr": float(psnr[i]),
                "ssim": float(ssim[i]),
                "lpips": float(lpips_vals[i]), # LPIPS 추가
            })
            if cfg["out_dir"]:
                out_img = tf.image.convert_image_dtype(preds[i], tf.uint8)
                name = f"{Path(hazy_list[idx]).stem}_pred.png"
                tf.io.write_file(os.path.join(cfg["out_dir"], name), tf.io.encode_png(out_img))
                
                out_img_clear = tf.image.convert_image_dtype(gt_b[i], tf.uint8)
                name_clear = f"{Path(ref_list[idx]).stem}_gt.png"
                tf.io.write_file(os.path.join(cfg["out_dir"], name_clear), tf.io.encode_png(out_img_clear))
            idx += 1
            
        print(f"  -> Processed {idx} images...")

    # --- 전체 데이터셋에 대한 FID 계산 ---
    print("[INFO] Calculating FID over all generated images...")
    real_features_concat = np.concatenate(real_features_all, axis=0)
    fake_features_concat = np.concatenate(fake_features_all, axis=0)
    fid_value = calculate_fid(real_features_concat, fake_features_concat)

    # CSV 저장
    with open(cfg["out_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index","hazy_path","gt_path","psnr","ssim", "lpips"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[INFO] Saved per-image metrics → {cfg['out_csv']}")

    # 요약 JSON
    mean_psnr = float(np.mean(psnr_all)) if psnr_all else 0.0
    mean_ssim = float(np.mean(ssim_all)) if ssim_all else 0.0
    mean_lpips = float(np.mean(lpips_all)) if lpips_all else 0.0
    
    summary_path = cfg["out_csv"].replace(".csv", ".json")
    summary = {
        "count": len(psnr_all),
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "mean_lpips": mean_lpips,   # 추가
        "fid": fid_value,           # 추가
        "diffusion_steps": cfg["diffusion_steps"],
        "weights": cfg["weights"],
        "hazy_dir": cfg["hazy_dir"],
        "ref_dir": cfg["ref_dir"],
        "image_size": cfg["image_size"],
        "batch_size": cfg["batch_size"],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        
    print("="*50)
    print(f"[SUMMARY] PSNR: {mean_psnr:.3f} | SSIM: {mean_ssim:.4f} | LPIPS: {mean_lpips:.4f} | FID: {fid_value:.2f}")
    print("="*50)
    print(f"[INFO] Summary JSON → {summary_path}")

if __name__ == "__main__":
    run_eval()