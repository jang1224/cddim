

##############################################################
# eval_run.py
# - 학습 때 만든 DiffusionModel을 import 해서
#   완전 노이즈에서 샘플링 → GT와 PSNR/SSIM 평가를 바로 수행
# - CLI 인자 필요 없음. 파일만 실행하면 동작.

import os, csv, json
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras

# ✅ 네 학습 코드 파일명에 맞게 수정하세요.
#    예: train_diffusion.py 안에 DiffusionModel 클래스가 있다고 가정
import only_dcp_ver5 as od
from only_dcp_ver5 import DiffusionModel  # ← 파일/클래스명 맞게 변경

# -------------------- 사용자 설정(여기만 바꾸면 됨) --------------------
CONFIG = {
    # 테스트셋 경로
    # "hazy_dir": "/home/jang/DDIM_python/RESIDE-b/test_imgs_8/hazy",
    # "ref_dir":  "/home/jang/DDIM_python/RESIDE-b/test_imgs_8/clear",
    "hazy_dir": "/home/jang/DDIM_python/RESIDE-b/test/hazy",
    "ref_dir":  "/home/jang/DDIM_python/RESIDE-b/test/clear",

    # 학습된 가중치 경로(EMA 권장: best.ema.weights.h5)
    "weights":  "./checkpoints/best.online.weights.h5",

    # 출력 (이미지 저장 폴더는 선택)
    "out_csv":  "./eval/eval_results.csv",
    "out_dir":  "./eval/preds",     # None 이면 이미지 저장 안 함

    # 모델/샘플링 설정
    "image_size": 128,
    "diffusion_steps": 50,          # 20~100 권장
    "batch_size": 8,
    "adapt_batches": 8,            # normalizer.adapt에 사용할 배치 수(속도용)
    # 네트워크 구조(학습과 동일해야 함)
    "widths": [64, 128, 256, 256],
    "block_depth": 2,
}

hazy_test_dir = '/home/jang/DDIM_python/RESIDE-b/test/hazy'
ref_test_dir =  '/home/jang/DDIM_python/RESIDE-b/test/clear'
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
#     hazy_files = sorted(str(p) for p in Path(hazy_dir).rglob("*") if Path(p).suffix.lower() in exts)
#     ref_files  = sorted(str(p) for p in Path(ref_dir).rglob("*")  if Path(p).suffix.lower() in exts)
#     assert len(hazy_files) == len(ref_files), f"파일 수 불일치: hazy({len(hazy_files)}), GT({len(ref_files)})"
#     hazy_files = tf.constant(hazy_files, dtype=tf.string)
#     ref_files  = tf.constant(ref_files, dtype=tf.string)
    hazy_files, ref_files = od.build_pairs(hazy_test_dir, ref_test_dir, expected_per_group=35)
    ds = tf.data.Dataset.from_tensor_slices((hazy_files, ref_files))
    ds = ds.map(lambda h, r: _load_pair(h, r, image_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, hazy_files, ref_files


def run_eval():
    cfg = CONFIG.copy()
    _gpu_memory_growth()

    # 출력 경로 준비
    if cfg["out_dir"]:
        os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["out_csv"]) or ".", exist_ok=True)

    # 데이터셋
    ds_adapt, _, _ = _make_dataset(cfg["hazy_dir"], cfg["ref_dir"], cfg["image_size"], cfg["batch_size"])
    ds_eval,  hazy_list, ref_list = _make_dataset(cfg["hazy_dir"], cfg["ref_dir"], cfg["image_size"], cfg["batch_size"])
    # tf.Tensor(dtype=string) -> Python list[str] 변환 (Path()에 EagerTensor 전달되어 발생한 TypeError 방지)   
    # hazy_list = [p.decode("utf-8") for p in hazy_list.numpy()]   
    # ref_list  = [p.decode("utf-8") for p in ref_list.numpy()]
    print("cardinality:", tf.data.experimental.cardinality(ds_eval).numpy())
    sample = next(iter(ds_eval.take(1)), None)
    print("take(1) ->", None if sample is None else (sample[0].shape, sample[1].shape))
    
    
    # 모델 로드(학습과 동일 구조 써야 함)
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
        print(f"[LOAD] EMA -> ema_network: {weights_path}")
    elif weights_path.endswith(".online.weights.h5") or "diffusion_model.weights.h5" in weights_path:
        model.load_weights(weights_path)
        USE_EMA = False
        print(f"[LOAD] ONLINE -> parent model: {weights_path}")
    else:
        try:
            model.load_weights(weights_path)
            USE_EMA = False
            print(f"[LOAD] -> parent model: {weights_path}")
        except Exception as e:
            print("[LOAD] parent failed, trying ema_network:", e)
            model.ema_network.load_weights(weights_path)
            USE_EMA = True
            print(f"[LOAD] -> ema_network: {weights_path}")

    model.hazy_norm.adapt(ds_adapt.map(lambda x, y: x).take(cfg["adapt_batches"]))
    if not hasattr(model, "clear_norm") or not isinstance(model.clear_norm, tf.keras.layers.Normalization):
        model.clear_norm = keras.layers.Normalization(axis=-1, dtype="float32")
    model.clear_norm.adapt(ds_adapt.map(lambda x, y: y).take(cfg["adapt_batches"]))
    #  EMA 덮어쓰기 방지: ONLINE일 때만 복사
    if not USE_EMA:
        model.ema_network.set_weights(model.network.get_weights())
        print("[INFO] Copied ONLINE -> EMA for inference")
        

    # 평가 루프
    results = []
    psnr_all, ssim_all = [], []
    idx = 0

    for hazy_b, gt_b in ds_eval:
        print("Pred start check:", hazy_b.shape, tf.reduce_min(hazy_b).numpy(), tf.reduce_max(hazy_b).numpy())
        b = int(hazy_b.shape[0])
        # 완전 노이즈에서 시작해 샘플링(조건: hazy_b)
        preds = model.generate(num_images=b,
                               diffusion_steps=cfg["diffusion_steps"],
                               hazy_img=hazy_b)  # [0,1]
        print("Pred output:", preds.shape, tf.reduce_min(preds).numpy(), tf.reduce_max(preds).numpy())
        psnr = tf.image.psnr(gt_b, preds, max_val=1.0).numpy()
        ssim = tf.image.ssim(gt_b, preds, max_val=1.0).numpy()
        psnr_all.extend(psnr.tolist())
        ssim_all.extend(ssim.tolist())

        # 저장/로깅
        for i in range(b):
            results.append({
                "index": idx,
                "hazy_path": hazy_list[idx],
                "gt_path":   ref_list[idx],
                "psnr": float(psnr[i]),
                "ssim": float(ssim[i]),
            })
            if cfg["out_dir"]:
                out_img = tf.image.convert_image_dtype(preds[i], tf.uint8)
                name = f"{Path(hazy_list[idx]).stem}_pred.png"
                tf.io.write_file(os.path.join(cfg["out_dir"], name), tf.io.encode_png(out_img))
                # GT도 저장 (선택 사항)-------------
                out_img_clear = tf.image.convert_image_dtype(gt_b[i], tf.uint8)
                name_clear = f"{Path(ref_list[idx]).stem}_gt.png"
                tf.io.write_file(os.path.join(cfg["out_dir"], name_clear), tf.io.encode_png(out_img_clear))
            idx += 1

    # CSV 저장
    with open(cfg["out_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index","hazy_path","gt_path","psnr","ssim"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[INFO] Saved per-image metrics → {cfg['out_csv']}")

    # 요약 JSON
    mean_psnr = float(np.mean(psnr_all)) if psnr_all else 0.0
    mean_ssim = float(np.mean(ssim_all)) if ssim_all else 0.0
    summary_path = cfg["out_csv"].replace(".csv", ".json")
    summary = {
        "count": len(psnr_all),
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "diffusion_steps": cfg["diffusion_steps"],
        "weights": cfg["weights"],
        "hazy_dir": cfg["hazy_dir"],
        "ref_dir": cfg["ref_dir"],
        "image_size": cfg["image_size"],
        "batch_size": cfg["batch_size"],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[SUMMARY] PSNR={mean_psnr:.3f}  SSIM={mean_ssim:.4f}")
    print(f"[INFO] Summary JSON → {summary_path}")

if __name__ == "__main__":
    run_eval()
