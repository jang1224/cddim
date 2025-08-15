
import huggingface_hub
huggingface_hub.login("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")

# ─── 2) 라이브러리 임포트 ───
import os, math
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers

from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipTextModel, BartConfig, TFBartForConditionalGeneration
from transformers.models.bart.modeling_tf_bart import TFBartAttention
from PIL import Image
import torch
import json
from datasets import load_dataset
# #context임베딩할 때 필요
from tqdm import tqdm
import collections

import gc # gc.collect()를 사용하기 위해 import -> 가비지 컬렉션을 위해 스크립트 상단에 추가

np.random.seed(None)
tf.random.set_seed(None)

gpus = tf.config.list_physical_devices('GPU')
print("GPU available:", gpus)
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True) #-> gpu사용량 줄이려고


# ─── 3) 하이퍼파라미터 ───
img_siz             = 128
batch_siz           = 8
gradient_accumulation_steps = 4
effective_batch_siz = batch_siz * gradient_accumulation_steps
kid_diffusion_steps = 100    # ← must be before class definition
min_signal_rate     = 1e-4 #0.02
max_signal_rate     = 0.95 #0.95
zdim                = 128
embed_max_freq      = 1000.0
widths              = [64, 128, 256, 256]#[160,320,768,768] # 768인 이유는 bottleneck 채널수와 crossattention(BART)의 d_model = 768로 같아야 가중치 로드가 에러 없이 됨.
block_depth         = 2
ctx_dim             = 256
seq_len             = 77    # tokenizer max length
num_epochs          = 100    # 학습 에폭


# 모델 준비
processor_blip = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_blip     = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to("cuda").eval()  #.to("cuda") -> gpu사용량 줄이려고

model_blip.trainable = False  #  꼭 이 줄 필요함!-> tf.keras.model을 쓰는데 blip의 tf-keras가 간섭을 일으킴. 그래서 학습 제거.

#텍스트를 임베딩하는 모델
txt_processor_blip = processor_blip.tokenizer  # BLIPProcessor 안에 tokenizer
txt_model_blip     = BlipTextModel.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                ).to("cuda").eval()  #.to("cuda") -> gpu사용량 줄이려고


# ——— 1) 데이터 경로 설정 ———
base_dir = '/mnt/c/Users/조장혁/Downloads/RESIDE-b/train/'
hazy_dir = os.path.join(base_dir, 'hazy')
ref_dir  = os.path.join(base_dir, 'clear')

base_dir_test = '/mnt/c/Users/조장혁/Downloads/RESIDE-b/test/'
hazy_test_dir = os.path.join(base_dir_test, 'hazy')
ref_test_dir  = os.path.join(base_dir_test, 'clear')

exts     = ('.png', '.jpg', '.jpeg')

def list_images(root):
    return [p for p in Path(root).rglob('*') if p.suffix.lower() in exts]


def check_duplicate_stems(paths, tag):
    stems = [p.stem for p in paths]
    counts = collections.Counter(stems)
    dups = {k:v for k,v in counts.items() if v > 1}
    if dups:
        print(f" {tag}: 중복된 파일명 {len(dups)}개!  (예시: {list(dups.keys())[:5]})")
        print("    같은 stem을 쓰면 .npy가 덮어쓰기 됩니다. 폴더 구조를 바꾸거나 stem 규칙을 바꾸세요.")
    else:
        print(f" {tag}: 파일명 중복 없음.")
        
        
def encode_context(text_str: str) -> np.ndarray:
    """BLIP 텍스트 인코더로 (seq_len, 768) 임베딩 생성"""
    inp = txt_processor_blip(
        [text_str],
        padding="max_length", truncation=True, max_length=seq_len,
        return_tensors="pt"
    ).to("cuda")  # txt_model_blip이 CUDA에 있으니 동일 장치
    with torch.no_grad():
        out = txt_model_blip(**inp)
        emb = out.last_hidden_state[0]  # (seq_len, 768)
    return emb.cpu().numpy().astype(np.float32)


def make_context(split: str):
    """
    split: 'train' 또는 'test'
    - train  → hazy_dir        / contexts_blip_embedding_
    - test   → hazy_test_dir   / contexts_blip_embedding_test
    JSON과 .npy를 모두 생성/저장.
    """
    assert split in ("train", "test")
    if split == "train":
        hazy_root = hazy_dir
        out_json  = "/home/jang/DDIM_python/paper/reside-b_contexts_blip_train.json"
        emb_dir   = "/home/jang/DDIM_python/paper/contexts_blip_embedding_RESIDE-b"
    else:
        hazy_root = hazy_test_dir
        out_json  = "/home/jang/DDIM_python/paper/reside-b_contexts_blip_test.json"
        emb_dir   = "/home/jang/DDIM_python/paper/contexts_blip_embedding_test_RESIDE-b"

    os.makedirs(emb_dir, exist_ok=True)

    # 1) 이미지 수집 & 중복 검사
    paths = list_images(hazy_root)
    print(f"[{split}] hazy 이미지 개수: {len(paths)}  (root={hazy_root})")
    check_duplicate_stems(paths, f"{split}/hazy")

    # 2) 캡션 생성 (BLIP) → JSON 저장
    context_dict = {}
    for p in tqdm(paths, desc=f"[{split}] captioning"):
        img = Image.open(p).convert("RGB")
        inputs = processor_blip(img, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model_blip.generate(**inputs, max_length=50)
            cap = processor_blip.decode(out[0], skip_special_tokens=True)
        context_dict[str(p)] = cap

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(context_dict, f, indent=2, ensure_ascii=False)
    print(f" [{split}] 캡션 JSON 저장: {out_json}  | 총 {len(context_dict)}개")

    # 3) 컨텍스트 임베딩 생성 → .npy 저장 (stem.npy)
    for img_path, caption in tqdm(context_dict.items(), desc=f"[{split}] embedding"):
        stem = Path(img_path).stem
        emb = encode_context(caption)  # (seq_len, 768)
        np.save(os.path.join(emb_dir, f"{stem}.npy"), emb)
    print(f" [{split}] 임베딩 npy 저장 완료: {emb_dir}")
# ================== 컨텍스트 생성 유틸 끝 ==================

make_context("test") # "train" or "test" 선택