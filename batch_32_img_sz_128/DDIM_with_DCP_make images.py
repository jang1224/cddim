# pip install --upgrade fsspec datasets huggingface_hub
# pip install transformers accelerate -> blip
# pip install opencv-python-headless -> cv2
# pip install tensorflow==2.12.0
# pip install pillow -> pil
# pip install matplotlib -> matplotlib
# pip uninstall -y numpy
# pip install numpy==1.26.4
# pip install tf-keras
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
from tensorflow.keras.losses import MeanAbsoluteError

from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipTextModel, BartConfig, TFBartForConditionalGeneration
from transformers.models.bart.modeling_tf_bart import TFBartAttention
from PIL import Image
import torch
import json
from datasets import load_dataset
# #context임베딩할 때 필요
# from tqdm import tqdm

import gc # gc.collect()를 사용하기 위해 import -> 가비지 컬렉션을 위해 스크립트 상단에 추가
import glob # 각 에폭마다 생성할때 필요
import re   # 각 에폭마다 생성할때 필요

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
widths              = [64, 128, 256, 256]    #[160,320,768,768] # 768인 이유는 bottleneck 채널수와 crossattention(BART)의 d_model = 768로 같아야 가중치 로드가 에러 없이 됨.
block_depth         = 2
ctx_dim             = 256
seq_len             = 77    # tokenizer max length
num_epochs          = 100    # 학습 에폭


# 모델 준비
processor_blip = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_blip     = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to("cpu").eval()  #.to("cuda") -> gpu사용량 줄이려고

model_blip.trainable = False  #  꼭 이 줄 필요함!-> tf.keras.model을 쓰는데 blip의 tf-keras가 간섭을 일으킴. 그래서 학습 제거.

#텍스트를 임베딩하는 모델
txt_processor_blip = processor_blip.tokenizer  # BLIPProcessor 안에 tokenizer
txt_model_blip     = BlipTextModel.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                ).to("cpu").eval()  #.to("cuda") -> gpu사용량 줄이려고


# ——— 1) 데이터 경로 설정 ———
base_dir = '/mnt/c/Users/조장혁/Downloads/RESIDE-6K/RESIDE-6K/train/'
hazy_dir = os.path.join(base_dir, 'hazy')
ref_dir  = os.path.join(base_dir, 'GT')

base_dir_test = '/mnt/c/Users/조장혁/Downloads/RESIDE-6K/RESIDE-6K/test/'
hazy_test_dir = os.path.join(base_dir_test, 'hazy')
ref_test_dir  = os.path.join(base_dir_test, 'GT')

# hazy 이미지 경로들
image_paths = list(Path(hazy_dir).rglob("*.jpg"))
exts     = ('.png', '.jpg', '.jpeg')

# # 이미지 하나씩 캡셔닝
# for img_path in image_paths:
#     raw_image = Image.open(img_path).convert('RGB')
#     inputs = processor_blip(raw_image, return_tensors="pt").to("cuda")

#     with torch.no_grad():
#         out = model_blip.generate(**inputs, max_length=50)
#         caption = processor_blip.decode(out[0], skip_special_tokens=True)

#     context_dict[str(img_path)] = caption
#     if tf.executing_eagerly():
#       tf.print(f"{img_path.name} → {caption}")

# # JSON 저장
# # Google Drive 등 원하는 경로에 저장
# output_json = "/content/RESIDE-6K/train/reside6k_contexts_blip.json"

# with open(output_json, "w") as f:
#     json.dump(context_dict, f, indent=2)

# if tf.executing_eagerly():
#    tf.print(f"context 저장 완료 : {output_json}")

json_path = "/home/jang/DDIM_python/paper/reside6k_contexts_blip.json"
with open(json_path, "r") as f:
  context_dict = json.load(f)


checkpoint_dir = '/home/jang/DDIM_python/paper/checkpoints_weights'
os.makedirs(checkpoint_dir, exist_ok=True)

exts = {'.jpg', '.jpeg', '.png'}

#train용
hazy_files = sorted(str(p) for p in Path(hazy_dir).rglob('*') if p.suffix.lower() in exts)
ref_files  = sorted(str(p) for p in Path(ref_dir).rglob('*')  if p.suffix.lower() in exts)
contexts   =  [''] * len(hazy_files)  # 리스트 순서 맞춤(학습용)

#test용
hazy_test_files = sorted(str(p) for p in Path(hazy_test_dir).rglob('*') if p.suffix.lower() in exts)
ref_test_files  = sorted(str(p) for p in Path(ref_test_dir).rglob('*')  if p.suffix.lower() in exts)
contexts_test = [''] * len(hazy_test_files) # 빈 캡션 리스트 (테스트용)

assert len(hazy_files)==len(ref_files), "파일 개수 불일치"
print("✔ 총 이미지 쌍:", len(hazy_files))

def encode_context(text_str: str) -> np.ndarray:
    """
    문자열(text_str) → BLIP 텍스트 인코더 last_hidden_state[0]
    → NumPy (seq_len, ctx_dim) float32 배열 반환
    """
    inputs = txt_processor_blip(
        [text_str],
        padding="max_length", truncation=True, max_length=seq_len,
        return_tensors="pt"
    )   #.to("cuda") -> gpu사용량 줄이려고

    with torch.no_grad():
        outputs = txt_model_blip(**inputs)
        emb = outputs.last_hidden_state[0]  # (seq_len, ctx_dim)

    return emb.cpu().numpy().astype(np.float32)


# # context 임베딩된거 저장------------------
# context_embedding_dir = '/home/jang/DDIM_python/paper/contexts_blip_embedding'
# os.makedirs(context_embedding_dir, exist_ok=True)
# captioner = processor_blip.tokenizer

# with open(json_path, "r") as f:
#     caption_dict = json.load(f)
    
# # 저장
# for img_path, caption in tqdm(caption_dict.items()):
#     filename = Path(img_path).stem
#     context_np = encode_context(caption)
#     np.save(os.path.join(context_embedding_dir, f"{filename}.npy"), context_np)
# # 여기까지---------------------
    
context_embedding_dir = '/home/jang/DDIM_python/paper/contexts_blip_embedding'

context_embedding_dir_test = '/home/jang/DDIM_python/paper/contexts_blip_embedding_test'
    
def _read_image(path):  
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return tf.image.convert_image_dtype(img, tf.float32)    # 여기서 /255.0 해줌


def _load_context(path_bytes):
    return np.load(path_bytes.decode("utf-8")).astype(np.float32)

def load_pair_train(h_path, r_path, _):  # dummy 세 번째 인자
    hazy  = tf.image.resize(_read_image(h_path),  [img_siz, img_siz])
    clear = tf.image.resize(_read_image(r_path), [img_siz, img_siz])

    stem = tf.strings.regex_replace(tf.strings.split(h_path, os.sep)[-1], r"\.(jpg|jpeg|png)$", "")
    ctx_path = tf.strings.join([context_embedding_dir, "/", stem, ".npy"])


    ctx = tf.numpy_function(_load_context, [ctx_path], tf.float32)  # 파일에서 직접 읽은 데이터의 실제 크기는 (77,768)
    ctx.set_shape([seq_len, 768])
    return hazy, clear, ctx


def load_pair_test(h_path, r_path, _):
    hazy  = tf.image.resize(_read_image(h_path),  [img_siz, img_siz])
    clear = tf.image.resize(_read_image(r_path), [img_siz, img_siz])

    # train과 동일하게 .npy 파일 경로를 만듬.
    stem = tf.strings.regex_replace(tf.strings.split(h_path, os.sep)[-1], r"\.(jpg|jpeg|png)$", "")
    ctx_path = tf.strings.join([context_embedding_dir_test, "/", stem, ".npy"])

    # def _generate_context(h_path_bytes):
    #     path = h_path_bytes.numpy().decode("utf-8")
    #     img  = Image.open(path).convert("RGB")
    #     inp  = processor_blip(img, return_tensors="pt") #.to("cuda") -> gpu 사용량 줄이려고
    #     out  = model_blip.generate(**inp, max_length=50)
    #     cap  = processor_blip.decode(out[0], skip_special_tokens=True)
    #     return encode_context(cap)

    # train과 동일하게 .npy 파일을 읽어옴.
    ctx = tf.numpy_function(_load_context, [ctx_path], tf.float32)
    ctx.set_shape([seq_len, 768]) # 원본 임베딩 차원
    
    return hazy, clear, ctx

def load_pair_test_ihaze(h_path, r_path, _):
    hazy  = tf.image.resize(_read_image(h_path),  [img_siz, img_siz])
    clear = tf.image.resize(_read_image(r_path), [img_siz, img_siz])

    def _generate_context(h_path_bytes):
        path = h_path_bytes.numpy().decode("utf-8")
        img  = Image.open(path).convert("RGB")
        inp  = processor_blip(img, return_tensors="pt") #.to("cuda") -> gpu 사용량 줄이려고
        out  = model_blip.generate(**inp, max_length=50)
        cap  = processor_blip.decode(out[0], skip_special_tokens=True)
        return encode_context(cap)

    # train과 동일하게 .npy 파일을 읽어옴.
    ctx = tf.py_function(_generate_context, [h_path], tf.float32)
    ctx.set_shape([seq_len, 768]) # 원본 임베딩 차원
    
    return hazy, clear, ctx


# ——— train/test Dataset 정의 ———
ds_train = tf.data.Dataset.from_tensor_slices((hazy_files, ref_files, contexts))
ds_test  = tf.data.Dataset.from_tensor_slices((hazy_test_files, ref_test_files, contexts_test))

total_count = len(hazy_files)
print("total_count", total_count)

train_ds = (
    ds_train
    .take(total_count)
    .map(load_pair_train, num_parallel_calls=tf.data.AUTOTUNE)
    .repeat()
    .shuffle(1024)
    .batch(batch_siz)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    ds_test
    .map(load_pair_test, num_parallel_calls=1)
    .repeat()
    .batch(batch_siz)
    .prefetch(tf.data.AUTOTUNE)
)
######################### ihaze도 보려고 따로 만든 데이터 셋
ihaze_base_dir = '/mnt/c/Users/조장혁/Downloads/RESIDE-6K/RESIDE-6K/test/I-haze'
ihaze_hazy_dir = os.path.join(ihaze_base_dir, 'I-haze_hazy')
ihaze_clear_dir  = os.path.join(ihaze_base_dir, 'I-haze_GT')

#test용
ihaze_hazy_files = sorted(str(p) for p in Path(ihaze_hazy_dir).rglob('*') if p.suffix.lower() in exts)
ihaze_clear_files  = sorted(str(p) for p in Path(ihaze_clear_dir).rglob('*')  if p.suffix.lower() in exts)
ihaze_empty_caps = [''] * len(ihaze_hazy_files) # 빈 캡션 리스트 (테스트용)

ihaze_ds_test  = tf.data.Dataset.from_tensor_slices((ihaze_hazy_files, ihaze_clear_files, ihaze_empty_caps))

ihaze_ds = (
    ihaze_ds_test
    .map(load_pair_test_ihaze, num_parallel_calls=1)
    .repeat()
    .batch(batch_siz)
    .prefetch(tf.data.AUTOTUNE)
)
###################################################

steps_per_epoch = int(tf.math.ceil(len(hazy_files) / batch_siz)) # 6000 / batch_siz
val_steps = kid_diffusion_steps #len(hazy_test_files)//batch_siz


# TensorFlow 버전의 Dark Channel
def tf_dark_channel(img, patch_size=15):
    min_rgb = tf.reduce_min(img, axis=-1, keepdims=True)  # (B, H, W, 1)
    return -tf.nn.max_pool2d(-min_rgb, ksize=patch_size, strides=1, padding='SAME')

# Transmission map 추정
def tf_transmission_estimate(img, airlight, patch_size=15, omega=0.95):
    img_norm = img / (airlight + 1e-6)
    dark = tf_dark_channel(img_norm, patch_size)
    return 1 - omega * dark  # (B, H, W, 1)

# 간단한 guided filter (box smoothing)
def tf_box_filter(img, r=15):   # 64x64 이미지에 40x40 필터는 너무 큼
    return tf.nn.avg_pool2d(img, ksize=r, strides=1, padding='SAME')

# 입력 이미지를 grayscale로 변환 후 guided filter 수행 -> 트랜스미션 맵을 더 자연스럽고 세밀하게 정제하기 위해서
def tf_transmission_refine(img, est_trans):
    gray = tf.image.rgb_to_grayscale(img)
    return tf_box_filter(est_trans * gray)

# 최종 마스크 예측 모델
def build_mask_predictor():
    inp = keras.Input((img_siz, img_siz, 3), dtype=tf.float32)

    # Airlight 추정 (평균값 사용: trainable하지 않음)
    airlight = layers.Lambda(lambda x: tf.reduce_max(x, axis=[1, 2], keepdims=True))(inp)  # (B, 1, 1, 3)

    t_est = layers.Lambda(lambda x: tf_transmission_estimate(x[0], x[1]))([inp, airlight])
    t_ref = layers.Lambda(lambda x: tf_transmission_refine(x[0], x[1]))([inp, t_est])

    return keras.Model(inputs=inp, outputs=t_ref, name='mask_predictor')

# ─── 8) Sinus Embedding & U‑Net 블록 ───
@tf.keras.utils.register_keras_serializable(package='ddpm')
# 입력이 (B,1,1,1)일때 (B,1,1,zdim)으로 출력
def sinusoidal_embedding(x):    # 위치 임베딩처럼, 각 스텝 t를 여러 주파수의 사인, 코사인 함수로 변환함.
    freqs = tf.exp(tf.linspace(tf.math.log(1.0),
                               tf.math.log(embed_max_freq),
                               zdim//2))
    ang   = 2*math.pi*freqs
    return tf.concat([tf.sin(ang*x), tf.cos(ang*x)], axis=3)

class ResidualBlock(layers.Layer):  # 입력 : (B, sz, sz, C)
    def __init__(self, w, **kwargs):
        super().__init__(**kwargs)
        self.proj = layers.Conv2D(w, 1)  # 필요 시
        self.norm = layers.BatchNormalization(center=False,scale=False)
        self.conv1 = layers.Conv2D(w, kernel_size = 3, padding='same', kernel_initializer='he_normal',activation='swish')
        self.conv2 = layers.Conv2D(w, kernel_size = 3, padding='same', kernel_initializer='he_normal')
        self.w = w
    def get_config(self):
        cfg = super().get_config(); cfg.update({"w": self.w}); return cfg
        
    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)
    
    def call(self, x, training=None):
        res = x if x.shape[-1] == self.conv2.filters else self.proj(x)  # (B, sz, sz, w[?]) -> x의 채널의 변환시킴
        y = self.norm(x, training=training)
        y = self.conv1(y)
        y = self.conv2(y)
        return res + y   # 출력 : (B, sz, sz, w[?])
    
class DownBlock(layers.Layer):  
    def __init__(self, w, bd, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [ResidualBlock(w) for _ in range(bd)]
        self.pool = layers.AveragePooling2D(2)
        self.w = w
        self.bd = bd
        
    def get_config(self):
        cfg = super().get_config(); cfg.update({"w": self.w, "bd": self.bd}); return cfg
        
    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)
    
    def call(self, x, training=None):  # 입력 : (B, sz, sz, C)
        for block in self.blocks:
            x = block(x, training = training)
        skip = x
        x = self.pool(x)    # for문 끝나고의 출력 : (B, sz / 2, sz / 2, w[2])
        return x, [skip]    
    
    
class UpBlock(layers.Layer):    
    def __init__(self, w, bd, **kwargs):
        super().__init__(**kwargs)
        self.upsample = layers.UpSampling2D(2, interpolation='bilinear')
        self.blocks = [ResidualBlock(w) for _ in range(bd)]
        self.concat = layers.Concatenate()
        self.w = w
        self.bd = bd
        
    def get_config(self):
        cfg = super().get_config(); cfg.update({"w": self.w, "bd": self.bd}); return cfg
        
    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)
    
    def call(self, x, skip, training=None):    # 입력 : (B, sz / 2, sz / 2, C) -> 여기서는 C = w[3] = 768이 들어옴
        x = self.upsample(x)     # (B, sz, sz, w[3])
        x = self.concat([x, skip])  # (B, sz, sz, w[2] + w[2])
        for block in self.blocks:
            x = block(x, training=training)
        return x    
    

# --- 수정 후: get_config 추가 ---
class CrossAttentionBlock(layers.Layer):
    """U-Net 특징맵과 텍스트 컨텍스트 간의 크로스 어텐션을 수행합니다.""" # <- Docstring도 추가 (원인 2 해결)
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels # 나중에 get_config에서 사용하기 위해 저장
        self.mha = layers.MultiHeadAttention(num_heads=8, key_dim=self.channels)
        self.layernorm = layers.LayerNormalization()
        self.add = layers.Add()

    def get_config(self):
        """레이어를 다시 생성하기 위한 설정 정보를 반환합니다."""
        config = super().get_config()
        config.update({
            "channels": self.channels,
        })
        return config

    def call(self, inputs):
        x, context = inputs
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x_flat = tf.reshape(x, [B, H * W, C])
        attn_output = self.mha(query=x_flat, key=context, value=context)
        attn_output = tf.reshape(attn_output, [B, H, W, C])
        x = self.add([x, self.layernorm(attn_output)])
        return x

class ConditionalUNet(tf.keras.Model):
    def __init__(self, sz, ws, bd, ctx_dim, seq_len=77, zdim=64, **kwargs):
        super().__init__(**kwargs)
        self.sz = sz
        self.ws = ws
        self.bd = bd
        self.ctx_dim = ctx_dim
        self.seq_len = seq_len
        self.zdim = zdim

        self.input_conv = layers.Conv2D(ws[0], kernel_size=1)
        self.sin_embed = layers.Lambda(sinusoidal_embedding)
        self.emb_proj   = layers.Conv2D(ws[0], kernel_size=1, bias_initializer="zeros")    # 임베딩을 ws[0](=160)으로 사상. 초기엔 영향 0으로 두고 싶으면 'zeros'로.
        self.e_norm     = layers.LayerNormalization(axis=[1,2,3])
        self.e_scale    = self.add_weight(
                            name="e_scale", shape=(ws[0],), dtype=tf.float32,
                            initializer=tf.keras.initializers.Constant(0.1),
                            trainable=True)
        self.up_embed = layers.UpSampling2D(sz , interpolation='nearest')
        self.out_conv = layers.Conv2D(3, 1, activation=None, kernel_initializer='zeros', bias_initializer="zeros") #kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.05)

        # 768차원 입력을 받아 우리가 사용할 ctx_dim(512)으로 변환합니다.
        self.context_projection = layers.Dense(ctx_dim, name="context_projection")
        
        # Residual 구조를 인스턴스로 담기
        self.down_blocks = [DownBlock(w, bd) for w in ws[:-1]]
        self.mid_blocks = [ResidualBlock(ws[-1]) for _ in range(bd)]
        self.up_blocks = [UpBlock(w, bd) for w in reversed(ws[:-1])]
        self.mid_attn = CrossAttentionBlock(ws[-1])
        
    def get_config(self):
        cfg = super().get_config()  # keras의 기본 속성들(name, trainable, dtype 등)을 포함시키기 위해
        cfg.update({
            "sz": self.sz,
            "ws": self.ws,
            "bd": self.bd,
            "ctx_dim": self.ctx_dim,
            "seq_len": self.seq_len,
            "zdim": self.zdim
        })
        return cfg
    
    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)
    
    def call(self, inputs, training=True):
        noised, hazy, gamma, context, mask_in = inputs
        
        # 입력된 768차원 context를 512차원으로 변환합니다.
        # 이 레이어의 가중치는 학습 과정에서 최적화됩니다.
        projected_context = self.context_projection(context)
        
        e = self.e_norm(self.emb_proj(self.up_embed(self.sin_embed(gamma))))
        e = e * tf.reshape(self.e_scale, [1, 1, 1, -1])  # (B,H,W,C) * (1,1,1,C)
        
        # 입력 합성 부분
        x = tf.concat([noised, hazy, mask_in], axis=-1)   # (B, sz, sz , 7)   # mask_in을  1 - mask_in을 안 하고 그냥 넣으면 뚜렷한 부분을 집중적으로 학습하겠다는 의미
        x = self.input_conv(x)  # (B, sz, sz, w[0])
        x = x + e  # (B, sz, sz, w[0])

        sk = []
        for block in self.down_blocks:
            x, skip = block(x, training=training)     # 이렇게 바꾼이유는 python list를 쓰는 코드는 @tf.function으로 trace될때 동작하지 않을 수 있음 -> run_eagerly=False하면 그래프 tracing 중 무시되서 그럼(True일때는 가능함)
            sk.extend(skip)

        for block in self.mid_blocks:
            x = block(x, training = training)
        x = self.mid_attn([x, projected_context])
        
        for block in self.up_blocks:
            skip = sk.pop()
            x = block(x, skip, training=training)

        out = self.out_conv(x)
        return out
    
# 전역 평균과 표준편차

FIXED_MEAN = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32)
FIXED_STD  = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32)

def normalize_img(img):
    """[0,1] 범위의 이미지를 고정된 평균/표준편차로 정규화"""
    return (img - FIXED_MEAN) / FIXED_STD

# epoch이 커짐에 따라서 noise loss에서 x0 loss로 점진적으로 가중을 해주기 위한 epoch tracker
class EpochTracker(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.current_epoch = epoch
        
        
class DiffusionModel(tf.keras.Model):
    def __init__(self, sz, ws, bd, ctx_dim):
        super().__init__()
        self.mask_pred = build_mask_predictor()
        self.mask_pred.trainable = False
        #self.network   = get_network_conditional_mask(sz,ws,bd,ctx_dim)
        self.network = ConditionalUNet(sz, ws, bd, ctx_dim)
        self.ema_network = tf.keras.models.clone_model(self.network)
        self.ema_network.set_weights(self.network.get_weights())
        self.ema_network.trainable = False
        self.normalizer = normalize_img
        self.loss_fn   = keras.losses.MeanAbsoluteError()   # L1으로 해볼게요
        self.grad_accumulator = None

    def call(self, inputs, training=True):
        return self.network(inputs, training=training)

    def compile(self, optimizer, **kwargs):
        super().compile(optimizer=optimizer, loss=self.loss_fn, **kwargs)
        # 손실 and 평가 지표 확인용
        self.noise_loss_tracker = keras.metrics.Mean(name="noise_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="image_loss")
        self.psnr_metric        = keras.metrics.Mean(name="psnr")
        self.ssim_metric        = keras.metrics.Mean(name="ssim")
        # 그래디언트 누적을 위한 변수 초기화
        self.step_counter = tf.Variable(0, trainable=False, dtype=tf.int64)
        
    @property #함수 메소드를 인자로 사용가능 
    def metrics(self):
        # 모델이 추적하는 리스트 반환
        #return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]
        return [self.noise_loss_tracker, self.image_loss_tracker, self.psnr_metric, self.ssim_metric]
        
    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = FIXED_MEAN + images * FIXED_STD
        return tf.cast(tf.clip_by_value(images, 0.0, 1.0), tf.float32)   # 혹시 범위를 벗어날 수 있으니 0~1로 클리핑

    # def diffusion_schedule(self, t):      # linear scheduler
    #     # 1) 시작·끝 신호 세기의 역코사인(arc-cosine) 값을 구해서 각도로 바꿈
    #     sa, ea = tf.acos(max_signal_rate), tf.acos(min_signal_rate)
    #     # 2) t 비율만큼 시작 각도(sa)에서 끝 각도(ea)으로 선형 보간(interpolation)
    #     ang    = sa + t*(ea-sa)
    #     # 3) 그 각도의 사인·코사인 값을 그대로 노이즈 비율(nr)과
    #     #  신호 비율(sr)로 반환합니다
    #     return tf.sin(ang), tf.cos(ang) #sin이 노이즈비율 , cos이 신호비율
    
    def diffusion_schedule(self, t, s=0.008):   # normalized cosine scheduler
        ft = tf.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        c0 = math.cos((s / (1 + s)) * math.pi / 2.0) ** 2
        alpha_bar = ft / c0                      # ← 정규화
        alpha_bar = tf.clip_by_value(alpha_bar, min_signal_rate, max_signal_rate)
        sr = tf.sqrt(alpha_bar)
        nr = tf.sqrt(1.0 - alpha_bar)
        return nr, sr
    
    def make_gamma(self, nr, sr, eps=1e-12):
        # nr = sqrt(1 - alpha_bar), sr = sqrt(alpha_bar)
        return tf.math.log(tf.square(sr) + eps) - tf.math.log(tf.square(nr) + eps)  # logSNR
    
    def train_step(self, data):
        hazy, clear, context = data
        hazy_n  = self.normalizer(hazy)
        clear_n = self.normalizer(clear)
        b       = tf.shape(clear_n)[0]
        noise   = tf.random.normal((b,img_siz,img_siz,3))
       # (1) 1,2,…,T 중 하나를 균일 샘플링 (정수)
        t = tf.random.uniform(
        shape=(b, 1, 1, 1), 
        minval=0.0, maxval=1.0,
        dtype=tf.float32
        )

        nr,sr   = self.diffusion_schedule(t)
        gamma = self.make_gamma(nr, sr)
        x_noi   = sr * clear_n + nr * noise
        
        mask = self.mask_pred(hazy_n, training=False)
        
        #  학습 전 가중치 복사 (Graph-safe) -> 디버깅용
        before = tf.identity(self.network.trainable_weights[0])
    
        with tf.GradientTape() as tape:
            
            pred = self.network([x_noi, hazy_n, gamma, context, mask], training=True)
            
            
            # #context가 적용이 제대로 되는지에 대한 디버깅 파트
            # pred_zero_context = self.network([x_noi, hazy_n, t, tf.zeros_like(context), mask], training=False)
            # if tf.executing_eagerly():
            #   tf.print("▶ train diff norm(context 적용되는가):", tf.norm(pred - pred_zero_context))
            
            # #mask 적용된게 끝까지 잘 가는지에 대한 디버깅 파트
            # zero_mask = tf.zeros_like(mask)
            # zero_mask = tf.cast(zero_mask, dtype=mask.dtype)
            # zero_mask.set_shape(mask.shape)
            # pred_zero_mask = self.network([x_noi, hazy_n, gamma, context, zero_mask], training=False)
            # diff   = tf.norm(pred - pred_zero_mask)
            # norm_r = tf.norm(pred)

            # mask_ratio = diff / (norm_r + 1e-8)
            # if tf.executing_eagerly():
            #     tf.print("\n▶ diff norm:", diff, "|| pred_r norm:", norm_r, 
            #         "▶ mask_ratio:", mask_ratio)      # 0.03 ~ 0.15까지 정상    0.4 위면 오버컨디셔닝 의심
            
        
     
            # # 디버깅: e 효과 확인
            # pn0 = self.network([x_noi, hazy_n, gamma*0, context, mask], training=False)  # or 내부에서 e_scale=0로 강제
            # t_ratio = tf.norm(pred - pn0) / (tf.norm(pred) + 1e-8)
            # tf.print("▶ t_ratio:", t_ratio)     # 0.03~ 0.12면 정상
            
            
            pred_x0 = (x_noi - nr * pred) / sr
            
            # -> 이거 안 하는게 좋을거 같다. -=> 초반에 튀게 해줘야지 학습이 제대로 될거 같음
            # if epoch > 35:
            #     pred_x0 = tf.clip_by_value(pred_x0, -3.0, 3.0)
            # else:
            #     pred_x0 = pred_x0

            # if tf.executing_eagerly():
            #     tf.print("pred shape:", tf.shape(pred), "min/max:", tf.reduce_min(pred), "/", tf.reduce_max(pred))            
                        
            noise_loss = self.loss_fn(pred, noise)      # 학습 기준
            image_loss = self.loss_fn(clear_n, pred_x0) # 모니터링용
            # epoch에 따라 w_noise 증감
            # epoch = getattr(self, 'current_epoch', 0)
            # r = epoch / num_epochs
            # w_noise = 0.9 * (1.0 - r) + 0.1 * r  # 학습 초반엔 noise 중심 → 후반엔 x0 중심
            # w_x0    = 1 - w_noise
            # loss = w_noise * noise_loss + w_x0  * image_loss


            total_loss = noise_loss / float(gradient_accumulation_steps) # 손실 스케일링
            # PSNR/SSIM: GT는 clear_n 기준으로
            psnr = tf.reduce_mean(tf.image.psnr(clear_n, pred_x0, max_val=2.0))
            ssim = tf.reduce_mean(tf.image.ssim(clear_n, pred_x0, max_val=2.0))
            
            
            # if tf.executing_eagerly():
            #     tf.print("pred_x0 mean/std:", tf.reduce_mean(pred_x0), "/", tf.math.reduce_std(pred_x0))    # pred_mean = -0.05 ~ 0.05, pred_std = 0.4 ~ 0.6이면 정상 범위
            #     tf.print("clear_n mean/std:", tf.reduce_mean(clear_n), "/", tf.math.reduce_std(clear_n))

        vars_ = self.network.trainable_weights
        grads = tape.gradient(total_loss, vars_)

        # None/NaN/Inf 방어
        safe_grads = []
        for g, v in zip(grads, vars_):
            if g is None:
                g = tf.zeros_like(v)
            else:
                g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(v))
            safe_grads.append(g)

        # 누적기 초기화
        if self.grad_accumulator is None:
            self.grad_accumulator = [tf.Variable(tf.zeros_like(v), trainable=False) for v in vars_]

        # 누적
        for acc, g in zip(self.grad_accumulator, safe_grads):
            acc.assign_add(g)
    

        # 스텝 카운터 증가 및 가중치 업데이트
        self.step_counter.assign_add(1)

        # tf.cond에 사용할 조건(predicate)
        condition = tf.equal(self.step_counter % gradient_accumulation_steps, 0)

        def apply_and_reset_gradients():
            """가중치를 업데이트하고 누적기를 리셋하는 함수"""
            self.optimizer.apply_gradients(zip(self.grad_accumulator, vars_))
            for acc in self.grad_accumulator:
                acc.assign(tf.zeros_like(acc))
            
            # EMA 업데이트 로직도 여기에 포함
            for w, ew in zip(self.network.weights, self.ema_network.weights):
                ew.assign(0.999 * ew + 0.001 * w)

            # # 디버깅용 가중치 변화량 출력 (Eager mode에서만 실행됨)
            # if tf.executing_eagerly():
            #     after = self.network.trainable_weights[0]
            #     diff = tf.reduce_mean(tf.abs(after - before))
            #     tf.print("가중치 변화량 : ", diff)
            
            return tf.constant(True) # tf.cond는 반환값이 필요함

        def do_nothing():
            """아무것도 하지 않는 함수"""
            # EMA 업데이트는 가중치 업데이트 시에만 수행하도록 변경
            # for w, ew in zip(self.network.weights, self.ema_network.weights):
            #     ew.assign(0.999 * ew + 0.001 * w)
            return tf.constant(False)

        # tf.cond를 사용하여 조건부로 가중치 업데이트 실행
        tf.cond(condition, apply_and_reset_gradients, do_nothing)


        # trackers update
        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)
        self.psnr_metric.update_state(psnr)
        self.ssim_metric.update_state(ssim)
    
        # if tf.executing_eagerly():
        #     tf.print("trainable_weights:", len(self.network.trainable_weights))
        #     tf.print("max grad:", tf.reduce_max([tf.reduce_max(tf.abs(g)) for g in grads if g is not None]))   
        

            
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        hazy, clear, context = data
        hazy_n  = self.normalizer(hazy)
        clear_n = self.normalizer(clear)
        b       = tf.shape(clear_n)[0]
        noise   = tf.random.normal((b, img_siz, img_siz, 3))
       # (1) 1,2,…,T 중 하나를 균일 샘플링 (정수)
        t = tf.random.uniform(
        shape=(b, 1, 1, 1), 
        minval=0.0, maxval=1.0,
        dtype=tf.float32
        )
        
        nr,sr   = self.diffusion_schedule(t)
        gamma = self.make_gamma(nr, sr)

        x_noi   = sr * clear_n + nr * noise

        mask = self.mask_pred(hazy_n, training=False)
        #  EMA로 예측
        pred_noise = self.ema_network([x_noi, hazy_n, gamma, context, mask], training=False)
        pred_x0 = (x_noi - nr * pred_noise) / sr
        
        noise_loss = self.loss_fn(noise, pred_noise)
        image_loss = self.loss_fn(clear_n, pred_x0)
        # PSNR/SSIM: GT는 clear_n 기준으로
        psnr = tf.reduce_mean(tf.image.psnr(clear_n, pred_x0, max_val=2.0))
        ssim = tf.reduce_mean(tf.image.ssim(clear_n, pred_x0, max_val=2.0))
        
        # trackers는 일반적으로 train 전용이지만, 간단히 여기서도 업데이트 가능(선택)
        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)
        self.psnr_metric.update_state(psnr)
        self.ssim_metric.update_state(ssim)

        return {m.name: m.result() for m in self.metrics}


    def dehaze(self, hazy_imgs, ctx, steps=kid_diffusion_steps):   
        hazy_n = self.normalizer(hazy_imgs)

        noise = tf.random.normal((tf.shape(hazy_n)[0], img_siz, img_siz, 3))
        next_x = noise
        mask = self.mask_pred(hazy_n, training=False)
        for i in reversed(range(steps)):

            x = next_x
            t = tf.fill([tf.shape(x)[0], 1, 1, 1], i / steps)
            nr, sr = self.diffusion_schedule(t)
            gamma = self.make_gamma(nr, sr)

            # 1) 네트워크에서 예측된 노이즈 -> train에서의 ema_network 사용
            pn = self.ema_network([x, hazy_n, gamma, ctx, mask], training=False)

            # 2) 학습 시와 동일하게 tanh 스케일링 적용
            px0_raw = (x - nr * pn) / sr
            
            #epoch = getattr(self, 'current_epoch', 0)
            # if epoch > 35:
            #     px0 = tf.clip_by_value(px0_raw, -3.0, 3.0)
            # else:
            #     px0 = px0_raw

            px0 = px0_raw

            # # 4) 디버깅용 출력 (분포 확인)
            # if tf.executing_eagerly():
            #     tf.print("clippingX px0 min/max:", tf.reduce_min(px0_raw), "/", tf.reduce_max(px0_raw))   # tf.tanh 적용시키기 전

            # 5) 다음 스텝으로 업데이트
            next_t = t - 1.0 / steps
            next_nr, next_sr = self.diffusion_schedule(next_t)
            next_x = next_sr * px0 + next_nr * pn
            
        return px0  # 최종 예측된 x0 반환
    
    
    def plot_images(self, epoch, dataset, dataset_tag, save_dir = "/home/jang/DDIM_python/paper/samples"):
        """
        주어진 하나의 데이터셋에 대해 이미지를 생성하고 저장합니다.
        epoch: 현재 에폭 번호 (파일 이름에 사용)
        dataset: tf.data.Dataset 객체 (이미지를 생성할 데이터)
        dataset_tag: 'RESIDE', 'I-HAZE' 등 출력물에 표시될 이름
        save_dir: 이미지를 저장할 폴더 경로
        """
        os.makedirs(save_dir, exist_ok=True)
        gc.collect() # 가비지 컬렉션으로 메모리 정리 시도

        # 데이터셋에서 한 배치만 가져옵니다.
        for hazy_b, clear_b, ctx_b in dataset.take(1):
            # 이미지 생성
            dehazed_b = self.dehaze(hazy_b, ctx_b)
            dehazed_b = tf.clip_by_value(dehazed_b, -1.0, 1.0)
            dehazed_b = self.denormalize(dehazed_b)

            # Matplotlib으로 이미지 그리기
            batch_size = hazy_b.shape[0]
            fig, axs = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))

            if batch_size == 1:
                axs = np.array([axs])

            for i in range(batch_size):
                axs[i, 0].imshow(hazy_b[i])
                axs[i, 0].set_title(f"{dataset_tag} • Hazy")
                axs[i, 0].axis("off")

                axs[i, 1].imshow(dehazed_b[i])
                axs[i, 1].set_title(f"Dehazed (Epoch {epoch})")
                axs[i, 1].axis("off")

                axs[i, 2].imshow(clear_b[i])
                axs[i, 2].set_title(f"{dataset_tag} • Clear")
                axs[i, 2].axis("off")

            # 파일 저장
            epoch_str = f"{epoch:03d}" if isinstance(epoch, int) else str(epoch)
            save_path = os.path.join(save_dir, f"generated_epoch_{epoch_str}_{dataset_tag}.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=200)
            plt.close(fig)
            print(f"이미지 저장 완료: {save_path}")  



reside_output_dir = "/home/jang/DDIM_python/paper/samples_reside"
ihaze_output_dir = "/home/jang/DDIM_python/paper/samples_ihaze"

print(" 저장된 최적 가중치로 이미지 생성을 시작합니다.")

# 1. 새로운 모델 인스턴스를 생성합니다.
model = DiffusionModel(img_siz, widths, block_depth, ctx_dim)

input_shape = [
    (batch_siz, img_siz, img_siz, 3), # noised
    (batch_siz, img_siz, img_siz, 3), # hazy
    (batch_siz, 1, 1, 1),             # gamma
    (batch_siz, seq_len, 768),        # context (원본 768 차원)
    (batch_siz, img_siz, img_siz, 1), # mask_in
]

print("모델을 빌드합니다...")
model.build(input_shape=input_shape)
print("모델 빌드 완료.")


checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch*.h5")))
print(f"총 {len(checkpoint_files)}개의 에폭 체크포인트를 찾았습니다.")

# 3. 모델을 컴파일합니다. (추론만 할 때는 간단하게 해도 됩니다)
model.compile(optimizer=optimizers.Adam())


# 5. 각 체크포인트에 대해 반복 작업 수행
for ckpt_path in checkpoint_files:
    
    # 파일명에서 에폭 번호 추출 (예: 'epoch045-...' -> 45)
    epoch_num_match = re.search(r"epoch(\d+)", os.path.basename(ckpt_path))
    if not epoch_num_match:
        continue # 'epoch' 패턴이 없는 파일은 건너뛰기
    epoch_num = int(epoch_num_match.group(1))
    
    print(f"--- Epoch {epoch_num} 가중치로 이미지 생성 중... ---")
    
    # 해당 에폭의 가중치 로드
    model.load_weights(ckpt_path)
    
    # 첫 번째 데이터셋(RESIDE)에 대한 이미지 생성
    print(f"--- RESIDE 데이터셋 생성 중... ---")
    model.plot_images(
        epoch=epoch_num,
        dataset=val_ds, # RESIDE 데이터셋 전달
        dataset_tag="RESIDE",
        save_dir=reside_output_dir
    )

    # 두 번째 데이터셋(I-HAZE)에 대한 이미지 생성
    print(f"--- I-HAZE 데이터셋 생성 중... ---")
    if ihaze_ds is not None: # ihaze_ds가 있을 경우에만 실행
        model.plot_images(
            epoch=epoch_num,
            dataset=ihaze_ds, # I-HAZE 데이터셋 전달
            dataset_tag="I-HAZE",
            save_dir=ihaze_output_dir
        )
    
    
print(f" 모든 이미지 생성이 완료되었습니다.")

    