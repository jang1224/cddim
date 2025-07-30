# ─── 0) 필수 라이브러리 설치 ───
!pip install transformers
!pip install opencv-python-headless

# ─── 1) Colab에 Google Drive 마운트 ───
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# ─── 2) 라이브러리 임포트 ───
import os, math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipTextModel
from PIL import Image
import torch
import json


# ─── 3) 하이퍼파라미터 ───
img_siz             = 64
batch_siz           = 16
kid_diffusion_steps = 50    # ← must be before class definition
min_signal_rate     = 0.02
max_signal_rate     = 0.95
zdim                = 32
embed_max_freq      = 1000.0
widths              = [320, 640, 1280, 1280]
block_depth         = 2
ctx_dim             = 768   # CLIP ViT-L/14
seq_len             = 77    # tokenizer max length
num_epochs          = 30    # 학습 에폭

# 모델 준비
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model     = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).cuda().eval()
#텍스트를 임베딩하는 모델
txt_processor = processor.tokenizer  # BLIPProcessor 안에 tokenizer
txt_model     = BlipTextModel.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                ).cuda().eval()


# ——— 1) 데이터 경로 설정 ———
base_dir = '/content/RESIDE-6K/train/'
hazy_dir = os.path.join(base_dir, 'hazy')
ref_dir  = os.path.join(base_dir, 'GT')

base_dir_test = '/content/RESIDE-6K/test/'
hazy_test_dir = os.path.join(base_dir_test, 'hazy')
ref_test_dir  = os.path.join(base_dir_test, 'GT')

# hazy 이미지 경로들
image_paths = list(Path(hazy_dir).rglob("*.jpg"))
exts     = ('.png', '.jpg', '.jpeg')

# # 이미지 하나씩 캡셔닝
# for img_path in image_paths:
#     raw_image = Image.open(img_path).convert('RGB')
#     inputs = processor(raw_image, return_tensors="pt").to("cuda")

#     with torch.no_grad():
#         out = model.generate(**inputs, max_length=50)
#         caption = processor.decode(out[0], skip_special_tokens=True)

#     context_dict[str(img_path)] = caption
#     print(f"{img_path.name} → {caption}")

# # JSON 저장
# # Google Drive 등 원하는 경로에 저장
# output_json = "/content/RESIDE-6K/train/reside6k_contexts_blip.json"

# with open(output_json, "w") as f:
#     json.dump(context_dict, f, indent=2)

# print(f"context 저장 완료 : {output_json}")

json_path = "/content/drive/Mydrive/reside6k_contexts_blip.json"
with open(json_path, "r") as f:
  context_dict = json.load(f)


checkpoint_dir = os.path.join(base_dir, 'checkpoints_weights')
os.makedirs(checkpoint_dir, exist_ok=True)

exts = {'.jpg', '.jpeg', '.png'}

#train용
hazy_files = sorted(str(p) for p in Path(hazy_dir).rglob('*') if p.suffix.lower() in exts)
ref_files  = sorted(str(p) for p in Path(ref_dir).rglob('*')  if p.suffix.lower() in exts)
contexts   = [context_dict.get(path, "") for path in hazy_files]  # 리스트 순서 맞춤(학습용)

#test용
hazy_test_files = sorted(str(p) for p in Path(hazy_test_dir).rglob('*') if p.suffix.lower() in exts)
ref_test_files  = sorted(str(p) for p in Path(ref_test_dir).rglob('*')  if p.suffix.lower() in exts)
empty_caps = [''] * len(hazy_test_files) # 빈 캡션 리스트 (테스트용)

assert len(hazy_files)==len(ref_files), "파일 개수 불일치"
print("✔ 총 이미지 쌍:", len(hazy_files))

def encode_context(text_str: str) -> np.ndarray:
    """
    문자열(text_str) → BLIP 텍스트 인코더 last_hidden_state[0]
    → NumPy (seq_len, ctx_dim) float32 배열 반환
    """
    inputs = processor.tokenizer(
        [text_str],
        padding="max_length", truncation=True, max_length=seq_len,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[0]  # (seq_len, ctx_dim)

    return emb.cpu().numpy().astype(np.float32)

# ─── 6) tf.data 파이프라인 ───
def load_pair(h_path, r_path, context_text):
    hazy  = tf.image.decode_jpeg(tf.io.read_file(h_path), channels=3)
    clear = tf.image.decode_jpeg(tf.io.read_file(r_path), channels=3)
    hazy  = tf.image.resize(hazy,  [img_siz,img_siz])
    clear = tf.image.resize(clear,[img_siz,img_siz])
    hazy  = tf.image.convert_image_dtype(hazy, tf.float32)
    clear = tf.image.convert_image_dtype(clear, tf.float32)
    
    # context 텍스트 -> BLIP 임베딩
    def _enc_or_gen(txt_bytes, img_path_bytes):
        txt = txt_bytes.decode("utf-8")
        if txt:
            # train: 미리 주어진 캡션
            return encode_context(txt)
        else:
            # test: 이미지로부터 캡션 생성
            path  = img_path_bytes.decode("utf-8")
            img   = Image.open(path).convert("RGB")
            inp   = processor(img, return_tensors="pt").to("cuda")
            out   = model.generate(**inp, max_length=50)
            cap   = processor.decode(out[0], skip_special_tokens=True)
            return encode_context(cap)

    ctx = tf.py_function(
        func=_enc_or_gen,
        inp=[context_text, h_path],
        Tout=tf.float32
    )
    ctx.set_shape([seq_len, ctx_dim]) #tf.py_function은 반환텐서의 형태와 데이터 타입을 자동으로 추론 못하기 때문에 set_shape로 형태 고정

    return hazy, clear, ctx

# ——— train/test Dataset 정의 ———
ds_train = tf.data.Dataset.from_tensor_slices((hazy_files, ref_files, contexts))
ds_test  = tf.data.Dataset.from_tensor_slices((hazy_test_files, ref_test_files, empty_caps))


train_ds = (
    ds_train
    .map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(len(hazy_files), seed=42)
    .batch(batch_siz)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    ds_test
    .map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_siz)
    .prefetch(tf.data.AUTOTUNE)
)


# ─── 7) Dark Channel Prior 기반 마스크 예측 함수 정의 ───
def DarkChannel(im, sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    return cv2.erode(dc, kernel)

def AtmLight(im, dark):
    h,w = im.shape[:2]
    imsz = h*w
    numpx = max(imsz//1000, 1)
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz,3)
    indices = np.argsort(darkvec)[-numpx:]
    return np.mean(imvec[indices], axis=0, keepdims=True)

def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)
    for i in range(3):
        im3[:,:,i] = im[:,:,i]/A[0,i]
    return 1 - omega * DarkChannel(im3, sz)

def Guidedfilter_gray(im, p, r=60, eps=1e-4):
    mean_I  = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_p  = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    cov_Ip  = cv2.boxFilter(im*p,cv2.CV_64F,(r,r)) - mean_I*mean_p
    var_I   = cv2.boxFilter(im*im,cv2.CV_64F,(r,r)) - mean_I*mean_I
    a       = cov_Ip/(var_I+eps)
    b       = mean_p - a*mean_I
    return cv2.boxFilter(a,cv2.CV_64F,(r,r))*im + cv2.boxFilter(b,cv2.CV_64F,(r,r))

def TransmissionRefine(im, et):
    gray = cv2.cvtColor((im*255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)/255
    return Guidedfilter_gray(gray, et)

def estimate_transmission(src):
    I = src.astype('float64')/255
    dark = DarkChannel(I,15)
    A    = AtmLight(I,dark)
    te   = TransmissionEstimate(I,A,15)
    t    = TransmissionRefine((src*255).astype(np.uint8), te)
    return t.astype(np.float32)

def estimate_transmission_batch(batch_np: np.ndarray) -> np.ndarray:
    masks = []
    for im in batch_np:
        t = estimate_transmission((im*255).astype(np.uint8))
        masks.append(t[...,None])
    return np.stack(masks, axis=0)  # (B,H,W,1)

def build_mask_predictor():
    inp = keras.Input((img_siz, img_siz, 3), dtype=tf.float32)
    mask = layers.Lambda(
        lambda batch: tf.numpy_function(
            func=estimate_transmission_batch,
            inp=[batch],
            Tout=tf.float32
        ),
        output_shape=(img_siz, img_siz, 1)
    )(inp)
    mask = layers.Lambda(lambda x: tf.ensure_shape(x, [None, img_siz, img_siz, 1]))(mask)
    return keras.Model(inputs=inp, outputs=mask, name='mask_predictor')

# ─── 8) Sinus Embedding & U‑Net 블록 ───
@tf.keras.utils.register_keras_serializable(package='ddpm')
def sinusoidal_embedding(x):
    freqs = tf.exp(tf.linspace(tf.math.log(1.0),
                               tf.math.log(embed_max_freq),
                               zdim//2))
    ang   = 2*math.pi*freqs
    return tf.concat([tf.sin(ang*x), tf.cos(ang*x)], axis=3)

def ResidualBlock(w):
    def f(x):
        res = x if x.shape[-1]==w else layers.Conv2D(w,1)(x)
        y   = layers.BatchNormalization(center=False,scale=False)(x)
        y   = layers.Conv2D(w,3,padding='same',activation='swish')(y)
        y   = layers.Conv2D(w,3,padding='same')(y)
        return layers.Add()([y,res])
    return f

def DownBlock(w,bd):
    def f(xs):
        x,sk = xs
        for _ in range(bd):
            x = ResidualBlock(w)(x); sk.append(x)
        return layers.AveragePooling2D(2)(x)
    return f

def UpBlock(w,bd):
    def f(xs):
        x,sk = xs
        x = layers.UpSampling2D(2,interpolation='bilinear')(x)
        for _ in range(bd):
            x = layers.Concatenate()([x,sk.pop()])
            x = ResidualBlock(w)(x)
        return x
    return f

# ─── 9) Cross‑Attn 블록 ───
class CrossAttentionBlock(layers.Layer):
    def __init__(self, channels, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.mha  = layers.MultiHeadAttention(num_heads=num_heads,
                                              key_dim=channels//num_heads)
        self.proj = layers.Dense(channels)
    def call(self, inputs):
        x, context = inputs
        B,H,W,C    = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x_flat     = tf.reshape(x, [B, H*W, C])
        attn       = self.mha(query=x_flat, value=context, key=context)
        attn       = self.proj(attn)
        attn       = tf.reshape(attn, [B,H,W,C])
        return layers.Add()([x, attn])

def get_network_conditional_mask(sz, ws, bd, ctx_dim):
    noised  = keras.Input((sz,sz,3), name='noised')
    hazy    = keras.Input((sz,sz,3), name='hazy')
    tstep   = keras.Input((1,1,1), name='time_emb')
    context = keras.Input((seq_len,ctx_dim), name='ctx')
    mask_in = keras.Input((sz,sz,1), name='mask')

    e = layers.Lambda(sinusoidal_embedding)(tstep)
    e = layers.UpSampling2D(sz, interpolation='nearest')(e)

    x = layers.Conv2D(ws[0],1)(noised)
    x = layers.Concatenate()([x, hazy*mask_in, e])

    sk = []
    # Down: Residual only
    for w in ws[:-1]:
        x = DownBlock(w, block_depth)([x, sk])

    # Bottleneck + single cross-attn
    for _ in range(block_depth):
        x = ResidualBlock(ws[-1])(x)
    x = CrossAttentionBlock(ws[-1])([x, context])

    # Up: Residual only
    for w in reversed(ws[:-1]):
        x = UpBlock(w, block_depth)([x, sk])

    out = layers.Conv2D(3, 1, kernel_initializer='zeros')(x)
    return keras.Model([noised, hazy, tstep, context, mask_in], out)

# ─── 11) DiffusionModel 정의 ───
def normalize_img(img):
    return (img - 0.5) / 0.5

class DiffusionModel(keras.Model):
    def __init__(self, sz, ws, bd, ctx_dim):
        super().__init__()
        self.mask_pred = build_mask_predictor()
        self.mask_pred.trainable = False
        self.network   = get_network_conditional_mask(sz,ws,bd,ctx_dim)
        self.loss_fn   = keras.losses.MeanSquaredError()

    def call(self, inputs, training=False):
        return self.network(inputs, training=training)

    def compile(self, optimizer):
        super().compile(optimizer=optimizer, loss=self.loss_fn)

    def diffusion_schedule(self, t):
        sa, ea = tf.acos(max_signal_rate), tf.acos(min_signal_rate)
        ang    = sa + t*(ea-sa)
        return tf.sin(ang), tf.cos(ang)

    def train_step(self, data):
        hazy, clear, context = data
        hazy_n  = normalize_img(hazy)
        clear_n = normalize_img(clear)
        b       = tf.shape(clear_n)[0]
        noise   = tf.random.normal((b,img_siz,img_siz,3))
        t       = tf.random.uniform((b,1,1,1))
        nr,sr   = self.diffusion_schedule(t)
        x_noi   = sr * clear_n + nr * noise

        with tf.GradientTape() as tape:
            mask = self.mask_pred(hazy_n, training=False)
            pred = self.network([x_noi, hazy_n, t, context, mask], training=True)
            loss = self.loss_fn(noise, pred)

        grads = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_weights))
        return {'loss': loss}

    def test_step(self, data):
        hazy, clear, context = data
        hazy_n  = normalize_img(hazy)
        clear_n = normalize_img(clear)
        b       = tf.shape(clear_n)[0]
        noise   = tf.random.normal((b, img_siz, img_siz, 3))
        t       = tf.random.uniform((b, 1, 1, 1))
        nr,sr   = self.diffusion_schedule(t)
        x_noi   = sr * clear_n + nr * noise

        mask = self.mask_pred(hazy_n, training=False)
        pred = self.network([x_noi, hazy_n, t, context, mask], training=False)
        loss = self.loss_fn(noise, pred)
        return {'loss': loss}

    def dehaze(self, hazy_imgs, ctx, steps=kid_diffusion_steps):
        x      = hazy_imgs
        hazy_n = normalize_img(hazy_imgs)
        mask   = self.mask_pred(hazy_n, training=False)
        for i in reversed(range(steps)):
            t     = tf.fill([tf.shape(x)[0],1,1,1], i/steps)
            nr,sr = self.diffusion_schedule(t)
            pn    = self.network([x, hazy_n, t, ctx, mask], training=False)
            x     = (x - nr*pn) / sr
        return x


# ─── 12) 모델 생성 & Build ───
model = DiffusionModel(img_siz, widths, block_depth, ctx_dim)
_ = model.mask_pred(tf.zeros((1,img_siz,img_siz,3)))
_ = model.network([
    tf.zeros((1,img_siz,img_siz,3)),
    tf.zeros((1,img_siz,img_siz,3)),
    tf.zeros((1,1,1,1)),
    tf.zeros((1,seq_len,ctx_dim)),
    tf.zeros((1,img_siz,img_siz,1))
])

# ─── 12.5) 모델 빌드(dummy call) ───
dummy_hazy  = tf.zeros((1, img_siz, img_siz, 3), dtype=tf.float32)
dummy_t     = tf.zeros((1, 1, 1, 1), dtype=tf.float32)
dummy_ctx   = tf.zeros((1, seq_len, ctx_dim), dtype=tf.float32)
dummy_mask  = tf.zeros((1, img_siz, img_siz, 1), dtype=tf.float32)
_ = model([dummy_hazy, dummy_hazy, dummy_t, dummy_ctx, dummy_mask], training=False)

# ─── 13) 학습 설정 & 실행 ───
model.compile(optimizer=keras.optimizers.Adam(1e-4))
model.run_eagerly = True

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "weights_epoch_{epoch:02d}.weights.h5"),
    save_weights_only=True,
    save_freq='epoch',
    verbose=1
)
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=num_epochs,
    callbacks=[checkpoint_cb]
)

# ─── 14) 테스트 & 시각화 ───
for hazy_b, _, ctx_b in val_ds.take(1):
    out = model.dehaze(hazy_b, ctx_b).numpy()
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1); plt.title("입력 hazy"); plt.imshow(hazy_b[0]); plt.axis('off')
    plt.subplot(1,2,2); plt.title("Dehazed"); plt.imshow(out[0]);   plt.axis('off')
    plt.show()
