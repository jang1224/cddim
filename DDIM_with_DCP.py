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
from keras import layers
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipTextModel, BartConfig, TFBartForConditionalGeneration
from transformers.models.bart.modeling_tf_bart import TFBartAttention
from PIL import Image
import torch
import json
from datasets import load_dataset
# #context임베딩할 때 필요
# from tqdm import tqdm

np.random.seed(None)
tf.random.set_seed(None)

print("GPU available:", tf.config.list_physical_devices('GPU'))


# ─── 3) 하이퍼파라미터 ───
img_siz             = 64
batch_siz           = 8
kid_diffusion_steps = 100    # ← must be before class definition
min_signal_rate     = 0.02
max_signal_rate     = 0.95
zdim                = 32
embed_max_freq      = 1000.0
widths              = [160,320,768,768] # 768인 이유는 bottleneck 채널수와 crossattention(BART)의 d_model = 768로 같아야 가중치 로드가 에러 없이 됨.
block_depth         = 2
ctx_dim             = 768   # CLIP ViT-L/14
seq_len             = 77    # tokenizer max length
num_epochs          = 50    # 학습 에폭

# 모델 준비
processor_blip = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_blip     = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).cuda().eval()
#텍스트를 임베딩하는 모델
txt_processor_blip = processor_blip.tokenizer  # BLIPProcessor 안에 tokenizer
txt_model_blip     = BlipTextModel.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                ).cuda().eval()


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
#     print(f"{img_path.name} → {caption}")

# # JSON 저장
# # Google Drive 등 원하는 경로에 저장
# output_json = "/content/RESIDE-6K/train/reside6k_contexts_blip.json"

# with open(output_json, "w") as f:
#     json.dump(context_dict, f, indent=2)

# print(f"context 저장 완료 : {output_json}")

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
empty_caps = [''] * len(hazy_test_files) # 빈 캡션 리스트 (테스트용)

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
    ).to("cuda")

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
    
def load_pair_train(h_path, r_path, _):  # dummy 세 번째 인자
    hazy  = tf.image.decode_jpeg(tf.io.read_file(h_path), channels=3)
    clear = tf.image.decode_jpeg(tf.io.read_file(r_path), channels=3)
    hazy  = tf.image.resize(hazy,  [img_siz,img_siz])
    clear = tf.image.resize(clear,[img_siz,img_siz])
    hazy  = tf.cast(hazy, tf.float32) / 255.0
    clear = tf.cast(clear, tf.float32) / 255.0

    stem     = tf.strings.split(h_path, os.sep)[-1]
    npy_name = tf.strings.regex_replace(stem, ".jpg", ".npy")
    ctx_path = tf.strings.join([context_embedding_dir, "/", npy_name])

    def _load_context(path_bytes):
        return np.load(path_bytes.decode("utf-8")).astype(np.float32)

    ctx = tf.numpy_function(_load_context, [ctx_path], tf.float32)
    ctx.set_shape([seq_len, ctx_dim])
    return hazy, clear, ctx


def load_pair_test(h_path, r_path, _):
    hazy  = tf.image.decode_jpeg(tf.io.read_file(h_path), channels=3)
    clear = tf.image.decode_jpeg(tf.io.read_file(r_path), channels=3)
    hazy  = tf.image.resize(hazy,  [img_siz,img_siz])
    clear = tf.image.resize(clear,[img_siz,img_siz])
    hazy  = tf.cast(hazy, tf.float32) / 255.0
    clear = tf.cast(clear, tf.float32) / 255.0

    def _generate_context(h_path_bytes):
        path = h_path_bytes.numpy().decode("utf-8")
        img  = Image.open(path).convert("RGB")
        inp  = processor_blip(img, return_tensors="pt").to("cuda")
        out  = model_blip.generate(**inp, max_length=50)
        cap  = processor_blip.decode(out[0], skip_special_tokens=True)
        return encode_context(cap)

    ctx = tf.py_function(_generate_context, [h_path], tf.float32)
    ctx.set_shape([seq_len, ctx_dim])
    return hazy, clear, ctx


# ——— train/test Dataset 정의 ———
ds_train = tf.data.Dataset.from_tensor_slices((hazy_files, ref_files, contexts))
ds_test  = tf.data.Dataset.from_tensor_slices((hazy_test_files, ref_test_files, empty_caps))

total_count = len(hazy_files)
print("total_count", total_count)

train_ds = (
    ds_train
    .take(total_count)
    .map(load_pair_train, num_parallel_calls=tf.data.AUTOTUNE)
    .repeat()
    .shuffle(batch_siz)
    .batch(batch_siz)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    ds_test
    .map(load_pair_test, num_parallel_calls=tf.data.AUTOTUNE)
    .repeat()
    .cache()
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
    return np.clip(t.astype(np.float32),0.0,1.0)

def estimate_transmission_batch(batch_np: np.ndarray) -> np.ndarray:
    masks = []
    for im in batch_np:
        t = estimate_transmission((im*255).astype(np.uint8))
        masks.append(t[...,None])
    return np.stack(masks, axis=0)  # (B,H,W,1)

def build_mask_predictor():
    inp = keras.Input((img_siz, img_siz, 3), dtype=tf.float32)

    mask = layers.Lambda(
        lambda batch: tf.ensure_shape(
            tf.numpy_function(
                func=estimate_transmission_batch,
                inp=[batch],
                Tout=tf.float32
            ),
            [None, img_siz, img_siz, 1]   # 배치는 None, H/W/1은 고정
        ),
        output_shape=(img_siz, img_siz, 1)  # Keras에도 알리기
    )(inp)

    return keras.Model(inputs=inp, outputs=mask, name='mask_predictor')

# ─── 8) Sinus Embedding & U‑Net 블록 ───
@tf.keras.utils.register_keras_serializable(package='ddpm')
def sinusoidal_embedding(x):    # 위치 임베딩처럼, 각 스텝 t를 여러 주파수의 사인, 코사인 함수로 변환함.
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


# 사전학습된 attnetion 불러옴.
class CrossAttentionBlock(layers.Layer):
    #layer_idx는 층의 인덱스를 나타내는데 BART는 총 attnetion이 6개가 있음. -> 인덱스가 커질수록 low level에서 high level의 의미 정보를 담고 있음
    def __init__(self, channels, hf_model="facebook/bart-base", layer_idx=0, **kwargs):
        super().__init__(**kwargs)
        # BART config: cross-attention 활성화
        # Config 로드
        config = BartConfig.from_pretrained(hf_model, add_cross_attention=True)
        self.d_model   = config.d_model                    # e.g. 768
        self.n_heads   = config.decoder_attention_heads    # e.g. 12
        self.attn_drop = config.attention_dropout          # 일반적으로 0.0~0.1
        # CrossAttentionBlock __init__에서 추가
        self.trainable = True             # 전체 Layer도 설정


        # 사전학습된 구조 그대로 만든 TF 레이어
        # 올바른 인자 전달
        self.cross_attn = TFBartAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=self.attn_drop,
            is_decoder=True,           # cross-attn이므로 decoder=True
            name="hf_cross_attn"
        )
        #self.cross_attn.trainable = True  # 반드시 명시해줘야 함
        # attention 출력 차원 → U-Net 채널로 매핑하는 프로젝션
        self.proj       = layers.Dense(channels)
        self.proj.trainable = True        # Dense projection도
        # 이후 build()에서 사용하기 위한 정보 저장
        self.hf_model   = hf_model
        self.layer_idx  = layer_idx

    def build(self, input_shape):
        super().build(input_shape)
        # 실제 BART 가중치 로드
        bart = TFBartForConditionalGeneration.from_pretrained(self.hf_model)
        # 디코더 지정 레이어의 encoder_attn weight 추출
        pretrained = bart.model.decoder.layers[self.layer_idx].encoder_attn
        # cross_attn 레이어를 실제 입력 형태로 "build" 시켜서 가중치 슬롯 생성
        # 입력은 (batch, seq_len, d_model)
        dummy_shape = (None, seq_len, self.d_model)
        self.cross_attn.build(dummy_shape)
        # 우리의 cross_attn 레이어에 가중치 덮어쓰기
        self.cross_attn.set_weights(pretrained.get_weights())
        self.cross_attn.trainable = True  # build 이후에도 한 번 더 설정

    def call(self, inputs):
        x, context = inputs
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        # 공간차원(H×W)을 토큰 길이로 flatten
        x_flat     = tf.reshape(x, [B, H*W, C])
        # HF cross-attn 출력 (tuple 중 첫 번째가 attn_output)
        attn_out   = self.cross_attn(
                        x_flat,
                        key_value_states=context
                    )[0]
        # U-Net 차원으로 프로젝션
        attn_proj  = self.proj(attn_out)
        # 다시 (B,H,W,C)로 복원
        attn_map   = tf.reshape(attn_proj, [B, H, W, C])
        # residual 연결
        return layers.Add()([x, attn_map])


def get_network_conditional_mask(sz, ws, bd, ctx_dim):
    noised  = keras.Input((sz,sz,3), name='noised') # train_step에서는 랜덤으로 만든 xt 이미지
    hazy    = keras.Input((sz,sz,3), name='hazy')
    tstep   = keras.Input((1,1,1), name='time_emb')
    context = keras.Input((seq_len,ctx_dim), name='ctx')
    mask_in = keras.Input((sz,sz,1), name='mask')

    e = layers.Lambda(sinusoidal_embedding)(tstep)  # 서로 다른 빈도로 변하는 사인,코사인 값들이 합쳐져, network가 몇 번째 스텝인지 구분하기 쉬워짐
    e = layers.UpSampling2D(sz, interpolation='nearest')(e)
    e = layers.Conv2D(ws[0], kernel_size=1, activation='swish')(e)  # 추가
    
    x = layers.Concatenate()([noised,hazy,e])
    x = layers.Conv2D(ws[0],kernel_size=1)(x)
    sk = []
    # Down: Residual only
    for w in ws[:-1]:
        x = DownBlock(w, bd)([x, sk])

    for _ in range(bd):
        x = ResidualBlock(ws[-1])(x)
    #x = CrossAttentionBlock(ws[-1])([x, context])

    # Up: Residual only
    for w in reversed(ws[:-1]):
        x = UpBlock(w, bd)([x, sk])

    out = layers.Conv2D(3, 1, activation='linear', kernel_initializer= 'he_normal')(x)       # 예측된 노이즈를 담고 있는 텐서 출력 (B, sz,sz, 3)
    return keras.Model([noised, hazy, tstep, context, mask_in], out) # out = 𝜖^𝜃


# 전역 평균과 표준편차
FIXED_MEAN = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32)
FIXED_STD  = tf.constant([0.25, 0.25, 0.25], dtype=tf.float32)

def normalize_img(img):
    """[0,1] 범위의 이미지를 고정된 평균/표준편차로 정규화"""
    return (img - FIXED_MEAN) / FIXED_STD

# epoch이 커짐에 따라서 noise loss에서 x0 loss로 점진적으로 가중을 해주기 위한 epoch tracker
class EpochTracker(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.current_epoch = epoch
        
        
class DiffusionModel(keras.Model):
    def __init__(self, sz, ws, bd, ctx_dim):
        super().__init__()
        self.mask_pred = build_mask_predictor()
        self.mask_pred.trainable = False
        self.network   = get_network_conditional_mask(sz,ws,bd,ctx_dim)
        self.normalizer = normalize_img
        self.loss_fn   = keras.losses.MeanSquaredError()

    def call(self, inputs, training=True):
        return self.network(inputs, training=training)

    def compile(self, optimizer):
        super().compile(optimizer=optimizer, loss=self.loss_fn)
        
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
    def diffusion_schedule(self, t, s=0.008):
        ft = tf.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = ft  # normalize 안 하고 그대로 사용하거나,
        # alpha_bar = ft / tf.reduce_max(ft)  # 이 방식도 가능
        alpha_bar = tf.clip_by_value(alpha_bar, 1e-4, 0.999)
        sr = tf.sqrt(alpha_bar)
        nr = tf.sqrt(1.0 - alpha_bar)
        return nr, sr
    
    def train_step(self, data):
        hazy, clear, context = data
        hazy_n  = self.normalizer(hazy)
        clear_n = self.normalizer(clear)
        b       = tf.shape(clear_n)[0]
        noise   = tf.random.normal((b,img_siz,img_siz,3))
       # (1) 1,2,…,T 중 하나를 균일 샘플링 (정수)
        t_int = tf.random.uniform(
        shape=(b,),
        minval=1,
        maxval=kid_diffusion_steps+1,  # maxval은 exclusive 이므로 +1
        dtype=tf.int32
        )
        # (2) 0~1로 정규화
        t = tf.cast(t_int, tf.float32) / tf.cast(kid_diffusion_steps, tf.float32)
        # (3) 네트워크 입력 형태로 reshape
        t = tf.reshape(t, [b, 1, 1, 1])
        
        nr,sr   = self.diffusion_schedule(t)
        x_noi   = sr * clear_n + nr * noise
        mask = self.mask_pred(hazy_n, training=False,)
        with tf.GradientTape() as tape:
            tape.watch(x_noi)
            pred = self.network([x_noi, hazy_n, t, context, mask], training=True)
            
            
            # #context가 적용이 제대로 되는지에 대한 디버깅 파트
            # pred_zero_context = self.network([x_noi, hazy_n, t, tf.zeros_like(context), mask], training=False)
            # tf.print("▶ train diff norm(context 적용되는가):", tf.norm(pred - pred_zero_context))
            
            #mask 적용된게 끝까지 잘 가는지에 대한 디버깅 파트
            zero_mask = tf.zeros_like(mask)
            zero_mask = tf.cast(zero_mask, dtype=mask.dtype)
            pred_zero_mask = self.network([x_noi, hazy_n, t, context, zero_mask], training=False)
            diff   = tf.norm(pred - pred_zero_mask)
            norm_r = tf.norm(pred)

            ratio = diff / (norm_r + 1e-8)
            tf.print("▶ diff norm:", diff, "|| pred_r norm:", norm_r, 
                    "▶ ratio(mask 적용되는가):", ratio)      # 0.05(5%) 이상으로 나온다면 mask가 예측에 눈에 띄게 기여하고 있는거임. 1% 미만이면 mask 영향이 거의 사라진 상태임
            
            epoch = getattr(self, 'current_epoch', 0)
            pred_x0 = (x_noi - nr * pred) / sr
            
            ## -> 이거 안 하는게 좋을거 같다. -=> 초반에 튀게 해줘야지 학습이 제대로 될거 같음
            # if epoch < 10:
            #     pred_x0 = tf.tanh(pred_x0 * 0.7)
            # else:
            #     pred_x0 = pred_x0
            
            tf.print("pred shape:", tf.shape(pred), "min/max:", tf.reduce_min(pred), "/", tf.reduce_max(pred))            
            
            #loss = 0.1 * self.loss_fn(pred, noise) + 0.9 * self.loss_fn(pred_x0, clear_n) # 1.0과 0.0이면 noise-only학습, 0.0과 1.0이면 x0-only 학습
            
            # epoch에 따라 w_noise 증감
            r = epoch / num_epochs
            w_noise = 0.7 * (1 - r) + 0.2 * r  # 학습 초반엔 noise 중심 → 후반엔 x0 중심
            w_x0    = 1 - w_noise
            loss = w_noise * self.loss_fn(pred, noise) \
                + w_x0    * self.loss_fn(pred_x0, clear_n)

            tf.print("pred_x0 mean/std:", tf.reduce_mean(pred_x0), "/", tf.math.reduce_std(pred_x0))
            tf.print("clear_n mean/std:", tf.reduce_mean(clear_n), "/", tf.math.reduce_std(clear_n))

        grads = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_weights))
        tf.print("max grad:", tf.reduce_max([tf.reduce_max(tf.abs(g)) for g in grads if g is not None]))
        return {'loss_train': loss}

    def test_step(self, data):
        hazy, clear, context = data
        hazy_n  = self.normalizer(hazy)
        clear_n = self.normalizer(clear)
        b       = tf.shape(clear_n)[0]
        noise   = tf.random.normal((b, img_siz, img_siz, 3))
       # (1) 1,2,…,T 중 하나를 균일 샘플링 (정수)
        t_int = tf.random.uniform(
        shape=(b,),
        minval=1,
        maxval=kid_diffusion_steps+1,  # maxval은 exclusive 이므로 +1
        dtype=tf.int32
        )
        # (2) 0~1로 정규화
        t = tf.cast(t_int, tf.float32) / tf.cast(kid_diffusion_steps, tf.float32)
        # (3) 네트워크 입력 형태로 reshape
        t = tf.reshape(t, [b, 1, 1, 1])
        
        nr,sr   = self.diffusion_schedule(t)
        x_noi   = sr * clear_n + nr * noise

        mask = self.mask_pred(hazy_n, training=False)
        pred = self.network([x_noi, hazy_n, t, context, mask], training=False)
        loss = self.loss_fn(noise, pred)
        return {'loss_test': loss}

    def dehaze(self, hazy_imgs, ctx, steps=kid_diffusion_steps):   
        hazy_n = self.normalizer(hazy_imgs)

        noise = tf.random.normal((tf.shape(hazy_n)[0], img_siz, img_siz, 3))
        next_x = noise
        # T_time = tf.fill([tf.shape(hazy_n)[0], 1, 1, 1], 1.0)
        # nr, sr = self.diffusion_schedule(T_time)
        # next_x = sr * next_x + nr * noise
        mask = self.mask_pred(hazy_imgs, training=False)
        for i in reversed(range(steps)):

            x = next_x
            t = tf.fill([tf.shape(x)[0], 1, 1, 1], i / steps)
            nr, sr = self.diffusion_schedule(t)

            # 1) 네트워크에서 예측된 노이즈
            pn = self.network([x, hazy_n, t, ctx, mask], training=False)

            # 2) 학습 시와 동일하게 tanh 스케일링 적용
            px0_raw = (x - nr * pn) / sr
            epoch = getattr(self, 'current_epoch', 0)
            # if epoch < 10:
            #     pred_x0 = tf.tanh(px0_raw * 0.7)
            #     # 3) (Optional) tanh 이후에만 클리핑
            #     px0 = tf.clip_by_value(pred_x0, -1.0, 1.0)
            # else:
            #     px0 = px0_raw
            px0 = px0_raw

            # 4) 디버깅용 출력 (분포 확인)
            tf.print("px0_raw min/max:", tf.reduce_min(px0_raw), "/", tf.reduce_max(px0_raw))   # tf.tanh 적용시키기 전
            tf.print("px0      min/max:", tf.reduce_min(px0),    "/", tf.reduce_max(px0))       # tf.tanh 적용시키고 난 후

            # 5) 다음 스텝으로 업데이트
            next_t = t - 1.0 / steps
            next_nr, next_sr = self.diffusion_schedule(next_t)
            next_x = next_sr * px0 + next_nr * pn
        return px0  # 최종 예측된 x0 반환
    
    
    def plot_images(self, epoch=None, logs=None, val_ds=None, num_rows=3, num_cols=6): #이 변수로 개수 조절
        # plot random generated images for visual evaluation of generation quality
        # 디렉토리 없으면 생성
        os.makedirs("/home/jang/DDIM_python/paper/samples", exist_ok=True)

        for hazy_b, clear_b, ctx_b in val_ds.take(1):
            dehazed_b = self.dehaze(hazy_b, ctx_b)  # [-1,1] 범위
            print("before clip x0 min/max:", dehazed_b.numpy().min(), dehazed_b.numpy().max())
            # dehazed_b = tf.clip_by_value(dehazed_b, -1.0, 1.0)
            print("after clip x0 min/max:", dehazed_b.numpy().min(), dehazed_b.numpy().max())
            dehazed_b = self.denormalize(dehazed_b).numpy()  #[0,1] 범위
            print("x0 min/max:", tf.reduce_min(dehazed_b).numpy(), "/", tf.reduce_max(dehazed_b).numpy())
            print("x0 mean/std:", tf.reduce_mean(dehazed_b).numpy(), "/", tf.math.reduce_std(dehazed_b).numpy())
            dehazed_b = tf.clip_by_value(dehazed_b, 0.0, 1.0)
            batch_size = hazy_b.shape[0]
            fig, axs = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))

            for i in range(batch_size):
                hazy = hazy_b[i].numpy()    # (img_siz, img_siz, 3), float32, 0~1
                clear = clear_b[i].numpy()  # (img_siz, img_siz, 3), float32, 0~1

                dehazed = dehazed_b[i]          # 이미 numpy 상태임

                axs[i, 0].imshow(hazy)
                axs[i, 0].set_title("Hazy")
                axs[i, 0].axis("off")

                axs[i, 1].imshow(dehazed)
                axs[i, 1].set_title("Dehazed (Output)")
                axs[i, 1].axis("off")

                axs[i, 2].imshow(clear)
                axs[i, 2].set_title("Clear (GT)")
                axs[i, 2].axis("off")

            plt.tight_layout()
            # 사진 저장
            plt.savefig(f"samples/generated_epoch_{epoch+1:03d}.png", dpi=200)
            plt.close(fig)



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
model.compile(
    optimizer=keras.optimizers.Adam(1e-4)       # 학습률을 보통 1e-4를 사용. 1e-5는 안정성 확보할때 사용
)
model.run_eagerly = False   # 디버깅할때는 True로 바꾸고

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "weights_epoch_{epoch:02d}.weights.h5"),
    save_weights_only=True,
    save_freq='epoch',
    verbose=1
)


for v in model.trainable_variables:
    if "hf_cross_attn" in v.name:
        print(f"✅ 학습 대상: {v.name}")
        
steps_per_epoch = 200 #len(hazy_files) // batch_siz
val_steps = kid_diffusion_steps #len(hazy_test_files)//batch_siz

print("steps_per_epoch : ", steps_per_epoch)
print("fit 이전")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,   #한 epoch 당 배치 개수
    validation_steps=val_steps,
    callbacks=[checkpoint_cb,
    keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, 
        logs: model.plot_images(epoch, logs, val_ds=val_ds)
    ),#이미지 생성
    EpochTracker()
]
)
print("fit 이후")
