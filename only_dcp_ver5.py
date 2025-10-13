## conda install matplotlib
## python -m pip install tensorflow-datasets
## python -m pip install opencv-python-headless -> import cv2
## python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -> import troch

import os
####dcp 추가 버전
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ.pop("TF_GPU_ALLOCATOR", None)   
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds #tfds 데이터셋 이용하는거
tf.config.optimizer.set_jit(False)

import keras
from keras import layers
#from keras import ops
##
#from keras import layers

import numpy as np
# import cv2
# import torch
from pathlib import Path
#import dcp
import dcp_tf
from keras.callbacks import CSVLogger
import json
#reside-b 전처리할때
import re
from collections import defaultdict
##################################
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
    
# #dataset_repetitions = 5
# dataset_repetitions = 1
num_epochs = 400  # train for at least 50 epochs for good results
image_size = 128


plot_diffusion_steps = 50 #생성시 샘플링 스텝

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# architecture
embedding_dims = 64 # widths[0]랑 같아야됨
embedding_max_frequency = 1000.0
widths = [64, 128, 256, 256]
block_depth = 2

# optimization
batch_size = 16 #64
VAL_BATCH_SIZE = 16
ema = 0.999
learning_rate = 1e-6
weight_decay = 1e-5

num_images = 4 #생성할 이미지 수
groups = 16; # groupNorm할때 몇으로 묶을건지

# ---- test-time sampling control ----
TEST_GENERATE_STEPS = 50   # test에서 generate()에 사용 할 역확산 스텝
TEST_NUM_IMAGES     = 4
#dataset의 쌍끼리 순서를 맞춤
############################################## RESIDE-6K########################################################
#─── ZIP에서 바로 읽는 데이터로더 (변수명/함수명 유지) ─────────────────────────

# base_dir = '/home/jang/DDIM_python/RESIDE-6K/RESIDE-6K/train'
# hazy_dir = os.path.join(base_dir, 'hazy')
# ref_dir  = os.path.join(base_dir, 'GT')

# base_dir_test = '/home/jang/DDIM_python/RESIDE-6K/RESIDE-6K/test'
# hazy_test_dir = os.path.join(base_dir_test, 'hazy')
# ref_test_dir  = os.path.join(base_dir_test, 'GT')

# # hazy 이미지 경로들
# image_paths = list(Path(hazy_dir).rglob("*.jpg"))
# exts     = ('.png', '.jpg', '.jpeg')

# hazy_files = sorted(str(p) for p in Path(hazy_dir).rglob('*') if p.suffix.lower() in exts)
# ref_files  = sorted(str(p) for p in Path(ref_dir).rglob('*')  if p.suffix.lower() in exts)

# hazy_test_files = sorted(str(p) for p in Path(hazy_test_dir).rglob('*') if p.suffix.lower() in exts)
# ref_test_files  = sorted(str(p) for p in Path(ref_test_dir).rglob('*')  if p.suffix.lower() in exts)

# assert len(hazy_files)==len(ref_files), "파일 개수 불일치"
# print("✔ 총 이미지 쌍:", len(hazy_files))

# def _read_image(path):  
#     img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
#     img.set_shape([None, None, 3])
#     return tf.image.convert_image_dtype(img, tf.float32)    # 여기서 /255.0 해줌

# def load_pair(h_path, r_path):  # dummy 세 번째 인자
#     hazy  = tf.image.resize(_read_image(h_path),  [image_size, image_size])
#     clear = tf.image.resize(_read_image(r_path), [image_size, image_size])
#     hazy  = tf.clip_by_value(hazy,  0.0, 1.0)
#     clear = tf.clip_by_value(clear, 0.0, 1.0)
#     return hazy, clear

# ds_train = tf.data.Dataset.from_tensor_slices((hazy_files, ref_files))
# ds_test  = tf.data.Dataset.from_tensor_slices((hazy_test_files, ref_test_files))

# total_count = len(hazy_files)
# print("total_count", total_count)

# train_dataset = (
#     ds_train
#     .shuffle(2048, reshuffle_each_iteration=True)
#     .map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
#     .repeat()
#     .batch(batch_size)
#     .prefetch(tf.data.AUTOTUNE)
# )

# val_dataset = (
#     ds_test
#     .shuffle(2048, reshuffle_each_iteration=False) # 한 번만 섞어서 고정
#     .map(load_pair, num_parallel_calls=1)
#     .batch(VAL_BATCH_SIZE)
#     .prefetch(tf.data.AUTOTUNE) 
# )

###################################RESIDE-b##############################################################

base_dir = '/home/jang/DDIM_python/RESIDE-b/train'
hazy_dir = os.path.join(base_dir, 'hazy')
ref_dir  = os.path.join(base_dir, 'clear')

base_dir_test = '/home/jang/DDIM_python/RESIDE-b/test'
hazy_test_dir = os.path.join(base_dir_test, 'hazy')
ref_test_dir  = os.path.join(base_dir_test, 'clear')

# hazy 이미지 경로들
image_paths = list(Path(hazy_dir).rglob("*.jpg"))
exts     = ('.png', '.jpg', '.jpeg')

def numkey(p) -> str:
    """파일명에서 '첫 숫자 블록' 추출 (str/Path 모두 허용)"""
    stem = Path(p).stem
    m = re.search(r'(\d+)', stem)
    return m.group(1) if m else None

def build_pairs(hazy_dir: str, gt_dir: str, expected_per_group: int | None = None):
    """
    hazy_dir에 있는 모든 hazy를 번호키로 묶고,
    같은 번호를 가진 GT 1장을 매칭해 리스트를 반환.
    - expected_per_group: 번호별 hazy 개수 기대치(예: train=35, test=1 또는 None)
    """
    hazy_paths = sorted([p for p in Path(hazy_dir).rglob('*') if p.suffix.lower() in exts])
    gt_paths   = sorted([p for p in Path(gt_dir).rglob('*')  if p.suffix.lower() in exts])

    # GT 인덱스(번호키 → GT Path)
    gt_index = {}
    for p in gt_paths:
        k = numkey(p)
        if k: gt_index[k] = p

    pairs_hz, pairs_gt = [], []
    unmatched = 0
    group_cnt = defaultdict(int)

    for hz in hazy_paths:
        k = numkey(hz)
        if not k or k not in gt_index:
            unmatched += 1
            continue
        pairs_hz.append(str(hz))
        pairs_gt.append(str(gt_index[k]))
        group_cnt[k] += 1

    print(f"✔ 매칭된 쌍: {len(pairs_hz)} | hazy 총 {len(hazy_paths)}, GT 총 {len(gt_paths)}, unmatched hazy {unmatched}")

    # (선택) 번호 그룹 크기 점검
    if expected_per_group is not None:
        bad = {k:v for k,v in group_cnt.items() if v != expected_per_group}
        if bad:
            sample = dict(list(bad.items())[:10])
            print(f" 번호별 hazy 개수 기대치({expected_per_group})와 다름 (일부): {sample}")
        else:
            print(f" 모든 번호 그룹이 {expected_per_group}장입니다.")

    return pairs_hz, pairs_gt

hazy_files, ref_files = build_pairs(hazy_dir, ref_dir, expected_per_group=35) 
hazy_test_files, ref_test_files = build_pairs(hazy_test_dir, ref_test_dir, expected_per_group=35)


print("✔ 최종 train_hazy 페어 길이:", len(hazy_files))
print("✔ 최종 train_ref 페어 길이:", len(ref_files))
   
def _read_image(path):  
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return tf.image.convert_image_dtype(img, tf.float32) 


def load_pair(h_path, r_path):  # dummy 세 번째 인자
    hazy  = tf.image.resize(_read_image(h_path),  [image_size, image_size])
    clear = tf.image.resize(_read_image(r_path), [image_size, image_size])
    hazy  = tf.clip_by_value(hazy,  0.0, 1.0)
    clear = tf.clip_by_value(clear, 0.0, 1.0)
    return hazy, clear


print(len(hazy_files), " , ", len(ref_files))
#assert len(hazy_files)== len(35*ref_files), "파일 개수 불일치"
print("✔ 총 이미지 쌍:", len(hazy_files))

# ——— train/test Dataset 정의 ———
ds_train = tf.data.Dataset.from_tensor_slices((hazy_files, ref_files))
ds_test  = tf.data.Dataset.from_tensor_slices((hazy_test_files, ref_test_files))

total_count = len(hazy_files)
print("total_count", total_count)


train_dataset = (
    ds_train
    .shuffle(2048, reshuffle_each_iteration=True)
    .map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    .repeat()
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    ds_test
    .shuffle(2048, reshuffle_each_iteration=False)  # 한 번만 섞어서 고정
    .map(load_pair, num_parallel_calls=1)
    .batch(VAL_BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE) 
)

################################################################################################################

train_steps = max(1, len(hazy_files)      // batch_size)
val_steps   = max(1, len(hazy_test_files) // VAL_BATCH_SIZE) 


"""
def build_mask_predictor():
    inp = keras.Input((image_size, image_size, 3), dtype=tf.float32)

    mask = layers.Lambda(
        lambda batch: tf.numpy_function(
            func=dcp.estimate_transmission_batch,
            inp=[batch], # 입력은 batch 텐서
            Tout=tf.float32   # 출력 타입은 float32 numpy 배열
        ),
        output_shape=(image_size, image_size, 1)   # 출력 형태 지정 (H, W, 1 채널 마스크)
    )(inp)
    mask = layers.Lambda(lambda x: tf.ensure_shape(x, [None, image_size, image_size, 1]))(mask)
    return keras.Model(inputs=inp, outputs=mask, name='mask_predictor')

"""

@keras.saving.register_keras_serializable()  #모델 정보를 json파일에  저장
def sinusoidal_embedding(x): #임베딩 정보를 리턴
    embedding_min_frequency = 1.0
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = tf.cast(2.0 * math.pi * frequencies, "float32")
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings

def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3] #x.shape = (batch_size, height, width, input_channels)
        if input_width == width: #채널 수 맞추기 같으면 그대로
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x) #1x1로 채널 수 맞추기

        G = min(groups, width)           # 채널보다 큰 그룹 방지
        if width % G != 0:               # 나눠떨어지도록 보정(예: 가장 가까운 약수로)
            for g in range(G, 0, -1):
                if width % g == 0:
                    G = g; break
                    
        # x = layers.GroupNormalization(groups=G, axis=-1, epsilon=1e-5)(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", activation="swish")(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual]) # 잔차 연결

        return x

    return apply


def DownBlock(width, block_depth): #채널 #2
    def apply(x):
        x, skips = x

        for _ in range(block_depth): #2번 수행
            x = ResidualBlock(width)(x)

            skips.append(x) #skips에 저장 나중에 concate에 사용

        x = layers.AveragePooling2D(pool_size=2)(x) # 풀링=> 크기1/2

        return x

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)#크기 2배
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()]) #여기서 skip 꺼내서 concate수행
            x = ResidualBlock(width)(x)

        return x
    return apply

def get_network(image_size, widths, block_depth):
    # 이미지의 가로/세로 크기, 채널 수 리스트,각 단계마다 residual block 몇 개 쓸지
    noisy_images = keras.Input(shape=(image_size, image_size, 7)) #노이즈가 섞인 RGB 이미지 (입력)
    noise_variances = keras.Input(shape=(1, 1, 1)) #이미지에 추가된 노이즈의 강도 (시간정보 t에 해당)

    #hazy_img =  keras.Input(shape=(image_size, image_size, 3))

    e = layers.Lambda(sinusoidal_embedding, output_shape=(1, 1, widths[0]))(noise_variances)
    #noise_variances를 sinusoidal embedding으로 변환 (32채널)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)
    #(1, 1, 32) → (image_size, image_size, 32)로 확장

    ###인코더###
    #x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)#e와 맞춰서 크기 채널수 맞춰줌 ############
    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)#e와 맞춰서 크기 채널수 맞춰줌 ###########
    x = layers.Concatenate()([x,e]) #임베딩 정보 더해줌+ mask 정보


    skips = []
    for width in widths[:-1]: #width [32,64,128,256]#마지막 제외
        x = DownBlock(width, block_depth)([x, skips]) #채널 수 넣어주며 대운블럭 수행
    ###중앙###
    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)
    ###디코더###
    for width in reversed(widths[:-1]):  #width [32,64,128,256]
        x = UpBlock(width, block_depth)([x, skips]) #apply(x,skips)
    ###출력###
    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)
    #채널 수 3 RGB // 처음에는 아무 정보 없으므로 → 가중치를 0으로 초기화해서, 모델 시작 시 출력이 0에 가까움

    #이때 x 에는 노이즈 정보가 담겨있음

    ###모델 생성###
    return keras.Model([noisy_images, noise_variances], x, name="residual_unet") # 입력, 출력, 이름

@keras.saving.register_keras_serializable()
class DiffusionModel(keras.Model):
    def __init__(self, image_size, widths, block_depth):
        super().__init__()
        # 입력 이미지를 역정규화하기 위한 레이어 (mean=0, std=1 형태로 변환) 정규화를 가져와서 내부에 역정규화 구현
        self.hazy_norm  = layers.Normalization()
        self.clear_norm = layers.Normalization()

        # U-Net 기반의 주 네트워크 구조 생성
        self.network = get_network(image_size, widths, block_depth)
        # EMA(지수이동평균) 네트워크: 학습이 완료된 안정된 파라미터를 추론 시 사용
        # U-net과 똑같은 모델을 하나 더만듦
        self.ema_network = keras.models.clone_model(self.network)
        self.ema_network.trainable = False#추가
        #dcp map
        #self.mask_pred = build_mask_predictor()
        self.mask_pred = dcp_tf.build_mask_predictor(image_size)
        
        self.mask_pred.trainable = False
        self._val_saved = False
        self._vis_cache = None   # (hazy_np, dehazed_np, gt_np) 저장
        
    def call(self, inputs, training=False):
        return self.network(inputs, training=training)


    #옵티마이저와 손실함수 설정
    def compile(self, **kwargs): #**kwargs 는 딕셔너리로 처리
        '''
        kwargs = {
        'optimizer': AdamW(...),
        'loss': MeanSquaredError()
        }
        '''
        # 여기다가 로스 추가해주면 됨 + metric에 가서 추적하기 위해 추가
        super().compile(**kwargs)# keras의 optimizer, loss를 그대로 사용
        # 노이즈 손실 추적: 예측된 노이즈가 실제 노이즈와 얼마나 가까운지 평균 손실 기록
        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        # 이미지 복원 성능 추적용 메트릭: 복원 이미지가 원본과 얼마나 가까운지 평균 손실 기록
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        # 이미지 생성 품질 평가를 위한 KID (Kernel Inception Distance)
        # self.kid = KID(name="kid")
        self.psnr_metric = keras.metrics.Mean(name="psnr")  # psnr
        self.ssim_metric = keras.metrics.Mean(name="ssim") #ssim
    @property #함수 메소드를 인자로 사용가능
    def metrics(self):
        # 모델이 추적하는 리스트 반환
        #return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]
        return [self.noise_loss_tracker, self.image_loss_tracker, self.psnr_metric, self.ssim_metric]

    def denormalize(self, images):
        # -1~1정규화된 이미지를 원래 스케일(픽셀값 0~1 범위)로 복원
        # 정규화 과정: normalized = (original - mean) / std
        # 역정규화: original = mean + normalized * std
        # 여기서 std = sqrt(variance)
        images = self.clear_norm.mean + images * self.clear_norm.variance ** 0.5 #역정규화
        #이렇게 복원된 픽셀 값들이 0~1 범위를 넘을 수 있기 때문에, 클리핑으로 안정성 확보를 합니다.
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times): #0~1사이 값 들어옴 range는 0부터 시작 => diffusion_times는 1부터 들어옴
        # diffusion times -> angles
        #t에 따른 신호(알파)와 노이즈(베타 = 1-알파)를 사인 코사인으로 구현
        #max_signal_rate와 min_signal_rate는 사전에 설정된 신호 비율의 최대/최소값입니다.
        #각 신호 비율에 대응하는 각도를 역삼각함수 arccos를 이용해 구합니다.
        #최대 신호 비율에 대응하는 작은 각도
        start_angle = tf.cast(tf.acos(max_signal_rate), "float32") #0.95
        #across(역코사인)는 각도,cast는 float32로 맞춤
        #최소 신호 비율에 대응하는 큰 각도
        end_angle = tf.cast(tf.acos(min_signal_rate), "float32") #0.02

        #선형으로 표현해서 t일 때 각도를 구함
        #t=0이면 diffusion_angles = start_angle,
        #t=1이면 diffusion_angles = end_angle이 되도록 선형적으로 값을 계산합니다.
        diffusion_angles = (end_angle - start_angle) *  diffusion_times + start_angle
        #d_angles는 diffusion_time과 비레함
        # 삼각함수로 알파와 베타, 원본 비율, 노이즈 비율을 구함  #1부터 점점 작게 들어오면
        #각이 점점 줄어듦
        #x좌표
        signal_rates = tf.cos(diffusion_angles) #점점 늘어남 ex) 0.02.. => 0.99
        #y좌표
        noise_rates = tf.sin(diffusion_angles) #점점 늘어남 ex)0.99=> 0.02
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    #노이즈를 예측 작업 => 입력된 이미지의 노이즈를 예측,이미지 복원, reverse_diffusion에서 사용
    def denoise(self, noisy_images, noise_rates, signal_rates,hazy_img, training, mask=None):
        # the exponential moving average weights are used at evaluation
        # 학습 중이면 self.network (현재 가중치 사용), 평가 시엔 self.ema_network (EMA 가중치 사용)
        if training:
            network = self.network
        else:
            network = self.ema_network
            #network = self.network
        
        if mask is None:
            with tf.device('/CPU:0'):
                mask = self.mask_pred(hazy_img)
        mask = tf.cast(mask, noisy_images.dtype)
        mask = tf.clip_by_value(mask, 0.0, 1.0)
        tf.debugging.check_numerics(noisy_images, "noisy_images NaN/Inf")
        tf.debugging.check_numerics(mask, "mask NaN/Inf")
        #print(tf.shape(mask).numpy())
        noisy_images1 = tf.concat([hazy_img, noisy_images, mask], axis=-1) # haze 이미지 concat GT에 노이즈 씌운거

        #헤이즈 이미지에 gt+노이즈를 concat // 그 후에 네트워크에서 차원 늘리고 임베딩 추가

        # 노이즈 예측: 네트워크는 noisy_images와 noise_rates**2를 입력받아 노이즈 성분을 예측함
        pred_noises = network([noisy_images1, noise_rates**2], training=training) #u net사용
        # 예측된 노이즈를 사용해 원래 이미지를 복원 (추정)
        #print(noisy_images.shape)  # 예: (32, 64, 64, 3)
        #print(noise_rates.shape)   # 보통 (32, 1, 1, 1) 이거나 (32,) 같은 shape
        #print(pred_noises.shape)   # 보통 (32, 64, 64, 3)
        #pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates ##################
        pred_images = tf.clip_by_value(pred_images, -3.0, 3.0)
        return pred_noises, pred_images

#####sampling #####
    #sampling 할 때만 사용 test_step도 사용
    #def generate 에서 사용
    #denoise를 몇 번 수행할지 t=>0  다시 0=> t-1, t-1=>0... 반복,
    def reverse_diffusion(self, initial_noise, diffusion_steps,hazy_img):
        #완전한 가우시안 노이즈,몇 단계로 역확산을 수행할지 설정
        #intial_noise.shape=(num_images, H, W, C)
        # reverse diffusion = sampling
        #num_images = initial_noise.shape[0] # 배치 크기 (생성할 이미지 개수)
        step_size = 1.0 / diffusion_steps #diffusion_step이 t 디퓨전 수행
        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        """next_noisy_images = initial_noise #랜덤 가우시안 노이즈에서 시작
        #다음 노이즈 상태t-1 이미지 만들기, 예측한 x0이미지에 이전 노이즈를 살짝 섞음
        """
        next_noisy_images = (initial_noise)
        #next_noisy_images = tf.concat([hazy_img, initial_noise], axis=-1) # haze 이미지 concat GT에 노이즈 씌운거

        """
        img_np = tf.clip_by_value(next_noisy_images[0], 0.0, 1.0).numpy()
        os.makedirs("samples", exist_ok=True)
        plt.imshow(img_np)
        plt.axis("off")
        plt.title("Initial noisy input (step -1)")
        plt.savefig("diffusion_steps_vis/step_-1.png")
        plt.close()
        """
    
        with tf.device('/CPU:0'):
            mask = self.mask_pred(hazy_img) 
        
        for step in range(diffusion_steps): #0~20  1- 현재스텝/ 전체스텝
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            #현재 시점의 t 계산, 1에서 시작해서 점점 줄어듦
            b = tf.shape(hazy_img)[0]  # 현재 배치 크기
            diffusion_times = tf.ones((b, 1, 1, 1)) - step * step_size
            #1==>0
            # -step*(1.0/diffusion_steps) 0.0~1.0 사이값으로 만들어서 스케쥴에 들어감
            #현재시점의 노이즈 비율 계산
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)#1.0~0.0  1 0.95 0.90 0.85
            #0.9=>0,3    #0.3=>0.9

            #노이즈 비율을 가지고 이미지와 노이즈 예측분리
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates,signal_rates,hazy_img,training=False, mask=mask
            )
            # network used in eval mode
            #다음 스텝 준비
            # remix the predicted components using the next signal and noise rates
            #다음 t시점
            next_diffusion_times = diffusion_times - step_size
            #다음 스텝 노이즈 비율
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )

            #다음 노이즈 상태t-1 이미지 만들기, 예측한 x0이미지에 이전 노이즈를 살짝 섞음
            next_noisy_images = (
               # next_signal_rates * pred_images + next_noise_rates * pred_noises
               next_signal_rates * pred_images + next_noise_rates * pred_noises
            )

            #next_noisy_images = tf.concat([hazy_img, next_noisy_images], axis=-1)

            # this new noisy image will be used in the next step
            #  시각화 (ex: 첫 번째 이미지만 저장)
            """
            img_np = tf.clip_by_value(pred_images[0], 0.0, 1.0).numpy()
            plt.imshow(img_np)
            plt.axis("off")
            plt.title(f"Step {step}")
            plt.savefig(f"diffusion_steps_vis/step_{step:03d}.png")
            plt.close()
            """
        return pred_images #생성된 이미지

    #generate는 sampling 할 때 만 사용 test_step도 사용
    #위에 reverse diffusion을 사용해서 생성

    def generate(self, num_images, diffusion_steps, hazy_img):
        # noise -> images -> denormalized images
        
        if num_images is None:
            num_images = tf.shape(hazy_img)[0]

        # 랜던 노이즈 설정 ##우리는 랜덤 노이즈가 아니라 haze이미지에 랜덤 노이즈 씌워서 시작
        initial_noise = keras.random.normal(
            shape=(num_images, image_size, image_size, 3)
        )

        #출력은 정규화된 이미지 (픽셀 값이 평균 0, 표준편차 1 범위)
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps,hazy_img[:num_images])
        #정규화된 값을 원래 스케일(0~1 범위)로 되돌립니다.
        generated_images = self.denormalize(generated_images)
        return generated_images
#####sampling #####여기까지

#데이터셋 (train_haze, train_gt), (val_input, val_target) 형태로 저장
#train_haze만 가져옴

    def train_step(self, images):#한 배치의 images를 입력으로 받음 0~1상태
        # normalize images to have standard deviation of 1, like the noises
        hazy_images, clear_images = images
        b = tf.shape(hazy_images)[0]
        #이미지 정규화 평균 0, 표준편차 1
        images = self.hazy_norm(hazy_images, training=True)
        images1 = self.clear_norm(clear_images, training=True)


        #동일한 크기의 가우시안 노이즈 샘플을 생성
        noises = tf.random.normal((b, image_size, image_size, 3))  # 수정
        diffusion_times = tf.random.uniform((b, 1, 1, 1),0.0, 1.0)  # 수정


        #t시점의 노이즈 와 원본 비율 계산
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

        # mix the images with noises accordingly
        #t시점의 gt이미지에 노이즈 낀 이미지 생성
        noisy_images = signal_rates * images1 + noise_rates * noises
        #noisy_images = signal_rates * concat_images + noise_rates * noises

        #print("hazy_norm:", images.shape)
        #print("clear_norm:", images1.shape)
        #concat_images = tf.concat([images, noisy_images], axis=-1) # haze 이미지 concat GT에 노이즈 씌운거
        #print("concat_images:", concat_images.shape)
        
        with tf.device('/CPU:0'):
            mask = self.mask_pred(hazy_images) 
        
        #예측과 손실 계산
        with tf.GradientTape() as tape: #tensorflow의 자동 미분 기능 사용 문법
            # train the network to separate noisy images to their components
            #이미지의 섞여있는 노이즈 예측, 노이즈 제거한 이미지x0
             #trans map
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates,hazy_img=hazy_images,training=True, mask=mask
                #concat_images, noise_rates, signal_rates,images,training=True
            )
            #실제 노이즈와 예측한 노이즈의 차이 계산
            noise_loss = self.loss(noises, pred_noises)  # used for training
            #학습에는 사용되지않지만 이미지의 차이 정도를 보기 위함
             # ★ 변경: 지표/모니터링은 픽셀 도메인(0~1)에서 계산
            pred_images_px = self.denormalize(pred_images)          # [-?,?] -> [0,1]
            gt_px          = tf.clip_by_value(clear_images, 0.0, 1.0)  # GT 픽셀

            # 모니터링용 이미지 손실(픽셀 기준)
            image_loss = self.loss(gt_px, pred_images_px)
            total_loss = noise_loss
            #####
            # 4. SSIM 손실 (clean 이미지와 예측 비교)
            ssim_scores = tf.image.ssim(gt_px, pred_images_px, max_val=1.0)
            ssim_loss = tf.reduce_mean(1.0 - ssim_scores)

            psnr_scores = tf.image.psnr(gt_px, pred_images_px, max_val=1.0)
            psnr_loss = tf.reduce_mean(1.0 - psnr_scores)

            # 5. 전체 손실 (SSIM 반영)
            total_loss = noise_loss   # ← alpha=0.1은 조절 가능

        # 6. Backprop
        var_list = self.network.trainable_weights
        grads = tape.gradient(total_loss, var_list)
        grads_vars = [(g, v) for g, v in zip(grads, var_list) if g is not None]
        self.optimizer.apply_gradients(grads_vars)

        #역전파로 파라미터 업데이트(noise_loss기준)
        """
        #gradients 계산
        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        #gradients를 통해 가중치 갱신
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        """
        #손실을 Keras 메트릭 시스템에 저장(모니터링 위함), 로스값 업데이트

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)
        psnr = tf.image.psnr(gt_px,  pred_images_px, max_val=1.0)
        ssim = tf.image.ssim(gt_px,  pred_images_px, max_val=1.0)
        self.psnr_metric.update_state(psnr)
        self.ssim_metric.update_state(ssim)
        # EMA 네트워크 업데이트
        #학습 중에는 네트워크 가중치가 요동치므로 안정적인 추론용 파라미터를 유지하기 위해 EMA 버전을 따로 관리

        # ---- EMA는 N스텝마다 ----
        for w, ew in zip(self.network.weights, self.ema_network.weights):
            ew.assign(ema * ew + (1.0 - ema) * w)

        # KID is not measured during the training phase for computational efficiency
        # 추적 중인 메트릭 리스트 반환 noise_loss_tracker, image_loss_tracker,KID제외(계산 비용이 큼)
        #반환값은 딕셔너리 형태:
        """{
        "n_loss": 평균 노이즈 손실,
        "i_loss": 평균 이미지 손실

        """
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images):
        
        # ---- 배치 언팩 ----
        hazy_img, clear_images = images                      # (B,H,W,3)
        b = tf.shape(hazy_img)[0]

        # ---- 완전 노이즈에서 역확산으로 생성(검증은 이것만) ----
        # 배치에서 TEST_NUM_IMAGES개만 생성(속도/VRAM 절약)
        num_eval = tf.minimum(tf.constant(TEST_NUM_IMAGES, tf.int32), b)
        tf.debugging.assert_positive(num_eval, message="num_eval==0 (검증 배치가 비어있음)")
        idx = tf.range(num_eval)                 # ← 텐서 인덱스 안전
        hazy_sel = tf.gather(hazy_img, idx)
        
        # generate()는 EMA 네트워크 사용(training=False) + denormalize까지 끝난 [0,1] 반환
        gen_imgs = self.generate(
            num_images=None,     
            diffusion_steps=TEST_GENERATE_STEPS,
            hazy_img=hazy_sel
        )

        #손실 계산: 노이즈와 이미지 로스
        # noise_loss = self.loss(noises, pred_noises)
        # ---- 메트릭 계산 ([0,1] 픽셀 도메인) ----
        k     = tf.shape(gen_imgs)[0]
        tf.debugging.assert_positive(k, message="gen_imgs==0 (생성 결과가 없음)")
        gt_idx = tf.range(k)
        gt_px = tf.clip_by_value(tf.gather(clear_images, gt_idx), 0.0, 1.0)

        image_loss = self.loss(gt_px, gen_imgs)
        psnr       = tf.image.psnr(gt_px,  gen_imgs, max_val=1.0)
        ssim       = tf.image.ssim(gt_px,  gen_imgs, max_val=1.0)

        self.image_loss_tracker.update_state(image_loss)
        self.psnr_metric.update_state(psnr)
        self.ssim_metric.update_state(ssim)
        

        # ---- 결과 리턴 ----
        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, epoch=None, logs=None,num_images= num_images,images=val_dataset):
       # hazy_images, clean_images = images
        for hazy_images, clean_images in images.take(1):
            hazy_images = hazy_images[:num_images]
            clean_images = clean_images[:num_images]

            generated_images = self.generate(
                num_images=num_images,
                diffusion_steps=plot_diffusion_steps,
                hazy_img=hazy_images,
            )

            plt.figure(figsize=(12, num_images*3))
            for i in range(num_images):
                plt.subplot(num_images , 3, i * 3 + 1)
                plt.imshow(hazy_images[i])
                plt.title("Hazy")
                plt.axis("off")

                # generated
                plt.subplot(num_images , 3, i * 3 + 2)
                plt.imshow(generated_images[i])
                plt.title("Dehazed")
                plt.axis("off")

                # clean
                plt.subplot(num_images , 3, i * 3 + 3)
                plt.imshow(clean_images[i])
                plt.title("GT Clean")
                plt.axis("off")
            plt.tight_layout()
            os.makedirs("samples", exist_ok=True)
            plt.savefig(f"samples/generated_epoch_{epoch}.png")
            plt.close()
            break

def _stack_triplet_grid(hazy, dehazed, gt):  # [N,H,W,3] in [0,1]
    # 가로로 붙여서 [N, H, 3W, 3]
    triplet = tf.concat([hazy, dehazed, gt], axis=2)
    # 세로로 이어붙여 [N*H, 3W, 3]
    shape = tf.shape(triplet)
    n, h, w3 = shape[0], shape[1], shape[2]
    grid = tf.reshape(triplet, [n * h, w3, 3])
    grid = tf.image.convert_image_dtype(grid, dtype=tf.uint8)
    return grid

def _save_grid_png(tensor, path_str):
    # 상위 폴더 생성/정규식 처리 모두 제거
    with tf.device('/CPU:0'):
        png = tf.image.encode_png(tensor)
        tf.io.write_file(path_str, png)  
        
if __name__ == "__main__":


    # create and compile the model
    model = DiffusionModel(image_size, widths, block_depth)
    # below tensorflow 2.9:
    # pip install tensorflow_addons
    # import tensorflow_addons as tfa
    # optimizer=tfa.optimizers.AdamW
    #모델을 학습 준비 상태로 만듦
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        ),
        loss=keras.losses.mean_absolute_error,
        jit_compile=False,
    )
    #model.run_eagerly = True

    dummy_noisy7 = tf.zeros((1, image_size, image_size, 7), dtype=tf.float32)
    dummy_t      = tf.zeros((1, 1, 1, 1), dtype=tf.float32)
    _ = model([dummy_noisy7, dummy_t], training=False)
    _ = model.ema_network([dummy_noisy7, dummy_t], training=False)  # EMA build
    model.ema_network.set_weights(model.network.get_weights())      # 가중치 동기화

    # save the best model based on the validation KID metric
    #model.build(input_shape=[(1, image_size, image_size, 3), (1, image_size, image_size, 1), (1, image_size, image_size, 3) ])
    #학습 중 가장 validation KID 값이 낮을 때 모델 가중치 저장
    checkpoint_path = "./checkpoints/diffusion_model.weights.h5"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor="val_i_loss",
        mode="min",
        save_best_only=True,
    )
    
    class SaveBothWeights(keras.callbacks.Callback):
        def __init__(self, dirpath="./checkpoints", monitor="val_i_loss", mode="min"):
            super().__init__()
            os.makedirs(dirpath, exist_ok=True)
            self.dirpath = dirpath
            self.monitor = monitor
            self.mode = mode
            self.best = np.inf if mode == "min" else -np.inf

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            # 1) 매 에폭 스냅샷(온라인/EMA)
            online_path = os.path.join(self.dirpath, f"epoch_{epoch+1:03d}.online.weights.h5")
            ema_path    = os.path.join(self.dirpath, f"epoch_{epoch+1:03d}.ema.weights.h5")
            self.model.save_weights(online_path)
            self.model.ema_network.save_weights(ema_path)

            # 2) best 갱신 시 별도 복사(온라인/EMA)
            val = logs.get(self.monitor)
            if val is not None:
                is_better = (val < self.best) if self.mode == "min" else (val > self.best)
                if is_better:
                    self.best = val
                    best_online = os.path.join(self.dirpath, "best.online.weights.h5")
                    best_ema    = os.path.join(self.dirpath, "best.ema.weights.h5")
                    self.model.save_weights(best_online)
                    self.model.ema_network.save_weights(best_ema)
                    
                    
    save_both = SaveBothWeights(dirpath="./checkpoints", monitor="val_i_loss", mode="min")   
             
    class SaveSamplesOnEpochEnd(keras.callbacks.Callback):
        def __init__(self, val_ds, num_images=2, steps=20):
            super().__init__()
            self.val_ds = val_ds
            self.num_images = num_images
            self.steps = steps
            os.makedirs("samples", exist_ok=True)

        def on_epoch_end(self, epoch, logs=None):
            for hazy, gt in self.val_ds.take(1):
                hazy = hazy[:self.num_images]
                gen  = self.model.generate(
                    num_images=None, diffusion_steps=self.steps, hazy_img=hazy
                )
                k = int(gen.shape[0])         # 텐서 → int
                grid = _stack_triplet_grid(hazy[:k], gen[:k], gt[:k])
                png  = tf.image.encode_png(grid)
                path = f"samples/val_epoch_{epoch+1:03d}.png"
                tf.io.write_file(path, png)
                break     

    class ValImageHook(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.model.current_epoch = int(epoch) + 1
            self.model._val_saved = False

        def on_test_begin(self, logs=None):
            # 샘플 디렉토리 보장만 하면 됩니다.
            self.model._save_dir = os.path.abspath("samples")
            tf.io.gfile.makedirs(self.model._save_dir)
            self.model._val_saved = False

        def on_test_end(self, logs=None):
            pass 


    #정규화를 위한 평균/분산 학습
    #Normalization() 레이어는 데이터를 (평균 0, 표준편차 1)로 표준화합니다.
    #adapt()는 훈련 데이터를 기준으로 평균/분산을 계산합니다.
    # calculate mean and variance of training dataset for normalization
    #model.normalizer.adapt(train_dataset)

    # hazy(x)만으로 평균/분산 추정 (유한 개수만 사용)
    normalizer_h_ds = train_dataset.take(train_steps).map(lambda x,y: x)
    normalizer_c_ds = train_dataset.take(train_steps).map(lambda x,y: y)
    model.hazy_norm.adapt(normalizer_h_ds)
    model.clear_norm.adapt(normalizer_c_ds)


    #hazy 만 뽑아서 정규화
    #hazy_only = train_dataset.map(lambda hazy, gt: hazy)
    #model.normalizer.adapt(hazy_only)
    #gt_only = train_dataset.map(lambda hazy, gt: gt)
    #model.normalizer.adapt(gt_only)

    # for hazy_img, gt_img in train_dataset.take(1):
    #     print(hazy_img.shape)  # (batch_size, 64, 64, 3)
    #     print(gt_img.shape)    # (batch_size, 64, 64, 3)
    #     print("hazy_b dtype:", hazy_img.dtype)        # tf.float32 인지 확인
    #     print("gt_img dtype:", gt_img.dtype)

    #     # 추가 완료
    #     print("hazy_b min/max:", tf.reduce_min(hazy_img).numpy(), tf.reduce_max(hazy_img).numpy())
    #     print("gt_img min/max:", tf.reduce_min(gt_img).numpy(), tf.reduce_max(gt_img).numpy())
    #     break

    # ---------------- 콘솔 프린터 ----------------
    def _fmt(v, nd=4, default='-'):
        try: return f"{float(v):.{nd}f}"
        except: return default

    def _pick(d, *keys):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    printer = keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: print(
            f"[E{epoch+1:03d}] "
            f"n_loss={_fmt(_pick(logs, 'n_loss','loss'),5)}  "
            f"i_loss={_fmt(_pick(logs, 'i_loss'),5)}  "
            f"psnr={_fmt(_pick(logs, 'psnr'),3)}  "
            f"ssim={_fmt(_pick(logs, 'ssim'),4)}  "
            # f"val_n_loss={_fmt(_pick(logs, 'val_n_loss','val_loss'),5)}  "
            f"val_i_loss={_fmt(_pick(logs, 'val_i_loss'),5)}  "
            f"val_psnr={_fmt(_pick(logs, 'val_psnr'),3)}  "
            f"val_ssim={_fmt(_pick(logs, 'val_ssim'),4)}"
        )
    )

    # ---------------- 학습 재개 스위치 ----------------
    RESUME = True
    
    ckpt_dir = "./tf_ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(0),
        optimizer=model.optimizer,
        online=model,              # online 네트워크(서브클래스 모델)
        ema=model.ema_network,     # EMA 네트워크
    )
    
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)

    if RESUME and manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        print(f"✔ 학습 재개: {manager.latest_checkpoint} 에서 복원됨")
    else:
        print("▶ 처음부터 학습 시작")

    class TFCheckpointSaver(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            ckpt.step.assign_add(1)
            manager.save(checkpoint_number=int(ckpt.step.numpy()))
            print(f" TF 체크포인트 저장 완료: {int(ckpt.step.numpy())}")


    # ---------------- 로그 저장 ----------------
    os.makedirs("logs", exist_ok=True)

    csv_logger = CSVLogger(
        filename="logs/training_log.csv", 
        separator=",", 
        append=True   # True면 이어쓰기, False면 새로쓰기
    ) 
    
    class MetricsLogger(keras.callbacks.Callback):
        def __init__(self, filepath="logs/metrics.json"):
            self.filepath = filepath
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            record = {"epoch": epoch+1}
            record.update({k: float(v) for k,v in logs.items() if v is not None})

            with open(self.filepath, "a") as f:
                f.write(json.dumps(record) + "\n")
    
    #모델 학습
    # run training and plot generated images periodically
    history = model.fit(
        train_dataset,
        epochs=num_epochs,                # 예: 100
        steps_per_epoch=train_steps,      # n_train // 64
        validation_data=val_dataset,
        validation_steps=val_steps // 5,   # 꼭 명시
        callbacks=[
            # keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images),
            ValImageHook(),
            checkpoint_callback,
            printer,
            save_both,
            TFCheckpointSaver(),
            csv_logger,
            MetricsLogger(),
            SaveSamplesOnEpochEnd(val_dataset, num_images=TEST_NUM_IMAGES,
                        steps=TEST_GENERATE_STEPS),
        ],
    )

    # ---------------- 학습 곡선 저장 ----------------  
    # 시각화
    plt.figure(figsize=(12, 6))

    # 기록된 메트릭 시각화
    for key in history.history:
        if key in ['n_loss', 'i_loss', 'psnr', 'ssim', #'val_n_loss',
                   'val_i_loss', 'val_psnr', 'val_ssim']:
            plt.plot(history.history[key], label=key)

    plt.xlabel("Epochs")
    plt.ylabel("Metric")
    plt.title("Training & Validation Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 저장
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/loss_curve.png", dpi=150)
    plt.close()
    
    # ---------------- 베스트 가중치 로드(추론용) ----------------
    #가장 좋은 가중치 로드
    model.load_weights(checkpoint_path)
    #이미지 저장
    model.plot_images()