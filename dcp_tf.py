import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Dark channel using TF ---
def dark_channel_tf(image, patch_size=15):
    # image: [B,H,W,3], float32 in [0,1]
    min_rgb = tf.reduce_min(image, axis=-1, keepdims=True)  # [B,H,W,1]
    # min-filter using max_pool on negated
    k = patch_size
    pad = (k - 1) // 2
    neg = -min_rgb
    neg_padded = tf.pad(neg, [[0,0],[pad,pad],[pad,pad],[0,0]], mode='REFLECT')
    local_neg_max = tf.nn.max_pool2d(neg_padded, ksize=k, strides=1, padding='VALID')
    local_min = -local_neg_max
    return local_min  # [B,H,W,1]

# --- Atmospheric light estimation in TF ---
def atm_light_tf(image, dark, top_percent=0.001):
    # image: [B,H,W,3], dark: [B,H,W,1], both float32
    b = tf.shape(image)[0]
    h = tf.shape(image)[1]
    w = tf.shape(image)[2]
    hw = h * w
    # k = max(floor(HW * top_percent), 1)
    k = tf.maximum(1, tf.cast(tf.math.floor(tf.cast(hw, tf.float32) * top_percent), tf.int32))

    flat_dark = tf.reshape(dark, [b, -1])            # [B, H*W]
    flat_img  = tf.reshape(image, [b, -1, 3])        # [B, H*W, 3]

    # get indices of top k brightest in dark channel
    # tf.math.top_k returns values and indices along last dim
    values, indices = tf.math.top_k(flat_dark, k=k, sorted=False)  # [B,k]
    # gather corresponding pixels per batch
    # batch_dims=1 lets us gather per-batch
    top_pixels = tf.gather(flat_img, indices, axis=1, batch_dims=1)  # [B, k, 3]
    A = tf.reduce_mean(top_pixels, axis=1, keepdims=True)  # [B,1,3]
    A = tf.reshape(A, [b, 1, 1, 3])                        # [B,1,1,3]
    return A

# --- Transmission estimate (normalized by A) ---
def transmission_estimate_tf(image, A, patch_size=15, omega=0.95):
    # image: [B,H,W,3] in [0,1], A: [B,1,1,3]
    A_safe = tf.maximum(A, 1e-6)
    norm = image / A_safe                       # broadcast: [B,H,W,3]
    dark_norm = tf.reduce_min(norm, axis=-1, keepdims=True)  # [B,H,W,1]
    t = 1.0 - omega * dark_norm
    t = tf.clip_by_value(t, 0.0, 1.0)
    return t  # [B,H,W,1]

# --- Guided filter using box/mean filters (vectorized) ---
def box_filter(x, r):
    # mean filter with window size r x r using avg_pool2d
    # x: [B,H,W,C]
    k = r
    # avg_pool2d expects NHWC and float
    return tf.nn.avg_pool2d(x, ksize=k, strides=1, padding='SAME')

def guided_filter_tf(I, p, r=60, eps=1e-4):
    # I: guidance image [B,H,W,1] (grayscale, float32 0..1)
    # p: filtering input [B,H,W,1]
    mean_I = box_filter(I, r)          # [B,H,W,1]
    mean_p = box_filter(p, r)
    mean_Ip = box_filter(I * p, r)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = box_filter(I * I, r)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)

    q = mean_a * I + mean_b
    return q

# --- Transmission refine (uses grayscale guidance) ---
def transmission_refine_tf(image, et, r=60, eps=1e-4):
    # image: [B,H,W,3] uint8 or float32; et: [B,H,W,1] float32
    # Convert image to grayscale normalized 0..1
    # If image is uint8 0..255, we expect caller to pass normalized floats.
    gray = tf.image.rgb_to_grayscale(image)  # [B,H,W,1]
    # ensure float32 in 0..1
    gray = tf.image.convert_image_dtype(gray, dtype=tf.float32)
    et = tf.cast(et, tf.float32)
    t = guided_filter_tf(gray, et, r=r, eps=eps)
    t = tf.clip_by_value(t, 0.0, 1.0)
    return t

# --- Full pipeline: estimate transmission for a batch ---
def estimate_transmission_batch_tf(batch_image, patch_size=15, r=60, omega=0.95):
    # batch_image: [B,H,W,3], expected float32 in [0,1]
    image = tf.clip_by_value(batch_image, 0.0, 1.0)
    dark = dark_channel_tf(image, patch_size=patch_size)            # [B,H,W,1]
    A = atm_light_tf(image, dark, top_percent=0.001)                # [B,1,1,3]
    te = transmission_estimate_tf(image, A, patch_size=patch_size, omega=omega)  # [B,H,W,1]
    t = transmission_refine_tf(batch_image, te, r=r, eps=1e-4)      # [B,H,W,1]
    return t

# --- Keras Layer / Model builder using TF ops (no tf.numpy_function) ---
class MaskPredictorLayer(keras.layers.Layer):
    def __init__(self, image_size, patch_size=15, r=60, omega=0.95, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.r = r
        self.omega = omega

    def call(self, inputs):
        # inputs assumed [B,H,W,3] float32 in [0,1]
        masks = estimate_transmission_batch_tf(inputs,
                                               patch_size=self.patch_size,
                                               r=self.r,
                                               omega=self.omega)
        # Ensure shape [B,H,W,1]
        masks = tf.ensure_shape(masks, [None, self.image_size, self.image_size, 1])
        return masks

def build_mask_predictor(image_size):
    inp = keras.Input((image_size, image_size, 3), dtype=tf.float32)
    # If your input values are 0..255, add a normalization layer:
    # x = layers.Lambda(lambda x: x / 255.0)(inp)
    x = inp  # assume caller will feed 0..1 floats
    mask = MaskPredictorLayer(image_size)(x)
    return keras.Model(inputs=inp, outputs=mask, name='mask_predictor')