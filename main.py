import logging
import io
import numpy as np
from pathlib import Path

import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

import tensorflow as tf
from tensorflow import keras

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Custom layers
class ReduceMeanLayer(keras.layers.Layer):
    def call(self, x):
        return tf.reduce_mean(x, axis=-1, keepdims=True)
    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (1,)

class ReduceMaxLayer(keras.layers.Layer):
    def call(self, x):
        return tf.reduce_max(x, axis=-1, keepdims=True)
    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (1,)

class ScaledDotProductLayer(keras.layers.Layer):
    def __init__(self, scale, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
    def call(self, inputs):
        q, k = inputs
        return tf.matmul(q, k, transpose_b=True) / self.scale
    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config

class MatMulLayer(keras.layers.Layer):
    def call(self, inputs):
        a, b = inputs
        return tf.matmul(a, b)

CUSTOM_OBJECTS = {
    "ReduceMeanLayer":       ReduceMeanLayer,
    "ReduceMaxLayer":        ReduceMaxLayer,
    "ScaledDotProductLayer": ScaledDotProductLayer,
    "MatMulLayer":           MatMulLayer,
}

# ── Конфигурация
MODEL_PATH      = Path("best_model_attention.keras")
FFT_PARAMS_PATH = Path("normalization_params.txt")
IMG_SIZE        = 224
THRESHOLD       = 0.5

# ── Загрузка модели при старте
logger.info("Загружаем модель: %s", MODEL_PATH)
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Файл модели не найден: {MODEL_PATH}. "
    )
model = keras.models.load_model(str(MODEL_PATH), custom_objects=CUSTOM_OBJECTS)
logger.info("Модель загружена успешно")

# ── Параметры нормализации FFT ────────────────────────────────────────────────
FFT_GLOBAL_MIN = None
FFT_GLOBAL_MAX = None

if FFT_PARAMS_PATH.exists():
    params = {}
    for line in FFT_PARAMS_PATH.read_text().splitlines():
        k, v = line.split("=")
        params[k.strip()] = float(v.strip())
    FFT_GLOBAL_MIN = params.get("global_min")
    FFT_GLOBAL_MAX = params.get("global_max")
    logger.info("FFT-параметры нормализации загружены: min=%.4f max=%.4f",
                FFT_GLOBAL_MIN, FFT_GLOBAL_MAX)
else:
    logger.warning(
        "normalization_params.txt не найден — FFT будет нормализоваться "
        "локально по каждому изображению)."
    )

# ── FastAPI app
app = FastAPI(
    title="Deepfake Detector API",
    description=(
        "Dual-Stream + Attention модель для детекции дипфейков. "
    ),
    version="2.0.0",
)

class PredictionResponse(BaseModel):
    label:            str
    fake_probability: float
    real_probability: float
    threshold:        float

# ── FFT-преобразование
def compute_fft_array(gray_img: np.ndarray) -> np.ndarray:
    img = cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    fft_shifted = np.fft.fftshift(np.fft.fft2(img))
    magnitude   = np.log1p(np.abs(fft_shifted))

    if FFT_GLOBAL_MIN is not None and FFT_GLOBAL_MAX is not None:
        mag_norm = (magnitude - FFT_GLOBAL_MIN) / (FFT_GLOBAL_MAX - FFT_GLOBAL_MIN)
    else:
        mag_min, mag_max = magnitude.min(), magnitude.max()
        mag_norm = (magnitude - mag_min) / (mag_max - mag_min + 1e-8)

    mag_norm = np.clip(mag_norm, 0.0, 1.0)
    return mag_norm[..., np.newaxis].astype(np.float32)

def decode_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Не удалось декодировать изображение")
    return img

# ── Эндпоинты ─────────────────────────────────────────────────────────────────
@app.get("/health", summary="Проверка работоспособности")
def health():
    return {"status": "OK"}


@app.post("/predict", response_model=PredictionResponse,
          summary="Определить: реальное лицо или дипфейк")
async def predict(
    image: UploadFile = File(..., description="Фото лица (jpg / png)"),
):
    """
    Принимает **одно** RGB-изображение лица.

    Сервис автоматически:
    1. Изменяет размер до 224x224
    2. Вычисляет FFT-спектр (точно как при обучении модели)
    3. Прогоняет оба потока через Dual-Stream + Attention модель

    Возвращает метку real / fake и вероятности.
    """
    logger.info("Получен запрос /predict  file=%s", image.filename)

    raw_bytes = await image.read()
    bgr_img   = decode_image(raw_bytes)

    rgb_img  = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_img  = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))
    rgb_arr  = rgb_img.astype(np.float32) / 255.0

    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    fft_arr  = compute_fft_array(gray_img)

    rgb_batch = rgb_arr[np.newaxis, ...]
    fft_batch = fft_arr[np.newaxis, ...]

    fake_prob = float(model.predict([rgb_batch, fft_batch], verbose=0)[0][0])
    real_prob = 1.0 - fake_prob
    label     = "fake" if fake_prob >= THRESHOLD else "real"

    logger.info("Результат: label=%s  fake_prob=%.4f", label, fake_prob)

    return PredictionResponse(
        label=label,
        fake_probability=round(fake_prob, 4),
        real_probability=round(real_prob, 4),
        threshold=THRESHOLD,
    )
