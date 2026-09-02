import os
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import argparse

# ============================================================
#  НАСТРОЙКИ
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", default="D:/DeepfakeDetection/real_fake_dataset")
parser.add_argument("--fft_root", default="D:/DeepfakeDetection/real_fake_fft")
parser.add_argument("--csv_path", default="D:/DeepfakeDetection/real_fake_dataset/metadata.csv")
args = parser.parse_args()

DATASET_ROOT = args.dataset_root
FFT_ROOT     = args.fft_root
CSV_PATH     = args.csv_path
# ============================================================


def compute_fft_raw(img_path):
    """
    Вычисляет логарифмический амплитудный спектр.
    Возвращает float32 массив без нормализации.
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    fft = np.fft.fft2(img.astype(np.float32))
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shifted))

    if magnitude.shape != (224, 224):
        magnitude = cv2.resize(magnitude, (224, 224))

    return magnitude


def process_dataset(csv_path, dataset_root, fft_root):
    df = pd.read_csv(csv_path)
    total = len(df)

    # ============================================================
    #  Сбор всех спектров в память и высчитывание глобальных
    #  min/max по всему датасету
    # ============================================================
    print("=" * 55)
    print("ШАГ 1: Вычисление глобального min/max")
    print("=" * 55)

    global_min = float('inf')
    global_max = float('-inf')
    cache = {}

    for i, row in df.iterrows():
        img_path = Path(dataset_root) / row["filename"]
        mag = compute_fft_raw(img_path)

        if mag is None:
            continue

        cache[row["filename"]] = mag
        global_min = min(global_min, mag.min())
        global_max = max(global_max, mag.max())

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{total}] min={global_min:.3f} max={global_max:.3f}")

    print(f"\nГлобальный min: {global_min:.4f}")
    print(f"Глобальный max: {global_max:.4f}")

    # ============================================================
    #  Нормализация относительно глобальных min/max и сохранение
    # ============================================================
    print("\n" + "=" * 55)
    print("ШАГ 2: Нормализация и сохранение")
    print("=" * 55)

    success = 0
    errors = 0

    for i, row in df.iterrows():
        if row["filename"] not in cache:
            errors += 1
            continue

        mag = cache[row["filename"]]

        mag_norm = (mag - global_min) / (global_max - global_min)
        mag_uint8 = (mag_norm * 255).astype(np.uint8)

        fft_path = Path(fft_root) / row["filename"]
        os.makedirs(fft_path.parent, exist_ok=True)
        fft_path = fft_path.with_suffix(".png")

        cv2.imwrite(str(fft_path), mag_uint8)
        success += 1

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{total}] сохранено: {success}")

    params_path = Path(fft_root) / "normalization_params.txt"
    os.makedirs(fft_root, exist_ok=True)
    with open(params_path, 'w') as f:
        f.write(f"global_min={global_min}\n")
        f.write(f"global_max={global_max}\n")
    print(f"\nПараметры нормализации сохранены: {params_path}")

    print("\n" + "=" * 55)
    print("ГОТОВО!")
    print(f"Успешно:  {success}")
    print(f"Ошибок:   {errors}")
    print(f"Результат в: {fft_root}")
    print("=" * 55)


if __name__ == "__main__":
    process_dataset(CSV_PATH, DATASET_ROOT, FFT_ROOT)