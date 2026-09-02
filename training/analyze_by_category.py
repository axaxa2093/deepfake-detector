import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

# ============================================================
#  НАСТРОЙКИ
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", default="D:/DeepfakeDetection/real_fake_dataset")
parser.add_argument("--fft_root", default="D:/DeepfakeDetection/real_fake_fft")
parser.add_argument("--csv_path", default="D:/DeepfakeDetection/real_fake_dataset/metadata.csv")
parser.add_argument("--model_path", default="D:/DeepfakeDetection/results_attention/best_model_attention.keras")
parser.add_argument("--results_dir", default="D:/DeepfakeDetection/results_attention")
args = parser.parse_args()

DATASET_ROOT = args.dataset_root
FFT_ROOT     = args.fft_root
CSV_PATH     = args.csv_path
MODEL_PATH   = args.model_path
RESULTS_DIR  = args.results_dir

IMG_SIZE     = 224
SEED         = 42
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
# ============================================================


# ============================================================
#  КАСТОМНЫЕ СЛОИ
# ============================================================
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


# ============================================================
#  ЗАГРУЗКА МОДЕЛИ
# ============================================================
print("Загрузка модели...")
model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "ReduceMeanLayer": ReduceMeanLayer,
        "ReduceMaxLayer": ReduceMaxLayer,
        "ScaledDotProductLayer": ScaledDotProductLayer,
        "MatMulLayer": MatMulLayer,
    }
)
print("Модель загружена!")


# ============================================================
#  РАЗБИВКА ДАТАСЕТА
# ============================================================
df = pd.read_csv(CSV_PATH)
train_df, temp_df = train_test_split(
    df, test_size=(VAL_RATIO + TEST_RATIO), random_state=SEED, stratify=df['label'])
val_df, test_df = train_test_split(
    temp_df, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
    random_state=SEED, stratify=temp_df['label'])
print(f"Test set: {len(test_df)} картинок")


# ============================================================
#  ПРЕДСКАЗАНИЯ
# ============================================================
print("Получение предсказаний...")
y_true, y_pred, models_list, categories_list = [], [], [], []

for _, row in test_df.iterrows():
    rgb_path = Path(DATASET_ROOT) / row['filename']
    fft_path = Path(FFT_ROOT) / Path(row['filename']).with_suffix('.png')

    rgb = tf.keras.preprocessing.image.load_img(
        str(rgb_path), target_size=(IMG_SIZE, IMG_SIZE))
    rgb = tf.keras.preprocessing.image.img_to_array(rgb) / 255.0
    rgb = np.expand_dims(rgb, 0)

    fft = tf.keras.preprocessing.image.load_img(
        str(fft_path), target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale')
    fft = tf.keras.preprocessing.image.img_to_array(fft) / 255.0
    fft = np.expand_dims(fft, 0)

    pred = model.predict([rgb, fft], verbose=0)[0][0]
    y_true.append(row['label'])
    y_pred.append(1 if pred >= 0.5 else 0)
    models_list.append(row['model'])
    categories_list.append(row['category'])

results_df = pd.DataFrame({
    'label':    y_true,
    'pred':     y_pred,
    'model':    models_list,
    'category': categories_list,
    'correct':  [t == p for t, p in zip(y_true, y_pred)]
})


# ============================================================
#  СТАТИСТИКА
# ============================================================
model_stats = results_df.groupby('model').agg(
    total=('correct', 'count'),
    correct=('correct', 'sum')
).reset_index()
model_stats['accuracy'] = model_stats['correct'] / model_stats['total']
model_stats['errors'] = model_stats['total'] - model_stats['correct']
model_stats = model_stats.sort_values('accuracy')

print("\nТОЧНОСТЬ ПО МОДЕЛЯМ:")
print(model_stats.to_string(index=False))


# ============================================================
#  ГРАФИКИ
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Анализ ошибок по категориям и моделям (с вниманием)', fontsize=14)

colors = ['#d32f2f' if acc < 0.85 else '#388e3c' for acc in model_stats['accuracy']]
bars = axes[0, 0].barh(model_stats['model'], model_stats['accuracy'], color=colors)
axes[0, 0].axvline(x=0.85, color='gray', linestyle='--', alpha=0.7, label='порог 85%')
axes[0, 0].set_xlim(0, 1.05)
axes[0, 0].set_title('Точность по моделям')
axes[0, 0].set_xlabel('Accuracy')
axes[0, 0].legend()
for bar, acc in zip(bars, model_stats['accuracy']):
    axes[0, 0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{acc:.2f}', va='center', fontsize=9)

axes[0, 1].barh(model_stats['model'], model_stats['errors'], color='#ef5350')
axes[0, 1].set_title('Количество ошибок по моделям')
axes[0, 1].set_xlabel('Количество ошибок')
for i, (errors, total) in enumerate(zip(model_stats['errors'], model_stats['total'])):
    axes[0, 1].text(errors + 0.3, i, f'{errors}/{total}', va='center', fontsize=9)

cat_stats = results_df.groupby('category').agg(
    total=('correct', 'count'),
    correct=('correct', 'sum')
).reset_index()
cat_stats['accuracy'] = cat_stats['correct'] / cat_stats['total']
cat_stats = cat_stats.sort_values('accuracy')
cat_colors = ['#d32f2f' if acc < 0.85 else '#388e3c' for acc in cat_stats['accuracy']]
bars3 = axes[1, 0].bar(cat_stats['category'], cat_stats['accuracy'], color=cat_colors)
axes[1, 0].axhline(y=0.85, color='gray', linestyle='--', alpha=0.7)
axes[1, 0].set_ylim(0, 1.1)
axes[1, 0].set_title('Точность по категориям')
axes[1, 0].set_ylabel('Accuracy')
for bar, acc in zip(bars3, cat_stats['accuracy']):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.2f}', ha='center', fontsize=10)

error_df = results_df[results_df['correct'] == False].copy()
if len(error_df) > 0:
    error_df['error_type'] = error_df.apply(
        lambda r: 'FP (real→fake)' if r['label'] == 0 else 'FN (fake→real)', axis=1)
    pivot = error_df.groupby(['model', 'error_type']).size().unstack(fill_value=0)
    sns.heatmap(pivot, annot=True, fmt='d', cmap='Reds',
                ax=axes[1, 1], linewidths=0.5)
    axes[1, 1].set_title('Типы ошибок по моделям')
    axes[1, 1].set_xlabel('Тип ошибки')
    axes[1, 1].set_ylabel('Модель')

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'analysis_by_category_attention.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nГрафик сохранён: {plot_path}")

stats_path = os.path.join(RESULTS_DIR, 'stats_by_model_attention.csv')
model_stats.to_csv(stats_path, index=False, encoding='utf-8')
print(f"Статистика сохранена: {stats_path}")