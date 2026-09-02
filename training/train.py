import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score,
                             precision_score, recall_score)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                         ModelCheckpoint)

# ============================================================
#  НАСТРОЙКИ
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", default="D:/DeepfakeDetection/real_fake_dataset")
parser.add_argument("--fft_root", default="D:/DeepfakeDetection/real_fake_fft")
parser.add_argument("--csv_path", default="D:/DeepfakeDetection/real_fake_dataset/metadata.csv")
parser.add_argument("--results_dir", default="D:/DeepfakeDetection/results_attention")
args = parser.parse_args()

DATASET_ROOT  = args.dataset_root
FFT_ROOT      = args.fft_root
CSV_PATH      = args.csv_path
RESULTS_DIR   = args.results_dir

IMG_SIZE      = 224
BATCH_SIZE    = 32
EPOCHS        = 30
LEARNING_RATE = 3e-5

TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15
SEED          = 42
# ============================================================

os.makedirs(RESULTS_DIR, exist_ok=True)
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ============================================================
#  1. ЗАГРУЗКА И РАЗБИВКА ДАТАСЕТА
# ============================================================
print("=" * 60)
print("1. ЗАГРУЗКА ДАТАСЕТА")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
print(f"Всего записей: {len(df)}")
print(f"Fake: {df['label'].sum()}, Real: {(df['label']==0).sum()}")

train_df, temp_df = train_test_split(
    df, test_size=(VAL_RATIO + TEST_RATIO), random_state=SEED, stratify=df['label']
)
val_df, test_df = train_test_split(
    temp_df, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
    random_state=SEED, stratify=temp_df['label']
)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


# ============================================================
#  2. ЗАГРУЗЧИК ДАННЫХ
# ============================================================
class DualStreamDataset(keras.utils.Sequence):
    def __init__(self, df, dataset_root, fft_root, batch_size, augment=False):
        self.df = df.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.fft_root = Path(fft_root)
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        rgb_batch, fft_batch, labels = [], [], []

        for _, row in batch.iterrows():
            rgb_path = self.dataset_root / row['filename']
            rgb = tf.keras.preprocessing.image.load_img(
                str(rgb_path), target_size=(IMG_SIZE, IMG_SIZE))
            rgb = tf.keras.preprocessing.image.img_to_array(rgb) / 255.0

            fft_filename = Path(row['filename']).with_suffix('.png')
            fft_path = self.fft_root / fft_filename
            fft = tf.keras.preprocessing.image.load_img(
                str(fft_path), target_size=(IMG_SIZE, IMG_SIZE),
                color_mode='grayscale')
            fft = tf.keras.preprocessing.image.img_to_array(fft) / 255.0

            if self.augment and np.random.rand() > 0.5:
                rgb = np.fliplr(rgb)
                fft = np.fliplr(fft)

            rgb_batch.append(rgb)
            fft_batch.append(fft)
            labels.append(row['label'])

        return (np.array(rgb_batch), np.array(fft_batch)), np.array(labels)

# ============================================================
#  КАСТОМНЫЕ СЛОИ (вместо Lambda — корректная сериализация)
# ============================================================

class ReduceMeanLayer(keras.layers.Layer):
    """Заменяет tf.reduce_mean внутри Functional API."""
    def call(self, x):
        return tf.reduce_mean(x, axis=-1, keepdims=True)
    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (1,)

class ReduceMaxLayer(keras.layers.Layer):
    """Заменяет tf.reduce_max внутри Functional API."""
    def call(self, x):
        return tf.reduce_max(x, axis=-1, keepdims=True)
    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (1,)

class ScaledDotProductLayer(keras.layers.Layer):
    """Scaled dot-product: Q * K^T / scale."""
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
    """Матричное умножение двух тензоров."""
    def call(self, inputs):
        a, b = inputs
        return tf.matmul(a, b)

# ============================================================
#  3. БЛОКИ ВНИМАНИЯ
# ============================================================

def se_block(x, ratio=16, name="se"):
    """
    Squeeze-and-Excitation: channel attention.
    Учит модель взвешивать важность каждого канала признаков.
    """
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    se = layers.Dense(channels // ratio, activation='relu',
                      name=f"{name}_fc1")(se)
    se = layers.Dense(channels, activation='sigmoid',
                      name=f"{name}_fc2")(se)
    se = layers.Reshape((1, 1, channels), name=f"{name}_reshape")(se)
    return layers.Multiply(name=f"{name}_scale")([x, se])


def cbam_block(x, ratio=16, name="cbam"):
    """
    CBAM: channel attention + spatial attention.
    Отвечает на вопросы: какие каналы важны и где смотреть.
    """
    channels = x.shape[-1]

    # Channel Attention
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)
    avg_pool = layers.Reshape((1, 1, channels))(avg_pool)
    max_pool = layers.Reshape((1, 1, channels))(max_pool)

    shared_fc1 = layers.Dense(channels // ratio, activation='relu',
                               name=f"{name}_ch_fc1")
    shared_fc2 = layers.Dense(channels, name=f"{name}_ch_fc2")

    avg_out = shared_fc2(shared_fc1(avg_pool))
    max_out = shared_fc2(shared_fc1(max_pool))

    channel_att = layers.Add()([avg_out, max_out])
    channel_att = layers.Activation('sigmoid',
                                     name=f"{name}_ch_sigmoid")(channel_att)
    x = layers.Multiply(name=f"{name}_ch_scale")([x, channel_att])

    # Spatial Attention
    avg_spatial = ReduceMeanLayer(name=f"{name}_avg_spatial")(x)
    max_spatial = ReduceMaxLayer(name=f"{name}_max_spatial")(x)
    spatial_concat = layers.Concatenate(axis=-1,
                                         name=f"{name}_sp_concat")(
        [avg_spatial, max_spatial])
    spatial_att = layers.Conv2D(1, kernel_size=7, padding='same',
                                 activation='sigmoid',
                                 name=f"{name}_sp_conv")(spatial_concat)
    x = layers.Multiply(name=f"{name}_sp_scale")([x, spatial_att])

    return x


def cross_attention(query, key_value, dim=256, name="cross_att"):
    """
    Cross-Attention между RGB и FFT потоками.
    Позволяет одному потоку "спрашивать" другой о важных признаках.
    """
    q = layers.Dense(dim, name=f"{name}_q")(query)
    k = layers.Dense(dim, name=f"{name}_k")(key_value)
    v = layers.Dense(dim, name=f"{name}_v")(key_value)

    q = layers.Reshape((1, dim), name=f"{name}_q_reshape")(q)
    k = layers.Reshape((1, dim), name=f"{name}_k_reshape")(k)
    v = layers.Reshape((1, dim), name=f"{name}_v_reshape")(v)

    scale = float(tf.math.sqrt(tf.cast(dim, tf.float32)).numpy())
    scores = ScaledDotProductLayer(scale=scale, name=f"{name}_scores")([q, k])
    weights = layers.Softmax(name=f"{name}_softmax")(scores)

    attended = MatMulLayer(name=f"{name}_attended")([weights, v])
    attended = layers.Reshape((dim,), name=f"{name}_out_reshape")(attended)

    # Residual connection + LayerNorm
    query_proj = layers.Dense(dim, name=f"{name}_query_proj")(query)
    attended = layers.Add(name=f"{name}_residual")([attended, query_proj])
    attended = layers.LayerNormalization(name=f"{name}_norm")(attended)
    return attended


# ============================================================
#  4. АРХИТЕКТУРА С ВНИМАНИЕМ
# ============================================================
def build_attention_model():
    # ── Поток 1: RGB ──────────────────────────────────────────
    rgb_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="rgb_input")

    base_model = EfficientNetB0(
        include_top=False, weights='imagenet', input_tensor=rgb_input
    )
    # Первые 100 слоёв заморожены — базовые признаки универсальны
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    rgb_x = se_block(base_model.output, ratio=16, name="rgb_se")
    rgb_x = cbam_block(rgb_x, ratio=16, name="rgb_cbam")
    rgb_vec = layers.GlobalAveragePooling2D(name="rgb_gap")(rgb_x)
    rgb_vec = layers.Dense(256, activation='relu', name="rgb_dense")(rgb_vec)
    rgb_vec = layers.Dropout(0.3, name="rgb_dropout")(rgb_vec)

    # ── Поток 2: FFT ──────────────────────────────────────────
    fft_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="fft_input")

    f = layers.Conv2D(32, 3, activation='relu', padding='same',
                      name="fft_conv1")(fft_input)
    f = layers.BatchNormalization()(f)
    f = layers.MaxPooling2D(2)(f)

    f = layers.Conv2D(64, 3, activation='relu', padding='same',
                      name="fft_conv2")(f)
    f = layers.BatchNormalization()(f)
    f = layers.MaxPooling2D(2)(f)

    f = layers.Conv2D(128, 3, activation='relu', padding='same',
                      name="fft_conv3")(f)
    f = layers.BatchNormalization()(f)
    f = layers.MaxPooling2D(2)(f)

    f = layers.Conv2D(256, 3, activation='relu', padding='same',
                      name="fft_conv4")(f)
    f = layers.BatchNormalization()(f)

    f = se_block(f, ratio=16, name="fft_se")
    f = cbam_block(f, ratio=16, name="fft_cbam")

    fft_vec = layers.GlobalAveragePooling2D(name="fft_gap")(f)
    fft_vec = layers.Dense(256, activation='relu', name="fft_dense")(fft_vec)
    fft_vec = layers.Dropout(0.3, name="fft_dropout")(fft_vec)

    # ── Cross-Attention ───────────────────────────────────────
    rgb_attended = cross_attention(rgb_vec, fft_vec, dim=256,
                                   name="rgb_to_fft")
    fft_attended = cross_attention(fft_vec, rgb_vec, dim=256,
                                   name="fft_to_rgb")

    # ── Объединение ───────────────────────────────────────────
    combined = layers.Concatenate(name="concat")([rgb_attended, fft_attended])
    combined = layers.Dense(256, activation='relu',
                             name="combined_dense1")(combined)
    combined = layers.Dropout(0.4, name="combined_dropout")(combined)
    combined = layers.Dense(64, activation='relu',
                             name="combined_dense2")(combined)
    output = layers.Dense(1, activation='sigmoid', name="output")(combined)

    model = keras.Model(
        inputs=[rgb_input, fft_input],
        outputs=output,
        name="DualStreamAttentionDetector"
    )
    return model, base_model


# ============================================================
#  5. ОБУЧЕНИЕ
# ============================================================
print("\n" + "=" * 60)
print("2. ПОСТРОЕНИЕ МОДЕЛИ С ВНИМАНИЕМ")
print("=" * 60)

model, base_model = build_attention_model()
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

train_loader = DualStreamDataset(train_df, DATASET_ROOT, FFT_ROOT,
                                  BATCH_SIZE, augment=True)
val_loader   = DualStreamDataset(val_df,   DATASET_ROOT, FFT_ROOT,
                                  BATCH_SIZE, augment=False)
test_loader  = DualStreamDataset(test_df,  DATASET_ROOT, FFT_ROOT,
                                  BATCH_SIZE, augment=False)

callbacks = [
    EarlyStopping(monitor='val_auc', patience=7,
                  restore_best_weights=True, mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=3, min_lr=1e-7),
    ModelCheckpoint(
        filepath=os.path.join(RESULTS_DIR, 'best_model_attention.keras'),
        monitor='val_auc', save_best_only=True, mode='max'
    )
]

print("\n" + "=" * 60)
print("3. ОБУЧЕНИЕ")
print("=" * 60)

history = model.fit(
    train_loader,
    validation_data=val_loader,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
#  6. ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ
# ============================================================
print("\n" + "=" * 60)
print("4. ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
print("=" * 60)

y_true, y_pred_proba = [], []
for (rgb_batch, fft_batch), labels in test_loader:
    preds = model.predict([rgb_batch, fft_batch], verbose=0)
    y_pred_proba.extend(preds.flatten())
    y_true.extend(labels)

y_true = np.array(y_true)
y_pred_proba = np.array(y_pred_proba)
y_pred = (y_pred_proba >= 0.5).astype(int)

accuracy  = (y_true == y_pred).mean()
auc       = roc_auc_score(y_true, y_pred_proba)
f1        = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall    = recall_score(y_true, y_pred)

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"AUC-ROC:   {auc:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))


# ============================================================
#  7. ГРАФИКИ
# ============================================================
print("\n" + "=" * 60)
print("5. СОХРАНЕНИЕ ГРАФИКОВ")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Dual-Stream + Attention Detector — Training Results', fontsize=14)

axes[0, 0].plot(history.history['loss'], label='Train Loss')
axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
axes[0, 0].set_title('Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].legend()

axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0, 1].set_title('Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].legend()

fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
axes[1, 0].plot(fpr, tpr, label=f'AUC = {auc:.4f}')
axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[1, 0].set_title('ROC Curve')
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].legend()

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'],
            ax=axes[1, 1])
axes[1, 1].set_title('Confusion Matrix')
axes[1, 1].set_ylabel('True Label')
axes[1, 1].set_xlabel('Predicted Label')

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'training_results_attention.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Графики сохранены: {plot_path}")

metrics_path = os.path.join(RESULTS_DIR, 'metrics_attention.txt')
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write("=== Dual-Stream + Attention Deepfake Detector ===\n\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"AUC-ROC:   {auc:.4f}\n")
    f.write(f"F1-score:  {f1:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n\n")
    f.write(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))
print(f"Метрики сохранены: {metrics_path}")

print("\n" + "=" * 60)
print("ГОТОВО! Модель сохранена в:",
      os.path.join(RESULTS_DIR, 'best_model_attention.keras'))
print("=" * 60)