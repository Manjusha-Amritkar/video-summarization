"""
Training script for BiLSTM-based video summarization
"""

import os
import tensorflow as tf
from tensorflow.keras import optimizers

from model import BiLSTMAttentionModel
from feature_extraction import load_features_with_sliding_windows


# =======================
# GPU & Precision Setup
# =======================
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

tf.keras.mixed_precision.set_global_policy("mixed_float16")


# =======================
# Configuration
# =======================
BASE_FEATURE_PATH = "/mnt/d/Mass Projects/Video_Summ/new_dataset_features_npy1"
TRAIN_PATH = os.path.join(BASE_FEATURE_PATH, "train")
VALID_PATH = os.path.join(BASE_FEATURE_PATH, "valid")

SEQ_LEN = 150
STRIDE = 125
FEATURE_DIM = 2048

NUM_EPOCHS = 50
BATCH_SIZE = 1
LEARNING_RATE = 1e-5

MODEL_SAVE_PATH = (
    "/mnt/d/Mass Projects/Video_Summ/RESULTS/Models_2/"
    "SUMME/video_summarization_4-heads_15k.keras"
)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)


# =======================
# Training Pipeline
# =======================
def train():
    print("📥 Loading training features...")
    train_features = load_features_with_sliding_windows(
        TRAIN_PATH, seq_len=SEQ_LEN, stride=STRIDE
    )

    print("📥 Loading validation features...")
    valid_features = load_features_with_sliding_windows(
        VALID_PATH, seq_len=SEQ_LEN, stride=STRIDE
    )

    print("Train windows:", train_features.shape)
    print("Valid windows:", valid_features.shape)

    model = BiLSTMAttentionModel(feature_dim=FEATURE_DIM)
    model.build(input_shape=(None, SEQ_LEN, FEATURE_DIM))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )

    print("🚀 Starting training...")
    model.fit(
        train_features,
        train_features,
        validation_data=(valid_features, valid_features),
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
    )

    model.save(MODEL_SAVE_PATH)
    print(f"✅ Model saved at {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
