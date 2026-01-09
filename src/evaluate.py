"""
Evaluation pipeline for video summarization
(Model loading + feature reconstruction)
"""

import os
import numpy as np
import tensorflow as tf
import gc

from model import BiLSTMAttentionModel


# =======================
# Configuration
# =======================
MODEL_PATH = (
    "/mnt/d/Mass Projects/Video_Summ/RESULTS/Final/Models/"
    "bilstm_multi_4-heads_60k_2.keras"
)

FEATURES_DIR = "/mnt/d/Mass Projects/Video_Summ/extracted_features/"
CHUNK_SIZE = 500


# =======================
# Load Model
# =======================
def load_trained_model():
    with tf.keras.utils.custom_object_scope(
        {"BiLSTMAttentionModel": BiLSTMAttentionModel}
    ):
        model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
    return model


# =======================
# Feature Reconstruction
# =======================
def reconstruct_video_features(model, video_features):
    """
    Reconstruct features in chunks to avoid OOM.

    Args:
        model: Trained reconstruction model
        video_features (np.ndarray): (num_frames, feature_dim)

    Returns:
        np.ndarray: Reconstructed features
    """
    num_frames = video_features.shape[0]
    video_features = video_features[np.newaxis, ...]

    reconstructed = np.zeros_like(video_features)

    for i in range((num_frames + CHUNK_SIZE - 1) // CHUNK_SIZE):
        s = i * CHUNK_SIZE
        e = min((i + 1) * CHUNK_SIZE, num_frames)
        reconstructed[:, s:e, :] = model.predict(
            video_features[:, s:e, :], verbose=0
        )

    return reconstructed.squeeze(axis=0)


# =======================
# Main
# =======================
def main():
    model = load_trained_model()

    sample_videos = [
        f for f in os.listdir(FEATURES_DIR) if f.endswith(".npy")
    ]

    for video_file in sample_videos:
        print(f"\n🔄 Reconstructing: {video_file}")
        feats = np.load(os.path.join(FEATURES_DIR, video_file))
        recon = reconstruct_video_features(model, feats)
        print("Reconstructed shape:", recon.shape)

        del feats, recon
        gc.collect()
        tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
