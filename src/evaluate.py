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


import scipy.io
import ruptures as rpt
import pandas as pd


def get_segments_from_indices(indices):
    if not indices:
        return []
    indices = sorted(set(indices))
    segments, start = [], indices[0]
    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1] + 1:
            segments.append((start, indices[i - 1]))
            start = indices[i]
    segments.append((start, indices[-1]))
    return segments


def expand_segment(seg):
    return list(range(seg[0], seg[1] + 1))


def calculate_segment_f1(user_segments, model_segments):
    tp = 0
    for m_seg in model_segments:
        m_range = set(expand_segment(m_seg))
        for u_seg in user_segments:
            u_range = set(expand_segment(u_seg))
            if m_range & u_range:
                tp += 1
                break

    precision = tp / len(model_segments) if model_segments else 0
    recall = tp / len(user_segments) if user_segments else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )
    return f1, precision, recall


def detect_kts_segments(features, penalty=500):
    algo = rpt.KernelCPD(kernel="linear").fit(features)
    change_points = algo.predict(pen=penalty)

    segments = []
    start = 0
    for cp in change_points:
        segments.append((start, cp - 1))
        start = cp

    if segments[-1][1] < features.shape[0] - 1:
        segments.append((start, features.shape[0] - 1))

    return segments


def knapsack(values, weights, capacity):
    n = len(values)
    W = int(capacity * 100)
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        w = int(weights[i - 1] * 100)
        for c in range(W + 1):
            if w <= c:
                dp[i][c] = max(values[i - 1] + dp[i - 1][c - w], dp[i - 1][c])
            else:
                dp[i][c] = dp[i - 1][c]

    selected = []
    c = W
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            selected.append(i - 1)
            c -= int(weights[i - 1] * 100)

    return selected[::-1]


# =======================
# Main
# =======================
MAT_DIR = "/mnt/d/Mass Projects/Video_Summ/mat/"
FPS = 25
SUMMARY_XLSX = "/mnt/d/Mass Projects/Video_Summ/RESULTS/Final/Results/SUMME/evaluation_results.xlsx"
PENALTY = 500


def main():
    model = load_trained_model()
    results = []

    video_files = [f for f in os.listdir(FEATURES_DIR) if f.endswith(".npy")]

    with pd.ExcelWriter(SUMMARY_XLSX, engine="xlsxwriter") as writer:
        for video_file in video_files:
            video_name = os.path.splitext(video_file)[0]
            print(f"\n🚀 Evaluating: {video_name}")

            feat_path = os.path.join(FEATURES_DIR, video_file)
            mat_path = os.path.join(MAT_DIR, f"{video_name}.mat")

            if not os.path.exists(mat_path):
                print("❌ Missing annotation file, skipping.")
                continue

            raw_feats = np.load(feat_path)
            recon = reconstruct_video_features(model, raw_feats)

            # Importance scores via reconstruction error
            errs = np.mean(np.abs(raw_feats - recon), axis=1)
            imp_scores = (errs - errs.min()) / (errs.max() - errs.min() + 1e-8)

            # KTS segmentation
            segments = detect_kts_segments(raw_feats, penalty=PENALTY)
            durations = [(e - s + 1) / FPS for s, e in segments]
            scores = [np.mean(imp_scores[s:e + 1]) for s, e in segments]

            total_duration = raw_feats.shape[0] / FPS
            summary_duration = total_duration * 0.15

            selected_idx = knapsack(scores, durations, summary_duration)
            selected_segments = [segments[i] for i in selected_idx]

            # Load ground truth
            mat = scipy.io.loadmat(mat_path)
            user_scores = mat["user_score"]

            metrics = []
            for u in range(user_scores.shape[1]):
                usr = user_scores[:, u]
                nonzero = np.where(usr > 0)[0]
                user_segments = get_segments_from_indices(nonzero.tolist())

                f1, p, r = calculate_segment_f1(user_segments, selected_segments)
                metrics.append((u + 1, f1, p, r))

            df = pd.DataFrame(metrics, columns=["User", "F1", "Precision", "Recall"])
            df.to_excel(writer, sheet_name=video_name[:31], index=False)

            results.append({
                "Video": video_name,
                "Avg F1": df["F1"].mean(),
                "Avg Precision": df["Precision"].mean(),
                "Avg Recall": df["Recall"].mean(),
            })

            del raw_feats, recon
            gc.collect()
            tf.keras.backend.clear_session()

        pd.DataFrame(results).to_excel(writer, sheet_name="Summary", index=False)

    print(f"\n✅ Evaluation completed. Results saved to {SUMMARY_XLSX}")




if __name__ == "__main__":
    main()
