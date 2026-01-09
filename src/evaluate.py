"""
Evaluation pipeline for video summarization
Includes reconstruction, KTS segmentation, knapsack selection,
and F1 / Precision / Recall evaluation on SumMe dataset.
"""

import os
import gc
import numpy as np
import tensorflow as tf
import scipy.io
import ruptures as rpt
import pandas as pd

from model import BiLSTMAttentionModel
from config import (
    MAT_DIR,
    FPS,
    SUMMARY_RATIO,
    KTS_PENALTY,
    CHUNK_SIZE,
    EVAL_RESULTS_PATH,
    MODEL_SAVE_PATH,
    BASE_FEATURE_PATH,
)


# =======================
# Load Model
# =======================
def load_trained_model():
    with tf.keras.utils.custom_object_scope(
        {"BiLSTMAttentionModel": BiLSTMAttentionModel}
    ):
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    print("✅ Model loaded successfully")
    return model


# =======================
# Feature Reconstruction
# =======================
def reconstruct_video_features(model, video_features):
    """
    Reconstruct features in chunks to avoid OOM.
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
# Helper Functions
# =======================
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
            if m_range & set(expand_segment(u_seg)):
                tp += 1
                break

    precision = tp / len(model_segments) if model_segments else 0
    recall = tp / len(user_segments) if user_segments else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    return f1, precision, recall


def detect_kts_segments(features):
    algo = rpt.KernelCPD(kernel="linear").fit(features)
    change_points = algo.predict(pen=KTS_PENALTY)

    segments, start = [], 0
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

    selected, c = [], W
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            selected.append(i - 1)
            c -= int(weights[i - 1] * 100)

    return selected[::-1]


# =======================
# Main Evaluation Loop
# =======================
def main():
    model = load_trained_model()
    results = []

    features_dir = os.path.join(BASE_FEATURE_PATH, "test")
    video_files = [f for f in os.listdir(features_dir) if f.endswith(".npy")]

    with pd.ExcelWriter(EVAL_RESULTS_PATH, engine="xlsxwriter") as writer:
        for video_file in video_files:
            video_name = os.path.splitext(video_file)[0]
            print(f"\n🚀 Evaluating: {video_name}")

            feat_path = os.path.join(features_dir, video_file)
            mat_path = os.path.join(MAT_DIR, f"{video_name}.mat")

            if not os.path.exists(mat_path):
                print("❌ Missing annotation file, skipping.")
                continue

            raw_feats = np.load(feat_path)
            recon = reconstruct_video_features(model, raw_feats)

            errs = np.mean(np.abs(raw_feats - recon), axis=1)
            imp_scores = (errs - errs.min()) / (errs.max() - errs.min() + 1e-8)

            segments = detect_kts_segments(raw_feats)
            durations = [(e - s + 1) / FPS for s, e in segments]
            scores = [np.mean(imp_scores[s:e + 1]) for s, e in segments]

            total_duration = raw_feats.shape[0] / FPS
            summary_duration = total_duration * SUMMARY_RATIO

            selected_idx = knapsack(scores, durations, summary_duration)
            selected_segments = [segments[i] for i in selected_idx]

            mat = scipy.io.loadmat(mat_path)
            user_scores = mat["user_score"]

            metrics = []
            for u in range(user_scores.shape[1]):
                usr = user_scores[:, u]
                user_segments = get_segments_from_indices(
                    np.where(usr > 0)[0].tolist()
                )
                f1, p, r = calculate_segment_f1(
                    user_segments, selected_segments
                )
                metrics.append((u + 1, f1, p, r))

            df = pd.DataFrame(
                metrics, columns=["User", "F1", "Precision", "Recall"]
            )
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

        pd.DataFrame(results).to_excel(
            writer, sheet_name="Summary", index=False
        )

    print(f"\n✅ Evaluation completed. Results saved to {EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()
