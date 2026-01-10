"""
TVSum evaluation script for unsupervised video summarization
"""

import os
import gc
import numpy as np
import tensorflow as tf
import pandas as pd
import h5py
import ruptures as rpt

from model import BiLSTMAttentionModel
from utils import get_segments_from_indices, calculate_segment_f1
from config import (
    MODEL_SAVE_PATH,
    CHUNK_SIZE,
    KTS_PENALTY,
)

# =======================
# TVSum-specific paths
# =======================
FEATURES_DIR = "data/tvsum/features"
MAT_FILE_PATH = "data/tvsum/ydata-tvsum50.mat"
EXCEL_OUT = "results/tvsum_evaluation.xlsx"

FPS = 25
SUMMARY_RATIO = 0.15


# =======================
# Load Model
# =======================
def load_model():
    with tf.keras.utils.custom_object_scope(
        {"BiLSTMAttentionModel": BiLSTMAttentionModel}
    ):
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    print("✅ Model loaded")
    return model


# =======================
# KTS + Knapsack helpers
# =======================
def detect_kts_segments(features):
    algo = rpt.KernelCPD(kernel="linear").fit(features)
    cps = algo.predict(pen=KTS_PENALTY)

    segments, start = [], 0
    for cp in cps:
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
# Main
# =======================
def main():
    model = load_model()

    with h5py.File(MAT_FILE_PATH, "r") as mat:
        tvsum = mat["tvsum50"]

        def decode(ref):
            return "".join(map(chr, mat[ref][()].flatten())).replace("\x00", "")

        video_names = [decode(r) for r in tvsum["video"][:, 0]]
        user_annos = [mat[r][:] for r in tvsum["user_anno"][:, 0]]

    with pd.ExcelWriter(EXCEL_OUT, engine="xlsxwriter") as writer:
        summary = []

        for idx, video in enumerate(video_names):
            print(f"\n🚀 Evaluating TVSum video: {video}")

            feat_path = os.path.join(FEATURES_DIR, f"{video}.npy")
            if not os.path.exists(feat_path):
                print("❌ Missing features, skipping")
                continue

            raw_feats = np.load(feat_path)
            num_frames = raw_feats.shape[0]
            video_feats = raw_feats[np.newaxis, ...]

            recon = np.zeros_like(video_feats)
            for i in range((num_frames + CHUNK_SIZE - 1) // CHUNK_SIZE):
                s, e = i * CHUNK_SIZE, min((i + 1) * CHUNK_SIZE, num_frames)
                recon[:, s:e, :] = model.predict(video_feats[:, s:e, :], verbose=0)

            errs = np.mean(np.abs(video_feats - recon), axis=2).flatten()
            imp = (errs - errs.min()) / (errs.max() - errs.min() + 1e-8)

            segments = detect_kts_segments(raw_feats)
            durations = [(e - s + 1) / FPS for s, e in segments]
            scores = [np.mean(imp[s:e + 1]) for s, e in segments]

            total_duration = num_frames / FPS
            selected = knapsack(scores, durations, total_duration * SUMMARY_RATIO)
            model_segments = [segments[i] for i in selected]

            metrics = []
            for u, usr in enumerate(user_annos[idx]):
                k = int(np.floor(num_frames * SUMMARY_RATIO))
                idxs = np.argsort(usr)[-k:]
                user_segments = get_segments_from_indices(idxs.tolist())

                f1, p, r = calculate_segment_f1(user_segments, model_segments)
                metrics.append((u + 1, f1, p, r))

            df = pd.DataFrame(metrics, columns=["User", "F1", "Precision", "Recall"])
            df.to_excel(writer, sheet_name=video[:31], index=False)

            summary.append({
                "Video": video,
                "Avg F1": df["F1"].mean(),
                "Avg Precision": df["Precision"].mean(),
                "Avg Recall": df["Recall"].mean(),
            })

            del video_feats, recon
            gc.collect()
            tf.keras.backend.clear_session()

        pd.DataFrame(summary).to_excel(writer, sheet_name="Summary", index=False)

    print(f"\n✅ TVSum evaluation saved to {EXCEL_OUT}")


if __name__ == "__main__":
    main()
