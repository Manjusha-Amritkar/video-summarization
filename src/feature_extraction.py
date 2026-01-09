"""
Feature loading and sliding-window generation for video summarization
"""

import os
import numpy as np


def load_features_with_sliding_windows(
    directory,
    seq_len=150,
    stride=125,
):
    """
    Loads .npy feature files and creates sliding windows.

    Args:
        directory (str): Path to directory containing .npy feature files
        seq_len (int): Sequence length for each window
        stride (int): Sliding window stride

    Returns:
        np.ndarray: Array of shape (num_windows, seq_len, feature_dim)
    """
    all_windows = []

    video_files = sorted(os.listdir(directory))
    for file in video_files:
        if not file.endswith(".npy"):
            continue

        file_path = os.path.join(directory, file)
        features = np.load(file_path)  # (num_frames, feature_dim)

        for start in range(0, len(features) - seq_len + 1, stride):
            window = features[start:start + seq_len]
            all_windows.append(window)

    return np.array(all_windows, dtype=np.float32)
