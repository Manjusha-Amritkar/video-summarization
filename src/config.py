"""
Central configuration file for video summarization project
"""

import os

# =======================
# Dataset paths
# =======================

# Base directory for extracted CNN features (.npy)
BASE_FEATURE_PATH = os.getenv(
    "FEATURE_BASE_PATH",
    "data/features"
)

TRAIN_PATH = os.path.join(BASE_FEATURE_PATH, "train")
VALID_PATH = os.path.join(BASE_FEATURE_PATH, "valid")
TEST_PATH = os.path.join(BASE_FEATURE_PATH, "test")

# Ground-truth annotations (SumMe .mat files)
MAT_DIR = os.getenv(
    "ANNOTATION_PATH",
    "data/annotations"
)

# =======================
# Model & training config
# =======================

FEATURE_DIM = 2048
SEQ_LEN = 150
STRIDE = 125

NUM_EPOCHS = 50
BATCH_SIZE = 1
LEARNING_RATE = 1e-5

# =======================
# Evaluation config
# =======================

FPS = 25
SUMMARY_RATIO = 0.15
KTS_PENALTY = 500
CHUNK_SIZE = 500

# =======================
# Output paths
# =======================

MODEL_SAVE_PATH = os.getenv(
    "MODEL_SAVE_PATH",
    "results/models/video_summarization.keras"
)

EVAL_RESULTS_PATH = os.getenv(
    "EVAL_RESULTS_PATH",
    "results/evaluation_results.xlsx"
)
