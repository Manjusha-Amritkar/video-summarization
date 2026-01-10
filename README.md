# Reconstruction-Aware Multi-Head Attention for Unsupervised Video Summarization

## Overview

This repository presents an **unsupervised video summarization framework** based on **Bidirectional LSTM (BiLSTM)** and **multi-head self-attention**, trained using **reconstruction error** as the learning signal.

The model learns temporal dependencies in video content by reconstructing CNN-extracted frame features. Frames and segments that are difficult to reconstruct are treated as more **informative**, enabling the generation of compact and meaningful video summaries **without requiring labeled training data**.

The approach is evaluated on benchmark datasets using **user-annotated summaries** and **F1-score–based metrics**.

---

## Methodology

The proposed pipeline consists of the following stages:

### 1. Feature Extraction

* Video frames are processed using a CNN backbone (ResNet).
* Frame-level features are extracted and stored as `.npy` files.
* Feature extraction is performed **offline** and is not included in this repository.

### 2. Temporal Modeling

* Sliding windows are applied to long feature sequences.
* A **Bidirectional LSTM (BiLSTM)** captures forward and backward temporal context.
* **Multi-Head Self-Attention** enhances long-range temporal dependency modeling.

### 3. Importance Estimation

* The model is trained to reconstruct input features using **Mean Squared Error (MSE)** loss.
* Frame-wise **reconstruction error** is used as an importance score.
* Higher reconstruction error ⇒ higher semantic importance.

### 4. Segmentation and Summary Generation

* **Kernel Temporal Segmentation (KTS)** detects scene boundaries.
* A **knapsack optimization** selects the most important segments under a fixed summary length budget (15% of the video duration).

---

## Dataset

This framework is evaluated on standard video summarization benchmarks:

### SumMe Dataset

* 25 consumer videos depicting real-life scenarios
* Multiple user-annotated ground-truth summaries per video
* Average video duration: ~2 minutes
* Evaluation performed using Precision, Recall, and F1-score

Dataset link:
[https://zenodo.org/records/4884870](https://zenodo.org/records/4884870)

### TVSum Dataset (optional extension)

* 50 videos across 10 semantic categories (e.g., sports, news, cooking)
* Each video annotated by 20 users with importance scores (1–5)

Dataset link:
[https://www.kaggle.com/datasets/andreymasyutin/tvsum-dataset](https://www.kaggle.com/datasets/andreymasyutin/tvsum-dataset)

📌 **Note:** Datasets and extracted features are **not included** in this repository due to licensing and storage constraints.

---

## Training

The model is trained in a **fully unsupervised manner** to reconstruct CNN-extracted frame features using reconstruction loss.

Run training using:

```bash
python src/train.py
```

### Configuration

Training configuration (paths, hyperparameters, and output locations) can be modified via:

```
src/config.py
```

This centralized configuration makes the code **machine-independent** and **reproducible** across different systems.

---

## Evaluation

Evaluation follows the standard protocol used in **SumMe** and **TVSum** datasets:

- Reconstruction error → frame importance
- KTS-based temporal segmentation
- Knapsack-based segment selection
- Comparison with user summaries using **F1-score, Precision, and Recall**

Due to differences in annotation formats, **separate evaluation scripts**
are provided for each dataset.

Run evaluation using:

```bash
python src/evaluate_summe.py   # SumMe evaluation
python src/evaluate_tvsum.py   # TVSum evaluation

```

### Evaluation Output

* Per-video evaluation results (per user)
* Aggregate summary metrics
* Results are saved as an **Excel file** in the `results/` directory

---

## Repository Structure

```
video-summarization/
├── src/
│   ├── model.py              # BiLSTM + Multi-Head Attention model
│   ├── feature_extraction.py # Sliding-window feature loader
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # Evaluation & metrics
│   ├── utils.py              # Helper functions
│   └── config.py             # Centralized configuration
├── data/        # datasets & features (not included)
├── results/     # saved models and evaluation outputs
└── README.md
```

---

## Requirements

Key dependencies include:

* TensorFlow / Keras
* NumPy
* SciPy
* Pandas
* Ruptures
* Matplotlib

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Citation

If you use this code or methodology in your research, please cite:

```
@misc{reconstruction_aware_video_summarization,
  title  = {Reconstruction-Aware Multi-Head Attention for Unsupervised Video Summarization},
  author = {Amritkar, Manjusha},
  year   = {2026}
}
```

---

## License

This repository is intended for **academic and research purposes only**.
