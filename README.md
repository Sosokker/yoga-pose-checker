# Yoga Pose Classification with Cross-Modal Attention Keypoint Encoder

A lightweight deep learning project for yoga pose classification using **MediaPipe keypoint extraction** and a custom **Cross-Modal Attention** architecture.

## Architecture Overview

```
RGB Image → MediaPipe Pose → 17 Keypoints (x, y, z, visibility)
                                    ↓
                          Joint Embedding (4 → 128) + Positional Encoding
                                    ↓
                     [Optional Joint Self-Attention (2 layers, 4 heads)]
                                    ↓
                     Cross-Modal Attention Bridge (2 layers)
                     (Pose tokens Q attend to learnable Class prototype K,V)
                                    ↓
                     Mean+Max Pool + Optional Raw Keypoint Skip
                                    ↓
                          Classification Head (128 → 64 → 5)
                                    ↓
                             Pose Class Prediction
```

## Final Results

| Model | Val Accuracy | Test Accuracy | Top-5 | Macro F1 | Params |
|-------|-------------|---------------|-------|----------|--------|
| **Baseline MLP** (68→256→128→5) | 88.46% | **90.45%** | 1.000 | 0.904 | 51k |
| **Cross-Modal + SA + Flatten Raw** (best test) | **89.74%** | 88.54% | 1.000 | 0.886 | 187k |
| Cross-Modal + Skip + MeanMax | 88.46% | 86.62% | 1.000 | 0.871 | 188k |
| Cross-Modal (no SA, flatten raw) | 88.46% | 87.26% | 1.000 | 0.875 | 55k |
| Cross-Modal (no SA, no skip) | 90.38% | 85.99% | 1.000 | 0.866 | 56k |

**Key Finding:** The Cross-Modal Attention model achieves higher validation accuracy (89.74%) than the Baseline MLP (88.46%), but the Baseline generalizes slightly better on the test set (90.45% vs 88.54%). On a 157-sample test set with 5 balanced classes, this 2-point gap corresponds to approximately 3 misclassified samples — **not statistically significant**.

### Why Cross-Modal Attention Still Matters

Even though it does not beat the MLP on this small dataset, the cross-modal architecture provides **interpretability** that the black-box MLP cannot:

- **Attention Maps**: Visualize which body joints the model attends to for each pose class (see `outputs/figures/attention_visualization.png`)
- **Gradient-Based Joint Importance**: Identify which joints contribute most to the model's decision (see `outputs/figures/joint_importance.png`)
- **Class Prototypes**: Learnable class embeddings act as interpretable anchors in the attention mechanism

## Dataset

- **Source**: Kaggle — [Yoga Poses Dataset](https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset)
- **Size**: 1,043 images across 5 classes
- **Classes**: `downdog`, `goddess`, `plank`, `tree`, `warrior2`
- **Split**: 70% train (730) / 15% validation (156) / 15% test (157), seeded at 42

## Project Structure

```
.
├── configs/
│   └── default.yaml          # Training hyperparameters
├── data/
│   ├── raw/                  # Raw yoga pose images (organized by class)
│   └── processed/
│       └── keypoints/        # Extracted 17×4 keypoint tensors (.npy)
├── notebooks/
│   └── training.ipynb        # Colab-friendly notebook
├── outputs/
│   ├── checkpoints/          # Model checkpoints (.pt)
│   ├── logs/                 # TensorBoard logs + sweep results
│   └── figures/              # Confusion matrices, attention viz, joint importance
├── scripts/
│   ├── download_data.py      # Dataset download helper
│   ├── hyperparam_sweep.py   # Random hyperparameter sweep
│   └── target_sweep.py       # Focused sweep on key hyperparameters
├── src/
│   ├── data/
│   │   ├── keypoint_extraction.py   # MediaPipe extractor + augmentations
│   │   └── dataset.py               # PyTorch Dataset / DataLoaders
│   ├── models/
│   │   └── cross_modal_attention.py # Cross-Modal + Baseline MLP
│   ├── training/
│   │   └── trainer.py               # Training loop with early stopping
│   ├── evaluation/
│   │   └── evaluator.py             # Metrics, confusion matrix, attention viz
│   ├── utils/
│   │   └── metrics.py               # Accuracy, Top-K, F1 helpers
│   └── main.py                      # CLI entry point
├── app.py                    # Streamlit demo application
├── pyproject.toml
├── tests/
│   └── test_smoke.py         # Smoke tests for model forward passes
└── README.md
```

## Setup (with uv)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
uv pip install -e ".[dev]"
```

## Workflow

### 1. Download Dataset

```bash
# Option A: Automatic (requires Kaggle API credentials)
uv run python scripts/download_data.py

# Option B: Manual — download from Kaggle and point to it
uv run python scripts/download_data.py --manual /path/to/dataset.zip
```

The expected raw structure is:
```
data/raw/
  ├── downdog/
  ├── goddess/
  ├── plank/
  ├── tree/
  └── warrior2/
```

### 2. Extract Keypoints (offline, one-time)

```bash
uv run python -m src.main --mode extract
```

This runs MediaPipe Pose on every image and saves `17 × 4` tensors as `.npy` files.

### 3. Train

```bash
# Cross-Modal Attention Model (your design)
uv run python -m src.main --mode train --model cross_modal

# Baseline MLP (for comparison)
uv run python -m src.main --mode train --model baseline
```

You can override any hyperparameter from the CLI:
```bash
uv run python -m src.main --mode train --model cross_modal \
  --d_model 128 --dropout 0.2 --batch_size 64 --label_smoothing 0.0
```

### 4. Evaluate

```bash
# Cross-modal evaluation
uv run python -m src.main --mode eval --model cross_modal \
  --checkpoint outputs/checkpoints/cross_modal_full_bs64_best.pt

# Baseline evaluation
uv run python -m src.main --mode eval --model baseline \
  --checkpoint outputs/checkpoints/baseline_best.pt
```

This produces:
- Test accuracy & top-5 accuracy
- Macro F1 score + per-class F1
- Confusion matrix heatmap (`outputs/figures/*_confusion_matrix.png`)
- Attention visualization (`outputs/figures/attention_visualization.png`)
- Gradient-based joint importance (`outputs/figures/joint_importance.png`)

### 5. Streamlit Demo

```bash
uv run streamlit run app.py
```

Upload a yoga pose image to get:
- Top-3 predictions with confidence scores
- Attention bar chart showing which joints the model focused on
- Keypoint overlay on the original image

### 6. Monitor Training

```bash
uv run tensorboard --logdir outputs/logs
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| MediaPipe preprocessing | Fast, lightweight; model trains on 17×4 vectors instead of raw pixels |
| Cross-modal attention | Class prototypes act as Keys/Values; joints attend to pose classes directly |
| Joint self-attention | Models anatomical relationships (e.g., knee-ankle-hip) before cross-modal bridge |
| Mean+max pooling + raw skip | Aggregates joint features and preserves low-level keypoint signal |
| ~50–190k parameters | Lightweight; trains in under 1 minute on Apple Silicon MPS |
| Keypoint augmentation | Simulates detector noise, camera distance, and left-right mirroring |

## Extending the Project

- **More classes**: The architecture scales naturally with `num_classes` via the prototype matrix.
- **Temporal extension**: Replace mean pooling with a small Transformer over video frames.
- **Graph structure**: Use a GNN over the 17 joints to explicitly encode the skeleton graph.
- **Larger dataset**: Test on Yoga-82 (82 classes, 28k images) where cross-modal attention should show clearer advantages.

## Citation / References

- MediaPipe Pose: [Google AI Blog](https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html)
- Yoga-82 Dataset: [Project Page](https://sites.google.com/view/yoga-82/home)
- Vaswani et al., "Attention Is All You Need", NeurIPS 2017
