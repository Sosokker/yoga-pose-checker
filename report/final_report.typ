#set text(font: "New Computer Modern", size: 11pt)
#set page(margin: (x: 2.5cm, y: 2.5cm))
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 18pt, weight: "bold")[Yoga Pose Classification with Cross-Modal Attention Keypoint Encoder]
  #v(0.5em)
  #text(size: 12pt)[Final Project Report — Deep Learning]
  #v(0.3em)
  #text(size: 10pt)[May 2026]
]

#outline()

= Project Topic

#strong[What is the project?]

This project builds a yoga pose classification system that takes an RGB image of a person performing a yoga pose, extracts 17 body keypoints using MediaPipe Pose, and classifies the pose into one of five categories: *downdog*, *goddess*, *plank*, *tree*, or *warrior2*.

#strong[Why is this topic interesting?]

Yoga pose classification sits at the intersection of computer vision and human health. A reliable automated classifier could power real-time feedback apps for home practitioners, reduce the need for in-person instructors, and enable large-scale form analysis in fitness research. Unlike generic image classification, yoga poses are defined by specific anatomical relationships — making them an ideal testbed for structured, interpretable deep learning architectures.

#strong[Why did I choose this topic?]

I chose this topic because it allows me to work with a clean, well-defined problem (5 classes, structured keypoint data) while still exploring an interesting architectural idea: cross-modal attention. Instead of treating pose classification as a black-box image classification task, I wanted to design a model that explicitly reasons about *which body joints matter* for each pose — a form of built-in interpretability.

= Why Deep Learning?

#strong[Why does this problem require deep learning?]

While rule-based systems could theoretically classify poses by checking joint angles (e.g., "if knee angle < 90 degrees, it's warrior2"), such hand-crafted rules are brittle. They fail when:
- The person is partially occluded
- The camera angle changes
- Body proportions vary across individuals
- The pose has subtle variations

Deep learning learns these rules *from data*, generalizing across viewpoints, body types, and lighting conditions without explicit programming.

#strong[Comparison with other approaches:]

#table(
  columns: (1fr, 1fr, 1fr),
  inset: 8pt,
  align: horizon,
  table.header([Approach], [Strengths], [Weaknesses]),
  [Hand-crafted rules (joint angles)], [Interpretable, no training data needed], [Brittle, doesn't generalize, hard to maintain],
  [Classical ML (SVM / Random Forest on keypoints)], [Fast training, works with small data], [Limited feature interactions, plateaus on complex poses],
  [CNN on raw images (ResNet, EfficientNet)], [End-to-end, no pose extraction needed], [Requires large datasets, black-box, sensitive to background clutter],
  [Deep learning on keypoints (this project)], [Structured input, lightweight, interpretable attention], [Requires pose detector, may miss pixel-level cues],
)

My approach combines the best of both worlds: the lightweight nature of keypoint-based methods with the representational power of deep learning through attention mechanisms.

= Deep Learning Architecture

#strong[Model: Cross-Modal Pose Classifier]

The architecture consists of four main stages:

```
Input: 17 keypoints x 4 dims = 68 values (x, y, z, visibility)

Stage 1 — Joint Embedding
  Linear(4 -> 128) + ReLU + Dropout + Positional Encoding
  Output: (batch, 17, 128)

Stage 2 — Joint Self-Attention (optional, 2 layers, 4 heads)
  MultiHeadAttention(d_model=128, heads=4) + LayerNorm + Residual
  Each joint attends to all other joints
  Output: (batch, 17, 128)

Stage 3 — Cross-Modal Attention Bridge (2 layers)
  Query = joint tokens (batch, 17, 128)
  Key/Value = learnable class prototypes (batch, 5, 128)
  Attention: softmax(Q @ K^T / sqrt(128)) @ V
  LayerNorm(x + context)
  Output: (batch, 17, 128), attention weights (batch, 17, 5)

Stage 4 — Classification Head
  Mean+Max pooling over 17 joints -> (batch, 256)
  Concatenate with flattened raw keypoints (68)
  Linear(324 -> 128) + ReLU + Dropout
  Linear(128 -> 64) + ReLU + Dropout
  Linear(64 -> 5)
```

#strong[Baseline for comparison:]

A 3-layer MLP: Linear(68 -> 256) + ReLU + Dropout -> Linear(256 -> 128) + ReLU + Dropout -> Linear(128 -> 5)

#strong[Key components:]

- *JointEmbedding*: Projects raw keypoints into a high-dimensional space with learned positional encodings for each joint.
- *JointSelfAttention*: Standard multi-head self-attention. Each joint token attends to all others, modeling anatomical relationships (e.g., knee-ankle-hip).
- *CrossModalAttention*: The core innovation. Instead of pooling and classifying, pose tokens (Queries) attend directly to learnable class prototypes (Keys/Values). This creates an interpretable mapping: "which joints attend to which class?"
- *ClassificationHead*: Mean+max pooling aggregates attended features, and a skip connection from flattened raw keypoints preserves low-level spatial information.

Activation functions: ReLU throughout. Final output is raw logits passed to CrossEntropyLoss.

= Code Structure

The project is organized into modular PyTorch components:

#strong[Data handling:] `src/data/`
- `keypoint_extraction.py` — `MediaPipeKeypointExtractor` class extracts 17 keypoints from images. Compatible with both old and new MediaPipe APIs. Also includes `KeypointAugmenter` for training-time data augmentation (horizontal flip, Gaussian noise, rotation, scaling).
- `dataset.py` — `YogaPoseDataset` loads `.npy` keypoint files. `get_dataloaders()` creates train/val/test splits with seed=42.

#strong[Model definition:] `src/models/cross_modal_attention.py`
- Defines `JointEmbedding`, `JointSelfAttention`, `CrossModalAttention`, `ClassificationHead`, `CrossModalPoseClassifier`, and `BaselineMLP`.
- All in pure PyTorch `nn.Module` classes.

#strong[Training:] `src/training/trainer.py`
- `Trainer` class handles the full training loop: forward pass, backward pass, Adam optimizer, ReduceLROnPlateau scheduler, early stopping, and checkpointing.
- Saves best and last checkpoints automatically.

#strong[Evaluation:] `src/evaluation/evaluator.py`
- `Evaluator` computes test accuracy, top-5 accuracy, macro F1, per-class F1.
- Generates confusion matrix heatmap, attention visualization (which joints attend to which class), and gradient-based joint importance bar chart.

#strong[Entry point:] `src/main.py`
- CLI with argparse. Supports three modes: `extract` (keypoint extraction), `train`, and `eval`.
- All hyperparameters can be overridden from the command line.

#strong[Public code repository:]

The full source code is available at the git repository in the project root. The repository includes all source files, configs, tests, and the Streamlit demo app.

= Training Method

#strong[Dataset:]

- Source: Kaggle — Yoga Poses Dataset (niharika41298/yoga-poses-dataset)
- 1,043 images across 5 classes: downdog, goddess, plank, tree, warrior2
- Split: 70% train (730) / 15% validation (156) / 15% test (157)
- Random seed: 42 (fixed for reproducibility)

#strong[Preprocessing:]

1. MediaPipe Pose extracts 17 body keypoints per image
2. Each keypoint: normalized (x, y, z, visibility)
3. Failed detections are skipped (~5% of images)
4. Training augmentations: horizontal flip (50%), Gaussian noise (std=0.01), random rotation (±5 degrees), random scaling (0.9–1.1x)

#strong[Training hyperparameters (best config):]

#table(
  columns: (1fr, 1fr),
  inset: 6pt,
  table.header([Parameter], [Value]),
  [Optimizer], [Adam],
  [Learning rate], [0.001],
  [Batch size], [64],
  [Weight decay], [0.0001],
  [Max epochs], [80],
  [Early stopping patience], [15],
  [LR scheduler], [ReduceLROnPlateau (factor=0.5, patience=8)],
  [Loss], [CrossEntropyLoss (label smoothing=0.0)],
  [Dropout], [0.2],
  [Device], [Apple Silicon MPS],
)

#strong[Training time:] All models train in under 1 minute on Apple Silicon MPS.

= Evaluation

#strong[Metrics:]

- Top-1 Accuracy
- Top-5 Accuracy
- Macro F1 Score
- Per-class F1 Scores

#strong[Final Results:]

#table(
  columns: (2fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  inset: 6pt,
  align: horizon,
  table.header([Model], [Val Acc], [Test Acc], [Top-5], [Macro F1], [Params]),
  [Baseline MLP], [88.46%], [90.45%], [1.000], [0.904], [51k],
  [Cross-Modal + SA (best test)], [89.74%], [88.54%], [1.000], [0.886], [187k],
  [Cross-Modal (no SA, flat raw)], [88.46%], [87.26%], [1.000], [0.875], [55k],
  [Cross-Modal + Skip + MeanMax], [88.46%], [86.62%], [1.000], [0.871], [188k],
)

#strong[Key finding:] The Cross-Modal model achieves higher validation accuracy (89.74%) than the Baseline MLP (88.46%), but the Baseline generalizes slightly better on the test set (90.45% vs 88.54%). On a 157-sample test set with 5 balanced classes, this 2-point gap corresponds to approximately 3 misclassified samples — not statistically significant.

#strong[Training curves:] Training loss steadily decreases while validation accuracy plateaus around epoch 40–50. Early stopping prevents overfitting.

#strong[Interpretability outputs:] The evaluation generates three visualizations:
1. Confusion matrix heatmap (per-model)
2. Attention visualization — bar charts showing which joints the model attends to for each predicted class
3. Gradient-based joint importance — mean absolute gradients per joint, revealing which body parts most influence predictions

= Related Work

#strong[MediaPipe Pose:]
Google's on-device pose estimation solution (Bazarevsky et al., 2020). Extracts 33 landmarks in real-time. We use a COCO-style subset of 17 landmarks for compatibility with standard pose datasets.

#strong[Attention Is All You Need:]
Vaswani et al. (NeurIPS 2017) introduced the Transformer architecture. Our cross-modal attention adapts this idea: instead of tokens attending to other tokens in the same sequence, pose tokens attend to learnable class prototypes — a form of "query-to-prototype" attention.

#strong[Yoga-82 Dataset:]
A large-scale yoga pose dataset with 82 classes and 28,000+ images (Verma et al., 2020). Our project uses a smaller 5-class subset for rapid experimentation, but the architecture scales naturally to more classes via the prototype matrix.

#strong[Pose-based Action Recognition:]
Works like ST-GCN (Yan et al., AAAI 2018) use Graph Neural Networks on skeletons. Our approach is simpler (attention instead of GNNs) but similarly leverages the structured nature of keypoint data.

= Individual Contribution

#table(
  columns: (1fr, 1fr, 1fr),
  inset: 8pt,
  align: horizon,
  table.header([Student ID], [Name], [Contribution]),
  [6510545730], [Sirin Phungkun], [Main implementor (50%). Led architecture design (Cross-Modal Attention), model implementation, training pipeline, hyperparameter sweeps, and Streamlit demo development. Managed all git commits and repository maintenance.],
  [6510545748], [Sukprachoke Leelapisuth], [Data collection & preprocessing (25%). Responsible for downloading and organizing the Kaggle yoga pose dataset, manual data cleaning (removing corrupted images, verifying class labels), and assisting with MediaPipe keypoint extraction testing across different image qualities.],
  [6510545683], [Phumrapee Chaowanapricha], [Evaluation & documentation (25%). Conducted model evaluation experiments (test set analysis, confusion matrix interpretation), contributed to the final report writing, and assisted with UI/UX feedback for the Streamlit demo interface.],
)

#strong[Note on collaboration:] This project was developed collaboratively with Sirin Phungkun as the primary developer handling all core implementation and git commits. The other two members contributed through data preparation, testing, evaluation analysis, and documentation — work that is reflected in the final deliverables but not captured in commit history.

= Conclusion

This project demonstrates that cross-modal attention on pose keypoints is a viable and interpretable approach to yoga pose classification. While it did not definitively outperform a strong MLP baseline on this small 5-class dataset, it provides unique interpretability (attention maps, gradient-based joint importance) that the black-box MLP cannot. On larger datasets with more classes, the cross-modal architecture's ability to learn class-specific joint relationships should shine more clearly.
