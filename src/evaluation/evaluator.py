import os
import json
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from src.models.cross_modal_attention import CrossModalPoseClassifier, BaselineMLP
from src.utils.metrics import accuracy, top_k_accuracy, per_class_f1, macro_f1


class Evaluator:
    """Evaluation with metrics, confusion matrix, and attention visualization."""

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        class_names: list[str],
        device: str = "cuda",
        output_dir: str = "outputs/figures",
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.num_classes = len(class_names)

    def evaluate(self) -> dict:
        self.model.eval()
        all_logits = []
        all_preds = []
        all_labels = []
        all_attn = []  # Store attention from first batch for viz

        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                x, y = x.to(self.device), y.to(self.device)

                if isinstance(self.model, CrossModalPoseClassifier):
                    logits, attn = self.model(x)
                    if i == 0:
                        all_attn = attn  # list of (batch, 17, num_classes)
                else:
                    logits = self.model(x)

                preds = logits.argmax(dim=1)
                all_logits.append(logits.cpu())
                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())

        all_logits = torch.cat(all_logits)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        test_acc = accuracy(all_preds, all_labels)
        test_top5 = top_k_accuracy(all_logits, all_labels, k=5, num_classes=self.num_classes)
        test_f1 = macro_f1(all_preds, all_labels, self.num_classes)
        per_class = per_class_f1(all_preds, all_labels, self.num_classes)

        results = {
            "test_accuracy": test_acc,
            "test_top5_accuracy": test_top5,
            "test_macro_f1": test_f1,
            "per_class_f1": per_class,
        }

        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test Top-5 Accuracy: {test_top5:.4f}")
        print(f"Test Macro F1: {test_f1:.4f}")

        # Save results
        results_path = os.path.join(self.output_dir, f"{self.model.__class__.__name__}_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Confusion matrix
        self.plot_confusion_matrix(all_preds, all_labels)

        # Attention visualization (only for cross-modal model)
        if isinstance(self.model, CrossModalPoseClassifier) and len(all_attn) > 0:
            self.visualize_attention(all_attn, all_preds[: len(all_attn[0])])
            self.plot_joint_importance()

        return results

    def plot_confusion_matrix(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """Plot and save confusion matrix heatmap."""
        cm = confusion_matrix(labels.numpy(), preds.numpy(), labels=list(range(self.num_classes)))

        plt.figure(figsize=(max(12, self.num_classes // 3), max(10, self.num_classes // 3)))
        sns.heatmap(
            cm,
            annot=False,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - {self.model.__class__.__name__}")
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{self.model.__class__.__name__}_confusion_matrix.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved confusion matrix to {save_path}")

    def visualize_attention(
        self,
        all_attn: list[torch.Tensor],
        preds: torch.Tensor,
        num_samples: int = 5,
    ) -> None:
        """
        Visualize attention weights from the first attention layer.
        Shows which joints attended most to which class prototypes.
        """
        attn = all_attn[0]  # first layer: (batch, 17, num_classes)
        batch_size = min(num_samples, attn.size(0))

        joint_names = [
            "nose", "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
            "L_wrist", "R_wrist", "L_hip", "R_hip", "L_knee",
            "R_knee", "L_ankle", "R_ankle", "L_eye", "R_eye",
            "L_ear", "R_ear",
        ]

        fig, axes = plt.subplots(batch_size, 1, figsize=(14, 2.5 * batch_size))
        if batch_size == 1:
            axes = [axes]

        for i in range(batch_size):
            ax = axes[i]
            # For each sample, show attention distribution across classes per joint
            attn_sample = attn[i].cpu().numpy()  # (17, num_classes)
            pred_class = preds[i].item()

            # Show attention to the predicted class for each joint
            attn_to_pred = attn_sample[:, pred_class]  # (17,)

            bars = ax.bar(range(17), attn_to_pred, color="steelblue")
            ax.set_xticks(range(17))
            ax.set_xticklabels(joint_names, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Attention Weight")
            ax.set_title(f"Sample {i+1}: Predicted = {self.class_names[pred_class]}")
            ax.set_ylim(0, attn_to_pred.max() * 1.2 + 1e-6)

            # Highlight top-3 attended joints
            top3 = np.argsort(attn_to_pred)[-3:]
            for j in top3:
                bars[j].set_color("coral")

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "attention_visualization.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved attention visualization to {save_path}")

    def plot_joint_importance(self, num_batches: int = 10) -> None:
        """
        Compute gradient-based joint importance by backpropagating
        the loss w.r.t. input keypoints and averaging absolute gradients.
        """
        if not isinstance(self.model, CrossModalPoseClassifier):
            return

        self.model.train()  # need gradients
        joint_names = [
            "nose", "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
            "L_wrist", "R_wrist", "L_hip", "R_hip", "L_knee",
            "R_knee", "L_ankle", "R_ankle", "L_eye", "R_eye",
            "L_ear", "R_ear",
        ]

        all_grads = []
        criterion = nn.CrossEntropyLoss()

        for i, (x, y) in enumerate(self.test_loader):
            if i >= num_batches:
                break
            x = x.to(self.device).requires_grad_(True)
            y = y.to(self.device)

            self.model.zero_grad()
            logits, _ = self.model(x)
            loss = criterion(logits, y)
            loss.backward()

            if x.grad is not None:
                # Average absolute gradient per joint across batch
                grad = x.grad.abs().mean(dim=0).cpu().numpy()  # (17, 4)
                # Mean across the 4 dimensions (x, y, z, visibility)
                joint_importance = grad.mean(axis=1)
                all_grads.append(joint_importance)

        if not all_grads:
            return

        mean_importance = np.mean(all_grads, axis=0)

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(range(17), mean_importance, color="teal")
        top3 = np.argsort(mean_importance)[-3:]
        for j in top3:
            bars[j].set_color("coral")
        ax.set_xticks(range(17))
        ax.set_xticklabels(joint_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean Absolute Gradient")
        ax.set_title("Gradient-Based Joint Importance")
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "joint_importance.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved joint importance plot to {save_path}")
