import torch
from typing import Optional


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute top-1 accuracy."""
    correct = (preds == labels).sum().item()
    return correct / len(labels)


def top_k_accuracy(
    logits_or_preds: torch.Tensor,
    labels: torch.Tensor,
    k: int = 5,
    num_classes: Optional[int] = None,
) -> float:
    """
    Compute top-k accuracy.
    If logits_or_preds is 1D (already argmaxed), we can't do top-k; return top-1.
    If 2D (logits), compute proper top-k.
    """
    if logits_or_preds.dim() == 1:
        # Already predictions
        if k == 1:
            return accuracy(logits_or_preds, labels)
        # For k > 1 with only preds, fallback to top-1
        return accuracy(logits_or_preds, labels)

    # logits: (batch, num_classes)
    num_classes_actual = logits_or_preds.size(1)
    k = min(k, num_classes_actual)
    _, top_k_preds = logits_or_preds.topk(k, dim=1)
    correct = top_k_preds.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
    return correct / len(labels)


def per_class_f1(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    """Compute per-class F1 score."""
    from sklearn.metrics import f1_score
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    f1s = f1_score(labels_np, preds_np, average=None, labels=list(range(num_classes)), zero_division=0)
    return {f"class_{i}": f1s[i] for i in range(num_classes)}


def macro_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    from sklearn.metrics import f1_score
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    return f1_score(labels_np, preds_np, average="macro", labels=list(range(num_classes)), zero_division=0)
