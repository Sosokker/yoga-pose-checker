import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.cross_modal_attention import CrossModalPoseClassifier
from src.utils.metrics import accuracy, top_k_accuracy


class Trainer:
    """Training loop with early stopping, LR scheduling, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int,
        device: str = "cuda",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
        patience: int = 10,
        lr_patience: int = 5,
        lr_factor: float = 0.5,
        label_smoothing: float = 0.0,
        checkpoint_dir: str = "outputs/checkpoints",
        log_dir: str = "outputs/logs",
        model_name: str = "cross_modal",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.model_name = model_name

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=lr_factor, patience=lr_patience
        )

        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, model_name))

        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
        self.history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_top5": []}

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for x, y in tqdm(self.train_loader, desc="Training"):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            if isinstance(self.model, CrossModalPoseClassifier):
                logits, _ = self.model(x)
            else:
                logits = self.model(x)

            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)

        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def validate(self) -> tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_preds = []
        all_labels = []

        for x, y in tqdm(self.val_loader, desc="Validation"):
            x, y = x.to(self.device), y.to(self.device)

            if isinstance(self.model, CrossModalPoseClassifier):
                logits, _ = self.model(x)
            else:
                logits = self.model(x)

            loss = self.criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            all_logits.append(logits.cpu())
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

        all_logits = torch.cat(all_logits)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        val_loss = total_loss / len(self.val_loader.dataset)
        val_acc = accuracy(all_preds, all_labels)
        val_top5 = top_k_accuracy(
            all_logits, all_labels, k=5,
            num_classes=self.model.num_classes if hasattr(self.model, 'num_classes') else max(all_labels).item() + 1
        )

        return val_loss, val_acc, val_top5

    def fit(self) -> dict:
        print(f"Training {self.model_name} on {self.device}")
        start_time = time.time()

        for epoch in range(1, self.max_epochs + 1):
            print(f"\nEpoch {epoch}/{self.max_epochs}")
            train_loss = self.train_epoch()
            val_loss, val_acc, val_top5 = self.validate()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_top5"].append(val_top5)

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)
            self.writer.add_scalar("Accuracy/val_top5", val_top5, epoch)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Top-5: {val_top5:.4f}")

            self.scheduler.step(val_loss)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.epochs_no_improve += 1
                self.save_checkpoint(epoch, is_best=False)

            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        elapsed = time.time() - start_time
        self.writer.close()
        print(f"Training completed in {elapsed / 60:.2f} minutes. Best val acc: {self.best_val_acc:.4f}")
        return self.history

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
        }
        path = os.path.join(self.checkpoint_dir, f"{self.model_name}_last.pt")
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pt")
            torch.save(state, best_path)
            print(f"  -> Saved best checkpoint (val_acc={self.best_val_acc:.4f})")
