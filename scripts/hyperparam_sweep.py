#!/usr/bin/env python3
"""
Focused hyperparameter sweep for Cross-Modal Pose Classifier.
Tests ~8 promising configurations to beat the baseline.
"""
import os
import sys
import json
import shutil
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.dataset import get_dataloaders
from src.models.cross_modal_attention import CrossModalPoseClassifier
from src.training.trainer import Trainer


def run_trial(cfg: dict, trial_id: int, train_loader, val_loader, num_classes: int, device: str) -> dict:
    model = CrossModalPoseClassifier(
        num_classes=num_classes,
        d_model=cfg["d_model"],
        hidden_dim=cfg.get("hidden_dim", 64),
        num_attention_layers=cfg["num_attention_layers"],
        dropout=cfg["dropout"],
        embed_dropout=cfg.get("embed_dropout", 0.1),
        use_joint_self_attention=True,
        num_self_attention_layers=cfg["num_self_attention_layers"],
        num_self_attention_heads=cfg["num_self_attention_heads"],
    )
    n_params = sum(p.numel() for p in model.parameters())

    model_name = f"sweep_trial_{trial_id:03d}"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        device=device,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        max_epochs=100,
        patience=20,
        lr_patience=10,
        lr_factor=0.5,
        label_smoothing=cfg.get("label_smoothing", 0.0),
        model_name=model_name,
        checkpoint_dir="outputs/checkpoints",
        log_dir="outputs/logs",
    )
    history = trainer.fit()

    best_val_acc = trainer.best_val_acc
    return {
        "trial_id": trial_id,
        **cfg,
        "n_parameters": n_params,
        "best_val_acc": best_val_acc,
        "final_val_acc": history["val_acc"][-1],
        "final_val_loss": history["val_loss"][-1],
    }


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    data_dir = "data/processed/keypoints"

    print("Loading data...")
    train_loader_32, val_loader_32, _, class_names = get_dataloaders(
        data_dir=data_dir, batch_size=32, num_workers=0, augmentation=True
    )
    train_loader_16, val_loader_16, _, _ = get_dataloaders(
        data_dir=data_dir, batch_size=16, num_workers=0, augmentation=True
    )
    num_classes = len(class_names)

    # Focused search: 8 configs targeting regularization + capacity reduction
    search_configs = [
        # 1: Halve capacity, increase dropout, add label smoothing
        {"lr": 5e-4, "batch_size": 32, "weight_decay": 5e-4, "d_model": 64, "dropout": 0.5,
         "embed_dropout": 0.2, "label_smoothing": 0.1, "num_self_attention_layers": 1,
         "num_attention_layers": 2, "num_self_attention_heads": 2, "hidden_dim": 64},
        # 2: Even more dropout, stronger weight decay
        {"lr": 5e-4, "batch_size": 32, "weight_decay": 1e-3, "d_model": 64, "dropout": 0.5,
         "embed_dropout": 0.2, "label_smoothing": 0.1, "num_self_attention_layers": 1,
         "num_attention_layers": 2, "num_self_attention_heads": 2, "hidden_dim": 64},
        # 3: Try smaller batch + higher LR
        {"lr": 1e-3, "batch_size": 16, "weight_decay": 5e-4, "d_model": 64, "dropout": 0.5,
         "embed_dropout": 0.2, "label_smoothing": 0.1, "num_self_attention_layers": 1,
         "num_attention_layers": 2, "num_self_attention_heads": 2, "hidden_dim": 64},
        # 4: Single cross-attention layer (reduce capacity further)
        {"lr": 5e-4, "batch_size": 32, "weight_decay": 5e-4, "d_model": 64, "dropout": 0.5,
         "embed_dropout": 0.2, "label_smoothing": 0.1, "num_self_attention_layers": 1,
         "num_attention_layers": 1, "num_self_attention_heads": 2, "hidden_dim": 64},
        # 5: Even smaller hidden_dim
        {"lr": 5e-4, "batch_size": 32, "weight_decay": 5e-4, "d_model": 64, "dropout": 0.5,
         "embed_dropout": 0.2, "label_smoothing": 0.1, "num_self_attention_layers": 1,
         "num_attention_layers": 2, "num_self_attention_heads": 2, "hidden_dim": 32},
        # 6: No self-attention, just cross-modal (like original but regularized)
        {"lr": 5e-4, "batch_size": 32, "weight_decay": 5e-4, "d_model": 64, "dropout": 0.5,
         "embed_dropout": 0.2, "label_smoothing": 0.1, "num_self_attention_layers": 0,
         "num_attention_layers": 2, "num_self_attention_heads": 2, "hidden_dim": 64},
        # 7: Two SA layers but with heavy dropout
        {"lr": 5e-4, "batch_size": 32, "weight_decay": 5e-4, "d_model": 64, "dropout": 0.5,
         "embed_dropout": 0.2, "label_smoothing": 0.1, "num_self_attention_layers": 2,
         "num_attention_layers": 2, "num_self_attention_heads": 2, "hidden_dim": 64},
        # 8: Higher LR with strong regularization
        {"lr": 1e-3, "batch_size": 32, "weight_decay": 1e-3, "d_model": 64, "dropout": 0.5,
         "embed_dropout": 0.2, "label_smoothing": 0.1, "num_self_attention_layers": 1,
         "num_attention_layers": 2, "num_self_attention_heads": 2, "hidden_dim": 64},
    ]

    results = []
    best_val_acc = 0.0
    best_config = None
    best_trial = -1

    for trial_id, cfg in enumerate(search_configs):
        print(f"\n{'='*60}")
        print(f"TRIAL {trial_id + 1}/{len(search_configs)}")
        print(f"{'='*60}")
        print(json.dumps(cfg, indent=2))

        tl = train_loader_16 if cfg["batch_size"] == 16 else train_loader_32
        vl = val_loader_16 if cfg["batch_size"] == 16 else val_loader_32

        result = run_trial(cfg, trial_id, tl, vl, num_classes, device)
        results.append(result)
        print(f"Result: best_val_acc={result['best_val_acc']:.4f} | params={result['n_parameters']:,}")

        if result["best_val_acc"] > best_val_acc:
            best_val_acc = result["best_val_acc"]
            best_config = cfg.copy()
            best_trial = trial_id
            src = os.path.join("outputs", "checkpoints", f"sweep_trial_{trial_id:03d}_best.pt")
            dst = os.path.join("outputs", "checkpoints", "sweep_best.pt")
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"  -> New best! Saved checkpoint to {dst}")

    os.makedirs("outputs/logs", exist_ok=True)
    with open("outputs/logs/sweep_results.json", "w") as f:
        json.dump({
            "best_config": best_config,
            "best_val_acc": best_val_acc,
            "best_trial": best_trial,
            "results": results,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best config: {json.dumps(best_config, indent=2)}")
    print(f"Results saved to outputs/logs/sweep_results.json")


if __name__ == "__main__":
    main()
