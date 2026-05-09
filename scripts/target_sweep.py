#!/usr/bin/env python3
"""
Targeted sweep: keep d_model=128 capacity but apply strong regularization.
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

    model_name = f"target_sweep_trial_{trial_id:03d}"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        device=device,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        max_epochs=120,
        patience=25,
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

    # Keep d_model=128 but add strong regularization
    search_configs = [
        # 1: d_model=128, heavy dropout + label smoothing, 1 SA layer
        {"lr": 5e-4, "batch_size": 32, "weight_decay": 5e-4, "d_model": 128, "dropout": 0.5,
         "embed_dropout": 0.2, "label_smoothing": 0.1, "num_self_attention_layers": 1,
         "num_attention_layers": 2, "num_self_attention_heads": 4, "hidden_dim": 64},
        # 2: Same but higher LR + smaller batch
        {"lr": 1e-3, "batch_size": 16, "weight_decay": 5e-4, "d_model": 128, "dropout": 0.5,
         "embed_dropout": 0.2, "label_smoothing": 0.1, "num_self_attention_layers": 1,
         "num_attention_layers": 2, "num_self_attention_heads": 4, "hidden_dim": 64},
        # 3: Even stronger regularization
        {"lr": 5e-4, "batch_size": 32, "weight_decay": 1e-3, "d_model": 128, "dropout": 0.5,
         "embed_dropout": 0.3, "label_smoothing": 0.1, "num_self_attention_layers": 1,
         "num_attention_layers": 2, "num_self_attention_heads": 4, "hidden_dim": 64},
        # 4: Try 2 SA layers but with heavy dropout
        {"lr": 5e-4, "batch_size": 32, "weight_decay": 5e-4, "d_model": 128, "dropout": 0.5,
         "embed_dropout": 0.2, "label_smoothing": 0.1, "num_self_attention_layers": 2,
         "num_attention_layers": 2, "num_self_attention_heads": 4, "hidden_dim": 64},
        # 5: Higher LR, strong WD, 1 SA
        {"lr": 1e-3, "batch_size": 16, "weight_decay": 1e-3, "d_model": 128, "dropout": 0.5,
         "embed_dropout": 0.2, "label_smoothing": 0.1, "num_self_attention_layers": 1,
         "num_attention_layers": 2, "num_self_attention_heads": 4, "hidden_dim": 64},
        # 6: Original config but with label smoothing only
        {"lr": 1e-3, "batch_size": 32, "weight_decay": 1e-4, "d_model": 128, "dropout": 0.3,
         "embed_dropout": 0.1, "label_smoothing": 0.1, "num_self_attention_layers": 2,
         "num_attention_layers": 2, "num_self_attention_heads": 4, "hidden_dim": 64},
        # 7: Try d_model=96 as a middle ground
        {"lr": 5e-4, "batch_size": 32, "weight_decay": 5e-4, "d_model": 96, "dropout": 0.5,
         "embed_dropout": 0.2, "label_smoothing": 0.1, "num_self_attention_layers": 1,
         "num_attention_layers": 2, "num_self_attention_heads": 4, "hidden_dim": 64},
        # 8: d_model=128, no SA, just cross-modal with heavy reg
        {"lr": 5e-4, "batch_size": 32, "weight_decay": 5e-4, "d_model": 128, "dropout": 0.5,
         "embed_dropout": 0.2, "label_smoothing": 0.1, "num_self_attention_layers": 0,
         "num_attention_layers": 2, "num_self_attention_heads": 4, "hidden_dim": 64},
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
            src = os.path.join("outputs", "checkpoints", f"target_sweep_trial_{trial_id:03d}_best.pt")
            dst = os.path.join("outputs", "checkpoints", "target_sweep_best.pt")
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"  -> New best! Saved checkpoint to {dst}")

    os.makedirs("outputs/logs", exist_ok=True)
    with open("outputs/logs/target_sweep_results.json", "w") as f:
        json.dump({
            "best_config": best_config,
            "best_val_acc": best_val_acc,
            "best_trial": best_trial,
            "results": results,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("TARGET SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best config: {json.dumps(best_config, indent=2)}")
    print(f"Results saved to outputs/logs/target_sweep_results.json")


if __name__ == "__main__":
    main()
