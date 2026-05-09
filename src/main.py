import os
import argparse
import json
import yaml
import torch

from src.data.dataset import get_dataloaders
from src.models.cross_modal_attention import CrossModalPoseClassifier, BaselineMLP
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device(config_device: str) -> str:
    """Auto-resolve device string to available hardware."""
    if config_device == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if config_device == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    return config_device


def main():
    parser = argparse.ArgumentParser(description="Yoga Pose Classification")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "extract"], default="train", help="Mode to run")
    parser.add_argument("--data_dir", type=str, default="data/processed/keypoints", help="Path to processed keypoints")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for eval")
    parser.add_argument("--model", type=str, choices=["cross_modal", "baseline"], default="cross_modal", help="Model type")
    # Hyperparameter overrides for sweep
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--weight_decay", type=float, default=None, help="Override weight decay")
    parser.add_argument("--d_model", type=int, default=None, help="Override d_model")
    parser.add_argument("--dropout", type=float, default=None, help="Override dropout")
    parser.add_argument("--label_smoothing", type=float, default=None, help="Override label smoothing")
    parser.add_argument("--embed_dropout", type=float, default=None, help="Override embed dropout")
    parser.add_argument("--num_self_attention_layers", type=int, default=None, help="Override SA layers")
    parser.add_argument("--num_attention_layers", type=int, default=None, help="Override cross-attn layers")
    parser.add_argument("--num_self_attention_heads", type=int, default=None, help="Override SA heads")
    parser.add_argument("--hidden_dim", type=int, default=None, help="Override classification head hidden dim")
    parser.add_argument("--use_joint_self_attention", type=lambda x: x.lower() in ('true', '1', 'yes'), default=None, help="Override use_joint_self_attention (true/false)")
    parser.add_argument("--use_flatten_raw", type=lambda x: x.lower() in ('true', '1', 'yes'), default=None, help="Flatten raw keypoints into classification head (true/false)")
    parser.add_argument("--model_name", type=str, default=None, help="Override checkpoint/model name")
    args = parser.parse_args()

    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = {}

    # Merge CLI args with config
    data_dir = args.data_dir
    batch_size = args.batch_size if args.batch_size is not None else config.get("batch_size", 32)
    lr = args.lr if args.lr is not None else config.get("lr", 1e-3)
    weight_decay = args.weight_decay if args.weight_decay is not None else config.get("weight_decay", 1e-4)
    max_epochs = config.get("max_epochs", 80)
    patience = config.get("patience", 20)
    lr_patience = config.get("lr_patience", 10)
    lr_factor = config.get("lr_factor", 0.5)
    d_model = args.d_model if args.d_model is not None else config.get("d_model", 64)
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else config.get("hidden_dim", 128)
    num_attention_layers = args.num_attention_layers if args.num_attention_layers is not None else config.get("num_attention_layers", 2)
    dropout = args.dropout if args.dropout is not None else config.get("dropout", 0.3)
    embed_dropout = args.embed_dropout if args.embed_dropout is not None else config.get("embed_dropout", 0.1)
    num_workers = config.get("num_workers", 0)
    device = get_device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    augmentation = config.get("augmentation", True)
    use_joint_self_attention = args.use_joint_self_attention if args.use_joint_self_attention is not None else config.get("use_joint_self_attention", True)
    use_flatten_raw = args.use_flatten_raw if args.use_flatten_raw is not None else config.get("use_flatten_raw", False)
    num_self_attention_layers = args.num_self_attention_layers if args.num_self_attention_layers is not None else config.get("num_self_attention_layers", 2)
    num_self_attention_heads = args.num_self_attention_heads if args.num_self_attention_heads is not None else config.get("num_self_attention_heads", 2)
    label_smoothing = args.label_smoothing if args.label_smoothing is not None else config.get("label_smoothing", 0.1)
    model_name = args.model_name if args.model_name else args.model

    if args.mode == "extract":
        from src.data.keypoint_extraction import MediaPipeKeypointExtractor

        print("Running keypoint extraction...")
        extractor = MediaPipeKeypointExtractor()
        raw_dir = config.get("raw_data_dir", "data/raw")
        output_dir = config.get("output_keypoint_dir", "data/processed/keypoints")

        for pose_class in os.listdir(raw_dir):
            class_dir = os.path.join(raw_dir, pose_class)
            if not os.path.isdir(class_dir):
                continue
            out_class_dir = os.path.join(output_dir, pose_class)
            result = extractor.extract_from_directory(class_dir, out_class_dir, label=pose_class)
            print(f"{pose_class}: {result['extracted']}/{result['total']} extracted")
        return

    # Load data
    print(f"Loading data from {data_dir}...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        augmentation=augmentation,
    )
    num_classes = len(class_names)
    print(f"Classes: {num_classes} | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # Build model
    if args.model == "cross_modal":
        model = CrossModalPoseClassifier(
            num_classes=num_classes,
            d_model=d_model,
            hidden_dim=hidden_dim,
            num_attention_layers=num_attention_layers,
            dropout=dropout,
            embed_dropout=embed_dropout,
            use_joint_self_attention=use_joint_self_attention,
            num_self_attention_layers=num_self_attention_layers,
            num_self_attention_heads=num_self_attention_heads,
            use_flatten_raw=use_flatten_raw,
        )
    else:
        model = BaselineMLP(
            input_dim=68,
            hidden_dims=[256, 128],
            num_classes=num_classes,
            dropout=dropout,
        )

    print(f"Model: {args.model} | Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Config: d_model={d_model}, heads={num_self_attention_heads}, sa_layers={num_self_attention_layers}, "
          f"ca_layers={num_attention_layers}, dropout={dropout}, embed_dropout={embed_dropout}, "
          f"wd={weight_decay}, lr={lr}, bs={batch_size}, ls={label_smoothing}")

    if args.mode == "train":
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
            label_smoothing=label_smoothing,
            model_name=model_name,
        )
        history = trainer.fit()

        # Save training history
        hist_path = os.path.join("outputs", f"{model_name}_history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Saved training history to {hist_path}")

    elif args.mode == "eval":
        if args.checkpoint is None:
            args.checkpoint = f"outputs/checkpoints/{model_name}_best.pt"
        if os.path.exists(args.checkpoint):
            state = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(state["model_state_dict"])
            print(f"Loaded checkpoint from {args.checkpoint}")
        else:
            print(f"Warning: checkpoint not found at {args.checkpoint}, using random weights.")

        evaluator = Evaluator(
            model=model,
            test_loader=test_loader,
            class_names=class_names,
            device=device,
        )
        results = evaluator.evaluate()

        # Print comparison if baseline results exist
        baseline_results_path = os.path.join("outputs", "figures", "BaselineMLP_results.json")
        cross_results_path = os.path.join("outputs", "figures", "CrossModalPoseClassifier_results.json")
        if os.path.exists(baseline_results_path) and os.path.exists(cross_results_path):
            with open(baseline_results_path) as f:
                baseline = json.load(f)
            with open(cross_results_path) as f:
                cross = json.load(f)
            print("\n" + "="*50)
            print("MODEL COMPARISON")
            print("="*50)
            print(f"Cross-Modal Attention:  Top-1 = {cross['test_accuracy']:.4f} | Top-5 = {cross['test_top5_accuracy']:.4f} | F1 = {cross['test_macro_f1']:.4f}")
            print(f"Baseline MLP:           Top-1 = {baseline['test_accuracy']:.4f} | Top-5 = {baseline['test_top5_accuracy']:.4f} | F1 = {baseline['test_macro_f1']:.4f}")
            winner = "Cross-Modal" if cross['test_accuracy'] > baseline['test_accuracy'] else "Baseline"
            print(f"Winner: {winner}")
            print("="*50)


if __name__ == "__main__":
    main()
