"""Quick smoke test for model forward pass and training step."""
import torch
import numpy as np
from src.models.cross_modal_attention import CrossModalPoseClassifier, BaselineMLP, JointEmbedding, CrossModalAttention
from src.data.dataset import YogaPoseDataset
from src.data.keypoint_extraction import KeypointAugmenter
from torch.utils.data import DataLoader


def test_joint_embedding():
    emb = JointEmbedding(input_dim=4, d_model=128, num_joints=17)
    x = torch.randn(4, 17, 4)
    out = emb(x)
    assert out.shape == (4, 17, 128), f"Expected (4, 17, 128), got {out.shape}"
    print("✓ JointEmbedding")


def test_cross_modal_attention():
    attn = CrossModalAttention(d_model=128, num_classes=10)
    x = torch.randn(4, 17, 128)
    out, weights = attn(x)
    assert out.shape == (4, 17, 128), f"Expected (4, 17, 128), got {out.shape}"
    assert weights.shape == (4, 17, 10), f"Expected (4, 17, 10), got {weights.shape}"
    print("✓ CrossModalAttention")


def test_full_model():
    model = CrossModalPoseClassifier(num_classes=10, d_model=128, hidden_dim=64, num_attention_layers=2)
    x = torch.randn(8, 17, 4)
    logits, all_attn = model(x)
    assert logits.shape == (8, 10), f"Expected (8, 10), got {logits.shape}"
    assert len(all_attn) == 2, f"Expected 2 attention layers, got {len(all_attn)}"
    print("✓ CrossModalPoseClassifier")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")


def test_baseline_mlp():
    model = BaselineMLP(input_dim=68, hidden_dims=[256, 128], num_classes=10)
    x = torch.randn(8, 17, 4)
    logits = model(x)
    assert logits.shape == (8, 10), f"Expected (8, 10), got {logits.shape}"
    print("✓ BaselineMLP")


def test_augmenter():
    aug = KeypointAugmenter()
    kp = np.random.rand(17, 4).astype(np.float32)
    kp_aug = aug(kp)
    assert kp_aug.shape == (17, 4)
    assert np.all(kp_aug[:, 0] >= 0) and np.all(kp_aug[:, 0] <= 1)
    assert np.all(kp_aug[:, 1] >= 0) and np.all(kp_aug[:, 1] <= 1)
    print("✓ KeypointAugmenter")


def test_training_step():
    model = CrossModalPoseClassifier(num_classes=5, d_model=64, hidden_dim=32, num_attention_layers=1)
    x = torch.randn(4, 17, 4)
    y = torch.randint(0, 5, (4,))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    logits, _ = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0
    print(f"✓ Training step (loss={loss.item():.4f})")


def test_dataset():
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create fake data
        for cls in ["warrior", "tree"]:
            cls_dir = os.path.join(tmpdir, cls)
            os.makedirs(cls_dir)
            for i in range(5):
                np.save(os.path.join(cls_dir, f"sample_{i}.npy"), np.random.rand(17, 4).astype(np.float32))

        ds = YogaPoseDataset(tmpdir, augmentation=True)
        assert len(ds) == 10
        x, y = ds[0]
        assert x.shape == (17, 4)
        assert isinstance(y, int)
        print("✓ YogaPoseDataset")


def test_end_to_end():
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        for cls in ["warrior", "tree", "dog"]:
            cls_dir = os.path.join(tmpdir, cls)
            os.makedirs(cls_dir)
            for i in range(10):
                np.save(os.path.join(cls_dir, f"sample_{i}.npy"), np.random.rand(17, 4).astype(np.float32))

        from src.data.dataset import get_dataloaders
        train_loader, val_loader, test_loader, class_names = get_dataloaders(
            data_dir=tmpdir, batch_size=4, num_workers=0
        )
        assert len(class_names) == 3
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0
        print("✓ get_dataloaders")

        model = CrossModalPoseClassifier(num_classes=3, d_model=32, hidden_dim=16, num_attention_layers=1)
        from src.training.trainer import Trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=3,
            device="cpu",
            max_epochs=2,
            patience=5,
            model_name="test",
        )
        history = trainer.fit()
        assert len(history["train_loss"]) == 2
        print("✓ End-to-end training")


if __name__ == "__main__":
    print("Running smoke tests...\n")
    test_joint_embedding()
    test_cross_modal_attention()
    test_full_model()
    test_baseline_mlp()
    test_augmenter()
    test_training_step()
    test_dataset()
    test_end_to_end()
    print("\n🎉 All tests passed!")
