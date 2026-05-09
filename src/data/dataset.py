import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, Callable
from .keypoint_extraction import KeypointAugmenter


class YogaPoseDataset(Dataset):
    """Dataset for pre-extracted keypoint .npy files."""

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        augmentation: bool = False,
    ):
        """
        Args:
            data_dir: Root directory with subdirectories per class containing .npy files.
            transform: Optional transform to apply.
            augmentation: Whether to apply keypoint augmentations.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.augmentation = augmentation
        self.augmenter = KeypointAugmenter() if augmentation else None

        self.samples = []  # list of (npy_path, label_idx)
        self.classes = sorted(
            [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.endswith(".npy"):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        npy_path, label = self.samples[idx]
        keypoints = np.load(npy_path).astype(np.float32)  # (17, 4)

        if self.augmentation and self.augmenter is not None:
            keypoints = self.augmenter(keypoints)

        if self.transform:
            keypoints = self.transform(keypoints)

        keypoints = torch.from_numpy(keypoints)
        return keypoints, label


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 4,
    augmentation: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Create train/val/test dataloaders with given split ratios.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    full_dataset = YogaPoseDataset(data_dir, augmentation=False)
    class_names = full_dataset.classes
    n_total = len(full_dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # Enable augmentation only for training
    train_ds.dataset = YogaPoseDataset(data_dir, augmentation=augmentation)
    train_indices = train_ds.indices
    train_ds = torch.utils.data.Subset(train_ds.dataset, train_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_names
