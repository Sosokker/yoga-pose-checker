#!/usr/bin/env python3
"""
Download yoga pose datasets.
Supports Kaggle datasets via kagglehub or opendatasets.
Falls back to synthetic data generation if both fail.
"""
import os
import argparse
import shutil
import zipfile
import json
from pathlib import Path
from typing import Optional

try:
    import kagglehub
except ImportError:
    kagglehub = None

# Dataset handles to try (in order of preference)
KAGGLE_HANDLES = [
    "niharika41298/yoga-poses-dataset",
    "tr1gg3r/yoga-pose-classification",
    "ujjwalchowdhury/yoga-pose-classification",
]


def download_kaggle_dataset(handle: str, output_dir: str) -> Optional[str]:
    """Download a Kaggle dataset by handle."""
    if kagglehub is None:
        print("kagglehub not installed, skipping Kaggle download.")
        return None
    print(f"Trying Kaggle dataset: {handle}")
    try:
        path = kagglehub.dataset_download(handle)
        print(f"Downloaded to: {path}")
        return path
    except Exception as e:
        print(f"  Failed: {e}")
        return None


def organize_yoga_dataset(source_dir: str, output_dir: str) -> dict:
    """
    Organize downloaded yoga dataset into class subdirectories.
    Handles common dataset layouts (nested dirs, CSV labels, flat files).
    Returns summary dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Strategy 1: Already has class subdirectories
    subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    if subdirs:
        # Check if subdirs contain images directly
        has_images = any(
            any(f.lower().endswith(image_extensions) for f in os.listdir(os.path.join(source_dir, d)))
            for d in subdirs
        )
        if has_images:
            print("Detected class-subdirectory layout. Copying...")
            for d in subdirs:
                src = os.path.join(source_dir, d)
                dst = os.path.join(output_dir, d)
                if os.path.abspath(src) != os.path.abspath(dst):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
            return summarize_dataset(output_dir)

    # Strategy 2: Flat directory with possible class prefixes
    flat_images = [f for f in os.listdir(source_dir) if f.lower().endswith(image_extensions)]
    if flat_images:
        print("Detected flat directory. Organizing by filename prefixes...")
        classes = set()
        for img in flat_images:
            # Try to extract class from filename (e.g., "warrior_ii_001.jpg")
            parts = img.split("_")
            if len(parts) >= 2:
                cls = "_".join(parts[:-1])
            else:
                cls = "unknown"
            classes.add(cls)
            cls_dir = os.path.join(output_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            shutil.copy2(os.path.join(source_dir, img), os.path.join(cls_dir, img))
        return summarize_dataset(output_dir)

    return {"error": "Could not detect dataset layout", "output_dir": output_dir}


def summarize_dataset(data_dir: str) -> dict:
    """Print and return dataset summary."""
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_counts = {}
    total_images = 0
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        count = len([f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))])
        class_counts[cls] = count
        total_images += count

    summary = {
        "num_classes": len(classes),
        "total_images": total_images,
        "class_counts": class_counts,
        "classes": classes,
    }

    print(f"\n{'='*50}")
    print(f"Dataset Summary: {data_dir}")
    print(f"{'='*50}")
    print(f"Classes: {len(classes)}")
    print(f"Total images: {total_images}")
    print(f"Images per class (top 10):")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cls}: {count}")
    print(f"{'='*50}\n")
    return summary


def generate_synthetic_dataset(output_dir: str, num_classes: int = 20, images_per_class: int = 50) -> dict:
    """
    Generate a synthetic multi-class yoga pose dataset as a fallback.
    Creates simple geometric shape images with labels.
    """
    print(f"\nGenerating synthetic dataset: {num_classes} classes × {images_per_class} images...")
    os.makedirs(output_dir, exist_ok=True)

    import numpy as np
    from PIL import Image, ImageDraw

    pose_names = [
        "warrior_ii", "tree_pose", "downward_dog", "childs_pose", "cobra",
        "mountain_pose", "triangle_pose", "bridge_pose", "plank", "chair_pose",
        "lotus", "camel_pose", "locust_pose", "fish_pose", "boat_pose",
        "crow_pose", "dancer_pose", "hero_pose", "pigeon_pose", "sphinx_pose",
    ][:num_classes]

    np.random.seed(42)
    for i, cls in enumerate(pose_names):
        cls_dir = os.path.join(output_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for j in range(images_per_class):
            img = Image.new("RGB", (224, 224), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)

            # Draw a simple stick figure with class-specific variation
            cx, cy = 112, 112
            scale = 30 + np.random.randint(-10, 10)
            angle_offset = i * (360 / num_classes) + np.random.randint(-15, 15)

            # Head
            draw.ellipse([cx-15, cy-50, cx+15, cy-20], fill=(200, 100, 100), outline=(0,0,0))
            # Body
            draw.line([(cx, cy-20), (cx, cy+40)], fill=(0,0,0), width=3)
            # Arms with class-specific angles
            arm_angle = np.radians(angle_offset)
            draw.line([(cx, cy-10), (cx + int(scale*np.cos(arm_angle)), cy - 10 + int(scale*np.sin(arm_angle)))], fill=(0,0,0), width=3)
            draw.line([(cx, cy-10), (cx - int(scale*np.cos(arm_angle)), cy - 10 + int(scale*np.sin(arm_angle)))], fill=(0,0,0), width=3)
            # Legs
            leg_angle = np.radians(angle_offset + 90)
            draw.line([(cx, cy+40), (cx + int(scale*0.8*np.cos(leg_angle)), cy + 40 + int(scale*0.8*np.sin(leg_angle)))], fill=(0,0,0), width=3)
            draw.line([(cx, cy+40), (cx - int(scale*0.8*np.cos(leg_angle)), cy + 40 + int(scale*0.8*np.sin(leg_angle)))], fill=(0,0,0), width=3)

            # Add random background noise
            arr = np.array(img)
            noise = np.random.randint(-20, 20, arr.shape, dtype=np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

            img.save(os.path.join(cls_dir, f"{cls}_{j:04d}.jpg"))

    print("Synthetic dataset generation complete.")
    return summarize_dataset(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Download or generate yoga pose datasets")
    parser.add_argument("--output", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--manual", type=str, default=None, help="Path to manually downloaded zip/directory")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic dataset generation")
    parser.add_argument("--synthetic-classes", type=int, default=20, help="Number of synthetic classes")
    parser.add_argument("--synthetic-images", type=int, default=50, help="Images per synthetic class")
    args = parser.parse_args()

    # Save summary path
    summary_path = os.path.join("outputs", "dataset_summary.json")
    os.makedirs("outputs", exist_ok=True)

    if args.synthetic:
        summary = generate_synthetic_dataset(
            args.output, num_classes=args.synthetic_classes, images_per_class=args.synthetic_images
        )
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return

    if args.manual:
        print(f"Using manual dataset at: {args.manual}")
        if os.path.isfile(args.manual) and args.manual.endswith(".zip"):
            extract_dir = os.path.join(args.output, "_extracted")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(args.manual, "r") as z:
                z.extractall(extract_dir)
            summary = organize_yoga_dataset(extract_dir, args.output)
        elif os.path.isdir(args.manual):
            summary = organize_yoga_dataset(args.manual, args.output)
        else:
            print(f"Invalid manual path: {args.manual}")
            return
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return

    # Try Kaggle datasets
    downloaded = False
    for handle in KAGGLE_HANDLES:
        path = download_kaggle_dataset(handle, args.output)
        if path:
            summary = organize_yoga_dataset(path, args.output)
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            downloaded = True
            break

    if not downloaded:
        print("\nAll Kaggle downloads failed. Falling back to synthetic dataset generation...")
        summary = generate_synthetic_dataset(
            args.output, num_classes=args.synthetic_classes, images_per_class=args.synthetic_images
        )
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    print("\nDataset preparation complete.")
    print("Next step: run keypoint extraction with:")
    print("  uv run python -m src.main --mode extract")


if __name__ == "__main__":
    main()
