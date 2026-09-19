import os
import re
import sys
import shutil
import random
import zipfile
import urllib.request
import argparse
from pathlib import Path
from typing import List, Dict
import io
from PIL import Image

# Microsoft Kaggle Cats and Dogs active dataset URL
DATASET_URL = (
    "https://download.microsoft.com/download/3/E/1/3E1C3F21-"
    "ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
)

# Regex matching PetImages filenames, matching the TensorFlow Datasets implementation
NAME_RE = re.compile(r"^PetImages[\\/](Cat|Dog)[\\/]\d+\.jpg$", re.IGNORECASE)


def download_dataset(download_url: str, dest_zip: Path) -> Path:
    """Downloads dataset zip file using stream chunks with custom User-Agent and progress display."""
    if dest_zip.exists() and dest_zip.stat().st_size > 100_000_000:
        print(f"Archive already downloaded at: {dest_zip} ({dest_zip.stat().st_size / (1024*1024):.1f} MB)")
        return dest_zip

    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Cats vs Dogs dataset from:\n  {download_url}")

    req = urllib.request.Request(
        download_url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )

    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 512  # 512 KB chunks

        with open(dest_zip, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = min(100.0, downloaded * 100.0 / total_size)
                    mb_down = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    sys.stdout.write(f"\rDownloading: {percent:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)")
                else:
                    sys.stdout.write(f"\rDownloading: {downloaded / (1024 * 1024):.1f} MB")
                sys.stdout.flush()

    print("\nDownload complete!")
    return dest_zip


def is_valid_image(image_bytes: bytes) -> bool:
    """
    Validates image data using both header inspection (JFIF magic bytes as in TFDS)
    and Pillow PIL decoding to ensure image is completely readable by PyTorch.
    """
    if len(image_bytes) < 10:
        return False

    # TFDS Check: valid JPEG files in Cats vs Dogs contain b'JFIF' in the initial header bytes
    if b"JFIF" not in image_bytes[:10]:
        return False

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.convert("RGB")
        return True
    except Exception:
        return False


def process_and_split_from_zip(
    zip_path: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    max_samples_per_class: int = None,
    seed: int = 42
):
    """
    Streams and filters images directly from the downloaded ZIP archive into
    train/val/test splits without extracting the entire 824MB archive onto disk twice.
    """
    print(f"Reading archive: {zip_path.name}...")

    cat_members: List[zipfile.ZipInfo] = []
    dog_members: List[zipfile.ZipInfo] = []

    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            match = NAME_RE.match(member.filename)
            if not match:
                continue

            label = match.group(1).lower()
            if label == "cat":
                cat_members.append(member)
            elif label == "dog":
                dog_members.append(member)

        print(f"Found {len(cat_members)} Cat images and {len(dog_members)} Dog images in archive.")

        # Shuffle deterministically
        random.seed(seed)
        random.shuffle(cat_members)
        random.shuffle(dog_members)

        splits_config = [
            ("train", 0.0, train_ratio),
            ("val", train_ratio, train_ratio + val_ratio),
            ("test", train_ratio + val_ratio, 1.0)
        ]

        # Prepare target directories
        for split_name, _, _ in splits_config:
            (output_dir / split_name / "cat").mkdir(parents=True, exist_ok=True)
            (output_dir / split_name / "dog").mkdir(parents=True, exist_ok=True)

        stats: Dict[str, Dict[str, int]] = {
            "cat": {"valid": 0, "corrupted": 0, "train": 0, "val": 0, "test": 0},
            "dog": {"valid": 0, "corrupted": 0, "train": 0, "val": 0, "test": 0}
        }

        for class_name, members in [("cat", cat_members), ("dog", dog_members)]:
            print(f"\nProcessing and filtering {class_name.upper()} images...")
            valid_buffers = []

            for member in members:
                try:
                    data = z.read(member)
                    if is_valid_image(data):
                        valid_buffers.append(data)
                        if max_samples_per_class and len(valid_buffers) >= max_samples_per_class:
                            break
                    else:
                        stats[class_name]["corrupted"] += 1
                except Exception:
                    stats[class_name]["corrupted"] += 1

            total_valid = len(valid_buffers)
            stats[class_name]["valid"] = total_valid
            print(f"  Valid: {total_valid} | Corrupted/Excluded: {stats[class_name]['corrupted']}")

            # Split indices
            train_end = int(total_valid * train_ratio)
            val_end = train_end + int(total_valid * val_ratio)

            split_data = {
                "train": valid_buffers[:train_end],
                "val": valid_buffers[train_end:val_end],
                "test": valid_buffers[val_end:]
            }

            for split_name, items in split_data.items():
                dest_dir = output_dir / split_name / class_name
                stats[class_name][split_name] = len(items)
                for idx, img_bytes in enumerate(items):
                    out_path = dest_dir / f"{class_name}_{idx:05d}.jpg"
                    with open(out_path, "wb") as f_out:
                        f_out.write(img_bytes)

    print("\n" + "=" * 58)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 58)
    print(f"{'Class':<8} | {'Valid':<8} | {'Corrupted':<10} | {'Train':<7} | {'Val':<6} | {'Test':<6}")
    print("-" * 58)
    for cls_name, s in stats.items():
        print(f"{cls_name:<8} | {s['valid']:<8} | {s['corrupted']:<10} | {s['train']:<7} | {s['val']:<6} | {s['test']:<6}")
    print("=" * 58)
    print(f"Output directory ready at: {output_dir.resolve()}\n")


def create_demo_dataset(output_dir: Path, samples_per_class: int = 30):
    """Generates synthetic test images for dry-runs and offline environments."""
    from PIL import ImageDraw
    print(f"Generating synthetic demo dataset in {output_dir} ({samples_per_class} per class)...")
    splits = {
        "train": int(samples_per_class * 0.7),
        "val": int(samples_per_class * 0.15),
        "test": samples_per_class - int(samples_per_class * 0.7) - int(samples_per_class * 0.15)
    }
    colors = {"cat": (220, 100, 80), "dog": (80, 140, 220)}

    for split_name, count in splits.items():
        for class_name, color in colors.items():
            folder = output_dir / split_name / class_name
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                img = Image.new("RGB", (224, 224), color=color)
                draw = ImageDraw.Draw(img)
                draw.text((20, 20), f"{class_name.upper()} #{i}", fill=(255, 255, 255))
                img.save(folder / f"{class_name}_{i:05d}.jpg")
    print("Demo dataset generated successfully!")


def main():
    parser = argparse.ArgumentParser(description="Prepare Cats vs Dogs Dataset with TFDS-grade validation.")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Target output directory")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio (default 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio (default 0.15)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max valid samples per class for quick dry-runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits")
    parser.add_argument("--url", type=str, default=DATASET_URL, help="Download URL for kagglecatsanddogs zip")
    parser.add_argument("--zip-path", type=str, default=None, help="Path to pre-downloaded kagglecatsanddogs.zip")
    parser.add_argument("--demo", action="store_true", help="Generate instant demo dataset without downloading 800MB")

    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    if args.demo:
        create_demo_dataset(out_dir, samples_per_class=args.max_samples or 30)
        return

    # Check for local zip or download
    if args.zip_path and Path(args.zip_path).exists():
        archive_path = Path(args.zip_path)
    else:
        cache_dir = Path("data/_temp_downloads")
        archive_path = cache_dir / "kagglecatsanddogs_5340.zip"
        download_dataset(args.url, archive_path)

    process_and_split_from_zip(
        zip_path=archive_path,
        output_dir=out_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_samples_per_class=args.max_samples,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
