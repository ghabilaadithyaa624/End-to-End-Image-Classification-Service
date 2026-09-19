import os
import sys
import shutil
import random
import zipfile
import urllib.request
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image

# Microsoft Kaggle Cats and Dogs dataset URL
DATASET_URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-EC20-4492-9A4E-0482386A45F3/kagglecatsanddogs_5340.zip"


def download_progress_hook(block_num: int, block_size: int, total_size: int):
    """Prints a clean CLI progress indicator during download."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\rDownloading dataset: {percent:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)")
    else:
        sys.stdout.write(f"\rDownloading dataset: {downloaded / (1024 * 1024):.1f} MB")
    sys.stdout.flush()


def download_dataset(download_url: str, dest_zip: Path) -> Path:
    """Downloads dataset zip file if not already present."""
    if dest_zip.exists() and dest_zip.stat().st_size > 1000000:
        print(f"Archive already exists at {dest_zip}. Skipping download.")
        return dest_zip

    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Cats vs Dogs dataset from:\n  {download_url}")
    urllib.request.urlretrieve(download_url, dest_zip, reporthook=download_progress_hook)
    print("\nDownload complete!")
    return dest_zip


def extract_dataset(zip_path: Path, extract_to: Path) -> Path:
    """Extracts the PetImages directory from the downloaded zip file."""
    print(f"Extracting {zip_path.name} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")
    return extract_to


def is_valid_image(image_path: Path) -> bool:
    """Verifies image integrity and confirms Pillow can decode RGB channels."""
    if not image_path.is_file() or image_path.stat().st_size == 0:
        return False
    try:
        # First verification check
        with Image.open(image_path) as img:
            img.verify()
        # Second decode check (Pillow verify leaves file pointer at EOF)
        with Image.open(image_path) as img:
            img.convert("RGB")
        return True
    except Exception:
        return False


def collect_and_clean_images(class_dir: Path, max_samples: int = None) -> Tuple[List[Path], int]:
    """Scans class directory, validates all images, and excludes corrupted files."""
    valid_images: List[Path] = []
    corrupted_count: int = 0

    all_files = [f for f in class_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
    print(f"Scanning {len(all_files)} files in {class_dir.name}...")

    for f in all_files:
        if is_valid_image(f):
            valid_images.append(f)
            if max_samples and len(valid_images) >= max_samples:
                break
        else:
            corrupted_count += 1

    return valid_images, corrupted_count


def split_dataset(
    items: List[Path],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Shuffles and splits items into train, validation, and test subsets."""
    random.seed(seed)
    shuffled = list(items)
    random.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_set = shuffled[:train_end]
    val_set = shuffled[train_end:val_end]
    test_set = shuffled[val_end:]

    return train_set, val_set, test_set


def copy_files(file_list: List[Path], target_dir: Path, class_name: str):
    """Copies a list of image files into the destination class folder."""
    dest_folder = target_dir / class_name
    dest_folder.mkdir(parents=True, exist_ok=True)
    for idx, src_file in enumerate(file_list):
        dest_file = dest_folder / f"{class_name}_{idx:05d}{src_file.suffix.lower()}"
        shutil.copy2(src_file, dest_file)


def prepare_cats_vs_dogs(
    raw_source_dir: Path = None,
    output_dir: Path = Path("data/raw"),
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    max_samples_per_class: int = None,
    seed: int = 42,
    download_url: str = DATASET_URL
):
    """Main orchestration pipeline to download, clean, split, and save the dataset."""
    temp_dir = Path("data/_temp_downloads")

    # Step 1: Obtain source images (download if no local directory specified)
    if raw_source_dir is None or not raw_source_dir.exists():
        zip_path = temp_dir / "kagglecatsanddogs.zip"
        download_dataset(download_url, zip_path)
        extracted_path = temp_dir / "extracted"
        if not (extracted_path / "PetImages").exists():
            extract_dataset(zip_path, extracted_path)
        source_pet_images = extracted_path / "PetImages"
    else:
        source_pet_images = raw_source_dir

    if not source_pet_images.exists():
        # Look for PetImages subfolder if user passed outer folder
        if (source_pet_images / "PetImages").exists():
            source_pet_images = source_pet_images / "PetImages"
        else:
            raise FileNotFoundError(f"Could not locate 'PetImages' folder in {source_pet_images}")

    classes = {"cat": ["Cat", "cat"], "dog": ["Dog", "dog"]}
    stats: Dict[str, Dict[str, int]] = {}

    # Target split folders
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    for standard_label, possible_names in classes.items():
        class_folder = None
        for name in possible_names:
            candidate = source_pet_images / name
            if candidate.exists():
                class_folder = candidate
                break

        if not class_folder:
            raise FileNotFoundError(f"Missing folder for {standard_label} in {source_pet_images}")

        print(f"\nProcessing class: {standard_label.upper()} from {class_folder}...")
        valid_files, corrupted = collect_and_clean_images(class_folder, max_samples=max_samples_per_class)
        print(f"  Valid: {len(valid_files)} | Corrupted/Excluded: {corrupted}")

        train_files, val_files, test_files = split_dataset(
            valid_files,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed
        )

        print(f"  Splits -> Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        copy_files(train_files, train_dir, standard_label)
        copy_files(val_files, val_dir, standard_label)
        copy_files(test_files, test_dir, standard_label)

        stats[standard_label] = {
            "valid": len(valid_files),
            "corrupted": corrupted,
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        }

    # Clean up temp downloads if created
    if temp_dir.exists() and raw_source_dir is None:
        print(f"\nCleaning up temporary download cache at {temp_dir}...")
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n" + "=" * 55)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 55)
    print(f"{'Class':<8} | {'Valid':<8} | {'Corrupted':<10} | {'Train':<7} | {'Val':<6} | {'Test':<6}")
    print("-" * 55)
    for cls_name, s in stats.items():
        print(f"{cls_name:<8} | {s['valid']:<8} | {s['corrupted']:<10} | {s['train']:<7} | {s['val']:<6} | {s['test']:<6}")
    print("=" * 55)
    print(f"Output directory ready at: {output_dir.resolve()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download, validate, and split Cats vs Dogs dataset.")
    parser.add_argument("--source-dir", type=str, default=None, help="Local path to PetImages if already downloaded")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Target output dataset directory")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train partition ratio (default 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation partition ratio (default 0.15)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max valid samples per class (for testing/dry-runs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--url", type=str, default=DATASET_URL, help="URL to download dataset zip from")

    args = parser.parse_args()

    prepare_cats_vs_dogs(
        raw_source_dir=Path(args.source_dir) if args.source_dir else None,
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_samples_per_class=args.max_samples,
        seed=args.seed,
        download_url=args.url
    )
