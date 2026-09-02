import os
from pathlib import Path

RAW_DIR = Path("data/raw")

def rename_sequentially(directory: Path, prefix: str = "img"):
    files = sorted([f for f in directory.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".jfif"]])
    for i, f in enumerate(files, start=1):
        new_name = directory / f"{prefix}_{i:03d}{f.suffix.lower()}"
        f.rename(new_name)
    print(f"{len(files)} bestanden hernoemd in {directory}")

if __name__ == "__main__":
    rename_sequentially(RAW_DIR)