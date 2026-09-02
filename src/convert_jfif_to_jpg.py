from pathlib import Path
from PIL import Image

RAW_DIR = Path("data/raw")


def convert_jfif_to_jpeg(directory: Path):
    converted = 0
    skipped = 0
    failed = 0

    for file in sorted(directory.iterdir()):
        if not file.is_file():
            continue

        extension = file.suffix.lower()

        if extension in [".jpeg", ".jpg", ".png"]:
            print(f"SKIP: {file.name}")
            skipped += 1
            continue

        if extension == ".jfif":
            output_file = file.with_suffix(".jpeg")

            try:
                with Image.open(file) as image:
                    image = image.convert("RGB")
                    image.save(output_file, "JPEG", quality=95)

                file.unlink()

                print(f"CONVERTED: {file.name} -> {output_file.name}")
                converted += 1

            except Exception as e:
                print(f"ERROR: {file.name} -> {e}")
                failed += 1

    print("\nDone.")
    print(f"Converted: {converted}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")


if __name__ == "__main__":
    if not RAW_DIR.exists():
        print(f"Directory does not exist: {RAW_DIR}")
    else:
        convert_jfif_to_jpeg(RAW_DIR)