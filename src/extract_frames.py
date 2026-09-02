# src/extract_frames.py
#
# Extracts frames from recorded gameplay video(s), spaced out in time and
# filtered to skip near-duplicate frames, so you get diverse screenshots
# fast instead of manually taking them one by one.
#
# Usage:
#   python src/extract_frames.py path\to\video.mp4
#   python src/extract_frames.py path\to\folder_with_videos
#
# Requires: opencv-python (already in requirements.txt)

import cv2
import sys
from pathlib import Path

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECONDS_BETWEEN_FRAMES = 3       # minimum time gap between saved frames
SIMILARITY_THRESHOLD = 0.97      # skip frame if too similar to the last saved one (0-1, higher = stricter)


def frame_similarity(frame_a, frame_b) -> float:
    """Quick similarity check using histogram comparison (fast, good enough for dedup)."""
    hist_a = cv2.calcHist([frame_a], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_b = cv2.calcHist([frame_b], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)


def extract_from_video(video_path: Path, start_index: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open {video_path}, skipping.")
        return start_index

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(fps * SECONDS_BETWEEN_FRAMES)

    frame_count = 0
    saved_count = start_index
    last_saved_frame = None

    print(f"\nProcessing {video_path.name} ({fps:.1f} fps)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Skip near-duplicate frames compared to the last one we saved
            if last_saved_frame is not None:
                sim = frame_similarity(frame, last_saved_frame)
                if sim > SIMILARITY_THRESHOLD:
                    frame_count += 1
                    continue

            out_path = OUTPUT_DIR / f"img_{saved_count:03d}.jpg"
            cv2.imwrite(str(out_path), frame)
            last_saved_frame = frame
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"  saved {saved_count - start_index} frames from this video")
    return saved_count


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/extract_frames.py <video_file_or_folder>")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if input_path.is_dir():
        video_files = sorted(
            [f for f in input_path.iterdir() if f.suffix.lower() in (".mp4", ".mkv", ".mov", ".avi")]
        )
    else:
        video_files = [input_path]

    if not video_files:
        print(f"No video files found at {input_path}")
        sys.exit(1)

    # Continue numbering from any existing images already in data/raw
    existing = list(OUTPUT_DIR.glob("img_*.jpg")) + list(OUTPUT_DIR.glob("img_*.png"))
    next_index = len(existing) + 1

    for video_path in video_files:
        next_index = extract_from_video(video_path, next_index)

    print(f"\nDone. Total images in {OUTPUT_DIR}: {next_index - 1}")


if __name__ == "__main__":
    main()