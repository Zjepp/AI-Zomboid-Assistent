from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/models/zomboid_detector/weights/best.pt")

    metrics = model.val(data="data/dataset/data.yaml", split="test")

    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall: {metrics.box.mr:.3f}")

if __name__ == "__main__":
    main()