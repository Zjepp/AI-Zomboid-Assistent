from ultralytics import YOLO
from pathlib import Path

def main():

    model = YOLO("runs/detect/models/zomboid_detector/weights/best.pt")

    test_images = Path("data/dataset/test/images")
    results = model.predict(source=str(test_images), save=True, conf=0.25, project="reports", name="predictions")

    print("Voorspellingen opgeslagen in reports/predictions/")

if __name__ == "__main__":
    main()