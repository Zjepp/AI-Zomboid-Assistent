from ultralytics import YOLO

def train_model(data_yaml: str, epochs: int = 100, imgsz: int = 960, model_size: str = "yolov8n.pt"):
    model = YOLO(model_size)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        rect=True,
        patience=20,
        cache="ram",            
        project="models",
        name="zomboid_detector",
        exist_ok=True,
    )
    return results

if __name__ == "__main__":
    train_model(data_yaml="data/dataset/data.yaml", epochs=200)
