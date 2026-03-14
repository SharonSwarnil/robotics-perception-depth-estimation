from ultralytics import YOLO
import os

def train_model():

    print("Loading YOLOv8 Nano model for edge-optimized training...")

    model = YOLO("yolov8n.pt")

    DATASET_YAML_PATH = "dataset/data.yaml"

    if not os.path.exists(DATASET_YAML_PATH):
        print(f"ERROR: Could not find {DATASET_YAML_PATH}")
        return

    print("Starting training...")

    results = model.train(
        data=DATASET_YAML_PATH,
        epochs=10, # Adjust as needed for better results. Mine is CPU so it takes more time, That's why I set it to just 10 epoches
        batch=10,
        name="robot_nav_v1",
        device="cpu" # Use 0 for GPU if available. Mine is CPU-only, so I set it to "cpu". Adjust as your configuration   
    )

    print("\nTraining finished.")
    print("Best model saved at: runs/detect/robot_nav_v1/weights/best.pt")

    print("\nExporting optimized ONNX model for edge devices...")

    exported_path = model.export(
        format="onnx",
        half=True
    )

    print(f"Optimized model saved at: {exported_path}")


if __name__ == "__main__":
    train_model()
    
    