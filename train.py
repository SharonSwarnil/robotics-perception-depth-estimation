import os
import torch
from ultralytics import YOLO

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
        device="cpu", # Use 0 for GPU if available. Mine is CPU-only, so I set it to "cpu". Adjust as your configuration 
        workers=0 # Set to 0 because Windows sometimes freezes during multi-threading on CPU
    )

    print("\nTraining finished.")
    print("Best model saved at: runs/detect/robot_nav_v1/weights/best.pt")

    print("\nExporting optimized ONNX model for edge devices...")

    # Automatically check for GPU to avoid forcing FP16 quantization on a CPU
    is_gpu = torch.cuda.is_available()
    
    exported_path = model.export(
        format="onnx",
        half=is_gpu
    )

    if is_gpu:
        print(f"Optimized FP16 ONNX model saved at: {exported_path}")
    else:
        print(f"Standard FP32 ONNX model saved at: {exported_path} (CPU detected)")

if __name__ == "__main__":
    train_model()
    
    