import cv2
import time
import os
from ultralytics import YOLO

# Real-world widths in meters
REAL_WIDTHS = {
    "cone": 0.3,       
    "barrier": 1.5,    
    "stop_sign": 0.75  
}

FOCAL_LENGTH = 800  

def estimate_distance(class_name, pixel_width):
    if class_name not in REAL_WIDTHS:
        return None
    
    real_width = REAL_WIDTHS[class_name]
    distance = (real_width * FOCAL_LENGTH) / pixel_width
    return round(distance, 2)

def run_inference(image_path, model_path, output_dir):
    print(f"Loading optimized edge model: {model_path}")
    model = YOLO(model_path, task='detect')
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image at {image_path}")
        return

    start_time = time.time()
    results = model(frame)[0]
    
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {round(fps, 1)}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        pixel_width = x2 - x1
        
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        
        distance = estimate_distance(class_name, pixel_width)
        
        if distance:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name.capitalize()}: {distance}m"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            print(f"Detected {class_name} at {distance} meters.")

    # --- NEW: Automated Output Handling ---
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the original filename and create the new save path
    base_name = os.path.basename(image_path)
    output_file = os.path.join(output_dir, f"detected_{base_name}")
    
    cv2.imwrite(output_file, frame)
    print(f"Saved prediction to {output_file}")
    
    cv2.imshow("Robotics Navigation - Distance Test", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    MODEL_WEIGHTS = r"C:\Folder\New folder\Sharon_files\l01-SharonSwarnil\runs\detect\robot_nav_v1\weights\best.onnx" 
    
    # The new output in this folder "outputs"
    OUTPUT_FOLDER = r"C:\Folder\New folder\Sharon_files\l01-SharonSwarnil\outputs"
    
    TEST_IMAGE = r"C:\Folder\New folder\Sharon_files\l01-SharonSwarnil\dataset\test\images\000010_jpg.rf.743a198151d460af8c26a6a0e05bbea1.jpg" 
    
    run_inference(TEST_IMAGE, MODEL_WEIGHTS, OUTPUT_FOLDER)