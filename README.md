# Robotics Perception & Depth Estimation

## Overview

This project is a simple perception module built for a robotics/navigation use case. The goal was to detect a few important road objects and estimate how far they are from the camera using basic computer vision techniques.

The system currently detects:

* Traffic Cones
* Road Barriers
* Stop Signs

Along with detection, it also estimates distance using a basic geometric approach.

---

## What I Built

I combined object detection with a simple depth estimation logic:

* Used **YOLOv8 Nano** for object detection
* Used a **pinhole camera approximation** to estimate distance
* Exported the model to **ONNX** so it can run on lightweight systems

This project was done on CPU, so I focused more on making things work efficiently rather than training very large models.

---

## Dataset & Approach

Initially, I planned to use the BDD100K dataset, but it was too large for my system.

So I:

* Collected smaller datasets from Roboflow
* Merged them into one dataset
* Fixed class labels manually using Python scripts
* Removed incorrect annotations

Final dataset:

* Train: 812 images
* Validation: 102 images
* Test: 102 images

Classes:

* Cone
* Barrier
* Stop Sign

---

## Model Training

* Model: YOLOv8 Nano
* Training: CPU
* Epochs: 10

Since I didn’t have GPU access, I kept training limited and focused on getting stable results.

---

## Training Results

![Training Results](runs/detect/robot_nav_v1/results.png)

### Observations

* Loss decreases gradually
* Precision and recall improve over time
* Final mAP50 is around **0.84**

For a small dataset and CPU training, this result is decent for a prototype.

---

## Distance Estimation

After detection, distance is calculated using:

Distance = (Real Width × Focal Length) / Pixel Width

Assumptions used:

* Cone ≈ 0.3 m
* Barrier ≈ 1.5 m
* Stop Sign ≈ 0.75 m
* Focal length = 800

This is not perfectly accurate because the camera is not calibrated, but it gives a reasonable estimate.

---

## Sample Outputs

### Street Scene

![Street Scene](outputs/street_scene.jpg)

### Barrier Detection

![Barrier](outputs/barrier.jpg)

### Stop Sign Detection

![Stop Sign](outputs/stop_sign.jpg)

Each output shows:

* Detected object
* Estimated distance

---

## Performance

Tested on:

* AMD Ryzen 5 5625U (CPU)

Inference speed:

* ~0.2 FPS

This is expected for CPU. It should run faster on GPU or edge devices like Jetson.

---

## Limitations

* Small dataset (especially barrier class)
* Distance estimation is approximate
* No camera calibration
* Limited training due to CPU

---

## What I Learned

* How to prepare and merge datasets
* How object detection models work in practice
* Basics of distance estimation using camera geometry
* Exporting models for deployment (ONNX)
* Handling real constraints like no GPU

---

## Future Improvements

* Train on larger datasets
* Use camera calibration for better depth accuracy
* Improve dataset quality
* Try deployment on edge devices

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train:

```bash
python train.py
```

Run detection:

```bash
python detect_distance.py
```

Outputs will be saved in the `outputs/` folder.

---

## Author

Sharon Swarnil
B.Tech - Artificial Intelligence & Data Science
