# PCB Bare-Board Defect Detection Using YOLO11n

A deep learning-based object detection project that detects and localizes seven types of defects on Printed Circuit Boards (PCBs) using **YOLO11n (YOLO11 Nano)** and the Ultralytics framework.

The project uses a pretrained YOLO11n model fine-tuned on a PCB bare-board defect dataset. It identifies defects, predicts their bounding boxes, and assigns confidence scores to each detection.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Project Objectives](#3-project-objectives)
4. [Defect Categories](#4-defect-categories)
5. [Dataset Information](#5-dataset-information)
6. [Dataset Structure](#6-dataset-structure)
7. [Technologies and Tools](#7-technologies-and-tools)
8. [Environment Setup](#8-environment-setup)
9. [Dataset Preparation](#9-dataset-preparation)
10. [YOLO Dataset Configuration](#10-yolo-dataset-configuration)
11. [Model Architecture](#11-model-architecture)
12. [Model Training](#12-model-training)
13. [Model Evaluation](#13-model-evaluation)
14. [Training Results and Visualizations](#14-training-results-and-visualizations)
15. [External Image Detection](#15-external-image-detection)
16. [Understanding the Evaluation Metrics](#16-understanding-the-evaluation-metrics)
17. [Project Output Structure](#17-project-output-structure)
18. [How to Run the Project](#18-how-to-run-the-project)
19. [Limitations](#19-limitations)
20. [Future Improvements](#20-future-improvements)
21. [Conclusion](#21-conclusion)

---

## 1. Project Overview

Printed Circuit Boards (PCBs) are essential components of electronic devices. Defects during PCB manufacturing can affect electrical connectivity, product reliability, and overall manufacturing quality.

Traditional PCB inspection methods often require manual examination or conventional image-processing techniques. These approaches can be time-consuming and may struggle with complex defect patterns.

This project implements an automated PCB defect detection system using YOLO11n, a lightweight object detection model.

The trained model detects seven categories of PCB defects:

- Missing Hole
- Mouse Bite
- Open Circuit
- Short
- Spur
- Spurious Copper
- Pin Hole

The model predicts the location and category of each detected defect using bounding boxes and confidence scores.

### Key Features

- Automated detection of seven PCB defect categories.
- YOLO11n-based object detection.
- Transfer learning using pretrained weights.
- Training on a dataset containing more than 20,000 PCB images.
- Bounding-box localization of defects.
- Validation and independent test-set evaluation.
- Precision, Recall, mAP@50, and mAP@50-95 analysis.
- Confusion matrix and training-curve visualization.
- Detection on external PCB images.
- GPU-accelerated training using Google Colab.

---

## 2. Problem Statement

Manual inspection of printed circuit boards can be time-consuming, particularly when manufacturing processes involve large numbers of boards and multiple defect categories.

The objective of this project is to develop a deep learning-based object detection system capable of automatically identifying and localizing common PCB manufacturing defects from input images.

The system uses YOLO11n to learn visual characteristics of different defect categories and generate bounding-box predictions for detected defects.

---

## 3. Project Objectives

1. Prepare and organize a labeled PCB defect detection dataset.
2. Configure the dataset in the YOLO annotation format.
3. Implement a YOLO11n-based object detection pipeline.
4. Fine-tune pretrained weights using the PCB dataset.
5. Evaluate the trained model using validation and test datasets.
6. Analyze class-wise detection performance.
7. Visualize training progress and detection performance.
8. Perform inference on external PCB images.
9. Save the trained model for future deployment and experimentation.

---

## 4. Defect Categories

The model is trained to recognize seven defect categories.

| Class ID | Defect Category | Description |
|---|---|---|
| 0 | Missing Hole | A required hole is missing from the PCB. |
| 1 | Mouse Bite | Irregular material loss along the PCB edge. |
| 2 | Open Circuit | An interruption in an intended conductive path. |
| 3 | Short | An unintended electrical connection between conductive regions. |
| 4 | Spur | An unwanted copper extension or protrusion. |
| 5 | Spurious Copper | Unwanted copper material present in an unintended region. |
| 6 | Pin Hole | A small hole or void in a PCB region. |

The class IDs are maintained consistently throughout dataset preparation, training, and inference.

---

## 5. Dataset Information

**Dataset Name:** PCB Bare-Board Defect Detection

**Dataset Source:** Hugging Face

**Dataset Repository:** `ParsaVictor/pcb-bare-board-defect-detection`

The dataset contains images of bare PCBs along with their corresponding object detection annotations.

### Dataset Split

| Dataset Split | Number of Images | Purpose |
|---|---:|---|
| Training | 18,276 | Model training |
| Validation | 1,090 | Model monitoring and selection |
| Testing | 1,017 | Final performance evaluation |
| **Total** | **20,383** | |

### Annotation Information

The dataset uses YOLO-format annotation files.

Each image has a corresponding `.txt` annotation file containing the object class and normalized bounding-box coordinates.

YOLO annotation format:

```text
class_id x_center y_center width height
```

All bounding-box coordinates are normalized between 0 and 1.

Example:

```text
2 0.512 0.438 0.125 0.096
```

This represents an object belonging to class 2 with its normalized center coordinates, width, and height.

### Empty Annotation Files

Some images contain no annotated defects.

These images have empty `.txt` annotation files and are treated as negative examples during object detection training.

Dataset counts:

| Split | Empty Annotation Files |
|---|---:|
| Training | 3,044 |
| Validation | 183 |
| Testing | 170 |
| **Total** | **3,397** |

These negative examples help the model learn to distinguish defect-free regions from actual defects.

### Published Defect Counts

| Defect Category | Published Count |
|---|---:|
| Open Circuit | 8,594 |
| Mouse Bite | 8,437 |
| Spur | 8,079 |
| Spurious Copper | 7,798 |
| Short | 7,755 |
| Missing Hole | 2,802 |
| Pin Hole | 2,633 |

These are the dataset's published defect counts, not a separately calculated count of test predictions.

---

## 6. Dataset Structure

The dataset is organized into separate training, validation, and testing directories.

```text
pcb_data/
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each image must have a corresponding annotation file with the same filename.

Example:

```text
images/train/image001.jpg
labels/train/image001.txt
```

The same naming convention is maintained for validation and test images.

---

## 7. Technologies and Tools

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| YOLO11n | Object detection model |
| Ultralytics | YOLO implementation and training framework |
| PyTorch | Deep learning framework |
| OpenCV | Image processing and visualization |
| NumPy | Numerical operations |
| Matplotlib | Training graphs and result visualization |
| Pandas | Training results analysis |
| Google Colab | Model training environment |
| NVIDIA Tesla T4 GPU | GPU-accelerated training |
| Hugging Face | Dataset hosting and retrieval |
| Git and GitHub | Version control and project hosting |

---

## 8. Environment Setup

The model was trained using Google Colab with an NVIDIA Tesla T4 GPU.

### Step 1: Open Google Colab

Open:

https://colab.research.google.com/

Create a new notebook.

### Step 2: Enable GPU

Navigate to:

Runtime → Change runtime type → T4 GPU

Verify GPU availability:

```python
import torch

print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

### Step 3: Install Required Libraries

```python
!pip install -q ultralytics huggingface_hub pyyaml pandas matplotlib opencv-python
```

Verify Ultralytics installation:

```python
import ultralytics

print("Ultralytics Version:", ultralytics.__version__)
```

---

## 9. Dataset Preparation

The dataset was retrieved directly from Hugging Face and extracted into the required YOLO directory structure.

The original dataset archives were:

```text
train_images_and_labels.zip
val_images_and_labels.zip
test_images_and_labels.zip
```

### Step 1: Download Dataset Archives

```python
from huggingface_hub import hf_hub_download

dataset_repo = "ParsaVictor/pcb-bare-board-defect-detection"

files_to_download = [
    "train_images_and_labels.zip",
    "val_images_and_labels.zip",
    "test_images_and_labels.zip"
]

for filename in files_to_download:

    file_path = hf_hub_download(
        repo_id=dataset_repo,
        filename=filename,
        repo_type="dataset"
    )

    print(filename, "downloaded:", file_path)
```

### Step 2: Extract Dataset Files

The archives contain image and annotation files under the `final_bare` directory.

The following is a general extraction example. It assumes the archives are available in the current Colab working directory.

```python
import os
import zipfile
from pathlib import Path

base_dir = Path("/content/pcb_data")

for split in ["train", "val", "test"]:

    (base_dir / "images" / split).mkdir(
        parents=True, exist_ok=True
    )

    (base_dir / "labels" / split).mkdir(
        parents=True, exist_ok=True
    )
```

Extract the image and label files into their respective split directories, retaining only `.jpg` and `.txt` files.

The archive layout should be checked before extraction to ensure the files are assigned to the correct split.

### Step 3: Verify Dataset Counts

```python
from pathlib import Path

base_dir = Path("/content/pcb_data")

for split in ["train", "val", "test"]:

    image_dir = base_dir / "images" / split
    label_dir = base_dir / "labels" / split

    images = list(image_dir.glob("*.jpg"))
    labels = list(label_dir.glob("*.txt"))

    print(f"\n{split.upper()} DATASET")
    print("Images:", len(images))
    print("Labels:", len(labels))
```

Expected counts:

```text
TRAIN:
Images: 18276
Labels: 18276

VAL:
Images: 1090
Labels: 1090

TEST:
Images: 1017
Labels: 1017
```

---

## 10. YOLO Dataset Configuration

YOLO requires a YAML configuration file containing the dataset paths, number of classes, and class names.

Create `data.yaml`:

```yaml
path: /content/pcb_data

train: images/train
val: images/val
test: images/test

nc: 7

names:
  0: missing_hole
  1: mouse_bite
  2: open_circuit
  3: short
  4: spur
  5: spurious_copper
  6: pin_hole
```

This file ensures that the model uses the correct image directories and class mapping during training and evaluation.

---

## 11. Model Architecture

### YOLO11n

This project uses YOLO11n, the Nano variant of the YOLO11 object detection family.

YOLO11n is a lightweight model designed to provide efficient object detection with relatively low computational requirements.

### Why YOLO11n?

- Lightweight architecture.
- Efficient inference.
- Suitable for GPU-based training.
- Supports bounding-box object detection.
- Provides an effective starting point for transfer learning.

### Model Configuration

| Parameter | Configuration |
|---|---|
| Model | YOLO11n |
| Pretrained Weights | `yolo11n.pt` |
| Input Image Size | 640 × 640 |
| Number of Classes | 7 |
| Task | Object Detection |
| Framework | Ultralytics |
| Training Method | Transfer Learning |

The pretrained YOLO11n weights are fine-tuned on the PCB defect dataset.

---

## 12. Model Training

The model was trained using Google Colab with an NVIDIA Tesla T4 GPU.

### Training Configuration

| Parameter | Value |
|---|---|
| Model | YOLO11n |
| Epochs | 20 |
| Batch Size | 16 |
| Image Size | 640 × 640 |
| GPU | NVIDIA Tesla T4 |
| Workers | 2 |
| Patience | 10 |
| Cache | False |
| Optimizer | Ultralytics default |
| Initial Weights | `yolo11n.pt` |

### Training Code

```python
from ultralytics import YOLO

# Load pretrained YOLO11n weights
model = YOLO("yolo11n.pt")

# Train the model
results = model.train(

    data="/content/data.yaml",

    epochs=20,

    imgsz=640,

    batch=16,

    device=0,

    workers=2,

    patience=10,

    cache=False,

    project="/content/pcb_runs",

    name="yolo11n_pcb_20ep_b16",

    plots=True,

    save=True,

    verbose=True
)
```

### Explanation of Training Parameters

- `data`: Specifies the dataset configuration file.
- `epochs`: Number of training epochs.
- `imgsz`: Input image resolution.
- `batch`: Number of images processed in one batch.
- `device`: Selects the GPU.
- `workers`: Number of data-loading workers.
- `patience`: Early-stopping patience based on validation performance.
- `cache`: Controls dataset caching.
- `project`: Directory for training outputs.
- `name`: Experiment name.
- `plots`: Generates training and evaluation plots.
- `save`: Saves model checkpoints.

The training process generates model checkpoints, training logs, and performance visualizations.

### Training Checkpoints

The trained model files are saved in:

```text
/content/pcb_runs/yolo11n_pcb_20ep_b16/weights/
```

Important files:

```text
best.pt
last.pt
```

- `best.pt`: Model checkpoint corresponding to the best validation fitness during training.
- `last.pt`: Model checkpoint from the final completed training epoch.

For inference and final evaluation, `best.pt` is used.

---

## 13. Model Evaluation

After training, the model was evaluated using the independent test dataset.

The test dataset contains:

- 1,017 images
- 2,404 annotated defect instances
- 170 background images

### Load the Best Model

```python
from ultralytics import YOLO

model = YOLO(
    "/content/pcb_runs/yolo11n_pcb_20ep_b16/weights/best.pt"
)
```

### Evaluate on the Test Dataset

```python
metrics = model.val(

    data="/content/data.yaml",

    split="test",

    imgsz=640,

    batch=8,

    device=0,

    plots=True,

    project="/content/pcb_runs",

    name="test_evaluation"
)
```

### Extract Evaluation Metrics

```python
print("Precision:", metrics.box.mp)

print("Recall:", metrics.box.mr)

print("mAP@50:", metrics.box.map50)

print("mAP@50-95:", metrics.box.map)
```

The evaluation generates class-wise metrics, confusion matrices, and other performance visualizations.

---

## 14. Training Results and Visualizations

### Overall Test Results

The trained YOLO11n model achieved the following test performance:

| Evaluation Metric | Result |
|---|---:|
| Precision | 90.69% |
| Recall | 85.17% |
| mAP@50 | 89.57% |
| mAP@50-95 | 60.17% |

### Class-Wise Performance

| Defect Category | Precision | Recall | mAP@50 |
|---|---:|---:|---:|
| Missing Hole | 93.13% | 97.57% | 99.00% |
| Mouse Bite | 89.49% | 79.14% | 84.15% |
| Open Circuit | 87.37% | 83.14% | 87.47% |
| Short | 87.10% | 69.15% | 80.89% |
| Spur | 92.41% | 83.63% | 87.27% |
| Spurious Copper | 93.09% | 84.05% | 88.80% |
| Pin Hole | 92.26% | 99.49% | 99.35% |

### Results Interpretation

**Precision — 90.69%**

The model demonstrates strong precision, indicating that a high proportion of its predicted defect detections are correct.

**Recall — 85.17%**

The model identifies a substantial proportion of the actual defects, although some defects remain undetected.

**mAP@50 — 89.57%**

The model achieves an overall mean Average Precision of approximately 89.57% at an IoU threshold of 0.50.

**mAP@50-95 — 60.17%**

This metric evaluates detection performance across multiple IoU thresholds, from 0.50 to 0.95, providing a stricter assessment of localization accuracy.

### Training Graphs

The training process generates `results.png`, which includes:

- Training box loss.
- Training classification loss.
- Training distribution focal loss.
- Validation box loss.
- Validation classification loss.
- Validation distribution focal loss.
- Precision.
- Recall.
- mAP@50.
- mAP@50-95.

### Confusion Matrix

The confusion matrix provides information about correct and incorrect class predictions.

It helps identify:

- Correctly classified defects.
- Misclassification between defect categories.
- False-positive detections.
- Missed defect instances.

### Additional Visualization Files

Typical generated files include:

```text
results.png
confusion_matrix.png
confusion_matrix_normalized.png
BoxP_curve.png
BoxR_curve.png
BoxPR_curve.png
BoxF1_curve.png
```

These visualizations help analyze model convergence and detection performance.

---

## 15. External Image Detection

The trained model can also detect defects in external PCB images that are not part of the training, validation, or test datasets.

### Step 1: Load the Trained Model

```python
from ultralytics import YOLO

model = YOLO(
    "/content/pcb_runs/yolo11n_pcb_20ep_b16/weights/best.pt"
)

print("Trained model loaded successfully!")
```

### Step 2: Upload an External Image

```python
from google.colab import files

uploaded = files.upload()

image_path = list(uploaded.keys())[0]

print("Uploaded image:", image_path)
```

### Step 3: Perform Defect Detection

```python
results = model.predict(

    source=image_path,

    imgsz=640,

    conf=0.25,

    iou=0.7,

    save=True,

    project="/content/pcb_predictions",

    name="external_image",

    exist_ok=True
)

print("Detection completed!")
```

### Step 4: Display the Annotated Image

```python
import cv2
import matplotlib.pyplot as plt

annotated_image = results[0].plot()

annotated_image = cv2.cvtColor(
    annotated_image,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(12, 8))

plt.imshow(annotated_image)

plt.axis("off")

plt.title("PCB Defect Detection Results")

plt.show()
```

### Step 5: Display Detected Defects and Confidence Scores

```python
result = results[0]

if len(result.boxes) > 0:

    print("Detected PCB Defects:\n")

    for box in result.boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        print(
            f"Defect: {class_name} | "
            f"Confidence: {confidence * 100:.2f}%"
        )

else:

    print("No PCB defects detected.")
```

### Step 6: Save and Download the Prediction

```python
from google.colab import files
import os

output_path = (
    "/content/pcb_predictions/external_image/"
    + os.path.basename(image_path)
)

if os.path.exists(output_path):

    files.download(output_path)
```

The output image contains the predicted bounding boxes, class names, and confidence scores.

**Note:** A model reporting no detections does not guarantee that an image is defect-free. Detection performance depends on image quality, defect appearance, and the confidence threshold.

---

## 16. Understanding the Evaluation Metrics

### 1. Precision

Precision measures the proportion of predicted positive detections that are correct.

\[
Precision = \frac{TP}{TP+FP}
\]

Where:

- TP = True Positives
- FP = False Positives

Higher precision indicates fewer incorrect defect detections.

### 2. Recall

Recall measures the proportion of actual defects that the model successfully detects.

\[
Recall = \frac{TP}{TP+FN}
\]

Where:

- TP = True Positives
- FN = False Negatives

Higher recall indicates fewer missed defects.

### 3. Intersection over Union (IoU)

IoU measures the overlap between the predicted bounding box and the ground-truth bounding box.

\[
IoU = \frac{Area\ of\ Intersection}{Area\ of\ Union}
\]

### 4. Mean Average Precision at 50 (mAP@50)

mAP@50 calculates the mean Average Precision across all classes using an IoU threshold of 0.50.

### 5. Mean Average Precision at 50-95 (mAP@50-95)

This metric calculates the mean Average Precision across IoU thresholds ranging from 0.50 to 0.95, with increments of 0.05.

It provides a more comprehensive evaluation of both detection and bounding-box localization performance.

---

## 17. Project Output Structure

The following is the recommended organization for the GitHub repository.

```text
PCB-Bare-Board-Defect-Detection/
│
├── README.md
│
├── notebooks/
│   └── PCB_Defect_Detection.ipynb
│
├── models/
│   └── best.pt
│
├── results/
│   ├── results.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── BoxP_curve.png
│   ├── BoxR_curve.png
│   ├── BoxPR_curve.png
│   └── BoxF1_curve.png
│
├── predictions/
│   └── external_image_prediction.jpg
│
├── requirements.txt
│
└── .gitignore
```

**Important:** The complete dataset is not included in this repository. It can be obtained from the original Hugging Face dataset repository.

The trained model can be included if its file size and GitHub hosting limits permit. Otherwise, provide a link to a suitable external model-storage location.

---

## 18. How to Run the Project

### Prerequisites

- Python 3.10 or later (the original experiment used Python 3.13.16).
- Google Colab or a compatible GPU-enabled environment.
- NVIDIA GPU recommended for efficient training.
- Access to the PCB dataset.

### Step 1: Clone the Repository

```bash
git clone https://github.com/rajputporika02/PCB-Bare-Board-Defect-Detection-Using-YOLO11n.git

cd PCB-Bare-Board-Defect-Detection-Using-YOLO11n
```



### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Prepare the Dataset

Download the dataset from:

https://huggingface.co/datasets/ParsaVictor/pcb-bare-board-defect-detection

Organize it using the YOLO directory structure described above.

### Step 4: Configure the Dataset

Update `data.yaml` with the correct dataset path.

### Step 5: Train the Model

Run the training code from Section 12.

### Step 6: Evaluate the Model

Run the evaluation code from Section 13.

### Step 7: Perform External Image Detection

Run the inference code from Section 15.

---

## 19. Limitations

- Detection performance depends on image quality, resolution, and defect visibility.
- Small or visually similar defects can be difficult to distinguish.
- Some defect categories have lower recall than others.
- Performance on images from different PCB manufacturing environments may vary.
- The model has been evaluated on the available dataset and should be tested on additional real-world PCB images before industrial deployment.
- The current implementation performs object detection on images and does not implement a real-time production-line inspection system.

---

## 20. Future Improvements

Potential extensions include:

1. Experimenting with larger YOLO model variants such as YOLO11s or YOLO11m.
2. Improving detection performance for difficult defect categories, particularly short and mouse bite.
3. Performing systematic hyperparameter optimization.
4. Exploring advanced data augmentation techniques.
5. Increasing robustness across different PCB image acquisition conditions.
6. Developing a real-time PCB inspection application.
7. Integrating a user interface for image upload and defect visualization.
8. Deploying the trained model using an inference API or edge device.

These are possible future directions, not features claimed as already implemented.

---

## 21. Conclusion

This project demonstrates the application of deep learning and computer vision to automated PCB bare-board defect detection.

A YOLO11n model was fine-tuned using a dataset containing 20,383 images across seven defect categories. The trained model was evaluated on an independent test set and achieved:

- **90.69% Precision**
- **85.17% Recall**
- **89.57% mAP@50**
- **60.17% mAP@50-95**

The implementation also includes class-wise performance analysis, confusion matrix visualization, training-curve analysis, and external image inference.

The project provides a practical foundation for automated PCB inspection and demonstrates the use of modern object detection techniques in electronics manufacturing quality control.

---

## Author

**Abid Singh Rajput Porika**



---

## Acknowledgements

- Ultralytics for the YOLO11 implementation.
- Hugging Face for hosting the PCB bare-board defect detection dataset.
- Google Colab for providing GPU-based development and training infrastructure.

