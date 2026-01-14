# Automated Glacial Algae Monitoring

![Automated Glacial Algae Monitoring Banner](media/image-1.png)

> ⚠️ **Active Research Project**
> 
> This project is part of ongoing MRes research at the University of Bristol. The pipeline is under active development and cannot be publicly released until the research is complete. Documentation will be updated as new features and improvements are added.
> 

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive modular pipeline for automated detection and classification of glacial algae cells from light microscopy images using state-of-the-art deep learning object detection models.

## Overview

This project is part of the MRes research of the sole contributor @lzyjo at the University of Bristol, focused on developing automated methods for monitoring glacial algae communities in the cryosphere. Glacial algae play a crucial role in glacier surface albedo dynamics and biogeochemical processes, yet their monitoring remains labor-intensive and time-consuming. This pipeline automates the detection and classification of glacial algae cells, enabling large-scale quantitative analysis of microscopy images.

### Why Automated Glacial Algae Monitoring?

- **Manual counting is impractical**: Analyzing microscopy images manually is extremely time-consuming
- **Reproducibility**: Automated detection provides consistent, reproducible results
- **Scalability**: Process large datasets efficiently for temporal and spatial studies
- **Quantitative analysis**: Further applications will enable precise biomass estimation and cell morphology characterization

## Key Features

- 🔬 **Data Pre-processing**: Robust pipeline for handling light microscopy images and annotation formats, including dataset structure (VOC and YOLO)
- 🔄 **Data Augmentation**: Comprehensive and modular data augmentation pipeline 
- 🏷️ **Label Conversion**: Flexible conversion between annotation formats (VOC and YOLO) and different levels of annotation detail 
- 🤖 **Multi-Model Training**: Support for state-of-the-art object detection architectures
- 📊 **Performance Tracking**: Automated logging of training metrics and losses for for Excel and TensorBoard
- 📈 **Visualization**: Comprehensive and modular pipeline for performance plots and model comparison tools
- ⚡ **Modular Design**: Easy to extend and customize for different datasets and models

## Supported Models

| Model | Framework | Key Features |
|-------|-----------|-------------|
| **Faster R-CNN** | PyTorch | Two-stage detector, Region Proposal Networks (RPNs) |
| **RetinaNet** | PyTorch | Single-stage, Focal Loss for class imbalance |
| **YOLO v5** | Ultralytics | Single-stage, high speed and accuracy |
| **YOLO v10** | Ultralytics | Latest YOLO architecture, improved speed/accuracy trade-off |

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Architecture](#pipeline-architecture)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (Highly recommended for training)

### Setting up a Virtual Environment

We strongly recommend using a virtual environment to avoid dependency conflicts:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

or otherwise in IDE such as VSCode

### Dependencies

Install the required packages using pip or uv:

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install uv package manager (optional but faster)
pip install uv

# Install dependencies with uv
uv pip install pycocotools tensorboard tifffile torchmetrics scikit-learn scipy Pillow tqdm torchvision torch numpy DateTime matplotlib pandas argparse ultralytics pytorch-lightning

# OR with pip
pip install pycocotools tensorboard tifffile torchmetrics scikit-learn scipy Pillow tqdm torchvision torch numpy DateTime matplotlib pandas argparse ultralytics pytorch_lightning
```

### Core Dependencies

| Package | Purpose |
|---------|---------|
| torch, torchvision, pytorch-lightning | Deep learning framework |
| ultralytics | YOLO models |
| pycocotools | - |
| torchmetrics | Evaluation metrics |
| tensorboard | Training visualization |
| scikit-learn | ML utilities and metrics |
| matplotlib | Plotting and visualization |
| tifffile | Microscopy image handling |

## Quick Start

### 1. Prepare Your Data

Organize your microscopy images and annotations following the naming convention:

```
<Location>_<Month>_<Year>_<Sample>/
├── <Month>/
│   ├── <Sample_ID>/
│   │   ├── Overlays/
│   │   │   ├── Image_<XXXXXX>.tif
│   │   │   ├── Image_<XXXXXX>.tif
│   │   │   └── ...
│   │   └── annotations/
│   │       ├── annotations.xml
│   │       └── ...
│   └── ...
└── ...
```

**Example**: `Greenland_24_July_5/` or `Mort_Sept_24_mA/`


### 2. Use dataset formatting pipeline to convert to desired format (VOC/YOLO):

```bash
python data_pre-processing.py  --split True --datasets_to_use Greenland_24_July_5,Mort_Sept_24_mA 
```


### 3. Convert Annotations (VOC -> YOLO or YOLO -> VOC)

Additionally pipeline allows for conversion between different levels of detail in annotations (which should be done prior to label conversion):
- Species and cell count
- Cell only
- Species only
- Cell count only


#### Annotation level conversion:
Conversions can only be carried out top-down

```bash
python annotation_level_conversion.py  --from_level species_and_cellcount --to_level species
```

#### Label format conversion:

```bash
python label_conversion.py  --from_format VOC --to_format YOLO --model_type classifier
```

### 4. Data Augmentation (Optional)

```bash
python data_augmentation.py --model_type classifier --augmentations horizontalflip --raw_dir /insert_path --augmented_dir /insert_path --visual_check True
```

### 5. Train a Model

```bash
python train.py --model_name faster_rcnn 
                   --model_type classifier
                   --data_folder /path/to/dataset
                   --data_format VOC
                   --loss_function BCE
                   --early_stopping True
```

### 6. Evaluate Performance

```bash
python evaluate.py --model /path/to/model --data /path/to/dataset --visualize True
```

## Pipeline Architecture

The pipeline consists of the following modular components:


```
┌──────────────────────┐
│  Raw Microscopy      │
│  Img + Annotations   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Data Preprocessing   │
│ - Label conversion   │
│ - Format conversion  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Data Augmentation    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Model Training      │
│ - Multiple models    │
│ - Multiple types     │
│ - Multiple losses    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Metrics & Logging    │
│ - Loss tracking      │
│ - TensorBoard        │
│ - Performance plots  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Model Evaluation     │
│ & Comparison         │
└──────────────────────┘
```


### Annotation Format

The pipeline accepts annotations using:
- **ImageJ** (.xml files)
- **MIA** (Modular Image Analysis)
- **FIJI** (Fiji Is Just ImageJ)

which gives you VOC formatted .xmls 


## Model Training

### Training Configuration

Customize training via command-line arguments and hyperparameter/configuration files:

```bash
python train.py --model_name faster_rcnn 
                   --model_type classifier
                   --data_folder /path/to/dataset
                   --data_format VOC
                   --loss_function BCE
                   --early_stopping True
```

### Monitoring Training

View training progress in real-time with TensorBoard or at the end in excel/csv files. 

## Evaluation Metrics

Model performance is evaluated using multiple metrics:

### Object Detection Metrics

- **mAP** (mean Average Precision)
- **mAP@0.5** (AP at IoU threshold 0.5)
- **mAP@0.75** (AP at IoU threshold 0.75)
- **IoU** (Intersection over Union)
- **Precision**
- **Recall**
- **Accuracy**
- **F1 Score**
- **Confusion Matrix**
- **Precision-Recall Curve**

### Loss Functions

- **PyTorch Built-in Losses**
- **Cross-Entropy (CE)**
- **Binary Cross-Entropy (BCE)**
- **Smooth L1 Loss**
- **Mean Squared Error (MSE)**
- **L1 Loss**
- **Hamming Loss**

### Performance Visualization

The pipeline generates:
- Training/validation loss curves
- Precision-recall curves
- Accuracy over epochs
- Confusion matrices
- Detection visualizations


## Contributing

This project is part of an MRes dissertation research. While this is currently a sole-contributor project for now, suggestions and feedback are welcome.

### Development

To contribute improvements or report issues:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **University of Bristol** - MRes Program support
- **PyTorch** and **Ultralytics** teams - Deep learning frameworks
- **Glaciology research community** - Domain expertise and datasets
- All open-source contributors whose libraries made this work possible

## Contact

For questions or collaborations:
- **Repository**: [github.com/lzyjo/Automated-Glacial-Algae-Monitoring](https://github.com/lzyjo/Automated-Glacial-Algae-Monitoring)
- **Issues**: [GitHub Issues](https://github.com/lzyjo/Automated-Glacial-Algae-Monitoring/issues)

---

