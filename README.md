# LHRF-YOLO
## 📁 Dataset Download

The dataset used in this project is available via the following cloud storage links:

- 🔗 **Baidu Netdisk**: [Download here](https://pan.baidu.com/s/1_Xti5AoIER3yZ5hQhToSZQ) (Extraction code: `m58i`)  
- 🌍 **Google Drive**: [Download here](https://drive.google.com/file/d/1VFYYXHbzDTtgjTUU8cl6qNgi8ZahxGSe/view?usp=sharing)  
- 📦 **Alternative Link**: [QuarkDrive](https://pan.quark.cn/s/c13d3a6251c0) (Extraction code: `4aii`)  

> ⚠️ Note: Please choose the appropriate download option based on your region. Some links may have slower access speeds depending on your network location.

## 🚀 Deployment Guide

### Requirements

- Python >= 3.8
- CUDA GPU (recommended for accelerated training and inference)
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/LHRF-YOLO.git
cd LHRF-YOLO

# Install dependencies
pip install -e .
```

### Quick Start

**Train Model**
```bash
yolo train model=LHRF.yaml data=data_fire-smoke.yaml epochs=100 imgsz=640
```

**Inference**
```bash
yolo predict model=runs/detect/train/weights/best.pt source=path/to/image.jpg
```

**Export Model**
```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx
```

### Python API

```python
from ultralytics import YOLO

# Load model
model = YOLO("LHRF.yaml")

# Train
model.train(data="data_fire-smoke.yaml", epochs=100, imgsz=640)

# Inference
results = model.predict(source="path/to/image.jpg")
```
