# Brain Tumor Detection

AI-powered MRI brain tumor classification built with PyTorch transfer learning (EfficientNet-B0) and a FastAPI web application. Upload an MRI scan and get an instant prediction with Grad-CAM heatmap visualization and a confidence warning badge.

**Live demo:** https://brain-tumor-detection-umba.onrender.com

---

## Features

- 4-class classification: Glioma, Meningioma, Pituitary Tumor, No Tumor
- Grad-CAM heatmap showing which regions influenced the prediction
- Confidence warning badge (High / Moderate / Low) with clinical guidance
- Probability breakdown bar chart for all classes
- Drag-and-drop image upload

## Tech Stack

| Layer | Technology |
|---|---|
| Model | EfficientNet-B0 (PyTorch, pretrained ImageNet) |
| Explainability | Grad-CAM (custom implementation, no external lib) |
| Web App | FastAPI + Jinja2 + Uvicorn |
| Dataset | [Brain Tumor Classification MRI — Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) |
| Deployment | Render.com |

## Project Structure

```
Brain Tumor Detection/
├── app/
│   ├── main.py              # FastAPI routes and startup
│   ├── predictor.py         # EfficientNet inference + Grad-CAM
│   ├── static/
│   │   ├── css/style.css
│   │   ├── js/main.js
│   │   └── uploads/         # Saved prediction images (gitignored)
│   └── templates/
│       └── index.html
├── ml_model/
│   ├── brain_tumor_efficientnet.pth   # Trained model weights
│   └── brain_tumor_detection.ipynb   # Full training notebook
├── data/                              # Dataset (gitignored)
├── requirements.txt                   # Dev dependencies
├── requirements-deploy.txt            # Production dependencies
├── Procfile                           # Render start command
└── render.yaml                        # Render deployment config
```

## Local Setup

### 1. Clone and install

```bash
git clone https://github.com/ayanchyaziz123/brain-tumor-detection.git
cd brain-tumor-detection
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the dataset

Download from [Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) and place it as:

```
data/
  Training/
    glioma/  meningioma/  notumor/  pituitary/
  Testing/
    glioma/  meningioma/  notumor/  pituitary/
```

### 3. Train the model (optional — weights included)

Open `ml_model/brain_tumor_detection.ipynb` in Jupyter and run all cells. The trained weights save to `ml_model/brain_tumor_efficientnet.pth`.

```bash
jupyter notebook ml_model/brain_tumor_detection.ipynb
```

### 4. Run the app

```bash
uvicorn app.main:app --reload
```

Open `http://localhost:8000`

## Model Architecture

- **Backbone:** EfficientNet-B0 pretrained on ImageNet (backbone frozen during initial training)
- **Head:** Dropout(0.4) → Linear(1280, 512) → SiLU → BatchNorm → Dropout(0.2) → Linear(512, 4)
- **Loss:** CrossEntropyLoss with label smoothing (0.1)
- **Optimizer:** AdamW with CosineAnnealingLR
- **Sampler:** WeightedRandomSampler to handle class imbalance
- **Augmentation:** Random crop, flip, rotation, color jitter, random erasing

## Confidence Badge

| Badge | Confidence | Guidance |
|---|---|---|
| HIGH CONFIDENCE | 85%+ | Result is reliable |
| MODERATE CONFIDENCE | 60–84% | Consider reviewing with a specialist |
| LOW CONFIDENCE | Below 60% | Please consult a radiologist |

> This tool is for educational purposes only and is not a substitute for professional medical diagnosis.

## Deployment

Deployed on Render free tier. Cold starts may take 20–30 seconds after inactivity. The frontend handles this with a 90-second timeout and a helpful loading message.

## Author

Built by **Rahman** — PyTorch · EfficientNet-B0 · FastAPI
