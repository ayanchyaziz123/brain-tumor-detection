# Brain Tumor Detection

AI-powered MRI brain tumor classification using Transfer Learning (EfficientNet-B0) and Flask.

## Classes
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

## Tech Stack
- **Model:** EfficientNet-B0 (PyTorch, Transfer Learning)
- **Web App:** Flask
- **Dataset:** [Brain Tumor Classification MRI](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

## Project Structure
```
Brain Tumor Detection/
├── notebook/
│   └── brain_tumor_detection.ipynb   # Training pipeline
├── flask_app/
│   ├── app.py                         # Flask server
│   ├── model/                         # Saved model (.pth)
│   ├── static/                        # CSS & JS
│   ├── templates/                     # HTML
│   └── utils/predictor.py             # Inference
├── requirements.txt                   # All dependencies
├── requirements-deploy.txt            # Production dependencies
└── render.yaml                        # Render deployment config
```

## Setup

### 1. Clone & install
```bash
git clone https://github.com/ayanchyaziz123/rahman_brain-tumor-detection.git
cd rahman_brain-tumor-detection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download dataset
Download from Kaggle and place as:
```
data/
  Training/  glioma/  meningioma/  notumor/  pituitary/
  Testing/   glioma/  meningioma/  notumor/  pituitary/
```

### 3. Train the model
```bash
cd notebook
jupyter notebook brain_tumor_detection.ipynb
```
Run all cells. Model saves to `flask_app/model/brain_tumor_efficientnet.pth`.

### 4. Run the app
```bash
cd flask_app
python app.py
```
Open `http://localhost:5000`

## Model Architecture
- **Backbone:** EfficientNet-B0 pretrained on ImageNet
- **Head:** Dropout → Linear(1280, 512) → SiLU → BN → Dropout → Linear(512, 4)
- **Strategy:** Freeze backbone → train head → unfreeze top layers
- **Augmentation:** Random flip, rotation, color jitter, random erasing

## Deployment
Deployed on Render. See `render.yaml` for configuration.
