import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import numpy as np
import os


class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes=4, dropout=0.4, freeze_backbone=False):
        super().__init__()
        self.backbone = efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class TumorPredictor:
    _instance = None

    @classmethod
    def get_instance(cls, model_path):
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance

    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f'Model not found at {model_path}. '
                'Please train the model using the Jupyter notebook first.'
            )
        ckpt = torch.load(model_path, map_location=self.device)
        self.class_names  = ckpt['class_names']
        self.num_classes  = ckpt['num_classes']
        self.img_size     = ckpt.get('img_size', 224)
        mean = ckpt.get('imagenet_mean', [0.485, 0.456, 0.406])
        std  = ckpt.get('imagenet_std',  [0.229, 0.224, 0.225])

        self.model = BrainTumorClassifier(num_classes=self.num_classes)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs  = F.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx  = int(np.argmax(probs))
        return {
            'predicted_class': self.class_names[pred_idx],
            'confidence':      float(probs[pred_idx]),
            'probabilities':   {
                cls: float(p) for cls, p in zip(self.class_names, probs)
            },
            'is_tumor': self.class_names[pred_idx] != 'No Tumor'
        }
