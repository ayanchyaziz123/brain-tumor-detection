import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0
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


class GradCAM:
    def __init__(self, model, target_layer):
        self.model      = model
        self.activations = None
        self.gradients   = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor, class_idx):
        self.model.eval()
        self.model.zero_grad()
        logits = self.model(tensor)
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = F.relu(cam).cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


class TumorPredictor:
    _instance = None

    @classmethod
    def get_instance(cls, model_path):
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance

    def __init__(self, model_path: str):
        self.device = torch.device('cpu')
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f'Model not found at {model_path}. '
                'Please train the model using the Jupyter notebook first.'
            )
        ckpt = torch.load(model_path, map_location=self.device)
        self.class_names = ckpt['class_names']
        self.num_classes  = ckpt['num_classes']
        self.img_size     = ckpt.get('img_size', 224)
        mean = ckpt.get('imagenet_mean', [0.485, 0.456, 0.406])
        std  = ckpt.get('imagenet_std',  [0.229, 0.224, 0.225])

        self.model = BrainTumorClassifier(num_classes=self.num_classes)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Grad-CAM on last EfficientNet feature block
        self.gradcam = GradCAM(self.model, self.model.backbone.features[-1])

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    @staticmethod
    def _jet_colormap(x: np.ndarray) -> np.ndarray:
        pts = [0.0, 0.125, 0.375, 0.625, 0.875, 1.0]
        r = np.interp(x, pts, [0.0, 0.0, 0.0, 1.0, 1.0, 0.5])
        g = np.interp(x, pts, [0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        b = np.interp(x, pts, [0.5, 1.0, 1.0, 0.0, 0.0, 0.0])
        return np.stack([r, g, b], axis=-1)

    def predict_with_gradcam(self, image: Image.Image, save_dir: str, base_name: str) -> dict:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize original for overlay
        orig_np = np.array(image.resize((self.img_size, self.img_size))) / 255.0
        tensor  = self.transform(image).unsqueeze(0).to(self.device)

        # Prediction (no grad)
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = F.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))

        # Grad-CAM (needs grad)
        cam = self.gradcam.generate(tensor, pred_idx)
        cam = np.array(
            Image.fromarray((cam * 255).astype(np.uint8))
                 .resize((self.img_size, self.img_size))
        ) / 255.0

        # Colormap + overlay
        heatmap = self._jet_colormap(cam)
        overlay = np.clip(0.5 * orig_np + 0.5 * heatmap, 0, 1)

        # Save images
        heatmap_name = f'{base_name}_heatmap.png'
        overlay_name = f'{base_name}_overlay.png'
        Image.fromarray((heatmap * 255).astype(np.uint8)).save(os.path.join(save_dir, heatmap_name))
        Image.fromarray((overlay * 255).astype(np.uint8)).save(os.path.join(save_dir, overlay_name))

        return {
            'predicted_class': self.class_names[pred_idx],
            'confidence':      float(probs[pred_idx]),
            'probabilities':   {cls: float(p) for cls, p in zip(self.class_names, probs)},
            'is_tumor':        self.class_names[pred_idx] != 'No Tumor',
            'heatmap_name':    heatmap_name,
            'overlay_name':    overlay_name,
        }
