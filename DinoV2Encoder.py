import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

class DinoV2Encoder:
    def __init__(self, device="cuda:0", model_name="dinov2_vits14"):
        self.device = torch.device(device if torch.cuda.is_available() or "cpu" in device else "cpu")

        # DINOv2 ViT-S/14 -> dim=384, 224x224 -> 14x14=196 patches
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(self.device)

        # Standard ImageNet normalization (DINOv2 uses this convention)
        self.tf = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def encode(self, img):
        # img: path / PIL / np(H,W,3)
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                x = img.astype(np.float32)
                if x.max() <= 1.5:
                    x = (x * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    x = x.clip(0, 255).astype(np.uint8)
                img = x
            img = Image.fromarray(img, mode="RGB")
        elif isinstance(img, Image.Image):
            img = img.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        x = self.tf(img).unsqueeze(0).to(self.device)  # (1,3,224,224)

        feats = self.model.forward_features(x)
        patch = feats["x_norm_patchtokens"]            # (1,196,384)
        return patch

