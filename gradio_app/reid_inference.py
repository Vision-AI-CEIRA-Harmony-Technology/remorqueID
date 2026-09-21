"""Inference utilities for the trained VeRi ResNet-50 ReID model."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T


DEFAULT_IMAGE_SIZE = (256, 256)
DEFAULT_CHECKPOINT = Path(__file__).parent / "models" / "resnet50_veri_final.pth"


class ResNet50ReID(nn.Module):
    """The same embedding architecture used by the training notebook."""

    def __init__(self, num_vehicles: int):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(2048)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(2048, num_vehicles, bias=False)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        features = self.gap(features).flatten(1)
        features = self.bn(features)
        return features, self.classifier(features)


class ReIDEmbedder:
    """Loads the checkpoint once and extracts normalized image embeddings."""

    def __init__(self, checkpoint_path: Path | str = DEFAULT_CHECKPOINT):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        num_vehicles = checkpoint.get("num_vehicles", 575)
        self.model = ResNet50ReID(num_vehicles=num_vehicles).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.transform = T.Compose([
            T.Resize(DEFAULT_IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.inference_mode()
    def embed(self, image) -> torch.Tensor:
        """Return one L2-normalized embedding for a PIL image."""
        image_tensor = self.transform(image.convert("RGB"))
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        embedding, _ = self.model(image_tensor)
        return F.normalize(embedding, dim=1)

    def compare(self, reference_image, query_image, threshold: float) -> dict:
        """Compare two images and apply the supplied cosine threshold."""
        reference_embedding = self.embed(reference_image)
        query_embedding = self.embed(query_image)
        similarity = torch.sum(reference_embedding * query_embedding).item()
        return {
            "similarity": similarity,
            "threshold": threshold,
            "same_vehicle": similarity >= threshold,
            "device": str(self.device),
        }