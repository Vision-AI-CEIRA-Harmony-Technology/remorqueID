"""Inference utilities for the trained VeRi ResNet-50 ReID model."""

import re
from pathlib import Path

try:
    from fast_plate_ocr import LicensePlateRecognizer
except Exception as e:
    print("Failed to import LicensePlateRecognizer", e)
    LicensePlateRecognizer = None

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image



DEFAULT_IMAGE_SIZE = (256, 256)
DEFAULT_CHECKPOINT = Path(__file__).parent / "models" / "resnet50_veri_final.pth"
DEFAULT_PLATE_MODEL = Path(__file__).parent / "models" / "plate_rtdetr.onnx"

# NOTE: the Gradio similarity slider is for cosine similarity only and must not be
# used as a detector/OCR threshold. Plate detection uses a separate internal cutoff.
PLATE_DETECTION_SCORE_THRESHOLD = 0.05
PLATE_MIN_AREA_RATIO = 0.0005
PLATE_MAX_CANDIDATES_PER_IMAGE = 1

_PLATE_SESSION = None
_PLATE_RECOGNIZER = None


def normalize_plate(value) -> str:
    """Strip formatting noise and normalize the license plate to an uppercase alphanumeric string."""
    if value is None:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", str(value).upper())
    return cleaned


def evaluate_vehicle_match(similarity: float, threshold: float, reference_plate: str, query_plate: str) -> dict:
    """Return a conservative verdict combining the embedding score and plate comparison."""
    ref_plate = normalize_plate(reference_plate)
    qry_plate = normalize_plate(query_plate)

    if not ref_plate and not qry_plate:
        return {
            "same_vehicle": False,
            "plate_status": "unavailable",
            "needs_review": True,
            "flags": ["Plate unavailable on both images; manual review required."],
        }

    if not ref_plate or not qry_plate:
        return {
            "same_vehicle": False,
            "plate_status": "unavailable",
            "needs_review": True,
            "flags": ["One image has no readable plate; comparison is inconclusive."],
        }

    if ref_plate != qry_plate:
        return {
            "same_vehicle": False,
            "plate_status": "mismatch",
            "needs_review": True,
            "flags": ["Plate mismatch detected between the two images."],
        }

    if similarity >= threshold:
        return {
            "same_vehicle": True,
            "plate_status": "match",
            "needs_review": False,
            "flags": [],
        }

    return {
        "same_vehicle": False,
        "plate_status": "match",
        "needs_review": False,
        "flags": [f"Plates match, but similarity {similarity:.4f} is below threshold {threshold:.2f}."],
    }


def _get_plate_session():
    global _PLATE_SESSION
    if _PLATE_SESSION is None:
        if not DEFAULT_PLATE_MODEL.exists():
            return None
        _PLATE_SESSION = ort.InferenceSession(str(DEFAULT_PLATE_MODEL), providers=["CPUExecutionProvider"])
    return _PLATE_SESSION


def _get_plate_recognizer():
    global _PLATE_RECOGNIZER
    if LicensePlateRecognizer is None:
        return None
    if _PLATE_RECOGNIZER is None:
        try:
            _PLATE_RECOGNIZER = LicensePlateRecognizer("cct-s-v2-global-model")
            #! debug
            print(_PLATE_RECOGNIZER is None)
            print(LicensePlateRecognizer is None)
        except Exception:
            return None
    return _PLATE_RECOGNIZER


def extract_plate_text(image) -> tuple[str, str, float]:
    """Detect a license plate in the image, OCR it, and return (text, status, confidence)."""
    if image is None:
        return "", "unavailable", 0.0

    if not hasattr(image, "convert"):
        image = Image.fromarray(np.asarray(image))

    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    min_plate_area = max(200, int(PLATE_MIN_AREA_RATIO * width * height))

    try:
        session = _get_plate_session()
        if session is None:
            return "", "unavailable", 0.0

        resized = np.asarray(rgb_image.resize((640, 640)), dtype=np.float32) / 255.0
        x = resized.transpose(2, 0, 1)[None]
        logits, boxes = session.run(None, {"pixel_values": x})
        scores = 1.0 / (1.0 + np.exp(-logits[0, :, 0]))
        keep = scores > PLATE_DETECTION_SCORE_THRESHOLD

        best_roi = None
        best_det_score = 0.0
        for (cx, cy, w, h), score in zip(boxes[0][keep], scores[keep]):
            x0 = max(0, min(width - 1, int((cx - w / 2) * width)))
            y0 = max(0, min(height - 1, int((cy - h / 2) * height)))
            x1 = max(0, min(width, int((cx + w / 2) * width)))
            y1 = max(0, min(height, int((cy + h / 2) * height)))
            if x1 <= x0 or y1 <= y0:
                continue
            box_width = x1 - x0
            box_height = y1 - y0
            roi_area = box_width * box_height
            if roi_area < min_plate_area:
                continue

            candidate_roi = (x0, y0, x1, y1)
            candidate_area = box_width * box_height
            if best_roi is None or candidate_area > (best_roi[2] - best_roi[0]) * (best_roi[3] - best_roi[1]):
                best_roi = candidate_roi
                best_det_score = float(score)

            if PLATE_MAX_CANDIDATES_PER_IMAGE == 1:
                break

        if best_roi is None:
            return "", "unavailable", 0.0

        recognizer = _get_plate_recognizer()
        if recognizer is None:
            # !debug
            print("hello4")
            return "", "unavailable", 0.0

        roi = rgb_image.crop(best_roi)
        try:
            pred = recognizer.run(np.asarray(roi), return_confidence=True)[0]
        except Exception:
            return "", "unavailable", 0.0

        candidate = getattr(pred, "plate", "")
        normalized = normalize_plate(candidate)
        if not normalized:
            return "", "unavailable", 0.0

        score = float(getattr(pred, "confidence", best_det_score) or best_det_score)
        return normalized, "detected", score
    except Exception:
        return "", "unavailable", 0.0


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
        """Compare two images and apply the supplied cosine threshold with a plate-aware safety check."""
        reference_embedding = self.embed(reference_image)
        query_embedding = self.embed(query_image)
        similarity = torch.sum(reference_embedding * query_embedding).item()

        reference_plate, _, _ = extract_plate_text(reference_image)
        query_plate, _, _ = extract_plate_text(query_image)
        # !debug
        print(reference_plate, query_plate, similarity, threshold)
        plate_result = evaluate_vehicle_match(similarity, threshold, reference_plate, query_plate)

        return {
            "similarity": similarity,
            "threshold": threshold,
            "plate_status": plate_result["plate_status"],
            "same_vehicle": plate_result["same_vehicle"],
            "needs_review": plate_result["needs_review"],
            "flags": plate_result["flags"],
            "reference_plate": reference_plate,
            "query_plate": query_plate,
            "device": str(self.device),
        }