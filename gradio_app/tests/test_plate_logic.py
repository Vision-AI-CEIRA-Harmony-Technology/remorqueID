import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gradio_app"))

from reid_inference import evaluate_vehicle_match, normalize_plate


def test_normalize_plate_removes_noise():
    assert normalize_plate(" ab-123  cd ") == "AB123CD"


def test_evaluate_vehicle_match_plate_mismatch():
    result = evaluate_vehicle_match(0.95, 0.70, "AB123CD", "XY999ZZ")
    assert result["plate_status"] == "mismatch"
    assert result["same_vehicle"] is False
    assert result["needs_review"] is True
    assert any("plate mismatch" in flag.lower() for flag in result["flags"])


def test_evaluate_vehicle_match_similarity_below_threshold_even_when_plates_match():
    result = evaluate_vehicle_match(0.42, 0.70, "AB123CD", "AB123CD")
    assert result["plate_status"] == "match"
    assert result["same_vehicle"] is False
    assert result["needs_review"] is False
    assert any("similarity" in flag.lower() for flag in result["flags"])
