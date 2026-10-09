from __future__ import annotations

import hashlib
import io
import os
import threading
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import av
import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image


VIDEO_MODEL_ID = "dima806/deepfake_vs_real_image_detection"
VIDEO_MODEL_REVISION = "29e4cf9efc543845610045f6ba7e88e5cf9d9301"
FACE_DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
FACE_DETECTOR_SHA256 = "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f"
FACE_DETECTOR_PATH = Path(__file__).resolve().parents[2] / ".tools" / "models" / "blaze_face_short_range.tflite"
MAX_VIDEO_DURATION_SECONDS = 15
MAX_VIDEO_DECODED_FRAMES = 1_800
VIDEO_SAMPLE_INTERVAL_SECONDS = 1
_model_lock = threading.RLock()
_processor: Any = None
_classifier: Any = None
_model_error: str | None = None
_face_detector: Any = None
_face_detector_error: str | None = None


def _load_video_model() -> None:
    global _processor, _classifier, _model_error
    if _classifier is not None:
        return
    if _model_error is not None:
        raise RuntimeError(_model_error)
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        processor = AutoImageProcessor.from_pretrained(
            VIDEO_MODEL_ID,
            revision=VIDEO_MODEL_REVISION,
        )
        classifier = AutoModelForImageClassification.from_pretrained(
            VIDEO_MODEL_ID,
            revision=VIDEO_MODEL_REVISION,
        )
        classifier.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        classifier.eval()
    except Exception as error:
        _model_error = f"Video classifier could not be loaded: {type(error).__name__}: {error}"
        raise RuntimeError(_model_error) from error
    _processor = processor
    _classifier = classifier


def video_model_status() -> dict[str, str]:
    with _model_lock:
        if _model_error is not None:
            return {
                "status": "unavailable",
                "model": VIDEO_MODEL_ID,
                "revision": VIDEO_MODEL_REVISION,
                "detail": _model_error,
            }
        if _face_detector_error is not None:
            return {
                "status": "unavailable",
                "model": VIDEO_MODEL_ID,
                "revision": VIDEO_MODEL_REVISION,
                "detail": _face_detector_error,
            }
        if _classifier is not None and _face_detector is not None:
            return {
                "status": "ready",
                "model": VIDEO_MODEL_ID,
                "revision": VIDEO_MODEL_REVISION,
                "license": "Apache-2.0",
            }
        return {
            "status": "not_loaded",
            "model": VIDEO_MODEL_ID,
            "revision": VIDEO_MODEL_REVISION,
            "license": "Apache-2.0",
        }


def warm_video_model() -> dict[str, str]:
    with _model_lock:
        try:
            _load_video_model()
            _get_face_detector()
        except RuntimeError:
            return video_model_status()
        return video_model_status()


def _get_face_detector() -> Any:
    global _face_detector, _face_detector_error
    if _face_detector is None:
        if _face_detector_error is not None:
            raise RuntimeError(_face_detector_error)
        try:
            FACE_DETECTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
            if not FACE_DETECTOR_PATH.exists():
                with urlopen(FACE_DETECTOR_URL, timeout=30) as response:
                    model_bytes = response.read(1_000_001)
                if len(model_bytes) > 1_000_000:
                    raise RuntimeError("The face-detection model exceeded its 1 MB download limit.")
                if hashlib.sha256(model_bytes).hexdigest() != FACE_DETECTOR_SHA256:
                    raise RuntimeError("The downloaded face-detection model checksum did not match.")
                partial_path = FACE_DETECTOR_PATH.with_suffix(".download")
                partial_path.write_bytes(model_bytes)
                os.replace(partial_path, FACE_DETECTOR_PATH)
            elif hashlib.sha256(FACE_DETECTOR_PATH.read_bytes()).hexdigest() != FACE_DETECTOR_SHA256:
                raise RuntimeError("The cached face-detection model checksum did not match.")
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            options = vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(FACE_DETECTOR_PATH)),
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=0.6,
            )
            _face_detector = vision.FaceDetector.create_from_options(options)
        except Exception as error:
            _face_detector_error = (
                f"Face detector could not be loaded: {type(error).__name__}: {error}"
            )
            raise RuntimeError(_face_detector_error) from error
    return _face_detector


def analyze_video_frame(frame: np.ndarray) -> dict[str, Any]:
    with _model_lock:
        _load_video_model()
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detections = _get_face_detector().detect(mp_image).detections
        faces: list[dict[str, Any]] = []
        for detection in detections[:4]:
            box = detection.bounding_box
            left = max(0, min(width, box.origin_x))
            top = max(0, min(height, box.origin_y))
            right = max(left, min(width, box.origin_x + box.width))
            bottom = max(top, min(height, box.origin_y + box.height))
            if right - left < 16 or bottom - top < 16:
                continue
            crop = rgb[top:bottom, left:right]
            inputs = _processor(images=Image.fromarray(crop), return_tensors="pt")
            device = next(_classifier.parameters()).device
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
            with torch.inference_mode():
                probabilities = torch.softmax(_classifier(**inputs).logits, dim=-1)[0]
            labels = _classifier.config.id2label
            fake_index = next(
                (
                    int(index)
                    for index, label in labels.items()
                    if any(
                        marker in str(label).casefold()
                        for marker in ("fake", "synthetic", "generated", "artificial")
                    )
                ),
                None,
            )
            real_index = next(
                (int(index) for index, label in labels.items() if "real" in str(label).casefold()),
                None,
            )
            if fake_index is None or real_index is None:
                raise RuntimeError(
                    f"Classifier labels do not identify real and fake classes: {labels!r}"
                )
            faces.append({
                "bbox": {
                    "x": left / width,
                    "y": top / height,
                    "width": (right - left) / width,
                    "height": (bottom - top) / height,
                },
                "detection_confidence": round(float(detection.categories[0].score), 4),
                "fake_score": round(float(probabilities[fake_index]), 4),
                "real_score": round(float(probabilities[real_index]), 4),
                "model": VIDEO_MODEL_ID,
                "model_revision": VIDEO_MODEL_REVISION,
            })
        return {
            "status": "analyzed" if faces else "no_face",
            "faces": faces,
            "timestamp_note": "Frame classifier scores are research signals, not calibrated probabilities.",
        }


def decode_video_frames(video_bytes: bytes) -> list[tuple[int, np.ndarray]]:
    """Decode a short in-memory recording into frames sampled once per second."""
    samples: list[tuple[int, np.ndarray]] = []
    decoded_count = 0
    next_sample_at = 0.0
    try:
        with av.open(io.BytesIO(video_bytes), mode="r") as container:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                raise ValueError("The submitted recording contains no video stream.")
            frame_rate = float(stream.average_rate or 30)
            for frame_index, frame in enumerate(container.decode(stream)):
                decoded_count += 1
                if decoded_count > MAX_VIDEO_DECODED_FRAMES:
                    raise ValueError("The submitted video contains too many frames to screen safely.")
                timestamp = (
                    float(frame.pts * frame.time_base)
                    if frame.pts is not None and frame.time_base is not None
                    else frame_index / frame_rate
                )
                if timestamp < 0 or timestamp > MAX_VIDEO_DURATION_SECONDS:
                    raise ValueError("The video recording must be 15 seconds or shorter.")
                if timestamp + 0.001 < next_sample_at:
                    continue
                image = frame.to_ndarray(format="bgr24")
                height, width = image.shape[:2]
                if width * height > 8_000_000:
                    scale = (8_000_000 / (width * height)) ** 0.5
                    image = cv2.resize(
                        image,
                        (int(width * scale), int(height * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                samples.append((round(timestamp * 1000), image))
                next_sample_at = timestamp + VIDEO_SAMPLE_INTERVAL_SECONDS
    except av.error.FFmpegError as error:
        raise ValueError(f"The submitted video could not be decoded: {error}") from error
    if not samples:
        raise ValueError("The submitted recording contains no decodable video frames.")
    if samples[-1][0] < 1_000:
        raise ValueError("The submitted video must contain at least one second of footage.")
    return samples


def summarize_video_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    analyzed = [frame for frame in frames if frame.get("status") == "analyzed"]
    scores = [
        face["fake_score"]
        for frame in analyzed
        for face in frame.get("faces", [])
    ]
    result: dict[str, Any] = {
        "status": "unavailable" if not frames else "inconclusive",
        "frames_sampled": len(frames),
        "frames_with_faces": len(analyzed),
        "median_fake_score": None,
        "score_range": None,
        "temporal_consistency": "unavailable",
        "detail": "A frame-level screening signal is not a determination that a person or video is authentic.",
        "frame_evidence": [],
    }
    if not scores:
        unavailable = next(
            (frame.get("detail") for frame in frames if frame.get("status") == "unavailable" and frame.get("detail")),
            None,
        )
        if unavailable:
            result["detail"] = str(unavailable)
        else:
            result["detail"] = "No face crops were scored in the sampled live frames."
        return result

    result["median_fake_score"] = round(float(np.median(scores)), 4)
    result["score_range"] = round(float(max(scores) - min(scores)), 4)
    frame_evidence = []
    for frame in analyzed:
        for face in frame.get("faces", []):
            frame_evidence.append({
                "elapsed_ms": frame["elapsed_ms"],
                "fake_score": face["fake_score"],
                "real_score": face["real_score"],
                "model_lean": "AI-like" if face["fake_score"] > face["real_score"] else "real-like",
                "bbox": face["bbox"],
            })
    result["frame_evidence"] = sorted(
        frame_evidence,
        key=lambda evidence: evidence["fake_score"],
        reverse=True,
    )[:8]
    if len(analyzed) >= 3:
        counts = [len(frame["faces"]) for frame in analyzed]
        result["temporal_consistency"] = (
            "review" if max(counts) - min(counts) > 1 or result["score_range"] >= 0.55 else "no_large_shift_observed"
        )
        result["status"] = "review" if result["temporal_consistency"] == "review" else "analyzed"
    else:
        result["detail"] = "Too few face-bearing frames were sampled to compare temporal consistency."
    return result
