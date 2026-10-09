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


VIDEO_MODEL_ID = "prithivMLmods/Deep-Fake-Detector-Model"
VIDEO_MODEL_REVISION = "c5cb24c6a159dd2b57ca15c6a1065bd0ce8fa380"
VIDEO_AI_THRESHOLD = 0.38
VIDEO_REAL_THRESHOLD = 0.30
MIN_VIDEO_FACE_SIZE = 36
VIDEO_FACE_CROP_PADDING = 0.35
FACE_DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
FACE_DETECTOR_SHA256 = "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f"
FACE_DETECTOR_PATH = Path(__file__).resolve().parents[2] / ".tools" / "models" / "blaze_face_short_range.tflite"
MAX_VIDEO_DURATION_SECONDS = 61
MAX_VIDEO_DECODED_FRAMES = 3_600
VIDEO_SAMPLE_INTERVAL_SECONDS = 0.5
MAX_LIVE_FACE_DETECTION_WIDTH = 640
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
            _get_face_detector()
            _load_video_model()
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


def detect_live_faces(frame: np.ndarray) -> list[dict[str, float]]:
    height, width = frame.shape[:2]
    scale = min(1.0, MAX_LIVE_FACE_DETECTION_WIDTH / width)
    detection_frame = (
        cv2.resize(
            frame,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1.0
        else frame
    )
    rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with _model_lock:
        detections = _get_face_detector().detect(mp_image).detections
    detection_height, detection_width = detection_frame.shape[:2]
    return [
        {
            "x": detection.bounding_box.origin_x / detection_width,
            "y": detection.bounding_box.origin_y / detection_height,
            "width": detection.bounding_box.width / detection_width,
            "height": detection.bounding_box.height / detection_height,
        }
        for detection in detections
    ]


def analyze_video_frame(frame: np.ndarray) -> dict[str, Any]:
    with _model_lock:
        _load_video_model()
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detections = _get_face_detector().detect(mp_image).detections[:4]
        total_frame_area = max(1, width * height)
        eligible: list[tuple[Any, int, int, int, int]] = []
        for detection in detections:
            box = detection.bounding_box
            left = max(0, min(width, box.origin_x))
            top = max(0, min(height, box.origin_y))
            right = max(left, min(width, box.origin_x + box.width))
            bottom = max(top, min(height, box.origin_y + box.height))
            box_width = right - left
            box_height = bottom - top
            if (
                box_width >= MIN_VIDEO_FACE_SIZE and box_height >= MIN_VIDEO_FACE_SIZE
            ) or (box_width * box_height) / total_frame_area >= 0.03:
                eligible.append((detection, left, top, right, bottom))

        if not eligible and detections:
            largest = max(
                detections,
                key=lambda detection: (
                    detection.bounding_box.width * detection.bounding_box.height
                ),
            )
            box = largest.bounding_box
            left = max(0, min(width, box.origin_x))
            top = max(0, min(height, box.origin_y))
            right = max(left, min(width, box.origin_x + box.width))
            bottom = max(top, min(height, box.origin_y + box.height))
            if right - left >= 32 and bottom - top >= 32:
                eligible.append((largest, left, top, right, bottom))

        eligible.sort(key=lambda face: (face[3] - face[1]) * (face[4] - face[2]), reverse=True)
        largest_face_area = (
            (eligible[0][3] - eligible[0][1]) * (eligible[0][4] - eligible[0][2])
            if eligible
            else 0
        )
        faces: list[dict[str, Any]] = []
        subject_scores: list[float] = []
        for detection, left, top, right, bottom in eligible:
            face_width = right - left
            face_height = bottom - top
            pad_x = int(face_width * VIDEO_FACE_CROP_PADDING)
            pad_y = int(face_height * VIDEO_FACE_CROP_PADDING)
            crop_left = max(0, left - pad_x)
            crop_top = max(0, top - pad_y)
            crop_right = min(width, right + pad_x)
            crop_bottom = min(height, bottom + pad_y)
            crop_width = crop_right - crop_left
            crop_height = crop_bottom - crop_top
            if crop_height > crop_width:
                extra = crop_height - crop_width
                crop_left = max(0, crop_left - extra // 2)
                crop_right = min(width, crop_right + extra - extra // 2)
            elif crop_width > crop_height:
                extra = crop_width - crop_height
                crop_top = max(0, crop_top - extra // 2)
                crop_bottom = min(height, crop_bottom + extra - extra // 2)
            crop = rgb[crop_top:crop_bottom, crop_left:crop_right]
            if crop.size == 0:
                continue
            inputs = _processor(images=Image.fromarray(crop), return_tensors="pt")
            device = next(_classifier.parameters()).device
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
            with torch.inference_mode():
                probabilities = torch.softmax(_classifier(**inputs).logits, dim=-1)[0]
            labels = _classifier.config.id2label or {}
            normalized_labels = {
                int(index): str(label).casefold()
                for index, label in labels.items()
                if str(index).isdigit()
            }
            fake_index = next(
                (
                    index
                    for index, label in normalized_labels.items()
                    if any(marker in label for marker in ("fake", "synthetic", "generated", "artificial", "deepfake"))
                ),
                None,
            )
            real_index = next(
                (
                    index
                    for index, label in normalized_labels.items()
                    if any(marker in label for marker in ("real", "authentic", "genuine", "human", "original"))
                ),
                None,
            )
            if fake_index is None or real_index is None:
                raise RuntimeError(
                    f"Classifier labels do not identify real and fake classes: {labels!r}"
                )
            if fake_index >= len(probabilities) or real_index >= len(probabilities):
                raise RuntimeError("Classifier labels refer to an output index that does not exist.")

            fake_score = round(float(probabilities[fake_index]), 4)
            real_score = round(float(probabilities[real_index]), 4)
            is_subject = face_width * face_height >= 0.25 * largest_face_area
            if is_subject:
                subject_scores.append(fake_score)
            faces.append({
                "bbox": {
                    "x": left / width,
                    "y": top / height,
                    "width": face_width / width,
                    "height": face_height / height,
                },
                "detection_confidence": round(float(detection.categories[0].score), 4),
                "fake_score": fake_score,
                "real_score": real_score,
                "threshold": VIDEO_AI_THRESHOLD,
                "subject_face": is_subject,
                "model": VIDEO_MODEL_ID,
                "model_revision": VIDEO_MODEL_REVISION,
            })
        frame_score = max(subject_scores) if subject_scores else None
        return {
            "status": "analyzed" if faces else "no_face",
            "faces": faces,
            "frame_score": frame_score,
            "frame_classification": (
                "AI-like" if frame_score is not None and frame_score >= VIDEO_AI_THRESHOLD
                else "real-like" if frame_score is not None
                else None
            ),
            "timestamp_note": "Frame classifier scores are uncalibrated research signals, not identity or authenticity determinations.",
        }


def decode_video_frames(video_bytes: bytes) -> list[tuple[int, np.ndarray]]:
    """Decode a complete short recording and sample frames through its final frame."""
    samples: list[tuple[int, np.ndarray]] = []
    decoded_count = 0
    next_sample_at = 0.0
    try:
        with av.open(io.BytesIO(video_bytes), mode="r") as container:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                raise ValueError("The submitted recording contains no video stream.")
            frame_rate = float(stream.average_rate or 30)
            last_frame = None
            last_timestamp = 0.0
            last_sampled_timestamp = None
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
                    raise ValueError("The video recording must be 60 seconds or shorter.")
                last_frame = frame
                last_timestamp = timestamp
                if timestamp + 0.001 < next_sample_at:
                    continue
                samples.append((round(timestamp * 1000), _bounded_video_frame(frame)))
                last_sampled_timestamp = timestamp
                next_sample_at += VIDEO_SAMPLE_INTERVAL_SECONDS
                while next_sample_at <= timestamp:
                    next_sample_at += VIDEO_SAMPLE_INTERVAL_SECONDS
            if last_frame is not None and (
                last_sampled_timestamp is None or last_timestamp - last_sampled_timestamp > 0.001
            ):
                samples.append((round(last_timestamp * 1000), _bounded_video_frame(last_frame)))
    except av.error.FFmpegError as error:
        raise ValueError(f"The submitted video could not be decoded: {error}") from error
    if not samples:
        raise ValueError("The submitted recording contains no decodable video frames.")
    if samples[-1][0] < 1_000:
        raise ValueError("The submitted video must contain at least one second of footage.")
    return samples


def _bounded_video_frame(frame: av.VideoFrame) -> np.ndarray:
    image = frame.to_ndarray(format="bgr24")
    height, width = image.shape[:2]
    if width * height > 8_000_000:
        scale = (8_000_000 / (width * height)) ** 0.5
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return image


def summarize_video_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    analyzed = [frame for frame in frames if frame.get("status") == "analyzed"]
    scores: list[float] = []
    for frame in analyzed:
        frame_score = frame.get("frame_score")
        if frame_score is not None:
            scores.append(float(frame_score))
            continue
        face_scores = [
            float(face["fake_score"])
            for face in frame.get("faces", [])
            if "fake_score" in face and face.get("subject_face", True)
        ]
        if face_scores:
            scores.append(max(face_scores))
    result: dict[str, Any] = {
        "status": "unavailable" if not frames else "inconclusive",
        "frames_sampled": len(frames),
        "frames_with_faces": len(analyzed),
        "median_fake_score": None,
        "aggregate_fake_score": None,
        "score_range": None,
        "flagged_frames": 0,
        "flagged_frame_ratio": 0.0,
        "model_verdict": "INCONCLUSIVE",
        "model_threshold": VIDEO_AI_THRESHOLD,
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

    median_score = float(np.median(scores))
    result["median_fake_score"] = round(median_score, 4)
    mean_score = float(np.mean(scores))
    aggregate_score = 0.60 * median_score + 0.40 * mean_score
    result["aggregate_fake_score"] = round(aggregate_score, 4)
    result["score_range"] = round(float(max(scores) - min(scores)), 4)
    flagged_frames = sum(score >= VIDEO_AI_THRESHOLD for score in scores)
    flagged_ratio = flagged_frames / len(scores)
    real_ratio = sum(score < VIDEO_REAL_THRESHOLD for score in scores) / len(scores)
    result["flagged_frames"] = flagged_frames
    result["flagged_frame_ratio"] = round(flagged_ratio, 4)
    if (
        len(scores) >= 2
        and flagged_frames >= 2
        and flagged_ratio >= 0.40
        and aggregate_score >= 0.40
        and real_ratio < 0.40
    ):
        result["model_verdict"] = "AI_GENERATED"
    elif (
        len(scores) >= 2
        and aggregate_score <= VIDEO_REAL_THRESHOLD
        and (flagged_frames == 0 or (len(scores) >= 5 and flagged_frames <= 1 and aggregate_score <= 0.20))
        and real_ratio >= 0.60
    ):
        result["model_verdict"] = "REAL_VIDEO"
    else:
        result["model_verdict"] = "INCONCLUSIVE"
    frame_evidence = []
    for frame in analyzed:
        for face in frame.get("faces", []):
            if not face.get("subject_face", True):
                continue
            frame_evidence.append({
                "elapsed_ms": frame["elapsed_ms"],
                "fake_score": face["fake_score"],
                "real_score": face["real_score"],
                "model_lean": (
                    "AI-like" if face["fake_score"] >= VIDEO_AI_THRESHOLD else "real-like"
                ),
                "threshold": VIDEO_AI_THRESHOLD,
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
    result["detail"] = (
        f"Experimental video-screening consensus: {result['model_verdict']} "
        f"(aggregate AI-like score {result['aggregate_fake_score']:.1%}; "
        f"{flagged_frames}/{len(scores)} frames at or above {VIDEO_AI_THRESHOLD:.0%}). "
        "The model scores are uncalibrated research signals, not identity or authenticity determinations."
    )
    return result
