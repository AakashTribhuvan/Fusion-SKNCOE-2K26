"""Video-only screening API for the Frame/Check frontend.

This app reuses Video Detection Model's face detector, image classifier, frame sampler, and
metadata analyzer. It does not load the identity database or perform matching.
"""

from __future__ import annotations

import os
import statistics
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config
from core.ai_detector import AIImageDetector
from core.encoder import extract_faces
from core.metadata_analyzer import MetadataAnalyzer
from core.video_processor import extract_video_metadata, sample_video_frames


ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = int(os.getenv("VIDEO_DETECTION_MODEL_MAX_VIDEO_BYTES", str(250 * 1024 * 1024)))
MAX_VIDEO_SECONDS = float(os.getenv("VIDEO_DETECTION_MODEL_MAX_VIDEO_SECONDS", "180"))
VIDEO_SAMPLE_INTERVAL = float(getattr(config, "VIDEO_SAMPLE_INTERVAL", 2.0))
DEPTH_MODEL_ID = os.getenv("VIDEO_DETECTION_MODEL_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
AI_THRESHOLD = float(getattr(config, "AI_DETECTOR_THRESHOLD", 0.38))
_inference_lock = threading.RLock()

app = FastAPI(
    title="Video Detection Model Video Screening API",
    description="Video-only sampled-frame authenticity screening; no identity enrollment or matching.",
    version="3.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        *[origin.strip() for origin in os.getenv("VIDEO_DETECTION_MODEL_FRONTEND_ORIGINS", "").split(",") if origin.strip()],
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_ai_detector: AIImageDetector | None = None
_metadata_analyzer: MetadataAnalyzer | None = None
_face_model_ready = False


class RelativeDepthAnalyzer:
    """Lazy monocular depth measurements; output is relative, not metric depth."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._processor: Any = None
        self._model: Any = None
        self._device: Any = None
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _load(self) -> None:
        if self._model is not None:
            return
        if self._load_error:
            raise RuntimeError(self._load_error)
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            self._processor = AutoImageProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
            self._model.eval()
        except Exception as error:
            self._load_error = f"Relative-depth model unavailable: {error}"
            raise RuntimeError(self._load_error) from error

    def analyze(self, frame: np.ndarray, boxes: list[dict[str, int]]) -> list[dict[str, float]]:
        if not boxes:
            return []
        self._load()
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=rgb, return_tensors="pt")
        inputs = {key: tensor.to(self._device) for key, tensor in inputs.items()}
        with torch.inference_mode():
            predicted = self._model(**inputs).predicted_depth
        predicted = torch.nn.functional.interpolate(
            predicted.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )[0, 0]
        depth = predicted.detach().float().cpu().numpy()
        finite = np.isfinite(depth)
        if not finite.any():
            raise ValueError("Depth model returned no finite values.")
        low, high = np.percentile(depth[finite], (2, 98))
        depth = np.clip((depth - low) / max(float(high - low), 1e-6), 0, 1)

        results = []
        for box in boxes:
            x, y, box_width, box_height = _clip_box(box, width, height)
            if box_width < 8 or box_height < 8:
                continue
            left, top = max(0, x - box_width // 2), max(0, y - box_height // 2)
            right = min(width, x + box_width + box_width // 2)
            bottom = min(height, y + box_height + box_height // 2)
            roi = depth[top:bottom, left:right]
            if roi.size == 0:
                continue

            yy, xx = np.ogrid[top:bottom, left:right]
            center_x, center_y = x + box_width / 2, y + box_height / 2
            radius_x, radius_y = max(box_width * 0.38, 1), max(box_height * 0.46, 1)
            face_mask = ((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2 <= 1
            if not face_mask.any():
                continue
            face_values = depth[top:bottom, left:right][face_mask]
            background_values = depth[top:bottom, left:right][~face_mask]
            face_median = float(np.median(face_values))
            background_median = float(np.median(background_values)) if background_values.size else face_median
            results.append({
                "face_mask_median_relative_depth": round(face_median, 4),
                "face_mask_depth_spread_iqr": round(float(np.percentile(face_values, 75) - np.percentile(face_values, 25)), 4),
                "face_background_depth_delta": round(face_median - background_median, 4),
            })
        return results


_depth_analyzer = RelativeDepthAnalyzer(DEPTH_MODEL_ID)


def _clip_box(area: dict[str, Any], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x = max(0, min(image_width, int(area.get("x", 0))))
    y = max(0, min(image_height, int(area.get("y", 0))))
    width = max(0, min(image_width - x, int(area.get("w", 0))))
    height = max(0, min(image_height - y, int(area.get("h", 0))))
    return x, y, width, height


def _crop_face(frame: np.ndarray, area: dict[str, Any]) -> np.ndarray:
    image_height, image_width = frame.shape[:2]
    x, y, width, height = _clip_box(area, image_width, image_height)
    pad_ratio = float(getattr(config, "FACE_CROP_PADDING", 0.55))
    pad_w = int(width * pad_ratio)
    pad_h = int(height * pad_ratio)

    left = max(0, x - pad_w)
    top = max(0, y - pad_h)
    right = min(image_width, x + width + pad_w)
    bottom = min(image_height, y + height + pad_h)

    # Balance aspect ratio towards square so 224x224 ViT preprocessing does not introduce geometric squish
    box_w = right - left
    box_h = bottom - top
    if box_h > box_w:
        diff = box_h - box_w
        left = max(0, left - diff // 2)
        right = min(image_width, right + (diff - diff // 2))
    elif box_w > box_h:
        diff = box_w - box_h
        top = max(0, top - diff // 2)
        bottom = min(image_height, bottom + (diff - diff // 2))

    return frame[top:bottom, left:right]


def _box_iou(first: dict[str, int], second: dict[str, int]) -> float:
    left = max(first["x"], second["x"])
    top = max(first["y"], second["y"])
    right = min(first["x"] + first["w"], second["x"] + second["w"])
    bottom = min(first["y"] + first["h"], second["y"] + second["h"])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = first["w"] * first["h"] + second["w"] * second["h"] - intersection
    return intersection / union if union > 0 else 0.0


def _associate_face_tracks(frames: list[dict[str, Any]]) -> dict[str, Any]:
    tracks: list[dict[str, Any]] = []
    next_track_id = 1
    total_faces = 0
    persistent_faces = 0

    for sample_index, frame in enumerate(frames):
        faces = frame["faces"]
        total_faces += len(faces)
        candidates = []
        for face_index, face in enumerate(faces):
            box = face["bbox"]
            face_center = (box["x"] + box["w"] / 2, box["y"] + box["h"] / 2)
            face_scale = max((box["w"] + box["h"]) / 2, 1)
            for track in tracks:
                if sample_index - track["last_sample"] > 1:
                    continue
                previous = track["last_bbox"]
                previous_center = (previous["x"] + previous["w"] / 2, previous["y"] + previous["h"] / 2)
                previous_scale = max((previous["w"] + previous["h"]) / 2, 1)
                distance = np.hypot(face_center[0] - previous_center[0], face_center[1] - previous_center[1])
                normalized_distance = float(distance / max(face_scale, previous_scale))
                iou = _box_iou(box, previous)
                if iou >= 0.05 or normalized_distance <= 0.65:
                    candidates.append((normalized_distance - iou * 0.5, track, face_index))

        assigned_tracks: set[int] = set()
        assigned_faces: set[int] = set()
        for _, track, face_index in sorted(candidates, key=lambda item: item[0]):
            track_id = track["track_id"]
            if track_id in assigned_tracks or face_index in assigned_faces:
                continue
            face = faces[face_index]
            track["last_bbox"] = face["bbox"]
            track["last_sample"] = sample_index
            track["sample_count"] += 1
            face["track_id"] = track_id
            assigned_tracks.add(track_id)
            assigned_faces.add(face_index)

        for face_index, face in enumerate(faces):
            if face_index in assigned_faces:
                continue
            face["track_id"] = next_track_id
            tracks.append({
                "track_id": next_track_id,
                "last_bbox": face["bbox"],
                "last_sample": sample_index,
                "sample_count": 1,
            })
            next_track_id += 1

    persistent_tracks = [track for track in tracks if track["sample_count"] >= 2]
    persistent_faces = sum(track["sample_count"] for track in persistent_tracks)
    return {
        "available": True,
        "samples_with_faces": sum(bool(frame["faces"]) for frame in frames),
        "tracks_detected": len(tracks),
        "persistent_tracks": len(persistent_tracks),
        "max_track_samples": max((track["sample_count"] for track in tracks), default=0),
        "continuity_ratio": round(persistent_faces / total_faces, 4) if total_faces else 0.0,
        "detail": "Face boxes are associated across sampled frames by overlap and normalized center distance. This is face-track continuity, not general object tracking or liveness proof.",
    }


def _analyze_video(path: str, original_filename: str) -> dict[str, Any]:
    global _ai_detector, _metadata_analyzer
    if _ai_detector is None:
        _ai_detector = AIImageDetector()
    if _metadata_analyzer is None:
        _metadata_analyzer = MetadataAnalyzer()

    try:
        video_meta = extract_video_metadata(path)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Invalid or unreadable video: {error}") from error
    if video_meta["duration"] <= 0 or video_meta["total_frames"] <= 0:
        raise HTTPException(status_code=400, detail="The uploaded file contains no readable video frames.")
    if video_meta["duration"] > MAX_VIDEO_SECONDS:
        raise HTTPException(status_code=413, detail=f"Video exceeds the {MAX_VIDEO_SECONDS:g}-second limit.")

    try:
        samples = list(sample_video_frames(path, VIDEO_SAMPLE_INTERVAL))
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Video frame sampling failed: {error}") from error
    if not samples:
        raise HTTPException(status_code=400, detail="No frames could be sampled from this video.")

    frame_results: list[dict[str, Any]] = []
    model_scores: list[float] = []
    flagged_frames = 0
    depth_rows: list[dict[str, Any]] = []

    for sample in samples:
        frame = sample["frame"]
        frame_number = int(sample["frame_number"])
        timestamp = float(sample["timestamp"])
        height, width = frame.shape[:2]
        frame_faces: list[dict[str, Any]] = []
        face_errors: list[str] = []
        frame_scores: list[float] = []
        frame_flagged = False

        with _inference_lock:
            try:
                detected_faces = extract_faces(frame)
            except Exception as error:
                detected_faces = []
                face_errors.append(f"Face detection failed: {error}")

            total_frame_area = max(1, width * height)
            min_size = int(getattr(config, "MIN_FACE_SIZE", 50))

            # Filter faces: keep only valid human faces (discard tiny background noise/false-positive artifacts)
            # A face must either have width and height >= min_size OR occupy >= 3% of total frame area
            valid_faces = [
                f for f in detected_faces
                if (
                    f.get("facial_area", {}).get("w", 0) >= min_size
                    and f.get("facial_area", {}).get("h", 0) >= min_size
                )
                or (
                    (f.get("facial_area", {}).get("w", 0) * f.get("facial_area", {}).get("h", 0)) / total_frame_area >= 0.03
                )
            ]
            if not valid_faces and detected_faces:
                # If all were filtered, preserve the single largest face if it has at least 32px
                largest = max(
                    detected_faces,
                    key=lambda f: f.get("facial_area", {}).get("w", 0) * f.get("facial_area", {}).get("h", 0)
                )
                if largest.get("facial_area", {}).get("w", 0) >= 32 and largest.get("facial_area", {}).get("h", 0) >= 32:
                    valid_faces = [largest]

            # Sort faces by area descending so primary subject face is first
            valid_faces.sort(
                key=lambda f: f.get("facial_area", {}).get("w", 0) * f.get("facial_area", {}).get("h", 0),
                reverse=True
            )

            max_face_area = (
                valid_faces[0].get("facial_area", {}).get("w", 0) * valid_faces[0].get("facial_area", {}).get("h", 0)
                if valid_faces else 0
            )

            primary_scores: list[float] = []
            for detected in valid_faces:
                area = detected.get("facial_area", {})
                x, y, box_width, box_height = _clip_box(area, width, height)
                face_area = box_width * box_height
                bbox = {"x": x, "y": y, "w": box_width, "h": box_height}
                crop = _crop_face(frame, area)
                ai_result: dict[str, Any] = {"performed": False, "reason": "MODEL_UNAVAILABLE"}

                # A face is considered a prominent subject face if its area is at least 25% of the largest face
                is_subject_face = face_area >= (0.25 * max_face_area) if max_face_area > 0 else True

                if crop.size:
                    try:
                        result = _ai_detector.analyze(crop)
                        if result.get("label") == "Error" or result.get("inference_status") == "ERROR":
                            raise RuntimeError(result.get("error") or "Image classifier returned an error.")
                        score = float(result.get("ai_probability") if result.get("ai_probability") is not None else result["ai_confidence"])
                        frame_scores.append(score)

                        # Only prominent subject faces contribute to primary score
                        if is_subject_face:
                            primary_scores.append(score)

                        ai_result = {
                            "performed": True,
                            "result": "AI_GENERATED" if score >= AI_THRESHOLD else "REAL",
                            "score": round(score, 4),
                            "ai_probability": round(score, 4),
                            "real_probability": result.get("real_probability", round(1.0 - score, 4)),
                            "model": result.get("model_name") or result.get("model"),
                        }
                    except Exception as error:
                        ai_result = {"performed": False, "error": str(error), "reason": "MODEL_ERROR"}

                frame_faces.append({
                    "bbox": bbox,
                    "confidence": round(float(detected.get("confidence", 0) or 0), 4),
                    "ai_analysis": ai_result,
                })

        frame_analysis: dict[str, Any]
        if frame_scores:
            eval_scores = primary_scores if primary_scores else frame_scores
            # In multi-face frames, if any prominent subject face exhibits synthetic manipulation, score the frame accordingly
            frame_score = float(max(eval_scores))
            frame_flagged = frame_score >= AI_THRESHOLD
            model_scores.append(frame_score)
            flagged_frames += int(frame_flagged)
            frame_analysis = {
                "performed": True,
                "result": "AI_GENERATED" if frame_flagged else "REAL",
                "score": round(frame_score, 4),
                "ai_probability": round(frame_score, 4),
                "real_probability": round(1.0 - frame_score, 4),
                "model": _ai_detector.model_name,
                "faces_scored": len(frame_scores),
            }
        elif frame_faces:
            frame_analysis = {"performed": False, "reason": "MODEL_ERROR", "error": "No detected face crop produced a classifier score."}
        elif face_errors:
            frame_analysis = {"performed": False, "reason": "FACE_DETECTION_ERROR", "error": face_errors[0]}
        else:
            frame_analysis = {"performed": False, "reason": "NO_FACE_DETECTED"}

        frame_results.append({
            "frame_number": frame_number,
            "timestamp": timestamp,
            "faces_detected": len(frame_faces),
            "identity_matches": [],
            "protected_identity_detected": False,
            "faces": frame_faces,
            "ai_analysis": frame_analysis,
        })

        if frame_faces:
            try:
                with _inference_lock:
                    measurements = _depth_analyzer.analyze(frame, [face["bbox"] for face in frame_faces])
                for face, measurement in zip(frame_faces, measurements):
                    face["depth_analysis"] = measurement
                if measurements:
                    depth_rows.extend({"frame_number": frame_number, "timestamp": timestamp, **item} for item in measurements)
            except Exception as error:
                for face in frame_faces:
                    face["depth_analysis"] = {"available": False, "reason": str(error)}

    temporal = _associate_face_tracks(frame_results)
    frames_analyzed = len(samples)
    frames_scored = len(model_scores)

    try:
        metadata = _metadata_analyzer.analyze_video_file(path)
    except Exception as error:
        metadata = {"flags": [], "confidence": "unavailable", "summary": f"Metadata analysis unavailable: {error}"}

    metadata_flagged = metadata.get("confidence") in {"medium", "high"}
    warnings: list[str] = []
    if metadata_flagged:
        meta_flags = ", ".join(metadata.get("flags", [])) or "AI container fingerprint"
        warnings.append(f"Container metadata contains potential generator markers: {meta_flags}")

    if frames_scored == 0:
        final_status = "INCONCLUSIVE"
        verdict = "INCONCLUSIVE"
        ai_probability = 0.0
        real_probability = 0.0
        confidence = 0.0
        flagged_ratio = 0.0
        flagged_frames = 0
        reason = "No scorable human face was detected across sampled video frames."
    elif frames_scored == 1:
        s0 = model_scores[0]
        ai_probability = round(s0, 4)
        real_probability = round(1.0 - s0, 4)
        confidence = round(max(ai_probability, real_probability), 4)
        flagged_frames = 1 if s0 >= AI_THRESHOLD else 0
        flagged_ratio = float(flagged_frames)
        final_status = "INCONCLUSIVE"
        verdict = "INCONCLUSIVE"
        reason = "Only 1 sampled frame contained a scorable face; at least 2 scored frames are required for a reliable video verdict."
    else:
        scores = model_scores
        s_mean = float(statistics.mean(scores))
        s_med = float(statistics.median(scores))
        flagged_frames = sum(1 for s in scores if s >= AI_THRESHOLD)
        flagged_ratio = flagged_frames / frames_scored
        real_count = sum(1 for s in scores if s < 0.30)
        real_ratio = real_count / frames_scored

        # Calibrated weighted consensus (resistant to single-frame outliers)
        ai_probability = round(float(0.60 * s_med + 0.40 * s_mean), 4)
        real_probability = round(1.0 - ai_probability, 4)

        # Consistent decision logic
        if flagged_frames >= 2 and flagged_ratio >= 0.40 and ai_probability >= 0.40 and real_ratio < 0.40:
            final_status = "AI_GENERATED"
            verdict = "AI-GENERATED VIDEO"
            confidence = round(ai_probability, 4)
            reason = f"Consistent AI-generation markers detected across {flagged_frames} of {frames_scored} scored frames (aggregate AI probability: {ai_probability:.1%})."
        elif ai_probability <= 0.30 and (flagged_frames == 0 or (frames_scored >= 5 and flagged_frames <= 1 and ai_probability <= 0.20)) and real_ratio >= 0.60:
            final_status = "REAL_VIDEO"
            verdict = "REAL VIDEO"
            confidence = round(real_probability, 4)
            reason = f"Consistent authentic real-video characteristics across {frames_scored} sampled frames with no sustained synthetic manipulation (aggregate real probability: {real_probability:.1%})."
        else:
            final_status = "INCONCLUSIVE"
            verdict = "INCONCLUSIVE"
            confidence = round(max(ai_probability, real_probability), 4)
            reason = f"Ambiguous or conflicting signals across {frames_scored} sampled frames ({flagged_frames} flagged, {real_count} real, aggregate AI probability: {ai_probability:.1%}); evidence is insufficient for a decisive verdict."

    summary = f"{verdict}: {reason}"

    successful_depth = [row for row in depth_rows if "face_background_depth_delta" in row]
    depth_summary = {
        "available": bool(successful_depth),
        "model": DEPTH_MODEL_ID,
        "frames_analyzed": len({row["frame_number"] for row in successful_depth}),
        "face_regions_analyzed": len(successful_depth),
        "median_face_background_depth_delta": round(float(statistics.median(row["face_background_depth_delta"] for row in successful_depth)), 4) if successful_depth else None,
        "median_face_mask_depth_spread_iqr": round(float(statistics.median(row["face_mask_depth_spread_iqr"] for row in successful_depth)), 4) if successful_depth else None,
        "samples": depth_rows,
        "detail": "Depth Anything V2 estimates relative monocular depth. Face-region values use an elliptical mask inside detected face boxes; these are measurements, not metric depth, occlusion verdicts, or proof of manipulation.",
    }
    if not successful_depth:
        depth_summary["reason"] = _depth_analyzer.load_error or "No face-region depth measurements were produced."

    return {
        "status": "success",
        "final_status": final_status,
        "verdict": verdict,
        "ai_probability": ai_probability,
        "real_probability": real_probability,
        "confidence": confidence,
        "frames_analyzed": frames_analyzed,
        "frames_scored": frames_scored,
        "reason": reason,
        "model_name": getattr(_ai_detector, "model_name", "dima806/deepfake_vs_real_image_detection"),
        "warnings": warnings,
        "video": {
            "duration": float(video_meta["duration"]),
            "fps": float(video_meta["fps"]),
            "total_frames": int(video_meta["total_frames"]),
            "sampled_frames": len(samples),
            "width": int(video_meta["width"]),
            "height": int(video_meta["height"]),
            "filename": original_filename,
        },
        "identity": {
            "protected_identity_detected": False,
            "person_ids": [],
            "frames_with_identity": 0,
            "identity_frame_ratio": 0.0,
        },
        "ai_analysis": {
            "frames_analyzed": frames_analyzed,
            "frames_scored": frames_scored,
            "frames_flagged": flagged_frames,
            "flagged_frame_ratio": round(flagged_ratio, 4),
            "aggregate_score": ai_probability,
            "ai_probability": ai_probability,
            "real_probability": real_probability,
            "threshold": AI_THRESHOLD,
            "status": final_status,
            "model": getattr(_ai_detector, "model_name", None),
        },
        "depth_analysis": depth_summary,
        "temporal_analysis": temporal,
        "frames": frame_results,
        "summary": summary,
        "metadata_forensics": metadata,
        "limitations": [
            "The image classifier is applied to sampled face crops and is evaluated as a frame-level visual screening model.",
            "Depth output is relative monocular depth from an estimated face-region mask, not metric depth or a 3D liveness check.",
            "Temporal output associates detected face boxes only; it is not general object permanence or identity tracking.",
            "Frame sampling may miss brief artifacts between samples; calibrated probability reflects consensus across sampled frames.",
        ],
    }


@app.on_event("startup")
def startup_event() -> None:
    global _ai_detector, _metadata_analyzer, _face_model_ready
    _ai_detector = AIImageDetector()
    _metadata_analyzer = MetadataAnalyzer()
    _face_model_ready = True


@app.get("/")
def health_check() -> dict[str, Any]:
    return {
        "status": "Video Detection Model video API is running",
        "version": app.version,
        "mode": "video-authenticity-screening",
        "identity_matching": False,
        "face_model_ready": _face_model_ready,
        "depth_model": DEPTH_MODEL_ID,
        "depth_model_loaded": _depth_analyzer.available,
    }


@app.post("/api/scan-video")
async def scan_video(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = Path(file.filename or "upload.mp4").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported video format.")
    if _ai_detector is None or _metadata_analyzer is None:
        raise HTTPException(status_code=503, detail="Video analysis models are not initialized.")

    temp_path: str | None = None
    size = 0
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=f"Video exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
                temp_file.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="The uploaded video is empty.")
        return await run_in_threadpool(_analyze_video, temp_path, filename)
    finally:
        await file.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)