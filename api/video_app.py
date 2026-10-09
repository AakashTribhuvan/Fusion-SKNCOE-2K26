"""Video-only screening API for the Frame/Check frontend.

This app reuses Swaraksha's face detector, image classifier, frame sampler, and
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
MAX_UPLOAD_BYTES = int(os.getenv("SWARAKSHA_MAX_VIDEO_BYTES", str(250 * 1024 * 1024)))
MAX_VIDEO_SECONDS = float(os.getenv("SWARAKSHA_MAX_VIDEO_SECONDS", "180"))
VIDEO_SAMPLE_INTERVAL = float(getattr(config, "VIDEO_SAMPLE_INTERVAL", 2.0))
DEPTH_MODEL_ID = os.getenv("SWARAKSHA_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
AI_THRESHOLD = float(getattr(config, "AI_DETECTOR_THRESHOLD", 0.85))
_inference_lock = threading.RLock()

app = FastAPI(
    title="SWARAKSHA Video Screening API",
    description="Video-only sampled-frame authenticity screening; no identity enrollment or matching.",
    version="3.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        *[origin.strip() for origin in os.getenv("SWARAKSHA_FRONTEND_ORIGINS", "").split(",") if origin.strip()],
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
    padding = int(max(width, height) * 0.15)
    left, top = max(0, x - padding), max(0, y - padding)
    right, bottom = min(image_width, x + width + padding), min(image_height, y + height + padding)
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

            for detected in detected_faces:
                area = detected.get("facial_area", {})
                x, y, box_width, box_height = _clip_box(area, width, height)
                bbox = {"x": x, "y": y, "w": box_width, "h": box_height}
                crop = _crop_face(frame, area)
                ai_result: dict[str, Any] = {"performed": False, "reason": "MODEL_UNAVAILABLE"}
                if crop.size:
                    try:
                        result = _ai_detector.analyze(crop)
                        if result.get("label") == "Error":
                            raise RuntimeError("Image classifier returned an error.")
                        score = float(result["ai_confidence"])
                        frame_scores.append(score)
                        frame_flagged = frame_flagged or bool(result["is_ai"])
                        ai_result = {
                            "performed": True,
                            "result": "AI_GENERATED" if result["is_ai"] else "REAL",
                            "score": round(score, 4),
                            "model": result.get("model"),
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
            frame_score = max(frame_scores)
            model_scores.append(frame_score)
            flagged_frames += int(frame_flagged)
            frame_analysis = {
                "performed": True,
                "result": "AI_GENERATED" if frame_flagged else "REAL",
                "score": round(frame_score, 4),
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
    frames_analyzed = len(model_scores)
    flagged_ratio = flagged_frames / frames_analyzed if frames_analyzed else 0.0
    aggregate_score = float(statistics.median(model_scores)) if model_scores else 0.0

    try:
        metadata = _metadata_analyzer.analyze_video_file(path)
    except Exception as error:
        metadata = {"flags": [], "confidence": "unavailable", "summary": f"Metadata analysis unavailable: {error}"}

    metadata_flagged = metadata.get("confidence") in {"medium", "high"}
    classifier_flagged = flagged_ratio >= 0.3 or (flagged_frames > 0 and aggregate_score >= AI_THRESHOLD)

    if frames_analyzed == 0:
        ai_status = "NOT_ANALYZED"
        final_status = "INCONCLUSIVE"
        summary = "INCONCLUSIVE: no sampled face crop produced a classifier score."
    elif classifier_flagged:
        ai_status = "POTENTIAL_AI_MANIPULATION"
        final_status = "POTENTIAL_AI_MANIPULATION"
        summary = f"REVIEW REQUIRED: {flagged_frames} of {frames_analyzed} analyzed frame(s) were flagged by the image classifier."
    elif flagged_frames:
        ai_status = "REVIEW_REQUIRED"
        final_status = "REVIEW_REQUIRED"
        summary = f"REVIEW REQUIRED: the classifier flagged {flagged_frames} sampled frame(s)."
    elif metadata_flagged:
        ai_status = "NO_STRONG_AI_EVIDENCE"
        final_status = "REVIEW_REQUIRED"
        summary = "REVIEW REQUIRED: container metadata has markers for review; the image classifier did not flag the sampled faces."
    else:
        ai_status = "NO_STRONG_AI_EVIDENCE"
        final_status = "NO_STRONG_AI_EVIDENCE"
        summary = f"No sampled face crop was flagged across {frames_analyzed} analyzed frame(s); this does not establish authenticity."

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
            "frames_flagged": flagged_frames,
            "flagged_frame_ratio": round(flagged_ratio, 4),
            "aggregate_score": round(aggregate_score, 4),
            "threshold": AI_THRESHOLD,
            "status": ai_status,
            "model": getattr(_ai_detector, "model_name", None),
        },
        "depth_analysis": depth_summary,
        "temporal_analysis": temporal,
        "final_status": final_status,
        "frames": frame_results,
        "summary": summary,
        "metadata_forensics": metadata,
        "limitations": [
            "The image classifier is applied to sampled face crops and is not a validated video-level deepfake detector.",
            "Depth output is relative monocular depth from an estimated face-region mask, not metric depth or a 3D liveness check.",
            "Temporal output associates detected face boxes only; it is not general object permanence or identity tracking.",
            "Frame sampling may miss brief artifacts between samples; model scores are not calibrated probabilities.",
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
        "status": "SWARAKSHA video API is running",
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