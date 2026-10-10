"""
Video Detection Model — Face Encoder Module (AI Branch 1: Identity)

Uses DeepFace with RetinaFace detector and ArcFace model to:
- Detect faces in images/video frames
- Generate 512-dimensional facial embeddings
- Provide L2-normalized embeddings for FAISS cosine similarity search
"""

import sys
import os

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import cv2
from deepface import DeepFace
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def generate_embedding(image_input) -> np.ndarray:
    """
    Generate a 512-d ArcFace embedding from a face image.

    Args:
        image_input: File path (str) OR numpy array (BGR from OpenCV)

    Returns:
        np.ndarray: L2-normalized 512-d embedding vector (float32)

    Raises:
        ValueError: If no face is detected and enforce_detection is True
    """
    # Convert BGR numpy array to RGB for DeepFace
    if isinstance(image_input, np.ndarray):
        img = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
    else:
        img = image_input

    results = DeepFace.represent(
        img_path=img,
        model_name=config.FACE_MODEL,
        detector_backend=config.DETECTOR_BACKEND,
        enforce_detection=True,
        align=True,
    )

    if not results:
        raise ValueError("No face detected in the image.")

    # Take the first (highest-confidence) face
    embedding = np.array(results[0]["embedding"], dtype=np.float32)

    # L2-normalize for cosine similarity in FAISS
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def extract_faces(frame: np.ndarray, min_confidence: float = None) -> list:
    """
    Extract all face crops from a video frame using RetinaFace.

    Args:
        frame: BGR numpy array (from OpenCV VideoCapture)
        min_confidence: Minimum detection confidence (default from config)

    Returns:
        List of dicts: [{
            'face': np.ndarray (float32, 0-1, RGB),
            'facial_area': {'x': int, 'y': int, 'w': int, 'h': int},
            'confidence': float
        }]
    """
    if min_confidence is None:
        min_confidence = config.FACE_CONFIDENCE_MIN

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        faces = DeepFace.extract_faces(
            img_path=rgb_frame,
            detector_backend=config.DETECTOR_BACKEND,
            enforce_detection=False,
            align=True,
        )
    except Exception:
        return []

    # Discard dummy whole-image crops or detections below confidence
    valid_faces = [f for f in faces if float(f.get("confidence", 0) or 0) >= min_confidence]

    # Sort faces by bounding box area descending so primary subject face comes first
    valid_faces.sort(
        key=lambda f: f.get("facial_area", {}).get("w", 0) * f.get("facial_area", {}).get("h", 0),
        reverse=True
    )
    return valid_faces


def generate_frame_embeddings(frame: np.ndarray) -> list:
    """
    Combined face extraction + embedding generation for a video frame.
    Designed for the video processing pipeline.

    Args:
        frame: BGR numpy array (from OpenCV VideoCapture)

    Returns:
        List of dicts: [{
            'embedding': np.ndarray (512-d, L2-normalized),
            'facial_area': {'x', 'y', 'w', 'h'},
            'confidence': float
        }]
    """
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        results = DeepFace.represent(
            img_path=rgb_frame,
            model_name=config.FACE_MODEL,
            detector_backend=config.DETECTOR_BACKEND,
            enforce_detection=False,
            align=True,
        )
    except Exception:
        return []

    embeddings = []
    for res in results:
        conf = res.get("face_confidence", res.get("confidence", 0))
        if conf < config.FACE_CONFIDENCE_MIN:
            continue

        emb = np.array(res["embedding"], dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        embeddings.append({
            "embedding": emb,
            "facial_area": res["facial_area"],
            "confidence": conf,
        })

    return embeddings


def preload_models():
    """
    Pre-download and cache model weights (RetinaFace + ArcFace).
    Call this once at application startup to avoid first-frame latency.
    """
    print("[VIDEO DETECTION MODEL] Pre-loading face detection models...")
    DeepFace.build_model(config.FACE_MODEL)
    print(f"  ✓ {config.FACE_MODEL} model loaded")
    print(f"  ✓ Detector backend: {config.DETECTOR_BACKEND}")
    print("[VIDEO DETECTION MODEL] Models ready.")
