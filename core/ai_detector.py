"""
AI-generated image detector module for the SWARAKSHA project.
Uses a HuggingFace image classification model to detect deepfakes/AI-generated faces.
"""
import sys
import os

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add parent directory to path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config

import numpy as np
import cv2
from PIL import Image
import torch
from transformers import AutoModelForImageClassification

# Try ViTImageProcessor first (works with SadraCoding model),
# fall back to AutoImageProcessor for other models.
try:
    from transformers import ViTImageProcessor as _ImageProcessor
except ImportError:
    from transformers import AutoImageProcessor as _ImageProcessor


class AIImageDetector:
    """
    Detector for identifying AI-generated or deepfake face images.
    """
    def __init__(self, model_name=None, device=None):
        """
        Initialize the AIImageDetector.
        
        Args:
            model_name (str): The HuggingFace model identifier.
            device (str): Compute device ('cuda' or 'cpu').
        """
        self.model_name = model_name or getattr(config, 'AI_DETECTOR_MODEL', 'SadraCoding/SDXL-Deepfake-Detector')
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"[SWARAKSHA] Loading AI Detector model '{self.model_name}' on {self.device}...")
        
        try:
            # Use ViTImageProcessor for SadraCoding model compatibility
            try:
                self.processor = _ImageProcessor.from_pretrained(self.model_name)
            except Exception:
                from transformers import AutoImageProcessor
                self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            
            # Parse id2label mapping (handling both integer and string keys)
            raw_id2label = getattr(self.model.config, 'id2label', {}) or {}
            self.id2label = {}
            for k, v in raw_id2label.items():
                try:
                    self.id2label[int(k)] = str(v)
                except (ValueError, TypeError):
                    self.id2label[k] = str(v)

            # Determine class indices for Real vs Fake
            self.ai_idx = -1
            self.real_idx = -1
            for idx, label in self.id2label.items():
                label_lower = str(label).lower()
                if any(kw in label_lower for kw in ['fake', 'artificial', 'ai', 'synthetic', 'generated', 'deepfake']):
                    self.ai_idx = int(idx)
                elif any(kw in label_lower for kw in ['real', 'authentic', 'genuine', 'human', 'original']):
                    self.real_idx = int(idx)

            if self.ai_idx == -1 and self.real_idx != -1:
                self.ai_idx = 1 - self.real_idx
            elif self.real_idx == -1 and self.ai_idx != -1:
                self.real_idx = 1 - self.ai_idx
            elif self.ai_idx == -1 and self.real_idx == -1:
                self.real_idx = 0
                self.ai_idx = 1

            print(f"  [OK] AI Detector loaded — labels: {self.id2label} (real_idx={self.real_idx}, ai_idx={self.ai_idx})")
        except Exception as e:
            print(f"  [ERROR] Error loading model: {e}")
            raise

    def analyze(self, face_image) -> dict:
        """
        Analyze a face crop image to determine if it is AI-generated.
        
        Args:
            face_image: Image path (str), BGR numpy array (cv2), or PIL Image.
            
        Returns:
            dict: Authoritative result dictionary containing normalized probabilities,
                  predicted class, inference status, and image quality metrics.
        """
        try:
            if face_image is None:
                raise ValueError("Input image is None.")

            # Process input image
            if isinstance(face_image, str):
                if not os.path.exists(face_image):
                    raise FileNotFoundError(f"Image file not found: {face_image}")
                image = Image.open(face_image).convert('RGB')
                sharpness = 100.0
            elif isinstance(face_image, np.ndarray):
                if face_image.size == 0:
                    raise ValueError("Empty image array.")
                image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image_rgb).convert('RGB')
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if face_image.ndim == 3 else face_image
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            elif isinstance(face_image, Image.Image):
                image = face_image.convert('RGB')
                sharpness = 100.0
            else:
                raise ValueError("Unsupported image type. Must be str (path), numpy array, or PIL Image.")

            # Prepare inputs
            inputs = self.processor(images=image, return_tensors='pt')
            if hasattr(inputs, 'to'):
                inputs = inputs.to(self.device)
            elif isinstance(inputs, dict):
                inputs = {k: (v.to(self.device) if hasattr(v, 'to') else v) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            
            raw_ai = float(probs[self.ai_idx].item())
            raw_real = float(probs[self.real_idx].item())
            
            # Normalize so that real_probability + ai_probability == 1.0 strictly
            total = raw_ai + raw_real
            if total > 0:
                ai_probability = round(raw_ai / total, 4)
                real_probability = round(1.0 - ai_probability, 4)
            else:
                ai_probability = 0.5
                real_probability = 0.5
            
            threshold = float(getattr(config, 'AI_DETECTOR_THRESHOLD', 0.65))
            is_ai = ai_probability >= threshold
            predicted_class = "AI-Generated" if is_ai else "Real"
            is_blurry = sharpness < 10.0
            
            return {
                # Authoritative fields
                'real_probability': real_probability,
                'ai_probability': ai_probability,
                'predicted_class': predicted_class,
                'model_name': self.model_name,
                'inference_status': 'SUCCESS',
                'error': None,
                # Backward-compatible fields
                'is_ai': is_ai,
                'ai_confidence': ai_probability,
                'real_confidence': real_probability,
                'sharpness': round(sharpness, 1),
                'is_blurry': is_blurry,
                'label': predicted_class,
                'model': self.model_name,
            }
            
        except Exception as e:
            return {
                # Authoritative fields
                'real_probability': None,
                'ai_probability': None,
                'predicted_class': 'Error',
                'model_name': getattr(self, 'model_name', 'Unknown'),
                'inference_status': 'ERROR',
                'error': str(e),
                # Backward-compatible fields
                'is_ai': False,
                'ai_confidence': 0.0,
                'real_confidence': 0.0,
                'sharpness': 0.0,
                'is_blurry': False,
                'label': 'Error',
                'model': getattr(self, 'model_name', 'Unknown'),
            }

    def analyze_batch(self, face_images: list) -> list:
        """
        Analyze a batch of face images.
        
        Args:
            face_images (list): List of images to analyze.
            
        Returns:
            list: List of result dictionaries.
        """
        return [self.analyze(img) for img in face_images]


def preload_detector() -> AIImageDetector:
    """
    Preload and return the AIImageDetector instance.
    
    Returns:
        AIImageDetector: Initialized detector instance.
    """
    print("Preloading AI Image Detector...")
    try:
        detector = AIImageDetector()
        return detector
    except Exception as e:
        print(f"Failed to preload AI Image Detector: {e}")
        return None
