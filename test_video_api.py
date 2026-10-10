import sys
import os

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np
import torch
from fastapi.testclient import TestClient

import config
import api.video_app as video_app
from core.ai_detector import AIImageDetector


class ClassifierUnitTests(unittest.TestCase):
    """Unit tests for core/ai_detector.py."""

    def test_classifier_label_mapping_when_indices_reversed(self):
        """1. Verify label mapping when index 0 is Fake and index 1 is Real."""
        detector = object.__new__(AIImageDetector)
        detector.model_name = "test-model"
        detector.device = "cpu"
        detector.model = MagicMock()
        detector.processor = MagicMock()

        # Reversed order: 0 = Fake, 1 = Real
        detector.model.config.id2label = {0: "Fake", 1: "Real"}
        detector.id2label = {0: "Fake", 1: "Real"}
        detector.ai_idx = 0
        detector.real_idx = 1

        # Mock model output: logit for 0 (Fake) is lower, logit for 1 (Real) is higher
        logits = torch.tensor([[1.0, 4.0]])  # softmax: ~0.047 for Fake, ~0.953 for Real
        detector.model.return_value = SimpleNamespace(logits=logits)
        detector.processor.return_value = {"images": torch.zeros((1, 3, 224, 224))}

        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.analyze(dummy_img)

        self.assertEqual(result["inference_status"], "SUCCESS")
        self.assertEqual(result["predicted_class"], "Real")
        self.assertAlmostEqual(result["real_probability"] + result["ai_probability"], 1.0, places=3)
        self.assertGreater(result["real_probability"], 0.90)
        self.assertLess(result["ai_probability"], 0.10)

    def test_classifier_label_mapping_normal_order(self):
        """1b. Verify label mapping when index 0 is Real and index 1 is Fake."""
        detector = object.__new__(AIImageDetector)
        detector.model_name = "test-model"
        detector.device = "cpu"
        detector.model = MagicMock()
        detector.processor = MagicMock()

        detector.model.config.id2label = {0: "Real", 1: "Fake"}
        detector.id2label = {0: "Real", 1: "Fake"}
        detector.real_idx = 0
        detector.ai_idx = 1

        logits = torch.tensor([[1.0, 4.0]])  # index 1 (Fake) is higher
        detector.model.return_value = SimpleNamespace(logits=logits)
        detector.processor.return_value = {"images": torch.zeros((1, 3, 224, 224))}

        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.analyze(dummy_img)

        self.assertEqual(result["inference_status"], "SUCCESS")
        self.assertEqual(result["predicted_class"], "AI-Generated")
        self.assertAlmostEqual(result["real_probability"] + result["ai_probability"], 1.0, places=3)
        self.assertGreater(result["ai_probability"], 0.90)
        self.assertLess(result["real_probability"], 0.10)

    def test_probability_calculation_from_synthetic_logits(self):
        """2. Verify exact softmax and normalization from synthetic logits."""
        detector = object.__new__(AIImageDetector)
        detector.model_name = "test-model"
        detector.device = "cpu"
        detector.model = MagicMock()
        detector.processor = MagicMock()
        detector.id2label = {0: "Real", 1: "Fake"}
        detector.real_idx = 0
        detector.ai_idx = 1

        # Equal logits -> exact 50/50 probabilities
        logits = torch.tensor([[2.5, 2.5]])
        detector.model.return_value = SimpleNamespace(logits=logits)
        detector.processor.return_value = {"images": torch.zeros((1, 3, 224, 224))}

        dummy_img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = detector.analyze(dummy_img)

        self.assertEqual(result["real_probability"], 0.5)
        self.assertEqual(result["ai_probability"], 0.5)
        self.assertEqual(result["real_probability"] + result["ai_probability"], 1.0)

    def test_classifier_failure_does_not_fabricate_scores(self):
        """10. Model inference failure returns ERROR status without fabricating scores."""
        detector = object.__new__(AIImageDetector)
        detector.model_name = "test-model"
        detector.device = "cpu"
        detector.model = MagicMock(side_effect=RuntimeError("CUDA out of memory"))
        detector.processor = MagicMock()
        detector.id2label = {0: "Real", 1: "Fake"}
        detector.real_idx = 0
        detector.ai_idx = 1

        dummy_img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = detector.analyze(dummy_img)

        self.assertEqual(result["inference_status"], "ERROR")
        self.assertIsNone(result["real_probability"])
        self.assertIsNone(result["ai_probability"])
        self.assertEqual(result["predicted_class"], "Error")
        self.assertIn("CUDA out of memory", result["error"])


class VideoAggregationTests(unittest.TestCase):
    """Unit tests for aggregation and decision logic in api/video_app.py."""

    def _run_pipeline(self, frame_scores_list, metadata_flags=None, metadata_conf="none"):
        """Helper to run _analyze_video with synthetic frame scores."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        sample_frames = []
        for i in range(len(frame_scores_list)):
            sample_frames.append({
                "frame_number": i * 60,
                "timestamp": float(i * 2.0),
                "frame": frame,
            })

        face_idx = [0]
        def mock_extract_faces(f):
            idx = face_idx[0]
            face_idx[0] += 1
            if idx < len(frame_scores_list):
                score = frame_scores_list[idx]
                if score is None:
                    return []
                return [{"facial_area": {"x": 20, "y": 20, "w": 40, "h": 40}, "confidence": 0.99}]
            return []

        call_idx = [0]
        def mock_analyze(crop):
            idx = call_idx[0]
            call_idx[0] += 1
            score = frame_scores_list[idx] if idx < len(frame_scores_list) else 0.5
            return {
                "real_probability": round(1.0 - score, 4),
                "ai_probability": round(score, 4),
                "predicted_class": "AI-Generated" if score >= 0.65 else "Real",
                "model_name": "test-detector",
                "inference_status": "SUCCESS",
                "error": None,
                "is_ai": score >= 0.65,
                "ai_confidence": round(score, 4),
                "real_confidence": round(1.0 - score, 4),
                "sharpness": 100.0,
                "is_blurry": False,
                "label": "AI-Generated" if score >= 0.65 else "Real",
                "model": "test-detector",
            }

        detector = SimpleNamespace(model_name="test-detector", analyze=mock_analyze)
        metadata = SimpleNamespace(analyze_video_file=lambda path: {
            "flags": metadata_flags or [],
            "confidence": metadata_conf,
        })

        with (
            patch.object(video_app, "extract_video_metadata", return_value={
                "width": 100, "height": 100, "fps": 30.0, "total_frames": len(frame_scores_list) * 60, "duration": len(frame_scores_list) * 2.0
            }),
            patch.object(video_app, "sample_video_frames", return_value=sample_frames),
            patch.object(video_app, "extract_faces", side_effect=mock_extract_faces),
            patch.object(video_app, "_ai_detector", detector),
            patch.object(video_app, "_metadata_analyzer", metadata),
            patch.object(video_app._depth_analyzer, "analyze", return_value=[]),
        ):
            return video_app._analyze_video("dummy.mp4", "dummy.mp4")

    def test_clearly_real_video_produces_real_video_verdict(self):
        """3. Clearly real-class set of frame predictions produces REAL_VIDEO."""
        scores = [0.05, 0.08, 0.04, 0.06, 0.07, 0.05]
        result = self._run_pipeline(scores)

        self.assertEqual(result["final_status"], "REAL_VIDEO")
        self.assertEqual(result["verdict"], "REAL VIDEO")
        self.assertLess(result["ai_probability"], 0.20)
        self.assertGreater(result["real_probability"], 0.80)
        self.assertEqual(result["confidence"], result["real_probability"])
        self.assertEqual(result["frames_scored"], 6)

    def test_clearly_ai_video_produces_ai_generated_verdict(self):
        """4. Clearly AI-class set of frame predictions produces AI_GENERATED."""
        scores = [0.88, 0.91, 0.85, 0.89, 0.94, 0.87]
        result = self._run_pipeline(scores)

        self.assertEqual(result["final_status"], "AI_GENERATED")
        self.assertEqual(result["verdict"], "AI-GENERATED VIDEO")
        self.assertGreater(result["ai_probability"], 0.80)
        self.assertLess(result["real_probability"], 0.20)
        self.assertEqual(result["confidence"], result["ai_probability"])
        self.assertEqual(result["frames_scored"], 6)

    def test_single_false_positive_frame_among_real_frames_remains_real(self):
        """5. One false-positive frame among several real frames does NOT flip verdict to AI."""
        # 6 frames: 5 clearly real (5%), 1 anomalous spike (72%)
        scores = [0.05, 0.08, 0.72, 0.04, 0.06, 0.05]
        result = self._run_pipeline(scores)

        self.assertEqual(result["final_status"], "REAL_VIDEO")
        self.assertEqual(result["verdict"], "REAL VIDEO")
        self.assertLessEqual(result["ai_probability"], 0.25)
        self.assertGreater(result["real_probability"], 0.75)

    def test_conflicting_predictions_produce_inconclusive(self):
        """6. Conflicting predictions (half real, half AI) produce INCONCLUSIVE."""
        scores = [0.85, 0.88, 0.10, 0.12]
        result = self._run_pipeline(scores)

        self.assertEqual(result["final_status"], "INCONCLUSIVE")
        self.assertEqual(result["verdict"], "INCONCLUSIVE")
        self.assertIn("Ambiguous or conflicting signals", result["reason"])

    def test_no_scored_frames_produces_inconclusive(self):
        """7. No detected face or no scored frames produces INCONCLUSIVE."""
        scores = [None, None, None]
        result = self._run_pipeline(scores)

        self.assertEqual(result["final_status"], "INCONCLUSIVE")
        self.assertEqual(result["verdict"], "INCONCLUSIVE")
        self.assertEqual(result["frames_scored"], 0)
        self.assertEqual(result["ai_probability"], 0.0)
        self.assertEqual(result["real_probability"], 0.0)

    def test_metadata_evidence_does_not_override_classifier(self):
        """11. Metadata flags alone do not override classifier verdict on authentic video."""
        scores = [0.05, 0.06, 0.04, 0.07, 0.05]
        result = self._run_pipeline(scores, metadata_flags=["FFmpeg AI container tag"], metadata_conf="high")

        self.assertEqual(result["final_status"], "REAL_VIDEO")
        self.assertEqual(result["verdict"], "REAL VIDEO")
        self.assertTrue(any("metadata" in w.lower() for w in result["warnings"]))
        self.assertEqual(result["metadata_forensics"]["confidence"], "high")

    def test_stable_json_response_schema(self):
        """12. Verify exact stable output schema fields and valid final status values."""
        scores = [0.05, 0.06, 0.04, 0.05]
        result = self._run_pipeline(scores)

        required_keys = [
            "status", "final_status", "verdict", "ai_probability", "real_probability",
            "confidence", "frames_analyzed", "frames_scored", "reason", "model_name", "warnings"
        ]
        for key in required_keys:
            self.assertIn(key, result, f"Missing required top-level key: {key}")

        self.assertIn(result["final_status"], ["REAL_VIDEO", "AI_GENERATED", "INCONCLUSIVE"])
        self.assertIn(result["verdict"], ["REAL VIDEO", "AI-GENERATED VIDEO", "INCONCLUSIVE"])
        self.assertAlmostEqual(result["ai_probability"] + result["real_probability"], 1.0, places=3)


class VideoApiEndpointTests(unittest.TestCase):
    """End-to-end API tests using FastAPI TestClient."""

    def setUp(self):
        video_app._ai_detector = SimpleNamespace(model_name="test-model", analyze=lambda x: {
            "ai_probability": 0.5, "real_probability": 0.5, "predicted_class": "Real",
            "inference_status": "SUCCESS", "is_ai": False, "ai_confidence": 0.5,
            "real_confidence": 0.5, "label": "Real", "model": "test-model"
        })
        video_app._metadata_analyzer = SimpleNamespace(analyze_video_file=lambda x: {"flags": [], "confidence": "none"})
        self.client = TestClient(video_app.app)

    def test_health_check_endpoint(self):
        """Health check returns 200 with mode and depth info."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "Video Detection Model video API is running")
        self.assertFalse(data["identity_matching"])

    def test_unsupported_file_extension_rejected(self):
        """9. Uploading non-video format returns HTTP 400."""
        response = self.client.post(
            "/api/scan-video",
            files={"file": ("report.pdf", b"%PDF-1.4...", "application/pdf")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported video format", response.json()["detail"])

    def test_empty_video_upload_rejected(self):
        """8. Uploading 0-byte video returns HTTP 400."""
        response = self.client.post(
            "/api/scan-video",
            files={"file": ("empty.mp4", b"", "video/mp4")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json()["detail"].lower())



class CandidateModelEvaluationTests(unittest.TestCase):
    """Tests evaluating the reference ResNeXt+LSTM model candidate and its blockers."""

    def test_candidate_dependency_audit_detects_missing_torchvision_and_dlib(self):
        """15. Candidate dependency check detects missing torchvision/dlib gracefully."""
        from candidate_resnext_lstm_evaluator import check_reference_dependencies
        deps = check_reference_dependencies()
        self.assertIn("torch", deps)
        self.assertIn("torchvision", deps)
        self.assertIn("dlib", deps)
        self.assertIn("face_recognition", deps)
        self.assertTrue(deps["torch"]["available"])

    def test_candidate_missing_checkpoint_does_not_fabricate_weights(self):
        """16. Missing checkpoint is correctly reported and no fabricated predictions are made."""
        from candidate_resnext_lstm_evaluator import search_for_checkpoints
        checkpoints = search_for_checkpoints()
        has_trained_weights = any("model_checkpoint.pt" in cp for cp in checkpoints)
        self.assertFalse(has_trained_weights)

    def test_candidate_evaluator_generates_valid_report(self):
        """17. Candidate evaluator executes cleanly and produces structured report."""
        from candidate_resnext_lstm_evaluator import evaluate_candidate
        report = evaluate_candidate()
        self.assertIn("Model A (Current Video Detection Model ViT)", report)
        self.assertIn("Model B (Reference ResNeXt-50 + LSTM)", report)
        model_b = report["Model B (Reference ResNeXt-50 + LSTM)"]
        self.assertEqual(model_b["Status"], "BLOCKED (Missing weights, broken dependencies, licensing conflict)")
        self.assertIn("GNU General Public License", model_b["License"])


if __name__ == "__main__":
    unittest.main()