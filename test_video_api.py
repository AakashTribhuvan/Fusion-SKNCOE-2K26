import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import api.video_app as video_app


class VideoApiAnalysisTests(unittest.TestCase):
    def test_video_health_does_not_advertise_identity_matching(self):
        response = TestClient(video_app.app).get("/")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["identity_matching"])
        self.assertEqual(response.json()["mode"], "video-authenticity-screening")

    def test_scan_endpoint_rejects_unsupported_files(self):
        response = TestClient(video_app.app).post(
            "/api/scan-video",
            files={"file": ("sample.txt", b"not a video", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Unsupported video format.")

    def test_video_scan_scores_face_crop_without_identity_lookup(self):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        face = {"facial_area": {"x": 20, "y": 15, "w": 30, "h": 36}, "confidence": 0.98}
        depth = {
            "face_mask_median_relative_depth": 0.72,
            "face_mask_depth_spread_iqr": 0.11,
            "face_background_depth_delta": 0.18,
        }
        detector = SimpleNamespace(
            model_name="test-image-classifier",
            analyze=lambda crop: {
                "label": "Real",
                "is_ai": False,
                "ai_confidence": 0.12,
                "model": "test-image-classifier",
            },
        )
        metadata = SimpleNamespace(analyze_video_file=lambda path: {"flags": [], "confidence": "none"})

        with (
            patch.object(video_app, "extract_video_metadata", return_value={
                "width": 100,
                "height": 80,
                "fps": 30.0,
                "total_frames": 60,
                "duration": 2.0,
            }),
            patch.object(video_app, "sample_video_frames", return_value=[{
                "frame_number": 0,
                "timestamp": 0.0,
                "frame": frame,
            }]),
            patch.object(video_app, "extract_faces", return_value=[face]),
            patch.object(video_app, "_ai_detector", detector),
            patch.object(video_app, "_metadata_analyzer", metadata),
            patch.object(video_app._depth_analyzer, "analyze", return_value=[depth]),
        ):
            result = video_app._analyze_video("unused.mp4", "sample.mp4")

        self.assertEqual(result["ai_analysis"]["frames_analyzed"], 1)
        self.assertEqual(result["frames"][0]["ai_analysis"]["score"], 0.12)
        self.assertFalse(result["identity"]["protected_identity_detected"])
        self.assertEqual(result["depth_analysis"]["face_regions_analyzed"], 1)
        self.assertTrue(result["frames"][0]["faces"][0]["depth_analysis"])

    def test_face_boxes_persist_across_adjacent_samples(self):
        frames = [
            {"faces": [{"bbox": {"x": 10, "y": 10, "w": 30, "h": 30}}]},
            {"faces": [{"bbox": {"x": 12, "y": 11, "w": 30, "h": 30}}]},
        ]

        summary = video_app._associate_face_tracks(frames)

        self.assertEqual(summary["tracks_detected"], 1)
        self.assertEqual(summary["persistent_tracks"], 1)
        self.assertEqual(frames[0]["faces"][0]["track_id"], frames[1]["faces"][0]["track_id"])

    def test_no_scored_face_is_inconclusive_not_clear(self):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        metadata = SimpleNamespace(analyze_video_file=lambda path: {"flags": [], "confidence": "none"})

        with (
            patch.object(video_app, "extract_video_metadata", return_value={
                "width": 100,
                "height": 80,
                "fps": 30.0,
                "total_frames": 60,
                "duration": 2.0,
            }),
            patch.object(video_app, "sample_video_frames", return_value=[{
                "frame_number": 0,
                "timestamp": 0.0,
                "frame": frame,
            }]),
            patch.object(video_app, "extract_faces", return_value=[]),
            patch.object(video_app, "_metadata_analyzer", metadata),
        ):
            result = video_app._analyze_video("unused.mp4", "sample.mp4")

        self.assertEqual(result["ai_analysis"]["status"], "NOT_ANALYZED")
        self.assertEqual(result["final_status"], "INCONCLUSIVE")

    def test_metadata_finding_does_not_masquerade_as_classifier_flag(self):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        face = {"facial_area": {"x": 20, "y": 15, "w": 30, "h": 36}, "confidence": 0.98}
        detector = SimpleNamespace(
            model_name="test-image-classifier",
            analyze=lambda crop: {
                "label": "Real",
                "is_ai": False,
                "ai_confidence": 0.12,
                "model": "test-image-classifier",
            },
        )
        metadata = SimpleNamespace(analyze_video_file=lambda path: {
            "flags": ["Container marker for review"],
            "confidence": "high",
        })

        with (
            patch.object(video_app, "extract_video_metadata", return_value={
                "width": 100,
                "height": 80,
                "fps": 30.0,
                "total_frames": 60,
                "duration": 2.0,
            }),
            patch.object(video_app, "sample_video_frames", return_value=[{
                "frame_number": 0,
                "timestamp": 0.0,
                "frame": frame,
            }]),
            patch.object(video_app, "extract_faces", return_value=[face]),
            patch.object(video_app, "_ai_detector", detector),
            patch.object(video_app, "_metadata_analyzer", metadata),
            patch.object(video_app._depth_analyzer, "analyze", return_value=[]),
        ):
            result = video_app._analyze_video("unused.mp4", "sample.mp4")

        self.assertEqual(result["ai_analysis"]["frames_flagged"], 0)
        self.assertEqual(result["ai_analysis"]["status"], "NO_STRONG_AI_EVIDENCE")
        self.assertEqual(result["final_status"], "REVIEW_REQUIRED")
        self.assertIn("metadata", result["summary"])


if __name__ == "__main__":
    unittest.main()