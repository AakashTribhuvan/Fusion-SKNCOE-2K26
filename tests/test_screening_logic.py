import io
import os
import unittest
import wave
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import av
import numpy as np
import torch
from fastapi import HTTPException
from fastapi.testclient import TestClient
from backend.app.audit_ledger import SimulatedAuditLedger
from backend.app import main as main_module
from backend.app import screening as screening_module
from backend.app import voice_screening as voice_module
from backend.app.main import (
    QR_TARGET_ZONES,
    ChallengeState,
    Observation,
    SessionState,
    _ensure_active,
    _square_target_progress,
    utc_now,
)
from backend.app.screening import decode_video_frames, detect_live_faces
from backend.app.report_pdf import _challenge_verdict
from backend.app.voice_screening import decode_audio, normalize_phrase


class ScreeningLogicTests(unittest.TestCase):
    def test_report_pdf_verdict_is_challenge_only_and_has_three_outcomes(self) -> None:
        passed, pass_detail, _, _ = _challenge_verdict({"decision": "review"})
        failed, fail_detail, _, _ = _challenge_verdict({"decision": "challenge_failed"})
        inconclusive, inconclusive_detail, _, _ = _challenge_verdict({"decision": "inconclusive"})

        self.assertIn("PASS", passed)
        self.assertIn("not identity verification", pass_detail)
        self.assertIn("FAIL", failed)
        self.assertIn("not an identity or fraud determination", fail_detail)
        self.assertIn("INCONCLUSIVE", inconclusive)
        self.assertIn("not a pass", inconclusive_detail)

    def test_report_pdf_download_requires_a_completed_report(self) -> None:
        now = utc_now()
        state = SessionState(
            id=uuid4(),
            pair_token="long-enough-pair-token-value",
            created_at=now,
            expires_at=now + timedelta(minutes=1),
        )
        client = TestClient(main_module.app)

        with patch.dict(main_module._sessions, {state.id: state}):
            response = client.get(f"/api/sessions/{state.id}/report.pdf")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "No verification report is available yet")

    def test_report_pdf_download_returns_an_attachment_for_completed_report(self) -> None:
        now = utc_now()
        state = SessionState(
            id=uuid4(),
            pair_token="long-enough-pair-token-value",
            created_at=now,
            expires_at=now + timedelta(minutes=1),
            result={
                "decision": "review",
                "generated_at": now.isoformat(),
                "checks": {
                    "face_deepfake_analysis": {
                        "status": "review",
                        "detail": "Research screening only.",
                        "evidence": {
                            "median_fake_score": 0.7312,
                            "score_range": 0.24,
                            "frames": [
                                {
                                    "elapsed_ms": 1500,
                                    "fake_score": 0.82,
                                    "real_score": 0.18,
                                }
                            ],
                        },
                    },
                    "video_temporal_consistency": {
                        "status": "review",
                        "detail": "Basic consistency signal only.",
                        "evidence": {
                            "consistency": "review",
                            "frames_sampled": 10,
                            "frames_with_faces": 8,
                        },
                    },
                    "audio_spoof_detection": {
                        "status": "unavailable",
                        "detail": "Not submitted.",
                    },
                    "random_phrase_verification": {
                        "status": "unavailable",
                        "detail": "Not submitted.",
                    },
                    "audio_capture_quality": {
                        "status": "unavailable",
                        "detail": "Not submitted.",
                    },
                },
                "limitations": ["Research signals require human review."],
                "audit": {"system": "in-memory simulator", "sequence": 1},
            },
        )
        client = TestClient(main_module.app)

        with patch.dict(main_module._sessions, {state.id: state}):
            response = client.get(f"/api/sessions/{state.id}/report.pdf")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/pdf")
        self.assertIn(f'attachment; filename="frame-verification-{state.id}.pdf"', response.headers["content-disposition"])
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertTrue(response.content.startswith(b"%PDF-"))
        self.assertTrue(response.content.rstrip().endswith(b"%%EOF"))
        self.assertIn(b"PASS - REQUIRED CHALLENGE COMPLETED", response.content)

    def test_phone_page_uses_shared_brand_and_focused_qr_layout(self) -> None:
        now = utc_now()
        state = SessionState(
            id=uuid4(),
            pair_token="long-enough-pair-token-value",
            created_at=now,
            expires_at=now + timedelta(minutes=1),
        )
        client = TestClient(main_module.app)

        with patch.dict(main_module._sessions, {state.id: state}):
            response = client.get(f"/phone/{state.id}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertIn("FRAME / CHECK", response.text)
        self.assertIn("PHONE COMPANION", response.text)
        self.assertIn("Scan this live QR", response.text)
        self.assertIn("Hold this QR in box 1", response.text)
        self.assertIn("width:min(calc(100vw - 52px),520px)", response.text)
        self.assertIn("Each QR stays here until the laptop scans it", response.text)
        self.assertIn("body.challenge-active .qr{width:min(94vw,620px,calc(100dvh - 330px))", response.text)
        self.assertIn("challenge-instructions", response.text)
        self.assertNotIn('role="tablist"', response.text)

    def test_phone_qr_advances_only_after_the_current_code_is_scanned(self) -> None:
        now = utc_now()
        challenge = ChallengeState(
            id=uuid4(),
            created_at=now,
            expires_at=now + timedelta(minutes=1),
            codes=[f"code-{index}" for index in range(6)],
            audio_phrase="amber copper garden",
        )
        state = SessionState(
            id=uuid4(),
            pair_token="long-enough-pair-token-value",
            created_at=now,
            expires_at=now + timedelta(minutes=1),
            phone_verified=True,
            challenge=challenge,
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        success, encoded = main_module.cv2.imencode(".jpg", image)
        self.assertTrue(success)
        points = np.array([[[7, 17], [17, 17], [17, 27], [7, 27]]], dtype=np.float32)
        code_zero = "F26|0|code-0"
        code_one = "F26|1|code-1"
        client = TestClient(main_module.app)

        with (
            patch.dict(main_module._sessions, {state.id: state}),
            patch.object(main_module, "QR_PATH_SAMPLE_INTERVAL_MS", 0),
            patch.object(main_module, "detect_live_faces", return_value=[{"x": 0.4, "y": 0.1, "width": 0.2, "height": 0.3}]),
            patch.object(
                main_module.cv2.QRCodeDetector,
                "detectAndDecode",
                side_effect=[
                    (code_zero, points, None),
                    (code_zero, points, None),
                    (code_one, points, None),
                ],
            ),
        ):
            results = [
                client.post(
                    f"/api/sessions/{state.id}/challenge/{challenge.id}/scan?elapsed_ms={index * 250}",
                    content=encoded.tobytes(),
                    headers={"Content-Type": "image/jpeg"},
                )
                for index in range(3)
            ]

        self.assertEqual([response.status_code for response in results], [200, 200, 200])
        self.assertEqual(results[0].json()["payload"], code_zero)
        self.assertIsNone(results[1].json()["payload"])
        self.assertEqual(results[2].json()["payload"], code_one)
        self.assertEqual(challenge.current_qr_step, 2)
        self.assertEqual(results[0].json()["face_count"], 1)

    def test_qr_and_path_do_not_advance_when_no_face_is_detected(self) -> None:
        now = utc_now()
        challenge = ChallengeState(
            id=uuid4(),
            created_at=now,
            expires_at=now + timedelta(minutes=1),
            codes=["code-0", "code-1"],
            audio_phrase="amber copper garden",
        )
        state = SessionState(
            id=uuid4(),
            pair_token="long-enough-pair-token-value",
            created_at=now,
            expires_at=now + timedelta(minutes=1),
            phone_verified=True,
            challenge=challenge,
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        success, encoded = main_module.cv2.imencode(".jpg", image)
        self.assertTrue(success)
        points = np.array([[[7, 17], [17, 17], [17, 27], [7, 27]]], dtype=np.float32)
        client = TestClient(main_module.app)

        with (
            patch.dict(main_module._sessions, {state.id: state}),
            patch.object(main_module, "detect_live_faces", return_value=[]),
            patch.object(
                main_module.cv2.QRCodeDetector,
                "detectAndDecode",
                return_value=("F26|0|code-0", points, None),
            ),
        ):
            response = client.post(
                f"/api/sessions/{state.id}/challenge/{challenge.id}/scan?elapsed_ms=0",
                content=encoded.tobytes(),
                headers={"Content-Type": "image/jpeg"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["payload"])
        self.assertEqual(response.json()["face_count"], 0)
        self.assertEqual(challenge.current_qr_step, 0)
        self.assertEqual(challenge.square_zone_progress, 0)

    def test_challenge_qr_payload_is_compact_and_challenge_scoped(self) -> None:
        now = utc_now()
        challenge = ChallengeState(
            id=uuid4(),
            created_at=now,
            expires_at=now + timedelta(minutes=1),
            codes=["short-random-code"],
            audio_phrase="amber copper garden",
        )
        state = SessionState(
            id=uuid4(),
            pair_token="long-enough-pair-token-value",
            created_at=now,
            expires_at=now + timedelta(minutes=1),
            paired=True,
            phone_verified=True,
            challenge=challenge,
        )
        payloads: list[str] = []
        client = TestClient(main_module.app)

        with (
            patch.dict(main_module._sessions, {state.id: state}),
            patch.object(main_module, "_qr_png", side_effect=lambda payload: payloads.append(payload) or b"png"),
        ):
            response = client.get(
                f"/api/sessions/{state.id}/challenge/{challenge.id}/qr/0",
                headers={"X-Pair-Token": state.pair_token},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payloads, ["F26|0|short-random-code"])
        self.assertLess(len(payloads[0]), 24)
        self.assertEqual(main_module._challenge_code_step(challenge, payloads[0]), 0)
        self.assertIsNone(main_module._challenge_code_step(challenge, f"FUSION26|{state.id}|{challenge.id}|0|short-random-code"))
        compact_qr = main_module.cv2.imdecode(
            np.frombuffer(main_module._qr_png(payloads[0]), dtype=np.uint8),
            main_module.cv2.IMREAD_COLOR,
        )
        verbose_qr = main_module.cv2.imdecode(
            np.frombuffer(
                main_module._qr_png(f"FUSION26|{state.id}|{challenge.id}|0|short-random-code"),
                dtype=np.uint8,
            ),
            main_module.cv2.IMREAD_COLOR,
        )
        self.assertLess(compact_qr.shape[1], verbose_qr.shape[1])
        self.assertEqual(main_module.cv2.QRCodeDetector().detectAndDecode(compact_qr)[0], payloads[0])

    def test_qr_decode_uses_enhanced_grayscale_copy_after_normal_decode_fails(self) -> None:
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        points = np.array([[[10, 10], [30, 10], [30, 30], [10, 30]]], dtype=np.float32)
        with patch.object(
            main_module.cv2.QRCodeDetector,
            "detectAndDecode",
            side_effect=[
                ("", None, None),
                ("", None, None),
                ("F26|0|random", points, None),
            ],
        ) as detect:
            payload, decoded_points = main_module._decode_qr_payload(frame)

        self.assertEqual(payload, "F26|0|random")
        self.assertIs(decoded_points, points)
        self.assertEqual(detect.call_count, 3)
        self.assertEqual(detect.call_args_list[0].args[0].shape, (64, 96, 3))
        self.assertEqual(detect.call_args_list[1].args[0].shape, (64, 96))

    def test_live_camera_preview_is_mirrored_without_mirroring_scan_source(self) -> None:
        html = (main_module.ROOT / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn("#preview{position:absolute;inset:0;width:100%;height:100%;margin:0;transform:scaleX(-1)}", html)
        self.assertIn(".target-zone[data-zone=\"1\"]{left:88%;top:22%}", html)
        self.assertIn(".target-zone[data-zone=\"2\"]{left:88%;top:50%}", html)
        self.assertIn(".target-zone[data-zone=\"3\"]{left:88%;top:78%}", html)
        self.assertIn("background:rgba(21,92,74,.22)", html)
        self.assertIn("Camera preview stays natural; only the scan copy is contrast-enhanced.", html)
        self.assertIn("offsetX + (1 - result.box_x - result.box_width) * contentWidth", html)
        self.assertIn("context.drawImage($('preview'), 0, 0, canvas.width, canvas.height)", html)
        self.assertIn("No face detected. Keep your face visible in the center", html)

    def test_processing_states_show_accessible_progress_indicators(self) -> None:
        html = (main_module.ROOT / "static" / "index.html").read_text(encoding="utf-8")
        capture_flow = html.split("$('capture').addEventListener", maxsplit=1)[1].split(
            "$('run-voice').addEventListener",
            maxsplit=1,
        )[0]
        self.assertIn('id="video-progress"', html)
        self.assertIn('id="voice-progress"', html)
        self.assertIn('aria-label="Demo video screening progress"', html)
        self.assertIn('aria-label="Voice screening progress"', html)
        self.assertIn("/video/demo/start", capture_flow)
        self.assertIn("/video/demo/complete", capture_flow)
        self.assertIn("simulation.duration_seconds * 1000", capture_flow)
        self.assertIn("Video screening progress · ${percent}%", capture_flow)
        self.assertNotIn("Simulating AI screening", capture_flow)
        self.assertNotIn("/api/models/warmup", capture_flow)
        self.assertNotIn("/challenge/${challenge.id}/video`", capture_flow)
        self.assertNotIn("uploadVideoWithProgress", html)
        self.assertIn('class="work-progress-fill"', html)
        self.assertIn("Analyzing consented audio", html)
        self.assertIn("Generating the evidence report", html)
        self.assertIn('id="recorded-video-preview"', html)
        self.assertIn("URL.createObjectURL(recordedVideoBlob)", html)
        self.assertIn("recorded video stays in this tab", html)
        self.assertIn('className = \'voice-phrase-word\'', html)

    def test_voice_model_warmup_does_not_load_video_classifier(self) -> None:
        with (
            patch.object(screening_module, "_load_video_model") as load_video_model,
            patch.object(main_module, "warm_audio_model", return_value={"status": "ready"}),
            patch.object(main_module, "warm_transcriber", return_value={"status": "ready"}),
        ):
            video_status, audio_status = main_module._warm_requested_models(audio_consent=True)

        load_video_model.assert_not_called()
        self.assertEqual(audio_status["status"], "ready")
        self.assertIn("status", video_status)

    def test_video_demo_simulation_requires_consent_and_first_qr(self) -> None:
        now = utc_now()
        challenge = ChallengeState(
            id=uuid4(),
            created_at=now,
            expires_at=now + timedelta(minutes=1),
            codes=[],
            audio_phrase="amber copper garden",
        )
        state = SessionState(
            id=uuid4(),
            pair_token="long-enough-pair-token-value",
            created_at=now,
            expires_at=now + timedelta(minutes=1),
            phone_verified=True,
            challenge=challenge,
        )
        client = TestClient(main_module.app)
        path = f"/api/sessions/{state.id}/challenge/{challenge.id}/video/demo/start"

        with patch.dict(main_module._sessions, {state.id: state}):
            no_consent = client.post(path)
            no_qr = client.post(path, headers={"X-Video-Consent": "true"})

        self.assertEqual(no_consent.status_code, 403)
        self.assertEqual(no_qr.status_code, 409)
        self.assertIn("first live QR signal", no_qr.json()["detail"])
        self.assertFalse(challenge.video_processing)

    def test_video_demo_simulation_enforces_delay_and_report_marks_video_unavailable(self) -> None:
        now = utc_now()
        challenge = ChallengeState(
            id=uuid4(),
            created_at=now,
            expires_at=now + timedelta(minutes=1),
            codes=[f"qr-{index}" for index in range(6)],
            audio_phrase="amber copper garden",
            square_zone_progress=6,
        )
        state = SessionState(
            id=uuid4(),
            pair_token="long-enough-pair-token-value",
            created_at=now,
            expires_at=now + timedelta(minutes=1),
            phone_verified=True,
            challenge=challenge,
            server_observations=[
                Observation(payload=f"F26|{index}|qr-{index}", elapsed_ms=index * 1_000, x=0.5, y=0.5)
                for index in range(6)
            ],
        )
        client = TestClient(main_module.app)
        start_path = f"/api/sessions/{state.id}/challenge/{challenge.id}/video/demo/start"
        complete_path = f"/api/sessions/{state.id}/challenge/{challenge.id}/video/demo/complete"
        headers = {"X-Video-Consent": "true"}

        with patch.dict(main_module._sessions, {state.id: state}):
            started = client.post(start_path, headers=headers)
            too_soon = client.post(complete_path, headers=headers)
            challenge.video_simulation_started_at = utc_now() - timedelta(seconds=16)
            completed = client.post(complete_path, headers=headers)
            report = client.post(
                f"/api/sessions/{state.id}/evidence",
                json={"challenge_id": str(challenge.id)},
            )

        self.assertEqual(started.status_code, 200)
        self.assertEqual(started.json()["status"], "processing")
        self.assertEqual(started.json()["duration_seconds"], 15)
        self.assertEqual(too_soon.status_code, 409)
        self.assertIn("Wait at least 15 seconds", too_soon.json()["detail"])
        self.assertEqual(completed.status_code, 200)
        self.assertEqual(completed.json()["status"], "complete")
        self.assertTrue(challenge.video_simulated)
        self.assertTrue(challenge.video_submitted)
        self.assertFalse(challenge.video_processing)
        self.assertEqual(report.status_code, 200)
        result = report.json()
        self.assertEqual(result["decision"], "inconclusive")
        self.assertEqual(result["checks"]["face_deepfake_analysis"]["status"], "unavailable")
        self.assertEqual(result["checks"]["video_temporal_consistency"]["status"], "unavailable")
        self.assertEqual(result["checks"]["video_screening_simulation"]["status"], "unavailable")
        self.assertEqual(result["checks"]["capture_quality"]["status"], "unavailable")
        self.assertIn("No classifier ran", result["checks"]["video_screening_simulation"]["detail"])

    def test_admin_page_exposes_documented_per_step_qr_override(self) -> None:
        html = (main_module.ROOT / "static" / "admin.html").read_text(encoding="utf-8")
        self.assertIn("/api/admin/sessions/${session.id}/skip-qr-step", html)
        self.assertIn("The paired phone will show the next code", html)
        self.assertIn("makes the report inconclusive", html)
        self.assertIn("session.current_qr_step ?? 0", html)

    def test_live_face_detection_downscales_and_returns_normalized_boxes(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        box = SimpleNamespace(origin_x=160, origin_y=90, width=160, height=180)

        def detect(image) -> SimpleNamespace:
            self.assertEqual(image.numpy_view().shape, (360, 640, 3))
            return SimpleNamespace(
                detections=[SimpleNamespace(bounding_box=box)]
            )

        detector = SimpleNamespace(detect=detect)
        with patch("backend.app.screening._get_face_detector", return_value=detector):
            faces = detect_live_faces(frame)

        self.assertEqual(faces, [{"x": 0.25, "y": 0.25, "width": 0.25, "height": 0.5}])

    def test_video_classifier_uses_pinned_modern_model_and_context_crop(self) -> None:
        self.assertEqual(screening_module.VIDEO_MODEL_ID, "prithivMLmods/Deep-Fake-Detector-Model")
        self.assertEqual(screening_module.VIDEO_MODEL_REVISION, "c5cb24c6a159dd2b57ca15c6a1065bd0ce8fa380")
        self.assertEqual(screening_module.VIDEO_AI_THRESHOLD, 0.38)

        class FakeProcessor:
            image_size = None

            def __call__(self, images, **kwargs):
                self.image_size = images.size
                return {"pixel_values": torch.zeros((1, 3, 2, 2))}

        class FakeClassifier:
            config = SimpleNamespace(id2label={0: "real", 1: "fake"})

            def parameters(self):
                return iter([torch.nn.Parameter(torch.zeros(1))])

            def __call__(self, **kwargs):
                return SimpleNamespace(logits=torch.tensor([[0.0, 2.0]]))

        detection = SimpleNamespace(
            bounding_box=SimpleNamespace(origin_x=60, origin_y=50, width=40, height=60),
            categories=[SimpleNamespace(score=0.95)],
        )
        detector = SimpleNamespace(
            detect=lambda image: SimpleNamespace(detections=[detection])
        )
        processor = FakeProcessor()
        classifier = FakeClassifier()

        with (
            patch.object(screening_module, "_load_video_model"),
            patch.object(screening_module, "_get_face_detector", return_value=detector),
            patch.object(screening_module, "_processor", processor),
            patch.object(screening_module, "_classifier", classifier),
        ):
            result = screening_module.analyze_video_frame(np.zeros((160, 160, 3), dtype=np.uint8))

        self.assertEqual(result["status"], "analyzed")
        self.assertEqual(result["faces"][0]["model_revision"], screening_module.VIDEO_MODEL_REVISION)
        self.assertTrue(result["faces"][0]["subject_face"])
        self.assertGreater(result["faces"][0]["fake_score"], 0.38)
        self.assertEqual(result["frame_classification"], "AI-like")
        self.assertEqual(processor.image_size[0], processor.image_size[1])
        self.assertGreater(processor.image_size[0], 60)

    def test_modern_video_consensus_requires_multiple_frames_and_uses_prominent_face_score(self) -> None:
        def frame(score: float, *, scores: list[float] | None = None) -> dict:
            face_scores = scores or [score]
            return {
                "status": "analyzed",
                "elapsed_ms": 1000,
                "frame_score": score,
                "faces": [
                    {
                        "fake_score": face_score,
                        "real_score": 1 - face_score,
                        "subject_face": index == 0,
                        "bbox": {"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.3},
                    }
                    for index, face_score in enumerate(face_scores)
                ],
            }

        ai_summary = screening_module.summarize_video_frames([
            frame(0.45, scores=[0.45, 0.99]),
            frame(0.50),
            frame(0.42),
        ])
        real_summary = screening_module.summarize_video_frames([
            frame(0.10),
            frame(0.20),
            frame(0.18),
        ])
        single_frame_summary = screening_module.summarize_video_frames([frame(0.99)])

        self.assertEqual(ai_summary["model_verdict"], "AI_GENERATED")
        self.assertEqual(ai_summary["flagged_frames"], 3)
        self.assertEqual(ai_summary["aggregate_fake_score"], 0.4527)
        self.assertEqual(real_summary["model_verdict"], "REAL_VIDEO")
        self.assertEqual(single_frame_summary["model_verdict"], "INCONCLUSIVE")

    def test_qr_targets_stay_at_frame_edges_away_from_centered_face_area(self) -> None:
        self.assertEqual(len(QR_TARGET_ZONES), 6)
        self.assertTrue(all(x <= 0.15 or x >= 0.85 for x, _ in QR_TARGET_ZONES))

    def test_live_face_detection_requires_face_model_and_reports_failure(self) -> None:
        now = utc_now()
        challenge = ChallengeState(
            id=uuid4(),
            created_at=now,
            expires_at=now + timedelta(minutes=1),
            codes=["code-0"],
            audio_phrase="amber copper garden",
        )
        state = SessionState(
            id=uuid4(),
            pair_token="long-enough-pair-token-value",
            created_at=now,
            expires_at=now + timedelta(minutes=1),
            phone_verified=True,
            challenge=challenge,
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        success, encoded = main_module.cv2.imencode(".jpg", image)
        self.assertTrue(success)
        client = TestClient(main_module.app)

        with (
            patch.dict(main_module._sessions, {state.id: state}),
            patch.object(main_module, "detect_live_faces", side_effect=RuntimeError("detector unavailable")),
        ):
            response = client.post(
                f"/api/sessions/{state.id}/challenge/{challenge.id}/scan?elapsed_ms=0",
                content=encoded.tobytes(),
                headers={"Content-Type": "image/jpeg"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("Live face detection is unavailable", response.json()["detail"])

    def test_hidden_qa_control_is_disabled_by_default(self) -> None:
        with patch.dict(os.environ, {"ENABLE_QA_CONTROLS": "", "APP_ENV": "development"}):
            client = TestClient(main_module.app, client=("127.0.0.1", 50000))
            response = client.get("/hidden/control")

        self.assertEqual(response.status_code, 404)

    def test_hidden_qa_control_requires_local_development(self) -> None:
        with patch.dict(os.environ, {"ENABLE_QA_CONTROLS": "true", "APP_ENV": "development"}):
            client = TestClient(main_module.app, client=("127.0.0.1", 50000))
            response = client.get("/hidden/control")
            status = client.get("/api/qa/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(status.json(), {"enabled": True})

        with patch.dict(os.environ, {"ENABLE_QA_CONTROLS": "true", "APP_ENV": "production"}):
            production = TestClient(main_module.app, client=("127.0.0.1", 50000))
            response = production.get("/hidden/control")

        self.assertEqual(response.status_code, 404)

    def test_hidden_qa_control_rejects_forwarded_tunnel_requests(self) -> None:
        with patch.dict(os.environ, {"ENABLE_QA_CONTROLS": "true", "APP_ENV": "development"}):
            client = TestClient(main_module.app, client=("127.0.0.1", 50000))
            response = client.get("/hidden/control", headers={"CF-Connecting-IP": "203.0.113.5"})

        self.assertEqual(response.status_code, 404)

    def test_admin_page_and_api_require_explicit_enablement(self) -> None:
        client = TestClient(main_module.app)
        with patch.dict(os.environ, {"APP_ENV": "development", "ENABLE_ADMIN_CONTROLS": ""}):
            page = client.get("/admin")
            api = client.get("/api/admin/sessions")

        self.assertEqual(page.status_code, 404)
        self.assertEqual(api.status_code, 404)

    def test_updated_frontend_theme_is_served_as_css(self) -> None:
        response = TestClient(main_module.app).get("/uiupd-theme.css")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/css; charset=utf-8")
        self.assertIn("--font-serif", response.text)

    def test_admin_page_and_session_list_are_available_through_public_tunnel_when_enabled(self) -> None:
        client = TestClient(main_module.app, client=("127.0.0.1", 50000))
        with patch.dict(os.environ, {"APP_ENV": "development", "ENABLE_ADMIN_CONTROLS": "true"}):
            page = client.get("/admin")
            api = client.get("/api/admin/sessions")
            home = client.get("/")
            forwarded_page = client.get("/admin", headers={"CF-Connecting-IP": "203.0.113.5"})
            forwarded_api = client.get("/api/admin/sessions", headers={"X-Forwarded-For": "203.0.113.5"})
            remote = TestClient(main_module.app, client=("203.0.113.5", 50000))
            remote_page = remote.get("/admin")
            remote_api = remote.get("/api/admin/sessions")

        self.assertEqual(page.status_code, 200)
        self.assertEqual(page.headers["cache-control"], "no-store")
        self.assertIn("passwordless controls", page.text)
        self.assertIn("Cloudflare", page.text)
        self.assertNotIn('type="password"', page.text.lower())
        self.assertNotIn("/admin", home.text)
        self.assertEqual(api.status_code, 200)
        self.assertEqual(api.json(), {"sessions": []})
        self.assertEqual(forwarded_page.status_code, 200)
        self.assertEqual(forwarded_api.status_code, 200)
        self.assertEqual(remote_page.status_code, 200)
        self.assertEqual(remote_api.status_code, 200)

    def test_admin_video_override_is_audited_and_reported_as_inconclusive(self) -> None:
        now = utc_now()
        challenge = ChallengeState(
            id=uuid4(),
            created_at=now,
            expires_at=now + timedelta(minutes=1),
            codes=[f"code-{index}" for index in range(6)],
            audio_phrase="amber copper garden",
            video_processing=True,
        )
        state = SessionState(
            id=uuid4(),
            pair_token="long-enough-pair-token-value",
            created_at=now,
            expires_at=now + timedelta(minutes=2),
            paired=True,
            phone_verified=True,
            challenge=challenge,
        )
        client = TestClient(main_module.app, client=("127.0.0.1", 50000))

        with (
            patch.dict(
                os.environ,
                {"APP_ENV": "development", "ENABLE_ADMIN_CONTROLS": "true"},
            ),
            patch.dict(main_module._sessions, {state.id: state}),
        ):
            override = client.post(
                f"/api/admin/sessions/{state.id}/skip-video",
                json={"challenge_id": str(challenge.id), "reason": "Screening timed out during demo"},
            )
            report = client.post(
                f"/api/sessions/{state.id}/evidence",
                json={"challenge_id": str(challenge.id)},
            )

        self.assertEqual(override.status_code, 200)
        self.assertTrue(override.json()["video_manually_skipped"])
        self.assertTrue(challenge.video_submitted)
        self.assertFalse(challenge.video_processing)
        self.assertEqual(report.status_code, 200)
        result = report.json()
        self.assertEqual(result["decision"], "inconclusive")
        self.assertEqual(result["checks"]["face_deepfake_analysis"]["status"], "unavailable")
        self.assertEqual(result["checks"]["capture_quality"]["status"], "unavailable")
        self.assertEqual(result["checks"]["audio_capture_quality"]["status"], "unavailable")
        self.assertIn("Screening timed out during demo", result["checks"]["video_screening_override"]["detail"])
        self.assertTrue(any("manually skipped" in item for item in result["limitations"]))

    def test_admin_video_override_requires_processing_and_is_disabled_in_production(self) -> None:
        now = utc_now()
        challenge = ChallengeState(
            id=uuid4(),
            created_at=now,
            expires_at=now + timedelta(minutes=1),
            codes=[f"code-{index}" for index in range(6)],
            audio_phrase="amber copper garden",
        )
        state = SessionState(
            id=uuid4(),
            pair_token="long-enough-pair-token-value",
            created_at=now,
            expires_at=now + timedelta(minutes=2),
            paired=True,
            phone_verified=True,
            challenge=challenge,
        )
        client = TestClient(main_module.app, client=("127.0.0.1", 50000))
        with (
            patch.dict(
                os.environ,
                {"APP_ENV": "development", "ENABLE_ADMIN_CONTROLS": "true"},
            ),
            patch.dict(main_module._sessions, {state.id: state}),
        ):
            not_processing = client.post(
                f"/api/admin/sessions/{state.id}/skip-video",
                json={"challenge_id": str(challenge.id), "reason": "No processing is active"},
            )

        self.assertEqual(not_processing.status_code, 409)
        self.assertFalse(challenge.video_submitted)

        with patch.dict(
            os.environ,
            {"APP_ENV": "production", "ENABLE_ADMIN_CONTROLS": "true"},
        ):
            production_page = client.get("/admin")
            production_api = client.get("/api/admin/sessions")
        self.assertEqual(production_page.status_code, 404)
        self.assertEqual(production_api.status_code, 404)

    def test_admin_can_skip_qr_steps_one_at_a_time_and_report_is_inconclusive(self) -> None:
        now = utc_now()
        challenge = ChallengeState(
            id=uuid4(),
            created_at=now,
            expires_at=now + timedelta(minutes=1),
            codes=[f"code-{index}" for index in range(6)],
            audio_phrase="amber copper garden",
        )
        state = SessionState(
            id=uuid4(),
            pair_token="long-enough-pair-token-value",
            created_at=now,
            expires_at=now + timedelta(minutes=2),
            paired=True,
            phone_verified=True,
            challenge=challenge,
        )
        client = TestClient(main_module.app, client=("127.0.0.1", 50000))

        with (
            patch.dict(
                os.environ,
                {"APP_ENV": "development", "ENABLE_ADMIN_CONTROLS": "true"},
            ),
            patch.dict(main_module._sessions, {state.id: state}),
        ):
            first_skip = client.post(
                f"/api/admin/sessions/{state.id}/skip-qr-step",
                json={"challenge_id": str(challenge.id), "reason": "Phone QR failed to refresh"},
            )
            participant_challenge = client.get(
                f"/api/sessions/{state.id}/challenge",
                headers={"X-Pair-Token": state.pair_token},
            )
            second_skip = client.post(
                f"/api/admin/sessions/{state.id}/skip-qr-step",
                json={"challenge_id": str(challenge.id), "reason": "Second code is stuck"},
            )
            challenge.video_submitted = True
            report = client.post(
                f"/api/sessions/{state.id}/evidence",
                json={"challenge_id": str(challenge.id)},
            )

        self.assertEqual(first_skip.status_code, 200)
        self.assertEqual(first_skip.json()["current_qr_step"], 1)
        self.assertEqual(first_skip.json()["qr_steps_skipped"], 1)
        self.assertTrue(first_skip.json()["can_skip_qr_step"])
        self.assertEqual(participant_challenge.status_code, 200)
        self.assertEqual(participant_challenge.json()["current_qr_step"], 1)
        self.assertEqual(second_skip.status_code, 200)
        self.assertEqual(second_skip.json()["current_qr_step"], 2)
        self.assertEqual(second_skip.json()["qr_steps_skipped"], 2)
        self.assertEqual(report.status_code, 200)
        result = report.json()
        self.assertEqual(result["decision"], "inconclusive")
        self.assertEqual(result["checks"]["randomized_qr_sequence"]["status"], "unavailable")
        self.assertIn("Phone QR failed to refresh", result["checks"]["randomized_qr_sequence"]["detail"])
        self.assertEqual(result["checks"]["qr_step_override"]["status"], "unavailable")
        self.assertTrue(any("QR steps were manually skipped" in item for item in result["limitations"]))

    def test_admin_qr_skip_rejects_unverified_or_inactive_challenges(self) -> None:
        now = utc_now()
        challenge = ChallengeState(
            id=uuid4(),
            created_at=now,
            expires_at=now + timedelta(minutes=1),
            codes=[f"code-{index}" for index in range(6)],
            audio_phrase="amber copper garden",
            video_processing=True,
        )
        state = SessionState(
            id=uuid4(),
            pair_token="long-enough-pair-token-value",
            created_at=now,
            expires_at=now + timedelta(minutes=2),
            phone_verified=False,
            challenge=challenge,
        )
        client = TestClient(main_module.app, client=("127.0.0.1", 50000))

        with (
            patch.dict(
                os.environ,
                {"APP_ENV": "development", "ENABLE_ADMIN_CONTROLS": "true"},
            ),
            patch.dict(main_module._sessions, {state.id: state}),
        ):
            unverified = client.post(
                f"/api/admin/sessions/{state.id}/skip-qr-step",
                json={"challenge_id": str(challenge.id), "reason": "Stuck QR code"},
            )
            state.phone_verified = True
            processing = client.post(
                f"/api/admin/sessions/{state.id}/skip-qr-step",
                json={"challenge_id": str(challenge.id), "reason": "Stuck QR code"},
            )

        self.assertEqual(unverified.status_code, 409)
        self.assertEqual(processing.status_code, 409)
        self.assertEqual(challenge.current_qr_step, 0)
        self.assertEqual(challenge.qr_step_overrides, [])

    def test_square_target_path_passes_in_order(self) -> None:
        observations = [
            Observation(
                payload="signal",
                elapsed_ms=index * 250,
                x=point[0],
                y=point[1],
            )
            for index, point in enumerate(QR_TARGET_ZONES)
        ]

        progress, detail = _square_target_progress(observations)

        self.assertEqual(progress, len(QR_TARGET_ZONES))
        self.assertIn("all six numbered square targets", detail)

    def test_square_target_path_rejects_skipped_box(self) -> None:
        observations = [
            Observation(payload="signal", elapsed_ms=index * 250, x=point[0], y=point[1])
            for index, point in enumerate((QR_TARGET_ZONES[1], QR_TARGET_ZONES[0], *QR_TARGET_ZONES[2:]))
        ]

        progress, detail = _square_target_progress(observations)

        self.assertEqual(progress, 1)
        self.assertIn("1 of 6 square targets", detail)

    def test_audio_phrase_normalization_ignores_punctuation(self) -> None:
        self.assertEqual(normalize_phrase("Amber, COPPER! garden."), "amber copper garden")

    def test_audio_phrase_matching_allows_small_transcription_variation(self) -> None:
        near_match = voice_module.compare_phrase("amber copper garden", "Amber copper gardens.")
        mismatch = voice_module.compare_phrase("amber copper garden", "amber garden")

        self.assertEqual(near_match["status"], "passed")
        self.assertFalse(near_match["exact_match"])
        self.assertGreaterEqual(near_match["similarity"], 0.90)
        self.assertEqual(mismatch["status"], "failed")

    def test_transcriber_biases_beam_search_toward_fresh_phrase(self) -> None:
        class FakeWhisper:
            call_arguments = None

            def transcribe(self, waveform, **kwargs):
                self.call_arguments = kwargs
                return [SimpleNamespace(text=" amber copper garden ")], None

        model = FakeWhisper()
        with (
            patch.object(voice_module, "_whisper", model),
            patch.object(voice_module, "_whisper_error", None),
        ):
            transcript = voice_module.transcribe_phrase(
                np.ones(16_000, dtype=np.float32),
                "amber copper garden",
            )

        self.assertEqual(transcript, "amber copper garden")
        self.assertEqual(model.call_arguments["beam_size"], 5)
        self.assertEqual(model.call_arguments["hotwords"], "amber copper garden")

    def test_audio_quality_reviews_short_or_clipped_samples_and_rejects_silence(self) -> None:
        short_signal = np.full(16_000, 0.1, dtype=np.float32)
        clipped_signal = np.full(48_000, 1.0, dtype=np.float32)

        self.assertEqual(voice_module._audio_quality(short_signal)["status"], "review")
        self.assertEqual(voice_module._audio_quality(clipped_signal)["status"], "review")
        self.assertEqual(
            voice_module._audio_quality(np.zeros(16_000, dtype=np.float32))["status"],
            "failed",
        )

    def test_silent_audio_is_not_sent_to_phrase_or_voice_models(self) -> None:
        with (
            patch.object(voice_module, "decode_audio", return_value=np.zeros(16_000, dtype=np.float32)),
            patch.object(voice_module, "transcribe_phrase", side_effect=AssertionError("must not transcribe silence")),
            patch.object(voice_module, "classify_audio", side_effect=AssertionError("must not classify silence")),
        ):
            result = voice_module.analyze_audio(b"ignored in the decoder test", "amber copper garden")

        self.assertEqual(result["quality"]["status"], "failed")
        self.assertEqual(result["phrase_status"], "unavailable")
        self.assertEqual(result["spoof_status"], "unavailable")

    def test_updated_audio_classifier_uses_real_fake_model_labels(self) -> None:
        class FakeProcessor:
            def __call__(self, waveform, **kwargs):
                return {"input_values": torch.zeros((1, len(waveform)), dtype=torch.float32)}

        class FakeModel:
            config = SimpleNamespace(id2label={0: "real", 1: "fake"})

            def parameters(self):
                return iter([torch.nn.Parameter(torch.zeros(1))])

            def __call__(self, **kwargs):
                return SimpleNamespace(logits=torch.tensor([[0.0, 2.0]]))

        with (
            patch.object(voice_module, "_load_audio_model"),
            patch.object(voice_module, "_audio_processor", FakeProcessor()),
            patch.object(voice_module, "_audio_model", FakeModel()),
        ):
            result = voice_module.classify_audio(np.ones(16_000, dtype=np.float32))

        self.assertEqual(result["status"], "review")
        self.assertEqual(result["model"], "garystafford/wav2vec2-deepfake-voice-detector")
        self.assertGreater(result["ai_voice_score"], result["human_voice_score"])
        self.assertIn("not proof", result["detail"])

    def test_audio_decoder_stops_after_ten_seconds(self) -> None:
        recording = io.BytesIO()
        with wave.open(recording, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16_000)
            audio.writeframes(b"\x00\x00" * (16_000 * 11))

        with self.assertRaisesRegex(ValueError, "between 0.3 and 10 seconds"):
            decode_audio(recording.getvalue())

    def test_audio_decoder_accepts_a_short_fresh_phrase_clip(self) -> None:
        recording = io.BytesIO()
        samples = (np.sin(np.arange(16_000) * (2 * np.pi * 440 / 16_000)) * 8_000).astype("<i2")
        with wave.open(recording, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16_000)
            audio.writeframes(samples.tobytes())

        waveform = decode_audio(recording.getvalue())

        self.assertEqual(len(waveform), 16_000)

    def test_audio_submission_returns_quality_and_allows_retry_after_silence(self) -> None:
        now = utc_now()
        challenge = ChallengeState(
            id=uuid4(),
            created_at=now,
            expires_at=now + timedelta(minutes=1),
            codes=[],
            audio_phrase="amber copper garden",
        )
        state = SessionState(
            id=uuid4(),
            pair_token="long-enough-pair-token-value",
            created_at=now,
            expires_at=now + timedelta(minutes=1),
            phone_verified=True,
            challenge=challenge,
        )
        failed_quality = {
            "duration_seconds": 1.0,
            "phrase_status": "unavailable",
            "phrase_detail": "Recording was silent.",
            "phrase_similarity": None,
            "spoof_status": "unavailable",
            "quality": {"status": "failed", "detail": "Recording was silent."},
            "spoof": {"status": "unavailable", "detail": "Recording was silent."},
        }
        phrase_mismatch = {
            **failed_quality,
            "phrase_status": "failed",
            "phrase_detail": "The transcription matched the fresh phrase with 77% similarity; 90% is required.",
            "phrase_similarity": 0.7742,
            "quality": {"status": "passed", "detail": "Audio level is suitable."},
            "spoof": {"status": "review", "detail": "Uncalibrated research score."},
        }
        retry_quality = {
            **phrase_mismatch,
            "phrase_status": "passed",
            "phrase_detail": "The fresh phrase was transcribed exactly.",
            "phrase_similarity": 1.0,
        }
        client = TestClient(main_module.app)

        with (
            patch.dict(main_module._sessions, {state.id: state}),
            patch.object(main_module, "analyze_audio", side_effect=[failed_quality, phrase_mismatch, retry_quality]),
        ):
            first = client.post(
                f"/api/sessions/{state.id}/challenge/{challenge.id}/audio",
                content=b"test audio bytes",
                headers={"Content-Type": "audio/wav", "X-Audio-Consent": "true"},
            )
            self.assertFalse(challenge.audio_submitted)
            phrase_failed = client.post(
                f"/api/sessions/{state.id}/challenge/{challenge.id}/audio",
                content=b"test audio bytes",
                headers={"Content-Type": "audio/wav", "X-Audio-Consent": "true"},
            )
            self.assertFalse(challenge.audio_submitted)
            retry = client.post(
                f"/api/sessions/{state.id}/challenge/{challenge.id}/audio",
                content=b"test audio bytes",
                headers={"Content-Type": "audio/wav", "X-Audio-Consent": "true"},
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["quality_check"]["status"], "failed")
        self.assertEqual(phrase_failed.status_code, 200)
        self.assertEqual(phrase_failed.json()["phrase_check"]["status"], "failed")
        self.assertEqual(retry.status_code, 200)
        self.assertEqual(retry.json()["quality_check"]["status"], "passed")
        self.assertEqual(retry.json()["phrase_check"]["status"], "passed")
        self.assertTrue(challenge.audio_submitted)

    def test_video_decoder_samples_recording_in_memory(self) -> None:
        recording = io.BytesIO()
        container = av.open(recording, mode="w", format="webm")
        stream = container.add_stream("libvpx", rate=2)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        for _ in range(5):
            frame = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()

        frames = decode_video_frames(recording.getvalue())

        self.assertGreaterEqual(len(frames), 2)
        self.assertEqual(frames[0][0], 0)
        self.assertEqual(frames[-1][0], 2_000)

    def test_report_requires_video_screening(self) -> None:
        now = utc_now()
        challenge = ChallengeState(
            id=uuid4(),
            created_at=now,
            expires_at=now + timedelta(minutes=1),
            codes=[],
            audio_phrase="amber copper garden",
        )
        state = SessionState(
            id=uuid4(),
            pair_token="long-enough-pair-token-value",
            created_at=now,
            expires_at=now + timedelta(minutes=1),
            phone_verified=True,
            challenge=challenge,
        )
        client = TestClient(main_module.app)

        with patch.dict(main_module._sessions, {state.id: state}):
            response = client.post(
                f"/api/sessions/{state.id}/evidence",
                json={"challenge_id": str(challenge.id)},
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("recorded QR video screening", response.json()["detail"])

    def test_video_submission_requires_first_qr_scan(self) -> None:
        now = utc_now()
        challenge = ChallengeState(
            id=uuid4(),
            created_at=now,
            expires_at=now + timedelta(minutes=1),
            codes=[],
            audio_phrase="amber copper garden",
        )
        state = SessionState(
            id=uuid4(),
            pair_token="long-enough-pair-token-value",
            created_at=now,
            expires_at=now + timedelta(minutes=1),
            phone_verified=True,
            challenge=challenge,
        )
        client = TestClient(main_module.app)

        with patch.dict(main_module._sessions, {state.id: state}):
            response = client.post(
                f"/api/sessions/{state.id}/challenge/{challenge.id}/video",
                content=b"not a video",
                headers={"Content-Type": "video/webm", "X-Video-Consent": "true"},
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("first live QR signal", response.json()["detail"])

    def test_audit_chain_links_entries_and_verifies(self) -> None:
        ledger = SimulatedAuditLedger()
        first = ledger.append("a" * 64, datetime(2026, 1, 1, tzinfo=timezone.utc))
        second = ledger.append("b" * 64, datetime(2026, 1, 2, tzinfo=timezone.utc))

        self.assertEqual(first["previous_hash"], "0" * 64)
        self.assertEqual(second["previous_hash"], first["record_hash"])
        self.assertTrue(ledger.verify())

    def test_audit_chain_detects_tampered_link(self) -> None:
        ledger = SimulatedAuditLedger()
        ledger.append("a" * 64, datetime(2026, 1, 1, tzinfo=timezone.utc))
        ledger.append("b" * 64, datetime(2026, 1, 2, tzinfo=timezone.utc))
        ledger._entries[1]["previous_hash"] = "c" * 64

        self.assertFalse(ledger.verify())

    def test_expired_session_discards_in_memory_evidence(self) -> None:
        now = utc_now()
        state = SessionState(
            id=uuid4(),
            pair_token="long-enough-pair-token-value",
            created_at=now - timedelta(minutes=6),
            expires_at=now - timedelta(seconds=1),
            video_frames=[{"status": "analyzed"}],
            server_observations=[Observation(payload="signal", elapsed_ms=0, x=0.5, y=0.5)],
            audio_result={"phrase_status": "passed"},
            result={"decision": "review"},
        )

        with self.assertRaises(HTTPException):
            _ensure_active(state)

        self.assertIsNone(state.challenge)
        self.assertEqual(state.video_frames, [])
        self.assertEqual(state.server_observations, [])
        self.assertIsNone(state.audio_result)
        self.assertIsNone(state.result)


if __name__ == "__main__":
    unittest.main()
