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
from fastapi import HTTPException
from fastapi.testclient import TestClient
from backend.app.audit_ledger import SimulatedAuditLedger
from backend.app import main as main_module
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
from backend.app.voice_screening import decode_audio, normalize_phrase


class ScreeningLogicTests(unittest.TestCase):
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

    def test_admin_page_and_session_list_are_passwordless_and_loopback_only(self) -> None:
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
        self.assertIn("local computer only", page.text)
        self.assertNotIn("password", page.text.lower())
        self.assertNotIn("/admin", home.text)
        self.assertEqual(api.status_code, 200)
        self.assertEqual(api.json(), {"sessions": []})
        self.assertEqual(forwarded_page.status_code, 404)
        self.assertEqual(forwarded_api.status_code, 404)
        self.assertEqual(remote_page.status_code, 404)
        self.assertEqual(remote_api.status_code, 404)

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

    def test_audio_decoder_stops_after_ten_seconds(self) -> None:
        recording = io.BytesIO()
        with wave.open(recording, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16_000)
            audio.writeframes(b"\x00\x00" * (16_000 * 11))

        with self.assertRaisesRegex(ValueError, "between 2 and 10 seconds"):
            decode_audio(recording.getvalue())

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
