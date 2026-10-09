import io
import math
import unittest
import wave
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import uuid4

import av
import numpy as np
from fastapi import HTTPException
from fastapi.testclient import TestClient
from backend.app.audit_ledger import SimulatedAuditLedger
from backend.app import main as main_module
from backend.app.main import ChallengeState, Observation, SessionState, _ensure_active, _figure_eight_coherence, utc_now
from backend.app.screening import decode_video_frames
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
        self.assertIn("keep the screen bright until the laptop shows QR LOCK", response.text)
        self.assertIn("width:min(82vw,370px)", response.text)
        self.assertNotIn('role="tablist"', response.text)

    def test_figure_eight_path_passes(self) -> None:
        observations = [
            Observation(
                payload="signal",
                elapsed_ms=index * 250,
                x=0.5 + 0.25 * math.sin(index * 2 * math.pi / 23),
                y=0.5 + 0.2 * math.sin(2 * index * 2 * math.pi / 23),
            )
            for index in range(24)
        ]

        passed, detail = _figure_eight_coherence(observations)

        self.assertTrue(passed)
        self.assertIn("crossed its center", detail)

    def test_stationary_path_does_not_pass(self) -> None:
        observations = [
            Observation(payload="signal", elapsed_ms=index * 250, x=0.5, y=0.5)
            for index in range(20)
        ]

        passed, detail = _figure_eight_coherence(observations)

        self.assertFalse(passed)
        self.assertIn("Too few distinct", detail)

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
        self.assertGreaterEqual(frames[-1][0], 1_000)

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
