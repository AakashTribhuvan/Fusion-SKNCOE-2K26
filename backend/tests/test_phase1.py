import io
from datetime import datetime
from unittest.mock import patch

import av
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app
from app.services.audio_quality_service import AudioQualityService
from app.services.challenge_service import ChallengeService
from app.services.phrase_match_service import PhraseMatchService

client = TestClient(app)


def _wav_bytes(duration_seconds=1.0, sample_rate=16000, amplitude=0.2):
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * 220 * t)
    buffer = io.BytesIO()
    sf.write(buffer, wave.astype(np.float32), sample_rate, format="WAV")
    return buffer.getvalue()


def _webm_opus_bytes(duration_seconds=1.0, sample_rate=48000):
    output = io.BytesIO()
    container = av.open(output, mode='w', format='webm')
    stream = container.add_stream('libopus', rate=sample_rate)
    stream.layout = 'mono'
    signal = (0.2 * np.sin(2 * np.pi * 240 * np.arange(sample_rate * duration_seconds) / sample_rate))
    samples = np.int16(signal * 32767).reshape(1, -1)
    frame = av.AudioFrame.from_ndarray(samples, format='s16', layout='mono')
    frame.sample_rate = sample_rate
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return output.getvalue()


def test_challenge_generation_and_expiration():
    challenge = ChallengeService.create(ttl_seconds=900)
    assert challenge.phrase
    assert len(challenge.phrase.split()) >= 3
    assert challenge.expires_at > challenge.created_at


def test_create_and_replace_challenge():
    response = client.post('/api/v1/sessions')
    assert response.status_code == 200, response.text
    payload = response.json()
    session_id = payload['session_id']
    old_phrase = payload['challenge_phrase']

    refreshed = client.post(f'/api/v1/sessions/{session_id}/challenge')
    assert refreshed.status_code == 200, refreshed.text
    refreshed_payload = refreshed.json()
    assert refreshed_payload['invalidated_previous'] is True
    assert refreshed_payload['challenge_phrase'] != old_phrase


def test_audio_upload_validation_rejects_empty_file():
    response = client.post('/api/v1/sessions')
    session_id = response.json()['session_id']
    upload = client.post(f'/api/v1/sessions/{session_id}/audio', files={'file': ('empty.wav', b'', 'audio/wav')})
    assert upload.status_code == 400


def test_audio_upload_rejects_unrecognized_media_bytes():
    session_id = client.post('/api/v1/sessions').json()['session_id']
    upload = client.post(
        f'/api/v1/sessions/{session_id}/audio',
        files={'file': ('recording.wav', b'not actually a WAV file', 'audio/wav')},
    )
    assert upload.status_code == 415
    assert 'valid PCM WAV' in upload.json()['detail']


def test_audio_upload_reports_decoded_duration():
    session_id = client.post('/api/v1/sessions').json()['session_id']
    upload = client.post(
        f'/api/v1/sessions/{session_id}/audio',
        files={'file': ('recording.wav', _wav_bytes(duration_seconds=1.0), 'audio/wav')},
    )
    assert upload.status_code == 200
    assert 0.99 <= upload.json()['duration_seconds'] <= 1.01


def test_browser_webm_opus_recording_is_decoded_from_media_content():
    session_id = client.post('/api/v1/sessions').json()['session_id']
    upload = client.post(
        f'/api/v1/sessions/{session_id}/audio',
        files={'file': ('recording.webm', _webm_opus_bytes(), 'audio/webm;codecs=opus')},
    )
    assert upload.status_code == 200, upload.text
    assert 0.9 <= upload.json()['duration_seconds'] <= 1.1


def test_audio_quality_service_handles_silence_and_pass():
    silent = AudioQualityService.analyze(np.zeros(1600), 16000)
    assert silent['status'] == 'fail'
    assert silent['silent'] is True

    tone = (0.5 * np.sin(2 * np.pi * 220 * np.linspace(0, 1, 16000, endpoint=False))).astype(np.float32)
    pass_result = AudioQualityService.analyze(tone, 16000)
    assert pass_result['status'] == 'pass'
    assert pass_result['speech_usable'] is True


def test_phrase_normalization_and_matching():
    comparison = PhraseMatchService.compare('Blue river seven', 'Blue river, seven!')
    assert comparison['exact_match'] is True
    assert comparison['phrase_verification_status'] == 'pass'

    wrong = PhraseMatchService.compare('Blue river seven', 'Blue forest five')
    assert wrong['exact_match'] is False
    assert wrong['phrase_verification_status'] == 'fail'

    missing_transcript = PhraseMatchService.compare('Blue river seven', None)
    assert missing_transcript['exact_match'] is None
    assert missing_transcript['phrase_verification_status'] == 'incomplete'


def test_model_status_route_and_unavailable_detection():
    health = client.get('/health')
    assert health.status_code == 200
    model_status = client.get('/api/v1/models/status')
    assert model_status.status_code == 200
    payload = model_status.json()
    assert payload['spoof_detector']['status'] in {'available', 'configured', 'disabled', 'error', 'unavailable'}


def test_verify_session_returns_structured_result():
    session = client.post('/api/v1/sessions').json()
    session_id = session['session_id']
    audio_bytes = _wav_bytes(duration_seconds=1.0)
    upload = client.post(
        f'/api/v1/sessions/{session_id}/audio',
        files={'file': ('sample.wav', audio_bytes, 'audio/wav')},
    )
    assert upload.status_code == 200, upload.text

    verify = client.post(f'/api/v1/sessions/{session_id}/verify')
    assert verify.status_code == 200, verify.text
    payload = verify.json()
    assert payload['session_id'] == session_id
    assert 'audio_quality' in payload
    assert 'speech_detection' in payload
    assert 'phrase_verification' in payload
    assert 'spoof_detection' in payload
    assert payload['spoof_detection']['status'] in {'completed', 'unavailable', 'error'}


def test_verify_skips_spoof_inference_when_audio_is_silent():
    session = client.post('/api/v1/sessions').json()
    session_id = session['session_id']
    audio_bytes = _wav_bytes(duration_seconds=1.0, amplitude=0.0)
    upload = client.post(
        f'/api/v1/sessions/{session_id}/audio',
        files={'file': ('silence.wav', audio_bytes, 'audio/wav')},
    )
    assert upload.status_code == 200, upload.text

    with patch('app.api.verification.SpoofDetectorService.run') as run_detector:
        verify = client.post(f'/api/v1/sessions/{session_id}/verify')

    assert verify.status_code == 200, verify.text
    run_detector.assert_not_called()
    spoof_result = verify.json()['spoof_detection']
    assert spoof_result['status'] == 'incomplete'
    assert spoof_result['predicted_class'] is None
    assert spoof_result['genuine_score'] is None
    assert spoof_result['spoof_score'] is None


def test_repeated_submission_is_rejected():
    session = client.post('/api/v1/sessions').json()
    session_id = session['session_id']
    audio_bytes = _wav_bytes(duration_seconds=1.0)
    client.post(f'/api/v1/sessions/{session_id}/audio', files={'file': ('sample.wav', audio_bytes, 'audio/wav')})
    first = client.post(f'/api/v1/sessions/{session_id}/verify')
    assert first.status_code == 200
    second = client.post(f'/api/v1/sessions/{session_id}/verify')
    assert second.status_code == 409
