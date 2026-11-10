"""Voice health check: the route answers, bad audio is refused with a message, and the result carries its notice."""
import math
import shutil
import struct
import wave

import pytest
from fastapi.testclient import TestClient

from api import voice_check
from main import app

client = TestClient(app)


def test_info_route_reports_availability():
    r = client.get("/voice-check")
    assert r.status_code == 200 and "Not a diagnosis" in r.json()["notice"]


def test_junk_audio_is_refused_with_a_reason():
    r = client.post("/voice-check", files={"file": ("a.webm", b"not audio", "audio/webm")}, data={"sex": "Male"})
    assert r.status_code == 422 and r.json()["detail"]


def test_sex_is_required():
    r = client.post("/voice-check", files={"file": ("a.webm", b"x", "audio/webm")})
    assert r.status_code == 422


@pytest.mark.skipif(not voice_check.available(), reason="model or ffmpeg missing")
def test_a_steady_tone_is_measured_or_refused_never_crashes(tmp_path):
    path = tmp_path / "tone.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 140 * i / 16000))) for i in range(16000 * 5)))
    try:
        out = voice_check.check(path.read_bytes(), "Female")
        assert out["notice"] == voice_check.NOTICE and out["label"]
    except voice_check.VoiceCheckError as e:
        assert str(e)


def test_silence_is_refused(tmp_path):
    if not voice_check.available() or not shutil.which("ffmpeg") and voice_check._ffmpeg() is None:
        pytest.skip("ffmpeg missing")
    path = tmp_path / "quiet.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * 16000 * 5)
    with pytest.raises(voice_check.VoiceCheckError):
        voice_check.check(path.read_bytes(), "Male")
