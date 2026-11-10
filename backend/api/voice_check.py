"""Voice health check (experimental): a 5-second sustained "aaah" -> Praat voice measures -> Parkinson's voice screening.

The measures are computed with praat-parselmouth exactly as Praat's voice report defines them, which is how the
training set (UCI, Sakar et al. 2018) was measured. The model failed the switch-on bar on held-out people
(AUC 0.76), was trained on studio microphones, and has never been tested on phone recordings, so it is only shown
in the Experimental section with a notice. It is never offered to model search or to patients.
"""
import json
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent / "voice_models"
META = HERE / "parkinson_voice.json"
NOTICE = ("Screening, experimental. Not a diagnosis. Trained on studio recordings of 252 adults in Istanbul and not tested on "
          "phone microphones. A changed voice has many causes, so talk to a doctor if you are worried.")
MIN_SECONDS = 3.0
MIN_VOICED = 0.6
MIN_HNR_DB = 3.0
MAX_Z = 6.0  # a measure this far from every training recording means the recording, not the voice, is unusual
F0_MIN, F0_MAX = 75, 500
_lock = threading.Lock()
_session: dict = {}


class VoiceCheckError(ValueError):
    """The recording can't be used. The message is shown to the user."""


def available() -> bool:
    return META.exists() and META.with_suffix(".onnx").exists() and _ffmpeg() is not None


def _ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - optional; fall back to a system ffmpeg
        return shutil.which("ffmpeg")


def _to_wav(audio: bytes, folder: Path) -> Path:
    src, dst = folder / "in", folder / "voice.wav"
    src.write_bytes(audio)
    run = subprocess.run([_ffmpeg() or "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "44100", str(dst)],
                         capture_output=True, timeout=30, check=False)
    if run.returncode or not dst.exists():
        raise VoiceCheckError("Couldn't read that recording. Please record again.")
    return dst


def measure(wav: Path) -> dict:
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(str(wav))
    if snd.get_total_duration() < MIN_SECONDS:
        raise VoiceCheckError("The recording was too short. Hold the 'aaah' for about 5 seconds.")
    # Drop the first and last half second, where the voice starts and stops.
    snd = snd.extract_part(0.5, snd.get_total_duration() - 0.5) if snd.get_total_duration() > 4 else snd
    pitch = call(snd, "To Pitch", 0.0, F0_MIN, F0_MAX)
    frames = pitch.selected_array["frequency"]
    if (frames > 0).mean() < MIN_VOICED:
        raise VoiceCheckError("I couldn't hear a steady 'aaah'. Please try again in a quiet room, about 15 cm from the phone.")
    pp = call(snd, "To PointProcess (periodic, cc)", F0_MIN, F0_MAX)
    j = (0, 0, 0.0001, 0.02, 1.3)
    harm = call(snd, "To Harmonicity (cc)", 0.01, F0_MIN, 0.1, 1.0)
    hnr = call(harm, "Get mean", 0, 0)
    if not np.isfinite(hnr) or hnr < MIN_HNR_DB:
        raise VoiceCheckError("There was too much background noise. Please try again somewhere quieter.")
    # Praat's voice report gives the mean autocorrelation and noise-to-harmonics ratio used in training.
    report = call([snd, pitch, pp], "Voice report", 0, 0, F0_MIN, F0_MAX, 1.3, 1.6, 0.03, 0.45)
    values = {}
    for line in report.splitlines():
        key, _, val = line.strip().partition(":")
        values[key.strip()] = val.strip()
    formant = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
    out = {
        "meanPeriodPulses": call(pp, "Get mean period", *j),
        "stdDevPeriodPulses": call(pp, "Get stdev period", *j),
        "locPctJitter": call(pp, "Get jitter (local)", *j),
        "locAbsJitter": call(pp, "Get jitter (local, absolute)", *j),
        "rapJitter": call(pp, "Get jitter (rap)", *j),
        "ppq5Jitter": call(pp, "Get jitter (ppq5)", *j),
        "ddpJitter": call(pp, "Get jitter (ddp)", *j),
        "locShimmer": call([snd, pp], "Get shimmer (local)", *j, 1.6),
        "locDbShimmer": call([snd, pp], "Get shimmer (local_dB)", *j, 1.6),
        "apq3Shimmer": call([snd, pp], "Get shimmer (apq3)", *j, 1.6),
        "apq5Shimmer": call([snd, pp], "Get shimmer (apq5)", *j, 1.6),
        "apq11Shimmer": call([snd, pp], "Get shimmer (apq11)", *j, 1.6),
        "ddaShimmer": call([snd, pp], "Get shimmer (dda)", *j, 1.6),
        "meanAutoCorrHarmonicity": float(values.get("Mean autocorrelation", "nan").split()[0]),
        "meanNoiseToHarmHarmonicity": float(values.get("Mean noise-to-harmonics ratio", "nan").split()[0]),
        "meanHarmToNoiseHarmonicity": float(hnr),
    }
    for i in range(1, 5):
        out[f"f{i}"] = call(formant, "Get mean", i, 0, 0, "hertz")
    bad = [k for k, v in out.items() if not np.isfinite(v)]
    if bad:
        raise VoiceCheckError("I couldn't measure your voice clearly. Please try again with one steady 'aaah'.")
    out["pitch_hz"] = float(np.median(frames[frames > 0]))
    return out


def _vector(meta: dict, m: dict, male: bool) -> np.ndarray:
    row = []
    for name in meta["features"]:
        v = 1.0 if (name == "male" and male) else 0.0 if name == "male" else float(m[name])
        row.append(np.log(max(v, 1e-9)) if name in meta["log"] else v)
    return np.array([row], dtype=np.float32)


def check(audio: bytes, sex: str | None) -> dict:
    if not available():
        raise VoiceCheckError("The voice check isn't available on this server.")
    if sex not in ("Male", "Female"):
        raise VoiceCheckError("Please add your sex (male or female) in your profile first: voice measures differ between them.")
    meta = json.loads(META.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:  # the recording is deleted straight after measuring
        m = measure(_to_wav(audio, Path(tmp)))
    x = _vector(meta, m, sex == "Male")
    z = np.abs((x[0] - np.array(meta["train_mean"])) / np.array(meta["train_std"]))
    if float(z.max()) > MAX_Z:
        raise VoiceCheckError("This recording is very different from the ones the check was trained on, so I can't use it. "
                              "Try again in a quiet room with one steady 'aaah'.")
    with _lock:
        if "s" not in _session:
            _session["s"] = ort.InferenceSession(str(META.with_suffix(".onnx")), providers=["CPUExecutionProvider"])
    s = _session["s"]
    p = float(s.run(None, {meta["input"]: x})[1][0][1])
    flagged = p >= meta["threshold"]
    return {
        "label": "Voice pattern similar to recordings from people with Parkinson's" if flagged else "No Parkinson's-like voice pattern found",
        "flagged": flagged,
        "score": round(p, 3),
        "measures": {
            "Pitch": f"{m['pitch_hz']:.0f} Hz",
            "Jitter (pitch wobble)": f"{100 * m['locPctJitter']:.2f}%",
            "Shimmer (loudness wobble)": f"{100 * m['locShimmer']:.1f}%",
            "Harmonics-to-noise": f"{m['meanHarmToNoiseHarmonicity']:.1f} dB",
        },
        "test": meta["metrics"],
        "trained_on": meta["trained_on"],
        "credit": "Approach from mohnabilfadl/Parkinson_Detection_Using_Voice_Analysis; data from Sakar et al. 2018 (UCI). Trained by the Curanex team.",
        "notice": NOTICE,
    }
