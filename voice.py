"""
Push-to-talk voice input for Pocket Jarvis.

Telegram voice notes arrive as Opus-in-Ogg (.oga), referenced by a
file_id — not raw audio bytes. This module:
  1. Resolves file_id -> a downloadable file_path via Telegram's getFile
  2. Downloads the .oga bytes
  3. Converts to 16kHz mono PCM WAV via ffmpeg (the exact format Vosk's
     KaldiRecognizer expects)
  4. Transcribes with a local, offline Vosk model

Deliberately offline: no external STT API call, so this doesn't touch
Gemini's rate limits and doesn't add a new internet-dependent service to
babysit. Only cost is local CPU + a one-time model download.

Setup (once, in Termux):
    pkg install ffmpeg
    pip install vosk
    # download e.g. vosk-model-small-en-us-0.15, unzip it somewhere, then
    # point VOSK_MODEL_PATH (.env) at the extracted folder.
"""

import os
import json
import wave
import subprocess
import tempfile
import logging

import requests
from vosk import Model, KaldiRecognizer

log = logging.getLogger("pocket-jarvis")

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"
FILE_BASE = f"https://api.telegram.org/file/bot{BOT_TOKEN}"
VOSK_MODEL_PATH = os.environ.get("VOSK_MODEL_PATH")

# Lazy-loaded once, reused across calls. Loading the model is the slow
# part (~seconds); transcribing a short push-to-talk clip is fast once
# it's loaded.
_model = None


class VoiceError(Exception):
    """Raised for any step (download/convert/transcribe) failing, so the
    caller can send a clean Telegram reply instead of a bare traceback."""


def _get_model() -> Model:
    global _model
    if _model is None:
        if not VOSK_MODEL_PATH or not os.path.isdir(VOSK_MODEL_PATH):
            raise VoiceError(
                f"VOSK_MODEL_PATH not set or not a directory: {VOSK_MODEL_PATH!r}. "
                "Download a model (e.g. vosk-model-small-en-us-0.15) and point "
                "VOSK_MODEL_PATH at the extracted folder in .env."
            )
        log.info("Loading Vosk model from %s ...", VOSK_MODEL_PATH)
        _model = Model(VOSK_MODEL_PATH)
    return _model


def _download_voice_file(file_id: str) -> str:
    """file_id -> local .oga path. Telegram requires the two-step
    getFile-then-download dance; file_id alone isn't a URL."""
    resp = requests.get(f"{API_BASE}/getFile", params={"file_id": file_id}, timeout=15)
    resp.raise_for_status()
    result = resp.json().get("result")
    if not result or "file_path" not in result:
        raise VoiceError(f"getFile returned no file_path for file_id={file_id!r}")
    remote_path = result["file_path"]

    file_resp = requests.get(f"{FILE_BASE}/{remote_path}", timeout=30)
    file_resp.raise_for_status()

    fd, local_path = tempfile.mkstemp(suffix=".oga")
    with os.fdopen(fd, "wb") as f:
        f.write(file_resp.content)
    return local_path


def _convert_to_wav(oga_path: str) -> str:
    """Opus/Ogg -> 16kHz mono PCM WAV. Raises VoiceError with ffmpeg's
    actual stderr (e.g. "ffmpeg not found") rather than a bare
    CalledProcessError, since that reason needs to be visible in the
    Telegram reply, not just the Termux log."""
    wav_path = oga_path.replace(".oga", ".wav")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", oga_path,
                "-ar", "16000", "-ac", "1", "-f", "wav", wav_path,
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except FileNotFoundError:
        raise VoiceError("ffmpeg not found — install it in Termux: pkg install ffmpeg")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace")[:300] if e.stderr else "unknown error"
        raise VoiceError(f"ffmpeg conversion failed: {stderr}")
    return wav_path


def _transcribe_wav(wav_path: str) -> str:
    model = _get_model()
    with wave.open(wav_path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise VoiceError("Converted WAV isn't mono 16-bit PCM — conversion step is broken.")
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(False)

        text_parts = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                text_parts.append(json.loads(rec.Result()).get("text", ""))
        text_parts.append(json.loads(rec.FinalResult()).get("text", ""))

    return " ".join(p for p in text_parts if p).strip()


def transcribe_voice_message(file_id: str) -> str:
    """
    Full pipeline: Telegram file_id -> transcribed text.
    Raises VoiceError on any failure. Cleans up temp files regardless of
    outcome so repeated voice notes don't litter Termux's tmp dir.
    """
    oga_path = _download_voice_file(file_id)
    wav_path = None
    try:
        wav_path = _convert_to_wav(oga_path)
        text = _transcribe_wav(wav_path)
        if not text:
            raise VoiceError("Heard nothing intelligible — try again, closer to the mic.")
        return text
    finally:
        for p in (oga_path, wav_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass