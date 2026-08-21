"""
Push-to-talk voice input for Pocket Jarvis.

Telegram voice notes arrive as Opus-in-Ogg (.oga), referenced by a
file_id — not raw audio bytes. This module:
  1. Resolves file_id -> a downloadable file_path via Telegram's getFile
  2. Downloads the .oga bytes
  3. Converts to 16kHz mono PCM WAV via ffmpeg (the format whisper.cpp
     expects)
  4. Transcribes by shelling out to a compiled whisper.cpp `whisper-cli`
     binary — deliberately NOT the `vosk` pip package: Vosk hasn't
     shipped a release since ~2022, and its PyPI wheels are only built
     for specific Python ABI versions, which likely doesn't include
     newer interpreters (e.g. 3.13) — pip would fall back to compiling
     from source, which needs a full Kaldi toolchain and is exactly the
     kind of fragile build step you don't want on a phone. whisper.cpp
     sidesteps this entirely: it's a native binary invoked via
     subprocess, same as ffmpeg already is below, so there's no Python
     package/wheel compatibility question at all.

Deliberately offline: no external STT API call, so this doesn't touch
Gemini's rate limits and doesn't add a new internet-dependent service to
babysit. Only cost is local CPU + a one-time model download.

Setup (once, in Termux):
    pkg install ffmpeg git cmake clang -y
    git clone https://github.com/ggerganov/whisper.cpp
    cd whisper.cpp
    cmake -B build && cmake --build build --config Release -j4
    bash ./models/download-ggml-model.sh tiny.en
    # then point .env at the resulting binary + model:
    #   WHISPER_CLI_PATH=/path/to/whisper.cpp/build/bin/whisper-cli
    #   WHISPER_MODEL_PATH=/path/to/whisper.cpp/models/ggml-tiny.en.bin
"""

import os
import subprocess
import tempfile
import logging

import requests

log = logging.getLogger("pocket-jarvis")

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"
FILE_BASE = f"https://api.telegram.org/file/bot{BOT_TOKEN}"
WHISPER_CLI_PATH = os.environ.get("WHISPER_CLI_PATH", "whisper-cli")
WHISPER_MODEL_PATH = os.environ.get("WHISPER_MODEL_PATH")


class VoiceError(Exception):
    """Raised for any step (download/convert/transcribe) failing, so the
    caller can send a clean Telegram reply instead of a bare traceback."""


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
    """Shells out to whisper-cli, which writes a plain-text transcript
    to <out_prefix>.txt (-otxt) rather than parsing stdout — stdout's
    exact formatting (timestamps, progress lines) has changed across
    whisper.cpp versions before, so the file output is the more stable
    contract to depend on."""
    if not WHISPER_MODEL_PATH or not os.path.isfile(WHISPER_MODEL_PATH):
        raise VoiceError(
            f"WHISPER_MODEL_PATH not set or not a file: {WHISPER_MODEL_PATH!r}. "
            "Run models/download-ggml-model.sh (e.g. tiny.en) in your whisper.cpp "
            "checkout and point WHISPER_MODEL_PATH at the resulting .bin in .env."
        )

    out_prefix = wav_path[:-4] if wav_path.endswith(".wav") else wav_path
    txt_path = out_prefix + ".txt"

    try:
        subprocess.run(
            [
                WHISPER_CLI_PATH,
                "-m", WHISPER_MODEL_PATH,
                "-f", wav_path,
                "-otxt", "-of", out_prefix,
                "-l", "en",
                "-nt",  # no timestamps in the output text
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
    except FileNotFoundError:
        raise VoiceError(
            f"whisper-cli not found at {WHISPER_CLI_PATH!r} — build whisper.cpp "
            "and set WHISPER_CLI_PATH to build/bin/whisper-cli in .env."
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace")[:300] if e.stderr else "unknown error"
        raise VoiceError(f"whisper-cli failed: {stderr}")

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        raise VoiceError(f"whisper-cli ran but produced no output file: {e}")
    finally:
        if os.path.exists(txt_path):
            os.remove(txt_path)

    return text.strip()


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