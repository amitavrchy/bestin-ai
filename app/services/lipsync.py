
"""Lip-sync timeline generation.

Preference order in auto mode:
  1. Visemes or word timings already returned by the TTS provider
  2. Whisper word-level timestamps on the generated audio (OpenAI)
  3. Rhubarb Lip Sync CLI, if `rhubarb` is on PATH (and ffmpeg for conversion)
  4. A phoneme-heuristic fallback that always works, optionally refined
     with amplitude when a WAV can be produced
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from mutagen import File as MutagenFile

from app.providers.base import STTProvider
from app.schemas import VisemeCue

# Rhubarb viseme set: A-H plus X (idle / rest).
#   A closed mouth (M, B, P)
#   B slight opening (most consonants, default)
#   C open (EH, AE, IH)
#   D wide open (AA)
#   E slightly rounded (ER)
#   F puckered (UW, OW, W)
#   G teeth on lip (F, V)
#   H tongue to teeth (L, TH)
#   X idle rest

MULTI_RULES = [
    ("ch", "B"),
    ("sh", "B"),
    ("zh", "B"),
    ("th", "H"),
    ("wh", "F"),
    ("qu", "F"),
    ("ng", "B"),
    ("ck", "B"),
    ("ph", "G"),
    ("gh", "B"),
    ("oo", "F"),
    ("ee", "C"),
    ("ea", "C"),
    ("oa", "F"),
    ("ou", "F"),
    ("ow", "F"),
    ("oy", "F"),
    ("oi", "F"),
    ("ai", "C"),
    ("ay", "C"),
    ("au", "D"),
    ("aw", "D"),
    ("ew", "F"),
    ("ie", "C"),
    ("ey", "C"),
    ("ar", "E"),
    ("er", "E"),
    ("ir", "E"),
    ("or", "E"),
    ("ur", "E"),
    ("ing", "C"),
]

SINGLE_RULES = {
    "a": "D",
    "e": "C",
    "i": "C",
    "o": "F",
    "u": "F",
    "y": "C",
    "b": "A",
    "m": "A",
    "p": "A",
    "f": "G",
    "v": "G",
    "w": "F",
    "l": "H",
    "r": "E",
}


def rhubarb_available() -> bool:
    return shutil.which("rhubarb") is not None


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def audio_duration_seconds(path: str) -> float:
    info = MutagenFile(path)
    if info is not None and getattr(info, "info", None) is not None:
        length = getattr(info.info, "length", None)
        if length:
            return float(length)
    return 0.0


def text_to_visemes(text: str) -> List[str]:
    """Map English text to a sequence of Rhubarb viseme letters."""
    visemes: List[str] = []
    i = 0
    lowered = text.lower()
    length = len(lowered)
    while i < length:
        ch = lowered[i]
        if ch in " \t":
            i += 1
            continue
        if ch in ".,!?;:—–-()[]{}\"'“”‘’":
            visemes.append("X")
            i += 1
            continue
        if not ch.isalpha():
            i += 1
            continue
        matched = False
        # Prefer the longest multi-letter rule that matches here.
        for token, viseme in sorted(MULTI_RULES, key=lambda item: len(item[0]), reverse=True):
            if lowered.startswith(token, i):
                visemes.append(viseme)
                i += len(token)
                matched = True
                break
        if matched:
            continue
        visemes.append(SINGLE_RULES.get(ch, "B"))
        i += 1
    return visemes or ["X"]


def _cues_from_pairs(pairs: List[Tuple[float, float, str]]) -> List[VisemeCue]:
    cues: List[VisemeCue] = []
    for start, end, viseme in pairs:
        if end < start:
            end = start
        letter = viseme if viseme in list("ABCDEFGHX") else "B"
        if cues and cues[-1].viseme == letter and abs(cues[-1].end - start) < 0.02:
            cues[-1].end = max(cues[-1].end, end)
        else:
            cues.append(VisemeCue(start=max(0.0, start), end=end, viseme=letter))
    return cues


def visemes_from_word_timings(words: List[Dict[str, Any]]) -> List[VisemeCue]:
    pairs: List[Tuple[float, float, str]] = []
    cursor = 0.0
    for item in words:
        word = str(item.get("word") or "")
        start = float(item.get("start", cursor))
        end = float(item.get("end", start))
        if start > cursor + 0.04:
            pairs.append((cursor, start, "X"))
        letters = [v for v in text_to_visemes(word) if v != "X"] or ["B"]
        span = max(end - max(start, 0.0), 0.04)
        slot = span / float(len(letters))
        for index, letter in enumerate(letters):
            s = start + slot * index
            e = start + slot * (index + 1)
            pairs.append((s, e, letter))
        cursor = max(end, cursor)
    return _cues_from_pairs(pairs)


def heuristic_timeline(text: str, duration: float, rms_frames: Optional[List[Tuple[float, float]]] = None) -> List[VisemeCue]:
    duration = max(duration, 0.2)
    letters = text_to_visemes(text)
    # Weight punctuation pauses a bit longer than a single phoneme.
    weights = []
    for letter in letters:
        weights.append(2.2 if letter == "X" else 1.0)
    total = sum(weights) or 1.0
    pairs: List[Tuple[float, float, str]] = []
    t = 0.0
    # Leave a tiny rest at the start and end.
    lead = min(0.08, duration * 0.05)
    tail = min(0.12, duration * 0.06)
    usable = max(duration - lead - tail, 0.08)
    if lead > 0:
        pairs.append((0.0, lead, "X"))
        t = lead
    for letter, weight in zip(letters, weights):
        span = usable * (weight / total)
        pairs.append((t, t + span, letter))
        t += span
    if t < duration:
        pairs.append((t, duration, "X"))

    if rms_frames:
        # Force a closed mouth during quiet frames, without destroying speech cues.
        quiet = {round(start, 2) for start, rms in rms_frames if rms < 0.02}
        if quiet:
            refined: List[Tuple[float, float, str]] = []
            hop = 0.02
            for start, end, letter in pairs:
                cursor = start
                while cursor < end - 1e-6:
                    nxt = min(cursor + hop, end)
                    key = round(cursor, 2)
                    viseme = "X" if key in quiet else letter
                    refined.append((cursor, nxt, viseme))
                    cursor = nxt
            pairs = refined
    return _cues_from_pairs(pairs)


def _wav_via_ffmpeg(src: str, wav_path: str) -> bool:
    if not ffmpeg_available():
        return False
    command = [
        "ffmpeg",
        "-y",
        "-i",
        src,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        wav_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0 and os.path.exists(wav_path)


def wav_rms_frames(wav_path: str, hop: float = 0.02) -> Optional[List[Tuple[float, float]]]:
    try:
        import array
        import wave

        with wave.open(wav_path, "rb") as handle:
            nchannels = handle.getnchannels()
            sampwidth = handle.getsampwidth()
            framerate = handle.getframerate()
            nframes = handle.getnframes()
            raw = handle.readframes(nframes)
        if sampwidth != 2:
            return None
        samples = array.array("h")
        samples.frombytes(raw)
        if nchannels > 1:
            mono = []
            for i in range(0, len(samples), nchannels):
                mono.append(int(sum(samples[i : i + nchannels]) / nchannels))
            samples = array.array("h", mono)
        hop_n = max(int(framerate * hop), 1)
        frames: List[Tuple[float, float]] = []
        peak = 1.0
        values = []
        for i in range(0, len(samples), hop_n):
            window = samples[i : i + hop_n]
            if not window:
                continue
            acc = math.sqrt(sum(float(s) * float(s) for s in window) / float(len(window)))
            peak = max(peak, acc)
            values.append((i / float(framerate), acc))
        if peak <= 0:
            return None
        return [(t, acc / peak) for t, acc in values]
    except Exception:
        return None


def rhubarb_timeline(audio_path: str, text: str) -> Optional[List[VisemeCue]]:
    if not rhubarb_available():
        return None
    tmp_dir = tempfile.mkdtemp(prefix="bestin-rhubarb-")
    try:
        wav_path = os.path.join(tmp_dir, "speech.wav")
        if not _wav_via_ffmpeg(audio_path, wav_path):
            # If the source is already wav, try it directly.
            if audio_path.lower().endswith(".wav"):
                wav_path = audio_path
            else:
                return None
        dialog_path = os.path.join(tmp_dir, "dialog.txt")
        with open(dialog_path, "w", encoding="utf-8") as handle:
            handle.write(text)
        out_path = os.path.join(tmp_dir, "cues.json")
        command = [
            "rhubarb",
            "-f",
            "json",
            "-o",
            out_path,
            "-q",
            "-d",
            dialog_path,
            wav_path,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.exists(out_path):
            return None
        with open(out_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        pairs: List[Tuple[float, float, str]] = []
        for cue in payload.get("mouthCues") or []:
            pairs.append(
                (
                    float(cue.get("start", 0.0)),
                    float(cue.get("end", 0.0)),
                    str(cue.get("value") or "X"),
                )
            )
        return _cues_from_pairs(pairs) if pairs else None
    except Exception:
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def available_backends() -> Dict[str, Any]:
    return {
        "mode": "auto",
        "rhubarb": rhubarb_available(),
        "ffmpeg": ffmpeg_available(),
        "whisper_alignment": True,
        "heuristic": True,
    }


def build_timeline(
    audio_path: str,
    text: str,
    mode: str = "auto",
    tts_info: Optional[Dict[str, Any]] = None,
    stt: Optional[STTProvider] = None,
) -> Tuple[List[VisemeCue], str]:
    """Return (cues, backend_name)."""
    tts_info = tts_info or {}
    duration = audio_duration_seconds(audio_path)

    def _heuristic() -> List[VisemeCue]:
        rms = None
        if ffmpeg_available():
            tmp = tempfile.mkdtemp(prefix="bestin-wav-")
            try:
                wav_path = os.path.join(tmp, "speech.wav")
                if _wav_via_ffmpeg(audio_path, wav_path):
                    rms = wav_rms_frames(wav_path)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        return heuristic_timeline(text, duration or 2.0, rms_frames=rms)

    # 1. Provider-supplied visemes or word timings.
    raw_visemes = tts_info.get("visemes") if tts_info else None
    if raw_visemes and mode in ("auto", "provider"):
        cues: List[VisemeCue] = []
        for item in raw_visemes:
            if isinstance(item, VisemeCue):
                cues.append(item)
            elif isinstance(item, dict):
                cues.append(VisemeCue(**item))
        if cues:
            return cues, "provider"

    word_timings = tts_info.get("word_timings") if tts_info else None
    if word_timings and mode in ("auto", "provider", "whisper"):
        cues = visemes_from_word_timings(word_timings)
        if cues:
            return cues, "provider-words"

    # 2. Whisper word alignment.
    if mode in ("auto", "whisper") and stt is not None and hasattr(stt, "transcribe_words"):
        try:
            words = stt.transcribe_words(audio_path)
            if words:
                cues = visemes_from_word_timings(words)
                if cues:
                    if duration and cues[-1].end < duration:
                        cues.append(VisemeCue(start=cues[-1].end, end=duration, viseme="X"))
                    return cues, "whisper"
        except Exception:
            if mode == "whisper":
                return _heuristic(), "heuristic (whisper failed)"

    # 3. Rhubarb CLI.
    if mode in ("auto", "rhubarb"):
        cues = rhubarb_timeline(audio_path, text)
        if cues:
            return cues, "rhubarb"
        if mode == "rhubarb":
            return _heuristic(), "heuristic (rhubarb unavailable)"

    # 4. Always-on fallback.
    return _heuristic(), "heuristic"
