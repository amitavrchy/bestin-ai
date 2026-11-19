import json
import re
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg

from app.config import settings

SHAPES = 'ABCDEFGHX'


def shape(char: str) -> str:
    if char in 'bmp': return 'A'
    if char in 'fv': return 'G'
    if char in 'ouqw': return 'F'
    if char in 'ae': return 'D'
    if char in 'iy': return 'C'
    if char in 'l': return 'H'
    return 'B'


def timeline(text: str, duration: float, words: list[dict]) -> list[dict]:
    if not words:
        tokens = re.findall(r"[A-Za-z']+", text) or ['']
        words = [{'word': w, 'start': duration * i / len(tokens), 'end': duration * (i + 1) / len(tokens)} for i, w in enumerate(tokens)]
    cues = []
    last = 0.0
    for word in words:
        start = max(last, min(duration, word['start']))
        end = max(start, min(duration, word['end']))
        if start - last > 0.12:
            cues.append({'start': last, 'end': start, 'value': 'X'})
        letters = re.sub('[^a-z]', '', word['word'].lower()) or 'a'
        # Limit very short segments to avoid visually noisy rapid changes.
        count = min(len(letters), max(1, int((end - start) / 0.06)))
        for i in range(count):
            a = start + (end - start) * i / count
            b = start + (end - start) * (i + 1) / count
            value = shape(letters[i * len(letters) // count])
            if cues and cues[-1]['value'] == value and abs(cues[-1]['end'] - a) < 0.001:
                cues[-1]['end'] = b
            elif b > a:
                cues.append({'start': a, 'end': b, 'value': value})
        last = end
    if duration - last > 0.12:
        cues.append({'start': last, 'end': duration, 'value': 'X'})
    return cues


def build(audio: bytes, text: str, duration: float, words: list[dict]) -> tuple[list[dict], str]:
    mode = settings.lipsync_backend
    binary = Path('tools/bin/rhubarb')
    if mode in ('auto', 'rhubarb') and binary.is_file():
        try:
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp)
                (p / 'audio.mp3').write_bytes(audio)
                (p / 'dialog.txt').write_text(text)
                subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i', str(p / 'audio.mp3'), '-ar', '16000', '-ac', '1', str(p / 'audio.wav')], check=True, capture_output=True, timeout=30)
                result = subprocess.run([str(binary.resolve()), '-f', 'json', '--extendedShapes', 'GHX', '-d', str(p / 'dialog.txt'), str(p / 'audio.wav')], check=True, capture_output=True, timeout=60)
                cues = json.loads(result.stdout)['mouthCues']
                return cues, 'rhubarb'
        except (OSError, subprocess.SubprocessError, ValueError, KeyError):
            pass
    use_words = bool(words) and mode != 'heuristic'
    return timeline(text, duration, words if use_words else []), 'words' if use_words else 'heuristic'
