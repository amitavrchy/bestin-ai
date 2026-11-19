from faster_whisper import WhisperModel
import os

if __name__ == "__main__":
    WhisperModel(os.getenv("LOCAL_STT_MODEL", "base.en"), device="cpu", compute_type="int8")
