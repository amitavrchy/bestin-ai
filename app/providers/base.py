from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError


class ChatProvider(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


class STTProvider(ABC):
    @abstractmethod
    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str,
        mime: Optional[str] = None,
    ) -> str:
        raise NotImplementedError

    def transcribe_words(self, audio_path: str) -> Optional[List[Dict[str, Any]]]:
        """Optional word timings: [{word, start, end}, ...]."""
        return None


class TTSProvider(ABC):
    @abstractmethod
    def synthesize(self, text: str, out_path: str) -> Dict[str, Any]:
        """Write audio to out_path.

        Return a dict that may include:
          path, mime, visemes, word_timings
        """
        raise NotImplementedError
