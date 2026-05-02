"""Pre-baked audio filler cache — synthesized at startup to minimize first-response latency."""
import asyncio
from collections import deque
from typing import Optional

from .tts_interface import TTSClient

FILLER_TEXTS = [
    "Let me think about that for a second.",
    "That's a good point.",
    "Could you say that again?",
    "I'll have to get back to you on that.",
    "Sounds good to me.",
    "That makes sense.",
    "I appreciate you bringing that up.",
    "Let me consider that.",
    "Understood.",
    "Can you elaborate on that?",
    "I see what you mean.",
    "That's worth looking into.",
    "I hear you.",
    "Fair enough.",
    "Give me just a moment.",
    "That's an interesting perspective.",
    "I want to make sure I understand correctly.",
    "Good to know.",
    "I'll keep that in mind.",
    "Let's come back to that.",
]


class AudioCache:
    """Pre-synthesizes filler audio; deque prevents repeating within DEQUE_SIZE turns."""

    DEQUE_SIZE = 5

    def __init__(self, tts: TTSClient) -> None:
        self._tts = tts
        self._cache: dict[str, bytes] = {}
        self._recent: deque[str] = deque(maxlen=self.DEQUE_SIZE)

    async def preload(self) -> None:
        """Synthesize all FILLER_TEXTS and store PCM bytes."""
        tasks = [self._synthesize_one(text) for text in FILLER_TEXTS]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _synthesize_one(self, text: str) -> None:
        chunks: list[bytes] = []
        async for chunk in self._tts.synthesize(text):
            chunks.append(chunk)
        self._cache[text] = b"".join(chunks)

    def get_filler(self) -> Optional[bytes]:
        """Return PCM bytes for the next unused filler; avoids recent repeats."""
        available = [t for t in FILLER_TEXTS if t not in self._recent and t in self._cache]
        if not available:
            # All fillers were recently used — pick any cached one
            available = [t for t in FILLER_TEXTS if t in self._cache]
        if not available:
            return None
        text = available[0]
        self._recent.append(text)
        return self._cache[text]
