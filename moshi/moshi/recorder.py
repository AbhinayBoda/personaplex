"""WAV recorder for saving PersonaPlex conversations.

Records both sides (inbound user audio and outbound model audio) and writes
stereo WAV files so conversations can be reviewed or archived.
"""

import os
import wave
from datetime import datetime
from typing import Optional

import numpy as np

from .utils.logging import setup_logger

logger = setup_logger(__name__)

RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", "recordings")


class ConversationRecorder:
    """Accumulates PCM audio from both sides and saves to WAV.

    Audio is expected as float32 numpy arrays at the given sample rate.
    The saved WAV is stereo: channel 0 = user (inbound), channel 1 = model (outbound).
    """

    def __init__(
        self,
        call_id: str,
        sample_rate: int = 24000,
        output_dir: Optional[str] = None,
    ):
        self.call_id = call_id
        self.sample_rate = sample_rate
        self.output_dir = output_dir or RECORDINGS_DIR
        self._inbound: list[np.ndarray] = []
        self._outbound: list[np.ndarray] = []

    def add_inbound(self, pcm: np.ndarray) -> None:
        """Append a chunk of user audio (float32)."""
        self._inbound.append(pcm.copy())

    def add_outbound(self, pcm: np.ndarray) -> None:
        """Append a chunk of model-generated audio (float32)."""
        self._outbound.append(pcm.copy())

    def save(self) -> Optional[str]:
        """Write the accumulated audio to a stereo WAV file.

        Returns the file path, or None if there is nothing to save.
        """
        if not self._inbound and not self._outbound:
            return None

        inbound = np.concatenate(self._inbound) if self._inbound else np.array([], dtype=np.float32)
        outbound = np.concatenate(self._outbound) if self._outbound else np.array([], dtype=np.float32)

        # Pad the shorter channel to match lengths
        max_len = max(len(inbound), len(outbound))
        if len(inbound) < max_len:
            inbound = np.pad(inbound, (0, max_len - len(inbound)))
        if len(outbound) < max_len:
            outbound = np.pad(outbound, (0, max_len - len(outbound)))

        # Interleave into stereo int16
        stereo = np.stack([inbound, outbound], axis=-1)
        stereo_int16 = np.clip(stereo * 32767, -32768, 32767).astype(np.int16)

        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{self.call_id}.wav"
        filepath = os.path.join(self.output_dir, filename)

        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(stereo_int16.tobytes())

        logger.info(f"saved recording: {filepath} ({max_len / self.sample_rate:.1f}s)")
        return filepath
