"""Audio format conversion between Twilio mulaw 8kHz and PersonaPlex PCM 24kHz.

Twilio Media Streams deliver audio as base64-encoded mu-law at 8 kHz mono.
PersonaPlex (Mimi codec) expects float32 PCM at 24 kHz mono.

This module provides bidirectional conversion with proper resampling.
"""

import audioop
import base64

import numpy as np
import sphn


TWILIO_SAMPLE_RATE = 8000
MODEL_SAMPLE_RATE = 24000


class TwilioAudioConverter:
    """Bidirectional converter: Twilio mulaw 8kHz  <-->  PCM float32 24kHz."""

    def __init__(
        self,
        twilio_rate: int = TWILIO_SAMPLE_RATE,
        model_rate: int = MODEL_SAMPLE_RATE,
    ):
        self.twilio_rate = twilio_rate
        self.model_rate = model_rate

    # ------------------------------------------------------------------
    # Twilio --> Model
    # ------------------------------------------------------------------

    def mulaw_bytes_to_pcm16(self, mulaw_bytes: bytes) -> np.ndarray:
        """Decode mu-law bytes to int16 PCM numpy array."""
        pcm_bytes = audioop.ulaw2lin(mulaw_bytes, 2)
        return np.frombuffer(pcm_bytes, dtype=np.int16)

    def pcm16_to_float32(self, pcm16: np.ndarray) -> np.ndarray:
        """Convert int16 PCM to float32 in [-1.0, 1.0]."""
        return pcm16.astype(np.float32) / 32768.0

    def resample_to_model(self, audio: np.ndarray) -> np.ndarray:
        """Resample from Twilio rate to model rate.

        ``sphn.resample`` expects shape (channels, samples), so we add/remove
        the channel dimension here.
        """
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]  # (1, T)
        resampled = sphn.resample(
            audio,
            src_sample_rate=self.twilio_rate,
            dst_sample_rate=self.model_rate,
        )
        return resampled[0]  # back to (T,)

    def twilio_payload_to_pcm(self, b64_payload: str) -> np.ndarray:
        """Full pipeline: base64 mulaw --> float32 PCM at model sample rate.

        Returns a 1-D float32 numpy array.
        """
        mulaw_bytes = base64.b64decode(b64_payload)
        pcm16 = self.mulaw_bytes_to_pcm16(mulaw_bytes)
        pcm_float = self.pcm16_to_float32(pcm16)
        return self.resample_to_model(pcm_float)

    # ------------------------------------------------------------------
    # Model --> Twilio
    # ------------------------------------------------------------------

    def resample_to_twilio(self, audio: np.ndarray) -> np.ndarray:
        """Resample from model rate to Twilio rate."""
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        resampled = sphn.resample(
            audio,
            src_sample_rate=self.model_rate,
            dst_sample_rate=self.twilio_rate,
        )
        return resampled[0]

    def float32_to_pcm16(self, audio: np.ndarray) -> np.ndarray:
        """Convert float32 [-1, 1] to int16 PCM."""
        clipped = np.clip(audio, -1.0, 1.0)
        return (clipped * 32767).astype(np.int16)

    def pcm16_to_mulaw_bytes(self, pcm16: np.ndarray) -> bytes:
        """Encode int16 PCM to mu-law bytes."""
        pcm_bytes = pcm16.tobytes()
        return audioop.lin2ulaw(pcm_bytes, 2)

    def pcm_to_twilio_payload(self, pcm_float: np.ndarray) -> str:
        """Full pipeline: float32 PCM at model rate --> base64 mulaw for Twilio.

        Input: 1-D float32 array at ``model_rate``.
        Returns: base64-encoded string ready for Twilio WebSocket.
        """
        resampled = self.resample_to_twilio(pcm_float)
        pcm16 = self.float32_to_pcm16(resampled)
        mulaw = self.pcm16_to_mulaw_bytes(pcm16)
        return base64.b64encode(mulaw).decode("ascii")
