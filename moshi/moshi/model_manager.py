"""Shared model singleton for PersonaPlex.

Loads the model once at startup and provides session-level locking
so that both the React frontend WebSocket and Twilio endpoints
share the same GPU-resident model.
"""

import asyncio
import os
import random
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Literal
import tarfile

import numpy as np
import sentencepiece
import torch

# Enable hf_transfer for 5-10x faster model downloads from HuggingFace.
# Must be set BEFORE importing huggingface_hub.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import hf_hub_download  # noqa: E402

from .models import loaders, MimiModel, LMModel, LMGen
from .utils.logging import setup_logger

logger = setup_logger(__name__)

DeviceString = Literal["cuda", "cpu"]


def torch_auto_device(requested: Optional[str] = None) -> torch.device:
    """Return a torch.device based on the requested string or availability."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def wrap_with_system_tags(text: str) -> str:
    """Add system tags as the model expects if they are missing."""
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def _get_voice_prompt_dir(
    voice_prompt_dir: Optional[str], hf_repo: str
) -> Optional[str]:
    """Download and extract voice prompts from HF if no local dir is given."""
    if voice_prompt_dir is not None:
        return voice_prompt_dir

    logger.info("retrieving voice prompts from HuggingFace")
    voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"

    if not voices_dir.exists():
        logger.info(f"extracting {voices_tgz} to {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)

    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")

    return str(voices_dir)


def _get_static_path(static: Optional[str]) -> Optional[str]:
    """Download React frontend dist from HF if no local path is given."""
    if static is None:
        logger.info("retrieving the static content")
        dist_tgz = hf_hub_download("nvidia/personaplex-7b-v1", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        return str(dist)
    elif static != "none":
        return static
    return None


class ModelManager:
    """Loads PersonaPlex model components once, shared across all endpoints.

    The model can only handle one session at a time (max_batch=1), so an
    asyncio.Lock gates access.  Call ``acquire_session`` to obtain exclusive
    access for the duration of one conversation.
    """

    def __init__(self) -> None:
        self.mimi: Optional[MimiModel] = None
        self.other_mimi: Optional[MimiModel] = None
        self.text_tokenizer: Optional[sentencepiece.SentencePieceProcessor] = None
        self.lm_gen: Optional[LMGen] = None
        self.device: Optional[torch.device] = None
        self.voice_prompt_dir: Optional[str] = None
        self.static_path: Optional[str] = None
        self.frame_size: int = 0
        self._lock = asyncio.Lock()
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def sample_rate(self) -> int:
        assert self.mimi is not None
        return int(self.mimi.sample_rate)

    @property
    def frame_rate(self) -> float:
        assert self.mimi is not None
        return float(self.mimi.frame_rate)

    def load(
        self,
        device: str = "cuda",
        hf_repo: str = loaders.DEFAULT_REPO,
        mimi_weight: Optional[str] = None,
        moshi_weight: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        voice_prompt_dir: Optional[str] = None,
        static_path: Optional[str] = None,
        cpu_offload: bool = False,
    ) -> None:
        """Load all model components.  Call once at startup."""
        self.device = torch_auto_device(device)
        seed_all(42424242)

        # Increment HF download counter
        hf_hub_download(hf_repo, "config.json")

        # --- Mimi codec (two instances: one for encode, one for decode) ---
        logger.info("loading mimi")
        if mimi_weight is None:
            mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, self.device)
        self.other_mimi = loaders.get_mimi(mimi_weight, self.device)
        logger.info("mimi loaded")

        # --- Text tokenizer ---
        if tokenizer_path is None:
            tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

        # --- Moshi LM ---
        logger.info("loading moshi LM")
        if moshi_weight is None:
            moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)
        lm = loaders.get_moshi_lm(
            moshi_weight, device=self.device, cpu_offload=cpu_offload
        )
        lm.eval()
        logger.info("moshi LM loaded")

        # --- LMGen wrapper ---
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lm_gen = LMGen(
            lm,
            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
            sample_rate=int(self.mimi.sample_rate),
            device=self.device,
            frame_rate=self.mimi.frame_rate,
            save_voice_prompt_embeddings=False,
        )

        # --- Streaming mode ---
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        # --- Voice prompts ---
        self.voice_prompt_dir = _get_voice_prompt_dir(voice_prompt_dir, hf_repo)
        if self.voice_prompt_dir is not None:
            assert os.path.exists(self.voice_prompt_dir), (
                f"Voice prompt directory missing: {self.voice_prompt_dir}"
            )
        logger.info(f"voice_prompt_dir = {self.voice_prompt_dir}")

        # --- Static frontend ---
        self.static_path = _get_static_path(static_path)
        logger.info(f"static_path = {self.static_path}")

        self._ready = True

    def warmup(self) -> None:
        """Run dummy iterations to compile CUDA graphs."""
        assert self._ready, "Call load() before warmup()"
        logger.info("warming up the model")
        for _ in range(4):
            chunk = torch.zeros(
                1, 1, self.frame_size, dtype=torch.float32, device=self.device
            )
            codes = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:9])
                _ = self.other_mimi.decode(tokens[:, 1:9])

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        logger.info("warmup complete")

    def resolve_voice_prompt(self, filename: str) -> str:
        """Return the full path to a voice prompt file, raising if not found."""
        if self.voice_prompt_dir is None:
            raise ValueError("No voice_prompt_dir configured")
        full_path = os.path.join(self.voice_prompt_dir, filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"Voice prompt '{filename}' not found in '{self.voice_prompt_dir}'"
            )
        return full_path

    def configure_prompts(
        self,
        voice_prompt: str,
        text_prompt: str,
        seed: Optional[int] = None,
    ) -> None:
        """Set voice/text prompts and optional seed before a session."""
        voice_path = self.resolve_voice_prompt(voice_prompt)
        if self.lm_gen.voice_prompt != voice_path:
            if voice_path.endswith(".pt"):
                self.lm_gen.load_voice_prompt_embeddings(voice_path)
            else:
                self.lm_gen.load_voice_prompt(voice_path)

        if text_prompt:
            self.lm_gen.text_prompt_tokens = self.text_tokenizer.encode(
                wrap_with_system_tags(text_prompt)
            )
        else:
            self.lm_gen.text_prompt_tokens = None

        if seed is not None and seed != -1:
            seed_all(seed)

    def reset_streaming(self) -> None:
        """Reset all streaming states before a new session."""
        self.mimi.reset_streaming()
        self.other_mimi.reset_streaming()
        self.lm_gen.reset_streaming()

    @asynccontextmanager
    async def acquire_session(self):
        """Context manager that acquires the model lock for one session."""
        async with self._lock:
            self.reset_streaming()
            yield self


# Module-level singleton
manager = ModelManager()
