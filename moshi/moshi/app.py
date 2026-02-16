"""FastAPI application for PersonaPlex.

Unified server that handles:
  - ``/health``              : Health check for Cloud Run / load balancers
  - ``/api/chat``            : React frontend WebSocket (binary Opus protocol)
  - ``/twilio/inbound``      : TwiML webhook for inbound Twilio calls
  - ``/twilio/outbound``     : REST endpoint to initiate outbound calls
  - ``/twilio/media-stream`` : Twilio Media Streams WebSocket
  - ``/``                    : Static files for React frontend

Launch with:
    uvicorn moshi.app:app --host 0.0.0.0 --port 8080
or:
    python -m moshi.app [--host 0.0.0.0] [--port 8080] [--device cuda]
"""

import argparse
import asyncio
import os
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import sphn
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .model_manager import manager
from .twilio_handler import router as twilio_router
from .utils.logging import setup_logger, ColorizedLog

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: load model on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup so it's ready before the first request."""
    device = os.environ.get("DEVICE", "cuda")
    hf_repo = os.environ.get("HF_REPO", "nvidia/personaplex-7b-v1")
    cpu_offload = os.environ.get("CPU_OFFLOAD", "0") == "1"
    static_path = os.environ.get("STATIC_PATH", None)

    with torch.no_grad():
        manager.load(
            device=device,
            hf_repo=hf_repo,
            cpu_offload=cpu_offload,
            static_path=static_path,
        )
        manager.warmup()

    # Mount static files for React frontend after model loads
    if manager.static_path and os.path.exists(manager.static_path):
        app.mount(
            "/static",
            StaticFiles(directory=manager.static_path),
            name="static-assets",
        )
        logger.info(f"serving static content from {manager.static_path}")

    logger.info("PersonaPlex ready")
    yield
    logger.info("PersonaPlex shutting down")


app = FastAPI(
    title="PersonaPlex API",
    description="Speech-to-speech conversational AI with voice and role control",
    version="0.1.0",
    lifespan=lifespan,
)

# Include Twilio routes
app.include_router(twilio_router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check endpoint for Cloud Run startup probes and load balancers."""
    if not manager.ready:
        return JSONResponse({"status": "loading"}, status_code=503)

    info = {
        "status": "healthy",
        "model": "personaplex-7b-v1",
        "device": str(manager.device),
    }
    if manager.device and manager.device.type == "cuda":
        info["gpu_memory_allocated_mb"] = round(
            torch.cuda.memory_allocated() / 1024 / 1024, 1
        )
    return info


# ---------------------------------------------------------------------------
# React Frontend: root page
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Serve the React frontend index.html."""
    if manager.static_path:
        index = os.path.join(manager.static_path, "index.html")
        if os.path.exists(index):
            return FileResponse(index)
    return {"message": "PersonaPlex API is running. No frontend deployed."}


# ---------------------------------------------------------------------------
# /api/chat -- React frontend WebSocket (binary Opus protocol)
# ---------------------------------------------------------------------------

@app.websocket("/api/chat")
async def websocket_chat(
    ws: WebSocket,
    voice_prompt: str = Query("NATF2.pt"),
    text_prompt: str = Query(""),
    seed: int = Query(-1),
):
    """WebSocket endpoint compatible with the existing React frontend.

    Binary protocol:
      0x00  Handshake (server -> client)
      0x01  Audio (Opus encoded, bidirectional)
      0x02  Text token (server -> client)
    """
    await ws.accept()
    clog = ColorizedLog.randomize()
    clog.log("info", "incoming React WebSocket connection")

    close = False

    async with manager.acquire_session():
        # Configure prompts
        manager.configure_prompts(
            voice_prompt, text_prompt, seed if seed != -1 else None
        )

        opus_writer = sphn.OpusStreamWriter(manager.sample_rate)
        opus_reader = sphn.OpusStreamReader(manager.sample_rate)

        # System prompts
        async def is_alive():
            if close:
                return False
            return True

        await manager.lm_gen.step_system_prompts_async(
            manager.mimi, is_alive=is_alive
        )
        manager.mimi.reset_streaming()
        clog.log("info", "system prompts done")

        if not close:
            # Send handshake
            await ws.send_bytes(b"\x00")
            clog.log("info", "sent handshake")

        # ----- recv loop -----
        async def recv_loop():
            nonlocal close
            try:
                while not close:
                    data = await ws.receive_bytes()
                    if len(data) == 0:
                        continue
                    kind = data[0]
                    if kind == 1:  # audio
                        opus_reader.append_bytes(data[1:])
                    else:
                        clog.log("warning", f"unknown message kind {kind}")
            except WebSocketDisconnect:
                pass
            except Exception as exc:
                clog.log("error", f"recv error: {exc}")
            finally:
                close = True

        # ----- opus processing loop -----
        async def opus_loop():
            all_pcm_data = None
            while not close:
                await asyncio.sleep(0.001)
                pcm = opus_reader.read_pcm()
                if pcm.shape[-1] == 0:
                    continue
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))

                while all_pcm_data.shape[-1] >= manager.frame_size:
                    chunk = all_pcm_data[: manager.frame_size]
                    all_pcm_data = all_pcm_data[manager.frame_size :]
                    chunk_t = torch.from_numpy(chunk).to(device=manager.device)[
                        None, None
                    ]
                    codes = manager.mimi.encode(chunk_t)
                    _ = manager.other_mimi.encode(chunk_t)
                    for c in range(codes.shape[-1]):
                        tokens = manager.lm_gen.step(codes[:, :, c : c + 1])
                        if tokens is None:
                            continue
                        assert (
                            tokens.shape[1]
                            == manager.lm_gen.lm_model.dep_q + 1
                        )
                        main_pcm = manager.mimi.decode(tokens[:, 1:9])
                        _ = manager.other_mimi.decode(tokens[:, 1:9])
                        main_pcm = main_pcm.cpu()
                        opus_writer.append_pcm(main_pcm[0, 0].numpy())

                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = manager.text_tokenizer.id_to_piece(
                                text_token
                            )
                            _text = _text.replace("\u2581", " ")
                            msg = b"\x02" + bytes(_text, encoding="utf8")
                            try:
                                await ws.send_bytes(msg)
                            except Exception:
                                break

        # ----- send loop -----
        async def send_loop():
            while not close:
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    try:
                        await ws.send_bytes(b"\x01" + msg)
                    except Exception:
                        break

        # Run all three loops concurrently
        tasks = [
            asyncio.create_task(recv_loop()),
            asyncio.create_task(opus_loop()),
            asyncio.create_task(send_loop()),
        ]
        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED
        )
        close = True
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        try:
            await ws.close()
        except Exception:
            pass
        clog.log("info", "session closed")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def cli():
    """CLI entrypoint for ``personaplex-serve``."""
    parser = argparse.ArgumentParser(description="PersonaPlex FastAPI Server")
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--hf-repo", default="nvidia/personaplex-7b-v1", type=str)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--static", type=str, default=None)
    args = parser.parse_args()

    os.environ.setdefault("DEVICE", args.device)
    os.environ.setdefault("HF_REPO", args.hf_repo)
    if args.cpu_offload:
        os.environ["CPU_OFFLOAD"] = "1"
    if args.static:
        os.environ["STATIC_PATH"] = args.static

    import uvicorn

    uvicorn.run(
        "moshi.app:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    cli()