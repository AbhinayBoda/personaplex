"""Twilio Media Streams integration for PersonaPlex.

Handles:
  - Inbound calls: TwiML webhook that connects a Twilio call to a
    bidirectional WebSocket media stream.
  - Outbound calls: REST endpoint that initiates a Twilio call and
    points its media stream back to our WebSocket.
  - WebSocket handler: Receives mulaw 8 kHz audio from Twilio, converts
    to PCM 24 kHz, feeds PersonaPlex, converts the response back to
    mulaw, and sends it to Twilio.
"""

import asyncio
import json
import os
import uuid
from typing import Optional

import numpy as np
import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, Query
from fastapi.responses import Response

from .audio_converter import TwilioAudioConverter
from .model_manager import manager
from .recorder import ConversationRecorder
from .utils.logging import setup_logger, ColorizedLog

logger = setup_logger(__name__)

router = APIRouter(prefix="/twilio", tags=["twilio"])

converter = TwilioAudioConverter()


# ---------------------------------------------------------------------------
# TwiML Webhook -- Inbound Calls
# ---------------------------------------------------------------------------

@router.post("/inbound")
async def twilio_inbound(request: Request):
    """Twilio voice webhook.  Returns TwiML that opens a media stream back
    to our ``/twilio/media-stream`` WebSocket endpoint.

    Configure your Twilio phone number's Voice webhook to POST to this URL.
    """
    server_url = os.environ.get("SERVER_URL", "")
    if not server_url:
        host = request.headers.get("host", "localhost:8080")
        scheme = "wss" if request.url.scheme == "https" else "ws"
        server_url = f"{scheme}://{host}"
    else:
        server_url = server_url.replace("https://", "wss://").replace("http://", "ws://")

    voice_prompt = os.environ.get("DEFAULT_VOICE_PROMPT", "NATF2.pt")
    text_prompt = os.environ.get("DEFAULT_TEXT_PROMPT", "")

    ws_url = (
        f"{server_url}/twilio/media-stream"
        f"?voice_prompt={voice_prompt}"
        f"&text_prompt={text_prompt}"
    )

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}" />
    </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


# ---------------------------------------------------------------------------
# Outbound Call
# ---------------------------------------------------------------------------

@router.post("/outbound")
async def twilio_outbound(
    request: Request,
    to: str = Query(..., description="Phone number to call (E.164)"),
    voice_prompt: str = Query("NATF2.pt", description="Voice prompt filename"),
    text_prompt: str = Query("", description="System text prompt"),
):
    """Initiate an outbound call via Twilio REST API.

    The call will connect its media stream back to our WebSocket endpoint.
    Requires TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER
    environment variables.
    """
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_number = os.environ.get("TWILIO_PHONE_NUMBER")

    if not all([account_sid, auth_token, from_number]):
        from fastapi.responses import JSONResponse
        return JSONResponse(
            {"error": "Twilio credentials not configured. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER."},
            status_code=500,
        )

    server_url = os.environ.get("SERVER_URL", "")
    if not server_url:
        host = request.headers.get("host", "localhost:8080")
        scheme = "https" if request.url.scheme == "https" else "http"
        server_url = f"{scheme}://{host}"

    ws_url = server_url.replace("https://", "wss://").replace("http://", "ws://")
    stream_url = (
        f"{ws_url}/twilio/media-stream"
        f"?voice_prompt={voice_prompt}"
        f"&text_prompt={text_prompt}"
    )

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{stream_url}" />
    </Connect>
</Response>"""

    from twilio.rest import Client

    client = Client(account_sid, auth_token)
    call = client.calls.create(
        to=to,
        from_=from_number,
        twiml=twiml,
    )

    return {"call_sid": call.sid, "status": call.status}


# ---------------------------------------------------------------------------
# WebSocket -- Twilio Media Stream
# ---------------------------------------------------------------------------

@router.websocket("/media-stream")
async def twilio_media_stream(
    ws: WebSocket,
    voice_prompt: str = Query("NATF2.pt"),
    text_prompt: str = Query(""),
    seed: int = Query(-1),
):
    """Bidirectional Twilio Media Stream WebSocket handler.

    Protocol:
      Twilio sends JSON messages with ``event`` field:
        - ``connected``  : stream metadata
        - ``start``      : contains streamSid, call metadata
        - ``media``      : contains base64 mulaw payload
        - ``stop``       : stream ended

      We send back JSON with ``event: "media"`` containing base64 mulaw.
    """
    await ws.accept()
    clog = ColorizedLog.randomize()
    call_id = str(uuid.uuid4())[:8]
    stream_sid: Optional[str] = None
    close = False

    clog.log("info", f"[twilio:{call_id}] media-stream WebSocket connected")

    async with manager.acquire_session():
        manager.configure_prompts(voice_prompt, text_prompt, seed if seed != -1 else None)

        recorder = ConversationRecorder(
            call_id=call_id, sample_rate=manager.sample_rate
        )

        # Buffer for incoming audio until we have a full Mimi frame
        pcm_buffer = np.array([], dtype=np.float32)

        # Outbound audio buffer (model -> Twilio)
        outbound_buffer = np.array([], dtype=np.float32)

        # Run system prompts (voice conditioning + text prompt)
        async def is_alive():
            return not close

        await manager.lm_gen.step_system_prompts_async(manager.mimi, is_alive=is_alive)
        manager.mimi.reset_streaming()
        clog.log("info", f"[twilio:{call_id}] system prompts loaded")

        async def process_audio_frame(pcm_chunk: np.ndarray):
            """Process one Mimi-frame-sized PCM chunk through the model."""
            nonlocal outbound_buffer

            chunk_tensor = torch.from_numpy(pcm_chunk).to(device=manager.device)[None, None]
            codes = manager.mimi.encode(chunk_tensor)
            _ = manager.other_mimi.encode(chunk_tensor)

            for c in range(codes.shape[-1]):
                tokens = manager.lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
                assert tokens.shape[1] == manager.lm_gen.lm_model.dep_q + 1
                main_pcm = manager.mimi.decode(tokens[:, 1:9])
                _ = manager.other_mimi.decode(tokens[:, 1:9])
                main_pcm_np = main_pcm[0, 0].cpu().numpy()
                recorder.add_outbound(main_pcm_np)
                outbound_buffer = np.concatenate([outbound_buffer, main_pcm_np])

        async def send_loop():
            """Send accumulated model audio back to Twilio."""
            nonlocal outbound_buffer
            # Twilio expects ~20ms chunks (160 samples at 8kHz = 480 at 24kHz)
            twilio_chunk_size = int(0.02 * manager.sample_rate)

            while not close:
                await asyncio.sleep(0.005)
                if len(outbound_buffer) < twilio_chunk_size:
                    continue

                chunk = outbound_buffer[:twilio_chunk_size]
                outbound_buffer = outbound_buffer[twilio_chunk_size:]

                payload = converter.pcm_to_twilio_payload(chunk)
                msg = {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": payload},
                }
                try:
                    await ws.send_json(msg)
                except Exception:
                    break

        send_task: Optional[asyncio.Task] = None

        try:
            while True:
                raw = await ws.receive_text()
                data = json.loads(raw)
                event = data.get("event")

                if event == "connected":
                    clog.log("info", f"[twilio:{call_id}] stream connected")

                elif event == "start":
                    stream_sid = data["start"]["streamSid"]
                    clog.log("info", f"[twilio:{call_id}] stream started, sid={stream_sid}")
                    send_task = asyncio.create_task(send_loop())

                elif event == "media":
                    payload = data["media"]["payload"]
                    pcm_24k = converter.twilio_payload_to_pcm(payload)
                    recorder.add_inbound(pcm_24k)
                    pcm_buffer = np.concatenate([pcm_buffer, pcm_24k])

                    while len(pcm_buffer) >= manager.frame_size:
                        frame = pcm_buffer[: manager.frame_size]
                        pcm_buffer = pcm_buffer[manager.frame_size :]
                        await process_audio_frame(frame)

                elif event == "stop":
                    clog.log("info", f"[twilio:{call_id}] stream stopped")
                    break

        except WebSocketDisconnect:
            clog.log("info", f"[twilio:{call_id}] WebSocket disconnected")
        except Exception as exc:
            clog.log("error", f"[twilio:{call_id}] error: {exc}")
        finally:
            close = True
            if send_task is not None:
                send_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass

            filepath = recorder.save()
            if filepath:
                clog.log("info", f"[twilio:{call_id}] recording saved: {filepath}")
            clog.log("info", f"[twilio:{call_id}] session ended")
