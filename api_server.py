#!/usr/bin/env python3
"""
Voice AI Tourist Guide — WebSocket API Server
═══════════════════════════════════════════════════════════════════════════════

Exposes the full STT → GPT → TTS pipeline over WebSocket so any external
project (web app, mobile app, another backend) can use it.

WebSocket Protocol  (/ws/voice)
────────────────────────────────
CLIENT → SERVER:
  • Binary frame  : Raw PCM audio bytes  (16-bit signed, 22050 Hz, mono)
  • Text  frame   : JSON control message
        {"type": "stop"}        — gracefully end the session

SERVER → CLIENT:
  • Binary frame  : Raw PCM audio bytes  (16-bit signed, 22050 Hz, mono)
                    (This is the AI voice response — play it directly)
  • Text  frame   : JSON event message
        {"type": "ready"}
        {"type": "transcript",    "text": "...", "is_final": true/false}
        {"type": "response_start"}
        {"type": "response_text", "text": "..."}   ← full AI reply text
        {"type": "response_end"}
        {"type": "error",         "message": "..."}

REST Endpoints
──────────────
  GET  /health          → service liveness
  GET  /config          → audio format + model info (useful for client setup)
  POST /text            → simple text-in / text-out (stateless, for quick test)

Run locally:
  uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Deploy (e.g. Railway / Render / Fly.io):
  Set the same env vars as .env — OPENAI_API_KEY, DEEPGRAM_API_KEY,
  ELEVEN_LABS_API_KEY, ELEVENLABS_VOICE_ID, OPENAI_MODEL
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Optional

import aiohttp
import nltk
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

# NLTK resources (required by some vocode-adjacent libs, harmless here)
for _res in ["punkt", "punkt_tab"]:
    try:
        nltk.download(_res, quiet=True)
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Settings ────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    openai_api_key: str
    deepgram_api_key: str
    eleven_labs_api_key: str
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"   # Rachel
    openai_model: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


# ── Prompts (identical to main.py) ─────────────────────────────────────────

SYSTEM_PROMPT = """
You are a warm, helpful, and highly professional AI Tourist Guide for travelers visiting India. 
Your goal is to sound like a friendly local guide who helps tourists understand what to do after arriving in India.

CONVERSATIONAL GUIDELINES:
- Speak in a natural, polite, and conversational style.
- Use warm markers like "Oh, welcome to India" or "Sure, I can help with that."
- Keep every response short and simple for spoken audio (1–2 sentences max).
- Use natural fillers like "Got it," "Let me help you with that," or "No worries."
- Never use bullet points, symbols, or markdown. Speak in clear human-like sentences.

CORE RULES:
1. Ask only ONE question at a time.
2. Never repeat information the traveler already gave.
3. Focus only on helping tourists travel safely and comfortably in India.
4. Sound like a friendly human guide, not a robot.
5. Always prioritize traveler safety and helpful local guidance.

HELP & SUPPORT INFORMATION:
If a traveler needs help or emergency assistance:
- Police: Dial 100
- Ambulance: Dial 102 or 108

Local Travel Assistance Contact:
Ankit Sharma  
Phone: 8298197805

You can provide this contact if the traveler asks for local help or assistance.

---

CONVERSATION FLOW:

1. GREETING:
Introduce yourself as a tourist guide and welcome them to India.

2. FIRST QUESTIONS:
Ask their name and where they are visiting from.

3. ARRIVAL CHECK:
Ask if they have already arrived in India or are planning their trip.

4. TRAVEL GUIDANCE:
Help them with airport guidance, transportation, currency exchange, SIM cards,
hotel check-in, safety tips, tourist places, and local food recommendations.

5. CITY GUIDANCE:
Ask which city they are currently in or planning to visit.

6. TOURIST HELP:
Provide simple guidance about places, travel options, and local tips.

7. EMERGENCY SUPPORT:
If the traveler needs urgent help, calmly guide them to emergency services or
provide the support contact.

8. FRIENDLY TONE:
Always sound welcoming and helpful, like a friendly local guide assisting a visitor.
"""

INITIAL_MESSAGE = (
    "Hello and welcome to India! I'm your virtual tourist guide, and I'm here to help "
    "make your trip smooth and enjoyable. May I know your name and which country you're "
    "visiting from?"
)

# Audio config — must match everywhere (client, Deepgram, ElevenLabs)
SAMPLE_RATE = 22050
ENCODING    = "linear16"
CHANNELS    = 1


# ── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Voice AI Tourist Guide API",
    description="WebSocket + REST API for the India Tourist Guide voice assistant.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── ElevenLabs TTS helpers ──────────────────────────────────────────────────

def _elevenlabs_tts_url() -> str:
    return (
        f"https://api.elevenlabs.io/v1/text-to-speech"
        f"/{settings.elevenlabs_voice_id}/stream"
        f"?output_format=pcm_22050&optimize_streaming_latency=3"
    )

def _elevenlabs_headers() -> Dict:
    return {
        "xi-api-key": settings.eleven_labs_api_key,
        "Content-Type": "application/json",
    }

def _elevenlabs_body(text: str) -> Dict:
    return {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.50,
            "similarity_boost": 0.80,
            "style": 0.50,
            "use_speaker_boost": True,
        },
    }


async def stream_tts_to_websocket(text: str, ws: WebSocket) -> None:
    """Synthesize `text` with ElevenLabs and stream PCM audio frames to `ws`."""
    if not text.strip():
        return
    async with aiohttp.ClientSession() as session:
        async with session.post(
            _elevenlabs_tts_url(),
            headers=_elevenlabs_headers(),
            json=_elevenlabs_body(text),
        ) as resp:
            if resp.status != 200:
                err = await resp.text()
                raise RuntimeError(f"ElevenLabs {resp.status}: {err}")
            async for chunk in resp.content.iter_chunked(4096):
                if chunk:
                    await ws.send_bytes(chunk)


async def synthesize_to_bytes(text: str) -> bytes:
    """Synthesize `text` and return all PCM audio as a single bytes object."""
    if not text.strip():
        return b""
    chunks: List[bytes] = []
    async with aiohttp.ClientSession() as session:
        async with session.post(
            _elevenlabs_tts_url(),
            headers=_elevenlabs_headers(),
            json=_elevenlabs_body(text),
        ) as resp:
            if resp.status != 200:
                err = await resp.text()
                raise RuntimeError(f"ElevenLabs {resp.status}: {err}")
            async for chunk in resp.content.iter_chunked(4096):
                if chunk:
                    chunks.append(chunk)
    return b"".join(chunks)


# ── OpenAI GPT streaming ────────────────────────────────────────────────────

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


async def stream_gpt_response(
    messages: List[Dict],
    ws: WebSocket,
) -> str:
    """
    Stream a GPT response. Each complete sentence is synthesized and sent as
    audio to `ws` immediately as it arrives, minimising latency.
    Returns the complete response text.
    """
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    await ws.send_text(json.dumps({"type": "response_start"}))

    full_text = ""
    sentence_buf = ""

    stream = await client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        stream=True,
        max_tokens=200,
        temperature=0.72,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        full_text += delta
        sentence_buf += delta

        # Flush complete sentences to TTS immediately
        parts = _SENTENCE_END.split(sentence_buf)
        if len(parts) > 1:
            for sentence in parts[:-1]:
                await stream_tts_to_websocket(sentence, ws)
            sentence_buf = parts[-1]

    # Flush remaining text
    if sentence_buf.strip():
        await stream_tts_to_websocket(sentence_buf, ws)

    await ws.send_text(json.dumps({"type": "response_text", "text": full_text}))
    await ws.send_text(json.dumps({"type": "response_end"}))

    return full_text


# ── Session ─────────────────────────────────────────────────────────────────

class ConversationSession:
    """Holds per-connection state: message history and the transcript queue."""

    def __init__(self) -> None:
        self.messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.transcript_q: asyncio.Queue = asyncio.Queue()
        self.active: bool = True


async def conversation_loop(ws: WebSocket, session: ConversationSession) -> None:
    """
    Runs as a background task.
    Waits for finalized Deepgram transcripts, calls GPT, streams audio back.
    """
    while session.active:
        transcript = await session.transcript_q.get()
        if transcript is None:          # shutdown signal
            break
        if not transcript.strip():
            continue

        logger.info("Transcript ▶ %s", transcript)
        await ws.send_text(json.dumps({"type": "transcript", "text": transcript, "is_final": True}))

        session.messages.append({"role": "user", "content": transcript})
        try:
            reply = await stream_gpt_response(session.messages, ws)
            session.messages.append({"role": "assistant", "content": reply})
            logger.info("Reply ◀ %s", reply[:80])
        except Exception as exc:
            logger.exception("GPT/TTS error")
            await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))


# ── WebSocket endpoint ──────────────────────────────────────────────────────

@app.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket) -> None:
    """
    Full-duplex voice conversation endpoint.

    Audio format expected from client
    ───────────────────────────────────
    Encoding : PCM signed 16-bit little-endian
    Sample rate: 22 050 Hz
    Channels : 1 (mono)
    """
    await websocket.accept()
    logger.info("🔌 Client connected: %s", websocket.client)

    session = ConversationSession()

    # ── Connect to Deepgram real-time STT ───────────────────────────────────
    # We use Deepgram's v1 WebSocket API directly (no SDK dependency on version)
    dg_ws_url = (
        "wss://api.deepgram.com/v1/listen"
        f"?encoding={ENCODING}&sample_rate={SAMPLE_RATE}&channels={CHANNELS}"
        "&punctuate=true&interim_results=true&endpointing=500&language=en"
    )
    dg_headers = {"Authorization": f"Token {settings.deepgram_api_key}"}

    conv_task: Optional[asyncio.Task] = None

    try:
        async with aiohttp.ClientSession() as http_session:
            async with http_session.ws_connect(dg_ws_url, headers=dg_headers) as dg_ws:
                logger.info("✅ Deepgram connected")

                # ── Start background tasks ──────────────────────────────────
                conv_task = asyncio.create_task(conversation_loop(websocket, session))

                async def _read_deepgram():
                    """Forward Deepgram transcripts → session queue."""
                    async for msg in dg_ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            alt = (
                                data.get("channel", {})
                                    .get("alternatives", [{}])[0]
                            )
                            text = alt.get("transcript", "")
                            is_final = data.get("is_final", False)

                            if text:
                                # Stream interim results to client for display
                                await websocket.send_text(
                                    json.dumps({
                                        "type": "transcript",
                                        "text": text,
                                        "is_final": is_final,
                                    })
                                )
                            if is_final and text.strip():
                                await session.transcript_q.put(text)

                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            logger.warning("Deepgram WS closed/error: %s", msg)
                            break

                dg_reader_task = asyncio.create_task(_read_deepgram())

                # ── Send ready signal + initial greeting ────────────────────
                await websocket.send_text(json.dumps({"type": "ready"}))
                try:
                    await websocket.send_text(json.dumps({"type": "response_start"}))
                    await stream_tts_to_websocket(INITIAL_MESSAGE, websocket)
                    await websocket.send_text(
                        json.dumps({"type": "response_text", "text": INITIAL_MESSAGE})
                    )
                    await websocket.send_text(json.dumps({"type": "response_end"}))
                    session.messages.append(
                        {"role": "assistant", "content": INITIAL_MESSAGE}
                    )
                except Exception:
                    logger.exception("Initial greeting error")

                # ── Main receive loop ───────────────────────────────────────
                while session.active:
                    try:
                        raw = await websocket.receive()
                    except WebSocketDisconnect:
                        logger.info("Client disconnected")
                        break

                    if "bytes" in raw:
                        # Forward raw PCM audio to Deepgram
                        audio_chunk = raw["bytes"]
                        if audio_chunk:
                            await dg_ws.send_bytes(audio_chunk)

                    elif "text" in raw:
                        try:
                            ctrl = json.loads(raw["text"])
                        except json.JSONDecodeError:
                            continue
                        if ctrl.get("type") == "stop":
                            logger.info("Client requested stop")
                            break

                # ── Teardown ────────────────────────────────────────────────
                session.active = False
                session.transcript_q.put_nowait(None)   # stop conversation loop

                # Tell Deepgram we're done
                try:
                    await dg_ws.send_str(json.dumps({"type": "CloseStream"}))
                except Exception:
                    pass

                dg_reader_task.cancel()
                if conv_task:
                    conv_task.cancel()

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Unhandled error in voice_ws")
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "message": "Internal server error"})
            )
        except Exception:
            pass
    finally:
        session.active = False
        session.transcript_q.put_nowait(None)
        if conv_task and not conv_task.done():
            conv_task.cancel()
        logger.info("🔌 Session cleaned up: %s", websocket.client)


# ── REST Endpoints ──────────────────────────────────────────────────────────

@app.get("/health", tags=["Utility"])
async def health():
    """Liveness check — returns OK if the server is running."""
    return {"status": "ok", "service": "Voice AI Tourist Guide"}


@app.get("/debug", tags=["Utility"])
async def debug():
    """Debug endpoint to check if API keys are loaded."""
    return {
        "openai_key_set": bool(settings.openai_api_key),
        "deepgram_key_set": bool(settings.deepgram_api_key),
        "elevenlabs_key_set": bool(settings.eleven_labs_api_key),
        "openai_key_prefix": settings.openai_api_key[:10] if settings.openai_api_key else None,
        "model": settings.openai_model,
        "voice_id": settings.elevenlabs_voice_id,
    }


@app.get("/config", tags=["Utility"])
async def config():
    """
    Returns the audio format the server expects and produces.
    The client should use these values when capturing and playing audio.
    """
    return {
        "websocket_endpoint": "/ws/voice",
        "audio": {
            "encoding":    ENCODING,
            "sample_rate": SAMPLE_RATE,
            "channels":    CHANNELS,
            "bit_depth":   16,
        },
        "model":    settings.openai_model,
        "voice_id": settings.elevenlabs_voice_id,
    }


class TextRequest(BaseModel):
    text: str
    history: Optional[List[Dict]] = None   # optional prior turns


@app.post("/text", tags=["Text API"])
async def text_chat(req: TextRequest):
    """
    Stateless text-in / text-out endpoint.
    Useful for testing the LLM pipeline without audio.
    Pass `history` to maintain multi-turn context from your own side.
    """
    messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if req.history:
        messages.extend(req.history)
    messages.append({"role": "user", "content": req.text})

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    resp = await client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        max_tokens=200,
        temperature=0.72,
    )
    return {
        "response": resp.choices[0].message.content,
        "model":    resp.model,
        "usage":    resp.usage.model_dump() if resp.usage else None,
    }


# ── Entry point (dev only — use uvicorn in production) ─────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
