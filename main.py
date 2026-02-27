#!/usr/bin/env python3
"""
Voice AI Assistant - Powered by Vocode
STT: Deepgram | LLM: OpenAI GPT | TTS: ElevenLabs (natural voice)
Run: python main.py
"""

import asyncio
import signal
import sys
import nltk
from typing import Optional

for resource in ['punkt', 'punkt_tab']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

from vocode.streaming.input_device.microphone_input import MicrophoneInput
from vocode.streaming.output_device.blocking_speaker_output import BlockingSpeakerOutput
from vocode.logging import configure_pretty_logging
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
from vocode.streaming.synthesizer.base_synthesizer import SynthesisResult
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber

configure_pretty_logging()


# ── Settings ───────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    openai_api_key: str
    deepgram_api_key: str
    eleven_labs_api_key: str
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel
    openai_model: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


SYSTEM_PROMPT = """You are an AI voice assistant for Eco Tech Pest Control. You speak in a friendly, professional, and conversational tone — like a helpful human representative. 

CORE RULES:
- Ask only ONE question at a time. Wait for the caller's response before moving on.
- Never repeat information the caller has already given.
- Keep responses to 1–2 sentences max. Keep it natural for spoken audio.
- Never use bullet points, symbols, or markdown.
- Stay on pest control topics only. If unrelated, say: "I'm sorry, I'm only able to help with pest control. Would you like me to connect you to a live agent?"
- Always sound warm and natural — avoid robotic phrasing.

---

LOOKUP DATA (Use this for verification):

Account Database:
- 555-867-5309 → Sarah Mitchell (Appt: March 15, 2025, 9 AM – 12 PM, General Pest Control)
- 555-234-7890 → James Ortega (No upcoming appointments)
- Other → Account not found

Zip Codes Serviced: 90210, 30301. (73301 is NOT serviced).

Pricing:
- Single-family All Season: $149 initial, then $89/quarter
- Single-family One-Time: $199
- Multi-unit One-Time: $249

---

CONVERSATION FLOW:
1. GREETING: State the greeting exactly as: "Thank you for calling Eco Tech Pest Control! Are you a current customer with us, or are you calling for the first time?"
2. CURRENT CUSTOMER: Ask for phone number -> Look up in database.
3. NEW CUSTOMER: Ask for zip code -> Check service area.
4. COLLECT INFO: Get Name, Number, Pest type, and Square footage (One question at a time).
5. PRICING: Recommend plans based on property type.
6. BOOKING: Collect Address, City, Day, Time, and Email.
7. CONFIRMATION: Read a short natural summary of all details.
8. AGENT TRANSFER: If asked for a human, say: "Of course, let me connect you with one of our specialists right now. Please hold for just a moment."
"""

INITIAL_MESSAGE = "Thank you for calling Eco Tech Pest Control! Are you a current customer with us, or are you calling for the first time?"


# ── Natural Voice Synthesizer ──────────────────────────────────────────────

class NaturalElevenLabsSynthesizer(ElevenLabsSynthesizer):
    """
    Overrides get_chunks and create_speech_uncached to:
    1. Force pcm_22050 (free tier compatible — pcm_44100 requires Pro)
    2. Inject full voice_settings including style + use_speaker_boost
       which Vocode's default implementation does NOT pass to ElevenLabs
    3. Use 22050Hz sample rate end-to-end (mic, speaker, TTS all match)
    """

    async def get_chunks(
        self,
        url: str,
        headers: dict,
        body: dict,
        chunk_size: int,
        chunk_queue: asyncio.Queue,
    ):
        # Force free-tier format in URL
        url = url.replace("pcm_44100", "pcm_22050")
        url = url.replace("pcm_24000", "pcm_22050")

        # Inject full voice_settings — Vocode only sends stability+similarity_boost
        # style and use_speaker_boost are what make the voice natural vs robotic
        body["voice_settings"] = {
            "stability": 0.50,           # 0.3-0.5 = expressive, 0.7+ = robotic/monotone
            "similarity_boost": 0.80,    # 0.7-0.85 = natural, faithful to voice
            "style": 0.35,               # 0.2-0.4 = subtle natural expression
            "use_speaker_boost": True,   # Clearer, more present voice
        }

        await super().get_chunks(url, headers, body, chunk_size, chunk_queue)

    async def create_speech_uncached(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        # Force free tier sample rate before URL is built
        self.output_format = "pcm_22050"
        self.sample_rate = 22050
        self.upsample = None
        return await super().create_speech_uncached(
            message, chunk_size, is_first_text_chunk, is_sole_text_chunk
        )


# ── Main ───────────────────────────────────────────────────────────────────

async def main():
    settings = Settings()

    print("\n" + "═" * 60)
    print("  🎙️  Voice AI Assistant  (Powered by Vocode)")
    print("  STT: Deepgram  |  LLM: GPT  |  TTS: ElevenLabs")
    print("═" * 60)
    print("  Speak naturally. Press Ctrl+C to exit.\n")

    # Use 22050Hz end-to-end — mic, speaker and ElevenLabs all match
    # This was your key fix — bypasses the helper that defaults to 44100Hz
    microphone_input  = MicrophoneInput.from_default_device(sampling_rate=22050)
    speaker_output    = BlockingSpeakerOutput.from_default_device(sampling_rate=22050)

    conversation = StreamingConversation(
        output_device=speaker_output,

        # ── STT: Deepgram ──────────────────────────────────────────────────
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_device(
                microphone_input,
                endpointing_config=PunctuationEndpointingConfig(),
                api_key=settings.deepgram_api_key,
            ),
        ),

        # ── LLM: OpenAI GPT ───────────────────────────────────────────────
        agent=ChatGPTAgent(
            ChatGPTAgentConfig(
                openai_api_key=settings.openai_api_key,
                model_name=settings.openai_model,
                prompt_preamble=SYSTEM_PROMPT,
                initial_message=BaseMessage(text=INITIAL_MESSAGE),
                generate_responses=True,
            )
        ),

        # ── TTS: ElevenLabs ────────────────────────────────────────────────
        synthesizer=NaturalElevenLabsSynthesizer(
            ElevenLabsSynthesizerConfig.from_output_device(
                speaker_output,
                api_key=settings.eleven_labs_api_key,
                voice_id=settings.elevenlabs_voice_id,
                model_id="eleven_multilingual_v2",  # Best quality model
                stability=0.50,
                similarity_boost=0.80,
                optimize_streaming_latency=1,        # 1 = quality/latency balance
                experimental_streaming=True,
            ),
        ),
    )

    await conversation.start()
    print("  ✅ Ready! Speak now...\n")

    signal.signal(
        signal.SIGINT,
        lambda *_: asyncio.create_task(conversation.terminate())
    )

    while conversation.is_active():
        chunk = await microphone_input.get_audio()
        conversation.receive_audio(chunk)

    print("\n  Session ended. Goodbye! 👋\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye! 👋\n")
        sys.exit(0)