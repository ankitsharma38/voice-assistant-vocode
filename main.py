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


SYSTEM_PROMPT = """You are a friendly, warm, intelligent AI voice assistant.
- Speak in complete natural sentences like a real human conversation
- Keep responses to 1-3 sentences unless more detail is truly needed
- Never use bullet points, markdown, lists, or symbols
- Be conversational, warm and natural
- Do not use filler phrases like Certainly, Of course, or Absolutely
"""


# ── Natural Voice Synthesizer ──────────────────────────────────────────────

class NaturalElevenLabsSynthesizer(ElevenLabsSynthesizer):
    """
    Full override of get_chunks to:
    1. Force pcm_22050 (free tier compatible)
    2. Inject full voice_settings including style + use_speaker_boost
       which Vocode's default implementation does NOT send
    """

    async def get_chunks(
        self,
        url: str,
        headers: dict,
        body: dict,
        chunk_size: int,
        chunk_queue: asyncio.Queue,
    ):
        # Force free-tier format
        url = url.replace("pcm_44100", "pcm_22050")
        url = url.replace("pcm_24000", "pcm_22050")

        # Override voice_settings with full natural parameters
        # stability (0.5): 0.5 is the ideal balance for natural but consistent speech
        # similarity_boost (0.8): higher = clearer original voice
        # style (0.3): adds subtle expressive style
        body["voice_settings"] = {
            "stability": 0.50,
            "similarity_boost": 0.80,
            "style": 0.35,
            "use_speaker_boost": True,
        }

        # Call parent get_chunks with patched url and body
        await super().get_chunks(url, headers, body, chunk_size, chunk_queue)

    async def create_speech_uncached(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        # Force free tier sample rate
        self.output_format = "pcm_22050"
        self.sample_rate = 22050
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

    # Force 22050Hz globally to match ElevenLabs Free Tier
    # Manual initialization bypasses helper limitations
    microphone_input = MicrophoneInput.from_default_device(sampling_rate=22050)
    speaker_output = BlockingSpeakerOutput.from_default_device(sampling_rate=22050)

    conversation = StreamingConversation(
        output_device=speaker_output,

        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_device(
                microphone_input,
                endpointing_config=PunctuationEndpointingConfig(),
                api_key=settings.deepgram_api_key,
            ),
        ),

        agent=ChatGPTAgent(
            ChatGPTAgentConfig(
                openai_api_key=settings.openai_api_key,
                model_name=settings.openai_model,
                prompt_preamble=SYSTEM_PROMPT,
                generate_responses=True,
            )
        ),

        synthesizer=NaturalElevenLabsSynthesizer(
            ElevenLabsSynthesizerConfig.from_output_device(
                speaker_output,
                api_key=settings.eleven_labs_api_key,
                voice_id=settings.elevenlabs_voice_id,
                model_id="eleven_multilingual_v2",
                stability=0.30,
                similarity_boost=0.75,
                optimize_streaming_latency=1,
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