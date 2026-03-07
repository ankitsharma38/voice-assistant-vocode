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

Example:
"Hello and welcome to India! I'm your virtual tourist guide and I'm here to help you travel safely and enjoy your visit."

2. FIRST QUESTIONS:
Ask their name and where they are visiting from.

Example:
"May I know your name and which country you're visiting from?"

3. ARRIVAL CHECK:
Ask if they have already arrived in India or are planning their trip.

Example:
"Have you already arrived in India, or are you planning your visit?"

4. TRAVEL GUIDANCE:
Help them with:
- Airport guidance
- Transportation like taxis, metro, or trains
- Currency exchange
- SIM cards
- Hotel check-in
- Safety tips
- Tourist places to visit
- Local food recommendations

5. CITY GUIDANCE:
Ask which city they are currently in or planning to visit.

Example:
"Which city in India are you currently visiting?"

6. TOURIST HELP:
Provide simple guidance about places, travel options, and local tips.

7. EMERGENCY SUPPORT:
If the traveler needs urgent help, calmly guide them to emergency services or provide the support contact.

Example:
"If you need immediate help, you can call the police at 100 or ambulance services at 108. If you need local travel assistance, you can also contact Ankit Sharma at 8298197805."

8. FRIENDLY TONE:
Always sound welcoming and helpful, like a friendly local guide assisting a visitor.
"""

INITIAL_MESSAGE = "Hello and welcome to India! I'm your virtual tourist guide, and I'm here to help make your trip smooth and enjoyable. May I know your name and which country you're visiting from?"

# ── Natural Voice Synthesizer ──────────────────────────────────────────────

class NaturalElevenLabsSynthesizer(ElevenLabsSynthesizer):
    """
    Overrides get_chunks and create_speech_uncached to:
    1. Force pcm_22050 (free tier compatible — pcm_44100 requires Pro)
    2. Inject full voice_settings including style + use_speaker_boost
    3. Use 22050Hz sample rate end-to-end
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
        body["voice_settings"] = {
            "stability": 0.50,
            "similarity_boost": 0.80,
            "style": 0.50,  # Increased for more expressive "Indian" warmth
            "use_speaker_boost": True,
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

    async def get_phrase_filler_audios(self) -> list:
        # """Synthesizes standard filler phrases (Hmm, Let me see...) on startup"""
        # from vocode.streaming.synthesizer.base_synthesizer import FILLER_PHRASES, FillerAudio
        
        # filler_audios = []
        # # Generate first 3 (Hmm, Uh..., Let me see)
        # for phrase in FILLER_PHRASES[:3]:
        #     res = await self.create_speech_uncached(phrase, 16384)
        #     audio_data = b""
        #     async for chunk_result in res.chunk_generator:
        #         audio_data += chunk_result.chunk
            
        #     filler_audios.append(
        #         FillerAudio(
        #             message=phrase,
        #             audio_data=audio_data,
        #             synthesizer_config=self.synthesizer_config,
        #             is_interruptible=True,
        #             seconds_per_chunk=1
        #         )
        #     )
        return filler_audios


# ── Main ───────────────────────────────────────────────────────────────────

async def main():
    settings = Settings()

    print("\n" + "═" * 60)
    print("  🎙️  Voice AI Assistant  (Powered by Eco Tech)")
    print("  STT: Deepgram  |  LLM: GPT  |  TTS: ElevenLabs")
    print("═" * 60)
    print("  Speak naturally. Press Ctrl+C to exit.\n")

    # Use 22050Hz end-to-end — mic, speaker and ElevenLabs all match
    # This was your key fix — bypasses the helper that defaults to 44100Hz
    microphone_input  = MicrophoneInput.from_default_device(sampling_rate=22050)
    speaker_output    = BlockingSpeakerOutput.from_default_device(sampling_rate=22050)

    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_device(
                microphone_input,
                endpointing_config=PunctuationEndpointingConfig(
                    time_cutoff_seconds=0.8,  # SNAPPY: stops fast after you stop speaking
                ),
                api_key=settings.deepgram_api_key,
            ),
        ),
        agent=ChatGPTAgent(
            ChatGPTAgentConfig(
                openai_api_key=settings.openai_api_key,
                model_name=settings.openai_model,
                prompt_preamble=SYSTEM_PROMPT,
                initial_message=BaseMessage(text=INITIAL_MESSAGE),
                generate_responses=True,
                send_filler_audio=False
            )
        ),
        synthesizer=NaturalElevenLabsSynthesizer(
            ElevenLabsSynthesizerConfig.from_output_device(
                speaker_output,
                api_key=settings.eleven_labs_api_key,
                voice_id=settings.elevenlabs_voice_id,
                model_id="eleven_multilingual_v2",
                stability=0.35,
                similarity_boost=0.75,
                optimize_streaming_latency=3,
                experimental_streaming=True,
            ),
        ),
        # ── Performance Tuning ─────────────────────────────────────────────
        speed_coefficient=1.0,      # Fixed speed (prevents "rushing")
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