# 🎙️ Voice AI Assistant — Vocode Edition

Real-time voice conversation powered by **Vocode** framework.
Vocode handles all the hard parts: audio I/O, streaming pipeline, barge-in, turn detection.

## Stack

| Layer     | Provider                            |
| --------- | ----------------------------------- |
| Framework | Vocode `StreamingConversation`      |
| STT       | Deepgram `nova-2`                   |
| LLM       | OpenAI `gpt-4o-mini`                |
| TTS       | ElevenLabs `eleven_multilingual_v2` |

---

## Setup

### 1. Install PortAudio (required for audio I/O)

```bash
# Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-dev

# macOS
brew install portaudio
```

### 2. Create & activate virtualenv

```bash
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

# Vocode needs the [io] extra for microphone/speaker support
pip install 'vocode[io]'
```

### 4. Configure API keys

```bash
cp .env.example .env
# Edit .env with your keys
```

### 5. Run

```bash
python main.py
```

---

## How Vocode Works

```
Microphone
    │
    ▼  [PCM audio chunks]
DeepgramTranscriber (WebSocket)
    │  partial + final transcripts
    ▼
ChatGPTAgent (streaming tokens)
    │  sentence-level responses
    ▼
ElevenLabsSynthesizer (experimental_streaming=True)
    │  audio chunks streamed as they're generated
    ▼
Speaker (blocking thread for smooth playback)
```

Vocode's `StreamingConversation` manages:

- **Barge-in**: user can interrupt AI mid-sentence
- **Turn detection**: `PunctuationEndpointingConfig` detects end of speech
- **Filler audio**: optional "thinking" sounds while LLM generates
- **Interruptible events**: all pipeline stages are cancellable

---

## Troubleshooting

**`No module named 'vocode'`**

```bash
pip install 'vocode[io]'
```

**Microphone not detected**

```bash
python3 -c "import pyaudio; p=pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]"
```

Then set `use_default_devices=False` in `main.py` to manually select your device.

**Echo/feedback** — Use headphones. Vocode doesn't have built-in echo cancellation for system audio.
