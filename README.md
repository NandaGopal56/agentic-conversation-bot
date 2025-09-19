# Agentic Conversation Bot

An agentic, local-first conversation bot that integrates STT (speech-to-text), an agent/orchestrator, and TTS (text-to-speech). It streams LLM responses, sends partial chunks to TTS, and persists conversations. Built with asyncio, LangGraph/LangChain, OpenAI APIs, SpeechRecognition, and PyDub.

## Features
- Streaming conversation with partial TTS output
- Wake word detection (scaffold)
- STT via OpenAI API or Whisper
- Agent orchestrator with in-memory bus
- TTS playback pipeline
- Rotating, colored logging with detailed debug diagnostics
- Persistence of conversation summaries and messages

## Tech Stack
- Python 3.10+
- Asyncio for concurrency
- LangChain/LangGraph for conversation orchestration
- OpenAI for STT/LLM
- SpeechRecognition + PyAudio for microphone input
- PyDub + simpleaudio for audio playback
- dotenv for configuration

## Project Structure
```
agetic_conversation_bot/
├── agents/
│   ├── agent_processor.py        # Agent service lifecycle, subscribes to bus
│   ├── bot.py                    # Conversation workflow runner (LangGraph)
│   ├── graph/
│   │   ├── nodes.py              # Graph nodes
│   │   ├── state.py              # Graph state dataclasses
│   │   └── workflow.py           # Workflow construction
│   ├── logger.py                 # Centralized logging (rotating file + colored console)
│   ├── receiver.py               # Bus message handlers (e.g., on_voice_command)
│   ├── storage.py                # Persistence helpers
│   └── text_writer.py            # Writes AI text to the bus for TTS
│
├── communication_bus/
│   └── inmemory_bus.py           # Simple pub/sub bus used across services
│
├── stt_audio_processor/
│   ├── audio_handler.py          # Microphone handling, ambient calibration, queueing
│   ├── config.py                 # STT and system configuration
│   ├── conversation_manager.py   # Bridges STT with agents (higher-level glue)
│   ├── openai_transcription.py   # OpenAI STT wrapper
│   ├── stt_processor.py          # STT service orchestrator (start/stop, bus IO)
│   ├── voice_assistant.py        # Wake-word and assistant loop (scaffold)
│   └── wake_word_detector.py     # Wake-word detection (scaffold)
│
├── tts_audio_processor/
│   ├── audio_player.py           # Playback worker thread and queue
│   ├── text_reader.py            # Converts text chunks to audio segments
│   ├── tts_processor.py          # TTS service orchestrator (start/stop, bus IO)
│   └── tts_service.py            # TTS provider abstraction (e.g., ElevenLabs/OpenAI)
│
├── conversation_bot/
│   ├── main1.py                  # Example conversation runner
│   └── main2.py                  # Example conversation runner
│
├── main.py                       # Entry point that starts STT, Agent, and TTS services
├── pyproject.toml                # Project metadata and dependencies
├── README.md                     # You are here
└── uv.lock                       # Lockfile for uv/pip
```

## How It Works (End-to-End)
1. Microphone input is handled by `stt_audio_processor/audio_handler.py`.
   - It calibrates ambient noise, then captures audio frames via SpeechRecognition’s background callback and pushes raw audio bytes into an asyncio queue.

2. The STT service (`stt_audio_processor/stt_processor.py`) drains queued audio and sends it to `openai_transcription.py`.
   - `transcribe_audio_by_openai()` converts audio to a WAV buffer and calls OpenAI’s transcription endpoint.
   - Results are published on the in-memory bus (e.g., topic `voice/commands`).

3. The Agent service (`agents/agent_processor.py`) subscribes to bus topics and orchestrates conversation logic.
   - `agents/bot.py` defines `run_conversation()`/`invoke_conversation()` that build and run a LangGraph workflow.
   - As the LLM streams responses, partial chunks are yielded and forwarded to TTS via `agents/text_writer.py`.
   - On completion, final messages are persisted with `agents/storage.py`.

4. The TTS service (`tts_audio_processor/tts_processor.py`) subscribes to the text stream and synthesizes speech using `tts_service.py`.
   - `text_reader.py` turns text into `AudioSegment` blocks.
   - `audio_player.py` plays segments sequentially in a background thread while temporarily muting the microphone to avoid feedback.

5. Everything is launched by `main.py`.
   - It initializes STT, Agent, and TTS services and runs them concurrently with `asyncio.gather()`.

## Logging
- Centralized logging is configured in `agents/logger.py`.
- Features include rotating file logs under `logs/`, colored console output, and optional TRACE-level diagnostics.
- Set environment variable `LOG_LEVEL` to `DEBUG` or `TRACE` for deep debugging.

## Configuration
Environment variables are loaded from `.env` (see `main.py`). Common settings:
- `OPENAI_API_KEY` — required for OpenAI APIs.
- `LOG_LEVEL` — e.g., `INFO`, `DEBUG`, `TRACE`.
- STT config in `stt_audio_processor/config.py` (e.g., sample rate, thresholds, mic).
- TTS provider keys (e.g., ElevenLabs) if used in `tts_audio_processor/tts_service.py`.

Example `.env`:
```
OPENAI_API_KEY=sk-...redacted...
LOG_LEVEL=INFO
```

## Installation
Ensure Python 3.10+ is installed, plus system audio deps:
- macOS: `brew install portaudio ffmpeg`
- Linux: `apt-get install portaudio19-dev ffmpeg` (or distro equivalent)

Install Python deps using uv (recommended) or pip:
```
# Using uv (https://github.com/astral-sh/uv)
uv sync

# Or using pip
python -m venv .venv && source .venv/bin/activate
pip install -r <(uv pip compile pyproject.toml)  # or translate to requirements.txt
```

## Running
1. Create and populate `.env` with your API keys.
2. Start the bot:
```
python -m agetic_conversation_bot.main
```
This launches all three services (STT, Agent, TTS). Speak into your microphone once calibration completes.

For testing conversation workflow only (without full services), you can run the examples in `conversation_bot/`.

## Key Modules
- `stt_audio_processor/audio_handler.py`
  - Singleton that manages mic calibration (`calibrate_microphone()`), `mute()`/`unmute()`, and audio queueing.
  - Background callback `_record_callback()` enqueues raw audio for STT.

- `stt_audio_processor/openai_transcription.py`
  - `transcribe_audio_by_openai()` converts numpy audio to WAV in-memory and calls OpenAI.

- `agents/bot.py`
  - `run_conversation()` streams LLM chunks; `invoke_conversation()` orchestrates end-to-end.

- `tts_audio_processor/audio_player.py`
  - Background thread reads queue and plays audio buffers using simpleaudio, muting/unmuting the mic.

## Troubleshooting
- Microphone not found
  - Ensure your mic is connected and accessible by the OS.
  - On Linux, check ALSA/pulse configs. Set default mic in `stt_audio_processor/config.py`.

- No audio playback
  - Verify `simpleaudio` works and speakers are selected.
  - Ensure TTS provider returns audio and `text_reader.py` pushes segments to the queue.

- Logs not appearing
  - Check the `logs/` directory.
  - Set `LOG_LEVEL=DEBUG` or `TRACE` for more verbosity.

## Roadmap / Ideas
- Add robust wake-word detection and VAD
- Add hotword training and better echo cancellation
- Support multiple TTS providers and voices
- Add REST/gRPC endpoints for remote control
- Persistent vector-based long-term memory

## License
MIT

