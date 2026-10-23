"""
Vocode + Clawdbot Voice Conversation
Complete local voice pipeline with free services

STT: Whisper.cpp (via pywhispercpp prebuilt binaries)
TTS: Piper TTS (primary — fast, ONNX, no PyTorch)
     Coqui TTS  (fallback if Piper model not found)
AI:  Clawdbot (via CLI: clawdbot agent --agent main --message "..." --json)
"""

import asyncio
import signal
import site
import os
from pathlib import Path

from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.logging import configure_pretty_logging
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.models.message import BaseMessage

# Local STT — Whisper.cpp via pywhispercpp (prebuilt libwhisper.so)
from vocode.streaming.transcriber.whisper_cpp_transcriber import WhisperCPPTranscriber
from vocode.streaming.models.transcriber import WhisperCPPTranscriberConfig

# Our Clawdbot integration
from clawdbot_agent import ClawdbotAgent, ClawdbotAgentConfig

configure_pretty_logging()


def _find_libwhisper() -> str:
    """Locate libwhisper.so — prebuilt by pywhispercpp or locally compiled"""
    for sp in site.getsitepackages():
        candidate = Path(sp) / "libwhisper.so"
        if candidate.exists():
            return str(candidate)
    local = Path(__file__).parent / "whisper.cpp" / "libwhisper.so"
    return str(local) if local.exists() else ""


def _get_synthesizer(speaker_output):
    """Return the best available local TTS synthesizer"""
    piper_model = Path(__file__).parent / "piper_models" / "en_US-amy-medium.onnx"

    if piper_model.exists():
        print(f"🗣️  Using Piper TTS: {piper_model.name}")
        from piper_synthesizer import PiperTTSSynthesizer, PiperTTSSynthesizerConfig
        return PiperTTSSynthesizer(
            PiperTTSSynthesizerConfig.from_output_device(
                speaker_output,
                model_path=str(piper_model),
            )
        )

    print("🗣️  Piper model not found, trying Coqui TTS...")
    try:
        from vocode.streaming.synthesizer.coqui_tts_synthesizer import CoquiTTSSynthesizer
        from vocode.streaming.models.synthesizer import CoquiTTSSynthesizerConfig
        return CoquiTTSSynthesizer(
            CoquiTTSSynthesizerConfig.from_output_device(
                speaker_output,
                tts_kwargs={"model_name": "tts_models/en/ljspeech/tacotron2-DDC_ph"},
            )
        )
    except Exception as e:
        raise RuntimeError(
            f"No TTS engine available.\n"
            f"Piper model missing: {piper_model}\n"
            f"Coqui TTS error: {e}\n\n"
            f"Fix: python3 -c \"\n"
            f"from piper.download_voices import download_voice\n"
            f"import pathlib; download_voice('en_US-amy-medium', pathlib.Path('piper_models'))\n"
            f"\""
        )


class VoiceSettings:
    """Configuration for the voice conversation"""

    CLAWDBOT_AGENT = "main"
    WHISPER_LIB_PATH = _find_libwhisper()
    WHISPER_MODEL_PATH = str(
        Path(__file__).parent / "whisper.cpp" / "models" / "ggml-tiny.en.bin"
    )

    def validate_paths(self) -> bool:
        ok = True
        if not self.WHISPER_LIB_PATH:
            print("❌ libwhisper.so not found. Run: uv pip install pywhispercpp")
            ok = False
        else:
            print(f"✅ Whisper lib:   {self.WHISPER_LIB_PATH}")

        if not Path(self.WHISPER_MODEL_PATH).exists():
            print(f"❌ Whisper model not found: {self.WHISPER_MODEL_PATH}")
            print("   Run: bash whisper.cpp/models/download-ggml-model.sh tiny.en")
            ok = False
        else:
            print(f"✅ Whisper model: {self.WHISPER_MODEL_PATH}")

        return ok


async def main():
    print("🎤 Starting Vocode + Clawdbot Voice Pipeline...")

    settings = VoiceSettings()
    if not settings.validate_paths():
        return

    print("🔊 Setting up audio devices...")
    try:
        microphone_input, speaker_output = create_streaming_microphone_input_and_speaker_output(
            use_default_devices=True,
        )
        print("✅ Audio devices ready")
    except Exception as e:
        print(f"❌ Audio setup failed: {e}")
        print("Make sure you have a microphone and speakers connected")
        return

    print("🎤 Configuring speech recognition (Whisper.cpp)...")
    transcriber = WhisperCPPTranscriber(
        WhisperCPPTranscriberConfig.from_input_device(
            microphone_input,
            libname=settings.WHISPER_LIB_PATH,
            fname_model=settings.WHISPER_MODEL_PATH,
        )
    )

    print(f"🤖 Connecting to Clawdbot agent '{settings.CLAWDBOT_AGENT}'...")
    agent = ClawdbotAgent(
        ClawdbotAgentConfig(
            agent_id=settings.CLAWDBOT_AGENT,
            initial_message=BaseMessage(text="Hello! I'm ready for voice chat."),
        )
    )

    try:
        synthesizer = _get_synthesizer(speaker_output)
        print("✅ Voice synthesis ready")
    except RuntimeError as e:
        print(f"❌ TTS setup failed:\n{e}")
        return

    print("🚀 Creating voice conversation...")
    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=transcriber,
        agent=agent,
        synthesizer=synthesizer,
    )

    print("🎉 Voice conversation started!")
    print("🎤 Start speaking! Press Ctrl+C to end\n")

    def signal_handler(sig, frame):
        print("\n👋 Ending conversation...")
        asyncio.create_task(conversation.terminate())

    signal.signal(signal.SIGINT, signal_handler)

    await conversation.start()

    while conversation.is_active():
        try:
            chunk = await microphone_input.get_audio()
            conversation.receive_audio(chunk)
        except Exception as e:
            print(f"Audio processing error: {e}")
            break

    print("👋 Conversation ended")


if __name__ == "__main__":
    asyncio.run(main())
