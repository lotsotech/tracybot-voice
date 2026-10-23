"""
Piper TTS Synthesizer for Vocode
Uses piper-tts (local, free, no PyTorch needed — runs on ONNX Runtime)

Usage:
    from piper_synthesizer import PiperTTSSynthesizer, PiperTTSSynthesizerConfig

    synthesizer = PiperTTSSynthesizer(
        PiperTTSSynthesizerConfig.from_output_device(
            speaker_output,
            model_path="piper_models/en_US-amy-medium.onnx",
        )
    )

Download a voice model first:
    python3 -c "
    from piper.download_voices import download_voice
    import pathlib
    download_voice('en_US-amy-medium', pathlib.Path('piper_models'))
    "
"""

import asyncio
import io
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import SynthesizerConfig, SynthesizerType
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer, SynthesisResult


class PiperTTSSynthesizerConfig(SynthesizerConfig, type="synthesizer_piper_tts"):  # type: ignore
    """Config for Piper TTS synthesizer"""
    model_path: str  # path to .onnx model file
    config_path: Optional[str] = None  # optional: path to model.onnx.json


class PiperTTSSynthesizer(BaseSynthesizer[PiperTTSSynthesizerConfig]):
    """
    Local TTS using piper-tts (ONNX Runtime — no PyTorch required).
    Fast, high quality, fully offline.
    """

    def __init__(self, synthesizer_config: PiperTTSSynthesizerConfig):
        super().__init__(synthesizer_config)
        from piper.voice import PiperVoice
        self.voice = PiperVoice.load(
            model_path=synthesizer_config.model_path,
            config_path=synthesizer_config.config_path,
            use_cuda=False,
        )
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _synthesize_wav_bytes(self, text: str) -> bytes:
        """Synthesize text to WAV bytes (runs in thread pool)"""
        buf = io.BytesIO()
        wav_out = wave.open(buf, "wb")
        self.voice.synthesize_wav(text, wav_out)
        wav_out.close()
        buf.seek(0)
        return buf.read()

    async def create_speech_uncached(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        wav_bytes = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._synthesize_wav_bytes,
            message.text,
        )
        return self.create_synthesis_result_from_wav(
            synthesizer_config=self.synthesizer_config,
            file=io.BytesIO(wav_bytes),
            message=message,
            chunk_size=chunk_size,
        )
