#!/usr/bin/env python3
"""
TracyBot Voice Chat Server — optimized
Pipeline: Whisper STT (in-memory) → Clawdbot → Piper TTS (in-memory, sentence-chunked)

Improvements over baseline:
- Whisper model loaded once at startup (~1.7s saved per turn)
- PiperVoice loaded once at startup (~1s saved per turn)
- Response split into sentences; first audio chunk sent as soon as sentence 1 is ready
"""

import asyncio
import base64
import io
import json
import re
import subprocess
import tempfile
import wave
import os
from pathlib import Path

import numpy as np
import whisper
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from piper import PiperVoice

PIPER_MODEL = Path(__file__).parent / "piper_models" / "en_US-libritts-high.onnx"
PIPER_SPEAKER_ID = 0

# Models loaded once at startup
_whisper_model = None
_piper_voice = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _whisper_model, _piper_voice
    print("Loading Whisper tiny model...", flush=True)
    _whisper_model = whisper.load_model("tiny")
    print("Loading Piper TTS model...", flush=True)
    _piper_voice = PiperVoice.load(PIPER_MODEL)
    print("Models ready.", flush=True)
    yield


app = FastAPI(title="TracyBot Voice Chat", lifespan=lifespan)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for chunked TTS."""
    text = text.replace("**", "").replace("*", "").replace("`", "").replace("#", "").strip()
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def audio_chunks_to_wav_bytes(chunks) -> bytes:
    """Convert PiperVoice AudioChunks to WAV bytes."""
    if not chunks:
        return b""
    sample_rate = chunks[0].sample_rate
    sample_width = chunks[0].sample_width
    raw = b"".join(c.audio_int16_bytes for c in chunks)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(raw)
    return buf.getvalue()


async def speech_to_text(wav_path: str) -> str:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: _whisper_model.transcribe(wav_path, language="en")
    )
    return result.get("text", "").strip()


def synthesize_sentence(text: str) -> bytes:
    """Synthesize one sentence to WAV bytes using in-memory Piper."""
    from piper.config import SynthesisConfig
    cfg = SynthesisConfig(speaker_id=PIPER_SPEAKER_ID)
    chunks = list(_piper_voice.synthesize(text, syn_config=cfg))
    return audio_chunks_to_wav_bytes(chunks)


async def send_to_clawdbot(text: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _clawdbot_sync, text)


def _clawdbot_sync(text: str) -> str:
    try:
        result = subprocess.run(
            ["clawdbot", "agent", "--message", text,
             "--session-id", "voice-session", "--json"],
            capture_output=True, text=True, timeout=120
        )
        raw = result.stdout.strip()
        if not raw:
            return "No response received."
        return _extract_text(raw)
    except subprocess.TimeoutExpired:
        return "Sorry, that took too long."
    except Exception as e:
        return f"Error: {e}"


def _extract_text(raw: str) -> str:
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            result = data.get("result")
            if isinstance(result, dict):
                payloads = result.get("payloads", [])
                if payloads:
                    return payloads[0].get("text", "")
            if "payloads" in data and data["payloads"]:
                return data["payloads"][0].get("text", "")
            if "result" in data and isinstance(data["result"], str):
                return data["result"]
            if "text" in data:
                return str(data["text"])
    except (json.JSONDecodeError, TypeError):
        pass
    return str(raw)


@app.get("/")
async def get_main_page():
    return HTMLResponse(content="""<!DOCTYPE html>
<html>
<head>
    <title>TracyBot Voice Chat</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
               max-width: 600px; margin: 0 auto; padding: 20px;
               background: #1a1a2e; color: #eee; min-height: 100vh; }
        h1 { text-align: center; margin: 20px 0; font-size: 1.5em; }
        .chat { border: 1px solid #333; border-radius: 12px; height: 50vh;
                overflow-y: auto; padding: 15px; margin: 15px 0;
                background: #16213e; }
        .msg { margin: 8px 0; padding: 10px 14px; border-radius: 18px;
               max-width: 85%; line-height: 1.4; font-size: 0.95em; }
        .user { background: #0f3460; margin-left: auto; text-align: right; }
        .bot { background: #533483; }
        .thinking { color: #888; font-style: italic; }
        .controls { text-align: center; padding: 20px 0; }
        #recordBtn { width: 80px; height: 80px; border-radius: 50%;
                     border: 3px solid #e94560; background: transparent;
                     color: #e94560; font-size: 32px; cursor: pointer;
                     transition: all 0.3s; }
        #recordBtn:hover { background: #e94560; color: white; }
        #recordBtn.recording { background: #e94560; color: white;
                               animation: pulse 1s infinite; }
        @keyframes pulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.1); } }
        #status { margin-top: 12px; color: #888; font-size: 0.85em; }
    </style>
</head>
<body>
    <h1>TracyBot Voice</h1>
    <div class="chat" id="chat"></div>
    <div class="controls">
        <button id="recordBtn" onclick="toggleRecording()">🎤</button>
        <div id="status">Tap to talk</div>
    </div>
<script>
let ws, mediaRecorder, chunks = [], recording = false;
let audioQueue = [], audioPlaying = false;

function init() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(proto + '//' + location.host + '/ws');
    ws.onopen = () => setStatus('Connected — tap to talk');
    ws.onmessage = async (e) => {
        const data = JSON.parse(e.data);
        if (data.type === 'transcript') {
            document.querySelector('.thinking')?.remove();
            addMsg(data.text, 'user');
        }
        if (data.type === 'response_start') {
            addMsg(data.text, 'bot');
            setStatus('Speaking...');
        }
        if (data.type === 'audio_chunk') {
            enqueueAudio(data.audio);
        }
        if (data.type === 'response_done') {
            // Nothing to do — audio queue drains itself
        }
        if (data.type === 'error') {
            setStatus('Error: ' + data.text);
        }
    };
    ws.onclose = () => setStatus('Disconnected — refresh to reconnect');
}

function enqueueAudio(b64) {
    audioQueue.push(b64);
    if (!audioPlaying) drainAudioQueue();
}

function drainAudioQueue() {
    if (audioQueue.length === 0) {
        audioPlaying = false;
        setStatus('Tap to talk');
        return;
    }
    audioPlaying = true;
    const b64 = audioQueue.shift();
    const audio = new Audio('data:audio/wav;base64,' + b64);
    audio.onended = drainAudioQueue;
    audio.onerror = drainAudioQueue;
    audio.play().catch(drainAudioQueue);
}

async function toggleRecording() {
    if (!recording) {
        const stream = await navigator.mediaDevices.getUserMedia({audio: true});
        mediaRecorder = new MediaRecorder(stream, {mimeType: 'audio/webm'});
        chunks = [];
        mediaRecorder.ondataavailable = e => chunks.push(e.data);
        mediaRecorder.onstop = async () => {
            stream.getTracks().forEach(t => t.stop());
            const blob = new Blob(chunks, {type: 'audio/webm'});
            const reader = new FileReader();
            reader.onload = () => {
                ws.send(JSON.stringify({type: 'audio', data: reader.result.split(',')[1]}));
                addMsg('...', 'thinking');
                setStatus('Thinking...');
            };
            reader.readAsDataURL(blob);
        };
        mediaRecorder.start();
        recording = true;
        document.getElementById('recordBtn').classList.add('recording');
        setStatus('Listening...');
    } else {
        mediaRecorder.stop();
        recording = false;
        document.getElementById('recordBtn').classList.remove('recording');
        setStatus('Processing...');
    }
}

function addMsg(text, cls) {
    const chat = document.getElementById('chat');
    const div = document.createElement('div');
    div.className = 'msg ' + cls;
    div.textContent = text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

function setStatus(s) { document.getElementById('status').textContent = s; }
window.onload = init;
</script>
</body>
</html>""")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_json()
            if raw.get("type") != "audio":
                continue

            # Decode + convert WebM → WAV
            audio_bytes = base64.b64decode(raw["data"])
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as wf:
                wf.write(audio_bytes)
                webm_path = wf.name
            wav_path = webm_path.replace(".webm", ".wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", wav_path],
                capture_output=True, timeout=10
            )
            os.unlink(webm_path)

            if not os.path.exists(wav_path):
                await websocket.send_json({"type": "error", "text": "Audio conversion failed"})
                continue

            # STT — Whisper in-memory
            transcript = await speech_to_text(wav_path)
            os.unlink(wav_path)

            if not transcript:
                await websocket.send_json({"type": "error", "text": "Could not understand audio"})
                continue

            await websocket.send_json({"type": "transcript", "text": transcript})

            # Claude via clawdbot
            response_text = await send_to_clawdbot(transcript)
            print(f"[VOICE] Q: {transcript}")
            print(f"[VOICE] A: {response_text[:200]}")

            # Send full text so UI can display it
            await websocket.send_json({"type": "response_start", "text": response_text})

            # TTS sentence-by-sentence — Piper in-memory
            sentences = split_sentences(response_text)
            loop = asyncio.get_event_loop()
            for sentence in sentences:
                wav_bytes = await loop.run_in_executor(None, synthesize_sentence, sentence)
                audio_b64 = base64.b64encode(wav_bytes).decode()
                await websocket.send_json({"type": "audio_chunk", "audio": audio_b64})

            await websocket.send_json({"type": "response_done"})

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    print("TracyBot Voice Server starting...")
    print(f"  TTS model: {PIPER_MODEL}")
    print(f"  Open http://localhost:8765 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8765)
