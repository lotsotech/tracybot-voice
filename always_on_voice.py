#!/usr/bin/env python3
"""
TracyBot Always-On Voice Assistant
- Serves HTTPS on port 8766 for LAN access
- Browser JS uses Web Speech API for continuous wake-word detection ("hey tracybot")
- On wake: records user utterance, sends WebM audio to server
- Server: ffmpeg converts WebM->WAV, Whisper STT, clawdbot CLI, Piper TTS
- Streamed WAV audio returned to browser and spoken aloud
"""

import asyncio
import base64
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import whisper
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from piper import PiperVoice

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
VENV_BIN = HERE / "voice_venv" / "bin"
PIPER_MODEL = HERE / "piper_models" / "en_US-libritts-high.onnx"
PIPER_SPEAKER_ID = 0
AUDIO_DIR = Path(tempfile.mkdtemp(prefix="tracybot_always_on_"))

# ── In-memory models (loaded once at startup) ─────────────────────────────────
_whisper_model = None
_piper_voice = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _whisper_model, _piper_voice
    print("[VOICE] Loading Whisper base model...", flush=True)
    _whisper_model = whisper.load_model("base")
    print("[VOICE] Loading Piper TTS model...", flush=True)
    _piper_voice = PiperVoice.load(PIPER_MODEL)
    print("[VOICE] Models ready.", flush=True)
    yield

LAN_IP = "192.168.0.112"
PORT = 8767

CERT_FILE = "/tmp/tracybot_cert.pem"
KEY_FILE = "/tmp/tracybot_key.pem"
SPEAKERS_FILE = HERE / "speaker_embeddings.json"
SPEAKER_THRESHOLD = 0.75  # cosine similarity threshold for a confident match

# ── Self-signed TLS cert ───────────────────────────────────────────────────────

def ensure_cert():
    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        return
    print(f"[VOICE] Generating self-signed cert for {LAN_IP}…")
    cmd = (
        f"openssl req -x509 -newkey rsa:2048 -keyout {KEY_FILE} -out {CERT_FILE} "
        f"-days 365 -nodes "
        f"-subj '/CN={LAN_IP}' "
        f"-addext 'subjectAltName=IP:{LAN_IP},IP:100.125.71.73,IP:127.0.0.1,DNS:localhost'"
    )
    ret = os.system(cmd)
    if ret != 0:
        print("[VOICE] WARNING: openssl cert generation failed")
    else:
        print(f"[VOICE] Cert written to {CERT_FILE}")

# ── Speaker ID ────────────────────────────────────────────────────────────────

_encoder = None

def get_encoder():
    global _encoder
    if _encoder is None:
        from resemblyzer import VoiceEncoder
        _encoder = VoiceEncoder()
    return _encoder

def load_speakers() -> dict:
    if SPEAKERS_FILE.exists():
        raw = json.loads(SPEAKERS_FILE.read_text())
        return {name: np.array(emb) for name, emb in raw.items()}
    return {}

def save_speakers(speakers: dict):
    SPEAKERS_FILE.write_text(json.dumps({n: e.tolist() for n, e in speakers.items()}))

def embed_wav(wav_path: str) -> np.ndarray:
    from resemblyzer import preprocess_wav
    wav = preprocess_wav(wav_path)
    return get_encoder().embed_utterance(wav)

def identify_speaker(embedding: np.ndarray) -> str | None:
    speakers = load_speakers()
    if not speakers:
        return None
    best_name, best_sim = None, -1.0
    for name, ref_emb in speakers.items():
        sim = float(np.dot(embedding, ref_emb) / (np.linalg.norm(embedding) * np.linalg.norm(ref_emb)))
        if sim > best_sim:
            best_sim, best_name = sim, name
    return best_name if best_sim >= SPEAKER_THRESHOLD else None

# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(title="TracyBot Always-On Voice", lifespan=lifespan)

# ── Audio pipeline helpers ─────────────────────────────────────────────────────

async def webm_to_wav(webm_bytes: bytes) -> bytes | None:
    """Convert WebM/Opus audio bytes to 16 kHz mono WAV using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as wf:
        wf.write(webm_bytes)
        webm_path = wf.name
    wav_path = webm_path.replace(".webm", ".wav")
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", webm_path,
            "-ar", "16000", "-ac", "1", "-f", "wav", wav_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
        if os.path.exists(wav_path):
            data = Path(wav_path).read_bytes()
            return data
        print(f"[VOICE] ffmpeg failed (rc={proc.returncode}): {stderr.decode(errors='replace')[-300:]}", flush=True)
        return None
    except asyncio.TimeoutError:
        print("[VOICE] ffmpeg timeout", flush=True)
        return None
    finally:
        for p in (webm_path, wav_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


async def speech_to_text(wav_bytes: bytes) -> str:
    """Whisper STT — in-memory model, no subprocess."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        wf.write(wav_bytes)
        wav_path = wf.name
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: _whisper_model.transcribe(wav_path, language="en", fp16=False)
        )
        return result.get("text", "").strip()
    finally:
        try:
            os.unlink(wav_path)
        except FileNotFoundError:
            pass


VOICE_SYSTEM_PROMPT_BASE = (
    "You are TracyBot, a voice assistant. Everything you say is spoken aloud through speakers — "
    "there is no screen. Rules: "
    "1. Plain spoken language only. No markdown, no bullet points, no headers, no emojis, no special characters. "
    "2. When asked to 'say' something, speak those words directly. For example, if asked to 'say hi to Tracy', respond with 'Hi Tracy!' — do not narrate or describe the action. "
    "3. Keep responses to 1-3 sentences unless more is clearly needed. "
    "4. Never use asterisks, pound signs, backticks, or any formatting characters — they sound terrible when read aloud."
)

def _clawdbot_sync(text: str, speaker: str | None = None) -> str:
    """Run claude -p for a clean one-shot voice response."""
    system_prompt = VOICE_SYSTEM_PROMPT_BASE
    if speaker:
        system_prompt += f" The person speaking is {speaker}."
    try:
        result = subprocess.run(
            ["claude", "-p", "--no-session-persistence", "--model", "claude-sonnet-4-6",
             "--system-prompt", system_prompt,
             text],
            capture_output=True, text=True, timeout=120,
        )
        response = result.stdout.strip()
        return response if response else "Sorry, I didn't get a response."
    except subprocess.TimeoutExpired:
        return "Sorry, I took too long to respond."
    except Exception as exc:
        return f"Error: {exc}"


async def query_clawdbot(text: str, speaker: str | None = None) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _clawdbot_sync, text, speaker)


def _clean_for_tts(text: str) -> str:
    return text.replace("**", "").replace("*", "").replace("`", "").replace("#", "").strip()


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def _synthesize_sentence(text: str) -> bytes:
    """Synthesize one sentence → WAV bytes using in-memory Piper."""
    from piper.config import SynthesisConfig
    cfg = SynthesisConfig(speaker_id=PIPER_SPEAKER_ID)
    chunks = list(_piper_voice.synthesize(text, syn_config=cfg))
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


async def text_to_speech(text: str) -> bytes | None:
    """Piper TTS — in-memory model, full response as single WAV."""
    text = _clean_for_tts(str(text))
    if not text:
        return None
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _synthesize_sentence, text)


async def text_to_speech_chunked(text: str, websocket: WebSocket) -> None:
    """Synthesize sentence-by-sentence and send audio_chunk messages as each is ready."""
    text = _clean_for_tts(str(text))
    if not text:
        return
    sentences = _split_sentences(text)
    if not sentences:
        return
    loop = asyncio.get_event_loop()
    for sentence in sentences:
        wav_bytes = await loop.run_in_executor(None, _synthesize_sentence, sentence)
        if wav_bytes:
            await websocket.send_json({
                "type": "audio_chunk",
                "audio": base64.b64encode(wav_bytes).decode(),
            })

# ── HTML page ──────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<meta name="theme-color" content="#0d0d1a">
<title>TracyBot Voice</title>
<style>
  :root {
    --bg: #0d0d1a;
    --surface: #161628;
    --accent: #7b5ea7;
    --accent2: #e94560;
    --text: #e8e8f0;
    --muted: #6b6b8a;
    --wake: #2a6b3a;
    --listen: #6b2a2a;
    --think: #2a4a6b;
    --speak: #5a3a6b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body {
    height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    overflow: hidden;
  }
  body {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0;
    padding: 16px;
  }

  /* ── Orb ── */
  #orb-wrap {
    position: relative;
    width: min(55vw, 200px);
    height: min(55vw, 200px);
    flex-shrink: 0;
  }
  #orb {
    width: 100%;
    height: 100%;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, var(--accent), #1a1a3a);
    box-shadow: 0 0 30px 4px rgba(123,94,167,0.3);
    transition: background 0.4s, box-shadow 0.4s, transform 0.2s;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  #orb.sleeping {
    background: radial-gradient(circle at 35% 35%, #3a3a5a, #111120);
    box-shadow: 0 0 15px 2px rgba(50,50,80,0.3);
  }
  #orb.responsive {
    background: radial-gradient(circle at 35% 35%, #4caf70, #1a3a2a);
    box-shadow: 0 0 25px 5px rgba(76,175,112,0.25);
    animation: pulse-soft 2s ease-in-out infinite;
  }
  @keyframes pulse-soft {
    0%,100% { transform: scale(1); }
    50% { transform: scale(1.03); }
  }
  #orb.wake {
    background: radial-gradient(circle at 35% 35%, #4caf70, var(--wake));
    box-shadow: 0 0 35px 8px rgba(76,175,112,0.4);
  }
  #orb.listening {
    background: radial-gradient(circle at 35% 35%, #e94560, var(--listen));
    box-shadow: 0 0 35px 8px rgba(233,69,96,0.5);
    animation: pulse 1.2s ease-in-out infinite;
  }
  #orb.thinking {
    background: radial-gradient(circle at 35% 35%, #4a9fd4, var(--think));
    box-shadow: 0 0 35px 8px rgba(74,159,212,0.4);
    animation: spin-glow 2s linear infinite;
  }
  #orb.speaking {
    background: radial-gradient(circle at 35% 35%, #b06de0, var(--speak));
    box-shadow: 0 0 35px 8px rgba(176,109,224,0.5);
    animation: pulse 0.8s ease-in-out infinite;
  }
  @keyframes pulse {
    0%,100% { transform: scale(1); }
    50% { transform: scale(1.06); }
  }
  @keyframes spin-glow {
    0% { box-shadow: 0 0 35px 8px rgba(74,159,212,0.4); }
    50% { box-shadow: 0 0 50px 14px rgba(74,159,212,0.6); }
    100% { box-shadow: 0 0 35px 8px rgba(74,159,212,0.4); }
  }

  /* ── Status text ── */
  #status-label {
    margin-top: 18px;
    font-size: clamp(0.85rem, 3.5vw, 1.1rem);
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
    min-height: 1.4em;
    text-align: center;
  }
  #status-label.active { color: var(--text); }

  /* ── Transcript / chat ── */
  #chat {
    width: 100%;
    max-width: 480px;
    max-height: 38vh;
    overflow-y: auto;
    margin-top: 14px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 0 4px;
    scroll-behavior: smooth;
  }
  .bubble {
    padding: 9px 14px;
    border-radius: 16px;
    font-size: clamp(0.8rem, 3.2vw, 0.95rem);
    line-height: 1.45;
    max-width: 88%;
    word-wrap: break-word;
  }
  .bubble.user {
    background: #1e2a4a;
    align-self: flex-end;
    border-bottom-right-radius: 4px;
  }
  .bubble.bot {
    background: #2a1e4a;
    align-self: flex-start;
    border-bottom-left-radius: 4px;
  }
  .bubble.error {
    background: #4a1e1e;
    align-self: center;
    font-size: 0.8rem;
    color: #ff8080;
  }

  /* ── Footer hint ── */
  #hint {
    position: fixed;
    bottom: 10px;
    left: 0; right: 0;
    text-align: center;
    font-size: 0.7rem;
    color: var(--muted);
    opacity: 0.5;
    pointer-events: none;
  }
  #wake-indicator {
    font-size: 0.72rem;
    color: var(--muted);
    text-align: center;
    margin-top: 6px;
    min-height: 1em;
  }
</style>
</head>
<body>

<div id="orb-wrap">
  <div id="orb" class="sleeping" onclick="orbClick()" title="Tap to talk"></div>
</div>

<div id="status-label">Initializing…</div>
<div id="wake-indicator"></div>

<div id="chat"></div>

<div id="hint">Say "hey tracybot" — or tap the orb</div>

<div id="enroll-panel" style="margin-top:16px;text-align:center;display:flex;flex-direction:column;align-items:center;gap:8px;">
  <div style="font-size:0.7rem;color:var(--muted);letter-spacing:0.05em;text-transform:uppercase;">Speaker enrollment</div>
  <div style="display:flex;gap:8px;">
    <button onclick="startEnroll('Greg')" style="padding:6px 14px;border-radius:8px;border:1px solid #444;background:#1a1a3a;color:#ccc;font-size:0.8rem;cursor:pointer;">Enroll Greg</button>
    <button onclick="startEnroll('Tracy')" style="padding:6px 14px;border-radius:8px;border:1px solid #444;background:#1a1a3a;color:#ccc;font-size:0.8rem;cursor:pointer;">Enroll Tracy</button>
  </div>
  <div id="enroll-status" style="font-size:0.75rem;color:var(--muted);min-height:1em;"></div>
</div>

<script>
// ── AudioContext (unlocked on first user gesture) ─────────────────────────────
let audioCtx = null;
function getAudioCtx() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === 'suspended') audioCtx.resume();
  return audioCtx;
}
document.addEventListener('click', () => getAudioCtx(), { once: false });

// ── State machine ──────────────────────────────────────────────────────────────
// States: INIT, SLEEPING, RESPONSIVE, WAKE, LISTENING, THINKING, SPEAKING, ERROR
let state = 'INIT';
let responsiveTimer = null;
let responsiveRecorder = null;
let responsiveChunks = [];
let responsiveStream = null;
let cancelResponsiveSilence = null;
let responsiveAccum = '';
let responsiveAccumTimer = null;

const orb = document.getElementById('orb');
const statusLabel = document.getElementById('status-label');
const wakeInd = document.getElementById('wake-indicator');
const chat = document.getElementById('chat');

function setState(s, detail) {
  state = s;
  orb.className = s.toLowerCase();
  const labels = {
    INIT:       'Initializing…',
    SLEEPING:   'Sleeping — say "hey tracybot"',
    RESPONSIVE: 'Recording reply — tap orb when done',
    WAKE:       'Wake word detected…',
    LISTENING:  'Listening…',
    THINKING:   'Thinking…',
    SPEAKING:   'Speaking…',
    ERROR:      detail || 'Error',
  };
  statusLabel.textContent = labels[s] || s;
  statusLabel.className = (s !== 'SLEEPING' && s !== 'INIT') ? 'active' : '';
}

// ── WebSocket ──────────────────────────────────────────────────────────────────
let ws = null;
let wsReady = false;

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host + '/ws');
  ws.onopen = () => {
    wsReady = true;
    console.log('[WS] connected');
  };
  ws.onclose = () => {
    wsReady = false;
    console.log('[WS] closed, reconnecting in 3s');
    setTimeout(connectWS, 3000);
  };
  ws.onerror = (e) => console.error('[WS] error', e);
  ws.onmessage = handleServerMsg;
}

let audioQueue = [];
let audioPlaying = false;

function handleServerMsg(evt) {
  const msg = JSON.parse(evt.data);
  if (msg.type === 'transcript') {
    addBubble(msg.text, 'user');
    setState('THINKING');
  } else if (msg.type === 'response') {
    addBubble(msg.text, 'bot');
    setState('SPEAKING');
    audioQueue = [];
    audioPlaying = false;
  } else if (msg.type === 'audio_chunk') {
    enqueueAudio(msg.audio);
  } else if (msg.type === 'response_done') {
    // Queue drains itself; go responsive when empty
    if (!audioPlaying && audioQueue.length === 0) goResponsive();
  } else if (msg.type === 'silence') {
    goSleep();
  } else if (msg.type === 'error') {
    addBubble('Error: ' + msg.text, 'error');
    goSleep();
  }
}

function enqueueAudio(b64) {
  audioQueue.push(b64);
  if (!audioPlaying) drainAudioQueue();
}

function drainAudioQueue() {
  if (audioQueue.length === 0) {
    audioPlaying = false;
    goResponsive();
    return;
  }
  audioPlaying = true;
  playAudio(audioQueue.shift(), drainAudioQueue);
}

// ── Silence detection via AnalyserNode ───────────────────────────────────────
// Returns a cancel function. Calls onSilence() when silence detected.
// Waits MIN_SPEECH_MS before silence can trigger, stops at MAX_MS.
const SILENCE_THRESHOLD = 8;    // 0-255 RMS — below this = silence
const SILENCE_DURATION_MS = 1500; // ms of continuous silence before stopping
const MIN_SPEECH_MS = 400;       // don't trigger silence before this
const MAX_DURATION_MS = 30000;   // hard cap

function startSilenceDetector(stream, onSilence) {
  const ctx = getAudioCtx();
  const source = ctx.createMediaStreamSource(stream);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 512;
  source.connect(analyser);
  const buf = new Uint8Array(analyser.frequencyBinCount);

  let silentFor = 0;
  const POLL_MS = 100;
  const startTime = Date.now();
  let fired = false;

  const interval = setInterval(() => {
    if (fired) return;
    analyser.getByteFrequencyData(buf);
    const rms = buf.reduce((s, v) => s + v, 0) / buf.length;
    const elapsed = Date.now() - startTime;

    if (elapsed > MAX_DURATION_MS) {
      fired = true;
      clearInterval(interval);
      source.disconnect();
      onSilence();
      return;
    }
    if (elapsed < MIN_SPEECH_MS) { silentFor = 0; return; }

    if (rms < SILENCE_THRESHOLD) {
      silentFor += POLL_MS;
      if (silentFor >= SILENCE_DURATION_MS) {
        fired = true;
        clearInterval(interval);
        source.disconnect();
        onSilence();
      }
    } else {
      silentFor = 0;
    }
  }, POLL_MS);

  return () => { fired = true; clearInterval(interval); try { source.disconnect(); } catch(_) {} };
}

// ── Continuous Speech Recognition (wake word) ─────────────────────────────────
let recognition = null;
let recognitionRunning = false;
let wakeArmed = true;   // false while we're actively listening/processing

function startContinuousRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    wakeInd.textContent = 'Speech recognition not available in this browser';
    return;
  }
  recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = 'en-US';
  recognition.maxAlternatives = 1;

  recognition.onstart = () => {
    recognitionRunning = true;
    wakeInd.textContent = 'Wake word listener active';
  };
  recognition.onend = () => {
    recognitionRunning = false;
    if (state === 'SLEEPING') {
      setTimeout(() => recognition.start(), 500);
    }
  };
  recognition.onerror = (e) => {
    recognitionRunning = false;
    if (e.error !== 'aborted' && e.error !== 'no-speech') {
      console.warn('[SR] error:', e.error);
    }
    if (state === 'SLEEPING') {
      setTimeout(() => recognition.start(), 1000);
    }
  };
  recognition.onresult = (evt) => {
    if (!wakeArmed || state !== 'SLEEPING') return;
    for (let i = evt.resultIndex; i < evt.results.length; i++) {
      const rawTranscript = evt.results[i][0].transcript.trim();
      const transcript = rawTranscript.toLowerCase();
      if (!transcript) continue;
      wakeInd.textContent = 'Heard: ' + transcript;
      if (isWakePhrase(transcript)) {
        console.log('[WAKE] triggered by:', transcript);
        triggerWake();
        return;
      }
    }
  };

  recognition.start();
}

function isWakePhrase(text) {
  // Match "hey tracybot" and common mis-hearings
  const patterns = [
    /hey\s+tracy\s*bot/,
    /hey\s+tracy/,
    /a\s+tracy\s*bot/,
    /tracybot/,
    /hey\s+gracie\s*bot/,
  ];
  return patterns.some(p => p.test(text));
}

// ── Wake & Record ──────────────────────────────────────────────────────────────
let mediaRecorder = null;
let audioChunks = [];
let micStream = null;
let silenceTimer = null;
let cancelSilenceDetector = null;

async function triggerWake() {
  if (state !== 'SLEEPING' && state !== 'RESPONSIVE') return;
  setState('WAKE');
  wakeArmed = false;

  // Stop the continuous recognition so it doesn't interfere with mic recording
  if (recognitionRunning) {
    recognition.abort();
    recognitionRunning = false;
  }

  setTimeout(() => startListening(), 50);
}

async function startListening() {
  setState('LISTENING');
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 }
    });
  } catch (err) {
    console.error('[MIC] error:', err);
    setState('ERROR', 'Microphone access denied');
    setTimeout(goSleep, 2000);
    return;
  }

  audioChunks = [];

  // Use webm/opus if available, fall back gracefully
  let mimeType = 'audio/webm;codecs=opus';
  if (!MediaRecorder.isTypeSupported(mimeType)) {
    mimeType = 'audio/webm';
    if (!MediaRecorder.isTypeSupported(mimeType)) {
      mimeType = '';
    }
  }

  const options = mimeType ? { mimeType } : {};
  mediaRecorder = new MediaRecorder(micStream, options);
  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) audioChunks.push(e.data);
  };
  mediaRecorder.onstop = sendAudio;
  mediaRecorder.start(200);

  cancelSilenceDetector = startSilenceDetector(micStream, stopListening);
}

function stopListening() {
  if (silenceTimer) { clearTimeout(silenceTimer); silenceTimer = null; }
  if (cancelSilenceDetector) { cancelSilenceDetector(); cancelSilenceDetector = null; }
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  if (micStream) {
    micStream.getTracks().forEach(t => t.stop());
    micStream = null;
  }
  setState('THINKING');
}

async function sendAudio() {
  if (!wsReady) {
    addBubble('Not connected to server', 'error');
    goSleep();
    return;
  }
  const mimeType = audioChunks[0]?.type || 'audio/webm';
  const blob = new Blob(audioChunks, { type: mimeType });
  if (blob.size < 1000) { goSleep(); return; }  // too small = empty recording
  const reader = new FileReader();
  reader.onload = () => {
    const b64 = reader.result.split(',')[1];
    ws.send(JSON.stringify({ type: 'audio', data: b64 }));
  };
  reader.readAsDataURL(blob);
}

// ── Audio playback ─────────────────────────────────────────────────────────────
function playAudio(b64, onended) {
  onended = onended || goResponsive;
  try {
    const ctx = getAudioCtx();
    const raw = atob(b64);
    const buf = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) buf[i] = raw.charCodeAt(i);
    ctx.decodeAudioData(buf.buffer, (decoded) => {
      const src = ctx.createBufferSource();
      src.buffer = decoded;
      src.connect(ctx.destination);
      src.onended = onended;
      src.start(0);
    }, (e) => {
      console.error('[AUDIO] decode error', e);
      onended();
    });
  } catch(e) {
    console.error('[AUDIO] playback failed', e);
    onended();
  }
}

// ── Responsive window — record directly, bypass speech recognizer ─────────────
async function goResponsive() {
  setState('RESPONSIVE');
  wakeArmed = false;
  wakeInd.textContent = '';
  responsiveAccum = '';
  if (responsiveAccumTimer) { clearTimeout(responsiveAccumTimer); responsiveAccumTimer = null; }

  try {
    responsiveStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 }
    });
    responsiveChunks = [];
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus' : 'audio/webm';
    responsiveRecorder = new MediaRecorder(responsiveStream, { mimeType });
    responsiveRecorder.ondataavailable = e => { if (e.data.size > 0) responsiveChunks.push(e.data); };
    responsiveRecorder.onstop = sendResponsiveAudio;
    responsiveRecorder.start(200);
    cancelResponsiveSilence = startSilenceDetector(responsiveStream, commitResponsive);
  } catch(e) {
    console.warn('[RESPONSIVE] mic failed:', e);
    goSleep();
    return;
  }

  // Hard cap at 30s (silence detector handles normal case)
  responsiveTimer = setTimeout(() => { responsiveTimer = null; commitResponsive(); }, 30000);
}

function commitResponsive() {
  if (responsiveTimer) { clearTimeout(responsiveTimer); responsiveTimer = null; }
  if (cancelResponsiveSilence) { cancelResponsiveSilence(); cancelResponsiveSilence = null; }
  if (responsiveRecorder && responsiveRecorder.state !== 'inactive') {
    responsiveRecorder.stop();  // onstop = sendResponsiveAudio
  } else {
    goSleep();
  }
  if (responsiveStream) { responsiveStream.getTracks().forEach(t => t.stop()); responsiveStream = null; }
  setState('THINKING');
}

function stopResponsiveRecorder() {
  if (cancelResponsiveSilence) { cancelResponsiveSilence(); cancelResponsiveSilence = null; }
  if (responsiveRecorder && responsiveRecorder.state !== 'inactive') {
    responsiveRecorder.stop();
  }
  if (responsiveStream) {
    responsiveStream.getTracks().forEach(t => t.stop());
    responsiveStream = null;
  }
  responsiveRecorder = null;
  responsiveChunks = [];
}

function sendResponsiveAudio() {
  if (responsiveStream) { responsiveStream.getTracks().forEach(t => t.stop()); responsiveStream = null; }
  if (!wsReady || responsiveChunks.length === 0) { goSleep(); return; }
  const mimeType = responsiveChunks[0]?.type || 'audio/webm';
  const blob = new Blob(responsiveChunks, { type: mimeType });
  responsiveChunks = [];
  if (blob.size < 1000) { goSleep(); return; }  // too small = empty recording
  const reader = new FileReader();
  reader.onload = () => {
    ws.send(JSON.stringify({ type: 'audio', data: reader.result.split(',')[1] }));
  };
  reader.readAsDataURL(blob);
}

// ── Go back to sleep ──────────────────────────────────────────────────────────
function goSleep() {
  if (responsiveTimer) { clearTimeout(responsiveTimer); responsiveTimer = null; }
  if (responsiveAccumTimer) { clearTimeout(responsiveAccumTimer); responsiveAccumTimer = null; }
  responsiveAccum = '';
  stopResponsiveRecorder();
  setState('SLEEPING');
  wakeArmed = true;
  wakeInd.textContent = '';
  if (!recognitionRunning && recognition) {
    setTimeout(() => { try { recognition.start(); } catch(_) {} }, 300);
  }
}

// ── Manual orb tap ────────────────────────────────────────────────────────────
function orbClick() {
  if (state === 'RESPONSIVE') {
    commitResponsive();
  } else if (state === 'SLEEPING' || state === 'ERROR') {
    triggerWake();
  } else if (state === 'LISTENING') {
    stopListening();
  } else if (state === 'SPEAKING') {
    goSleep();
  }
}

// ── Chat UI ────────────────────────────────────────────────────────────────────
function addBubble(text, cls) {
  const div = document.createElement('div');
  div.className = 'bubble ' + cls;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

// ── Speaker enrollment ────────────────────────────────────────────────────────
async function startEnroll(name) {
  const statusEl = document.getElementById('enroll-status');
  statusEl.textContent = `Recording 8s for ${name}… speak naturally`;
  statusEl.style.color = '#e94560';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/webm';
    const rec = new MediaRecorder(stream, { mimeType });
    const chunks = [];
    rec.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
    rec.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      statusEl.textContent = 'Processing…';
      const blob = new Blob(chunks, { type: mimeType });
      const b64 = await new Promise(res => {
        const r = new FileReader();
        r.onload = () => res(r.result.split(',')[1]);
        r.readAsDataURL(blob);
      });
      try {
        const resp = await fetch('/enroll', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name, audio: b64 }),
        });
        const data = await resp.json();
        if (data.ok) {
          statusEl.textContent = `${name} enrolled (${data.total} speaker${data.total !== 1 ? 's' : ''} saved)`;
          statusEl.style.color = '#4caf70';
        } else {
          statusEl.textContent = `Error: ${data.error}`;
          statusEl.style.color = '#e94560';
        }
      } catch (e) {
        statusEl.textContent = `Upload failed: ${e.message}`;
        statusEl.style.color = '#e94560';
      }
    };
    rec.start(200);
    setTimeout(() => rec.stop(), 8000);
  } catch (e) {
    statusEl.textContent = `Mic error: ${e.message}`;
    statusEl.style.color = '#e94560';
  }
}

// ── Boot ───────────────────────────────────────────────────────────────────────
async function boot() {
  setState('INIT');
  connectWS();

  // Request mic permission early so Chrome doesn't block first wake
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    stream.getTracks().forEach(t => t.stop());
  } catch (e) {
    console.warn('[BOOT] Mic pre-request failed:', e.message);
  }

  setState('SLEEPING');
  startContinuousRecognition();
}

window.addEventListener('load', boot);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML)


@app.post("/enroll")
async def enroll(req: Request):
    request_data = await req.json()
    """Enroll a speaker. Body: {name: str, audio: base64-webm}"""
    name = request_data.get("name", "").strip()
    audio_b64 = request_data.get("audio", "")
    if not name or not audio_b64:
        return JSONResponse({"ok": False, "error": "name and audio required"}, status_code=400)
    try:
        audio_bytes = base64.b64decode(audio_b64)
        wav_bytes = await webm_to_wav(audio_bytes)
        if not wav_bytes:
            return JSONResponse({"ok": False, "error": "audio conversion failed"}, status_code=400)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            wav_path = f.name
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, embed_wav, wav_path)
        os.unlink(wav_path)
        speakers = load_speakers()
        speakers[name] = embedding
        save_speakers(speakers)
        print(f"[SPEAKER] Enrolled: {name}")
        return JSONResponse({"ok": True, "speaker": name, "total": len(speakers)})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/speakers")
async def list_speakers():
    return JSONResponse({"speakers": list(load_speakers().keys())})


# ── WebSocket endpoint ─────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[VOICE] WebSocket client connected")
    try:
        while True:
            raw = await websocket.receive_json()

            # Text shortcut from RESPONSIVE state (Web Speech API transcript)
            if raw.get("type") == "text":
                transcript = raw.get("data", "").strip()
                if not transcript:
                    continue
                print(f"[VOICE] Text input: {transcript}")
                response_text = await query_clawdbot(transcript)
                print(f"[VOICE] Response: {response_text[:200]}")
                await websocket.send_json({"type": "response", "text": response_text})
                await text_to_speech_chunked(response_text, websocket)
                await websocket.send_json({"type": "response_done"})
                continue

            if raw.get("type") != "audio":
                continue

            # 1. Decode base64 audio
            try:
                audio_bytes = base64.b64decode(raw["data"])
            except Exception:
                await websocket.send_json({"type": "error", "text": "Bad audio encoding"})
                continue

            # 2. WebM → WAV
            wav_bytes = await webm_to_wav(audio_bytes)
            if not wav_bytes:
                await websocket.send_json({"type": "silence"})
                continue

            # 3. STT
            print(f"[VOICE] Running STT on {len(wav_bytes)} bytes WAV…")
            transcript = await speech_to_text(wav_bytes)
            if not transcript:
                await websocket.send_json({"type": "silence"})
                continue

            # 3b. Speaker ID
            speaker = None
            if load_speakers():
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(wav_bytes)
                    tmp_wav = f.name
                loop = asyncio.get_event_loop()
                try:
                    emb = await loop.run_in_executor(None, embed_wav, tmp_wav)
                    speaker = identify_speaker(emb)
                except Exception as e:
                    print(f"[SPEAKER] ID failed: {e}")
                finally:
                    os.unlink(tmp_wav)

            print(f"[VOICE] Transcript: {transcript} (speaker: {speaker or 'unknown'})")
            await websocket.send_json({"type": "transcript", "text": transcript, "speaker": speaker})

            # 4. Claude
            print(f"[VOICE] Querying Claude…")
            response_text = await query_clawdbot(transcript, speaker=speaker)
            print(f"[VOICE] Response ({len(response_text)} chars): {response_text[:200]}")

            # 5. TTS — sentence-by-sentence, send chunks as each is ready
            await websocket.send_json({"type": "response", "text": response_text})
            await text_to_speech_chunked(response_text, websocket)
            await websocket.send_json({"type": "response_done"})

    except WebSocketDisconnect:
        print("[VOICE] WebSocket disconnected")
    except Exception as exc:
        print(f"[VOICE] WebSocket error: {exc}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    ensure_cert()

    print()
    print("=" * 60)
    print("  TracyBot Always-On Voice Assistant")
    print("=" * 60)
    print(f"  TTS model : {PIPER_MODEL}")
    print(f"  Audio dir : {AUDIO_DIR}")
    print(f"  LAN URL   : https://{LAN_IP}:{PORT}")
    print(f"  Local URL : https://localhost:{PORT}")
    print()
    print("  Steps on Android Chrome:")
    print(f"  1. Open https://{LAN_IP}:{PORT}")
    print("  2. Tap 'Advanced' → 'Proceed to ...' to accept cert")
    print("  3. Allow microphone when prompted")
    print("  4. Say 'hey tracybot' or tap the orb")
    print("=" * 60)
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        ssl_keyfile=KEY_FILE,
        ssl_certfile=CERT_FILE,
        log_level="info",
    )
