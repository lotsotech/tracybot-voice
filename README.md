# Vocode + Clawdbot Voice Integration 🎤🤖

Complete local voice pipeline connecting Vocode's voice capabilities to your Clawdbot instance. **100% free** - no API costs!

## Features ✨

- **🎤 Speech-to-Text**: Whisper.cpp (local, free)
- **🗣️ Text-to-Speech**: Coqui TTS (local, free)  
- **🤖 AI Brain**: Your existing Clawdbot instance
- **📞 Future**: Phone call capability ready
- **💰 Cost**: $0 - completely local

## Quick Start 🚀

### 1. Prerequisites

- **Clawdbot running** on `http://localhost:18789`
- **Python 3.8+** installed
- **Microphone and speakers** connected
- **Git** for cloning repositories

### 2. Setup

```bash
# Make setup script executable and run it
chmod +x setup.sh
./setup.sh
```

This will:
- Install Python packages (vocode, TTS, etc.)
- Clone and compile Whisper.cpp
- Download speech recognition models
- Test TTS models

### 3. Test Integration

```bash
# Test API connectivity and voice components
python3 test_voice_integration.py

# Test text-only chat (no voice)
python3 text_conversation.py
```

### 4. Start Voice Conversation

```bash
# Full voice pipeline
python3 voice_conversation.py
```

## How It Works 🔧

```
[Your Voice] → [Whisper.cpp STT] → [ClawdbotAgent] → [Clawdbot API] 
                                                           ↓
[Speakers] ← [Coqui TTS] ← [Text Response] ← [HTTP Response]
```

1. **Speak** into microphone
2. **Whisper.cpp** converts speech to text (local)
3. **ClawdbotAgent** sends text to your Clawdbot instance
4. **Clawdbot** processes and responds
5. **Coqui TTS** converts response to speech (local)
6. **Speakers** play the voice response

## File Structure 📁

```
vocode_clawdbot/
├── setup.sh                    # Installation script
├── requirements.txt             # Python dependencies
├── clawdbot_agent.py           # Bridge between Vocode and Clawdbot
├── voice_conversation.py       # Main voice chat interface  
├── text_conversation.py        # Text-only testing
├── test_voice_integration.py   # Integration tests
└── whisper.cpp/                # Created during setup
    ├── libwhisper.so           # Speech recognition library
    └── models/                 # Downloaded speech models
```

## Configuration ⚙️

### ClawdbotAgent Settings

Edit `voice_conversation.py` to customize:

```python
# Clawdbot connection
CLAWDBOT_URL = "http://localhost:18789"  # Your Clawdbot URL

# Voice models
TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC_ph"  # Fast, good quality

# Session management  
session_key = "voice-session-123"  # Optional session tracking
```

### Voice Quality Options

**Faster (recommended for testing):**
```python
TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC_ph"
```

**Higher Quality (slower):**
```python  
TTS_MODEL = "tts_models/en/ljspeech/vits"
```

## Troubleshooting 🛠️

### "Clawdbot connection failed"
- ✅ Make sure Clawdbot is running: `clawdbot status`
- ✅ Check URL: `curl http://localhost:18789/api/health`

### "Audio setup failed" 
- ✅ Check microphone: `arecord -l`
- ✅ Check speakers: `aplay -l`  
- ✅ Install audio tools: `sudo apt-get install alsa-utils`

### "Whisper library not found"
- ✅ Run setup script: `./setup.sh`
- ✅ Compile manually: `cd whisper.cpp && make`

### "TTS model failed"
- ✅ Install TTS: `pip install TTS`
- ✅ Test models: `tts --list_models`

## Advanced Usage 🔥

### Custom Session Management

```python
agent = ClawdbotAgent(
    ClawdbotAgentConfig(
        clawdbot_url="http://localhost:18789",
        session_key="voice-user-greg",  # Persistent session
        timeout=30
    )
)
```

### Phone Integration (Future)

The setup is ready for phone calls via Twilio/telephony providers:

```python
from vocode.streaming.telephony.server.base import TelephonyServer

server = TelephonyServer(
    agent=ClawdbotAgent(...),
    # Add phone configuration
)
```

## Performance Notes 📊

### Resource Usage (Local)
- **Whisper.cpp (tiny model)**: ~100MB RAM
- **Coqui TTS**: ~500MB RAM first load, ~50MB after
- **Total**: ~1GB RAM for complete voice pipeline

### Latency
- **Speech recognition**: ~200-500ms
- **Clawdbot processing**: Variable (depends on your setup)
- **Voice synthesis**: ~300-800ms
- **Total round-trip**: ~1-3 seconds

## Next Steps 🎯

1. **Test basic voice chat** - verify everything works
2. **Customize voice models** - try different TTS voices
3. **Add phone capability** - integrate telephony services
4. **Deploy remotely** - host on server for phone access

## Support 💬

Having issues? Check:
- `test_voice_integration.py` output for specific errors
- Clawdbot logs: `clawdbot logs`  
- Audio device setup with `alsa-utils`

Enjoy talking to your Clawdbot! 🎉