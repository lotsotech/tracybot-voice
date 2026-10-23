#!/bin/bash
# Vocode + Clawdbot Voice Integration Setup Script
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Setting up Vocode + Clawdbot Voice Pipeline..."

# Create virtualenv
echo "🐍 Creating Python virtualenv..."
uv venv
source .venv/bin/activate

# Install Python packages
echo "📦 Installing Python packages..."
# pywhispercpp provides prebuilt libwhisper.so — no compiler needed
uv pip install vocode TTS httpx numpy sounddevice soundfile pywhispercpp pydantic

# Download Whisper.cpp model (no compilation needed)
echo "🎤 Downloading Whisper model..."
if [ ! -d "whisper.cpp" ]; then
    git clone --depth=1 https://github.com/ggerganov/whisper.cpp
fi

if [ ! -f "whisper.cpp/models/ggml-tiny.en.bin" ]; then
    bash whisper.cpp/models/download-ggml-model.sh tiny.en
else
    echo "✅ Whisper model already downloaded"
fi

# Verify libwhisper.so is available from pywhispercpp
echo "🔍 Verifying libwhisper.so..."
python3 -c "
import site, pathlib
for sp in site.getsitepackages():
    p = pathlib.Path(sp) / 'libwhisper.so'
    if p.exists():
        print(f'✅ Found: {p}')
        break
else:
    print('❌ libwhisper.so not found — pywhispercpp may not have installed correctly')
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  python3 test_voice_integration.py   # run integration tests"
echo "  python3 text_conversation.py        # text-only test"
echo "  python3 voice_conversation.py       # full voice pipeline"
