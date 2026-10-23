"""
Test script for Vocode + Clawdbot integration
Verifies all components before running voice conversation
"""

import asyncio
import site
from pathlib import Path


async def test_clawdbot_connection():
    """Test text connectivity to Clawdbot via CLI"""
    print("🔍 Testing Clawdbot connection...")

    from clawdbot_agent import ClawdbotAgent, ClawdbotAgentConfig

    agent = ClawdbotAgent(ClawdbotAgentConfig(agent_id="main", clawdbot_timeout=30))

    test_messages = [
        "Hello, can you hear me?",
        "What's your name?",
        "What can you help me with?",
    ]

    for i, message in enumerate(test_messages, 1):
        print(f"\n--- Test {i} ---")
        print(f"👤 Human: {message}")
        try:
            parts = []
            async for gr in agent.generate_response(message, conversation_id="test"):
                parts.append(gr.message.text)
                flag = "(interruptible)" if gr.is_interruptible else "(final)"
                print(f"🤖 Clawdbot: {gr.message.text!r} {flag}")
            print(f"✅ Full response: {' '.join(parts)}")
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    return True


def test_tts_models():
    """Test available Coqui TTS models"""
    print("\n🗣️  Testing TTS models...")
    try:
        from TTS.api import TTS
        tts = TTS()
        models = tts.list_models()
        english_models = [m for m in models if "en" in m and ("ljspeech" in m or "vits" in m)][:5]
        print("Available English TTS models:")
        for model in english_models:
            print(f"  📢 {model}")
        return True
    except ImportError:
        print("❌ Coqui TTS not installed. Run: uv pip install TTS")
        return False
    except Exception as e:
        print(f"❌ TTS test failed: {e}")
        return False


def test_whisper_setup():
    """Check Whisper.cpp library and model"""
    print("\n🎤 Testing Whisper.cpp setup...")

    # Find prebuilt libwhisper.so from pywhispercpp
    lib_path = ""
    for sp in site.getsitepackages():
        candidate = Path(sp) / "libwhisper.so"
        if candidate.exists():
            lib_path = str(candidate)
            break

    model_path = Path(__file__).parent / "whisper.cpp" / "models" / "ggml-tiny.en.bin"

    if lib_path:
        print(f"  ✅ Whisper lib:   {lib_path}")
    else:
        print("  ❌ libwhisper.so not found")
        print("     Run: uv pip install pywhispercpp")
        return False

    if model_path.exists():
        print(f"  ✅ Whisper model: {model_path}")
    else:
        print(f"  ❌ Whisper model missing: {model_path}")
        print("     Run: bash whisper.cpp/models/download-ggml-model.sh tiny.en")
        return False

    # Quick functional test
    try:
        from pywhispercpp.model import Model
        m = Model(str(model_path), redirect_whispercpp_logs_to=None)
        sample = Path(__file__).parent / "whisper.cpp" / "samples" / "jfk.wav"
        if sample.exists():
            result = m.transcribe(str(sample))
            text = " ".join(seg.text for seg in result)
            print(f"  ✅ Test transcription: {text[:80]!r}")
        else:
            print("  ✅ Model loaded (no sample wav to transcribe)")
        return True
    except Exception as e:
        print(f"  ❌ Whisper functional test failed: {e}")
        return False


async def main():
    print("🧪 Vocode + Clawdbot Integration Tests")
    print("=" * 50)

    tests_passed = 0
    total_tests = 3

    # Test 1: Clawdbot API connectivity
    if await test_clawdbot_connection():
        tests_passed += 1
        print("\n✅ Clawdbot connection: PASSED")
    else:
        print("\n❌ Clawdbot connection: FAILED")
        print("   Make sure Clawdbot gateway is running")

    # Test 2: Coqui TTS
    if test_tts_models():
        tests_passed += 1
        print("✅ TTS models: PASSED")
    else:
        print("❌ TTS models: FAILED")

    # Test 3: Whisper.cpp setup
    if test_whisper_setup():
        tests_passed += 1
        print("✅ Whisper.cpp setup: PASSED")
    else:
        print("❌ Whisper.cpp setup: FAILED")

    print("\n" + "=" * 50)
    print(f"🏁 Tests completed: {tests_passed}/{total_tests} passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed! Ready for voice conversation.")
        print("   Activate venv: source .venv/bin/activate")
        print("   Run: python3 voice_conversation.py")
    else:
        print("⚠️  Some tests failed. Check errors above.")
        if tests_passed >= 1:
            print("   Text-only mode: python3 text_conversation.py")


if __name__ == "__main__":
    asyncio.run(main())
