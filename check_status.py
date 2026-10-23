"""
Quick status check for Vocode + Clawdbot setup
"""

import subprocess
import sys
from pathlib import Path

def check_python_packages():
    """Check if required packages are installed"""
    print("📦 Checking Python packages...")
    
    required_packages = [
        "vocode", "TTS", "httpx", "asyncio"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing.append(package)
    
    return len(missing) == 0

def check_whisper_setup():
    """Check Whisper.cpp installation (pywhispercpp prebuilt or locally compiled)"""
    print("\n🎤 Checking Whisper.cpp...")

    import site
    # Prefer prebuilt libwhisper.so from pywhispercpp
    lib_path = next(
        (Path(sp) / "libwhisper.so" for sp in site.getsitepackages()
         if (Path(sp) / "libwhisper.so").exists()),
        Path("./whisper.cpp/libwhisper.so"),  # fallback: local compile
    )
    model_path = Path(__file__).parent / "whisper.cpp" / "models" / "ggml-tiny.en.bin"

    if lib_path.exists():
        print(f"  ✅ Library: {lib_path}")
    else:
        print(f"  ❌ Library missing (install pywhispercpp or compile whisper.cpp)")
        return False

    if model_path.exists():
        print(f"  ✅ Model: {model_path}")
    else:
        print(f"  ❌ Model missing: {model_path}")
        print("     Run: bash whisper.cpp/models/download-ggml-model.sh tiny.en")
        return False

    return True

def check_clawdbot():
    """Check if Clawdbot is running via CLI health command"""
    print("\n🤖 Checking Clawdbot...")

    try:
        result = subprocess.run(
            ["clawdbot", "health"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 or "ok" in result.stdout.lower():
            print("  ✅ Clawdbot is running")
            print(f"     {result.stdout.strip().splitlines()[0]}")
            return True
        else:
            print("  ❌ Clawdbot not healthy")
            print(f"     {result.stderr.strip()[:80]}")
            return False
    except FileNotFoundError:
        print("  ❌ clawdbot CLI not found in PATH")
        return False
    except Exception as e:
        print(f"  ❌ Cannot check Clawdbot: {e}")
        print("     Make sure Clawdbot is running: clawdbot gateway start")
        return False

def main():
    """Run all status checks"""
    print("🔍 Vocode + Clawdbot Status Check")
    print("=" * 40)
    
    checks_passed = 0
    total_checks = 3
    
    # Check 1: Python packages
    if check_python_packages():
        checks_passed += 1
    else:
        print("   Run: pip install -r requirements.txt")
    
    # Check 2: Whisper.cpp
    if check_whisper_setup():
        checks_passed += 1
    else:
        print("   Run: ./setup.sh")
    
    # Check 3: Clawdbot
    if check_clawdbot():
        checks_passed += 1
    else:
        print("   Start Clawdbot: clawdbot gateway start")
    
    # Summary
    print("\n" + "=" * 40)
    print(f"Status: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("🎉 All systems ready!")
        print("   Run: python3 voice_conversation.py")
    elif checks_passed >= 1:
        print("⚠️  Partial setup - some components missing")
        if checks_passed >= 2:
            print("   Try: python3 test_voice_integration.py")
    else:
        print("❌ Setup incomplete")
        print("   Run: ./setup.sh")

if __name__ == "__main__":
    main()