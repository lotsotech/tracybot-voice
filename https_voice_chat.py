#!/usr/bin/env python3
"""
HTTPS Voice Chat Interface
Self-signed certificate for microphone access
"""

import ssl
import tempfile
import os
from pathlib import Path

# Generate self-signed certificate
def create_self_signed_cert():
    cert_file = "/tmp/cert.pem"
    key_file = "/tmp/key.pem"
    
    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        os.system(f"""
openssl req -x509 -newkey rsa:4096 -keyout {key_file} -out {cert_file} -days 365 -nodes \
-subj '/CN=100.125.71.73' \
-addext 'subjectAltName=IP:100.125.71.73,IP:127.0.0.1,DNS:localhost'
        """)
    
    return cert_file, key_file

# Import the existing voice chat app
import sys
sys.path.append('/home/twindekn/clawd/vocode_clawdbot')
from simple_voice_chat import app

if __name__ == "__main__":
    import uvicorn
    
    # Create self-signed certificate
    cert_file, key_file = create_self_signed_cert()
    
    print("🔒 Starting TracyBot Voice Chat Server with HTTPS...")
    print("📱 Open https://100.125.71.73:8000 in your browser")
    print("⚠️  You'll need to accept the self-signed certificate warning")
    print("🎤 Then microphone access should work!")
    
    # Run with SSL
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        ssl_keyfile=key_file,
        ssl_certfile=cert_file
    )