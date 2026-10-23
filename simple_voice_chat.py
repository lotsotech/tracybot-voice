#!/usr/bin/env python3
"""
Simple Voice Chat Interface
Uses Clawdbot's existing TTS instead of local TTS libraries
"""

import asyncio
import json
import subprocess
import tempfile
import os
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import httpx

app = FastAPI(title="TracyBot Voice Chat")

# Clawdbot API configuration
CLAWDBOT_API = "http://localhost:3000"  # Adjust as needed

class VoiceChat:
    def __init__(self):
        self.client = httpx.AsyncClient()
    
    async def speech_to_text(self, audio_file: bytes) -> str:
        """Convert speech to text using Whisper (system install or API)"""
        # For now, return placeholder - we'll implement Whisper separately
        # or use a speech-to-text service
        return "Hello, this is placeholder text from voice input"
    
    async def send_to_clawdbot(self, text: str) -> str:
        """Send text to Clawdbot and get response"""
        try:
            response = await self.client.post(
                f"{CLAWDBOT_API}/chat",
                json={"message": text},
                timeout=30.0
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "Sorry, I didn't understand that.")
            else:
                return "Sorry, I'm having trouble connecting right now."
        except Exception as e:
            print(f"Error connecting to Clawdbot: {e}")
            return "Sorry, I'm having technical difficulties."
    
    async def text_to_speech_clawdbot(self, text: str) -> str:
        """Use Clawdbot's TTS tool to convert text to speech"""
        try:
            # Use Clawdbot's tts tool via subprocess
            result = subprocess.run([
                "clawdbot", "tts", text
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Extract the MEDIA: path from output
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.startswith('MEDIA:'):
                        return line.replace('MEDIA:', '').strip()
            
            return None
        except Exception as e:
            print(f"Error with Clawdbot TTS: {e}")
            return None

voice_chat = VoiceChat()

@app.get("/")
async def get_main_page():
    """Serve the main voice chat interface"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>TracyBot Voice Chat</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .chat-container { border: 1px solid #ddd; height: 400px; overflow-y: scroll; padding: 10px; margin: 10px 0; }
        .voice-controls { text-align: center; margin: 20px 0; }
        .btn { padding: 10px 20px; margin: 5px; font-size: 16px; cursor: pointer; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .recording { background-color: #ff4444; color: white; }
        .idle { background-color: #4444ff; color: white; }
        .message { margin: 10px 0; padding: 8px; border-radius: 5px; }
        .user { background-color: #e3f2fd; text-align: right; }
        .bot { background-color: #f3e5f5; }
    </style>
</head>
<body>
    <h1>🎤 TracyBot Voice Chat</h1>
    <div class="chat-container" id="chatContainer"></div>
    
    <div class="voice-controls">
        <button id="recordBtn" class="btn idle" onclick="toggleRecording()">
            🎤 Click to Talk
        </button>
        <br><br>
        <div id="status">Ready to chat!</div>
    </div>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        let ws;

        // Initialize WebSocket connection
        function initWebSocket() {
            ws = new WebSocket(`ws://localhost:8000/ws`);
            
            ws.onopen = function() {
                updateStatus("Connected - ready to chat!");
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'response') {
                    addMessage(data.text, 'bot');
                    if (data.audio_url) {
                        playAudio(data.audio_url);
                    }
                }
            };
            
            ws.onclose = function() {
                updateStatus("Disconnected - refresh to reconnect");
            };
        }

        async function toggleRecording() {
            if (!isRecording) {
                await startRecording();
            } else {
                await stopRecording();
            }
        }

        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    await sendAudio(audioBlob);
                };

                mediaRecorder.start();
                isRecording = true;
                
                document.getElementById('recordBtn').textContent = '🛑 Stop Recording';
                document.getElementById('recordBtn').className = 'btn recording';
                updateStatus("Recording... speak now!");

            } catch (err) {
                console.error('Error accessing microphone:', err);
                updateStatus("Error: Could not access microphone");
            }
        }

        async function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                isRecording = false;
                
                document.getElementById('recordBtn').textContent = '🎤 Click to Talk';
                document.getElementById('recordBtn').className = 'btn idle';
                updateStatus("Processing your message...");
            }
        }

        async function sendAudio(audioBlob) {
            const reader = new FileReader();
            reader.onload = function() {
                const audioData = reader.result.split(',')[1]; // Remove data:audio/wav;base64,
                ws.send(JSON.stringify({
                    type: 'audio',
                    data: audioData
                }));
            };
            reader.readAsDataURL(audioBlob);
        }

        function addMessage(text, sender) {
            const chatContainer = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            messageDiv.textContent = `${sender === 'user' ? 'You' : 'TracyBot'}: ${text}`;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function playAudio(audioUrl) {
            const audio = new Audio(audioUrl);
            audio.play().catch(err => console.error('Audio playback failed:', err));
        }

        function updateStatus(message) {
            document.getElementById('status').textContent = message;
        }

        // Initialize when page loads
        window.onload = initWebSocket;
    </script>
</body>
</html>
    """)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for real-time voice chat"""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "audio":
                # Decode base64 audio data
                import base64
                audio_data = base64.b64decode(data["data"])
                
                # Convert speech to text (placeholder for now)
                user_text = await voice_chat.speech_to_text(audio_data)
                
                # Get response from Clawdbot
                response_text = await voice_chat.send_to_clawdbot(user_text)
                
                # Convert response to speech using Clawdbot TTS
                audio_path = await voice_chat.text_to_speech_clawdbot(response_text)
                
                # Send response back to client
                response = {
                    "type": "response",
                    "text": response_text,
                    "audio_url": f"/audio/{Path(audio_path).name}" if audio_path else None
                }
                
                await websocket.send_json(response)
                
    except WebSocketDisconnect:
        print("WebSocket disconnected")

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve audio files generated by Clawdbot TTS"""
    # Clawdbot TTS files are typically in a media directory
    audio_path = Path(f"/tmp/{filename}")  # Adjust path as needed
    if audio_path.exists():
        return FileResponse(audio_path, media_type="audio/mpeg")
    return {"error": "Audio file not found"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting TracyBot Voice Chat Server...")
    print("📱 Open http://localhost:8000 in your browser")
    print("🎤 Click 'Click to Talk' to start voice conversation")
    uvicorn.run(app, host="0.0.0.0", port=8000)