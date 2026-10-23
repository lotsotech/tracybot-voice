#!/usr/bin/env python3
"""
Debug Voice Chat - Simplified with detailed logging
"""

import asyncio
import json
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI(title="TracyBot Voice Chat - Debug")

@app.get("/")
async def get_main_page():
    """Serve the debug voice chat interface"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>TracyBot Voice Chat - Debug</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .chat-container { border: 1px solid #ddd; height: 300px; overflow-y: scroll; padding: 10px; margin: 10px 0; }
        .debug-container { border: 1px solid #f44; height: 200px; overflow-y: scroll; padding: 10px; margin: 10px 0; background: #ffe; }
        .voice-controls { text-align: center; margin: 20px 0; }
        .btn { padding: 10px 20px; margin: 5px; font-size: 16px; cursor: pointer; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .recording { background-color: #ff4444; color: white; }
        .idle { background-color: #4444ff; color: white; }
        .message { margin: 10px 0; padding: 8px; border-radius: 5px; }
        .user { background-color: #e3f2fd; text-align: right; }
        .bot { background-color: #f3e5f5; }
        .debug { font-family: monospace; font-size: 12px; color: #666; }
    </style>
</head>
<body>
    <h1>🎤 TracyBot Voice Chat - Debug Mode</h1>
    
    <div class="debug-container" id="debugContainer">
        <strong>Debug Log:</strong><br>
    </div>
    
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

        function debugLog(message) {
            const debugContainer = document.getElementById('debugContainer');
            const timestamp = new Date().toLocaleTimeString();
            debugContainer.innerHTML += `<div class="debug">[${timestamp}] ${message}</div>`;
            debugContainer.scrollTop = debugContainer.scrollHeight;
            console.log(`[DEBUG] ${message}`);
        }

        // Initialize WebSocket connection
        function initWebSocket() {
            debugLog("Initializing WebSocket connection...");
            ws = new WebSocket(`wss://100.125.71.73:8001/ws`);
            
            ws.onopen = function() {
                debugLog("✅ WebSocket connected successfully!");
                updateStatus("Connected - ready to chat!");
            };
            
            ws.onmessage = function(event) {
                debugLog(`📥 Received message: ${event.data}`);
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'response') {
                        addMessage(data.text, 'bot');
                        debugLog(`🤖 Bot response: "${data.text}"`);
                    } else if (data.type === 'error') {
                        debugLog(`❌ Error: ${data.message}`);
                        updateStatus(`Error: ${data.message}`);
                    } else if (data.type === 'debug') {
                        debugLog(`🔍 Server debug: ${data.message}`);
                    }
                } catch (e) {
                    debugLog(`❌ Failed to parse message: ${e}`);
                }
            };
            
            ws.onclose = function() {
                debugLog("❌ WebSocket connection closed");
                updateStatus("Disconnected - refresh to reconnect");
            };
            
            ws.onerror = function(error) {
                debugLog(`❌ WebSocket error: ${error}`);
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
                debugLog("🎤 Requesting microphone access...");
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                debugLog("✅ Microphone access granted");
                
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                    debugLog(`📊 Audio chunk received: ${event.data.size} bytes`);
                };

                mediaRecorder.onstop = async () => {
                    debugLog("🛑 Recording stopped, processing audio...");
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    debugLog(`📦 Audio blob created: ${audioBlob.size} bytes`);
                    await sendAudio(audioBlob);
                };

                mediaRecorder.start();
                isRecording = true;
                
                document.getElementById('recordBtn').textContent = '🛑 Stop Recording';
                document.getElementById('recordBtn').className = 'btn recording';
                updateStatus("Recording... speak now!");
                debugLog("🎙️ Recording started");

            } catch (err) {
                debugLog(`❌ Microphone error: ${err.message}`);
                updateStatus("Error: Could not access microphone");
            }
        }

        async function stopRecording() {
            if (mediaRecorder && isRecording) {
                debugLog("⏹️ Stopping recording...");
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                isRecording = false;
                
                document.getElementById('recordBtn').textContent = '🎤 Click to Talk';
                document.getElementById('recordBtn').className = 'btn idle';
                updateStatus("Processing your message...");
            }
        }

        async function sendAudio(audioBlob) {
            debugLog("📤 Converting audio to base64...");
            const reader = new FileReader();
            reader.onload = function() {
                const audioData = reader.result.split(',')[1]; // Remove data:audio/wav;base64,
                debugLog(`📡 Sending audio data: ${audioData.length} characters`);
                
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: 'audio',
                        data: audioData
                    }));
                    debugLog("✅ Audio data sent to server");
                } else {
                    debugLog("❌ WebSocket not connected - cannot send audio");
                }
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

        function updateStatus(message) {
            document.getElementById('status').textContent = message;
            debugLog(`📍 Status: ${message}`);
        }

        // Initialize when page loads
        window.onload = function() {
            debugLog("🚀 Page loaded, initializing...");
            initWebSocket();
        };
    </script>
</body>
</html>
    """)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for real-time voice chat with debug logging"""
    await websocket.accept()
    print("🔌 WebSocket connection accepted")
    
    # Send debug message to client
    await websocket.send_json({
        "type": "debug",
        "message": "WebSocket connection established"
    })
    
    try:
        while True:
            data = await websocket.receive_json()
            print(f"📥 Received data: {data.get('type', 'unknown')} type")
            
            if data["type"] == "audio":
                try:
                    # Send debug update
                    await websocket.send_json({
                        "type": "debug", 
                        "message": "Processing audio data..."
                    })
                    
                    # Decode base64 audio data
                    audio_data = base64.b64decode(data["data"])
                    print(f"🎵 Decoded audio: {len(audio_data)} bytes")
                    
                    # For now, just echo back a simple response
                    response_text = f"I heard you! Your audio was {len(audio_data)} bytes. (This is a debug response - speech recognition not yet implemented)"
                    
                    await websocket.send_json({
                        "type": "debug",
                        "message": f"Sending response: {len(response_text)} chars"
                    })
                    
                    # Send response back to client
                    await websocket.send_json({
                        "type": "response",
                        "text": response_text
                    })
                    
                    print("✅ Response sent successfully")
                    
                except Exception as e:
                    print(f"❌ Error processing audio: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Processing error: {str(e)}"
                    })
                
    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    print("🐛 Starting Debug Voice Chat Server...")
    uvicorn.run(app, host="0.0.0.0", port=8001, ssl_keyfile="/tmp/key.pem", ssl_certfile="/tmp/cert.pem")