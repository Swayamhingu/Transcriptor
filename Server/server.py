import asyncio
import websockets
import json
import numpy as np
import io
import wave
from openai import OpenAI
import os

# Install required packages:
# pip install websockets numpy openai

class SimpleVAD:
    """Simple energy-based Voice Activity Detection"""
    def __init__(self, threshold=0.01, sample_rate=16000):
        self.threshold = threshold
        self.sample_rate = sample_rate

    def detect_speech(self, audio_data):
        """Detect speech based on audio energy"""
        try:
            # Convert bytes to numpy array
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
            # Convert to float32 normalized to [-1, 1]
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio_float32**2))

            return rms > self.threshold
        except Exception as e:
            print(f"VAD detection error: {e}")
            return False

class TranscriptionServer:
    def __init__(self, openai_api_key, vad_threshold=0.02):
        print("Initializing VAD (Voice Activity Detection)...")
        self.vad = SimpleVAD(threshold=vad_threshold)
        print("VAD initialized successfully!")

        self.client = OpenAI(api_key=openai_api_key)
        self.sample_rate = 16000

    def detect_speech(self, audio_data):
        """Detect if audio contains speech"""
        return self.vad.detect_speech(audio_data)

    def transcribe_audio(self, audio_buffer):
        """Send audio to OpenAI Whisper API and get transcription"""
        try:
            # Convert buffer to WAV format
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_buffer)

            wav_buffer.seek(0)
            wav_buffer.name = "audio.wav"

            # Send to OpenAI Whisper API
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=wav_buffer,
                language="en"  # Optional: remove for auto-detection
            )

            return transcript.text.strip()

        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    async def handle_client(self, websocket):
        """Handle WebSocket client connection - FIXED: removed path parameter"""
        print(f"Client connected from {websocket.remote_address}")

        audio_buffer = bytearray()
        silence_frames = 0
        speech_detected = False
        min_speech_duration = 0.5  # Minimum 0.5 seconds of speech
        silence_threshold = 2   # Frames of silence to trigger transcription (~500ms)

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Audio data received
                    audio_buffer.extend(message)

                    # Check for speech every ~320ms
                    check_interval = int(self.sample_rate * 2 * 0.32)  # 320ms of 16-bit audio

                    if len(audio_buffer) >= check_interval:
                        # Check the most recent audio chunk
                        recent_audio = bytes(audio_buffer[-check_interval:])
                        is_speech = self.detect_speech(recent_audio)

                        if is_speech:
                            speech_detected = True
                            silence_frames = 0
                        else:
                            if speech_detected:
                                silence_frames += 1

                        # If we have speech followed by sufficient silence, transcribe
                        if speech_detected and silence_frames >= silence_threshold:
                            min_bytes = int(self.sample_rate * 2 * min_speech_duration)

                            # ------------------
                            min_bytes = int(self.sample_rate * 2 * min_speech_duration)
                            if len(audio_buffer) < min_bytes:
                                # Skip transcription, probably noise
                                audio_buffer.clear()
                                speech_detected = False
                                silence_frames = 0
                                continue
                            #------------------
                            if len(audio_buffer) >= min_bytes:
                                duration = len(audio_buffer) / (self.sample_rate * 2)
                                print(f"Transcribing {len(audio_buffer)} bytes ({duration:.2f}s)...")

                                # Run transcription in thread pool to avoid blocking
                                loop = asyncio.get_event_loop()
                                text = await loop.run_in_executor(
                                    None,
                                    self.transcribe_audio,
                                    bytes(audio_buffer)
                                )

                                if text:
                                    await websocket.send(json.dumps({
                                        'type': 'transcription',
                                        'text': text
                                    }))
                                    print(f"Transcribed: {text}")

                            # Reset for next segment
                            audio_buffer.clear()
                            speech_detected = False
                            silence_frames = 0

                        # Prevent buffer from growing too large (30 seconds max)
                        max_buffer_bytes = self.sample_rate * 2 * 30
                        if len(audio_buffer) > max_buffer_bytes:
                            # Keep last 10 seconds
                            keep_bytes = self.sample_rate * 2 * 10
                            audio_buffer = audio_buffer[-keep_bytes:]
                            print("Buffer trimmed to prevent overflow")

                elif isinstance(message, str):
                    data = json.loads(message)
                    if data.get('type') == 'start':
                        print("Recording started")
                        audio_buffer.clear()
                        speech_detected = False
                        silence_frames = 0
                        await websocket.send(json.dumps({'type': 'status', 'message': 'Ready'}))
                    elif data.get('type') == 'stop':
                        print("Recording stopped")
                        # Final transcription if any audio remains
                        min_bytes = int(self.sample_rate * 2 * 0.3)
                        if len(audio_buffer) > min_bytes:
                            print("Transcribing final segment...")
                            loop = asyncio.get_event_loop()
                            text = await loop.run_in_executor(
                                None,
                                self.transcribe_audio,
                                bytes(audio_buffer)
                            )
                            if text:
                                await websocket.send(json.dumps({
                                    'type': 'transcription',
                                    'text': text
                                }))
                        audio_buffer.clear()
                        speech_detected = False
                        silence_frames = 0

        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected from {websocket.remote_address}")
        except Exception as e:
            print(f"Error handling client: {e}")
            import traceback
            traceback.print_exc()
            try:
                await websocket.send(json.dumps({'type': 'error', 'message': str(e)}))
            except:
                pass

async def main():
    # Set your OpenAI API key here or use environment variable
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        print("=" * 60)
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("=" * 60)
        print("Linux/Mac: export OPENAI_API_KEY='your-api-key-here'")
        print("Windows: set OPENAI_API_KEY=your-api-key-here")
        print("=" * 60)
        return

    # Initialize server with adjustable VAD threshold
    # Lower = more sensitive (0.005), Higher = less sensitive (0.02)
    server = TranscriptionServer(openai_api_key=api_key, vad_threshold=0.01)

    print("=" * 60)
    print("WebSocket Live Transcription Server")
    print("=" * 60)

    ################
    port = os.environ.get("PORT", 8765)
    print(f"Server listening on ws://0.0.0.0:{port}")


    print(f"STT: OpenAI Whisper API")
    print(f"VAD: Energy-based detection (threshold: 0.01)")
    print("=" * 60)
    print("Open index.html in your browser to start!")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)

    # Start WebSocket server with proper handler
    ############
    async with websockets.serve(server.handle_client, "0.0.0.0", int(os.environ.get("PORT", 8765))):
        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            print("\n" + "=" * 60)
            print("Server stopped by user")
            print("=" * 60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")