import os
import sys
import time
import requests
import base64
import tempfile
import speech_recognition as sr
import pygame

# Fix Windows console encoding for Unicode/emoji characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Initialize audio player
pygame.mixer.init()

API_BASE_URL = "http://127.0.0.1:8000"

def play_audio(audio_b64, content_type):
    """Decodes base64 audio and plays it seamlessly without opening external players."""
    audio_bytes = base64.b64decode(audio_b64)
    suffix = ".mp3" if "mpeg" in content_type else ".wav"
    
    temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(temp_fd)
    
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)
        
    try:
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    finally:
        pygame.mixer.music.unload()
        try:
            os.remove(temp_path)
        except Exception:
            pass

def main():
    print("==================================================")
    print("*  TALKING TO BRIDGES - HANDS-FREE VOICE AGENT  *")
    print("==================================================")
    
    recognizer = sr.Recognizer()
    # Adjust for ambient noise briefly
    with sr.Microphone() as source:
        print("\n[System] Calibrating background noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
    # Start the continuous conversational loop
    while True:
        try:
            with sr.Microphone() as source:
                print("\n[Listening...] (Speak now, will auto-stop on silence)")
                # phrase_time_limit prevents it from recording forever if background noise is loud
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
                
            print("[Processing your voice...]")
            
            # Save the recorded audio to a temporary file
            wav_data = audio.get_wav_data()
            temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(temp_fd)
            with open(temp_path, "wb") as f:
                f.write(wav_data)
                
            # 1. Transcribe (Speech-to-Text)
            with open(temp_path, "rb") as f:
                files = {"audio": ("mic.wav", f, "audio/wav")}
                resp = requests.post(f"{API_BASE_URL}/transcribe", files=files, timeout=60.0)
                
            os.remove(temp_path)
            
            if resp.status_code != 200:
                print(f"[Error] Transcription failed: {resp.text}")
                continue
                
            user_text = resp.json().get("text", "").strip()
            print(f"\nYOU: {user_text}")
            
            if not user_text:
                continue
                
            # Exit command
            if user_text.lower() in ["stop", "exit", "quit", "goodbye"]:
                print("\n[System] Shutting down voice agent. Goodbye!")
                break
                
            print("[Thinking...]")
            
            # 2. Chat (LLM via backend)
            chat_resp = requests.post(f"{API_BASE_URL}/api/chat", json={"message": user_text}, timeout=120.0)
            if chat_resp.status_code != 200:
                print(f"[Error] Chat failed: {chat_resp.text}")
                continue
                
            answer = chat_resp.json().get("answer", "")
            print(f"\nASSISTANT: {answer}\n")
            
            # 3. Speak (Text-to-Speech)
            speak_resp = requests.post(f"{API_BASE_URL}/speak", json={"message": answer}, timeout=60.0)
            if speak_resp.status_code == 200:
                audio_b64 = speak_resp.json().get("audio_base64")
                content_type = speak_resp.json().get("content_type", "audio/wav")
                if audio_b64:
                    play_audio(audio_b64, content_type)
                    
        except sr.WaitTimeoutError:
            # Just loop back and listen again if no speech was detected
            pass
        except KeyboardInterrupt:
            print("\n[System] Voice Agent stopped by user.")
            break
        except Exception as e:
            print(f"\n[Error] {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
