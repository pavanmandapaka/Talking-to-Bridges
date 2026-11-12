import os
import re
import base64
import requests
import asyncio
import tempfile
import wave
from dotenv import load_dotenv

load_dotenv()

# Set your preferred engine here or in the .env file
# Options: "elevenlabs", "edge", "piper"
DEFAULT_TTS_ENGINE = os.getenv("TTS_ENGINE", "edge")

# ---------------------------------------------------------
# TEXT CLEANING FOR SPEECH (Removes markdown / asterisks)
# ---------------------------------------------------------
def clean_text_for_speech(text: str) -> str:
    """Clean markdown formatting, asterisks, and noisy symbols so TTS speaks natural English."""
    if not text:
        return ""

    # 1. Remove fenced code blocks ```...```
    text = re.sub(r"```[\s\S]*?```", "", text)

    # 2. Inline code backticks: `code` -> code
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # 3. Markdown links: [label](url) -> label
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # 4. Headers: ### Header -> Header
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # 5. Bold & italic formatting (***text***, **text**, *text*, ___text___, __text__)
    text = re.sub(r"\*{3}(.*?)\*{3}", r"\1", text)
    text = re.sub(r"_{3}(.*?)_{3}", r"\1", text)
    text = re.sub(r"\*{2}(.*?)\*{2}", r"\1", text)
    text = re.sub(r"_{2}(.*?)_{2}", r"\1", text)
    text = re.sub(r"\*([^*\n]+)\*", r"\1", text)

    # 6. Any remaining standalone asterisks (* or **)
    text = text.replace("*", "")

    # 7. Tables: remove divider rows like |---|---|
    text = re.sub(r"\|[-:\s|]+\|", "", text)
    # Replace table pipe delimiters with commas/spaces for natural speech cadence
    text = re.sub(r"\|", " , ", text)

    # 8. Blockquotes: > Quote -> Quote
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)

    # 9. Convert Sensor_1 .. Sensor_5 to "Sensor 1" so TTS does not say "Sensor underscore 1"
    text = re.sub(r"\bSensor_(\d+)\b", r"Sensor \1", text)

    # 10. Clean multiple underscores
    text = re.sub(r"_{2,}", " ", text)

    # 11. Bullet point markers at line start
    text = re.sub(r"^\s*[-+]\s+", "", text, flags=re.MULTILINE)

    # 12. Normalize multiple spaces and extra newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)

    return text.strip()

# ---------------------------------------------------------
# PIPER INITIALIZATION (Loads once on startup if chosen)
# ---------------------------------------------------------
_piper_voice = None
def get_piper_voice():
    global _piper_voice
    if _piper_voice is None:
        try:
            from piper.voice import PiperVoice
            import requests
            import tempfile
            import shutil
            
            model_name = "en_US-lessac-high"
            base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/high/{model_name}"
            model_path = os.path.join(os.path.dirname(__file__), f"{model_name}.onnx")
            config_path = f"{model_path}.json"
            
            def download_file(url, out_path):
                print(f"Downloading {out_path}...")
                with requests.get(url, stream=True, timeout=30.0) as r:
                    r.raise_for_status()
                    fd, tmp = tempfile.mkstemp()
                    os.close(fd)
                    with open(tmp, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    shutil.move(tmp, out_path)
            
            if not os.path.exists(model_path):
                download_file(f"{base_url}.onnx", model_path)
                download_file(f"{base_url}.onnx.json", config_path)
            
            _piper_voice = PiperVoice.load(model_path)
        except Exception as e:
            print(f"Error loading Piper: {e}")
    return _piper_voice

# ---------------------------------------------------------
# ENGINE FUNCTIONS
# ---------------------------------------------------------
def _speak_elevenlabs(text: str) -> tuple[str, str]:
    """Generates audio via ElevenLabs API and returns (base64_string, content_type)"""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY is missing in .env")
        
    url = "https://api.elevenlabs.io/v1/text-to-speech/pNInz6obpgDQGcFmaJgB" # Adam
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }
    headers = {"Accept": "audio/mpeg", "xi-api-key": api_key}
    
    response = requests.post(url, json=payload, headers=headers, timeout=10.0)
    if response.status_code == 200:
        audio_b64 = base64.b64encode(response.content).decode("ascii")
        return audio_b64, "audio/mpeg"
    raise Exception(f"ElevenLabs Error: {response.text}")

def _speak_edge(text: str) -> tuple[str, str]:
    """Generates audio via Edge-TTS and returns (base64_string, content_type)"""
    import edge_tts
    
    async def generate_edge():
        communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return audio_data

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            audio_bytes = pool.submit(lambda: asyncio.run(generate_edge())).result()
    else:
        audio_bytes = asyncio.run(generate_edge())

    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    return audio_b64, "audio/mpeg"

def _speak_piper(text: str) -> tuple[str, str]:
    """Generates audio via local Piper and returns (base64_string, content_type)"""
    voice = get_piper_voice()
    if not voice:
        raise Exception("Piper voice could not be loaded.")
        
    temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(temp_fd)
    
    try:
        with wave.open(temp_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(voice.config.sample_rate)
            voice.synthesize_wav(text, wav_file)
            
        with open(temp_path, "rb") as f:
            audio_bytes = f.read()
            
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        return audio_b64, "audio/wav"
    finally:
        os.remove(temp_path)

# ---------------------------------------------------------
# MAIN EXPORT
# ---------------------------------------------------------
def generate_speech(text: str, engine: str = None) -> tuple[str, str]:
    """
    Day 1 Agreement for /speak
    What goes in: Text
    What comes out: Audio (Base64), Content-Type
    """
    if not text or not text.strip():
        raise ValueError("Cannot synthesize speech from empty text.")

    cleaned_text = clean_text_for_speech(text)
    if not cleaned_text:
        cleaned_text = text.strip()

    engine = engine or DEFAULT_TTS_ENGINE
    
    print(f"[TTS Service] Generating speech using engine: {engine}")
    
    try:
        if engine == "elevenlabs":
            return _speak_elevenlabs(cleaned_text)
        elif engine == "edge":
            return _speak_edge(cleaned_text)
        elif engine == "piper":
            return _speak_piper(cleaned_text)
        else:
            raise ValueError(f"Unknown engine: {engine}")
    except Exception as e:
        print(f"[TTS Service Error] {e}. Falling back to Piper...")
        return _speak_piper(cleaned_text)
