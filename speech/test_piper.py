import os
import urllib.request
import wave
import time
from piper.voice import PiperVoice
import sounddevice as sd
import soundfile as sf

def download_human_voice():
    """
    Downloads a 'high' quality Piper voice for the most human-sounding results.
    'en_US-ryan-high' is a highly rated, natural-sounding male voice. 
    (You can also try 'en_US-lessac-high' for a female voice)
    """
    # Let's switch to a highly expressive female voice
    model_name = "en_US-lessac-high"
    base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/high/{model_name}"
    
    model_path = f"{model_name}.onnx"
    config_path = f"{model_name}.onnx.json"
    
    if not os.path.exists(model_path):
        print(f"Downloading high-quality human voice model ({model_name})...")
        print("This is about 70MB and only needs to happen once.")
        urllib.request.urlretrieve(f"{base_url}.onnx", model_path)
        urllib.request.urlretrieve(f"{base_url}.onnx.json", config_path)
        print("Download complete!\n")
        
    return model_path

def test_tts(text="Hello! I am your structural health monitoring assistant. The bridge sensors are currently looking stable."):
    model_path = download_human_voice()
    
    print("Loading voice model...")
    voice = PiperVoice.load(model_path)
    
    output_wav = "test_output.wav"
    print(f"Synthesizing speech: '{text}'")
    
    start_time = time.time()
    
    # Piper synthesizes the text and saves it directly to a .wav file
    with wave.open(output_wav, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
        
    end_time = time.time()
    print(f"Speech generated in {end_time - start_time:.2f} seconds!")
    
    print("Playing audio out loud...")
    # Read the audio file and play it through your laptop speakers
    data, fs = sf.read(output_wav)
    sd.play(data, fs)
    sd.wait() # Wait until the audio finishes playing
    
    print("Finished playing!")

if __name__ == "__main__":
    # You can change this text to test different sentences
    test_sentence = "The anomaly detection pipeline found a slight vibration spike on sensor 4."
    test_tts(test_sentence)
