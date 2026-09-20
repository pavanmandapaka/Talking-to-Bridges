import torch
from TTS.api import TTS
import time
import os

def test_xtts():
    # Make sure we use the RTX 3050 GPU!
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading XTTS v2 model on {device.upper()}...")
    print("Note: The first run will download the model (approx 3GB). Please be patient!")
    
    # Initialize the XTTS v2 model
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    
    text = "Well, hello there! I am your structural health monitoring assistant. Everything is looking perfectly stable today."
    output_wav = "test_xtts_output.wav"
    
    print("Synthesizing ultra-realistic speech...")
    start_time = time.time()
    
    # Generate speech. XTTS is so advanced it actually clones voices. 
    # We will use one of its high-quality built-in voices ("Ana Florence").
    tts.tts_to_file(text=text, speaker="Ana Florence", language="en", file_path=output_wav)
    
    end_time = time.time()
    print(f"Speech generated in {end_time - start_time:.2f} seconds!")
    
    print("Playing audio...")
    # Open the WAV file in the default Windows media player
    os.system(f"start {output_wav}")

if __name__ == "__main__":
    test_xtts()
