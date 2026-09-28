from faster_whisper import WhisperModel
import os

# ---------------------------------------------------------
# GLOBAL MODEL INITIALIZATION
# We load the model outside the function so it only takes
# time to load once when the app starts, making transcriptions 
# instant when the user actually speaks.
# ---------------------------------------------------------
print("Loading Speech-to-Text Model (faster-whisper, base, GPU)...")
try:
    # Try loading on GPU
    model = WhisperModel("base", device="cuda", compute_type="float16")
except Exception as e:
    print(f"GPU load failed ({e}). Falling back to CPU...")
    model = WhisperModel("base", device="cpu", compute_type="int8")

def transcribe(audio_file_path):
    """
    Day 1 Agreement for /transcribe
    What goes in: Audio (file path)
    What comes out: The text that was spoken
    """
    if not os.path.exists(audio_file_path):
        return "Error: Audio file not found."
        
    # Transcribe the audio
    segments, info = model.transcribe(audio_file_path, beam_size=5)
    
    # Combine the segments into a single string
    text = ""
    for segment in segments:
        text += segment.text + " "
        
    return text.strip()

# Quick test if you run this file directly
if __name__ == "__main__":
    # You can test this with the sample audio you created earlier
    test_audio = "sample_test_audio.wav"
    if os.path.exists(test_audio):
        print(f"\nTesting transcription on: {test_audio}")
        result = transcribe(test_audio)
        print("Result:", result)
    else:
        print(f"\nCould not find {test_audio} to run a quick test.")
