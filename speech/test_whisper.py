import whisper
import time
import os

def test_local_whisper(audio_file_path, model_size="base"):
    """
    Test script to load a local Whisper model and transcribe an audio file.
    """
    print(f"Loading Whisper '{model_size}' model...")
    # Using 'base' for quick testing, but you can try 'tiny', 'small', 'medium', or 'large'
    model = whisper.load_model(model_size)
    
    print(f"Transcribing '{audio_file_path}'...")
    start_time = time.time()
    
    # Perform the transcription
    result = model.transcribe(audio_file_path)
    
    end_time = time.time()
    
    print("\n--- Transcription Result ---")
    print(result["text"].strip())
    print("-" * 28)
    print(f"Time taken: {end_time - start_time:.2f} seconds\n")

if __name__ == "__main__":
    # Create a dummy audio file path for testing purposes
    sample_audio = "sample_test_audio.wav"
    
    if os.path.exists(sample_audio):
        test_local_whisper(sample_audio)
    else:
        print(f"Error: Could not find '{sample_audio}'.")
        print("Please place a short sample audio file in this folder and rename it to 'sample_test_audio.wav' to run the test.")
