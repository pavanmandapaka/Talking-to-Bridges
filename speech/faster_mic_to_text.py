import os
# Fix for OpenMP duplicate library error on Windows/Anaconda
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from faster_whisper import WhisperModel
import speech_recognition as sr
import tempfile
import time

def transcribe_with_faster_whisper(model_size="base"):
    """
    Records audio from the microphone and transcribes it using faster-whisper.
    """
    print(f"Loading faster-whisper '{model_size}' model on CPU (int8 optimized)...")
    # Using CPU with int8 quantization — still much faster than standard Whisper
    # Switch to device="cuda", compute_type="float16" once CUDA libraries are installed
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 2.0
    recognizer.phrase_threshold = 0.3
    
    with sr.Microphone() as source:
        print("\nAdjusting for ambient noise... please wait a second.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("\n🎤 Listening... Speak now! (Stop talking for a moment to end recording)")
        audio_data = recognizer.listen(source)
        print("Done listening. Processing audio...")
        
    temp_audio_path = "temp_faster_recording.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_data.get_wav_data())
        
    print("Transcribing (faster-whisper)...")
    start_time = time.time()
    
    # Transcribe the audio
    segments, info = model.transcribe(temp_audio_path, beam_size=5)
    
    print(f"\n[Detected language '{info.language}' with {info.language_probability:.2f} probability]")
    
    text = ""
    # faster-whisper returns a generator, so we iterate through the segments
    for segment in segments:
        text += segment.text + " "
        
    end_time = time.time()
    
    print("\n" + "="*30)
    print("🗣️  YOU SAID:")
    print(text.strip())
    print("="*30)
    print(f"Transcription took: {end_time - start_time:.2f} seconds\n")
    
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
        
    return text.strip()

if __name__ == "__main__":
    transcribe_with_faster_whisper()
