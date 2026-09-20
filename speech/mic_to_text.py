import whisper
import speech_recognition as sr
import tempfile
import os

def transcribe_from_mic(model_size="base"):
    """
    Records audio from the microphone and transcribes it using Whisper.
    This fulfills the Week 2 goal: "Build microphone recording → speech-to-text pipeline"
    """
    print(f"Loading Whisper '{model_size}' model...")
    model = whisper.load_model(model_size)
    
    recognizer = sr.Recognizer()
    
    # Wait 2 seconds of silence before considering the speech "done"
    # (default is 0.8s which cuts off mid-sentence)
    recognizer.pause_threshold = 2.0
    
    # Minimum length of speech to count as a phrase (filters out random noise)
    recognizer.phrase_threshold = 0.3
    
    with sr.Microphone() as source:
        print("\nAdjusting for ambient noise... please wait a second.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("\n🎤 Listening... Speak now! (Stop talking for a moment to end recording)")
        # Listens until it detects silence
        audio_data = recognizer.listen(source)
        print("Done listening. Processing audio...")
        
    # Save the recorded audio to a temporary WAV file
    # Whisper requires a file path (or a NumPy array, but a temp file is easiest to start with)
    temp_audio_path = "temp_recording.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_data.get_wav_data())
        
    print("Transcribing...")
    # Transcribe the saved temporary file
    result = model.transcribe(temp_audio_path)
    
    print("\n" + "="*30)
    print("🗣️  YOU SAID:")
    print(result["text"].strip())
    print("="*30 + "\n")
    
    # Clean up the temporary file
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
        
    return result["text"].strip()

if __name__ == "__main__":
    transcribe_from_mic()
