import os
import time
from dotenv import load_dotenv
import requests

# Load API keys from the .env file in the root folder
load_dotenv(dotenv_path="../.env")
api_key = os.getenv("ELEVENLABS_API_KEY")

def test_elevenlabs():
    if not api_key:
        print("Error: Could not find ELEVENLABS_API_KEY in the .env file.")
        print("Please add it to your .env file like this: ELEVENLABS_API_KEY=your_key_here")
        return

    print("Connecting to ElevenLabs API...")
    
    # 1. Fetch available voices to get the ID for "Brian"
    headers = {
        "Accept": "application/json",
        "xi-api-key": api_key
    }
    
    voices_response = requests.get("https://api.elevenlabs.io/v1/voices", headers=headers)
    if voices_response.status_code != 200:
        print("Failed to fetch voices. Please check your API key.")
        return
        
    voices = voices_response.json().get("voices", [])
    
    # "Adam" is a highly realistic, professional male voice that is 
    # included in the default free tier. 
    voice_name = "Adam"
    voice_id = next((v["voice_id"] for v in voices if v["name"] == voice_name), "pNInz6obpgDQGcFmaJgB") # Fallback to Adam's exact ID
    
    test_text = "Good morning! I am the structural health monitoring assistant. All bridge sensors are currently operating within normal parameters."
    
    print(f"Synthesizing hyper-realistic speech (Voice: {voice_name})...")
    start_time = time.time()
    
    # 2. Generate the audio
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = {
        "text": test_text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    
    response = requests.post(tts_url, json=payload, headers=headers)
    
    if response.status_code == 200:
        output_file = "test_elevenlabs_output.mp3"
        with open(output_file, "wb") as f:
            f.write(response.content)
            
        end_time = time.time()
        print(f"Speech generated in {end_time - start_time:.2f} seconds!")
        
        print("Playing audio out loud...")
        os.system(f"start {output_file}")
    else:
        print(f"Error generating speech: {response.text}")

if __name__ == "__main__":
    test_elevenlabs()
