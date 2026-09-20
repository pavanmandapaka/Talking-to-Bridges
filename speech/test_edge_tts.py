import asyncio
import edge_tts
import time
import os

async def test_edge():
    test_sentence = "Hello! I am your structural health monitoring assistant. The bridge sensors are currently looking stable, but I will keep an eye on them."
    
    # 'en-US-AriaNeural' is a highly expressive and natural female voice.
    # If you prefer a male voice, you can try 'en-US-ChristopherNeural'.
    voice = "en-US-AriaNeural" 
    
    output_audio = "test_edge_output.mp3"
    
    print(f"Synthesizing speech with Microsoft Edge Neural TTS (Voice: {voice})...")
    start_time = time.time()
    
    # Generate the speech
    communicate = edge_tts.Communicate(test_sentence, voice)
    await communicate.save(output_audio)
    
    end_time = time.time()
    print(f"Speech generated in {end_time - start_time:.2f} seconds!")
    
    print("Playing audio out loud...")
    
    # On Windows, 'start' automatically opens and plays the file in your default media player
    os.system(f"start {output_audio}")

if __name__ == "__main__":
    # edge_tts is asynchronous, so we run it using asyncio
    asyncio.run(test_edge())
