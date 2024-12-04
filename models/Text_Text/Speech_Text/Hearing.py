import whisper
import pyaudio
import numpy as np
import torch

model = whisper.load_model("turbo")
P = pyaudio.PyAudio()
stream = P.open(format = pyaudio.paInt16,
                channels = 1,
                rate = 16000,
                input = True,
                frames_per_buffer = 1024)
    
def get_text(voice):
    #Confirming that audio is being listened for
    print("Listening for audio...")
    
    while True:
        #Audio chunk capture
        
        audio_input = np.frombuffer(voice, dtype = np.int16)
        audio_tensor = torch.from_numpy(audio_input).float()
        audio_tensor = audio_tensor / 32768.0
        
        #Whisper processing
        mel = whisper.log_mel_spectrogram(audio_tensor).to(model.device)

        # detects and prints the spoken language
        _, probs = model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")

        # decodes the audio chunks
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)

        # prints and returns the transcribed text
        print(result.text) 
        return result.text 