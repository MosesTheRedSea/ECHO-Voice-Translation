import pyaudio
import wave
import time
from Speech_Text import Hearing
from Text_Speech import Speaking
from Text_Text import transformer_model
import os
import torch
import torch.nn.functional as F
import pyttsx3
import tempfile


engine = pyttsx3.init()
listener = Hearing
speaker = Speaking
translator = transformer_model


P = pyaudio.PyAudio()
stream = P.open(format = pyaudio.paInt16,
                channels = 1,
                rate = 44100,
                input = True,
                frames_per_buffer = 1024)
    

# Audio Capture
def capture_audio(stream, duration = 5):
    frames = []
    
    for _ in range(0, int(44100 / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
        
    return b''.join(frames)
    
    # Save audio data (bytes) to a temporary WAV file
def save_audio_to_tempfile(audio_data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_data)  # Write the bytes to the temporary file
        
        return tmpfile.name  # Return the path to the temporary file
    
# Speech to Text   
def listen(voice):
    #temp_filename = save_audio_to_tempfile(voice)
    text = listener.get_text(voice)
    #os.remove(temp_filename)
    
    return text

# Text to Text Transation
def translate(text):
    return translator.Seq2Seq(text)
    

# Text to Speech
def echo(text):
    engine.say(text)
    engine.runAndWait()
    #os.system(f'say "{text}"')
    
    
# The real-time translator
def real_time_translate():
    while True:
        #The audio capture
        audio_data = capture_audio(stream)
        
        #The transcription from the audio data
        text = listen(audio_data)
        print("Recognized text:" + text)
        
        #The translation from the recognized text
        translation = translate(text)
        print("Translated text:" + translation)
        
        #The audio for the translation
        echo(translation)
        
        time.sleep(1)

#Starts the translation
if __name__ == "__main__":
    try:
        print("Starting the real-time translation...")
        real_time_translate()
    except KeyboardInterrupt:
        print("Stopping real-time translation")
        stream.stop_stream()
        stream.close()
        P.terminate()
        
