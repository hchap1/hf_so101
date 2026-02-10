import sounddevice as sd
import vosk
import queue
import json
import numpy as np
import time

# Queue to hold audio data
_audio_queue = queue.Queue()

# Load Vosk model (download first!)
MODEL_PATH = "vosk_model"
model = vosk.Model(MODEL_PATH)

def listen_until_silence(samplerate: int = 16000, silence_threshold: float = 200, 
                         silence_duration: float = 2.0, max_recording: float = 30.0) -> str:
    """
    Records audio until silence is detected and returns recognized speech as a string.
    
    Parameters:
    - samplerate: audio sample rate
    - silence_threshold: amplitude threshold below which is considered silence
    - silence_duration: seconds of continuous silence to stop recording
    - max_recording: safety cap for maximum recording time
    """
    rec = vosk.KaldiRecognizer(model, samplerate)
    silence_start = None
    start_time = time.time()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, flush=True)
        _audio_queue.put(bytes(indata))

    print("Listening... speak now.")
    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        while True:
            if time.time() - start_time > max_recording:
                print("Max recording time reached.")
                break
            data = _audio_queue.get()
            # feed to Vosk
            rec.AcceptWaveform(data)
            # check amplitude for silence
            audio_array = np.frombuffer(data, dtype=np.int16)
            amplitude = np.abs(audio_array).mean()
            if amplitude < silence_threshold:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > silence_duration:
                    # stop if enough silence
                    break
            else:
                silence_start = None

    result = rec.FinalResult()
    text = json.loads(result).get("text", "")
    print(f"Recognized (offline): {text}")
    return text


# Example usage:
if __name__ == "__main__":
    spoken_text = listen_until_silence()
    print("You said:", spoken_text)
