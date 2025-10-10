import os
import sounddevice as sd
import soundfile as sf

def record_voice(filename, duration=3, sr=16000):
    """
    Record audio from microphone and save to a WAV file.
    """
    print(f"Recording {filename} for {duration} seconds...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    recording = recording.flatten()
    sf.write(filename, recording, sr)
    print(f"Saved: {filename}")

def record_multiple_clips(folder="voices/silence", num_clips=5, duration=3, sr=16000):
    os.makedirs(folder, exist_ok=True)

    for i in range(1, num_clips + 1):
        filename = os.path.join(folder, f"clip_{i}.wav")
        record_voice(filename, duration, sr)

if __name__ == "__main__":
    # Record 5 clips by default
    record_multiple_clips(folder="voices/Amgad", num_clips=10, duration=3)