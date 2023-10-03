import librosa
import soundfile as sf
import numpy as np

input_audio_file = "input_audio.wav"
output_audio_file = "resolution_audio.wav"

target_sampling_rate = 44100

# Load the low-resolution audio and resample it to the target rate
y, sr = librosa.load(input_audio_file, sr=None)
y_high_res = librosa.resample(y, sr, target_sampling_rate)

# Save the super-resolved audio
sf.write(output_audio_file, y_high_res, target_sampling_rate)

print(output_audio_file)
