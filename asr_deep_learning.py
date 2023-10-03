import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
import soundfile as sf


class AudioSuperResolutionModel(nn.Module):
    def __init__(self):
        super(AudioSuperResolutionModel, self).__init__()
        self.fc1 = nn.Linear(44100, 88200)  # Increase the sampling rate by 2x

    def forward(self, x):
        x = self.fc1(x)
        return x

input_audio_file = "input_audio.wav"
target_sampling_rate = 88200  # Define your target sampling rate (e.g., 2x the original)

y, sr = librosa.load(input_audio_file, sr=None)

# Create an instance of your model
model = AudioSuperResolutionModel()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert audio data to a PyTorch tensor
input_audio = torch.tensor(y, dtype=torch.float32)

# Perform audio super-resolution
output_audio = model(input_audio.unsqueeze(0)).squeeze().detach().numpy()

# Save the super-resolved audio
output_audio_file = "output_audio.wav"
sf.write(output_audio_file, output_audio, target_sampling_rate)

print(output_audio_file)
