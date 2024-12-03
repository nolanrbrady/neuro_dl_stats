import sys
print(sys.executable)
import pandas as pd
import torch
import numpy as np
import statsmodels.api as sm
from signal_generator.synthesizer_builder import SynthesizerBuilder
from models.model_factory import ModelFactory

def generate_offset_cosine_matrix(num_samples=100, num_waves=500, frequency=0.5, sampling_rate=100, offset_steps=5):
    t = np.linspace(0, num_samples / sampling_rate, num_samples, endpoint=False)
    wave_matrix = np.zeros((num_samples, num_waves))  # Initialize matrix
    for i in range(num_waves):
        wave_matrix[:, i] = np.cos(2 * np.pi * frequency * t + (i + 1))  # Generate cosine wave with phase offset

    return wave_matrix

# Create a Synthesizer with custom parameters using the builder
synth_builder = SynthesizerBuilder()
synth = (synth_builder
       .add_length(500)
       .add_intercept()
       .add_drift(0, 0.3)
       .add_heart_rate()
       .add_signal(30, 60, 3, 15, "1")
       .add_signal(30, 60, 3, 50, "2")
       .build())
signal, design_matrix = synth.get()
print("Signal shape: ", signal.shape)
print("Design matrix: ", design_matrix.shape)
# synth.plot_signal()
# synth.plot_design_matrix()

# Get the GLM Values
beta_values_target = np.array([1, 1.5, 12, 3, 0.5])
# print(design_matrix.shape, beta_values.shape)
y_true = np.dot(design_matrix, beta_values_target)
# print("Y_true: ", y_true, type(y_true), y_true.shape)
design_matrix = np.concatenate([design_matrix, signal], axis=1)

# Cosine VAE test
wave_data = generate_offset_cosine_matrix()
print("Wave data shape: ", wave_data.shape)

design_matrix = wave_data

# Run the model
print(design_matrix.shape)
# Create the model
VAE = ModelFactory.create_model('autoencoder', design_matrix.shape)

# Train the model
model = VAE.train(design_matrix.T, learning_rate=0.001, n_epochs=10000)

# Get the latent value
x_sample = torch.FloatTensor(design_matrix.T)
latent_values = model.get_latent_value(x_sample)
print("Learned latent value: ", latent_values, latent_values.shape)