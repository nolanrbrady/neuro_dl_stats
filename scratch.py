import sys
print(sys.executable)
import pandas as pd
import torch
import numpy as np
import statsmodels.api as sm
from signal_generator.synthesizer_builder import SynthesizerBuilder
from models.model_factory import ModelFactory


# Create a Synthesizer with custom parameters using the builder
synth_builder = SynthesizerBuilder()
synth = (synth_builder
       .add_length(500)
       .add_intercept()
       .add_drift(0, 0.5)
       .add_signal(30, 60, 3, 15, "1")
       .add_signal(30, 60, 3, 50, "2")
       .build())
signal, design_matrix = synth.get()
print("Signal shape: ", signal.shape)
print("Design matrix: ", design_matrix.shape)
# synth.plot_signal()
# synth.plot_design_matrix()

# Get the GLM Values
beta_values_target = np.array([1, 1.5, 12, 3])
# print(design_matrix.shape, beta_values.shape)
y_true = np.dot(design_matrix, beta_values_target)
# print("Y_true: ", y_true, type(y_true), y_true.shape)



# Run the model
print(design_matrix.shape)
# Create the model
VAE = ModelFactory.create_model('autoencoder', design_matrix.shape)

# Train the model
model = VAE.train(design_matrix, learning_rate=0.001, n_epochs=2000)

# Get the latent value
x_sample = torch.FloatTensor(design_matrix)
latent_values = model.get_latent_value(x_sample)
print("Learned latent value: ", latent_values.shape)