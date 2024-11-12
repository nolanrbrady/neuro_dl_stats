import numpy as np
import matplotlib.pyplot as plt

class Synthesizer:
    def __init__(self, task='block', noise=0.5, length=500):
        self.signal_names = []
        self.drift = None
        self.sampling_rate = 1 # in Hz
        self.task = task
        self.noise = noise
        self.length = length
        self.signal = []
        self.noise = []

    def get(self):
        """Simulates playing a note by printing the waveform, frequency, and volume."""
        output = np.array(self.signal).T
        return output

    def plot(self):
        signal = np.array(self.signal)
        # Plot each time series
        plt.figure(figsize=(12, 6))
        for i in range(signal.shape[0]):
            plt.plot(signal[i], label=f"Series {i + 1}")

        # Add labels and legend
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Time Series Data")
        plt.legend()
        plt.show()

