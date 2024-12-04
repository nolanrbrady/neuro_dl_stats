import unittest
import numpy as np

from signal_generator.signal_preprocessor import SignalPreprocessorBuilder


def calculate_snr(signal):
    mean_signal = np.mean(signal, axis=1, keepdims=True)
    signal_power = np.var(mean_signal)
    residual_noise = signal - mean_signal
    noise_power = np.var(residual_noise)
    snr = signal_power / (noise_power + 1e-10)  # Avoid division by zero
    return snr

class TestFNIRSPreprocessingBuilder(unittest.TestCase):
    def setUp(self):
        # Simulate bad fNIRS data
        np.random.seed(42)
        true_signal = np.sin(np.linspace(0, 2 * np.pi, 500))
        self.noisy_signal = true_signal + 0.5 * np.random.randn(32, 500)
        self.fs = 10

    def test_snr_improvement(self):
        # Calculate initial SNR
        initial_snr = calculate_snr(self.noisy_signal)

        # Apply preprocessing using the builder
        processed_signal = (
            SignalPreprocessorBuilder(self.noisy_signal)
            .bandpass_filter(low_cutoff=0.01, high_cutoff=0.2, fs=self.fs)
            .baseline_drift_correction()
            .remove_spiking_artifacts(threshold=2.0)
            .build()
        )

        # Calculate SNR after preprocessing
        processed_snr = calculate_snr(processed_signal)

        # Assert that SNR improves after preprocessing
        self.assertGreater(processed_snr, initial_snr, "SNR did not improve after preprocessing")

    def test_pipeline_output_shape(self):
        # Apply preprocessing using the builder
        processed_signal = (
            SignalPreprocessorBuilder(self.noisy_signal)
            .bandpass_filter(low_cutoff=0.01, high_cutoff=0.2, fs=self.fs)
            .baseline_drift_correction()
            .remove_spiking_artifacts(threshold=2.0)
            .build()
        )

        # Assert that the output shape matches the input shape
        self.assertEqual(
            processed_signal.shape,
            self.noisy_signal.shape,
            "Processed signal shape does not match the input signal shape"
        )