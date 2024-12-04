import numpy as np
from scipy.signal import butter, filtfilt

class SignalPreprocessor:
    """
    Class to hold fNIRS signal data and apply preprocessing steps.
    """
    def __init__(self, signal):
        self.signal = signal

    def set_signal(self, signal):
        self.signal = signal

    def get_signal(self):
        return self.signal


class SignalPreprocessorBuilder:
    def __init__(self, signal):
        self._preprocessor = SignalPreprocessor(signal)

    def bandpass_filter(self, low_cutoff, high_cutoff, fs, order=4):
        nyquist = 0.5 * fs
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        b, a = butter(order, [low, high], btype="band")
        filtered_signal = filtfilt(b, a, self._preprocessor.get_signal(), axis=0)
        self._preprocessor.set_signal(filtered_signal)
        return self

    def baseline_drift_correction(self):
        baseline_corrected_signal = self._preprocessor.get_signal() - np.mean(self._preprocessor.get_signal(), axis=0)
        self._preprocessor.set_signal(baseline_corrected_signal)
        return self

    def remove_spiking_artifacts(self, threshold):
        signal = self._preprocessor.get_signal()
        signal[np.abs(signal) > threshold] = threshold
        self._preprocessor.set_signal(signal)
        return self

    def normalize(self):
        signal = self._preprocessor.get_signal()
        min_val = np.min(signal)
        max_val = np.max(signal)
        normalized_signal = (signal - min_val) / (max_val - min_val)
        self._preprocessor.set_signal(normalized_signal)
        return self

    def build(self):
        return self._preprocessor.get_signal()
