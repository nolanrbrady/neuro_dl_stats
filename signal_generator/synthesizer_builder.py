from .synthesizer import Synthesizer
import numpy as np
from scipy.signal import convolve

class SynthesizerBuilder:
    def __init__(self):
        self.synth = Synthesizer()  # Start with a default Synthesizer instance

    def add_signal(self, time_on, time_off, repeats, name):
        print("Adding signal...")
        baseline_level = 0
        elevated_level = 1

        # Generate the timeseries mask
        block_on = np.ones(time_on) * elevated_level
        block_off = np.ones(time_off) * baseline_level
        block = np.concatenate([block_on, block_off])
        task_signal = np.tile(block, repeats)

        # Convolve into BOLD signal
        bold_signal = self.__bold_response(task_signal)

        # Pad the series to match the signal length
        padded_task = np.pad(task_signal, (0, max(0, self.synth.length - len(task_signal))), 'constant')
        padded_bold = np.pad(bold_signal, (0, max(0, self.synth.length - len(bold_signal))), 'constant')

        # Add the signal
        self.__add_signal(padded_task, f"task_{name}")
        self.__add_signal(padded_bold, f"signal_{name}")

        return self

    def add_intercept(self):
        intercept = np.ones(self.synth.length)
        self.__add_signal(intercept, "intercept")
        return self

    def add_length(self, length):
        self.synth.length = length
        return self

    def add_drift(self, start, end):
        drift = np.linspace(start, end, self.synth.length)
        self.__add_signal(drift, "drift 1")
        return self

    def build(self):
        return self.synth  # Return the fully configured Synthesizer instance

    def __add_signal(self, signal, name):
        old_signal = self.synth.signal
        new_signal = old_signal.append(signal)
        self.signal = new_signal

        # Save the names for reference
        self.signal_names = self.synth.signal_names.append(name)

    def __bold_response(self, signal):
        time_to_peak = 6  # typical peak for the HRF in seconds
        response_length = 20  # duration of HRF (in seconds)
        hrf = np.array([(t / time_to_peak) ** 8 * np.exp(-t / time_to_peak)
                        for t in np.arange(0, response_length, 1 / self.synth.sampling_rate)])
        hrf /= hrf.sum()  # normalize the HRF

        # Step 3: Convolve the Block Design Signal with the HRF
        convolved_signal = convolve(signal, hrf, mode='full')[:len(signal)]
        return convolved_signal
