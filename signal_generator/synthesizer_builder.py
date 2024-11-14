from .synthesizer import Synthesizer
import numpy as np
from scipy.signal import convolve
from scipy.special import gamma

class SynthesizerBuilder:
    def __init__(self):
        self.synth = Synthesizer()  # Start with a default Synthesizer instance

    def add_signal(self, time_on, time_off, repeats, offset, name):
        baseline_level = 0
        elevated_level = 1

        # Generate the timeseries mask
        start_delay = np.ones(offset) * baseline_level
        block_on = np.ones(time_on) * elevated_level
        block_off = np.ones(time_off) * baseline_level
        block = np.concatenate([start_delay, block_on, block_off])
        task_signal = np.tile(block, repeats)
        print("Task signal shape: ", task_signal.shape)

        # Convolve into BOLD signal
        bold_signal = self.__bold_response(task_signal)

        # Pad the series to match the signal length
        padded_task = np.pad(task_signal, (0, max(0, self.synth.length - len(task_signal))), 'constant')
        padded_bold = np.pad(bold_signal, (0, max(0, self.synth.length - len(bold_signal))), 'constant')

        # Add the signal
        self.__add_to_design_matrix(padded_task, f"task_{name}")
        self.__add_signal(padded_bold, f"signal_{name}")

        return self

    def add_intercept(self):
        intercept = np.ones(self.synth.length)
        self.__add_to_design_matrix(intercept, "intercept")
        return self

    def add_length(self, length):
        self.synth.length = length
        return self

    def add_drift(self, start, end):
        drift = np.linspace(start, end, self.synth.length)
        self.__add_to_design_matrix(drift, "drift 1")
        return self

    def build(self):
        return self.synth  # Return the fully configured Synthesizer instance

    # Private functions -- No Touchy --
    def __add_signal(self, signal, name):
        old_signal = self.synth.signal
        new_signal = old_signal.append(signal)
        self.signal = new_signal

        # Save the names for reference
        self.signal_names = self.synth.signal_names.append(name)

    def __add_to_design_matrix(self, params, name):
        # Add the design matrix parameters
        old_params = self.synth.design_matrix
        new_params = old_params.append(params)
        self.design_matrix = new_params

        # Keep track of the names of the various inputs
        self.design_matrix_names = self.synth.design_matrix_names.append(name)

    # Original Version
    # def __bold_response(self, signal):
    #     time_to_peak = 6  # typical peak for the HRF in seconds
    #     response_length = 20  # duration of HRF (in seconds)
    #     hrf = np.array([(t / time_to_peak) ** 8 * np.exp(-t / time_to_peak)
    #                     for t in np.arange(0, response_length, 1 / self.synth.sampling_rate)])
    #     hrf /= hrf.sum()  # normalize the HRF
    #
    #     # Step 3: Convolve the Block Design Signal with the HRF
    #     convolved_signal = convolve(signal, hrf, mode='full')[:len(signal)]
    #     return convolved_signal

    # ChatGPT Edit Version
    def __bold_response(self, signal):
        # HRF Parameters
        tr = 1.0 / self.synth.sampling_rate  # Time resolution in seconds
        response_length = 32  # Total duration of HRF in seconds
        time = np.arange(0, response_length, tr)
        peak_delay = 6  # Time to peak of the response in seconds
        under_delay = 16  # Time to peak of the undershoot in seconds
        peak_disp = 1  # Dispersion of the peak
        under_disp = 1  # Dispersion of the undershoot
        p_u_ratio = 6  # Ratio of peak to undershoot amplitude

        # Calculate the HRF using a double gamma function
        # Ensure time starts from zero to keep HRF causal
        peak = ((time / peak_delay) ** (peak_delay * peak_disp)) * np.exp(- (time - peak_delay) / peak_disp)
        peak[time < 0] = 0  # Zero out negative times to maintain causality
        peak /= (gamma(peak_delay * peak_disp) * peak_disp)

        undershoot = ((time / under_delay) ** (under_delay * under_disp)) * np.exp(- (time - under_delay) / under_disp)
        undershoot[time < 0] = 0
        undershoot /= (gamma(under_delay * under_disp) * under_disp)

        hrf = peak - undershoot / p_u_ratio
        hrf[time < 0] = 0
        hrf /= np.max(hrf)  # Normalize to peak at 1

        # Convolve the Block Design Signal with the HRF
        convolved_signal = np.convolve(signal, hrf, mode='full')

        # Adjust the convolution output to account for the HRF delay
        hrf_delay_samples = int(peak_delay * self.synth.sampling_rate)
        convolved_signal = convolved_signal[hrf_delay_samples:hrf_delay_samples + len(signal)]

        # Ensure the convolved signal has the same length as the input signal
        convolved_signal = convolved_signal[:len(signal)]

        return convolved_signal