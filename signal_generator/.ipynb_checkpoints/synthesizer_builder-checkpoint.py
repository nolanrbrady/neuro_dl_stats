from .synthesizer import Synthesizer
import numpy as np

class SynthesizerBuilder:
    def __init__(self):
        self.synth = Synthesizer()  # Start with a default Synthesizer instance

    def add_signal(self, waveform):
        self.synth.waveform = waveform
        return self

    def add_intercept(self):
        length = self.synth.length
        intercept = np.ones(length)
        current_signal = self.synth.signal
        self.synth.signal = current_signal.append(intercept)

    def add_length(self, length):
        self.synth.length = length
        return self

    def add_drift(self):
        self.synth.drift
        return self

    def build(self):
        return self.synth  # Return the fully configured Synthesizer instance