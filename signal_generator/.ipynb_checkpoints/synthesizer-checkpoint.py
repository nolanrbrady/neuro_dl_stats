class Synthesizer:
    def __init__(self, waveform='cosine', noise=0.5, length=500):
        self.waveform = waveform
        self.noise = noise
        self.length = length
        self.signal = []

    def play(self):
        """Simulates playing a note by printing the waveform, frequency, and volume."""
        print(self.signal)
        return self.signal

    def stop(self):
        """Stops playing the note."""
        print("Synthesizer stopped.")

    def change_waveform(self, waveform):
        """Changes the waveform type."""
        self.waveform = waveform
        print(f"Waveform changed to {self.waveform}")

    def adjust_frequency(self, frequency):
        """Adjusts the frequency."""
        self.frequency = frequency
        print(f"Frequency adjusted to {self.frequency} Hz")
