import unittest
import numpy as np
from unittest.mock import MagicMock

from signal_generator.synthesizer import Synthesizer
from signal_generator.synthesizer_builder import SynthesizerBuilder


class TestSynthesizerBuilder(unittest.TestCase):

    def setUp(self):
        self.builder = SynthesizerBuilder()
        self.builder.synth = Synthesizer()  # Mock Synthesizer instance

    def test_add_signal(self):
        self.builder.add_length(100)
        self.builder.add_signal(5, 10, 2, 0, "test_signal")
        self.assertIn("task_test_signal", self.builder.synth.design_matrix_names)
        self.assertIn("signal_test_signal", self.builder.synth.signal_names)

    def test_add_intercept(self):
        self.builder.add_length(100)
        self.builder.add_intercept()
        self.assertIn("intercept", self.builder.synth.design_matrix_names)
        intercept = self.builder.synth.design_matrix[-1]
        np.testing.assert_array_equal(intercept, np.ones(100))

    def test_add_length(self):
        self.builder.add_length(150)
        self.assertEqual(self.builder.synth.length, 150)

    def test_add_drift(self):
        self.builder.add_length(100)
        self.builder.add_drift(0, 1)
        self.assertIn("drift 1", self.builder.synth.design_matrix_names)
        drift = self.builder.synth.design_matrix[-1]
        expected_drift = np.linspace(0, 1, 100)
        np.testing.assert_array_almost_equal(drift, expected_drift)

    def test_add_heart_rate(self):
        self.builder.add_length(100)
        self.builder.add_heart_rate()
        self.assertIn("heart_rate", self.builder.synth.design_matrix_names)
        hr_wave = self.builder.synth.design_matrix[-1]
        self.assertEqual(len(hr_wave), 100)

    def test_build(self):
        synth = self.builder.build()
        self.assertIsInstance(synth, Synthesizer)

    def test_bold_response(self):
        signal = np.ones(100)
        bold_signal = self.builder._SynthesizerBuilder__bold_response(signal)
        self.assertEqual(len(bold_signal), 100)
        self.assertTrue(np.any(bold_signal > 0))  # Ensure some signal is present

    def test_private_add_signal(self):
        signal = np.ones(100)
        self.builder._SynthesizerBuilder__add_signal(signal, "test_signal")
        self.assertIn("test_signal", self.builder.synth.signal_names)
        self.assertEqual(len(self.builder.synth.signal), 1)

    def test_private_add_to_design_matrix(self):
        params = np.ones(100)
        self.builder._SynthesizerBuilder__add_to_design_matrix(params, "test_params")
        self.assertIn("test_params", self.builder.synth.design_matrix_names)
        self.assertEqual(len(self.builder.synth.design_matrix), 1)


if __name__ == "__main__":
    unittest.main()