import unittest
import numpy as np
from models.model_factory import ModelFactory
from models.cnn_glm import SeqCNN
from models.variational_autoencoder import VariationalAutoEncoder


class TestModelFactory(unittest.TestCase):
    def test_create_glm_model(self):
        data_shape = (100, 10)
        design_matrix = np.random.rand(*data_shape)
        model = ModelFactory.create_model("glm", design_matrix)
        self.assertEqual(model.data_shape, data_shape)
        self.assertEqual(model.n_predictors, data_shape[1])

    def test_create_cnn_model(self):
        data_shape = (500, 10)
        design_matrix = np.random.rand(*data_shape)
        model = ModelFactory.create_model("cnn", design_matrix)
        self.assertIsInstance(model, SeqCNN)
        self.assertEqual(model.kernel_size, data_shape[1])
        self.assertEqual(model.hidden_size, data_shape[1])

    def test_create_autoencoder_model(self):
        data_shape = (500, 5)
        # design_matrix = np.random.rand(*data_shape)
        model = ModelFactory.create_model("variational_autoencoder", data_shape)
        self.assertIsInstance(model, VariationalAutoEncoder)
        self.assertEqual(model.input_dim, data_shape[0])

    def test_invalid_model_type(self):
        data_shape = (10, 10)
        with self.assertRaises(ValueError) as context:
            ModelFactory.create_model("invalid_type", data_shape)
        self.assertEqual(
            str(context.exception),
            "Model type 'invalid_type' is not supported. Choose from: ['glm', 'cnn', 'autoencoder']"
        )
