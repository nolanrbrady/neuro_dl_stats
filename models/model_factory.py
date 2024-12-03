from .cnn_glm import SeqCNN
from .deep_glm import BoldGLM
from .variational_autoencoder import VariationalAutoEncoder
from .autoencoder import AutoEncoder

class ModelFactory:
    @staticmethod
    def create_model(model_type, data_shape, **kwargs):
        models = {
            "glm": BoldGLM,
            "cnn": SeqCNN,
            "variational_autoencoder": VariationalAutoEncoder,
            "autoencoder": AutoEncoder,
        }

        if model_type not in models:
            raise ValueError(f"Model type '{model_type}' is not supported. Choose from: {list(models.keys())}")

        # Pass data_shape and kwargs to the model's constructor
        return models[model_type](data_shape, **kwargs)