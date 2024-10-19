import numpy as np

class MultiLayerPerceptron():
    """
    a simple implementation of 3-layer MLP with scalar realized back propagation
    """
    def __init__(self, Layers, activation_function, leanring_rate):
        # init weights and bias for each layers
        self.Weights_hidden
        self.Bias_hidden = np.zeros((1, Layers[1]))
        self.Weights_output
        self.Bias_output = np.zeros((1, Layers[2]))
        self.activation_function = activation_function
        self.learning_rate = leanring_rate