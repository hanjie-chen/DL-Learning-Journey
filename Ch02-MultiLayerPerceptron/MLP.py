class MultipleLayerPerceptron():
    """
    a simple implementation of MLP with scalar realized back propagation
    """
    def __init__(self, Layers, activation_function, leanring_rate):
        # init weights and bias for each layers
        
        self.activation_function = activation_function
        self.learning_rate = leanring_rate