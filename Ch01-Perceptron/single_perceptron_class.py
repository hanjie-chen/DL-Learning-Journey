import numpy as np

class Perceptron():
    def __init__(self, input_size, activation_function, learning_rate) -> None:
        self.weights = np.random.uniform(-0.5, 0.5, input_size)
        self.bias = 0
        self.activation_function = activation_function
        self.learning_rate = learning_rate

    def forward(self, input):
        return self.activation_function(np.dot(input, self.weights.T) + self.bias)
    
    def adjust_weight(self, input, target, output):
        return self.learning_rate * (target - output) * input
    
    def adjust_bias(self, target, ouput):
        return self.learning_rate * (target - ouput) * 1


    