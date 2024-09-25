import numpy as np

class Perceptron():
    """
    A simple implementation of the Perceptron algorithm
    """

    def __init__(self, input_size, activation_function, learning_rate) -> None:
        self.weights = np.random.uniform(-0.5, 0.5, input_size)
        self.bias = 0
        self.activation_function = activation_function
        self.learning_rate = learning_rate
    
    # forward process
    def forward(self, input):
        return self.activation_function(np.dot(input, self.weights.T) + self.bias)
    
    # Delta learning rule
    def adjust_weight(self, input, target, output):
        return self.learning_rate * (target - output) * input
    
    def adjust_bias(self, target, output):
        return self.learning_rate * (target - output) * 1
    
    # train 1 input
    def train_once(self, input, target):
        output = self.forward(input)
        if output != target:
            self.weights += self.adjust_weight(input, target, output)
            self.bias += self.adjust_bias(target, output)
        return output
    
    # train all input, Input is the all input matrix
    def train(self, epochs, Input, Target):
        for i in range(epochs):
            Output = []
            for j in range(Input.shape[0]):
                output = self.train_once(input=Input[j], target=Target[j])
                Output.append(output)
            # print the weight and bias every 10 times
            if i % 10 == 0:
                print(f"{i} epoch, Weights:{self.weights}, Bias: {self.bias}")
            if Output == Target:
                print(f"Training completed in {epochs + 1} epochs")
                return True
        print(f"Training did not converage in {epochs} epochs")
        return False
    
    # predict for the new inputs
    def predict(self, Input):
        Output = []
        for input in Input:
            Output.append(self.forward(Input))
        return Output
