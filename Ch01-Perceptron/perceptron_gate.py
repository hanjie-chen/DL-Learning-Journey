from single_perceptron_class import Perceptron
from activation_function import step_fucntion, sigmoid_function, sign_function, ReLU_function

import numpy as np


Input = np.array([
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
])

Target_AND_Gate = np.array([
    1,
    0,
    0,
    0
])

Target_OR_Gate = np.array([
    1,
    1,
    1,
    0
])

learning_rate = 0.1

perceptron = Perceptron(Input.shape[1], step_fucntion, learning_rate)
perceptron.train(epochs=100, Input=Input, Target=Target_AND_Gate)
# Output = perceptron.predict(Input)
# print(f"here is my training Output: {Output}")
