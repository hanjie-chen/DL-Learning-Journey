# this file define the activation function that forward propagation use
# now these function only used for signle number deal with
import math

def step_fucntion(x):
    return 1 if x > 0 else 0

def sigmoid_function(x):
    return 1 / (1+math.exp(-x))

def ReLU_function(x):
    return x if x > 0 else 0
