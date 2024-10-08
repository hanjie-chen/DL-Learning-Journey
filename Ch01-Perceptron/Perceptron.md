# Perceptron

Neural networks are a fundamental concept in artificial intelligence and machine learning. They are inspired by the human brain's structure and function, consisting of interconnected nodes (neurons) that process and transmit information. Understanding neural networks is crucial to delve into the field of deep learning. 

In this note, we'll focus on perceptrons, which are the basic building blocks of neural networks. We'll explore their structure, how they learn, and their capabilities and limitations.

# Perceptron Structure and Forward Pass

In the 1960s, Frank Rosenblatt proposed an artificial neural network structure called the Perceptron. The basic unit of this network can be described by a simple formula, known as a single perceptron.

for example:

![single perceptron](../images/Ch01/single-perceptron.png)
$$
output =f_{a}( bias + \sum\limits_{i=1}^{3}input_{i}\times weight_{i})
$$
Here is the explain of the formula

- $output$: the output of the unit.

- $f_{a}$: the activation function of the unit.

- $bias$: the bias of the unit, one of parameter of the network

- $input_{i}$: the input signal

- $weight_{i}$: the weight of the input signal, the most important parameter of the network.

In this example, we only have 3 inputs, when we have more inputs, such as we have n inputs, it will be like that:
$$
output =f_{a}( bias + \sum\limits_{i=1}^{n}input_{i}\times weight_{i})
$$

The bias term is crucial as it allows the perceptron to make non-zero decisions even when all inputs are zero, effectively shifting the decision boundary.

# Activation Functions $f_a$

Activation functions introduce non-linearity into the neuron's output, allowing neural networks to learn complex patterns. Let's explore some common activation functions:

## Common Activation Functions

Let's look at a few simple activation functions:

### 1. Step Function

This is the simplest activation function, often used in the original perceptron model:

$$
f(x) = \begin{cases} 
1 & \text{if } x \geq \text{0} \\
0 & \text{otherwise}
\end{cases}
$$

It's like a simple on/off switch, useful for binary classification tasks.

### 2. Sigmoid Function

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

This function "squashes" the input into a range between 0 and 1, making it useful for probabilities or binary classification.

### 3. Rectified Linear Unit (ReLU)

$$
f(x) = \max(0, x)
$$

This function is simple but very effective. It outputs the input directly if it's positive, and 0 otherwise.

## Impact on Learning

The choice of activation function can affect how our perceptron learns:

- With a step function, the perceptron makes clear binary decisions.
- With a sigmoid function, we can get more nuanced, probability-like outputs.
- ReLU allows for faster learning in many cases.

In our upcoming code example, we'll use both the step function and the sigmoid function to see how they affect our perceptron's behavior.

# How Perceptron Learn

Training means using existing data to get suitable weights and bias for the perceptron. The goal is to adjust these parameters so that the perceptron can correctly classify input data.

## The Learning Process

1. **Forward Pass**: First, we input data into the perceptron and calculate its output.
2. **Error Calculation**: We compare this output with the target (which is the expected output).
3. **Parameter Update**: Based on the error, we adjust the weights and bias.

So, the key question is: How do we adjust the weights and bias?

## Intuition Behind the Learning Rule

The basic idea is simple and intuitive:
- If the perceptron's prediction is too high, we need to decrease the weights.
- If the prediction is too low, we need to increase the weights.
- The amount of adjustment should be proportional to the error and the input value.

This intuition leads us to the Delta rule, also known as the Perceptron Learning Rule.

## The Delta Rule

The Delta rule, proposed by Frank Rosenblatt along with the perceptron, provides a method to update weights and bias:

For weights:
$$
\Delta weight_{i} = learning\_rate * (target - output) * input_{i} \\
weight_{i} = weight_{i} + \Delta weight_{i}
$$

For bias:
$$
\Delta bias = learning\_rate * (target - output) * 1 \\
bias = bias + \Delta bias
$$

Here's what each part means:
- $learning\_rate$: A small positive number that controls the size of each adjustment.
- $(target - output)$: The prediction error. This determines the direction and magnitude of the update.
- $input_{i}$: The input value. This ensures that inputs contributing more to the output are adjusted more.

This rule ensures that:
1. Weights associated with larger inputs are adjusted more.
2. Larger errors lead to larger adjustments.
3. The direction of adjustment (increase or decrease) is determined by the sign of the error.

> A more explain about the different between $\Delta weight$ and $\Delta bias$ 
>
> if we treat the bias as $weight_{0}$, and there is a default $input_{0}=1$, then the formula will be more succinct
> $$
> output =f_{a}(\sum\limits_{i=0}^{3}input_{i}\times weight_{i})
> $$
> For computer programming, it is highly recommended because it's easier to achieve, but it maybe harder for people to understand
>
> and we can use Delta rule to conclude the second bias change rule.

By repeatedly applying this rule on many examples, the perceptron gradually improves its performance, adjusting its decision boundary to better classify the input data.

# Learning Rate

The learning rate is a hyperparameter that determines the step size of each parameter update. Choosing an appropriate learning rate is crucial: 

- A large learning rate may lead to faster convergence but can cause parameter oscillation. 
- A small learning rate results in slower updates but may provide more precise convergence.

For beginners, it's common to start with a static learning rate (e.g., 0.1). More advanced techniques involve dynamic learning rates, which we'll explore in later chapters.

# Talk is cheap, show me the code

Let's implement a perceptron to realize the AND and OR logic gates.

## About Initialize the parameters

about initialize the the weights and bias, we first use simple way. 

for weight, we use randomly and uniform distribution, such as `[-0.5, 0.5]`, and for bias we set it to 0

more complex initialize will discuss in next chapter

## Use perceptron to realize the AND / OR Gate

### Input and Target data

```python
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
```

### Single Perceptron class

```python
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
            
            # print the weight and bias every time, it used to be 10 times but I find that it usually only not more than 10 times
            print(f"{i} epoch, Weights:{self.weights}, Bias: {self.bias}")
            
            if np.array_equal(np.array(Output), Target):
                print(f"Training completed in {i} epochs")
                break
            else:
                print(f"Training did not converage in {epochs} epochs")
    
    # predict for the new inputs
    def predict(self, Input):
        Output = []
        for input in Input:
            Output.append(self.forward(input))
        return Output
```

### activation fucntion

```python
# this file define the activation function that forward propagation use
# now these function only used for signle number deal with
import math

def step_fucntion(x):
    return 1 if x > 0 else 0

def sign_function(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    elif x < 0:
        return -1

def sigmoid_function(x):
    return 1 / (1+math.exp(-x))

def ReLU_function(x):
    return x if x > 0 else 0
```

### Run and draw picture

to run the code, we should:

```python
learning_rate = 0.1

perceptron = Perceptron(Input.shape[1], step_fucntion, learning_rate)
perceptron.train(epochs=100, Input=Input, Target=Target_OR_Gate)
Output = perceptron.predict(Input)
print(f"here is my training Output: {Output}")
```

for we can draw the pciture to show more directly about the training, we should do a little bit change for the Perceptron class

```python
# previous code
import matplotlib.pyplot as plt

class Perceptron():
    """
    A simple implementation of the Perceptron algorithm
    """

    # ... exist code
    
    # train all input, Input is the all input matrix
    def train(self, epochs, Input, Target):
        # start plt interactive mode
        plt.ion()
        
        # ... exist code
        
        # close matplot interactive mode
        plt.ioff()
        # show last picture
        plt.show()
    
    # ... exist code
    
    # draw picture, provide by o1
    def plot_decision_boundary(self, Input, Target, epoch):
        # 2-dimension data only
        if Input.shape[1] != 2:
            print("sorry, only can draw 2-dimension data")
            return

        # clear all
        plt.clf()

        # draw point
        for i in range(len(Input)):
            if Target[i] == 1:
                plt.scatter(Input[i][0], Input[i][1], color='blue', marker='o', label='Class 1' if i == 0 else "")
            else:
                plt.scatter(Input[i][0], Input[i][1], color='red', marker='x', label='Class 0' if i == 0 else "")

        # create grid
        x_min, x_max = Input[:, 0].min() - 1, Input[:, 0].max() + 1
        y_min, y_max = Input[:, 1].min() - 1, Input[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = np.array([self.forward(point) for point in grid])
        Z = Z.reshape(xx.shape)

        # boundary
        plt.contourf(xx, yy, Z, alpha=0.2, levels=np.linspace(0, 1, 3), colors=['red', 'blue'])
        plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--')

        plt.xlabel('Input 1')
        plt.ylabel('Input 2')
        plt.title(f'Decision Boundary at Epoch {epoch}')
        plt.legend()

        # renew every 0.3 s
        plt.pause(0.3)

```

### End of the example

in this example, we use perceptron to realize the AND and OR gate, and we use `numpy` to manipulate the matrix multiplication.

But in our activation function, it only deal with 1 number, and in our Perceptron class, the `adjust_weight` , `adjust_bias` and `train_once` also just deal with single number.

If we use large-scale use the `np.array` these operation will be more succinct code, we will try in next chapter / example.

## Limitations: XOR Gate

Having already used a single perceptron to implement the AND and OR gates, let's attempt to implement the XOR gate

so we have the Input and Target:

```python
Input = np.array([
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
])
Target_XOR_Gate = np.array([
    0,
    1,
    1,
    0
])
```

But when we train, we get following:

```powershell
0 epoch, Weights:[ 0.33238207 -0.33962529], Bias: 0.0
1 epoch, Weights:[ 0.33238207 -0.23962529], Bias: 0.0
2 epoch, Weights:[ 0.23238207 -0.23962529], Bias: 0.0
3 epoch, Weights:[ 0.23238207 -0.13962529], Bias: 0.0
4 epoch, Weights:[ 0.13238207 -0.13962529], Bias: 0.0
5 epoch, Weights:[ 0.13238207 -0.03962529], Bias: 0.0
...
99 epoch, Weights:[ 0.13238207 -0.03962529], Bias: 0.0
Training did not converage in 100 epochs
here is my training Output: [1, 1, 0, 0]
```

when it comes to a set of weights and bias, it not be trained but stuck.

let us try to analyze this situation, we use the Wights `[ 0.13238207, -0.03962529]` and bias `0`

### Turn 1

1. forward process
   $$
   f_a([1, 1] * [ 0.13238207, -0.03962529] + 0 = 0.09275677999999998) = 1
   $$

2. Delta rule
   $$
   \Delta weight = 0.1 * (0 - 1) * [1, 1] = [-0.1, -0.1] \\
   \Delta bias = 0.1 * (0 - 1) * 1 = -0.1
   $$

3. current weights and bias
   $$
   weight = [0.03238207, -0.13962529]\\
   bias = -0.1
   $$

### Turn 2

1. forward process: 
   $$
   f_a([1, 0] * [0.03238207, -0.13962529] + (-0.1) = -0.06761793) = 0
   $$
   
2. Delta rule: 
   $$
   \Delta weight = 0.1 * (1 - 0) * [1, 0] = [0.1, 0] \\
   \Delta bias = 0.1 * (1 - 0) * 1 = 0.1
   $$
   
3. weights and bias: 
   $$
   weight = [0.13238207, -0.13962529]\\
   bias = 0
   $$
   

### Turn 3

1. forward process: 
   $$
   f_a([0, 1] * [0.13238207, -0.13962529] + 0 = -0.13962529) = 0
   $$
   
2. Delta rule: 
   $$
   \Delta weight = 0.1 * (1 - 0) * [0, 1] = [0, 0.1] \\
   \Delta bias = 0.1 * (1 - 0) * 1 = 0.1
   $$
   
3. weight and bias: 
   $$
   weight = [0.13238207, -0.03962529]\\
   bias = 0.1
   $$
   

### Turn 4

1. forward process: 
   $$
   f_a([0, 0] * [0.13238207, -0.03962529] + 0.1 = 0.1) = 1
   $$
   
2. Delta rule: 
   $$
   \Delta weight = 0.1 * (0 - 1) * [0, 0] = [0, 0] \\
   \Delta bias = 0.1 * (0 - 1) * 1 = -0.1
   $$
   
3. weight and bias: 
   $$
   weight = [0.13238207, -0.03962529]\\
   bias = 0
   $$
   

In this process, we can see a full process that how the weights and bias change and change back.

And one more thing we should notice is that, why the parameter will be change back, because we use only 1 input to update the parameter, and then use the updated paramter to forward next input.

If we use batch inputs and adjust the weights after evaluating all data points, we might think it could help. However, since the XOR problem is fundamentally non-linear, even batch training won't enable a single-layer perceptron to solve it.

### The essence of XOR problem

While a single perceptron can successfully learn linear decision boundaries (like AND and OR gates), it fails to learn the XOR function. This is because the XOR problem is not linearly separable.

Which means we cannot draw a single straight line to separate the classes. A single-layer perceptron can only learn linear decision boundaries, so it fails to solve the XOR problem. As shown in the follwoing picture

![XOR](../images/Ch01/XOR_Gate_Failed.png)

# Deeper think about the XOR

if we have AND Gate and OR Gate, actually we can construct a XOR gate, like following graph:

![AN-OR](../images/Ch01/AND-OR-Gate.png)

so that means use MLP(multiple layer perceptron) may can resolve the problem, and let's jump into Ch02, which will introduce MLP, and most important BP algorithm

## Next step

we will go to build a multiple layer perceptorn try to resolve the XOR problem in Ch02-MultiLayerPerceptron
