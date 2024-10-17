# Multiple Layer Perceptron

so, we will continue explore the XOR Gate, and we will try to use a simple 2 layer(hidden layer and output layer, these 2 layer all have weight and bias parameter) perceptron to, just like following:

![png](../images/Ch02/XOR-Neural-Network.png)

we have 2 node to input and 3 hidden node and 1 output node.

and now, let's focus on this 

# Forward Pass

for the forward pass, it very similar to the perceptorn, just use the weights and bias to calcualte the vaule and move to next steps. A little bit different is it become the array calculate.

## 1 input forward pass

### Input --> Hidden

$$
Hidden_{1\times3} = f_a(Input_{1\times2} \cdot Weights_{2\times3}^{hidden}+Bias_{1\times3}^{hidden})
$$

### Hidden --> Output

$$
Output_{1\times1}=f_a(Hidden_{1\times3}\cdot Weights_{3\times1}^{output}+Bias_{1\times1}^{output})
$$

$f_a$: the activation function

$Input_{1\times3}$: 1 group input

$Weights_{2\times3}^{hidden}$: hidden layer perceptron weights

$Bias_{1\times3}^{hidden}$: hidden layer perceptron bias

$Weights_{1\times3}^{output}$: output layer perceptron weights

$Bias_{1\times1}^{output}$: output layer perceptorn bias

we will consider the batch input forward pass after, which Input matrix will expand to $Input_{4\times2}$

# How MLP learn

## The trouble about MLP learn

But when it comes to the learning process, we have trouble.

Compared with the single perceptron, where the output can be directly influenced by the weight and bias, the MLP is different. The output is decided not only by the output layer perceptron but also by the hidden layer perceptrons.

## Something sparked by old Delta rule

Recall the old Delta rule, we used the following guide to create the delta rule:

- If the perceptron's prediction is too high, we need to decrease the weights.
- If the prediction is too low, we need to increase the weights.
- The amount of adjustment should be proportional to the error and the input value.

So the important things are error and input.

## Extending the idea to MLP

While the Delta rule works well for single layer perceptrons, we need to extend this idea for MLPs. We still want to adjust our weights and bias based on the error, but now we have multiple layers of weights to consider.

The key question becomes: How do we quantify the error for each layer, especially the hidden layers?

To answer this, let's start by looking at the output layer. We can easily calculate the error here by comparing our prediction with the actual target value. But how do we measure this error precisely?

## Introducing the Loss Function

This is where the concept of a loss function comes in. A loss function helps us quantify how "wrong" our predictions are. It gives us a single number that represents the error of our entire network.

One of the most common and intuitive loss functions is the Mean Squared Error (MSE). Which is
$$
\begin{align}
Error &= \frac{1}{N}\times(Target-Output)^2\\
      &=\frac{1}{N}\times\sum_{i=1}^{N}(target_{i}-output_{i})^2
\end{align}
$$
Obviously

$N$: is the number of output sample and target sample

$Output, Target$: Output is our predictions and Target is our goal



By using MSE, we now have a clear, numerical representation of our network's performance.

The next question is: How do we use this loss function to adjust the weights in all layers of our network, including the hidden layers? This leads us to the concept of back propagation, which we'll explore next.

# Back propagation

our goal is simple, we should get the minum of the loss function, so we just analyse the loss function as the mathematical.

[back propagation mathematical derivation](./BackPropagation.md)

> this file derivate every 

# Talk is cheap, show me the code

again, we come to code part







# Next step

我们应该继续学习XOR问题。这是一个非常重要的步骤，原因如下：

1. XOR问题的重要性：
   
2. 理解线性可分与非线性可分
   
3. 引入多层感知器
   
4. 理解隐藏层的作用：
   
5. 学习反向传播
   
6. 可视化和理解：
   XOR问题在二维平面上很容易可视化，这有助于我们直观地理解神经网络如何学习复杂的决策边界。

接下来的步骤可以是：

1. 尝试用单个感知器解决XOR问题，观察其失败。
2. 设计一个简单的多层感知器（通常两层足够）来解决XOR问题。
3. 实现前向传播算法。
4. 学习并实现反向传播算法。
5. 训练网络并观察其如何成功解决XOR问题。
6. 可视化结果，包括决策边界的变化过程。

通过这个过程，我们可以自然地过渡到更复杂的神经网络结构和算法，为进一步学习深度学习打下坚实的基础。您觉得这个计划如何？我们可以从哪一步开始？