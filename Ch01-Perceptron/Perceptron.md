# Perceptron

To begin with deep learning and nerual network, we should learn the base: perceptron

In the 1960s, Frank Rosenblatt propose a artificial neural network structure, named Perceptron. The basic unit of this netwrok can simplely describe as a formula, usually called single perceptron

for example:

![single perceptron](../images/single-perceptron.png)
$$
output =f_{a}( bias + \sum\limits_{i=1}^{3}input_{i}\times weight_{i})
$$
Here is the explain of the formula

- $output$: the output of the unit.

- $f_{a}$: the activation function of the unit.

- $bias$: the bias of the unit, one of parameter of the network

- $input_{i}$: the input singal

- $weight_{i}$: the weight of the input singal, the most important parameter of the netwrok.

In this example, we only have 3 inputs, when we have more inputs, such as we have n inputs, it will be like that:
$$
output =f_{a}( bias + \sum\limits_{i=1}^{n}input_{i}\times weight_{i})
$$

# How Perceptron Learn

Training means using existing data to get suitable weights and bias for the perceptron. The goal is to adjust these parameters so that the perceptron can correctly classify input data.

## The Learning Process

1. **Forward Pass**: First, we input data into the perceptron and calculate its output.
2. **Error Calculation**: We compare this output with the target (which is excepted output).
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
> for computer programming, it highly recommand because more easy to archieve; but hard for people understand
>
> and we can use Delta rule to conclude the second bias change rule.

By repeatedly applying this rule on many examples, the perceptron gradually improves its performance, adjusting its decision boundary to better classify the input data.