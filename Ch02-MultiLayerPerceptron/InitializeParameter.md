

## 随机初始化（Random Initialization）

原理：
从一个固定范围内的均匀分布中随机抽取权重值。

实现：
```python
def random_init(n_in, n_out):
    return np.random.uniform(-0.1, 0.1, (n_in, n_out))
```

优点：
- 简单易实现
- 打破对称性，使不同神经元学习不同的特征

缺点：
- 可能导致梯度消失或爆炸，特别是在深层网络中
- 初始化范围的选择较为随意，不同问题可能需要不同的范围

适用场景：
- 浅层网络
- 作为基准方法进行比较

## 正态分布初始化（Normal Distribution Initialization）

原理：
从均值为0，标准差为固定值（通常为0.01或0.1）的正态分布中抽取权重值。

实现：
```python
def normal_init(n_in, n_out, std=0.01):
    return np.random.normal(0, std, (n_in, n_out))
```

优点：
- 比均匀分布更符合自然现象
- 可以通过调整标准差来控制权重的范围

缺点：
- 固定的标准差可能不适用于所有层，特别是在深层网络中

适用场景：
- 一般用途
- 当其他方法效果不佳时可以尝试

## Xavier/Glorot初始化

原理：
保持每一层输入和输出的方差一致，防止信号在前向传播和反向传播过程中逐渐消失或爆炸。

实现：
```python
def xavier_init(n_in, n_out):
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_in, n_out))

# 或者使用正态分布版本
def normal_xavier_init(n_in, n_out):
    std = np.sqrt(2 / (n_in + n_out))
    return np.random.normal(0, std, (n_in, n_out))
```

优点：
- 在使用线性激活函数、sigmoid或tanh激活函数的网络中表现良好
- 有效缓解梯度消失问题

缺点：
- 对于ReLU激活函数可能不是最优的

适用场景：
- 使用sigmoid或tanh激活函数的网络
- 较深的网络

## He初始化

原理：
专为ReLU激活函数设计，考虑到ReLU在负半轴的输出为零的特性。

实现：
```python
def he_init(n_in, n_out):
    std = np.sqrt(2 / n_in)
    return np.random.normal(0, std, (n_in, n_out))
```

优点：
- 特别适合ReLU及其变体（如Leaky ReLU）
- 在深层网络中表现优异

缺点：
- 对于非ReLU激活函数可能不是最优的

适用场景：
- 使用ReLU或其变体作为激活函数的网络
- 深层网络

选择建议：

1. 如果你的MLP使用sigmoid或tanh激活函数，选择Xavier/Glorot初始化。
2. 如果你的MLP使用ReLU或其变体作为激活函数，选择He初始化。
3. 如果你不确定或者想要一个通用的起点，可以先尝试Xavier/Glorot初始化。
4. 如果你的网络较浅或者你想要一个基准比较，可以使用简单的随机初始化或正态分布初始化。

最后，记住初始化方法虽然重要，但它只是神经网络优化的一个方面。在实际应用中，你可能需要结合其他技术（如正则化、批量归一化等）并进行实验，以找到最适合你特定问题的方法。