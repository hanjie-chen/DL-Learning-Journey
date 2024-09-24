import numpy as np
import matplotlib.pyplot as plt

# 定义激活函数
def step_function(x):
    return np.where(x >= 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 定义Perceptron类
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def forward(self, inputs, activation_function):
        return activation_function(np.dot(inputs, self.weights) + self.bias)

    def train(self, X, y, epochs, activation_function):
        for _ in range(epochs):
            for inputs, target in zip(X, y):
                prediction = self.forward(inputs, activation_function)
                error = target - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

# 准备数据
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
y_or = np.array([0, 1, 1, 1])

# 训练AND感知器
perceptron_and = Perceptron(input_size=2)
perceptron_and.train(X, y_and, epochs=100, activation_function=step_function)

# 训练OR感知器
perceptron_or = Perceptron(input_size=2)
perceptron_or.train(X, y_or, epochs=100, activation_function=step_function)

# 测试函数
def test_perceptron(perceptron, X, y, gate_type):
    predictions = [perceptron.forward(x, step_function) for x in X]
    accuracy = np.mean(predictions == y)
    print(f"{gate_type} Gate Results:")
    print(f"Inputs: {X}")
    print(f"Expected: {y}")
    print(f"Predicted: {predictions}")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")

# 测试AND和OR感知器
test_perceptron(perceptron_and, X, y_and, "AND")
test_perceptron(perceptron_or, X, y_or, "OR")

# 可视化决策边界
def plot_decision_boundary(perceptron, X, y, title):
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    
    x1 = np.linspace(-0.5, 1.5, 100)
    x2 = -(perceptron.weights[0] * x1 + perceptron.bias) / perceptron.weights[1]
    
    plt.plot(x1, x2, 'k-', lw=2)
    plt.fill_between(x1, x2, 1.5, alpha=0.1)
    plt.fill_between(x1, x2, -0.5, alpha=0.1)
    
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.title(title)
    plt.show()

plot_decision_boundary(perceptron_and, X, y_and, "AND Gate Decision Boundary")
plot_decision_boundary(perceptron_or, X, y_or, "OR Gate Decision Boundary")
