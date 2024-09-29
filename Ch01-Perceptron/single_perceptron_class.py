import numpy as np
import matplotlib.pyplot as plt

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
        print(output, end="")
        return output
    
    # train all input, Input is the all input matrix
    def train(self, epochs, Input, Target):
        # start plt interactive mode
        plt.ion()

        for i in range(epochs):
            Output = []
            for j in range(Input.shape[0]):
                output = self.train_once(input=Input[j], target=Target[j])
                Output.append(output)
            
            # print the weight and bias every time, it used to be 10 times but I find that it usually only not more than 10 times
            print(f" now {i} epoch, Weights:{self.weights}, Bias: {self.bias}")
            
            # draw decisiion boundary every epoch
            self.plot_decision_boundary(Input, Target, i)

            # if it already finished
            if np.array_equal(np.array(Output), Target):
                print(f"Training completed in {i} epochs")
                break
        # if for iteration finished
        else:
            print(f"Training did not converage in {epochs} epochs")
        
        # close matplot interactive mode
        plt.ioff()
        # show last picture
        plt.show()
    
    # predict for the new inputs
    def predict(self, Input):
        # print(f"current weights is {self.weights}, current bias is {self.bias}, output is {np.dot(Input, self.weights)+self.bias}")
        Output = []
        for j in range(Input.shape[0]):
            output = self.forward(Input[j])
            Output.append(output)
        return Output
    
    # draw picture, provide by o1
    def plot_decision_boundary(self, Input, Target, epoch):
        # 2-demension data only
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
