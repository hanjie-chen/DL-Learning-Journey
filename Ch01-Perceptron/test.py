import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
    # ...（之前的代码）

    # 修改绘制决策边界的方法
    def plot_decision_boundary(self, Input, Target, epoch):
        # 确保输入是二维数据
        if Input.shape[1] != 2:
            print("只能绘制二维数据的决策边界。")
            return

        # 清除当前的图形
        plt.clf()

        # 绘制数据点
        for i in range(len(Input)):
            if Target[i] == 1:
                plt.scatter(Input[i][0], Input[i][1], color='blue', marker='o', label='Class 1' if i == 0 else "")
            else:
                plt.scatter(Input[i][0], Input[i][1], color='red', marker='x', label='Class 0' if i == 0 else "")

        # 创建网格来绘制决策边界
        x_min, x_max = Input[:, 0].min() - 1, Input[:, 0].max() + 1
        y_min, y_max = Input[:, 1].min() - 1, Input[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = np.array([self.forward(point) for point in grid])
        Z = Z.reshape(xx.shape)

        # 绘制决策边界
        plt.contourf(xx, yy, Z, alpha=0.2, levels=np.linspace(0, 1, 3), colors=['red', 'blue'])
        plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--')

        plt.xlabel('Input 1')
        plt.ylabel('Input 2')
        plt.title(f'Decision Boundary at Epoch {epoch}')
        plt.legend()

        # 实时更新显示
        plt.pause(0.1)
