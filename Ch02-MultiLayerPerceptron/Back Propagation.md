# Back Propagation

let's consideer a general MLP topology, show as following:

![general-topology](../images/Ch02/general_nerual_network_topology.png)

Input layer n nodes and Hidden layer m nodes, and output layer l nodes

# Expend matrix for all layer

$Input_{1\times n}$
$$
Input_{1\times n} = 
\begin{bmatrix}
input_0 & input_1 & \cdots input_{n-1}
\end{bmatrix}_{1\times n}
$$


Hidden Layer $Weights^{hidden}$ and $Bias^{hidden}$ is 
$$
\begin{bmatrix}
Bias^{hidden}_{1\times m}\\
Weights^{hidden}_{n\times m}
\end{bmatrix}_{(n+1)\times m}=
\begin{bmatrix}
b_0^{hidden} & b_1^{hidden} & \cdots & b_{m-1}^{hidden} \\
w_{00}^{hidden} & w_{01}^{hidden} & \cdots & w_{0(m-1)}^{hidden} \\
w_{10}^{hidden} & w_{11}^{hidden} & \cdots & w_{1(m-1)}^{hidden} \\
\vdots & \vdots & \ddots & \vdots \\
w_{(n-1)0}^{hidden} & w_{(n-1)1}^{hidden} & \cdots & w_{(n-1)(m-1)}^{hidden}
\end{bmatrix}_{(n+1)\times m}
$$


Output Layer  $Weights^{output}$ and $Bias^{output}$ is 
$$
\begin{bmatrix}
Bias^{output}_{1\times l}\\
Weights^{output}_{m\times l}
\end{bmatrix}_{(m+1)\times l}=
\begin{bmatrix}
b_0^{output} & b_1^{output} & \cdots & b_{l-1}^{output} \\
w_{00}^{output} & w_{01}^{output} & \cdots & w_{0(l-1)}^{output} \\
w_{10}^{output} & w_{11}^{output} & \cdots & w_{1(l-1)}^{output} \\
\vdots & \vdots & \ddots & \vdots \\
w_{(m-1)0}^{output} & w_{(m-1)1}^{output} & \cdots & w_{(m-1)(l-1)}^{output}
\end{bmatrix}_{(m+1)\times l}
$$
$Output_{1\times l}$
$$
Output_{1\times l}=
\begin{bmatrix}
output_0 & output_1 & \cdots & output_{l-1}
\end{bmatrix}_{1\times l}
$$
$Target_{1\times l}$ is our trainning goal
$$
Target_{1\times l}=
\begin{bmatrix}
target_0 & target_1 & \cdots & target_{l-1}
\end{bmatrix}_{1\times l}
$$
$Loss$ is our loss function
$$
\begin{align}
Loss &= \frac{1}{l}\times(Target-Output)^2\\
      &=\frac{1}{l}\times\sum_{i=0}^{l-1}(target_{i}-output_{i})^2
\end{align}
$$

# Forward propagate



## Input --> Hidden

it should be
$$
Net_{1\times m}=f_a(Input_{1\times n} \cdot Weights_{n\times m}^{hidden} + Bias_{1\times m}^{hidden})
$$


