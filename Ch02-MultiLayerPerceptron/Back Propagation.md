# Back Propagation

let's consideer a general full connection MLP topology, show as following:

![general-topology](../images/Ch02/general_nerual_network_topology.png)

Input layer n nodes and Hidden layer m nodes, and output layer l nodes

# Expand matrix for all layer

## $Input_{1\times n}$

$$
Input_{1\times n} = 
\begin{bmatrix}
input_0 & input_1 & \cdots & input_{n-1}
\end{bmatrix}_{1\times n}
$$

## $Weights^{hidden} \& Bias^{hidden}$ 

Hidden Layer weights and bias
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

> why we use this kind of format of Bias and Weights, it will be used for code and programing realize, we will see later Forward propagete

## $Weights^{output} \& Bias^{output}$

Output Layer  weights and bias
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
## $Output_{1\times l}$

$$
Output_{1\times l}=
\begin{bmatrix}
output_0 & output_1 & \cdots & output_{l-1}
\end{bmatrix}_{1\times l}
$$
## $Target_{1\times l}$ 

our trainning goal
$$
Target_{1\times l}=
\begin{bmatrix}
target_0 & target_1 & \cdots & target_{l-1}
\end{bmatrix}_{1\times l}
$$
## $Loss$

our loss function
$$
\begin{align}
Loss &= \frac{1}{l}\times(Target-Output)^2\\
      &=\frac{1}{l}\times\sum_{i=0}^{l-1}(target_{i}-output_{i})^2
\end{align}
$$

$f_a$ is activation funciton

# Forward propagate

let's go through again for the forward propagete and use some special symbol to mark some intermediate calcualtion results. This symbol can help us to get succinct result in back propagation

## Input --> Hidden

let's first define $Net_{1\times m}^{hidden}$ which is the input value of the Hidden layer actication function
$$
Net_{1\times m}^{hidden} \triangleq Input_{1\times n} \cdot Weights_{n\times m}^{hidden} + Bias_{1\times m}^{hidden}
$$

and then we define $Hidden_{1\times m}$ which is the output of the Hidden layer
$$
Hidden_{1\times m} \triangleq f_a(Net_{1\times m}^{hidden})
$$
as finally we will use matrix multiplication to express this process (because it more convinence to code and programing). we will have following:
$$
\begin{align}

Net_{1\times m}^{hidden} 

&\triangleq 
Input_{1\times n} \cdot Weights_{n\times m}^{hidden} + Bias_{1\times m}^{hidden}

\\ 
\\

&= 
\begin{bmatrix}
1 & Input_{1\times n}
\end{bmatrix}_{1\times n+1}
\cdot
\begin{bmatrix}
Bias^{hidden}_{1\times m}\\
Weights^{hidden}_{n\times m}
\end{bmatrix}_{(n+1)\times m}

\\
\\

&=
\begin{bmatrix}
1 & input_0 & input_1 & \cdots & input_{n-1}
\end{bmatrix}_{1\times n+1}
\cdot
\begin{bmatrix}
b_0^{hidden} & b_1^{hidden} & \cdots & b_{m-1}^{hidden} \\
w_{00}^{hidden} & w_{01}^{hidden} & \cdots & w_{0(m-1)}^{hidden} \\
w_{10}^{hidden} & w_{11}^{hidden} & \cdots & w_{1(m-1)}^{hidden} \\
\vdots & \vdots & \ddots & \vdots \\
w_{(n-1)0}^{hidden} & w_{(n-1)1}^{hidden} & \cdots & w_{(n-1)(m-1)}^{hidden}
\end{bmatrix}_{(n+1)\times m}

\\
\\

&=
\begin{bmatrix}
b_0^{hidden}+\sum_{i=0}^{n-1} (input_i\times w_{i0}^{hidden}) &
\cdots &
b_{m-1}^{hidden}+\sum_{i=0}^{n-1} (input_i\times w_{i(m-1)}^{hidden})
\end{bmatrix}_{1\times m}

\\
\\

&\triangleq 
\begin{bmatrix}
net_0^{hidden} & net_1^{hidden} & \cdots & net_{m}^{hidden}
\end{bmatrix}_{1\times m}

\end{align}
$$
in summary we have
$$
Hidden_{1\times m} = f_a(Net_{1\times m}^{hidden}) \quad \Rightarrow \quad hidden_i = f_a(net_i^{hidden}) \quad i \in [0, m-1]
$$


## Hidden --> Output

similar to the above, we have
$$
\begin{align}
Net_{1\times l}^{output} 
&\triangleq 

Hidden_{1\times m} \cdot Weights_{m\times l}^{output}+Bias_{1\times l}^{output}
\\
\\
&=
\begin{bmatrix}
1 & Hidden_{1\times m}
\end{bmatrix}_{1\times (m+1)}
\cdot
\begin{bmatrix}
Bias_{1\times l}^{output} \\
Weights_{m\times l}^{output}
\end{bmatrix}_{(m+1)\times l}

\\
\\
&=
\begin{bmatrix}
1 & hidden_0 & hidden_1 & \cdots & hiddens_{m-1}
\end{bmatrix}_{1\times (m+1)}
\cdot
\begin{bmatrix}
b_0^{output} & b_1^{output} & \cdots & b_{l-1}^{output} \\
w_{00}^{output} & w_{01}^{output} & \cdots & w_{0(l-1)}^{output} \\
w_{10}^{output} & w_{11}^{output} & \cdots & w_{1(l-1)}^{output} \\
\vdots & \vdots & \ddots & \vdots \\
w_{(m-1)0}^{output} & w_{(m-1)1}^{output} & \cdots & w_{(m-1)(l-1)}^{output}
\end{bmatrix}_{(m+1)\times l}
\\
\\
&=
\begin{bmatrix}
b_0^{output}+\sum_{i=0}^{m-1}hidden_i\times w_{i0}^{output} & \cdots & b_{l-1}^{output}+\sum_{i=0}^{m-1}hidden_i\times w_{i(l-1)}^{output}
\end{bmatrix}_{1\times l}
\\
\\
&\triangleq
\begin{bmatrix}
net_0^{output} & net_1^{output} & \cdots & net_{l-1}^{output}
\end{bmatrix}_{1\times l}
\end{align}
$$
In summary we have
$$
Output_{1\times l} = f_a(Net_{1\times l}^{output}) \quad \Rightarrow \quad output_i = f_a(net_i^{output}) \quad i \in [0, l-1]
$$


# Expand Loss function

don't forget our goal, we want to find a way to adjust weights and bias to make the loss function smallest.

now, we have expanded forward propagate and, let's try to use exist matrix and variable to expand the loss function
$$
\begin{align}
Loss 
&= \frac{1}{l}\times(Target-Output)^2\\
&=\frac{1}{l}\times\sum_{i=0}^{l-1}(target_{i}-output_{i})^2 \\
&= \frac{1}{l}\times\sum_{i=0}^{l-1}(target_{i}-f_a(net_i^{output}) )^2 \\
&= \frac{1}{l}\times\sum_{i=0}^{l-1}(target_{i}-f_a(b_i^{output}+\sum_{j=0}^{m-1}hidden_j\times w_{ji}^{output}) )^2 \\
&= \frac{1}{l}\times\sum_{i=0}^{l-1}(target_{i}-f_a(b_i^{output}+\sum_{j=0}^{m-1}(f_a(net_j^{hidden}) )\times w_{ji}^{output}) )^2 \\
&= \frac{1}{l}\times\sum_{i=0}^{l-1}(target_{i}-f_a(b_i^{output}+\sum_{j=0}^{m-1}(f_a(b_j^{hidden}+\sum_{k=0}^{n-1} (input_k\times w_{kj}^{hidden})) )\times w_{ji}^{output}) )^2
\end{align}
$$
as we alrady full expand the loss function, now let't confirm which is variable and which is constant.

as we want to adjust the weights and bias, so the variable is 
$$
b_i^{output}, b_j^{hidden}, w_{kj}^{hidden}, w_{ji}^{output}
$$
constant is (because we focues on the 1 input, so input is also constant)
$$
target_i, input_k
$$
so, finally the loss function is
$$
Loss = Loss(b_i^{output}, b_j^{hidden}, w_{kj}^{hidden}, w_{ji}^{output})
$$
now our problem become: how we adjust the variable of a Multivariate functions, which can make the function value smallest?

mostly we use Gradient Descent

# Gradient Descent
