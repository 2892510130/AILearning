## Chapter 5 : Multilayer Perceptrons
- Activation Function: relu, sigmoid, tanh ($\frac{1 - \exp(-2x)}{1 + \exp(-2x)}$)
- Numerical stability: vanish and explode are common
  - Symmetry: linear layer and conv (with no share weight) layer are symmetric so we can not tell apart from different weight and try to explain it (for example 2 hidden unit with same initial value, they will update the same way), so we need to **Bread the Symmetry** (like using a dropout)
  - Xavier initilization: get from distrubution of zero mean and variance $\sigma = \sqrt{2 / (n_{in} + n_{out})}$
  - Dropout, shared param...
- (Rolnick et al., 2017) has revealed that in the setting of label noise, neural networks tend to fit cleanly labeled data **first** and only subsequently to interpolate the mislabeled data.
  - so we can use early stop once error on val is minimal or the patience hit. usually combined with regularization.
- Dropout:
  - $h^{'} = \left \{ 
  \begin{array}{lll}
  & 0, p \\
  & \frac{h}{1-p}, 1-p
  \end{array} 
  \right .$, now $E[h^{'}] = E[h]$
  - We do not use dropout in test, except we want to know the uncertainty of the model output (by comparing different dropout)
  - Use lower p in lower layer (to get lower feature), higher p in higher layer