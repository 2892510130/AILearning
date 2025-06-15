### Chapter 3 : Linear Neural Network
- Minibatch stochastic gradient descent (小批量随机梯度下降)
- 一般的训练过程
  - model.forward() 与 y_hat 做差，然后反向传播，优化器根据导数去更新参数
- Machine Learning Concept
  - lasso regression: l1 norm; ridge regression: l2 norm;