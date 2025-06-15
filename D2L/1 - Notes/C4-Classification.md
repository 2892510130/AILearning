## Chapter 4 : Classification
- softmax:
  $y_i = \frac{\exp(o_i)}{\sum_j \exp(o_j)}$, often minus max(oj) to get numerical stable
- Information theory
  - cross-entropy loss：$l(y, \hat y) = - \sum y_i * \log(\hat y_i)$
  - amount of information $\log{\frac{1}{P(j)}} = - \log{P(j)}$ 
  - entorpy $H[P] = \sum -P(j) \log{P(j)}$
  - cross-entorpy $H(P, Q) = \sum -P(j) \log{Q(j)}, ~ P=Q \rightarrow H(P, Q) = H(P, P) = H(P)$. In pytorch, F.cross_entropy will do the softmax for you.
- Image Classification Rules:
  - image stored in (channel, height, weight) manner.
- Distrubution shift:
  - Covariate Shift (feature shift): $p(x) \neq q(x), p(y|x) = q(y|x)$
    - For example: p(x) and q(x) are features of oral and urban house, y is the price, we assume the feature and label relation is the same
    - Method: weighted by $$\beta(x) = p(x) / q(x) \rightarrow \int\int l(f(x), y)p(y|x)p(x)dxdy = \int\int l(f(x), y)q(y|x)q(x) \frac{p(x)}{q(x)}dxdy \rightarrow \sum_i \beta_i l(f(x_i), y_i)$$ $\beta$ can be obtained with logistic regression.
  - Label Shift, $p(y) \neq q(y), p(x|y) = q(x|y)$, the same method $\beta(y) = p(y) / q(y)$, but now $q(y)$ is hard to get, we need compute a confusion matrix on the val data then use the model to pridcit the distrubution of the $q(y)$
  - Concept Shift (the concept of the label)