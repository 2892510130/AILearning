## Chapter 18 : Gaussian Processes
- Definition: any subset of a GP is multivariate Gaussian distribution. It is defined solely on its mean and covariance function.
  - For a given mean function $m(x)$, a covariance function $k(x,x^{'})$, noise $w \sim \mathcal N(0, \sigma^2)$, conditioned on Data $(X,y)$, for new data point $x^*$, we have:
    $$ 
    \begin{array}{lll}
    \mu(x^*) &=& m(x^*) + k(x^*, X) \left [ k(X,X) + \sigma^2 I\right]^{-1}(y - m(x^*)) \\
    v(x^*)   &=& k(x^*,x^*) - k(x^*, X)\left [ k(X,X) + \sigma^2 I\right]^{-1}k(X, x^*)
    \end{array}
    $$
- Priors:
  - For any function with linear parameters, which takes this form $f(x) = \sum_i w_i \phi_i(x)$, is a GP, where $w \sim \mathcal N(u, S)$. And the mean function is $m(x) = u^{\top}\phi(x)$, covariance function is $k(x, x^{'}) = \phi(x)^{\top}S\phi(x)$. Any GP can take this form, but $\phi(x)$ maybe infinite dimension, for example, RBF kernel.
    $$
    \begin{array}{lll}
    k(x, x^{'}) &=& E\left[(f(x)-E[f(x)])(f(x^{'})-E[f(x^{'})])\right] \\
                &&\rightarrow  E[f(x)] = E[W^T\phi(x)] = U^T\phi(x), W=[w_0, w_1, \ldots], U = [u_0, u_1, \ldots] \\
                &=& E\left[ (W^T-U^T) \phi(x) (W^T-U^T) \phi(x^{'}) \right] \\
                &&\rightarrow  \hat W = W - U \\
                &=& E\left[ \hat W^T \phi(x) \hat W^T \phi(x^{'}) \right] \\
                &=& E\left[ \phi(x)^T \hat W \hat W^T \phi(x^{'}) \right] \\
                &=& \phi(x)^T E[\hat W \hat W^T] \phi(x^{'}) \\
                &=& \phi(x)^T S \phi(x^{'})
    \end{array}
    $$
    The above prove use the fact that $w_i$ is independent (iid).
  - *RBF kernel* (stationary kernel): can be derived with the above form
    $$f(x) = \sum_{i=1}^Jw_i\phi_i(x)，w_i \sim \mathcal N\left (0, \frac{\sigma^2}{J} \right), \phi_i(x) = \exp \left(-\frac{(x-c_i)^2}{2l^2} \right)$$
    push the dimension to infinity, we get:
    $$k(x,x^{'}) = \sigma^2 \exp \left( -\frac{(x-x^{'})^2}{2l^2} \right)$$
  - *Neural Network Kernel* is non-stationary kernel:
    $$f(x) = b + \sum_{i=1}^Jv_ih(x;u_i), b\sim\mathcal N(0,\sigma_b^2), v\sim\mathcal N(0,\sigma_v^2/J), u\sim \mathcal N(0,\Sigma).$$
    We get:
    $$k(x, x^{'}) = \frac{2}{\pi} \sin\left( \frac{2\bar x^{\top} \Sigma \bar x^{'}}{\sqrt{(1+2\bar x^{\top} \Sigma \bar x)(1+2\bar x^{'\top} \Sigma \bar x^{'})}} \right)$$
- Inference:
  - With the prior, we get $y|f,x \sim \mathcal N \left(0, k(X,X)+\sigma^2I \right)$ for $y=f(x)+\epsilon$, $f(x) \sim \mathcal{GP}, \epsilon \sim \mathcal N(0, \sigma^2)$. So:
    $$p(y|f,x) = \frac{1}{(2\pi)^{n/2}|k(X,X)+\sigma^2I|^{1/2}} \exp \left(-\frac{1}{2}y^T(k(X,X)+\sigma^2I)^{-1}y \right)$$
    Take log on it, we get:
    $$\log p(y|f,x) = -\frac{1}{2}(k(X,X)+\sigma^2I)^{-1}y - \frac{1}{2}|k(X,X)+\sigma^2I| - \frac{n}{2}\log2\pi$$