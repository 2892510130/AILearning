## Chapter 7 : CNN
1. **Invariance**: translation equivariance, locality -> The earliest layers should respond similarly to the same patch and focus on local regions.
2. **Convolution**: math is $(f * g)(i, j) = \sum_a \sum_b f(a, b)  g(i - a, j - b)$, remind that **cross-correlation** is $(f * g)(i, j) = \sum_a \sum_b f(a, b)  g(i + a, j + b)$
   - The difference is not important as we will learn the kernel, `k_conv_learned = k_corr_learned.T`, or `conv(X, k_conv_learned) = corr(X, k_corr_learned)`
3. **Receptive Field**： for any element (tensors on the conv layer) x, all the elements that may effect x in the previous layers in the forward population.
4. **Padding, Stride**: $\lfloor (n_h - k_h + p_h + s_h) / s_h \rfloor \times \lfloor (n_w - k_w + p_w + s_w) / s_w \rfloor$, often `p_h = k_h - 1`, the same for `p_w`. `p_h = p_h_upper + p_h_lower`
5. **Channel**:
   - multi in $c_i$ -> kernel must also have the same channels ($c_i \times k_h \times k_w$), then add them up.
   - multi out $c_o$ -> kernel with $c_o \times c_i \times k_h \times k_w$, get $c_o$ output channels.
6. use `torch.stack` to stack tensors
7. **Pooling**: mitigating the sensitivity of convolutional layers to location and of spatially downsampling representations.