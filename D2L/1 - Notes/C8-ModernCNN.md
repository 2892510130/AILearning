## Chapter 8 : Modern CNN
1. **AlexNet**: first deep conv successful, using dropout, Relu, polling
2. **VGG**: multiple 3 * 3 conv layers (two 3 * 3 conv touch 5 * 5 input as a 5 * 5 conv, but 2 * 3 * 3  = 18 < 25 = 5 * 5)
3. **NiN**: to handle 2 problem (1. much ram for the MLP at the end; 2. can not add MLP between the conv to increase the degree of nonlinearity as it will destroy the spatial information)
   - use 1 * 1 conv layer to add local nonlinearities across the channel activations
   - use global average pooling to integrate across all locations in the last representation layer. (must combine with added nonlinearities)
4. **GoogleNet**: Inception layer, parallel conv multi scales, and then concate them
5. **Batch Normalization**:
   - $BN(\mathbf x) = \mathbf{\gamma} \bigodot \frac{\mathbf x - \mathbf{\mu_B}}{\sigma^2_B} + \mathbf \beta$, $\mathbf{\mu_B} = \frac{1}{|B|}\sum_{x \in B} \mathbf x$,
     $\sigma^2_B = \frac{1}{|B|} \sum_{x \in B} (x - \mathbf{\mu_B})^2 + \epsilon$
   - On linear layer [N, D] it will get across D (different features in D will not do calculations), on conv layer [N, C, H, W] it will across C (save the difference between channels)
     - For example, [N, C, H, W] shape input x, for x[N, 0, H, W], get it's mean mu and std and do (x[N, 0, H, W] - mu) / std, here mu and std are scalar.
   - At the testing stage, we will use the global (whole) data mean and varience, instead of minibatch mean and varience. Just like dropout.
   - So BN also serves as a noise introducer! (minibatch information != true mean and var) Teye et al. (2018) and Luo et al. (2018).
   - So it best works for batch size of 50 ~ 100, higher the noise is small, lower it is too high.
   - Moving global mean and var: when testing, no minibatch, so we use a global one that is stored during training.
     - It is a kind of exp weighted mean, closest batch has higer weight
     - $\mu_m = \mu_m * (1 - \tau) + \mu * \tau, \Sigma_m = \Sigma_m * (1 - \tau) + \Sigma * \tau$, $\tau$ is called momentum term.
6. **Layer Normalization**: often used in NLP
   - For features like [N, A, B] it will save difference between N, A and B are typically seq_len, hidden_size.
7. **ResNet**: residual block, pass x as one of the branch before a activation function (for the original paper, and later it is changed to BN -> AC -> Conv)
   - To get the passed x has the correct shape to add up, we can use 1 * 1 conv if it is needed
   - **Idea**: nested-function class, shallower net (like ResNet-20) is subclass of depper net (like ResNet-50). Because in ResNet-50 if the layers after 20th layer are f(x) = x, then it is the same as RestNet-20! So we can make sure f' (the best we can get in ResNet-50 for certain data) will be better than f (ResNet-20 on the same data) or at least the same.
   - <figure style="text-align: center;">
      <img alt="Residul Block" src="https://d2l.ai/_images/resnet-block.svg" style="background-color: white; display: inline-block;">
      <figcaption> Residul Block </figcaption>
    </figure>
    
   - **ResNeXt**: use g groups of 3 * 3 conv layers between two 1 * 1 conv of channel $b$ and $c_o$, so $\mathcal O(c_i c_o) \rightarrow \mathcal O(g ~ c_i / g ~ c_o / g) = \mathcal O(c_ic_o/g)$
     - This is a **Bottleneck** arch if $b < c_i$
       </br>
   - <figure style="text-align: center;">
      <img alt="ResNeXt Block" src="https://d2l.ai/_images/resnext-block.svg" style="background-color: white; display: inline-block;">
      <figcaption> ResNeXt Block </figcaption>
    </figure>
    
8. **DenseNet**: instead of plus x, we concatenate x repeatedly.
   - For example (\<channel\> indicates the channel): x\<c_1\> -> f_1(x)\<c_2\> end up with [x, f_1(x)]\<c_1 + c_2\> -> f_2([x, f_1(x)])\<c_3\> end up with [x, f_1(x), f_2([x, f_1(x)])]\<c_1 + c_2 + c_3\>
   - Too many of this layer will cause the dimeansion too big, so we need some layer to reduce it. **Translation** layer use 1 * 1 conv to reduce channel and avgpool to half the H and W.
9. **RegNet**:
   - AnyNet: network with **stem** -> **body** -> **head**.
   - Distrubution of net: $F(e,Z)=∑_{i=1}^{n}1(e_i<e)$, use this empirical CDF to approximate $F(e, p)$, $p$ is the net arch distrubution. $Z$ is a sample of net sample from $p$, if $F(e, Z_1) < F(e, Z_2)$ then we say $Z_1$ is better, it's parameters are better.
   - So for RegNet, they find that we should use same k (k = 1, no bottlenet, is best, says in paper) and g for the ResNeXt blocks with no harm, and increase the network depth d and weight c along the stage. And keep the c change linearly with $c_j = c_o + c_aj$ with slope $c_a$
   - neural architecture search (NAS) : with certain search space, use RL (NASNet), evolution alg (AmoebaNet), gradient based (DARTS) or shared weight (ENAS) to get the model. But it takes to much computation resource.
   - <figure style="text-align: center;">
      <img src="https://d2l.ai/_images/anynet.svg" style="background-color: white; display: inline-block;">
      <figcaption> AnyNet Structure (Search Space) </figcaption>
    </figure>
   </br>