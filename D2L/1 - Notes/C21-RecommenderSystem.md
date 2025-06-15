## Chapter 21 : Recommender System
- Dataset:
  - *seq-aware* : use the recent data to predict, history data for train.
- *Matrix Factorization*:
  - Let $\mathbf R \in \mathbb R^{m\times n}$ denote the interaction matrix with $m$ users and $n$ items. *Latent Matrix* $\mathbf P \in \mathbb R^{m\times k}$ and $\mathbf Q \in \mathbb R^{n\times k}$, $k \ll m,n$ is the latent factor size. Each row in $\mathbf P$ may represent a user's interest in these latent features, each row in $\mathbf Q$ represent an item's value in these latent features. The predicted ratings are $\hat{\mathbf R} = \mathbf P \mathbf Q^{\top}$. But this way we can not introduce bias, so the predicted one becomes $\hat{\mathbf R_{ui}} = \mathbf p_u \mathbf q_i^{\top} + b_u + b_i$. Then the objective function is:
    $${\arg\min}_{\mathbf{P}, \mathbf{Q}, b} \sum_{(u,i) \in \mathcal{K}} \lVert \mathbf{R}_{ui} - \hat{\mathbf{R}}_{ui} \rVert^2 + \lambda \left( \| \mathbf P \|_F^2 + \| \mathbf Q \|_F^2 + b^2_u + b_i^2 \right)$$
- *AutoRec*:
  - Let $\mathbf R_{*i}$ denote the $i^{th}$ column of the ration matrix, the neural arch is:
    $$h(\mathbf R_{*i}) = f \left ( \mathbf W \cdot g(\mathbf{VR_{*i}}+\mu) + b \right)$$
    Then the objective function is:
    $${\arg\min}_{\mathbf{W}, \mathbf{V}, \mu, b} \sum_{i=1}^M \lVert \mathbf{R}_{*i} - \hat{\mathbf{R}}_{*i} \rVert^2_{\mathcal O} + \lambda \left( \| \mathbf W \|_F^2 + \| \mathbf V \|_F^2 \right)$$
    Where $\|\|_{\mathcal O}$ means only the contribution of observed ratings are considered.
- Type of learning to rank:
    - *Pointwise* approache: MF and AutoRec are both this type, at one time only one individual preference are trained, the order is ignored.
    - *Pairwise* approache: input is pair, we need to select which one is better (consider the order).
    - *Listwise* approache: consider the entire list.
- *Bayeisan Personalized Ranking Loss*:
  - Traning data consist of both positive and negitive pairs (missing values), $(u,i,j)$ represents user $u$ prefer item $i$ over $j$. BPR aims to maximize:
    $$p(\Theta \mid >_u) \propto p(>_u \mid \Theta)p(\Theta)$$
    Where $\Theta$ represents the parameter, $>_u$ represents the desired personalized total ranking of all items for user $u$. We can formulate the maximum posterior estimator (MAE):
    $$
    \begin{array}{lll}
    \text{BPR-OPT} &:=& \ln p(\Theta \mid >_u) \\
                   &\propto& \ln p(>_u \mid \Theta)p(\Theta) \\
                   &=& \ln \prod_{(u,i,j) \in D} \sigma(\hat{y}_{ui} - \hat{y}_{uj}) p(\Theta) \\
                   &=& \sum_{(u,i,j) \in D} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \ln p(\Theta) \\
                   &=& \sum_{(u,i,j) \in D} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) - \lambda_\Theta \lVert \Theta \rVert^2
    \end{array}
    $$
    Where we assume $\Theta \sim \mathcal N(0, \lambda_{\Theta} I)$, and $i$ is positive item, $j$ is negitive and missing item.
    This graph shows details:
    <figure style="text-align: center;">
      <img src="https://d2l.ai/_images/rec-ranking.svg" alt="rec-ranking" style="background-color: white; display: inline-block;"/>
      <figcaption> rec-ranking </figcaption>
    </figure>
- *Hinge Loss*: $\sum_{(u,i,j) \in D} \max (m-\hat{y}_{ui}+\hat{y}_{uj}, 0)$ often used in SVM, where $m$ is the safety margin size.
- *Neural Collaborative Filtering*: consist of 2 subnets.
  - *GMF* subnet: $x = p_u \odot q_i$, $\hat y_{ui} = \alpha(h^{\top}x)$, $p$ and $q$ are latent matrix row, $\hat y_{ui}$ is user $u$ score on item $i$.
  - *MLP* subnet: its input is concatenation of user and item embeddings.
    $$
    \begin{align*}
    \mathbf{z}^{(1)} &= \phi_1(\mathbf{U}_u, \mathbf{V}_i) = [\mathbf{U}_u, \mathbf{V}_i] \\
    \phi^{(2)}(\mathbf{z}^{(1)}) &= \alpha^{1}(\mathbf{W}^{(2)} \mathbf{z}^{(1)} + \mathbf{b}^{(2)}) \\
    &\vdots \\
    \phi^{(L)}(\mathbf{z}^{(L-1)}) &= \alpha^{L}(\mathbf{W}^{(L)} \mathbf{z}^{(L-1)} + \mathbf{b}^{(L)}) \\
    \hat{y}_{ui} &= \alpha(\mathbf{h}^\top \phi^{L}(\mathbf{z}^{(L-1)}))
    \end{align*}
    $$
  - To combine them, we take the last layer and concatenate: $\hat y_{ui} = \sigma \left ( h^{\top}[x, \phi^L(z^{L-1})] \right )$.
  - To evaluate the result, we use a hit rate at given cutting off $\mathscr l$: $\frac{1}{m}\sum_{u\in \mathcal U} \mathbf 1(\text{rank}_{u,g_u} \leq \mathscr l)$ and area under the ROC curve (AUC):
    $$ \text{AUC} = \frac 1 m \sum_{u\in \mathcal U} \frac{1}{|\mathcal I \setminus S_u|} \sum_{j\in \mathcal I \setminus S_u} \mathbf 1(\text{rank}_{u,g_u} < \text{rank}_{u,j})$$
    Where $\mathcal I$ is the item set, $S_u$ is the candidate items of user $u$.
- *Caser*: The goal of Caser is to recommend item by considering user general tastes as well as short-term intention.
  - Let $S^u=(S^u_1, \ldots, S^u_{|S_u|})$ denotes the ordered sequence, consider $L$ recent items:
    $$E^{(u,t)} = \left [ qS^u_{t-L}, \ldots, qS^u_{t-2}, S^u_{t-1} \right ]^{\top}$$
    where $Q\in \mathbb R^{n\times k}$ represents items embeddings, $E^{(u,t)} \in \mathbb R^{L\times k}$.
    <figure style="text-align: center;">
      <img src="https://d2l.ai/_images/rec-caser.svg" alt="rec-caser" style="background-color: white; display: inline-block;"/>
      <figcaption> rec-caser </figcaption>
    </figure>
    Left is horizontal convolutional layer, right is vertical. $p_u$ and $v_i$ (another one for item) are user and item embedding. So basically, we input the $E$ input convolution layer get feature $z$, concatenates with user embedding $p_u$, get feature $f$, then with item embedding $v_i^{pos}$ and $v_i^{neg}$ to get $\hat y^{pos} = f^{\top}v_i^{pos} + b^{pos}, \hat y^{neg} = f^{\top}v_i^{neg} + b^{neg}$, get the BPR loss.
- Feature-Rich Recommender System:
  - *Click-through rate*: click / impression (show times) * 100%.
  - *Factorizatin Machine*: given $x \in \mathbb R^d$ with $d$ fields, feature embedding $V \in \mathbb R^{d\times k}$
    $$ \hat{y}(x) = \mathbf{w}_0 + \sum_{i=1}^d \mathbf{w}_i x_i + \sum_{i=1}^d\sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j $$
    Then reformulate the equation to decrease the complexity:
    $$
    \begin{split}\begin{aligned}
    &\sum_{i=1}^d \sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j \\
     &= \frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d\langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j - \frac{1}{2}\sum_{i=1}^d \langle\mathbf{v}_i, \mathbf{v}_i\rangle x_i x_i \\
     &= \frac{1}{2} \big (\sum_{i=1}^d \sum_{j=1}^d \sum_{l=1}^k\mathbf{v}_{i, l} \mathbf{v}_{j, l} x_i x_j - \sum_{i=1}^d \sum_{l=1}^k \mathbf{v}_{i, l} \mathbf{v}_{i, l} x_i x_i \big)\\
     &=  \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i) (\sum_{j=1}^d \mathbf{v}_{j, l}x_j) - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2 \big ) \\
     &= \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i)^2 - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2)
     \end{aligned}\end{split}
    $$
  - *Deep FM*: $\hat{y} = \sigma(\hat{y}^{(FM)} + \hat{y}^{(DNN)})$
    <figure style="text-align: center;">
      <img src="https://d2l.ai/_images/rec-deepfm.svg" alt="rec-deepfm" style="background-color: white; display: inline-block;"/>
      <figcaption> rec-deepfm </figcaption>
    </figure>
- FAQ:
  - What is the field? Q: for example, we have 100 samples, total shape is [100, 500], so it has 500 fields, then we put it into embedding layer get [100, 500, embedding_dims].
<!-- <figure style="text-align: center;">
  <img src="https://d2l.ai/_images/gan.svg" alt="GAN" style="background-color: white; display: inline-block;"/>
  <figcaption> GAN </figcaption>
</figure> -->