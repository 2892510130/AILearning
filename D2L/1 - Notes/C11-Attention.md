## Chapter 11 : Attention Mechanisms and Transformers
- **Idea** : attention first come up in encoder - decoder design, rather than tranform the input to a fixed size feature and the feed it to all the decoder step, we want to create a representation that has the same length of input and decoder at each time step can pay attention to different input sequence (with it's weight). And transfromer give up residual connection, instead use attention at all.
- **Q, K, V**
  - Define database $\mathcal{D} \stackrel{\mathrm{d e f}} {=} \{( \mathbf{k}_{1}, \mathbf{v}_{1} ), \ldots( \mathbf{k}_{m}, \mathbf{v}_{m} ) \} $, give some query, the attention is $\text{Attention}(q, D) \stackrel{\mathrm{d e f}} {=} \sum_{i=1}^m \alpha(q, k_i)v_i $, where $\alpha(q, k_i)$ are scalar attention weight, this operation is also called attention pooling. We want this attention weight to be larger than 0 and sum up to 1, so we can use a softmax to transfrom it.
- **Attention pooling with similarity**
  - In a regression task, we can use kernel output as the attention weight (after normalization).
  - When directly compute loss $(f(x_i) - y_i)^2$， because we have $y_i$, the $\sigma$ will go to zero, causing overfitting. Even if we remove $y_i$ from the loss, if the dataset is large enough, we may still overfit.
  - <figure style="text-align: center;">
      <img alt="Attention Pooling" src="https://d2l.ai/_images/attention-output.svg" style="background-color: white; display: inline-block;">
      <figcaption> Attention Pooling </figcaption>
    </figure>
- **Attention Scoring Function**
  - **Dot Product Attention**: note that kernel of gaussian is $a(q, k_i) = q^{\intercal}k_i - 0.5 * ||q||^2 - 0.5 * ||k_i||^2$, and after the softmax normalization, second one is cancled out. Then if we get $k_i$ with batch or layer normalization, it's length will be bounded and often constant, so we can get rid of last term without penalty. This leads to $a(q, k_i) = q^{\intercal}k_i$, ann assume both $q$ and $k_i$ have zero mean and unit variance, this attention weight will have a variance of $d$ (which is query feature size), so we can further normalize it : $a(q, k_i) = q^{\intercal}k_i / \sqrt{d}$, this is the common one used in transformer. At last, we do a softmax on it.
  - Because we do not want consider \<pad\>, so we can do musk softmax. And for Batch Matrix Multiplication, use torch.bmm. bmm(Q, K) take Q(n, a, b) and K(n, b, c), which return [Q1 @ K1, ..., Q_n @ K_n].
  - **Additive Attention**: when q and k has different feature size, we can do a transform ($q^{\intercal}Mk$), or use this additive attention $a(\mathbf{q},\mathbf{k})=\mathbf{w}_v^\top\tanh(\mathbf{W}_q\mathbf{q}+\mathbf{W}_k\mathbf{k})\in\mathbb{R}$, inside the () we add by broadcasting.
- **Bahdanau Attention**
  - Attention function will work between encoder hidden state and decoder hidden state, $c_{t'} = \sum_t^T a(s_{t'-1}, h_t)h_t$, and this will used to generate $s_{t'}$.
  - <figure style="text-align: center;">
      <img alt="Bahdanau Attention" src="https://d2l.ai/_images/seq2seq-details-attention.svg" style="background-color: white; display: inline-block;">
      <figcaption> Bahdanau Attention </figcaption>
    </figure>
  - Seq2SeqAttentionDecoder first use encoder last layer hidden state as the query, and later use it's own hidden state from it's rnn model. Keys and values are all encoder outputs (last layer hidden state at all time step), then concatenate the embed X with this context (attention result) then feed to it's rnn.
- **Multi-head Attention**
  - <figure style="text-align: center;">
      <img alt="Multi-head Attention" src="https://d2l.ai/_images/multi-head-attention.svg" style="background-color: white; display: inline-block;">
      <figcaption> Multi-head Attention </figcaption>
    </figure>
  - We want the same Q, K, V to have different behaviour with the same attention mechanism, so we have to copy them $h$ times and first pass them into FC layer which has learnable param that can change the QKV, then feed to attention, get $h$ results, concatenate them. $h_i = f(W_i^{q}q_i, W_i^kk_i, W_i^vv_i)$, $f$ is attention pooling, each $W$ have shape of $(p_q, d_q), (p_k, d_k)$ and $(p_v, d_v)$, $h_i$ is of shape $(p_v,)$. We concatenate these h to a $(h \times p_v,)$ shape. And we use a big learnable matrix of shape $(p_o, h \times p_v)$ times the concatenated result, which finnally return output of shape $(p_o, )$. For the purpose of parallel computation, we set $hp_q = hp_k = hp_v = p_o$.
  - Impl of d2l is the same idea, but for parrallel, it use some trick, hidden_size is h * p_q, put num_head into batch_size, make the batch_size = batch_size * num_head, and in the output reverse it.
- **Self Attention**
  - Do the attention(X, X, X) to get encoder. Compare CNN, RNN and self-attention, given n sequence with d dimension. CNN : choose kernel of 3, computation is O(knd^2), longest connect path is O(n/k). RNN : compute O(nd^2), path O(n). Self attention : compute O(n^2d), path O(1).
  - Positional Encoding: self attention does not contain position (order) information, so a token at time step 1 and 5 are the same (but it should not!). So we need to add something to keep the position information. First we use fixed position encoding with sine and cosine. $X^{n \times d}$ is the input representation of n tokens, and the position encoding is $X + P$, where $p_{i,2j} = \sin \left( \frac{i}{10000^{2j/d}} \right)$ and $p_{i,2j+1} = \cos \left( \frac{i}{10000^{2j/d}} \right)$. This works because the function abouve contain different frequency information.
  - Relative position encoding : $\begin{bmatrix}
\cos(\delta\omega_j) & \sin(\delta\omega_j) \\
-\sin(\delta\omega_j) & \cos(\delta\omega_j)
\end{bmatrix}
\begin{bmatrix}
p_{i,2j} \\
p_{i,2j+1}
\end{bmatrix} = \begin{bmatrix}
p_{i+\delta,2j} \\
p_{i+\delta,2j+1}
\end{bmatrix}$, we can just add a (1, step, hidden) param, and add it to the embed(X) to learn the position.
- **Transformer**
  - <figure style="text-align: center;">
      <img alt="Transformer" src="https://d2l.ai/_images/transformer.svg" style="background-color: white; display: inline-block;">
      <figcaption> Transformer Arch </figcaption>
    </figure>
  - The encoder-decoder attention layer take decoder self-attention layer output as query, and encoder output as key and value.
  - Note that in decoder self-attention, we will carefully mask output to reserve the autoregreesive nature, we do not take position in the outpt (later as the input of decoder self-attention layer) after the position we are calculating.
  - Before pos encoding we first multiply sqrt(d) with embed(X) to rescale it, maybe because embed(X) has small variace.
  - For prediction, we need to cache the input X for the decoder, in training we can just compute all time step all together. In impl, it is cached in state[2].
  - **!!** only the last output of the encoder will do attention on all block of decoder.
- **Vision Transformer**
  - <figure style="text-align: center;">
      <img alt="Vision Transformer" src="https://d2l.ai/_images/vit.svg" style="background-color: white; display: inline-block;">
      <figcaption> Vision Transformer </figcaption>
    </figure>
  - patch embeding will feed to a conv then flatten it, return shape of (batch, patch, hidden)
  - Do the normalization before the attention is better for the efficient learning of transformer. The vit mlp layer use GELU and dropout is applied to the output of each fully connected layer in the MLP for regularization.

- Large Scale Pre-training
  - Encoder only, ViT, BERT. BERT use masked language modeling, and for a token, tokens at left and right can all attend to this masked token. So it is a bidirection encoder --- in the figure below, each token along the vertical axis attends to all input tokens along the horizontal axis.
  - <figure style="text-align: center;">
      <img alt="BERT" src="https://d2l.ai/_images/bert-encoder-only.svg" style="background-color: white; display: inline-block;">
      <figcaption> BERT </figcaption>
    </figure>
  - Encoder-Decoder, BART & T5, both attempt to reconstruct original text in their pretraining objectives, while the former emphasizes noising input (e.g., masking, deletion, permutation, and rotation) and the latter highlights multitask unification with comprehensive ablation studies.
  - <figure style="text-align: center;">
      <img alt="T5" src="https://d2l.ai/_images/t5-encoder-decoder.svg" style="background-color: white; display: inline-block;">
      <figcaption> T5 </figcaption>
    </figure>
  - Decoder only, GPT. **In-context learning** : conditional on an input sequence with the task description, task-specific input–output examples, and a prompt (task input). 
  - <figure style="text-align: center;">
      <img alt="GPT" src="https://d2l.ai/_images/gpt-decoder-only.svg" style="background-color: white; display: inline-block;">
      <figcaption> GPT </figcaption>
    </figure>
  - <figure style="text-align: center;">
      <img alt="x - shot" src="https://d2l.ai/_images/gpt-3-xshot.svg" style="background-color: white; display: inline-block;">
      <figcaption> x-shot </figcaption>
    </figure>
- Efficient Transformer design (see that survey)
  - Sparse attention：Longformer：使用滑动窗口和全局注意力，降低复杂度到O(n)。BigBird：结合随机注意力、窗口注意力和全局注意力，适合长序列。
  - Low rank approximation：Linformer：通过低秩分解将注意力矩阵投影到较低维度，复杂度从O(n²)降到O(n)。
  - Memory：Transformer-XL：引入循环记忆，处理长序列时重用之前的隐藏状态，避免重复计算。
  - Efficient attention：Performer：使用核方法（Favor+）近似点积注意力，复杂度降为O(n)。
  - Model compress：Distillation：将大Transformer蒸馏为小模型（如DistilBERT）。量化：减少参数精度，降低内存占用。
<!-- <img alt="ResNeXt Block" src="https://d2l.ai/_images/rnn.svg" style="background-color: white; display: inline-block;"> -->
<!-- <img alt="ResNeXt Block" src="https://d2l.ai/_images/rnn-bptt.svg" style="background-color: white; display: inline-block;"> -->