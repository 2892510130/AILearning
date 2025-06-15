## Chapter 15 : NLP Pretraining
- Word Embedding (word2vec)
  - one-hot is not good, we can not get similarity from it (for example cosine similarity).
  - Skip-Gram Model:
    - use one word (center word) to generate other word (context word) with a probability, each word has two vectors to represent itself, $v_i$ when use as center word, $u_i$ when use as context word.
    - Here we use: $P(w_o | w_c) = \frac{\exp(u_o^{\top}v_c)}{\sum_{i \in \mathcal V} \exp(u_i^{\top}v_c)}$. And given a sequence of length $T$, a context window $m$, the skip-gram model is $\prod_{t=1}^T\prod_{-m \leq j \leq m, j \neq 0} \log P(w^{t+j}|w^t)$. Note that index less than 0 or larger than T will be omitted.
    - Maximize this model, we get the trained params (v and u), and in nlp, v (center word representation) is often used as word representation.
  - The Continuous Bag of Words (CBOW) Model
    - different from Skip-Gram, CBOW center word is generated from context words.
    - Switch the meaning of $v$ and $u$, $P(w_c\mid w_{o_1},\ldots,w_{o_{2m}})=\frac{\exp\left(\frac{1}{2m}\mathbf{u}_c^\top(\mathbf{v}_{o_1}+\ldots+\mathbf{v}_{o_{2m}})\right)}{\sum_{i\in\mathcal{V}}\exp\left(\frac{1}{2m}\mathbf{u}_i^\top(\mathbf{v}_{o_1}+\ldots+\mathbf{v}_{o_{2m}})\right)}$, denote $\mathcal W_o = \{w_{o1}, \dots, w_{o2m}\}$ and $\bar v_o = (v_{o1} + \dots + v_{o2m}) / 2m$ then we get $P(w_c \mid \mathcal W_o) = \frac{\exp(u_c^{\top} \bar v_o)}{\sum_{i \in \mathcal V} \exp(u_i^{\top} \bar v_o)}$
    - Given a sequence with length $T$, the model is $\prod_{t=1}^T P(w^t \mid w^{t-m}, \dots, w^{t+m})$, after traning, it will use the context word vector as the word representation.
- Approximate Training
  - In the softmax above, the denominator containt all the vectors from the vocabulary, so the gradient calculation is hard to get, we need to approximate it.
  - Negative Sampling:
    - given a center word $w_c$, any context word $w_o$ coms from the context window is considered as an event: $P(D=1 \mid w_c, w_o) = \sigma(u_o^{\top}v_c)$, $\sigma$ use sigmoid activation function. So we train to maximize $\prod_{t=1}^T\prod_{-m \leq j \leq m, j \neq 0} \log P(D=1|w^t, w^{t+j})$.
    - But this only contains positive items, we need to add negitive ones. Change the above $P(D=1|w^t, w^{t+j})$ to $P(D=1|w^t, w^{t+j}) \prod_{k=1, w_k \sim P(w)}^K P(D=0 \mid w^t, w_k) \rightarrow P(w^{t+j} \mid w^t)$, where $P(w)$ is a predefined distribution that will sample $K$ noise words that is negitive (not in the context window).
    - so $-\log P(w^{t+j} \mid w^t) = -\log \sigma\left( \mathbf{u}_{i_t+j}^{\top} \mathbf{v}_{i_t} \right) - \sum_{k=1, \, w_k \sim P(w)}^{K} \log \sigma\left( -\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t} \right)$, only depend on the hyperparameter $K$, not vocab size involve.
  - Hierarchical Softmax:
    - Use a binary tree where leaf node represents a word. $L(w)$ is the path length from root to leaf represents word $w$, $n(w,j)$ is the $j$th node in this path with it's context word vector $u_{n(w,j)}$.
    - We use $P(w_o \mid w_c) = \prod_{j=1}^{L(w_o)-1} \sigma \left( \left[ n(w_o, j+1) = \text{leftChild}(n(w_o, j)) \right] \cdot \mathbf{u}_{n(w_o, j)}^\top \mathbf{v}_c \right)$, $[*]$ return 1 if true, otherwise -1.
    - since $L(w_o) - 1$ is on the order of $\mathcal O(\log_2|\mathcal V|)$, when vocab size is huge, this can reduce the traning cost.
- Dataset
  - Subsampling: some word like 'a', 'to' is too common, it's context word varies, so we want to sub-sample (discard) them with a probability $P(w_i) = \max(1 - \sqrt{t / f(w_i)}, 0)$, where $f(w_i)$ is the ratio of word $w_i$.
  - Minibatch: $i^{th}$ example include a center word and its $n_i$ context words and $m_i$ noise words. For different $i$, $n_i + m_i$ varies, so we pad them with 0, so for the loss calculation, we need a mask. And for distinguish between context word and noise word, we need another variable as label (work like mask).
  - When the corpus is huge, we often sample context words and noise words for the center words in the current minibatch.
- Word Embedding with Global Vectors (Glove)
  - Start from skip-gram: let $x_{ij}$ be times $w_i$ (as the center word) and $w_j$ (as the context word) occur in corpus in certain window size, the loss is the same as $-\sum_i\sum_jx_{ij}\log q_{ij}$, here $q_{ij} = p(w_j \mid w_i)$. Now write $p_{ij} = x_{ij} / x_i$, we get $-\sum_ix_i\sum_jp_{ij}\log q_{ij}$, which takes the form of a weighted cross entropy loss.
  - However, the softmax has the problem we mentioned above, and cross entropy may assign large weight to rare events. So GloVe makes three changes:
    - Do not use probability distribution: $p_{ij}^{'} = x_{ij}, q_{ij}^{'} = \exp(u_j^{\top}v_i)$, and use **Squared Loss** $\left(\log p_{ij}^{'} - \log q_{ij}^{'} \right)^2 = \left(u_j^{\top}v_i - \log x_{ij} \right)^2$
    - Add two bias to the center word and context word parameters, $b_i$ and $c_i$.
    - replace the weight to $h(x_{ij})$ which increase in $[0, 1]$.
    - Then we get the loss $=-\sum_i\sum_jh(x_{ij})\left(u_j^{\top}v_i + b_i + c_j - \log x_{ij} \right)^2$. $h(x) = (x/c)^{\alpha}$ ($\alpha = 0.75$, $c = 100$) if $x < c$ else 1 is a suggestion.
  - The model is symmetric as $x_{ij} = x_{ji}$, so the two vector of word is mathmetically same, but during traning they may get different value (initialization), so the **word vector** is sum of them.
  - Other way to get the same loss function: consider ratio of co-occurrence probability $p_{ij} / p_{ik}$, we want some function $f(u_j,u_k,v_i)$ to fit it, based on the exchange of $j$ and $k$, we find $f(x)f(-x)=1$, so one posibility is $f = \frac{\exp(u_j^{\top}v_i)}{\exp(u_k^{\top}v_i)}$, let $\exp(u_j^{\top}v_i) \approx \alpha p_{ij}$, we get $u_j^{\top}v_i + b_i + c_j - \log x_{ij} \approx 0$
  - **GloVe if build from counting, and skip-gram and CBOW is build from probability**.
- Subword Embedding
  - fastText model: add \< and \> to a word, then divide the word into different length subword. For a word $w$, let $\mathcal G_w$ the union of all its subword of length between 3 and 6 and its special subword (\<word w\>). Then the word vector of it is $v_w = \sum_{g \in \mathcal G_w}z_g$.
  - Byte Pair Encoding (BPE): iteratively merge most frequent pairs, so we can control the size of the vocab.
  - fastText and GloVe, 2 web of pretrained word embeddings.
- Bidirectional Encoder Representations from Transformers (BERT)
  - The above word embedding method is context-independent, namely f(x) is only a function of word x, this is a limitation as the same word in different context may have different meanings. So we have context-sensitive representations like TagLM, CoVe, and ELMo, f(x, c(x)). ELMo combing all the middle layer of the bidirection LSTM as its output, and serve as additional feature for downstream tasks, during the downstream task training, the ELMo layers are frozen.
  - But these models is task-specific. We need to design a architecture for every NLP tasks, this is hard. GPT based on Transformer decoder build a task-agnostic model, its output will be passed to a output layer, and fine-tune the model for the downstream task. However it can only look forward because of the natural of autoregressive, so for the same word the context may differ in the future, so does the meaning.
  - BERT combine them, can see the left and right context, and same as GPT fine-tune, fed to a output layer (train from scrath).
  - <figure style="text-align: center;">
      <img src="https://d2l.ai/_images/elmo-gpt-bert.svg" alt="nlp embedding" style="background-color: white; display: inline-block;"/>
      <figcaption> A comparison of ELMo, GPT, and BERT </figcaption>
    </figure>
  - BERT use [cls] at the top, and then a sequence, then a [seq] spetial token. In order to fit different tasks, there are two way to construct the BERT input. 1 is [cls] sequnces [seq] for task like emotion analyse, 2 is [cls] sequences 1 [seq] sequences 2 [seq] for task like nautral language inference. And to distinguish text pairs, BERT add a learned segment embedding $e_A$ and $e_B$ to the token embedding, see figure below. And BERT use a learned positional embedding.
  <!-- - <img alt="BERT input" src="https://d2l.ai/_images/bert-input.svg" style="background-color: white; display: inline-block;"> BERT input -->
  - <figure style="text-align: center;">
      <img src="https://d2l.ai/_images/bert-input.svg" alt="BERT input" style="background-color: white; display: inline-block;"/>
      <figcaption> BERT input </figcaption>
    </figure>
  - Pretraining Tasks:
    - Masked Language Modeling: get the BERT output, add mask on certain position, add a mlp layer to predict these position. The mask can be a special token [\<mask\>] or the original token or a randome token, we can use a probability of [0.8, 0.1, 0.1] for all three cases. Note than special token [cls] and [sep] will not be the prediction target.
    - Next Sentence Prediction: MLM does not understant the relationship of two sequences. So we implement this, add one layer after [cls] token representation to train a binary task (whether next sequence is logically the next one for this sequence). In training, we sample 50% ramdom sequence and 50% true next sequence.
    - BERT pretraning is a linear combination of these 2 tasks.