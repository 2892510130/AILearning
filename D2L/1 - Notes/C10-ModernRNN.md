## Chapter 10 : Modern RNN
- **LSTM**
  - The structure is :
  - <figure style="text-align: center;">
      <img alt="LSTM Arch" src="https://zh.d2l.ai/_images/lstm-3.svg" style="background-color: white; display: inline-block;">
      <figcaption> LSTM Arch </figcaption>
    </figure>
- **GRU**
  - <figure style="text-align: center;">
      <img alt="GRU Arch" src="https://d2l.ai/_images/gru-3.svg" style="background-color: white; display: inline-block;">
      <figcaption> GRU Arch </figcaption>
    </figure>
  - Reset gates help capture short-term dependencies in sequences.
  - Update gates help capture long-term dependencies in sequences.
- **Deep RNN**
  - <figure style="text-align: center;">
      <img alt="Deep RNN" src="https://d2l.ai/_images/deep-rnn.svg" style="background-color: white; display: inline-block;">
      <figcaption> Deep RNN </figcaption>
    </figure>
  - In deep rnn, the output is the last layer of the hidden state with every timestep, and state is the last time step hidden state with all layer of rnn.
- **Bidirection RNN**, it is slow and gradient chain is long
  - $P(x_1,\ldots,x_T,h_1,\ldots,h_T)=\prod_{t=1}^TP(h_t\mid h_{t-1})P(x_t\mid h_t),\mathrm{~where~}P(h_1\mid h_0)=P(h_1)$, it is a hidden markov model. We can use dynamic programming method compute is from start to end, also from end to start. Just how B-RNN is capable of.
  - <figure style="text-align: center;">
      <img alt="Bidirection RNN" src="https://zh.d2l.ai/_images/birnn.svg" style="background-color: white; display: inline-block;">
      <figcaption> B-RNN </figcaption>
    </figure>
  - And we just need to concatenate these two H.
- **Machine translation**
  - non-breaking space, some space should not split to new line, like Mr. Smith.
  - Teacher Forcing : all the input will be pad with \<pad\>, source token no special treat, decoder input (target seq use as input) will start with \<bos\>, and label is shift by 1 (no \<bos\> at the begining).
  - **Important**: when use teacher forcing, the truth target is feed to the decoder. This will make the traning faster and stable, but it will make training and predicting different (because when predicting we do not have truth target label, we have to repeatedly predict). We can make them the same, but the tranning will be harder.
- **Sequence to Sequence**
  - We use this Encoder - Decoder Arch to get varied length input and varied length output.
  - We do not use one-hot, instead we use nn.Embed layer, which will take token i, and return ith row of the matrix of this embeding layer.
  - From the encoder, we get the hidden states, and use a funcion $c = q(h_1, \cdots, h_T)$, for example, just use the $h_T$. And in the decoder, we concatenate this with the target embed output, and feed to rnn.
  - When calculating the loss, we should not take \<pad\> into acount. So we need to musk the loss with the tokens.
  - <figure style="text-align: center;">
      <img alt="Encoder Decoder" src="https://d2l.ai/_images/seq2seq-details.svg" style="background-color: white; display: inline-block;">
      <figcaption> Encoder Decoder </figcaption>
    </figure>
  - Bilingual Evaluation Understudy, BLEU evaluates whether this n-gram in the predicted sequence appears in the target sequence. For example, target sequence ABCDEF, predict sequence ABBCD, $p_1 = 4/5$, we have ABCD in the target sequence, $p_2 = 3 / 4$, we have AB, BC, CD. So we get BLEU as $\exp\left(\min\left(0,1-\frac{\mathrm{len}_{\mathrm{label}}}{\mathrm{len}_{\mathrm{pred}}}\right)\right)\prod_{n=1}^kp_n^{1/2^n}$, higher n will have higher weight, small length of predict length takes lower.
- **Beam Search**
  - Before this section, we use greedy search to get prediction, use argmax on the prediction vector : $y_{t^{\prime}}=\underset{y\in\mathcal{Y}}{\operatorname*{\operatorname*{argmax}}}P(y\mid y_1,\ldots,y_{t^{\prime}-1},\mathbf{c})$, where $\mathcal Y$ is the vacab. Once our model outputs “<eos>” (or we reach the maximum length $T'$) the output sequence is completed.
  - However, use the most likely tokens is not the same with the most likely sequence : $\prod_{t^{\prime}=1}^{T^{\prime}}P(y_{t^{\prime}}\mid y_1,\ldots,y_{t^{\prime}-1},\mathbf{c})$. For example, in this figure below, ACB will have this probability of 0.5 * 0.3 * 0.6 = 0.09. On the other hand, greedy search choose ABC which is 0.5 * 0.4 * 0.4 = 0.08, it is lower, not optimal!
  - <figure style="text-align: center;">
      <img alt="Max sequence" src="https://d2l.ai/_images/s2s-prob2.svg" style="background-color: white; display: inline-block;"><img alt="Max token" src="https://d2l.ai/_images/s2s-prob1.svg" style="background-color: white; display: inline-block;">
      <figcaption> Max sequence (left), Max token (right) </figcaption>
    </figure>
  - If we want the optimal one, we need to do exhaustive search, search all possible sequence, it is not possible!
  - The most straightforward type of beam search is keep k candidates. In time step 2, we get $P ( A, y_{2} \mid\mathbf{c} )=P ( A \mid\mathbf{c} ) P ( y_{2} \mid A, \mathbf{c} )$ for the top, and $P ( C, y_{2} \mid\mathbf{c} )=P ( C \mid\mathbf{c} ) P ( y_{2} \mid C, \mathbf{c} ) $ for the bottom, then choose most 2 from them. And then choose sequence that maximize $$\frac{1} {L^{\alpha}} \mathrm{l o g} \, P ( y_{1}, \ldots, y_{L} \mid\mathbf{c} )=\frac{1} {L^{\alpha}} \sum_{t^{\prime}=1}^{L} \mathrm{l o g} \, P ( y_{t^{\prime}} \mid y_{1}, \ldots, y_{t^{\prime}-1}, \mathbf{c} ) ; $$ Note tha we have **6** candidates (A, C ..).
  <figure style="text-align: center;">
      <img alt="Max sequence" src="https://d2l.ai/_images/beam-search.svg" style="background-color: white; display: inline-block;">
      <figcaption> beam-search </figcaption>
    </figure>