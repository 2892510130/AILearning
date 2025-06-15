## Chapter 9 : RNN
- Two form of sequence to sequence task:
  - **aligned**: input at certain time step aligns with corrsponding output, like tagging (fight -> verb)
  - **unaligned**: no step-to-step correspondence, like maching translation
- **Autoregressive** model: regress value based on previous value
  - latent autoregressive models (since $h_t$ is never observed): estimate $P(x_t | x_{t-1} \dots x_1)$ with $\hat x_t = P(x_t | h_t)$ and $h_t = g(h_{t-1}, x_{t-1})$
- **Sequence Model**: to get joint probablity of a sequence $p(x_1, \dots, x_T)$, we change it to a form like autoregressive one: $p(x_1) \prod_{t=2}^T p(x_t|x_{t-1}, \dots, x_1)$
  - **Markov Condition**: if we can make the condition above into $x_{t-1}, \dots, x_{t-\tau}$ without any loss, aka the future is conditionally independent of the past, given the recent history, then the sequence satisfies a Markov condition. And it is $\tau^{th}$-order Markov model.
- Zipf’s law: the frequency of words will decrease exponentially, n-grams too (with smaller slope).
  - So use word frequency to construct the probility is not good, for example. $\hat p(learning|deep) = n(deep, learning) / n(deep)$, $n(deep, learning)$ will be very small compared to denominator. We can use so called **Laplace Smooth** but that will not help too much.
- **Perplexity**: (how confusion it is), given a true test data, the cross-entropy is $J = \frac{1}{n} \sum_{t=1}^n -\log P(x_t | x_{t-1}, \dots, x_1)$, and the perplexity is $\exp(J)$.
- Partioning the sequence: for a $T$ token indices sequence, we add some randomness, discard first $d \in U(0, n]$ tokens and partion the rest into $m = \lfloor (T-d) / n \rfloor$ group. For a sequence $x_t$ the target sequence is shifted by one token $x_{t+1}$.
---
- **RNN**: for a vocab with size $|\mathcal V|$, the model parameters should go up to $|\mathcal V|^n$, $n$ is the sequence length.So we $P(x_t | x_{t-1} \dots x_1) \approx P(x_t | h_{t-1})$，$h$ is a **hidden state**, it varies at different time step and contains information of previous time steps. Hidden layer, on the other hand, is a structure, it dose not change in forward calculation.
  - recurrent: $H_t = \phi (X_tW_{th} + H_{t-1}W_{hh} + b_h)$, output is $O_t = H_tW_{tq} + b_q$.
  - <figure style="text-align: center;">
      <img alt="ResNeXt Block" src="https://d2l.ai/_images/rnn.svg" style="background-color: white; display: inline-block;">
      <figcaption> ResNeXt Block </figcaption>
    </figure>
  - clip the gradient: $g = \min(1, \frac{\theta}{|| g ||}) g$, it is a hack but useful.
  - **Warm-up**: When predicting, we can first feed a prefix (now called prompt I think), just iter the prefix into the network without generating output until we need to predict.
- For RNN: the input shape is (sequence_length, batch_size, feature_size), first is time_step, third is one-hot dim or word2vec dim.
- **Backpropagation through time**
  - <figure style="text-align: center;">
      <img alt="ResNeXt Block" src="https://d2l.ai/_images/rnn-bptt.svg" style="background-color: white; display: inline-block;">
      <figcaption> Computation graph of RNN </figcaption>
    </figure>
  - How to reduce gradient explosion or vanishing: truncate the gradient propagete at certain time step.
  - In the img above: $$\begin{array}{lll}\frac{\partial L}{\partial h_T} = W_{qh}^{\intercal} \frac{\partial L}{\partial o_T} \\\frac{\partial L}{\partial h_t} = \sum_{i=t}^T (W_{hh}^{\intercal})^{T-i} W_{qh}^{\intercal} \frac{\partial L}{\partial o_{T+t-i}} \\ \frac{\partial L}{\partial W_{hx}} = \sum_{t=1}^T \frac{\partial L}{\partial h_t} x_t^{\intercal} \\ \frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^T \frac{\partial L}{\partial h_t} h_{t-1}^{\intercal}\end{array}$$