## Chapter 16 : NLP Applications
<figure style="text-align: center;">
  <img alt="nlp-map-app" src="https://d2l.ai/_images/nlp-map-app.svg" style="background-color: white; display: inline-block;">
  <figcaption> NLP Applications </figcaption>
</figure>

- Sentiment Analysis using RNN
  - Glove (embedding, frozen) -> Bidirection RNN (encoder) -> linear (decoder)
- Using CNN:
  - *One dimensional correlation of mutiple channel is same as two dimensional corelation*.
  - *Max-over-time Pool*: pool the sequence into certain width (different channel no operation).
  - <figure style="text-align: center;">
      <img alt="textcnn" src="https://d2l.ai/_images/textcnn.svg" style="background-color: white; display: inline-block;">
      <figcaption> Text CNN </figcaption>
    </figure>
- Natural Language Processing:
  - Three type: entailment (positive next), contradiction (negitive next), neutral (not next).
  - Given premise and hypohesis tokens $A = (a_1,  \ldots, a_m), B = (b_1,  \ldots, b_m)$, where $a_i, b_j \in \mathbb R^d$ is word vector. The attenton is $e_{ij} = f(a_i)^{\top}f(b_j)$, $f$ is a mlp layer. We define the soft aligned representation:
    $$\boldsymbol{\beta}_i = \sum_{j=1}^{n}\frac{\exp(e_{ij})}{ \sum_{k=1}^{n} \exp(e_{ik})} \mathbf{b}_j.$$
    $$\boldsymbol{\alpha}_j = \sum_{i=1}^{m}\frac{\exp(e_{ij})}{ \sum_{k=1}^{m} \exp(e_{kj})} \mathbf{a}_i.$$
    Then we compare this representations with word vectors (again $g$ is a mlp layer):
    $$\begin{split}\mathbf{v}_{A,i} = g([\mathbf{a}_i, \boldsymbol{\beta}_i]), i = 1, \ldots, m\\ \mathbf{v}_{B,j} = g([\mathbf{b}_j, \boldsymbol{\alpha}_j]), j = 1, \ldots, n.\end{split}$$
    Then we aggregating the information:
    $$\mathbf{v}_A = \sum_{i=1}^{m} \mathbf{v}_{A,i}, \quad \mathbf{v}_B = \sum_{j=1}^{n}\mathbf{v}_{B,j}.$$
    and then feed them into a mlp layer then get the classification result: $\hat{\mathbf{y}} = h([\mathbf{v}_A, \mathbf{v}_B]).$
- Fine-tune BERT: single text, text pair, text tag (on every token representation add a shared dense layer), Q&A (first sequence as question, second as passage, from passage find out answer for question, we also need additional two output layer for predicting start pos and end pos, and the prediction is[start, end] pair).
<!-- <img alt="ResNeXt Block" src="https://d2l.ai/_images/rnn.svg" style="background-color: white; display: inline-block;"> -->
<!-- <img alt="ResNeXt Block" src="https://d2l.ai/_images/rnn-bptt.svg" style="background-color: white; display: inline-block;"> -->