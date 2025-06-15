## Chapter 20 : Generative Adversarial Networks
<figure style="text-align: center;">
  <img src="https://d2l.ai/_images/gan.svg" alt="GAN" style="background-color: white; display: inline-block;"/>
  <figcaption> GAN </figcaption>
</figure>

- The discriminator is a binary classifier which outputs a saclar prediction $o$, so $D(x) = 1/(1+e^{-o})$, assume the label for $y$ for true data is 1, and 0 for fake data, we have:
  $$\min_D\left\{ -y\log D(x)-(1-y)\log (1-D(x)) \right\}$$
  The generator is $x^{'} = G(z)$, where $z \sim \mathcal N(0,1)$ is called latent variable. The goal is find $G$ that maximize the loss when $y=0$:
  $$\max_G \left \{ -(1-y)\log (1-D(G(z))) \right \} = \max_G \left \{ -\log (1-D(G(z))) \right \}$$
  But in this way if the generator is good, the gradient will be too small for discriminator to evolve, so we choos to minimize this (given $y=1$):
  $$\min_G \left \{ -y\log D(G(z)) \right \}=\min_G \left \{ -\log D(G(z)) \right \}$$
  So the overall objective function is:
  $$\min_D\max_G \left \{ -E_{x\sim \text{Data}}\log D(x) - E_{z \sim \text{Noise}}\log (1-D(G(z)))\right\}$$
- Deep Convolutional GAN
  - Generator use transposed convolution, discriminator use convolution, and the learning rate of them is same, adam smooth term $\beta$ set to 0.5 to deal with rapidly changing gradient. A leakly RELU is used.