

# Pytorch-stable-diffusion

PyTorch implementation of Stable Diffusion from scratch

Download weights and tokenizer files:

- Download vocab.json and merges.txt from [https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/tokenizer](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/tokenizer) and save them in the data folder
- Download v1-5-pruned-emaonly.ckpt from [https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main) and save it in the data folder



Generative models aim to learn the underlying data distribution and sample new data from it. Popular approaches such as generative adversarial networks (GANs), variational auto-encoders (VAEs) and flow-based models each have strengths and weaknesses: GANs rely on adversarial training, VAEs optimize a surrogate loss and flow models need reversible transformations. Diffusion models are a newer class of generative models inspired by non-equilibrium thermodynamics. They define a Markov chain that gradually adds Gaussian noise to data and then learn to reverse that diffusion process to recover the original data. Because the latent variable has the **same dimensionality as the data**, diffusion models offer a flexible yet tractable way to model complex data distributions.

---

## Forward diffusion process

Given a data sample $x_0$ drawn from the true data distribution $q(x)$, the forward diffusion process adds a small amount of Gaussian noise in $T$ discrete steps. Each step produces a noisy sample $x_t$ from the previous sample $x_{t-1}$ using a variance schedule ${\beta_t \in (0,1)}_{t=1}^T$:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\bigl(\sqrt{1-\beta_t},x_{t-1}, \beta_t\mathbf{I}\bigr),
\qquad
q(x_{1:T} \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1}).
$$

As noise is repeatedly added, the sample loses its structure; in the limit $T \to \infty$, the distribution $x_T$ approaches an isotropic Gaussian. A useful property is that one can sample $x_t$ in closed form directly from $x_0$ using the cumulative product $\bar{\alpha}_t = \prod_{i=1}^t (1 - \beta_i)$. The resulting marginal distribution is Gaussian with mean $\sqrt{\bar{\alpha}_t},x_0$ and variance $(1-\bar{\alpha}_t)\mathbf{I}$. This closed-form formula enables efficient forward simulation.

The connection to stochastic gradient Langevin dynamics (SGLD) is instructive. SGLD samples from a probability density $p(x)$ by taking stochastic gradient steps and injecting Gaussian noise:

$$
x_t = x_{t-1} + \tfrac{\delta}{2}\nabla_{x}\log p(x_{t-1}) + \sqrt{\delta},\boldsymbol{\epsilon}_t.
$$

When the step size $\delta$ is small and the number of steps large, the sample converges to the true density. Diffusion models can be viewed as learning to reverse such a stochastic process.

---

## Reverse diffusion and training

### Reverse process

While the forward process is fixed, sampling requires reversing the chain to recover $x_0$ from isotropic noise $x_T$. To do so, we would ideally sample from the true reverse conditional $q(x_{t-1}\mid x_t)$. Unfortunately this distribution depends on the entire dataset and is intractable. Diffusion models instead train a neural network $p_\theta$ to approximate it. The reverse process is modeled as:

$$
p_\theta(x_{0:T}) = p(x_T).\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t),
\qquad
p_\theta(x_{t-1}\mid x_t) = \mathcal{N}\bigl(\boldsymbol{\mu}_\theta(x_t,t),\boldsymbol{\Sigma}_\theta(x_t,t)\bigr).
$$

Because the true conditional $q(x_{t-1}\mid x_t,x_0)$ is tractable for fixed $x_0$, it can be written as a Gaussian with mean $\tilde{\boldsymbol{\mu}}_t(x_t,x_0)$ and variance $\tilde{\beta}_t$. These parameters involve the forward variance schedule $\beta_t$, cumulative product $\bar{\alpha}_t$ and the injected noise at step $t$. The goal of training is to make $p_\theta(x_{t-1}\mid x_t)$ match the true reverse conditional.

---

### Parameterizing the training objective

The network learns to predict the mean of the reverse process by predicting the noise added at each step. Rewriting the true mean yields:

$$
\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}}\Bigl(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_t\Bigr).
$$

Instead of directly predicting $\tilde{\boldsymbol{\mu}}_t$, we train $\boldsymbol{\mu}_\theta$ to output a prediction of the noise $\boldsymbol{\epsilon}_t$. The parameterization becomes:

$$
\boldsymbol{\mu}_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\Bigl(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(x_t,t)\Bigr).
$$

The training loss at step $t$ is the expected squared error between true and predicted noise:

$$
L_{\text{simple}} = \mathbb{E}_{t,x_0,\boldsymbol{\epsilon}_t}\Bigl[\big|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_t,,t)\big|^2\Bigr].
$$

---

### Noise-conditioned score networks

Score-based diffusion relates noise prediction to the score function:

$$
\mathbf{s}_\theta(x_t,t) \approx \nabla_{x_t}\log q(x_t)
= -\frac{\boldsymbol{\epsilon}_\theta(x_t,t)}{\sqrt{1-\bar{\alpha}_t}}.
$$

---

## Variance schedules and parameterization

### Choosing the noise schedule $\beta_t$

Ho et al. use a linear schedule from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$.
Nichol & Dhariwal propose a cosine schedule:

$$
\beta_t = \text{clip}\Bigl(1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}},0.999\Bigr),
\qquad
\bar{\alpha}_t \propto \cos^2\Bigl(\frac{t/T+s}{1+s}\frac{\pi}{2}\Bigr).
$$

---

### Reverse process variance $\boldsymbol{\Sigma}_\theta$

Original DDPM sets:

$$
\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t.
$$

Nichol & Dhariwal interpolate:

$$
\boldsymbol{\Sigma}_\theta(x_t,t)
= \exp\bigl(\mathbf{v}\log \beta_t + (1-\mathbf{v})\log \tilde{\beta}_t\bigr).
$$

---

## Conditioned generation

### Classifier-guided diffusion

Noise is modified using classifier gradient:

$$
\bar{\boldsymbol{\epsilon}}_\theta(x_t,t) = \boldsymbol{\epsilon}_\theta(x_t,t) - \sqrt{1-\bar{\alpha}_t} w \nabla_{x_t}\log f_\phi(y\mid x_t).
  $$

---

### Classifier-free guidance

The gradient of an implicit classifier can be expressed using both conditional and unconditional score estimators. After substituting these into the classifier-guided score formulation, the resulting score no longer relies on an external classifier.

$$
\nabla_{x_t} \log p(y|x_t)
 = \nabla_{x_t} \log p(x_t|y) - \nabla_{x_t} \log p(x_t) = -\frac{1}{\sqrt{1 - \bar{\alpha}_t}}
\left( \epsilon_{\theta}(x_t,t,y) - \epsilon_{\theta}(x_t,t) \right)
$$

$$
\bar{\epsilon}_{\theta}(x_t,t,y) = \epsilon_{\theta}(x_t,t,y) - \sqrt{1 - \bar{\alpha}_t}\, w \nabla_{x_t}\log p(y|x_t) = (w+1)\epsilon_{\theta}(x_t,t,y) - w\epsilon_{\theta}(x_t,t)
$$




