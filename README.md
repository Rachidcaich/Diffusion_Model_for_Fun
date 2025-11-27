

## Introduction

Generative models aim to learn the underlying data distribution and sample new data from it. Popular approaches such as generative adversarial networks (GANs), variational auto-encoders (VAEs) and flow-based models each have strengths and weaknesses: GANs rely on adversarial training, VAEs optimize a surrogate loss and flow models need reversible transformations. Diffusion models are a newer class of generative models inspired by non-equilibrium thermodynamics. They define a Markov chain that gradually adds Gaussian noise to data and then learn to reverse that diffusion process to recover the original data. Because the latent variable has the **same dimensionality as the data**, diffusion models offer a flexible yet tractable way to model complex data distributions.

The figure below summarizes where diffusion models sit relative to GANs, VAEs and flow-based models. In GANs, a discriminator tries to distinguish real and synthetic samples while a generator learns to fool it. VAEs use an encoder–decoder pair to maximize a variational lower bound. Flow-based models learn an invertible transformation between data and latent variables. Diffusion models, in contrast, repeatedly corrupt a data point with Gaussian noise and then learn to reverse the noise process back to a sample.

![Comparative overview of generative models]({{file\:file-6DMnKfiDZrYVZsvvqCBDq3}})

---

## Forward diffusion process

Given a data sample $\mathbf{x}_0$ drawn from the true data distribution $q(\mathbf{x})$, the forward diffusion process adds a small amount of Gaussian noise in $T$ discrete steps. Each step produces a noisy sample $\mathbf{x}*t$ from the previous sample $\mathbf{x}*{t-1}$ using a variance schedule ${\beta_t \in (0,1)}*{t=1}^T$:

$$
q(\mathbf{x}*t \mid \mathbf{x}*{t-1}) = \mathcal{N}\bigl(\sqrt{1-\beta_t},\mathbf{x}*{t-1},; \beta_t\mathbf{I}\bigr),
\qquad
q(\mathbf{x}*{1:T} \mid \mathbf{x}*0) = \prod*{t=1}^T q(\mathbf{x}*t \mid \mathbf{x}*{t-1}).
$$

As noise is repeatedly added, the sample loses its structure; in the limit $T \to \infty$, the distribution $\mathbf{x}_T$ approaches an isotropic Gaussian. A useful property is that one can sample $\mathbf{x}_t$ in closed form directly from $\mathbf{x}_0$ using the cumulative product $\bar{\alpha}*t = \prod*{i=1}^t (1 - \beta_i)$. The resulting marginal distribution is Gaussian with mean $\sqrt{\bar{\alpha}_t},\mathbf{x}_0$ and variance $(1-\bar{\alpha}_t)\mathbf{I}$. This closed-form formula enables efficient forward simulation.

The connection to stochastic gradient Langevin dynamics (SGLD) is instructive. SGLD samples from a probability density $p(\mathbf{x})$ by taking stochastic gradient steps and injecting Gaussian noise:

$$
\mathbf{x}*t = \mathbf{x}*{t-1} + \tfrac{\delta}{2}\nabla_{\mathbf{x}}\log p(\mathbf{x}_{t-1}) + \sqrt{\delta},\boldsymbol{\epsilon}_t.
$$

When the step size $\delta$ is small and the number of steps large, the sample converges to the true density. Diffusion models can be viewed as learning to reverse such a stochastic process.

---

## Reverse diffusion and training

### Reverse process

While the forward process is fixed, sampling requires reversing the chain to recover $\mathbf{x}_0$ from isotropic noise $\mathbf{x}*T$. To do so, we would ideally sample from the true reverse conditional $q(\mathbf{x}*{t-1}\mid\mathbf{x}*t)$. Unfortunately this distribution depends on the entire dataset and is intractable. Diffusion models instead train a neural network $p*\theta$ to approximate it. The reverse process is modeled as:

$$
p_\theta(\mathbf{x}*{0:T}) = p(\mathbf{x}*T),\prod*{t=1}^T p*\theta(\mathbf{x}*{t-1}\mid\mathbf{x}*t),
\qquad
p*\theta(\mathbf{x}*{t-1}\mid\mathbf{x}*t) = \mathcal{N}\bigl(\boldsymbol{\mu}*\theta(\mathbf{x}*t,t),;\boldsymbol{\Sigma}*\theta(\mathbf{x}_t,t)\bigr).
$$

Because the true conditional $q(\mathbf{x}_{t-1}\mid \mathbf{x}_t,\mathbf{x}_0)$ is tractable for fixed $\mathbf{x}_0$, it can be written as a Gaussian with mean $\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t,\mathbf{x}_0)$ and variance $\tilde{\beta}*t$. These parameters involve the forward variance schedule $\beta_t$, cumulative product $\bar{\alpha}*t$ and the injected noise at step $t$. The goal of training is to make $p*\theta(\mathbf{x}*{t-1}\mid\mathbf{x}_t)$ match the true reverse conditional.

---

### Parameterizing the training objective

The network learns to predict the mean of the reverse process by predicting the noise added at each step. Rewriting the true mean yields:

$$
\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}}\Bigl(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_t\Bigr).
$$

Instead of directly predicting $\tilde{\boldsymbol{\mu}}*t$, we train $\boldsymbol{\mu}*\theta$ to output a prediction of the noise $\boldsymbol{\epsilon}_t$. The parameterization becomes:

$$
\boldsymbol{\mu}_\theta(\mathbf{x}_t,t) = \frac{1}{\sqrt{\alpha_t}}\Bigl(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}*t}}\boldsymbol{\epsilon}*\theta(\mathbf{x}_t,t)\Bigr).
$$

The training loss at step $t$ is the expected squared error between true and predicted noise:

$$
L_{\text{simple}} = \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}_t}\Bigl[\big|\boldsymbol{\epsilon}*t - \boldsymbol{\epsilon}*\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_t,,t)\big|^2\Bigr].
$$

---

### Noise-conditioned score networks

Score-based diffusion relates noise prediction to the score function:

$$
\mathbf{s}_\theta(\mathbf{x}*t,t) \approx \nabla*{\mathbf{x}_t}\log q(\mathbf{x}*t)
= -\frac{\boldsymbol{\epsilon}*\theta(\mathbf{x}_t,t)}{\sqrt{1-\bar{\alpha}_t}}.
$$

---

## Variance schedules and parameterization

### Choosing the noise schedule $\beta_t$

Ho et al. use a linear schedule from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$.
Nichol & Dhariwal propose a cosine schedule:

$$
\beta_t = \text{clip}\Bigl(1 - \frac{\bar{\alpha}*t}{\bar{\alpha}*{t-1}},,0.999\Bigr),
\qquad
\bar{\alpha}_t \propto \cos^2\Bigl(\frac{t/T+s}{1+s}\frac{\pi}{2}\Bigr).
$$

---

### Reverse process variance $\boldsymbol{\Sigma}_\theta$

Original DDPM sets:

$$
\tilde{\beta}*t = \frac{1-\bar{\alpha}*{t-1}}{1-\bar{\alpha}_t}\beta_t.
$$

Nichol & Dhariwal interpolate:

$$
\boldsymbol{\Sigma}_\theta(\mathbf{x}_t,t)
= \exp\bigl(\mathbf{v}\log \beta_t + (1-\mathbf{v})\log \tilde{\beta}_t\bigr).
$$

---

## Conditioned generation

### Classifier-guided diffusion

Noise is modified using classifier gradient:

$$
\bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}*t,t) = \boldsymbol{\epsilon}*\theta(\mathbf{x}_t,t)  \sqrt{1-\bar{\alpha}*t},w,\nabla*{\mathbf{x}*t}\log f*\phi(y\mid\mathbf{x}_t).
  $$

---

### Classifier-free guidance

$$
\bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}*t,t,y)
= (w+1),\boldsymbol{\epsilon}*\theta(\mathbf{x}_t,t,y)

* w,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t).
  $$

---

## Speeding up diffusion sampling

### DDIM

$$
q_\sigma(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0),
\qquad
\sigma_t^2 = \eta,\tilde{\beta}_t.
$$

---

## Latent diffusion

Encoder & decoder:

$$
\mathbf{x}\in\mathbb{R}^{H\times W\times 3}
\ \xrightarrow{\mathcal{E}}
\mathbf{z}\in\mathbb{R}^{h\times w\times c}
\ \xrightarrow{\mathcal{D}}
\mathbf{x}.
$$



Would you like a clean final README output? 📄🚀
