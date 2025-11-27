

## Introduction

Generative models aim to learn the underlying data distribution and sample new data from it.  Popular approaches such as generative adversarial networks (GANs), variational auto‑encoders (VAEs) and flow‑based models each have strengths and weaknesses: GANs rely on adversarial training, VAEs optimize a surrogate loss and flow models need reversible transformations.  Diffusion models are a newer class of generative models inspired by non‑equilibrium thermodynamics.  They define a Markov chain that gradually adds Gaussian noise to data and then learn to reverse that diffusion process to recover the original data.  Because the latent variable has the same dimensionality as the data, diffusion models offer a flexible yet tractable way to model complex data distributions.

The figure below summarizes where diffusion models sit relative to GANs, VAEs and flow‑based models.  In GANs, a discriminator tries to distinguish real and synthetic samples while a generator learns to fool it.  VAEs use an encoder–decoder pair to maximize a variational lower bound.  Flow‑based models learn an invertible transformation between data and latent variables.  Diffusion models, in contrast, repeatedly corrupt a data point with Gaussian noise and then learn to reverse the noise process back to a sample.

![Comparative overview of generative models]({{file\:file-6DMnKfiDZrYVZsvvqCBDq3}})

## Forward diffusion process

Given a data sample (\mathbf{x}*0) drawn from the true data distribution (q(\mathbf{x})), the forward diffusion process adds a small amount of Gaussian noise in (T) discrete steps.  Each step produces a noisy sample (\mathbf{x}*t) from the previous sample (\mathbf{x}*{t-1}) using a variance schedule ({\beta_t \in (0,1)}*{t=1}^T):

[
q(\mathbf{x}*t\vert\mathbf{x}*{t-1}) = \mathcal{N}\bigl(\sqrt{1-\beta_t},\mathbf{x}*{t-1},; \beta_t\mathbf{I}\bigr),
\quad q(\mathbf{x}*{1:T}\vert\mathbf{x}*0) = \prod*{t=1}^T q(\mathbf{x}*t\vert\mathbf{x}*{t-1}).
]

As noise is repeatedly added, the sample loses its structure; in the limit (T \to \infty), the distribution (\mathbf{x}_T) approaches an isotropic Gaussian.  A useful property is that one can sample (\mathbf{x}_t) in closed form directly from (\mathbf{x}_0) using the cumulative product $bar{\alpha}*t = \prod*{i=1}^t (1 - \beta_i)$.  The resulting marginal distribution is Gaussian with mean (\sqrt{\bar{\alpha}_t},\mathbf{x}_0) and variance ((1-\bar{\alpha}_t)\mathbf{I}).  This closed‑form formula enables efficient forward simulation.

The connection to stochastic gradient Langevin dynamics (SGLD) is instructive.  SGLD samples from a probability density (p(\mathbf{x})) by taking stochastic gradient steps and injecting Gaussian noise: (\mathbf{x}*t = \mathbf{x}*{t-1} + \tfrac{\delta}{2}\nabla_{\mathbf{x}}\log p(\mathbf{x}_{t-1}) + \sqrt{\delta},\boldsymbol{\epsilon}_t).  When the step size (\delta) is small and the number of steps large, the sample converges to the true density.  Diffusion models can be viewed as learning to reverse such a stochastic process.

## Reverse diffusion and training

### Reverse process

While the forward process is fixed, sampling requires reversing the chain to recover (\mathbf{x}_0) from isotropic noise (\mathbf{x}*T).  To do so, we would ideally sample from the true reverse conditional (q(\mathbf{x}*{t-1}\vert\mathbf{x}*t)).  Unfortunately this distribution depends on the entire dataset and is intractable.  Diffusion models instead train a neural network (p*\theta) to approximate it.  The reverse process is modeled as

[
p_\theta(\mathbf{x}*{0:T}) = p(\mathbf{x}*T),\prod*{t=1}^T p*\theta(\mathbf{x}*{t-1}\vert\mathbf{x}*t),
\quad p*\theta(\mathbf{x}*{t-1}\vert\mathbf{x}*t) = \mathcal{N}\bigl(\boldsymbol{\mu}*\theta(\mathbf{x}*t,t),;\boldsymbol{\Sigma}*\theta(\mathbf{x}_t,t)\bigr)
.
]

Because the true conditional (q(\mathbf{x}_{t-1}\vert\mathbf{x}_t,\mathbf{x}_0)) is tractable for fixed (\mathbf{x}_0), it can be written as a Gaussian with mean (\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t,\mathbf{x}_0)) and variance (\tilde{\beta}*t).  These parameters involve the forward variance schedule (\beta_t), cumulative product (\bar{\alpha}*t) and the noise at step (t).  The goal of training is to make (p*\theta(\mathbf{x}*{t-1}\vert\mathbf{x}_t)) match this true reverse conditional.

### Parameterizing the training objective

The network learns to predict the mean of the reverse process by predicting the noise added at each step.  Rewriting the true mean yields

[
\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}}\Bigl(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}},\boldsymbol{\epsilon}_t\Bigr).
]

Instead of directly predicting (\tilde{\boldsymbol{\mu}}*t), we train (\boldsymbol{\mu}*\theta) to output a prediction of the noise (\boldsymbol{\epsilon}_t).  The parameterization becomes

[
\boldsymbol{\mu}_\theta(\mathbf{x}_t,t) = \frac{1}{\sqrt{\alpha_t}}\Bigl(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}*t}},\boldsymbol{\epsilon}*\theta(\mathbf{x}_t,t)\Bigr).
]

The training loss at step (t), derived from the variational lower bound, is the expected squared error between the true and predicted mean scaled by the variance of the reverse distribution.  Simplifying and ignoring weighting yields the commonly used “simple” objective

[
L_{\text{simple}} = \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}_t}\bigl[|\boldsymbol{\epsilon}*t - \boldsymbol{\epsilon}*\theta(\sqrt{\bar{\alpha}_t},\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t},\boldsymbol{\epsilon}_t, t)|^2\bigr].
]

Empirically, this simplified loss leads to stable and effective training.  The network is thus trained to denoise a noisy input and predict the injected noise, encouraging it to reverse the forward diffusion.

### Noise‑conditioned score networks

Song and Ermon proposed an alternative score‑based generative model that estimates the score (\nabla_{\mathbf{x}}\log q(\mathbf{x}_t)) of the perturbed data distribution and samples via Langevin dynamics.  To stabilize score estimation, they add noise at multiple levels so that perturbed data covers the full space and train a noise‑conditioned score network jointly across noise levels.  In diffusion notation, the score network approximates

[
\mathbf{s}_\theta(\mathbf{x}*t,t) \approx \nabla*{\mathbf{x}_t}\log q(\mathbf{x}*t) = -\frac{\boldsymbol{\epsilon}*\theta(\mathbf{x}_t,t)}{\sqrt{1-\bar{\alpha}_t}}.
]

This connection shows that diffusion models and score‑based models are closely related; both learn to estimate the gradient of the data density at different noise scales.

## Variance schedules and parameterization

### Choosing the noise schedule (\beta_t)

The forward variance schedule (\beta_t) determines how much noise is added at each step.  Ho et al. proposed using a linear schedule from (\beta_1=10^{-4}) to (\beta_T=0.02).  Nichol & Dhariwal observed that this choice does not optimize likelihood well and introduced a cosine schedule defined by (\beta_t = \text{clip}\bigl(1 - \tfrac{\bar{\alpha}*t}{\bar{\alpha}*{t-1}}, 0.999\bigr)) with (\bar{\alpha}_t\propto \cos^2\bigl(\tfrac{t/T+s}{1+s},\tfrac{\pi}{2}\bigr)).  The cosine schedule provides a near‑linear drop in the middle of training and subtle changes near the ends, improving sample quality.

### Reverse process variance (\boldsymbol{\Sigma}_\theta)

Originally, DDPM fixes the reverse variance (\boldsymbol{\Sigma}_\theta) to a constant diagonal matrix based on (\beta_t) or (\tilde{\beta}*t = \frac{1-\bar{\alpha}*{t-1}}{1-\bar{\alpha}*t},\beta_t).  Learning a diagonal covariance leads to unstable training.  Nichol & Dhariwal proposed learning (\boldsymbol{\Sigma}*\theta) via an interpolation between (\beta_t) and (\tilde{\beta}*t): (\boldsymbol{\Sigma}*\theta(\mathbf{x}_t,t) = \exp\bigl(\mathbf{v}\log \beta_t + (1-\mathbf{v})\log \tilde{\beta}_t\bigr)) where (\mathbf{v}) is predicted by the model.  They train this variance using a hybrid objective combining the simple objective and a variational bound term weighted by a small (\lambda), and apply time‑averaging with importance sampling to stabilize optimisation.

## Conditioned generation

Diffusion models can generate images conditioned on class labels or descriptive text.  Two principal methods have emerged.

### Classifier‑guided diffusion

Dhariwal & Nichol train a separate classifier (f_\phi(y\vert\mathbf{x}*t,t)) on noisy data and use its gradient (\nabla*{\mathbf{x}*t}\log f*\phi(y\vert\mathbf{x}*t)) to guide the reverse process towards a target label.  The joint score (\nabla*{\mathbf{x}_t}\log q(\mathbf{x}_t,y)) combines the unconditional score and the classifier gradient, resulting in a modified noise prediction

[
\bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}*t,t) = \boldsymbol{\epsilon}*\theta(\mathbf{x}_t,t) - \sqrt{1-\bar{\alpha}*t}, w \nabla*{\mathbf{x}*t}\log f*\phi(y\vert\mathbf{x}_t),
]

where (w) controls the strength of the classifier guidance.  This classifier‑guided model, referred to as ADM‑G, achieves state‑of‑the‑art results on image synthesis when combined with architectural improvements like larger U‑Nets, attention modules and BigGAN‑style residual blocks.

### Classifier‑free guidance

Ho & Salimans showed that conditional generation can be performed without a separate classifier.  A single network is trained to produce both unconditional predictions (\boldsymbol{\epsilon}_\theta(\mathbf{x}*t,t)) and conditional predictions (\boldsymbol{\epsilon}*\theta(\mathbf{x}_t,t,y)) by randomly dropping the conditioning information during training.  The implicit classifier gradient can be computed from the difference between conditional and unconditional scores.  The resulting guidance is

[
\bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}*t,t,y) = (w+1),\boldsymbol{\epsilon}*\theta(\mathbf{x}*t,t,y) - w,\boldsymbol{\epsilon}*\theta(\mathbf{x}_t,t).
]

Classifier‑free guidance avoids adversarial effects from an external classifier and has been found to produce a better balance between image diversity and fidelity.  GLIDE and other text‑to‑image diffusion models adopt this strategy and further explore CLIP guidance and cross‑attention for text conditioning.

## Speeding up diffusion sampling

### Fewer sampling steps and DDIM

Generating samples from DDPM requires following a long Markov chain ((T) can be thousands of steps), making inference slow.  Strided sampling schedules take updates every (\lceil T/S \rceil) steps to reduce the chain length.  Song et al. derive a parameterization (q_{\sigma}(\mathbf{x}_{t-1}\vert\mathbf{x}_t,\mathbf{x}_0)) that introduces a tunable standard deviation (\sigma_t).  Setting (\sigma_t^2 = \eta,\tilde{\beta}_t) allows controlling the sampling stochasticity; the special case (\eta = 0) yields the **denoising diffusion implicit model** (DDIM) which follows the same marginal distribution as DDPM but performs deterministic sampling.

With DDIM, one can sample using a subset of timesteps.  Experiments show that DDIM produces higher‑quality samples when the number of sampling steps (S) is small, while DDPM performs better when using the full chain.  DDIM also has a consistency property: samples obtained from the same latent variable share high‑level features and allow semantically meaningful interpolation.

### Progressive distillation and consistency models

Salimans & Ho proposed **progressive distillation**, which distills a deterministic sampler into a new model that uses half the number of sampling steps.  In each iteration, the student model learns to match two teacher steps with one student step; repeated distillation halves the sampling cost while retaining quality.

Consistency models extend this idea by training a network to map any noisy sample (\mathbf{x}*t) along the diffusion trajectory directly to the same origin (\mathbf{x}*\epsilon).  The model outputs must satisfy (f(\mathbf{x}*t,t) = f(\mathbf{x}*{t'},t')) for any (t,t') in the interval.  The parameterization uses time‑dependent skip and output coefficients (c_{\text{skip}}(t)) and (c_{\text{out}}(t)) such that (c_{\text{skip}}(\epsilon)=1) and (c_{\text{out}}(\epsilon)=0).  Two training schemes exist: *consistency distillation*, which distills a pre‑trained diffusion model using pairs of points on the same trajectory, and *consistency training*, which trains from scratch by estimating the score function with an unbiased estimator and using an ODE solver.  Consistency models can generate high‑quality samples in a single or few steps and provide a flexible trade‑off between speed and quality.

## Latent diffusion and scaling to high resolution

### Latent diffusion models

Rombach et al. observed that most bits in an image describe perceptual details; compressing images into a latent space retains semantic content while reducing dimensionality.  **Latent diffusion models (LDMs)** first use an autoencoder with encoder (\mathcal{E}) to compress an image (\mathbf{x}\in\mathbb{R}^{H\times W\times 3}) into a latent representation (\mathbf{z}\in\mathbb{R}^{h\times w\times c}), and a decoder (\mathcal{D}) to reconstruct the image.  Regularization such as KL penalties or vector quantization prevents the latent from having unbounded variance.  The diffusion process runs in the latent space, which greatly reduces training cost and speeds up inference.  A time‑conditioned U‑Net with cross‑attention handles conditioning information (class labels, semantic maps, etc.) by projecting the conditioning input through a domain‑specific encoder and injecting it into the U‑Net via attention.

### Cascaded diffusion and unCLIP

To synthesize high‑resolution images, Ho et al. propose a cascade of multiple diffusion models operating at increasing resolutions.  Strong data augmentation (adding Gaussian noise at low resolution and Gaussian blur at high resolution) to the conditioning inputs helps reduce error propagation across the pipeline.  Truncated and non‑truncated conditioning augmentations modify how the low‑resolution reverse process is corrupted before being fed to a super‑resolution model.

The two‑stage model **unCLIP** leverages a pretrained CLIP text–image encoder.  A prior model produces a CLIP image embedding given a text prompt, and a decoder generates an image conditioned on that embedding.  This decouples text‑to‑image synthesis into a latent prior and a decoder, allowing unCLIP to perform zero‑shot image manipulation and variation.

**Imagen** replaces the CLIP encoder with a large language model (T5‑XXL) to encode text and finds that larger text encoders lead to better image‑text alignment.  Dynamic thresholding clips predictions during sampling to handle train–test mismatch and improve fidelity.  Additional architectural tweaks such as shifting parameters toward low‑resolution blocks and scaling skip connections improve efficiency.

## Model architectures

### U‑Net and ControlNet

Most diffusion models use a U‑Net backbone.  U‑Net consists of a downsampling stack of repeated 3×3 convolutions with ReLU activation and max pooling, followed by an upsampling stack that upsamples feature maps and halves the number of channels.  Skip connections concatenate feature maps from the encoder to the decoder, preserving high‑resolution details.

For conditional generation requiring additional input (e.g., edges, human poses or depth maps), ControlNet augments U‑Net by inserting a trainable copy of each encoder layer and two zero‑initialized 1×1 convolutions.  The original parameters are frozen and the conditioned branch learns to modulate features via the zero convolutions, effectively controlling the generation while protecting the base model from noisy gradients.

### Diffusion Transformer (DiT)

DiT brings Transformer architectures to diffusion models.  It operates on latent patches: the latent representation (\mathbf{z}) is divided into patches and processed by a sequence of Transformer blocks.  Conditioning information such as the timestep (t) or class label (c) is injected via adaptive layer normalization (adaLN‑Zero).  The transformer predicts both the noise and a diagonal covariance and scales efficiently with model size.  Experiments show that scaling DiT models improves generation quality and computational efficiency.

## Quick summary and outlook

Diffusion models strike a balance between tractability and flexibility.  They are analytically tractable and allow exact likelihood evaluation, yet can model rich, high‑dimensional data.  Their main drawback is the cost of sampling: reversing a long Markov chain requires many neural network evaluations.  Research on fast sampling—including DDIM, progressive distillation, consistency models and latent‑space diffusion—makes diffusion models increasingly practical.  Meanwhile, innovations in conditioning (classifier‑free guidance), noise schedules and architectures (U‑Net, ControlNet, DiT) continue to improve sample quality, efficiency and controllability.  Diffusion models have become a central tool for text‑to‑image generation and offer a flexible framework that continues to evolve.

---

> **Note**: The image reference uses a placeholder (`{{file:file-6DMnKfiDZrYVZsvvqCBDq3}}`). In your GitHub repository, you should include the image file (e.g., `generative-overview.png`) and update the image link to its relative path.
