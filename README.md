<!-- MathJax for LaTeX rendering -->
<script type="text/javascript" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>




[Updated on 2021-09-19: Highly recommend this blog post on score-based generative modeling by Yang Song (author of several key papers in the references)].\\
[Updated on 2022-08-27: Added classifier-free guidance, GLIDE, unCLIP and Imagen.]\\
[Updated on 2022-08-31: Added latent diffusion model.]\\
[Updated on 2024-04-13: Added progressive distillation, consistency models, and the Model Architecture section.]
\end{abstract}

\section{Introduction}

So far, I’ve written about three types of generative models: GANs, VAEs, and flow-based models. They have shown great success in generating high-quality samples, but each has some limitations of its own. GAN models are known for potentially unstable training and less diversity in generation due to their adversarial training nature. VAEs rely on a surrogate loss. Flow models have to use specialized architectures to construct reversible transforms.

Diffusion models are inspired by non-equilibrium thermodynamics. They define a Markov chain of diffusion steps to slowly add random noise to data and then learn to reverse the diffusion process to construct desired data samples from the noise. Unlike VAEs or flow models, diffusion models are learned with a fixed procedure and the latent variable has high dimensionality (same as the original data).

Several diffusion-based generative models have been proposed with similar ideas underneath, including diffusion probabilistic models~\cite{sohl2015deep}, noise-conditioned score networks (NCSN;~\cite{song2019generative}), and denoising diffusion probabilistic models (DDPM;~\cite{ho2020denoising}).

\section{Forward Diffusion Process}

Given a data point sampled from a real data distribution
\[
\mathbf{x}_0 \sim q(\mathbf{x}),
\]
we define a forward diffusion process in which we add a small amount of Gaussian noise to the sample in $T$ steps, producing a sequence of noisy samples
\[
\mathbf{x}_1, \dots, \mathbf{x}_T.
\]
The step sizes are controlled by a variance schedule
\[
\{\beta_t \in (0, 1)\}_{t=1}^T.
\]

We define
\begin{align}
q(\mathbf{x}_t \vert \mathbf{x}_{t-1})
  &= \mathcal{N}\bigl(\mathbf{x}_t;\, \sqrt{1 - \beta_t}\,\mathbf{x}_{t-1},\, \beta_t \mathbf{I}\bigr), \\
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)
  &= \prod_{t=1}^T q(\mathbf{x}_t \vert \mathbf{x}_{t-1}).
\end{align}

As $t$ increases, the sample $\mathbf{x}_t$ gradually loses its distinguishable features. Eventually, when $T \to \infty$, $\mathbf{x}_T$ becomes equivalent to an isotropic Gaussian distribution.

\subsection{Closed-form Sampling at Arbitrary Timestep}

A nice property of the above process is that we can sample $\mathbf{x}_t$ at any arbitrary timestep $t$ in closed form using the reparameterization trick. Let
\[
\alpha_t = 1 - \beta_t,\qquad
\bar{\alpha}_t = \prod_{i=1}^t \alpha_i.
\]

We can unroll the Markov chain:
\begin{align}
\mathbf{x}_t
  &= \sqrt{\alpha_t}\,\mathbf{x}_{t-1}
     + \sqrt{1 - \alpha_t}\,\bm{\epsilon}_{t-1},
     &&\text{where }\bm{\epsilon}_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
  &= \sqrt{\alpha_t \alpha_{t-1}}\,\mathbf{x}_{t-2}
     + \sqrt{1 - \alpha_t \alpha_{t-1}}\,\bar{\bm{\epsilon}}_{t-2}
     &&\text{($\bar{\bm{\epsilon}}_{t-2}$ merges two Gaussians)} \\
  &\;\dots \\
  &= \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0
     + \sqrt{1 - \bar{\alpha}_t}\,\bm{\epsilon},
\end{align}
where $\bm{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. Therefore,
\begin{equation}
q(\mathbf{x}_t \vert \mathbf{x}_0)
  = \mathcal{N}\bigl(\mathbf{x}_t;\, \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\,
                     (1 - \bar{\alpha}_t)\mathbf{I}\bigr).
\end{equation}

\paragraph{Merging Gaussians.}
Recall that when we merge two Gaussians with different variances,
\[
\mathcal{N}(\mathbf{0}, \sigma_1^2 \mathbf{I}) \quad\text{and}\quad
\mathcal{N}(\mathbf{0}, \sigma_2^2 \mathbf{I}),
\]
the result is
\[
\mathcal{N}\bigl(\mathbf{0}, (\sigma_1^2 + \sigma_2^2)\mathbf{I}\bigr).
\]
Here the merged standard deviation is
\[
\sqrt{(1 - \alpha_t) + \alpha_t(1-\alpha_{t-1})}
  = \sqrt{1 - \alpha_t \alpha_{t-1}}.
\]

Usually, we can afford a larger update step when the sample gets noisier, so
\[
\beta_1 < \beta_2 < \dots < \beta_T
\quad\Longrightarrow\quad
\bar{\alpha}_1 > \bar{\alpha}_2 > \dots > \bar{\alpha}_T.
\]

\section{Connection with Stochastic Gradient Langevin Dynamics}

Langevin dynamics is a concept from physics, developed for statistically modeling molecular systems. Combined with stochastic gradient descent, stochastic gradient Langevin dynamics (SGLD;~\cite{welling2011bayesian}) can produce samples from a probability density $p(\mathbf{x})$ using only the gradients $\nabla_{\mathbf{x}} \log p(\mathbf{x})$ in a Markov chain of updates:
\begin{equation}
\mathbf{x}_t
  = \mathbf{x}_{t-1} + \frac{\delta}{2}
    \nabla_{\mathbf{x}} \log p(\mathbf{x}_{t-1})
    + \sqrt{\delta}\,\bm{\epsilon}_t,
\quad \bm{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),
\end{equation}
where $\delta$ is the step size. When $T \to \infty$ and $\delta \to 0$, the distribution of $\mathbf{x}_T$ converges to the true density $p(\mathbf{x})$.

Compared to standard SGD, SGLD injects Gaussian noise into the parameter updates to avoid collapsing into local minima.

\section{Reverse Diffusion Process}

If we can reverse the forward diffusion process and sample from $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$, we would be able to recreate the true sample from a Gaussian noise input $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

If $\beta_t$ is small enough, $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$ is also Gaussian. Unfortunately, we cannot easily estimate $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$ because it depends on the entire dataset. Therefore we learn a model $p_\theta$ to approximate these conditional probabilities in order to run the reverse diffusion process:
\begin{align}
p_\theta(\mathbf{x}_{0:T})
  &= p(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t), \\
p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)
  &= \mathcal{N}\bigl(\mathbf{x}_{t-1};
                      \bm{\mu}_\theta(\mathbf{x}_t, t),
                      \bm{\Sigma}_\theta(\mathbf{x}_t, t)\bigr).
\end{align}

It is noteworthy that the reverse conditional probability is tractable when conditioned on $\mathbf{x}_0$:
\begin{equation}
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)
  = \mathcal{N}\bigl(\mathbf{x}_{t-1};
                     \tilde{\bm{\mu}}(\mathbf{x}_t, \mathbf{x}_0),
                     \tilde{\beta}_t \mathbf{I}\bigr).
\end{equation}

Using Bayes' rule, we can derive
\begin{align}
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)
  &= q(\mathbf{x}_t \vert \mathbf{x}_{t-1}, \mathbf{x}_0)
     \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_0)}
          {q(\mathbf{x}_t \vert \mathbf{x}_0)} \\
  &\propto \exp\Big(-\frac{1}{2}\big(
    \frac{(\mathbf{x}_t - \sqrt{\alpha_t}\,\mathbf{x}_{t-1})^2}{\beta_t}
    + \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0)^2}
           {1-\bar{\alpha}_{t-1}}
    - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0)^2}
           {1-\bar{\alpha}_t}
  \big)\Big).
\end{align}

By collecting terms in $\mathbf{x}_{t-1}^2$ and $\mathbf{x}_{t-1}$ and matching to the standard Gaussian form, we obtain
\begin{align}
\tilde{\beta}_t
  &= \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\,\beta_t, \\
\tilde{\bm{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0)
  &= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\,\mathbf{x}_t
   + \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1 - \bar{\alpha}_t}\,\mathbf{x}_0.
\end{align}

Using the reparameterization
\[
\mathbf{x}_t
  = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0
    + \sqrt{1 - \bar{\alpha}_t}\,\bm{\epsilon}_t,
\]
we can also express the mean as
\begin{equation}
\tilde{\bm{\mu}}_t
  = \frac{1}{\sqrt{\alpha_t}}
    \Bigl(\mathbf{x}_t
          - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\,
            \bm{\epsilon}_t\Bigr).
\end{equation}

\section{Variational Lower Bound}

The diffusion setup is very similar to a VAE, so we can use a variational lower bound to optimize the negative log-likelihood:
\begin{align}
- \log p_\theta(\mathbf{x}_0)
  &\le - \log p_\theta(\mathbf{x}_0)
        + D_{\mathrm{KL}}\bigl(
          q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)
          \,\|\, p_\theta(\mathbf{x}_{1:T}\vert\mathbf{x}_0)
        \bigr) \\
  &= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Bigl[
      \log \frac{q(\mathbf{x}_{1:T}\vert \mathbf{x}_0)}
                {p_\theta(\mathbf{x}_{0:T})}
     \Bigr]
   =: L_{\mathrm{VLB}}.
\end{align}

This can be decomposed into a sum of KL terms:
\begin{align}
L_{\mathrm{VLB}}
  &= \mathbb{E}_q\Bigl[
       D_{\mathrm{KL}}\bigl(
         q(\mathbf{x}_T \vert \mathbf{x}_0)
         \,\|\, p_\theta(\mathbf{x}_T)
       \bigr)
       + \sum_{t=2}^T
         D_{\mathrm{KL}}\bigl(
           q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)
           \,\|\, p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)
         \bigr)
       - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)
     \Bigr] \\
  &= \mathbb{E}_q\Bigl[
       L_T + \sum_{t=1}^{T-1} L_t + L_0
     \Bigr],
\end{align}
where
\begin{align}
L_T &= D_{\mathrm{KL}}\bigl(
         q(\mathbf{x}_T \vert \mathbf{x}_0)
         \,\|\, p_\theta(\mathbf{x}_T)
       \bigr), \\
L_t &= D_{\mathrm{KL}}\bigl(
         q(\mathbf{x}_t \vert \mathbf{x}_{t+1}, \mathbf{x}_0)
         \,\|\, p_\theta(\mathbf{x}_t \vert \mathbf{x}_{t+1})
       \bigr),\qquad 1 \le t \le T-1, \\
L_0 &= -\log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1).
\end{align}

Every KL term (except $L_0$) compares two Gaussians and thus can be computed in closed form. $L_T$ is constant w.r.t.\ $\theta$ and can be ignored during training.

\section{Parameterization of the Training Loss}

We approximate the reverse conditionals with a neural network:
\[
p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)
  = \mathcal{N}\bigl(
      \mathbf{x}_{t-1};
      \bm{\mu}_\theta(\mathbf{x}_t, t),
      \bm{\Sigma}_\theta(\mathbf{x}_t, t)
    \bigr).
\]

We would like $\bm{\mu}_\theta$ to match $\tilde{\bm{\mu}}_t$, and because $\mathbf{x}_t$ is available at training time, it is convenient to let the network predict the noise $\bm{\epsilon}_t$ instead:
\begin{equation}
\bm{\mu}_\theta(\mathbf{x}_t, t)
  = \frac{1}{\sqrt{\alpha_t}}
    \Bigl(\mathbf{x}_t
          - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}
            \bm{\epsilon}_\theta(\mathbf{x}_t, t)
    \Bigr).
\end{equation}

Then
\[
\mathbf{x}_t
  = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0
    + \sqrt{1-\bar{\alpha}_t}\,\bm{\epsilon}_t,
\]
and the loss term $L_t$ becomes
\begin{align}
L_t
  &= \mathbb{E}_{\mathbf{x}_0,\,\bm{\epsilon}}\Bigl[
       \frac{(1-\alpha_t)^2}
            {2\alpha_t (1-\bar{\alpha}_t)\|\bm{\Sigma}_\theta\|_2^2}
       \,\bigl\|\bm{\epsilon}_t
                - \bm{\epsilon}_\theta(\mathbf{x}_t, t)
         \bigr\|^2
     \Bigr] \\
  &= \mathbb{E}_{\mathbf{x}_0,\,\bm{\epsilon}}\Bigl[
       \lambda_t\,
       \bigl\|\bm{\epsilon}_t
              - \bm{\epsilon}_\theta(
                  \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0
                  + \sqrt{1-\bar{\alpha}_t}\,\bm{\epsilon}_t,\; t
                )
       \bigr\|^2
     \Bigr],
\end{align}
for some scalar weighting $\lambda_t$.

\subsection{Simplified Objective}

Empirically, Ho et al.~\cite{ho2020denoising} found that training works better with a simplified objective that ignores the weighting term:
\begin{equation}
L_t^{\text{simple}}
  = \mathbb{E}_{t,\,\mathbf{x}_0,\,\bm{\epsilon}}\Bigl[
      \bigl\|\bm{\epsilon}_t
             - \bm{\epsilon}_\theta(\mathbf{x}_t, t)
      \bigr\|^2
    \Bigr],
\end{equation}
so that the final training loss is simply
\begin{equation}
L_{\text{simple}} = L_t^{\text{simple}} + C,
\end{equation}
where $C$ is a constant independent of $\theta$.
