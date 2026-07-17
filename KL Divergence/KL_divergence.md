## Introduction to KL Divergence

### Definition

KL divergence (also known as relative entropy) is used to measure the difference between two probability distributions $p(x)$ and $q(x)$.

As for two discrete probability distributions $P$ and $Q$ defined on the same probability space, the KL divergence is defined as: 
$$
D_{\text{KL}}(P \ \Vert \ Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}.
$$

As for two continuous probability distributions $p$ and $q$:
$$
D_{\text{KL}}(p \ \Vert \ q) = \int_{\mathcal{X}} p(x) \log \frac{p(x)}{q(x)} dx = \mathbb{E}_{x \sim p(x)} \left[ \log \frac{p(x)}{q(x)} \right].
$$
In the following sections, we mainly focus on the continuous version.

### Key Properties

1. **Non-negativity (Gibbs' Inequality)**: $D_{\text{KL}}(p \ \Vert \ q) \geq 0$ iff $p(x) = q(x)$ (almost anywhere). 
2. **Asymmetry**: Generally, $D_{\text{KL}}(p \ \Vert \ q) \neq D_{\text{KL}}(q \ \Vert \ p)$. Therefore, KL divergence is **not** a true distance metric.

## Forward KL vs. Reverse KL

In machine learning, we often aim to minimize the discrepancy between a true distribution $p(x)$ and a model distribution $q_\theta(x)$. The optimization objective varies depending on which distribution is placed in the expectation operator. Define the two optimization objectives:
1. forward KL divergence: $\min_{q} D_{\text{KL}}(p \ \Vert \ q)$
2. reverse KL divergence: $\min_{q} D_{\text{KL}}(q \ \Vert \ p)$

### Ideal Cases
In ideal scenario, we have
$$
\begin{gathered}
\arg\min_{q} D_{\text{KL}}(p \ \Vert \ q) = 0 \ \Rightarrow \ q^*(x) = p(x) \\
\arg\min_{q} D_{\text{KL}}(q \ \Vert \ p) = 0 \ \Rightarrow \ q^*(x) = p(x)
\end{gathered}
$$
Under this ideal mathematical assumption, the optimal solution is the target distribution itself. Therefore, in this case, optimizing both forward and backward KL divergence are equivalent.

### Practical Cases
In practice, the model distribution $q_\theta(x)$ is determined by the parameter $\theta$ (e.g., a single Gaussian distribution or a neural network with a specific architecture), and it constitutes a restricted hypothesis space $\mathcal{Q}_\theta$. The true distribution $p(x)$, however, is often extremely complex and does not belong to the set $\mathcal{Q}_\theta$ at all. This implies that $\min D_{\text{KL}} > 0$.

In this case, the objective functions for the forward and backward KL divergences will cause the parameters $\theta^*$ to converge in completely different directions. We will perform a rigorous derivation using the example of fitting an arbitrary complex distribution with a single Gaussian distribution.

Suppose the target distribution $p(x)$ is an unknown complex multimodal distribution, and the model distribution is a Gaussian distribution $q_\theta(x) = \mathcal{N}(x; \mu, \Sigma)$, with parameters $\theta = \{\mu, \Sigma\}$.

#### Forward KL Divergence
If we take the forward KL divergence as the training objective, we have
$$
\min_\theta D_{\text{KL}}(p \ \Vert \ q_\theta) = \mathbb{E}_{x \sim p(x)} \left[\log \frac{p(x)}{q_\theta(x)}\right] = \mathbb{E}_{x \sim p(x)} [\log p(x) - \log q_\theta(x)].
$$
Since $\mathbb{E}_{x \sim p(x)}[\log p(x)]$ is constant w.r.t $\theta$, this is equivalent to **Maximum Likelihood Estimation (MLE)**:
$$
\theta^* = \arg\max_\theta \mathbb{E}_{x \sim p(x)}[\log q_\theta(x)]
$$
Substituting the logarithm of the probability density of a single Gaussian distribution, $\log q_\theta(x) = -\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu) - \frac{1}{2}\log\vert{}\Sigma\vert{} + C$, we have
$$
\mu^* = \arg\max_\mu \mathbb{E}_{x \sim p(x)}\left[ -\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu) \right]
$$
Taking the derivative w.r.t. $\mu$ and setting it to zero, we have
$$0 = \nabla_\mu \mathbb{E}_{x \sim p(x)} \left[ -\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu) \right] = \mathbb{E}_{x \sim p(x)}[\Sigma^{-1}(x-\mu)]$$
Since $\Sigma^{-1}$ is full rank, we have
$$
\mu^* = \mathbb{E}_{x \sim p(x)}[x]
$$
**Conclusion**: The optimal mean is the **global mean** of the true distribution, covering all modes. If the target distribution consists of two peaks that are very far apart (such as the distributions of cat and dog images), the model’s mean will be firmly locked at the physical center between these two peaks (which is often a meaningless “valley” region with no probability mass at all). This is why it is called **Mode Covering** (also known as Mean-Seeking, Zero-Avoiding).

#### Reverse KL Divergence
If we take the reverse KL divergence as the training objective, we have
$$
\min_\theta D_{\text{KL}}(q_\theta \ \Vert \ p) = \mathbb{E}_{x \sim q_\theta(x)} \left[\log \frac{q_\theta(x)}{p(x)}\right] = -H(q_\theta(x)) + \mathbb{E}_{x \sim q_\theta(x)} \left[ -\log p(x) \right]
$$
The objective function here consists of two parts, which is the essence of maximizing the ELBO in variational inference:
1. The cross-entropy term $\mathbb{E}_{q_\theta}[-\log p(x)]$: This requires that $q_\theta(x)$ be concentrated in regions where $p(x)$ is large. If $q_\theta$ allocates probability mass to the “valley” regions where $p(x) \approx 0$, causing $-\log p(x) \to \infty$, the penalty from this term will instantly explode.
2. Negative entropy term $-H(q_\theta)$: This requires that the variance of $q_\theta(x)$ itself be as small as possible (the more concentrated the distribution, the smaller the negative entropy).

**Conclusion**: The model collapses to a single mode to avoid low-density regions of $p(x)$. Its optimal solution is to select, from among the numerous peaks in $p(x)$, the local peak (mode) with the largest “area” and the easiest to fit, then place $\mu^*$ at the vertex of this peak and shrink the variance $\Sigma$ so that it is completely contained within this single peak. This is why it is called **Mode-Seeking** (also known as Zero-Forcing).