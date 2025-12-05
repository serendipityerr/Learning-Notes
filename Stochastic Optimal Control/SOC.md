# Stochastic Optimal Control Theory

Control theory is a mathematical description of how to act optimally to gain future rewards.

## Deterministic Discrete Time Control
We start by discussing the most simple control case, which is the finite horizon discrete time deterministic control problem.

Consider the control of a discrete time dynamical system: 
$$
x_{t+1} = x_{t} + f(t, x_t, u_t), \quad t=0,1,\cdots,T-1
$$
where $x_t$ is an $n$-dimensional vector describing the *state* of the system and $u_t$ is an $m$-dimensional vector that specifies the *control* or *action* at time $t$. If we specify $x_0$ at $t = 0$ and we specify a sequence of controls $u_{0:T−1} = u_0, u_1, \cdots , u_{T−1}$, we can compute future states of the system $x_{1:T}$ recursively.

Define the cost function $\mathcal{C}$ as:
$$
\mathcal{C}(x_0, u_{0:T-1}) = \sum_{t=0}^{T-1} R (x_t, u_t, t) + \phi(x_T) 
$$
where $R (t, x_t, u_t)$ is the cost that is associated with taking action $u_t$ at time $t$ in state $x_t$ (trajectory cost), and $\phi(x_T)$ is the cost associated with ending up in state $x_T$ at time $T$ (terminal cost). The problem of optimal control is to find the sequence $u_{0:T−1}$ that minimizes $\mathcal{C}(x_0, u_{0:T−1})$:
$$
\begin{gathered}
\min_{u_{0:T−1}} \mathcal{C}(x_0, u_{0:T−1}) \\
x_{t+1} = x_{t} + f(x_t, u_t, t), \ t=0,1,\cdots,T-1, \ x_{0} = x_{0}
\end{gathered}
$$
Now, we introduce the optimal *cost to go* function $J$:
$$
J(x_t, t) = \min_{u_{t:T−1}} \left( \sum_{s=t}^{T-1} R (x_s, u_s, s) + \phi(x_T) \right)
$$
which solves the optimal control problem from an intermediate time $t$ until the fixed end time $T$ , starting at an arbitrary location $x_t$. We have
$$
\begin{gathered}
J(x_0, 0) = \min_{u_{0:T−1}} \mathcal{C}(x_0, u_{0:T−1})\\
J(x_T, T) = \phi(x_T) 
\end{gathered}
$$
Consider recursively computing $J(x_t, t)$ as
$$
\begin{aligned}
J(x_t, t) &= \min_{u_{t:T−1}} \left( \sum_{s=t}^{T-1} R (x_s, u_s, s) + \phi(x_T) \right) \\
&= \min_{u_{t}} \left[ R (x_t, u_t, t) + \min_{u_{t+1:T−1}} \left( \sum_{s=t+1}^{T-1} R (x_s, u_s, s) + \phi(x_T) \right) \right] \\
&= \min_{u_{t}} \left[ R (x_t, u_t, t) + J(x_{t+1}, t+1) \right] \\
&= \min_{u_{t}} \left[ R (x_t, u_t, t) + J(x_{t} + f(x_t, u_t, t), t+1) \right] 
\end{aligned}
$$
**Note:** the minimization over the whole path $u_{t:T−1}$ has reduced to a sequence of minimizations over $u_t$. This simplification is due to the **Markovian** nature of the problem: the future depends on the past and vise versa only through the present. The algorithm to compute the optimal control $u^*_{0:T−1}$, the optimal trajectory  $x^*_{1:T}$ and the optimal cost is given by
1. Initialize $J(x_T, T) = \phi(x_T)$
2. Backwards: For $t=T-1, \cdots, 1, 0$, and For all $x_t$, compute
$$
\begin{gathered}
u^*_t(x_t) = \arg\min_u \{ R(x_t, u, t) + J(t+1, x_t + f(x_t, u, t)) \} \\
J(t, x_t) = R(x_t, u^*_t(x_t), t) + J(t+1, x_t + f(x_t, u^*_t(x_t), t)
\end{gathered}
$$
3. Forwards: For $t=T-1, \cdots, 1, 0$ compute
$$
x^*_{t+1} = x^*_{t} + f(x^*_{t}, u^*_{t}, t), \quad x^*_0 = x_0
$$

This is just like **Dynamic Programming (DP)** algorithm. It is linear in the horizon time $T$ and linear in the size of the state and action spaces.

## Deterministic Continuous Time Control

Now, consider the optimal control problem in continuous time:
$$
x_{t+\mathrm{d}t} = x_{t} + f(x_t, u_t, t)\mathrm{d}t, \quad \mathrm{d}x_t = f(x_t, u_t, t)\mathrm{d}t
$$
We have two methods for solving the optimal control problem: 1) Pontryagin Minimum Principle (PMP) (a pair of ordinary differential equations); 2) Hamilton-Jacobi-Bellman (HJB) equation (a partial differential equation).

### Hamilton-Jacobi-Bellman Equation
The initial state is fixed: $x_0 = x_0$ and the final state is free. The problem is to find a control signal $u_t, 0 < t < T$ , which we denote as $u(0 \rightarrow T)$, such that minimizing
$$
C(x_0, u(0 \rightarrow T)) = \int_0^T R(x_t, u_t, t) \mathrm{d} t + \phi(x_T)
$$
Then, the *cost to go* function can be extended into
$$
J(x_t, t) = \min_{u_{t \rightarrow T}} \left( \int_{t}^{T} R (x_z, u_z, z) \mathrm{d} z + \phi(x_T) \right) 
$$
Again consider recursively computing $J(x_t, t)$ as
$$
\begin{aligned}
J(x_t, t) &= \min_{u_{t}} \left( R (x_t, u_t, t) \mathrm{d} t + \min_{u_{t+\mathrm{d}t \rightarrow T}} \left( \int_{t+\mathrm{d} t}^{T} R (x_z, u_z, z) \mathrm{d} z + \phi(x_T)\right) \right) \\
&= \min_{u_{t}} \left( R (x_t, u_t, t) \mathrm{d} t + J(x_{t+\mathrm{d}t}, t+\mathrm{d}t) \right)  \\
&\approx \min_{u_{t}} \left( R (x_t, u_t, t) \mathrm{d} t + J(x_t, t) + \partial_t J(x_{t}, t) \mathrm{d} t + \partial_{x_t}J(x_{t}, t)f(x_t, u_t, t) \mathrm{d} t \right) 
\end{aligned}
$$
which implies
$$
-\partial_t J(x_{t}, t) = \min_{u_{t}} \left( R (x_t, u_t, t) + f(x_t, u_t, t) \partial_{x_t}J(x_{t}, t) \right)
$$
which is a partial differential equation, known as the *Hamilton-Jacobi-Bellman (HJB)* equation that describes the evolution of $J$ as a function of $x$ and $t$ and must be solved with boundary condition $J(x_T, T) = \phi(x)$. $\partial_t$ and $\partial_x$ denote partial derivatives with respect to $t$ and $x$, respectively.

The optimal control $u_t(x_t)$ at the current $x_t$, $t$ is given by
$$
u_t(x_t) = \arg \min_{u} \left( R(x_t, u, t) + \partial_{x_t}J(x_{t}, t) f(x_t, u, t) \right)
$$
**Note:** in order to compute the optimal control at the initial state $x_0$ at $t = 0$, one must compute $J(x_t, t)$ for all values of $x_t$ and $t$.

### Pontryagin Minimum Principle

An alternative approach is a variational method that directly finds the optimal trajectory and optimal control and bypasses the expensive computation of the cost-to-go. We can write the optimal control problem as a constrained optimization problem:
$$
\begin{gathered}
\min_{u_t} \int_0^T R(x_t, u_t, t) \mathrm{d} t + \phi(x_T) \\
\text{s.t.} \ \mathrm{d} x_t = f(x_t, u_t, t) \mathrm{d} t, \ x_0 = x_0
\end{gathered}
$$
We can solve the constraint optimization problem by introducing the Lagrange multiplier function $\lambda_t$ that ensures the ODE constraint for all $t$:
$$
\begin{aligned}
\mathcal{C} &= \int_0^T \left[R(x_t, u_t, t) - \lambda_t \left(f(x_t, u_t, t) - \frac{\mathrm{d} x_t}{\mathrm{d} t} \right) \right] \mathrm{d} t + \phi(x_T) \\
&= \int_0^T \left[-H(x_t, u_t, \lambda_t, t) + \lambda_t \frac{\mathrm{d} x_t}{\mathrm{d} t} \right] \mathrm{d} t + \phi(x_T) \\
\end{aligned}
$$
where we defined $H(x_t, u_t, \lambda_t, t) = -R(x_t, u_t, t) + \lambda_t f(x_t, u_t, t)$. The solution is found by extremizing $\mathcal{C}$. We can find the extremal solution if we compute the change $\delta \mathcal{C}$ in $\mathcal{C}$ that results from independent variation of $x_t$, $u_t$, and $\lambda_t$ at all times $t$ between $0$ and $T$. We denote these variations by $\delta x_t$, $\delta u_t$, and $\delta \lambda_t$, respectively. We have: 
$$
\delta \mathcal{C} = \int_0^T \left[-\partial_{x_t} H \delta x_t - \partial_{u_t} H \delta u_t + \left(-\partial_{\lambda_t} H + \frac{\mathrm{d} x_t}{\mathrm{d} t}\right) \delta \lambda_t + \lambda_t \delta \frac{\mathrm{d} x_t}{\mathrm{d} t} \right] \mathrm{d} t + \partial_{x_T} \phi(x_T) \delta \phi(x_T) 
$$
Take partial integration:
$$
\int_0^T \lambda_t \delta \frac{\mathrm{d} x_t}{\mathrm{d} t} \mathrm{d} t = \int_0^T \lambda_t \frac{\mathrm{d}\delta x_t}{\mathrm{d} t} \mathrm{d} t = -\int_0^T \delta x_t \frac{\mathrm{d} \lambda_t}{\mathrm{d} t} \mathrm{d} t + \lambda_T \delta x_T - \lambda_0 \delta x_0
$$
and $\delta x_0 = 0$ as $x_0$ is given.

The stationary solution $\delta \mathcal{C} = 0$ is obtained when the coefficients of the independent variations $\delta x_t$, $\delta u_t$, $\delta \lambda_t$, and $\delta x_T$ are zero. Thus, 
$$
\begin{aligned}
0 &= \partial_{u_t} H(x_t, u_t, \lambda_t, t) \\
\frac{\mathrm{d} \lambda_t}{\mathrm{d} t} &= - \partial_{x_t} H(x_t, u_t, \lambda_t, t) \\
\frac{\mathrm{d} x_t}{\mathrm{d} t} &= \partial_{\lambda_t} H(x_t, u_t, \lambda_t, t) \\
\lambda_T &= - \partial_{x_T} \phi(x_T)
\end{aligned}
$$
The first ODE implies the optimal controller $u_t^*(x_t, \lambda_t, t)$. The remaining ODEs are:
$$
\begin{aligned}
\frac{\mathrm{d} \lambda_t}{\mathrm{d} t} &= - \partial_{x_t} H(x_t, u_t^*, \lambda_t, t) \\
\frac{\mathrm{d} x_t}{\mathrm{d} t} &= \partial_{\lambda_t} H(x_t, u_t^*, \lambda_t, t)
\end{aligned}
$$
with the boundary conditions $x_0 = x_0$ and $\lambda_T = - \partial_{x_T} \phi(x_T)$. There remains two coupled ODEs that describe the dynamics of $x_t$ and $\lambda_t$ over time with boundary conditions.

> [!exr|1] 
> Consider the following problem:
> $$
> \begin{gathered}
> \min_{u_t} \int_0^T \frac{1}{2} \|\mathbf{u}_t\|_2^2 \mathrm{d} t + \frac{\gamma}{2}\| \boldsymbol{x}_T - x_T \|_2^2 \\
> \text{s.t.} \ \mathrm{d} x_t = \left( f_t \boldsymbol{x}_t + h_t \mathbf{m} + g_t \mathbf{u}_t \right) \mathrm{d} t, \ \boldsymbol{x}_0 = x_0,
> \end{gathered}
> $$
> Denote $d_{t, \gamma} = \gamma^{-1} + e^{2\bar{f}_{T}} \bar{g}^2_{t:T}$, $\bar{f}_{s:t} = \int_{s}^{t} f_z dz$, $\bar{h}_{s:t} = \int_{s}^{t} e^{-\bar{f}_{z}} h_z dz$ and $\bar{g}^2_{s:t} = \int_{s}^{t} e^{-2\bar{f}_{z}}g^2_z dz$, denote $\bar{f}_{t}$, $\bar{h}_{t}$ and $\bar{g}^2_{t}$ for simplification when $s=0$, then the closed-form optimal controller $\mathbf{u}_t^*$ is
> $$
> \mathbf{u}_{t, \gamma}^{*} = g_t e^{\bar{f}_{t:T}} \frac{x_{T} - e^{\bar{f}_{t:T}} \mathbf{x}_t - \mathbf{m} e^{\bar{f}_{T}} \bar{h}_{t:T}}{d_{t, \gamma}}.
> $$

`\begin{proof}`
Please refer to Theorem 4.1 in [UniDB](https://arxiv.org/pdf/2502.05749) for detailed proof.
`\end{proof}`

## Stochastic Optimal Control

In this section, we consider the extension of the continuous control problem to the case that the dynamics is subject to noise and is given by a Stochastic Differential Equation (SDE).

Consider the following controlled SDE:
$$
\mathrm{d}x_t = f(x_t, u_t, t) \mathrm{d} t + g_t \mathrm{d}w_t,
$$
where $f: \mathbb{R}^d \times \mathbb{R}^d \times [0, T] \rightarrow \mathbb{R}^d$ is the vector-valued drift function, $g:[0, T] \rightarrow \mathbb{R}$ signifies the scalar-valued diffusion coefficient and $w_t \in \mathbb{R}^d$ is the standard Wiener process, also known as Brownian motion.

Since the dynamics is stochastic, it is no longer the case that when $x$ at time $t$ and the full control path $u(t \rightarrow T)$ are given, we know the future path $x(t \rightarrow T)$. Therefore, we could only hope to be able to minimize its expectation value over all possible future realizations of the Wiener process:
$$
\mathcal{C}(x_0, u(0 \rightarrow T)) = \mathbb{E}_{x_0} \left[ \int_0^T R(x_t, u_t, t) \mathrm{d} t + \phi(x_T) \right]
$$
where the subscript $x_0$ on the expectation value is to remind us that the expectation is over all stochastic trajectories that start in $x_0$. The solution of the control problem proceeds very similar as in the deterministic case: the *cost to go* function can be extended into
$$
J(x_t, t) = \min_{u_{t \rightarrow T}} \mathbb{E}_{x_t} \left[ \int_{t}^{T} R (x_z, u_z, z) \mathrm{d} z + \phi(x_T) \right] 
$$
Again consider recursively computing $J(x_t, t)$ as
$$
\begin{aligned}
J(x_t, t) &= \min_{u_{t}} \mathbb{E}_{x_t} \left[ R (x_t, u_t, t) \mathrm{d} t + \min_{u_{t+\mathrm{d}t \rightarrow T}} \left( \int_{t+\mathrm{d} t}^{T} R (x_z, u_z, z) \mathrm{d} z  + \phi(x_T)\right) \right] \\
&= \min_{u_{t}} \left( R (x_t, u_t, t) \mathrm{d} t + \mathbb{E}_{x_t} \left[ J(x_{t+\mathrm{d}t}, t+\mathrm{d}t) \right] \right)
\end{aligned}
$$
Again make a Taylor expansion:
$$
\begin{aligned}
\mathbb{E}_{x_t} \left[ J(x_{t+\mathrm{d}t}, t+\mathrm{d}t) \right] &= \int \mathcal{N}(x_{t+\mathrm{d}t} \mid x_t) J(x_{t+\mathrm{d}t}, t+\mathrm{d}t) \mathrm{d} x_{t+\mathrm{d}t} \\
&\approx J(x_{t}, t) + \partial_t J(x_{t}, t) \mathrm{d}t + \partial_{x_t} J(x_{t}, t) \langle \mathrm{d}x_t \rangle + \frac{1}{2} \partial^2_{x_t} J(x_{t}, t) \langle \mathrm{d}x_t^2 \rangle \\
\end{aligned}
$$
where we have $\langle \mathrm{d}x_t \rangle = f(x_t, u_t, t) \mathrm{d} t$ and $\langle \mathrm{d}x_t^2 \rangle = g_t \mathrm{d} t$. Therefore, 
$$
-\partial_t J(x_t, t) = \min_{u_{t}} \left( R (x_t, u_t, t) + f(x_t, u_t, t) \partial_{x_t}J(x_{t}, t) + \frac{g_t^2}{2} \partial^2_{x_t} J(x_{t}, t) \right)
$$
We can easily find that when $g_t \rightarrow 0$, the equation above would degrade to the deterministic one as we discussed in the section above (Deterministic Continuous Time Control).


