## Stochastic Process

### Poisson Process 泊松过程
随机变量 $X$ 是定义在样本空间 $\Omega$ 上的函数，当 $x$ 是 $X$ 的观测值时，存在 $\Omega$ 中的 $\omega$ s.t. $x = X(\omega)$。

随机向量 $(X_1, X_2, \cdots, X_n)$ 是定义在样本空间 $\Omega$ 上的 $n$ 元函数。当 $(x_1, x_2, \cdots, x_n)$ 是 $(X_1, X_2, \cdots, X_n)$ 的观测值时，存在 $\omega$ s.t. $(x_1, x_2, \cdots, x_n) = (X_1(\omega), X_2(\omega), \cdots, X_n(\omega))$ 。这时称 $(x_1, x_2, \cdots, x_n)$ 是 $(X_1(\omega), X_2(\omega), \cdots, X_n(\omega))$ 的一次观测/一次实现。

#### 计数过程和泊松过程
##### 随机过程与随机变量
**随机过程**：def: 设 $T$ 是 $(-\infty, \infty)$ 的一个子集，如果对于每个 $t \in T$，$X_t$ 是随机变量，则称随机变量的集合 $\{ X_t \mid t\in T \}$ 是随机过程。
**指标集**： $T$ 为该随机过程的指标集。
**一次观测/一次实现**：将 $t$ 视为时间，如果对每个 $t \in T$，都得到了 $X_t$ 的观测值 $x_t$，则称 $\{ x_t \mid t\in T \}$  是 $\{ X_t \mid t\in T \}$ 一次观测/一次实现。
**轨迹/轨道**：当 $T = [0, \infty)$ 或 $T = [-\infty, \infty)$ 时，$\{ X_t \mid t\in T \}$ 的一次实现就是一条曲线，所以又被称为一条轨迹/轨道。
**有限维分布**：对于 $\forall \ m \in \mathbb{Z}^+$ 和 $T$ 中互不相同的 $t_1, t_2, \cdots, t_m$，称
$$
(X_{t_1}, X_{t_2}, \cdots, X_{t_m})
$$
的联合分布为随机过程 $\{ X_t \mid t\in T \}$ 的一个有限维分布。
**随机分布的概率分布**：称全体有限维分布为该随机分布的概率分布。
**随机序列**：如果对于每个 $n\ge0$，$X_n$ 是随机变量，则称 $\{ X_n \mid n=0,1,\cdots \}$ 是随机序列。随机序列是随机过程的一个特例。
**计数过程**：用 $N(t)$ 表示时间段 $[0,t]$ 内某类事件发生的个数，则 $N(t)$ 是随机变量。由于 $N(t)$ 记录了时间段 $[0,t]$ 内发生的事件数，所以称  $\{ N(t); t \ge 0 \}$ 是计数过程。以后简记 $\{ N(t)\}$。计数过程满足以下条件：
1. 对 $t \ge 0$，$N(t)$ 是取非负整数值的随机变量；
2. 对 $t > s \ge 0$，$N(t) \ge N(s)$；
3. 对 $t > s \ge 0$，$N(t) - N(s)$ 是时间段 $(s,t]$ 中的事件发生数；
4. $\{ N(t) \}$ 的轨迹是单调不减右连续的阶梯函数；

**独立增量性**：对于计数过程 $\{ N(t)\}$，用 $N(s,t]$ 表示区间 $(s,t]$ 内发生的事件数，则有
$$
N(s,t] = N(t) - N(s), \quad s<t
$$
如果在互不相交的时间段内发生事件的个数是相互独立的，则称相应的计数过程 $\{ N(t)\}$ 具有独立增量性。用数学表示：对 $\forall \ n \in \mathbb{Z}^+$ 和 $0 \le t_1 < t_2 < \cdots < t_n$，随机变量
$$
N(0), \ N(0,t_1], \ N(t_1,t_2], \cdots, N(t_{n-1},t_n] 
$$
相互独立。
**独立增量过程**：具有独立增量性的计数过程称为独立增量过程。
**平稳增量性**：如果在长度相等的时间段内，事件发生个数的概率分布是相同的，则称相应的计数过程具有平稳增量性。用数学表示：对 $\forall \ s>0, t_1 > t_2 \ge 0$，随机变量
$$
N(t_1,t_2] \ \text{和} \ N(t_1 + s, t_2 + s] \ \text{同分布}
$$
**平稳增量过程**：具有平稳增量性的计数过程称为平稳增量过程。

##### 泊松过程
**泊松过程**：def：称满足下面条件的计数过程 $\{ N(t)\}$ 是强度为 $\lambda$ 的泊松过程：
1.  $N(0) = 0$；
2.  $\{ N(t)\}$ 是独立增量过程；
3. 对 $\forall \ t,s \ge 0$，$N(s, t+s]$ 服从参数为 $\lambda t$ 的泊松分布，即
$$
	P(N(s, t+s] = k) = \frac{\lambda t}{k!} e^{-\lambda t}, \quad k=0,1,\cdots
$$
其中常数 $\lambda > 0$ 称为泊松分布 $\{ N(t)\}$ 的**强度**。

性质3说明泊松过程是平稳增量过程（概率分布相同），而且时间段 $(s,s+t]$ 中发生事件的个数服从泊松分布。

**为什么常数 $\lambda > 0$ 称为泊松分布 $\{ N(t)\}$ 的强度？** 容易计算：
$$
E(N(t)) = E(N(0,t]) = \lambda t, \quad Var(N(t)) = \lambda t
$$
于是 
$$
\lambda = \frac{E(N(t))}{t}
$$
是单位时间内事件发生的平均数，因此称为强度。$\lambda$ 越大，单位时间内平均发生的事件越多。


**e.g.** 上海证券交易所开盘后，股票买卖的依次成交构成一个泊松过程。如果每10分钟平均有12万次成交，计算该泊松过程的强度 $\lambda$ 和1秒内成交100次的概率。
**Sol:** 用 $\{ N(t)\}$ 表示所述的泊松过程，10分钟内的平均成交次数：
$$
E(N(t, t+10]) = 10\lambda = 120000
$$
于是 $\lambda = 12000$ 次/分钟。
用 $\{ N_1(t)\}$ 表示以秒为单位的泊松过程，强度为 $\lambda_1 = \lambda / 60 = 200$。于是1秒内成交100次的概率为：
$$
P(N_1(1) = 100) = \frac{\lambda_1^{100}}{100!} e^{-\lambda_1} = \frac{200^{100}}{100!} e^{-200}
$$
若要计算5秒内成交次数大于 $k=1050$ 次的概率，则要计算：
$$
P(N_1(5) > k) = 1 - P(N_1(5) \le k) = 1 - \sum_{j=0}^k P(N_1(5) = j) = 1 - \sum_{j=0}^k \frac{(5 \times 200)^j}{j!} e^{-5 \times 200}
$$

泊松过程 *Def 2*: 设 $\lambda > 0$ 是常数。如果计数过程 $\{ N(t)\}$ 满足下面条件，则称他是强度为 $\lambda$ 的泊松过程：
1. $N(0) = 0$；
2. $\{ N(t)\}$ 是独立增量过程，有平稳增量性；
3. 普通性：对任何 $t \ge 0$，当正数 $h \rightarrow 0$ 时，有
$$
\left\{
\begin{array}{ll}
P(N(h) = 1) = \lambda h + o(h) \\
P(N(h) \ge 2) = o(h) \\
\end{array}
\right.
$$
其中隐含了 $P(N(h) = 0) = 1 - \lambda h + o(h)$。



**Theorem 1**：泊松过程 Def 1 和 Def 2 等价。

*proof:* “$\Leftarrow$” (Def 2 推导 Def 1)：设 $\{ N(t)\}$ 满足 Def 2，只需证明在Def 1中的条件3成立。 
对确定的正数 $t$，将区间 $(0, t]$ 进行 $n$ 等分，每段区间长度 $t/n$，等分点设为：
$$
t_j = \frac{jt}{n}, \quad j=0,1,\cdots,n.
$$
用 $Y_j = N(t_{j-1}, t_j]$ 表示区间 $(t_{j-1}, t_j]$ 中的事件数，则 $Y_1, Y_2, \cdots, Y_n$ 独立同分布，同时
$$
\begin{gathered}
P(Y_j \ge 2) = o(t_j - t_{j-1}) = o(t/n) \\
p_n = P(Y_j = 1) = \lambda t / n + o(t/n) \\
q_n = P(Y_j = 0) = 1 - \lambda t / n + o(t/n)
\end{gathered}
$$
对非负整数 $k$，引入事件：
$$
\begin{gathered}
A_n = \{ \sum_{j=1}^n Y_j = k, \text{其中有} k \text{个} Y_j = 1, \text{其余的} Y_j = 0; 1 \le j \le n \} \\
B_n = \{ \sum_{j=1}^n Y_j = k, \text{至少有一个} Y_j \ge 2\}
\end{gathered}
$$
则有 $A_n$ 和 $B_n$ 独立（$A_n \cap B_n = \emptyset$），以及 $B_n \subset \bigcup_{j=1}^n \{ Y_j \ge 2\}$。当 $n \rightarrow \infty$，
$$
\begin{gathered}
P(A_n) = C_n^k p_n^k q_n^{n-k} \\
P(B_n) \le P \left(\bigcup_{j=1}^n \{ Y_j \ge 2\} \right) \le n P(Y_j \ge 2) = no(t/n) \rightarrow 0 \\
np_n = n(\lambda t/n + o(t/n)) \rightarrow \lambda t, \quad q_n \rightarrow 1, \\
q_n^n = (1 - \lambda t/n + o(t/n))^n = \left( 1 - \frac{\lambda t}{n} \right) \left( 1 + \frac{o(t/n)}{1 - \lambda t/n} \right) \rightarrow e^{-\lambda t}
\end{gathered}
$$
所以用 $\{ N(0,t] = k\} = \left\{ \sum_{j=1}^n Y_j = k \right\} = A_n \cup B_n$ 得到
$$
\begin{aligned}
P(N(s, s+t] = k) &= P(N(0, t] = k) \\
&= P(A_n \cup B_n) = \lim_{n \rightarrow \infty} [P(A_n) + P(B_n)] \\
&= \lim_{n \rightarrow \infty} P(A_n) = \lim_{n \rightarrow \infty} C_n^k p_n^k q_n^{n-k}\\
&= \lim_{n \rightarrow \infty} \frac{1}{k!}[n(n-1)\cdots(n-k+1)p_n^k] q_n^{n-k}\\
&= \frac{(\lambda t)^k}{k!} e^{-\lambda t}.
\end{aligned}
$$
“$\Rightarrow$” (Definition 1 推导 Definition 2) 同样只需证明在Def 2中的条件3成立。 
用Taylor公式得到：
$$
\begin{aligned}
P(N(h) = 1) &= \lambda h e^{-\lambda h} = \lambda h(1 - \lambda h + o(h)) \\
&= \lambda h + o(h) \\
P(N(h) \ge 2) &= 1 - P(N(h) = 0) - P(N(h) = 1) = 1 - e^{-\lambda h} - \lambda h e^{-\lambda h} \\
&= 1 - [1 - \lambda h + o(h)] - [\lambda h + o(h)] \\
&= o(h)
\end{aligned}
$$
这就完成了证明。


**习题1**： 设 $\{ N(t)\}$ 是强度为 $\lambda$ 的泊松过程，$0 \le s < t$，验证在条件 $N(t) = n$ 下，$N(s)$ 服从二项分布 $\mathcal{B}(n,s/t)$。

*proof:* 考虑概率 $P(N(s) = k \mid N(t) = n)$：
$$
\begin{aligned}
P(N(s) = k \mid N(t) = n) &= \frac{P(N(s) = k) P(N(t) = n \mid N(s) = k)}{P(N(t) = n)} \\
&= \frac{P(N(s) = k) P(N(t-s) = n-k)}{P(N(t) = n)} \\
&= \frac{\frac{(\lambda s)^k}{k!} e^{-\lambda s} \frac{(\lambda (t - s))^{n-k}}{(n - k)!} e^{-\lambda (t - s)}}{\frac{(\lambda t)^k}{k!} e^{-\lambda t}} \\
&= \frac{n!}{k!(n-k)!} \frac{s^k (t-s)^{n-k}}{t^k} \\
&= C^k_n \left(\frac{s}{t}\right)^k \left(1-\frac{s}{t}\right)^{n-k} 
\end{aligned}
$$
因此，在条件 $N(t) = n$ 下，$N(s)$ 服从二项分布 $\mathcal{B}(n,s/t)$。

**习题2**：对于泊松过程 $\{ N(t)\}$，计算 $E[N(t)N(t+s)]$ 和 $E[N(t+s) \mid N(t)]$。

*proof:* 对于 $E[N(t)N(t+s)]$
$$
\begin{aligned}
E[N(t)N(t+s)] &= E[N(t)\left( N(t+s) - N(t) + N(t) \right)] \\
&= E[N(t)\left( N(t,t+s] + N(t) \right)] = E(N(t)N(t)) + E(N(t)N(t,t+s]) \\ 
&= Var(N(t)) + E^2(N(t)) + E(N(t))E(N(t,t+s]) \\
&= \lambda t + (\lambda t)^2 + \lambda t \lambda s \\
&= \lambda^2 t (t+s) + \lambda t
\end{aligned}
$$
对于 $E[N(t+s) \mid N(t)]$
$$
\begin{aligned}
E[N(t+s) \mid N(t)] &= N(t) + E[N(t,t+s] \mid N(t)] \\
&= N(t) + \lambda s 
\end{aligned}
$$
这里可以理解为 当条件为 $N(t) = k$ 时，$N(t,t+s]$ 的取值不受条件影响，所以 $E[N(t,t+s] \mid N(t)] = E[N(0,s]) = \lambda s$。




