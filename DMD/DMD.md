## DMD
### What is Distillation?

扩散模型（Diffusion Models）可以生成高质量图像，却往往需要几十次网络计算才能完成一张图。一个自然的问题是：能否把这些计算压缩到很少的次数甚至是一次？

这就是蒸馏（Distillation）要做的事情：将一个预训练好的多步扩散模型压缩为少步或一步生成模型，通过知识蒸馏保持生成质量同时显著加速推理。比如说，现在已经有了一个图像生成的扩散模型（教师 Teacher 网络），他已经能够通过100步的反向过程生成出一张质量不错的图像，但是100步的生成过程的时间开销是非常大的，可能是几分钟，我们希望用一个新的网络（学生 Student 网络）学习这个教师生成图像的知识，使其能用很少的步数（例如1-4步）生成出一张和教师100步生成的质量差不多的图像。

传统蒸馏的一种非常直接的做法是：给教师和学生相同的初始随机噪声，让学生一步生成的图像接近教师多步采样的结果。
$$z\xrightarrow{\text{教师多步采样}}y, \ z\xrightarrow{G_\theta}x, \ \min_\theta\ell(x,y).$$
但实际上教师的噪声到图像映射非常复杂，如果纯粹逐样本之间的拟合，单步学生难以准确复现。这篇文章 DMD（Distribution Matching Distillation，分布匹配蒸馏）给出的答案是：训练一个单步的学生生成模型，用预训练的教师生成模型指导它的输出分布，再用教师生成的配对数据稳定训练。相比上面的传统蒸馏方法， DMD 放松了逐样本对应的要求：学生可以生成与教师某次输出不同的图像，但是只要整体上足够真实，并且分布接近目标，我们认为也是很好的。

### Preliminary

回归一下 DDPMs 中的一些定义：我们定义了这样的前向加噪过程（这里与之前blog里的记号略有不同，这里的 $\alpha_t$ 指的是之前blog里的 $\sqrt{\bar{\alpha}_t}$$，$$\sigma_t$ 指的是之前blog里的 $\sqrt{1 - \bar{\alpha}_t}$）：
$$x_t = \alpha_t x_0 + \sigma_t \varepsilon, \ \varepsilon \sim \mathcal N(0,I)$$
我们通过逐步在数据样本中加入随机噪声把真实数据 $x_0$ 变成接近标准正态分布的 $x_T$，再训练网络去反转这个过程，进而学会从噪声中生成数据：
$$x_0\sim p_{\mathrm{real}} \xrightarrow{\text{Forward Process}} x_T\sim\mathcal N(0,I), \ x_T\xrightarrow{\text{Reverse Process}}\hat x_0.$$
这里的 $p_{\mathrm{real}}$ 就是真实数据分布。我们认为一个预训练好的 Teacher 生成模型生成出的数据分布已经能够很好的拟合真实数据分布 $p_{\mathrm{real}}$ 了，于是在蒸馏中，我们用这个 Teacher 生成模型提供这个分布的信息，我们把这个 Teacher 网络称为 $\mu^{\mathrm{base}}$ 。DDPMs 通常用噪声网络参数化 $\varepsilon^{\mathrm{base}}(x_t,t)$ 预测噪声，而 DMD 为了方便推导，使用 data prediction：$\mu^{\mathrm{base}}(x_t,t)\approx x_0$。也就是说输入一张带噪图像 $x_t$ 和时间步 $t$，直接输出对干净样本的估计。事实上，$\epsilon_\theta(x_t,t)$ 和 $\mu^{\mathrm{base}}$ 可以根据 Tweedie's Formula 和前向加噪公式相互转化，所以基于不同的网络参数化进行推导本质没有太多区别：
$$\mu^{\mathrm{base}}(x_t,t)=\frac{x_t-\sigma_t \varepsilon^{\mathrm{base}}(x_t,t)}{\alpha_t}, \ \varepsilon^{\mathrm{base}}(x_t,t)=\frac{x_t-\alpha_t\mu^{\mathrm{base}}(x_t,t)}{\sigma_t}.$$
我们先定义要蒸馏的模型与单步生成过程，然后再考虑如何构造 Training Loss。Teacher 生成模型已经能够从随机噪声 $z \sim \mathcal N(0,I)$ 中生成图像，但需要反复调用去噪网络。我们希望训练一个新的 Student 生成模型 $G_\theta$（神经网络参数为 $\theta$），直接完成这个映射，也就是：
$$z\sim\mathcal N(0,I) \xrightarrow{G_\theta} x=G_\theta(z).$$
如果是 text-to-image 文生图或其他一些条件生成模型，他们会带有类别条件或文本条件输入，这样的话 Student 模型可以写成 $G_\theta(z,c)$，为了简化公式，下面先省略条件 $c$。原本的 Teacher 生成模型需要根据不同的时间步 $t$ 处理不同噪声等级的输入 $x_t$ ，而因为 Student 作为单步生成模型，不再通过变化的时间步执行多轮采样，因此 $G_\theta$ 只需要接收初始噪声作为网络输入。

那么接下来我们应该如何训练 $G_\theta$？

### Distribution Matching Loss

#### 从 KL Divergence 开始

记 Student 生成样本的分布为 $p_{\mathrm{fake}}$。它由 $G_\theta$ 决定，因此也可以记为 $p_\theta$。DMD 的思想是希望 Student 生成样本的分布 $p_{\mathrm{fake}}$ 和预训练 Teacher 生成模型的输出分布 $p_{\mathrm{real}}$ ，于是可以用 KL Divergence 定义目标：
$$
\begin{aligned}
D_{\mathrm{KL}}(p_{\mathrm{fake}}(x) \| p_{\mathrm{real}}(x)) &=\mathbb E_{x\sim p_{\mathrm{fake}}} \left[\log\frac{p_{\mathrm{fake}}(x)}{p_{\mathrm{real}}(x)}\right] \\
&=\mathbb E_{z\sim\mathcal N(0,I)} \left[\log p_{\mathrm{fake}}(G_\theta(z)) - \log p_{\mathrm{real}}(G_\theta(z))\right].
\end{aligned}
$$

第二步就是重参数化：从 $p_{\mathrm{fake}}$ 采样，等价于先采样高斯噪声 $z$，再通过 Student 生成模型得到 $x=G_\theta(z)$。但这里有一个问题：我们既不知道 $p_{\mathrm{real}}(x)$ 的具体表达式，也不知道 $p_{\mathrm{fake}}(x)$ 的具体表达式。Student 生成模型虽然可以生成图像，但我们不能计算这些图像的实际概率密度。

但是！我们在神经网络训练的时候，真正需要用来梯度下降更新的其实是 Loss 对参数 $\theta$ 的梯度，那么如果我们考虑 $\nabla_\theta D_{\mathrm{KL}}(p_{\mathrm{fake}} \| p_{\mathrm{real}})$ 呢？
$$
\nabla_\theta D_{\mathrm{KL}}(p_{\mathrm{fake}}(x) \| p_{\mathrm{real}}(x)) =\mathbb E_z\left[ J_\theta(z)^\top\nabla_x\log p_{\mathrm{fake}}(x) - J_\theta(z)^\top\nabla_x\log p_{\mathrm{real}}(x)
\right] +\mathbb E_{x\sim p_{\mathrm{fake}}} \left[\partial_\theta\log p_{\mathrm{fake}}(x)\right].
$$
其中 $J_\theta(z)=\frac{\partial G_\theta(z)}{\partial\theta}$。最后一项在允许交换积分与求导等正则条件下其实是0：

$$
\mathbb E_{x\sim p_{\mathrm{fake}}}
\left[\partial_\theta\log p_{\mathrm{fake}}(x)\right] = \int p_{\mathrm{fake}}(x)\partial_\theta\log p_{\mathrm{fake}}(x)\,dx = \int\partial_\theta p_{\mathrm{fake}}(x)\,dx =\partial_\theta\int p_{\mathrm{fake}}(x)\,dx = 0.
$$
因此，我们有
$$
\nabla_\theta D_{\mathrm{KL}}(p_{\mathrm{fake}}(x) \| p_{\mathrm{real}}(x)) = \mathbb E_z\left[ J_\theta(z)^\top(\nabla_x\log p_{\mathrm{fake}}(x) - \nabla_x\log p_{\mathrm{real}}(x)) \right].
$$

现在的 Student 生成模型已经能生成一张图像 $x = G_\theta(z)$ 了，但是可能不够与 Teacher 生成的图像分布接近，接下来需要 Teacher 生成模型判断它应该怎样改进，而 Teacher 能提供的有效指导，是不同噪声等级下的去噪预测（输入为带噪的 $x_t$ 和相应时间步 $t$，输出是干净图像的预测 $\mu^{\mathrm{base}}$）。因此，DMD 先把 Student 生成的图像 $x$ 用一样的前向过程加噪，再输入 score 函数进行预测、更新、改进：
$$
q(x_t|x)=\mathcal N(x_t;\alpha_t x,\sigma_t^2I) \ \Leftrightarrow \ x_t=\alpha_t x+\sigma_t\varepsilon, \ \varepsilon\sim\mathcal N(0,I)
$$
相应的 Loss 也要从 $t = 0$ 时刻（干净分布）KL 梯度改成任意 $0 \le t \le 1$ 时刻的：
$$
\begin{aligned}
\nabla_\theta D_{\mathrm{KL}}(p_{\mathrm{fake}}(x_t) \| p_{\mathrm{real}}(x_t)) &= \mathbb E_z \left[ \frac{\partial x_t}{\partial x} J_\theta(z)^\top(\nabla_{x_t}\log p_{\mathrm{fake}}(x_t) - \nabla_{x_t}\log p_{\mathrm{real}}(x_t)) \right] \\
&= \mathbb E_z \left[ \alpha_t J_\theta(z)^\top(\nabla_{x_t}\log p_{\mathrm{fake}}(x_t) - \nabla_{x_t}\log p_{\mathrm{real}}(x_t)) \right]
\end{aligned}
$$

同时在 diffusion model 中，$\nabla_{x_t} \log p(x_t)$ 被称为分数 (score)，其网络估计值（简记为 $s_\theta(x_t, t)$）与一个（相对应的）训练好的 diffusion 噪声网络 $\varepsilon_\theta$ / data prediction 网络有关系：
$$
\nabla_x\log p(x) \approx s_\theta(x_t, t) \approx - \frac{\varepsilon_\theta (x_t, t)}{\sigma_t} = -\frac{x_t-\alpha_t \mu_{\theta}(x_t,t)}{\sigma_t^2}.
$$
于是，原来对两个概率密度的计算，转换成了计算两个 score：$s_{\mathrm{fake}}(x_t, t) \approx \nabla_{x_t}\log p_{\mathrm{fake}}(x_t)$ 和 $s_{\mathrm{real}}(x_t, t) \approx \nabla_{x_t}\log p_{\mathrm{real}}(x_t)$。$J_\theta$ 可以通过 Student 生成模型的自动微分（auto_grad）获得。
接下来的问题是：应该如何用现有的东西计算/估计这两个 score？

#### Real Score：由预训练 Teacher 生成模型直接估计

我们已经有一个在真实数据上训练好的 Teacher 生成模型 $\mu^{\mathrm{base}}$，如果能把它转换为 score，那么就可以直接利用它了。根据我们刚刚给出的 score 与 data prediction 网络的关系，将 Teacher 生成模型的输出代入，就可以得到：
$$
s_{\mathrm{real}}(x_t,t) \approx -\frac{x_t-\alpha_t \mu^{\mathrm{base}}(x_t,t)}{\sigma_t^2}.
$$
我们假定 Teacher 模型学出来生成的真实分布是固定的，因此 Teacher 模型 $s_{\mathrm{real}}(x_t,t)$ 在训练中保持冻结。

#### Fake Score：为 Student 生成模型训练一个去噪器

虽然我们知道 $x=G_\theta(z)$，但光有生成网络 $G_\theta$ 不足以估计 Student 生成模型生成分布的 score $\nabla_{x}\log p_{\mathrm{fake},t}(x)$ ，因为 score 描述概率密度怎样随样本位置变化。

所以 DMD 引入了另一个去噪网络 $\mu^{\mathrm{fake}}_\phi$（fake diffusion model，网络参数为 $\phi$），让它学习当前 Student 生成模型生成的分布。它使用 Student 的输出作为训练数据，然后用 $\ell_2$ norm 让 $\mu^{\mathrm{fake}}_\phi$ 去拟合 Student 的输出：
$$
\begin{gathered}
x=G_\theta(z),\ x_t=\alpha_t x+\sigma_t\varepsilon, \\
\mathcal L_\phi = \mathbb E\left[ \omega(t)\left\| \mu^{\mathrm{fake}}_\phi(x_t,t)-\operatorname{sg}(x) \right\|_2^2 \right]
\end{gathered}
$$
其中 $\omega(t)$ 是训练权重，$\operatorname{sg}$ 表示 stop-gradient。这里的监督目标不是 Teacher 生成的图像，而是 Student 生成的图像，并且 Student 生成的图像外面用了 $\operatorname{sg}$，所以这个训练目标只会更新 fake diffusion model 的参数 $\phi$ ，而不会更新 Student 生成模型的参数 $\theta$。

当 fake diffusion model 训练拟合比较充分时，$\mu^{\mathrm{fake}}_\phi$ 能近似当前分布的条件均值，于是再用前面 score 与 data prediction 网络的关系可得：
$$
s_{\mathrm{fake}}(x_t,t) \approx -\frac{x_t-\alpha_t \mu^{\mathrm{fake}}_\phi(x_t,t)}{\sigma_t^2}
$$
与 Teacher 生成模型不同，由于 $p_{\mathrm{fake}}$ 会随着 Student 生成模型的更新而不断变化，所以 $\mu^{\mathrm{fake}}_\phi$ 也需要持续训练。

整合前面的计算，我们可得最终使用的分布匹配梯度为
$$
\begin{aligned}
\nabla_\theta \mathbb{E}_t [D_{\mathrm{KL}}(p_{\mathrm{fake}}(x_t) \| p_{\mathrm{real}}(x_t))] &= \mathbb{E}_{z, t, \varepsilon}\left[ w_t \alpha_t J_\theta(z)^\top(\nabla_{x_t}\log p_{\mathrm{fake}}(x_t) - \nabla_{x_t}\log p_{\mathrm{real}}(x_t)) \right] \\
& \approx \mathbb{E}_{z,t,\varepsilon}\left[ w_t \alpha_t J_\theta(z)^\top \left(s_{\mathrm{fake}}(x_t,t)-s_{\mathrm{real}}(x_t,t)\right) \right] \\
\end{aligned}
$$
其中 $w_t$ 是额外引入的用来控制不同时间步 Loss 的权重的， $t$ 默认在 $[0.02T,0.98T]$ 内均匀采样，$T=1000$ 与 DDPMs 中的时间 $T$ 一致。如果进一步将两个 score 展开：
$$
s_{\mathrm{fake}}(x_t,t) - s_{\mathrm{real}}(x_t,t)  = \frac{\alpha_t\mu^{\mathrm{fake}}_\phi (x_t,t) - x_t}{\sigma_t^2} - \frac{\alpha_t \mu^{\mathrm{base}}(x_t,t) - x_t}{\sigma_t^2} = \frac{\alpha_t}{\sigma_t^2}
\left(\mu^{\mathrm{fake}}_\phi(x_t,t)-\mu^{\mathrm{base}}(x_t,t)\right)
$$
这样，共同的 $x_t$ 项被消去，最终只需要计算两个去噪预测的差，并进行尺度归一化。至此，DMD 的最核心的部分已经构建完成。

#### Regression Loss and Final Objective

现在已经可以用刚刚构建的训练目标来蒸馏 Student 单步生成模型了，但仅仅这样做，DMD 的作者发现在实际实验中仍然会出现训练不稳定和 mode collapse / mode dropping。

尤其在噪声较小时，生成样本附近的真实密度可能非常低，Teacher 生成模型给出的 score 估计不够可靠。因此，论文又加回了一个逐样本的回归目标。先从标准正态分布采样 $z_{\mathrm{ref}}$，用 Teacher 生成模型的完整多步确定性采样得到对应图像：
$$
z_{\mathrm{ref}}\sim\mathcal N(0,I), \ y_{\mathrm{ref}}=\operatorname{TeacherSampler}(z_{\mathrm{ref}})
$$
把这些噪声和最终图像保存成配对数据集：
$$
\mathcal D=\{(z_{\mathrm{ref}}^{(i)},y_{\mathrm{ref}}^{(i)})\}_{i=1}^N.
$$
构建配对数据集这一步是离线（offline）的。训练时读取 $\mathcal D$ 中的配对，将配对中的 $z_{\mathrm{ref}}$ 输入 Student 生成模型，得到 $x_{\mathrm{ref}}=G_\theta(z_{\mathrm{ref}})$，然后计算回归目标：
$$
\mathcal L_{\mathrm{reg}} = \mathbb E_{(z_{\mathrm{ref}},y_{\mathrm{ref}})\sim\mathcal D} \left[\ell(G_\theta(z_{\mathrm{ref}}),y_{\mathrm{ref}})\right]
$$
论文默认用 LPIPS 作为 $\ell$，让 Student 与 Teacher 生成模型的输出在感知结构上接近。这样，两种目标分别提供不同约束：分布匹配允许 Student 生成模型改善整体输出分布的真实性，而回归目标则提供稳定的逐样本参考，帮助保持 Teacher 生成模型映射的大尺度结构。

于是，DMD 蒸馏的最终目标写成：
$$
\mathcal L_\theta = D_{\mathrm{KL}}+\lambda_{\mathrm{reg}}\mathcal L_{\mathrm{reg}}
$$
其中 $\lambda_{\mathrm{reg}}=0.25$ 作为默认设置。实际网络更新中，第一项使用前面得到的近似梯度 $\nabla_\theta \mathbb{E}_t [D_{\mathrm{KL}}(p_{\mathrm{fake}}(x_t) \| p_{\mathrm{real}}(x_t))]$，第二项直接自动微分即可。

#### Training Algorithm

现在把前面的推导对应到一轮训练。回顾一下我们一共有三个核心网络：冻结的 Teacher 生成模型 $\mu^{\mathrm{base}}$、用于单步生成的 Student 生成模型 $G_\theta$，以及学习 Student 生成分布的 fake 去噪器 $\mu^{\mathrm{fake}}_\phi$。训练流程可以总结为：

1. 采样新噪声 $z$，通过 Student 单步生成 $x=G_\theta(z)$；同时从 $\mathcal D$ 读取 $(z_{\mathrm{ref}},y_{\mathrm{ref}})$，得到 $x_{\mathrm{ref}}=G_\theta(z_{\mathrm{ref}})$。
2. 随机采样时间 $t$ 和高斯噪声 $\varepsilon$，给 $x$ 加噪得到 $x_t = \alpha_t x_0 + \sigma_t \varepsilon$。
3. 将同一个 $x_t$ 输入 Teacher 生成模型 $\mu^{\mathrm{base}}$ 和 fake 去噪器 $\mu^{\mathrm{fake}}_\phi$，计算分布匹配梯度 $\nabla_\theta D_{\mathrm{KL}}(p_{\mathrm{fake}}(x_t) \| p_{\mathrm{real}}(x_t))$。
4. 加上 $\operatorname{LPIPS}(x_{\mathrm{ref}},y_{\mathrm{ref}})$ 的梯度，更新 Student 的参数 $\theta$。
5. 对本轮 Student 生成的 $x$ 停止梯度，再采样一个时间 $t^\prime$ 并加噪，用 $\mathcal L_\phi$ 更新 fake 去噪器 $\mu^{\mathrm{fake}}_\phi$ 的参数 $\phi$。
6. 下一轮由更新后的 Student 生成样本，重复以上过程。

#### Sampling / Inference Algorithm

训练结束后，采样就很简单了，采样一个随机噪声然后输入 Student 网络得到结果即可：
$$
z\sim\mathcal N(0,I),\ x=G_\theta(z).
$$
Teacher 生成模型和 fake 去噪器 $\mu^{\mathrm{fake}}_\phi$ 都不再参与生成。整个方法的难点主要留在训练阶段，部署后的核心生成过程则被压缩成一次 Student 生成模型的调用。

## 参考文献 Reference

1. **One-step Diffusion with Distribution Matching Distillation (DMD).** [[pdf](https://arxiv.org/pdf/2311.18828)]
2. **Improved Distribution Matching Distillation for Fast Image Synthesis (DMD2).** [[pdf]([2405.14867](https://arxiv.org/pdf/2405.14867))]






