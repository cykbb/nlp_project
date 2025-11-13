#!/usr/bin/env python
# coding: utf-8

# In[13]:


import random
import torch
from torch.distributions.multinomial import Multinomial


# In[14]:


num_tosses = 1000
heads = sum([random.random() > 0.5 for _ in range(num_tosses)])
tails = num_tosses - heads
print("heads, tails: ", [heads, tails])


# In[15]:


fair_probs = torch.tensor([0.5, 0.3, 0.2])
Multinomial(100, fair_probs).sample()


# In[16]:


Multinomial(100, fair_probs).sample() / 100


# In[17]:


counts = Multinomial(10000, fair_probs).sample()
counts / 10000


# 
# 
# ## 1. 贝叶斯公式的推导
# 
# * 条件概率的定义：
# 
#   $$
#   P(A,B) = P(A|B)P(B) = P(B|A)P(A)
#   $$
# * 两边相等，得到：
# 
#   $$
#   P(A|B) = \frac{P(B|A)P(A)}{P(B)}.
#   $$
# * 这就是经典的 **贝叶斯公式**。
# 
# ---
# 
# ## 2. 贝叶斯公式的直观意义
# 
# * 它可以 **反转条件概率的方向**。
#   通常我们容易得到 $P(B|A)$（例如：已知疾病 $A$ 时出现症状 $B$ 的概率），但我们真正想要的是 $P(A|B)$（出现症状后有病的概率）。
# * 贝叶斯公式帮我们通过 $P(B|A)$、先验 $P(A)$、以及边缘概率 $P(B)$ 计算出需要的 $P(A|B)$。
# 
# ---
# 
# ## 3. 简化版公式
# 
# 有时候我们不知道 $P(B)$，但可以用比例的方式写：
# 
# $$
# P(A|B) \propto P(B|A)P(A).
# $$
# 
# * 意思是：后验概率和“似然 × 先验”成正比。
# * 最后还要通过归一化确保所有可能的 $A$ 概率和为 1。
# 
# ---
# 
# ## 4. 归一化（正规化）
# 
# 为了把“成比例”变成“等于”，需要归一化：
# 
# $$
# P(A|B) = \frac{P(B|A)P(A)}{\sum_a P(B|A=a)P(A=a)}.
# $$
# 
# * 分母就是所有可能 $A$ 的情况求和，保证结果是一个真正的概率分布。
# 
# ---
# 
# ## 5. 在贝叶斯统计里的解释
# 
# * **Prior 先验** $P(H)$：在看到数据前，对假设 $H$ 的主观信念。
# * **Likelihood 似然** $P(E|H)$：如果假设 $H$ 成立，看到证据 $E$ 的可能性。
# * **Posterior 后验** $P(H|E)$：看到数据后，更新对假设的信念。
# * 公式：
# 
#   $$
#   P(H|E) = \frac{P(E|H)P(H)}{P(E)}.
#   $$
# 
#   可以理解为：**后验 = 先验 × 似然 ÷ 证据**。
# 
# ---
# 
# ## 6. 边缘化 (Marginalization)
# 
# 分母 $P(E)$ 实际上是把所有可能的假设都考虑进来：
# 
# $$
# P(E) = \sum_H P(E|H)P(H).
# $$
# 
# 这个求和过程叫做 **边缘化**，本质就是把联合概率 $P(E,H)$ 对 $H$ 求和，得到边缘分布 $P(E)$。
# 
# ---
# 
# ## 7. 小结
# 
# * 贝叶斯公式 = 条件概率公式的直接推论。
# * 它的价值在于**反转条件概率方向**，把 $P(B|A)$ 变成 $P(A|B)$。
# * 在贝叶斯统计里：
# 
#   * **先验**：主观信念。
#   * **似然**：假设解释数据的能力。
#   * **后验**：结合数据更新后的信念。
# * 分母 $P(B)$ 通过边缘化得到，确保归一化。
# 
# ---
# 

# 
# ---
# 
# ## 1. 什么是期望 (Expectation)
# 
# * 期望就是“平均结果”。
# * 在随机变量 $X$ 上，它表示 **长期重复实验时的平均值**。
# * 定义（离散型）：
# 
#   $$
#   E[X] = \sum_x x \, P(X=x).
#   $$
# * 定义（连续型）：
# 
#   $$
#   E[X] = \int x \, p(x)\, dx.
#   $$
# 
# 例子：投资可能 50% 完全失败 (收益 0)，40% 有 2 倍收益，10% 有 10 倍收益：
# 
# $$
# E[X] = 0.5\cdot 0 + 0.4\cdot 2 + 0.1\cdot 10 = 1.8.
# $$
# 
# 所以期望回报是 1.8 倍。
# 
# ---
# 
# ## 2. 函数的期望
# 
# 不仅可以算随机变量本身的期望，还可以算它的函数 $f(X)$ 的期望：
# 
# $$
# E[f(X)] = \sum_x f(x)P(x)\quad\text{或}\quad \int f(x)p(x)\, dx.
# $$
# 
# 在经济学里，这个 $f$ 常常表示 **效用函数 (utility)**，即“幸福感”。
# 
# * 人们对钱的效用往往是**非线性**的（损失更痛苦，收益边际递减）。
# * 例如“金钱效用是对数函数”的说法，就是这个原因。
# 
# ---
# 
# ## 3. 投资效用的例子
# 
# 如果设定：
# 
# * 完全失败：效用 = -1
# * 收益 1 倍 → 效用 1
# * 收益 2 倍 → 效用 2
# * 收益 10 倍 → 效用 4
# 
# 那么期望效用：
# 
# $$
# E[f(X)] = 0.5\cdot (-1) + 0.4\cdot 2 + 0.1\cdot 4 = 0.7.
# $$
# 
# 代表预期幸福值是 **负的**（比不投资更糟），所以理性的选择可能是不要投资。
# 
# ---
# 
# ## 4. 风险与方差 (Variance)
# 
# 除了期望，还要考虑 **风险性**，即结果波动有多大。
# 
# * 定义：
# 
#   $$
#   \text{Var}[X] = E\left[(X-E[X])^2\right] = E[X^2] - (E[X])^2.
#   $$
# * 标准差 $\sigma = \sqrt{\text{Var}[X]}$ 具有和 $X$ 相同的单位，更直观。
# 
# 例子：投资例子中的方差：
# 
# $$
# \text{Var}[X] = 0.5\cdot 0^2 + 0.4\cdot 2^2 + 0.1\cdot 10^2 - 1.8^2 = 8.36.
# $$
# 
# 说明投资虽然期望收益高，但波动性极大 → 风险高。
# 
# ---
# 
# ## 5. 多元情况：均值向量与协方差矩阵
# 
# * 对向量随机变量 $\mathbf{x}$，期望就是分量逐一的平均：
# 
#   $$
#   \mu = E[\mathbf{x}] \quad (\mu_i = E[x_i]).
#   $$
# * 协方差矩阵：
# 
#   $$
#   \Sigma = E\big[(\mathbf{x}-\mu)(\mathbf{x}-\mu)^T\big].
#   $$
# * 性质：对任意向量 $\mathbf{v}$，
# 
#   $$
#   \mathbf{v}^T \Sigma \mathbf{v} = \text{Var}(\mathbf{v}^T\mathbf{x}).
#   $$
# 
#   这表示我们能通过 $\Sigma$ 计算任何线性组合的方差。
# * 协方差矩阵的对角线是方差，非对角线表示变量之间的相关性（0=无关，正/负值表示正/负相关）。
# 
# ---
# 
# ## 6. 总结
# 
# 这一节的主线：
# 
# 1. **期望**：平均值，衡量“长期收益”。
# 2. **效用期望**：考虑心理价值，效用函数可非线性。
# 3. **方差**：衡量不确定性/风险。
# 4. **多元扩展**：期望向量、协方差矩阵，用来刻画多个变量的均值和相关性。
# 
# 

# # Exercises

# 
# ### 1. 什么时候更多数据能把不确定性降到任意低？
# 
# **例子**：掷一枚可能存在偏差的硬币。
# 
# * 我们不知道硬币正面朝上的真实概率 $p$。
# * 如果我们不断收集掷硬币的数据（例如上万次投掷），我们就可以把对 $p$ 的估计精确到非常小的误差。
# * 这时的不确定性来自 **参数未知**，属于 **epistemic uncertainty**。随着数据量无限增大，我们对参数的估计可以无限逼近真实值。
# 
# ➡️ 解释：这是 **模型参数的不确定性**，可以通过数据收集逐渐消除。
# 
# ---
# 
# ### 2. 什么时候更多数据只能降低到一定程度？
# 
# **例子**：预测下一次公平硬币投掷的结果。
# 
# * 即使我们完全知道概率 $p=0.5$，下一次投掷结果仍然不可预测，只能说有 50% 的概率是正面。
# * 这种不确定性来自事件本身的 **随机性 (aleatoric uncertainty)**，数据再多也无法消除。
# 
# ➡️ 解释：这种不确定性是问题固有的噪声，无法通过收集更多数据进一步减少。
# 
# * 在这个例子里，收集数据最多只能让我们确定“硬币是公平的”，之后不确定性停留在 $p=0.5$ 上，再也无法减少。
# 
# ---
# 
# ## 总结：
# 
# 1. **更多数据可无限降低的不确定性** → 参数未知 (epistemic)。例：硬币偏向概率估计。
# 2. **更多数据无法完全消除的不确定性** → 问题固有随机性 (aleatoric)。例：下一次硬币投掷结果。
# 
# ---
# 
# 

# 设 $X_i\sim \text{Bernoulli}(p)$ 表示第 $i$ 次掷硬币是否为正面（1 为正面，0 为反面），独立同分布。
# 用样本频率
# 
# $$
# \hat p=\frac1n\sum_{i=1}^n X_i
# $$
# 
# 估计正面概率 $p$。
# 
# ## (1) 方差随样本数的缩放
# 
# $$
# \mathbb E[\hat p]=p,\qquad 
# \operatorname{Var}(\hat p)=\operatorname{Var}\!\left(\frac1n\sum X_i\right)
# =\frac{1}{n^2}\sum \operatorname{Var}(X_i)
# =\frac{n\cdot p(1-p)}{n^2}=\frac{p(1-p)}{n}.
# $$
# 
# → 方差 **$O(1/n)$**；样本数翻 10 倍，方差缩至 1/10。最坏情况下 $p(1-p)\le 1/4$。
# 
# ## (2) 用切比雪夫不等式给出偏离期望的上界
# 
# 对任意 $\varepsilon>0$：
# 
# $$
# \Pr\!\left(|\hat p-p|\ge \varepsilon\right)
# \le \frac{\operatorname{Var}(\hat p)}{\varepsilon^2}
# = \frac{p(1-p)}{n\varepsilon^2}
# \le \frac{1}{4n\varepsilon^2}.
# $$
# 
# 这是分布无关（只用到了方差）的保守上界。
# 
# ## (3) 与中心极限定理（CLT）的关系
# 
# CLT 给出更精细的近似：
# 
# $$
# \sqrt{n}\,\frac{\hat p-p}{\sqrt{p(1-p)}} \ \xrightarrow{d}\ \mathcal N(0,1),
# $$
# 
# 等价地，
# 
# $$
# \hat p \approx \mathcal N\!\left(p,\ \frac{p(1-p)}{n}\right)\quad (n\ \text{大}).
# $$
# 
# 因此可得近似置信区间
# 
# $$
# \hat p \pm z_{\alpha/2}\sqrt{\frac{p(1-p)}{n}}
# $$
# 
# （实际常用 $\hat p$ 代替 $p$ 做 plug-in）。
# 和切比雪夫相比，CLT 在样本足够大时给出**更紧**的概率界与区间（例如 95% 时用 $z_{0.025}\approx1.96$，而切比雪夫要 $k\approx 4.47$ 才到 95%）。
# 

# 你问的这句在说：**虽然对每个固定的 (m) 都能用切比雪夫不等式给 (z_m) 做上界，但不能把不同 (m) 的 (z_m) 当作“互相独立”来同时使用这些界**。原因是这些均值之间**强相关**。
# 
# ---
# 
# ## 1) 先看单个 (m) 时为什么能用切比雪夫
# 
# 设 (x_i) 独立同分布，(E[x_i]=0,\ \mathrm{Var}(x_i)=1)。样本均值
# [
# z_m=\frac1m\sum_{i=1}^m x_i,\quad E[z_m]=0,\quad \mathrm{Var}(z_m)=\frac{1}{m}.
# ]
# 切比雪夫给出（对任意 (\varepsilon>0)）：
# [
# \Pr\big(|z_m|\ge \varepsilon\big)\le \frac{\mathrm{Var}(z_m)}{\varepsilon^2}
# =\frac{1}{m,\varepsilon^2}.
# ]
# ——**对“固定的某个 (m)”这是完全合法的**。
# 
# ---
# 
# ## 2) 为什么不能把不同 (m) 的界“当独立地”同时用？
# 
# 因为 (z_m) 和 (z_{m+1}) 不是独立随机变量。事实上
# [
# z_{m+1}=\frac{m z_m + x_{m+1}}{m+1},
# ]
# 它们**共享了前 (m) 个样本**，所以强相关。我们甚至可以算出相关性：
# 
# [
# \mathrm{Cov}(z_m,z_{m+1})
# =\mathrm{Cov}!\Big(z_m,\frac{m}{m+1}z_m+\frac{1}{m+1}x_{m+1}\Big)
# =\frac{m}{m+1}\mathrm{Var}(z_m)
# =\frac{1}{m+1}>0,
# ]
# 因为 (\mathrm{Var}(z_m)=1/m) 且 (z_m) 与 (x_{m+1}) 独立。
# 
# [
# \mathrm{Corr}(z_m,z_{m+1})
# =\frac{\mathrm{Cov}(z_m,z_{m+1})}
# {\sqrt{\mathrm{Var}(z_m)\mathrm{Var}(z_{m+1})}}
# =\frac{1/(m+1)}{\sqrt{(1/m)(1/(m+1))}}
# =\sqrt{\frac{m}{m+1}}\approx 1.
# ]
# 
# > 相关系数几乎是 1 —— 这叫“强相关”。
# > 因此你**不能**像独立事件那样把“好事件”的概率相乘（或把“坏事件”的上界相乘/累加为独立），那样会严重低估（或误估）联合事件的概率。
# 
# 直觉：既然 (z_{m+1}) 就是在 (z_m) 的基础上稍微加了一点点新信息，它们的大小、符号高度一致，当然不独立。
# 
# ---
# 
# ## 3) 那想要“同时对许多 (m)”给出保证怎么办？
# 
# 不能用“独立相乘”，但可以用**并联界（union bound）**：
# 
# [
# \Pr!\Big(\max_{1\le m\le M}|z_m|\ge \varepsilon\Big)
# \le \sum_{m=1}^M \Pr(|z_m|\ge \varepsilon)
# \le \frac{1}{\varepsilon^2}\sum_{m=1}^M \frac{1}{m}
# \le \frac{1+\ln M}{\varepsilon^2}.
# ]
# 
# 想让上式 (\le \delta)，选
# [
# \varepsilon ;\ge; \sqrt{\frac{1+\ln M}{\delta}}.
# ]
# （当然，这个界很保守；若 (x_i) 还满足有界或次高斯，可用 Hoeffding/Bernstein 等更紧的集中不等式；更系统的工具还有鞅不等式、强大数定律等。）
# 
# ---
# 
# ### 小结
# 
# * **单个 (m)**：切比雪夫可用，(\Pr(|z_m|\ge\varepsilon)\le 1/(m\varepsilon^2))。
# * **多个 (m) 同时**：(z_m) 之间强相关（(\mathrm{Corr}\approx!1)），**不能**把它们当独立来处理。
# * **正确做法**：用 union bound（或更强的集中不等式）给出“同时对所有 (m)”的统一概率保证。
# 
# 
# 

# 下面把“并联界（union bound）”讲清楚，并把你看到的不等式一步步推出来。
# 
# ---
# 
# ## 1) 并联界是什么？
# 
# 并联界（又叫 **Boole’s inequality**）是最基础的概率上界之一：
# 
# [
# \boxed{\Pr\Big(\bigcup_{i=1}^M A_i\Big)\ \le\ \sum_{i=1}^M \Pr(A_i)}
# ]
# 
# 对**任意**事件 (A_1,\dots,A_M) 都成立，不需要独立性。
# 直觉：多个事件的“并”发生的概率，最多不超过把每个事件概率**直接加起来**（虽然会双计重叠，但这是上界，所以安全）。
# 
# ### 一个两行证明（指示变量法）
# 
# 令 (\mathbf{1}*{A}) 为事件 (A) 的指示变量（发生=1，不发生=0）。有：
# [
# \mathbf{1}*{\cup_i A_i}\ \le\ \sum_{i=1}^M \mathbf{1}*{A_i}.
# ]
# 两边取期望（期望=概率）即得
# [
# \Pr\Big(\bigcup_i A_i\Big)=\mathbb E[\mathbf{1}*{\cup_i A_i}]
# \le \sum_i \mathbb E[\mathbf{1}_{A_i}]
# =\sum_i \Pr(A_i).
# ]
# 
# （也可把它看作“容斥原理”的第一阶截断。）
# 
# ---
# 
# ## 2) 把并联界用在“最大值事件”上
# 
# 令
# [
# A_m={|z_m|\ge \varepsilon}.
# ]
# 则
# [
# {\max_{1\le m\le M}|z_m|\ge \varepsilon}=\bigcup_{m=1}^M A_m.
# ]
# 套用并联界：
# [
# \Pr!\Big(\max_{1\le m\le M}|z_m|\ge \varepsilon\Big)
# \le \sum_{m=1}^M \Pr(|z_m|\ge \varepsilon).
# \tag{★}
# ]
# 
# ---
# 
# ## 3) 用 Chebyshev 给每个 (\Pr(|z_m|\ge \varepsilon)) 上界
# 
# 前面已算过（均值 0，方差 (\mathrm{Var}(z_m)=1/m)）：
# [
# \Pr(|z_m|\ge \varepsilon)\ \le\ \frac{\mathrm{Var}(z_m)}{\varepsilon^2}
# =\frac{1}{m,\varepsilon^2}.
# ]
# 代回 (★)：
# [
# \Pr!\Big(\max_{1\le m\le M}|z_m|\ge \varepsilon\Big)
# \le \frac{1}{\varepsilon^2}\sum_{m=1}^M \frac{1}{m}.
# \tag{★★}
# ]
# 
# ---
# 
# ## 4) 把调和级数再上界：(\sum_{m=1}^M\frac1m\le 1+\ln M)
# 
# 用积分比较（或调和数 (H_M\le 1+\ln M)）：
# [
# \sum_{m=1}^M \frac1m \ \le\ 1+\ln M.
# ]
# 带入 (★★)：
# [
# \boxed{\Pr!\Big(\max_{1\le m\le M}|z_m|\ge \varepsilon\Big)
# \ \le\ \frac{1+\ln M}{\varepsilon^2}.}
# ]
# 
# 若想让左边 (\le \delta)，解出 (\varepsilon)：
# [
# \boxed{\ \varepsilon \ \ge\ \sqrt{\dfrac{1+\ln M}{\delta}}\ }.
# ]
# 
# ---
# 
# ## 5) 关键点与常见误区
# 
# * 并联界**不需要**事件独立；正因为 (z_m) 之间强相关，我们**不能**把概率当独立相乘，只能用并联界这样的保守工具。
# * 并联界常**偏松**；若 (x_i) 有更强的性质（有界、次高斯等），可以用 Hoeffding/Bernstein/鞅不等式得到更紧的“同时”界。
# 
# 需要的话，我可以再给你画个小图，展示“并集 ≤ 概率和”的直觉，或者用一个数值模拟检查这个上界。
# 

# **有效前沿（Efficient Frontier）**是马科维茨均值–方差理论中的那条“最优组合曲线”：
# 在给定风险（方差/标准差）水平下，**期望收益最大**的那些投资组合，或等价地，在给定收益目标下**方差最小**的那些组合，构成的集合。
# 
# ---
# 
# ## 1) 形式化定义
# 
# 令组合权重 (\alpha\in\mathbb{R}^d)、(\mathbf{1}^\top\alpha=1)（预算约束），
# [
# \mathbb E[r_p]=\alpha^\top\mu,\qquad
# \mathrm{Var}(r_p)=\alpha^\top\Sigma\alpha.
# ]
# 
# * **有效组合**：不存在另一个可行 (\tilde\alpha) 使
#   (\tilde\alpha^\top\mu \ge \alpha^\top\mu) 且
#   (\tilde\alpha^\top\Sigma\tilde\alpha \le \alpha^\top\Sigma\alpha)，并且至少一项严格不等。
# * **有效前沿**：所有有效组合在“风险–收益平面”（横轴 (\sigma=\sqrt{\alpha^\top\Sigma\alpha})，纵轴 (\alpha^\top\mu)）上的轨迹。
# 
# ---
# 
# ## 2) 曲线长什么样？
# 
# 当允许做空且 (\Sigma\succ 0) 时，有闭式表达：
# 解最小方差问题 (\min_\alpha \alpha^\top\Sigma\alpha) s.t.
# (\alpha^\top\mu = R,\ \mathbf{1}^\top\alpha=1) 得
# [
# \alpha^\star(R)=\lambda,\Sigma^{-1}\mu+\gamma,\Sigma^{-1}\mathbf{1},
# ]
# 并得到**前沿方差**（抛物线）：
# [
# \sigma^2(R)=\frac{A R^2-2 B R + C}{D},\quad
# A=1^\top\Sigma^{-1}1,\ B=1^\top\Sigma^{-1}\mu,\ C=\mu^\top\Sigma^{-1}\mu,\ D=AC-B^2>0.
# ]
# 图形上是一条**上包络**的抛物线；其下方任何点都“被支配”（同风险收益更低或同收益风险更高）。
# 
# ---
# 
# ## 3) 两个基点
# 
# * **全球最小方差（GMV）组合**：前沿上最左点（风险最低）。
#   (R) 取使 (\sigma^2(R)) 最小的值即可求得，对应权重 (\propto \Sigma^{-1}\mathbf{1})（再归一化）。
# * **切线/最大夏普比组合（含无风险资产 (r_f) 时）**：
#   用超额收益 (\mu-r_f \mathbf{1}) 最大化夏普比 (\frac{\alpha^\top(\mu-r_f\mathbf{1})}{\sqrt{\alpha^\top\Sigma\alpha}})，得到
#   (\alpha^\star \propto \Sigma^{-1}(\mu-r_f\mathbf{1}))。
#   与无风险资产合成的**资本市场线（CML）**是一条直线，主导仅含风险资产的前沿上方部分。
# 
# ---
# 
# ## 4) 如何“走”出前沿（数值）
# 
# * **方差受限最大化收益**：
#   (\max_\alpha \alpha^\top\mu) s.t. (\alpha^\top\Sigma\alpha\le\sigma_{\max}^2,,1^\top\alpha=1)。
# * **收益达标最小化方差**：
#   (\min_\alpha \alpha^\top\Sigma\alpha) s.t. (\alpha^\top\mu\ge R,,1^\top\alpha=1)。
# * 或**标量化**：(\min_\alpha \alpha^\top\Sigma\alpha - \lambda,\alpha^\top\mu) s.t. (1^\top\alpha=1)。
#   不同 (\sigma_{\max})/(R)/(\lambda) 对应前沿上的不同点。
# 
# ---
# 
# ## 5) 约束与现实考量
# 
# * **不允许做空**（(\alpha_i\ge 0)）或有**持仓上限/交易成本**时，前沿仍存在但不再是简单抛物线，需用二次规划（QP）或锥规划数值求解。
# * 估计误差（(\mu,\Sigma) 的样本误差）会导致“极端权重”；常见对策：正则化/稳健优化、Black–Litterman、对角加载、因子协方差等。
# 
# ---
# 
# ## 6) 一句话记忆
# 
# > **有效前沿 = 在均值–方差平面里“不可被改进”的投资组合集合**：
# > 给定风险取最高收益，或给定收益取最低风险；引申出 GMV、含无风险资产的切线组合与资本市场线。
# 

# In[18]:


from IPython.display import Math, display
display(Math(r"\mathbb E[r_p]=\alpha^\top\mu,\ \mathrm{Var}(r_p)=\alpha^\top\Sigma\alpha."))

