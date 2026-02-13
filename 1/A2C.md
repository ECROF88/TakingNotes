# A2C - Synchronous Advantage Actor-Critic

## 从 Policy Gradient 到 Actor-Critic

### 回顾 REINFORCE (蒙特卡洛策略梯度)

策略梯度的基本形式：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot G_t \right]
$$

其中 $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$ 是从时刻 $t$ 开始的折扣回报。

问题：用蒙特卡洛采样估计 $G_t$，方差很大（因为要等整条轨迹结束才能算回报）。

### 引入 Baseline 降低方差

减去一个 baseline $b(s)$（通常选状态价值函数 $V(s)$）：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot (G_t - b(s_t)) \right]
$$

减去 baseline 不改变梯度期望（因为 $\mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)] = 0$），但能显著降低方差。

### Advantage Function 优势函数

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

含义：在状态 $s$ 下选动作 $a$ 比平均水平好多少。
- $A > 0$：这个动作比平均好，应该增加概率
- $A < 0$：这个动作比平均差，应该减少概率

## Actor-Critic 架构

两个组件：
1. Actor（策略网络）：$\pi_\theta(a|s)$，负责选动作
2. Critic（价值网络）：$V_\phi(s)$，负责评估状态好坏

Actor 用策略梯度更新，Critic 用 TD 学习更新。

### 为什么需要 Critic？

REINFORCE 用蒙特卡洛回报 $G_t$ 来估计 $Q(s,a)$，方差大。
Actor-Critic 用 Critic 网络来估计价值，用 TD 误差代替蒙特卡洛回报，方差小但引入偏差（bias-variance tradeoff）。

## A2C 算法细节

### TD 误差作为 Advantage 的估计

不需要显式计算 $Q(s,a)$，用 TD 误差近似 Advantage：

$$
\hat{A}_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

这是 Advantage 的一个无偏估计（在 $V_\phi$ 准确的前提下）。

也可以用 n-step 回报来估计：

$$
\hat{A}_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V_\phi(s_{t+n}) - V_\phi(s_t)
$$

n 越大方差越大偏差越小，n=1 就是纯 TD，n=∞ 就退化成蒙特卡洛。

### GAE (Generalized Advantage Estimation)

GAE 是对不同步长 TD 误差的指数加权平均，用 $\lambda$ 控制 bias-variance tradeoff：

$$
\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}
$$

其中 $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ 是 TD 误差。

- $\lambda = 0$：退化为 1-step TD，低方差高偏差
- $\lambda = 1$：退化为蒙特卡洛，高方差低偏差
- 实践中 $\lambda = 0.95$ 左右效果好

### A2C 的 "Synchronous" 含义

A2C 是 A3C (Asynchronous Advantage Actor-Critic) 的同步版本。

**A3C（异步）：** 多个 worker 异步并行采集数据，各自计算梯度后异步更新全局参数。
- 问题：异步更新导致不同 worker 用的策略参数不一致（stale gradients 过时梯度），训练不稳定
- stale gradients 解释：Worker A 在采样时用的是参数 θ₀，但采样期间 Worker B 已经把参数更新到 θ₁ 了，当 Worker A 计算完梯度准备更新时，它的梯度是基于旧参数 θ₀ 算的（已经"过时"了），用这种过时梯度更新参数会导致训练方向不准确

**A2C（同步）：** 多个 worker 同步并行采集数据，等所有 worker 都采完后，汇总计算梯度，统一更新参数。
- 优点：没有 stale gradients 问题，所有 worker 用的都是同一版本参数，计算的梯度都基于当前参数，训练更稳定
- 实践中 A2C 效果和 A3C 差不多甚至更好，实现也更简单

### 损失函数

A2C 的总损失由三部分组成：

$$
L = L_{policy} + c_1 \cdot L_{value} - c_2 \cdot H(\pi_\theta)
$$

1. 策略损失（Actor Loss）：
$$
L_{policy} = -\mathbb{E}_t \left[ \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t \right]
$$

2. 价值损失（Critic Loss）：
$$
L_{value} = \mathbb{E}_t \left[ (V_\phi(s_t) - V_t^{target})^2 \right]
$$

其中 $V_t^{target} = \hat{A}_t + V_\phi(s_t)$ 或直接用 n-step 回报。

3. 熵正则化（Entropy Bonus）：
$$
H(\pi_\theta) = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)
$$

鼓励探索，防止策略过早收敛到确定性策略。$c_2$ 通常取 0.01。

### 算法流程

```
初始化 Actor 参数 θ，Critic 参数 ϕ（可以共享底层网络）
初始化 N 个并行环境

for each iteration:
    1. 所有 N 个环境用当前策略 π_θ 同步采集 T 步数据
       收集 {s_t, a_t, r_t, s_{t+1}} for t = 1..T, env = 1..N
    
    2. 对每个环境计算 Advantage 估计（用 GAE 或 n-step TD）
    
    3. 计算总损失 L = L_policy + c1 * L_value - c2 * H(π)
    
    4. 反向传播，更新 θ 和 ϕ
```

## A2C vs 其他算法对比

| 特性 | REINFORCE | A2C | PPO |
|------|-----------|-----|-----|
| 回报估计 | 蒙特卡洛 | TD / GAE | TD / GAE |
| 方差 | 高 | 低 | 低 |
| 偏差 | 无 | 有（来自 Critic） | 有 |
| 数据效率 | 低（on-policy） | 中（on-policy） | 中（on-policy，但可多次复用） |
| 稳定性 | 差 | 中 | 好（有 clip 机制） |
| 并行 | 无 | 同步多环境 | 同步多环境 |

## 关键点总结

1. Advantage = Q - V，衡量动作相对平均水平的好坏
2. 用 Critic 网络估计 V，用 TD 误差近似 Advantage，降低方差
3. Synchronous 指多个 worker 同步采集、统一更新，比 A3C 的异步方式更稳定
4. 熵正则化防止策略坍缩，鼓励探索
5. Actor 和 Critic 可以共享底层特征提取网络，只在最后一层分叉
6. A2C 是 PPO 的前身，PPO 在 A2C 基础上加了 clipped surrogate objective 来限制策略更新幅度
