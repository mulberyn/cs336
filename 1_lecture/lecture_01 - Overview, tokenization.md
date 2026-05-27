---
type: lecture
tags:
  - lecture
  - cs336
  - llm
  - tokenization
course: CS336
created: 2026-05-26
---
# Overview, tokenization

> [!NOTE]
> 链接：[[lecture_01.pdf]]

## 一、概述
![[file-20260527162746748.png]]

课程讲解的机制、思想是可以跨规模迁移的，不过关于哪些数据是否好，是否能够帮助决策的直觉需要根据不同的数据规模而定的，也需要大量的实验来验证。
机制上可以了解**并行是如何实现的**、**内核是如何加速的**，但是哪些建模能够改善模型的直觉需要通过大量的实验来进行验证。
**效率和成本是在大规模训练和实验的过程中很重要的一点。**
efficency 既需要硬件性能，也需要算法优化。

![[file-20260527162703835.png]]

### 1.1 课程存在的原因

课程提出了一个非常重要的问题：

> AI 研究者正在逐渐与底层技术脱节。

发展过程：

- 2016：研究者自己实现和训练模型。
- 2018：研究者下载 BERT 等模型进行 fine tune。
- 2022~2026：研究者更多通过 API prompt 使用 GPT、Claude、Gemini 等模型。

抽象层级提高虽然提升了生产力，但是这些抽象不像编程语言、操作系统那样稳定可靠，它们是“泄漏的（leaky abstraction）”。
很多真正重要的问题仍然需要深入到底层：

- Transformer 到底如何工作？
- Attention 为什么有效？
- GPU kernel 为什么会影响训练速度？
- 为什么某些结构在大规模下才有效？

因此课程核心思想：

> Understanding via Building
>
> 通过“从零构建”来真正理解语言模型。

### 1.2 Frontier Model 的工业化问题

当前前沿模型已经进入工业化阶段。

例如：

- GPT-4 训练成本据称超过 1 亿美元。
- xAI 在 2025 年使用约 23 万张 GPU 训练 Grok。

这意味着：

- 普通研究者无法复现 frontier model。
- 大量关键工程细节没有公开。
- 很多经验来自工业界长期 trial-and-error。

但是课程强调：

即使无法训练 GPT-5 规模模型，小模型中的很多“机制”和“系统思想”仍然是可以迁移的。

### 1.3 三类知识

课程中提到了三种知识：

#### 1.3.1 Mechanics（机制）

即系统如何运作。

例如：

- Transformer 的计算方式
- Attention 的实现
- Parallelism 的实现
- FlashAttention 的原理
- Kernel fusion 的意义

这些知识是可以跨规模迁移的。

#### 1.3.2 Mindset（工程思维）

即：

> 如何极致压榨硬件性能。

包括：

- 减少 data movement
- 减少 memory bandwidth 开销
- 提高 GPU occupancy
- 用 profiling 找瓶颈
- 用 scaling law 降低试错成本

这类思维在大模型时代极其重要。

#### 1.3.3 Intuition（直觉）

例如：

- 什么数据最好
- 什么结构效果最好
- 为什么 SwiGLU 比 ReLU 好
- 为什么某些 trick 只在大规模下有效

很多内容目前并没有理论解释，而是：

> 依赖大量实验。

因此 Intuition 不一定能跨规模迁移。

### 1.4 Bitter Lesson

课程提到了著名观点：

> The Bitter Lesson

错误理解：

> scale 最重要，算法不重要。

正确理解：

> 能够扩展到大规模的算法才重要。

课程中给出一个很重要的公式：

$$accuracy = efficiency \times resources$$

随着规模增大：

- efficiency 的重要性会越来越高
- 大规模训练无法容忍浪费
- 系统优化的重要性会越来越高

例如：

- FlashAttention
- fused kernel
- MoE
- KV Cache
- Quantization
- Parallelism

本质上都在提升 efficiency。

---

## 二、现代大模型发展

MLE 和 RL 相关知识在 AI 领域发挥作用。

### 2.1 Pre-neural Era（神经网络之前）

早期语言模型主要是：

#### N-gram Language Model

核心思想：

$$
P(w_t | w_{t-1}, w_{t-2}, ...)
$$

只依赖有限历史。

应用：

- 机器翻译
- 语音识别
- 搜索推荐

问题：

- 无法建模长距离依赖
- 数据稀疏严重
- 泛化能力差

### 2.2 Neural Era（2010s）

#### LSTM

Long Short-Term Memory：

- 解决 RNN 梯度消失问题
- 可以建模更长上下文
- 在 NLP 中统治很长时间

#### Seq2Seq

用于机器翻译：

- Encoder 编码输入
- Decoder 生成输出

#### Attention

Bahdanau Attention：

- 解决固定长度 bottleneck
- Decoder 可以关注输入不同部分

Attention 的提出实际上为 Transformer 铺平了道路。

### 2.3 Transformer 时代

2017 年：

> Attention Is All You Need

Transformer 成为现代 LLM 的核心。

关键特性：

- 并行计算能力强
- 更适合 GPU
- 长距离依赖能力强
- scaling 性能优秀

### 2.4 Foundation Model 时代

#### BERT

特点：

- 双向 Transformer
- Masked Language Modeling
- 更适合理解任务

#### GPT 系列

GPT-2：

- 展现流畅文本生成能力
- 初步展现 zero-shot 能力

GPT-3：

- In-context learning
- Prompt engineering 兴起

#### Chinchilla Scaling Law

DeepMind 提出：

> 模型参数和训练 token 数应该平衡。

经典经验：

$$
D \approx 20N
$$

其中：

- D：token 数
- N：参数量

例如：

- 70B 模型应该训练约 1.4T token

### 2.5 Open Model 的崛起

课程中重点强调了开源的重要性。

代表模型：

- Llama
- Mistral
- DeepSeek
- Qwen
- Kimi
- GLM
- Olmo
- Nemotron

意义：

- 推动学术研究
- 提供可复现性
- 促进技术创新
- 让课程成为可能

### 2.6 Language Model 的定义变化

课程中给出的演化：

- 2018：需要 finetune 的模型
- 2020：可以 prompt 的模型
- 2022：可以对话的模型
- 2026：可以 autonomous action 的 agent

但是底层 fundamentals 没变：

- attention
- optimization
- kernels
- systems

变化的是：

- context 更长
- inference 更重要
- latency 更关键
- memory pressure 更大

- （贴上你跑通的代码，并注释它的作用）
- （例如：这里的 flash attention 前向传播，mask 是如何传入的）

---

## 三、课程相关

课程和作业相对应：

1. basic: Assignment 1: tokenization, model architecture, training
2. system: Assignment 2: kernels, parallelism, inference
3. scaling_laws: Assignment 3: scaling laws
4. data: Assignment 4: evaluation, curation, transformation, filtering, deduplication, mixing
5. alignment: Assignment 5: RLHF, RL algorithms, RL systems

课程贯穿始终的核心问题：

> 在固定资源下，如何训练出最好的模型？

资源包括：

- compute
- memory
- bandwidth
- data

因此：

- systems 是 efficiency
- tokenization 是 efficiency
- data filtering 是 efficiency
- scaling law 也是 efficiency

### 3.1 从零训练一个语言模型

#### 3.1.1 Tokenization

Tokenizer 的目标：

> 将字符串映射为 token id。

形式化定义：

$$
Tokenizer: text \leftrightarrow integers
$$

课程中强调：

Transformer 的 attention 是 quadratic 的：$O(n^2)$

因此：

- token 越少越好
- sequence 越短越好

Tokenizer 本质上是 compression。

#### 3.1.2 Transformer

Transformer 是现代语言模型的基础。

核心结构：

- Multi-head Attention
- MLP
- Residual
- LayerNorm

Transformer 的一些改进：（深入了解可以关注一下 291 行相关的几个模型）

![[file-20260527171418265.png]]

##### Activation Function

- ReLU
- GELU
- SwiGLU

SwiGLU 在现代模型中非常常见。

##### Positional Encoding

- Sinusoidal
- RoPE

RoPE 是当前主流。

优点：

- 更适合 extrapolation
- 更适合长上下文

##### Attention 改进

- Sparse Attention
- Sliding Window Attention
- GQA
- MLA

这些都在解决：

> attention 太贵的问题。

##### 新架构探索

- Mamba
- State Space Model
- Linear Attention

目标：

- 降低 attention quadratic complexity
- 提高长上下文效率

##### MoE

Mixture of Experts：

核心思想：

- 参数量很大
- 每次只激活少量 expert

实现：

- 增加 capacity
- 降低 FLOPs

#### 3.1.3 训练

![[file-20260527172106868.png|679]]

训练本质：

> 优化模型参数，使 loss 最小。

核心组件：

##### Loss Function

通常使用：

- Cross Entropy Loss
- Multi-token Prediction

##### Optimizer

常见的优化器：Adam（现在也有 Kimi K2 模型使用 Muon 优化器）

现代常见：

- AdamW
- SOAP
- Muon

##### Initialization

例如：

- Xavier Init
- muP

初始化非常重要。

因为：

- 影响训练稳定性
- 影响 gradient scale
- 影响 hyperparameter transfer

##### Learning Rate Schedule

例如：

- cosine decay
- warmup-stable-decay

##### Regularization

例如：

- dropout
- weight decay

##### Batch Size

大模型训练中：

- batch size 极大
- communication 成本很高

critical batch size 是一个重要概念。

##### 三者平衡

课程给出了一个非常关键的观点：

模型训练始终在平衡：

1. Expressivity
2. Stability
3. Efficiency

###### Expressivity

模型表达复杂模式的能力。

###### Stability

训练稳定。

例如：

- gradient 不爆炸
- parameter norm 合理

###### Efficiency

训练快、推理快。

#### 3.1.4 作业1

从零完成一个语言模型

![[file-20260527172211991.png]]

作业内容：

- 实现 BPE tokenizer
- 实现 Transformer
- 实现 AdamW
- 实现训练循环
- 在 TinyStories/OpenWebText 上训练

Leaderboard：

- 固定训练预算
- 最低 perplexity 获胜

这是一个非常偏 research engineering 的课程。

---

### 3.2 System

本节的目的就是在完成训练一个语言模型的基础上，对这个模型进行 kernel 级别的优化，让该模型变快。

注意一下 [How to Scale your model?](https://jax-ml.github.io/scaling-book/) 这本书

#### 3.2.1 Resource Accounting

课程强调：

> 训练前必须先算账。

例如：

$$
FLOPs = 6ND
$$

其中：

- N：参数量
- D：token 数

例如：

- 70B 参数
- 1T token

需要：

$$
4.2 \times 10^{23} FLOPs
$$

#### 3.2.2 Roofline Analysis

GPU 有两个核心瓶颈：

- compute throughput
- memory bandwidth

例如 B200：

- 2.25 PFLOP/s
- 8TB/s bandwidth

问题：

到底是：

- compute bound
- memory bound

#### 3.2.3 Kernel

Kernel 是运行在 GPU 上的函数。

PyTorch 中每个 primitive operation：

- 都会 launch kernel

问题：

kernel launch 很贵。

因此需要：

##### Kernel Fusion

例如：

不要：

- matmul
- 写回 HBM
- activation
- 再读 HBM

而是：

- fused matmul + activation

减少 data movement。

##### FlashAttention

核心思想：

- tiling
- 避免 materialize 完整 attention matrix
- 减少 HBM IO

这也是现代 LLM inference/training 的核心优化之一。

#### 3.2.4 Parallelism

当 GPU 数量巨大时：

- GPU 间通信成为瓶颈

常见并行：

- Data Parallelism
- Tensor Parallelism
- Pipeline Parallelism
- Sequence Parallelism
- Expert Parallelism

核心原则仍然是：

> minimize data movement

#### 3.2.5 Inference

Inference 分为：

##### Prefill

特点：

- 可以并行
- compute-bound

##### Decode

特点：

- 一次生成一个 token
- memory-bound

因此 decode 是 inference 的核心瓶颈。

#### 3.2.6 加速 Decode

课程提到：

##### Quantization

例如：

- int8
- fp8
- 4bit

##### Distillation

用小模型模仿大模型。

##### Speculative Decoding

使用 draft model 先生成。

full model 再验证。

##### Continuous Batching

提升 GPU utilization。

### 3.3 Scaling Law

第一节的目的是创建一个大模型，第二节的目的是让大模型变快。而本节的目的是如何让大模型的规模变大。

该节的目的仍然是为了 efficency，让大模型的训练变得更加**可预测**。用训练小模型/做小实验的方式来验证该方法是否能优化大模型，这样可以节省算力和时间的开销，并且也可以用于预测一些方法在大模型上会有怎么样的效果，并且简化大模型训练的过程（将任务转变成超参数的调整）

#### 3.3.1 核心思想

如果目标是：

$$
10^{25} FLOPs
$$

不可能直接进行超参数搜索。

因此：

- 先做小实验
- 拟合 scaling law
- 再 extrapolate 到大规模

#### 3.3.2 Scaling Recipe

关键概念：

> 不要只考虑单个模型。

而是：

> 一整套 scaling recipe。

例如：

- hidden dim 如何变化
- depth 如何变化
- learning rate 如何变化
- batch size 如何变化

#### 3.3.3 Predictability

课程强调：

> Predictability >= Optimality

原因：

大规模实验太贵。

稳定可预测：

- 比偶然最优更重要。

### 3.4 Data

本节的重点就是大模型训练的材料——数据 Data，数据质量决定了模型质量，数据方向决定了模型的功能。

#### 3.4.1 Evaluation

评测有两类：

##### Internal Evaluation

用于：

- 模型开发
- 对比实验
- scaling 分析

##### External Evaluation

用于：

- 真实场景效果
- 实际能力验证

例如：

- SWE-Bench
- Terminal-Bench
- GPQA

#### 3.4.2 Data Source

数据来源：

- Common Crawl
- GitHub
- arXiv
- books
- forums

#### 3.4.3 Data Processing

##### Transformation

例如：

- HTML -> text
- PDF -> text

##### Filtering

移除：

- 低质量数据
- harmful content

##### Deduplication

目标：

- 减少重复训练
- 防止 memorization
- 节省 compute

常见：

- Bloom Filter
- MinHash

##### Data Mixing

不同数据比例如何分配。

#### 3.4.4 Synthetic Data

现代模型中：

- synthetic data 越来越重要
- self-play 越来越重要

例如：

- reasoning trace
- code synthesis
- tool-use trajectory

### 3.5 Alignment

后训练，对齐一些下游任务

#### 3.5.1 为什么需要 Alignment

预训练目标只是：

> next token prediction

但真实需求：

- helpful
- harmless
- honest

因此需要 alignment。

#### 3.5.2 基本流程

1. 模型生成回答
2. 人类/模型评分
3. 更新模型偏好

#### 3.5.3 常见算法

##### PPO

经典 RLHF 方法。

##### DPO

更简单。

直接优化 preference。

##### GRPO

去掉 value model。

#### 3.5.4 Alignment 的困难

- RL 不稳定
- rollout 很贵
- inference infra 很复杂
- on-policy 数据昂贵

---

## 四、Tokenization

[[Let's build the GPT Tokenizer]]

Tokenization 是整个课程第一部分。

课程强调：

> Tokenizer 本质上是 compression + abstraction。

### 4.1 Tokenizer 的定义

Tokenizer 负责：

- encode
- decode

即：

$$
text \leftrightarrow token ids
$$

课程中抽象接口：

```python
class Tokenizer(ABC):
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError
```

### 4.2 为什么需要 Tokenization

Transformer 无法直接处理字符串。

模型需要：

- 离散 token
- embedding lookup
- sequence modeling

Tokenization 的目标：

#### 压缩序列长度

例如：

- 1000 bytes
- 压缩成约 250 token

这样 attention 开销会显著下降。

#### 提供语义 chunk

例如：

- “hello”
- “ization”
- “function”

让模型更容易学习语义。

### 4.3 Character Tokenizer

课程首先介绍 Character-based Tokenizer。

```python
list(map(ord, string))
```

优点：

- 简单
- 可逆

缺点：

#### Vocabulary 太大

Unicode 大约 15 万字符。

#### Rare Character 太多

例如：

- emoji
- 生僻字

#### Compression Ratio 很差

sequence 太长。

因此：

- attention cost 太高

### 4.4 Byte Tokenizer

Byte Tokenizer 使用 UTF-8 bytes。

```python
string.encode("utf-8")
```

优点：

- vocab 固定 256
- 不会 OOV
- 完全可逆

缺点：

#### Compression Ratio = 1

意味着：

- token 数过多
- sequence 过长
- attention 开销巨大

课程强调：

attention 是 quadratic 的。

因此 byte-level Transformer 很贵。

### 4.5 Word Tokenizer

按单词切分：

```python
regex.findall(r"\w+|.", string)
```

优点：

- token 语义强
- compression 好

缺点：

#### Vocabulary 巨大

所有单词都可能进入 vocab。

#### OOV 问题

未见单词：

- 使用 UNK token
- 影响 perplexity
- 泛化能力差

#### 长尾问题

很多词极少出现。

### 4.6 BPE（Byte Pair Encoding）

课程核心。

#### 4.6.1 核心思想

从 byte 开始。

不断合并：

> 最频繁出现的 token pair。

例如：

```text
h e l l o
```

可能逐渐合并：

```text
h + e -> he
he + l -> hel
```

最终形成常见 chunk。

#### 4.6.2 BPE 的优势

##### 可学习

vocab 来自数据。

##### 自适应

高频内容：

- 使用更少 token

低频内容：

- 退化成 byte

##### 不会真正 OOV

因为最终一定能退回 byte。

#### 4.6.3 Merge 过程

课程中的 merge：

```python
if indices[i] == pair[0] and indices[i + 1] == pair[1]:
```

作用：

- 找到相邻 pair
- 替换为新 token

#### 4.6.4 Train BPE

训练流程：

##### Step 1

初始化 byte vocab：

```python
vocab = {x: bytes([x]) for x in range(256)}
```

##### Step 2

统计 pair frequency：

```python
counts[(index1, index2)] += 1
```

##### Step 3

找到最频繁 pair：

```python
pair = max(counts, key=counts.get)
```

##### Step 4

merge pair。

##### Step 5

重复。

#### 4.6.5 Compression Ratio

课程定义：

$$
compression\ ratio = bytes / token
$$

越大说明：

- sequence 越短
- attention 越省

但是：

- vocab 会变大
- embedding 更 sparse

因此需要 balance。

### 4.7 GPT Tokenizer 的特点

课程中提到：

- 空格和单词通常绑定
- 开头和中间 token 不同
- 数字通常按几位切分

例如：

```text
" hello"
```

和：

```text
"hello"
```

可能是不同 token。

这是因为：

Tokenizer 在学习统计规律。

### 4.8 Tokenizer-Free 的未来

课程最后提到：

未来可能：

> end-to-end byte model

例如：

- Byte Latent Transformer
- Byte-level Transformer
- tokenizer-free architecture

但是目前：

- compute 成本仍然太高
- attention 太贵

因此 BPE 仍然是主流。

---

## 五、课程核心思想总结

### 5.1 CS336 的真正目标

这门课并不是：

- 教你调用 API
- 教你 prompt engineering
- 教你快速做 AI application

而是：

> 从底层真正理解现代语言模型。

包括：

- 数学
- 系统
- kernel
- distributed training
- scaling
- data
- RLHF

### 5.2 整门课贯穿的主线

课程实际上始终围绕一个问题：

> 如何在有限资源下训练最好的模型？

因此：

- Tokenization 是 efficiency
- Kernel 是 efficiency
- Parallelism 是 efficiency
- Scaling law 是 efficiency
- Data filtering 是 efficiency

### 5.3 当前 LLM 研究趋势

从课程内容可以看出当前趋势：

#### 更长上下文

例如：

- RoPE
- Sliding Window Attention
- KV Cache

#### 更高推理效率

例如：

- Quantization
- Speculative Decoding
- Continuous Batching

#### 更大的模型

例如：

- MoE
- Expert Parallelism
- Scaling Law

#### 更强的 Agent 能力

例如：

- tool use
- RL
- autonomous agent

### 5.4 我目前需要重点掌握的内容

对于后续学习来说，本 lecture 最重要的内容包括：

#### 必须真正理解

- Tokenization
- Attention
- Transformer
- AdamW
- Scaling Law
- GPU Memory/Compute
- Parallelism

#### 需要重点实践

- 手写 BPE
- 手写 Transformer
- 手写 Training Loop
- Profiling
- Triton Kernel

#### 需要逐渐建立 intuition

- 为什么某些 trick 有效
- 为什么某些结构能 scale
- 为什么某些数据更重要

### 5.5 Lecture 01 的一句话总结

> 现代大模型的核心不是“会调用 API”，而是：
>
> 在有限算力、有限数据、有限带宽下，通过系统、算法和工程优化，把 efficiency 压榨到极致。

