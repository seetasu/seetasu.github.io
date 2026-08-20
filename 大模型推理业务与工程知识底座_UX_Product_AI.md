# 大模型推理业务与工程知识底座（面向 UX / Product / AI）

> **用途**：供 AI 在后续生成推理相关 HTML、解释材料、研究框架或产品讨论材料时使用。  
> **目标读者**：需要理解大模型推理业务与工程场景的 UX 设计师、产品设计师，以及辅助其工作的 AI。  
> **内容范围**：从模型训练完成后的推理准备，到单次推理执行、在线 Serving、分布式扩展、性能指标、优化、观测诊断、容量与生产运行，以及主流推理工具生态。  
> **不包含**：当前项目的具体页面、信息架构、导航、交互方案、Dashboard/Report 设计、具体产品功能方案。  
> **版本说明**：竞品与工具生态基于截至 2026-08-20 可查的官方公开资料整理；具体特性可能随版本变化。

---

## 1. 先建立一个总的 Mental Model

大模型推理不是“把训练好的模型运行起来”这么简单。进入真实业务后，它同时包含三个层面：

1. **模型计算**：给定输入 token，执行 Transformer，生成下一个 token。
2. **系统执行**：在有限计算、显存和通信资源上，持续调度大量不同长度、不同状态的请求。
3. **服务优化**：在质量、延迟、吞吐、容量、成本和可靠性等约束下寻找合适配置。

可以把整个场景理解为：

```text
Trained Model
    ↓
Inference Adaptation
    ↓
Engine / Compile / Runtime Preparation
    ↓
Single-request Execution
    ↓
Multi-request Serving
    ↓
Distributed Serving
    ↓
Observe / Benchmark / Diagnose
    ↓
Optimize / Validate
    ↓
Production Operation
```

推理 Serving 的目标不是一个传统训练 loss，而是一个**受约束的多目标系统优化问题**：

```text
maximize:
  throughput / hardware efficiency / service capacity

minimize:
  TTFT / TPOT / tail latency / memory / cost

subject to:
  output quality
  accuracy loss
  hardware capacity
  SLO / SLA
  reliability requirements
```

因此不存在脱离场景的“最快配置”或“最佳推理配置”。任何性能结论都必须放回具体的 **Model × Workload × Hardware × Configuration** 中理解。

---

## 2. 推理场景的核心上下文与对象

### 2.1 一个推理 Scenario 由什么构成

一个可被比较、分析或优化的推理场景，至少需要四类上下文：

```text
Inference Scenario
├── Model / Variant
├── Workload
├── Hardware
└── Configuration
```

### Model / Variant

包括：

- 模型架构：Dense / MoE、Attention 形式等
- 参数规模
- 模型版本
- Context Length
- 权重精度
- Quantization 版本
- 是否经过 Pruning、Distillation 等处理
- 是否存在 LoRA / Adapter
- 模型的算子组成与硬件支持情况

### Workload

描述系统真正面对的请求，而不是只描述模型：

- Input Sequence Length，ISL
- Output Sequence Length，OSL
- Request Rate
- Concurrency
- Prompt 长度分布
- 输出长度分布
- Long-context 比例
- 多轮会话比例
- Shared Prefix 比例
- Streaming / Non-streaming
- 请求优先级
- Traffic burst / steady traffic
- 在线业务还是离线批处理

### Hardware

包括：

- GPU / NPU 型号
- 单机 Device 数量
- 多机节点数量
- HBM 容量与带宽
- Device 间互联
- Node 间网络
- 拓扑结构
- CPU / Host 能力
- Storage / Offload 能力

### Configuration

包括：

- Inference Engine
- Compiler / Runtime
- Precision
- TP / DP / PP / EP 等并行策略
- KV Cache 配置
- Batch / Continuous Batching 参数
- Scheduler 参数
- Prefill / Decode 策略
- Routing
- Serving topology
- Replica 数量
- Worker 资源分配
- Offloading
- Speculative Decoding
- Prefix Cache 等

**一个性能数字如果没有这些上下文，通常不具有可比较性。**

---

### 2.2 推理世界中的核心对象

对 UX / Product 来说，理解“用户操作和判断的对象”非常重要。推理领域的核心对象可以分为以下几组。

#### 模型对象

```text
Model
└── Model Variant
    ├── Precision Variant
    ├── Quantized Variant
    ├── Parallel Variant
    └── Engine-specific Artifact
```

#### 请求对象

```text
Request
├── Prompt
├── Input Tokens
├── Output Tokens
├── Sampling Parameters
├── Priority
└── Runtime State
```

#### 执行对象

```text
Run
├── Request
├── Batch
├── Prefill
├── Decode Step
├── Operator
└── Kernel
```

#### 系统对象

```text
Endpoint
├── Router
├── Queue
├── Scheduler
├── Worker
├── Runtime
└── Device
```

#### 内存对象

```text
Tensor
KV Cache
KV Block / Page
Buffer
HBM
Host Memory
```

#### 观测与实验对象

```text
Metric
Log
Trace
Profile
Hardware Counter
Benchmark
Baseline
Experiment
Comparison
```

这些对象处在不同抽象层。一个典型问题经常需要在它们之间建立关系，而不是只研究单一对象。

例如：

```text
P99 TTFT 变差
    ↓
某类 Request 排队时间变长
    ↓
Prefill Queue 拥塞
    ↓
长 Prompt 占据 Prefill Worker
    ↓
当前 P/D 资源比例不适合该 Workload
```

---

## 3. 从训练好的模型到可在线提供服务

训练完成只是推理生命周期的起点。

### 3.1 模型产物

训练完成后通常会得到：

- Model checkpoint / weights
- Model configuration
- Tokenizer
- Architecture definition
- 可能还包括 Adapter / LoRA 等附加参数

此时模型具有“模型能力”，但并不意味着它已经：

- 能在目标硬件运行
- 能高效运行
- 能满足服务性能要求
- 能在多用户并发环境稳定工作

---

### 3.2 推理适配

目标是把训练产物转成目标推理栈可以执行的形式。

典型活动包括：

- 模型格式转换
- 算子支持检查
- 不支持算子的替换或实现
- 权重转换
- Precision / Quantization
- Layout 转换
- 图优化
- Kernel 选择或生成
- 并行策略规划
- 多 Device / 多 Node 适配

这里首先要回答的是：

> **这个模型能否在目标软硬件栈上正确执行？**

随后才是：

> **能否以满足业务约束的方式高效执行？**

---

### 3.3 推理准备与执行

通常会涉及：

```text
Model
↓
Graph / Engine Representation
↓
Operator Lowering
↓
Kernel
↓
Runtime Execution
↓
Device
```

具体系统边界因框架而异。一些 Engine 会同时承担图优化、Kernel 选择、Runtime、Scheduler 等能力，因此不要把上述层级机械理解成必须对应五个独立软件产品。

---

### 3.4 服务化

当模型从“可以运行”变成“可以被业务调用”，系统需要增加：

- API / Endpoint
- Request parsing
- Authentication / protocol
- Queue
- Batching
- Scheduler
- KV Cache management
- Routing
- Replica / Worker management
- Load balancing
- Failure handling
- Streaming response
- Metrics / logs

这一步把问题从“模型执行”升级成了“有限资源上的并发服务系统”。

---

## 4. 一次 LLM 推理内部发生什么

### 4.1 自回归生成

典型 Decoder-only LLM 的生成过程：

```text
Prompt
  ↓
Tokenize
  ↓
Prefill
  ↓
KV Cache
  ↓
Logits
  ↓
Sampling
  ↓
First Output Token
  ↓
Decode
  ↓
Sampling
  ↓
Next Token
  ↓
Decode...
  ↓
EOS / Stop Condition
```

模型每生成一个新 token，都把它加入上下文，然后继续预测下一个 token。

---

### 4.2 Prefill

Prefill 接收完整输入 Prompt。

主要工作：

- 一次处理整个输入序列
- 计算每层 Transformer
- 产生第一轮输出
- 建立后续 Decode 所需的 KV Cache

Prefill 往往包含较大的矩阵计算。

由于一次处理很多 token，矩阵规模较大、数据复用较充分，硬件计算单元更容易被充分利用，因此常被概括为：

> **Prefill 更偏 compute-bound。**

但这只是典型特征，不是任何模型、任何硬件、任何输入长度下都绝对成立。

Prefill 对用户体验最直接的影响通常体现在 **TTFT**。

---

### 4.3 Decode

Decode 从 Prefill 已经建立的上下文状态出发，逐 token 生成。

每一步通常：

```text
new token
  ↓
model forward
  ↓
read existing KV Cache
  ↓
append new K / V
  ↓
produce logits
  ↓
sample next token
```

和 Prefill 相比，一次 Decode step 处理的 token 很少，矩阵计算更小，但需要反复读取模型权重和不断增长的 KV Cache。

因此常见特征是：

> **Decode 更容易受到显存带宽、KV Cache 访问和内存系统限制。**

Decode 对用户持续阅读体验最直接的影响通常体现在 **TPOT / ITL**。

---

### 4.4 Prefill 与 Decode 的关键区别

| 维度 | Prefill | Decode |
|---|---|---|
| 输入 | 整段 Prompt | 通常每步新增 token |
| 计算形态 | 大矩阵计算更多 | 小步重复计算 |
| 典型瓶颈 | Compute | Memory / bandwidth |
| KV Cache | 创建 | 读取并追加 |
| 主要业务变量 | Input Length | Concurrency、Output Length |
| 用户体验关联 | TTFT | TPOT / ITL |
| 调度风险 | 长 Prompt 阻塞其他请求 | 大量活跃 request 竞争内存与带宽 |

理解这两个阶段的不同，是理解后续 batching、scheduler、KV Cache、PD Disaggregation 和容量规划的基础。

---

### 4.5 Sampling

模型输出的是 logits，而不是直接输出最终文本。

Sampling 会根据 logits 得到下一个 token，相关参数可能包括：

- Temperature
- Top-k
- Top-p
- Greedy decoding
- Beam search
- repetition penalty
- stop condition

因此“推理速度”除了模型 forward，也可能受 decoding strategy 和后处理影响。

---

## 5. KV Cache：LLM Serving 的核心运行状态

### 5.1 为什么需要 KV Cache

Attention 在生成第 N 个 token 时，需要访问之前 token 的 Key 和 Value。

如果每一步都重新计算所有历史 token 的 K/V，会产生大量重复计算。

KV Cache 保存已经算过的 K/V：

```text
Prefill
  ↓
生成历史 token 的 K/V
  ↓
写入 KV Cache

Decode step N
  ↓
读取已有 KV
  ↓
只计算新 token 对应的 K/V
  ↓
追加到 KV Cache
```

KV Cache 用空间换时间，是现代 LLM 推理最核心的优化机制之一。

---

### 5.2 KV Cache 生命周期

```text
Request admitted
      ↓
KV space allocated
      ↓
Prefill writes KV
      ↓
Decode repeatedly reads/appends KV
      ↓
Request finishes
      ↓
KV released / reused
```

如果存在 prefix cache，则部分 KV 生命周期可能跨越单个请求。

---

### 5.3 为什么 KV Cache 会成为系统瓶颈

KV Cache 大小受到多种因素影响：

```text
Model architecture
× Number of layers
× KV heads
× Head dimension
× Sequence length
× Active requests
× KV precision
```

因此：

```text
Context Length ↑
      ↓
KV per request ↑
      ↓
Memory pressure ↑
      ↓
Maximum concurrency ↓
      ↓
Throughput / queue behavior changes
```

长上下文、高并发场景下，KV Cache 常常比“模型权重是否装得下”更早成为容量限制。

---

### 5.4 常见 KV Cache 优化方向

#### Paged / Block-based KV Cache

将 KV 空间按 block/page 管理，而不是给每个请求预留完整最大长度。

价值：

- 减少浪费
- 动态分配
- 提高并发容量
- 更容易回收和复用

#### Prefix Caching

多个请求存在相同前缀时，复用已经计算的 KV。

适合：

- 共享 System Prompt
- 多轮对话
- Agent workflow 中重复上下文

#### KV Quantization

降低 KV dtype，减少显存和带宽压力。

Trade-off：

```text
Memory ↓
Bandwidth pressure ↓
Potential capacity ↑
vs.
Numerical quality risk
```

#### Offloading

把部分 KV 放到：

- Host memory
- 其他 GPU/NPU
- 分层存储

代价是传输延迟与管理复杂度。

#### Eviction / Preemption / Recompute

当 KV 容量不足时，系统可能：

- 淘汰缓存
- 抢占请求
- 重算部分上下文

这些行为可能直接表现为 TTFT / TPOT / throughput 波动。

---

## 6. 从单请求变成 Serving：Queue、Batch 与 Scheduler

### 6.1 一个在线请求的生命周期

可以抽象成：

```text
Arrival
  ↓
Queue
  ↓
Admission
  ↓
Prefill
  ↓
Decode Loop
  ↓
Finish
  ↓
Release resources
```

真实系统中，很多请求同时处在不同阶段：

```text
R1: Decode
R2: Decode
R3: Prefill
R4: Waiting
R5: Decode
R6: Waiting for KV capacity
```

Scheduler 的任务，就是决定有限资源下一刻给谁。

---

### 6.2 Static Batching

传统 batching 会等一组请求凑成 batch，然后一起运行。

问题在于 LLM 请求长度差异很大：

- Prompt 不同
- Output Length 不同
- 请求结束时间不同

固定 batch 中，如果必须等待最长 request，资源会被浪费。

---

### 6.3 Continuous Batching / In-flight Batching

LLM Serving 通常采用 token-step 或 iteration-level 的动态 batching：

```text
Step 1:
R1 R2 R3 R4

Step 2:
R1 R2 R4 + R5 joins

Step 3:
R1 R4 R5 + R6 joins
```

已经完成的 request 随时退出，新请求可以进入。

价值：

- 提高硬件利用率
- 减少因请求长度不一致产生的空闲
- 提升吞吐

代价：

- Scheduler 更复杂
- 内存管理更复杂
- 大 batch 可能影响 latency
- Prefill 和 Decode 之间可能互相干扰

---

### 6.4 Scheduler 本质上在解决什么

Scheduler 不是简单决定“谁先谁后”，而是在进行动态资源分配。

典型决策包括：

- 新 request 是否允许进入
- 哪些 request 组成当前 batch
- Prefill 还是 Decode 优先
- 每轮处理多少 token
- KV 空间不足时抢占谁
- 是否 chunk prefill
- 不同优先级请求如何公平调度

它需要同时平衡：

```text
Latency
Throughput
Fairness
Memory
KV capacity
Hardware utilization
SLO
```

所以 scheduler policy 本身也是推理优化的重要组成部分。

---

## 7. Inference Engine、Runtime、Scheduler 与 Serving 的边界

这些概念在现实产品中经常重叠，但理解逻辑边界非常重要。

### 7.1 Compiler

主要处理模型如何被转换为可高效执行的计算：

- Graph optimization
- Operator fusion
- Layout transformation
- Operator lowering
- Kernel generation / selection
- Memory planning

---

### 7.2 Inference Engine

负责把模型以适合推理的方式执行。

常见能力：

- 模型加载
- 高性能 operator / kernel
- KV Cache
- Batch execution
- Sampling
- Quantization execution
- Parallel execution
- 有些 Engine 同时包含 Scheduler

例如 vLLM、SGLang、TensorRT-LLM、MindIE LLM 都覆盖了部分 engine/runtime/serving 能力，因此不要把“Engine”理解成严格统一的软件层。

---

### 7.3 Runtime

更靠近真实执行：

- Device memory
- Kernel dispatch
- Stream
- Synchronization
- Host / Device coordination
- Communication
- Resource management

Runtime 回答的是：

> **已经决定要执行的计算，实际上怎样在硬件上发生？**

---

### 7.4 Scheduler

Scheduler 决定：

> **有限资源下一步执行哪些请求或计算。**

它连接 workload 与 runtime。

---

### 7.5 Serving System

Serving 面向在线业务：

- API / Endpoint
- Request handling
- Routing
- Queue
- Load balancing
- Workers / replicas
- Autoscaling
- Fault tolerance
- Metrics
- Deployment

Serving 关注的是：

> **如何让模型成为一个稳定、高性能、可扩展的在线服务。**

---

### 7.6 一个可用的逻辑层级

```text
Application / Client
        ↓
Serving / Distributed Inference
        ↓
Scheduler / Routing
        ↓
Inference Engine
        ↓
Runtime
        ↓
Compiler-generated execution / kernels
        ↓
Hardware
```

现实系统里这些边界可以融合，但分析问题时仍然有价值。

---

## 8. 分布式推理与并行策略

当模型或 workload 超出单 Device 能力时，需要扩展到多 Device / 多 Node。

### 8.1 Data Parallelism（DP）

核心：

> 把不同请求或 batch 分到多个模型副本。

可简单理解为：

```text
GPU 0: Request A
GPU 1: Request B
GPU 2: Request C
GPU 3: Request D
```

典型目标：

- 提高服务吞吐
- 提高并发容量

代价：

- 每个副本通常都需要模型权重
- 负载均衡成为问题

---

### 8.2 Tensor Parallelism（TP）

核心：

> 把一次模型计算中的 Tensor / Matrix 拆到多个 Device。

例如一个大型矩阵乘：

```text
one layer
↓
split across GPU0 + GPU1 + GPU2 + GPU3
```

价值：

- 单 Device 放不下模型时可扩展
- 聚合多 Device 算力

代价：

- 高频通信
- AllReduce / AllGather
- 通信与计算同步
- TP 越大不一定越快

---

### 8.3 Pipeline Parallelism（PP）

核心：

> 按模型层拆分到多个 stage。

```text
Layers 1–20  → Device Group A
Layers 21–40 → Device Group B
Layers 41–60 → Device Group C
```

主要问题：

- Pipeline bubble
- Stage imbalance
- 中间激活传输
- latency 增加

---

### 8.4 Expert Parallelism（EP）

MoE 模型中：

> 不同 Expert 分布到不同 Device。

请求 token 会经过 Router 被发送到不同 Expert。

主要问题：

- Expert load imbalance
- All-to-All communication
- Hot expert
- 跨节点通信

---

### 8.5 一个简单记忆方式

```text
DP：拆请求
TP：拆一次计算
PP：拆模型层
EP：拆 Expert
```

实际大型模型常使用 Hybrid Parallelism：

```text
TP × DP × EP
```

甚至叠加 PP / CP 等。

扩展 Device 数量时，需要同时考虑：

```text
More compute capacity
        vs.
More communication
        vs.
More synchronization
```

因此“更多卡”并不自动意味着更低 latency 或更高 efficiency。

---

## 9. Prefill / Decode Disaggregation（PD 分离）

### 9.1 为什么会出现 PD 分离

Prefill 与 Decode 的资源特征不同：

```text
Prefill
→ prompt length sensitive
→ often compute-heavy

Decode
→ active request sensitive
→ KV / memory-bandwidth heavy
```

在统一 Worker 中，两者共享资源。

长 Prefill 可能阻塞正在 Decode 的请求，造成 token generation 抖动。

PD Disaggregation 将它们放到不同 Worker pool：

```text
Request
  ↓
Prefill Worker
  ↓
KV Cache Transfer
  ↓
Decode Worker
  ↓
Response
```

---

### 9.2 它解决什么问题

- P 与 D 可以独立扩缩容
- 可以采用不同并行策略
- 减少长 Prefill 对 Decode 的干扰
- 根据 workload 分别分配不同硬件资源
- 更容易控制 TTFT 与 TPOT 的 trade-off

---

### 9.3 它新增什么问题

最重要的是：

> **KV Cache 必须从 Prefill 侧到达 Decode 侧。**

于是增加：

- KV transfer latency
- Network bandwidth
- RDMA / interconnect
- KV layout compatibility
- Routing
- Worker pairing
- Transfer failure
- More complex scheduling
- More operational failure domains

因此：

```text
PD separation
≠
always faster
```

短 Prompt、低并发、小模型、慢网络、KV transfer 成本高的场景，聚合部署可能更简单甚至更快。

---

### 9.4 PD 分离背后的产品判断

用户真正做的不是“打开 PD”。

而是在判断：

```text
当前 workload 的 Prefill / Decode 压力是否显著不同？
      ↓
统一资源池是否产生干扰？
      ↓
独立扩容带来的收益是否大于 KV transfer 成本？
```

这是一类典型的跨层优化决策。

---

## 10. 推理指标：不要把所有数字混在一起

### 10.1 用户感知延迟

#### TTFT — Time To First Token

从请求发出到收到第一个输出 token 的时间。

通常包含：

```text
network
+ queue
+ scheduling
+ prefill
+ first token generation
```

因此：

> TTFT 高不等于 Prefill Kernel 一定慢。

---

#### TPOT — Time Per Output Token

一般用于描述首 token 之后生成 token 的平均速度。

常见近似：

```text
TPOT ≈ (End-to-End Latency - TTFT) / (Output Tokens - 1)
```

不同工具的具体定义可能略有差异，比较前必须确认 metric definition。

---

#### ITL — Inter Token Latency

相邻输出 token 之间的时间。

它更适合描述流式生成时的“连续输出是否顺滑”。

TPOT 通常是一个 request 级平均值；ITL 可以进一步观察 token 间抖动。

---

#### End-to-End Latency

从 request 开始到完整 response 结束。

它受到：

```text
TTFT
+ output length
+ decode rate
+ network / streaming
```

共同影响。

---

#### Tail Latency

例如：

- P95
- P99

在线服务不能只看平均值，因为真实用户可能主要被 long-tail 问题影响。

---

### 10.2 吞吐

常见两个维度：

#### Requests per second

适合请求大小相对可比的场景。

#### Tokens per second

对 LLM 更常用，因为不同请求 token 数差异很大。

还需要区分：

- Input token throughput
- Output token throughput
- Total token throughput

---

### 10.3 GPU / NPU 利用率不是 Throughput

高利用率只说明硬件忙。

可能存在：

```text
Utilization high
but
useful throughput low
```

例如：

- 大量低效 kernel
- 通信等待
- 重算
- memory stall
- 不合理工作负载

反过来，低 utilization 也不一定直接意味着系统应该“把 batch 拉满”，因为可能会破坏 latency SLO。

---

### 10.4 内存指标

重要指标包括：

- Model weight memory
- KV Cache usage
- KV block utilization
- HBM usage
- Host offload
- Cache hit rate
- Eviction
- Preemption
- Recompute

---

### 10.5 服务与调度指标

包括：

- Queue length
- Queue time
- Active requests
- Waiting requests
- Batch size
- Prefill batch
- Decode batch
- Request rate
- Success / error rate
- Cancellation
- Worker load
- Replica utilization

---

### 10.6 成本指标

可用：

- Cost / request
- Cost / output token
- Cost / 1M tokens
- Required accelerator count
- Energy / compute cost
- Capacity per accelerator

最终业务通常希望的是：

> **在满足质量和 SLO 的前提下，以尽可能少的资源服务尽可能多的有效请求。**

---

## 11. 推理本质：受约束的多目标优化

### 11.1 为什么不能只追求 Throughput

Batch size 增大：

```text
Batch ↑
→ Hardware utilization ↑
→ Throughput ↑
```

但同时可能：

```text
Queue / scheduling delay ↑
→ TTFT ↑
```

因此最大吞吐不一定是业务最优点。

---

### 11.2 典型 Trade-off

#### Throughput vs Latency

更 aggressive 的 batching 往往有利吞吐，但可能增加等待。

#### Memory vs Concurrency

单请求使用更多 KV：

```text
Memory per request ↑
→ Maximum concurrent requests ↓
```

#### Accuracy vs Performance

Quantization：

```text
precision ↓
→ model / KV memory ↓
→ bandwidth pressure ↓
→ performance potential ↑
→ accuracy risk ↑
```

#### Compute vs Communication

并行规模扩大：

```text
TP ↑
→ compute distributed
→ communication ↑
```

#### Cache Reuse vs Memory

保留更多 prefix cache：

```text
Cache hit opportunity ↑
vs.
Available KV capacity ↓
```

---

### 11.3 Pareto Frontier

对一个场景，可以存在很多无法互相绝对支配的方案：

```text
Config A: latency best
Config B: throughput best
Config C: cost best
Config D: balanced
```

如果 A 的 throughput 更低，但 latency 更好，就不能简单说 B“更优”。

因此推理优化通常是在约束条件下寻找 Pareto-optimal candidates，再依据业务目标做决策。

---

### 11.4 Goal 与 Constraint

例如一个 Chat 场景可能定义：

```text
Goal:
  maximize output throughput

Constraints:
  P99 TTFT < 800 ms
  P99 TPOT < 50 ms
  quality loss < 1%
  HBM usage < capacity
```

这种表达比“我要优化性能”更有意义。

---

## 12. Workload 决定最优策略

同一个模型，不同 workload 可能需要完全不同的部署。

### Interactive Chat

典型特征：

- 对 TTFT 敏感
- 对 TPOT 敏感
- Streaming
- 多轮对话
- Prefix reuse 可能高

### Long-context / RAG

典型特征：

- Input Length 大
- Prefill 压力高
- KV 大
- TTFT 成为主要压力

### Long-form Generation

典型特征：

- Output Length 大
- Decode 长
- KV 长时间占用
- Decode capacity 成为关键

### Offline Batch

典型特征：

- 可以牺牲单请求 latency
- 追求最大 throughput / cost efficiency
- 更 aggressive batching

### Agent / Multi-turn Workload

典型特征：

- 一个任务产生多轮模型调用
- 上下文不断增长
- Shared prefix / KV reuse 价值提高
- 外部 tool latency 与模型推理交织
- 需要从单请求性能升级到 workflow-level latency

因此：

> **模型不是 Scenario 的全部；Workload 才决定系统在真实业务中的资源压力。**

---

## 13. 推理优化的层级

推理优化可以跨越多个层级。真正的性能问题往往不是单一层可以解释。

### 13.1 Model / Algorithm

包括：

- Model architecture
- Distillation
- Pruning
- Sparse model
- Attention algorithm
- Speculative decoding
- MoE routing optimization

改变这一层可能直接改变计算量和模型质量。

---

### 13.2 Numerical Representation

包括：

- FP16
- BF16
- FP8
- INT8
- INT4
- Weight-only quantization
- KV Cache quantization

优化目标：

- 减少 memory
- 减少 bandwidth
- 提高特定硬件吞吐

约束：

- Accuracy / quality

---

### 13.3 Compiler / Graph

包括：

- Graph optimization
- Operator fusion
- Constant folding
- Layout optimization
- Memory planning
- Kernel selection
- Lowering

---

### 13.4 Operator / Kernel

包括：

- Tiling
- Tensor layout
- Memory access
- Vectorization
- Tensor Core / AI Core utilization
- Pipeline
- Kernel fusion
- Custom kernel

这是 Nsight Compute 等 kernel profiler 更关注的层级。

---

### 13.5 Memory / KV Cache

包括：

- Paged KV
- Prefix cache
- KV quantization
- Offload
- Cache eviction
- Cache reuse
- Block sizing
- Fragmentation reduction

---

### 13.6 Parallelism / Communication

包括：

- TP / DP / PP / EP
- AllReduce
- AllGather
- All-to-All
- Communication / computation overlap
- Topology-aware placement

---

### 13.7 Runtime / Scheduler

包括：

- Continuous batching
- Chunked prefill
- Scheduling policy
- Request preemption
- Batch size
- Memory allocation
- Runtime dispatch
- CUDA/NPU graphs

---

### 13.8 Serving / Cluster

包括：

- Routing
- Replica count
- Load balancing
- KV-aware routing
- PD disaggregation
- Autoscaling
- Worker placement
- Fault handling

---

### 13.9 为什么要分层

一个现象可能在高层出现，却由低层造成。

例如：

```text
P99 TPOT ↑
↓
某些 Decode batch 变慢
↓
MoE communication ↑
↓
Expert load imbalance
↓
EP placement / routing problem
```

也可能：

```text
P99 TTFT ↑
↓
queue time ↑
↓
prefill workers saturated
↓
traffic ISL distribution changed
```

所以优化前必须先定位瓶颈所在层级。

---

## 14. Correctness 与整网精度

性能不是推理唯一目标。

### 14.1 四个不同层面的“成功”

```text
Compile success
≠
Runtime success
≠
Numerical correctness
≠
Model quality acceptable
```

一个模型可以成功编译、成功执行，但输出数值已经错误。

也可以 numerical error 很小，但在整网累计后影响最终质量。

---

### 14.2 常见正确性问题

- NaN / Inf
- Wrong tensor value
- Wrong shape
- Wrong stride / layout
- Quantization deviation
- Operator implementation mismatch
- Different backend produces inconsistent outputs
- Nondeterminism
- Sampling / postprocess difference

---

### 14.3 算子精度与整网精度

单算子误差不代表模型最终一定失败。

但多个 layer 的误差可能：

```text
small local error
↓
propagation
↓
accumulation
↓
model output degradation
```

整网精度诊断通常需要逐层或逐 Tensor 比对，将问题逐渐缩小到：

```text
Model output
↓
Layer
↓
Operator
↓
Tensor
↓
Kernel / numerical implementation
```

---

### 14.4 Quantization 的质量判断

Quantization 不是只看“模型能不能跑”。

需要同时验证：

- Tensor / operator error
- End-to-end model quality
- Task-specific evaluation
- Performance gain
- Memory gain

因此它仍然属于 multi-objective optimization。

---

## 15. Benchmark、Monitoring、Tracing、Profiling 分别回答什么

这些词经常混用，但它们服务不同问题。

### 15.1 Benchmark

回答：

> **在某个明确 workload 下，这个系统整体表现怎么样？**

输入：

- Model
- Workload
- Hardware
- Configuration

输出：

- TTFT
- TPOT / ITL
- E2E latency
- Throughput
- Success rate
- 资源指标

Benchmark 适合：

- 建立 baseline
- 比较配置
- 容量测试
- 验证优化

---

### 15.2 Monitoring

回答：

> **线上系统现在是否健康？**

通常长期运行、低开销。

关注：

- Traffic
- Queue
- Latency
- Throughput
- Error
- Resource
- Worker health
- SLO

Monitoring 主要用于发现异常，而不是直接解释所有 root cause。

---

### 15.3 Trace

回答：

> **某个请求或某段执行具体经历了什么？**

它强调事件之间的时间和因果关系：

```text
request
→ queue
→ router
→ worker
→ prefill
→ KV transfer
→ decode
→ response
```

也可以进一步下钻到：

```text
CPU
→ runtime API
→ GPU/NPU kernel
```

---

### 15.4 Profiling

回答：

> **时间和硬件资源具体花在哪里？**

Profiling 可以分层：

#### Request / serving profiling
看 request stage。

#### Runtime / system profiling
看 thread、runtime、kernel launch、communication、timeline。

#### Kernel profiling
看单个 kernel 内：

- compute utilization
- memory
- occupancy
- stalls
- instruction / pipeline
- hardware counters

---

### 15.5 Logging

回答：

> **系统发生了哪些离散事件或错误？**

适合：

- error
- warning
- request state
- configuration
- crash context

---

### 15.6 Evaluation

回答：

> **模型输出质量是否满足业务要求？**

它和 performance benchmark 是不同维度。

---

### 15.7 Replay

将真实或受控 workload 保存下来，再重放。

作用：

- 重现问题
- Regression testing
- Before / after comparison
- Crash diagnosis
- 真实流量近似 benchmark

---

## 16. Evidence：从现象到根因需要怎样的证据

对推理诊断来说，一个核心原则是：

> **Metric ≠ Diagnosis ≠ Root Cause ≠ Proof。**

### 16.1 证据层级

```text
Business / Service Metric
          ↓
Request-level Metric
          ↓
Request Trace
          ↓
Runtime Event
          ↓
Operator / Kernel Profile
          ↓
Hardware Counter
```

越往下：

- 粒度更细
- 解释能力更强
- 数据量更大
- 采集成本通常也更高

---

### 16.2 一个完整诊断例子

现象：

```text
P99 TTFT = 1.8s
```

这只是 symptom。

继续看：

```text
queue = 1.2s
prefill = 0.4s
other = 0.2s
```

可以判断问题首先不是 Prefill kernel。

再看：

```text
waiting requests ↑
prefill workers saturated
decode workers underutilized
```

形成 hypothesis：

> Prefill capacity 不足或 P/D 资源比例不适合当前 workload。

修改资源比例后重新跑同一 workload：

```text
P99 TTFT:
1.8s → 0.7s
```

同时 throughput / TPOT 没有恶化，才形成更完整的验证证据。

---

### 16.3 “看到相关性”不等于找到因果

例如：

```text
GPU utilization low
```

可能原因非常多：

- Queue 没有足够请求
- Batch 太小
- CPU feeding slow
- Kernel launch gap
- Communication
- Dependency
- Memory stall
- Load imbalance

所以好的诊断过程是：

```text
Symptom
↓
Relevant signal
↓
Narrow component
↓
Execution evidence
↓
Hypothesis
↓
Controlled change
↓
Validation
```

---

## 17. Instrumentation Overhead：观测不是免费的

系统采集越深入，对被测系统干扰通常越大。

可能产生：

- CPU overhead
- Device overhead
- Synchronization
- I/O
- Memory
- Trace buffer
- Storage
- Timing perturbation
- Kernel replay

因此不能认为：

> “把所有 trace、profile、counter 永久打开”就是最好的 observability。

可以建立一个简单层级：

```text
Production Monitoring
  ↓ low overhead

Selective Request Tracing
  ↓

Targeted Runtime Trace
  ↓

Deep Profiling
  ↓

Kernel Counter / Replay
  ↓ high overhead
```

这也是为什么 Monitoring、Tracing 和 Profiling 通常被设计成不同采集模式。

---

## 18. Controlled Inference 与 Online Production

推理分析存在两种根本不同的运行环境。

### 18.1 Controlled Environment

例如：

- Benchmark
- Profiling
- Evaluation
- Replay
- Optimization experiment

特点：

- Workload 可控
- 环境可控
- 可以 warm-up
- 可以重复
- 可以固定随机因素
- 可以开启较高 overhead 的 instrumentation

目标是：

> **理解系统、比较方案、验证因果。**

---

### 18.2 Online Production

特点：

- Traffic 不稳定
- Workload distribution 变化
- 多租户
- 有真实用户
- 对 instrumentation overhead 敏感
- 故障和性能问题可能只在特定流量条件出现

目标是：

> **持续知道系统是否健康，并在必要时捕获足够证据。**

---

### 18.3 二者如何连接

典型闭环：

```text
Online anomaly
↓
Identify affected workload / request cohort
↓
Capture evidence
↓
Reproduce with controlled workload
↓
Deep profile
↓
Change
↓
Benchmark
↓
Deploy
↓
Observe production again
```

这比直接在线上开启重型 profiler 更可靠。

---

## 19. Benchmark 与实验设计原则

性能比较必须先保证“可比较”。

### 19.1 需要控制的变量

至少包括：

- Model version
- Precision
- Engine version
- Runtime version
- Hardware
- Number of devices
- Parallel strategy
- Input Length
- Output Length
- Request Rate
- Concurrency
- Dataset
- Sampling params
- Streaming
- Prefix cache state
- Warm-up
- Network path
- Software configuration

---

### 19.2 一个常见错误

```text
System A: 8000 token/s
System B: 6000 token/s
```

不能直接推出：

```text
A faster than B
```

除非：

- 相同模型
- 相同 token workload
- 相同硬件
- 相同 quality
- 相同 latency constraint
- 相同 metric definition

---

### 19.3 Saturation Curve 比单点数字更有意义

服务系统常见：

```text
Request Rate ↑
↓
Throughput ↑
↓
approach saturation
↓
Queue grows rapidly
↓
P99 latency explodes
```

因此评估服务时，更有意义的是看：

```text
Load
vs.
Throughput
vs.
Latency
```

而不是只看一个最高 throughput。

---

### 19.4 Baseline

任何优化最好从 baseline 开始。

Baseline 的作用：

- 提供对照
- 避免“优化后感觉更快”
- 判断 regression
- 确认优化收益来自哪个变量

---

## 20. 典型 Failure Modes 与问题归因

用户通常从“现象”进入，而不是从技术组件进入。

### 20.1 模型无法运行

可能来自：

- Unsupported operator
- Compile failure
- Wrong model format
- Runtime incompatibility
- Out of memory
- Device mismatch
- Parallel configuration invalid

---

### 20.2 结果错误

可能来自：

- Operator correctness
- Quantization
- Numerical overflow
- Layout / shape
- Backend implementation
- Sampling / postprocess
- Distributed communication error

---

### 20.3 TTFT 高

可能来自：

```text
Network
Queue
Scheduling
Long prompt
Prefill compute
Prefix cache miss
Worker saturation
Communication
Cold start
```

---

### 20.4 TPOT / ITL 高

可能来自：

```text
Decode kernel
KV bandwidth
KV capacity pressure
Batch composition
Communication
MoE routing
Scheduler interference
Memory contention
```

---

### 20.5 Throughput 低

可能来自：

- Workload 不够
- Batch 太小
- Hardware underutilization
- Kernel inefficient
- Communication overhead
- Scheduler overhead
- Poor load balancing
- Resource fragmentation

---

### 20.6 P99 抖动

可能来自：

- Traffic burst
- Long prompt
- Heavy prefill blocks decode
- GC / runtime behavior
- Network
- Load imbalance
- Preemption / recompute
- Noisy neighbor
- Worker degradation

---

### 20.7 OOM / KV 不足

可能来自：

- Context Length
- Concurrency
- KV dtype
- Fragmentation
- Cache retention
- Model memory
- Parallel strategy
- Incorrect capacity configuration

---

### 20.8 PD 分离后更慢

可能来自：

- KV transfer slow
- P/D ratio wrong
- Routing ineffective
- Network fallback
- Extra coordination cost
- Workload 不适合 disaggregation

---

### 20.9 性能 Regression

Regression 分析不能只比较两次总耗时。

需要确认：

```text
Model
Workload
Hardware
Engine
Kernel
Runtime
Config
Traffic
```

到底哪个发生了变化。

---

## 21. Capacity Planning：性能最终要转换成“能服务多少业务”

Serving 不只是优化单 request。

一个更业务化的问题是：

> **在给定 SLO 下，这套系统能支撑多少 traffic，需要多少硬件？**

---

### 21.1 基本因果关系

```text
Traffic
× Input Length
× Output Length
× Concurrency
        ↓
Prefill + Decode Demand
        ↓
Compute / Memory / KV / Network Demand
        ↓
Worker / Device Capacity
        ↓
Replica Count
        ↓
Cost
```

---

### 21.2 Saturation

系统存在最大稳定服务区间。

接近 saturation 后：

```text
small traffic increase
↓
queue grows
↓
tail latency grows sharply
```

所以容量规划通常需要 headroom，而不是让 accelerator 永远跑到理论 100%。

---

### 21.3 Autoscaling

Autoscaling 需要决定：

- 用什么指标触发
- 扩什么资源
- 扩容速度
- 模型加载时间
- Prefill / Decode 是否独立 scaling
- 高峰过后何时 scale down

如果模型启动时间长，传统 Web service 的 autoscaling 逻辑不一定直接适用。

---

### 21.4 Cost

最终优化可能表现为：

```text
Same SLO
+ Same quality
+ fewer accelerators
```

它可能比纯粹的单卡 benchmark improvement 更有业务价值。

---

## 22. Production Reliability 与运行风险

分布式推理增加很多 failure domains：

```text
Client
↓
Gateway
↓
Router
↓
Scheduler
↓
Worker
↓
Runtime
↓
Device
↓
Network
↓
KV Transfer
```

典型生产问题：

- Worker crash
- Device fault
- OOM
- Model load failure
- Request timeout
- Network failure
- KV transfer failure
- Replica imbalance
- Traffic spike
- Partial degradation
- Failed rollout

因此 production inference 的目标并不只有性能，还包括：

```text
Reliability
Availability
Recoverability
Observability
Cost predictability
```

---

## 23. 推理中的用户角色、任务与决策

推理系统通常跨多个角色协作。

### 23.1 Model / Algorithm Engineer

关注：

- 模型质量
- 模型结构
- Quantization
- 模型兼容性
- Output correctness

典型决策：

- 是否接受某种量化
- 是否修改模型结构
- 是否换推理友好的模型版本

---

### 23.2 Inference Engineer

关注：

- Engine
- KV Cache
- Scheduler
- Parallelism
- Performance
- Serving configuration

典型决策：

- TP / DP 如何配置
- Batch 如何设
- KV 策略
- 是否采用 speculative decoding
- 是否采用 PD

---

### 23.3 Compiler / Kernel Engineer

关注：

- Operator
- IR
- Kernel
- Memory access
- Hardware counter
- Kernel efficiency

典型决策：

- 是否需要 custom kernel
- Tiling / layout
- fusion
- compute / memory bottleneck

---

### 23.4 Serving / Platform Engineer

关注：

- Request routing
- Workers
- Replicas
- Cluster
- Traffic
- Capacity
- Reliability

典型决策：

- replica count
- load balancing
- autoscaling
- P/D worker ratio
- placement

---

### 23.5 Performance Engineer

跨层工作：

```text
Benchmark
→ Observe
→ Locate bottleneck
→ Profile
→ Form hypothesis
→ Experiment
→ Compare
```

通常需要在多个工具之间切换。

---

### 23.6 SRE / Operations

关注：

- SLO
- Incidents
- Error
- Tail latency
- Capacity
- Service health
- Rollout / rollback

---

### 23.7 Application Developer

更多关注服务接口：

- API compatibility
- latency
- streaming
- reliability
- quota
- cost

---

### 23.8 对 UX/Product 最重要的判断

同一个指标，对不同角色意味着不同的下一步行动。

例如：

```text
TTFT ↑
```

对 SRE：

> 服务是不是已经违反 SLO？

对 Serving Engineer：

> Queue / Scheduler / Worker capacity 哪个出了问题？

对 Inference Engineer：

> Prefill 是否需要优化？

对 Kernel Engineer：

> Prefill kernel 是否出现 regression？

所以底层工具体验的核心，不是“展示更多数据”，而是帮助角色将证据转成决策。

---

## 24. 推理优化的业务闭环

这是推理工作的自然过程，不代表任何具体产品导航。

```text
Define Scenario
        ↓
Define Goal & Constraints
        ↓
Establish Baseline
        ↓
Run
        ↓
Observe
        ↓
Diagnose
        ↓
Form Hypothesis
        ↓
Change one or limited variables
        ↓
Benchmark / Validate
        ↓
Compare
        ↓
Decision
        ↓
Deploy
        ↓
Observe Again
```

几个重要原则：

1. **Observe before Optimize**：不知道瓶颈在哪里就直接改参数，容易变成盲目调优。
2. **Baseline first**：没有 baseline 就无法证明 improvement。
3. **Context must stay comparable**：Workload 或硬件变了，结果可能失去可比性。
4. **Optimize under constraints**：最大 throughput 不是唯一目标。
5. **Validate causality**：一个变化与性能改善同时发生，不等于它一定是原因。
6. **Production closes the loop**：Controlled benchmark 最终仍需回到真实流量验证。

---

# 25. 竞品与推理工具生态

## 25.1 先不要把这些工具都当成“同类竞品”

推理领域的工具覆盖不同层级：

```text
Application
    │
    ▼
Distributed Serving / Cluster
    │
    ▼
Inference Engine / Runtime
    │
    ▼
Kernel / Hardware
```

横向还有：

```text
Benchmark
Observability
Tracing
System Profiling
Kernel Profiling
Operations
```

因此：

> NVIDIA Dynamo、vLLM、Nsight Compute、AIPerf 不是四个直接竞争的同类产品。

更有价值的分析方法是：

- 服务谁
- 解决哪个任务
- 处在哪个抽象层
- 用户从什么触点使用
- 暴露哪些对象
- 给出原始数据还是解释
- 是运行系统还是分析系统
- 用户需要具备多少底层知识

---

## 25.2 NVIDIA Dynamo

### 定位

Dynamo 是面向生成式 AI 的开源分布式推理框架，重点处在 **distributed inference / serving orchestration** 层。

它可以和 vLLM、SGLang、TensorRT-LLM 等推理后端组合，而不是取代所有 inference engine。

### 主要关注对象

- Request
- Frontend
- Router
- Worker
- Prefill Worker
- Decode Worker
- KV Cache
- KV Transfer
- Replica
- Distributed deployment

### 核心能力

截至 2026-08 官方文档重点包括：

- KV Cache-aware routing
- Disaggregated Serving
- KV Cache Offloading
- Autoscaling
- Fault tolerance
- Observability
- Benchmarking
- 多种 inference backend 集成

### 典型触点

- CLI
- Python
- Kubernetes deployment / CRD
- Configuration
- OpenAI-compatible serving endpoint
- Metrics
- Benchmark tooling

### 值得 UX/Product 理解的地方

Dynamo 把优化对象从“单个 Engine”提升成：

> **整个 distributed inference system。**

例如 KV-aware routing 并不是优化 kernel，而是在请求到来时，结合：

- 哪个 worker 已经持有可复用 KV
- 哪个 worker 当前 decode load 较高

做 request placement。

PD Disaggregation 则把 Prefill / Decode 独立 scaling。

因此它代表了推理优化从 **execution optimization** 向 **system orchestration optimization** 扩展。

---

## 25.3 TensorRT-LLM

### 定位

TensorRT-LLM 是 NVIDIA 面向 LLM 的高性能 inference stack，更靠近 **模型执行、runtime 与性能优化**。

### 核心对象

- Model
- Engine
- Request
- Batch
- KV Cache
- Kernel
- Parallel execution

### 核心能力

官方文档包含：

- Quantization
- In-flight Batching
- Paged Attention
- Chunked Context / Prefill
- KV Cache Reuse
- Speculative Decoding
- TP / PP / EP 等并行
- Multi-GPU / Multi-node
- Disaggregated Serving
- Benchmarking
- Telemetry

### 触点

- Python API
- CLI
- Runtime / executor
- Model / engine artifact
- Serving commands
- Benchmark CLI
- Configuration

### 对知识底座的重要性

TensorRT-LLM 很适合帮助理解：

> **“一个模型如何变成一个针对特定 NVIDIA 硬件高度优化的 inference execution system。”**

和 Dynamo 相比，它更靠近执行层；和 Nsight Compute 相比，它又是一个“运行系统”，不是 profiling 工具。

---

## 25.4 vLLM

### 定位

vLLM 是高吞吐 LLM inference / serving engine，也是理解现代 LLM Serving 机制的重要参考。

### 核心能力

官方文档包括：

- PagedAttention
- Continuous Batching
- Chunked Prefill
- Prefix Caching
- 多种 Quantization
- Speculative Decoding
- TP / PP / DP / EP / Context Parallelism
- Disaggregated Prefill / Decode
- High-throughput serving

### 典型触点

- Python API
- `vllm serve` CLI
- OpenAI-compatible API
- Engine / server configuration
- Prometheus metrics
- OpenTelemetry tracing
- Benchmark commands

### Observability 特点

vLLM 的 metrics 明确区分：

- Server-level metrics
- Request-level metrics

Server-level 用于理解整个 engine 状态和容量，例如 queue、KV usage、throughput；request-level 则解释具体请求体验。

其官方设计也明确把：

- Prometheus 用于 production monitoring
- Logging 用于 ad-hoc debugging / development
- OpenTelemetry 用于 tracing

作为不同用途。

### 对 UX/Product 的价值

vLLM 很适合研究：

> **一个 inference engine 如何同时向上提供服务 API，向下管理 KV / batching / scheduler，并向旁路输出 production observability。**

---

## 25.5 SGLang

### 定位

SGLang 是面向 LLM / multimodal inference 的高性能 serving system。

### 关注能力

包括：

- 高性能 serving
- Prefix / Radix cache 相关机制
- Speculative decoding
- Scheduling
- Distributed execution
- PD Disaggregation
- Benchmark
- Production observability

### 触点

- Python
- CLI
- Native endpoint
- OpenAI-compatible API
- Benchmark CLI
- Prometheus metrics
- Request dump / replay
- Profiler

### Benchmark 能力

`bench_serving` 可以控制：

- Dataset
- Input / output length
- Request rate
- Concurrency
- Streaming
- Cache state

并输出：

- Request throughput
- Token throughput
- E2E latency
- TTFT
- ITL
- TPOT

这很好地体现了：

> **Benchmark 的输入不只是模型，而是 workload。**

### Observability / Reproducibility

SGLang 还提供：

- Request dump and replay
- Crash dump and replay
- Prometheus metrics

这体现出推理系统的“复盘”对象不仅是 profiler report，还可以是：

> **真实请求现场。**

---

## 25.6 NVIDIA Triton Inference Server

### 定位

Triton 是更通用的 production inference server，不只服务 LLM。

### 核心对象

- Model Repository
- Model Configuration
- Model Instance
- Request
- Batch
- Endpoint

### 核心能力

- 多框架 Backend
- HTTP / gRPC Serving
- Model Repository
- Model lifecycle
- Dynamic Batching
- Multiple model instances
- Metrics
- Performance Analyzer

### 触点

- Model Repository
- `config.pbtxt`
- CLI
- HTTP
- gRPC
- Metrics endpoint
- Performance Analyzer

### 对 UX/Product 的价值

Triton 适合理解：

> **“如何把模型运行能力标准化成 production inference service。”**

它与 LLM-specific Engine 的最大区别之一，是它更强调通用模型部署与 serving contract。

---

## 25.7 NVIDIA Nsight Systems

### 定位

Nsight Systems 是 **system-level profiler**。

它不直接把用户对象定义成“LLM request”，而更关注：

- Process
- Thread
- CPU
- CUDA API
- GPU
- Stream
- Kernel
- Memory copy
- Communication
- Timeline

### 触点

- CLI
- GUI
- Trace / report artifact

### 它回答的问题

例如：

- CPU 在什么时候调用 GPU？
- Kernel launch 之间为什么有 gap？
- CPU / GPU 是否并行？
- Communication 与 compute 是否重叠？
- 多 stream 实际如何执行？
- 哪个 kernel 成为 long pole？

### 对 UX/Product 的价值

它代表了一个重要下钻边界：

```text
LLM-specific abstraction
↓
System execution abstraction
```

当 inference engine 层的 metric 无法解释问题时，性能工程师会进一步进入这种 system timeline。

---

## 25.8 NVIDIA Nsight Compute

### 定位

Nsight Compute 是 **CUDA kernel-level profiler**。

它比 Nsight Systems 更深一层。

### 核心对象

- Kernel
- Source
- Instruction
- Memory
- SM
- Occupancy
- Warp
- Pipeline
- Hardware counter

### 触点

- GUI
- `ncu` CLI
- Profile report
- Exported report
- Analysis rules

### 典型能力

- Detailed kernel metrics
- Memory analysis
- Occupancy
- Roofline
- Result comparison
- Baseline
- Rule-based analysis

### 对 UX/Product 的价值

Nsight Compute 很值得研究的一点，是它不是只显示 raw counter，而是在逐渐建立：

```text
Counter
↓
Derived metric
↓
Performance issue
↓
Prioritized rule / recommendation
```

这是专家工具“从 observability 走向 explainability”的典型方式。

---

## 25.9 NVIDIA AIPerf / Triton Performance Analyzer

### 定位

属于 **workload generation + inference benchmarking** 层。

它们不是 runtime，也不是 profiler。

核心逻辑：

```text
Define workload
↓
Send requests
↓
Measure client-observed performance
↓
Compare configurations
```

### AIPerf

当前 NVIDIA 的 GenAI benchmark 工具链正在向 AIPerf 迁移；Triton Performance Analyzer 文档已经提示 GenAI-Perf deprecated。

AIPerf 可测量包括：

- TTFT
- TTFO
- ITL / TPOT
- Request latency
- Request throughput
- Output token throughput
- Success / error
- Quality goodput

它还能够结合部分 server metrics。

### 对 UX/Product 的价值

它清晰说明：

> **Benchmark 是一个独立用户任务，而不是 profiler 的附属功能。**

---

## 25.10 Huawei MindIE

### 整体定位

MindIE 是华为昇腾推理加速套件，覆盖多个推理层级。

截至 MindIE 2.3 官方文档，和 LLM 场景最相关的是：

```text
MindIE
├── MindIE LLM
├── MindIE Motor
└── MindIE Turbo
```

---

### MindIE LLM

定位：

> Ascend 上的大模型 inference component。

官方架构中包含：

- Server
- LLM Manager
- Scheduler
- Block Manager
- Executor
- Text Generator
- Modeling

核心能力包括：

- Concurrent request scheduling
- Continuous Batching
- PageAttention
- FlashDecoding
- KV Cache management
- Quantization
- TP / DP / EP / CP / SP 等并行能力

触点包括：

- RESTful APIs
- C++ API
- Configuration
- Service runtime

它与 vLLM / TensorRT-LLM 类似，跨越 inference engine、scheduler 和一定 serving 能力。

---

### MindIE Motor

定位：

> 面向 Serving / Cluster 的请求调度与 PD Disaggregation 框架。

官方能力包括：

- Prefill / Decode request scheduling
- Load balancing
- Coordinator
- Controller
- Worker / instance health
- RAS
- Cluster deployment

在 PD 场景中，Coordinator 是外部请求入口，Prefill 和 Decode 实例独立运行。

这使 MindIE Motor 更接近：

```text
Distributed serving / orchestration
```

而 MindIE LLM 更接近：

```text
Inference execution / scheduling
```

---

### MindIE 的研究价值

对 Ascend 场景来说，MindIE 最重要的参考意义不是“它有什么页面”，而是：

> **Ascend 技术栈如何划分 inference engine、scheduler、PD serving、cluster operation 的职责边界。**

特别值得关注的业务能力包括：

- KV Cache 管理
- Continuous Batching
- TTFT / TPOT 目标下的调度
- PD Disaggregation
- Load balancing
- RAS
- Ascend-specific parallelism / communication

---

## 25.11 一张工具层级地图

```text
                     INFERENCE TOOLING STACK

Application / Client
       │
       ▼
API / Serving / Distributed Orchestration
       │
       ├── NVIDIA Dynamo
       ├── MindIE Motor
       └── NVIDIA Triton Inference Server
       │
       ▼
Inference Engine / Scheduler / Runtime
       │
       ├── vLLM
       ├── SGLang
       ├── TensorRT-LLM
       └── MindIE LLM
       │
       ▼
Runtime / Kernel / Device
       │
       └── CUDA / Ascend runtime + optimized kernels
       │
       ▼
Hardware
```

旁路工具：

```text
Benchmark
├── NVIDIA AIPerf
├── Triton Performance Analyzer
├── vLLM benchmark
└── SGLang bench_serving

Production Observability
├── Engine metrics
├── Prometheus
├── OpenTelemetry
└── Logs

System Profiling
└── NVIDIA Nsight Systems

Kernel Profiling
└── NVIDIA Nsight Compute
```

---

## 25.12 从竞品得到的领域判断

### 1. 推理工具天然分层

不同工具之所以分开，不只是组织或上市节奏问题，而是因为：

- 观测对象不同
- 使用阶段不同
- 采集 overhead 不同
- 专业深度不同
- 用户要解决的问题不同

---

### 2. “一站式”不等于“一个工具把所有能力重新做一遍”

真实工具生态更常见的是：

```text
High-level system
→ expose context
→ allow drill-down
→ specialist profiler
```

重点是上下文能否串联，而不是把所有专业能力压成一张 Dashboard。

---

### 3. LLM-specific observability 正在向 Request / KV / Scheduler 层上移

传统 profiler 更关注：

```text
CPU
GPU
Kernel
Memory
```

现代 LLM engine 开始暴露：

```text
Request
Queue
TTFT
TPOT
KV Cache
Prefix hit
Preemption
Scheduler
```

说明推理工具的核心观测对象已经从“硬件执行”向“业务请求 × 模型状态 × 系统资源”扩展。

---

### 4. 深度分析仍然需要跨层

LLM metric 能告诉用户：

> 什么 request 慢。

System profiler 能告诉用户：

> 时间花在哪里。

Kernel profiler 能进一步告诉用户：

> 为什么这个 kernel 没跑好。

因此真正的诊断能力依赖：

```text
Request
↕
Engine
↕
Runtime
↕
Kernel
↕
Hardware
```

之间的映射。

---

# 26. 关键因果链

这一节作为整个知识底座的压缩索引。

### Context Length

```text
Context Length ↑
→ Prefill compute ↑
→ KV Cache per request ↑
→ Memory pressure ↑
→ Max concurrency ↓
→ Capacity / queue changes
```

---

### Output Length

```text
Output Length ↑
→ Decode duration ↑
→ KV residency time ↑
→ Active request pressure ↑
→ Decode capacity ↓
```

---

### Request Rate

```text
Request Rate ↑
→ Queue opportunity ↑
→ Batch opportunity ↑
→ Throughput may ↑
→ Saturation reached
→ Queue grows
→ P99 latency ↑ sharply
```

---

### Batch Size

```text
Batch Size ↑
→ Hardware utilization ↑
→ Throughput ↑
→ memory pressure ↑
→ scheduling / waiting ↑
→ latency may ↑
```

---

### Quantization

```text
Precision ↓
→ Weight / KV memory ↓
→ bandwidth pressure ↓
→ capacity / performance potential ↑
→ quality risk ↑
```

---

### TP

```text
TP ↑
→ more compute devices per request
→ per-device memory pressure ↓
→ communication ↑
→ synchronization ↑
→ latency may improve or worsen
```

---

### Prefix Cache

```text
Shared Prefix ↑
→ Cache hit ↑
→ Prefill work ↓
→ TTFT potential ↓
→ cache residency ↑
→ KV capacity pressure ↑
```

---

### PD Disaggregation

```text
P/D resource specialization
→ interference ↓
→ independent scaling
→ potential SLO improvement

but

KV transfer
+ network
+ routing
+ coordination
→ additional cost / failure modes
```

---

### Optimization

```text
Observed symptom
≠
Root cause

Metric
→ Evidence
→ Component
→ Hypothesis
→ Controlled change
→ Validation
```

---

# 27. 面向 UX / Product 的最终领域原则

1. **推理是一个受约束的多目标优化系统，不是一个单指标性能问题。**
2. **任何性能结论都必须绑定 Model × Workload × Hardware × Configuration。**
3. **Prefill 与 Decode 是理解 LLM Serving 的两个基本执行阶段。**
4. **KV Cache 是连接模型计算、memory、scheduler、capacity 与 routing 的核心运行状态。**
5. **Serving 的本质是有限资源上的动态请求调度。**
6. **Parallelism 获得更多计算资源的同时，也引入通信与同步成本。**
7. **PD Disaggregation 是 workload-driven 的资源解耦，不是默认更优的部署方式。**
8. **GPU/NPU utilization、throughput、latency、cost 不能互相替代。**
9. **平均值不足以代表在线服务，P95/P99 和 saturation behavior 同样关键。**
10. **Benchmark 必须控制 workload 和上下文，否则比较无意义。**
11. **Metric 只能说明发生了什么，不能自动证明 root cause。**
12. **Monitoring、Tracing、Profiling、Benchmark、Evaluation 是不同任务。**
13. **观测本身有 overhead，数据粒度越深越需要有选择地采集。**
14. **Controlled environment 用于证明因果；Production observation 用于发现真实问题。**
15. **正确性、性能、容量、成本与可靠性共同构成推理工程质量。**
16. **用户真正做的是判断和决策，不是单纯配置参数。**
17. **专家工具也需要清晰的对象、上下文、证据关系与因果链。**
18. **推理工具生态天然分层；不同工具应按任务与抽象层理解，而不是简单做功能多少比较。**
19. **高层 LLM observability 和底层 system/kernel profiling 需要通过上下文映射形成完整诊断链。**
20. **优化闭环最终必须回到真实 workload 与业务 SLO，而不是停留在实验室峰值。**

---

# 28. 核心术语速查

| 术语 | 含义 |
|---|---|
| Inference | 使用训练好的模型生成预测或输出 |
| Serving | 将推理能力以在线服务形式提供给请求 |
| Prefill | 处理输入 Prompt 并建立 KV Cache 的阶段 |
| Decode | 基于已有上下文逐 token 生成的阶段 |
| KV Cache | 保存历史 Attention K/V 的运行时状态 |
| TTFT | 请求到首个输出 token 的延迟 |
| TPOT | 首 token 后平均每个输出 token 的处理时间 |
| ITL | 相邻输出 token 之间的延迟 |
| Throughput | 单位时间完成的 request 或 token 数量 |
| Continuous Batching | 允许 request 在 iteration/token step 动态进入退出 batch |
| Scheduler | 决定有限资源下一步执行哪些请求/阶段 |
| TP | Tensor Parallelism，拆一次模型计算 |
| DP | Data Parallelism，拆请求或 batch |
| PP | Pipeline Parallelism，拆模型层 |
| EP | Expert Parallelism，拆 MoE Expert |
| PD Disaggregation | Prefill 与 Decode 独立部署和扩缩容 |
| Prefix Cache | 复用共享 Prompt 前缀的 KV Cache |
| Quantization | 用更低数值精度表示权重、激活或 KV |
| Benchmark | 在明确 workload 下测量整体性能 |
| Monitoring | 长期观察线上服务健康状态 |
| Trace | 记录一次请求/执行的事件与时间关系 |
| Profiling | 深入分析时间和硬件资源消耗 |
| SLO | Service Level Objective，服务目标 |
| Saturation | 系统接近容量极限、queue 和 latency 开始快速恶化的状态 |
| Baseline | 用于后续比较的基准配置或结果 |
| Regression | 相比既有 baseline 出现性能或正确性退化 |

---

# 29. 竞品官方资料索引

以下资料用于校验本知识底座中的竞品事实，建议 AI 在处理“最新版本、最新能力、当前接口”等问题时重新访问官方资料，而不是依赖本文件静态内容。

- NVIDIA Dynamo Documentation  
  https://docs.nvidia.com/dynamo/
- NVIDIA Dynamo — Disaggregated Serving  
  https://docs.nvidia.com/dynamo/latest/user-guides/disaggregated-serving
- NVIDIA Dynamo — KV Cache Aware Routing  
  https://docs.nvidia.com/dynamo/latest/user-guides/kv-cache-aware-routing
- TensorRT-LLM Documentation  
  https://nvidia.github.io/TensorRT-LLM/
- vLLM Documentation  
  https://docs.vllm.ai/en/stable/
- vLLM Metrics  
  https://docs.vllm.ai/en/latest/design/metrics/
- SGLang Documentation  
  https://docs.sglang.ai/
- SGLang Bench Serving  
  https://docs.sglang.ai/developer_guide/bench_serving
- SGLang Observability  
  https://docs.sglang.ai/advanced_features/observability.html
- NVIDIA Triton Inference Server  
  https://docs.nvidia.com/deeplearning/triton-inference-server/
- NVIDIA Nsight Systems  
  https://docs.nvidia.com/nsight-systems/
- NVIDIA Nsight Compute  
  https://docs.nvidia.com/nsight-compute/
- NVIDIA AIPerf  
  https://docs.nvidia.com/aiperf/
- Huawei MindIE 2.3 Documentation  
  https://www.hiascend.com/document/detail/en/mindie/230/index/index.html

---

## 文档使用约束（给后续 AI）

使用本知识底座生成内容时：

- 不要把推理简化为“模型部署”。
- 不要把模型、Inference Engine、Runtime、Serving、Profiler 混成同一个层级。
- 不要脱离 workload 比较性能。
- 不要把 GPU/NPU utilization 直接当 throughput。
- 不要看到 TTFT/TPOT 异常就直接断言 root cause。
- 不要把所有推理优化都归结为 Kernel 优化。
- 不要把 PD Disaggregation 描述成默认更优。
- 不要把所有观测数据都视为可以无成本长期采集。
- 不要把 Benchmark、Monitoring、Tracing、Profiling 混为一谈。
- 不要按竞品界面结构反推领域结构。
- 需要最新版本信息时，重新核对官方文档。
- 生成 UX / Product 内容时，优先从“用户正在判断什么、需要什么证据、改变什么系统对象”出发，而不是从“有哪些参数和图表”出发。
