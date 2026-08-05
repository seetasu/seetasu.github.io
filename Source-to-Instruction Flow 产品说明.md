# Source-to-Instruction Flow 产品说明

> 面向产品、设计与研发团队  
> 当前性质：体验 Pattern 与产品能力定义  
> 首个验证案例：Conv + Bias + ReLU

## 1. 用户问题

Ascend C 源码展示的是 API 调用，开发者还需要自行还原：

- 指令按什么顺序执行；
- 哪些代码属于搬运、计算或同步；
- 指令由哪个硬件单元执行；
- 循环、分支和依赖如何连接。

源码、硬件知识和执行关系彼此分散，用户难以从一行代码建立完整的执行理解。

## 2. 对产品团队的简短介绍

> Source-to-Instruction Flow 以源码为入口，把分散的 API 调用恢复为可阅读的执行过程。第一阶段先连接源码、Instruction Flow 与硬件单元，让用户理解“这行代码在做什么、位于哪里、由谁执行”；后续再加入 Tensor 与 Memory，解释数据如何产生、搬运和复用。系统明确区分已识别、推断和未知内容，不把静态分析包装成真实运行结果。

## 3. 产品价值

- 把源码阅读从“识别 API”推进到“理解执行过程”；
- 连接源码、Instruction 与硬件 mental model；
- 缩短从异常现象到相关代码和执行环节的定位路径；
- 为后续 Tensor、Memory、Event 诊断和 Profiling 提供统一入口。

## 4. 特性定义

用户选中源码后，系统将其定位到 Instruction Flow，并联动展示当前指令涉及的硬件单元，帮助用户回答：

> 这行代码在做什么，处于执行过程的哪里，由谁执行，前后依赖什么？

特性由三个连续能力组成：

```text
源码标记
定位当前解释对应的代码
    ↓
Instruction Recovery
识别 Move / Compute / Sync 等执行语义
    ↓
Instruction Flow Visualization
展示顺序、循环、分支、依赖与硬件参与
```

## 5. 产品特性

这里描述产品能力的发布顺序，不替代 Demo 的开发里程碑。

### 阶段 1：Instruction Flow × Hardware

首阶段聚焦最短理解闭环：

```text
选中源码
→ 定位对应 Instruction
→ 查看前后执行关系
→ 查看参与的硬件单元
→ 返回源码继续阅读
```

阶段 1 包含：

- Source 与 Instruction 双向定位；
- Instruction 顺序、Loop、Branch；
- Data、Event、Control dependency；
- Host、Scalar、MTE、Cube、Vector、Fixpipe、Event 等执行角色；
- 当前 Instruction 与硬件单元联动；
- `Recognized / Inferred / Unknown` 识别状态。

阶段 1 不展示真实耗时、利用率、stall 或 overlap。

### 后续阶段：Tensor × Memory

后续逐步补充：

- Tensor 的产生、有效、消费与复用；
- Tensor Data Dump：查看一个具体 Tile（如 `16×16`）包含的实际数据；
- GM、L1、L0、UB 等存储位置；
- 数据搬运方向、地址范围和 Buffer 占用；
- Instruction、Tensor、Memory 与 Hardware 的统一执行上下文；
- 仿真或 Profiling 数据形成的运行观测层。

Tensor 和 Memory 是对 Instruction Flow 的下钻解释，不阻塞阶段 1 建立基本执行理解。

Data Dump 需关联具体运行、Instruction、Tile、dtype、layout 和存储位置。没有 dump 数据时只显示 shape 与元数据，不生成或猜测矩阵内容。

## 6. 能力边界

- 逻辑顺序不等于真实时间；
- 静态源码不能证明某次运行实际走过的路径；
- 不承诺自动理解所有任意代码；
- 无法确认的代码保留为 `Unknown / Source-only`；
- AI 推断必须显式标记，不作为编译或运行事实；
- Tensor 实际数值只能来自 Data Dump 或其他明确的运行数据；
- 真实 duration、等待和 overlap 只能来自仿真或 Profiling。

## 7. 通用 Instruction Visualization Pattern

### 7.1 目标

对能够识别语义角色的 Instruction，使用与算子类型无关的视觉语法生成执行图；无法识别的内容保留源码位置，不隐藏、不猜测。

通用性来自统一的语义、视觉规则和降级方式，不要求不同算子生成相同形状的图。

### 7.2 最小语义模型

| 维度 | 内容 |
| --- | --- |
| Instruction | `DataCopy`、`LoadData3D`、`Mmad`、`Fixpipe` 等 |
| Execution role | Host、Scalar、MTE2、MTE1、Cube、Vector、Fixpipe、Event |
| Action type | Configure、Allocate、Move、Transform、Compute、Sync、Store |
| Operands | 主要输入 → 输出 |
| Scope | Once、Per core、Per tile、Per iteration、Conditional |
| Structure | Sequence、Loop、Branch |
| Dependency | Data、Event、Control |
| Recognition | Recognized、Inferred、Unknown |
| Evidence | Source、Compiler/IR、API Registry、Runtime |

阶段 1 必需的是 Instruction、Execution role、Action type、Structure、Dependency 和 Recognition。Tensor、Memory 语义可在后续阶段补充。

### 7.3 恢复流程

```text
区分 Host / Kernel
→ 识别 Sequence / Loop / Branch
→ 保持结构内逻辑顺序
→ 识别 Instruction 与 Action type
→ 映射 Execution role
→ 连接明确的 Data / Event / Control dependency
→ 压缩重复循环
→ 保留 Unknown
→ 默认聚焦主执行路径
```

该流程描述产品所需的语义结果，不限定解析器必须使用 AST、Compiler IR、规则或大模型。

### 7.4 视觉语法

1. **泳道**：按 Execution role 分组，隐藏未参与的空泳道；
2. **逻辑轴**：只表达执行顺序，不用宽度表示耗时；
3. **Instruction block**：显示名称、Action type 和主要输入→输出；
4. **硬件联动**：选中节点时，高亮参与的硬件单元；
5. **关系**：区分 Data、Event 和 Control，不共用同一种视觉编码；
6. **循环**：表达 Initialize、Repeated body ×N、Tail/Finalize；
7. **分支**：展示条件和可能路径，不冒充真实已执行路径；
8. **Event**：表达 signal/wait 或依赖线，不画成普通计算；
9. **Unknown**：保留源码位置和 unresolved 状态，不补画未经确认的硬件或依赖。

### 7.5 Source 联动

- 点击源码，聚焦对应 Instruction；
- 点击 Instruction，返回对应源码；
- 一个 Instruction 可以关联多处源码；
- 切换 Host / Kernel 文件不丢失当前执行上下文；
- 当前文件没有对应引用时，不伪造高亮；
- Loop 折叠或展开后，保持当前源码和硬件上下文。

### 7.6 识别与 AI 边界

优先使用确定性材料：

```text
AST / CFG / Compiler IR
+ API 语义注册表
+ 类型、符号和定义—使用关系
```

大模型用于辅助解释陌生封装、复杂函数和未登记 API。其结果只能进入 `Inferred`，不能覆盖确定性事实。

| 状态 | 含义 |
| --- | --- |
| Recognized | 有源码、编译材料或已登记 API 语义支持 |
| Inferred | 根据上下文生成的候选解释，需要保留依据 |
| Unknown | 当前证据不足，只显示源码位置 |

推荐的演进闭环：

```text
确定性识别
→ 发现 Unknown
→ AI 提出候选解释
→ 用户确认或修正
→ 保存为项目规则
→ 下次确定性识别
```

### 7.7 阶段 1 验收

- 用户能从源码进入对应 Instruction；
- 用户能看出主要执行顺序、循环和依赖；
- 用户能识别当前参与的硬件单元；
- 用户能从 Instruction 返回源码；
- Unknown 不被隐藏或包装成事实；
- 逻辑轴不暗示真实耗时；
- 不依赖 Tensor、Memory 视图也能完成基本执行理解。
