### **最合理的消融实验设计原则：只验证“论文声称的核心贡献”，不要把所有代码开关都做成消融**

你这个项目已经不是“小模型实验”，而是一个完整系统（system paper / pipeline paper）。这种论文里，消融实验的目标不是“穷举所有组件”，而是：

1. 验证核心创新是否真实有效；
2. 证明性能提升来自你的设计，而不是参数偶然；
3. 证明各模块之间具有互补性；
4. 证明你的方法论（CV高召回 + LLM约束推理）是正确路线。

所以最合理的设计，不是做十几个小 ablation，而是：

> “围绕论文贡献点，做 4~6 个关键消融”。

这是 reviewer 最喜欢的结构。

---

### **你这篇论文真正的“核心贡献”是什么**

从 `myself.md` 来看，你真正的创新有四个：

| 创新 | 是否必须做消融 | 原因 |
|---|---|---|
| 三模态融合（layout + trajectory + smart device） | 必须 | 最大卖点 |
| CV + constrained-LLM hybrid | 必须 | 方法论核心 |
| 双源家具召回（hatch + trajectory loop） | 必须 | CV贡献 |
| 分层规则推理（7-layer / 4-stage funnel） | 建议 | 可解释性贡献 |

其它细节：
- Gaussian heatmap
- DBSCAN-like
- IoU suppression

这些不需要单独消融。

否则 reviewer 会觉得：
“作者在堆 implementation details。”

---

# 最推荐的论文级消融设计（非常适合你）

建议做：

# 4组主消融 + 1组补充实验

这是最稳的。

---

# 1. Multi-modal Ablation（最重要）

这是必须的。

因为你论文最大的 claim 是：

> heterogeneous multi-modal fusion

所以 reviewer 一定会问：

“每个模态真的有贡献吗？”

因此必须设计：

| Variant | Layout | Trajectory | Smart Device |
|---|---|---|---|
| Full | ✓ | ✓ | ✓ |
| w/o Trajectory | ✓ | ✗ | ✓ |
| w/o Device Prior | ✓ | ✓ | ✗ |
| Layout Only | ✓ | ✗ | ✗ |

---

## 为什么这样设计最合理

因为它对应：

| 模态 | 解决的问题 |
|---|---|
| Layout | 几何结构 |
| Trajectory | 行为语义 |
| Smart Device | 强先验语义 |

三者功能不同。

因此 reviewer 会认为：
这是“orthogonal contribution”。

非常合理。

---

## 这里应该观察什么指标

重点：

| 指标 | 原因 |
|---|---|
| RTA | 房间类型 |
| FNA | 家具命名 |
| CLS | 综合性能 |

不要只看 IoU。

因为 trajectory/device 对 IoU 提升未必大，
但对 semantic accuracy 极其重要。

---

# 2. Hybrid CV-LLM Ablation（第二重要）

这是你方法论的核心。

你真正的创新不是“用了LLM”，而是：

> constrained action-based LLM correction

这是 reviewer 最可能感兴趣的点。

建议：

| Variant | CV Recall | LLM Geometry Fix | Hard Rules |
|---|---|---|---|
| Full | ✓ | ✓ | ✓ |
| CV-only | ✓ | ✗ | ✓ |
| Free-form LLM | ✓ | unrestricted | ✗ |
| Direct LLM Generation | ✗ | full generation | ✗ |

---

## 这组实验特别关键

因为它能证明：

### 1. 纯CV：
- recall高
- semantic弱
- merge/split错误多

### 2. unrestricted LLM：
- hallucination严重
- consistency差

### 3. direct generation：
- geometry最不稳定

### 4. 你的 constrained action protocol：
- 最稳定

这会直接支撑你的核心claim：

> “LLM should refine perception instead of replacing perception.”

这句话非常论文化。

---

# 3. Furniture Detection Ablation（CV核心）

这是你 CV 部分最值得发的内容。

你不是单一 detector。

而是：

- hatch shadow
- trajectory enclosed region

双源互补。

建议：

| Variant | Hatch | Trajectory-loop |
|---|---|---|
| Full | ✓ | ✓ |
| w/o Hatch | ✗ | ✓ |
| w/o Trajectory Loop | ✓ | ✗ |

指标：

| Metric | 作用 |
|---|---|
| Recall | 最重要 |
| F1_IoU | 综合 |
| FN Area | 漏检面积 |

---

## reviewer会喜欢什么

你需要强调：

### hatch:
更适合：
- 大家具
- 规则家具
- 阴影明显区域

### trajectory-loop:
更适合：
- 椅子
- 桌腿
- 非规则家具
- 阴影弱区域

这叫：

> complementary perception signals

这是标准论文术语。

---

# 4. Reasoning Strategy Ablation（建议）

这一组是“增强论文解释性”的。

不是最核心，
但会提升 paper quality。

---

## Room reasoning

| Variant | Layered Priority | Behavior Prior |
|---|---|---|
| Full | ✓ | ✓ |
| Flat Prompt | ✗ | ✓ |
| w/o Behavior | ✓ | ✗ |

这里重点看：

- bedroom precision
- living room confusion
- bathroom recall

---

## Furniture naming

| Variant | Allowed List | Shape Constraint |
|---|---|---|
| Full | ✓ | ✓ |
| w/o Allowed List | ✗ | ✓ |
| w/o Shape Constraint | ✓ | ✗ |

重点看：

- hallucination rate
- invalid label rate

这个 reviewer 会非常喜欢。

因为现在很多 LLM paper 最大问题就是：
“不控制 hallucination”。

---

# 5. 一个非常加分的实验（强烈建议）

# Robustness / Noise Experiment

这是很多人会漏掉的。

但你项目特别适合做。

---

## 方法

人为加入：

| 噪声类型 | 方式 |
|---|---|
| trajectory noise | 坐标随机偏移 |
| device missing | 删除部分设备 |
| wall segmentation noise | mask腐蚀/膨胀 |

看性能下降。

---

## 为什么这特别重要

因为你的输入来自：
- 扫地机器人
- 真实行为轨迹
- 智能家居设备

本身就是 noisy real-world data。

所以 robustness 会非常加分。

很多 reviewer 会直接提高评价。

---

# 最终推荐的实验章节结构（非常合理）

```text
5. Experiments

5.1 Experimental Setup
    Dataset
    Metrics
    Implementation Details

5.2 Comparison with Baselines
    VLM Zero-shot
    VLM-CoT
    CV-only
    ...

5.3 Ablation Studies

    5.3.1 Multi-modal Fusion Ablation
    5.3.2 Hybrid CV-LLM Ablation
    5.3.3 Dual-source Furniture Detection Ablation
    5.3.4 Reasoning Strategy Ablation

5.4 Robustness Analysis

5.5 Qualitative Results

5.6 Failure Cases
```

这是非常标准、非常“像顶会”的结构。

---

# 最后一个关键建议（非常重要）

你现在最应该避免的是：

# “过多细碎消融”

例如：

- 去掉 Gaussian blur
- 去掉 Otsu
- 去掉 IoU suppression
- 去掉 DBSCAN-like

这些 reviewer 不关心。

因为这些属于：
implementation engineering。

---

reviewer 真正关心的是：

# “你的 conceptual contribution 是否成立”

也就是：

- 多模态是否有效；
- constrained LLM 是否优于 free-form；
- 双源检测是否互补；
- 分层推理是否必要。

围绕这四点设计，
你的论文实验部分会非常成熟。

*内容由 AI 生成仅供参考*