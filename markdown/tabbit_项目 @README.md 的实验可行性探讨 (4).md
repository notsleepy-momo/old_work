### **你现在最需要做的，不是继续“优化模块”，而是把论文从“算法贡献叙事”重构成“真实智能家居语义建模叙事”。**

这是一个非常关键的转折。

因为你的实验已经告诉你：

- 真正稳定有效的是：
  - IoT semantic anchors
  - CV geometry
  - constrained semantic correction

- trajectory topology：
  - 有帮助
  - 但不是决定性核心

所以 narrative 必须顺着实验走，而不是强行让实验适配原始故事。

下面我会直接给你一个“可投稿”的 narrative 重构方案。

---

# 一、你现在的 narrative（问题在哪里）

你现在实际上在讲：

> 我提出了一个 topology-aware CV+LLM spatial reasoning framework。

这个 narrative 的隐含要求是：

- trajectory topology 必须显著提升指标
- LLM reasoning 必须非常强
- layered reasoning 必须明显优于简单规则

但实验并没有支持这些。

所以现在会出现：

| Narrative | 实验 |
|---|---|
| trajectory 是核心 | trajectory 提升很弱 |
| layered reasoning 很强 | device prior 直接锁死 |
| constrained anti-hallucination | 提升有限 |

于是论文开始“不协调”。

---

# 二、正确 narrative 应该是什么？

你现在真正成立的故事，其实是：

---

# ✅ 新 Narrative（核心）

> Consumer smart homes already contain enough ambient signals to reconstruct usable semantic spatial models without CAD, RGB imaging, or dedicated infrastructure.

这是你真正强的地方。

注意：

### 核心已经从：

❌ “新算法”

变成：

✅ “现实世界 semantic reconstruction feasibility”

这是典型 UbiComp narrative。

---

# 三、重构后的论文核心问题（非常重要）

原问题：

❌ 如何设计 topology-aware reasoning？

新问题：

✅ 如何利用消费级智能家居已有信号自动构建 semantic home model？

这是完全不同的论文。

而且后者更适合 UbiComp。

---

# 四、重构后的 Contribution（建议版本）

你现在应该把 contribution 改成：

---

## ✅ Contribution 1（主贡献）

### A practical semantic home reconstruction system using consumer smart-home signals

强调：

- 科沃斯地图
- IoT devices
- ambient trajectories

无需：

- CAD
- RGB
- LiDAR

这是最大的现实价值。

---

## ✅ Contribution 2

### Ambient semantic anchors from IoT devices substantially stabilize room-level inference

这里直接承认：

> IoT prior 很强。

不要躲。

因为：

在 UbiComp 里：

“现实系统中的强先验”

不是缺点。

是优点。

---

## ✅ Contribution 3

### Constrained semantic correction improves structural consistency under noisy spatial observations

这里强调：

- consistency
- structure
- noise robustness

不要强调：

❌ “LLM reasoning intelligence”

因为实验不支持。

---

## ✅ Contribution 4（降级）

### Trajectory topology provides complementary cues under incomplete geometric observations

注意：

### complementary
### incomplete observations

trajectory 从：

❌ 核心创新

降级为：

✅ 辅助感知信号

这就和实验一致了。

---

# 五、Abstract 应该怎么改（核心）

你现在 abstract 可能偏：

- topology
- reasoning
- hybrid CV-LLM

应该改成：

---

## ✅ 新版 Abstract 主线

### 第一段：

现实问题：

> Smart homes generate fragmented spatial signals but lack semantic spatial models.

---

### 第二段：

你的 insight：

> Existing consumer devices already provide sufficient ambient cues for semantic reconstruction.

---

### 第三段：

你的系统：

- robot maps
- IoT anchors
- behavioral trajectories
- constrained correction

---

### 第四段：

强调：

- no CAD
- no RGB
- no additional hardware

这会非常 UbiComp。

---

# 六、Method 怎么重构？

你现在 method 太像：

“算法模块”。

应该改成：

---

# ✅ 新 Method 结构

---

## 4.1 Consumer Spatial Signals

介绍：

- robot topology
- device priors
- trajectories

强调：

### heterogeneous ambient sensing

---

## 4.2 Geometry Recovery

这里：

CV detection 是主角。

trajectory：

只是补充。

不要再写：

“trajectory-driven object reasoning”

而写：

✅ “trajectory-assisted geometry completion”

差别巨大。

---

## 4.3 Semantic Anchoring

这里重点讲：

device prior。

因为实验已经证明：

它最有效。

---

## 4.4 Constrained Semantic Correction

这里强调：

- consistency
- validity
- anti-invalid-layout

不要强调：

“高级 reasoning”。

---

# 七、Evaluation 也必须重构

现在不要再以：

“trajectory 提升了多少”

为主线。

而应该：

---

# ✅ 新 Evaluation 主线

---

## 1️⃣ Feasibility

是否能从消费级数据重建 usable layout？

---

## 2️⃣ Stability

跨家庭是否稳定？

---

## 3️⃣ Robustness

轨迹稀疏时是否还能工作？

---

## 4️⃣ Semantic Value

是否提升 activity understanding？

---

## 5️⃣ Mechanism Validation

最后再放：

- modality ablation
- constrained vs free-form

而不是前面大讲 trajectory。

---

# 八、最关键的一件事

你必须接受：

## trajectory 不是核心。

这是目前实验已经告诉你的。

很多论文真正成熟的标志，就是：

> 作者终于接受“真实 contribution”是什么。

---

# 九、你现在最像哪类 UbiComp 论文？

你现在真正像的是：

### “Practical semantic modeling from ambient sensing”

而不是：

### “Novel topology reasoning framework”

这是一个非常重要的转向。

---

# 十、重构后的标题建议

你现在标题不要再强调：

❌ topology reasoning  
❌ LLM framework  

建议：

---

## 方向 1（最稳）

### Semantic Home Reconstruction from Consumer Robot Maps and Ambient Smart-Home Signals

---

## 方向 2（更 UbiComp）

### Reconstructing Semantic Home Layouts from Ambient Consumer Sensing

---

## 方向 3（保留 trajectory）

### Topology-Assisted Semantic Home Reconstruction from Consumer Smart-Home Signals

注意：

Topology-Assisted  
不是  
Topology-Driven

这个差别非常重要。

---

# 十一、最终一句话（核心）

你现在不是：

> “trajectory topology 没做出来”

而是：

> 实验帮你发现了真正成立的论文故事。

而那个故事其实更适合 UbiComp。

*内容由 AI 生成仅供参考*