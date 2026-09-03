### **整体趋势是合理的，但目前结果暴露出两个会被 reviewer 重点质疑的问题：系统对 smart device 过敏、以及 hatch 分支存在单点失效**

从 reviewer 视角看，你这组 B 实验“像真实系统结果”，不是那种过于完美的假结果，因此这是好事。尤其：

- 轨迹删除 30% 仅轻微下降
- Gaussian blur 基本稳定
- 稀疏行为数据性能下降
- device noise 导致语义错误

这些都符合真实系统行为。

但目前有几个指标组合，会让 reviewer 立刻追问。

---

# 一、B1 设备坐标扰动：趋势合理，但说明系统过度依赖 device prior

你现在：

| Noise | CLS |
|---|---|
| 0dm | 90.0% |
| 5dm | 66.5% |
| 10dm | 59.0% |

这说明：

> ±5dm 就已经造成严重语义崩塌。

这是比较危险的。

---

## 为什么 reviewer 会质疑

因为家庭尺度下：

- 5dm = 50cm
- 实际 smart home 坐标误差通常就在这个量级

如果 50cm 就让：

- FNA: 88.9% → 57.1%
- CLS: 90 → 66

reviewer 会认为：

> “系统并不 robust，只是在依赖高精度 device anchor。”

---

# 二、但 RTA 很高，这说明问题不在房间分类，而在家具语义传播

这是你结果里最重要的信息：

| Noise | RTA |
|---|---|
| 0dm | 100 |
| 10dm | 85.7 |

实际上：

- 房间分类没完全崩
- 崩的是 FNA

即：

| 指标 | 问题 |
|---|---|
| F1 | 小幅下降 |
| RTA | 基本稳定 |
| FNA | 大幅下降 |

这非常像：

> “错误 device label 污染了 furniture naming”

这是合理现象。

---

# 三、你现在最大的风险：系统“语义锚点单点依赖”

README 里：

```text
priority ≥4 的设备直接确定房间类型
```

这是 reviewer 会攻击的点。

因为：

- Refrigerator 偏 50cm
- 直接掉进客厅 bbox
- 整个房间被误判 kitchen

这种 cascading failure 很典型。

---

# 四、你应该如何修实验（非常关键）

目前不要删结果。

应该：

> 把它转化为“发现系统局限性”的 evidence。

这是 UbiComp 很喜欢的写法。

---

# 五、正确写法（非常重要）

你不能写：

> “系统 robust under device noise”

因为不成立。

应该写：

> “The system remains structurally stable under moderate coordinate perturbation, while semantic naming accuracy degrades due to strong reliance on device priors.”

这就合理了。

---

# 六、你还需要一个“改进版实验”

现在 reviewer 会问：

> “那你怎么解决？”

你需要补一个：

---

## ✅ Device Confidence Decay 实验（强烈建议补）

加入：

```python
device_confidence *= exp(-distance/error)
```

或者：

```python
priority -= noise_penalty
```

再做：

| Method | CLS@10dm |
|---|---|
| Original | 59 |
| Confidence-aware | 74 |

这会一下把论文质量提升很多。

因为：

> reviewer 不怕系统有问题  
> reviewer 怕作者不知道问题在哪

---

# 七、B2 行为轨迹稀疏度：结果合理，但有一个异常点

这里：

| Days | CLS |
|---|---|
| 1天 | 69.3 |
| 3天 | 69.3 |
| 7天 | 66.5 |
| 14天 | 90 |

问题：

> 7 天反而比 1 天差。

这是 reviewer 一定会问的。

---

# 八、为什么会出现这种现象

其实很可能是真实现象。

原因通常：

---

## 情况 A：行为模式冲突

7 天数据可能：

- 混入异常轨迹
- 周末行为
- 客人轨迹

导致：

```text
living_room ↔ bedroom
```

混淆。

---

## 情况 B：LLM 过拟合行为 prior

你现在：

```text
night → bedroom
day → living_room
```

如果 7 天数据：

- 某人长时间在客厅工作

LLM 可能误推。

---

# 九、这反而是“好论文素材”

你不要隐藏它。

应该写：

> Intermediate-duration trajectories occasionally introduce behavioral ambiguity due to mixed activity patterns.

然后：

```text
Future work:
temporal consistency modeling
```

这会显得你很成熟。

---

# 十、B3 图像退化：大部分合理，但 hatch contrast 实验有点危险

这里：

| Condition | CLS |
|---|---|
| Clean | 90 |
| Reduced Hatch Contrast | 24 |

这是：

> catastrophic failure

reviewer 会立刻意识到：

```text
你的系统高度依赖 hatch extraction
```

---

# 十一、但这个结果“不是不合理”

因为你的系统：

```text
红框 confidence = 0.90
```

而且：

```text
蓝框会删除与红框 overlap 的区域
```

所以：

一旦 hatch 消失：

- 大家具召回崩
- naming 崩
- room inference 崩

这是符合 pipeline 的。

---

# 十二、你现在真正需要的是“Topology Contribution”实验

你 README 里已经设计了：

```text
w/o Hatch
w/o Robot Trajectory Loop
```

你必须跑。

因为现在 reviewer 会问：

> “既然 hatch 这么脆弱，trajectory loop 是否能补偿？”

如果你能证明：

| Variant | Small Recall |
|---|---|
| w/o loop | 62 |
| full | 82 |

那这个 catastrophic failure 就会变成：

> “双源互补设计的必要性”

---

# 十三、目前实验的总体评价

| 部分 | 评价 |
|---|---|
| 趋势真实性 | ✅ 很真实 |
| 系统行为合理性 | ✅ 基本合理 |
| 是否像真实 research | ✅ 是 |
| 是否太完美 | ❌ 没有 |
| 是否会被 reviewer challenge | ✅ 会 |
| 是否致命 | ❌ 不致命 |

实际上：

> “有缺陷但可解释”  
比  
> “所有实验都完美”

更像真正 UbiComp 论文。

---

# 十四、你下一步最重要的事

不是“调高分”。

而是：

## ✅ 把失败模式理论化

你现在已经开始有：

- semantic anchor collapse
- behavior ambiguity
- hatch single-point failure

这其实已经进入：

> systems research 的层次了。

这是好事。

---

# 十五、我最建议你立刻补的两个实验

## ✅ 必补 1：Topology Contribution

验证：

- hatch
- trajectory loop

互补性。

这是你方法的核心创新之一。

---

## ✅ 必补 2：Confidence-aware device matching

这是修复：

```text
device noise sensitivity
```

的关键。

如果补上这两个：

论文质量会提升非常明显。

*内容由 AI 生成仅供参考*