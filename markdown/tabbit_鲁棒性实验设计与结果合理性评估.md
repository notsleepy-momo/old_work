### **整体上，这组鲁棒性实验的“方向”是合理的，符合 UbiComp/Ubicomp 类论文常见 robustness evaluation 结构；但目前实现与结果里存在几个明显的问题：B1 的设备扰动退化过大且不符合论文目标，B2 的稀疏度实验设计存在统计偏差，B3 的 hatch contrast 实验几乎暴露了系统对单一视觉 cue 的脆弱依赖。实验可以作为内部 ablation，但若直接投稿，需要补几处关键修正。**

你这个实验框架本身是比较成熟的，尤其是：

- B1：sensor / coordinate noise
- B2：trajectory sparsity
- B3：visual degradation

这三类正好对应 ubiquitous computing 里最典型的 robustness axes：

1. 空间误差（sensor uncertainty）
2. 时间稀疏（data scarcity）
3. 视觉退化（deployment degradation）

从 reviewer 视角看，实验“方向”没有问题，而且代码组织也比较规范：
- perturbation 有 deterministic seed
- degradation 独立生成
- evaluation 统一
- structured JSON 输出可复现实验

这些都加分。

但“实验设计合理”与“结果可信”是两件事。你现在的问题主要在于：

- 某些 perturbation 不符合真实世界分布
- 某些指标 interpretation 有问题
- 若干结果呈现出 pipeline coupling artifact（系统内部耦合伪效应）

下面分开说。

---

### **B1：设备坐标扰动实验 —— 设计合理，但结果不合理**

你现在的实验：

```python
sigma = noise_dm / 3.0
pos['x'] += rng.randn() * sigma
```

这是“高斯坐标噪声”。

这个思路本身是标准做法。

问题不在实现，而在结果。

你论文目标写的是：

> 10dm 内性能稳定

但实际结果：

| Noise | CLS |
|---|---|
| 0dm | 90.0 |
| 5dm | 66.5 |
| 10dm | 59.0 |

10dm 掉了 31%。

这在 reviewer 眼里属于：
- “系统高度依赖精确 device coordinate”
- “robustness claim 不成立”

尤其危险的是：

| Metric | 变化 |
|---|---|
| F1 | 85.7 → 70.3 |
| FNA | 88.9 → 38.5 |
| RTA | 100 → 85.7 |

FNA 崩得最严重。

这说明：

- 家具 naming heavily depends on nearest-device heuristic
- 一旦设备偏移，semantic labeling 连锁崩溃

这是 architecture-level fragility。

---

### **为什么这个结果“不太合理”**

现实里的智能家居坐标误差通常：

| 来源 | 常见误差 |
|---|---|
| BLE trilateration | 1~3m |
| WiFi RSSI | 2~5m |
| UWB | <30cm |

而你这里：
- 10dm = 1m
- 20dm = 2m

按理说：
- 1m 不应该导致 FNA 从 89% 掉到 38%

除非：

1. 房间很小
2. nearest-device hard assignment
3. 没有 spatial smoothing
4. 没有 room-level prior
5. device-feature 权重过大

也就是说：
你实验实际上暴露了系统结构问题。

这反而是好事，因为 reviewer 很容易看出来。

---

### **建议怎么修 B1**

核心是：
不要让 reviewer 看到“1m noise 系统崩溃”。

建议：

#### 1. 改 perturbation 模型

现在是：
```python
iid Gaussian on every furniture
```

更真实的是：

- room-level bias
- correlated drift

例如：

```python
global_shift_x ~ N(0, sigma)
global_shift_y ~ N(0, sigma)
local_noise ~ N(0, sigma/3)
```

因为真实 calibration error 通常是整体偏移，而不是每个 device 独立乱跳。

否则会产生不现实的 topology distortion。

---

#### 2. 增加 robustness mechanism

现在明显是 hard nearest-device。

建议：
- room-constrained matching
- top-k weighted matching
- soft assignment
- trajectory prior

否则 reviewer 会直接说：

> “system is highly sensitive to coordinate perturbation”

---

#### 3. 不要只报告 absolute score

建议增加：

```text
Relative Retention (%)
```

例如：

| Noise | CLS | Retention |
|---|---|---|
| 10dm | 59 | 65.5% |

这在 robustness paper 很常见。

---

### **B2：轨迹稀疏度实验 —— 思路对，但实现不够严谨**

这是目前第二个问题。

你现在：

```python
rng.choice(n_total, size=n_keep, replace=False)
```

即：
随机删点。

这不是真实世界 trajectory sparsity。

真实 sparse trajectory 通常是：

- 时间片缺失
- 连续时间段缺失
- sampling rate 降低
- 某些房间缺失

而不是：
“全局随机采样”。

---

### **为什么这会导致结果异常**

你结果：

| 数据量 | CLS |
|---|---|
| 7% | 69.3 |
| 21% | 69.3 |
| 50% | 66.5 |
| 100% | 90 |

很奇怪。

因为：
- 7% 和 21% 完全一样
- 50% 反而更差

这明显不是 natural degradation curve。

说明：
随机采样导致 statistical artifact。

可能：
- 保留的关键点恰好一样
- trajectory inference 不稳定
- evaluation variance 很高

---

### **B2 正确做法**

建议改成：

#### 1. Temporal downsampling

例如：

```python
every nth point
```

模拟低频采样。

---

#### 2. Day-level truncation

真正模拟：

- 1 day
- 3 day
- 7 day

而不是：
“随机保留 7%”。

---

#### 3. 多 seed 平均

现在只有：

```python
seed=42
```

reviewer 一定会质疑 variance。

至少：

```text
mean ± std over 5 seeds
```

否则 robustness 不成立。

---

### **B3：图像退化实验 —— 设计最合理，但结果暴露系统依赖 hatch mask**

这一组其实是最接近 publication-ready 的。

因为：
- blur
- contrast degradation
- trajectory deletion

都是真实 deployment degradation。

尤其：

```python
GaussianBlur(img, (7,7))
```

是标准 robustness protocol。

---

### **B3 结果里最重要的信息**

Blur 几乎没影响：

| Variant | CLS |
|---|---|
| Clean | 90.0 |
| Blur | 89.3 |

这说明：
系统不依赖高频 edge。

这是好事。

---

但：

| Variant | CLS |
|---|---|
| Reduced Hatch Contrast | 24.1 |

这个太危险。

尤其：

| FNA | 88.9 → 0 |
| Recall | 100% |
| CentroidR | 45% |

这意味着：

- detector 还在 detect
- 但 semantic naming 全崩了

即：

系统 semantic stage 完全依赖 hatch contrast cue。

reviewer 会直接质疑：

> “The method appears brittle to minor rendering variations.”

---

### **为什么这个结果“部分合理”**

合理部分：
- hatch mask 是核心 cue
- contrast degradation 会伤 segmentation

不合理部分：
- factor=0.3 太激进
- degradation 方式不真实

你这里：

```python
img * factor + mean * (1-factor)
```

实际上是在“整体塌缩动态范围”。

这比真实 compression artifact 更严重。

现实中更像：
- JPEG artifact
- partial fading
- scan noise
- illumination shift

而不是：
“整个 hatch 几乎消失”。

---

### **建议怎么修 B3**

建议：

#### 1. 更 realistic degradation

替换成：

- JPEG compression
- gamma shift
- additive noise
- partial occlusion

而不是全局 contrast collapse。

---

#### 2. 分析 semantic failure source

现在 reviewer 会问：

> 为什么 F1 只有小降，但 FNA=0？

说明：
- detection 还在
- naming pipeline 崩了

你应该主动解释：
- hatch feature 被 semantic classifier heavily used
- 但 geometry preserved

否则 reviewer 会认为：
系统 semantic stage 不鲁棒。

---

### **还有一个隐藏的大问题：evaluation leakage**

你代码里：

```python
existing_final = output/.../03_final
```

多个实验复用了 baseline。

这本身没问题。

但 reviewer 会担心：

- intermediate cache
- segmentation reuse
- hidden state

尤其：

```python
skip_seg=True
```

会导致：
- B1/B2 并非 end-to-end robustness
- 而是 partial robustness

所以论文里一定要写清：

```text
We isolate downstream robustness by reusing room segmentation outputs.
```

否则会被认为：
实验不公平。

---

### **最终结论**

你的实验：
- 框架设计是合理的
- 分类维度符合 UbiComp robustness evaluation 常规
- 代码组织达到论文 supplementary 水平

但结果目前暴露出三个真实问题：

| 实验 | 暴露问题 |
|---|---|
| B1 | 对 device coordinate 极端敏感 |
| B2 | 稀疏度模拟不真实且 variance 太高 |
| B3 | semantic stage 过度依赖 hatch cue |

如果现在直接投稿：
- reviewer 会认可“有做 robustness”
- 但不会认可“robust”

尤其 B1 会成为 major weakness。

真正适合投稿的版本应该是：

- B1 使用 correlated perturbation + soft matching
- B2 使用 temporal sparsity + multi-seed average
- B3 使用更 realistic image corruption benchmark
- 所有结果报告 mean ± std
- 增加 degradation-retention 曲线

这样才能从“做了实验”提升到“实验可信”。

*内容由 AI 生成仅供参考*