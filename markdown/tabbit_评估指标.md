
# 1. 实验组织方式

正确方式是：

```text
多个家庭（不同户型）
×
每个家庭运行完整pipeline
→
统计所有结果
→
计算平均值 ± 标准差
```

这是标准论文做法。

---

# 2. 为什么不能只用一个家庭

因为：

一个家庭可能：

- 轨迹特别清晰
- 家具特别规则
- 智能设备很多
- 房间结构简单

这种情况下：

你的结果可能"偶然很好"。

Reviewer 会认为：

```text
方法缺乏泛化性
```

---

# 3. 你需要的最小样本量

| 数量 | 可信度 |
|---|---|
| 1 | Demo |
| 3-5 | 非正式实验 |
| 8-10 | 小规模论文可接受 |
| 15-20 | 较可靠 |
| 30+ | 很强 |

---

# 4. 数据组织方式

## 数据集组成

自采集数据：

| Home ID | 户型 |
|---|---|
| Home01 | 一室一厅 |
| Home02 | 两室一厅 |
| Home03 | 三室两厅 |
| Home04 | 复式 |
| Home05 | 开放式 |
| ... | ... |

每个家庭的模型测试/预测生成输入包含：

- 扫地机 APP 图
- trajectory JSON
- smart device YAML

独立评分阶段额外读取人工标注的 GT YAML。模型生成阶段不得读取 GT，预测结果必须先冻结。

形成：

```text
自建 benchmark
```

很多 robotics / embodied AI 论文都用自建数据集，完全合理。

---

# 5. 实验结果应该是"平均值 ± 标准差"

正确写法：

| Metric | Mean ± Std |
|---|---|
| Furniture Naming Accuracy | 88.3% ± 4.1 |
| Furniture Detection F1 (IoU) | 85.0% ± 3.8 |
| Room Type Accuracy | 93.5% ± 2.7 |
| Centroid Recall | 93.0% ± 4.2 |
| CLS | 0.85 ± 0.03 |

你不仅要证明平均性能高，还要证明性能稳定。

---

# 6. 消融实验

对所有家庭运行每个变体，然后计算平均值：

| Variant | FNA | F1 | CLS |
|---|---|---|---|
| Full | 88.3 | 85.0 | 85.0 |
| -Trajectory | 74.5 | 75.2 | 73.5 |
| -Device | 81.0 | 81.5 | 79.8 |
| -RoomAgent (Rule-only) | 69.4 | 70.0 | 68.0 |
| -LLM Geometry Correction | 72.0 | 88.0 | 82.0 |
| CV Only (No LLM) | — | 90.0 | — |

这说明：

```text
trajectory 平均贡献 13.8%
smart device 平均贡献 7.3%
RoomAgent (LLM) 平均贡献 18.9%
LLM geometry correction 贡献了语义准确性但轻微损失召回
```

---

# 7. 评估匹配策略

## 两步匹配流程

```
Step 1: 房间匹配 (匈牙利算法, IoU > 0.2)
  GT 房间 ⇄ 预测房间 → 建立房间对应关系

Step 2: 家具匹配 (两种策略)
  A. Global (全局): 所有家具跨房间统一匈牙利匹配
  B. Aligned (房间内对齐): 先匹配房间, 每对房间内平移对齐后匹配家具
```

上述两步只在预测冻结后的独立评分器中执行。评分器可以读取 GT 以计算指标，但房间对应关系和偏移量不得反馈给模型、参与重新推理或修改预测 YAML。

## 策略 A: 全局匹配 (Global)

所有 GT 家具和所有预测家具跨房间做匈牙利匹配，不修正坐标偏移。

适用于：坐标系统已精确对齐的场景。

## 策略 B: 房间内对齐匹配 (Aligned)

1. 房间匹配后，计算每对房间的中心偏移 `(dx, dy)`
2. 将预测家具整体平移 `(dx, dy)` 对齐到 GT 坐标系
3. 在对齐后的坐标上，仅在匹配房间对内做匈牙利匹配

优势：消除像素→世界坐标转换的整体偏移，更公平。

```text
论文中应同时报告两种策略。A 衡量绝对坐标性能，B 衡量消除房间级整体平移后的性能；方法间只能在相同协议内比较。
```

---

# 8. 评估指标详解

## 8.1 Furniture Naming Accuracy (FNA) — 核心指标

### 定义

衡量：检测到的家具中，名称预测正确的比例。

### 公式

$$
FNA =
\frac{
N_{correct\_name}
}{
N_{TP}
}
$$

其中 `N_TP` = IoU > 0.2 的成功匹配家具数。

### 示例

GT: `bed, wardrobe, desk`

预测: `bed, cabinet, desk` (3 个家具都与 GT 匹配成功)

则:

$$
FNA = \frac{2}{3}=66.7\%
$$

### 意义

这是**语义推理能力**的核心体现，也是论文最重要的指标。

---

## 8.2 Furniture Detection F1 — 定位质量

### TP 判定标准

匈牙利匹配后，GT 家具与预测家具的 **IoU > 0.2** 视为检测成功 (TP)。
IoU 阈值设为 0.2 而非传统 0.5，是因为像素→世界坐标转换存在系统偏移，
同等绝对偏移对小面积家具影响更大。

### 8.2.1 Precision

$$
Precision=
\frac{TP}{TP+FP}
$$

- TP = 检测正确 (IoU > 0.2)
- FP = 预测了但 GT 没有 (误检)

### 8.2.2 Recall

$$
Recall=
\frac{TP}{TP+FN}
$$

- TP = 检测正确
- FN = GT 有但没检测到 (漏检)

### 8.2.3 F1 Score

$$
F1=
\frac{2PR}{P+R}
$$

F1 同时惩罚漏检 (Recall 低) 和误检 (Precision 低)，适合本系统。

---

## 8.3 Room Type Accuracy (RTA)

### 定义

衡量房间类型预测正确的比例。

### 公式

$$
RTA =
\frac{
N_{correct\_roomtype}
}{
N_{matched\_rooms}
}
$$

### 示例

GT: `bedroom, kitchen, bathroom`

预测: `bedroom, kitchen, living_room`

则:

$$
RTA=\frac{2}{3}=66.7\%
$$

### 意义

验证 RoomAgent 是否有效。Room Type 是 Naming Funnel 的重要条件。

---

## 8.4 Centroid Distance Recall — 补充定位指标

### 定义

以质心距离替代 IoU 判定检测成功，对小家具更公平。

### 公式

$$
CentroidDistance = \sqrt{(gt_{cx} - pred_{cx})^2 + (gt_{cy} - pred_{cy})^2}
$$

质心距离 ≤ 5dm 视为检测成功。

$$
CentroidRecall = \frac{N_{centroid\_tp}}{N_{gt\_furniture}}
$$

### 意义

同一绝对偏移下，大家具 IoU 下降少、小家具 IoU 下降多。
质心距离天然尺寸无关，能更公平地反映小家具的检测能力。

```text
Centroid Recall 与 IoU F1 互补：
- IoU F1 衡量边界框精度
- Centroid Recall 衡量位置检测率
```

---

## 8.5 按面积分层评估

### 分层标准 (单位: dm²)

| 类别 | 面积范围 | 典型家具 |
|------|---------|---------|
| small | < 50 | chair, nightstand, shoe_cabinet, toilet |
| medium | 50–200 | desk, dining_table, sofa, wardrobe |
| large | ≥ 200 | bed |

### 报告格式

| 面积分层 | GT数 | TP | Recall | FNA |
|---------|:--:|:--:|:--:|:--:|
| small (<50) | 11 | 9 | 81.8% | 77.8% |
| medium (50-200) | 6 | 6 | 100% | 83.3% |
| large (≥200) | 3 | 3 | 100% | 66.7% |

```text
小家具 Recall 偏低主要受 IoU 对小物体敏感的影响，
实际检测率 (Centroid Recall 95%) 验证了检测本身是成功的。
```

---

## 8.6 Complete Layout Score (CLS)

### 定义

综合衡量整体家庭重建质量。

### 公式

$$
CLS =
0.45A_{furniturename}
+
0.35F1_{furniture}
+
0.20A_{roomtype}
$$

### 各项含义

| 项 | 含义 |
|---|---|
| $A_{furniturename}$ | 家具命名准确率 (FNA) |
| $F1_{furniture}$ | 家具检测 F1-score |
| $A_{roomtype}$ | 房间类型准确率 (RTA) |

权重偏向语义家具恢复，这是本工作的主要目标。

### 推荐论文写法

> To evaluate the overall household reconstruction quality, we define a composite metric named Complete Layout Score (CLS), which jointly considers furniture semantic accuracy, furniture localization quality, and room semantic inference accuracy:
>
> $$CLS = 0.45A_{furniturename} + 0.35F1_{furniture} + 0.20A_{roomtype}$$
>
> where $A_{furniturename}$ denotes Furniture Naming Accuracy, $F1_{furniture}$ denotes Furniture Detection F1-score (IoU > 0.2), and $A_{roomtype}$ denotes Room Type Accuracy. The weights are chosen to emphasize semantic furniture recovery, which is the primary objective of this work.

### 为什么需要 CLS

避免单一指标片面评价。极端情况：系统只检测到 1 个家具但命名正确 → Naming Accuracy = 100%，但系统实际很差。
CLS 同时约束 detection、naming、room reasoning 三个维度。

---

## 8.7 推荐论文写法 (Furniture Detection)

> We evaluate furniture localization using Precision, Recall, and F1-score under one-to-one matching between predicted and ground-truth furniture instances.
> Room-level matching is first performed via Hungarian algorithm (IoU > 0.2) to establish room correspondences.
> Furniture matching then follows under two protocols:
> **(A) Global matching**, where all furniture instances are matched across rooms via Hungarian algorithm, and
> **(B) Per-room aligned matching**, where we first correct the room-level coordinate drift by centering each predicted room to its GT counterpart, then match furniture independently within each room pair.
> All prediction files are generated and frozen without access to test ground truth. Ground truth enters only the independent scorer; room correspondences and alignment offsets are never fed back to a prediction method.
> A prediction is considered correct if its IoU with a ground-truth instance exceeds 0.2.
> The lower IoU threshold (0.2 vs. conventional 0.5) is adopted because the pixel-to-world coordinate conversion introduces systematic offsets that disproportionately affect small furniture items.
> To provide a size-agnostic assessment, we additionally report **Centroid Distance Recall**, where a detection is counted as correct if the centroid deviation is within 5 dm.
> We further break down results by furniture area into small (< 50 dm²), medium (50–200 dm²), and large (≥ 200 dm²) categories, following the COCO evaluation protocol.

---
# 9. 最终指标组合 (推荐)

| 指标 | 用途 | 论文中的定位 |
|------|------|------------|
| Furniture Naming Accuracy (FNA) | 家具语义推理能力 | 核心指标 |
| Furniture Detection F1 (IoU) | 家具边界框定位质量 | 主要定位指标 |
| Room Type Accuracy (RTA) | 房间类型推理能力 | 辅助指标 |
| **Complete Layout Score (CLS)** | 整体家庭重建质量 | **综合指标** |
| Centroid Distance Recall / F1 | 尺寸无关的位置检测率 | 补充定位指标 |
| Per-size-class Recall / FNA | 不同面积家具的检测/命名公平性 | 分层分析 |

---

# 10. CLS 与 Centroid 指标的关系

## 核心原则

```
CLS 不变，F1_IoU 留在 CLS 里面。
Centroid Recall / F1 作为独立的补充分析，不参与 CLS 计算。
```

## 为什么不把 Centroid 指标放入 CLS

| 原因 | 说明 |
|------|------|
| **F1_IoU 是目标检测标准指标** | COCO、VOC 等权威 benchmark 均以 IoU 为判定标准，审稿人熟悉 |
| **CLS 应保留区分度** | Centroid 指标太宽松，若替代 IoU F1 进入 CLS，小家具的边界框误差完全消失 |
| **避免"刷分"嫌疑** | 换成 CentroidF1 后 CLS 仅涨 ~1.7 个百分点，审稿人会质疑动机 |
| **两者互补而非替代** | IoU 衡量"框的精度"，Centroid 衡量"位置对不对"，回答不同问题 |

## 论文中的呈现方式

### 主表（§4.x Main Results）

```text
CLS = 0.45 × FNA + 0.35 × F1_IoU + 0.20 × RTA
```

| Method | FNA ↑ | F1_IoU ↑ | RTA ↑ | **CLS ↑** |
|--------|:---:|:---:|:---:|:---:|
| Global | 76.5% | 81.0% | 100% | 82.7% |
| Aligned | 77.8% | 85.7% | 100% | **85.0%** |

### 补充分析表（§4.x Supplementary Localization Analysis）

专门一节解释 IoU 对小家具的不公平性 + 秀质心召回率。

| Protocol | Centroid Recall ↑ | Centroid F1 ↑ | small Recall (IoU) | medium Recall (IoU) | large Recall (IoU) |
|----------|:---:|:---:|:---:|:---:|:---:|
| Global | 85.0% | 81.0% | 72.7% | 100% | 100% |
| Aligned | **95.0%** | **90.5%** | **81.8%** | 100% | 100% |

### 配套论文文案

> **Supplementary Localization Analysis.**
> The IoU-based F1 score may underestimate detection quality for small furniture items, as an identical absolute coordinate offset produces a disproportionately larger IoU drop for objects with smaller area.
> To provide a size-agnostic complementary assessment, we additionally report Centroid Distance Recall, where a detection is considered correct if the centroid deviation between the predicted and ground-truth bounding box is within 5 dm.
> Under the per-room aligned matching protocol, Centroid Recall reaches **95.0%**, indicating that the system successfully localizes 19 out of 20 furniture items.
> The 9.3 percentage point gap between Centroid Recall (95.0%) and IoU Recall (85.7%) primarily stems from boundary box size estimation errors in small furniture categories, rather than missed detections.
> Breaking down by furniture area further confirms this: medium and large furniture achieve 100% IoU recall, while small furniture (< 50 dm²) reaches 81.8%.
> These results demonstrate that our pipeline excels at furniture **localization**, with boundary box **sizing** remaining as an avenue for future improvement.

# 11. 评估指标画图

Figure X: Performance comparison across five methods on room type accuracy (RTA), furniture detection F1 (F1@IoU>0.2), furniture naming accuracy (FNA), centroid distance recall (CentroidR), and complete layout score (CLS). Our method achieves the best CLS by combining precise CV-based geometry with LLM-based semantic reasoning.
