### **你的项目最合理的准确率验证方式，是“分层验证 + 拓扑验证 + 消融验证”三套体系同时做。**

因为 ymj_TS 不是单一任务。

它同时包含：

- 房间分割
- 家具检测
- 家具命名
- 房间语义推理
- 行为分析
- 拓扑恢复

所以不能只用一个 mAP 或 Accuracy。

你需要建立：
“多层级评估体系”。

你现在已经有：
- 真实扫地机轨迹图
- GT 手工标注 YAML

这已经足够做完整评估。

下面我给你一个最适合论文的方案。

---

# 1. 总体验证框架（推荐直接写论文）

建议论文实验部分：

```text
4. Experiments

4.1 Dataset and Ground Truth
4.2 Evaluation Metrics
4.3 Overall Quantitative Results
4.4 Ablation Study
4.5 Robustness Analysis
4.6 Error Analysis
```

核心是：

```text
Evaluation Metrics
```

这里决定论文质量。

---

# 2. 房间级（Room-level）准确率

这是最重要的第一层。

---

## 2.1 Room Detection IoU

验证：

```text
预测房间位置
vs
GT房间位置
```

使用：

$$
IoU = \frac{Area(P \cap G)}{Area(P \cup G)}
$$

其中：

- P = predicted room
- G = ground truth room

---

### 推荐参数

| 参数 | 推荐值 |
|---|---|
| IoU threshold | 0.5 |
| 严格版本 | 0.7 |

---

### 输出指标

你应该统计：

| 指标 | 说明 |
|---|---|
| Room IoU mean | 平均IoU |
| Room Recall | GT房间被找到比例 |
| Room Precision | 预测房间正确比例 |
| Room F1 | 综合指标 |

---

### 示例

```text
Bedroom_01:
GT box = [0,0,30,40]
Pred = [1,2,31,39]

IoU = 0.87
```

---

# 3. 房间类型准确率（Room Semantic Accuracy）

这是：

```text
bedroom / kitchen / bathroom
```

是否正确。

---

## 指标

### Top-1 Accuracy

$$
Accuracy = \frac{CorrectRooms}{TotalRooms}
$$

---

### 推荐增加：

#### confusion matrix

特别重要。

因为：

最容易错的是：

| GT | Pred |
|---|---|
| dining room | living room |
| studyroom | bedroom |
| bathroom | laundry |

你需要：

```text
Confusion Matrix
```

作为论文图。

---

# 4. 家具检测准确率（Furniture Detection）

这是第二大核心。

---

## 4.1 Furniture IoU

和目标检测一样。

---

### 推荐 IoU threshold

| 阈值 | 用途 |
|---|---|
| 0.5 | 标准 |
| 0.75 | 严格 |
| AP@[0.5:0.95] | COCO风格（高级） |

---

## 4.2 Precision / Recall

特别适合你的：

因为你强调：

```text
CV 高召回
```

所以：

Recall 非常关键。

---

### 公式

$$
Precision = \frac{TP}{TP+FP}
$$

$$
Recall = \frac{TP}{TP+FN}
$$

---

### 你的论文里应该强调：

```text
本方法优先保证 Recall，
再通过 LLM 和规则降低 FP。
```

这是你的核心思想。

---

# 5. 家具命名准确率（Furniture Naming Accuracy）

这个是：

```text
bed / sofa / wardrobe
```

是否正确。

---

## 指标

### 分类 Accuracy

$$
Accuracy = \frac{CorrectFurnitureNames}{DetectedFurniture}
$$

---

## 推荐增加：

### Top-k Accuracy

因为：

很多家具本身模糊：

- sofa vs lounge chair
- cabinet vs wardrobe

所以：

| 指标 | 推荐 |
|---|---|
| Top-1 | 必须 |
| Top-3 | 推荐 |

---

# 6. 拓扑结构准确率（非常重要）

这是你的亮点。

很多论文没有。

---

# 6.1 Room Adjacency Accuracy

验证：

```text
房间连通关系
```

例如：

```text
kitchen connected to dining room
```

---

## 表示方式

GT：

```python
Adj_GT = {
  (kitchen, dining),
  (living, bedroom)
}
```

预测：

```python
Adj_Pred = {...}
```

---

## 指标

### Graph Edge Accuracy

$$
Accuracy = \frac{|E_{pred} \cap E_{gt}|}{|E_{gt}|}
$$

---

# 6.2 Door Connectivity Accuracy

验证：

门是否连接正确房间。

这个很适合你的项目。

因为你有：

```text
doors + walls + topology
```

---

# 7. 布局整体相似度（高级指标）

这个非常适合论文。

---

# 7.1 Hausdorff Distance

比较：

整体布局轮廓。

---

## 适合验证：

```text
整体户型恢复质量
```

---

# 7.2 Graph Edit Distance

把家庭布局看成：

```text
Scene Graph
```

节点：
- room
- furniture

边：
- adjacency
- containment

---

## 验证：

预测图距离 GT 图多远。

这是很高级的做法。

非常适合：

CVPR / NeurIPS 风格。

---

# 8. 行为分析准确率（可选）

这个难定量。

建议：

---

## 方法1：人工标注

GT：

```yaml
bedroom:
  main_activity: sleeping
```

预测：

```yaml
sleeping
```

计算 Accuracy。

---

## 方法2：LLM-as-a-Judge

GPT-4o 判断：

```text
预测行为是否合理
```

---

# 9. 消融实验（必须）

这是论文最重要部分之一。

---

## 推荐参数

| Variant | 去掉什么 |
|---|---|
| Full | 完整 |
| -LLM | 去掉LLM几何修正 |
| -Trajectory | 去掉行为轨迹 |
| -Device | 去掉智能设备 |
| CV-only | 只有CV |
| LLM-only | 只有VLM |

---

## 重点观察

### Furniture Recall

去掉 trajectory 后：

Recall 是否下降。

这是验证：

```text
trajectory_surrounded
```

是否真的有效。

---

# 10. 鲁棒性实验（非常推荐）

这个能大幅提高论文质量。

---

# 10.1 轨迹稀疏度

随机删：

| 删除比例 |
|---|
| 20% |
| 50% |
| 80% |

观察：

- Room Acc
- Furniture Recall

---

# 10.2 设备缺失率

隐藏：

| 覆盖率 |
|---|
| 100% |
| 50% |
| 20% |
| 0% |

观察：

Kitchen / Bathroom 准确率。

---

# 10.3 噪声实验

给轨迹加入：

```python
Gaussian Noise
```

验证鲁棒性。

---

# 11. 最推荐你使用的最终指标（论文可直接采用）

下面这个组合最合理。

---

## 房间层

| 指标 |
|---|
| Room IoU |
| Room Type Accuracy |
| Room Recall |
| Room Precision |
| Room F1 |

---

## 家具层

| 指标 |
|---|
| Furniture IoU@0.5 |
| Furniture Recall |
| Furniture Precision |
| Furniture F1 |
| Furniture Naming Accuracy |

---

## 拓扑层

| 指标 |
|---|
| Room Adjacency Accuracy |
| Door Connectivity Accuracy |
| Graph Edit Distance |

---

## 整体层

| 指标 |
|---|
| Layout Hausdorff Distance |
| End-to-End Layout Accuracy |

---

# 12. 最关键：你的“End-to-End”指标

这是你区别于别人的地方。

建议定义：

---

## Complete Layout Score (CLS)

例如：

$$
CLS =
0.25R_{room}
+
0.25R_{furniture}
+
0.2A_{roomtype}
+
0.15A_{furniturename}
+
0.15A_{topology}
$$

这是：

“完整家庭恢复质量”。

很多顶会喜欢这种：

Composite Metric。

---

# 13. 你最应该重点展示的结果

不要只强调：

```text
分类准确率
```

你真正强的是：

---

## 1. Recall 高

因为：

```text
CV 高召回
```

是你核心设计。

---

## 2. 拓扑恢复能力

你的：

- room adjacency
- door relation
- furniture containment

比很多生成论文更真实。

---

## 3. 弱感知恢复能力

你的输入非常弱：

- trajectory
- sparse device prior

却恢复：

完整 household structure。

这是最大亮点。

---

# 14. 论文里最建议画的图

---

## 图1：完整Pipeline

你已经有了。

---

## 图2：消融实验柱状图

例如：

```text
Furniture Recall
```

对比：

- Full
- -Trajectory
- -LLM
- CV-only

---

## 图3：Confusion Matrix

房间类型。

---

## 图4：Topology Recovery Example

GT vs Pred。

---

## 图5：轨迹稀疏度鲁棒性曲线

非常像顶会论文。

---

# 15. 最推荐的最终评价指标组合（最重要）

如果你时间有限。

至少做：

| 必做 | 指标 |
|---|---|
| ✅ | Room IoU |
| ✅ | Room Type Accuracy |
| ✅ | Furniture IoU@0.5 |
| ✅ | Furniture Recall/F1 |
| ✅ | Furniture Naming Accuracy |
| ✅ | Adjacency Accuracy |
| ✅ | 消融实验 |
| ✅ | 轨迹稀疏鲁棒性 |

这已经足够形成完整论文实验体系。

*内容由 AI 生成仅供参考*