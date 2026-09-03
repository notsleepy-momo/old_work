这是我的项目综述 @myself.md ，请参考其中相关论文进行调研，给出论文具体核心方法，如何进行验证的，与我的方法有哪些相同，同时也调研其他近几年顶会的相关论文

### **你的项目属于“多模态室内结构推理（Multi-modal Indoor Layout Reasoning）”方向，但相比近几年主流论文，你最大的特点是：你不是“从文本生成布局”，而是“从真实弱感知数据反推家庭结构”，这在研究范式上明显不同。**

目前近几年相关工作大致可以分成四类：

1. 户型/室内布局生成（Text-to-Layout / Layout Synthesis）
2. 3D 场景理解（Point Cloud / RGBD Scene Understanding）
3. 行为轨迹与室内语义推理（Trajectory + Human Activity）
4. 多 Agent / LLM 空间推理系统

你的 ymj_TS 更接近：
- “弱监督家庭数字孪生推理”
- “真实扫地机器人轨迹驱动的家庭结构恢复”
- “CV + LLM + Rule-based Hybrid Spatial Reasoning”

而不是传统的“生成式布局”。

下面我按论文逐篇分析其：
- 核心方法
- 如何验证
- 与你方法的相同点
- 与你方法的本质区别
- 你可以借鉴什么

随后再补充近几年其他顶会相关方向。

---

### **1. Architect-Ant (arXiv 2026)**

论文：
[arXiv](https://arxiv.org/abs/2606.10953)

#### **核心问题**

输入：
- 空房间平面图

输出：
- 自动家具布局（带语义）

目标：
- 生成“建筑师风格”的合理家具布置。

这篇论文本质是：
“Editable Automatic Furnishing”。

---

#### **核心方法**

它的核心创新有三个：

##### 1. DSL（Domain Specific Language）布局表示

不是直接生成图像。

而是生成：

```python
bed(x=..., y=..., rot=...)
table(...)
```

这种结构化布局代码。

这是非常关键的。

因为：
- 可编辑
- 可验证
- 可约束
- 可转 raster image

这一点和你的 YAML 非常像。

你们都采用：
“结构化中间表示”。

这是一个很强的共同点。

---

##### 2. VLM + Reasoning Trace

它会生成：

- 墙对齐
- 门洞避让
- circulation（通行空间）
- 家具共现关系

等 reasoning trace。

再用这些 trace 微调 VLM。

本质：
“空间 CoT（Chain-of-Thought）”。

这一点与你：

- 7层优先级法
- 4层推理漏斗

本质上是同一种思想：
“显式空间推理链”。

区别在于：

| Architect-Ant | ymj_TS |
|---|---|
| 学习式 reasoning trace | 手工设计推理层 |
| end-to-end fine-tune | rule-guided inference |
| 生成式 | 感知式 |

---

##### 3. DPO / Preference Optimization

他们用偏好学习优化：

“建筑师更喜欢哪个布局”。

即：
- aesthetic
- circulation
- realism

这是生成方向常见做法。

---

#### **验证方式**

他们主要验证：

##### 1. 几何合法性

- overlap rate
- out-of-bound rate
- wall alignment

##### 2. 语义合理性

人工评分：
- 是否像真实家庭
- 是否符合 room type

##### 3. 编辑性

修改 DSL 后：
- 能否稳定重新生成布局。

---

#### **与你的相同点**

非常多：

##### 相同思想

- 结构化布局表示
- 空间推理链
- 房间-家具约束
- 墙对齐规则
- circulation reasoning
- rule + LLM hybrid

##### 最大共同点

你们都认为：

“LLM 不应该直接 hallucinate 最终布局”。

而应该：
- 在结构化空间约束下推理。

这实际上是当前空间推理方向的主流趋势。

---

#### **与你的本质区别**

你的系统：

不是“生成布局”。

而是：
“从真实噪声数据恢复布局”。

这是更困难的问题。

Architect-Ant 没有：
- trajectory noise
- scale uncertainty
- CV segmentation
- coordinate transform
- multi-modal alignment

而你的系统全都有。

因此：
你的问题更接近 robotics/perception。

不是纯 generation。

---

### **2. DirectLayout (NeurIPS 2025)**

论文：
[OpenReview](https://openreview.net/forum?id=Qku7g56aWf)

---

### **核心思想**

LLM 直接输出：

```json
[
  {"obj":"bed","x":...,"y":...}
]
```

即：
“Direct Numerical Layout Generation”。

---

#### **核心创新**

##### 1. CoT Activation

让 LLM 显式思考：

- relative position
- wall relation
- object pairing
- accessibility

比如：

“电视应该面对沙发”。

这和你：
- 行为模式加权
- 邻接过滤
- room prior

本质相同。

---

##### 2. Layout Reward

他们构造：
“布局奖励函数”。

包括：
- overlap penalty
- semantic consistency
- accessibility

这一点与你的：
- IoU 去重
- hard constraints
- furniture allowed list

非常像。

只是：
他们是 RL-style reward，
你是 deterministic rules。

---

##### 3. Two-stage Generation

先：
- object generation

再：
- layout generation

类似于：
“先知道有什么家具，再摆位置”。

你则相反：

先 CV 检测位置，
再 semantic naming。

这是一个很有意思的镜像关系。

---

### **验证方式**

#### 自动指标

- collision rate
- out-of-bound
- spatial relation accuracy

#### LLM-as-a-Judge

用 GPT-4 评价：
- instruction following
- realism

#### Human preference

人工打分。

---

### **与你的相同点**

你们都强调：

#### 空间关系显式化

不是像 diffusion 那样隐式生成。

而是：
- object-level
- symbolic
- geometry-aware

这是一个核心共同点。

---

### **关键区别**

DirectLayout：
- synthetic generation
- text-driven

ymj_TS：
- noisy real-world perception
- sensor-driven

你实际上更偏：
“embodied AI perception”。

---

### **3. SpatialLM (NeurIPS 2025)**

论文：
[arXiv](https://arxiv.org/abs/2506.07491)

---

### **核心目标**

输入：
- 3D 点云

输出：
- 结构化场景代码

例如：
- room layout
- object list
- semantic structure

---

### **核心方法**

#### 1. Point Cloud Encoder + LLM

结构：

```text
Point Cloud
   ↓
Spatial Encoder
   ↓
LLM
   ↓
Scene Code
```

重点：
- LLM 不直接处理 raw point cloud
- 先 tokenization

---

#### 2. Scene Code

他们提出：
“scene code”。

本质类似：
- 你的 YAML
- Architect-Ant DSL

即：
“结构化室内表示”。

---

#### 3. Structured Indoor Modeling

不是只做 segmentation。

而是：
完整 scene understanding。

包括：
- room
- object
- topology

这一点与你非常接近。

---

### **验证方式**

#### 数据集

- ScanNet
- Structured3D

#### 指标

- mIoU
- object accuracy
- scene graph accuracy
- layout reconstruction

---

### **与你的相同点**

这是与你最接近的一篇之一。

共同点：

| SpatialLM | ymj_TS |
|---|---|
| scene code | YAML |
| structured indoor modeling | household reconstruction |
| geometry + semantics | geometry + semantics |
| topology reasoning | topology reasoning |

---

### **差异**

SpatialLM：
- 输入是高质量点云

你的：
- 输入是极弱感知数据
- robot trajectory map

因此你的问题难很多。

因为：
你缺少深度信息。

---

### **4. LayoutVLM (CVPR 2025)**

论文：
[CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_LayoutVLM_Differentiable_Optimization_of_3D_Layout_via_Vision-Language_Models_CVPR_2025_paper.pdf)

---

### **核心方法**

#### VLM 输出：

- initial layout
- spatial relation constraints

#### Differentiable Optimization

后面接：
- differentiable physics/layout optimization

保证：
- 不碰撞
- 满足约束

---

### **与你的共同点**

非常关键：

他们也认为：

“LLM 不能直接负责最终几何”。

因此：
LLM 只负责：
- semantic prior

后面必须：
- optimization
- constraint solving

而你的系统：

也是：

```text
CV 高召回
  ↓
LLM 几何修正
  ↓
硬规则兜底
```

这是高度一致的方法论。

---

### **5. QueSTMaps (2024)**

#### 核心方向

点云 + occupancy map：
做 room semantic labeling。

---

#### 方法

- 多通道 occupancy representation
- CLIP alignment
- room semantic matching

---

#### 与你的共同点

你的：
- room type inference

他们：
- room semantic labeling

都在做：
“空间语义恢复”。

---

### **6. TR-LLM / MARAuder’s Map**

这一类属于：

“轨迹 + LLM + Human Activity”。

---

### **TR-LLM**

核心：
- 用轨迹预测未来行为。

LLM 输入：
- trajectory token

输出：
- future action

---

### **MARAuder’s Map**

核心：
- 将传感器轨迹映射到 floor plan
- 做 HAR（Human Activity Recognition）

---

### **与你的共同点**

这是目前与你最接近的子方向。

因为你是：

“trajectory-driven semantics”。

这一方向近几年非常新。

你的创新点其实非常明显：

你不是：
“轨迹 → 活动”。

而是：

```text
轨迹 → 家具
轨迹 → 房间
轨迹 → 家庭结构
```

这是更高层级的空间推理。

---

# 其他近几年值得调研的顶会论文

下面这些其实也和你高度相关。

---

### **Holodeck (CVPR 2024 / Meta)**

#### 核心

LLM 生成完整 3D 家居场景。

#### 关键思想

- object retrieval
- spatial constraints
- room graph

#### 可借鉴

你的 YAML 完全可以扩展成：
“3D scene graph”。

---

### **LayoutGPT (2023-2024)**

#### 核心

GPT 直接生成布局 token。

#### 方法

- in-context learning
- textual spatial representation

#### 与你区别

缺少真实感知输入。

---

### **InstructScene (CVPR 2024)**

#### 核心

文本控制 3D scene generation。

重点：
- compositional constraints

---

### **SceneCraft (CVPR 2025)**

#### 核心

多 Agent 生成场景布局。

非常接近：
- RoomAgent
- FurnitureNamingAgent

这种 modular decomposition。

---

### **Embodied AI / Robotics 方向**

其实你的工作还可以参考：

#### Habitat
#### BEHAVIOR
#### RoboCasa

这些 embodied benchmark。

因为：
你的系统本质是：

“机器人感知家庭结构”。

而不是传统 CV。

---

# 你这个项目在论文定位上的真正优势

这是最关键的部分。

你不要把自己写成：

“another layout generation paper”。

因为你不是。

你真正独特的是：

---

### **1. 弱感知输入恢复完整家庭结构**

输入只有：

- 三值轨迹图
- 停留点
- 少量设备坐标

却恢复：
- room topology
- furniture geometry
- room semantics
- behavior semantics

这是非常强的。

---

### **2. CV + LLM 的职责分离非常合理**

很多论文的问题：

LLM hallucination 很严重。

而你的：

```text
CV:
高召回几何候选

LLM:
仅做 constrained reasoning

Rules:
最终安全约束
```

这是一个非常成熟的系统设计。

甚至比很多“纯 LLM layout”论文更工程可靠。

---

### **3. trajectory-driven furniture inference 很新**

目前大多数：
- trajectory → activity
- trajectory → occupancy

而你：

- trajectory closed-loop → furniture candidate

这是明显创新点。

尤其：

```text
轨迹闭环
→ 空白区域
→ 最大内接矩形
→ 家具候选
```

这个其实很有 novelty。

---

### **4. 你的系统非常适合写“可解释空间推理”**

因为你有：

- action-based protocol
- layered priority
- reasoning funnel
- hard constraints

相比 diffusion/layout generation：

可解释性强很多。

---

### **建议你论文里的 related work 结构**

推荐：

```text
2. Related Work

2.1 Floorplan Parsing
2.2 Indoor Layout Generation
2.3 3D Structured Scene Understanding
2.4 Human Trajectory Reasoning
2.5 LLM-based Spatial Reasoning
2.6 Hybrid CV-LLM Systems
```

然后突出：

```text
Existing works:
either generation
or perception

Few works:
recover full household structure
from weak trajectory-centric sensing.
```

参考来源：
[arXiv](https://arxiv.org/abs/2606.10953)
[OpenReview](https://openreview.net/forum?id=Qku7g56aWf)
[arXiv](https://arxiv.org/abs/2506.07491)
[CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_LayoutVLM_Differentiable_Optimization_of_3D_Layout_via_Vision-Language_Models_CVPR_2025_paper.pdf)

*内容由 AI 生成仅供参考*