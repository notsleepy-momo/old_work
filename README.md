# ymj_TS — 基于机器人轨迹拓扑的家庭轮廓智能推理系统

从扫地机器人 APP 图、智能设备坐标、和人的行为轨迹中，自动推理出完整的家庭布局 YAML：包含所有房间、家具、名称和尺寸。

核心思路：**CV 负责高召回 + LLM 负责约束修正 + 代码硬规则兜底**，5 步管线串联。

---

## 项目介绍

### 目标

给定三类异构输入，全自动生成一份完整的家庭布局 YAML：

| 输入 | 来源 | 说明 |
|------|------|------|
| 户型图 | 扫地机器人 APP 建图结果（彩色，经预处理转为灰度） | 墙=0 / 地板=128 / 机器人行驶轨迹=255，含房间轮廓和机器人行驶轨迹 |
| 智能设备坐标 | Smart_device YAML | 已知家具（冰箱、电视等）的世界坐标位置 |
| 人的行为轨迹 | trajectory JSON | 多天/多人的停留点数据，含停留时长和位置 |

输出 `output/yaml/03_final_{code}.yaml` 包含：

- 所有房间：id、position（世界坐标）、room_type（卧室/厨房/客厅...）、walls、doors
- 所有家具：id、position（世界坐标）、name（床/沙发/水槽...）、confidence
- 行为分析：每个房间的 main_activities、frequent_areas、time_based_patterns

### 核心方法

```
户型图 (经预处理的扫地机 APP 建图结果)
    │
    ├─ Step 1: CV 房间分割 (Hough 线检测 → Union-Find 网格合并 + 门检测 + hatch 阴影提取)
    │
    ├─ Step 2: CV 双源家具召回 + LLM 几何修正 + 智能设备匹配 + 坐标统一
    │     ├── CV: hatch 阴影内接矩形 (红框, confidence=0.90)
    │     ├── CV: 机器人轨迹闭环包围区检测 (蓝框, 互补召回)
    │     ├── 置信度融合 + IoU 去重 + 结构特征注解
    │     ├── LLM: action-based 约束修正 (merge/delete/adjust, 不发明家具)
    │     ├── 智能设备: 像素坐标匹配家具框 + 多设备框均分
    │     └── 坐标统一: 像素 → 世界坐标
    │
    ├─ Step 3: LLM 房间类型推理 (7 层优先级: 无家具→智能设备→阳台→行为→连通性→bathroom→缺失分配)
    │
    ├─ Step 4: LLM 行为模式分析 (主要活动/高频区域/时段模式)
    │
    └─ Step 5: LLM 家具命名 (4 层推理漏斗: 房间类型→尺寸形状→位置邻接→行为模式) + 最终合并输出
```

**设计要点：**
- CV 阶段高召回（宁多勿漏），双源互补：hatch 阴影适合规则大家具，机器人轨迹闭环覆盖非规则/阴影弱区域
- LLM 采用 constrained action protocol：只能对 CV 候选执行 merge/delete/adjust，禁止发明新家具
- 红框（hatch）不删除不合并，蓝框（robot trajectory loop）保守合并
- 代码硬规则处理门遮挡、蓝框去重（与红框重叠即删除）、强信号自动合并
- 蓝框合并引入机器人轨迹拓扑语义：独立圆环簇（椅子，loop_cluster_count≥3）不与大矩形（桌子）合并

---

## 项目结构

```
ymj_TS/
├── main.py                          ★ 唯一入口 — python main.py
├── pipeline.py                      Pipeline 类（5 步编排 + 消融控制）
├── evaluation.py                    ★ 评估系统 (RTA / F1 / FNA / CentroidR / CLS)
├── visualize_layout.py              YAML → PNG/PDF 可视化
├── layout_viewer.html               交互式户型查看器（浏览器）
│
├── agents/                          核心 Agent 模块
│   ├── furniture_detection_agent.py  Step 2: CV + LLM 混合家具检测 (~2300行)
│   │   ├── _extract_hatch_furniture_rects()         红框: hatch 内接矩形分解
│   │   ├── _detect_trajectory_surrounded_regions()  蓝框: 机器人轨迹闭环包围区检测
│   │   ├── _fuse_and_dedup_candidates()             置信度融合 + IoU 去重
│   │   ├── _annotate_structure_features()           只读结构特征注解
│   │   ├── _auto_merge_strong_signal_pairs()        代码确定性强合并
│   │   ├── _expand_hatch_to_trajectory()            红框向机器人轨迹边界扩展
│   │   ├── _match_devices_to_furniture()            智能设备→家具框匹配
│   │   ├── _parse_room_furniture_actions()          LLM action 解析
│   │   └── _process_room()                          单房间完整处理
│   ├── room_agent.py                Step 3: 房间类型推理（7 层优先级 + 消融支持）
│   ├── behavior_agent.py            Step 4: 行为模式分析
│   ├── furniture_naming_agent.py    Step 5: 家具命名（4 层推理漏斗 + 消融支持）
│   ├── coordinate_converter.py      像素 ↔ 世界坐标双向转换
│   ├── data_models.py               Pydantic 数据模型
│   └── prompts.py                   所有 LLM 提示词模板
│       ├── FURNITURE_DETECTION_AGENT_PROMPT   action-based 协议
│       ├── ROOM_AGENT_PROMPT                  7 层优先级 prompt
│       ├── BEHAVIOR_AGENT_PROMPT              行为分析 prompt
│       └── FURNITURE_AGENT_PROMPT             4 层漏斗 prompt
│
├── config/                          配置文件（代码通用，外部可修改）
│   ├── __init__.py                  配置加载器（带缓存）
│   ├── smart_devices.yaml           智能设备→房间类型映射（含优先级）
│   └── furniture_allowed.yaml       家具命名允许列表（按房间类型）
│
├── baselines/                       对比实验 & 消融实验
│   ├── compare_baselines.py         对比实验主脚本（5 方法对比）
│   ├── ablation_experiments.py      ★ 消融实验主脚本（2 组核心消融）
│   ├── vlm_baseline.py              VLM Zero-shot & Chain-of-Thought 基线
│   ├── cv_only_baseline.py          CV-Only (无 LLM) 基线
│   ├── plot_comparison.py           论文对比柱状图（PDF/PNG）
│   └── plot_ablation.py             消融实验柱状图（PDF/PNG）
│
├── photo2yaml/                      Step 1: 房间分割
│   ├── run_seg.py                   Hough 线检测 → 网格合并 → YAML
│   └── extract_hatch.py             斜线阴影提取（Gabor 滤波器）
│
├── input/                           输入数据
│   ├── photo/room_0622.png          户型图（彩色扫地机 APP 建图结果，预处理后为灰度）
│   ├── Smart_device/smartDevice_0622.yaml  智能设备坐标
│   └── traj/0622/                   多天行为轨迹 JSON + 合并脚本
│
├── GT/                              Ground Truth 标注
│   └── layout_0622.yaml             含完整 rooms + furniture + doors 的世界坐标 YAML
│
├── output/                          ★ 统一输出目录
│   ├── yaml/                        01_seg_pixel → 02_furniture_world → 03_final
│   ├── masks/                       CV 中间可视化图（房间裁剪/热度图/候选/全图）
│   ├── baselines/                   VLM / CV-Only 基线预测结果
│   ├── evaluation/                  评估结果 & 对比实验输出 & 消融结果
│   ├── ablation/                    各消融变体独立输出子目录
│   ├── viz/                         全图可视化
│   └── figures/                     论文图表（PDF/PNG）
│
├── markdown/                        论文设计文档
│   ├── tabbit_消融实验设计.md        消融实验设计指南
│   ├── tabbit_评估指标.md           评估指标详细定义
│   └── tabbit_论文结构.md           论文结构规划
│
└── realroom/                        真实房间参考图片
```

---

## 安装

```bash
# 要求 Python 3.10+
pip install opencv-python numpy pyyaml langchain-openai scikit-learn
```

### API Key 配置

创建 `.env` 文件（支持 UTF-8 或 UTF-16 LE 编码）。所有 Agent 统一使用
`gpt-5.6-sol` 和同一个 API key：

```bash
FURNITURE_API_KEY=your-api-key
FURNITURE_BASE_URL=https://your-furniture-api-url/v1
```

---

## 快速开始

### 全量运行（默认输入）

```bash
cd ymj_TS
python main.py
```

使用 `input/` 目录下的默认数据：`room_0622.png` + `smartDevice_0622.yaml` + `0622-Engineer-all_trajectory.json`。

### 指定输入

```bash
python main.py \
  --image input/photo/room_0622.png \
  --smart-device input/Smart_device/smartDevice_0622.yaml \
  --trajectory input/traj/0622/0622-Engineer-all_trajectory.json \
  --output-dir output
```

### 增量运行

```bash
# 跳过房间分割（使用已有的 output/yaml/01_seg_pixel_0622.yaml）
python main.py --skip-seg

# 仅做分割 + 坐标转换（不调用 LLM）
python main.py --skip-agents
```

---

## 5 步管线数据流

```
输入                                    处理                                   输出 (output/)
─────────────────────────────────────────────────────────────────────────────────────────
户型图 (room_0622.png)   → Step 1: run_seg.py                            → yaml/01_seg_pixel_0622.yaml
                          │  (Hough线检测 + Union-Find网格合并            (像素坐标，含房间+墙+门
                          │   + 门检测 + SCALE计算 + hatch提取)            + metadata SCALE/bl)
                          │                                                (输入为彩色 APP 建图结果，经预处理转灰度)
                          │
                          → Step 2: FurnitureDetectionAgent + Converter  → yaml/02_furniture_world_0622.yaml
                          │  CV:  hatch阴影内接矩形(红框, conf=0.90)       (世界坐标，含智能设备名称)
                          │      + 机器人轨迹闭环包围区(蓝框, 互补召回)          → masks/ (CV中间图)
                          │      + 置信度融合 + IoU去重
                          │      + 结构特征注解(loop/arc/rect/size)
                          │  LLM: action-based 约束修正
                          │  Device: 智能设备匹配 + 多设备框均分
                          │  Converter: 像素→世界坐标统一
                          │
智能设备 YAML             → Step 3: RoomAgent (LLM)                      → (内存: room_analyses)
行为轨迹 JSON            │  (7层优先级: 无家具→other → 智能设备确定
                          │   → 阳台检测 → 行为模式 → 连通性
                          │   → bathroom推断 → 缺失类型分配)
                          │
                          → Step 4: BehaviorAgent (LLM)                  → (内存: behavior_analyses)
                          │  (主活动/高频区域/时段模式)
                          │
                          → Step 5: FurnitureNamingAgent (LLM) + Merge  → yaml/03_final_0622.yaml
                             (智能设备名称保护 → 4层漏斗命名               → viz/ (全图可视化)
                              → 合并最终输出)
```

---

## 配置系统

项目使用外部 YAML 配置文件，可脱离代码修改：

### `config/smart_devices.yaml` — 智能设备→房间类型映射

| 设备名 | 优先级 | 映射房间类型 |
|--------|--------|------------|
| Fr / Refrigerator / Stove / Oven / Microwave | 5 (确定) | kitchen |
| Washing Machine | 5 (确定) | bathroom |
| TV / TV Cabinet / TV Stand / Treadmill | 3 (歧义) | [living_room, bedroom] 等多类型 |

优先级 ≥4 的设备直接确定房间类型。这些设备名称同时也作为**名称保护名单**——已有智能设备名的家具不参与 LLM 命名。

### `config/furniture_allowed.yaml` — 家具命名允许列表

| 房间类型 | 允许的家具名称 |
|---------|--------------|
| bathroom | toilet, washbasin, shower |
| kitchen | sink, cabinet, dining_table |
| bedroom | bed, wardrobe, nightstand, desk, chair, coat_rack |
| living_room | sofa, coffee_table, dining_table, shoe_cabinet, chair |
| studyroom | desk, bookshelf, chair, bed |
| diningroom | dining_table, chair |
| other | table, chair, shelf |

---

## 评估系统

### 评估指标

| 指标 | 缩写 | 公式 / 说明 |
|------|------|------------|
| Room Type Accuracy | RTA | 类型正确且定位匹配的房间数 / GT 房间总数 |
| Room Semantic F1 | Room F1 | 房间语义 Precision 与 Recall 的调和平均，同时惩罚漏检和多检 |
| Furniture Detection F1 | F1@IoU>0.2 | 匈牙利匹配后 IoU>0.2 为 TP |
| Semantic F1 | SemF1 | IoU 达标且名称正确才算 TP 的端到端家具语义 F1 |
| Conditional Naming Accuracy | CondFNA | 定位 TP 中名称正确数 / 定位 TP 总数，仅用于错误归因 |
| Centroid Distance Recall | CentroidR | 质心距离 ≤5dm 的检测率（对小家具更公平） |
| Legacy Complete Layout Score | CLS | 0.45×FNA + 0.35×F1 + 0.20×RTA；权重未经验证，仅保留为探索性兼容指标 |

支持**按面积分层**统计（small <50dm² / medium 50-200 / large ≥200）。

### 两种评估策略

| 策略 | 说明 |
|------|------|
| A. 全局匹配 (Global) | 使用预测的绝对坐标，将所有家具跨房间统一匈牙利匹配 |
| B. 房间内对齐匹配 (Aligned) | 先匹配房间，再在每对房间内平移对齐后匹配家具，用于衡量消除房间级整体平移后的性能 |

模型测试/预测生成阶段不允许读取 GT，输出 YAML 必须先冻结。随后独立评分器读取 GT 计算指标，并对每种方法同时应用 A 和 B；B 的房间匹配和偏移量不得反馈给模型或用于修改预测。方法之间只能在相同协议下比较。

### 评估单个预测

```bash
python evaluation.py \
  --pred output/yaml/03_final_0622.yaml \
  --gt GT/layout_0622.yaml \
  --output output/evaluation/evaluation_0622.txt
```

### 批量评估

```bash
python evaluation.py --pred_dir output/yaml/ --gt_dir GT/
```

---

## 对比实验 (Comparison with Baselines)

运行四种方法的公平对比，并为每种方法分别输出 Global 与 Aligned 结果：

```bash
# 完整运行（含 VLM API 调用，耗时较长）
python baselines/compare_baselines.py --code 0622

# 仅评分（跳过 API 调用，使用已有基线输出）
python baselines/compare_baselines.py --code 0622 --skip-vlm

# 批量对比
python baselines/compare_baselines.py --code 0622 --code 0701
```

### 对比方法

| 方法 | 说明 |
|------|------|
| VLM Zero-shot | 户型图直接输入 VLM，一次 Prompt 输出完整 YAML |
| VLM Chain-of-Thought | 分 3 步：房间检测 → 家具定位 → 按家具 ID 命名；后一步不能改写前两步几何 |
| CV-Only (No LLM) | 纯 CV 检测 + 规则推断（智能设备映射 + 面积/连通性），家具为 "unknown" |
| Ours | 本方法，使用与所有基线相同的 Global 和 Aligned 评分协议 |

VLM baseline 的生成阶段只读取 `input/Smart_device/smartDevice_{code}.yaml` 中的当前家庭尺寸与设备坐标，不读取 GT。所有预测冻结后，GT 才进入独立评分器；同一协议表中的四种方法使用完全相同的评分流程。

VLM 响应只有通过结构、名称、坐标和尺寸校验后才会原子替换输出文件。若本次 API 调用、解析或校验失败，该方法显示为 `N/A`，比较脚本不会评分磁盘上的旧预测。

对比结果保存至 `output/{code}/evaluation/comparison_{code}.txt`。

### 画对比柱状图

```bash
python baselines/plot_comparison.py --code 0622
# 输出: output/0622/figures/comparison_baselines_global.pdf/.png
#       output/0622/figures/comparison_baselines_aligned.pdf/.png
```

---

## 消融实验 (Ablation Studies)

**UbiComp 精简版**：2-3 组核心消融，每组 1 张表，总共 ≤1 页。验证"有方法设计的系统论文"中每个组件的必要性。

### 运行

```bash
# UbiComp 最小安全版本（仅必须消融）
python baselines/ablation_experiments.py --code 0622 --group modal --group llm

# 全部 3 组
python baselines/ablation_experiments.py --code 0622

# 跳过房间分割
python baselines/ablation_experiments.py --code 0622 --skip-seg

# 批量
python baselines/ablation_experiments.py --codes 0622,0701
```

### 2 组核心消融实验

**1. 模态贡献消融 (Modal Contribution) — 必须**

累加式验证各模态的独立与互补贡献：

| Variant | Layout | Device | Human Activity | 说明 |
|---------|--------|--------|---------------|------|
| Layout Only | ✓ | ✗ | ✗ | 仅户型图 |
| + Device | ✓ | ✓ | ✗ | 户型图 + 智能设备 |
| + Human Activity | ✓ | ✗ | ✓ | 户型图 + 人的行为轨迹 |
| Full (All Modalities) | ✓ | ✓ | ✓ | 全部三种输入 |

主输出指标：Room F1 / Localization F1 / Semantic F1 / Conditional FNA；Legacy CLS 仅作探索性参考。

**2. Constrained vs Free-form LLM — 强烈建议**

验证约束协议减少词表协议违规，并同时观察端到端语义误报：

| Variant | 说明 | 核心指标 |
|---------|------|---------|
| Free-form LLM | 无约束 prompt，LLM 可任意增删改家具 | Protocol Violation Rate / Semantic FDR |
| Constrained (Ours) | action-based 约束协议，只能 merge/delete/adjust | Protocol Violation Rate / Semantic FDR |

输出指标：Protocol Violation Rate、Semantic FDR、Room F1、Localization F1、Semantic F1。

Protocol Violation Rate = 违反房间允许词表的非智能设备预测 / 非智能设备预测总数；Semantic FDR = 1 - Semantic Precision。前者是机制检查，不是独立性能证明。

**3. 拓扑贡献消融 (Topology Contribution) — 可选**

验证双源召回（hatch 阴影 + 机器人轨迹闭环）的互补性：

| Variant | Hatch 阴影 | Robot Trajectory Loop | 说明 |
|---------|-----------|----------------------|------|
| Full | ✓ | ✓ | 双源互补 |
| w/o Hatch | ✗ | ✓ | 仅机器人轨迹闭环 |
| w/o Robot Traj. Loop | ✓ | ✗ | 仅 hatch 阴影 |

重点看 CentroidR（质心距离召回率），验证拓扑检测对小家具 recall 的提升。

每组实验自动运行 Pipeline → 评估主指标、Protocol Violation Rate 与 Semantic FDR → 输出对比表格并保存到 `output/evaluation/ablation_results_*.txt`。

### 画消融柱状图

```bash
# 自动查找最新消融结果并画图
python baselines/plot_ablation.py --code 0622

# 或指定结果文件
python baselines/plot_ablation.py --input output/0622/evaluation/ablation_results_xxx.txt
# 输出: output/{code}/figures/ablation_modal_{code}.pdf + ablation_llm_{code}.pdf
```

---

## 消融实验控制标志

`Pipeline` 接受 `ablation: dict` 参数，支持以下消融控制：

| 标志 | 所属消融组 | 效果 |
|------|-----------|------|
| `skip_trajectory` | 模态贡献 | 跳过人的行为轨迹输入 |
| `skip_smart_device` | 模态贡献 | 跳过智能设备输入 |
| `skip_hatch` | 拓扑贡献 | 跳过 hatch 阴影家具检测 |
| `skip_trajectory_loop` | 拓扑贡献 | 跳过机器人轨迹闭环家具检测 |
| `skip_llm_correction` | LLM | 跳过 LLM 修正，使用纯 CV 候选 |
| `use_free_form_llm` | LLM | 使用无约束 LLM prompt（核心消融） |
| `use_direct_llm_gen` | LLM | 跳过 CV，LLM 直接生成家具 |
| `skip_layered_priority` | 推理 | 跳过 7 层优先级，直接使用 LLM 输出 |
| `skip_behavior_prior` | 推理 | 跳过行为先验推理 |
| `skip_allowed_list` | 推理 | 跳过家具允许名单约束 |
| `skip_shape_constraint` | 推理 | 跳过形状约束 |

---

## Agent 核心设计

### 家具检测 Agent (FurnitureDetectionAgent)

**双源 CV 召回：**

| 来源 | 方法 | 置信度 | 覆盖场景 |
|------|------|--------|---------|
| 红框 (Hatch) | Gabor 滤波 → 连通域 → 最大内接矩形分解 | 0.90 | 大家具、规则家具、阴影明显 |
| 蓝框 (Robot Trajectory Loop) | 机器人轨迹闭运算 → findContours → 内接矩形 | 0.6×loop_score | 椅子、桌腿、非规则家具、阴影弱 |

**蓝框后处理：**
- 相邻同尺寸合并（并查集，width/height 比 ≥0.6，间距 ≤3dm）
- 死角过滤（长宽比 >5 且窄边 <12px）
- 世界坐标过滤（单边 ≥3dm，面积 ≥20dm²）
- 与红框重叠 → 直接删除

**结构特征注解**（只读，供 LLM 参考）：
- `rectangularity`: 内接矩形填充率
- `arc_score`: bbox 四周被机器人轨迹环绕比例 (0~1)
- `loop_cluster_count`: bbox 内机器人轨迹形成的闭环数 (≥3 ≈ 椅子)
- `size_class`: small / medium / large

### 房间推理 Agent (RoomAgent)

**7 层优先级：**

| 步骤 | 逻辑 | 置信度 |
|------|------|--------|
| Step 1 | 无家具 → other | 0.8 |
| Step 2 | 智能设备 priority≥4 → 确定房间类型 | 0.9 |
| Step 2.5 | 阳台检测：靠边 + 面积底部1/3 + 家具≤2 | 0.7-0.9 |
| Step 3 | 行为模式：night→bedroom, day→living_room | 0.4-0.7 |
| Step 4 | 连通性：connectivity≥4 + 大面积 + 有活动 → living_room | 0.5-0.6 |
| Step 5 | bathroom 推断：主卫 + 内套卫浴 + 第二卫 | 0.55-0.8 |
| Step 6 | 缺失必需类型评分分配 | 0.5-0.55 |
| Step 7 | other 重分配：有家具但未分类 → 按家具+行为+面积评分 | 0.4-0.5 |

含 3 次重试验证和自动修复。

### 家具命名 Agent (FurnitureNamingAgent)

**智能设备名称保护：** 已有设备名的家具（Refrigerator、TV 等）跳过 LLM，直接保留高置信度。

**4 层推理漏斗：**

| 层级 | 过滤器 | 作用 |
|------|--------|------|
| Level 1 | Room Type | 从允许列表中获取候选集（硬约束） |
| Level 2 | Size + Shape | 面积和长宽比条件排除不匹配候选 |
| Level 3 | Position + Adjacency | 贴墙/角落/中心位置 + 邻近关系排除 |
| Level 4 | Behavior | 日/夜行为模式提升/降低置信度 |

冲突解决优先级: Room Type > Size+Shape > Position > Behavior

额外硬约束：同房间不重复命名、沙发长边必须贴墙、沙发必须面对 TV Stand。

---

## 可视化工具

### visualize_layout.py

将任意 YAML 布局渲染为 PNG/PDF：

```bash
python visualize_layout.py --input output/yaml/03_final_0622.yaml --output output/viz/floorplan.png --dpi 150
```

### layout_viewer.html

交互式户型查看器，支持浏览器中查看房间类型（颜色区分）、家具名称、门窗位置。直接用浏览器打开即可。

---

## 输出格式

`output/yaml/03_final_0622.yaml` 示例：

```yaml
house:
  size: { x: 80, y: 110 }
  rooms:
    - name: bedroom
      position: { x: 0, y: 40, width: 30, height: 20 }
      walls: { top: true, right: true, bottom: true, left: true }
      doors:
        - id: door1
          points: [[30, 41], [30, 50], [21, 41]]
      furniture:
        - id: bedroom_fur1
          name: bed
          position: { x: 1, y: 50, width: 10, height: 9 }
        - id: bedroom_fur2
          name: wardrobe
          position: { x: 15, y: 50, width: 14, height: 9 }
    - name: kitchen
      position: ...
      furniture:
        - id: kitchen_fur1
          name: Refrigerator
          position: { x: 21, y: 71, width: 7, height: 7 }
    ...
```

---

## License

MIT
