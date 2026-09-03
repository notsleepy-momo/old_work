# ymj_TS — 室内家具布局智能校正系统

## 项目概述

本项目通过**知识图谱（KG）** 和 **Viterbi 序列校正**，对 AI Agent 自动生成的室内家具布局标注进行合理性校验与修正。

**核心思想**：人的行为在室内空间中呈现"家具→家具"的转移规律（如从 Fridge 去 Stove 做饭），且每个家具有固定的房间归属（Fridge 只在 Kitchen）。利用这些规律构建知识图谱，可以检测并修正 Agent 标注中的异常。

### 完整流程

```
┌──────────────────────────────────────────────────────────────────────┐
│                        输入数据层                                       │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────┐                    │
│  │ 室内轮廓  │  │ 智能设备布局  │  │  行为轨迹数据   │                    │
│  │ fin_map  │  │ smartDevice  │  │  trajectory    │                    │
│  └────┬─────┘  └──────┬───────┘  └───────┬────────┘                    │
│       │               │                  │                             │
├───────┼───────────────┼──────────────────┼─────────────────────────────┤
│       ▼               ▼                  ▼                             │
│  ┌──────────────────────────────────────────────────────┐              │
│  │         Agent（大模型）生成初始家具标注                  │              │
│  │         → 输出 updated_fin_map_*.yaml                │              │
│  └────────────────────────┬─────────────────────────────┘              │
│                           │                                           │
│                           ▼                                           │
│  ┌──────────────────────────────────────────────────────┐              │
│  │  校正流水线 (correct_agent_map.py)                    │              │
│  │                                                      │              │
│  │  ① 房间自动推断（根据家具类型投票）                      │              │
│  │  ② 轨迹点→家具槽位匹配                                 │              │
│  │  ③ Viterbi 全局最优序列校正（融合 KG 规则 + Agent 先验）│              │
│  │  ④ 房间兼容性过滤（拒绝跨房间的不合理修改）                │              │
│  └────────────────────────┬─────────────────────────────┘              │
│                           │                                           │
│                           ▼                                           │
│  ┌──────────────────────────────────────────────────────┐              │
│  │    输出：corrected_fin_map_*.yaml                     │              │
│  │    更合理的室内设备图                                  │              │
│  └──────────────────────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────────────────┘
                           ▲
                           │  知识图谱提供"房间规则" + "转移规则"
                           │
                  ┌────────┴────────┐
                  │  家具转移知识图谱  │
                  │  knowledge_graph │
                  └────────┬────────┘
                           │
                  ┌────────┴────────┐
                  │  训练阶段         │
                  │  pipeline.py     │
                  │  --mode train    │
                  └─────────────────┘
```

---

## 项目结构

```
ymj_TS/
├── allreadme.md                        # 本文件 — 项目总文档
├── README.md                           # 原始 TSGen 文档
│
├── input/                              # ★ 输入数据
│   ├── map/                            # 室内轮廓（fin_map_*.yaml）
│   ├── Smart_device/                   # 智能设备布局（smartDevice_*.yaml）
│   ├── initial_map/                    # 初始户型（无家具名）
│   ├── traj/
│   │   ├── Continuous_time/            # Agent 推断用的连续行为轨迹
│   │   └── oneday/                     # 单日行为轨迹（备用）
│   └── map_furniture_mapper.py         # 户型→fin_map 生成脚本
│
├── traindata/                          # ★ 训练数据
│   └── traj/
│       ├── 2/Engineer/                 # 户型2 × Engineer × 7天
│       ├── 3/Engineer/                 # 户型3 × Engineer × 7天
│       ├── 4/Engineer/                 # 户型4 × Engineer × 7天
│       └── 5/Engineer/                 # 户型5 × Engineer × 7天
│
├── output/                             # ★ 输出
│   └── Merge_day/
│       ├── 1/updated_fin_map_1-Engineer-all.yaml  # Agent 生成的家具标注
│       ├── 2/updated_fin_map_2-Engineer-all.yaml
│       ├── 3/updated_fin_map_3-Engineer-all.yaml
│       ├── 4/updated_fin_map_4-Engineer-all.yaml
│       └── 5/updated_fin_map_5-Engineer-all.yaml
│
└── furniture_graph/                    # ★ 核心代码
    ├── graph_builder.py                # 知识图谱构建
    ├── rule_engine.py                  # 规则引擎 + Viterbi 校正
    ├── correct_agent_map.py            # 校正流水线（主入口）
    ├── pipeline.py                     # 训练/推断/校正 CLI
    ├── map_inferrer.py                 # 地图推断器
    ├── gnn_enhancer.py                 # 可选 GNN 增强
    └── output/                         # 输出目录
        ├── knowledge_graph.json        # 训练好的知识图谱
        └── corrected_fin_map_*.yaml    # 校正后的家具地图
```

---

## 三步完整工作流

### 第 1 步：训练知识图谱

从 TSGen 生成的**行为序列数据**中统计家具间的转移规律和房间归属。

```
输入：traindata/traj/{map_id}/{person}/*.json
     （从 TSGen-main/output/{map_id}/ 经 getTraindata_filter_stationary.py 提取）
输出：furniture_graph/output/knowledge_graph.json
```

```bash
# 单户型训练（户型 2 的 7 天数据）
cd furniture_graph
python pipeline.py --mode train --train_dir ../traindata/traj/2

# 跨户型合并训练（推荐 — 4 个户型 × 7 天 = 649 个行为点）
python pipeline.py --mode train --train_dir ../traindata/traj --train_all_maps
```

#### 训练数据来源

TSGen 输出的事件流 → `getTraindata_filter_stationary.py`（位于 TSGen-main/）→ 行为点 JSON

```bash
# 在 TSGen-main/ 中执行
python getTraindata_filter_stationary.py
```

每个行为点格式：
```json
{
  "id": 1,
  "start_time": "2025-05-25 09:00:00",
  "end_time": "2025-05-25 09:01:00",
  "duration": 1.0,
  "furniture": "Bed_002",
  "room": "Bedroom",
  "center_position": {"x": 50.0, "y": 19.0}
}
```

#### 知识图谱内容

```
KG 结构:
  nodes: 19 种家具 → {rooms, dur_mean, dur_std, time_slot_distribution, ...}
  edges: 145 条 → {source, target, count, prob}

示例节点:
  Fridge   → room=[Kitchen]          dur=(6.7±5.0min)  n=51
  Bed      → room=[Bedroom]          dur=(223.4±292.1min) n=48
  Sofa     → room=[Living Room]      dur=(69.0±79.7min) n=81
  Stove    → room=[Kitchen]          dur=(18.8±16.3min) n=41

高概率转移规则:
  Fridge → Stove     P=1.00  (n=15)   ← 取冰箱后必定去灶台
  Wardrobe → Bed     P=0.67  (n=4)    ← 换衣后大概率去床
  Bed → Washbasin    P=0.60  (n=6)    ← 起床后去洗漱
```

### 第 2 步：Agent 生成初始标注

> 注意：Agent 部分在 `ymj_TS` 项目外部，由大模型（如 GPT-4o）基于以下输入生成：

```
输入：
  ├── fin_map_{id}.yaml         — 室内轮廓（房间边界 + 家具位置）
  ├── smartDevice_{id}.yaml     — 智能设备列表（已知家具类型）
  ├── {id}-Engineer-0525_trajectory.json — 连续行为轨迹

输出：
  ├── updated_fin_map_{id}-Engineer-all.yaml  — Agent 标注的家具地图
```

Agent 对每个家具槽位进行类型推断，结果保存为 YAML。这些标注**可能包含错误**（如将 cabinet 标注在 Kitchen 但不属于 KG、将 Treadmill 误标为 Fridge）。

### 第 3 步：KG 校正

```
输入：
  ├── fin_map_{id}.yaml                         — 房间边界
  ├── updated_fin_map_{id}-Engineer-all.yaml    — Agent 标注
  ├── {id}-Engineer-0525_trajectory.json        — 行为轨迹
  └── knowledge_graph.json                      — 预训练 KG

输出：corrected_fin_map_{id}.yaml
```

```bash
# 方式 1：按地图 ID（自动拼接默认路径）
python correct_agent_map.py --map_id 2

# 方式 2：手动指定任意户型
python correct_agent_map.py \
  --layout ../input/map/fin_map_5.yaml \
  --agent ../output/Merge_day/5/updated_fin_map_5-Engineer-all.yaml \
  --traj ../input/traj/Continuous_time/5/5-Engineer-0525_trajectory.json
```

#### 校正原理

| 步骤 | 方法 | 说明 |
|------|------|------|
| ① 房间推断 | 家具类型投票 | 根据房间内家具名自动映射（如 Fridge+Stove→Kitchen） |
| ② 轨迹匹配 | 最近邻欧氏距离 | 每个轨迹点匹配最近的家具槽位 |
| ③ 序列校正 | Viterbi 解码 | 融合 KG 转移概率 + Agent 先验权重，求全局最优序列 |
| ④ 房间过滤 | KG 房间约束 | 拒绝跨房间的不合理修改（如 Treadmill→Fridge 被 Living Room≠Kitchen 拒绝） |

**Viterbi 参数**：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--agent_weight` | 0.7 | Agent 先验权重（0=完全信任 KG, 1=完全信任 Agent） |
| `--tolerance` | 8.0 | 轨迹点到家具槽位的匹配容差 |
| `--strategy` | viterbi | `viterbi`(全局最优) 或 `pointwise`(逐点覆盖) |

---

## 关键输入文件说明

### 室内轮廓 `fin_map_{id}.yaml`

```
house:
  size: { x: 70, y: 70 }
  rooms:
    - id: room1
      position: { x: 0, y: 40, width: 20, height: 20 }
      furniture:
        - id: room1_fur1          # 家具槽位 ID
          position: { x: 1, y: 50, width: 6, height: 8 }
```

字段说明：
| 字段 | 含义 | 数据来源 |
|------|------|---------|
| rooms[].id | 房间 ID（如 room1） | 固定 |
| rooms[].position | 房间矩形边界 (x, y, width, height) | 室内设计 |
| furniture[].id | 家具槽位 ID | 固定 |
| furniture[].name | **家具类型**（Agent 标注/校正后填充） | AGENT→KG_CORRECTED |
| furniture[].position | 家具位置和尺寸 | 室内设计 |

### 智能设备 `smartDevice_{id}.yaml`

```
house:
  furniture:
    - name: Fr           # 大模型短名：Fr → Fridge
      id: Refrigerator_001
      position: { x: 21, y: 71 }
    - name: Stove
      id: Stove_001
      position: { x: 21, y: 80 }
```

用于 `map_furniture_mapper.py` 将 `initial_map` 与设备布局融合生成 `fin_map`。

### 行为轨迹 `Continuous_time/{id}-Engineer-0525_trajectory.json`

```json
{
  "id": 1,
  "start_time": "2025-05-25 08:30:00",
  "end_time": "2025-05-25 08:35:00",
  "duration": 5.0,
  "center_position": {"x": 44.0, "y": 34.0}
}
```

> 注意：推断用的轨迹文件**不含** `furniture` 和 `room` 字段。`room` 通过坐标匹配 `fin_map` 房间边界自动推断，`furniture` 从 Agent 标注 YAML 的槽位匹配获得。

---

## 房间自动推断

支持任意户型，根据房间内家具类型自动映射：

| 家具 | 投票给 |
|------|--------|
| Fridge, Stove, Sink, Microwave, Oven | Kitchen |
| Bed, Wardrobe, Nightstand | Bedroom |
| Sofa, Coffee_Table, TV_Stand, Treadmill, Table, Dining Table | Living Room |
| Toilet, Washbasin, Bathtub, Shower | Bathroom |
| Desk, Bookshelf, Technical Manual | Study |

### 房间名归一化

跨户型训练时，不同户型中的同功能房间名被统一：

| 原始名 | 归一化名 |
|--------|----------|
| Bedroom, Bedroom 1, Master Bedroom, Guest Bedroom | Bedroom |
| Living Room, Living & Dining Room, Dining Room | Living Room |
| Kitchen | Kitchen |
| Bathroom | Bathroom |
| Study, Study Room | Study |

---

## 家具名归一化

Agent 输出的命名风格多样，统一映射到 KG 通用名：

| Agent 原始名 | 归一化名 |
|-------------|----------|
| Fr, Refrigerator | Fridge |
| TV Stand, TV Cabinet, TV | TV_Stand |
| Coffee Table, coffee_table | Coffee_Table |
| bed, Bed | Bed |
| sofa, Sofa | Sofa |

---

## 运行示例

### 完整流程（户型 2）

```bash
cd e:\python\PythonProject\ymj_TS\furniture_graph

# Step 1: 训练知识图谱（合并 4 个户型）
python pipeline.py --mode train --train_dir ../traindata/traj --train_all_maps

# Step 2: 校正 Agent 标注
python correct_agent_map.py --map_id 2
```

### 校正前后对比示例

```
校正前:
  room2_fur3  (Kitchen)  → cabinet    ← agent 标注为 cabinet（不在 KG 中）
  room4_fur4  (Living Room) → Treadmill ← agent 标注正确，但 Viterbi 误判为 Fridge

KG 房间规则:
  Fridge → [Kitchen]
  cabinet → (不在 KG 中)

校正后:
  room2_fur3  (Kitchen)  → Stove    ← kitchen 中合理类型 ✅
  room4_fur4  (Living Room) → Treadmill ← 拒绝修改（Fridge 不在 Living Room）✅
```

---

## 常见问题

**Q: 训练数据从哪里来？**
A: 来自 [TSGen](https://github.com/tsgen) 项目生成的仿真行为数据。TSGen 通过 LLM 驱动的人物行为模拟生成 24 小时室内轨迹，输出事件流 → 经 `getTraindata_filter_stationary.py` 提取 stationary 行为点。

**Q: 训练好的 KG 可以用于其他户型吗？**
A: 可以。KG 基于**房间类型**（非坐标），只要户型房间可映射为标准类型（Kitchen、Bedroom 等），KG 规则即适用。目前内置 4 个户型的合并训练数据，覆盖了 19 种家具和 5 种房间类型。

**Q: Agent 标注如何生成？**
A: Agent 标注由大模型（GPT-4o）基于 `fin_map`（房间轮廓）、`smartDevice`（已知设备）和 `Continuous_time` 行为轨迹推断得出。本项目的 `correct_agent_map.py` 负责用 KG 校验并修正。

**Q: Viterbi 校正的 agent_weight 如何选择？**
A: `agent_weight=0.7` 表示 30% 信任 KG 规则、70% 保留 Agent 原始标注。如果 Agent 质量较差可降低到 0.5 或 0.3。

---

## 技术栈

- **Python** 3.10+
- **numpy** — 数值计算
- **PyYAML** — YAML 解析
- **NumPy** — 高斯评分、Viterbi 算法
- **collections** — Counter, defaultdict
