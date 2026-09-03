# ymj\_TS -- 基于轨迹拓扑的家庭轮廓智能推理系统

## 1. 项目总体目标

该项目从三类输入数据自动推理出完整的家庭布局 YAML 文件：

| 输入               | 来源                                | 说明                                      |
| ---------------- | --------------------------------- | --------------------------------------- |
| 户型轨迹图 (greyroom) | 扫地机器人 APP 截图                      | 三值图像：墙=0（黑色），地板=128（灰色），扫地机行驶轨迹=255（白色） |
| 智能设备坐标           | `Smart_device/smartDevice_*.yaml` | 已知家具（冰箱、电视、烤箱、洗衣机等）的世界坐标位置（单位: dm）      |
| 人的行为轨迹           | `traj/*_trajectory.json`          | 多天/多人的停留点数据，包含停留时长（分钟）和位置（世界坐标 dm）      |

最终输出 `output/yaml/03_final.yaml`（世界坐标），包含：所有房间（id、position、room\_type、walls、doors）、所有家具（id、position、name、confidence）、以及行为分析结果。

核心设计理念：**CV 负责高召回（宁多勿漏），LLM 只负责几何修正（不发明家具），代码硬规则兜底**。

***

## 2. 管线 / 架构

项目采用 `main.py` 作为唯一入口，调用 `pipeline.py` 中的 `Pipeline` 类，编排 7 步管线（在 `allreadme.md` 中记为 7 步，在 `pipeline.py` 中合并为 5 步）：

### 7步数据流 (来自 `allreadme.md` 和 `main.py`)：

```
输入:
  1. 户型图 (room_real.png)
  2. 智能设备 YAML (smartDevice.yaml, 世界坐标 dm)
  3. 行为轨迹 JSON (trajectory.json, 世界坐标 dm)

Step 1 -- CV 房间分割 (photo2yaml/run_seg.py)
  |-- Hough 线段检测 -> 水平/垂直线合并 -> 网格分割
  |-- Union-Find 无墙邻接合并 -> 候选房间
  |-- Gabor 滤波器提取斜线阴影 (hatch_mask.png) -- 标记家具位置
  |-- 门洞检测 (wall_raw 中扫描白色间隙)
  |-- 自动推算 SCALE (门宽法 / --ref 参考边)
  |-- 输出: output/yaml/01_seg_pixel.yaml (像素坐标)
       + output/masks/greyroom.png (三值图)
       + output/masks/hatch_mask.png (阴影家具掩码)

Step 2 -- 家具检测 + 智能设备匹配 (FurnitureDetectionAgent)
  |-- 源1: hatch 阴影 -> 最大内接矩形分解 -> 红框 (hatch_furniture, 高可靠)
  |-- 源2: 轨迹闭环内部空白 -> findContours 内轮廓 -> 蓝框 (trajectory_surrounded)
  |-- 世界坐标过滤 (单边 >= 3dm, 面积 >= 20dm²)
  |-- IoU 去重 + 置信度融合
  |-- 结构特征注解 (loop_cluster_count, arc_score, rectangularity, size_class)
  |-- 行为停留热度图生成 (Gaussian扩散, INFERNO色图叠加原图)
  |-- LLM action-based 几何修正 (merge/delete/adjust -- 不发明家具)
  |-- 智能设备匹配: 按距离+尺寸兼容匹配到最近的家具框 (多设备时分割矩形)
  |-- 坐标转换 -> 世界坐标
  |-- 输出: output/yaml/02_furniture_world.yaml

Step 3 -- 房间类型推理 (RoomAgent, LLM gpt-4o)
  |-- 7 层优先级法: 无家具->other > 智能设备(priority>=4) > 歧义设备+行为
  |-- > 行为统计(夜间/白天) > 连通性(score>=4) > 卫生间推理(主卫+内套)
  |-- > 缺失类型评分分配 > 有家具但other/unknown重分配
  |-- validate->fix->retry (最多3次)
  |-- 输出: room_analyses (内存)

Step 4 -- 行为模式分析 (BehaviorAgent, LLM gpt-4o)
  |-- 分析每个房间的主要活动、高频区域、时段模式
  |-- 输出: behavior_analyses (内存)

Step 5 -- 家具命名 (FurnitureNamingAgent, LLM gpt-5.4)
  |-- 智能设备名称保护: 已有名称的家具跳过 LLM, 置信度 0.95
  |-- 4 层推理漏斗: 房间类型过滤 > 尺寸形状过滤 > 位置邻接过滤 > 行为模式加分
  |-- 严格允许列表约束 (按房间类型从 config/furniture_allowed.yaml 读取)
  |-- validate->fix (去重、替换非法名称)
  |-- 输出: furniture_namings (内存)

Step 6 -- 合并输出 -> output/yaml/03_final.yaml
```

***

## 3. 每个 Agent 的角色

### 3.1 `FurnitureDetectionAgent` (`agents/furniture_detection_agent.py`)

- **定位**：家具位置检测（CV + LLM），只推理家具的**位置和大小**，不推理家具类别。
- **CV 阶段**（双源召回）：
  - **红框 (hatch\_furniture)**：从 `hatch_mask.png`（Gabor 滤波器提取的斜线阴影）中，通过连通域标记 + 贪婪最大内接矩形分解（直方图+单调栈）提取。高可靠性（confidence 0.90）。
  - **蓝框 (trajectory\_surrounded)**：对轨迹 mask 做闭运算 + `findContours` (RETR\_CCOMP)，找内部闭环区域，再进行最大内接矩形分解。低置信，需 LLM 验证。
  - 扩展 hatch 框到轨迹边界（`_expand_hatch_to_trajectory`）。
  - 相邻同尺寸蓝框自动合并（`_auto_merge_strong_signal_pairs`）。
- **行为轨迹处理**：
  - 世界坐标 -> 像素坐标转换。
  - 按自适应网格聚合停留时长。
  - 生成全图停留热度图（Gaussian 扩散 + INFERNO 色图 + 原图叠加）。
  - 筛选每个房间内的轨迹点，构建 LLM 可读的文本信息。
- **结构特征注解** (`_annotate_structure_features`)：为每个候选框附加 4 个只读特征：`loop_cluster_count`（闭环数，判断椅子腿环绕）、`arc_score`（轨迹弧包围度）、`rectangularity`（填充率）、`size_class`（small/medium/large）。
- **LLM 阶段**：使用 `FURNITURE_DETECTION_AGENT_PROMPT`（在 `prompts.py` 中），以 action-based 协议输出 `delete / merge / adjust` JSON。红框不删不合并（只能裁剪），蓝框可删可合并需验证。核心规则包括：结构分离规则（圆环簇不合并）、同尺度弧包围合并规则、四圆圈独立规则、尺寸比 > 2.5 不合并。
- **后处理**：置信度融合（0.4\*shadow\_score + 0.4\*loop\_score）、IoU > 0.5 去重、世界坐标过滤。
- **智能设备匹配** (`_match_devices_to_furniture`)：统一最近邻匹配 + 尺寸兼容检查，多设备在同一个框时按坐标分布方向分割矩形。

### 3.2 `RoomAgent` (`agents/room_agent.py`)

- **定位**：根据房间内的家具信息、行为模式和连通性推断房间类型（bedroom/living\_room/kitchen/bathroom/studyroom/diningroom/other）。
- **7 层优先级推理** (`_apply_layered_priority`)：
  1. 无家具 -> other（锁定）
  2. 智能设备 priority >= 4 -> 直接确定（如 Fr/Refrigerator/Stove/Oven -> kitchen）
  3. priority=3 的歧义设备 + 行为模式加权
  4. 行为统计：夜间长停留 -> bedroom，白天长停留 -> living\_room
  5. 连通性推理：连通度 >= 4 + 面积 > 30 -> living\_room
  6. 卫生间推理：主卫（最小 + 全小家具）+ 内套卫浴（全小 + 连大家具）
  7. 有家具但 other/unknown 的重新评分分配
- **验证与修正**：`validate_room_analyses` 检查必要类型存在性、卫生间数量约束（< 8 间房 1 个，>= 8 间房最多 2 个）、重复类型去重。`fix_room_analyses` 自动修复，含重试机制（最多 3 次）。
- **智能设备映射配置**：在 `config/smart_devices.yaml` 中定义，每个设备名 -> {types, priority}。

### 3.3 `BehaviorAgent` (`agents/behavior_agent.py`)

- **定位**：从轨迹数据中分析每个房间的行为模式。
- 利用 `BEHAVIOR_AGENT_PROMPT`，输入房屋布局 YAML + 轨迹 JSON + 房间类型信息，输出每个房间的 `main_activities`、`frequent_areas`（如 "center"、"near door"）、`time_based_patterns`（主要时段、高峰小时、总时长、访问次数）。

### 3.4 `FurnitureNamingAgent` (`agents/furniture_naming_agent.py`)

- **定位**：为无名的家具赋予类型名称。
- **智能设备名称保护**：已有名称（如洗衣机、冰箱等，来自 `config/smart_devices.yaml` 的设备名列表）的家具跳过 LLM，直接保留原名（confidence=0.95）。
- **4 层推理漏斗** (`FURNITURE_AGENT_PROMPT`)：
  1. 房间类型过滤：根据允许列表 `config/furniture_allowed.yaml`（如 bathroom: toilet/washbasin/shower）
  2. 尺寸形状过滤：每种家具类型有面积和长宽比的硬约束（如 bed: area > 200, ratio < 1.5; sofa: area > 100, ratio >= 1.8, 长边必须贴墙）
  3. 位置邻接过滤：靠墙/角落/居中的兼容性规则、邻接关系（如 nightstand 必须挨着 bed）
  4. 行为模式加分：夜晚活动 -> bedroom 家具加分
  5. 冲突解决使用判别规则（如 sofa vs dining\_table、toilet vs washbasin）

### 3.5 `CoordinateConverter` (`agents/coordinate_converter.py`)

- **定位**：像素坐标（图像 top-left 原点，y 向下）与 世界坐标（房屋 bottom-left 原点，y 向上，单位 dm）之间的双向转换。
- 从 YAML metadata 读取 SCALE（dm/px）、bl\_x、bl\_y。
- 转换范围覆盖：房间 position、门 points、墙 segments、家具 position、house.size。

***

## 4. 数据格式

### 输入格式

- **户型图** (`greyroom.png`)：三值灰度 PNG，0=墙，128=地板，255=轨迹
- **智能设备 YAML** (`smartDevice_0622.yaml`)：
  ```yaml
  house:
    size: { x: 63, y: 137 }
    furniture:
      - name: Refrigerator
        id: Refrigerator_001
        position: { x: 57, y: 104 }   # 世界坐标 (dm)
  ```
- **轨迹 JSON** (`trajectory.json`)：
  ```json
  [{
    "id": 1,
    "start_time": "2025-05-25 09:00:00",
    "end_time": "2025-05-25 09:01:00",
    "duration": 1.0,
    "center_position": {"x": 50.0, "y": 19.0}
  }]
  ```

### 中间输出格式

- `01_seg_pixel.yaml`：像素坐标的房间分割结果（含 walls、doors）
- `02_furniture_world.yaml`：世界坐标的家具位置 + 智能设备名称
- `03_final.yaml`：最终输出，世界坐标，含房间类型、家具名称、行为模式

### 最终输出格式 (`03_final.yaml`)

```yaml
house:
  size: { x: 63.0, y: 137.0 }
  rooms:
    - name: bedroom
      position: { x: 0, y: 15, width: 30, height: 42 }
      walls: { top: true, right: true, bottom: false, left: true }
      doors:
        - id: door1
          points: [[18, 57], [27, 57], [18, 48]]
      behavior:
        main_activities: [sleeping, getting dressed]
        frequent_areas: [center, near window]
        time_based_patterns: { primary_time_period: night, ... }
      furniture:
        - id: bedroom_fur1
          name: bed
          position: { x: 1, y: 25, width: 20, height: 18 }
        - id: bedroom_fur2
          name: wardrobe
          position: { x: 1, y: 51, width: 17, height: 6 }
```

***

## 5. 方法论

### CV 技术栈

- **Hough 线段检测** + Union-Find 网格合并（房间分割）
- **Gabor 滤波器**（45度/-45度斜线阴影提取，标记家具位置）
- **Otsu 阈值** + 连通域标记 + 贪婪最大内接矩形分解（直方图+单调栈解法）
- **findContours (RETR\_CCOMP)** 轨迹闭环检测
- **Gaussian 扩散**热度图生成（人体停留点可视化）
- **DBSCAN-like** 自适应网格聚合轨迹点
- **IoU 去重** + 包含抑制

### LLM 技术栈

- **LangChain + ChatOpenAI** 框架
- 4 个专用 Prompt（家具检测、房间推理、行为分析、家具命名），合计约 700 行精心设计的指令
- **Action-based 协议**（家具检测阶段）：LLM 只输出 delete/merge/adjust 动作 JSON，不直接输出最终家具列表，确保 CV 结果默认保留
- **4 层推理漏斗**（家具命名阶段）：逐步缩小候选集
- **7 层优先级法**（房间推理阶段）：从硬约束逐步到软约束
- 智能设备名称保护机制：已有名称的家具不参与 LLM 重新命名

### 硬规则兜底

- 红框不可删除、不可合并（只能裁剪重叠部分）
- 蓝框结构分离规则（圆环簇不合并）
- 世界坐标尺寸过滤（单边 < 3dm 删除，面积 < 20dm² 删除）
- 房间类型必须包含 kitchen/1 bathroom（< 8 间房）或 <= 2 bathrooms（>= 8 间房）
- 家具名称严格限制在允许列表内
- sofa 的长边必须贴墙 + ratio >= 1.8

***

## 6. 现有评估/验证方法

### 已实现的验证机制

1. **RoomAgent 验证**：`validate_room_analyses()` 检查必要房间类型存在性、bathroom 数量约束、无效类型名、置信度范围。验证失败时自动调用 `fix_room_analyses()`，然后重新生成（最多 3 次重试）。通过 `analyze_rooms()` 方法封装完整的生成-验证-修复-重试循环。
2. **FurnitureNamingAgent 验证**：`validate_furniture_namings()` 检查同房间重名、名称是否在允许列表中、placeholder 名称（none/unknown）、置信度范围、是否有未命名的家具。`fix_furniture_namings()` 自动修复（去重、替换非法名）。
3. **BehaviorAgent 验证**：`validate_behavior_analyses()` 检查 main\_activities/frequent\_areas/time\_based\_patterns 的类型正确性和完整性。
4. **CV 阶段验证**：
   - IoU 去重（> 0.5 的去重）
   - 世界坐标尺寸过滤（单边 >= 3dm，面积 >= 20dm²）
   - 轨迹点落点诊断（统计在房间内/外的比例，> 30% 在房间外则警告 SCALE 可能偏差）
   - 包含抑制（外层框包含内层框时删除外层）
5. **真实数据对比**：项目包含 `realroom/layout0622.yaml`（人工标注的 Ground Truth），包含 7 个房间（Balcony, Second Bedroom, Master Bedroom, Bathroom, Dining Room, Living Room, Kitchen）的完整家具标注，可用于评估算法准确性。
6. **可视化系统**：
   - `visualize_layout.py`：用 matplotlib 渲染最终的 YAML 为 PNG 户型图（温暖奶油色调，匹配 Figure\_1 风格）
   - `layout_viewer.html`：纯前端 HTML/JS YAML 查看器（支持拖放、粘贴、下载 PNG）
   - FurnitureDetectionAgent 在每个房间处理和最终阶段自动生成多种可视化图（候选框图、最终结果图、全图概览图、热度图）保存到 `output/masks/`
7. **配置文件驱动**（可复现）：`config/furniture_allowed.yaml` 和 `config/smart_devices.yaml` 将允许列表和映射从代码中解耦，修改配置无需改代码。
8. **历史子系统**：`furniture_graph/` 目录包含一个基于知识图谱的 Viterbi 序列校正子系统（使用从 TSGen 仿真数据训练的家具转移概率和房间归属规则），可用于对 Agent 标注进行二次校正。这是独立的、但目前未集成到主 7 步管线中的评估/校正机制。

***

## 7. 项目完整文件清单

| 文件路径                                                                      | 功能                                  |
| ------------------------------------------------------------------------- | ----------------------------------- |
| `e:\python\PythonProject\ymj_TS\main.py`                                  | 唯一入口，参数解析，调用 Pipeline               |
| `e:\python\PythonProject\ymj_TS\pipeline.py`                              | Pipeline 类，编排 7 步管线                 |
| `e:\python\PythonProject\ymj_TS\allreadme.md`                             | 项目总文档（详细设计文档）                       |
| `e:\python\PythonProject\ymj_TS\README.md`                                | 用户文档（安装、快速开始、输出格式）                  |
| `e:\python\PythonProject\ymj_TS\agents\furniture_detection_agent.py`      | Step 2: CV+LLM 家具检测（\~2127行，最复杂的模块） |
| `e:\python\PythonProject\ymj_TS\agents\room_agent.py`                     | Step 4: 房间类型推理（7层优先级，\~1375行）       |
| `e:\python\PythonProject\ymj_TS\agents\behavior_agent.py`                 | Step 5: 行为模式分析                      |
| `e:\python\PythonProject\ymj_TS\agents\furniture_naming_agent.py`         | Step 6: 家具命名（4层推理漏斗）                |
| `e:\python\PythonProject\ymj_TS\agents\coordinate_converter.py`           | Step 3: 像素<->世界坐标转换                 |
| `e:\python\PythonProject\ymj_TS\agents\prompts.py`                        | 所有 LLM Prompt 模板（\~714行）            |
| `e:\python\PythonProject\ymj_TS\agents\data_models.py`                    | Pydantic 数据模型                       |
| `e:\python\PythonProject\ymj_TS\agents\__init__.py`                       | 模块导出                                |
| `e:\python\PythonProject\ymj_TS\photo2yaml\run_seg.py`                    | Step 1: CV 房间分割（\~844行）             |
| `e:\python\PythonProject\ymj_TS\photo2yaml\extract_hatch.py`              | Gabor 滤波器斜线阴影提取                     |
| `e:\python\PythonProject\ymj_TS\config\furniture_allowed.yaml`            | 家具命名允许列表（按房间类型）                     |
| `e:\python\PythonProject\ymj_TS\config\smart_devices.yaml`                | 智能设备->房间类型映射 + 名称保护列表               |
| `e:\python\PythonProject\ymj_TS\visualize_layout.py`                      | matplotlib 户型图可视化                   |
| `e:\python\PythonProject\ymj_TS\layout_viewer.html`                       | 纯前端 YAML 查看器                        |
| `e:\python\PythonProject\ymj_TS\input\Smart_device\smartDevice_0622.yaml` | 示例智能设备输入                            |
| `e:\python\PythonProject\ymj_TS\realroom\layout0622.yaml`                 | 人工标注的真实布局（Ground Truth）             |

# ymj\_TS 项目论文分析与实验设计方案

***

## 一、项目整体描述（用于论文写作）

### 1.1 问题定义

本项目解决的是**多模态家庭轮廓智能推理**问题：给定三类异构输入 —— **户型轨迹图**（扫地机器人APP截图，三值灰度图：墙/地板/轨迹）、**智能设备坐标**（冰箱、洗衣机等的世界坐标位置）和**多日人体行为轨迹**（停留点 + 时长），自动推理出完整的家庭布局 YAML，包含：房间分割与类型推断、家具位置/类别/名称、行为模式分析。

### 1.2 核心贡献/创新点

1. **CV+LLM 混合管线架构**：CV 负责高召回（宁多勿漏），LLM 只负责几何修正和语义推理（不发明家具），硬规则兜底。解决了纯 CV 方法误检率高、纯 LLM 方法幻觉严重的问题。
2. **三模态融合机制**：首次将户型图、智能设备坐标、人体轨迹三种异构数据统一融合进行室内布局推理。
3. **Gabor 滤波器阴影提取 + 最大内接矩形分解**：从户型图斜线阴影中提取家具候选区域（红框高可靠）。
4. **轨迹闭环检测**：利用扫地机器人轨迹包围的空白区发现家具（蓝框），与阴影提取互补。
5. **行为驱动的家具/房间语义推理**：利用人体停留时长、时段分布辅助推断房间类型和家具名称。
6. **7 层优先级法 + 4 层推理漏斗**：可解释、层级递进的推理策略。
7. **智能设备名称保护机制**：已有先验知识的家具不参与 LLM 重命名，避免幻觉。

### 1.3 技术架构图（论文需要）

```
输入层:  户型图 (PNG)  +  智能设备 YAML  +  行为轨迹 JSON
          ↓                  ↓                 ↓
Step 1:  Hough线段检测 → Union-Find网格合并 → 房间分割 + 门洞检测
          + Gabor滤波器阴影提取 (hatch_mask)
          ↓
Step 2:  双源家具检测 (红框hatch+蓝框trajectory)
          → LLM几何修正(delete/merge/adjust) → 智能设备匹配
          → 坐标转换(像素→世界坐标)
          ↓
Step 3:  7层优先级房间类型推理 (LLM + 规则)
          ↓
Step 4:  行为模式分析 (LLM)
          ↓
Step 5:  4层推理漏斗家具命名 (LLM + 规则约束)
          ↓
输出:   03_final.yaml (世界坐标, 完整家庭布局)
```

***

## 二、近年顶会相关论文对比

### 2.1 直接相关论文

| 论文                 | 会议/期刊              | 年份   | 核心方法                        | 与本项目的异同                     |
| ------------------ | ------------------ | ---- | --------------------------- | --------------------------- |
| **Architect-Ant**  | arXiv 2026         | 2026 | VLM微调 + DSL家具布局生成 + DPO偏好优化 | 同：户型图家具推理；异：纯VLM方法，无轨迹/设备融合 |
| **DirectLayout**   | NeurIPS 2025       | 2025 | LLM直接生成数值3D布局 + CoT激活       | 同：LLM空间推理；异：文本→3D生成，不涉及图像输入 |
| **SpatialLM**      | arXiv 2025         | 2025 | LLM处理3D点云输出结构化场景描述          | 同：结构化室内建模；异：点云输入，不含轨迹       |
| **DisCo-Layout**   | arXiv 2025         | 2025 | 多Agent框架解耦语义/物理优化           | 同：多Agent推理；异：3D布局合成，不涉及真实输入 |
| **LLplace**        | arXiv 2024         | 2024 | Llama3微调3D室内布局生成            | 同：LLM空间布局；异：纯生成任务，无感知输入     |
| **LayoutAgent**    | CVPR Workshop 2025 | 2025 | VLM + 组合扩散模型空间布局            | 同：VLM空间规划；异：图像生成场景          |
| **QueSTMaps**      | arXiv 2024         | 2024 | 多通道占用表示 + CLIP对齐房间标签        | 同：房间分割+语义标注；异：3D点云输入        |
| **TR-LLM**         | arXiv 2025         | 2025 | 轨迹+LLM人类行为预测                | 同：轨迹+LLM融合；异：预测未来行为，不推理布局   |
| **MARAuder's Map** | arXiv 2025         | 2025 | 传感器轨迹投影到户型图做HAR             | 同：轨迹+户型图对齐；异：活动识别，不推理家具     |
| **FloorNet**       | IJDAR 2024         | 2024 | CNN+边界注意力=分割房间/墙            | 同：平面图分析；异：仅CV分割，无语义推理       |