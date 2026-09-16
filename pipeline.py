"""
全流程管线 — 5 步串联

输入:
  1. 户型轨迹图 (room_processed.png)
  2. 智能设备位置 (smartDevice.yaml, 世界坐标)
  3. 行为轨迹数据 (trajectory.json, 世界坐标)

流程:
  Step 1: 房间分割 → 房间 + 门 (像素坐标) → 01_seg_pixel.yaml
  Step 2: 家具检测 + 坐标统一 + 智能设备匹配 → 家具位置+设备名 (世界坐标) → 02_furniture_world.yaml
  Step 3: 房间类型推理 (RoomAgent)
  Step 4: 行为模式分析 (确定性, 基于 activity_info)
  Step 5: 家具命名 → 补充无名家具名称 → 03_final.yaml

用法:
  python pipeline.py
  python pipeline.py --image photo/room_processed.png --output output/final_layout.yaml
"""
import sys, os, logging, json, yaml, argparse, subprocess, copy
from datetime import datetime

# ─── 路径设置 ────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # ymj_TS/
sys.path.insert(0, PROJECT_ROOT)

from agents.furniture_detection_agent import FurnitureDetectionAgent
from agents.room_agent import RoomAgent
from agents.behavior_agent import BehaviorAgent
from agents.furniture_naming_agent import FurnitureNamingAgent
from agents.coordinate_converter import CoordinateConverter
from agents.data_models import RoomAnalysis, BehaviorAnalysis, FurnitureNaming
from llm_config import (
    DEFAULT_LLM_MODEL,
    require_shared_model,
    resolve_api_key,
    resolve_base_url,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('pipeline')


def normalize_trajectory_durations(trajectory_data):
    """Return a copy whose trajectory durations are expressed in seconds.

    Project trajectory files store ``duration`` in minutes, while RoomAgent
    and BehaviorAgent prompts and thresholds use seconds.  Timestamps are
    treated as the source of truth when available; the project schema
    (minutes) is used as the fallback for older records without timestamps.
    """
    normalized = copy.deepcopy(trajectory_data)

    if isinstance(normalized, list):
        points = normalized
    elif isinstance(normalized, dict):
        points = None
        for key in ('trajectories', 'trajectory', 'points', 'data'):
            if isinstance(normalized.get(key), list):
                points = normalized[key]
                break
        if points is None:
            points = [normalized]
    else:
        return normalized

    for point in points:
        if not isinstance(point, dict):
            continue

        raw_duration = point.get('duration', point.get('stay_duration'))
        try:
            raw_duration = float(raw_duration)
        except (TypeError, ValueError):
            continue

        elapsed_seconds = None
        start_time = point.get('start_time')
        end_time = point.get('end_time')
        if start_time and end_time:
            try:
                start = datetime.fromisoformat(str(start_time))
                end = datetime.fromisoformat(str(end_time))
                elapsed_seconds = max(0.0, (end - start).total_seconds())
            except (TypeError, ValueError):
                logger.warning("无法解析轨迹时间戳，按分钟解释 duration")

        if elapsed_seconds is not None:
            tolerance = max(1.0, elapsed_seconds * 0.05)
            if abs(raw_duration - elapsed_seconds) <= tolerance:
                duration_seconds = raw_duration
            elif abs(raw_duration * 60.0 - elapsed_seconds) <= tolerance:
                duration_seconds = raw_duration * 60.0
            else:
                logger.warning(
                    "轨迹 duration=%s 与时间戳跨度 %.1fs 不一致，采用时间戳",
                    raw_duration, elapsed_seconds,
                )
                duration_seconds = elapsed_seconds
        else:
            unit = str(point.get('duration_unit', '')).strip().lower()
            duration_seconds = raw_duration if unit in {'s', 'sec', 'second', 'seconds'} else raw_duration * 60.0

        point['duration'] = duration_seconds
        point['duration_unit'] = 'seconds'

    return normalized


class Pipeline:
    """全流程管线

    ablation 参数 (dict) 支持消融实验控制:
      - skip_trajectory:         跳过轨迹输入 (多模态消融)
      - skip_smart_device:       跳过智能设备输入 (多模态消融)
      - skip_hatch:              跳过 hatch 阴影家具检测 (双源消融)
      - skip_trajectory_loop:    跳过轨迹闭环家具检测 (双源消融)
      - skip_llm_correction:     跳过 LLM action 修正, 纯 CV 候选 (CV-LLM 消融)
      - use_free_form_llm:       使用无约束 LLM prompt (CV-LLM 消融)
      - use_direct_llm_gen:      跳过 CV, LLM 直接生成家具位置 (CV-LLM 消融)
      - skip_room_module:        跳过房间语义模块, 房间类型全为 unknown (单模块移除消融)
      - skip_behavior_module:    跳过行为语义模块, 行为证据置空 (单模块移除消融)
      - skip_naming_module:      跳过家具命名模块, 非设备家具标 unknown (单模块移除消融)
      - skip_layered_priority:   跳过7层优先级, 直接使用 LLM 输出 (推理策略消融)
      - skip_behavior_prior:     跳过行为先验推理 (推理策略消融)
      - skip_allowed_list:       跳过家具允许名单约束 (推理策略消融)
      - skip_shape_constraint:   跳过形状约束 (推理策略消融)
      - skip_naming_funnel:      使用无约束命名基线，跳过词表和几何后处理
    """

    def __init__(self, api_key: str = None, base_url: str = None,
                 furniture_api_key: str = None, furniture_base_url: str = None,
                 ablation: dict = None, code: str = None,
                 callbacks: list = None, model: str = DEFAULT_LLM_MODEL):
        # The legacy furniture_* parameters remain accepted, but all agents
        # receive one shared model, credential, and endpoint.
        self.api_key = resolve_api_key(furniture_api_key or api_key)
        self.base_url = resolve_base_url(furniture_base_url or base_url)
        self.furniture_api_key = self.api_key
        self.furniture_base_url = self.base_url
        self.model = require_shared_model(model)
        self.ablation = ablation or {}
        self.code = code or '0622'
        self.callbacks = callbacks or []
        self.furniture_agent = None
        self.room_agent = None
        self.behavior_agent = None
        self.naming_agent = None
        self._init_agents()

    def _init_agents(self):
        shared_llm_kwargs = {}
        if self.furniture_api_key:
            shared_llm_kwargs["api_key"] = self.furniture_api_key
        if self.furniture_base_url:
            shared_llm_kwargs["base_url"] = self.furniture_base_url
        if self.callbacks:
            shared_llm_kwargs["callbacks"] = self.callbacks

        self.furniture_agent = FurnitureDetectionAgent(
            model=self.model, **shared_llm_kwargs)
        self.room_agent = RoomAgent(model=self.model, **shared_llm_kwargs)
        self.behavior_agent = BehaviorAgent(model=self.model, **shared_llm_kwargs)
        self.naming_agent = FurnitureNamingAgent(
            model=self.model, **shared_llm_kwargs)

    # ==========================================================
    # Step 1: 房间分割 — 房间 + 门
    # ==========================================================
    def run_seg_step(self, image_path: str, output_dir: str) -> str:
        """调用 run_seg.py 生成房间分割 YAML"""
        run_seg_path = os.path.join(PROJECT_ROOT, 'photo2yaml', 'run_seg.py')
        yaml_dir = os.path.join(output_dir, 'yaml')

        if not os.path.exists(run_seg_path):
            raise FileNotFoundError(f"run_seg.py not found: {run_seg_path}")

        logger.info("=== Step 1: 房间分割 (run_seg) ===")
        os.makedirs(yaml_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

        # run_seg.py 直接输出到 output_dir (masks/ 和 floorplan_real.yaml)
        result = subprocess.run(
            [sys.executable, run_seg_path, image_path, output_dir],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            cwd=os.path.join(PROJECT_ROOT, 'photo2yaml')
        )
        if result.returncode != 0:
            logger.error(f"run_seg failed:\n{result.stderr}")
            raise RuntimeError("run_seg.py failed")

        logger.info(f"run_seg output:\n{result.stdout}")

        # 将 YAML 移到 yaml/ 子目录下，重命名为 01_seg_pixel_{code}.yaml，并注入 id 字段
        src_yaml = os.path.join(output_dir, 'floorplan_real.yaml')
        dst_yaml = os.path.join(yaml_dir, f'01_seg_pixel_{self.code}.yaml')
        if os.path.exists(src_yaml):
            import shutil
            shutil.move(src_yaml, dst_yaml)
            
            # ── 注入 room id 字段（后续所有 agent 依赖此字段）──
            with open(dst_yaml, 'r', encoding='utf-8') as f:
                seg_layout = yaml.safe_load(f)
            for room in seg_layout.get('house', {}).get('rooms', []):
                if 'id' not in room and 'name' in room:
                    room['id'] = room['name']
            with open(dst_yaml, 'w', encoding='utf-8') as f:
                yaml.dump(seg_layout, f, default_flow_style=None,
                          allow_unicode=True, sort_keys=False, indent=2)
            logger.info(f"房间分割 YAML (像素) -> {dst_yaml} (已注入 room.id)")
        else:
            raise FileNotFoundError(f"run_seg output not found: {src_yaml}")

        return dst_yaml

    # ==========================================================
    # Step 2: 家具检测 + 智能设备匹配 + 坐标统一
    #   输出: 02_furniture_world.yaml (世界坐标, 家具位置 + 智能设备名称)
    #   智能设备匹配在 FurnitureDetectionAgent 内部完成 (像素坐标, 红框匹配)
    # ==========================================================
    def step2_furniture_and_devices(self, image_path: str, seg_yaml_path: str,
                                     trajectory_path: str, smart_device_path: str,
                                     hatch_mask_path: str, output_dir: str) -> tuple:
        """Step 2: 家具位置大小识别 + 智能设备名匹配 + 坐标统一

        FurnitureDetectionAgent 内部完成:
          - 家具检测 (像素坐标)
          - 智能设备红框匹配 + 多设备框均分 (像素坐标)
          - 输出临时 YAML (像素坐标, 含设备名)

        本方法追加:
          - CoordinateConverter → 坐标统一 (世界坐标)
          - 写入 02_furniture_world.yaml

        Returns:
            (yaml_path, trajectory_data): 02_furniture_world.yaml 路径 + 轨迹数据
        """
        yaml_dir = os.path.join(output_dir, 'yaml')
        os.makedirs(yaml_dir, exist_ok=True)

        # ── 2a. 家具检测 + 智能设备匹配 (像素坐标, 一步完成) ──
        logger.info("=== Step 2a: 家具检测 + 智能设备匹配 (FurnitureDetectionAgent) ===")
        temp_pixel_path = os.path.join(yaml_dir, '_temp_furniture_pixel.yaml')

        # ── 消融控制：按需跳过输入 ──
        _traj_path = trajectory_path
        if self.ablation.get('skip_trajectory'):
            logger.info("[ABLATION] 跳过轨迹输入")
            _traj_path = None
        _device_path = smart_device_path
        if self.ablation.get('skip_smart_device'):
            logger.info("[ABLATION] 跳过智能设备输入")
            _device_path = None
        _hatch_path = hatch_mask_path
        if self.ablation.get('skip_hatch'):
            logger.info("[ABLATION] 跳过 hatch 阴影家具检测")
            _hatch_path = None

        self.furniture_agent.process(image_path, seg_yaml_path,
                                     output_path=temp_pixel_path,
                                     trajectory_json_path=_traj_path,
                                     hatch_mask_path=_hatch_path,
                                     smart_device_path=_device_path,
                                     intermediate_yaml_dir=yaml_dir,
                                     ablation=self.ablation)

        # ── 2b. 坐标统一 (像素 → 世界) ──
        logger.info("=== Step 2b: 坐标统一 (像素 → 世界) ===")
        img_shape = None
        if image_path and os.path.exists(image_path):
            import cv2
            img = cv2.imread(image_path)
            if img is not None:
                img_shape = img.shape
        converter = CoordinateConverter.from_yaml(seg_yaml_path, img_shape=img_shape)
        with open(temp_pixel_path, 'r', encoding='utf-8') as f:
            layout = yaml.safe_load(f)
        world_layout = converter.convert_layout_to_world(layout)
        os.remove(temp_pixel_path)  # 清理像素坐标临时文件

        # 加载轨迹数据
        trajectory_data = []
        if _traj_path and os.path.exists(_traj_path):
            with open(_traj_path, 'r', encoding='utf-8') as f:
                trajectory_data = normalize_trajectory_durations(json.load(f))
            logger.info(f"行为轨迹: {len(trajectory_data)} 个停驻点")

        # ── 写入 02_furniture_world_{code}.yaml ──
        world_path = os.path.join(yaml_dir, f'02_furniture_world_{self.code}.yaml')
        with open(world_path, 'w', encoding='utf-8') as f:
            yaml.dump(world_layout, f, default_flow_style=None,
                      allow_unicode=True, sort_keys=False, indent=2)
        logger.info(f"Step 2 输出 -> {world_path}")
        return world_path, trajectory_data

    # ==========================================================
    # Step 3: 房间类型识别 (RoomAgent)
    # ==========================================================
    def room_analysis_step(self, world_yaml_path: str, trajectory_data: list) -> list:
        """推理每个房间的类型"""
        logger.info("=== Step 3: 房间类型识别 (RoomAgent) ===")

        with open(world_yaml_path, 'r', encoding='utf-8') as f:
            world_layout = yaml.safe_load(f)

        result = self.room_agent.analyze_rooms(world_layout, trajectory_data,
                                                ablation=self.ablation)
        room_analyses = result

        logger.info(f"房间类型推理完成: {len(room_analyses)} 个房间")
        for ra in room_analyses:
            logger.info(f"  {ra.room_id}: {ra.room_type} (conf={ra.confidence:.2f})")

        return room_analyses

    # ==========================================================
    # Step 4: 行为模式分析 (BehaviorAgent, temperature=0.0)
    # ==========================================================
    def behavior_analysis_step(self, world_yaml_path: str, trajectory_data: list,
                                room_analyses: list) -> list:
        """分析每个房间的行为模式（LLM 以 temperature=0.0 输出，最大化确定性）

        ablation['skip_behavior_module'] 为 True 时跳过行为语义模块，
        返回空行为证据（单模块移除消融）。
        """
        logger.info("=== Step 4: 行为模式预测 (BehaviorAgent) ===")

        if self.ablation.get('skip_behavior_module'):
            logger.info("[ABLATION] 跳过行为语义模块，行为证据置空")
            return []

        with open(world_yaml_path, 'r', encoding='utf-8') as f:
            world_layout = yaml.safe_load(f)

        result = self.behavior_agent.generate_behavior_analyses(
            world_layout, trajectory_data, room_analyses
        )
        behavior_analyses = result.behavior_analyses

        logger.info(f"行为分析完成: {len(behavior_analyses)} 个房间")
        return behavior_analyses

    # ==========================================================
    # Step 5: 家具命名 → 补充名称, 写入最终 YAML
    # ==========================================================
    def step5_furniture_naming_and_merge(self, world_yaml_path: str,
                                          room_analyses: list,
                                          behavior_analyses: list,
                                          output_dir: str) -> str:
        """Step 5: 家具命名 + 合并最终输出 → 03_final_{code}.yaml"""
        yaml_dir = os.path.join(output_dir, 'yaml')
        os.makedirs(yaml_dir, exist_ok=True)
        final_path = os.path.join(yaml_dir, f'03_final_{self.code}.yaml')

        logger.info("=== Step 5a: 家具命名 (FurnitureNamingAgent) ===")
        with open(world_yaml_path, 'r', encoding='utf-8') as f:
            world_layout = yaml.safe_load(f)

        # 设置 LLM 推理输出保存目录
        self.naming_agent._output_dir = yaml_dir

        furniture_namings = self.naming_agent.name_furniture(
            world_layout, room_analyses, behavior_analyses,
            ablation=self.ablation
        )
        # The ablation runner reports ProtV at the pre-final naming stage.
        # Keep that immutable snapshot in memory; do not write it into the
        # final layout or let subsequent merge/repair logic mutate it.
        self.last_pre_final_namings = list(
            getattr(self.naming_agent, 'last_pre_final_namings', []))
        self.last_pre_final_room_analyses = list(room_analyses)
        logger.info(f"家具命名完成: {len(furniture_namings)} 件")

        # ── 5b. 合并最终输出 ──
        logger.info("=== Step 5b: 合并最终输出 ===")
        room_type_map = {ra.room_id: ra.room_type for ra in room_analyses}
        furniture_name_map = {fn.furniture_id: fn.name for fn in furniture_namings}

        for room in world_layout.get('house', {}).get('rooms', []):
            rid = room.get('id', room.get('name', ''))
            orig_name = room.get('name', rid)
            room_type = room_type_map.get(rid)
            if room_type:
                room['name'] = room_type
            room.pop('id', None)
            room.pop('behavior', None)

            new_room_name = room_type if room_type else orig_name
            fur_idx = 0
            for fur in room.get('furniture', []):
                fur_idx += 1
                fid = fur.get('id', f'{orig_name}_fur{fur_idx}')
                orig_fur_name = (fur.get('name', '') or '').strip()
                if fid in furniture_name_map:
                    pos = fur.pop('position', None)
                    fur['id'] = f'{new_room_name}_fur{fur_idx}'
                    fur['name'] = furniture_name_map[fid]
                    if pos is not None:
                        fur['position'] = pos
                # 强制保护：智能设备名不可被覆盖
                if orig_fur_name and self.naming_agent._is_smart_device(orig_fur_name):
                    if fur['name'] != orig_fur_name:
                        logger.warning(f"  智能设备名被覆盖: {orig_fur_name} → {fur['name']}, 已恢复")
                        fur['name'] = orig_fur_name

        world_layout.pop('smart_devices', None)

        # ── 5c. 家具重叠检测 & 自动修正 ──
        logger.info("=== Step 5c: 家具重叠检测 ===")
        self._check_and_fix_furniture_overlaps(world_layout)

        # 保存最终输出
        with open(final_path, 'w', encoding='utf-8') as f:
            yaml.dump(world_layout, f, default_flow_style=None,
                      allow_unicode=True, sort_keys=False, indent=2)
        logger.info(f"最终输出 -> {final_path}")

        return final_path

    # ==========================================================
    # Step 5c 辅助: 家具重叠检测 & 自动修正
    # ==========================================================
    @staticmethod
    def _furniture_bboxes_overlap(pos_a: dict, pos_b: dict) -> float:
        """计算两个家具的 IoU (Intersection over Union)。
        
        Args:
            pos_a, pos_b: {'x', 'y', 'width', 'height'} 字典
        
        Returns:
            IoU 值 (0.0 ~ 1.0)，0 表示不重叠
        """
        ax1, ay1 = pos_a['x'], pos_a['y']
        ax2, ay2 = ax1 + pos_a['width'], ay1 + pos_a['height']
        bx1, by1 = pos_b['x'], pos_b['y']
        bx2, by2 = bx1 + pos_b['width'], by1 + pos_b['height']

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = ix2 - ix1
        ih = iy2 - iy1
        if iw <= 0 or ih <= 0:
            return 0.0

        inter = iw * ih
        area_a = pos_a['width'] * pos_a['height']
        area_b = pos_b['width'] * pos_b['height']
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _resolve_furniture_overlap(fur_smaller: dict, fur_larger: dict):
        """将较小的家具从重叠区域中缩出。

        计算两个家具的重叠区域，将较小的家具从重叠方向缩到不再重叠。
        沿重叠方向（x 或 y，选重叠较少的方向）缩小较小家具的尺寸。

        Args:
            fur_smaller: 被缩小的家具（面积较小者），就地修改 position
            fur_larger:   保留不变的家具
        """
        pos_s = fur_smaller['position']
        pos_l = fur_larger['position']

        sx1, sy1 = pos_s['x'], pos_s['y']
        sx2, sy2 = sx1 + pos_s['width'], sy1 + pos_s['height']
        lx1, ly1 = pos_l['x'], pos_l['y']
        lx2, ly2 = lx1 + pos_l['width'], ly1 + pos_l['height']

        # 重叠区域
        ox1 = max(sx1, lx1)
        oy1 = max(sy1, ly1)
        ox2 = min(sx2, lx2)
        oy2 = min(sy2, ly2)
        overlap_w = ox2 - ox1
        overlap_h = oy2 - oy1

        if overlap_w <= 0 or overlap_h <= 0:
            return  # 无重叠

        # 选择重叠较浅的方向来缩（减少修改量）
        # 同时保证缩小后还有最小尺寸（0.5 dm）
        MIN_SIZE = 0.5

        if overlap_w <= overlap_h:
            # 水平方向缩：判断 smaller 在 larger 的哪一侧
            if sx1 + pos_s['width'] / 2 < lx1 + pos_l['width'] / 2:
                # smaller 在左侧 → 缩右边
                shrink = overlap_w
                new_w = pos_s['width'] - shrink
                if new_w >= MIN_SIZE:
                    pos_s['width'] = round(new_w, 2)
                else:
                    # 宽度不够缩，尝试调整 x
                    pos_s['x'] = round(pos_s['x'] - max(0, MIN_SIZE - new_w), 2)
                    pos_s['width'] = round(MIN_SIZE, 2)
            else:
                # smaller 在右侧 → 缩左边
                shrink = overlap_w
                new_w = pos_s['width'] - shrink
                if new_w >= MIN_SIZE:
                    pos_s['x'] = round(pos_s['x'] + shrink, 2)
                    pos_s['width'] = round(new_w, 2)
                else:
                    pos_s['width'] = round(MIN_SIZE, 2)
        else:
            # 垂直方向缩：判断 smaller 在 larger 的哪一侧
            if sy1 + pos_s['height'] / 2 < ly1 + pos_l['height'] / 2:
                # smaller 在上方 → 缩底部
                shrink = overlap_h
                new_h = pos_s['height'] - shrink
                if new_h >= MIN_SIZE:
                    pos_s['height'] = round(new_h, 2)
                else:
                    pos_s['y'] = round(pos_s['y'] - max(0, MIN_SIZE - new_h), 2)
                    pos_s['height'] = round(MIN_SIZE, 2)
            else:
                # smaller 在下方 → 缩顶部
                shrink = overlap_h
                new_h = pos_s['height'] - shrink
                if new_h >= MIN_SIZE:
                    pos_s['y'] = round(pos_s['y'] + shrink, 2)
                    pos_s['height'] = round(new_h, 2)
                else:
                    pos_s['height'] = round(MIN_SIZE, 2)

    def _check_and_fix_furniture_overlaps(self, world_layout: dict, iou_threshold: float = 0.01):
        """检测并修复所有房间内家具之间的重叠。

        对每个房间内的家具逐对检查 IoU，若有重叠则缩小面积较小的家具。

        Args:
            world_layout: 房屋布局字典，包含 house.rooms[].furniture[]
            iou_threshold: IoU 阈值，超过此值视为重叠（默认 0.01 = 任何重叠）
        """
        rooms = world_layout.get('house', {}).get('rooms', [])
        fixed_count = 0

        for room in rooms:
            room_name = room.get('name', room.get('id', 'unknown'))
            furniture = room.get('furniture', [])
            n = len(furniture)
            if n < 2:
                continue

            # 逐对检查
            for i in range(n):
                for j in range(i + 1, n):
                    fur_a = furniture[i]
                    fur_b = furniture[j]
                    pos_a = fur_a.get('position')
                    pos_b = fur_b.get('position')
                    if not pos_a or not pos_b:
                        continue

                    iou = self._furniture_bboxes_overlap(pos_a, pos_b)
                    if iou < iou_threshold:
                        continue

                    name_a = fur_a.get('name', '?')
                    name_b = fur_b.get('name', '?')

                    # 面积小的家具缩
                    area_a = pos_a['width'] * pos_a['height']
                    area_b = pos_b['width'] * pos_b['height']

                    if area_a <= area_b:
                        victim, keeper = fur_a, fur_b
                        v_name, k_name = name_a, name_b
                    else:
                        victim, keeper = fur_b, fur_a
                        v_name, k_name = name_b, name_a

                    logger.warning(
                        f"[OVERLAP] {room_name}: {v_name} 与 {k_name} 重叠 "
                        f"(IoU={iou:.1%})，缩小 {v_name}"
                    )

                    # 修复前记录
                    old_pos = dict(victim['position'])
                    self._resolve_furniture_overlap(victim, keeper)

                    # 检查修复后是否仍然重叠（最多重试 3 次）
                    retry = 0
                    while retry < 3:
                        iou_after = self._furniture_bboxes_overlap(
                            victim['position'], keeper['position'])
                        if iou_after < iou_threshold:
                            break
                        self._resolve_furniture_overlap(victim, keeper)
                        retry += 1

                    new_pos = victim['position']
                    logger.info(
                        f"  {v_name}: ({old_pos['x']:.1f},{old_pos['y']:.1f}) "
                        f"{old_pos['width']:.1f}x{old_pos['height']:.1f}"
                        f" → ({new_pos['x']:.1f},{new_pos['y']:.1f}) "
                        f"{new_pos['width']:.1f}x{new_pos['height']:.1f}"
                    )
                    fixed_count += 1

        if fixed_count > 0:
            logger.info(f"共修复 {fixed_count} 处家具重叠")
        else:
            logger.info("未检测到家具重叠")

    # ==========================================================
    # 全流程运行 (5 步)
    # ==========================================================
    def run(self, image_path: str, smart_device_path: str = None,
            trajectory_path: str = None, output_dir: str = None,
            skip_seg: bool = False, skip_agents: bool = False) -> str:
        """执行全流程 5 步管线

        Step 1: 房间分割 → 01_seg_pixel.yaml
        Step 2: 家具检测 + 坐标统一 + 智能设备匹配 → 02_furniture_world.yaml
        Step 3: 房间类型识别 (RoomAgent)
        Step 4: 行为模式分析 (确定性)
        Step 5: 家具命名 + 合并输出 → 03_final.yaml
        """
        if output_dir is None:
            output_dir = os.path.join(PROJECT_ROOT, 'output', self.code)

        yaml_dir = os.path.join(output_dir, 'yaml')
        os.makedirs(yaml_dir, exist_ok=True)

        # ── Step 1: 房间分割 ──
        if skip_seg:
            # 先检查当前 output_dir，再检查持久化目录 (多轮评估等场景)
            seg_path = os.path.join(yaml_dir, f'01_seg_pixel_{self.code}.yaml')
            if not os.path.exists(seg_path):
                seg_path = os.path.join(PROJECT_ROOT, 'output', self.code, 'yaml',
                                        f'01_seg_pixel_{self.code}.yaml')
            if not os.path.exists(seg_path):
                raise FileNotFoundError(f"skip_seg=True but no existing YAML: "
                                        f"{output_dir}/yaml/ or output/{self.code}/yaml/")
            logger.info("=== Step 1: 跳过 run_seg, 使用已有 YAML ===")
        else:
            seg_path = self.run_seg_step(image_path, output_dir)

        # ── Step 2: 家具检测 + 坐标统一 + 智能设备匹配 ──
        greyroom_path = os.path.join(output_dir, 'masks', 'greyroom.png')
        hatch_mask_path = os.path.join(output_dir, 'masks', 'hatch_mask.png')
        # skip_seg 时，masks 可能只在持久化目录中
        if not os.path.exists(greyroom_path):
            greyroom_persist = os.path.join(PROJECT_ROOT, 'output', self.code,
                                            'masks', 'greyroom.png')
            if os.path.exists(greyroom_persist):
                greyroom_path = greyroom_persist
        if not os.path.exists(hatch_mask_path):
            hatch_persist = os.path.join(PROJECT_ROOT, 'output', self.code,
                                         'masks', 'hatch_mask.png')
            if os.path.exists(hatch_persist):
                hatch_mask_path = hatch_persist
        hm_path = hatch_mask_path if os.path.exists(hatch_mask_path) else None
        seg_image = greyroom_path if os.path.exists(greyroom_path) else image_path

        if not skip_agents:
            world_yaml_path, trajectory_data = self.step2_furniture_and_devices(
                seg_image, seg_path, trajectory_path,
                smart_device_path, hm_path, output_dir)
        else:
            logger.info("=== Step 2: 跳过家具检测 ===")
            # 需要至少做坐标转换以保持后续兼容
            import shutil
            world_yaml_path = os.path.join(yaml_dir, f'02_furniture_world_{self.code}.yaml')
            shutil.copy2(seg_path, world_yaml_path)
            trajectory_data = []
            if trajectory_path and os.path.exists(trajectory_path):
                with open(trajectory_path, 'r', encoding='utf-8') as f:
                    trajectory_data = json.load(f)

        if skip_agents:
            logger.info("=== LLM Agents 全部跳过 ===")
            final_path = os.path.join(yaml_dir, f'03_final_{self.code}.yaml')
            import shutil
            shutil.copy2(world_yaml_path, final_path)
            return final_path

        # ── Step 3: 房间类型识别 ──
        room_analyses = self.room_analysis_step(world_yaml_path, trajectory_data)

        # ── Step 4: 行为模式预测 ──
        behavior_analyses = self.behavior_analysis_step(
            world_yaml_path, trajectory_data, room_analyses)

        # ── Step 5: 家具命名 + 合并输出 ──
        final_path = self.step5_furniture_naming_and_merge(
            world_yaml_path, room_analyses, behavior_analyses, output_dir)

        return final_path


# ══════════════════════════════════════════════════════════════
# 命令行入口
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='全流程管线')
    parser.add_argument('--image', default=None,
                        help='户型轨迹图 (默认: photo2yaml/photo/room_processed.png)')
    parser.add_argument('--smart-device', default=None,
                        help='智能设备位置 YAML (默认: input/Smart_device/smartDevice_1.yaml)')
    parser.add_argument('--trajectory', default=None,
                        help='行为轨迹 JSON (默认: input/traj/oneday/1-Engineer_trajectory.json)')
    parser.add_argument('--output-dir', default=None,
                        help='输出目录 (默认: photo2yaml/output)')
    parser.add_argument('--skip-seg', action='store_true',
                        help='跳过 run_seg (使用已有的 01_seg_pixel.yaml)')
    parser.add_argument('--skip-agents', action='store_true',
                        help='跳过所有 LLM Agents (仅分割+坐标统一)')
    args = parser.parse_args()

    # 默认路径
    img = args.image or os.path.join(
        PROJECT_ROOT, 'photo2yaml', 'photo', 'room_processed.png'
    )
    device = args.smart_device or os.path.join(
        PROJECT_ROOT, 'input', 'Smart_device', 'smartDevice_1.yaml'
    )
    traj = args.trajectory or os.path.join(
        PROJECT_ROOT, 'input', 'traj', 'oneday', '1-Engineer_trajectory.json'
    )
    out_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, 'photo2yaml', 'output'
    )

    # 检查输入文件
    if not os.path.exists(img):
        print(f"[错误] 图片不存在: {img}")
        sys.exit(1)
    if device and not os.path.exists(device):
        print(f"[警告] 智能设备文件不存在: {device}，跳过")
        device = None
    if traj and not os.path.exists(traj):
        print(f"[警告] 轨迹文件不存在: {traj}，跳过")
        traj = None

    # API key：从环境变量读取
    api_key = os.environ.get("FURNITURE_API_KEY")
    base_url = os.environ.get("FURNITURE_BASE_URL")
    if not api_key:
        print("[错误] 环境变量 FURNITURE_API_KEY 未设置，请在 .env 文件中配置")
        sys.exit(1)

    pipeline = Pipeline(api_key=api_key, base_url=base_url)
    result = pipeline.run(
        image_path=img,
        smart_device_path=device,
        trajectory_path=traj,
        output_dir=out_dir,
        skip_seg=args.skip_seg,
        skip_agents=args.skip_agents,
    )
    print(f"\n✅ 全流程完成！输出: {result}")
