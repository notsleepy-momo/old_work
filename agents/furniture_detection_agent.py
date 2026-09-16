"""
家具检测Agent — 扫地机轨迹 + 人停留点 + 房间轮廓 → 家具位置

输入:
  - output/masks/greyroom.png     (3值图: 墙=0黑, 地板=128灰, 轨迹=255白)
  - output/floorplan_real.yaml    (房间分割，像素坐标 + metadata SCALE/bl)
  - input/traj/.../*_trajectory.json (人的停留点，世界坐标dm)

流程:
  1. metadata 读取 SCALE/bl_x/bl_y → 世界坐标(dm)转像素
  2. 轨迹 JSON 加载 → 按5px网格聚合停留时长
  3. 逐房间: YAML坐标裁剪 + CV(Otsu检测不可达区) + LLM推理
  4. CV 预处理的中间结果保存到 output/masks/
  5. 去重后写入 room_real.yaml
  ⚠ 只推理位置和大小，不推理家具类别
"""
import sys, os, json, re, base64, logging
import cv2
import numpy as np
import yaml
from typing import Dict, List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from sklearn.cluster import DBSCAN

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llm_config import (
    DEFAULT_LLM_MODEL,
    require_shared_model,
    resolve_api_key,
    resolve_base_url,
)

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import FURNITURE_DETECTION_AGENT_PROMPT


class FurnitureDetectionAgent:
    """家具位置检测（CV Otsu + LLM 推理），输出 room_real.yaml"""

    # ── 智能设备真实尺寸（世界坐标 dm，1 dm = 10 cm）────────────────
    # 用于判断检测框是否与设备类型匹配
    DEVICE_EXPECTED_SIZES = {
        'refrigerator':    {'min_w': 4, 'max_w': 10,  'min_h': 4, 'max_h': 10},
        'oven':            {'min_w': 3, 'max_w': 10,  'min_h': 3,  'max_h': 10},
        'stove':           {'min_w': 3, 'max_w': 10,  'min_h': 3,  'max_h': 10},
        'tv_stand':        {'min_w': 4, 'max_w': 7, 'min_h': 6,  'max_h': 20},
        'tv':              {'min_w': 4, 'max_w': 7, 'min_h': 6,  'max_h': 20},
        'microwave':       {'min_w': 3, 'max_w': 10,  'min_h': 3,  'max_h': 10},
        'washing_machine': {'min_w': 3, 'max_w': 10,  'min_h': 3,  'max_h': 10},
        'treadmill':       {'min_w': 6, 'max_w': 15,  'min_h': 6,  'max_h': 15},
    }

    def __init__(self, api_key: str = None, base_url: str = None,
                 model: str = DEFAULT_LLM_MODEL, callbacks: list = None):
        self.api_key = resolve_api_key(api_key, required=False)
        self.base_url = resolve_base_url(base_url)
        self.model = require_shared_model(model)
        self.callbacks = callbacks or []
        self.llm = None
        self._init_llm()
        self._output_path = None
        self._scale = 1.0
        self._bl_x = 0
        self._bl_y = 0

    def _init_llm(self):
        try:
            common_kwargs = {"max_retries": 5, "timeout": 300}
            key = self.api_key
            if key:
                common_kwargs["api_key"] = key
            if self.base_url:
                common_kwargs["base_url"] = self.base_url
            if self.callbacks:
                common_kwargs["callbacks"] = self.callbacks
            if "api_key" in common_kwargs:
                self.llm = ChatOpenAI(model=self.model, temperature=0.0, **common_kwargs)
                logger.info(f"LLM initialized (model={self.model})")
            else:
                logger.warning("No API key — LLM classification disabled")
        except Exception as e:
            logger.warning(f"LLM init failed: {e}")

    # ==============================================================
    # 坐标转换 (世界 → 像素)
    # ==============================================================
    def _load_coordinate_params(self, layout: Dict):
        """YAML metadata → SCALE(dm/px), bl"""
        meta = layout.get('metadata', {})
        self._scale = float(meta.get('SCALE', 1.0))
        self._bl_x = float(meta.get('bl_x', 0))
        self._bl_y = float(meta.get('bl_y', 0))
        logger.info(f"坐标转换: SCALE={self._scale} dm/px, bl=({self._bl_x},{self._bl_y})")

    def _world_to_pixel(self, wx: float, wy: float, img_h: int = None) -> Tuple[float, float]:
        """世界坐标(dm) → 像素坐标。SCALE单位dm/px"""
        px = wx / self._scale + self._bl_x
        py = self._bl_y - wy / self._scale
        if img_h is not None and (py < 0 or py >= img_h):
            logger.debug(f"点({wx:.1f},{wy:.1f})→像素({px:.0f},{py:.0f})越界")
        return px, py

    # ==============================================================
    # 轨迹加载与转换
    # ==============================================================
    def _load_trajectory(self, json_path: str, img_h: int = None) -> List[Dict]:
        """加载行为轨迹 JSON，世界坐标→像素坐标，按5px网格聚合停留时长"""
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_traj = json.load(f)

        stay_map: Dict[Tuple[int, int], float] = {}
        # 自适应网格大小：基于 SCALE，高分辨(scale小)用小网格，低分辨(scale大)用大网格
        grid_size = max(3, min(10, int(5 / max(self._scale, 0.1))))
        for entry in raw_traj:
            center = entry.get('center_position', {})
            wx = center.get('x', 0)
            wy = center.get('y', 0)
            duration_min = float(entry.get('duration', 0))
            if duration_min <= 0:
                continue
            px_raw, py_raw = self._world_to_pixel(wx, wy, img_h)
            # 聚合到自适应网格
            grid_x = int(round(px_raw / grid_size) * grid_size)
            grid_y = int(round(py_raw / grid_size) * grid_size)
            key = (grid_x, grid_y)
            stay_map[key] = stay_map.get(key, 0) + duration_min

        logger.info(f"轨迹: {len(raw_traj)} 条 → 自适应{grid_size}px网格聚合 {len(stay_map)} 个停留点")
        return [{'pixel_x': k[0], 'pixel_y': k[1], 'duration_min': v}
                for k, v in stay_map.items() if v > 1.0]

    # ==============================================================
    # 停留热度图生成
    # ==============================================================
    def _generate_stay_heatmap(self, traj_points: List[Dict], img_shape: Tuple[int, int],
                                layout: Dict, bg_image: np.ndarray = None) -> np.ndarray:
        """根据行为停留点生成全图热度图（叠加在原图上）

        热度值 = log1p(stay_minutes)，颜色越亮/红 = 停留越长。

        Args:
            bg_image: 原始户型图（可选），若提供则热力值叠加在原图上，便于判断对齐

        Returns:
            BGR 热度图 (uint8, same size as original image)
        """
        h_img, w_img = img_shape[:2]
        heat_layer = np.zeros((h_img, w_img), dtype=np.float32)

        if not traj_points:
            bg = bg_image.copy() if bg_image is not None else np.zeros((h_img, w_img, 3), dtype=np.uint8)
            return bg

        # 自适应扩散半径
        base_sigma = max(12.0, min(w_img, h_img) / 60.0)

        kr = int(base_sigma * 2.5)
        ksize = kr * 2 + 1
        gauss_kernel = cv2.getGaussianKernel(ksize, base_sigma)
        gauss_kernel = gauss_kernel @ gauss_kernel.T

        for pt in traj_points:
            px = int(pt['pixel_x'])
            py = int(pt['pixel_y'])
            dur = pt['duration_min']
            if not (0 <= px < w_img and 0 <= py < h_img):
                continue

            log_dur = np.log1p(dur)

            x_start = max(0, px - kr)
            x_end = min(w_img, px + kr + 1)
            y_start = max(0, py - kr)
            y_end = min(h_img, py + kr + 1)

            for yy in range(y_start, y_end):
                ky = yy - py + kr
                for xx in range(x_start, x_end):
                    kx = xx - px + kr
                    weight = gauss_kernel[ky, kx]
                    heat_layer[yy, xx] = max(heat_layer[yy, xx], log_dur * weight)

        # === 墙体掩码（greyroom: 墙=0黑, 地板=128, 轨迹=255） ===
        if bg_image is not None:
            gray = cv2.cvtColor(bg_image, cv2.COLOR_BGR2GRAY)
            # 纯黑背景（室外）→ 不可通行
            _, bg_mask = cv2.threshold(gray, 10, 1, cv2.THRESH_BINARY)
            # 墙=0 → 不可通行
            wall_mask = (gray < 5).astype(np.uint8)
            walkable = (bg_mask.astype(bool) & (wall_mask == 0))
            heat_layer[~walkable] = 0

        if heat_layer.max() <= 0:
            bg = bg_image.copy() if bg_image is not None else np.zeros((h_img, w_img, 3), dtype=np.uint8)
            return bg

        # 归一化并 gamma 校正
        heat_norm = (heat_layer / heat_layer.max() * 255).astype(np.uint8)
        heat_norm = (255 * (heat_norm.astype(np.float32) / 255) ** 0.4).astype(np.uint8)

        # INFERNO 色图
        heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_INFERNO)

        # === 核心：将热度图叠加到原图上 ===
        if bg_image is not None:
            bg = bg_image.copy()
            # 创建热力区域的二值掩码（有热度的像素）
            _, heat_mask = cv2.threshold(heat_norm, 1, 255, cv2.THRESH_BINARY)
            # 将灰度掩码转为 3 通道 BGR
            heat_mask_3ch = cv2.cvtColor(heat_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

            # 热度区域：70% 热力色 + 30% 原图（保持可见纹理）
            # 非热度区域：30% 原图亮度（变暗以突出热力区）
            fg = cv2.addWeighted(heat_color, 0.7, bg, 0.3, 0)
            bg_dim = cv2.addWeighted(bg, 0.3, np.zeros_like(bg), 0.7, 0)

            # 按掩码合成：热力区用 fg，非热力区用变暗背景
            overlay = (fg * heat_mask_3ch + bg_dim * (1 - heat_mask_3ch)).astype(np.uint8)
            heat_color = overlay

        # 叠加房间边界框（白色）
        for room in layout.get('house', {}).get('rooms', []):
            pos = room.get('position', {})
            rx = int(pos.get('x', 0))
            ry = int(pos.get('y', 0))
            rw = int(pos.get('width', 0))
            rh = int(pos.get('height', 0))
            cv2.rectangle(heat_color, (rx, ry), (rx + rw, ry + rh), (255, 255, 255), 1)

        # 叠加房间编号
        for i, room in enumerate(layout.get('house', {}).get('rooms', [])):
            pos = room.get('position', {})
            rx = int(pos.get('x', 0))
            ry = int(pos.get('y', 0))
            cv2.putText(heat_color, f"R{i+1}", (rx + 3, ry + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return heat_color

    def _crop_heatmap_for_room(self, heatmap: np.ndarray, 
                                px_left: int, px_top: int,
                                px_right: int, px_bot: int) -> np.ndarray:
        """从全图热度图中裁剪房间区域"""
        h, w = heatmap.shape[:2]
        x1 = max(0, px_left)
        y1 = max(0, px_top)
        x2 = min(w, px_right)
        y2 = min(h, px_bot)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((10, 10, 3), dtype=np.uint8)
        return heatmap[y1:y2, x1:x2].copy()

    def _filter_trajectory_for_room(self, traj_points: List[Dict],
                                     rx: float, ry: float, rw: float, rh: float,
                                     crop_left: float = None, crop_top: float = None,
                                     crop_right: float = None, crop_bot: float = None,
                                     margin: int = 0) -> List[Dict]:
        """筛选落在房间范围内的轨迹点。

        优先使用裁剪区域（基于红色墙线检测，更准确），
        回退到 YAML 房间边界框。
        
        Args:
            crop_left/top/right/bot: 裁剪区域边界（基于红墙检测）
            margin: 在边界外扩的容忍像素数
        """
        # 确定筛选边界：优先使用裁剪区域（红墙检测更准确）
        if crop_left is not None:
            x1, y1 = crop_left - margin, crop_top - margin
            x2, y2 = crop_right + margin, crop_bot + margin
        else:
            x1, y1 = rx, ry
            x2, y2 = rx + rw, ry + rh

        result = []
        for pt in traj_points:
            px = pt['pixel_x']
            py = pt['pixel_y']
            if x1 <= px <= x2 and y1 <= py <= y2:
                result.append(pt)
        return result

    def _build_trajectory_info(self, room_traj: List[Dict]) -> str:
        """构建轨迹信息字符串（供 LLM 使用）"""
        if not room_traj:
            return "(No human stay points in this room)"

        # 按停留时长降序排列
        sorted_pts = sorted(room_traj, key=lambda p: p['duration_min'], reverse=True)
        lines = []
        for pt in sorted_pts:
            dur = pt['duration_min']
            px = pt['pixel_x']
            py = pt['pixel_y']
            if dur >= 60:
                tag = "LONG_SLEEP/REST"
            elif dur >= 20:
                tag = "MEDIUM_ACTIVITY"
            elif dur >= 5:
                tag = "SHORT_ACTIVITY"
            else:
                tag = "PASSING"
            lines.append(f"  ({px:.0f}, {py:.0f}) d={dur:.0f}min [{tag}]")

        header = f"Total stay points: {len(sorted_pts)}, Total duration: {sum(p['duration_min'] for p in sorted_pts):.0f}min"
        return header + "\n" + "\n".join(lines)

    # ==============================================================
    # CV 预处理：轨迹覆盖空洞检测 → 连通域标记 → 种子矩形扩张
    # ==============================================================
    # 【可调参数 — 绝对值阈值通过 _compute_adaptive_thresholds 动态计算】
    _BBOX_EXPAND        = 1            # bbox 向外扩 px（固定值，与分辨率无关，不宜过大以免吞轨迹）
    _MIN_SOLIDITY       = 0.38         # 实体度下限（比例值，自动跨分辨率适配）
    _MIN_EXTENT         = 0.5          # 填充率下限（比例值，自动跨分辨率适配）
    _DILATE_ITERS       = 2            # 轨迹膨胀迭代次数

    # ==============================================================
    # Furniture Topology Refinement — 图拓扑重建
    # ==============================================================
    def _expand_hatch_to_trajectory(self, candidates: List[Dict],
                                     traj_mask: np.ndarray,
                                     room_w: int, room_h: int) -> List[Dict]:
        """将 hatch 候选框向外扩展到最近的轨迹边界。

        对每条边：向外逐像素扫描，直到遇到轨迹像素或到达房间边界。
        如果该方向在检查范围内没有轨迹，则扩展到轨迹边界。

        Args:
            candidates: hatch 候选列表 [{bbox: {x,y,w,h}, ...}]
            traj_mask: 轨迹二值掩码
            room_w, room_h: 房间裁剪尺寸
        """
        if not candidates:
            return candidates

        h, w = traj_mask.shape[:2]
        MAX_EXPAND = 30  # 单方向最大扩展30px

        result = []
        for c in candidates:
            b = c['bbox']
            bx, by = b['x'], b['y']
            bw, bh = b['width'], b['height']

            # 四个方向分别扩展
            # 左
            expand_left = 0
            for d in range(1, MAX_EXPAND + 1):
                check_x = bx - d
                if check_x < 0:
                    break
                col = traj_mask[by:by + bh, check_x]
                if np.any(col > 0):
                    expand_left = d
                    break

            # 右
            expand_right = 0
            for d in range(1, MAX_EXPAND + 1):
                check_x = bx + bw + d - 1
                if check_x >= w:
                    break
                col = traj_mask[by:by + bh, check_x]
                if np.any(col > 0):
                    expand_right = d
                    break

            # 上
            expand_top = 0
            for d in range(1, MAX_EXPAND + 1):
                check_y = by - d
                if check_y < 0:
                    break
                row = traj_mask[check_y, bx:bx + bw]
                if np.any(row > 0):
                    expand_top = d
                    break

            # 下
            expand_bottom = 0
            for d in range(1, MAX_EXPAND + 1):
                check_y = by + bh + d - 1
                if check_y >= h:
                    break
                row = traj_mask[check_y, bx:bx + bw]
                if np.any(row > 0):
                    expand_bottom = d
                    break

            # 应用扩展
            new_x = bx - expand_left
            new_y = by - expand_top
            new_w = bw + expand_left + expand_right
            new_h = bh + expand_top + expand_bottom

            c['bbox'] = {'x': new_x, 'y': new_y, 'width': new_w, 'height': new_h}

        return candidates

    # ==============================================================
    # 轨迹家具检测（简化版）：findContours 寻找轨迹闭环内部空白
    # ==============================================================
    def _detect_trajectory_surrounded_regions(self, room_img: np.ndarray,
                                               traj_mask: np.ndarray,
                                               hatch_exclude: np.ndarray,
                                               wall_mask: np.ndarray,
                                               name: str = "",
                                               mask_dir: str = None) -> List[Dict]:
        """检测轨迹闭环包围的空白区域（补充家具候选）。

        简化方案 (v2):
        1. 对轨迹掩码做 findContours → 找内部闭环
        2. 闭环内部面积较大 → 可能是家具占地区
        3. 排除 hatch 已有区域 + 墙体
        4. minAreaRect 输出

        Returns:
            [{'bbox': {x,y,width,height}, 'loop_score':float,
              'candidate_type': 'trajectory_surrounded', ...}, ...]
        """
        h, w = room_img.shape[:2]
        if h < 5 or w < 5:
            return []

        # 1) 轨迹闭运算 + 膨胀 → 形成连续闭环
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        traj_closed = cv2.morphologyEx(traj_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

        # 2) findContours 找闭环内部
        contours, hierarchy = cv2.findContours(traj_closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None or len(contours) == 0:
            return []

        # 3) 生成闭环内部掩码
        interior_mask = np.zeros((h, w), dtype=np.uint8)
        hierarchy = hierarchy[0]
        for i, (cnt, hrec) in enumerate(zip(contours, hierarchy)):
            # hrec[3] == parent index: 内部轮廓（有父轮廓的子轮廓）
            if hrec[3] >= 0:
                cv2.drawContours(interior_mask, [cnt], -1, 255, -1)

        # 排除 hatch + 墙体
        interior_mask = cv2.bitwise_and(interior_mask, cv2.bitwise_not(hatch_exclude))
        interior_mask = cv2.bitwise_and(interior_mask, cv2.bitwise_not(wall_mask))

        # 4) 连通域 → 家具候选
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(interior_mask, connectivity=8)
        if num_labels < 2:
            return []

        room_area = h * w
        min_area = max(30, int(room_area * 0.001))
        max_area = int(room_area * 0.5)
        # 内接矩形最小面积：对应世界坐标面积 20（dm²），按 scale(dm/px) 折算为像素面积
        # 扫地机器人半径为17.6dm，内接矩形最小面积为5dm²
        scale = getattr(self, '_scale', 0.08)
        MIN_RECT = max(1, int(5.0 / (scale * scale)))

        # 原始轨迹（非膨胀版，用于框内轨迹检查）
        gray_local = cv2.cvtColor(room_img, cv2.COLOR_BGR2GRAY)
        traj_raw_local = (gray_local > 200).astype(np.uint8)

        candidates = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue

            comp_mask = (labels == i).astype(np.uint8)
            x0, y0 = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
            cw, ch = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
            crop = comp_mask[y0:y0 + ch, x0:x0 + cw].copy()

            # 最大内接矩形分解（同 hatch 算法）
            remaining = crop.copy()
            while np.sum(remaining) > MIN_RECT:
                ch_r, cw_r = remaining.shape
                heights = np.zeros((ch_r, cw_r), dtype=np.int32)
                for col in range(cw_r):
                    run = 0
                    for row in range(ch_r):
                        if remaining[row, col] > 0:
                            run += 1
                        else:
                            run = 0
                        heights[row, col] = run

                best_area = 0
                best_rect = None
                for row in range(ch_r):
                    h_row = heights[row]
                    stack = []
                    for col in range(cw_r + 1):
                        cur_h = h_row[col] if col < cw_r else 0
                        start = col
                        while stack and stack[-1][0] > cur_h:
                            prev_h, prev_col = stack.pop()
                            width = col - prev_col
                            area_c = prev_h * width
                            if area_c > best_area:
                                best_area = area_c
                                best_rect = (row - prev_h + 1, prev_col, width, prev_h)
                            start = prev_col
                        if col < cw_r:
                            stack.append((cur_h, start))

                if best_rect is None or best_area < MIN_RECT:
                    break

                ry, rx, rw, rh = best_rect
                gx, gy = x0 + rx, y0 + ry

                # 死角过滤
                aspect = max(rw, rh) / max(min(rw, rh), 1)
                if aspect > 5.0 and min(rw, rh) < 12:
                    remaining[ry:ry + rh, rx:rx + rw] = 0
                    continue
                if rw * rh > room_area * 0.5:
                    remaining[ry:ry + rh, rx:rx + rw] = 0
                    continue

                # 框内轨迹检查
                bx1, by1 = max(0, gx), max(0, gy)
                bx2, by2 = min(w, gx + rw), min(h, gy + rh)
                if bx2 > bx1 and by2 > by1:
                    traj_inside = int(np.sum(traj_raw_local[by1:by2, bx1:bx2] > 0))
                    if traj_inside > 0:
                        remaining[ry:ry + rh, rx:rx + rw] = 0
                        continue
                else:
                    remaining[ry:ry + rh, rx:rx + rw] = 0
                    continue

                enclosure = self._compute_enclosure(comp_mask, traj_closed, ring=12)
                fill_ratio = best_area / max(rw * rh, 1)
                loop_score = enclosure * 0.6 + fill_ratio * 0.4

                candidates.append({
                    'bbox': {'x': gx, 'y': gy, 'width': rw, 'height': rh},
                    'area': best_area,
                    'centroid': (gx + rw / 2, gy + rh / 2),
                    'candidate_type': 'trajectory_surrounded',
                    'loop_score': round(loop_score, 3),
                    'confidence': round(0.6 * loop_score, 2),
                    'enclosure_score': round(enclosure, 3),
                })
                remaining[ry:ry + rh, rx:rx + rw] = 0

        # 包含抑制：外层框包含内层框 → 删除外层框，保留内层
        if len(candidates) >= 2:
            to_remove = set()
            for i in range(len(candidates)):
                if i in to_remove:
                    continue
                bi = candidates[i]['bbox']
                ai_x1, ai_y1 = bi['x'], bi['y']
                ai_x2, ai_y2 = ai_x1 + bi['width'], ai_y1 + bi['height']
                ai_area = bi['width'] * bi['height']
                for j in range(len(candidates)):
                    if i == j or j in to_remove:
                        continue
                    bj = candidates[j]['bbox']
                    aj_x1, aj_y1 = bj['x'], bj['y']
                    aj_x2, aj_y2 = aj_x1 + bj['width'], aj_y1 + bj['height']
                    # i 完全包含 j
                    if (ai_x1 <= aj_x1 and ai_y1 <= aj_y1 and
                        ai_x2 >= aj_x2 and ai_y2 >= aj_y2 and
                        ai_area > 1.5 * (bj['width'] * bj['height'])):
                        to_remove.add(i)
                        break

        # 世界坐标过滤：单边 < 3dm → 删除
        scale = getattr(self, '_scale', 0.08)
        candidates = [c for c in candidates
                      if min(c['bbox']['width'], c['bbox']['height']) * scale >= 2.0]

        # 相邻同尺寸合并：相邻的 2~4 个大小相近矩形 → 合并为一个外接框
        candidates = self._merge_adjacent_similar_boxes(
            candidates, room_img=room_img, name=name, mask_dir=mask_dir)

        # 世界坐标过滤：单边 < 3dm 或 总面积 < 20dm² → 删除
        scale = getattr(self, '_scale', 0.08)
        def _keep_world(c):
            w = c['bbox']['width']
            h = c['bbox']['height']
            min_side_dm = min(w, h) * scale
            area_dm2 = w * h * scale * scale
            return min_side_dm >= 3.0 and area_dm2 >= 20.0
        before = len(candidates)
        candidates = [c for c in candidates if _keep_world(c)]
        removed = before - len(candidates)
        if removed > 0:
            logger.info(f"  世界坐标过滤: 删除 {removed} 个 (保留 {len(candidates)} 个)")

        logger.info(f"  {name}: 轨迹闭环 {len(candidates)} 个")
        return candidates

    def _merge_adjacent_similar_boxes(self, candidates: List[Dict],
                                       gap_length: float = 3.8,
                                       size_ratio: float = 0.6,
                                       room_img: np.ndarray = None,
                                       name: str = "",
                                       mask_dir: str = None) -> List[Dict]:
        """将相邻且大小相近的 2~4 个矩形框合并为一个外接矩形框。

        判定条件（两框视为可合并）：
        1. 大小相近：较小框的宽、高分别 ≥ 较大框对应边的 size_ratio 倍
        2. 相邻：两框在 x、y 方向的间距均 ≤ gap_length 像素
        用并查集把互相可合并的框聚成组，每组 2~4 个则合并为外接框。

        Args:
            candidates: 候选框列表（含 'bbox'）
            gap_length: 相邻间距阈值（世界坐标 dm），内部通过 scale 转为像素
            size_ratio: 大小相近阈值系数
            room_img: 房间图像（用于可视化）
            name: 房间名称
            mask_dir: 输出目录
        Returns:
            合并后的候选框列表
        """
        n = len(candidates)
        scale = getattr(self, '_scale', 0.08)
        thr = gap_length / scale
        # ---- 合并前可视化：在 room_img 上画出所有候选框及标号 ----
        # if mask_dir and room_img is not None:
        #     viz = room_img.copy()
        #     for idx, c in enumerate(candidates):
        #         b = c['bbox']
        #         cv2.rectangle(viz, (b['x'], b['y']),
        #                       (b['x'] + b['width'], b['y'] + b['height']), (0, 255, 255), 1)
        #         cx, cy = b['x'] + b['width'] // 2, b['y'] + b['height'] // 2
        #         cv2.putText(viz, str(idx), (cx - 10, cy + 4),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        #     cv2.imwrite(os.path.join(mask_dir, f'{name}_traj_rects_before_merge.png'), viz)
        #     logger.info(f"  [merge_debug] 合并前可视化已保存: {name}_traj_rects_before_merge.png")
        # ------------------------------------------------------------

        if n < 2:
            return candidates

        def similar(b1, b2):
            w1, h1 = b1['width'], b1['height']
            w2, h2 = b2['width'], b2['height']
            wr = min(w1, w2) / max(w1, w2, 1)
            hr = min(h1, h2) / max(h1, h2, 1)
            return wr >= size_ratio and hr >= size_ratio

        def adjacent(b1, b2):
            # 各方向间距（负数=重叠）
            gap_x = max(b2['x'] - (b1['x'] + b1['width']),
                        b1['x'] - (b2['x'] + b2['width']))
            gap_y = max(b2['y'] - (b1['y'] + b1['height']),
                        b1['y'] - (b2['y'] + b2['height']))
            return gap_x <= thr and gap_y <= thr

        # 并查集
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # ---- debug: 打印每对框的 similar / adjacent 结果 ----
        # logger.info(f"  [merge_debug] 共 {n} 个候选框")
        for i in range(n):
            for j in range(i + 1, n):
                bi, bj = candidates[i]['bbox'], candidates[j]['bbox']
                sim = similar(bi, bj)
                adj = adjacent(bi, bj)
                w1, h1 = bi['width'], bi['height']
                w2, h2 = bj['width'], bj['height']
                wr = min(w1, w2) / max(w1, w2, 1)
                hr = min(h1, h2) / max(h1, h2, 1)
                gap_x = max(bj['x'] - (bi['x'] + bi['width']),
                            bi['x'] - (bj['x'] + bj['width']))
                gap_y = max(bj['y'] - (bi['y'] + bi['height']),
                            bi['y'] - (bj['y'] + bj['height']))
                # logger.info(f"  [merge_debug] ({i},{j}) "
                #             f"size=({w1}x{h1}) vs ({w2}x{h2}) "
                #             f"wr={wr:.3f} hr={hr:.3f} similar={sim} | "
                #             f"gap_x={gap_x:.1f} gap_y={gap_y:.1f} "
                #             f"thr={thr:.1f} adjacent={adj}")
                if sim and adj:
                    union(i, j)
        # --------------------------------------------------------

        # 分组
        groups: Dict[int, List[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        # ---- debug: 打印最终分组 ----
        # for members in groups.values():
        #     logger.info(f"  [merge_debug] 组: {members} (共{len(members)}人)")
        # --------------------------

        result = []
        merged_count = 0
        for members in groups.values():
            # 仅合并 2~4 个成组的框，其余原样保留
            if 2 <= len(members) <= 4:
                xs1 = [candidates[m]['bbox']['x'] for m in members]
                ys1 = [candidates[m]['bbox']['y'] for m in members]
                xs2 = [candidates[m]['bbox']['x'] + candidates[m]['bbox']['width'] for m in members]
                ys2 = [candidates[m]['bbox']['y'] + candidates[m]['bbox']['height'] for m in members]
                nx1, ny1, nx2, ny2 = min(xs1), min(ys1), max(xs2), max(ys2)
                base = candidates[members[0]].copy()
                base['bbox'] = {'x': nx1, 'y': ny1,
                                'width': nx2 - nx1, 'height': ny2 - ny1}
                base['area'] = (nx2 - nx1) * (ny2 - ny1)
                base['centroid'] = ((nx1 + nx2) / 2, (ny1 + ny2) / 2)
                # 置信度取组内最大
                base['confidence'] = max(candidates[m].get('confidence', 0.5) for m in members)
                result.append(base)
                merged_count += 1
            else:
                for m in members:
                    result.append(candidates[m])

        # if merged_count > 0:
        #     logger.info(f"  相邻同尺寸合并: {n} → {len(result)} 个（合并 {merged_count} 组）")
        return result

    def _compute_enclosure(self, region_mask: np.ndarray, traj_mask: np.ndarray,
                            ring: int = 15) -> float:
        """计算区域被轨迹包围的程度 (0~1)"""
        h, w = region_mask.shape[:2]
        ys, xs = np.where(region_mask > 0)
        if len(xs) == 0:
            return 0.0
        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max() + 1, ys.max() + 1

        sides = 0
        sides_checked = 0
        # top
        if y1 - ring >= 0:
            sides_checked += 1
            strip = traj_mask[y1 - ring:y1, x1:x2]
            if float(np.sum(strip)) / max(strip.size, 1) > 0.05:
                sides += 1
        # bottom
        if y2 + ring <= h:
            sides_checked += 1
            strip = traj_mask[y2:y2 + ring, x1:x2]
            if float(np.sum(strip)) / max(strip.size, 1) > 0.05:
                sides += 1
        # left
        if x1 - ring >= 0:
            sides_checked += 1
            strip = traj_mask[y1:y2, x1 - ring:x1]
            if float(np.sum(strip)) / max(strip.size, 1) > 0.05:
                sides += 1
        # right
        if x2 + ring <= w:
            sides_checked += 1
            strip = traj_mask[y1:y2, x2:x2 + ring]
            if float(np.sum(strip)) / max(strip.size, 1) > 0.05:
                sides += 1

        return sides / max(sides_checked, 1)

    # ==============================================================
    # Post-processing — NMS, Containment, Merge
    # ==============================================================
    def _iou(self, box1, box2):
        """计算 IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = a1 + a2 - inter
        return inter / max(union, 1)

    def _annotate_structure_features(self, candidates: List[Dict],
                                     traj_mask: np.ndarray,
                                     scale: float) -> None:
        """为候选原地附加只读结构特征（供 LLM 判断，不参与任何过滤/去重/合并）。

        新增字段：
          - rectangularity: 内接矩形填充率 (best_area/bbox_area)，越接近1越像实心矩形
          - arc_score:      bbox 四周被轨迹环绕的比例 (0~1)，越高越像被弧形轨迹包围
          - loop_cluster_count: bbox 内轨迹形成的闭环数量估计（≈圆环/椅腿环数）
          - size_class:     'small' / 'medium' / 'large'（按世界坐标面积 dm²）

        traj_mask: 裁剪坐标系下的轨迹二值掩码（与候选 bbox 同坐标系）
        scale:     dm/px
        """
        if not candidates:
            return
        if traj_mask is None:
            th, tw = 0, 0
        else:
            th, tw = traj_mask.shape[:2]

        for c in candidates:
            b = c['bbox']
            bw, bh = int(b['width']), int(b['height'])
            bx1, by1 = int(b['x']), int(b['y'])
            bx2, by2 = bx1 + bw, by1 + bh

            # rectangularity: 用已知 area(内接矩形面积) / bbox 面积
            bbox_area = max(bw * bh, 1)
            rect = min(1.0, c.get('area', bbox_area) / bbox_area)
            c['rectangularity'] = round(rect, 3)

            # arc_score: 复用 enclosure_score（四周环绕程度），无则默认0
            c['arc_score'] = round(float(c.get('enclosure_score', 0.0)), 3)

            # loop_cluster_count: bbox 内轨迹的闭环数（findContours 内轮廓计数）
            loop_count = 0
            if traj_mask is not None and tw > 0 and th > 0:
                ex1, ey1 = max(0, bx1), max(0, by1)
                ex2, ey2 = min(tw, bx2), min(th, by2)
                if ex2 > ex1 and ey2 > ey1:
                    sub = traj_mask[ey1:ey2, ex1:ex2].astype(np.uint8)
                    if sub.max() > 0:
                        contours, hierarchy = cv2.findContours(
                            sub, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                        if hierarchy is not None:
                            # 统计有父轮廓的内部闭环（hrec[3] >= 0）
                            loop_count = sum(1 for hrec in hierarchy[0] if hrec[3] >= 0)
                            # 若无内轮廓，用外轮廓数近似（弧段数）
                            if loop_count == 0:
                                loop_count = len(contours)
            c['loop_cluster_count'] = int(loop_count)

            # size_class: 按世界坐标面积 dm²
            area_dm2 = bw * bh * scale * scale
            if area_dm2 < 30:
                c['size_class'] = 'small'
            elif area_dm2 < 120:
                c['size_class'] = 'medium'
            else:
                c['size_class'] = 'large'

    def _build_cv_candidates_info(self, candidates: List[Dict]) -> str:
        """构建结构化候选信息文本（供 LLM 输入）"""
        if not candidates:
            return "(No CV candidates detected in this room)"

        lines = [f"CV detected {len(candidates)} candidate(s):"]
        for i, c in enumerate(candidates):
            b = c['bbox']
            ctype = c.get('candidate_type', 'obstacle')
            conf = c.get('confidence', 0.5)
            traj = c.get('traj_ratio', 0)
            encl = c.get('enclosure_score', 0)
            wall = c.get('wall_contact_ratio', 0)
            grey = c.get('grey_density', 0)
            asp = c.get('aspect_ratio', 1)
            area = c.get('area', b['width'] * b['height'])
            topo = c.get('topology_score', 0)
            # 结构特征（供 merge/separation 判断）
            loop_cnt = c.get('loop_cluster_count', 0)
            arc = c.get('arc_score', 0)
            rect = c.get('rectangularity', 0)
            size_cls = c.get('size_class', 'unknown')

            lines.append(
                f"  Candidate {c.get('candidate_id', i)}: "
                f"type={ctype} "
                f"bbox=({b['x']},{b['y']},{b['width']},{b['height']}) "
                f"area={area} "
                f"confidence={conf} "
                f"traj_ratio={traj} "
                f"enclosure={encl} "
                f"wall_contact={wall} "
                f"grey_density={grey} "
                f"aspect_ratio={asp} "
                f"topology_score={topo} "
                f"loop_cluster_count={loop_cnt} "
                f"arc_score={arc} "
                f"rectangularity={rect} "
                f"size_class={size_cls}"
            )

        return "\n".join(lines)

    def _prepare_candidates_for_correction(
        self,
        candidates: List[Dict],
        traj_mask: np.ndarray,
        scale: float,
        room_name: str,
    ) -> List[Dict]:
        """Run deterministic CV cleanup shared by the full and no-LLM variants."""
        self._annotate_structure_features(candidates, traj_mask, scale)

        filtered_candidates = []
        for candidate in candidates:
            bbox = candidate['bbox']
            area_dm2 = bbox['width'] * bbox['height'] * scale * scale
            is_corner_noise = (
                candidate.get('candidate_type') == 'trajectory_surrounded'
                and area_dm2 < 30.0
                and candidate.get('loop_cluster_count', 0) == 0
                and candidate.get('arc_score', 0) < 0.35
            )
            if is_corner_noise:
                logger.info(
                    f"  {room_name}: filter corner-noise candidate "
                    f"ID={candidate.get('candidate_id')} (area={area_dm2:.1f}dm2)"
                )
            else:
                filtered_candidates.append(candidate)

        return self._auto_merge_strong_signal_pairs(filtered_candidates, room_name)

    # ==============================================================
    # 主流程
    # ==============================================================
    def process(self, image_path: str, yaml_path: str, trajectory_json_path: str = None,
                output_path: str = None, hatch_mask_path: str = None,
                smart_device_path: str = None,
                intermediate_yaml_dir: str = None,
                ablation: dict = None) -> Dict:
        """主入口

        Args:
            image_path: greyroom.png (0=墙/128=地板/255=轨迹)
            yaml_path: floorplan_real.yaml (房间分割 + metadata)
            trajectory_json_path: 人停留点 JSON (世界坐标)
            output_path: 默认 yaml 同目录 room_real.yaml
            hatch_mask_path: hatch_mask.png (斜线阴影提取的二值掩码)
            smart_device_path: smartDevice YAML (世界坐标，用于细化家具位置)
            intermediate_yaml_dir: 中间 YAML 输出目录，非 None 时保存分步结果
            ablation: 消融实验控制标志
                - skip_trajectory_loop: 跳过轨迹闭环家具检测
                - skip_llm_correction: 跳过 LLM action 修正，纯 CV 候选
                - use_free_form_llm: 使用无约束 LLM prompt，并允许其增删候选
                - use_direct_llm_gen: 跳过 CV 候选，LLM 直接生成家具位置
        """
        ablation = ablation or {}
        self._ablation = ablation
        # Enforce input-removal flags at the agent boundary as well as in
        # Pipeline.  This keeps direct agent calls and the experiment runner
        # on the same protocol, and prevents a disabled modality from being
        # reintroduced by a caller bypassing Pipeline.step2.
        if ablation.get('skip_trajectory'):
            trajectory_json_path = None
        if ablation.get('skip_hatch'):
            hatch_mask_path = None
        if ablation.get('skip_smart_device'):
            smart_device_path = None
        if ablation.get('use_direct_llm_gen'):
            # Direct generation must not even load the CV hatch source.
            hatch_mask_path = None
        if output_path is None:
            # 默认输出到输入 YAML 同目录下的 room_real.yaml
            yaml_dir = os.path.dirname(yaml_path) if os.path.dirname(yaml_path) else '.'
            output_path = os.path.join(yaml_dir, 'room_real.yaml')
        self._output_path = output_path

        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        # 读取 hatch_mask（斜线阴影 = 家具位置）
        hatch_mask = None
        if hatch_mask_path and os.path.exists(hatch_mask_path):
            hatch_mask = cv2.imread(hatch_mask_path, cv2.IMREAD_GRAYSCALE)
            if hatch_mask is not None:
                _, hatch_mask = cv2.threshold(hatch_mask, 127, 255, cv2.THRESH_BINARY)
                logger.info(f"hatch_mask loaded: {hatch_mask_path}, furniture pixels: {np.sum(hatch_mask > 0)}")
            else:
                logger.warning(f"Cannot read hatch_mask: {hatch_mask_path}")
        else:
            logger.info("No hatch_mask provided, skipping hatch-based furniture detection")

        # 读取 YAML（容错：兼容已有的 numpy 标签）
        with open(yaml_path, 'r', encoding='utf-8') as f:
            try:
                layout = yaml.safe_load(f)
            except yaml.constructor.ConstructorError:
                f.seek(0)
                layout = yaml.full_load(f)

        # 读取坐标转换参数
        self._load_coordinate_params(layout)

        # 加载并转换轨迹数据
        all_traj_points = []
        stay_heatmap = None
        if trajectory_json_path and os.path.exists(trajectory_json_path):
            all_traj_points = self._load_trajectory(trajectory_json_path, img_h=img.shape[0])

            # 诊断：统计落点是否在房间内（坐标转换是否准确）
            rooms = layout.get('house', {}).get('rooms', [])
            inside = 0
            for pt in all_traj_points:
                px, py = pt['pixel_x'], pt['pixel_y']
                for rm in rooms:
                    pos = rm.get('position', {})
                    rx, ry, rw, rh = pos.get('x',0), pos.get('y',0), pos.get('width',0), pos.get('height',0)
                    if rx <= px <= rx + rw and ry <= py <= ry + rh:
                        inside += 1
                        break
            outside = len(all_traj_points) - inside
            out_pct = outside / max(len(all_traj_points), 1) * 100
            logger.info(f"轨迹点: {inside}/{len(all_traj_points)} 在房间内, "
                        f"{outside} 在房间外 ({out_pct:.1f}%)")
            if out_pct > 30:
                logger.warning("超过30%轨迹点在房间外，SCALE参数可能偏差，建议 "
                               "run_seg.py --ref <真实长度m>")

            # 生成全图停留热度图
            stay_heatmap = self._generate_stay_heatmap(all_traj_points, img.shape, layout, bg_image=img)
            heatmap_dir = os.path.join(os.path.dirname(self._output_path), 'masks')
            os.makedirs(heatmap_dir, exist_ok=True)
            cv2.imwrite(os.path.join(heatmap_dir, 'stay_heatmap.png'), stay_heatmap)
        else:
            logger.info("无轨迹数据，仅使用图像分析")

        # 清空旧家具
        for r in layout.get('house', {}).get('rooms', []):
            r.pop('furniture', None)

        # 生成户型全图缩略图（带房间编号标注）
        context_thumb = self._create_context_thumbnail(img, layout)
        _, thumb_encoded = cv2.imencode('.jpg', context_thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
        thumb_b64 = base64.b64encode(thumb_encoded.tobytes()).decode('utf-8')

        # 处理每个房间
        rooms = layout.get('house', {}).get('rooms', [])

        # ── 加载智能设备，世界→像素，按房间分配 ──
        room_devices_map = {}
        if smart_device_path and os.path.exists(smart_device_path):
            try:
                with open(smart_device_path, 'r', encoding='utf-8') as f:
                    device_data = yaml.safe_load(f)
                devices = device_data.get('house', {}).get('furniture', [])
                scale = getattr(self, '_scale', 0.08)
                for dev in devices:
                    pos = dev.get('position', {})
                    if 'x' not in pos or 'y' not in pos:
                        continue
                    # 世界坐标 → 像素坐标
                    wx, wy = float(pos['x']), float(pos['y'])
                    px = wx / scale + self._bl_x
                    py = self._bl_y - wy / scale
                    dw = float(pos.get('width', 0)) / scale
                    dh = float(pos.get('height', 0)) / scale
                    # 分配到房间
                    for r in rooms:
                        rpos = r.get('position', {})
                        rx, ry = rpos.get('x', 0), rpos.get('y', 0)
                        rw, rh = rpos.get('width', 0), rpos.get('height', 0)
                        if rx <= px <= rx + rw and ry <= py <= ry + rh:
                            room_devices_map.setdefault(r['name'], []).append({
                                'name': dev.get('name', dev.get('id', '?')),
                                'px': px, 'py': py, 'pw': dw, 'ph': dh,
                            })
                            break
                total = sum(len(v) for v in room_devices_map.values())
                logger.info(f"智能设备: {len(devices)} 件, {total} 件分配到房间")
            except Exception as e:
                logger.warning(f"智能设备加载失败: {e}")

        all_furniture = []
        for room in rooms:
            room_devices = room_devices_map.get(room['name'], [])
            room_furs = self._process_room(room, img, layout, all_traj_points, 
                                           stay_heatmap, thumb_b64, hatch_mask,
                                           room_devices)
            all_furniture.extend(room_furs)

        # ── 中间保存: 家具检测完成（像素坐标，无设备名）──
        # if intermediate_yaml_dir:
        #     os.makedirs(intermediate_yaml_dir, exist_ok=True)
        #     # 先写入家具位置（不含设备名）
        #     self._write_furniture(layout, all_furniture, name_map={})
        #     intermediate_furniture_path = os.path.join(intermediate_yaml_dir, '02_furniture_pixel.yaml')
        #     self._save_intermediate(layout, intermediate_furniture_path, '家具检测(像素, 无设备名)')

        # 智能设备匹配：将设备匹配到家具框 + 标记设备名
        img_h, img_w = img.shape[:2]
        all_furniture, name_map = self._match_devices_to_furniture(
            all_furniture, room_devices_map, img_w, img_h)

        # 写入 YAML（含设备名）
        self._write_furniture(layout, all_furniture, name_map=name_map)

        # # ── 中间保存: 设备匹配完成（像素坐标，含设备名）──
        # if intermediate_yaml_dir:
        #     intermediate_matched_path = os.path.join(intermediate_yaml_dir, '03_device_matched_pixel.yaml')
        #     self._save_intermediate(layout, intermediate_matched_path, '设备匹配(像素, 含设备名)')

        # 生成全图可视化：所有房间 + 家具框 + 智能设备
        self._draw_all_furniture_overview(img, layout, all_furniture,
                                           room_devices_map=room_devices_map)

        self._save_yaml(layout, output_path)

        total = len(all_furniture)
        logger.info(f"Updated layout saved: {output_path}")
        print(f"家具检测完成: {total} 件, 已写入 {output_path}")
        return layout

    # ==============================================================
    # 户型全图缩略图（带房间编号）
    # ==============================================================
    def _create_context_thumbnail(self, img: np.ndarray, layout: Dict, max_size: int = 400) -> np.ndarray:
        """生成带房间编号的户型全图缩略图"""
        h, w = img.shape[:2]
        scale = min(max_size / w, max_size / h)
        thumb_w, thumb_h = int(w * scale), int(h * scale)
        thumb = cv2.resize(img, (thumb_w, thumb_h))

        rooms = layout.get('house', {}).get('rooms', [])
        for i, room in enumerate(rooms):
            pos = room.get('position', {})
            rx = int(pos.get('x', 0) * scale)
            ry = int(pos.get('y', 0) * scale)
            rw = int(pos.get('width', 0) * scale)
            rh = int(pos.get('height', 0) * scale)

            label = f"R{i+1}"
            font_scale = max(0.5, min(1.0, rw / 100))
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            cx, cy = rx + rw // 2, ry + rh // 2
            cv2.rectangle(thumb, (cx - tw // 2 - 4, cy - th // 2 - 4),
                          (cx + tw // 2 + 4, cy + th // 2 + 4), (0, 0, 0), -1)
            cv2.putText(thumb, label, (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

        return thumb

    def _draw_all_furniture_overview(self, img: np.ndarray, layout: Dict,
                                       furniture_list: List[Tuple],
                                       room_devices_map: Dict[str, List[Dict]] = None):
        """生成全图可视化：直接读取 layout 中家具结果，有名（智能设备）→蓝框，无名→红框"""
        if len(img.shape) == 2:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            vis = img.copy()

        rooms = layout.get('house', {}).get('rooms', [])
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0),
                  (255, 0, 255), (0, 255, 255), (128, 128, 0), (0, 128, 128)]

        # 画房间边界
        for i, room in enumerate(rooms):
            pos = room.get('position', {})
            rx = int(pos.get('x', 0))
            ry = int(pos.get('y', 0))
            rw = int(pos.get('width', 0))
            rh = int(pos.get('height', 0))
            color = colors[i % len(colors)]
            cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), color, 2)
            cv2.putText(vis, room.get('name', ''), (rx + 3, ry + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 画家具框：有名（智能设备）→ 蓝框 + 设备名，无名 → 红框
            for fur in room.get('furniture', []):
                fpos = fur.get('position', {})
                fx = int(fpos.get('x', 0))
                fy = int(fpos.get('y', 0))
                fw = int(fpos.get('width', 0))
                fh = int(fpos.get('height', 0))
                device_name = fur.get('name')

                if device_name:
                    # 智能设备匹配的框 → 蓝色加粗 + 设备名标签
                    cv2.rectangle(vis, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 3)
                    cv2.putText(vis, device_name, (fx + 3, fy + fh - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                else:
                    # 普通家具框 → 红色
                    cv2.rectangle(vis, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)

        mask_dir = os.path.join(os.path.dirname(self._output_path), 'masks')
        cv2.imwrite(os.path.join(mask_dir, 'all_furniture_overview.png'), vis)
        print(f"全图家具可视化: {os.path.join(mask_dir, 'all_furniture_overview.png')}")

    # ==============================================================
    # Hatch Mask Furniture Extraction — 最大内接矩形分解 + 删离散小块
    # ==============================================================
    def _extract_hatch_furniture_rects(self, hatch_crop: np.ndarray,
                                        wall_mask: np.ndarray = None,
                                        min_comp_area: int = 800,
                                        min_rect_area: int = 200) -> List[Dict]:
        """从 hatch mask 裁剪图中提取家具（最大内接矩形分解）。

        流程:
        1. 连通域标记 → 删除离散小块 (area < min_comp_area)
        2. 对每个连通域: 贪婪最大内接矩形分解（直方图+单调栈）
        3. 每个矩形 = 一个家具候选

        Args:
            hatch_crop: 当前房间的 hatch_mask 裁剪 (0=背景, 255=阴影/家具)
            wall_mask: 墙体掩码 (用于 touch_wall_ratio 过滤)
            min_comp_area: 最小连通域保留面积
            min_rect_area: 最小内接矩形面积

        Returns:
            [{'bbox': {x,y,width,height}, 'candidate_type': 'hatch_furniture', ...}, ...]
        """
        if hatch_crop is None or np.sum(hatch_crop > 0) < min_comp_area:
            return []

        h_c, w_c = hatch_crop.shape[:2]
        hatch_bin = (hatch_crop > 127).astype(np.uint8)

        # Step1: 连通域 → 删离散小块
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hatch_bin, connectivity=8)
        if num_labels < 2:
            return []

        rects = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_comp_area:
                continue

            comp_mask = (labels == i).astype(np.uint8)
            x0 = int(stats[i, cv2.CC_STAT_LEFT])
            y0 = int(stats[i, cv2.CC_STAT_TOP])
            cw = int(stats[i, cv2.CC_STAT_WIDTH])
            ch = int(stats[i, cv2.CC_STAT_HEIGHT])
            crop = comp_mask[y0:y0 + ch, x0:x0 + cw].copy()

            # Step2: 贪婪最大内接矩形分解
            remaining = crop.copy()
            while np.sum(remaining) > min_rect_area:
                ch_r, cw_r = remaining.shape
                heights = np.zeros((ch_r, cw_r), dtype=np.int32)
                for col in range(cw_r):
                    run = 0
                    for row in range(ch_r):
                        if remaining[row, col] > 0:
                            run += 1
                        else:
                            run = 0
                        heights[row, col] = run

                # 单调栈求全局最大矩形
                best_area = 0
                best_rect = None  # (row, col, w, h)
                for row in range(ch_r):
                    h_row = heights[row]
                    stack = []
                    for col in range(cw_r + 1):
                        cur_h = h_row[col] if col < cw_r else 0
                        start = col
                        while stack and stack[-1][0] > cur_h:
                            prev_h, prev_col = stack.pop()
                            width = col - prev_col
                            area_c = prev_h * width
                            if area_c > best_area:
                                best_area = area_c
                                best_rect = (row - prev_h + 1, prev_col, width, prev_h)
                            start = prev_col
                        if col < cw_r:
                            stack.append((cur_h, start))

                if best_rect is None or best_area < min_rect_area:
                    break

                ry, rx, rw, rh = best_rect
                # 全局坐标
                gx = x0 + rx
                gy = y0 + ry
                # 死角过滤
                aspect = max(rw, rh) / max(min(rw, rh), 1)
                if aspect > 6.0:
                    remaining[ry:ry + rh, rx:rx + rw] = 0
                    continue
                bbox_area = rw * rh
                if bbox_area > h_c * w_c * 0.85:
                    remaining[ry:ry + rh, rx:rx + rw] = 0
                    continue

                rects.append({
                    'bbox': {'x': gx, 'y': gy, 'width': rw, 'height': rh},
                    'area': best_area,
                    'centroid': (gx + rw / 2, gy + rh / 2),
                    'candidate_type': 'hatch_furniture',
                    'shadow_score': round(best_area / max(bbox_area, 1), 3),
                    'confidence': 0.90,
                })
                remaining[ry:ry + rh, rx:rx + rw] = 0

        # 世界坐标过滤：单边 < 3dm → 删除
        scale = getattr(self, '_scale', 0.08)
        rects = [r for r in rects
                 if min(r['bbox']['width'], r['bbox']['height']) * scale >= 3.0]

        rects.sort(key=lambda r: r['area'], reverse=True)
        return rects

    # ==============================================================
    # 置信度融合 + IoU 去重
    # ==============================================================
    def _fuse_and_dedup_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """置信度融合 + IoU 去重。

        置信度公式:
          confidence = 0.4 * shadow_score + 0.4 * loop_score

        然后按 IoU > 0.5 去重，保留置信度更高的。
        """
        if not candidates:
            return []

        # 1) 置信度融合
        for c in candidates:
            shadow_score = c.get('shadow_score', 0.0)
            loop_score = c.get('loop_score', 0.0)
            fused_conf = 0.4 * shadow_score + 0.4 * loop_score
            # 保留原始类型置信度作为下限
            original_conf = c.get('confidence', 0.5)
            c['confidence'] = round(max(fused_conf, original_conf * 0.7), 3)

        # 2) IoU > 0.5 去重（红框不受去重影响）
        n = len(candidates)
        to_remove = set()
        for i in range(n):
            if i in to_remove:
                continue
            bi = candidates[i]['bbox']
            box_i = [bi['x'], bi['y'], bi['x'] + bi['width'], bi['y'] + bi['height']]
            for j in range(i + 1, n):
                if j in to_remove:
                    continue
                bj = candidates[j]['bbox']
                box_j = [bj['x'], bj['y'], bj['x'] + bj['width'], bj['y'] + bj['height']]
                if self._iou(box_i, box_j) > 0.5:
                    # 红框不参与去重
                    is_red_i = candidates[i].get('candidate_type') == 'hatch_furniture'
                    is_red_j = candidates[j].get('candidate_type') == 'hatch_furniture'
                    if is_red_i and is_red_j:
                        continue  # 两个红框都不删（可能是相邻真实家具）
                    if is_red_i:
                        to_remove.add(j)  # 保留红框，删另一个
                    elif is_red_j:
                        to_remove.add(i)  # 保留红框，删另一个
                        break
                    # 保留置信度高的
                    elif candidates[i]['confidence'] >= candidates[j]['confidence']:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break  # i 被淘汰，跳出内层循环

        result = [c for idx, c in enumerate(candidates) if idx not in to_remove]
        if len(result) < len(candidates):
            logger.info(f"  IoU 去重: {len(candidates)} → {len(result)}")

        return result

    def _process_room(self, room: Dict, img: np.ndarray, layout: Dict,
                      all_traj: List[Dict], stay_heatmap: np.ndarray = None,
                      thumb_b64: str = None, hatch_mask: np.ndarray = None,
                      room_devices: List[Dict] = None) -> List[Tuple]:
        """处理一个房间，返回绝对像素坐标的家具框列表。"""
        name = room['name']
        pos = room.get('position', {})
        rx, ry, rw_room, rh_room = pos.get('x', 0), pos.get('y', 0), pos.get('width', 0), pos.get('height', 0)

        if rw_room < 2 or rh_room < 2:
            return []

        h_img, w_img = img.shape[:2]

        # 直接用 YAML 坐标裁剪（greyroom 无红墙，无需红墙检测收紧）
        px_left  = max(0, int(rx) + 2)
        px_right = min(w_img - 1, int(rx + rw_room) - 2)
        px_top   = max(0, int(ry) + 2)
        px_bot   = min(h_img - 1, int(ry + rh_room) - 2)
        if px_right <= px_left or px_bot <= px_top:
            return []

        room_img = img[px_top:px_bot, px_left:px_right].copy()

        # 画红色边框供 LLM 看清边界
        cv2.rectangle(room_img, (0, 0), (px_right - px_left - 1, px_bot - px_top - 1), (0, 0, 255), 4)

        crop_h, crop_w = room_img.shape[:2]
        mask_dir = os.path.join(os.path.dirname(self._output_path), 'masks')
        os.makedirs(mask_dir, exist_ok=True)

        # ========== Hatch Mask 家具提取 ==========
        gray = cv2.cvtColor(room_img, cv2.COLOR_BGR2GRAY)
        wall_mask = (gray < 5).astype(np.uint8)

        hatch_furniture = []
        hatch_crop = None
        if (not getattr(self, '_ablation', {}).get('use_direct_llm_gen')
                and hatch_mask is not None and hatch_mask.size > 0):
            h_hm, w_hm = hatch_mask.shape[:2]
            hx1 = max(0, px_left)
            hy1 = max(0, px_top)
            hx2 = min(w_hm, px_right)
            hy2 = min(h_hm, px_bot)
            if hx2 > hx1 and hy2 > hy1:
                hatch_crop = hatch_mask[hy1:hy2, hx1:hx2].copy()
                hatch_furniture = self._extract_hatch_furniture_rects(hatch_crop, wall_mask=wall_mask)
                logger.info(f"  {name}: hatch 阴影家具 {len(hatch_furniture)} 个")

        # ========== CV 预处理: 提取 traj_mask ==========
        traj_raw = (gray > 200).astype(np.uint8)

        # 红框线
        B = room_img[:, :, 0].astype(np.int32)
        G = room_img[:, :, 1].astype(np.int32)
        R = room_img[:, :, 2].astype(np.int32)
        red_line = ((R > 80) & (G < 100) & (B < 100)).astype(np.uint8)
        traj_mask_full = np.clip(traj_raw + red_line, 0, 1).astype(np.uint8)

        dilate_kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        traj_mask_dilated = cv2.dilate(traj_mask_full, dilate_kernel_small, iterations=1)

        # ========== Hatch 候选扩展到轨迹边界 ==========
        if hatch_furniture and np.sum(traj_mask_full) > 0:
            hatch_furniture = self._expand_hatch_to_trajectory(
                hatch_furniture, traj_mask_full, crop_w, crop_h)

        # ========== 新 Greyroom 处理：排除 hatch 区域，检测轨迹包围区 ==========
        hatch_exclude_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        if (not getattr(self, '_ablation', {}).get('use_direct_llm_gen')
                and hatch_crop is not None and np.sum(hatch_crop > 0) > 0):
            hatch_exclude_mask = (hatch_crop > 127).astype(np.uint8)
            hatch_exclude_dilated = cv2.dilate(hatch_exclude_mask,
                                                np.ones((3, 3), np.uint8), iterations=1)
        else:
            hatch_exclude_dilated = np.zeros((crop_h, crop_w), dtype=np.uint8)

        ablation = getattr(self, '_ablation', {})
        trajectory_surrounded_furniture = []
        if ablation.get('use_direct_llm_gen'):
            logger.info(
                f"  [ABLATION] {name}: skip all CV candidate sources "
                "for direct LLM generation"
            )
        elif not ablation.get('skip_trajectory_loop'):
            trajectory_surrounded_furniture = self._detect_trajectory_surrounded_regions(
                room_img, traj_mask_dilated, hatch_exclude_dilated, wall_mask, name, mask_dir)
        else:
            logger.info(f"  [ABLATION] {name}: 跳过轨迹闭环家具检测")

        # ========== 合并候选 ==========
        hatch_candidates = list(hatch_furniture)
        greyroom_candidates = list(trajectory_surrounded_furniture)

        # ========== 合并 + 去重 ==========
        # 重叠去重：蓝框(greyroom)与红框(hatch)存在重叠 → 删除蓝框
        filtered_greyroom = []
        for gc in greyroom_candidates:
            gb = gc['bbox']
            overlapped = False
            for hc in hatch_candidates:
                hb = hc['bbox']
                overlap_x = max(0, min(gb['x'] + gb['width'], hb['x'] + hb['width']) - max(gb['x'], hb['x']))
                overlap_y = max(0, min(gb['y'] + gb['height'], hb['y'] + hb['height']) - max(gb['y'], hb['y']))
                if overlap_x > 0 and overlap_y > 0:
                    overlapped = True
                    break
            if not overlapped:
                filtered_greyroom.append(gc)
        removed = len(greyroom_candidates) - len(filtered_greyroom)
        if removed > 0:
            logger.info(f"  蓝框去重: 删除 {removed} 个与红框重叠的蓝框")

        all_candidates = []
        all_candidates.extend(hatch_candidates)
        all_candidates.extend(filtered_greyroom)

        # 为每个 candidate 分配唯一 ID
        for idx, c in enumerate(all_candidates):
            c['candidate_id'] = idx + 1

        if all_candidates:
            all_candidates = self._fuse_and_dedup_candidates(all_candidates)

        for i, c in enumerate(all_candidates):
            c['candidate_id'] = i

        logger.info(f"  {name}: CV 最终候选 {len(all_candidates)} 个")

        # 编码图片为 base64
        _, img_encoded = cv2.imencode('.png', room_img)
        img_b64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

        room_heatmap_b64 = None
        if stay_heatmap is not None and stay_heatmap.size > 0:
            room_heatmap = self._crop_heatmap_for_room(
                stay_heatmap, px_left, px_top, px_right, px_bot)
            _, hm_encoded = cv2.imencode('.png', room_heatmap)
            room_heatmap_b64 = base64.b64encode(hm_encoded.tobytes()).decode('utf-8')

        # Deterministic CV cleanup is part of candidate preparation, not of
        # the remote LLM correction module.  Run it once for every variant so
        # Full and No-LLM are compared from the same candidate set.
        scale = getattr(self, '_scale', 0.08)
        if not ablation.get('use_direct_llm_gen'):
            all_candidates = self._prepare_candidates_for_correction(
                all_candidates, traj_mask_full, scale, name)

        if self.llm is None or ablation.get('skip_llm_correction'):
            if ablation.get('skip_llm_correction'):
                logger.info(
                    f"  [ABLATION] {name}: skip LLM geometry correction; "
                    "keep deterministic CV post-processing"
                )
            elif self.llm is None:
                logger.warning(
                    f"No LLM available for {name}, using deterministic CV candidates"
                )
            direct_furniture = []
            for c in all_candidates:
                b = c['bbox']
                direct_furniture.append((name,
                    px_left + b['x'], px_top + b['y'],
                    b['width'], b['height']))
            return direct_furniture

        # 构建轨迹信息
        traj_margin = max(2, int(5 / max(self._scale, 0.1)))
        room_traj = self._filter_trajectory_for_room(
            all_traj, rx, ry, rw_room, rh_room,
            crop_left=px_left, crop_top=px_top,
            crop_right=px_right, crop_bot=px_bot,
            margin=traj_margin)
        trajectory_info = self._build_trajectory_info(room_traj)
        cv_candidates_info = self._build_cv_candidates_info(all_candidates)

        room_info = {
            "room_name": name,
            "position": {"x": rx, "y": ry, "width": rw_room, "height": rh_room},
            "crop_offset": {"left": px_left, "top": px_top},
            "crop_size_pixels": {"width": crop_w, "height": crop_h},
            "walls": room.get('walls', {}),
            "house_size": layout.get('house', {}).get('size', {}),
        }
        room_yaml = yaml.dump(room_info, default_flow_style=None, allow_unicode=True, sort_keys=False)

        # 消融与调用分支
        if ablation.get('use_direct_llm_gen'):
            direct_prompt = f"""You are a furniture detection system. Look at this room image and detect ALL furniture objects.
            
Room context (pixel coordinates):
```yaml
{room_yaml}
```

            {trajectory_info}

Output a JSON array of furniture objects. Each object has:
- "action": "keep"
- "bbox": {{"x": int, "y": int, "width": int, "height": int}}  (pixel coordinates relative to the room crop image)

Rules:
- Detect all visible furniture in the room
- Output bounding boxes that tightly enclose each furniture item
- Do not detect walls or empty spaces as furniture
- Return ONLY the JSON array, no other text

Example:
[{{"action": "keep", "bbox": {{"x": 50, "y": 100, "width": 80, "height": 60}}}}]
"""
            try:
                msg_content_direct = [
                    {"type": "text", "text": direct_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{img_b64}",
                        "detail": "high"
                    }},
                ]
                if room_heatmap_b64:
                    msg_content_direct.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{room_heatmap_b64}",
                        "detail": "high"
                    }})
                msg_direct = HumanMessage(content=msg_content_direct)
                result_direct = self.llm.invoke([msg_direct])
                content_direct = result_direct.content if hasattr(result_direct, 'content') else str(result_direct)
                logger.info(f"  {name}: LLM 直接生成输出:\n{content_direct}")
                # 尝试解析 JSON
                import re as _re
                json_match = _re.search(r'\[.*?\]', content_direct, _re.DOTALL)
                if json_match:
                    direct_actions = json.loads(json_match.group(0))
                    furniture = self._parse_direct_furniture_generation(
                        json.dumps(direct_actions), name,
                        px_left, px_top, crop_w, crop_h)
                    if furniture:
                        logger.info(f"  [ABLATION] {name}: 直接 LLM 生成 {len(furniture)} 件家具")
                        return furniture
            except Exception as e:
                logger.warning(f"  [ABLATION] {name}: 直接 LLM 生成失败: {e}")
            # Do not fall back to CV here: that would contaminate the direct
            # generation baseline with the very source it is meant to ablate.
            logger.warning(
                f"  [ABLATION] {name}: 直接生成失败，返回空家具结果（不回退 CV）"
            )
            return []

        # ── 消融: Free-form LLM (无约束 prompt) ──
        if ablation.get('use_free_form_llm'):
            logger.info(f"  [ABLATION] {name}: 使用无约束 Free-form LLM prompt")
            freeform_prompt = f"""You are a furniture detection and correction system.
            
Below are CV-detected furniture candidates with their structural features.
You may delete, merge, adjust, or add furniture as you see fit.

{trajectory_info}

CV Candidates:
{cv_candidates_info}

Room context (pixel coordinates):
```yaml
{room_yaml}
```

Output a JSON array of actions. Each action has:
- "action": "keep" | "delete" | "merge" | "adjust" | "add"
- For "keep": {{"action": "keep", "candidate_ids": [int]}}
- For "delete": {{"action": "delete", "candidate_ids": [int]}}
- For "merge": {{"action": "merge", "candidate_ids": [int, int], "bbox": {{"x": int, "y": int, "width": int, "height": int}}}}
- For "adjust": {{"action": "adjust", "candidate_id": int, "bbox": {{"x": int, "y": int, "width": int, "height": int}}}}
- For "add": {{"action": "add", "bbox": {{"x": int, "y": int, "width": int, "height": int}}}}

All coordinates are relative to the room crop image.
Return ONLY the JSON array, no other text.
"""
            try:
                msg_content_free = [
                    {"type": "text", "text": freeform_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{img_b64}",
                        "detail": "high"
                    }},
                ]
                msg_content_free.insert(1, {"type": "text", "text": f"## Room Context (pixel coordinates)\n```yaml\n{room_yaml}\n```"})
                if room_heatmap_b64:
                    msg_content_free.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{room_heatmap_b64}",
                        "detail": "high"
                    }})
                if thumb_b64:
                    msg_content_free.append({"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{thumb_b64}",
                        "detail": "low"
                    }})
                msg_free = HumanMessage(content=msg_content_free)
                result_free = self.llm.invoke([msg_free])
                content = result_free.content if hasattr(result_free, 'content') else str(result_free)
            except Exception as e:
                logger.warning(f"LLM call failed for {name}: {e}")
                fallback = []
                for c in all_candidates:
                    b = c['bbox']
                    fallback.append((name,
                        px_left + b['x'], px_top + b['y'],
                        b['width'], b['height']))
                return fallback
        else:
            # 调用 LLM — 使用新的 action-based 协议
            text_prompt = (FURNITURE_DETECTION_AGENT_PROMPT
                           .replace('{trajectory_info}', trajectory_info)
                           .replace('{cv_candidates_info}', cv_candidates_info))

            msg_content = [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                    "detail": "high"
                }},
            ]
            msg_content.insert(1, {"type": "text", "text": f"## Room Context (pixel coordinates)\n```yaml\n{room_yaml}\n```"})
            if room_heatmap_b64:
                msg_content.append({"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{room_heatmap_b64}",
                    "detail": "high"
                }})
            if thumb_b64:
                msg_content.append({"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{thumb_b64}",
                    "detail": "low"
                }})

            try:
                msg = HumanMessage(content=msg_content)
                result = self.llm.invoke([msg])
                content = result.content if hasattr(result, 'content') else str(result)
            except Exception as e:
                logger.warning(f"LLM call failed for {name}: {e}")
                # LLM 调用失败 → 使用全部 CV 候选
                fallback = []
                for c in all_candidates:
                    b = c['bbox']
                    fallback.append((name,
                        px_left + b['x'], px_top + b['y'],
                        b['width'], b['height']))
                return fallback

        # 解析 action-based LLM 输出
        logger.info(f"  {name}: LLM 原始输出:\n{content}")
        furniture = self._parse_room_furniture_actions(
            content, all_candidates, name,
            rx, ry, px_left, px_top, crop_w, crop_h,
            allow_unrestricted=bool(ablation.get('use_free_form_llm')))

        if furniture:
            print(f"  {name}: {len(furniture)} 件家具")
            for _, fx, fy, fw, fh in furniture:
                print(f"    ({fx:.1f}, {fy:.1f}) {fw:.1f}x{fh:.1f}")
        else:
            logger.warning(f"  {name}: LLM 处理后无家具输出 "
                           f"(CV 候选 {len(all_candidates)} 个)")

        return furniture

    def _parse_direct_furniture_generation(
        self,
        text: str,
        room_name: str,
        crop_left: int,
        crop_top: int,
        crop_w: int,
        crop_h: int,
    ) -> List[Tuple]:
        """Parse the direct-generation baseline's bbox-only response.

        The action parser below intentionally requires CV candidate IDs and
        therefore cannot parse direct-generation output. Keeping a separate
        parser makes this baseline independent of the CV candidate list
        instead of silently falling back to CV on every valid response.
        """
        actions = self._extract_json_array(text)
        if not actions:
            return []

        furniture = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            action_type = str(action.get("action", "keep")).lower()
            if action_type not in {"", "keep", "add"}:
                continue
            bbox = action.get("bbox") or action.get("position")
            if not isinstance(bbox, dict):
                continue
            try:
                x = float(bbox.get("x", 0))
                y = float(bbox.get("y", 0))
                width = float(bbox.get("width", 0))
                height = float(bbox.get("height", 0))
            except (TypeError, ValueError):
                continue
            if crop_w:
                x = max(0.0, min(x, float(crop_w)))
            else:
                x = max(0.0, x)
            if crop_h:
                y = max(0.0, min(y, float(crop_h)))
            else:
                y = max(0.0, y)
            if crop_w:
                width = min(width, float(crop_w) - x)
            if crop_h:
                height = min(height, float(crop_h) - y)
            if width < 5.0 or height < 5.0:
                continue
            furniture.append((
                room_name, crop_left + x, crop_top + y, width, height
            ))
        return furniture

    # ==============================================================
    # 解析 LLM action-based 输出 → 家具列表
    # ==============================================================
    def _auto_merge_strong_signal_pairs(self, candidates: List[Dict],
                                         room_name: str) -> List[Dict]:
        """自动合并满足 STRONG MERGE SIGNAL 的蓝框对，保证确定性。

        当房间内只有蓝框且满足以下全部条件时，预合并后再交给 LLM：
        1. 所有候选均为 blue (trajectory_surrounded)
        2. 恰好 2 个候选
        3. 相同 size_class
        4. arc_score 均 >= 0.4
        5. loop_cluster_count 均 == 0 (无独立闭环)
        6. 面积比 <= 2.5

        Returns:
            合并后的 candidates 列表（原地修改）
        """
        if len(candidates) != 2:
            return candidates

        # 检查是否全部为蓝框
        blue_candidates = [c for c in candidates
                           if c.get('candidate_type') == 'trajectory_surrounded']
        if len(blue_candidates) != 2:
            return candidates

        # 结构特征检查
        c1, c2 = blue_candidates
        sc1 = c1.get('size_class', '')
        sc2 = c2.get('size_class', '')
        if sc1 != sc2:
            return candidates

        arc1 = c1.get('arc_score', 0)
        arc2 = c2.get('arc_score', 0)
        if arc1 < 0.4 or arc2 < 0.4:
            return candidates

        loop1 = c1.get('loop_cluster_count', 0)
        loop2 = c2.get('loop_cluster_count', 0)
        if loop1 > 0 or loop2 > 0:
            return candidates

        # 面积比检查
        b1, b2 = c1['bbox'], c2['bbox']
        area1 = b1['width'] * b1['height']
        area2 = b2['width'] * b2['height']
        if area1 < 1 or area2 < 1:
            return candidates
        ratio = max(area1, area2) / min(area1, area2)
        if ratio > 2.5:
            return candidates

        # 全部条件满足 → 自动合并
        ux = min(b1['x'], b2['x'])
        uy = min(b1['y'], b2['y'])
        uw = max(b1['x'] + b1['width'], b2['x'] + b2['width']) - ux
        uh = max(b1['y'] + b1['height'], b2['y'] + b2['height']) - uy

        merged = {
            'candidate_id': 1,
            'candidate_type': 'trajectory_surrounded',
            'bbox': {'x': ux, 'y': uy, 'width': uw, 'height': uh},
            'area': uw * uh,
            'enclosure_score': max(arc1, arc2),
            'rectangularity': max(c1.get('rectangularity', 0),
                                  c2.get('rectangularity', 0)),
            'size_class': sc1,
            'loop_cluster_count': 0,
            'arc_score': max(arc1, arc2),
            'confidence': max(c1.get('confidence', 0.5),
                             c2.get('confidence', 0.5)),
            '_auto_merged': True,
        }
        logger.info(
            f"  {room_name}: 自动合并 2 个蓝框 (size={sc1}, arc={arc1:.2f}/{arc2:.2f}, "
            f"ratio={ratio:.1f}) → 1 个合并框 ({uw}x{uh})"
        )
        return [merged]

    def _parse_room_furniture_actions(self, text: str, candidates: List[Dict],
                                       room_name: str,
                                       rx: float, ry: float,
                                       crop_left: int, crop_top: int,
                                       crop_w: int = 0, crop_h: int = 0,
                                       allow_unrestricted: bool = False) -> List[Tuple]:
        """从 action-based LLM 回复中提取家具 bbox。

        受限协议下，CV 检测的所有框默认视为家具，LLM 只做
        merge/delete/adjust 修正；Free-form 消融则允许 add、删除红框，
        并可省略未提及的候选。

        Returns:
            [(room_name, px, py, pw, ph), ...]  绝对像素坐标
        """
        # 建立 candidate_id → candidate 的映射
        candidate_map = {c.get('candidate_id', i): c for i, c in enumerate(candidates)}

        actions = self._extract_json_array(text)
        # Free-form is deliberately allowed to delete any candidate and to
        # omit untouched candidates.  The constrained protocol keeps red
        # hatch boxes protected and retains candidates not explicitly acted
        # on; these are different experimental boundaries.
        allow_unrestricted = bool(
            allow_unrestricted
            or getattr(self, '_ablation', {}).get('use_free_form_llm')
        )
        # A free-form model may add furniture even when CV produced no
        # candidates.  Constrained mode still requires the CV candidate set.
        if not candidates and not (allow_unrestricted and actions):
            return []
        if not actions:
            # LLM 无有效输出 → 全部 CV 候选直接使用
            logger.info(f"  {room_name}: LLM 无有效输出，使用全部 {len(candidates)} 个 CV 候选")
            result = []
            for c in candidates:
                b = c['bbox']
                result.append((
                    room_name,
                    crop_left + b['x'], crop_top + b['y'],
                    b['width'], b['height'],
                ))
            return result

        # 内部用 dict 记录每个输出框，含红/蓝类型，便于门碰撞后处理
        # {'x','y','w','h','red'}  坐标为裁剪相对坐标
        furniture_boxes = []
        used_ids = set()       # 已被 LLM 处理过的 candidate（合并/删除/调整）

        def _is_red(cand: Dict) -> bool:
            return cand.get('candidate_type', '') == 'hatch_furniture'

        for action in actions:
            act_type = action.get('action', '')

            if act_type == 'merge':
                cids = action.get('candidate_ids', [])
                if len(cids) < 2:
                    continue
                if any(cid in used_ids for cid in cids):
                    continue
                valid = [candidate_map[cid] for cid in cids if cid in candidate_map]
                if len(valid) < 2:
                    continue
                is_red = any(_is_red(c) for c in valid)
                merged = action.get('merged_bbox') or action.get('bbox')
                if merged:
                    mx, my = merged.get('x', 0), merged.get('y', 0)
                    mw, mh = merged.get('width', 0), merged.get('height', 0)
                    if mw >= 5 and mh >= 5:
                        furniture_boxes.append({'x': mx, 'y': my, 'w': mw, 'h': mh, 'red': is_red})
                        for cid in cids:
                            used_ids.add(cid)
                        continue
                # 回退：union
                xs = [c['bbox']['x'] for c in valid]
                ys = [c['bbox']['y'] for c in valid]
                xe = [c['bbox']['x'] + c['bbox']['width'] for c in valid]
                ye = [c['bbox']['y'] + c['bbox']['height'] for c in valid]
                ux1, uy1 = min(xs), min(ys)
                ux2, uy2 = max(xe), max(ye)
                furniture_boxes.append({'x': ux1, 'y': uy1, 'w': ux2 - ux1, 'h': uy2 - uy1, 'red': is_red})
                for cid in cids:
                    used_ids.add(cid)

            elif act_type == 'delete':
                raw_ids = action.get('candidate_ids')
                if raw_ids is None:
                    raw_ids = [action.get('candidate_id')]
                if not isinstance(raw_ids, (list, tuple, set)):
                    raw_ids = [raw_ids]
                for cid in raw_ids:
                    if cid is None or cid in used_ids:
                        continue
                    c = candidate_map.get(cid)
                    if c is None:
                        continue
                    # Red hatch boxes are protected only by the constrained
                    # protocol; free-form is intentionally unrestricted.
                    if _is_red(c) and not allow_unrestricted:
                        logger.warning(
                            f"  {room_name}: LLM 尝试删除红框 candidate {cid}, 已拦截"
                            f" — {action.get('reason', '')}"
                        )
                        continue
                    used_ids.add(cid)
                    logger.info(
                        f"  {room_name}: LLM 删除 candidate {cid} — "
                        f"{action.get('reason', '')}"
                    )

            elif act_type == 'keep':
                # Free-form prompts may explicitly keep a list of candidates.
                raw_ids = action.get('candidate_ids')
                if raw_ids is None:
                    raw_ids = [action.get('candidate_id')]
                if not isinstance(raw_ids, (list, tuple, set)):
                    raw_ids = [raw_ids]
                for cid in raw_ids:
                    c = candidate_map.get(cid)
                    if c is None or cid in used_ids:
                        continue
                    b = c['bbox']
                    furniture_boxes.append({
                        'x': b['x'], 'y': b['y'], 'w': b['width'],
                        'h': b['height'], 'red': _is_red(c),
                    })
                    used_ids.add(cid)

            elif act_type == 'add' and allow_unrestricted:
                bbox = action.get('bbox') or action.get('position')
                if not isinstance(bbox, dict):
                    continue
                try:
                    ax = float(bbox.get('x', 0))
                    ay = float(bbox.get('y', 0))
                    aw = float(bbox.get('width', 0))
                    ah = float(bbox.get('height', 0))
                except (TypeError, ValueError):
                    continue
                ax = max(0.0, ax)
                ay = max(0.0, ay)
                if crop_w:
                    ax = min(ax, float(crop_w))
                    aw = min(aw, float(crop_w) - ax)
                if crop_h:
                    ay = min(ay, float(crop_h))
                    ah = min(ah, float(crop_h) - ay)
                if aw >= 5.0 and ah >= 5.0:
                    furniture_boxes.append({
                        'x': ax, 'y': ay, 'w': aw, 'h': ah, 'red': False,
                    })

            elif act_type == 'adjust':
                cid = action.get('candidate_id')
                if cid is None or cid in used_ids:
                    continue
                c = candidate_map.get(cid)
                if c is None:
                    continue
                b = c['bbox']
                if allow_unrestricted and isinstance(action.get('bbox'), dict):
                    adjusted = action['bbox']
                    try:
                        new_x = float(adjusted.get('x', b['x']))
                        new_y = float(adjusted.get('y', b['y']))
                        new_w = float(adjusted.get('width', b['width']))
                        new_h = float(adjusted.get('height', b['height']))
                    except (TypeError, ValueError):
                        continue
                    new_x = max(0.0, new_x)
                    new_y = max(0.0, new_y)
                    if crop_w:
                        new_x = min(new_x, float(crop_w))
                        new_w = min(new_w, float(crop_w) - new_x)
                    if crop_h:
                        new_y = min(new_y, float(crop_h))
                        new_h = min(new_h, float(crop_h) - new_y)
                    if new_w >= 5.0 and new_h >= 5.0:
                        used_ids.add(cid)
                        furniture_boxes.append({
                            'x': new_x, 'y': new_y, 'w': new_w, 'h': new_h,
                            'red': _is_red(c),
                        })
                    continue
                expand = action.get('expand', {})
                # 支持正负值：正值扩展，负值收缩
                el = int(expand.get('left', 0))
                er = int(expand.get('right', 0))
                et = int(expand.get('top', 0))
                eb = int(expand.get('bottom', 0))
                # 限制扩展/收缩幅度 ≤ 该边长度的 10%
                max_expand_w = int(b['width'] * 0.1)
                max_expand_h = int(b['height'] * 0.1)
                el = max(-max_expand_w, min(el, max_expand_w))
                er = max(-max_expand_w, min(er, max_expand_w))
                et = max(-max_expand_h, min(et, max_expand_h))
                eb = max(-max_expand_h, min(eb, max_expand_h))
                # 计算新坐标
                new_x = max(0, b['x'] - el)
                new_y = max(0, b['y'] - et)
                new_w = b['width'] + el + er
                new_h = b['height'] + et + eb
                # 不能超出裁剪范围
                new_w = min(new_w, crop_w - new_x) if crop_w else new_w
                new_h = min(new_h, crop_h - new_y) if crop_h else new_h
                if new_w < 5 or new_h < 5:
                    continue
                used_ids.add(cid)
                furniture_boxes.append({'x': new_x, 'y': new_y, 'w': new_w, 'h': new_h, 'red': _is_red(c)})

        # 未被 LLM 提及的 candidate → 默认保留
        if not allow_unrestricted:
            for c in candidates:
                cid = c.get('candidate_id', -1)
                if cid in used_ids:
                    continue
                b = c['bbox']
                furniture_boxes.append({'x': b['x'], 'y': b['y'], 'w': b['width'], 'h': b['height'], 'red': _is_red(c)})

        # 安全网：防止 LLM 删光所有候选框导致房间无家具输出
        if not furniture_boxes and candidates and (not allow_unrestricted or not actions):
            logger.warning(
                f"  {room_name}: LLM 删除了全部 {len(candidates)} 个候选框，"
                f"回退保留所有 CV 候选"
            )
            for c in candidates:
                b = c['bbox']
                furniture_boxes.append({
                    'x': b['x'], 'y': b['y'], 'w': b['width'], 'h': b['height'],
                    'red': _is_red(c)
                })

        # 转为返回用的绝对像素坐标元组
        furniture = []
        for fb in furniture_boxes:
            furniture.append((
                room_name,
                crop_left + fb['x'], crop_top + fb['y'],
                fb['w'], fb['h'],
            ))

        return furniture

    def _extract_json_array(self, text: str) -> List[Dict]:
        """从文本中提取 JSON 数组"""
        m = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', text, re.DOTALL)
        if m:
            json_str = m.group(1)
        else:
            start = text.find('[')
            end = text.rfind(']')
            if start >= 0 and end > start:
                json_str = text[start:end+1]
            else:
                logger.warning(f"No JSON array in LLM output: {text[:200]}")
                return []
        try:
            data = json.loads(json_str)
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}\nText snippet: {text[:300]}")
            return []

    # ==============================================================
    # 智能设备匹配：将设备名分配到对应的家具框
    # ==============================================================
    @staticmethod
    def _normalize_device_name(name: str) -> str:
        """将设备名标准化为查找 key"""
        key = name.lower().strip()
        # 常见同义词映射
        synonyms = {
            'fridge': 'refrigerator',
            'Refrigerator': 'refrigerator',
            'Fr': 'refrigerator',
            'wash_machine': 'washing_machine',
            'washer': 'washing_machine',
            'tv cabinet': 'tv_stand',
            'tv_cabinet': 'tv_stand',
            'tv cabinet ': 'tv_stand',
            'television': 'tv',
            'cocktail_table': 'coffee_table',
            'center_table': 'coffee_table',
            'din_table': 'dining_table',
            'din table': 'dining_table',
        }
        return synonyms.get(key, key)

    def _get_device_size_spec(self, device_name: str) -> dict:
        """获取设备类型的期望尺寸规格。未知类型返回 None。"""
        key = self._normalize_device_name(device_name)
        # 部分匹配：遍历所有已知类型
        for known_key, spec in self.DEVICE_EXPECTED_SIZES.items():
            if known_key in key or key in known_key:
                return spec
        # 模糊匹配：去掉空格/特殊字符再比
        clean = ''.join(c for c in key if c.isalnum())
        for known_key, spec in self.DEVICE_EXPECTED_SIZES.items():
            known_clean = ''.join(c for c in known_key if c.isalnum())
            if known_clean in clean or clean in known_clean:
                return spec
        return None

    def _box_matches_device(self, box_w_px: float, box_h_px: float, device_name: str) -> bool:
        """检查检测框的真实尺寸是否与设备类型匹配"""
        spec = self._get_device_size_spec(device_name)
        if spec is None:
            return True  # 未知类型→不做尺寸检查

        # 像素 → 世界坐标(dm)
        s = max(self._scale, 0.001)
        real_w = box_w_px * s
        real_h = box_h_px * s

        # 尺寸范围检查（确保宽≤高）
        # 使用两步赋值：先取 min 覆盖 w，再用覆盖后的 w 去 max h
        # 宽>高时 real_h 也会被压缩为较小值，使尺寸检查更保守，
        # 从而让设备匹配在 LLM 输出波动时保持跨轮稳定
        real_w = min(real_w, real_h)
        real_h = max(real_w, real_h)
        if real_w < spec['min_w'] or real_w > spec['max_w']:
            logger.debug(f"  尺寸不匹配 {device_name}: 宽 {real_w:.1f}dm "
                         f"(期望 {spec['min_w']}~{spec['max_w']}dm)")
            return False
        if real_h < spec['min_h'] or real_h > spec['max_h']:
            logger.debug(f"  尺寸不匹配 {device_name}: 高 {real_h:.1f}dm "
                         f"(期望 {spec['min_h']}~{spec['max_h']}dm)")
            return False

        return True

    def _match_devices_to_furniture(self, furniture_list: List[Tuple],
                                     room_devices_map: Dict[str, List[Dict]],
                                     img_w: int, img_h: int
                                     ) -> Tuple[List[Tuple], Dict[Tuple, str]]:
        """将智能设备匹配到家具框。

        流程:
          1. 对所有设备，统一按"临近位置 + 尺寸兼容"匹配到最近的框（一次匹配）
          2. 逐框检查：单个设备→标记名；多个设备→按设备坐标相对位置平分矩形
          3. 未匹配到任何框的设备→仅记录日志

        Returns:
            (修正后的 furniture_list, Dict[furniture_tuple -> device_name])
            furniture_list 会在原地修改（分裂框时增加新条目）
        """
        if not room_devices_map:
            return furniture_list, {}

        s = max(self._scale, 0.001)

        # ── 展平所有设备（含实际尺寸 pw/ph，单位 dm）──
        all_devices = []  # [(room_name, device_name, px, py, pw_dm, ph_dm)]
        for room_name, devices in room_devices_map.items():
            for dev in devices:
                pw_dm = float(dev.get('pw', 0))
                ph_dm = float(dev.get('ph', 0))
                all_devices.append((
                    room_name,
                    dev.get('name', '?'),
                    float(dev['px']), float(dev['py']),
                    max(pw_dm, 0.01), max(ph_dm, 0.01),
                ))

        # ── 统一匹配阶段：每个设备找到最近且尺寸兼容的框 ──
        # 评分规则：距离框中心越近越好，类型不匹配 3 倍惩罚，装不下则跳过
        box_to_devices = [[] for _ in range(len(furniture_list))]
        unmatched_devices = []

        # 设备最大匹配距离（dm），超过此距离视为匹配失败
        _DEVICE_MAX_DIST_DM = 20

        for dev_idx, (d_room, d_name, d_px, d_py, d_pw, d_ph) in enumerate(all_devices):
            best_idx, best_score, best_raw_dist = -1, float('inf'), float('inf')
            dev_w_px = int(d_pw / s)
            dev_h_px = int(d_ph / s)

            for i, (f_room, f_x, f_y, f_w, f_h) in enumerate(furniture_list):
                fw_i, fh_i = int(f_w), int(f_h)
                # 跳过装不下设备实际尺寸的框
                if fw_i < dev_w_px or fh_i < dev_h_px:
                    continue
                # 计算到框中心的距离
                cx = int(f_x) + fw_i / 2
                cy = int(f_y) + fh_i / 2
                raw_dist = ((d_px - cx) ** 2 + (d_py - cy) ** 2) ** 0.5
                # 距离超过上限（世界坐标 dm）→ 跳过
                raw_dist_dm = raw_dist * s
                if raw_dist_dm > _DEVICE_MAX_DIST_DM:
                    continue
                dist = raw_dist
                # 类型不匹配 → 3 倍距离惩罚
                if not self._box_matches_device(fw_i, fh_i, d_name):
                    dist *= 3
                if dist < best_score:
                    best_score = dist
                    best_idx = i
                    best_raw_dist = raw_dist

            if best_idx >= 0:
                box_to_devices[best_idx].append(dev_idx)
            else:
                unmatched_devices.append(all_devices[dev_idx])

        name_map = {}
        split_indices = set()
        new_entries = []  # [(furniture_tuple, device_name), ...]

        # ── 逐框处理（单设备→直接标记, 多设备→分割）──
        for i, f_tuple in enumerate(furniture_list):
            dev_indices = box_to_devices[i]

            if not dev_indices:
                continue

            devs_in_box = [all_devices[idx] for idx in dev_indices]

            if len(devs_in_box) == 1:
                # ── 单个设备：框不变，仅标记设备名 ──
                name_map[f_tuple] = devs_in_box[0][1]

            else:
                # ── 多个设备 → 根据设备分布方向分割 ──
                f_room, f_x, f_y, f_w, f_h = f_tuple
                # 取该框中所有设备的平均像素尺寸作为最小分割尺寸
                avg_dev_w = max(int(sum(d[4] for d in devs_in_box) / len(devs_in_box) / s), 5)
                avg_dev_h = max(int(sum(d[5] for d in devs_in_box) / len(devs_in_box) / s), 5)
                min_split_size = min(avg_dev_w, avg_dev_h)
                fi_x, fi_y, fi_w, fi_h = int(f_x), int(f_y), int(f_w), int(f_h)
                px_vals = [d[2] for d in devs_in_box]
                py_vals = [d[3] for d in devs_in_box]
                spread_x = max(px_vals) - min(px_vals)
                spread_y = max(py_vals) - min(py_vals)

                if spread_x >= spread_y:
                    # 垂直分割（按 X 轴划分左右，矩形均分）
                    devs_in_box.sort(key=lambda d: d[2])
                    split_pos = fi_x + fi_w / 2  # 矩形中点均分
                    split_pos = max(fi_x + min_split_size,
                                    min(int(split_pos), fi_x + fi_w - min_split_size))

                    left_w = split_pos - fi_x
                    if left_w >= min_split_size:
                        left_tuple = (f_room, float(fi_x), float(fi_y),
                                      float(left_w), float(fi_h))
                        new_entries.append((left_tuple, devs_in_box[0][1]))
                    else:
                        unmatched_devices.append(devs_in_box[0])

                    right_x = split_pos
                    right_w = fi_x + fi_w - right_x
                    if right_w >= min_split_size:
                        right_tuple = (f_room, float(right_x), float(fi_y),
                                       float(right_w), float(fi_h))
                        new_entries.append((right_tuple, devs_in_box[-1][1]))
                    else:
                        if len(devs_in_box) > 1:
                            unmatched_devices.append(devs_in_box[-1])

                    logger.info(f"  设备匹配 {f_room}: 框内 {len(devs_in_box)} 个设备，"
                                f"垂直分割 (x={split_pos}), spread_x={spread_x}")
                else:
                    # 水平分割（按 Y 轴划分上下，矩形均分）
                    devs_in_box.sort(key=lambda d: d[3])
                    split_pos = fi_y + fi_h / 2  # 矩形中点均分
                    split_pos = max(fi_y + min_split_size,
                                    min(int(split_pos), fi_y + fi_h - min_split_size))

                    top_h = split_pos - fi_y
                    if top_h >= min_split_size:
                        top_tuple = (f_room, float(fi_x), float(fi_y),
                                     float(fi_w), float(top_h))
                        new_entries.append((top_tuple, devs_in_box[0][1]))
                    else:
                        unmatched_devices.append(devs_in_box[0])

                    bottom_y = split_pos
                    bottom_h = fi_y + fi_h - bottom_y
                    if bottom_h >= min_split_size:
                        bottom_tuple = (f_room, float(fi_x), float(bottom_y),
                                        float(fi_w), float(bottom_h))
                        new_entries.append((bottom_tuple, devs_in_box[-1][1]))
                    else:
                        if len(devs_in_box) > 1:
                            unmatched_devices.append(devs_in_box[-1])

                    logger.info(f"  设备匹配 {f_room}: 框内 {len(devs_in_box)} 个设备，"
                                f"水平分割 (y={split_pos}), spread_y={spread_y}")

                split_indices.add(i)

        # ── 未匹配设备 → 仅记录日志，不做处理 ──
        for d_room, d_name, d_px, d_py, d_pw, d_ph in unmatched_devices:
            logger.info(f"  设备匹配 {d_room}: 设备 {d_name} 未找到兼容的框")

        # ── 更新 furniture_list（仅插入新分裂的条目）──
        if new_entries:
            new_list = [t for i, t in enumerate(furniture_list) if i not in split_indices]
            for new_tuple, dev_name in new_entries:
                new_list.append(new_tuple)
                name_map[new_tuple] = dev_name
            furniture_list[:] = new_list

        return furniture_list, name_map

    # ==============================================================
    # 写入 YAML
    # ==============================================================
    def _write_furniture(self, layout: Dict, furniture_list: List[Tuple],
                          name_map: Dict[Tuple, str] = None):
        """将家具写入 layout，写 id + name + position（像素坐标）

        Args:
            name_map: 由 _match_devices_to_furniture 返回的 {furniture_tuple -> device_name} 映射
        """
        if name_map is None:
            name_map = {}

        by_room: Dict[str, list] = {}
        for name, px, py, pw, ph in furniture_list:
            # 确保转为原生 Python 类型，避免 numpy 标量污染 YAML
            by_room.setdefault(name, []).append({
                "position": {
                    "x": float(px), "y": float(py),
                    "width": float(pw), "height": float(ph),
                },
                "tuple": (name, px, py, pw, ph),
            })
        for r in layout.get('house', {}).get('rooms', []):
            name = r['name']
            items = by_room.get(name, [])
            if not items:
                continue
            furs = []
            for i, item in enumerate(items, 1):
                entry = {"id": f"{name}_fur{i}"}
                # 如果有匹配的设备名，写入 name 字段
                device_name = name_map.get(item["tuple"])
                if device_name:
                    entry["name"] = device_name
                entry["position"] = item["position"]
                furs.append(entry)
            r['furniture'] = furs

    def _save_yaml(self, layout: Dict, output_path: str):
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # 注册 numpy 标量的 representer，防止写出 python/object 标签
        def _np_float_repr(dumper, data):
            return dumper.represent_float(float(data))
        def _np_int_repr(dumper, data):
            return dumper.represent_int(int(data))

        yaml.add_representer(np.float64, _np_float_repr)
        yaml.add_representer(np.float32, _np_float_repr)
        yaml.add_representer(np.int64, _np_int_repr)
        yaml.add_representer(np.int32, _np_int_repr)

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(layout, f, default_flow_style=None, allow_unicode=True, sort_keys=False, indent=2)

    def _save_intermediate(self, layout: Dict, output_path: str, label: str = None):
        """保存中间 YAML 快照（使用独立的 YAML dumper，不影响全局状态）"""
        import yaml as _yaml_module
        import numpy as _np

        def _np_float_repr(dumper, data):
            return dumper.represent_float(float(data))

        def _np_int_repr(dumper, data):
            return dumper.represent_int(int(data))

        _yaml_module.add_representer(_np.float64, _np_float_repr)
        _yaml_module.add_representer(_np.float32, _np_float_repr)
        _yaml_module.add_representer(_np.int64, _np_int_repr)
        _yaml_module.add_representer(_np.int32, _np_int_repr)

        with open(output_path, 'w', encoding='utf-8') as f:
            _yaml_module.dump(layout, f, default_flow_style=None, allow_unicode=True, sort_keys=False, indent=2)

        desc = f" -> {output_path}" if label is None else f" ({label}) -> {output_path}"
        logger.info(f"中间 YAML{desc}")


# ─── 独立运行入口 ─────────────────────────────────────────────
if __name__ == '__main__':
    from dotenv import load_dotenv

    def _load_env():
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
        if not os.path.exists(env_path):
            return
        for enc in ('utf-8-sig', 'utf-16-le', 'utf-16'):
            try:
                load_dotenv(dotenv_path=env_path, encoding=enc)
                return
            except UnicodeDecodeError:
                continue
        load_dotenv(dotenv_path=env_path, encoding='utf-8', errors='ignore')
    _load_env()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_image = os.path.join(base, 'photo2yaml', 'output', 'masks', 'greyroom.png')
    default_yaml = os.path.join(base, 'photo2yaml', 'output', 'floorplan_real.yaml')
    default_traj = os.path.join(base, 'input', 'traj', 'Continuous_time', '0622',
                                '0622-Engineer-all_trajectory.json')
    default_hatch_mask = os.path.join(base, 'photo2yaml', 'output', 'masks', 'hatch_mask.png')

    image_path = sys.argv[1] if len(sys.argv) > 1 else default_image
    yaml_path = sys.argv[2] if len(sys.argv) > 2 else default_yaml
    traj_path = sys.argv[3] if len(sys.argv) > 3 else default_traj
    output_path = sys.argv[4] if len(sys.argv) > 4 else yaml_path
    hatch_mask_path = sys.argv[5] if len(sys.argv) > 5 else default_hatch_mask

    api_key = os.environ.get("FURNITURE_API_KEY")
    base_url = os.environ.get("FURNITURE_BASE_URL")
    if not api_key:
        print("[错误] 环境变量 FURNITURE_API_KEY 未设置，请在 .env 文件中配置")
        print("   格式: FURNITURE_API_KEY=your_key_here")
        sys.exit(1)

    agent = FurnitureDetectionAgent(api_key=api_key, base_url=base_url)
    agent.process(image_path, yaml_path, trajectory_json_path=traj_path, 
                  output_path=output_path, hatch_mask_path=hatch_mask_path)
