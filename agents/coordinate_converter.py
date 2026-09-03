"""
坐标统一转换模块

功能:
  在像素坐标 (图像 top-left 原点) 和世界坐标 (bottom-left 原点, 单位:m)
  之间双向转换。转换范围涵盖: 房间、家具、门、墙段、子房间等全部几何数据。

用法:
  converter = CoordinateConverter.from_yaml('floorplan_real.yaml')
  pixel_x, pixel_y = converter.world_to_pixel(wx, wy)
  world_x, world_y = converter.pixel_to_world(px, py)
"""
from typing import Dict, Tuple, List
import yaml
import copy
import logging

logger = logging.getLogger(__name__)


class CoordinateConverter:
    """像素坐标 ↔ 世界坐标转换器

    像素坐标: (0,0) = image top-left, y 向下增长
    世界坐标: (0,0) = house bottom-left (bl_x, bl_y), y 向上增长

    转换公式:
      world_x = (pixel_x - bl_x) * SCALE
      world_y = (bl_y - pixel_y) * SCALE
      pixel_x = world_x / SCALE + bl_x
      pixel_y = bl_y - world_y / SCALE
    """

    def __init__(self, bl_x: int, bl_y: int, scale: float):
        self.bl_x = bl_x
        self.bl_y = bl_y
        self.scale = scale

    @classmethod
    def from_yaml(cls, yaml_path: str, img_shape: tuple = None) -> 'CoordinateConverter':
        """从 YAML 文件读取 metadata 创建转换器"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            layout = yaml.safe_load(f)
        meta = layout.get('metadata', {})
        bl_x = meta.get('bl_x')
        bl_y = meta.get('bl_y')
        scale = meta.get('SCALE')
        if scale is None:
            scale = 1.0
        else:
            scale = float(scale)

        if bl_x is None or bl_y is None:
            if img_shape is not None:
                h, w = img_shape[:2]
                bl_x = int(bl_x) if bl_x is not None else 0
                bl_y = int(bl_y) if bl_y is not None else h
                print(f"[坐标转换] metadata 缺少 bl_x/bl_y，从图像尺寸推算: ({bl_x}, {bl_y})")
            else:
                bl_x = 0
                bl_y = 1000
                print(f"[坐标转换] metadata 缺少 bl_x/bl_y，使用默认值: ({bl_x}, {bl_y})")
        else:
            bl_x = int(bl_x)
            bl_y = int(bl_y)

        return cls(bl_x, bl_y, scale)

    # ─── 世界 → 像素 ────────────────────────────────────────

    def world_to_pixel(self, wx: float, wy: float) -> Tuple[float, float]:
        """单个点: 世界坐标 → 像素坐标"""
        px = wx / self.scale + self.bl_x
        py = self.bl_y - wy / self.scale
        return px, py

    def world_rect_to_pixel(
        self, wx: float, wy: float, ww: float, wh: float
    ) -> Tuple[float, float, float, float]:
        """矩形: 世界坐标 (x,y,w,h) → 像素坐标 (x,y,w,h)"""
        px = wx / self.scale + self.bl_x
        py = self.bl_y - (wy + wh) / self.scale
        pw = ww / self.scale
        ph = wh / self.scale
        return px, py, pw, ph

    # ─── 像素 → 世界 ────────────────────────────────────────

    def pixel_to_world(self, px: float, py: float) -> Tuple[float, float]:
        """单个点: 像素坐标 → 世界坐标"""
        wx = (px - self.bl_x) * self.scale
        wy = (self.bl_y - py) * self.scale
        return round(wx, 2), round(wy, 2)

    def pixel_rect_to_world(
        self, px: float, py: float, pw: float, ph: float
    ) -> Tuple[float, float, float, float]:
        """矩形: 像素坐标 (x,y,w,h, top-left 原点) → 世界坐标 (x,y,w,h, bottom-left 原点)"""
        wx = (px - self.bl_x) * self.scale
        wy = (self.bl_y - (py + ph)) * self.scale
        ww = pw * self.scale
        wh = ph * self.scale
        return round(wx, 2), round(wy, 2), round(ww, 2), round(wh, 2)

    # ==========================================================
    # 门坐标转换
    # ==========================================================
    def _convert_door_pixel_to_world(self, door: Dict) -> None:
        """原地转换门坐标: 像素 → 世界

        支持两种格式:
          - 扁平点: {id: door1, x: 676.0, y: 213.0}
          - 多点数组: {name: "...", points: [[x1,y1],[x2,y2],[x3,y3]]}
        """
        if not isinstance(door, dict):
            return

        # 格式 1: 扁平 x, y
        if 'x' in door and 'y' in door:
            wx, wy = self.pixel_to_world(door['x'], door['y'])
            door['x'] = wx
            door['y'] = wy

        # 格式 2: points 数组
        if 'points' in door and isinstance(door['points'], list):
            new_points = []
            for pt in door['points']:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    wx, wy = self.pixel_to_world(pt[0], pt[1])
                    new_points.append([wx, wy])
                else:
                    new_points.append(pt)
            door['points'] = new_points

    # ==========================================================
    # 墙坐标转换
    # ==========================================================
    def _convert_walls_pixel_to_world(self, walls: Dict) -> None:
        """原地转换墙坐标: 像素 → 世界

        转换 walls 中的 extra 墙段（如果有）。
        walls 的 top/right/bottom/left 布尔标志不需要转换。
        """
        if not isinstance(walls, dict):
            return

        # 转换 extra 墙段中的 points
        extra = walls.get('extra', [])
        for ew in extra:
            if isinstance(ew, dict) and 'points' in ew:
                new_pts = []
                for pt in ew['points']:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        wx, wy = self.pixel_to_world(pt[0], pt[1])
                        new_pts.append([wx, wy])
                    else:
                        new_pts.append(pt)
                ew['points'] = new_pts

    # ==========================================================
    # wall_segments 坐标转换
    # ==========================================================
    def _convert_wall_segments_pixel_to_world(self, segments: List[Dict]) -> None:
        """原地转换 wall_segments: 像素 → 世界

        wall_segments 格式: [{x1, y1, x2, y2}, ...]
        """
        for seg in segments:
            if isinstance(seg, dict):
                if all(k in seg for k in ('x1', 'y1', 'x2', 'y2')):
                    wx1, wy1 = self.pixel_to_world(seg['x1'], seg['y1'])
                    wx2, wy2 = self.pixel_to_world(seg['x2'], seg['y2'])
                    seg['x1'] = wx1; seg['y1'] = wy1
                    seg['x2'] = wx2; seg['y2'] = wy2

    # ==========================================================
    # 完整 layout 转换 (像素 → 世界)
    # ==========================================================
    def convert_layout_to_world(self, layout: Dict) -> Dict:
        """将整个 layout 从像素坐标转为世界坐标

        转换范围:
          - house.size (尺寸缩放)
          - 每个房间的 position (矩形转世界)
          - 每个房间的 doors (点坐标转世界)
          - 每个房间的 walls.extra (点坐标转世界)
          - 每个房间的 furniture (矩形转世界)
          - 每个房间的 subrooms (递归转换)
          - metadata (保留不转换)
        """
        result = copy.deepcopy(layout)

        # house.size: 尺寸直接乘 SCALE
        house_size = result.get('house', {}).get('size', {})
        if 'x' in house_size and 'y' in house_size:
            house_size['x'] = round(house_size['x'] * self.scale, 2)
            house_size['y'] = round(house_size['y'] * self.scale, 2)

        # 遍历每个房间
        for room in result.get('house', {}).get('rooms', []):
            self._convert_room_to_world(room)

        return result

    def _convert_room_to_world(self, room: Dict) -> None:
        """原地转换单个房间的所有坐标: 像素 → 世界"""
        if not isinstance(room, dict):
            return

        # 1. 房间 position
        pos = room.get('position', {})
        if all(k in pos for k in ('x', 'y', 'width', 'height')):
            wx, wy, ww, wh = self.pixel_rect_to_world(
                pos['x'], pos['y'], pos['width'], pos['height']
            )
            pos['x'] = wx; pos['y'] = wy
            pos['width'] = ww; pos['height'] = wh

        # 2. 门坐标
        for door in room.get('doors', []):
            self._convert_door_pixel_to_world(door)

        # 3. 墙坐标 (extra 墙段)
        walls = room.get('walls', {})
        self._convert_walls_pixel_to_world(walls)

        # 4. wall_segments (子房间中的墙段)
        wall_segs = room.get('wall_segments', [])
        if wall_segs:
            self._convert_wall_segments_pixel_to_world(wall_segs)

        # 5. 家具坐标
        for fur in room.get('furniture', []):
            fpos = fur.get('position', {})
            if all(k in fpos for k in ('x', 'y', 'width', 'height')):
                wfx, wfy, wfw, wfh = self.pixel_rect_to_world(
                    fpos['x'], fpos['y'], fpos['width'], fpos['height']
                )
                fpos['x'] = wfx; fpos['y'] = wfy
                fpos['width'] = wfw; fpos['height'] = wfh

        # 6. 子房间 (递归)
        for sr in room.get('subrooms', []):
            self._convert_room_to_world(sr)
