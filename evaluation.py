"""
家庭布局推理系统单轮评估脚本
===========================
根据 tabbit_评估指标.md 中定义的四个指标进行评估：
  1. Furniture Naming Accuracy (FNA)  - 家具命名准确率
  2. Furniture Detection F1           - 家具检测 F1 (基于匈牙利匹配 + IoU>0.2)
  3. Room Type Accuracy (RTA)         - 房间类型准确率
  4. Complete Layout Score (CLS)      - 综合布局得分

评估策略 (两种，并排对比):
  A. 全局匹配 (Global)          - 所有家具跨房间统一匈牙利匹配
  B. 房间内对齐匹配 (Aligned)  - 先匹配房间，每对房间内平移对齐后匹配家具

  预测生成阶段禁止读取 GT。预测冻结后，独立评分器读取 GT 计算指标：A 衡量
  绝对坐标性能，B 衡量消除房间级整体平移后的性能。所有方法均报告 A 和 B，
  方法间比较必须使用相同协议。

# 评估单个布局：
    python evaluation.py --pred output/yaml/03_final_{code}.yaml --gt GT/layout_{code}.yaml

使用方式：
  python evaluation.py --pred output/yaml/03_final_0622.yaml --gt GT/layout_0622.yaml

支持批量：
  python evaluation.py --pred_dir output/yaml/ --gt_dir GT/
"""

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import yaml


# ============================================================
# 数据结构
# ============================================================

class BBox:
    """轴对齐边界框 (x, y 为左下角, width, height)"""
    __slots__ = ('x', 'y', 'width', 'height')

    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    def iou(self, other: 'BBox') -> float:
        """计算与另一个 BBox 的 IoU (Intersection over Union)"""
        ix = max(self.x, other.x)
        iy = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        iw = ix2 - ix
        ih = iy2 - iy
        if iw <= 0 or ih <= 0:
            return 0.0

        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def translate(self, dx: float, dy: float) -> 'BBox':
        """平移，返回新 BBox"""
        return BBox(self.x + dx, self.y + dy, self.width, self.height)

    def __repr__(self):
        return f"BBox(x={self.x:.1f}, y={self.y:.1f}, w={self.width:.1f}, h={self.height:.1f})"


class Room:
    __slots__ = ('name', 'bbox', 'furniture', 'index')

    def __init__(self, name: str, bbox: BBox, furniture: List['Furniture'], index: int = -1):
        self.name = name
        self.bbox = bbox
        self.furniture = furniture
        self.index = index


class Furniture:
    __slots__ = ('name', 'bbox', 'room_name', 'is_smart_device')

    def __init__(self, name: str, bbox: BBox, room_name: str = '', is_smart_device: bool = False):
        self.name = name
        self.bbox = bbox
        self.room_name = room_name
        self.is_smart_device = is_smart_device

    def clone(self) -> 'Furniture':
        return Furniture(self.name, self.bbox, self.room_name, self.is_smart_device)


# ============================================================
# 数据加载
# ============================================================

def load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_rooms(data: dict) -> List[Room]:
    """从 YAML 数据中解析房间列表"""
    rooms = []
    house_data = data.get('house', data)
    for i, room_data in enumerate(house_data.get('rooms', [])):
        pos = room_data['position']
        bbox = BBox(
            float(pos['x']), float(pos['y']),
            float(pos['width']), float(pos['height'])
        )
        furniture_list = []
        for fur_data in room_data.get('furniture', []):
            fpos = fur_data['position']
            fbbox = BBox(
                float(fpos['x']), float(fpos['y']),
                float(fpos['width']), float(fpos['height'])
            )
            fur_name = str(fur_data.get('name', '')).strip()
            furniture_list.append(Furniture(
                name=fur_name,
                bbox=fbbox,
                room_name=room_data.get('name', '')
            ))
        rooms.append(Room(
            name=str(room_data.get('name', '')).strip(),
            bbox=bbox,
            furniture=furniture_list,
            index=i
        ))
    return rooms


# ============================================================
# 匈牙利匹配算法 (手动实现，避免 scipy 依赖)
# ============================================================

def hungarian_matching(cost_matrix: List[List[float]]) -> List[Tuple[int, int]]:
    """
    匈牙利算法求解二分图最小权匹配。
    cost_matrix: n_rows x n_cols 的代价矩阵
    返回匹配对列表 [(row_idx, col_idx), ...]
    
    对于非方阵，通过添加虚节点来处理。
    """
    n_rows = len(cost_matrix)
    n_cols = len(cost_matrix[0]) if cost_matrix else 0

    if n_rows == 0 or n_cols == 0:
        return []

    n = max(n_rows, n_cols)

    INF = 1e9
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i < n_rows and j < n_cols:
                mat[i][j] = cost_matrix[i][j]
            else:
                mat[i][j] = INF

    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = mat[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    matches = []
    for j in range(1, n + 1):
        i = p[j]
        if i >= 1 and i <= n_rows and j <= n_cols:
            if mat[i - 1][j - 1] < INF * 0.5:
                matches.append((i - 1, j - 1))

    return matches


# ============================================================
# 名称规范化
# ============================================================

def normalize_name(name: str) -> str:
    """规范化名称用于比较：小写 + 去空格 + 去下划线"""
    name = name.strip().lower()
    name = name.replace('_', ' ')
    name = re.sub(r'\s+', ' ', name)
    return name


ROOM_NAME_ALIASES = {
    'living room': 'living_room', 
    'livingroom': 'living_room',
    'dining room': 'dining_room', 
    'diningroom': 'dining_room',
    'second bedroom': 'bedroom', 
    'master bedroom': 'bedroom',
    'bed room': 'bedroom', 
    'bath room': 'bathroom',
    'washroom': 'bathroom', 
    'study room': 'studyroom',
    'study': 'studyroom', 
    'balcony': 'balcony',
    'kitchen': 'kitchen', 
    'bedroom': 'bedroom',
    'bathroom': 'bathroom', 
    'studyroom': 'studyroom',
    'other': 'other',
}


def normalize_room_name(name: str) -> str:
    n = normalize_name(name)
    return ROOM_NAME_ALIASES.get(n, n)


FURNITURE_NAME_ALIASES = {
    'coffee table': 'coffee_table', 
    'dining table': 'dining_table',
    'diningtable': 'dining_table', 
    'tv stand': 'tv_stand',
    'tvstand': 'tv_stand', 
    'shoe cabinet': 'shoe_cabinet',
    'shoecabinet': 'shoe_cabinet', 
    'washing machine': 'washing_machine',
    'washingmachine': 'washing_machine', 
    'coat rack': 'coat_rack',
    'coatrack': 'coat_rack', 
    'book shelf': 'bookshelf',
    'night stand': 'nightstand',
    'cabinet': 'cabinet', 
    'wardrobe': 'wardrobe', 
    'closet': 'wardrobe',
}


def normalize_furniture_name(name: str) -> str:
    n = normalize_name(name)
    return FURNITURE_NAME_ALIASES.get(n, n)


# ============================================================
# 核心评估逻辑
# ============================================================

class FurnitureMatchResult:
    """单对家具匹配的详细结果"""
    __slots__ = ('gt_name', 'pred_name', 'gt_room', 'pred_room',
                 'iou', 'name_correct', 'gt_norm', 'pred_norm',
                 'offset_applied', 'gt_area', 'centroid_dist')

    def __init__(self, gt_name, pred_name, gt_room, pred_room,
                 iou, name_correct, gt_norm, pred_norm, offset_applied=(0, 0),
                 gt_area=0, centroid_dist=0):
        self.gt_name = gt_name
        self.pred_name = pred_name
        self.gt_room = gt_room
        self.pred_room = pred_room
        self.iou = iou
        self.name_correct = name_correct
        self.gt_norm = gt_norm
        self.pred_norm = pred_norm
        self.offset_applied = offset_applied
        self.gt_area = gt_area
        self.centroid_dist = centroid_dist


class EvaluationResult:
    """单次评估结果 (针对一种匹配策略)"""

    # 家具匹配 IoU 阈值
    FURNITURE_IOU_THRESHOLD = 0.2

    # 按面积分层: 小 < 50, 中 50-200, 大 >= 200 (单位 dm²)
    SMALL_MAX = 50
    LARGE_MIN = 200

    # 质心距离阈值: 质心偏移 ≤ 5dm 视为检测成功 (尺寸无关，对小家具更公平)
    CENTROID_THRESHOLD = 5.0

    def __init__(self, strategy_name: str = ""):
        self.strategy_name = strategy_name

        # 房间层面
        self.n_gt_rooms: int = 0
        self.n_pred_rooms: int = 0
        self.n_matched_rooms: int = 0
        self.n_correct_room_type: int = 0
        self.room_matches: List[Dict] = []

        # 家具层面
        self.n_gt_furniture: int = 0
        self.n_pred_furniture: int = 0
        self.tp_furniture: int = 0
        self.tp_correct_name: int = 0
        self.furniture_matches: List[FurnitureMatchResult] = []
        self.gt_furniture_areas: List[float] = []

        # 按面积分层的统计
        # {class_label: {'gt': N, 'tp': N, 'tp_correct_name': N}}
        self.per_class: Dict[str, dict] = {}

        # 计算后的指标
        self.room_type_accuracy: float = 0.0
        self.room_precision: float = 0.0
        self.room_recall: float = 0.0
        self.room_f1: float = 0.0
        self.furniture_precision: float = 0.0
        self.furniture_recall: float = 0.0
        self.furniture_f1: float = 0.0
        self.furniture_naming_accuracy: float = 0.0
        self.semantic_precision: float = 0.0
        self.semantic_recall: float = 0.0
        self.semantic_f1: float = 0.0
        self.cls: float = 0.0

        # 质心距离指标 (尺寸无关，对小家具更公平)
        self.centroid_tp: int = 0
        self.centroid_f1: float = 0.0
        self.centroid_recall: float = 0.0

    @staticmethod
    def _area_class(area: float) -> str:
        if area < EvaluationResult.SMALL_MAX:
            return 'small (<50)'
        elif area < EvaluationResult.LARGE_MIN:
            return 'medium (50-200)'
        else:
            return 'large (>=200)'

    def compute_metrics(self):
        """计算所有指标，含按面积分层"""
        # 房间语义 TP 同时要求房间被定位匹配且类型正确。RTA 保留原字段名，
        # 但分母使用完整 GT，避免漏检房间不进入分母而抬高结果。
        if self.n_gt_rooms > 0:
            self.room_type_accuracy = self.n_correct_room_type / self.n_gt_rooms
            self.room_recall = self.room_type_accuracy
        if self.n_pred_rooms > 0:
            self.room_precision = self.n_correct_room_type / self.n_pred_rooms
        if self.room_precision + self.room_recall > 0:
            self.room_f1 = (2 * self.room_precision * self.room_recall /
                            (self.room_precision + self.room_recall))

        # TP = IoU > 0.2 的匹配数
        self.tp_furniture = sum(1 for m in self.furniture_matches if m.iou > self.FURNITURE_IOU_THRESHOLD)

        if self.n_pred_furniture > 0:
            self.furniture_precision = self.tp_furniture / self.n_pred_furniture
        if self.n_gt_furniture > 0:
            self.furniture_recall = self.tp_furniture / self.n_gt_furniture
        if self.furniture_precision + self.furniture_recall > 0:
            self.furniture_f1 = (2 * self.furniture_precision * self.furniture_recall /
                                 (self.furniture_precision + self.furniture_recall))

        # 命名准确率
        if self.tp_furniture > 0:
            self.tp_correct_name = sum(
                1 for m in self.furniture_matches
                if m.iou > self.FURNITURE_IOU_THRESHOLD and m.name_correct
            )
            self.furniture_naming_accuracy = self.tp_correct_name / self.tp_furniture

        # 端到端语义检测：位置达标且名称正确才算 TP。该指标同时惩罚
        # 漏检、多检和命名错误，适合作为主语义指标。
        if self.n_pred_furniture > 0:
            self.semantic_precision = self.tp_correct_name / self.n_pred_furniture
        if self.n_gt_furniture > 0:
            self.semantic_recall = self.tp_correct_name / self.n_gt_furniture
        if self.semantic_precision + self.semantic_recall > 0:
            self.semantic_f1 = (2 * self.semantic_precision * self.semantic_recall /
                                (self.semantic_precision + self.semantic_recall))

        # 按面积分层统计。分母从完整 GT 初始化，不能仅依赖匈牙利匹配对，
        # 否则 GT 多于预测时未匹配的 GT 会从分层分母消失。
        per_class = {}
        for area in self.gt_furniture_areas:
            cls = self._area_class(area)
            if cls not in per_class:
                per_class[cls] = {'gt': 0, 'tp': 0, 'tp_correct_name': 0}
            per_class[cls]['gt'] += 1
        for m in self.furniture_matches:
            cls = self._area_class(m.gt_area)
            if cls not in per_class:
                per_class[cls] = {'gt': 0, 'tp': 0, 'tp_correct_name': 0}
            if m.iou > self.FURNITURE_IOU_THRESHOLD:
                per_class[cls]['tp'] += 1
                if m.name_correct:
                    per_class[cls]['tp_correct_name'] += 1
        self.per_class = per_class

        # 质心距离指标: TP = 质心距离 ≤ 5dm 的匹配数
        self.centroid_tp = sum(1 for m in self.furniture_matches
                               if m.centroid_dist <= self.CENTROID_THRESHOLD)
        if self.n_gt_furniture > 0:
            self.centroid_recall = self.centroid_tp / self.n_gt_furniture
        if self.n_pred_furniture > 0:
            centroid_prec = self.centroid_tp / self.n_pred_furniture
            if self.centroid_recall + centroid_prec > 0:
                self.centroid_f1 = (2 * self.centroid_recall * centroid_prec /
                                    (self.centroid_recall + centroid_prec))

        # 旧 CLS 仅为探索性兼容指标；权重未经验证，主结论应使用 Room F1、
        # Localization F1 和 Semantic F1。
        self.cls = (0.45 * self.furniture_naming_accuracy +
                    0.35 * self.furniture_f1 +
                    0.20 * self.room_type_accuracy)


def match_rooms(gt_rooms: List[Room], pred_rooms: List[Room],
                iou_threshold: float = 0.2) -> Tuple[Dict[int, int], set]:
    """
    房间匈牙利匹配。返回 (gt_idx -> pred_idx 映射, 已匹配的 pred 集合)。
    """
    n_gt = len(gt_rooms)
    n_pred = len(pred_rooms)
    room_cost = [[0.0] * n_pred for _ in range(n_gt)]
    for i in range(n_gt):
        for j in range(n_pred):
            room_cost[i][j] = 1.0 - gt_rooms[i].bbox.iou(pred_rooms[j].bbox)

    raw_matches = hungarian_matching(room_cost)

    room_match_map = {}
    pred_matched_set = set()
    for gi, pj in raw_matches:
        if gt_rooms[gi].bbox.iou(pred_rooms[pj].bbox) > iou_threshold:
            room_match_map[gi] = pj
            pred_matched_set.add(pj)
    return room_match_map, pred_matched_set


def match_furniture_global(gt_furniture: List[Furniture],
                           pred_furniture: List[Furniture]) -> List[FurnitureMatchResult]:
    """
    策略 A: 全局家具匹配。所有家具跨房间统一匈牙利匹配，不进行坐标对齐。
    """
    n_gt = len(gt_furniture)
    n_pred = len(pred_furniture)

    if n_gt == 0 or n_pred == 0:
        return []

    cost = [[0.0] * n_pred for _ in range(n_gt)]
    for i in range(n_gt):
        for j in range(n_pred):
            cost[i][j] = 1.0 - gt_furniture[i].bbox.iou(pred_furniture[j].bbox)

    raw_matches = hungarian_matching(cost)
    results = []
    for gi, pj in raw_matches:
        gt_fur = gt_furniture[gi]
        pred_fur = pred_furniture[pj]
        iou = gt_fur.bbox.iou(pred_fur.bbox)
        # 质心距离
        cdist = ((gt_fur.bbox.center_x - pred_fur.bbox.center_x) ** 2 +
                 (gt_fur.bbox.center_y - pred_fur.bbox.center_y) ** 2) ** 0.5
        gt_norm = normalize_furniture_name(gt_fur.name)
        pred_norm = normalize_furniture_name(pred_fur.name)
        name_correct = (gt_norm == pred_norm) if iou > EvaluationResult.FURNITURE_IOU_THRESHOLD else False
        results.append(FurnitureMatchResult(
            gt_name=gt_fur.name, pred_name=pred_fur.name,
            gt_room=gt_fur.room_name, pred_room=pred_fur.room_name,
            iou=iou, name_correct=name_correct,
            gt_norm=gt_norm, pred_norm=pred_norm,
            offset_applied=(0.0, 0.0),
            gt_area=gt_fur.bbox.area,
            centroid_dist=cdist
        ))
    return results


def match_furniture_per_room_aligned(gt_rooms: List[Room], pred_rooms: List[Room],
                                     room_match_map: Dict[int, int]) -> List[FurnitureMatchResult]:
    """
    策略 B: 房间内对齐匹配。
    1. 对每个匹配的房间对，计算 GT 房间中心 vs 预测房间中心的偏移量
    2. 将预测家具按该偏移整体平移（消除房间级系统误差）
    3. 在平移后的坐标上做匈牙利匹配 + IoU 判定

    此函数仅属于预测冻结后的独立评分器。房间匹配和偏移量不得反馈给模型，
    也不得用于修改预测文件。

    未匹配房间中的家具不参与匹配（统一算 FN/FP）。
    """
    all_results = []

    for gi, pj in room_match_map.items():
        gt_room = gt_rooms[gi]
        pred_room = pred_rooms[pj]

        # 计算房间中心偏移
        dx = gt_room.bbox.center_x - pred_room.bbox.center_x
        dy = gt_room.bbox.center_y - pred_room.bbox.center_y

        gt_furs = gt_room.furniture
        pred_furs = pred_room.furniture

        if not gt_furs or not pred_furs:
            continue

        # 对齐预测家具
        aligned_preds = [
            Furniture(f.name, f.bbox.translate(dx, dy), f.room_name)
            for f in pred_furs
        ]

        n_gt = len(gt_furs)
        n_pred = len(aligned_preds)
        cost = [[0.0] * n_pred for _ in range(n_gt)]
        for i in range(n_gt):
            for j in range(n_pred):
                cost[i][j] = 1.0 - gt_furs[i].bbox.iou(aligned_preds[j].bbox)

        raw_matches = hungarian_matching(cost)
        for fi, fj in raw_matches:
            gt_fur = gt_furs[fi]
            pred_fur = pred_furs[fj]  # 原始预测家具
            aligned_pred = aligned_preds[fj]
            iou = gt_fur.bbox.iou(aligned_pred.bbox)
            # 质心距离 (对齐后)
            cdist = ((gt_fur.bbox.center_x - aligned_pred.bbox.center_x) ** 2 +
                     (gt_fur.bbox.center_y - aligned_pred.bbox.center_y) ** 2) ** 0.5
            gt_norm = normalize_furniture_name(gt_fur.name)
            pred_norm = normalize_furniture_name(pred_fur.name)
            name_correct = (gt_norm == pred_norm) if iou > EvaluationResult.FURNITURE_IOU_THRESHOLD else False
            all_results.append(FurnitureMatchResult(
                gt_name=gt_fur.name, pred_name=pred_fur.name,
                gt_room=gt_room.name, pred_room=pred_room.name,
                iou=iou, name_correct=name_correct,
                gt_norm=gt_norm, pred_norm=pred_norm,
                offset_applied=(dx, dy),
                gt_area=gt_fur.bbox.area,
                centroid_dist=cdist
            ))

    return all_results


def evaluate_single(pred_path: str, gt_path: str,
                    verbose: bool = True) -> Tuple[EvaluationResult, EvaluationResult]:
    """
    评估单个家庭，返回两种策略的结果: (全局匹配, 房间内对齐匹配)。

    调用前提：pred_path 已由不读取测试 GT 的生成流程产生并冻结。gt_path 只在
    本评分函数中用于计算指标，不得反馈给预测流程。
    """
    # ---- 加载数据 ----
    pred_data = load_yaml(pred_path)
    gt_data = load_yaml(gt_path)

    gt_rooms = parse_rooms(gt_data)
    pred_rooms = parse_rooms(pred_data)

    # ---- 公共: 房间匹配 ----
    room_match_map, pred_matched_set = match_rooms(gt_rooms, pred_rooms)

    n_gt_rooms = len(gt_rooms)
    n_pred_rooms = len(pred_rooms)
    n_matched = len(room_match_map)

    # ---- 收集全局家具列表 (用于策略 A) ----
    gt_furniture_all = []
    for room in gt_rooms:
        for fur in room.furniture:
            fur.room_name = room.name
            gt_furniture_all.append(fur)

    pred_furniture_all = []
    for room in pred_rooms:
        for fur in room.furniture:
            fur.room_name = room.name
            pred_furniture_all.append(fur)

    # ============================================================
    # 策略 A: 全局匹配
    # ============================================================
    result_a = EvaluationResult("A. 全局匹配 (Global)")
    result_a.n_gt_rooms = n_gt_rooms
    result_a.n_pred_rooms = n_pred_rooms
    result_a.n_matched_rooms = n_matched
    result_a.n_gt_furniture = len(gt_furniture_all)
    result_a.n_pred_furniture = len(pred_furniture_all)
    result_a.gt_furniture_areas = [fur.bbox.area for fur in gt_furniture_all]

    # 房间类型正确性
    for gi, pj in room_match_map.items():
        gt_name = normalize_room_name(gt_rooms[gi].name)
        pred_name = normalize_room_name(pred_rooms[pj].name)
        is_correct = (gt_name == pred_name)
        if is_correct:
            result_a.n_correct_room_type += 1
        result_a.room_matches.append({
            'gt_room': gt_rooms[gi].name, 'pred_room': pred_rooms[pj].name,
            'gt_normalized': gt_name, 'pred_normalized': pred_name,
            'iou': round(gt_rooms[gi].bbox.iou(pred_rooms[pj].bbox), 4),
            'correct': is_correct
        })
    for gi in range(n_gt_rooms):
        if gi not in room_match_map:
            result_a.room_matches.append({
                'gt_room': gt_rooms[gi].name, 'pred_room': '(未匹配)',
                'gt_normalized': normalize_room_name(gt_rooms[gi].name),
                'pred_normalized': '-', 'iou': 0.0, 'correct': False
            })
    for pj in range(n_pred_rooms):
        if pj not in pred_matched_set:
            result_a.room_matches.append({
                'gt_room': '(未匹配)', 'pred_room': pred_rooms[pj].name,
                'gt_normalized': '-',
                'pred_normalized': normalize_room_name(pred_rooms[pj].name),
                'iou': 0.0, 'correct': False
            })

    result_a.furniture_matches = match_furniture_global(gt_furniture_all, pred_furniture_all)
    result_a.compute_metrics()

    # ============================================================
    # 策略 B: 房间内对齐匹配
    # ============================================================
    result_b = EvaluationResult("B. 房间内对齐匹配 (Aligned)")
    result_b.n_gt_rooms = n_gt_rooms
    result_b.n_pred_rooms = n_pred_rooms
    result_b.n_matched_rooms = n_matched
    result_b.room_matches = result_a.room_matches  # 房间匹配结果共享
    result_b.n_correct_room_type = result_a.n_correct_room_type

    # 策略 B 中，家具的 n_gt 和 n_pred 仍为全量，但匹配只在已匹配房间内进行
    result_b.n_gt_furniture = len(gt_furniture_all)
    result_b.n_pred_furniture = len(pred_furniture_all)
    result_b.gt_furniture_areas = list(result_a.gt_furniture_areas)

    result_b.furniture_matches = match_furniture_per_room_aligned(
        gt_rooms, pred_rooms, room_match_map
    )
    result_b.compute_metrics()

    return result_a, result_b


# ============================================================
# 批量评估
# ============================================================

def find_file_pairs(pred_dir: str, gt_dir: str) -> List[Tuple[str, str]]:
    pairs = []
    pred_path = Path(pred_dir)
    gt_path = Path(gt_dir)

    pred_files = {}
    pred_pattern = re.compile(r'03_final_(\w+)\.ya?ml$')
    for f in sorted(pred_path.glob('03_final_*.yaml')):
        m = pred_pattern.match(f.name)
        if m:
            pred_files[m.group(1)] = str(f)
    for f in sorted(pred_path.glob('03_final_*.yml')):
        m = pred_pattern.match(f.name)
        if m:
            pred_files[m.group(1)] = str(f)

    gt_pattern = re.compile(r'layout_(\w+)\.ya?ml$')
    for f in sorted(gt_path.glob('layout_*.yaml')):
        m = gt_pattern.match(f.name)
        if m:
            code = m.group(1)
            if code in pred_files:
                pairs.append((pred_files[code], str(f)))
    for f in sorted(gt_path.glob('layout_*.yml')):
        m = gt_pattern.match(f.name)
        if m:
            code = m.group(1)
            if code in pred_files and (pred_files[code], str(f)) not in pairs:
                pairs.append((pred_files[code], str(f)))
    return pairs


def evaluate_batch(pred_dir: str, gt_dir: str, verbose: bool = True) -> dict:
    pairs = find_file_pairs(pred_dir, gt_dir)
    if not pairs:
        print(f"[错误] 在 {pred_dir} 和 {gt_dir} 中没有找到匹配的预测/GT文件对。")
        sys.exit(1)

    all_a: List[EvaluationResult] = []
    all_b: List[EvaluationResult] = []
    codes: List[str] = []

    for pred_path, gt_path in pairs:
        code_match = re.search(r'(\w+)\.ya?ml$', gt_path)
        code = code_match.group(1).replace('layout_', '') if code_match else Path(gt_path).stem
        codes.append(code)

        if verbose:
            print(f"\n{'='*60}")
            print(f"  评估家庭: {code}")
            print(f"{'='*60}")

        ra, rb = evaluate_single(pred_path, gt_path, verbose=False)
        all_a.append(ra)
        all_b.append(rb)

    def mean_std(vals):
        if len(vals) == 0:
            return 0, 0
        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        return mean, std

    print(f"\n{'='*90}")
    print(f"  批量汇总 (N = {len(all_a)} 个家庭)")
    print(f"{'='*90}")

    metrics_names = ['Room F1', 'Loc F1', 'Semantic F1', 'Cond. FNA', 'Legacy CLS']
    a_vals = [
        [r.room_f1 for r in all_a],
        [r.furniture_f1 for r in all_a],
        [r.semantic_f1 for r in all_a],
        [r.furniture_naming_accuracy for r in all_a],
        [r.cls for r in all_a],
    ]
    b_vals = [
        [r.room_f1 for r in all_b],
        [r.furniture_f1 for r in all_b],
        [r.semantic_f1 for r in all_b],
        [r.furniture_naming_accuracy for r in all_b],
        [r.cls for r in all_b],
    ]

    print("\n  预测生成阶段禁止读取 GT；以下 GT 只用于预测冻结后的独立评分。")
    print("  A 衡量绝对坐标，B 先匹配房间，再在每对房间内平移对齐后匹配家具。")
    print(f"\n  {'指标':<14} {'Global Mean ± Std':>22} {'Aligned Mean ± Std':>22}")
    print(f"  {'-'*58}")
    for i, name in enumerate(metrics_names):
        am, as_ = mean_std(a_vals[i])
        bm, bs = mean_std(b_vals[i])
        print(f"  {name:<12} {am:>9.1%} ±{as_:>6.1%}        {bm:>9.1%} ±{bs:>6.1%}")

    # 逐家庭
    print(f"\n  {'家庭':<8} {'策略':<8} {'RoomF1':>8} {'LocF1':>8} {'SemF1':>8} {'CondFNA':>9} {'LegacyCLS':>10}")
    print(f"  {'-'*72}")
    for idx, code in enumerate(codes):
        for result, tag in [(all_a[idx], 'Global'), (all_b[idx], 'Aligned')]:
            print(f"  {code:<8} {tag:<8} {result.room_f1:>7.1%} {result.furniture_f1:>7.1%} "
                  f"{result.semantic_f1:>7.1%} {result.furniture_naming_accuracy:>8.1%} "
                  f"{result.cls:>9.1%}")

    return {'results_a': all_a, 'results_b': all_b, 'codes': codes}


# ============================================================
# 详细报告输出
# ============================================================

def print_detailed_report(result_a: EvaluationResult, result_b: EvaluationResult):
    """输出单次评估的详细报告，并排对比两种策略"""
    OK = '[OK]'
    FAIL = '[X]'

    print(f"\n  {'='*90}")
    print(f"                         评 估 结 果 详 情")
    print(f"  {'='*90}")

    # --- 基本统计 ---
    print(f"\n  [基本统计]")
    print(f"    GT 房间数: {result_a.n_gt_rooms}    预测房间数: {result_a.n_pred_rooms}    匹配房间数: {result_a.n_matched_rooms}")
    print(f"    GT 家具数: {result_a.n_gt_furniture}    预测家具数: {result_a.n_pred_furniture}")

    # --- 房间匹配 ---
    print(f"\n  [房间匹配详情]")
    print(f"    {'GT房间':<20} {'预测房间':<20} {'IoU':>8}  {'类型正确'}")
    print(f"    {'-'*60}")
    for m in result_a.room_matches:
        flag = OK if m.get('correct', False) else FAIL
        print(f"    {m['gt_room']:<20} {m['pred_room']:<20} {m['iou']:>7.3f}  {flag}")

    # --- 家具匹配对比 ---
    print(f"\n  {'='*90}")
    print(f"  [家具匹配对比]  A. 全局匹配 (Global)  vs  B. 房间内对齐匹配 (Aligned)")
    print(f"  {'='*90}")

    T = EvaluationResult.FURNITURE_IOU_THRESHOLD
    tp_a = [m for m in result_a.furniture_matches if m.iou > T]
    tp_b = [m for m in result_b.furniture_matches if m.iou > T]

    print(f"\n  --- 策略 A: 全局匹配 (TP@IoU>{T}={len(tp_a)}, FN={result_a.n_gt_furniture - len(tp_a)}, FP={result_a.n_pred_furniture - len(tp_a)}) ---")
    print(f"    {'GT家具':<18} {'-> 预测家具':<18} {'IoU':>8}  {'GT房间':<14} {'命名正确'}")
    print(f"    {'-'*75}")
    for m in result_a.furniture_matches:
        flag = OK if (m.iou > T and m.name_correct) else (FAIL if m.iou > T else '-')
        gt_name = m.gt_name[:16]
        pred_name = m.pred_name[:16]
        gt_room = m.gt_room[:13]
        print(f"    {gt_name:<18} -> {pred_name:<16} {m.iou:>7.3f}  {gt_room:<14} {flag}")

    print(f"\n  --- 策略 B: 房间内对齐匹配 (TP@IoU>{T}={len(tp_b)}, FN={result_b.n_gt_furniture - len(tp_b)}, FP={result_b.n_pred_furniture - len(tp_b)}) ---")
    if result_b.furniture_matches:
        print(f"    {'GT家具':<18} {'-> 预测家具':<18} {'IoU':>8}  {'偏移(dx,dy)':<16} {'GT房间':<14} {'命名正确'}")
        print(f"    {'-'*85}")
        for m in result_b.furniture_matches:
            flag = OK if (m.iou > T and m.name_correct) else (FAIL if m.iou > T else '-')
            gt_name = m.gt_name[:16]
            pred_name = m.pred_name[:16]
            gt_room = m.gt_room[:13]
            offset_str = f"({m.offset_applied[0]:.1f},{m.offset_applied[1]:.1f})" if m.offset_applied != (0, 0) else "-"
            print(f"    {gt_name:<18} -> {pred_name:<16} {m.iou:>7.3f}  {offset_str:<16} {gt_room:<14} {flag}")
    else:
        print(f"    (无匹配房间对的家具)")

    # --- 核心指标对比 ---
    print(f"\n  {'='*90}")
    print(f"                        核 心 指 标 对 比")
    print(f"  {'='*90}")
    print("    注意: 预测生成阶段不读取 GT；GT 仅在预测冻结后进入此独立评分器。")
    print("    A 使用绝对坐标；B 先匹配房间，再在每对房间内平移对齐后匹配家具。")
    print(f"    {'指标':<35} {'A Global':>16} {'B Aligned':>16}")
    print(f"    {'-'*69}")
    print(f"    {'Room Type Accuracy (RTA)':<35} {result_a.room_type_accuracy:>15.1%} {result_b.room_type_accuracy:>15.1%}")
    print(f"    {'Room Semantic Precision':<35} {result_a.room_precision:>15.1%} {result_b.room_precision:>15.1%}")
    print(f"    {'Room Semantic Recall':<35} {result_a.room_recall:>15.1%} {result_b.room_recall:>15.1%}")
    print(f"    {'Room Semantic F1':<35} {result_a.room_f1:>15.1%} {result_b.room_f1:>15.1%}")
    print(f"    {'Furniture Precision':<35} {result_a.furniture_precision:>15.1%} {result_b.furniture_precision:>15.1%}")
    print(f"    {'Furniture Recall':<35} {result_a.furniture_recall:>15.1%} {result_b.furniture_recall:>15.1%}")
    print(f"    {'Furniture Detection F1':<35} {result_a.furniture_f1:>15.1%} {result_b.furniture_f1:>15.1%}")
    print(f"    {'Furniture Naming Accuracy (FNA)':<35} {result_a.furniture_naming_accuracy:>15.1%} {result_b.furniture_naming_accuracy:>15.1%}")
    print(f"    {'Semantic Precision (end-to-end)':<35} {result_a.semantic_precision:>15.1%} {result_b.semantic_precision:>15.1%}")
    print(f"    {'Semantic Recall (end-to-end)':<35} {result_a.semantic_recall:>15.1%} {result_b.semantic_recall:>15.1%}")
    print(f"    {'Semantic F1 (end-to-end)':<35} {result_a.semantic_f1:>15.1%} {result_b.semantic_f1:>15.1%}")
    print(f"    {'  (TP={result_a.tp_furniture}/{result_b.tp_furniture})':<35}")
    print(f"    {'-'*69}")
    print(f"    {'Legacy CLS (exploratory only)':<35} {result_a.cls:>15.1%} {result_b.cls:>15.1%}")
    print(f"  {'='*90}")

    # --- 按面积分层报告 (说明小家具受IoU偏移影响) ---
    print(f"\n  [按家具面积分层评估]  (说明: 小家具同等偏移量下 IoU 天然偏低)")
    print(f"    {'面积分层':<16} {'策略':<8} {'GT数':>5} {'TP':>5} {'召回率':>8} {'检出命名率':>10}")
    print(f"    {'-'*58}")
    size_order = ['small (<50)', 'medium (50-200)', 'large (>=200)']
    for size_cls in size_order:
        for result, tag in [(result_a, 'Global'), (result_b, 'Align ')]:
            pc = result.per_class.get(size_cls, {'gt': 0, 'tp': 0, 'tp_correct_name': 0})
            recall = pc['tp'] / pc['gt'] if pc['gt'] > 0 else 0
            fna_cls = pc['tp_correct_name'] / pc['tp'] if pc['tp'] > 0 else 0
            print(f"    {size_cls:<16} {tag:<8} {pc['gt']:>5} {pc['tp']:>5} {recall:>7.1%} {fna_cls:>9.1%}")

    # --- 质心距离指标 (尺寸无关，对小家具更公平) ---
    print(f"\n  [质心距离指标]  阈值: 质心偏移 ≤ {EvaluationResult.CENTROID_THRESHOLD}dm")
    print(f"    {'指标':<20} {'策略A (全局)':>16} {'策略B (对齐)':>16}")
    print(f"    {'-'*54}")
    print(f"    {'Centroid TP':<20} {result_a.centroid_tp:>16} {result_b.centroid_tp:>16}")
    print(f"    {'Centroid Recall':<20} {result_a.centroid_recall:>15.1%} {result_b.centroid_recall:>15.1%}")
    print(f"    {'Centroid F1':<20} {result_a.centroid_f1:>15.1%} {result_b.centroid_f1:>15.1%}")

    # 对齐策略中偏移量大于阈值的警告
    large_offsets = [(m.gt_room, m.offset_applied)
                     for m in result_b.furniture_matches
                     if abs(m.offset_applied[0]) > 2 or abs(m.offset_applied[1]) > 2]
    if large_offsets:
        print(f"\n  [注意] 以下房间存在较大坐标偏移 (>2 dm):")
        for room_name, (dx, dy) in set(large_offsets):
            print(f"    {room_name}: dx={dx:.1f}, dy={dy:.1f} dm")


# ============================================================
# 入口
# ============================================================

def main():
    if sys.platform == 'win32':
        os.environ['PYTHONIOENCODING'] = 'utf-8'

    parser = argparse.ArgumentParser(
        description='家庭布局推理系统评估脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单文件评估
  python evaluation.py --pred output/yaml/03_final_{code}.yaml --gt GT/layout_{code}.yaml

  # 批量评估 (按文件名编码自动匹配)
  python evaluation.py --pred_dir output/yaml/ --gt_dir GT/
  # 简洁输出
  python evaluation.py --pred output/yaml/03_final_{code}.yaml --gt GT/layout_{code}.yaml --quiet
"""
    )
    parser.add_argument('--pred', type=str, help='预测 YAML 文件路径')
    parser.add_argument('--gt', type=str, help='Ground Truth YAML 文件路径')
    parser.add_argument('--pred_dir', type=str, help='预测 YAML 文件目录 (批量模式)')
    parser.add_argument('--gt_dir', type=str, help='GT YAML 文件目录 (批量模式)')
    parser.add_argument('--code', type=str, default='0622',
                        help='家庭编码，用于默认路径拼接 (默认: 0622)')
    parser.add_argument('--quiet', '-q', action='store_true', help='简洁输出模式')

    parser.add_argument('--output', '-o', type=str, default=None,
                        help='将结果保存到文件 (同时仍输出到终端)')

    args = parser.parse_args()
    _main_impl(args)


def _main_impl(args):
    # --- 确定输出文件路径 ---
    output_path = args.output
    if not output_path:
        # 自动生成: 从 GT 文件名提取编码
        if args.pred_dir or args.gt_dir:
            output_path = None  # 批量模式暂不自动保存
        else:
            code = args.code
            pred_path = args.pred or f'output/{code}/yaml/03_final_{code}.yaml'
            gt_path = args.gt or f'GT/layout_{code}.yaml'
            out_dir = f'output/{code}/evaluation'
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, f'evaluation_{code}.txt')

    # Tee 模式
    output_file = None
    original_stdout = sys.stdout
    if output_path:
        output_file = open(output_path, 'w', encoding='utf-8')
        class Tee:
            def write(self, data):
                original_stdout.write(data)
                output_file.write(data)
            def flush(self):
                original_stdout.flush()
                output_file.flush()
        sys.stdout = Tee()

    try:
        _run_evaluation(args)
    finally:
        if output_file:
            sys.stdout = original_stdout
            output_file.close()
            print(f"\n[结果已保存至: {output_path}]")


def _run_evaluation(args):
    if args.pred_dir or args.gt_dir:
        pred_dir = args.pred_dir or 'output/yaml/'
        gt_dir = args.gt_dir or 'GT/'
        return evaluate_batch(pred_dir, gt_dir, verbose=not args.quiet)

    # 从 GT 路径提取 code，否则用 args.code
    gt_path = args.gt
    code = getattr(args, 'code', '0622')
    if gt_path:
        m = re.search(r'layout_(\w+)\.ya?ml$', gt_path)
        if m:
            code = m.group(1)
    
    pred_path = args.pred or (f'output/{code}/yaml/03_final_{code}.yaml' if code else None)
    gt_path = gt_path or (f'GT/layout_{code}.yaml' if code else None)
    
    if not pred_path or not os.path.exists(pred_path):
        print(f"[错误] 预测文件不存在: {pred_path}")
        sys.exit(1)
    if not gt_path or not os.path.exists(gt_path):
        print(f"[错误] GT文件不存在: {gt_path}")
        sys.exit(1)

    print(f"预测文件: {pred_path}")
    print(f"GT文件:   {gt_path}")

    result_a, result_b = evaluate_single(pred_path, gt_path)

    if not args.quiet:
        print_detailed_report(result_a, result_b)
    else:
        print("\n  预测生成阶段禁止读取 GT；GT 仅用于预测冻结后的独立评分。")
        print("  A 使用绝对坐标；B 先匹配房间，再在每对房间内平移对齐后匹配家具。")
        print(f"\n  {'策略':<12} {'RoomF1':>8} {'LocF1':>8} {'SemF1':>8} {'CondFNA':>9} {'LegacyCLS':>10}")
        print(f"  {'-'*65}")
        print(f"  {'Global':<12} {result_a.room_f1:>7.1%} {result_a.furniture_f1:>7.1%} "
              f"{result_a.semantic_f1:>7.1%} {result_a.furniture_naming_accuracy:>8.1%} "
              f"{result_a.cls:>9.1%}")
        print(f"  {'Aligned':<12} {result_b.room_f1:>7.1%} {result_b.furniture_f1:>7.1%} "
              f"{result_b.semantic_f1:>7.1%} {result_b.furniture_naming_accuracy:>8.1%} "
              f"{result_b.cls:>9.1%}")
        
        # 简洁版按面积分层
        print(f"\n  {'面积分层':<16} {'策略':<8} {'GT':>4} {'TP':>4} {'Recall':>8} {'检出命名率':>9}")
        print(f"  {'-'*53}")
        for size_cls in ['small (<50)', 'medium (50-200)', 'large (>=200)']:
            for result, tag in [(result_a, 'Global'), (result_b, 'Align ')]:
                pc = result.per_class.get(size_cls, {'gt': 0, 'tp': 0, 'tp_correct_name': 0})
                rec = pc['tp'] / pc['gt'] if pc['gt'] > 0 else 0
                fna = pc['tp_correct_name'] / pc['tp'] if pc['tp'] > 0 else 0
                print(f"  {size_cls:<16} {tag:<8} {pc['gt']:>4} {pc['tp']:>4} {rec:>7.1%} {fna:>9.1%}")


if __name__ == '__main__':
    main()
