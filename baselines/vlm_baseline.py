"""
VLM Baseline — 纯视觉语言模型对比实验

两种 baseline:
  VLM Zero-shot:   一次 Prompt 直接输出完整家庭布局 YAML
  VLM Chain-of-Thought: 分 3 步推理 (房间 → 家具位置 → 家具命名)

输入:
  - 扫地机器人轨迹图 (PNG, 三值灰度: 0=墙, 128=地板, 255=轨迹)
  - 智能设备 YAML (坐标 + 设备名)
  - 行为轨迹 JSON (停留点 + 时长)

输出:
  - 完整家庭布局 YAML (与 pipeline 输出格式一致)
  - 用于与 evaluation.py 做对比评分

用法:
  python baselines/vlm_baseline.py --image input/photo/room_0622.png \
      --devices input/Smart_device/smartDevice_0622.yaml \
      --trajectory input/traj/0622/0622-Engineer-all_trajectory.json
"""
import argparse
import base64
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import load_furniture_allowed
from llm_config import (
    DEFAULT_LLM_MODEL,
    require_shared_model,
    resolve_api_key,
    resolve_base_url,
)

# 与项目其他实验脚本一致，兼容 UTF-8/UTF-16 的 .env。
def _load_env() -> None:
    env_path = Path(PROJECT_ROOT) / '.env'
    if not env_path.exists():
        return

    raw = env_path.read_bytes()
    if raw.startswith((b'\xff\xfe', b'\xfe\xff')):
        encodings = ('utf-16',)
    elif raw.startswith(b'\xef\xbb\xbf'):
        encodings = ('utf-8-sig',)
    else:
        encodings = ('utf-8', 'utf-8-sig', 'utf-16')

    for encoding in encodings:
        try:
            text = raw.decode(encoding)
        except UnicodeError:
            continue
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)
        return

_load_env()

API_KEY = resolve_api_key(required=False)
BASE_URL = resolve_base_url()
DEFAULT_MODEL = DEFAULT_LLM_MODEL

ROOM_NAMES = {
    'bedroom', 'living_room', 'kitchen', 'bathroom', 'diningroom',
    'studyroom', 'balcony', 'other',
}
FURNITURE_NAMES = {
    'bed', 'wardrobe', 'nightstand', 'desk', 'chair', 'bookshelf', 'sofa',
    'coffee_table', 'dining_table', 'tv_stand', 'shoe_cabinet', 'coat_rack',
    'refrigerator', 'washing_machine', 'toilet', 'shower', 'washbasin',
    'kitchen_countertop', 'oven', 'stove', 'table', 'shelf',
}
for _configured_names in load_furniture_allowed().values():
    FURNITURE_NAMES.update(_normalised for _normalised in _configured_names)


class VlmBaselineError(RuntimeError):
    """A VLM response or input cannot produce a valid baseline prediction."""


# ============================================================
# VLM Zero-shot Prompt
# ============================================================
VLM_ZEROSHOT_SYSTEM = """You are a house layout reconstruction system. 
Given a robot vacuum trajectory map (a color floor plan image showing walls,
rooms, and robot cleaning trajectories) and smart device positions, output a 
complete YAML describing all rooms and furniture in this home.

CRITICAL RULES:
1. Output ONLY valid YAML, no markdown, no code fences, no explanations.
2. All coordinates are in world units (decimeters dm).
3. Each room must have: name, position{x, y, width, height}, walls, doors, furniture.
4. Use ONLY these furniture names: bed, wardrobe, nightstand, desk, chair,
   bookshelf, sofa, coffee_table, dining_table, tv_stand, shoe_cabinet,
   coat_rack, refrigerator, washing_machine, toilet, shower, washbasin,
   kitchen_countertop, oven, stove.
5. Each furniture must have: id, name, position{x, y, width, height}.
6. Room types: bedroom, living_room, kitchen, bathroom, diningroom,
   studyroom, balcony.
7. Smart devices provided below are known furniture — include them at
   their exact X/Y positions with their given names. Their width/height are
   not observed; estimate those dimensions from the image.
8. Estimate other furniture from the image — look for rectangular
   shadows/patterns on the floor.
9. The house outer boundary is defined by walls.
   Y=0 is the bottom of the image, Y increases upward."""


def make_zeroshot_prompt(devices_text: str, trajectory_summary: str) -> str:
    return f"""Reconstruct the full house layout from this image.

SMART DEVICES (known positions, include these exactly):
{devices_text}

BEHAVIOR TRAJECTORY SUMMARY:
{trajectory_summary}

Output a complete house layout YAML with all rooms and furniture.
Format:
house:
  size: {{x: <total_width>, y: <total_height>}}
  rooms:
  - name: <room_type>
    position: {{x: <x>, y: <y>, width: <w>, height: <h>}}
    walls: {{top: true, right: true, bottom: true, left: true}}
    doors:
    - {{id: doorX, x: <x>, y: <y>}}
    furniture:
    - id: <room>_furX
      name: <furniture_name>
      position: {{x: <x>, y: <y>, width: <w>, height: <h>}}

OUTPUT ONLY THE YAML, NO OTHER TEXT:"""


# ============================================================
# VLM CoT Prompts (3-step)
# ============================================================
VLM_COT_SYSTEM = """You are a house layout reconstruction system.
Given a robot vacuum trajectory map (a color floor plan image showing walls,
rooms, and robot cleaning trajectories) and smart device positions, analyze 
the home step by step.

CRITICAL RULES:
1. All coordinates MUST be in world units (decimeters dm), NEVER pixel values.
2. The image maps to a real house. Use the provided house dimensions and scale.
3. Output ONLY valid JSON as instructed in each step, no markdown, no explanations.
4. Y=0 is the bottom of the image, Y increases upward.
5. Smart devices are known furniture at given positions — use those coordinates as spatial anchors."""

VLM_COT_STEP1 = """STEP 1: Identify all rooms in the house.

Look at the floor plan image. Identify walls, rooms, and the robot's cleaning trajectories.
Use the smart devices below as spatial anchors — they have known world coordinates (dm).
Locate each device on the image, then estimate room boundaries relative to these anchor points.

- Identify each room's bounding box (x, y, width, height in dm, NOT pixels).
- Assign a room type: bedroom, living_room, kitchen, bathroom, diningroom, studyroom, balcony.
- Give every room a stable unique id such as room_1.
- Note door positions between rooms.
- Use the smart device positions to calibrate your coordinate estimates.

Output as JSON:
{{
  "house_size": {{"x": <total_width_dm>, "y": <total_height_dm>}},
  "rooms": [
    {{"id": "room_1", "name": "<type>",
      "position": {{"x": <x_dm>, "y": <y_dm>, "width": <w_dm>, "height": <h_dm>}},
      "doors": [{{"id": "door_1", "x": <x_dm>, "y": <y_dm>}}]}}
  ]
}}

Smart devices (known world coordinates in dm, use as spatial anchors):
{devices}

{trajectory}

OUTPUT ONLY VALID JSON:"""

VLM_COT_STEP2 = """STEP 2: Detect furniture in each room.

Now that rooms are identified, estimate furniture positions within each room.
- Refer to every room by its Step 1 room id. Do not redefine room geometry.
- Give every furniture item a stable id unique in the whole house.
- Look for rectangular dark areas on the floor (shadows of furniture).
- Use the trajectory path (white lines) — furniture often sits in areas
  the robot does NOT clean.
- Include smart devices at their exact specified X/Y positions and estimate
  their unknown width/height from the image.

Rooms from Step 1:
{rooms_json}

Smart devices:
{devices}

Output as JSON (add furniture to each room):
{{
  "rooms": [
    {{"room_id": "room_1", "furniture": [
      {{"id": "room_1_fur1", "name": "unknown",
       "position": {{"x": <x>, "y": <y>, "width": <w>, "height": <h>}}}}
    ]}}
  ]
}}

OUTPUT ONLY VALID JSON:"""

VLM_COT_STEP3 = """STEP 3: Name the furniture in each room.

Given the room type, furniture positions/sizes, and smart devices, assign
semantic furniture names. Use ONLY these names:
bed, wardrobe, nightstand, desk, chair, bookshelf, sofa, coffee_table,
dining_table, tv_stand, shoe_cabinet, coat_rack, refrigerator,
washing_machine, toilet, shower, washbasin, kitchen_countertop, oven, stove.

Rules:
- Smart devices keep their names (do not rename).
- Large rectangle in bedroom → bed.
- Medium rectangle in living_room → sofa or dining_table (by size).
- Small rectangles near desk → chair.
- Rectangle in bathroom → toilet, shower, or washbasin (by position/size).
- Rectangle in kitchen → kitchen_countertop, oven, or stove.

Furniture with positions:
{furniture_json}

Output only a mapping from furniture id to the assigned name. Do not output
rooms, positions, sizes, walls, or doors.

Output as JSON:
{{
  "furniture_names": [
    {{"id": "room_1_fur1", "name": "<assigned_name>"}}
  ]
}}

OUTPUT ONLY VALID JSON:"""


# ============================================================
# Helpers
# ============================================================
def _load_device_input(path: str) -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise VlmBaselineError(f"无法读取智能设备输入 {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise VlmBaselineError(f"智能设备输入必须是 YAML 对象: {path}")
    return data


def load_devices_text(path: str) -> str:
    """读取设备名列表，兼容逐户输入和旧的词表配置。"""
    data = _load_device_input(path)
    house_devices = data.get('house', {}).get('furniture', [])
    if house_devices:
        return '\n'.join(
            f"  - {item.get('name')}" for item in house_devices
            if isinstance(item, dict) and item.get('name')
        )
    return '\n'.join(
        f"  - {name}" for name in sorted(data.get('smart_devices', {}))
    )


def _load_house_size_from_devices(path: str) -> dict:
    """Read public house dimensions from the smart-device input."""
    data = _load_device_input(path)
    house = data.get('house', {})
    return house.get('size', {}) if isinstance(house, dict) else {}


def _load_known_devices(path: str) -> List[dict]:
    data = _load_device_input(path)
    house = data.get('house', {})
    raw_devices = house.get('furniture', []) if isinstance(house, dict) else []
    if raw_devices is None:
        return []
    if not isinstance(raw_devices, list):
        raise VlmBaselineError("house.furniture 必须是列表")

    devices = []
    for index, item in enumerate(raw_devices, 1):
        if not isinstance(item, dict):
            raise VlmBaselineError(f"智能设备 #{index} 必须是对象")
        name = str(item.get('name', '')).strip()
        position = item.get('position')
        if not name or not isinstance(position, dict):
            raise VlmBaselineError(f"智能设备 #{index} 缺少 name 或 position")
        x = _finite_number(position.get('x'), f"智能设备 {name}.x")
        y = _finite_number(position.get('y'), f"智能设备 {name}.y")
        devices.append({
            'id': str(item.get('id', '')).strip(),
            'name': name,
            'x': x,
            'y': y,
        })
    return devices


def load_devices_for_prompt(path: str, image_path: str = None) -> tuple:
    """读取逐户设备坐标和公开房屋尺寸，不访问测试 GT。"""
    device_input = _load_device_input(path)
    house = device_input.get('house', {})
    if not isinstance(house, dict):
        house = {}
    hsize = house.get('size', {}) or {}
    result = []
    house_size_info = ""

    if hsize:
        width = _finite_number(hsize.get('x'), 'house.size.x')
        height = _finite_number(hsize.get('y'), 'house.size.y')
        if width <= 0 or height <= 0:
            raise VlmBaselineError("智能设备输入中的房屋尺寸必须为正数")
        scale_hint = ""
        if image_path and os.path.exists(image_path):
            from PIL import Image
            try:
                with Image.open(image_path) as img:
                    pw, ph = img.size
                scale_hint = (
                    f"The image is {pw}×{ph} pixels.\n"
                    f"Image width {pw}px = house width {width}dm, "
                    f"so each pixel represents {width / pw:.3f}dm.\n"
                    f"Image height {ph}px = house height {height}dm, "
                    f"so each pixel represents {height / ph:.3f}dm.\n"
                )
            except OSError as exc:
                raise VlmBaselineError(f"无法读取户型图 {image_path}: {exc}") from exc

        house_size_info = (
            "COORDINATE SYSTEM: All output coordinates must be in world units (decimeters dm).\n"
            f"The image shows a house of {width:g} dm wide × {height:g} dm tall.\n"
            f"{scale_hint}"
            "Origin (0,0) is bottom-left. X increases right and Y increases up.\n"
            "Do not output pixel coordinates."
        )

    for device in _load_known_devices(path):
        result.append(
            f"  - {device['name']}: at ({device['x']:g}, {device['y']:g}) "
            f"[id={device['id'] or 'unspecified'}]; exact observed X/Y; "
            "width and height are unknown and must be estimated from the image"
        )

    if not result:
        for name in sorted(device_input.get('smart_devices', {})):
            result.append(f"  - {name} (position unavailable)")

    return '\n'.join(result) or "  (none)", house_size_info


def load_trajectory_summary(path: str, max_lines: int = 15) -> str:
    """从轨迹 JSON 提取摘要"""
    if not path or not os.path.exists(path):
        return "(no trajectory data)"
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            traj = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise VlmBaselineError(f"无法读取轨迹文件 {path}: {exc}") from exc
    
    if isinstance(traj, list):
        points = traj
    elif isinstance(traj, dict):
        points = traj.get('trajectory', traj.get('points', traj.get('data', [])))
    else:
        return "(unparseable trajectory)"
    
    if not points:
        return "(empty trajectory)"
    
    # 提取摘要
    lines = [f"Total trajectory points: {len(points)}"]
    
    # 停留点统计
    def _get_pos(p):
        cp = p.get('center_position', {})
        return p.get('x', cp.get('x', 0)), p.get('y', cp.get('y', 0))

    def _get_dur_seconds(p):
        raw = p.get('duration', p.get('stay_duration', 0))
        try:
            raw = float(raw)
        except (TypeError, ValueError):
            return 0.0
        try:
            start = datetime.fromisoformat(str(p.get('start_time', '')))
            end = datetime.fromisoformat(str(p.get('end_time', '')))
            return max(0.0, (end - start).total_seconds())
        except (TypeError, ValueError):
            unit = str(p.get('duration_unit', '')).strip().lower()
            return raw if unit in {'s', 'sec', 'second', 'seconds'} else raw * 60.0

    stay_points = [p for p in points if isinstance(p, dict) and _get_dur_seconds(p) > 60]
    if stay_points:
        stay_points.sort(key=_get_dur_seconds, reverse=True)
        lines.append(f"\nTop {min(5, len(stay_points))} stay points (duration > 1min):")
        for sp in stay_points[:5]:
            x, y = _get_pos(sp)
            dur = _get_dur_seconds(sp)
            lines.append(f"  ({x:.1f}, {y:.1f}) - {dur:.0f}s")
    
    return '\n'.join(lines[:max_lines])


def image_to_base64(image_path: str, max_size: int = 512) -> str:
    """Convert image to base64, resize to max_size on longest side"""
    from PIL import Image
    import io
    try:
        with Image.open(image_path) as source:
            img = source.convert('RGB')
            w, h = img.size
            if max(w, h) > max_size:
                scale = max_size / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=85)
    except OSError as exc:
        raise VlmBaselineError(f"无法编码户型图 {image_path}: {exc}") from exc
    return base64.b64encode(buf.getvalue()).decode()


def _make_openai_client(api_key: str = None, base_url: str = None,
                        timeout: int = 600):
    from openai import OpenAI

    resolved_key = resolve_api_key(api_key, required=False) or API_KEY
    if not resolved_key:
        raise VlmBaselineError("未设置 FURNITURE_API_KEY，无法调用 VLM")
    kwargs = {'api_key': resolved_key, 'timeout': timeout}
    resolved_url = resolve_base_url(base_url) or BASE_URL
    if resolved_url:
        kwargs['base_url'] = resolved_url
    return OpenAI(**kwargs)


def _response_content(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise VlmBaselineError("VLM 返回了不完整的响应对象") from exc
    if not isinstance(content, str) or not content.strip():
        raise VlmBaselineError("VLM 返回了空文本")
    return content


def call_vlm(system_prompt: str, user_prompt: str, image_path: str,
             model: str = DEFAULT_MODEL, temperature: float = 0.2,
             api_key: str = None, base_url: str = None) -> str:
    """调用 VLM，发送图片 + 文本"""
    client = _make_openai_client(api_key, base_url)

    img_b64 = image_to_base64(image_path)
    
    response = client.chat.completions.create(
        model=require_shared_model(model),
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}",
                    "detail": "high"
                }}
            ]}
        ],
        max_tokens=4096,
    )
    
    return _response_content(response)


def call_vlm_text_only(system_prompt: str, user_prompt: str,
                       model: str = DEFAULT_MODEL, temperature: float = 0.2,
                       api_key: str = None, base_url: str = None) -> str:
    """纯文本 LLM 调用"""
    client = _make_openai_client(api_key, base_url)

    response = client.chat.completions.create(
        model=require_shared_model(model),
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4096,
    )
    
    return _response_content(response)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith('```'):
        return text
    lines = text.splitlines()
    if lines:
        lines = lines[1:]
    if lines and lines[-1].strip().startswith('```'):
        lines = lines[:-1]
    return '\n'.join(lines).strip()


def extract_yaml(text: str) -> str:
    """从 VLM 输出中提取 YAML 内容（去除 markdown 代码块等）"""
    return _strip_code_fence(text)


def extract_json(text: str) -> str:
    """从 VLM 输出中提取 JSON 内容"""
    text = _strip_code_fence(text)
    # 找到第一个 { 到最后一个 }
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise VlmBaselineError(f"{label} 必须是有限数字")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise VlmBaselineError(f"{label} 必须是有限数字") from exc
    if not math.isfinite(number):
        raise VlmBaselineError(f"{label} 必须是有限数字")
    return number


def _normalise_house_size(raw_size: Any) -> dict:
    if not isinstance(raw_size, dict):
        raise VlmBaselineError("house.size 必须是对象")
    width = _finite_number(raw_size.get('x'), 'house.size.x')
    height = _finite_number(raw_size.get('y'), 'house.size.y')
    if width <= 0 or height <= 0:
        raise VlmBaselineError("house.size 的 x/y 必须为正数")
    return {'x': width, 'y': height}


def _normalise_key(value: Any) -> str:
    key = str(value or '').strip().lower().replace('-', '_').replace(' ', '_')
    return re.sub(r'_+', '_', key)


ROOM_ALIASES = {
    'livingroom': 'living_room',
    'living': 'living_room',
    'dining_room': 'diningroom',
    'study_room': 'studyroom',
    'study': 'studyroom',
    'bath_room': 'bathroom',
}
FURNITURE_ALIASES = {
    'coffee_table': 'coffee_table',
    'dining_table': 'dining_table',
    'tv_cabinet': 'tv_stand',
    'tvstand': 'tv_stand',
    'shoe_cabinet': 'shoe_cabinet',
    'washingmachine': 'washing_machine',
    'wash_basin': 'washbasin',
    'book_shelf': 'bookshelf',
    'night_stand': 'nightstand',
    'closet': 'wardrobe',
}


def _normalise_room_name(value: Any) -> str:
    name = ROOM_ALIASES.get(_normalise_key(value), _normalise_key(value))
    if name not in ROOM_NAMES:
        raise VlmBaselineError(f"不允许的房间名称: {value!r}")
    return name


def _normalise_furniture_name(value: Any, room_name: str,
                              known_device_names: Dict[str, str]) -> str:
    key = _normalise_key(value)
    if key in known_device_names:
        return known_device_names[key]
    key = FURNITURE_ALIASES.get(key, key)
    if key == 'sink':
        key = 'washbasin' if room_name == 'bathroom' else 'kitchen_countertop'
    elif key in {'cabinet', 'cupboard'}:
        key = {
            'kitchen': 'kitchen_countertop',
            'bedroom': 'wardrobe',
            'living_room': 'shoe_cabinet',
        }.get(room_name, 'shelf')
    if key not in FURNITURE_NAMES:
        raise VlmBaselineError(f"不允许的家具名称: {value!r}")
    return key


def _bounded_bbox(raw_position: Any, label: str, house_size: dict) -> dict:
    if not isinstance(raw_position, dict):
        raise VlmBaselineError(f"{label} 必须是对象")
    x = _finite_number(raw_position.get('x'), f'{label}.x')
    y = _finite_number(raw_position.get('y'), f'{label}.y')
    width = _finite_number(raw_position.get('width'), f'{label}.width')
    height = _finite_number(raw_position.get('height'), f'{label}.height')
    max_x, max_y = house_size['x'], house_size['y']
    if width <= 0 or height <= 0:
        raise VlmBaselineError(f"{label} 的 width/height 必须为正数")

    tolerance_x = max_x * 0.05
    tolerance_y = max_y * 0.05
    if x < -tolerance_x or y < -tolerance_y or x >= max_x or y >= max_y:
        raise VlmBaselineError(f"{label} 的起点超出合理房屋边界")
    if width > max_x * 2 or height > max_y * 2:
        raise VlmBaselineError(f"{label} 的尺寸超出合理房屋边界")

    x = max(0.0, x)
    y = max(0.0, y)
    width = min(width, max_x - x)
    height = min(height, max_y - y)
    if width <= 0 or height <= 0:
        raise VlmBaselineError(f"{label} 裁剪后没有有效面积")
    return {'x': x, 'y': y, 'width': width, 'height': height}


def _normalise_doors(raw_doors: Any, room_id: str, house_size: dict) -> List[dict]:
    if raw_doors is None:
        return []
    if not isinstance(raw_doors, list):
        raise VlmBaselineError(f"{room_id}.doors 必须是列表")
    doors = []
    for index, raw_door in enumerate(raw_doors, 1):
        if not isinstance(raw_door, dict):
            raise VlmBaselineError(f"{room_id}.doors[{index}] 必须是对象")
        x = _finite_number(raw_door.get('x'), f'{room_id}.doors[{index}].x')
        y = _finite_number(raw_door.get('y'), f'{room_id}.doors[{index}].y')
        if not (0 <= x <= house_size['x'] and 0 <= y <= house_size['y']):
            raise VlmBaselineError(f"{room_id}.doors[{index}] 超出房屋边界")
        doors.append({
            'id': str(raw_door.get('id') or f'{room_id}_door{index}'),
            'x': x,
            'y': y,
        })
    return doors


def _normalise_room_shell(raw_room: Any, index: int, house_size: dict) -> dict:
    if not isinstance(raw_room, dict):
        raise VlmBaselineError(f"rooms[{index}] 必须是对象")
    room_id = str(raw_room.get('id') or raw_room.get('room_id') or f'room_{index}').strip()
    if not room_id:
        raise VlmBaselineError(f"rooms[{index}] 缺少有效 id")
    walls = raw_room.get('walls')
    if not isinstance(walls, dict):
        walls = {'top': True, 'right': True, 'bottom': True, 'left': True}
    return {
        'id': room_id,
        'name': _normalise_room_name(raw_room.get('name')),
        'position': _bounded_bbox(raw_room.get('position'), f'{room_id}.position', house_size),
        'walls': {
            side: bool(walls.get(side, True))
            for side in ('top', 'right', 'bottom', 'left')
        },
        'doors': _normalise_doors(raw_room.get('doors', []), room_id, house_size),
        'furniture': [],
    }


def validate_layout(layout: Any, public_house_size: Optional[dict] = None,
                    known_devices: Optional[List[dict]] = None) -> dict:
    """Validate and canonicalize a prediction using only non-GT inputs."""
    if not isinstance(layout, dict) or not isinstance(layout.get('house'), dict):
        raise VlmBaselineError("预测顶层必须包含 house 对象")
    raw_house = layout['house']
    size_source = public_house_size or raw_house.get('size')
    house_size = _normalise_house_size(size_source)
    raw_rooms = raw_house.get('rooms')
    if not isinstance(raw_rooms, list) or not raw_rooms:
        raise VlmBaselineError("house.rooms 必须是非空列表")

    devices = known_devices or []
    known_names = {_normalise_key(item['name']): item['name'] for item in devices}
    clean_rooms = []
    room_ids = set()
    furniture_ids = set()
    for room_index, raw_room in enumerate(raw_rooms, 1):
        room = _normalise_room_shell(raw_room, room_index, house_size)
        if room['id'] in room_ids:
            raise VlmBaselineError(f"重复房间 id: {room['id']}")
        room_ids.add(room['id'])
        raw_furniture = raw_room.get('furniture', [])
        if not isinstance(raw_furniture, list):
            raise VlmBaselineError(f"{room['id']}.furniture 必须是列表")
        for furniture_index, raw_item in enumerate(raw_furniture, 1):
            if not isinstance(raw_item, dict):
                raise VlmBaselineError(
                    f"{room['id']}.furniture[{furniture_index}] 必须是对象")
            item_id = str(
                raw_item.get('id') or f"{room['id']}_fur{furniture_index}").strip()
            if not item_id or item_id in furniture_ids:
                raise VlmBaselineError(f"无效或重复家具 id: {item_id!r}")
            furniture_ids.add(item_id)
            room['furniture'].append({
                'id': item_id,
                'name': _normalise_furniture_name(
                    raw_item.get('name'), room['name'], known_names),
                'position': _bounded_bbox(
                    raw_item.get('position'), f'{item_id}.position', house_size),
            })
        clean_rooms.append(room)

    used_items = set()
    for device in devices:
        key = _normalise_key(device['name'])
        matches = []
        for room in clean_rooms:
            for item in room['furniture']:
                if id(item) in used_items:
                    continue
                if item['id'] == device['id'] or _normalise_key(item['name']) == key:
                    matches.append(item)
        if not matches:
            raise VlmBaselineError(f"预测缺少已知智能设备: {device['name']}")
        exact_id = [item for item in matches if device['id'] and item['id'] == device['id']]
        target = exact_id[0] if exact_id else matches[0]
        used_items.add(id(target))
        target['name'] = device['name']
        observed_position = dict(target['position'])
        observed_position.update({'x': device['x'], 'y': device['y']})
        target['position'] = _bounded_bbox(
            observed_position, f"智能设备 {device['name']}.position", house_size)

    return {'house': {'size': house_size, 'rooms': clean_rooms}}


def _parse_yaml_layout(text: str) -> dict:
    try:
        parsed = yaml.safe_load(extract_yaml(text))
    except yaml.YAMLError as exc:
        raise VlmBaselineError(f"VLM 输出不是有效 YAML: {exc}") from exc
    if not isinstance(parsed, dict):
        raise VlmBaselineError("VLM YAML 输出必须是对象")
    return parsed


def _parse_json_object(text: str, stage: str) -> dict:
    try:
        parsed = json.loads(extract_json(text))
    except json.JSONDecodeError as exc:
        raise VlmBaselineError(f"{stage} 输出不是有效 JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise VlmBaselineError(f"{stage} JSON 顶层必须是对象")
    return parsed


def _dump_layout(layout: dict) -> str:
    return yaml.safe_dump(
        layout, allow_unicode=True, sort_keys=False, default_flow_style=None)


def _atomic_save_layout(layout: dict, output_path: str) -> str:
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', newline='\n', delete=False,
                dir=str(destination.parent), prefix=f'.{destination.name}.',
                suffix='.tmp') as temporary:
            temporary_path = Path(temporary.name)
            yaml.safe_dump(
                layout, temporary, allow_unicode=True, sort_keys=False,
                default_flow_style=None)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
    return str(destination)


def save_yaml(yaml_text: str, output_path: str,
              public_house_size: Optional[dict] = None,
              known_devices: Optional[List[dict]] = None) -> str:
    """Validate first, then atomically replace the prediction file."""
    layout = validate_layout(
        _parse_yaml_layout(yaml_text), public_house_size, known_devices)
    return _atomic_save_layout(layout, output_path)


def _validate_input_paths(image_path: str, devices_path: str) -> None:
    if not image_path or not os.path.isfile(image_path):
        raise VlmBaselineError(f"户型图不存在: {image_path}")
    if not devices_path or not os.path.isfile(devices_path):
        raise VlmBaselineError(f"智能设备输入不存在: {devices_path}")


def _call_stage(label: str, function, *args, **kwargs) -> str:
    try:
        return function(*args, **kwargs)
    except VlmBaselineError:
        raise
    except Exception as exc:
        raise VlmBaselineError(f"{label} 调用失败: {exc}") from exc


def _normalise_step1(data: dict, public_house_size: dict) -> Tuple[dict, List[dict]]:
    size_source = public_house_size or data.get('house_size')
    house_size = _normalise_house_size(size_source)
    raw_rooms = data.get('rooms')
    if not isinstance(raw_rooms, list) or not raw_rooms:
        raise VlmBaselineError("CoT Step 1 必须返回至少一个房间")
    rooms = []
    room_ids = set()
    for index, raw_room in enumerate(raw_rooms, 1):
        room = _normalise_room_shell(raw_room, index, house_size)
        if room['id'] in room_ids:
            raise VlmBaselineError(f"CoT Step 1 返回重复房间 id: {room['id']}")
        room_ids.add(room['id'])
        rooms.append(room)
    return house_size, rooms


def _merge_step2_furniture(data: dict, rooms: List[dict],
                           house_size: dict) -> List[dict]:
    raw_rooms = data.get('rooms')
    if not isinstance(raw_rooms, list):
        raise VlmBaselineError("CoT Step 2 的 rooms 必须是列表")
    merged = [{**room, 'furniture': []} for room in rooms]
    by_id = {room['id']: room for room in merged}
    assigned_rooms = set()
    furniture_ids = set()

    for index, raw_room in enumerate(raw_rooms):
        if not isinstance(raw_room, dict):
            raise VlmBaselineError(f"CoT Step 2 rooms[{index}] 必须是对象")
        room_id = str(raw_room.get('room_id') or raw_room.get('id') or '').strip()
        if not room_id and len(raw_rooms) == len(rooms):
            room_id = rooms[index]['id']
        if room_id not in by_id:
            raise VlmBaselineError(f"CoT Step 2 引用了未知房间 id: {room_id!r}")
        if room_id in assigned_rooms:
            raise VlmBaselineError(f"CoT Step 2 重复返回房间: {room_id}")
        assigned_rooms.add(room_id)
        raw_furniture = raw_room.get('furniture', [])
        if not isinstance(raw_furniture, list):
            raise VlmBaselineError(f"CoT Step 2 {room_id}.furniture 必须是列表")
        for item_index, raw_item in enumerate(raw_furniture, 1):
            if not isinstance(raw_item, dict):
                raise VlmBaselineError(
                    f"CoT Step 2 {room_id}.furniture[{item_index}] 必须是对象")
            item_id = str(raw_item.get('id') or f'{room_id}_fur{item_index}').strip()
            if not item_id or item_id in furniture_ids:
                raise VlmBaselineError(f"CoT Step 2 返回无效或重复家具 id: {item_id!r}")
            furniture_ids.add(item_id)
            by_id[room_id]['furniture'].append({
                'id': item_id,
                'name': str(raw_item.get('name', 'unknown')).strip() or 'unknown',
                'position': _bounded_bbox(
                    raw_item.get('position'), f'{item_id}.position', house_size),
            })
    return merged


def _extract_step3_names(data: dict) -> Dict[str, str]:
    raw_names = data.get('furniture_names')
    if isinstance(raw_names, dict):
        entries = [{'id': key, 'name': value} for key, value in raw_names.items()]
    elif isinstance(raw_names, list):
        entries = raw_names
    else:
        # Accept the old nested response shape, but never consume its geometry.
        entries = []
        for room in data.get('rooms', []) if isinstance(data.get('rooms'), list) else []:
            if isinstance(room, dict) and isinstance(room.get('furniture'), list):
                entries.extend(room['furniture'])

    result = {}
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            raise VlmBaselineError(f"CoT Step 3 furniture_names[{index}] 必须是对象")
        item_id = str(entry.get('id', '')).strip()
        name = str(entry.get('name', '')).strip()
        if not item_id or not name:
            raise VlmBaselineError("CoT Step 3 的每个命名必须包含 id 和 name")
        if item_id in result and result[item_id] != name:
            raise VlmBaselineError(f"CoT Step 3 对 {item_id} 返回了冲突名称")
        result[item_id] = name
    return result


def _build_cot_layout(house_size: dict, rooms: List[dict],
                      named_data: dict, known_devices: List[dict]) -> dict:
    name_map = _extract_step3_names(named_data)
    known_by_name = {
        _normalise_key(item['name']): item['name'] for item in known_devices
    }
    known_by_id = {
        item['id']: item['name'] for item in known_devices if item['id']
    }

    expected_ids = {
        item['id'] for room in rooms for item in room.get('furniture', [])
    }
    unknown_ids = set(name_map) - expected_ids
    if unknown_ids:
        raise VlmBaselineError(
            f"CoT Step 3 返回未知家具 id: {sorted(unknown_ids)[0]}")

    layout_rooms = []
    for room in rooms:
        output_room = {key: value for key, value in room.items() if key != 'furniture'}
        output_room['furniture'] = []
        for item in room.get('furniture', []):
            current_key = _normalise_key(item.get('name'))
            if item['id'] in known_by_id:
                assigned_name = known_by_id[item['id']]
            elif current_key in known_by_name:
                assigned_name = known_by_name[current_key]
            else:
                assigned_name = name_map.get(item['id'])
            if not assigned_name or _normalise_key(assigned_name) in {'unknown', 'none'}:
                raise VlmBaselineError(f"CoT Step 3 未命名家具: {item['id']}")
            output_room['furniture'].append({
                'id': item['id'],
                'name': assigned_name,
                'position': item['position'],
            })
        layout_rooms.append(output_room)

    return validate_layout(
        {'house': {'size': house_size, 'rooms': layout_rooms}},
        public_house_size=house_size,
        known_devices=known_devices,
    )


# ============================================================
# VLM Zero-shot
# ============================================================
def run_zeroshot(image_path: str, devices_path: str, trajectory_path: str,
                 output_path: str = None,
                 model: str = DEFAULT_MODEL, api_key: str = None,
                 base_url: str = None) -> dict:
    """VLM Zero-shot: 一次调用直接输出完整 YAML"""
    print(f"\n{'='*60}")
    print(f"  VLM Zero-shot Baseline")
    print(f"  Model: {model}")
    print(f"{'='*60}")
    
    _validate_input_paths(image_path, devices_path)
    public_house_size = _load_house_size_from_devices(devices_path)
    known_devices = _load_known_devices(devices_path)
    devices_text, house_size_info = load_devices_for_prompt(devices_path, image_path)
    trajectory_summary = load_trajectory_summary(trajectory_path)
    
    # 把房屋尺寸信息注入 prompt
    if house_size_info:
        trajectory_summary = house_size_info + "\n" + trajectory_summary
    
    prompt = make_zeroshot_prompt(devices_text, trajectory_summary)
    
    print("  Calling VLM...")
    raw_output = _call_stage(
        'Zero-shot', call_vlm, VLM_ZEROSHOT_SYSTEM, prompt, image_path,
        model=model, api_key=api_key, base_url=base_url)
    print(f"  VLM response length: {len(raw_output)} chars")
    layout = validate_layout(
        _parse_yaml_layout(raw_output), public_house_size, known_devices)
    yaml_text = _dump_layout(layout)
    
    if output_path:
        _atomic_save_layout(layout, output_path)
        print(f"  Output saved: {output_path}")
    
    return {
        "method": "vlm_zeroshot", "model": model,
        "yaml_text": yaml_text, "layout": layout,
    }


# ============================================================
# VLM Chain-of-Thought (3-step)
# ============================================================
def run_cot(image_path: str, devices_path: str, trajectory_path: str,
            output_path: str = None,
            model: str = DEFAULT_MODEL, api_key: str = None,
            base_url: str = None) -> dict:
    """VLM CoT: 3 步推理"""
    print(f"\n{'='*60}")
    print(f"  VLM Chain-of-Thought Baseline")
    print(f"  Model: {model}")
    print(f"{'='*60}")
    
    _validate_input_paths(image_path, devices_path)
    public_house_size = _load_house_size_from_devices(devices_path)
    known_devices = _load_known_devices(devices_path)
    devices_text, house_size_info = load_devices_for_prompt(devices_path, image_path)
    trajectory_summary = load_trajectory_summary(trajectory_path)
    
    # 把房屋尺寸信息注入 trajectory 摘要（Step 1 prompt 使用 trajectory 变量）
    if house_size_info:
        trajectory_summary = house_size_info + "\n" + trajectory_summary
    
    print("  Step 1/3: Room detection...")
    step1_prompt = VLM_COT_STEP1.format(devices=devices_text, trajectory=trajectory_summary)
    step1_raw = _call_stage(
        'CoT Step 1', call_vlm, VLM_COT_SYSTEM, step1_prompt, image_path,
        model=model, api_key=api_key, base_url=base_url)
    step1_data = _parse_json_object(step1_raw, 'CoT Step 1')
    house_size, rooms = _normalise_step1(step1_data, public_house_size)
    print(f"  Step 1 done: {len(rooms)} rooms detected")
    
    # Step 2: 家具位置
    print("  Step 2/3: Furniture position detection...")
    step2_prompt = VLM_COT_STEP2.format(
        rooms_json=json.dumps(
            {'house_size': house_size, 'rooms': rooms},
            indent=2, ensure_ascii=False),
        devices=devices_text
    )
    step2_raw = _call_stage(
        'CoT Step 2', call_vlm, VLM_COT_SYSTEM, step2_prompt, image_path,
        model=model, api_key=api_key, base_url=base_url)
    furniture_data = _parse_json_object(step2_raw, 'CoT Step 2')
    rooms_with_furniture = _merge_step2_furniture(
        furniture_data, rooms, house_size)
    total_furniture = sum(len(room['furniture']) for room in rooms_with_furniture)
    print(f"  Step 2 done: {total_furniture} furniture candidates")
    
    # Step 3: 家具命名 (纯文本，不需要图片)
    print("  Step 3/3: Furniture naming...")
    step3_prompt = VLM_COT_STEP3.format(
        furniture_json=json.dumps(
            {'rooms': rooms_with_furniture}, indent=2, ensure_ascii=False)
    )
    step3_raw = _call_stage(
        'CoT Step 3', call_vlm_text_only, VLM_COT_SYSTEM, step3_prompt,
        model=model, api_key=api_key, base_url=base_url)
    named_data = _parse_json_object(step3_raw, 'CoT Step 3')
    layout = _build_cot_layout(
        house_size, rooms_with_furniture, named_data, known_devices)
    yaml_text = _dump_layout(layout)
    
    if output_path:
        _atomic_save_layout(layout, output_path)
        print(f"  Output saved: {output_path}")
    
    print(f"  Step 3 done")
    return {
        "method": "vlm_cot", "model": model,
        "yaml_text": yaml_text, "layout": layout,
    }


# ============================================================
# Main
# ============================================================
def main() -> int:
    parser = argparse.ArgumentParser(description='VLM Baseline — 纯视觉语言模型对比实验')
    parser.add_argument('--image', type=str, default='input/photo/room_0622.png',
                        help='户型图路径')
    parser.add_argument('--devices', type=str, default='input/Smart_device/smartDevice_0622.yaml',
                        help='当前家庭的智能设备输入（含坐标和房屋尺寸）')
    parser.add_argument('--trajectory', type=str,
                        default='input/traj/0622/merge_trajectory.json',
                        help='轨迹 JSON 路径')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--method', type=str, choices=['zeroshot', 'cot', 'both'],
                        default='both', help='Baseline 方法')
    parser.add_argument('--code', type=str, default=None,
                        help='输出文件中的家庭编码；默认从图片文件名提取')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        choices=[DEFAULT_MODEL],
                        help='模型名称')
    parser.add_argument('--api-key', type=str, default=None)
    parser.add_argument('--base-url', type=str, default=None)

    args = parser.parse_args()

    # 提取编码（如 0622）
    code_match = re.search(r'(\d{4})', os.path.basename(args.image))
    code = args.code or (code_match.group(1) if code_match else 'unknown')

    output_dir = args.output_dir or f'output/{code}/baselines'

    api_key = args.api_key or API_KEY
    base_url = args.base_url or BASE_URL

    try:
        if args.method in ('zeroshot', 'both'):
            out_path = os.path.join(output_dir, f'vlm_zeroshot_{code}.yaml')
            run_zeroshot(
                args.image, args.devices, args.trajectory,
                output_path=out_path, model=args.model,
                api_key=api_key, base_url=base_url)

        if args.method in ('cot', 'both'):
            out_path = os.path.join(output_dir, f'vlm_cot_{code}.yaml')
            run_cot(
                args.image, args.devices, args.trajectory,
                output_path=out_path, model=args.model,
                api_key=api_key, base_url=base_url)
    except VlmBaselineError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
