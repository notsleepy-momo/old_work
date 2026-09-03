"""
CV-Only Baseline — 纯 CV + 规则，无 LLM 模块

生成 CV-only 预测用于消融/对比实验:
  - 房间类型: 纯规则推断 (智能设备映射 + 面积/连通性)
  - 家具命名: 仅智能设备保留名称，其余为 "unknown"
  - 不使用 RoomAgent / BehaviorAgent / FurnitureNamingAgent

输入: 02_furniture_world_{code}.yaml (CV 家具检测 + 坐标统一)
输出: cv_only_{code}.yaml (世界坐标, 规则推断的房间类型和家具名)

用法:
  python baselines/cv_only_baseline.py --code 0622
"""
import os, sys, yaml, argparse, re, copy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 智能设备 → 房间类型 映射 (priority >= 5 的明确设备)
DEVICE_ROOM_MAP = {
    'Refrigerator': 'kitchen', 'Fr': 'kitchen',
    'Stove': 'kitchen', 'Oven': 'kitchen', 'Microwave': 'kitchen',
    'Washing Machine': 'bathroom',
}

# 歧义设备 → 可能类型列表
AMBIGUOUS_DEVICES = {
    'TV': ['living_room', 'bedroom'],
    'TV Cabinet': ['living_room', 'bedroom'],
    'TV Stand': ['living_room', 'bedroom'],
}


def load_furniture_world(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def infer_room_types(data: dict) -> dict:
    """纯规则推断房间类型"""
    rooms = data.get('house', {}).get('rooms', [])
    room_types = {}  # room_id → type
    typed_rooms = set()

    # Pass 1: 智能设备明确映射
    for room in rooms:
        rid = room.get('id', '')
        furniture = room.get('furniture', [])
        for fur in furniture:
            name = fur.get('name', '')
            if name in DEVICE_ROOM_MAP:
                room_types[rid] = DEVICE_ROOM_MAP[name]
                typed_rooms.add(rid)
                break

    # Pass 2: 歧义设备 → 按面积/位置判断
    for room in rooms:
        rid = room.get('id', '')
        if rid in typed_rooms:
            continue
        pos = room.get('position', {})
        room_area = pos.get('width', 0) * pos.get('height', 0)
        furniture = room.get('furniture', [])
        for fur in furniture:
            name = fur.get('name', '')
            if name in AMBIGUOUS_DEVICES:
                # TV 设备 → 面积大的 → living_room, 小的 → bedroom
                if room_area > 500:
                    room_types[rid] = 'living_room'
                else:
                    room_types[rid] = 'bedroom'
                typed_rooms.add(rid)
                break

    # Pass 3: 剩余房间 — 按面积和连通性推断
    untyped = [(r, r.get('position', {}).get('width', 0) * r.get('position', {}).get('height', 0))
               for r in rooms if r.get('id', '') not in typed_rooms]
    untyped.sort(key=lambda x: x[1], reverse=True)

    needed = ['bedroom', 'living_room', 'kitchen', 'bathroom']
    assigned = set(room_types.values())

    # Fill missing types
    for room, area in untyped:
        rid = room.get('id', '')
        # Pick first needed type not yet assigned
        found = False
        for t in needed:
            if t not in assigned:
                room_types[rid] = t
                assigned.add(t)
                found = True
                break
        if not found:
            # All main types filled, use remaining rooms as studyroom/diningroom
            # Small rooms → bathroom, medium → studyroom, large → diningroom
            if area < 200:
                room_types[rid] = 'bathroom'
            elif area < 500:
                room_types[rid] = 'studyroom'
            else:
                room_types[rid] = 'diningroom'

    # Pass 4: 找最小房间 → balcony
    all_rooms_with_area = [(r.get('id', ''), r.get('position', {}).get('width', 0) * r.get('position', {}).get('height', 0))
                           for r in rooms]
    all_rooms_with_area.sort(key=lambda x: x[1])
    if all_rooms_with_area and all_rooms_with_area[0][1] < 300:
        smallest_id = all_rooms_with_area[0][0]
        # Only if not already typed as kitchen/bathroom (high confidence)
        if room_types.get(smallest_id, '') not in ('kitchen', 'bathroom'):
            room_types[smallest_id] = 'balcony'

    return room_types


def apply_cv_only(data: dict) -> dict:
    """应用 CV-only 规则: 房间类型 + 家具命名"""
    result = copy.deepcopy(data)
    room_types = infer_room_types(data)

    for room in result.get('house', {}).get('rooms', []):
        rid = room.get('id', '')
        new_type = room_types.get(rid, 'other')
        room['name'] = new_type

        # 家具: 智能设备保留名, 其余 "unknown" 后加类型名
        for fur in room.get('furniture', []):
            if 'name' not in fur or not fur['name']:
                fur['name'] = 'unknown'

    return result


def main():
    parser = argparse.ArgumentParser(description='CV-Only Baseline')
    parser.add_argument('--code', type=str, default='0622', help='家庭编码')
    parser.add_argument('--input', type=str, default=None,
                        help='02_furniture_world YAML (默认: output/yaml/02_furniture_world_{code}.yaml)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录')
    args = parser.parse_args()

    output_dir = args.output_dir or f'output/{args.code}/baselines'
    input_path = args.input or f'output/{args.code}/yaml/02_furniture_world_{args.code}.yaml'
    output_path = os.path.join(output_dir, f'cv_only_{args.code}.yaml')

    if not os.path.exists(input_path):
        print(f"[ERROR] 输入文件不存在: {input_path}")
        print(f"  请先运行 pipeline Step 1-2 生成家具检测中间文件")
        sys.exit(1)

    print(f"CV-Only Baseline — 家庭 {args.code}")
    print(f"  输入: {input_path}")
    print(f"  房间类型: 纯规则 (智能设备映射 + 面积/连通性)")
    print(f"  家具命名: 仅智能设备保留名, 其余 unknown")

    data = load_furniture_world(input_path)
    result = apply_cv_only(data)

    # 统计
    room_types = {}
    unknown_count = 0
    named_count = 0
    for room in result.get('house', {}).get('rooms', []):
        t = room.get('name', '?')
        room_types[t] = room_types.get(t, 0) + 1
        for fur in room.get('furniture', []):
            if fur.get('name', '') == 'unknown':
                unknown_count += 1
            else:
                named_count += 1

    print(f"  房间类型: {room_types}")
    print(f"  家具: {named_count} 命名 + {unknown_count} unknown")

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(result, f, default_flow_style=None,
                  allow_unicode=True, sort_keys=False, indent=2)
    print(f"  输出: {output_path}")


if __name__ == '__main__':
    main()
