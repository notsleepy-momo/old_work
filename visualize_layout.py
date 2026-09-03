"""
户型图 YAML 可视化脚本 — 完全匹配 Figure_1.png 风格
基于 photo2yaml/draw/floorplan_renderer.py 渲染逻辑

用法:
  python visualize_layout.py --code 0622
  python visualize_layout.py --input output/yaml/05_final_layout.yaml --output output/masks/floorplan.png
"""
import sys, os, argparse, yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ── 配色（匹配 Figure_1.png 风格：温暖奶油色调，优雅简约）──
ROOM_COLORS = [
    '#FDFBEB', '#F7F3E3', '#F5F1E1', '#FBF9EA', '#FEFDF5',
    '#FBF9F1', '#F6F3CA', '#F5F2CA', '#F5F2E2', '#F4F0E0',
    '#FCFAEA', '#FDFBEC', '#F7F4E4', '#F8F5E6', '#F6F3D5',
]

WALL_COLOR = '#999999'
WALL_WIDTH = 2.5
DOOR_COLOR = '#DEB78A'
DOOR_FILL = '#F5E6D3'
DOOR_ALPHA = 0.35
DOOR_MARKER_SIZE = 2

FURNITURE_COLOR = '#D8D8D8'
FURNITURE_EDGE = '#BBBBBB'
FURNITURE_ALPHA = 0.5
FURNITURE_FONT_SIZE = 5
FURNITURE_HATCH = ''


def load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def draw_wall(ax, x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color=WALL_COLOR, linewidth=WALL_WIDTH,
            solid_capstyle='round', zorder=3)


def draw_boolean_walls(ax, x, y, w, h, walls):
    if walls.get('top', False):
        draw_wall(ax, x, y + h, x + w, y + h)
    if walls.get('right', False):
        draw_wall(ax, x + w, y, x + w, y + h)
    if walls.get('bottom', False):
        draw_wall(ax, x, y, x + w, y)
    if walls.get('left', False):
        draw_wall(ax, x, y, x, y + h)


def draw_door_at_point(ax, door_x, door_y, room_bbox):
    """在门洞中心点画门符号，自动推断朝向（与 floorplan_renderer.py 一致）"""
    rx, ry, rw, rh = room_bbox
    dists = {
        'top': abs(door_y - (ry + rh)),
        'bottom': abs(door_y - ry),
        'left': abs(door_x - rx),
        'right': abs(door_x - (rx + rw)),
    }
    wall_side = min(dists, key=dists.get)
    door_len = DOOR_MARKER_SIZE * 2
    gap = door_len * 0.5

    if wall_side == 'top':
        p1 = np.array([door_x - gap, door_y])
        p2 = np.array([door_x + gap, door_y])
        swing_dir = np.array([0, -door_len])
    elif wall_side == 'bottom':
        p1 = np.array([door_x - gap, door_y])
        p2 = np.array([door_x + gap, door_y])
        swing_dir = np.array([0, door_len])
    elif wall_side == 'left':
        p1 = np.array([door_x, door_y - gap])
        p2 = np.array([door_x, door_y + gap])
        swing_dir = np.array([door_len, 0])
    else:  # right
        p1 = np.array([door_x, door_y - gap])
        p2 = np.array([door_x, door_y + gap])
        swing_dir = np.array([-door_len, 0])
    p3 = p1 + swing_dir
    _draw_door_from_points(ax, [p1.tolist(), p2.tolist(), p3.tolist()])


def _draw_door_from_points(ax, pts):
    """画门（与 floorplan_renderer.py 完全一致）"""
    p1 = np.array(pts[0])
    p2 = np.array(pts[1])
    p3 = np.array(pts[2])
    door_len = np.linalg.norm(p2 - p1)
    if door_len < 0.1:
        return
    v_door = p2 - p1
    v_swing = p3 - p1
    door_angle = np.degrees(np.arctan2(v_door[1], v_door[0])) % 360
    swing_angle = np.degrees(np.arctan2(v_swing[1], v_swing[0])) % 360
    theta1 = door_angle
    theta2 = swing_angle
    diff = (theta2 - theta1) % 360
    if diff > 180:
        theta1, theta2 = theta2, theta1
    dx_ = v_door / door_len if door_len > 0 else np.array([1, 0])
    perp = np.array([-dx_[1], dx_[0]])
    door_w = 0.6
    corners = [p1, p2, p2 + perp * door_w, p1 + perp * door_w]
    door_poly = patches.Polygon(corners, closed=True,
                                facecolor=DOOR_FILL, edgecolor=DOOR_COLOR,
                                linewidth=1.2, zorder=4)
    ax.add_patch(door_poly)
    if door_len > 0.5:
        wedge = patches.Wedge(p1, door_len, theta1, theta2,
                              facecolor=DOOR_COLOR, edgecolor=DOOR_COLOR,
                              alpha=DOOR_ALPHA, linewidth=0.8, zorder=3)
        ax.add_patch(wedge)
    ax.plot(p1[0], p1[1], 'o', color=DOOR_COLOR, markersize=3, zorder=5)


def draw_furniture(ax, furniture_list):
    """画家俱（匹配 Figure_1.png 风格：浅灰填充、无纹理）"""
    for fur in furniture_list:
        pos = fur.get('position', {})
        fx = pos.get('x', 0)
        fy = pos.get('y', 0)
        fw = pos.get('width', 1)
        fh = pos.get('height', 1)
        if fw < 0.3 or fh < 0.3:
            continue
        rect_kwargs = dict(
            xy=(fx, fy), width=fw, height=fh,
            facecolor=FURNITURE_COLOR,
            edgecolor=FURNITURE_EDGE,
            linewidth=1.2,
            alpha=FURNITURE_ALPHA,
            zorder=4,
        )
        if FURNITURE_HATCH:
            rect_kwargs['hatch'] = FURNITURE_HATCH
        rect = patches.Rectangle(**rect_kwargs)
        ax.add_patch(rect)
        label = fur.get('name') or fur.get('id', '')
        if label:
            fs = min(FURNITURE_FONT_SIZE, max(fw, fh) * 1.5)
            if fs >= 3:
                ax.text(fx + fw / 2, fy + fh / 2, label,
                        ha='center', va='center',
                        fontsize=fs, color='#555555',
                        weight='bold', zorder=6)


def draw_floorplan(layout: dict, output_path: str, dpi: int = 150):
    """绘制户型图"""
    house = layout.get('house', {})
    rooms = house.get('rooms', [])
    size = house.get('size', {})
    max_w = size.get('x', 100)
    max_h = size.get('y', 100)

    fig, ax = plt.subplots(figsize=(size['x'] * 0.12, size['y'] * 0.12), dpi=dpi)
    ax.set_aspect('equal')
    margin = max(size['x'], size['y']) * 0.05
    ax.set_xlim(-margin, size['x'] + margin)
    ax.set_ylim(-margin, size['y'] + margin)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    drawn_doors = set()
    color_idx = 0

    for room in rooms:
        pos = room.get('position', {})
        x, y = pos.get('x', 0), pos.get('y', 0)
        w, h = pos.get('width', 10), pos.get('height', 10)
        color = ROOM_COLORS[color_idx % len(ROOM_COLORS)]

        # 房间填充
        rect = patches.Rectangle((x, y), w, h, facecolor=color,
                                 edgecolor='none', alpha=0.85, zorder=1)
        ax.add_patch(rect)

        # 画墙
        walls = room.get('walls', {})
        draw_boolean_walls(ax, x, y, w, h, walls)

        # 画门（去重）
        for door in room.get('doors', []):
            door_id = door.get('id', f"{x}_{y}")
            if door_id in drawn_doors:
                continue
            drawn_doors.add(door_id)
            pts = door.get('points', [])
            if len(pts) == 3:
                _draw_door_from_points(ax, pts)
            elif 'x' in door and 'y' in door:
                draw_door_at_point(ax, door['x'], door['y'], (x, y, w, h))

        # 画家俱
        furniture_list = room.get('furniture', [])
        if furniture_list:
            draw_furniture(ax, furniture_list)

        color_idx += 1

        # 房间名标签
        cx, cy = x + w / 2, y + h / 2
        ax.text(cx, cy, room['name'], ha='center', va='center',
                fontsize=7, color='#1B1B1B', weight='bold', zorder=5,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.7))

    plt.tight_layout(pad=0)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1,
                facecolor='white')
    plt.close(fig)
    print(f"户型图已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='户型图 YAML 可视化 (Figure_1 风格)')
    parser.add_argument('--code', default='0622',
                        help='家庭编码，用于默认路径拼接 (默认: 0622)')
    parser.add_argument('--input', '-i', default=None,
                        help='输入 YAML 文件路径 (默认: output/{code}/yaml/03_final_{code}.yaml)')
    parser.add_argument('--output', '-o', default=None,
                        help='输出图片路径 (默认: output/{code}/viz/floorplan_render_{code}.png)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='输出图片 DPI (默认 150)')
    args = parser.parse_args()

    code = args.code
    input_path = args.input or f'output/{code}/yaml/03_final_{code}.yaml'
    output_path = args.output or f'output/{code}/viz/floorplan_render_{code}.png'

    if not os.path.exists(input_path):
        print(f"[错误] 输入文件不存在: {input_path}")
        sys.exit(1)

    layout = load_yaml(input_path)
    draw_floorplan(layout, output_path, args.dpi)


if __name__ == '__main__':
    main()
