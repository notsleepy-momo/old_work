"""
Hough墙线网格 + 无墙合并分割房间

# 默认（和原来一样）
python run_seg.py

# 自定义图片 + 默认输出目录
python run_seg.py my_photo/test.png

# 自定义图片 + 自定义输出目录
python run_seg.py my_photo/test.png my_output

# 自定义图片 + 自定义输出目录 + 自定义比例
python run_seg.py my_photo/test.png my_output 0.08

# 或用 --scale 指定比例（从任意位置）
python run_seg.py my_photo/test.png --scale 0.08

# 你知道房屋高度 = 137dm，代码自动用检测到的房屋总高像素计算 SCALE
python ./photo2yaml/run_seg.py  --ref 137 --ref-on height
"""
import sys, os, cv2, numpy as np
import yaml
sys.stdout.reconfigure(encoding='utf-8')
from extract_hatch import extract_hatch_mask

# 解析命令行参数
INPUT_IMG_PATH = "photo2yaml/photo/room_real2.png"
OUTPUT_DIR = "photo2yaml/output"
SCALE = None  # None = 自动推算，可通过 --scale 直接指定
REF_LENGTH = 137.0  # 房屋某条边的真实长度（dm），代码自动检测对应边像素长
REF_DIM = "height"  # 参考哪条边：width（房屋总宽）或 height（房屋总高）

i = 1
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg == '--scale' and i+1 < len(sys.argv):
        SCALE = float(sys.argv[i+1])
        i += 2
    elif arg == '--ref' and i+1 < len(sys.argv):
        REF_LENGTH = float(sys.argv[i+1])
        i += 2
    elif arg == '--ref-on' and i+1 < len(sys.argv):
        REF_DIM = sys.argv[i+1]
        i += 2
    elif not arg.startswith('--'):
        if INPUT_IMG_PATH == "photo2yaml/photo/room_real2.png":      
            INPUT_IMG_PATH = arg
        elif OUTPUT_DIR == "photo2yaml/output":
            OUTPUT_DIR = arg
        i += 1
    else:
        i += 1

DOOR_MIN, DOOR_MAX = 10, 200
THRESHOLD_WALL = 120
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(INPUT_IMG_PATH, cv2.IMREAD_GRAYSCALE)
h, w = img.shape
print(f"输入: {w}x{h}")

# ========== 1. wall.png + 房屋轮廓 ==========
_, bin = cv2.threshold(img, THRESHOLD_WALL, 255, cv2.THRESH_BINARY)
inv = 255 - bin
inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), 2)
inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), 1)
nl, lbls, sts, _ = cv2.connectedComponentsWithStats(inv, 8)
for i in range(1, nl):
    if sts[i,cv2.CC_STAT_AREA] < 30: inv[lbls==i] = 0
wall_img = 255 - inv
cv2.imwrite(f"{OUTPUT_DIR}/viz/wall.png", wall_img)
walls_white = 255 - wall_img

_, bg = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
cnts, _ = cv2.findContours(bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bg_cnt = max(cnts, key=cv2.contourArea)
hx, hy, hw, hh = cv2.boundingRect(bg_cnt)
print(f"房屋: ({hx},{hy}) {hw}x{hh}")
house_mask = np.zeros((h,w), np.uint8)
cv2.drawContours(house_mask, [bg_cnt], -1, 255, -1)

wall_mid = cv2.dilate(walls_white, np.ones((5,5),np.uint8), 6)
wall_raw = walls_white  # 原始阈值，未膨胀，保留门洞间隙

# ========== 2. Hough墙线检测 ==========
edges = cv2.Canny(cv2.morphologyEx(walls_white, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), 2), 5, 30, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=15, maxLineGap=10)

h_raw, v_raw = {}, {}
if lines is not None:
    for ln in lines:
        x1,y1,x2,y2 = ln[0]
        dx, dy = abs(x2-x1), abs(y2-y1)
        if np.sqrt(dx*dx+dy*dy) < 20: continue
        mx, my = (x1+x2)//2, (y1+y2)//2
        if not (hx <= mx <= hx+hw and hy <= my <= hy+hh): continue
        if dx > dy:
            y = y1
            if y not in h_raw: h_raw[y] = []
            h_raw[y].append((min(x1,x2), max(x1,x2)))
        else:
            x = x1
            if x not in v_raw: v_raw[x] = []
            v_raw[x].append((min(y1,y2), max(y1,y2)))

# 合并线段
def merge_segs(segs, gap=20):
    if not segs: return []
    segs = sorted(segs); m = [list(segs[0])]
    for s,e in segs[1:]:
        if s - m[-1][1] <= gap: m[-1][1] = max(m[-1][1], e)
        else: m.append([s,e])
    return [(s,e) for s,e in m]

def merge_parallel(d, tol=5):
    keys = sorted(d.keys()); res = {}; used = set()
    for i, k1 in enumerate(keys):
        if k1 in used: continue
        segs = list(d[k1]); grp = [k1]
        for k2 in keys[i+1:]:
            if k2 in used: continue
            if abs(k2-k1) <= tol:
                segs.extend(d[k2]); grp.append(k2); used.add(k2)
        avg = int(np.mean(grp))
        ms = merge_segs(segs, 25)
        ms = [(s,e) for s,e in ms if e-s >= 40]
        if ms: res[avg] = ms
    return res

h_walls = merge_parallel(h_raw)
v_walls = merge_parallel(v_raw)
print(f"Hough墙线: {len(h_walls)} H, {len(v_walls)} V")

# ========== 3. 筛选墙线构建网格 ==========
# 水平墙线：总覆盖长度 > 房屋宽度*10%
h_sel = sorted([y for y, segs in h_walls.items() 
                if sum(e-s for s,e in segs) > hw*0.10 and hy+10 < y < hy+hh-10])
# 垂直墙线：总覆盖长度 > 房屋高度*10%
v_sel = sorted([x for x, segs in v_walls.items()
                if sum(e-s for s,e in segs) > hh*0.10 and hx+10 < x < hx+hw-10])

# 合并间距<15px的相近墙线
def dedup_close(lst, tol=15):
    if not lst: return []
    res = [lst[0]]
    for v in lst[1:]:
        if v - res[-1] <= tol:
            res[-1] = (res[-1] + v) // 2
        else:
            res.append(v)
    return res

h_sel = dedup_close(h_sel, 15)
v_sel = dedup_close(v_sel, 15)

# 加入房屋边界
all_ys = sorted(set([hy, hy+hh] + h_sel))
all_xs = sorted(set([hx, hx+hw] + v_sel))

print(f"网格: {len(all_ys)-1}行 x {len(all_xs)-1}列, H={h_sel}, V={v_sel}")

# ========== 4. 网格分割 + 基于grid边界的无墙合并（直接使用grid坐标） ==========
nrows = len(all_ys) - 1
ncols = len(all_xs) - 1

# 每个单元格是否属于房屋内部
cells = []
for r in range(nrows):
    for c in range(ncols):
        rx = all_xs[c]; ry = all_ys[r]
        rw = all_xs[c+1] - rx
        rh = all_ys[r+1] - ry
        cy_center = int(ry + rh/2)
        cx_center = int(rx + rw/2)
        if house_mask[cy_center, cx_center] > 0:
            cells.append({'row': r, 'col': c, 'x': float(rx), 'y': float(ry), 
                          'w': float(rw), 'h': float(rh), 'area': float(rw*rh)})

print(f"网格单元: {len(cells)}")

# union-find
uf_parent = list(range(len(cells)))
def find(u):
    while uf_parent[u] != u:
        uf_parent[u] = uf_parent[uf_parent[u]]
        u = uf_parent[u]
    return u
def union(u, v):
    ru, rv = find(u), find(v)
    if ru != rv: uf_parent[rv] = ru

# 检查相邻grid位置之间是否有墙
def has_wall_at_grid_boundary(r1, c1, r2, c2):
    if r1 == r2 and abs(c1 - c2) == 1:  # 垂直墙线
        wx = all_xs[max(c1, c2)]
        y0 = all_ys[r1]; y1 = all_ys[r1+1]
        ys = y0 + 5; ye = y1 - 5
        if ye - ys < 10: return True
        strip = np.max(wall_mid[int(ys):int(ye), max(0,int(wx)-5):min(w,int(wx)+6)], axis=1)
        return np.mean(strip > 0) > 0.08
    elif c1 == c2 and abs(r1 - r2) == 1:  # 水平墙线
        wy = all_ys[max(r1, r2)]
        x0 = all_xs[c1]; x1 = all_xs[c1+1]
        xs = x0 + 5; xe = x1 - 5
        if xe - xs < 10: return True
        strip = np.max(wall_mid[max(0,int(wy)-5):min(h,int(wy)+6), int(xs):int(xe)], axis=0)
        return np.mean(strip > 0) > 0.08
    return True

# 遍历所有相邻grid位置
for r in range(nrows):
    for c in range(ncols):
        idx = next((i for i,cell in enumerate(cells) if cell['row']==r and cell['col']==c), None)
        if idx is None: continue
        if c+1 < ncols:
            ridx = next((i for i,cell in enumerate(cells) if cell['row']==r and cell['col']==c+1), None)
            if ridx is not None and not has_wall_at_grid_boundary(r, c, r, c+1):
                union(idx, ridx)
        if r+1 < nrows:
            bidx = next((i for i,cell in enumerate(cells) if cell['row']==r+1 and cell['col']==c), None)
            if bidx is not None and not has_wall_at_grid_boundary(r, c, r+1, c):
                union(idx, bidx)

# 用grid border计算每个group的bbox
g2r = {}
for idx, c in enumerate(cells):
    root = find(idx)
    if root not in g2r: g2r[root] = set()
    g2r[root].add((c['row'], c['col']))

# 生成候选房间（每个group一个初始bbox）
init_rooms = []
for root, rcs in g2r.items():
    rows = [rc[0] for rc in rcs]
    cols = [rc[1] for rc in rcs]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    rx = all_xs[min_c];  ry = all_ys[min_r]
    rw = all_xs[max_c+1] - rx
    rh = all_ys[max_r+1] - ry
    init_rooms.append({'x': float(rx), 'y': float(ry), 'w': float(rw), 'h': float(rh)})

# 对每个初始房间，用CC在wall_mid内分割出子房间
# 对于大房间，拆分为父房间+子房间结构
rooms = []
MAX_SPLIT_AREA = 250000
MAX_SPLIT_CELLS = 8
room_parent = {}  # subroom_id -> parent_id  (父房间id->空字符串)

def merge_adjacent_by_row(cells_list):
    """合并同一行内相邻的cells为更宽的subroom"""
    rows_cells = {}
    for c in cells_list:
        r = c['row']
        if r not in rows_cells: rows_cells[r] = []
        rows_cells[r].append(c)
    result = []
    for r in sorted(rows_cells.keys()):
        row_cells = sorted(rows_cells[r], key=lambda x: x['col'])
        cur = dict(row_cells[0])
        for c in row_cells[1:]:
            if c['col'] == cur['col'] + 1:
                cur['w'] += c['w']; cur['area'] += c['area']
            else:
                result.append(cur); cur = dict(c)
        result.append(cur)
    # 重写x,y为grid坐标（确保对齐）
    for sr in result:
        sr['x'] = float(all_xs[sr['col']])
        sr['y'] = float(all_ys[sr['row']])
    return result

for ir in init_rooms:
    rx, ry, rw, rh = ir['x'], ir['y'], ir['w'], ir['h']
    area = rw * rh
    covered = [c for c in cells if 
               c['x'] >= rx and c['x']+c['w'] <= rx+rw and
               c['y'] >= ry and c['y']+c['h'] <= ry+rh]
    
    if area > MAX_SPLIT_AREA or len(covered) > MAX_SPLIT_CELLS:
        group_cells = [c for c in cells if find(cells.index(c)) == find(cells.index(covered[0]))] if covered else covered
        sub_rooms = merge_adjacent_by_row(group_cells)
        px = min(s['x'] for s in sub_rooms); py = min(s['y'] for s in sub_rooms)
        px2 = max(s['x']+s['w'] for s in sub_rooms); py2 = max(s['y']+s['h'] for s in sub_rooms)
        parent_id = f'room{len(rooms)+1}'
        for si, sr in enumerate(sub_rooms):
            sr['id'] = f'{parent_id}_tmp_{si+1}'
            room_parent[sr['id']] = parent_id
        room_parent[parent_id] = ''
        parent_room = {'id': parent_id, 'x': px, 'y': py, 'w': px2-px, 'h': py2-py, 'area': (px2-px)*(py2-py),
                  'subrooms': sub_rooms}
        rooms.append(parent_room)
        print(f"  [父房间] ({px:.0f},{py:.0f}) {px2-px:.0f}x{py2-py:.0f} → {len(sub_rooms)}个子房间")
        for sr in sub_rooms:
            print(f"    子: {sr['id']}: ({sr['x']:.0f},{sr['y']:.0f}) {sr['w']:.0f}x{sr['h']:.0f}")
    else:
        rooms.append({'id': f'room{len(rooms)+1}', 'x': rx, 'y': ry, 'w': rw, 'h': rh, 'area': rw*rh})

rooms.sort(key=lambda r: (r['y']//80, r['x']))
seen_parent = set()
final_rooms = []
for r in rooms:
    pid = r.get('id', '')
    if pid in room_parent and room_parent[pid] != '':
        continue  # 子房间不单独作为顶层房间
    if pid in seen_parent: continue
    # 重新编号
    r['id'] = f'room{len(final_rooms)+1}'
    # 子房间也重新编号
    if 'subrooms' in r:
        for sr in r['subrooms']:
            sr['id'] = f'{r["id"]}_sub_{r["subrooms"].index(sr)+1}'
            room_parent[sr['id']] = r['id']
    seen_parent.add(pid)
    final_rooms.append(r)
rooms = final_rooms
print(f"最终房间: {len(rooms)}")
for r in rooms:
    print(f"  {r['id']}: ({r['x']:.0f},{r['y']:.0f}) {r['w']:.0f}x{r['h']:.0f}")
    if 'subrooms' in r:
        for sr in r['subrooms']:
            print(f"    ├ {sr['id']}: ({sr['x']:.0f},{sr['y']:.0f}) {sr['w']:.0f}x{sr['h']:.0f}")

# ========== 5. 墙+门检测 ==========
for r in rooms: r['walls_bool'] = {'top':False,'right':False,'bottom':False,'left':False}

pairs = set()
for i, a in enumerate(rooms):
    for j, b in enumerate(rooms):
        if i >= j: continue
        ar, ab = a['x']+a['w'], a['y']+a['h']
        br, bb = b['x']+b['w'], b['y']+b['h']
        if abs(ar-b['x']) < 100 and a['y'] < bb and ab > b['y']:
            pairs.add((a['id'], b['id'], 'v'))
        elif abs(br-a['x']) < 100 and b['y'] < ab and bb > a['y']:
            pairs.add((a['id'], b['id'], 'v'))
        if abs(ab-b['y']) < 100 and a['x'] < br and ar > b['x']:
            pairs.add((a['id'], b['id'], 'h'))
        elif abs(bb-a['y']) < 100 and b['x'] < ar and br > a['x']:
            pairs.add((a['id'], b['id'], 'h'))

# 补充子房间与相邻房间的pair
for r in rooms:
    if 'subrooms' not in r: continue
    for sr in r['subrooms']:
        sr_r = sr['x']+sr['w']; sr_b = sr['y']+sr['h']
        for r2 in rooms:
            if r2['id'] == r['id']: continue
            r2_r = r2['x']+r2['w']; r2_b = r2['y']+r2['h']
            if abs(sr_r-r2['x']) < 100 and sr['y'] < r2_b and sr_b > r2['y']:
                if (r['id'], r2['id'], 'v') not in pairs:
                    pairs.add((r['id'], r2['id'], 'v'))
            if abs(sr_b-r2['y']) < 100 and sr['x'] < r2_r and sr_r > r2['x']:
                if (r['id'], r2['id'], 'h') not in pairs:
                    pairs.add((r['id'], r2['id'], 'h'))

doors = []; dc = [0]

def _safe_max(arr, axis):
    """np.max 的安全版本，切片为空时返回空数组而非崩溃"""
    if arr.size == 0:
        return np.array([], dtype=arr.dtype)
    return np.max(arr, axis)

def detect_door_between(ra, rb, o):
    """检测两个房间（或子房间）之间的门。
    ra,rb 是包含 x,y,w,h 的dict（可以是subroom）。
    返回 (found, door_list)。
    注意：重叠区域的墙可能很薄，从grid全边界检查墙的存在。
    """
    if o == 'h':
        # 确定共享墙线位置：支持正向（a.bottom vs b.top）和反向（b.bottom vs a.top）
        ab = ra['y']+ra['h']; bb = rb['y']+rb['h']
        if abs(ab-rb['y']) < 30:
            yf = int((ab+rb['y'])/2)
        elif abs(bb-ra['y']) < 30:
            yf = int((bb+ra['y'])/2)
        else:
            yf = int((ab+rb['y'])/2)
        xs = int(max(ra['x'], rb['x'])); xe = int(min(ra['x']+ra['w'], rb['x']+rb['w']))
        if xe-xs < 10: return False, []
        # 墙检查：用完整grid边界范围（不仅重叠区域）
        wy0 = int(min(ra['y'], rb['y'])); wy1 = int(max(ra['y']+ra['h'], rb['y']+rb['h']))
        ms = _safe_max(wall_mid[max(0,wy0-5):min(h,wy1+6), xs:xe], 0)
        if ms.size == 0 or np.mean(ms>0) < 0.03:
            # 在更窄的重叠范围再试一次（门可能占满整段）
            ys = int(max(ra['y'], rb['y'])); ye = int(min(ra['y']+ra['h'], rb['y']+rb['h']))
            if ye-ys < 10: return False, []
            ms2 = _safe_max(wall_mid[max(0,ys-5):min(h,ye+6), xs:xe], 0)
            if ms2.size == 0 or np.mean(ms2>0) < 0.03: return False, []
        ts = _safe_max(wall_raw[max(0,wy0-1):min(h,wy1+2), xs:xe], 0)
        # 门扫描用重叠范围
        ys = int(max(ra['y'], rb['y'])); ye = int(min(ra['y']+ra['h'], rb['y']+rb['h']))
        ts2 = _safe_max(wall_raw[max(0,ys-1):min(h,ye+2), xs:xe], 0)
        
        fd = []; ig = False; gs = 0
        for k in range(len(ts2)):
            if ts2[k]==0 and not ig: ig=True; gs=k
            elif ts2[k]>0 and ig:
                gl=k-gs
                if DOOR_MIN<=gl<=DOOR_MAX:
                    dc[0]+=1
                    pts = [[xs+gs, yf], [(xs+gs+xs+k)/2, yf], [xs+k, yf]]
                    fd.append({'id': f'door{dc[0]}', 'points': pts, 'between': (ra['id'], rb['id'])})
                ig=False
        if ig:
            gl=len(ts2)-gs
            if DOOR_MIN<=gl<=DOOR_MAX:
                dc[0]+=1
                pts = [[xs+gs, yf], [(xs+gs+xs+len(ts2))/2, yf], [xs+len(ts2), yf]]
                fd.append({'id': f'door{dc[0]}', 'points': pts, 'between': (ra['id'], rb['id'])})
        return True, fd
    else:  # 'v'
        # 确定共享墙线位置：支持正向（a.right vs b.left）和反向（b.right vs a.left）
        ar = ra['x']+ra['w']; br = rb['x']+rb['w']
        if abs(ar-rb['x']) < 30:
            xf = int((ar+rb['x'])/2)
        elif abs(br-ra['x']) < 30:
            xf = int((br+ra['x'])/2)
        else:
            xf = int((ar+rb['x'])/2)
        ys = int(max(ra['y'], rb['y'])); ye = int(min(ra['y']+ra['h'], rb['y']+rb['h']))
        if ye-ys < 10: return False, []
        # 墙检查：用完整y范围（上下两房间并集，门可能占满整个重叠区）
        full_ys = int(min(ra['y'], rb['y'])); full_ye = int(max(ra['y']+ra['h'], rb['y']+rb['h']))
        ms = _safe_max(wall_mid[full_ys:full_ye, max(0,xf-5):min(w,xf+6)], 1)
        if ms.size == 0 or np.mean(ms>0) < 0.03:
            # 更宽的x范围再试
            ms2 = _safe_max(wall_mid[full_ys:full_ye, max(0,xf-15):min(w,xf+16)], 1)
            if ms2.size == 0 or np.mean(ms2>0) < 0.03: return False, []
        ts = _safe_max(wall_raw[ys:ye, max(0,xf-1):min(w,xf+2)], 1)
        if ts.size == 0: return False, []
        
        fd = []; ig = False; gs = 0
        for k in range(len(ts)):
            if ts[k]==0 and not ig: ig=True; gs=k
            elif ts[k]>0 and ig:
                gl=k-gs
                if DOOR_MIN<=gl<=DOOR_MAX:
                    dc[0]+=1
                    pts = [[xf, ys+gs], [xf, (ys+gs+ys+k)/2], [xf, ys+k]]
                    fd.append({'id': f'door{dc[0]}', 'points': pts, 'between': (ra['id'], rb['id'])})
                ig=False
        if ig:
            gl=len(ts)-gs
            if DOOR_MIN<=gl<=DOOR_MAX:
                dc[0]+=1
                pts = [[xf, ys+gs], [xf, (ys+gs+ys+len(ts))/2], [xf, ys+len(ts)]]
                fd.append({'id': f'door{dc[0]}', 'points': pts, 'between': (ra['id'], rb['id'])})
        return True, fd

sd={}; sn={}
for a,b,o in sorted(pairs):
    ra = next(r for r in rooms if r['id']==a)
    rb = next(r for r in rooms if r['id']==b)
    ok, det = detect_door_between(ra, rb, o)
    if not ok: continue
    as_,bs_ = ('right','left') if o=='v' else ('bottom','top')
    ka,kb=(a,as_),(b,bs_)
    if det: sd[ka]=sd.get(ka,0)+1; sd[kb]=sd.get(kb,0)+1; doors.extend(det)
    else: sn[ka]=sn.get(ka,0)+1; sn[kb]=sn.get(kb,0)+1

# 补充子房间门检测（用子房间实际边界，避免gap导致xf落在空隙中）
for r in rooms:
    if 'subrooms' not in r: continue
    for sr in r['subrooms']:
        sr_right = sr['x']+sr['w']; sr_bot = sr['y']+sr['h']
        for r2 in rooms:
            if r2['id'] == r['id']: continue
            r2_right = r2['x']+r2['w']; r2_bot = r2['y']+r2['h']

            # 检查4个方向：subroom.right vs room.left, subroom.left vs room.right,
            # subroom.bottom vs room.top, subroom.top vs room.bottom
            cands = []
            # v: 子右 vs 房左
            if abs(sr_right-r2['x']) < 100 and sr['y'] < r2_bot and sr_bot > r2['y']:
                cands.append(('v', sr_right, r2['x']))
            # v: 子左 vs 房右
            if abs(sr['x']-r2_right) < 100 and sr['y'] < r2_bot and sr_bot > r2['y']:
                cands.append(('v', sr['x'], r2_right))
            # h: 子底 vs 房顶
            if abs(sr_bot-r2['y']) < 100 and sr['x'] < r2_right and sr_right > r2['x']:
                cands.append(('h', sr_bot, r2['y']))
            # h: 子顶 vs 房底
            if abs(sr['y']-r2_bot) < 100 and sr['x'] < r2_right and sr_right > r2['x']:
                cands.append(('h', sr['y'], r2_bot))

            for (o, sr_side, r2_side) in cands:
                if o == 'v':
                    ys = int(max(sr['y'], r2['y']))
                    ye = int(min(sr_bot, r2_bot))
                    if ye-ys < 10: continue
                    for xf in [int(sr_side), int(r2_side)]:
                        if xf < 0 or xf >= w: continue
                        ms = np.max(wall_mid[ys:ye, max(0,xf-5):min(w,xf+6)], 1)
                        if np.mean(ms>0) < 0.03: continue
                        ts = np.max(wall_raw[ys:ye, max(0,xf-1):min(w,xf+2)], 1)
                        sc_fd = []; ig = False; gs = 0
                        for k in range(len(ts)):
                            if ts[k]==0 and not ig: ig=True; gs=k
                            elif ts[k]>0 and ig:
                                gl=k-gs
                                if DOOR_MIN<=gl<=DOOR_MAX:
                                    dc[0]+=1
                                    pts = [[xf, ys+gs], [xf, (ys+gs+ys+k)/2], [xf, ys+k]]
                                    sc_fd.append({'id': f'door{dc[0]}', 'points': pts, 'between': (r['id'], r2['id'])})
                                ig=False
                        if ig:
                            gl=len(ts)-gs
                            if DOOR_MIN<=gl<=DOOR_MAX:
                                dc[0]+=1
                                pts = [[xf, ys+gs], [xf, (ys+gs+ys+len(ts))/2], [xf, ys+len(ts)]]
                                sc_fd.append({'id': f'door{dc[0]}', 'points': pts, 'between': (r['id'], r2['id'])})
                        if sc_fd:
                            for d in sc_fd:
                                if d not in doors:
                                    doors.append(d)
                            as_, bs_ = ('right','left')
                            r['walls_bool'][as_] = True
                            r2['walls_bool'][bs_] = True
                else:  # 'h'
                    xs = int(max(sr['x'], r2['x'])); xe = int(min(sr_right, r2_right))
                    if xe-xs < 10: continue
                    for yf in [int(sr_side), int(r2_side)]:
                        if yf < 0 or yf >= h: continue
                        ms = np.max(wall_mid[max(0,yf-5):min(h,yf+6), xs:xe], 0)
                        if np.mean(ms>0) < 0.03: continue
                        ts = np.max(wall_raw[max(0,yf-1):min(h,yf+2), xs:xe], 0)
                        sc_fd = []; ig = False; gs = 0
                        for k in range(len(ts)):
                            if ts[k]==0 and not ig: ig=True; gs=k
                            elif ts[k]>0 and ig:
                                gl=k-gs
                                if DOOR_MIN<=gl<=DOOR_MAX:
                                    dc[0]+=1
                                    pts = [[xs+gs, yf], [(xs+gs+xs+k)/2, yf], [xs+k, yf]]
                                    sc_fd.append({'id': f'door{dc[0]}', 'points': pts, 'between': (r['id'], r2['id'])})
                                ig=False
                        if ig:
                            gl=len(ts)-gs
                            if DOOR_MIN<=gl<=DOOR_MAX:
                                dc[0]+=1
                                pts = [[xs+gs, yf], [(xs+gs+xs+len(ts))/2, yf], [xs+len(ts), yf]]
                                sc_fd.append({'id': f'door{dc[0]}', 'points': pts, 'between': (r['id'], r2['id'])})
                        if sc_fd:
                            for d in sc_fd:
                                if d not in doors:
                                    doors.append(d)
                            as_, bs_ = ('bottom','top')
                            r['walls_bool'][as_] = True
                            r2['walls_bool'][bs_] = True

for r in rooms:
    for s in ('top','bottom','left','right'):
        if sn.get((r['id'],s),0)>0 or sd.get((r['id'],s),0)>0: r['walls_bool'][s]=True

paired=set()
for a,b,o in pairs: paired.add((a,'right'if o=='v'else'bottom')); paired.add((b,'left'if o=='v'else'top'))
for r in rooms:
    rx,ry,rww,rhh=int(r['x']),int(r['y']),int(r['w']),int(r['h'])
    for s in ('top','bottom','left','right'):
        if r['walls_bool'][s] or (r['id'],s) in paired: continue
        if s=='top': strip=wall_mid[max(0,ry-8):min(h,ry+4), rx:rx+rww]
        elif s=='bottom': strip=wall_mid[max(0,ry+rhh-4):min(h,ry+rhh+8), rx:rx+rww]
        elif s=='left': strip=wall_mid[ry:ry+rhh, max(0,rx-8):min(w,rx+4)]
        else: strip=wall_mid[ry:ry+rhh, max(0,rx+rww-4):min(w,rx+rww+8)]
        if strip.size>0 and np.mean(strip)>0.05: r['walls_bool'][s]=True

uniq=[]; seen=[]
for d in doors:
    p=d['points'][1]; k=(round(p[0]/5)*5, round(p[1]/5)*5)
    if k not in seen: seen.append(k); uniq.append(d)
doors=uniq
for i,d in enumerate(doors): d['id']=f'door{i+1}'
print(f"门洞: {len(doors)}")

# ========== 6. 可视化 ==========
vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# ── 生成 greyroom（严格3类像素：墙=0, 地板=128, 轨迹=255） ──
# 输入: room_real.png (img, 已读取为灰度图)
# 墙体: 复用上方已生成的 wall.png (白色=墙) — wall_img 已在第1步生成
# 轨迹: 直接从 room_real.png 提取灰度 > TRAJ_THRESHOLD 且非墙的像素
# 其余: 全部赋值为128（地板/家具/阴影/杂斑一律不区分）
#
# 【可调参数】
#   TRAJ_THRESHOLD = 220  轨迹判定灰度下限 (输入图为灰度图，>220 视为纯白轨迹)
TRAJ_THRESHOLD = 220

# === 1) 墙体掩码: wall.png 中黑像素(0)即为墙 ===
#     wall_img: 墙=0(黑), 地板=255(白) — 在第1步由 img 二值化生成
wall_mask = (wall_img < 128)

# === 2) 墙边缘区域掩码: 墙附近的区域（用于将轨迹像素填充为地板）===
#     膨胀墙得到墙附近区域，但不改变原始墙的位置
wall_edge_mask = cv2.dilate(wall_mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=2)
wall_edge_mask = (wall_edge_mask > 0) & (~wall_mask)  # 仅墙附近区域，不含墙本身

# === 3) 轨迹掩码: 灰度 > TRAJ_THRESHOLD 且非墙体 且在房屋轮廓内 ===
#     房屋外背景可能是白色，用 house_mask 排除
#     house_mask: 房屋内部=255, 外部=0 (第1步通过轮廓提取生成)
traj_mask = (img > TRAJ_THRESHOLD) & (~wall_mask) & (house_mask > 0)

# === 4) 构建 greyroom: 严格只有 0 / 128 / 255 三种值 ===
grey = np.full((h, w), 128, dtype=np.uint8)  # 默认全部地板=128
grey[wall_mask] = 0    # 墙体=0（保持原始墙位置不变）
# 轨迹像素：墙附近区域内的轨迹填充为地板，其余保持为轨迹
traj_final = traj_mask & (~wall_edge_mask)
grey[traj_final] = 255  # 轨迹=255（排除墙附近区域）
# 注意: 无 dark_unreached 分支 — 非墙非轨迹像素固定128，不还原原图灰度

cv2.imwrite(f"{OUTPUT_DIR}/masks/greyroom.png", grey)
print(f"greyroom saved (3-class: wall=0 floor=128 traj=255, TRAJ_THRESHOLD={TRAJ_THRESHOLD}): {OUTPUT_DIR}/masks/greyroom.png")

# ========== 斜线阴影检测：生成 hatch_mask.png ==========
# 在原始灰度图上检测 Gabor 纹理（45°/-45°斜线阴影），
# 阴影区域 = 家具位置，输出二值掩码供 furniture_detection_agent 使用
hatch_mask_path = f"{OUTPUT_DIR}/masks/hatch_mask.png"
print("开始斜线阴影检测 (Gabor filter)...")
extract_hatch_mask(
    img_path=INPUT_IMG_PATH,
    output_path=hatch_mask_path,
    ksize=21,
    sigma=4.0,
    lambd=8.0,
    gamma=0.5,
    morph_kernel_size=5,
    min_area=200,
)
print(f"hatch_mask saved: {hatch_mask_path}")

# ── 填平内部小黑孔（噪声） ──
hatch_img = cv2.imread(hatch_mask_path, cv2.IMREAD_GRAYSCALE)
if hatch_img is not None:
    # 闭运算填平白色区域内部的黑色小孔
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    hatch_filled = cv2.morphologyEx(hatch_img, cv2.MORPH_CLOSE, kernel_fill)
    cv2.imwrite(hatch_mask_path, hatch_filled)
    filled_px = int(np.sum(hatch_filled > 0)) - int(np.sum(hatch_img > 0))
    print(f"hatch_mask 填平内部黑孔: +{filled_px} 像素")

# 灰度调色板参考图
palette = np.zeros((50, 256, 3), dtype=np.uint8)
for i in range(256):
    palette[:, i] = [i, i, i]
cv2.putText(palette, "0=Wall", (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
cv2.putText(palette, "128=Floor", (100, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
cv2.putText(palette, "255=Traj", (215, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
cv2.imwrite(f"{OUTPUT_DIR}/masks/greyroom_palette.png", palette)

# 用merged_mask生成CSV网格
colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),
          (128,0,128),(0,128,128),(128,128,0)]
merged_mask = np.zeros((h,w), np.uint8)
for idx, r in enumerate(rooms):
    if 'subrooms' in r:
        # 子房间合并绘制
        for sr in r['subrooms']:
            cv2.rectangle(merged_mask, (int(sr['x']),int(sr['y'])), 
                         (int(sr['x']+sr['w']),int(sr['y']+sr['h'])), idx+1, -1)
    else:
        cv2.rectangle(merged_mask, (int(r['x']),int(r['y'])), 
                     (int(r['x']+r['w']),int(r['y']+r['h'])), idx+1, -1)
for i, r in enumerate(rooms):
    mask2 = (merged_mask==i+1).astype(np.uint8)*255
    # 闭操作填平小缝隙，确保轮廓连续
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), 1)
    c,_ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if c: cv2.drawContours(vis, c, -1, colors[i%len(colors)], 2)
    # 标签放在父房间bbox左上角
    label_x = int(r['x']) + 3
    label_y = int(r['y']) + 15
    cv2.putText(vis, r['id'], (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i%len(colors)], 1)
for d in doors:
    p=d['points']
    cv2.line(vis, (int(p[0][0]),int(p[0][1])), (int(p[2][0]),int(p[2][1])), (0,255,255), 3)
    cv2.circle(vis, (int(p[1][0]),int(p[1][1])), 4, (0,255,255), -1)
cv2.rectangle(vis, (hx,hy), (hx+hw,hy+hh), (0,0,255), 1)
cv2.imwrite(f"{OUTPUT_DIR}/viz/room_final.png", vis)

# ========== 7. 自动推算SCALE + YAML ==========

# 用房屋总宽/总高推算 SCALE（用户提供真实长度，代码自动取检测到的像素长度）
# SCALE = REF_LENGTH / hw（宽）或 / hh（高）
if REF_LENGTH is not None:
    if REF_DIM == "height" and hh > 0:
        SCALE = REF_LENGTH / hh
        print(f"参考边 SCALE: {SCALE:.6f} (房屋总高 {REF_LENGTH}m / {hh}px)")
    elif hw > 0:
        SCALE = REF_LENGTH / hw
        print(f"参考边 SCALE: {SCALE:.6f} (房屋总宽 {REF_LENGTH}m / {hw}px)")
    else:
        print(f"⚠ 未检测到房屋边界，跳过参考边推算")

if SCALE is None:
    STANDARD_DOOR_WIDTH = 3.25  # dm (标准门宽)
    door_gaps_px = []
    for d in doors:
        pts = d.get('points', [])
        if len(pts) >= 2:
            p1, p2 = pts[0], pts[1]
            gap = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            if gap > 5:
                door_gaps_px.append(gap)
    if door_gaps_px:
        avg_gap = np.mean(door_gaps_px)
        SCALE = float(STANDARD_DOOR_WIDTH / avg_gap)
        print(f"门宽法 SCALE: {SCALE:.6f} m/px (平均门宽 {avg_gap:.0f}px)")
    else:
        SCALE = 0.01  # m/px (保守默认值)
        print(f"未检测到门，使用默认 SCALE={SCALE} m/px")
# 确保 YAML 输出为普通 Python 类型
e={}
for r in rooms: e[r['id']]={'l':float(r['x']),'r':float(r['x']+r['w']),'t':float(r['y']),'b':float(r['y']+r['h'])}
for a,b,o in sorted(pairs):
    ae,be=e[a],e[b]
    if o=='v':
        g=be['l']-ae['r']
        if 2<g<100: m=(be['l']+ae['r'])/2; ae['r']=max(ae['r'],m); be['l']=min(be['l'],m)
    else:
        g=be['t']-ae['b']
        if 2<g<100: m=(be['t']+ae['b'])/2; ae['b']=max(ae['b'],m); be['t']=min(be['t'],m)
for r in rooms:
    ee=e[r['id']]; r['x']=float(ee['l']); r['y']=float(ee['t'])
    r['w']=float(max(30,ee['r']-ee['l'])); r['h']=float(max(30,ee['b']-ee['t']))

# 坐标原点改为房屋左下角 (hx, hy+hh)
bl_x = hx
bl_y = hy + hh
yd={
    "house":{
        "size":{"x":hw,"y":hh},
        "rooms":[]
    },
    "metadata":{
        "SCALE":round(SCALE,6),
        "bl_x":int(bl_x),
        "bl_y":int(bl_y),
    }
}
for r in rooms:
    rx_=float(r['x']); ry_=float(r['y'])
    rw_=float(r['w']); rh_=float(r['h'])
    dl=[]
    for d in doors:
        if r['id'] in d['between']:
            p=d['points'][1]
            dl.append({"id":d['id'],"x":float(p[0]),"y":float(p[1])})
    room_data = {
        "name":r['id'],"position":{"x":rx_,"y":ry_,"width":rw_,"height":rh_},
        "walls":{"top":r['walls_bool']['top'],"right":r['walls_bool']['right'],
                 "bottom":r['walls_bool']['bottom'],"left":r['walls_bool']['left']},
        "doors":dl
    }
    if 'subrooms' in r:
        # 判断是否为规则矩形房间（子房间总面积 ≈ 父房间面积，无 L 型缺口）
        sub_total_area = sum(sr['w'] * sr['h'] for sr in r['subrooms'])
        parent_area = r['w'] * r['h']
        is_regular_rect = parent_area > 0 and sub_total_area / parent_area > 0.99
        if is_regular_rect:
            # 规则矩形：不需要 subrooms 和 wall_segments，直接使用父房间的 4 个墙标志
            pass
        else:
            # 不规则形状：为每个子房间计算墙线段（非共享的部分即为外墙线段）
            def get_edge_seg(sr, side):
                """返回 (x1,y1,x2,y2)"""
                if side == 'top':    return (sr['x'], sr['y'], sr['x']+sr['w'], sr['y'])
                if side == 'bottom': return (sr['x'], sr['y']+sr['h'], sr['x']+sr['w'], sr['y']+sr['h'])
                if side == 'left':   return (sr['x'], sr['y'], sr['x'], sr['y']+sr['h'])
                if side == 'right':  return (sr['x']+sr['w'], sr['y'], sr['x']+sr['w'], sr['y']+sr['h'])

            def segs_overlap(s1, s2, tol=2):
                """两条线段是否共线重叠（返回重叠长度）"""
                x1,y1,x2,y2 = s1; u1,v1,u2,v2 = s2
                if y1==y2 and v1==v2 and abs(y1-v1)<tol:
                    lo=max(min(x1,x2),min(u1,u2)); hi=min(max(x1,x2),max(u1,u2))
                    return max(0, hi-lo)
                if x1==x2 and u1==u2 and abs(x1-u1)<tol:
                    lo=max(min(y1,y2),min(v1,v2)); hi=min(max(y1,y2),max(v1,v2))
                    return max(0, hi-lo)
                return 0

            def subtract_seg(full, remove):
                """从 full 线段中减去 remove 线段的重叠部分，返回剩余线段列表"""
                x1,y1,x2,y2 = full; rx1,ry1,rx2,ry2 = remove
                ol = segs_overlap(full, remove)
                if ol < 2: return [full]
                if y1==y2:  # 水平
                    lo=max(min(x1,x2),min(rx1,rx2)); hi=min(max(x1,x2),max(rx1,rx2))
                    out=[]
                    if min(x1,x2) < lo-2: out.append((min(x1,x2),y1,lo,y2))
                    if max(x1,x2) > hi+2: out.append((hi,y1,max(x1,x2),y2))
                    return out if out else []
                else:  # 垂直
                    lo=max(min(y1,y2),min(ry1,ry2)); hi=min(max(y1,y2),max(ry1,ry2))
                    out=[]
                    if min(y1,y2) < lo-2: out.append((x1,min(y1,y2),x2,lo))
                    if max(y1,y2) > hi+2: out.append((x1,hi,x2,max(y1,y2)))
                    return out if out else []

            sub_list = []
            for si, sr in enumerate(r['subrooms']):
                # 初始4条墙线段
                wall_segs = {side: [get_edge_seg(sr, side)] for side in ['top','bottom','left','right']}
                # 对其他子房间，减去共享部分（sr的每条边 vs other的每条边）
                for sj, other in enumerate(r['subrooms']):
                    if si == sj: continue
                    for sr_side in ['top','bottom','left','right']:
                        for other_side in ['top','bottom','left','right']:
                            oseg = get_edge_seg(other, other_side)
                            new_list = []
                            for ws in wall_segs[sr_side]:
                                subbed = subtract_seg(ws, oseg)
                                new_list.extend(subbed if subbed else [])
                            wall_segs[sr_side] = new_list

                # 检查每条边是否有剩余墙段（长度>2的）
                sr_walls = {}
                sr_wall_segs = []
                for side in ['top','bottom','left','right']:
                    sr_walls[side] = any(s[2]-s[0] > 2 or s[3]-s[1] > 2 for s in wall_segs[side])
                    for ws in wall_segs[side]:
                        if ws[2]-ws[0] > 2 or ws[3]-ws[1] > 2:
                            wx1=float(ws[0]); wy1=float(ws[1])
                            wx2=float(ws[2]); wy2=float(ws[3])
                            if abs(wy1-wy2) < 0.01:
                                sr_wall_segs.append({"x1":wx1,"y1":wy1,"x2":wx2,"y2":wy1})
                            else:
                                sr_wall_segs.append({"x1":wx1,"y1":wy1,"x2":wx1,"y2":wy2})

                srx_=float(sr['x']); sry_=float(sr['y'])
                srw_=float(sr['w']); srh_=float(sr['h'])
                sub_list.append({
                    "name":sr['id'],"position":{"x":srx_,"y":sry_,"width":srw_,"height":srh_},
                    "walls":sr_walls,
                    "wall_segments":sr_wall_segs
                })
            room_data["subrooms"] = sub_list
    yd["house"]["rooms"].append(room_data)
with open(f"{OUTPUT_DIR}/floorplan_real.yaml","w",encoding="utf-8") as f:
    yaml.dump(yd,f,sort_keys=False,default_flow_style=None,allow_unicode=True)
print(f"\nYAML: {OUTPUT_DIR}/floorplan_real.yaml")
