import json
import re
import os

traj_dir = os.path.dirname(os.path.abspath(__file__))

subdirs = sorted([
    os.path.join(traj_dir, d)
    for d in os.listdir(traj_dir)
    if os.path.isdir(os.path.join(traj_dir, d)) and d.isdigit()
])

if not subdirs:
    subdirs = [traj_dir]

for subdir in subdirs:
    map_num = os.path.basename(subdir)
    prefix = f"{map_num}-"

    date_files = []
    for f in os.listdir(subdir):
        if not f.endswith('.json') or 'all' in f:
            continue
        m = re.match(rf"(?:layout)?{prefix}.+-(\d{{4}})(?:_trajectory)?\.json$", f)
        if m:
            date_str = m.group(1)
            date_files.append((date_str, os.path.join(subdir, f)))

    date_files.sort(key=lambda x: x[0])

    if not date_files:
        print(f"[{map_num}] No matching files")
        continue

    all_points = []
    next_id = 1

    for date_str, filepath in date_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            records = data
        elif isinstance(data, dict) and 'trajectories' in data:
            records = []
            for t in data['trajectories']:
                for p in t.get('points', []):
                    records.append({
                        "x": p.get("x", 0),
                        "y": p.get("y", 0),
                        "timestamp": p.get("timestamp", "")
                    })
        else:
            print(f"  [{map_num}] Skip {date_str}: unsupported format ({type(data).__name__})")
            continue

        for item in records:
            item["id"] = next_id
            next_id += 1
        all_points.extend(records)
        print(f"  [{map_num}] {date_str}: {len(records)} records")

    if not all_points:
        print(f"[{map_num}] No data extracted")
        continue

    output_path = os.path.join(subdir, f"{prefix}all_trajectory.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_points, f, ensure_ascii=False, indent=2)

    print(f"[{map_num}] Done: {len(all_points)} records -> {os.path.basename(output_path)}")
