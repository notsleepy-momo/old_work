"""
ymj_TS 完整管线 — 唯一入口

输入 (来自 input/):
  1. 户型图 (机器人 APP 图 / greyroom.png)
  2. 智能设备位置坐标 (Smart_device/smartDevice_*.yaml, 世界坐标)
  3. 人的行为统计信息 (traj/*_trajectory.json, 世界坐标)

7 步处理后输出到 output/:
  Step 1 — 房间分割    (photo2yaml/run_seg.py, CV Hough线检测)
  Step 2 — 家具检测    (FurnitureDetectionAgent, CV+LLM)
  Step 3 — 坐标统一    (CoordinateConverter, 像素→世界)
  Step 4 — 房间类型推理 (RoomAgent, LLM)
  Step 5 — 行为模式分析 (BehaviorAgent, LLM)
  Step 6 — 家具命名    (FurnitureNamingAgent, LLM)
  Step 7 — 合并输出    (final_layout.yaml, 含房间/家具/行为)

输出: output/yaml/final_layout.yaml
  - 所有房间 (id, position, room_type, walls, doors, behavior)
  - 所有家具 (id, position, name, confidence)
  - 世界坐标

用法:
  # 使用默认输入 (input/ 目录)
  python main.py

  # 指定输入
  python main.py --image input/photo/room_real2.png \\
                 --smart-device input/Smart_device/smartDevice_0622.yaml \\
                 --trajectory input/traj/0622/0622-Engineer-all_trajectory.json

    python main.py --code 0622
  # 跳过分割 (使用已有的 floorplan_real.yaml)
  python main.py --skip-seg

  # 仅做分割+坐标统一 (跳过所有 LLM)
  python main.py --skip-agents
"""
import sys, os, logging, argparse
from dotenv import load_dotenv

# ─── 加载 .env 环境变量 ────────────────────────────────────
def _load_env():
    """加载 .env 文件，自动检测编码 (UTF-8 / UTF-16)"""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if not os.path.exists(env_path):
        return
    for enc in ('utf-8-sig', 'utf-16-le', 'utf-16'):
        try:
            load_dotenv(dotenv_path=env_path, encoding=enc)
            return
        except UnicodeDecodeError:
            continue
    # 所有编码都失败，用二进制跳过
    print("[警告] 无法解码 .env 文件，尝试以 UTF-8 读取（忽略错误）")
    load_dotenv(dotenv_path=env_path, encoding='utf-8', errors='ignore')

_load_env()

# ─── 路径设置 ────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from pipeline import Pipeline
from llm_config import DEFAULT_LLM_MODEL

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('main')


# ══════════════════════════════════════════════════════════════
# 默认输入路径
# ══════════════════════════════════════════════════════════════
DEFAULT_CODE = '0622'
DEFAULT_IMAGE = os.path.join(
    PROJECT_ROOT, 'input', 'photo', f'room_{DEFAULT_CODE}.png'
)
DEFAULT_SMART_DEVICE = os.path.join(
    PROJECT_ROOT, 'input', 'Smart_device', f'smartDevice_{DEFAULT_CODE}.yaml'
)
DEFAULT_TRAJECTORY = os.path.join(
    PROJECT_ROOT, 'input', 'traj', DEFAULT_CODE, f'{DEFAULT_CODE}-Engineer-all_trajectory.json'
)
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', DEFAULT_CODE)


def _get_env_or_exit(key: str, description: str) -> str:
    """从环境变量读取配置，缺失则报错退出"""
    val = os.environ.get(key)
    if not val:
        print(f"[错误] 环境变量 {key} 未设置 ({description})")
        print(f"  请在 .env 文件中添加: {key}=your_key_here")
        sys.exit(1)
    return val


def main():
    # ─── 从环境变量读取 API 配置（代码中不出现任何 key） ────
    api_key = _get_env_or_exit("FURNITURE_API_KEY", f"{DEFAULT_LLM_MODEL} API key")
    base_url = os.environ.get("FURNITURE_BASE_URL")

    parser = argparse.ArgumentParser(
        description='ymj_TS — 家庭轮廓完整管线 (户型图 → 房间分割 → 家具检测 → 类型推理 → 行为分析 → 家具命名 → YAML)'
    )
    parser.add_argument('--code', default=DEFAULT_CODE,
                        help=f'家庭编码，用于输入输出路径 (默认: {DEFAULT_CODE})')
    parser.add_argument('--image', default=None,
                        help=f'户型轨迹图 (默认: input/photo/room_{{code}}.png)')
    parser.add_argument('--smart-device', default=None,
                        help=f'智能设备 YAML (默认: input/Smart_device/smartDevice_{{code}}.yaml)')
    parser.add_argument('--trajectory', default=None,
                        help=f'行为轨迹 JSON (默认: input/traj/{{code}}/...)')
    parser.add_argument('--output-dir', default=None,
                        help=f'输出目录 (默认: output/{{code}}/)')
    parser.add_argument('--skip-seg', action='store_true',
                        help='跳过房间分割 (使用已有 01_seg_pixel_{code}.yaml)')
    parser.add_argument('--skip-agents', action='store_true',
                        help='跳过所有 LLM Agents')
    args = parser.parse_args()

    # 用 code 拼接默认路径
    code = args.code
    image = args.image or os.path.join(PROJECT_ROOT, 'input', 'photo', f'room_{code}.png')
    smart_device = args.smart_device or os.path.join(PROJECT_ROOT, 'input', 'Smart_device', f'smartDevice_{code}.yaml')
    traj_dir = os.path.join(PROJECT_ROOT, 'input', 'traj', code)
    traj = args.trajectory
    if not traj and os.path.isdir(traj_dir):
        jsons = sorted(
            [f for f in os.listdir(traj_dir) if f.endswith('.json')],
            key=lambda f: os.path.getsize(os.path.join(traj_dir, f)), reverse=True)
        if jsons:
            traj = os.path.join(traj_dir, jsons[0])
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, 'output', code)

    # 验证输入
    if not os.path.exists(image):
        print(f"[错误] 户型图不存在: {image}")
        sys.exit(1)
    if smart_device and not os.path.exists(smart_device):
        print(f"[警告] 智能设备文件不存在, 跳过: {smart_device}")
        smart_device = None
    if traj and not os.path.exists(traj):
        print(f"[警告] 轨迹文件不存在, 跳过: {traj}")
        traj = None

    # ── 运行 ──
    print("=" * 60)
    print("  ymj_TS — 家庭轮廓管线")
    print("=" * 60)
    print(f"  家庭编码:      {code}")
    print(f"  输入:")
    print(f"    户型图:      {image}")
    print(f"    智能设备:    {smart_device or '(无)'}")
    print(f"    行为轨迹:    {traj or '(无)'}")
    print(f"  输出目录:      {output_dir}")
    print(f"  跳过分割:      {args.skip_seg}")
    print(f"  跳过 LLM:      {args.skip_agents}")
    print(f"  LLM 模型:      {DEFAULT_LLM_MODEL}")
    print(f"  API key:       FURNITURE_API_KEY (来自 .env)")
    print("=" * 60)

    pipeline = Pipeline(
        api_key=api_key, base_url=base_url,
        model=DEFAULT_LLM_MODEL,
        code=code,
    )
    result = pipeline.run(
        image_path=image,
        smart_device_path=smart_device,
        trajectory_path=traj,
        output_dir=output_dir,
        skip_seg=args.skip_seg,
        skip_agents=args.skip_agents,
    )

    print(f"\n✅ 全流程完成!")
    print(f"   最终输出: {result}")
    print(f"   输出目录: {args.output_dir}")
    print(f"     masks/   — CV 中间可视化图")
    print(f"     yaml/    — 所有 YAML 中间产物")
    print(f"     viz/     — 全图可视化")


if __name__ == '__main__':
    main()
