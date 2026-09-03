# 6.负责家具命名
from typing import Dict, List
import json
import os
import yaml
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .data_models import FurnitureNaming, ValidationResult, FurnitureNamingGenerationResult, RoomAnalysis, BehaviorAnalysis
from .prompts import FURNITURE_AGENT_PROMPT
from config import load_furniture_allowed, get_smart_device_names
from llm_config import (
    DEFAULT_LLM_MODEL,
    require_shared_model,
    resolve_api_key,
    resolve_base_url,
)

logger = logging.getLogger(__name__)

# 从配置文件加载家具命名允许列表（首次加载后缓存）
FURNITURE_ALLOWED_LISTS = load_furniture_allowed()

class FurnitureNamingAgent:
    """Furniture naming agent"""
    
    # 从配置文件加载智能设备名称保护列表
    SMART_DEVICE_NAMES = get_smart_device_names()
    SMART_DEVICE_NAMES_LOWER = {n.lower() for n in get_smart_device_names()}

    @classmethod
    def _is_smart_device(cls, name: str) -> bool:
        """检查名称是否为智能设备（大小写不敏感）。"""
        n = name.strip().lower()
        return n in cls.SMART_DEVICE_NAMES_LOWER
    
    def __init__(self, api_key: str = None, base_url: str = None,
                 model: str = DEFAULT_LLM_MODEL, output_dir: str = None,
                 callbacks: list = None):
        """Initialize furniture naming agent"""
        common_llm_kwargs = {
            "api_key": resolve_api_key(api_key),
            "max_retries": 5,
            "timeout": 120,
        }
        resolved_base_url = resolve_base_url(base_url)
        if resolved_base_url:
            common_llm_kwargs["base_url"] = resolved_base_url
        if callbacks:
            common_llm_kwargs["callbacks"] = callbacks
        
        self.prompt = ChatPromptTemplate.from_template(FURNITURE_AGENT_PROMPT)
        self.llm = ChatOpenAI(
            model=require_shared_model(model), temperature=0.0,
            **common_llm_kwargs)
        self.chain = self.prompt | self.llm
        self._output_dir = output_dir
    
    def generate_furniture_namings(self, house_layout: Dict, room_analyses: List[RoomAnalysis], behavior_analyses: List[BehaviorAnalysis], ablation: dict = None) -> FurnitureNamingGenerationResult:
        """Generate furniture namings
        
        智能设备名称保护：已有智能设备名称的家具不参与 LLM 命名，
        直接保留原名称并标记高置信度。
        
        ablation:
            - skip_allowed_list: 跳过家具允许名单约束
            - skip_shape_constraint: 跳过形状约束
        """
        ablation = ablation or {}
        self._ablation = ablation
        try:
            # ─── 智能设备名称保护：提取已有名称的家具，从待命名列表中移除 ───
            preserved_namings = []
            unnamed_furniture_map = {}  # {furniture_id: {room_id, position, ...}}
            
            for room in house_layout.get('house', {}).get('rooms', []):
                room_id = room.get('id')
                for fur in room.get('furniture', []):
                    fid = fur.get('id')
                    existing_name = (fur.get('name', '') or '').strip()
                    if self._is_smart_device(existing_name):
                        preserved_namings.append(FurnitureNaming(
                            furniture_id=fid,
                            name=existing_name,
                            room_id=room_id,
                            confidence=0.95
                        ))
                        logger.info(f"  智能设备名称保护: {fid} -> {existing_name} (跳过 LLM 命名)")
                    else:
                        unnamed_furniture_map[fid] = {
                            'room_id': room_id,
                            'position': fur.get('position', {})
                        }
        except Exception as e:
            logger.error(f"Furniture naming failed during smart device protection: {str(e)}")
            # 回退：直接使用默认命名
            default_namings = self._get_default_furniture_namings(house_layout, room_analyses)
            return FurnitureNamingGenerationResult(furniture_namings=default_namings, confidence=0.5)
        
        # 如果所有家具都已有智能设备名，直接返回
        if not unnamed_furniture_map:
            return FurnitureNamingGenerationResult(
                furniture_namings=preserved_namings, confidence=0.95
            )
        
        # ─── 构建仅含未命名家具的布局数据发给 LLM ───
        # 同时预计算结构特征（墙距、位置类型等），避免 LLM 计算错误
        naming_layout = json.loads(json.dumps(house_layout))
        structured_features = []  # 供 prompt 使用的结构化特征文本
        for room in naming_layout.get('house', {}).get('rooms', []):
            room_pos = room.get('position', {})
            rx, ry = float(room_pos.get('x', 0)), float(room_pos.get('y', 0))
            rw, rh = float(room_pos.get('width', 1)), float(room_pos.get('height', 1))
            room['furniture'] = [
                fur for fur in room.get('furniture', [])
                if fur.get('id') in unnamed_furniture_map
            ]
            for fur in room.get('furniture', []):
                fid = fur.get('id', '?')
                fpos = fur.get('position', {})
                fx, fy = float(fpos.get('x', 0)), float(fpos.get('y', 0))
                fw, fh = float(fpos.get('width', 1)), float(fpos.get('height', 1))
                area = fw * fh
                ratio = max(fw, fh) / (min(fw, fh) + 0.1)
                # 墙距
                dl, dr = fx - rx, (rx + rw) - (fx + fw)
                db, dt = fy - ry, (ry + rh) - (fy + fh)
                # 位置类型
                wall_contacts = sum(1 for d in (dl, dr, db, dt) if d <= 2)
                if wall_contacts >= 2:
                    placement = 'corner'
                elif wall_contacts == 1:
                    placement = 'against_wall'
                elif min(dl, dr, db, dt) <= 6:
                    placement = 'near_wall'
                else:
                    placement = 'center'
                # 长边接触
                long_side = 'width' if fw >= fh else 'height'
                long_side_wall = ''
                if long_side == 'height' and dl <= 2:
                    long_side_wall = 'left'
                elif long_side == 'height' and dr <= 2:
                    long_side_wall = 'right'
                elif long_side == 'width' and db <= 2:
                    long_side_wall = 'bottom'
                elif long_side == 'width' and dt <= 2:
                    long_side_wall = 'top'
                
                fur['_features'] = {
                    'area': round(area, 1),
                    'aspect_ratio': round(ratio, 2),
                    'placement': placement,
                    'wall_dist': f'L={dl:.0f} R={dr:.0f} B={db:.0f} T={dt:.0f}',
                    'long_side': long_side,
                    'long_side_wall': long_side_wall or 'none',
                }
                structured_features.append(
                    f"  {fid}: area={area:.0f} ratio={ratio:.2f} placement={placement} "
                    f"wall_dist=(L:{dl:.0f} R:{dr:.0f} B:{db:.0f} T:{dt:.0f}) "
                    f"long_side={long_side}" +
                    (f" touches_{long_side_wall}_wall" if long_side_wall else "")
                )
        
        try:
            # Convert data to strings
            house_layout_yaml = yaml.dump(naming_layout)
            room_analyses_json = json.dumps([room.dict() for room in room_analyses])
            behavior_analyses_json = json.dumps([behavior.dict() for behavior in behavior_analyses])
            
            # Invoke LLM
            result = self.chain.invoke({
                "house_layout_yaml": house_layout_yaml,
                "room_analyses": room_analyses_json,
                "behavior_analyses": behavior_analyses_json,
                "structured_features": "\n".join(structured_features),
            })
            
            # ─── 保存 LLM 原始推理输出到文件 ───
            if self._output_dir and hasattr(result, 'content') and result.content:
                try:
                    os.makedirs(self._output_dir, exist_ok=True)
                    response_path = os.path.join(self._output_dir, 'furniture_naming_llm_response.txt')
                    with open(response_path, 'w', encoding='utf-8') as f:
                        f.write(result.content)
                    logger.info(f"LLM 推理输出已保存: {response_path}")
                except Exception as save_err:
                    logger.warning(f"保存 LLM 推理输出失败: {save_err}")
            
            # Debug: print the result content
            logger.debug(f"LLM response type: {type(result)}")
            logger.debug(f"LLM response: {result}")
            
            # Check if result has content
            if not hasattr(result, 'content'):
                logger.error(f"LLM response has no content attribute: {result}")
                # Return default furniture namings (合并保留的智能设备名)
                default_namings = self._get_default_furniture_namings(house_layout, room_analyses)
                all_namings = preserved_namings + [n for n in default_namings if n.furniture_id in unnamed_furniture_map]
                return FurnitureNamingGenerationResult(furniture_namings=all_namings, confidence=0.5)
            
            # Debug: print the result content
            logger.debug(f"LLM response content: {result.content}")
            logger.debug(f"LLM response content type: {type(result.content)}")
            logger.debug(f"LLM response content length: {len(result.content) if result.content else 0}")
            
            # Parse result
            if not result.content:
                logger.error("LLM returned empty content")
                # Return default furniture namings (合并保留的智能设备名)
                default_namings = self._get_default_furniture_namings(house_layout, room_analyses)
                all_namings = preserved_namings + [n for n in default_namings if n.furniture_id in unnamed_furniture_map]
                return FurnitureNamingGenerationResult(furniture_namings=all_namings, confidence=0.5)
            
            # Try to extract JSON from response
            import re
            # Look for JSON array or object
            json_match = re.search(r'\[.*?\]|\{.*?\}', result.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                logger.debug(f"Extracted JSON: {json_str}")
                try:
                    furniture_namings_data = json.loads(json_str)
                    furniture_namings = [FurnitureNaming(**naming) for naming in furniture_namings_data]
                    all_namings = preserved_namings + furniture_namings
                    total = len(all_namings)
                    conf_sum = sum(n.confidence for n in all_namings)
                    confidence = conf_sum / total if total > 0 else 0.5
                    return FurnitureNamingGenerationResult(furniture_namings=all_namings, confidence=confidence)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"JSON parsing failed: {str(e)}")
                    # Return default furniture namings (合并保留的智能设备名)
                    default_namings = self._get_default_furniture_namings(house_layout, room_analyses)
                    all_namings = preserved_namings + [n for n in default_namings if n.furniture_id in unnamed_furniture_map]
                    return FurnitureNamingGenerationResult(furniture_namings=all_namings, confidence=0.5)
            else:
                # Try to parse the entire content
                logger.debug(f"Trying to parse entire content as JSON")
                try:
                    furniture_namings_data = json.loads(result.content)
                    furniture_namings = [FurnitureNaming(**naming) for naming in furniture_namings_data]
                    all_namings = preserved_namings + furniture_namings
                    total = len(all_namings)
                    conf_sum = sum(n.confidence for n in all_namings)
                    confidence = conf_sum / total if total > 0 else 0.5
                    return FurnitureNamingGenerationResult(furniture_namings=all_namings, confidence=confidence)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"JSON parsing failed: {str(e)}")
                    # Return default furniture namings (合并保留的智能设备名)
                    default_namings = self._get_default_furniture_namings(house_layout, room_analyses)
                    all_namings = preserved_namings + [n for n in default_namings if n.furniture_id in unnamed_furniture_map]
                    return FurnitureNamingGenerationResult(furniture_namings=all_namings, confidence=0.5)
        except Exception as e:
            logger.error(f"Furniture naming failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Return default furniture namings (合并保留的智能设备名)
            default_namings = self._get_default_furniture_namings(house_layout, room_analyses)
            all_namings = preserved_namings + [n for n in default_namings if n.furniture_id in unnamed_furniture_map]
            return FurnitureNamingGenerationResult(furniture_namings=all_namings, confidence=0.5)
    
    def validate_furniture_namings(self, furniture_namings: List[FurnitureNaming], house_layout: Dict, room_analyses: List[RoomAnalysis]) -> ValidationResult:
        """Validate furniture namings"""
        errors = []
        suggestions = []
        
        try:
            # Create room type map
            room_type_map = {room.room_id: room.room_type for room in room_analyses}
            
            # Check for duplicate names in each room
            room_furniture_names = {}
            for naming in furniture_namings:
                room_id = naming.room_id
                if room_id not in room_furniture_names:
                    room_furniture_names[room_id] = []
                if naming.name in room_furniture_names[room_id]:
                    errors.append(f"Duplicate furniture name '{naming.name}' in room {room_id}")
                    suggestions.append(f"Use a different name for furniture {naming.furniture_id} in room {room_id}")
                else:
                    room_furniture_names[room_id].append(naming.name)
            
            # Check if furniture names are from allowed lists
            skip_allowed = getattr(self, '_ablation', {}).get('skip_allowed_list', False)
            if not skip_allowed:
                for naming in furniture_namings:
                    room_id = naming.room_id
                    room_type = room_type_map.get(room_id, "other")
                    allowed_furniture = FURNITURE_ALLOWED_LISTS.get(room_type, FURNITURE_ALLOWED_LISTS["other"])
                    if naming.name.lower() in ("none", "unknown", ""):
                        errors.append(f"Furniture name '{naming.name}' is a placeholder (none/unknown/empty) for {naming.furniture_id} in room {room_id}")
                        suggestions.append(f"Replace placeholder name with a valid name from: {allowed_furniture}")
                    elif naming.name not in allowed_furniture:
                        errors.append(f"Furniture name '{naming.name}' not in allowed list for {room_type} in room {room_id}")
                        suggestions.append(f"Use a name from the allowed list: {allowed_furniture}")
            
            # Check if confidence scores are valid
            for naming in furniture_namings:
                if not 0 <= naming.confidence <= 1:
                    errors.append(f"Invalid confidence score {naming.confidence} for furniture {naming.furniture_id}")
                    suggestions.append("Confidence score should be between 0 and 1")
            
            # Check if all furniture items are named
            furniture_ids = []
            for room in house_layout.get('house', {}).get('rooms', []):
                for furniture in room.get('furniture', []):
                    furniture_ids.append(furniture.get('id'))
            
            named_furniture_ids = [naming.furniture_id for naming in furniture_namings]
            for furniture_id in furniture_ids:
                if furniture_id not in named_furniture_ids:
                    errors.append(f"Furniture {furniture_id} not named")
                    suggestions.append(f"Add naming for furniture {furniture_id}")
            
            is_valid = len(errors) == 0
            return ValidationResult(is_valid=is_valid, errors=errors, suggestions=suggestions)
        except Exception as e:
            logger.error(f"Furniture naming validation failed: {str(e)}")
            return ValidationResult(is_valid=False, errors=[str(e)], suggestions=["Fix the validation process"])
    
    def _score_candidate_name(self, name: str, area: float, aspect_ratio: float) -> float:
        """根据家具实际尺寸为候选名称打分，分数越高越匹配。"""
        # 典型家具尺寸特征 (面积 dm², 长宽比 width/height)
        profiles = {
            'bed':          lambda a, r: self._match(a, 150, 400, r, 0.5, 1.5),
            'wardrobe':     lambda a, r: self._match(a, 80, 250, r, 1.0, 3.0),
            'sofa':         lambda a, r: self._match(a, 60, 300, r, 1.5, 5.0),
            'dining_table': lambda a, r: self._match(a, 80, 250, r, 0.6, 1.8),
            'desk':         lambda a, r: self._match(a, 30, 150, r, 0.8, 3.0),
            'coffee_table': lambda a, r: self._match(a, 30, 120, r, 0.6, 2.5),
            'tv_stand':     lambda a, r: self._match(a, 30, 150, r, 1.8, 5.0),
            'bookshelf':    lambda a, r: self._match(a, 30, 150, r, 1.0, 3.0),
            'cabinet':      lambda a, r: self._match(a, 30, 120, r, 0.5, 3.0),
            'shoe_cabinet': lambda a, r: self._match(a, 10, 60, r, 1.0, 3.0),
            'nightstand':   lambda a, r: self._match(a, 15, 60, r, 0.5, 2.0),
            'chair':        lambda a, r: self._match(a, 10, 50, r, 0.5, 2.0),
            'table':        lambda a, r: self._match(a, 20, 200, r, 0.5, 2.5),
            'shelf':        lambda a, r: self._match(a, 15, 80, r, 1.5, 4.0),
            'toilet':       lambda a, r: self._match(a, 15, 60, r, 0.5, 2.0),
            'washbasin':    lambda a, r: self._match(a, 15, 50, r, 0.8, 2.5),
            'shower':       lambda a, r: self._match(a, 30, 100, r, 0.5, 1.8),
            'kitchen_countertop': lambda a, r: self._match(a, 40, 200, r, 1.5, 5.0),
            'coat_rack':    lambda a, r: self._match(a, 5, 30, r, 0.5, 2.0),
        }
        scorer = profiles.get(name)
        return scorer(area, aspect_ratio) if scorer else 0.0

    @staticmethod
    def _match(value: float, lo: float, hi: float, ratio: float, r_lo: float, r_hi: float) -> float:
        """评分：value 在 [lo, hi] 内得高分，ratio 在 [r_lo, r_hi] 内加分。"""
        area_score = 1.0 if lo <= value <= hi else max(0.0, 1.0 - min(abs(value - lo), abs(value - hi)) / max(lo, 1))
        ratio_score = 1.0 if r_lo <= ratio <= r_hi else max(0.0, 1.0 - min(abs(ratio - r_lo), abs(ratio - r_hi)) / max(r_lo, 0.1))
        return area_score * 0.6 + ratio_score * 0.4

    def fix_furniture_namings(self, furniture_namings: List[FurnitureNaming], house_layout: Dict, room_analyses: List[RoomAnalysis]) -> List[FurnitureNaming]:
        """Fix furniture namings"""
        skip_allowed = getattr(self, '_ablation', {}).get('skip_allowed_list', False)
        try:
            # Create room type map
            room_type_map = {room.room_id: room.room_type for room in room_analyses}
            
            # 构建家具尺寸查找表: furniture_id → (area, aspect_ratio)
            fur_size_map = {}
            for room in house_layout.get('house', {}).get('rooms', []):
                for fur in room.get('furniture', []):
                    fid = fur.get('id', '')
                    pos = fur.get('position', {})
                    w = float(pos.get('width', 1) or 1)
                    h = float(pos.get('height', 1) or 1)
                    fur_size_map[fid] = (w * h, w / h if h > 0 else 1.0)
            
            # Get all furniture IDs from house layout
            all_furniture = []
            for room in house_layout.get('house', {}).get('rooms', []):
                room_id = room.get('id')
                for furniture in room.get('furniture', []):
                    all_furniture.append((room_id, furniture.get('id')))
            
            # Create a map of existing namings
            existing_namings = {naming.furniture_id: naming for naming in furniture_namings}
            
            # Add missing furniture namings
            for room_id, furniture_id in all_furniture:
                if furniture_id not in existing_namings:
                    room_type = room_type_map.get(room_id, "other")
                    allowed_furniture = FURNITURE_ALLOWED_LISTS.get(room_type, FURNITURE_ALLOWED_LISTS["other"])
                    # Use a default name from the allowed list (or "furniture" if skip_allowed)
                    default_name = allowed_furniture[0] if not skip_allowed and allowed_furniture else "furniture"
                    furniture_namings.append(FurnitureNaming(
                        furniture_id=furniture_id,
                        name=default_name,
                        room_id=room_id,
                        confidence=0.5
                    ))
            
            # Fix duplicate names in each room — keep name for the piece with higher confidence
            # First pass: collect all pieces grouped by room and detect duplicates
            room_pieces = {}  # {room_id: [(naming, index), ...]}
            for idx, naming in enumerate(furniture_namings):
                room_pieces.setdefault(naming.room_id, []).append((naming, idx))
            
            for room_id, pieces in room_pieces.items():
                name_to_pieces = {}  # {name: [(naming, idx, confidence), ...]}
                for naming, idx in pieces:
                    name_to_pieces.setdefault(naming.name, []).append((naming, idx, naming.confidence))
                
                for name, entries in name_to_pieces.items():
                    if len(entries) <= 1:
                        continue  # no duplicate
                    # Sort by confidence descending — keep name for highest confidence, rename rest
                    entries.sort(key=lambda x: x[2], reverse=True)
                    loser_entries = entries[1:]  # lower confidence pieces get renamed
                    
                    room_type = room_type_map.get(room_id, "other")
                    allowed_furniture = FURNITURE_ALLOWED_LISTS.get(room_type, FURNITURE_ALLOWED_LISTS["other"])
                    used_names = {n for n, _, _ in entries}
                    
                    for naming, _, _ in loser_entries:
                        # 尺寸感知重命名: 按家具实际尺寸匹配度排序候选名
                        fid = naming.furniture_id
                        area, ratio = fur_size_map.get(fid, (100.0, 1.0))
                        # 从允许列表中找出未使用名，按尺寸匹配度排序
                        candidates = [(c, self._score_candidate_name(c, area, ratio))
                                     for c in allowed_furniture if c not in used_names]
                        candidates.sort(key=lambda x: -x[1])  # 高分优先
                        if candidates and candidates[0][1] > 0:
                            candidate = candidates[0][0]
                        else:
                            # 无可用名或得分全为 0，回退到顺序取第一个
                            for candidate in allowed_furniture:
                                if candidate not in used_names:
                                    break
                            else:
                                # No unused name, keep with suffix
                                i = 2
                                while f"{name}_{i}" in used_names:
                                    i += 1
                                naming.name = f"{name}_{i}"
                                continue
                        old_name = naming.name
                        naming.name = candidate
                        naming.confidence = max(0.3, naming.confidence - 0.2)
                        used_names.add(candidate)
                        logger.info(f"  去重: {naming.furniture_id} {old_name} → {candidate} "
                                    f"(同房间已存在更高置信度的 {name})")
            
            # Fix invalid furniture names (skip if skip_allowed)
            if not skip_allowed:
                for naming in furniture_namings:
                    room_id = naming.room_id
                    room_type = room_type_map.get(room_id, "other")
                    allowed_furniture = FURNITURE_ALLOWED_LISTS.get(room_type, FURNITURE_ALLOWED_LISTS["other"])
                    if naming.name not in allowed_furniture:
                        # 尺寸感知：按匹配度选最佳候选名
                        fid = naming.furniture_id
                        area, ratio = fur_size_map.get(fid, (100.0, 1.0))
                        best = max(allowed_furniture, key=lambda c: self._score_candidate_name(c, area, ratio))
                        naming.name = best if best else (allowed_furniture[0] if allowed_furniture else "table")
                        naming.confidence = 0.5
            
            # Fix invalid confidence scores
            for naming in furniture_namings:
                if not 0 <= naming.confidence <= 1:
                    naming.confidence = 0.5
            
            # ─── 硬几何后处理：纠正 LLM 违反物理约束的命名 ───
            furniture_namings = self._apply_hard_geometry_rules(
                furniture_namings, house_layout, room_type_map)
            
            return furniture_namings
        except Exception as e:
            logger.error(f"Furniture naming fixing failed: {str(e)}")
            return self._get_default_furniture_namings(house_layout, room_analyses)
    
    def _apply_hard_geometry_rules(
        self, furniture_namings: List[FurnitureNaming],
        house_layout: Dict, room_type_map: Dict[str, str]
    ) -> List[FurnitureNaming]:
        """硬几何后处理规则：纠正 LLM 违反基本物理约束的命名。
        
        这些规则基于预计算的几何特征（面积、长宽比、墙距、位置类型），
        不依赖 LLM 判断，保证确定性输出。
        
        规则：
        1. 卫生间 toilet/washbasin：corner + min wall distance 小 → toilet
        2. 客厅 sofa：area>100 + ratio≥1.8 + 长边贴墙 → 必须是 sofa
        3. 客厅 tv_stand/dining_table：贴墙 + ratio≥1.5 + 非sofa → tv_stand
        """
        try:
            # ── 构建几何特征表：furniture_id → {area, ratio, placement, min_wall_dist, long_side_wall} ──
            geo_features = {}
            for room in house_layout.get('house', {}).get('rooms', []):
                room_pos = room.get('position', {})
                rx = float(room_pos.get('x', 0))
                ry = float(room_pos.get('y', 0))
                rw = float(room_pos.get('width', 1))
                rh = float(room_pos.get('height', 1))
                for fur in room.get('furniture', []):
                    fid = fur.get('id', '')
                    fpos = fur.get('position', {})
                    fx = float(fpos.get('x', 0))
                    fy = float(fpos.get('y', 0))
                    fw = float(fpos.get('width', 1))
                    fh = float(fpos.get('height', 1))
                    area = fw * fh
                    ratio = max(fw, fh) / (min(fw, fh) + 0.1)
                    dl = fx - rx
                    dr = (rx + rw) - (fx + fw)
                    db = fy - ry
                    dt = (ry + rh) - (fy + fh)
                    min_wall = min(dl, dr, db, dt)
                    wall_contacts = sum(1 for d in (dl, dr, db, dt) if d <= 2)
                    if wall_contacts >= 2:
                        placement = 'corner'
                    elif wall_contacts == 1:
                        placement = 'against_wall'
                    elif min_wall <= 6:
                        placement = 'near_wall'
                    else:
                        placement = 'center'
                    long_side = 'width' if fw >= fh else 'height'
                    long_side_wall = False
                    if long_side == 'height' and (dl <= 2 or dr <= 2):
                        long_side_wall = True
                    elif long_side == 'width' and (db <= 2 or dt <= 2):
                        long_side_wall = True
                    geo_features[fid] = {
                        'area': area, 'ratio': ratio, 'placement': placement,
                        'min_wall_dist': min_wall, 'long_side_wall': long_side_wall,
                    }

            naming_map = {n.furniture_id: n for n in furniture_namings}

            # ── 规则 1: 卫生间 toilet/washbasin 硬纠错 ──
            for room in house_layout.get('house', {}).get('rooms', []):
                room_id = room.get('id', '')
                room_type = room_type_map.get(room_id, '')
                if room_type != 'bathroom':
                    continue

                bath_namings = []
                for fur in room.get('furniture', []):
                    fid = fur.get('id', '')
                    if fid in naming_map:
                        bath_namings.append((fid, naming_map[fid]))

                toilet_candidates = [n for fid, n in bath_namings if n.name == 'toilet']
                washbasin_candidates = [n for fid, n in bath_namings if n.name == 'washbasin']

                if len(toilet_candidates) == 1 and len(washbasin_candidates) == 1:
                    t_naming = toilet_candidates[0]
                    w_naming = washbasin_candidates[0]
                    t_feat = geo_features.get(t_naming.furniture_id, {})
                    w_feat = geo_features.get(w_naming.furniture_id, {})

                    t_is_corner = t_feat.get('placement') == 'corner'
                    w_is_corner = w_feat.get('placement') == 'corner'
                    t_min_wall = t_feat.get('min_wall_dist', 99)
                    w_min_wall = w_feat.get('min_wall_dist', 99)

                    t_score = (1 if t_is_corner else 0) + (1 if t_min_wall <= 0 else 0)
                    w_score = (1 if w_is_corner else 0) + (1 if w_min_wall <= 0 else 0)

                    if w_score > t_score or (w_score == t_score and w_min_wall < t_min_wall):
                        t_naming.name, w_naming.name = w_naming.name, t_naming.name
                        t_naming.confidence = 0.9
                        w_naming.confidence = 0.9
                        logger.info(
                            f"  [硬规则] bathroom 纠错: {t_naming.furniture_id}↔{w_naming.furniture_id} "
                            f"toilet/washbasin 交换 (corner={w_is_corner}/{t_is_corner}, "
                            f"min_wall={w_min_wall:.0f}/{t_min_wall:.0f})")

            # ── 规则 2: 客厅 sofa 硬约束 + 规则 3: tv_stand 约束 ──
            for room in house_layout.get('house', {}).get('rooms', []):
                room_id = room.get('id', '')
                room_type = room_type_map.get(room_id, '')
                if room_type != 'living_room':
                    continue

                for fur in room.get('furniture', []):
                    fid = fur.get('id', '')
                    if fid not in naming_map:
                        continue
                    naming = naming_map[fid]
                    feat = geo_features.get(fid, {})
                    area = feat.get('area', 0)
                    ratio = feat.get('ratio', 1)
                    placement = feat.get('placement', '')
                    long_side_wall = feat.get('long_side_wall', False)

                    # 规则 2: 必须是 sofa
                    if (area > 100 and ratio >= 1.8 and long_side_wall
                            and naming.name not in ('sofa',)
                            and not self._is_smart_device(naming.name)):
                        old = naming.name
                        naming.name = 'sofa'
                        naming.confidence = 0.95
                        logger.info(
                            f"  [硬规则] living_room 纠错: {fid} {old} → sofa "
                            f"(area={area:.0f}>100, ratio={ratio:.2f}>=1.8, long_side_wall=True)")
                        continue

                    # 规则 3: tv_stand vs dining_table — 贴墙的 elongated 非沙发 → tv_stand
                    if (ratio >= 1.5 and 30 < area < 150
                            and placement in ('against_wall', 'corner')
                            and naming.name == 'dining_table'
                            and not self._is_smart_device(naming.name)):
                        naming.name = 'tv_stand'
                        naming.confidence = 0.85
                        logger.info(
                            f"  [硬规则] living_room 纠错: {fid} dining_table → tv_stand "
                            f"(against_wall, area={area:.0f}, ratio={ratio:.2f}>=1.5)")
        except Exception as e:
            logger.warning(f"[硬规则] 几何后处理异常，保留原始命名: {e}")

        return furniture_namings
    
    def name_furniture(self, house_layout: Dict, room_analyses: List[RoomAnalysis], behavior_analyses: List[BehaviorAnalysis], ablation: dict = None) -> List[FurnitureNaming]:
        """Name furniture items"""
        # Generate furniture namings
        generation_result = self.generate_furniture_namings(house_layout, room_analyses, behavior_analyses, ablation=ablation)
        furniture_namings = generation_result.furniture_namings
        
        # Validate furniture namings
        validation_result = self.validate_furniture_namings(furniture_namings, house_layout, room_analyses)
        
        # Fix furniture namings if invalid
        if not validation_result.is_valid:
            logger.warning(f"Furniture namings validation failed: {validation_result.errors}")
            furniture_namings = self.fix_furniture_namings(furniture_namings, house_layout, room_analyses)
        
        return furniture_namings
    
    def _get_default_furniture_namings(self, house_layout: Dict, room_analyses: List[RoomAnalysis]) -> List[FurnitureNaming]:
        """Get default furniture namings when LLM fails
        
        智能设备名称保护：已有智能设备名的家具保留原名，不覆盖。
        """
        furniture_namings = []
        rooms = house_layout.get('house', {}).get('rooms', [])
        room_type_map = {room.room_id: room.room_type for room in room_analyses}
        for room in rooms:
            room_id = room.get('id')
            room_type = room_type_map.get(room_id, 'living_room')
            allowed_furniture = FURNITURE_ALLOWED_LISTS.get(room_type, FURNITURE_ALLOWED_LISTS['other'])

            unnamed_idx = 0  # 仅对无名家具递增编号
            for furniture in room.get('furniture', []):
                furniture_id = furniture.get('id')
                existing_name = (furniture.get('name', '') or '').strip()
                
                # 智能设备名称保护
                if self._is_smart_device(existing_name):
                    furniture_namings.append(FurnitureNaming(
                        furniture_id=furniture_id,
                        name=existing_name,
                        room_id=room_id,
                        confidence=0.95
                    ))
                else:
                    default_name = allowed_furniture[unnamed_idx % len(allowed_furniture)]
                    furniture_namings.append(FurnitureNaming(
                        furniture_id=furniture_id,
                        name=default_name,
                        room_id=room_id,
                        confidence=0.5
                    ))
                    unnamed_idx += 1
        return furniture_namings
