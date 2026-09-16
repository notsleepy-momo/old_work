# 4.负责房间类型推理
from typing import Dict, List
import json
import yaml
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .data_models import RoomAnalysis, ValidationResult, RoomGenerationResult
from .prompts import ROOM_AGENT_PROMPT
from config import load_smart_devices
from llm_config import (
    DEFAULT_LLM_MODEL,
    require_shared_model,
    resolve_api_key,
    resolve_base_url,
)

logger = logging.getLogger(__name__)

class RoomAgent:
    """Room agent for inferring room types"""
    
    # 从配置文件加载智能设备映射（首次加载后缓存）
    FURNITURE_PRIORITY_MAPPINGS = load_smart_devices()
    
    def __init__(self, api_key: str = None, base_url: str = None,
                 model: str = DEFAULT_LLM_MODEL, callbacks: list = None):
        """Initialize room agent"""
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
        
        self.prompt = ChatPromptTemplate.from_template(ROOM_AGENT_PROMPT)
        self.llm = ChatOpenAI(
            model=require_shared_model(model), temperature=0.0,
            **common_llm_kwargs)
        self.chain = self.prompt | self.llm
    
    def generate_room_analyses(self, house_layout: Dict, trajectory_data: any) -> RoomGenerationResult:
        """Generate room analyses"""
        try:
            # Validate and normalize trajectory data
            if not trajectory_data:
                logger.error("Empty trajectory data")
                default_analyses = self._get_default_room_analyses(house_layout)
                return RoomGenerationResult(room_analyses=default_analyses, confidence=0.5)
            
            # Ensure trajectory_data is a list
            if isinstance(trajectory_data, dict):
                # If it's a dict, check if it has a 'trajectories' or similar key
                if 'trajectories' in trajectory_data:
                    trajectory_data = trajectory_data['trajectories']
                else:
                    # Convert dict to list
                    trajectory_data = [trajectory_data]
            elif not isinstance(trajectory_data, list):
                logger.error(f"Invalid trajectory data type: {type(trajectory_data)}")
                default_analyses = self._get_default_room_analyses(house_layout)
                return RoomGenerationResult(room_analyses=default_analyses, confidence=0.5)
            
            # Add furniture information to house layout
            augmented_house_layout = self._augment_house_layout_with_furniture_info(house_layout)
            
            # Add activity information to house layout
            augmented_house_layout = self._augment_house_layout_with_activity_info(augmented_house_layout, trajectory_data)
            
            # Convert data to strings
            house_layout_yaml = yaml.dump(augmented_house_layout)
            trajectory_data_json = json.dumps(trajectory_data)
            
            # Invoke LLM
            result = self.chain.invoke({
                "house_layout_yaml": house_layout_yaml,
                "trajectory_data": trajectory_data_json
            })
            
            # Debug: print the result content
            logger.debug(f"LLM response type: {type(result)}")
            logger.debug(f"LLM response: {result}")
            
            # Check if result has content
            if not hasattr(result, 'content'):
                logger.error(f"LLM response has no content attribute: {result}")
                # Return default room analyses
                default_analyses = self._get_default_room_analyses(house_layout)
                return RoomGenerationResult(room_analyses=default_analyses, confidence=0.5)
            
            # Debug: print the result content
            logger.debug(f"LLM response content: {result.content}")
            logger.debug(f"LLM response content type: {type(result.content)}")
            logger.debug(f"LLM response content length: {len(result.content) if result.content else 0}")
            
            # Parse result
            if not result.content:
                logger.error("LLM returned empty content")
                # Return default room analyses
                default_analyses = self._get_default_room_analyses(house_layout)
                return RoomGenerationResult(room_analyses=default_analyses, confidence=0.5)
            
            # Try to extract JSON from response
            import re
            # Look for JSON array or object
            json_match = re.search(r'\[.*?\]|\{.*?\}', result.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                logger.debug(f"Extracted JSON: {json_str}")
                try:
                    room_analyses_data = json.loads(json_str)
                    room_analyses = [RoomAnalysis(**room) for room in room_analyses_data]
                    confidence = sum([room.confidence for room in room_analyses]) / len(room_analyses) if room_analyses else 0.5
                    return RoomGenerationResult(room_analyses=room_analyses, confidence=confidence)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"JSON parsing failed: {str(e)}")
                    # Return default room analyses
                    default_analyses = self._get_default_room_analyses(house_layout)
                    return RoomGenerationResult(room_analyses=default_analyses, confidence=0.5)
            else:
                # Try to parse the entire content
                logger.debug(f"Trying to parse entire content as JSON")
                try:
                    room_analyses_data = json.loads(result.content)
                    room_analyses = [RoomAnalysis(**room) for room in room_analyses_data]
                    confidence = sum([room.confidence for room in room_analyses]) / len(room_analyses) if room_analyses else 0.5
                    return RoomGenerationResult(room_analyses=room_analyses, confidence=confidence)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"JSON parsing failed: {str(e)}")
                    # Return default room analyses
                    default_analyses = self._get_default_room_analyses(house_layout)
                    return RoomGenerationResult(room_analyses=default_analyses, confidence=0.5)
        except Exception as e:
            logger.error(f"Room analysis failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Return default room analyses
            default_analyses = self._get_default_room_analyses(house_layout)
            return RoomGenerationResult(room_analyses=default_analyses, confidence=0.5)
    
    def validate_room_analyses(self, room_analyses: List[RoomAnalysis], house_layout: Dict) -> ValidationResult:
        """Validate room analyses"""
        errors = []
        suggestions = []
        
        try:
            # Get all room IDs from house layout
            room_ids = [room.get('id') for room in house_layout.get('house', {}).get('rooms', [])]
            
            # Check if all rooms are analyzed
            analyzed_room_ids = [analysis.room_id for analysis in room_analyses]
            for room_id in room_ids:
                if room_id not in analyzed_room_ids:
                    errors.append(f"Room {room_id} not analyzed")
                    suggestions.append(f"Add analysis for room {room_id}")
            
            # Check if room types are valid
            valid_room_types = ["bedroom", "living_room", "kitchen", "bathroom", "studyroom","diningroom","balcony","other","unknown"]
            for analysis in room_analyses:
                if analysis.room_type not in valid_room_types:
                    errors.append(f"Invalid room type {analysis.room_type} for room {analysis.room_id}")
                    suggestions.append(f"Use a valid room type from: {valid_room_types}")
            
            # Check if confidence scores are valid
            for analysis in room_analyses:
                if not 0 <= analysis.confidence <= 1:
                    errors.append(f"Invalid confidence score {analysis.confidence} for room {analysis.room_id}")
                    suggestions.append("Confidence score should be between 0 and 1")
            
            # Check if all required room types are present
            # bathroom count depends on total room count:
            #   < 8 rooms → exactly 1 bathroom
            #   >= 8 rooms → at most 2 bathrooms (main + ensuite)
            total_rooms = len(room_ids)
            required_room_types = ["bedroom", "living_room", "kitchen", "bathroom"]
            unique_required = {"kitchen"}
            if total_rooms < 8:
                unique_required.add("bathroom")
            existing_room_types = [analysis.room_type for analysis in room_analyses]
            type_counts = {}
            for t in existing_room_types:
                type_counts[t] = type_counts.get(t, 0) + 1
            for required_type in required_room_types:
                if required_type not in existing_room_types:
                    errors.append(f"Required room type {required_type} is missing")
                    suggestions.append(f"Assign {required_type} to an appropriate room")
            for unique_type in unique_required:
                count = type_counts.get(unique_type, 0)
                if count > 1:
                    errors.append(f"Duplicate {unique_type}: {count} rooms have this type, only 1 allowed")
                    suggestions.append(f"Change the lower confidence {unique_type} to a different type")
            # >= 8 rooms: at most 2 bathrooms
            if total_rooms >= 8:
                bathroom_count = type_counts.get("bathroom", 0)
                if bathroom_count > 2:
                    errors.append(f"Too many bathrooms: {bathroom_count} rooms have this type, at most 2 allowed")
                    suggestions.append("Change the lowest confidence bathroom to a different type")
            
            is_valid = len(errors) == 0
            return ValidationResult(is_valid=is_valid, errors=errors, suggestions=suggestions)
        except Exception as e:
            logger.error(f"Room validation failed: {str(e)}")
            return ValidationResult(is_valid=False, errors=[str(e)], suggestions=["Fix the validation process"])
    
    def fix_room_analyses(self, room_analyses: List[RoomAnalysis], house_layout: Dict,
                          skip_behavior_prior: bool = False) -> List[RoomAnalysis]:
        """Fix room analyses"""
        try:
            # Get all room IDs from house layout
            room_ids = [room.get('id') for room in house_layout.get('house', {}).get('rooms', [])]
            analyzed_room_ids = [analysis.room_id for analysis in room_analyses]
            
            # Add missing room analyses
            for room_id in room_ids:
                if room_id not in analyzed_room_ids:
                    room_analyses.append(RoomAnalysis(
                        room_id=room_id,
                        room_type="living_room",
                        confidence=0.5
                    ))
            
            # Fix invalid room types
            valid_room_types = ["bedroom", "living_room", "kitchen", "bathroom", "studyroom","diningroom","balcony","other","unknown"]
            for analysis in room_analyses:
                if analysis.room_type not in valid_room_types:
                    analysis.room_type = "living_room"
                    analysis.confidence = 0.5
            
            # Fix invalid confidence scores
            for analysis in room_analyses:
                if not 0 <= analysis.confidence <= 1:
                    analysis.confidence = 0.5
            
            # Calculate room areas for deduplication logic
            layout_rooms = house_layout.get('house', {}).get('rooms', [])
            room_areas = {}
            layout_rooms_map = {}
            for room in layout_rooms:
                room_id = room.get('id')
                layout_rooms_map[room_id] = room
                area = room.get('position', {}).get('width', 0) * room.get('position', {}).get('height', 0)
                room_areas[room_id] = area
            
            # Deduplicate required types that should only appear once
            # bathroom: only unique when total rooms < 8
            total_rooms = len(layout_rooms)
            unique_required = {"kitchen"}
            if total_rooms < 8:
                unique_required.add("bathroom")
            type_counts = {}
            for a in room_analyses:
                type_counts[a.room_type] = type_counts.get(a.room_type, 0) + 1
            for unique_type in unique_required:
                if type_counts.get(unique_type, 0) > 1:
                    duplicates = [a for a in room_analyses if a.room_type == unique_type]
                    if unique_type == "bathroom":
                        # Keep the one with best bathroom clues (small furniture, ensuite pattern)
                        duplicates.sort(key=lambda a: 0 if any(
                            kw in item.get('name', '').lower()
                            for item in layout_rooms_map.get(a.room_id, {}).get('furniture', [])
                            for kw in ['toilet', 'washbasin', 'shower']
                        ) else 1)
                    else:
                        # Keep the one with kitchen furniture
                        duplicates.sort(key=lambda a: 1 if any(
                            kw in item.get('name', '').lower()
                            for item in layout_rooms_map.get(a.room_id, {}).get('furniture', [])
                            for kw in ['fr', 'refrigerator', 'stove', 'oven', 'microwave']
                        ) else 0, reverse=True)
                    keep = duplicates[0]
                    for analysis in duplicates[1:]:
                        logger.info(f"Deduplicating: {analysis.room_id} has duplicate {unique_type}, keeping {keep.room_id}")
                        analysis.room_type = "other"
                        analysis.confidence = 0.3

            # >= 8 rooms: at most 2 bathrooms
            if total_rooms >= 8:
                bathroom_count = type_counts.get("bathroom", 0)
                if bathroom_count > 2:
                    duplicates = [a for a in room_analyses if a.room_type == "bathroom"]
                    # Keep the 2 with best bathroom clues
                    duplicates.sort(key=lambda a: (
                        0 if any(kw in item.get('name', '').lower()
                            for item in layout_rooms_map.get(a.room_id, {}).get('furniture', [])
                            for kw in ['toilet', 'washbasin', 'shower'])
                        else 1,
                        -a.confidence
                    ))
                    keep = duplicates[:2]
                    for analysis in duplicates[2:]:
                        logger.info(f"Deduplicating: {analysis.room_id} has duplicate bathroom (max 2), keeping {[k.room_id for k in keep]}")
                        analysis.room_type = "other"
                        analysis.confidence = 0.3

            # Ensure all required room types are present
            required_room_types = ["bedroom", "living_room", "kitchen", "bathroom"]
            existing_room_types = [analysis.room_type for analysis in room_analyses]
            missing_types = [rt for rt in required_room_types if rt not in existing_room_types]
            
            # Get house layout rooms
            layout_rooms = house_layout.get('house', {}).get('rooms', [])
            room_info = {}
            room_areas = {}
            for room in layout_rooms:
                room_id = room.get('id')
                room_info[room_id] = room
                area = room.get('position', {}).get('width', 0) * room.get('position', {}).get('height', 0)
                room_areas[room_id] = area
            
            # Assign missing types to appropriate rooms
            for missing_type in missing_types:
                # 从未确定的房间（非other）中选 bathroom：小家具优先 + 内套线索
                if missing_type == "bathroom":
                    candidates = []
                    for analysis in room_analyses:
                        if analysis.room_type == "other":
                            continue
                        if analysis.room_type in required_room_types and analysis.room_type != missing_type:
                            continue
                        room = room_info.get(analysis.room_id, {})
                        
                        has_conflict = False
                        for item in room.get('furniture', []):
                            name = item.get('name', '').lower()
                            if any(kw in name for kw in ['fr', 'refrigerator', 'stove', 'oven', 'microwave']):
                                has_conflict = True
                                break
                        
                        # 检查是否所有家具都是小型（< 150 units²）—— bathroom 典型特征
                        all_small = True
                        for item in room.get('furniture', []):
                            w = item.get('position', {}).get('width', 0)
                            h = item.get('position', {}).get('height', 0)
                            if w * h >= 150:
                                all_small = False
                                break
                        
                        # 检查是否连接到有大型家具（>= 200 units²）的房间——内套卫浴
                        connected_to_large = False
                        connectivity_info = room.get('connectivity_info', {})
                        connected_ids = connectivity_info.get('connected_rooms', [])
                        if not connected_ids:
                            pos = room.get('position', {})
                            rx, ry, rw, rh = pos.get('x', 0), pos.get('y', 0), pos.get('width', 0), pos.get('height', 0)
                            for other_room in room_info.values():
                                oid = other_room.get('id')
                                if oid == analysis.room_id:
                                    continue
                                opos = other_room.get('position', {})
                                ox, oy, ow, oh = opos.get('x', 0), opos.get('y', 0), opos.get('width', 0), opos.get('height', 0)
                                shares_horizontal = (ry == oy + oh or ry + rh == oy) and (rx < ox + ow and rx + rw > ox)
                                shares_vertical = (rx == ox + ow or rx + rw == ox) and (ry < oy + oh and ry + rh > oy)
                                if shares_horizontal or shares_vertical:
                                    connected_ids.append(oid)
                        for connected_id in connected_ids:
                            connected_room = room_info.get(connected_id, {})
                            for item in connected_room.get('furniture', []):
                                w = item.get('position', {}).get('width', 0)
                                h = item.get('position', {}).get('height', 0)
                                if w * h >= 200:
                                    connected_to_large = True
                                    break
                            if connected_to_large:
                                break
                        
                        area = room_areas.get(analysis.room_id, 0)
                        candidates.append((analysis, area, has_conflict, all_small, connected_to_large))
                    
                    # 排序：无冲突 > 全小型家具(bathroom特征) > 连大型家具(内套) > 面积小
                    candidates.sort(key=lambda x: (x[2], not x[3], not x[4], x[1]))
                    if candidates:
                        best = candidates[0][0]
                        logger.info(f"Bathroom assignment: {best.room_id} (all_small={candidates[0][3]},"
                                    f" ensuite={candidates[0][4]}, area={candidates[0][1]})")
                        best.room_type = "bathroom"
                        best.confidence = 0.9
                        continue
                
                # Find the most suitable room for this type
                best_room = None
                best_score = -1
                
                for analysis in room_analyses:
                    if analysis.room_type == "other":
                        continue
                    room_id = analysis.room_id
                    room = room_info.get(room_id, {})
                    
                    # Calculate suitability score
                    score = 0
                    
                    # Check furniture
                    furniture = room.get('furniture', [])
                    for item in furniture:
                        name = item.get('name', '').lower()
                        if missing_type == "kitchen" and any(kw in name for kw in ['fr', 'refrigerator', 'stove', 'kitchen_countertop']):
                            score += 5
                        elif missing_type == "bathroom" and any(kw in name for kw in ['toilet', 'washbasin', 'shower']):
                            score += 5
                        elif missing_type == "bedroom" and any(kw in name for kw in ['bed', 'wardrobe', 'nightstand']):
                            score += 5
                        elif missing_type == "living_room" and any(kw in name for kw in ['tv', 'sofa', 'treadmill']):
                            score += 5
                    
                    # Check activity info
                    activity_info = {} if skip_behavior_prior else room.get('activity_info', {})
                    primary_period = activity_info.get('primary_time_period', 'none')
                    if missing_type == "bedroom" and primary_period == "night":
                        score += 4
                    elif missing_type == "living_room" and primary_period == "day":
                        score += 4
                    
                    # Check if room already has a required type
                    if analysis.room_type in required_room_types and analysis.room_type != missing_type:
                        score -= 10  # Avoid overriding other required types
                    
                    # Bonus: neutral rooms (no smart devices, low activity) are flexible candidates
                    if not any(item.get('name', '').strip() for item in room.get('furniture', [])):
                        score += 2  # No smart devices, easy to reassign
                    total_activity = activity_info.get('total_duration', 0)
                    if total_activity == 0 or total_activity is None:
                        score += 2  # No activity, flexible room
                    
                    if score > best_score:
                        best_score = score
                        best_room = analysis
                
                # Assign the missing type to the best room
                if best_room:
                    logger.info(f"Assigning missing room type {missing_type} to room {best_room.room_id}")
                    best_room.room_type = missing_type
                    best_room.confidence = 0.7  # Set a reasonable confidence
            
            # Fallback: if missing types still remain (e.g., all rooms have required types),
            # force-assign by replacing duplicate room types
            existing_room_types = [analysis.room_type for analysis in room_analyses]
            still_missing = [rt for rt in required_room_types if rt not in existing_room_types]
            
            for missing_type in still_missing:
                # Count occurrences of each room type
                type_counts = {}
                for a in room_analyses:
                    type_counts[a.room_type] = type_counts.get(a.room_type, 0) + 1
                
                # First look for rooms with non-required types (skip "other")
                candidate = None
                for analysis in room_analyses:
                    if analysis.room_type not in required_room_types and analysis.room_type != "other":
                        candidate = analysis
                        break
                
                # If missing bathroom, find smallest non-other room
                if candidate is None and missing_type == "bathroom":
                    candidates = []
                    for analysis in room_analyses:
                        if analysis.room_type == "other":
                            continue
                        area = room_areas.get(analysis.room_id, 0)
                        candidates.append((analysis, area))
                    if candidates:
                        candidates.sort(key=lambda x: x[1])
                        candidate = candidates[0][0]
                
                # If no non-required rooms, find a duplicate required type to replace
                if candidate is None:
                    for analysis in room_analyses:
                        if analysis.room_type in required_room_types and analysis.room_type != missing_type:
                            if type_counts.get(analysis.room_type, 0) > 1:
                                candidate = analysis
                                break
                
                # Last resort: pick the room with lowest confidence
                if candidate is None:
                    candidate = min(room_analyses, key=lambda a: a.confidence)
                
                if candidate:
                    logger.info(f"Fallback: assigning missing type {missing_type} to room {candidate.room_id} (was: {candidate.room_type})")
                    candidate.room_type = missing_type
                    candidate.confidence = 0.6

            # Final pass: reassign "other" rooms that have furniture (e.g. dedup leftovers)
            for analysis in room_analyses:
                if analysis.room_type != "other":
                    continue
                room = layout_rooms_map.get(analysis.room_id, {})
                furniture = room.get('furniture', [])
                if not furniture:
                    continue

                type_scores = {"bedroom": 0, "living_room": 0, "kitchen": 0, "bathroom": 0}
                for item in furniture:
                    name = item.get('name', '').lower()
                    if any(kw in name for kw in ['bed', 'wardrobe', 'nightstand']):
                        type_scores["bedroom"] += 5
                    if any(kw in name for kw in ['tv', 'sofa', 'coffee_table', 'treadmill']):
                        type_scores["living_room"] += 3
                    if any(kw in name for kw in ['fr', 'refrigerator', 'stove', 'oven', 'microwave', 'kitchen_countertop']):
                        type_scores["kitchen"] += 5
                    if any(kw in name for kw in ['toilet', 'washbasin', 'shower']):
                        type_scores["bathroom"] += 5
                    w = item.get('position', {}).get('width', 0)
                    h = item.get('position', {}).get('height', 0)
                    area = w * h
                    ratio = max(w, h) / (min(w, h) + 0.1)
                    if area >= 200:
                        type_scores["bedroom"] += 2
                    elif area >= 100:
                        if ratio < 1.5:
                            type_scores["living_room"] += 2
                        elif ratio >= 2.0:
                            type_scores["bedroom"] += 2
                        else:
                            type_scores["bedroom"] += 1
                            type_scores["living_room"] += 1

                activity_info = {} if skip_behavior_prior else room.get('activity_info', {})
                primary_period = activity_info.get('primary_time_period', 'none')
                if primary_period == 'night':
                    type_scores["bedroom"] += 4
                elif primary_period == 'day':
                    type_scores["living_room"] += 4

                best_type = max(type_scores, key=type_scores.get)
                best_score = type_scores[best_type]
                if best_score > 0:
                    logger.info(f"Final reassign: {analysis.room_id} other→{best_type} (score={best_score})")
                    analysis.room_type = best_type
                    analysis.confidence = min(0.4 + best_score * 0.02, 0.8)
                else:
                    all_tiny = all(
                        (item.get('position', {}).get('width', 0) * item.get('position', {}).get('height', 0)) < 100
                        for item in furniture
                    )
                    if all_tiny:
                        analysis.room_type = "other"
                        analysis.confidence = 0.5
                        logger.info(f"Final reassign: {analysis.room_id} other→other (all furniture tiny)")
                    else:
                        analysis.room_type = "living_room"
                        analysis.confidence = 0.4
                        logger.info(f"Final reassign: {analysis.room_id} other→living_room (default)")

            return room_analyses
        except Exception as e:
            logger.error(f"Room analysis fixing failed: {str(e)}")
            return self._get_default_room_analyses(house_layout)
    
    def analyze_rooms(self, house_layout: Dict, trajectory_data: any,
                      ablation: dict = None) -> List[RoomAnalysis]:
        """Analyze rooms and infer types with retry mechanism to ensure all required types are present

        ablation:
            - skip_room_module: 跳过房间语义模块，全部房间退化为 "unknown"
                                （单模块移除消融；下游按 unknown 处理）
            - skip_layered_priority: 跳过7层优先级，直接使用 LLM 输出
            - skip_behavior_prior: 跳过行为先验推理
        """
        ablation = ablation or {}
        if ablation.get('skip_room_module'):
            logger.info("[ABLATION] 跳过房间语义模块，全部房间类型标为 unknown")
            rooms = house_layout.get('house', {}).get('rooms', [])
            return [RoomAnalysis(room_id=room.get('id'), room_type='unknown',
                                 confidence=0.3)
                    for room in rooms]
        required_room_types = ["bedroom", "living_room", "kitchen", "bathroom"]
        max_retries = 3
        
        for attempt in range(max_retries):
            if attempt == 0:
                # First attempt: generate and apply layered priority
                generation_result = self.generate_room_analyses(house_layout, trajectory_data)
                room_analyses = generation_result.room_analyses

                if ablation.get('skip_layered_priority'):
                    logger.info("[ABLATION] 跳过7层优先级，直接使用 LLM 输出")
                elif ablation.get('skip_behavior_prior'):
                    logger.info("[ABLATION] 跳过行为先验推理（仅使用家具和设备先验）")
                    # 只执行 Step1 (无家具→other) + Step2 (智能家具确定)，跳过行为相关步骤
                    self._apply_partial_priority(house_layout, room_analyses)
                else:
                    logger.info("Applying layered priority method for room type inference...")
                    room_analyses = self._apply_layered_priority(house_layout, room_analyses)
            else:
                if ablation.get('skip_layered_priority'):
                    # 不应用优先级，直接 validate/fix
                    pass
                elif ablation.get('skip_behavior_prior'):
                    # 保持行为先验消融：重试时也只能使用部分优先级，
                    # 不能回退到包含行为步骤的完整 7 层规则。
                    logger.info(
                        f"Retry attempt {attempt + 1}/{max_retries}: "
                        "re-apply priority without behavior prior..."
                    )
                    room_analyses = self._apply_partial_priority(
                        house_layout, room_analyses)
                else:
                    # Retry: re-apply layered priority and force-fix missing types
                    logger.info(f"Retry attempt {attempt + 1}/{max_retries} for room analysis...")
                    room_analyses = self._apply_layered_priority(house_layout, room_analyses)
            
            # Validate room analyses
            validation_result = self.validate_room_analyses(room_analyses, house_layout)
            
            if validation_result.is_valid:
                logger.info("Room analysis validation passed")
                break
            
            logger.warning(f"Room analyses validation failed (attempt {attempt + 1}/{max_retries}): {validation_result.errors}")
            
            # Fix room analyses
            room_analyses = self.fix_room_analyses(
                room_analyses, house_layout,
                skip_behavior_prior=bool(ablation.get('skip_behavior_prior')))
            
            # Re-validate after fix
            validation_result = self.validate_room_analyses(room_analyses, house_layout)
            if validation_result.is_valid:
                logger.info("Room analysis validation passed after fix")
                break
            
            # If still invalid after fix and this is the last attempt, force-assign missing types
            if attempt == max_retries - 1:
                logger.warning("Force-assigning missing room types...")
                existing_room_types = [a.room_type for a in room_analyses]
                missing_types = [rt for rt in required_room_types if rt not in existing_room_types]
                
                # Get room info from layout
                layout_rooms = house_layout.get('house', {}).get('rooms', [])
                room_info = {}
                for room in layout_rooms:
                    room_id = room.get('id')
                    room_info[room_id] = room
                
                for missing_type in missing_types:
                    # Find rooms already assigned to non-required types or duplicate types (skip "other")
                    candidate = None
                    for analysis in room_analyses:
                        if analysis.room_type not in required_room_types and analysis.room_type != "other":
                            candidate = analysis
                            break
                    
                    if candidate is None:
                        # If all rooms have required types, replace the one with lowest confidence (skip "other")
                        type_counts = {}
                        for a in room_analyses:
                            type_counts[a.room_type] = type_counts.get(a.room_type, 0) + 1
                        
                        for analysis in room_analyses:
                            if type_counts.get(analysis.room_type, 0) > 1 and analysis.room_type != missing_type:
                                candidate = analysis
                                break
                        
                        if candidate is None:
                            non_other = [a for a in room_analyses if a.room_type != "other"]
                            if non_other:
                                candidate = min(non_other, key=lambda a: a.confidence)
                    
                    if candidate:
                        logger.info(f"Force-assigning {missing_type} to room {candidate.room_id} (was: {candidate.room_type})")
                        candidate.room_type = missing_type
                        candidate.confidence = 0.55
            
            # Final validation check
            final_validation = self.validate_room_analyses(room_analyses, house_layout)
            if final_validation.is_valid:
                logger.info("All required room types satisfied after force assignment")
            else:
                logger.error(f"Unable to satisfy all room types: {final_validation.errors}")
        
        # Log room types
        logger.info("Identified room types:")
        for analysis in room_analyses:
            logger.info(f"- Room {analysis.room_id}: {analysis.room_type} (confidence: {analysis.confidence:.2f})")
        
        return room_analyses
    
    # 数据预处理——为房屋布局数据添加智能设备信息，LLM推理之前
    def _augment_house_layout_with_furniture_info(self, house_layout: Dict) -> Dict:
        """Augment house layout with furniture information for better room type inference"""
        # Create a copy of the house layout
        augmented_layout = json.loads(json.dumps(house_layout))
        
        # Define furniture to room type mappings with priority
        # 使用类级别的统一家具映射常量
        furniture_room_mappings = self.FURNITURE_PRIORITY_MAPPINGS
        
        # Augment each room with furniture information
        rooms = augmented_layout.get('house', {}).get('rooms', [])
        for room in rooms:
            furniture = room.get('furniture', [])
            room['furniture_info'] = {
                'count': len(furniture),
                'items': [],
                'suggested_room_types': [],
                'primary_room_type': None,
                'primary_priority': 0
            }
            
            # 只分析智能设备（有名称的家具）
            smart_devices = [item for item in furniture if item.get('name', '').strip()]
            
            # Analyze each smart device
            for item in smart_devices:
                furniture_name = item.get('name', 'unknown')
                room['furniture_info']['items'].append(furniture_name)
                
                # Add suggested room type based on smart device
                if furniture_name in furniture_room_mappings:
                    mapping = furniture_room_mappings[furniture_name]
                    suggested_type = mapping["types"][0]  # 取第一个类型作为建议
                    priority = mapping["priority"]
                    
                    # Update primary room type if this smart device has higher priority
                    if priority > room['furniture_info']['primary_priority']:
                        room['furniture_info']['primary_room_type'] = suggested_type
                        room['furniture_info']['primary_priority'] = priority
                    
                    if suggested_type not in room['furniture_info']['suggested_room_types']:
                        room['furniture_info']['suggested_room_types'].append(suggested_type)
            
            # Update count to reflect only smart devices
            room['furniture_info']['count'] = len(smart_devices)
        return augmented_layout
    
    def _get_default_room_analyses(self, house_layout: Dict) -> List[RoomAnalysis]:
        """Get default room analyses when LLM fails"""
        room_analyses = []
        rooms = house_layout.get('house', {}).get('rooms', [])
        for room in rooms:
            room_id = room.get('id')
            # Default to living_room for simplicity
            room_analyses.append(RoomAnalysis(
                room_id=room_id,
                room_type="living_room",
                confidence=0.5
            ))
        return room_analyses
    
    def _augment_house_layout_with_activity_info(self, house_layout: Dict, trajectory_data: List[Dict]) -> Dict:
        """增强房屋布局数据，添加轨迹活动信息"""
        augmented_layout = json.loads(json.dumps(house_layout))
        
        # 统计每个房间的活动信息
        room_activities = self._analyze_room_activities(house_layout, trajectory_data)
        
        # 分析房间连通性和转换模式
        room_transitions = self._analyze_room_transitions(trajectory_data, house_layout)
        
        # 为每个房间添加活动信息
        rooms = augmented_layout.get('house', {}).get('rooms', [])
        for room in rooms:
            room_id = room.get('id')
            room['activity_info'] = room_activities.get(room_id, {
                'total_duration': 0,
                'day_duration': 0,
                'night_duration': 0,
                'visit_count': 0,
                'primary_time_period': 'none'
            })
            room['transition_info'] = room_transitions.get(room_id, {
                'incoming_transitions': {},
                'outgoing_transitions': {},
                'total_transitions': 0
            })
        
        # 分析房间连通性
        self._analyze_room_connectivity(augmented_layout)
        
        return augmented_layout
    
    def _analyze_room_activities(self, house_layout: Dict, trajectory_data: List[Dict]) -> Dict:
        """分析每个房间的轨迹活动"""
        room_activities = {}
        rooms = house_layout.get('house', {}).get('rooms', [])
        
        # 为每个房间初始化活动统计
        for room in rooms:
            room_id = room.get('id')
            room_activities[room_id] = {
                'total_duration': 0,
                'day_duration': 0,  # 6:00-22:00
                'night_duration': 0, # 22:00-6:00
                'visit_count': 0,
                'primary_time_period': 'none'
            }
        
        # 分析每个轨迹点
        for trajectory in trajectory_data:
            center = trajectory.get('center_position', {})
            x, y = center.get('x'), center.get('y')
            duration = trajectory.get('duration', 0)
            start_time = trajectory.get('start_time', '')
            
            # 确定轨迹点所在的房间
            room_id = self._get_room_by_position(rooms, x, y)
            if room_id:
                # 更新活动统计
                room_activities[room_id]['total_duration'] += duration
                room_activities[room_id]['visit_count'] += 1
                
                # 区分白天和夜间活动
                try:
                    hour = int(start_time.split(' ')[1].split(':')[0])
                    if 6 <= hour < 22:
                        room_activities[room_id]['day_duration'] += duration
                    else:
                        room_activities[room_id]['night_duration'] += duration
                except (ValueError, IndexError):
                    # 如果时间格式不正确，默认为白天活动
                    room_activities[room_id]['day_duration'] += duration
        
        # 确定每个房间的主要活动时间段
        for room_id, activity in room_activities.items():
            if activity['night_duration'] > activity['day_duration']:
                activity['primary_time_period'] = 'night'
            elif activity['day_duration'] > activity['night_duration']:
                activity['primary_time_period'] = 'day'
        
        return room_activities
    
    def _get_room_by_position(self, rooms: List[Dict], x: float, y: float) -> str:
        """根据位置坐标确定房间ID"""
        for room in rooms:
            pos = room.get('position', {})
            room_x = pos.get('x', 0)
            room_y = pos.get('y', 0)
            width = pos.get('width', 0)
            height = pos.get('height', 0)
            
            # 检查点是否在房间内
            if room_x <= x < room_x + width and room_y <= y < room_y + height:
                return room.get('id')
        return None
    
    def _analyze_room_transitions(self, trajectory_data: List[Dict], house_layout: Dict) -> Dict:
        """分析房间之间的转换模式"""
        room_transitions = {}
        rooms = house_layout.get('house', {}).get('rooms', [])
        
        # 初始化转换统计
        for room in rooms:
            room_id = room.get('id')
            room_transitions[room_id] = {
                'incoming_transitions': {},
                'outgoing_transitions': {},
                'total_transitions': 0
            }
        
        # 分析轨迹序列
        if len(trajectory_data) < 2:
            return room_transitions
        
        # 按时间排序轨迹
        sorted_trajectories = sorted(trajectory_data, key=lambda t: t.get('start_time', ''))
        
        # 分析相邻轨迹的房间转换
        for i in range(len(sorted_trajectories) - 1):
            current_traj = sorted_trajectories[i]
            next_traj = sorted_trajectories[i + 1]
            
            # 获取当前和下一个轨迹的房间
            current_center = current_traj.get('center_position', {})
            next_center = next_traj.get('center_position', {})
            
            current_room = self._get_room_by_position(rooms, current_center.get('x'), current_center.get('y'))
            next_room = self._get_room_by_position(rooms, next_center.get('x'), next_center.get('y'))
            
            # 记录转换
            if current_room and next_room and current_room != next_room:
                # 更新当前房间的外出转换
                if next_room not in room_transitions[current_room]['outgoing_transitions']:
                    room_transitions[current_room]['outgoing_transitions'][next_room] = 0
                room_transitions[current_room]['outgoing_transitions'][next_room] += 1
                
                # 更新下一个房间的进入转换
                if current_room not in room_transitions[next_room]['incoming_transitions']:
                    room_transitions[next_room]['incoming_transitions'][current_room] = 0
                room_transitions[next_room]['incoming_transitions'][current_room] += 1
                
                # 更新总转换次数
                room_transitions[current_room]['total_transitions'] += 1
                room_transitions[next_room]['total_transitions'] += 1
        
        return room_transitions
    
    def _analyze_room_connectivity(self, house_layout: Dict):
        """分析房间连通性"""
        rooms = house_layout.get('house', {}).get('rooms', [])
        
        # 为每个房间添加连通性信息
        for room in rooms:
            room_id = room.get('id')
            room['connectivity_info'] = {
                'connected_rooms': [],
                'connectivity_score': 0
            }
        
        # 基于门的位置分析连通性
        for room in rooms:
            room_id = room.get('id')
            doors = room.get('doors', [])
            
            for door in doors:
                # 简化处理：假设门连接到相邻房间
                # 这里可以根据门的位置更精确地计算连接的房间
                # 为了简化，我们假设每个门连接到一个其他房间
                # 在实际应用中，应该根据门的坐标计算连接的房间
                pass
        
        # 基于转换数据增强连通性信息
        for room in rooms:
            room_id = room.get('id')
            transition_info = room.get('transition_info', {})
            outgoing = transition_info.get('outgoing_transitions', {})
            
            # 将有转换的房间视为连通
            for connected_room in outgoing.keys():
                if connected_room not in room['connectivity_info']['connected_rooms']:
                    room['connectivity_info']['connected_rooms'].append(connected_room)
            
            # 计算连通性得分
            room['connectivity_info']['connectivity_score'] = len(room['connectivity_info']['connected_rooms'])

# 结果修正——使用分层优先级法修正房间类型，LLM推理之后
    def _apply_layered_priority(self, house_layout: Dict, room_analyses: List[RoomAnalysis]) -> List[RoomAnalysis]:
        """
        应用分层优先级法确定房间类型
        优先级顺序：
        1. 最高优先级：家具信息（明确的家具指示器）
        2. 次优先级：行为模式（白天/夜间活动）
        3. 最低优先级：房间连通性 + 面积辅助判定
        """
        # 使用类级别的统一家具映射常量
        furniture_priority_mappings = self.FURNITURE_PRIORITY_MAPPINGS

        room_info = {}
        room_areas = {}  # 预计算房间面积（世界坐标）
        all_rooms = house_layout.get('house', {}).get('rooms', [])
        for room in all_rooms:
            room_id = room.get('id')
            room_info[room_id] = room
            pos = room.get('position', {})
            room_areas[room_id] = pos.get('width', 0) * pos.get('height', 0)

        total_rooms = len(all_rooms)
        required_types = ["bedroom", "living_room", "kitchen", "bathroom"]
        existing_types = set()
        determined_rooms = set()  # 已确定的房间不再参与后续分配

        # 计算面积排名（用于辅助判定）
        sorted_by_area = sorted(room_areas.items(), key=lambda x: x[1])
        area_rank = {rid: rank for rank, (rid, _) in enumerate(sorted_by_area)}  # 0 = 最小

        # ─── 第1步：无家具 → other（锁定，后续不参与修改）───
        for analysis in room_analyses:
            room_id = analysis.room_id
            room = room_info.get(room_id, {})
            furniture = room.get('furniture', [])
            if not furniture:
                analysis.room_type = "other"
                analysis.confidence = 0.8
                logger.info(f"Layered Priority [Step1 No Furniture]: Room {room_id} → other")
                determined_rooms.add(room_id)

        # ─── 第2步：智能家具确定房间类型 ───
        for analysis in room_analyses:
            room_id = analysis.room_id
            if room_id in determined_rooms:
                continue
            room = room_info.get(room_id, {})
            furniture = room.get('furniture', [])

            highest_priority = 0
            furniture_based_type = None
            smart_devices = [item for item in furniture if item.get('name', '').strip()]

            for item in smart_devices:
                furniture_name = item.get('name', '')
                if furniture_name in furniture_priority_mappings:
                    mapping = furniture_priority_mappings[furniture_name]
                    if mapping["priority"] > highest_priority:
                        highest_priority = mapping["priority"]
                        furniture_based_type = mapping["types"][0]

            if highest_priority >= 4 and furniture_based_type:
                analysis.room_type = furniture_based_type
                analysis.confidence = min(0.5 + (highest_priority * 0.1), 0.95)
                logger.info(f"Layered Priority [Step2 Furniture]: Room {room_id} → {furniture_based_type} (priority: {highest_priority})")
                existing_types.add(furniture_based_type)
                determined_rooms.add(room_id)
                continue

            if highest_priority == 3 and furniture_based_type:
                activity_info = room.get('activity_info', {})
                primary_period = activity_info.get('primary_time_period', 'none')
                day_duration = activity_info.get('day_duration', 0)
                night_duration = activity_info.get('night_duration', 0)

                possible_types = []
                for item in smart_devices:
                    furniture_name = item.get('name', '')
                    if furniture_name in furniture_priority_mappings:
                        mapping = furniture_priority_mappings[furniture_name]
                        if mapping["priority"] == 3:
                            possible_types.extend(mapping["types"])

                selected_type = None
                if primary_period == 'night' and night_duration > day_duration:
                    if 'bedroom' in possible_types:
                        selected_type = 'bedroom'
                elif primary_period == 'day' and day_duration > night_duration:
                    if 'living_room' in possible_types:
                        selected_type = 'living_room'

                if selected_type:
                    analysis.room_type = selected_type
                    analysis.confidence = min(0.4 + (highest_priority * 0.05), 0.7)
                    logger.info(f"Layered Priority [Step2 Ambiguous + Behavior]: Room {room_id} → {selected_type}")
                    existing_types.add(selected_type)
                    determined_rooms.add(room_id)

        # ─── Step 2.5: 阳台检测 (提前于行为推理, 靠边+小面积+少家具信号强度高于行为模式) ───
        # 阳台特征: 靠近房屋外侧边缘 + 家具 ≤2 + 面积在底部 1/3
        house_size = house_layout.get('house', {}).get('size', {})
        house_w = house_size.get('x', 0)
        house_h = house_size.get('y', 0)
        EDGE_THRESHOLD = 3.0  # 距房屋边缘 3dm 以内视为靠边
        max_balcony_rank = max(total_rooms // 3, 2)  # 底部 1/3 或至少第 3 小

        for analysis in room_analyses:
            if analysis.room_id in determined_rooms:
                continue

            room = room_info.get(analysis.room_id, {})
            pos = room.get('position', {})
            rx, ry = pos.get('x', 0), pos.get('y', 0)
            rw, rh = pos.get('width', 0), pos.get('height', 0)
            area_rk = area_rank.get(analysis.room_id, total_rooms)
            furniture = room.get('furniture', [])

            # 面积不在允许范围内 → 不是阳台
            if area_rk > max_balcony_rank:
                continue

            # 家具 > 2 → 不是阳台
            if len(furniture) > 2:
                continue

            # 检查靠房屋外侧边缘
            at_edge_left = abs(rx) <= EDGE_THRESHOLD
            at_edge_bottom = abs(ry) <= EDGE_THRESHOLD
            at_edge_right = abs(rx + rw - house_w) <= EDGE_THRESHOLD
            at_edge_top = abs(ry + rh - house_h) <= EDGE_THRESHOLD
            at_house_edge = at_edge_left or at_edge_bottom or at_edge_right or at_edge_top
            if not at_house_edge:
                continue

            # 厨房设备冲突 → 不是阳台
            if any(kw in item.get('name', '').lower()
                   for item in furniture
                   for kw in ['fr', 'refrigerator', 'stove', 'oven', 'microwave']):
                continue

            # 全小无名 + 最小面积 → 留给 bathroom (Step 5 会用到)
            all_small_unnamed = all(
                not item.get('name', '').strip()
                and (item.get('position', {}).get('width', 0) *
                     item.get('position', {}).get('height', 0)) < 150
                for item in furniture
            ) if furniture else True
            if all_small_unnamed and area_rk == 0:
                continue

            # 通过所有检查 → balcony
            analysis.room_type = "balcony"
            if len(furniture) == 0 and area_rk <= 1:
                analysis.confidence = 0.9
            elif len(furniture) <= 1:
                analysis.confidence = 0.8
            else:
                analysis.confidence = 0.7
            existing_types.add("balcony")
            determined_rooms.add(analysis.room_id)
            logger.info(
                f"Layered Priority [Step2.5 Balcony]: Room {analysis.room_id} → balcony"
                f" (area_rank={area_rk}, furniture={len(furniture)},"
                f" edge_left={at_edge_left}, edge_right={at_edge_right},"
                f" edge_bottom={at_edge_bottom}, edge_top={at_edge_top})"
            )

        # ─── 第3步：行为统计信息推理 ───
        for analysis in room_analyses:
            room_id = analysis.room_id
            if room_id in determined_rooms:
                continue
            room = room_info.get(room_id, {})
            furniture = room.get('furniture', [])
            room_area = room_areas.get(room_id, 0)

            activity_info = room.get('activity_info', {})
            primary_period = activity_info.get('primary_time_period', 'none')
            day_duration = activity_info.get('day_duration', 0)
            night_duration = activity_info.get('night_duration', 0)
            total_duration = activity_info.get('total_duration', 0)

            # 检查是否全小无名家具（降低置信度但不跳过）
            all_small_unnamed = all(
                not item.get('name', '').strip()
                and (item.get('position', {}).get('width', 0) *
                     item.get('position', {}).get('height', 0)) < 150
                for item in furniture
            ) if furniture else False

            # 行为信号强度
            night_ratio = night_duration / max(total_duration, 1)
            day_ratio = day_duration / max(total_duration, 1)
            behavior_strong = total_duration > 0 and (night_ratio > 0.6 or day_ratio > 0.6)

            if primary_period == 'night' and night_duration > day_duration:
                base_conf = min(0.4 + night_ratio * 0.3, 0.7)
                # 全小无名家具降权
                if all_small_unnamed and not behavior_strong:
                    base_conf = min(base_conf, 0.4)
                analysis.room_type = 'bedroom'
                analysis.confidence = base_conf
                logger.info(f"Layered Priority [Step3 Behavior]: Room {room_id} → bedroom (conf={base_conf:.2f})")
                existing_types.add('bedroom')
                if behavior_strong and not all_small_unnamed:
                    determined_rooms.add(room_id)  # 仅强信号锁定
            elif primary_period == 'day' and day_duration > night_duration:
                base_conf = min(0.4 + day_ratio * 0.3, 0.7)
                if all_small_unnamed and not behavior_strong:
                    base_conf = min(base_conf, 0.4)
                # 小房间 + 白天活动 → 可能是 study/bathroom，不是 living_room
                if room_area < 50 and all_small_unnamed:
                    continue  # 不判定为 living_room，留给后续步骤
                analysis.room_type = 'living_room'
                analysis.confidence = base_conf
                logger.info(f"Layered Priority [Step3 Behavior]: Room {room_id} → living_room (conf={base_conf:.2f})")
                existing_types.add('living_room')
                if behavior_strong and not all_small_unnamed:
                    determined_rooms.add(room_id)

        # ─── 第4步：连通性推理 ───
        for analysis in room_analyses:
            room_id = analysis.room_id
            if room_id in determined_rooms:
                continue
            room = room_info.get(room_id, {})
            room_area = room_areas.get(room_id, 0)
            activity_info = room.get('activity_info', {})
            total_duration = activity_info.get('total_duration', 0)
            connectivity_info = room.get('connectivity_info', {})
            connectivity_score = connectivity_info.get('connectivity_score', 0)
            connected_rooms = connectivity_info.get('connected_rooms', [])

            # 连通度 >= 4 且房间面积较大 + 有活动 → living_room
            if connectivity_score >= 4 and room_area > 30 and total_duration > 0:
                analysis.room_type = 'living_room'
                analysis.confidence = min(0.3 + (connectivity_score * 0.1), 0.6)
                logger.info(f"Layered Priority [Step4 Connectivity]: Room {room_id} → living_room (connectivity={connectivity_score})")
                existing_types.add('living_room')
                determined_rooms.add(room_id)
            # 连通到厨房且有白天活动 → living_room
            elif 'kitchen' in connected_rooms and total_duration > 0 and room_area > 25:
                analysis.room_type = 'living_room'
                analysis.confidence = 0.5
                logger.info(f"Layered Priority [Step4 Connectivity]: Room {room_id} → living_room (connected to kitchen)")
                existing_types.add('living_room')
                determined_rooms.add(room_id)

        # ─── 第5步：推断bathroom（主卫+内套卫浴） ───
        # 收集所有未确定房间的家具特征
        undetermined = []
        for analysis in room_analyses:
            if analysis.room_id in determined_rooms:
                continue
            room = room_info.get(analysis.room_id, {})
            area = room.get('position', {}).get('width', 0) * room.get('position', {}).get('height', 0)

            # 检查是否有厨房设备冲突
            has_conflict = False
            for item in room.get('furniture', []):
                name = item.get('name', '').lower()
                if any(kw in name for kw in ['fr', 'refrigerator', 'stove', 'oven', 'microwave']):
                    has_conflict = True
                    break

            # 检查是否所有家具都是无名且小型（< 150 units²）—— bathroom 典型特征
            all_small_unnamed = True
            for item in room.get('furniture', []):
                if item.get('name', '').strip():
                    all_small_unnamed = False
                    break
                w = item.get('position', {}).get('width', 0)
                h = item.get('position', {}).get('height', 0)
                if w * h >= 150:
                    all_small_unnamed = False
                    break

            # 检查是否连接到有大型家具（>= 200 units²）的房间—— 内套卫浴线索
            connected_to_large_furniture = False
            connectivity_info = room.get('connectivity_info', {})
            connected_ids = connectivity_info.get('connected_rooms', [])
            if not connected_ids:
                # 轨迹数据不足时，用空间相邻检测作为fallback
                pos = room.get('position', {})
                rx, ry, rw, rh = pos.get('x', 0), pos.get('y', 0), pos.get('width', 0), pos.get('height', 0)
                for other_room in room_info.values():
                    oid = other_room.get('id')
                    if oid == room.get('id'):
                        continue
                    opos = other_room.get('position', {})
                    ox, oy, ow, oh = opos.get('x', 0), opos.get('y', 0), opos.get('width', 0), opos.get('height', 0)
                    # 检查共享墙：room底边 == other顶边 OR room顶边 == other底边 OR room左边==other右边 OR room右边==other左边
                    shares_horizontal = (ry == oy + oh or ry + rh == oy) and (rx < ox + ow and rx + rw > ox)
                    shares_vertical = (rx == ox + ow or rx + rw == ox) and (ry < oy + oh and ry + rh > oy)
                    if shares_horizontal or shares_vertical:
                        connected_ids.append(oid)
            for connected_id in connected_ids:
                connected_room = room_info.get(connected_id, {})
                for item in connected_room.get('furniture', []):
                    w = item.get('position', {}).get('width', 0)
                    h = item.get('position', {}).get('height', 0)
                    if w * h >= 200:
                        connected_to_large_furniture = True
                        break
                if connected_to_large_furniture:
                    break

            undetermined.append((analysis, area, has_conflict, all_small_unnamed, connected_to_large_furniture))

        # 5a. 主卫：bathroom缺失时，选最合适的未确定房间
        if "bathroom" not in existing_types:
            candidates = [u for u in undetermined]
            candidates.sort(key=lambda x: (x[2], not x[3], not x[4], x[1]))
            if candidates:
                best = candidates[0][0]
                logger.info(f"Layered Priority [Step5a Primary Bathroom]: Room {best.room_id} → bathroom"
                            f" (all_small_unnamed={candidates[0][3]},"
                            f" connected_to_large={candidates[0][4]})")
                best.room_type = "bathroom"
                best.confidence = 0.9
                existing_types.add("bathroom")
                determined_rooms.add(best.room_id)
                # 从待选列表移除
                undetermined = [u for u in undetermined if u[0].room_id != best.room_id]

        # 5b. 内套卫浴：仅房间数 >= 8 时才启用，扫描所有房间找全小家具+连大家具的bathroom模式
        if total_rooms >= 8:
            # 安全边界：真正卧室/客厅一定有>=200的大家具，不会误判
            for analysis in room_analyses:
                if analysis.room_id in determined_rooms and analysis.room_type == "kitchen":
                    continue  # 厨房不可能有内套卫浴
                room = room_info.get(analysis.room_id, {})
                furniture = room.get('furniture', [])
                if not furniture:
                    continue  # 无家具的房间已在Step1设为other, 不参与浴室推断

                # 全小无名检查
                all_small_unnamed = True
                for item in furniture:
                    if item.get('name', '').strip():
                        all_small_unnamed = False
                        break
                    w = item.get('position', {}).get('width', 0)
                    h = item.get('position', {}).get('height', 0)
                    if w * h >= 150:
                        all_small_unnamed = False
                        break

                # 厨房冲突检查
                has_conflict = False
                for item in furniture:
                    name = item.get('name', '').lower()
                    if any(kw in name for kw in ['fr', 'refrigerator', 'stove', 'oven', 'microwave']):
                        has_conflict = True
                        break

                # 连接大家具的房间检查
                connected_to_large = False
                connectivity_info = room.get('connectivity_info', {})
                connected_ids = connectivity_info.get('connected_rooms', [])
                if not connected_ids:
                    pos = room.get('position', {})
                    rx, ry, rw, rh = pos.get('x', 0), pos.get('y', 0), pos.get('width', 0), pos.get('height', 0)
                    for other_room in room_info.values():
                        oid = other_room.get('id')
                        if oid == analysis.room_id:
                            continue
                        opos = other_room.get('position', {})
                        ox, oy, ow, oh = opos.get('x', 0), opos.get('y', 0), opos.get('width', 0), opos.get('height', 0)
                        shares_horizontal = (ry == oy + oh or ry + rh == oy) and (rx < ox + ow and rx + rw > ox)
                        shares_vertical = (rx == ox + ow or rx + rw == ox) and (ry < oy + oh and ry + rh > oy)
                        if shares_horizontal or shares_vertical:
                            connected_ids.append(oid)
                for connected_id in connected_ids:
                    connected_room = room_info.get(connected_id, {})
                    for item in connected_room.get('furniture', []):
                        w = item.get('position', {}).get('width', 0)
                        h = item.get('position', {}).get('height', 0)
                        if w * h >= 200:
                            connected_to_large = True
                            break
                    if connected_to_large:
                        break

                if all_small_unnamed and connected_to_large and not has_conflict:
                    if analysis.room_type != "bathroom":
                        logger.info(f"Layered Priority [Step5b Ensuite Bathroom]: Room {analysis.room_id} {analysis.room_type}→bathroom"
                                    f" (ensuite: all_small + connected_to_large)")
                    analysis.room_type = "bathroom"
                    analysis.confidence = 0.85
                    existing_types.add("bathroom")
                    determined_rooms.add(analysis.room_id)

        # 5c. 第二卫浴：房间数>=8且已有bathroom时，全小无名家具的房间也设为bathroom
        if total_rooms >= 8 and "bathroom" in existing_types:
            current_bathroom_count = sum(1 for a in room_analyses if a.room_type == "bathroom")
            if current_bathroom_count < 2:
                for analysis in room_analyses:
                    if analysis.room_id in determined_rooms:
                        continue
                    room = room_info.get(analysis.room_id, {})
                    furniture = room.get('furniture', [])
                    if not furniture:
                        continue
                    all_small_unnamed = all(
                        not item.get('name', '').strip()
                        and (item.get('position', {}).get('width', 0) *
                             item.get('position', {}).get('height', 0)) < 150
                        for item in furniture
                    ) if furniture else False
                    if all_small_unnamed:
                        has_conflict = any(
                            kw in item.get('name', '').lower()
                            for item in furniture
                            for kw in ['fr', 'refrigerator', 'stove', 'oven', 'microwave']
                        )
                        if not has_conflict:
                            analysis.room_type = "bathroom"
                            analysis.confidence = 0.7
                            existing_types.add("bathroom")
                            determined_rooms.add(analysis.room_id)
                            logger.info(f"Layered Priority [Step5c Secondary Bathroom]: Room {analysis.room_id} → bathroom (all_small_unnamed)")
                            current_bathroom_count += 1
                            if current_bathroom_count >= 2:
                                break

        # ─── 第6步：缺失类型评分分配 ───
        missing_types = [rt for rt in required_types if rt not in existing_types]
        for missing_type in missing_types:
            best_room = None
            best_score = -1

            for analysis in room_analyses:
                if analysis.room_id in determined_rooms:
                    continue

                room_id = analysis.room_id
                room = room_info.get(room_id, {})
                score = 0

                for item in room.get('furniture', []):
                    name = item.get('name', '').lower()
                    if missing_type == "kitchen" and any(kw in name for kw in ['fr', 'refrigerator', 'stove', 'kitchen_countertop']):
                        score += 5
                    elif missing_type == "bathroom" and any(kw in name for kw in ['toilet', 'washbasin', 'shower']):
                        score += 5
                    elif missing_type == "bedroom" and any(kw in name for kw in ['bed', 'wardrobe', 'nightstand']):
                        score += 5
                    elif missing_type == "living_room" and any(kw in name for kw in ['tv', 'sofa', 'treadmill']):
                        score += 5

                activity_info = room.get('activity_info', {})
                primary_period = activity_info.get('primary_time_period', 'none')
                if missing_type == "bedroom" and primary_period == "night":
                    score += 4
                elif missing_type == "living_room" and primary_period == "day":
                    score += 4

                if score > best_score:
                    best_score = score
                    best_room = analysis

            if best_room:
                logger.info(f"Layered Priority [Step6 Assign]: Room {best_room.room_id} → {missing_type} (score: {best_score})")
                best_room.room_type = missing_type
                best_room.confidence = 0.6

        # ─── 第7步：有家具但类型为other/unknown的房间，重新评分分配 ───
        for analysis in room_analyses:
            if analysis.room_type not in ("other", "unknown"):
                continue
            room = room_info.get(analysis.room_id, {})
            furniture = room.get('furniture', [])
            if not furniture:
                continue

            room_area = room_areas.get(analysis.room_id, 0)
            area_rk = area_rank.get(analysis.room_id, total_rooms)

            # 有家具的房间不应是 other，计算最佳匹配类型
            type_scores = {"bedroom": 0, "living_room": 0, "kitchen": 0, "bathroom": 0}
            
            # 1. 家具名称评分
            for item in furniture:
                name = item.get('name', '').lower()
                if any(kw in name for kw in ['fr', 'refrigerator', 'stove', 'oven', 'microwave', 'kitchen_countertop']):
                    type_scores["kitchen"] += 5
                if any(kw in name for kw in ['toilet', 'washbasin', 'shower']):
                    type_scores["bathroom"] += 5
                if any(kw in name for kw in ['bed', 'wardrobe', 'nightstand']):
                    type_scores["bedroom"] += 5
                if any(kw in name for kw in ['tv', 'sofa', 'coffee_table', 'treadmill']):
                    type_scores["living_room"] += 3
            
            # 2. 家具大小模式评分（无 name 时通过尺寸推断）
            has_large_furniture = False
            all_small = True
            for item in furniture:
                w = item.get('position', {}).get('width', 0)
                h = item.get('position', {}).get('height', 0)
                area = w * h
                ratio = max(w, h) / (min(w, h) + 0.1)
                if area >= 200:
                    has_large_furniture = True
                    all_small = False
                    if ratio < 2.0:
                        type_scores["bedroom"] += 3  # 床的大小和比例
                    else:
                        type_scores["bedroom"] += 2  # 衣柜或窄长家具
                elif area >= 100:
                    all_small = False
                    if ratio < 1.5:
                        type_scores["living_room"] += 2  # 方形餐桌/茶几
                    elif ratio >= 2.0:
                        type_scores["bedroom"] += 2  # 窄长书桌/书架
                    else:
                        type_scores["bedroom"] += 1
                        type_scores["living_room"] += 1

            # 3. 行为评分
            activity_info = room.get('activity_info', {})
            primary_period = activity_info.get('primary_time_period', 'none')
            if primary_period == 'night':
                type_scores["bedroom"] += 4
            elif primary_period == 'day':
                type_scores["living_room"] += 4

            # 4. 面积辅助评分
            # 最小房间 + 全小家具 → 强烈倾向 bathroom
            if area_rk <= 1 and all_small and not has_large_furniture:
                type_scores["bathroom"] += 5
            # 大房间 + 有大家具 → 倾向 bedroom
            if has_large_furniture and room_area > 100:
                type_scores["bedroom"] += 3
            # 中等面积 + 连到多个房间 → 倾向 living_room
            if room_area > 40 and room_area < 200:
                type_scores["living_room"] += 1

            # 选择最高分类型
            best_type = max(type_scores, key=type_scores.get)
            best_score = type_scores[best_type]

            if best_score > 0:
                logger.info(f"Layered Priority [Step7 Reassign]: Room {analysis.room_id} other→{best_type} (score: {best_score}, area_rank={area_rk})")
                analysis.room_type = best_type
                analysis.confidence = min(0.5 + best_score * 0.02, 0.85)
            else:
                # 所有分数为0：检查是否所有家具都是微型（< 100）
                all_tiny = all(
                    (item.get('position', {}).get('width', 0) * item.get('position', {}).get('height', 0)) < 100
                    for item in furniture
                )
                if all_tiny:
                    # 微型家具 + 最小房间 → bathroom；否则 → other
                    if area_rk <= 1 and total_rooms < 8 and "bathroom" not in existing_types:
                        analysis.room_type = "bathroom"
                        analysis.confidence = 0.6
                        existing_types.add("bathroom")
                        logger.info(f"Layered Priority [Step7 Default]: Room {analysis.room_id} → bathroom (tiny furniture + smallest room)")
                    else:
                        analysis.room_type = "other"
                        analysis.confidence = 0.5
                        logger.info(f"Layered Priority [Step7 Default]: Room {analysis.room_id} → other (all furniture tiny)")
                else:
                    # 还是无法判断，设为 living_room 作为默认
                    analysis.room_type = "living_room"
                    analysis.confidence = 0.4
                    logger.info(f"Layered Priority [Step7 Default]: Room {analysis.room_id} → living_room (default for furnished room)")

        return room_analyses

    def _apply_partial_priority(self, house_layout: Dict, room_analyses: List[RoomAnalysis]) -> List[RoomAnalysis]:
        """消融实验: 仅家具+设备先验，跳过行为推理 —— 只执行 Step1 + Step2"""
        furniture_priority_mappings = self.FURNITURE_PRIORITY_MAPPINGS

        room_info = {}
        all_rooms = house_layout.get('house', {}).get('rooms', [])
        for room in all_rooms:
            room_id = room.get('id')
            room_info[room_id] = room

        determined_rooms = set()

        # Step 1: 无家具 → other
        for analysis in room_analyses:
            room_id = analysis.room_id
            room = room_info.get(room_id, {})
            furniture = room.get('furniture', [])
            if not furniture:
                analysis.room_type = "other"
                analysis.confidence = 0.8
                determined_rooms.add(room_id)
                logger.info(f"Partial Priority [Step1]: Room {room_id} → other (no furniture)")

        # Step 2: 智能家具确定房间类型
        for analysis in room_analyses:
            room_id = analysis.room_id
            if room_id in determined_rooms:
                continue
            room = room_info.get(room_id, {})
            furniture = room.get('furniture', [])

            highest_priority = 0
            furniture_based_type = None
            smart_devices = [item for item in furniture if item.get('name', '').strip()]

            for item in smart_devices:
                furniture_name = item.get('name', '')
                if furniture_name in furniture_priority_mappings:
                    mapping = furniture_priority_mappings[furniture_name]
                    if mapping["priority"] > highest_priority:
                        highest_priority = mapping["priority"]
                        furniture_based_type = mapping["types"][0]

            if highest_priority >= 4 and furniture_based_type:
                analysis.room_type = furniture_based_type
                analysis.confidence = min(0.5 + (highest_priority * 0.1), 0.95)
                determined_rooms.add(room_id)
                logger.info(f"Partial Priority [Step2]: Room {room_id} → {furniture_based_type} (priority: {highest_priority})")

        # 其余房间保持 LLM 原始推断
        for analysis in room_analyses:
            if analysis.room_id not in determined_rooms:
                logger.info(f"Partial Priority: Room {analysis.room_id} → {analysis.room_type} (LLM original, conf={analysis.confidence})")

        return room_analyses
