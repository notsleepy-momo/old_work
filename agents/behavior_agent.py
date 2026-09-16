# 5.负责行为分析
import re
import time
from typing import Any, Dict, List
import json
import yaml
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .data_models import BehaviorAnalysis, BehaviorGenerationResult
from .prompts import BEHAVIOR_AGENT_ROOM_PROMPT
from llm_config import (
    DEFAULT_LLM_MODEL,
    require_shared_model,
    resolve_api_key,
    resolve_base_url,
)

logger = logging.getLogger(__name__)


def _room_id_at(rooms: List[Dict], x: float, y: float):
    """Return the id of the room containing point (x, y).

    Same point-in-rectangle rule as RoomAgent._get_room_by_position.
    """
    for room in rooms:
        pos = room.get('position', {})
        room_x = pos.get('x', 0)
        room_y = pos.get('y', 0)
        width = pos.get('width', 0)
        height = pos.get('height', 0)
        if room_x <= x < room_x + width and room_y <= y < room_y + height:
            return room.get('id')
    return None


class BehaviorAgent:
    """Behavior agent analyzing each room with one small LLM request per room.

    A single request covering all rooms needs 55-90s of generation, which API
    gateways cut off at their ~60s connection limit; per-room requests stay
    within a few seconds and degrade per room instead of wholesale.
    """

    def __init__(self, api_key: str = None, base_url: str = None,
                 model: str = DEFAULT_LLM_MODEL, callbacks: list = None):
        """Initialize behavior agent"""
        common_llm_kwargs = {
            "api_key": resolve_api_key(api_key),
            # Per-room requests finish in seconds; 2 client retries bound the
            # worst case per room without stalling the whole step for minutes.
            "max_retries": 2,
            "timeout": 60,
        }
        resolved_base_url = resolve_base_url(base_url)
        if resolved_base_url:
            common_llm_kwargs["base_url"] = resolved_base_url
        if callbacks:
            common_llm_kwargs["callbacks"] = callbacks

        self.prompt = ChatPromptTemplate.from_template(BEHAVIOR_AGENT_ROOM_PROMPT)
        self.llm = ChatOpenAI(
            model=require_shared_model(model), temperature=0.0,
            **common_llm_kwargs)
        self.chain = self.prompt | self.llm

    def generate_behavior_analyses(self, house_layout: Dict, trajectory_data: List[Dict],
                                   room_analyses: List[Any] = None) -> BehaviorGenerationResult:
        """Generate one behavior analysis per room via one LLM request per room.

        Rooms whose request fails (or that have no stop points) fall back to
        the deterministic default analysis individually, so one flaky request
        no longer discards the whole step's result.
        """
        rooms = house_layout.get('house', {}).get('rooms', [])
        if not rooms:
            return BehaviorGenerationResult(behavior_analyses=[], confidence=0.5)

        stops_by_room = self._slice_trajectory_by_room(rooms, trajectory_data)
        type_by_room = {getattr(a, 'room_id', None): a for a in (room_analyses or [])}

        behavior_analyses = []
        n_llm_ok = 0
        for room in rooms:
            room_id = room.get('id')
            stops = stops_by_room.get(room_id, [])
            analysis = None
            if stops:
                analysis = self._analyze_single_room(room, stops,
                                                     type_by_room.get(room_id))
            if analysis is not None:
                n_llm_ok += 1
            else:
                reason = "无停驻点" if not stops else "LLM 失败"
                logger.info(f"[Behavior] {room_id}: 回退默认分析 ({reason})")
                analysis = self._default_behavior_for_room(room)
            behavior_analyses.append(analysis)

        logger.info(f"[Behavior] 全部完成: LLM 成功 {n_llm_ok}/{len(rooms)} 个房间")
        confidence = 0.7 if n_llm_ok == len(rooms) else 0.5
        return BehaviorGenerationResult(behavior_analyses=behavior_analyses,
                                        confidence=confidence)

    def _slice_trajectory_by_room(self, rooms: List[Dict],
                                  trajectory_data: List[Dict]) -> Dict:
        """Group trajectory stop points by the room they fall into."""
        stops_by_room = {room.get('id'): [] for room in rooms}

        points = trajectory_data
        if isinstance(points, dict):
            for key in ('trajectories', 'trajectory', 'points', 'data'):
                if isinstance(points.get(key), list):
                    points = points[key]
                    break
            else:
                points = [points]
        if not isinstance(points, list):
            return stops_by_room

        for stop in points:
            if not isinstance(stop, dict):
                continue
            center = stop.get('center_position', {}) or {}
            x, y = center.get('x'), center.get('y')
            if x is None or y is None:
                continue
            room_id = _room_id_at(rooms, x, y)
            if room_id in stops_by_room:
                stops_by_room[room_id].append(stop)
        return stops_by_room

    def _analyze_single_room(self, room: Dict, stops: List[Dict],
                             room_analysis) -> BehaviorAnalysis:
        """Request the behavior analysis of one room; None on any failure."""
        room_id = room.get('id')
        t0 = time.time()
        try:
            room_types_info = ""
            if room_analysis is not None:
                room_types_info = (f"- Room {room_id}: {room_analysis.room_type} "
                                   f"(confidence: {room_analysis.confidence:.2f})")

            result = self.chain.invoke({
                "room_layout_yaml": yaml.dump(room, allow_unicode=True, sort_keys=False),
                "trajectory_data": json.dumps(stops, ensure_ascii=False),
                "room_types_info": room_types_info,
            })

            data = self._extract_json_object(getattr(result, 'content', None))
            if data is None:
                logger.warning(f"[Behavior] {room_id}: LLM 输出无法解析为 JSON "
                               f"({time.time() - t0:.1f}s)")
                return None
            analysis = self._to_behavior_analysis(room, data)
            logger.info(f"[Behavior] {room_id}: OK ({time.time() - t0:.1f}s, "
                        f"{len(stops)} 个停驻点)")
            return analysis
        except Exception as e:
            logger.warning(f"[Behavior] {room_id}: 请求失败 ({time.time() - t0:.1f}s): "
                           f"{type(e).__name__}: {e}")
            return None

    @staticmethod
    def _extract_json_object(content: str) -> Dict:
        """Extract the first JSON object from an LLM response."""
        if not content:
            return None
        match = re.search(r'\{.*\}|\[.*\]', content, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group(0))
        if isinstance(data, list):
            data = data[0] if data and isinstance(data[0], dict) else None
        return data if isinstance(data, dict) else None

    def _to_behavior_analysis(self, room: Dict, data: Dict) -> BehaviorAnalysis:
        """Build a BehaviorAnalysis from the model output, coercing types to
        the BehaviorAnalysis schema and filling gaps from activity_info."""
        activity = room.get('activity_info', {}) or {}
        patterns = data.get('time_based_patterns') or {}

        frequent_areas = []
        for area in (data.get('frequent_areas') or []):
            frequent_areas.append(area if isinstance(area, dict)
                                  else {"description": str(area)})
        main_activities = [str(a) for a in (data.get('main_activities') or [])]

        return BehaviorAnalysis(
            room_id=room.get('id'),
            main_activities=main_activities or ["no_significant_activity"],
            frequent_areas=frequent_areas,
            time_based_patterns={
                "primary_time_period": patterns.get(
                    'primary_time_period', activity.get('primary_time_period', 'none')),
                "peak_hours": patterns.get('peak_hours', []),
                "total_duration": patterns.get(
                    'total_duration', activity.get('total_duration', 0)),
                "visit_count": patterns.get(
                    'visit_count', activity.get('visit_count', 0)),
            },
        )

    def _default_behavior_for_room(self, room: Dict) -> BehaviorAnalysis:
        """Deterministic fallback for a single room."""
        room_id = room.get('id')
        activity = room.get('activity_info', {}) or {}
        primary = activity.get('primary_time_period', 'none')
        total_dur = activity.get('total_duration', 0)

        if primary == 'night':
            activities = ["sleeping", "resting"]
        elif primary == 'day' and total_dur > 1800:
            activities = ["working", "entertaining", "socializing"]
        elif total_dur > 0:
            activities = ["passing_through", "brief_stay"]
        else:
            activities = ["no_significant_activity"]

        return BehaviorAnalysis(
            room_id=room_id,
            main_activities=activities,
            frequent_areas=[],
            time_based_patterns={
                "primary_time_period": primary,
                "total_duration": total_dur,
                "visit_count": activity.get('visit_count', 0),
            }
        )
