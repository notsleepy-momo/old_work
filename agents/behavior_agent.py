# 5.负责行为分析
from typing import Dict, List, Any
import json
import yaml
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .data_models import BehaviorAnalysis, ValidationResult, BehaviorGenerationResult
from .prompts import BEHAVIOR_AGENT_PROMPT
from llm_config import (
    DEFAULT_LLM_MODEL,
    require_shared_model,
    resolve_api_key,
    resolve_base_url,
)

logger = logging.getLogger(__name__)

class BehaviorAgent:
    """Behavior agent for analyzing trajectory data"""

    def __init__(self, api_key: str = None, base_url: str = None,
                 model: str = DEFAULT_LLM_MODEL, callbacks: list = None):
        """Initialize behavior agent"""
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

        self.prompt = ChatPromptTemplate.from_template(BEHAVIOR_AGENT_PROMPT)
        self.llm = ChatOpenAI(
            model=require_shared_model(model), temperature=0.0,
            **common_llm_kwargs)
        self.chain = self.prompt | self.llm

    def generate_behavior_analyses(self, house_layout: Dict, trajectory_data: List[Dict], room_analyses: List[Any] = None) -> BehaviorGenerationResult:
        """Generate behavior analyses"""
        try:
            house_layout_yaml = yaml.dump(house_layout)
            trajectory_data_json = json.dumps(trajectory_data)

            room_types_info = ""
            if room_analyses:
                room_types_info = "\n\n## Room Types\n"
                for analysis in room_analyses:
                    room_types_info += f"- Room {analysis.room_id}: {analysis.room_type} (confidence: {analysis.confidence:.2f})\n"

            result = self.chain.invoke({
                "house_layout_yaml": house_layout_yaml,
                "trajectory_data": trajectory_data_json,
                "room_types_info": room_types_info
            })

            if not hasattr(result, 'content') or not result.content:
                default_analyses = self._get_default_behavior_analyses(house_layout)
                return BehaviorGenerationResult(behavior_analyses=default_analyses, confidence=0.5)

            import re
            json_match = re.search(r'\[.*?\]|\{.*?\}', result.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    behavior_analyses_data = json.loads(json_str)
                    behavior_analyses = [BehaviorAnalysis(**b) for b in behavior_analyses_data]
                    return BehaviorGenerationResult(behavior_analyses=behavior_analyses, confidence=0.7)
                except (json.JSONDecodeError, TypeError):
                    pass

            default_analyses = self._get_default_behavior_analyses(house_layout)
            return BehaviorGenerationResult(behavior_analyses=default_analyses, confidence=0.5)
        except Exception as e:
            logger.error(f"Behavior analysis failed: {str(e)}")
            default_analyses = self._get_default_behavior_analyses(house_layout)
            return BehaviorGenerationResult(behavior_analyses=default_analyses, confidence=0.5)

    def _get_default_behavior_analyses(self, house_layout: Dict) -> List[BehaviorAnalysis]:
        behavior_analyses = []
        rooms = house_layout.get('house', {}).get('rooms', [])
        for room in rooms:
            room_id = room.get('id')
            activity = room.get('activity_info', {})
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

            behavior_analyses.append(BehaviorAnalysis(
                room_id=room_id,
                main_activities=activities,
                frequent_areas=[],
                time_based_patterns={
                    "primary_time_period": primary,
                    "total_duration": total_dur,
                    "visit_count": activity.get('visit_count', 0),
                }
            ))
        return behavior_analyses
