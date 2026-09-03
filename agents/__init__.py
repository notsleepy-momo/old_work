# 8.包含所有的智能体定义
from .data_models import (
    RoomAnalysis,
    BehaviorAnalysis,
    FurnitureNaming,
    ValidationResult,
    RoomGenerationResult,
    BehaviorGenerationResult,
    FurnitureNamingGenerationResult
)
from .room_agent import RoomAgent
from .behavior_agent import BehaviorAgent
from .furniture_naming_agent import FurnitureNamingAgent
from .furniture_detection_agent import FurnitureDetectionAgent
from .prompts import (
    ROOM_AGENT_PROMPT,
    BEHAVIOR_AGENT_PROMPT,
    FURNITURE_AGENT_PROMPT,
)

__all__ = [
    "RoomAnalysis",
    "BehaviorAnalysis",
    "FurnitureNaming",
    "ValidationResult",
    "RoomGenerationResult",
    "BehaviorGenerationResult",
    "FurnitureNamingGenerationResult",
    "RoomAgent",
    "BehaviorAgent",
    "FurnitureNamingAgent",
    "FurnitureDetectionAgent",
    "ROOM_AGENT_PROMPT",
    "BEHAVIOR_AGENT_PROMPT",
    "FURNITURE_AGENT_PROMPT",
]
