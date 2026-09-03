# 1.包含所有的数据模型定义
from typing import Dict, List
from pydantic import BaseModel, Field

# Data models
class RoomAnalysis(BaseModel):
    """Room analysis result"""
    room_id: str
    room_type: str
    confidence: float

class BehaviorAnalysis(BaseModel):
    """Behavior analysis result"""
    room_id: str
    main_activities: List[str]
    frequent_areas: List[Dict]
    time_based_patterns: Dict

class FurnitureNaming(BaseModel):
    """Furniture naming result"""
    furniture_id: str
    name: str
    room_id: str
    confidence: float

class ValidationResult(BaseModel):
    """Validation result"""
    is_valid: bool
    errors: List[str]
    suggestions: List[str]

class RoomGenerationResult(BaseModel):
    """Room generation result"""
    room_analyses: List[RoomAnalysis]
    confidence: float

class BehaviorGenerationResult(BaseModel):
    """Behavior generation result"""
    behavior_analyses: List[BehaviorAnalysis]
    confidence: float

class FurnitureNamingGenerationResult(BaseModel):
    """Furniture naming generation result"""
    furniture_namings: List[FurnitureNaming]
    confidence: float
