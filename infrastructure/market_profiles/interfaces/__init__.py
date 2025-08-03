"""
Интерфейсы для модуля market_profiles.
"""

from .storage_interfaces import (
    IBehaviorHistoryStorage,
    IPatternAnalyzer,
    IPatternStorage,
)

__all__ = ["IPatternStorage", "IBehaviorHistoryStorage", "IPatternAnalyzer"]
