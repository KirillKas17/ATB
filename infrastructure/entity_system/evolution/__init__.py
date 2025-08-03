"""
Модуль эволюционной оптимизации системы аналитики.
Включает:
- Генетические алгоритмы
- Эволюционные стратегии
- Адаптивное обучение
- Мета-обучение
- Автоматическую оптимизацию параметров
"""

from .learning.adaptive_learning import AdaptiveLearning
from .learning.meta_learning import MetaLearning

__all__ = [
    "Individual",
    "Population",
    "EvolutionEngine",
    "GeneticOptimizer",
    "AdaptiveLearning",
    "MetaLearning",
]
