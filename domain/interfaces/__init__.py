"""
Интерфейсы для внешних зависимостей доменного слоя.
Этот модуль содержит протоколы и интерфейсы для взаимодействия
с внешними слоями (infrastructure, shared) без создания циклических зависимостей.
"""

# Базовые интерфейсы
from .base_service import BaseService
from .cache_protocol import CacheProtocol
from .orderbook_protocol import OrderbookProtocol
from .price_pattern_extractor import PricePatternExtractor

# Новые протоколы для исправления DDD нарушений
from .prediction_protocols import (
    BaseMarketPhasePredictor,
    BasePatternPredictor,
    BaseReversalPredictor,
    EnhancedPredictionResult,
    MarketPhasePredictorProtocol,
    PatternPredictorProtocol,
    ReversalPredictorProtocol,
)
from .risk_protocols import (
    BaseLiquidityAnalyzer,
    BasePortfolioOptimizer,
    BaseRiskAnalyzer,
    BaseStressTester,
    LiquidityAnalyzerProtocol,
    LiquidityGravityMetrics,
    PortfolioOptimizerProtocol,
    RiskAnalyzerProtocol,
    RiskAssessmentResult,
    StressTesterProtocol,
)
from .signal_protocols import (
    BaseMarketMakerSignalEngine,
    BaseSignalEngine,
    MarketMakerSignal,
    MarketMakerSignalProtocol,
    SessionInfluenceSignal,
    SignalEngineProtocol,
)
from .strategy_protocols import (
    BaseMarketFollower,
    BaseStrategyAdvisor,
    BaseSymbolSelector,
    FollowResult,
    FollowSignal,
    MarketFollowerProtocol,
    MirrorMap,
    StrategyAdvisorProtocol,
    SymbolSelectionResult,
    SymbolSelectorProtocol,
)

__all__ = [
    # Базовые интерфейсы
    "BaseService",
    "CacheProtocol",
    "OrderbookProtocol",
    "PricePatternExtractor",
    # Протоколы предсказаний
    "EnhancedPredictionResult",
    "PatternPredictorProtocol",
    "ReversalPredictorProtocol",
    "MarketPhasePredictorProtocol",
    "BasePatternPredictor",
    "BaseReversalPredictor",
    "BaseMarketPhasePredictor",
    # Протоколы риск-менеджмента
    "RiskAssessmentResult",
    "LiquidityGravityMetrics",
    "RiskAnalyzerProtocol",
    "LiquidityAnalyzerProtocol",
    "StressTesterProtocol",
    "PortfolioOptimizerProtocol",
    "BaseRiskAnalyzer",
    "BaseLiquidityAnalyzer",
    "BaseStressTester",
    "BasePortfolioOptimizer",
    # Протоколы сигналов
    "SessionInfluenceSignal",
    "MarketMakerSignal",
    "SignalEngineProtocol",
    "MarketMakerSignalProtocol",
    "BaseSignalEngine",
    "BaseMarketMakerSignalEngine",
    # Протоколы стратегий
    "MirrorMap",
    "FollowSignal",
    "FollowResult",
    "SymbolSelectionResult",
    "StrategyAdvisorProtocol",
    "MarketFollowerProtocol",
    "SymbolSelectorProtocol",
    "BaseStrategyAdvisor",
    "BaseMarketFollower",
    "BaseSymbolSelector",
]
