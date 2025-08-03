"""
Unit тесты для domain/interfaces/__init__.py.

Покрывает:
- Проверку импортов
- Валидацию экспортов
- Доступность всех интерфейсов
"""

import pytest
from typing import Any

from domain.interfaces import (
    # Базовые интерфейсы
    BaseService,
    OrderbookProtocol,
    PricePatternExtractor,
    # Протоколы предсказаний
    EnhancedPredictionResult,
    PatternPredictorProtocol,
    ReversalPredictorProtocol,
    MarketPhasePredictorProtocol,
    BasePatternPredictor,
    BaseReversalPredictor,
    BaseMarketPhasePredictor,
    # Протоколы риск-менеджмента
    RiskAssessmentResult,
    LiquidityGravityMetrics,
    RiskAnalyzerProtocol,
    LiquidityAnalyzerProtocol,
    StressTesterProtocol,
    PortfolioOptimizerProtocol,
    BaseRiskAnalyzer,
    BaseLiquidityAnalyzer,
    BaseStressTester,
    BasePortfolioOptimizer,
    # Протоколы сигналов
    SessionInfluenceSignal,
    MarketMakerSignal,
    SignalEngineProtocol,
    MarketMakerSignalProtocol,
    BaseSignalEngine,
    BaseMarketMakerSignalEngine,
    # Протоколы стратегий
    MirrorMap,
    FollowSignal,
    FollowResult,
    SymbolSelectionResult,
    StrategyAdvisorProtocol,
    MarketFollowerProtocol,
    SymbolSelectorProtocol,
    BaseStrategyAdvisor,
    BaseMarketFollower,
    BaseSymbolSelector,
)


class TestInterfacesImports:
    """Тесты для импортов интерфейсов."""

    def test_basic_interfaces_imports(self):
        """Тест импорта базовых интерфейсов."""
        assert BaseService is not None
        assert OrderbookProtocol is not None
        assert PricePatternExtractor is not None

    def test_prediction_protocols_imports(self):
        """Тест импорта протоколов предсказаний."""
        assert EnhancedPredictionResult is not None
        assert PatternPredictorProtocol is not None
        assert ReversalPredictorProtocol is not None
        assert MarketPhasePredictorProtocol is not None
        assert BasePatternPredictor is not None
        assert BaseReversalPredictor is not None
        assert BaseMarketPhasePredictor is not None

    def test_risk_protocols_imports(self):
        """Тест импорта протоколов риск-менеджмента."""
        assert RiskAssessmentResult is not None
        assert LiquidityGravityMetrics is not None
        assert RiskAnalyzerProtocol is not None
        assert LiquidityAnalyzerProtocol is not None
        assert StressTesterProtocol is not None
        assert PortfolioOptimizerProtocol is not None
        assert BaseRiskAnalyzer is not None
        assert BaseLiquidityAnalyzer is not None
        assert BaseStressTester is not None
        assert BasePortfolioOptimizer is not None

    def test_signal_protocols_imports(self):
        """Тест импорта протоколов сигналов."""
        assert SessionInfluenceSignal is not None
        assert MarketMakerSignal is not None
        assert SignalEngineProtocol is not None
        assert MarketMakerSignalProtocol is not None
        assert BaseSignalEngine is not None
        assert BaseMarketMakerSignalEngine is not None

    def test_strategy_protocols_imports(self):
        """Тест импорта протоколов стратегий."""
        assert MirrorMap is not None
        assert FollowSignal is not None
        assert FollowResult is not None
        assert SymbolSelectionResult is not None
        assert StrategyAdvisorProtocol is not None
        assert MarketFollowerProtocol is not None
        assert SymbolSelectorProtocol is not None
        assert BaseStrategyAdvisor is not None
        assert BaseMarketFollower is not None
        assert BaseSymbolSelector is not None

    def test_all_interfaces_are_callable_or_classes(self):
        """Тест что все интерфейсы являются классами или протоколами."""
        interfaces = [
            # Базовые интерфейсы
            BaseService,
            OrderbookProtocol,
            PricePatternExtractor,
            # Протоколы предсказаний
            EnhancedPredictionResult,
            PatternPredictorProtocol,
            ReversalPredictorProtocol,
            MarketPhasePredictorProtocol,
            BasePatternPredictor,
            BaseReversalPredictor,
            BaseMarketPhasePredictor,
            # Протоколы риск-менеджмента
            RiskAssessmentResult,
            LiquidityGravityMetrics,
            RiskAnalyzerProtocol,
            LiquidityAnalyzerProtocol,
            StressTesterProtocol,
            PortfolioOptimizerProtocol,
            BaseRiskAnalyzer,
            BaseLiquidityAnalyzer,
            BaseStressTester,
            BasePortfolioOptimizer,
            # Протоколы сигналов
            SessionInfluenceSignal,
            MarketMakerSignal,
            SignalEngineProtocol,
            MarketMakerSignalProtocol,
            BaseSignalEngine,
            BaseMarketMakerSignalEngine,
            # Протоколы стратегий
            MirrorMap,
            FollowSignal,
            FollowResult,
            SymbolSelectionResult,
            StrategyAdvisorProtocol,
            MarketFollowerProtocol,
            SymbolSelectorProtocol,
            BaseStrategyAdvisor,
            BaseMarketFollower,
            BaseSymbolSelector,
        ]

        for interface in interfaces:
            assert interface is not None, f"Interface {interface} is None"
            assert hasattr(interface, '__name__'), f"Interface {interface} has no __name__"

    def test_protocol_attributes(self):
        """Тест атрибутов протоколов."""
        # Проверяем что протоколы имеют необходимые атрибуты
        assert hasattr(PatternPredictorProtocol, '__protocol_attrs__')
        assert hasattr(RiskAnalyzerProtocol, '__protocol_attrs__')
        assert hasattr(SignalEngineProtocol, '__protocol_attrs__')
        assert hasattr(StrategyAdvisorProtocol, '__protocol_attrs__')

    def test_base_class_attributes(self):
        """Тест атрибутов базовых классов."""
        # Проверяем что базовые классы имеют необходимые атрибуты
        assert hasattr(BaseRiskAnalyzer, '__abstractmethods__')
        assert hasattr(BaseLiquidityAnalyzer, '__abstractmethods__')
        assert hasattr(BaseSignalEngine, '__abstractmethods__')
        assert hasattr(BaseStrategyAdvisor, '__abstractmethods__')

    def test_dataclass_attributes(self):
        """Тест атрибутов dataclass."""
        # Проверяем что dataclass имеют необходимые атрибуты
        assert hasattr(RiskAssessmentResult, '__dataclass_fields__')
        assert hasattr(LiquidityGravityMetrics, '__dataclass_fields__')
        assert hasattr(SessionInfluenceSignal, '__dataclass_fields__')
        assert hasattr(MarketMakerSignal, '__dataclass_fields__')

    def test_module_docstring(self):
        """Тест наличия документации модуля."""
        import domain.interfaces as interfaces_module
        assert interfaces_module.__doc__ is not None
        assert "Интерфейсы для внешних зависимостей доменного слоя" in interfaces_module.__doc__

    def test_all_list_completeness(self):
        """Тест полноты списка __all__."""
        import domain.interfaces as interfaces_module
        
        # Проверяем что все экспортируемые элементы присутствуют в __all__
        expected_exports = [
            # Базовые интерфейсы
            "BaseService",
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
        
        for export in expected_exports:
            assert export in interfaces_module.__all__, f"Export {export} missing from __all__"

    def test_no_cyclic_imports(self):
        """Тест отсутствия циклических импортов."""
        # Проверяем что модуль может быть импортирован без ошибок
        try:
            import domain.interfaces
            assert True, "Module imported successfully"
        except ImportError as e:
            pytest.fail(f"Cyclic import detected: {e}")

    def test_interface_categories(self):
        """Тест категоризации интерфейсов."""
        # Проверяем что интерфейсы правильно категоризированы
        prediction_interfaces = [
            EnhancedPredictionResult,
            PatternPredictorProtocol,
            ReversalPredictorProtocol,
            MarketPhasePredictorProtocol,
            BasePatternPredictor,
            BaseReversalPredictor,
            BaseMarketPhasePredictor,
        ]
        
        risk_interfaces = [
            RiskAssessmentResult,
            LiquidityGravityMetrics,
            RiskAnalyzerProtocol,
            LiquidityAnalyzerProtocol,
            StressTesterProtocol,
            PortfolioOptimizerProtocol,
            BaseRiskAnalyzer,
            BaseLiquidityAnalyzer,
            BaseStressTester,
            BasePortfolioOptimizer,
        ]
        
        signal_interfaces = [
            SessionInfluenceSignal,
            MarketMakerSignal,
            SignalEngineProtocol,
            MarketMakerSignalProtocol,
            BaseSignalEngine,
            BaseMarketMakerSignalEngine,
        ]
        
        strategy_interfaces = [
            MirrorMap,
            FollowSignal,
            FollowResult,
            SymbolSelectionResult,
            StrategyAdvisorProtocol,
            MarketFollowerProtocol,
            SymbolSelectorProtocol,
            BaseStrategyAdvisor,
            BaseMarketFollower,
            BaseSymbolSelector,
        ]
        
        # Проверяем что все интерфейсы в категориях не None
        all_interfaces = prediction_interfaces + risk_interfaces + signal_interfaces + strategy_interfaces
        for interface in all_interfaces:
            assert interface is not None, f"Interface {interface} is None in category"

    def test_interface_naming_convention(self):
        """Тест соглашения об именовании интерфейсов."""
        # Проверяем что протоколы заканчиваются на Protocol
        protocol_interfaces = [
            PatternPredictorProtocol,
            ReversalPredictorProtocol,
            MarketPhasePredictorProtocol,
            RiskAnalyzerProtocol,
            LiquidityAnalyzerProtocol,
            StressTesterProtocol,
            PortfolioOptimizerProtocol,
            SignalEngineProtocol,
            MarketMakerSignalProtocol,
            StrategyAdvisorProtocol,
            MarketFollowerProtocol,
            SymbolSelectorProtocol,
        ]
        
        for protocol in protocol_interfaces:
            assert protocol.__name__.endswith('Protocol'), f"Protocol {protocol.__name__} should end with 'Protocol'"

        # Проверяем что базовые классы начинаются с Base
        base_classes = [
            BasePatternPredictor,
            BaseReversalPredictor,
            BaseMarketPhasePredictor,
            BaseRiskAnalyzer,
            BaseLiquidityAnalyzer,
            BaseStressTester,
            BasePortfolioOptimizer,
            BaseSignalEngine,
            BaseMarketMakerSignalEngine,
            BaseStrategyAdvisor,
            BaseMarketFollower,
            BaseSymbolSelector,
        ]
        
        for base_class in base_classes:
            assert base_class.__name__.startswith('Base'), f"Base class {base_class.__name__} should start with 'Base'"

        # Проверяем что dataclass заканчиваются на Result, Signal, Metrics
        dataclass_interfaces = [
            RiskAssessmentResult,
            LiquidityGravityMetrics,
            SessionInfluenceSignal,
            MarketMakerSignal,
            MirrorMap,
            FollowSignal,
            FollowResult,
            SymbolSelectionResult,
        ]
        
        for dataclass in dataclass_interfaces:
            name = dataclass.__name__
            assert any(name.endswith(suffix) for suffix in ['Result', 'Signal', 'Metrics', 'Map']), \
                f"Dataclass {name} should end with 'Result', 'Signal', 'Metrics', or 'Map'" 