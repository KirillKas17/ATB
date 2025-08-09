"""
Unit тесты для strategy_types.

Покрывает:
- Перечисления (StrategyDirection, StrategyType, MarketRegime)
- Датаклассы (StrategyMetrics, Signal, StrategyConfig, StrategyAnalysis)
- TypedDict (StrategyValidationResult, StrategyOptimizationResult, StrategyPerformanceResult)
- Протоколы (StrategyServiceProtocol, StrategyProtocol, StrategyFactoryProtocol)
- Эволюционные типы (EvolutionConfig, EvolutionMetrics)
- Адаптивные типы (AdaptationConfig, MarketContext)
- ML типы (MLModelConfig, MLPrediction)
- Риск-менеджмент типы (RiskConfig, RiskAssessment)
"""

import pytest
from datetime import datetime
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock
import pandas as pd

from domain.type_definitions.strategy_types import (
    StrategyDirection,
    StrategyType,
    MarketRegime,
    StrategyMetrics,
    Signal,
    StrategyConfig,
    StrategyAnalysis,
    StrategyValidationResult,
    StrategyOptimizationResult,
    StrategyPerformanceResult,
    StrategyServiceProtocol,
    StrategyProtocol,
    StrategyFactoryProtocol,
    EvolutionConfig,
    EvolutionMetrics,
    AdaptationConfig,
    MarketContext,
    MLModelConfig,
    MLPrediction,
    RiskConfig,
    RiskAssessment,
)


class TestStrategyDirection:
    """Тесты для StrategyDirection."""

    def test_enum_values(self):
        """Тест значений перечисления."""
        assert StrategyDirection.LONG == "long"
        assert StrategyDirection.SHORT == "short"
        assert StrategyDirection.HOLD == "hold"
        assert StrategyDirection.CLOSE == "close"

    def test_enum_membership(self):
        """Тест принадлежности к перечислению."""
        assert "long" in StrategyDirection
        assert "short" in StrategyDirection
        assert "hold" in StrategyDirection
        assert "close" in StrategyDirection

    def test_enum_iteration(self):
        """Тест итерации по перечислению."""
        directions = list(StrategyDirection)
        assert len(directions) == 4
        assert all(isinstance(d, StrategyDirection) for d in directions)


class TestStrategyType:
    """Тесты для StrategyType."""

    def test_enum_values(self):
        """Тест значений перечисления."""
        expected_types = [
            "trend",
            "mean_reversion",
            "breakout",
            "scalping",
            "arbitrage",
            "pairs_trading",
            "statistical_arbitrage",
            "momentum",
            "volatility",
            "grid",
            "martingale",
            "hedging",
            "manipulation",
            "reversal",
            "sideways",
            "adaptive",
            "evolvable",
            "deep_learning",
            "random_forest",
            "regime_adaptive",
        ]

        for strategy_type in expected_types:
            assert hasattr(StrategyType, strategy_type.upper().replace("_", ""))

    def test_enum_membership(self):
        """Тест принадлежности к перечислению."""
        assert "trend" in StrategyType
        assert "mean_reversion" in StrategyType
        assert "breakout" in StrategyType

    def test_enum_count(self):
        """Тест количества типов стратегий."""
        assert len(StrategyType) == 20


class TestMarketRegime:
    """Тесты для MarketRegime."""

    def test_enum_values(self):
        """Тест значений перечисления."""
        expected_regimes = [
            "trending_up",
            "trending_down",
            "sideways",
            "volatile",
            "low_volatility",
            "breakout",
            "reversal",
            "manipulation",
        ]

        for regime in expected_regimes:
            assert hasattr(MarketRegime, regime.upper().replace("_", ""))

    def test_enum_membership(self):
        """Тест принадлежности к перечислению."""
        assert "trending_up" in MarketRegime
        assert "sideways" in MarketRegime
        assert "volatile" in MarketRegime

    def test_enum_count(self):
        """Тест количества рыночных режимов."""
        assert len(MarketRegime) == 8


class TestStrategyMetrics:
    """Тесты для StrategyMetrics."""

    @pytest.fixture
    def sample_metrics(self) -> StrategyMetrics:
        """Тестовые метрики."""
        return StrategyMetrics(
            total_signals=100,
            successful_signals=65,
            failed_signals=35,
            avg_profit=0.02,
            avg_loss=-0.01,
            win_rate=0.65,
            profit_factor=1.5,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            max_drawdown=0.15,
            recovery_factor=2.0,
            expectancy=0.008,
            risk_reward_ratio=2.0,
            kelly_criterion=0.3,
            volatility=0.25,
            mar_ratio=1.8,
            ulcer_index=0.12,
            omega_ratio=1.4,
            gini_coefficient=0.3,
            tail_ratio=1.1,
            skewness=0.5,
            kurtosis=3.2,
            var_95=0.08,
            cvar_95=0.12,
            drawdown_duration=30.0,
            max_equity=1.5,
            min_equity=0.85,
            median_trade=0.005,
            median_duration=24.0,
            profit_streak=8,
            loss_streak=3,
            stability=0.85,
            calmar_ratio=1.6,
            treynor_ratio=1.3,
            information_ratio=0.8,
            kappa_ratio=1.2,
            gain_loss_ratio=2.5,
            additional={"custom_metric": 0.95},
        )

    def test_creation(self, sample_metrics):
        """Тест создания метрик."""
        assert sample_metrics.total_signals == 100
        assert sample_metrics.successful_signals == 65
        assert sample_metrics.win_rate == 0.65
        assert sample_metrics.additional["custom_metric"] == 0.95

    def test_default_values(self):
        """Тест значений по умолчанию."""
        metrics = StrategyMetrics()
        assert metrics.total_signals == 0
        assert metrics.successful_signals == 0
        assert metrics.win_rate == 0.0
        assert isinstance(metrics.additional, dict)

    def test_immutability(self, sample_metrics):
        """Тест неизменяемости полей."""
        # Проверяем, что поля можно изменять (dataclass по умолчанию изменяемый)
        sample_metrics.total_signals = 200
        assert sample_metrics.total_signals == 200


class TestSignal:
    """Тесты для Signal."""

    @pytest.fixture
    def sample_signal(self) -> Signal:
        """Тестовый сигнал."""
        return Signal(
            direction=StrategyDirection.LONG,
            entry_price=50000.0,
            stop_loss=48000.0,
            take_profit=52000.0,
            volume=0.1,
            confidence=0.85,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            metadata={"source": "technical_analysis"},
            strategy_type=StrategyType.TREND,
            market_regime=MarketRegime.TRENDING_UP,
            risk_score=0.3,
            expected_return=0.04,
            holding_period=24,
            position_size=0.05,
        )

    def test_creation(self, sample_signal):
        """Тест создания сигнала."""
        assert sample_signal.direction == StrategyDirection.LONG
        assert sample_signal.entry_price == 50000.0
        assert sample_signal.stop_loss == 48000.0
        assert sample_signal.confidence == 0.85
        assert sample_signal.metadata["source"] == "technical_analysis"

    def test_default_values(self):
        """Тест значений по умолчанию."""
        signal = Signal(direction=StrategyDirection.SHORT, entry_price=100.0)
        assert signal.stop_loss is None
        assert signal.take_profit is None
        assert signal.confidence == 1.0
        assert isinstance(signal.timestamp, datetime)
        assert isinstance(signal.metadata, dict)

    def test_optional_fields(self):
        """Тест опциональных полей."""
        signal = Signal(
            direction=StrategyDirection.HOLD, entry_price=100.0, stop_loss=None, take_profit=None, volume=None
        )
        assert signal.stop_loss is None
        assert signal.take_profit is None
        assert signal.volume is None


class TestStrategyConfig:
    """Тесты для StrategyConfig."""

    @pytest.fixture
    def sample_config(self) -> StrategyConfig:
        """Тестовая конфигурация."""
        return StrategyConfig(
            strategy_type=StrategyType.TREND,
            name="Moving Average Crossover",
            description="Стратегия на основе пересечения скользящих средних",
            parameters={"fast_period": 10, "slow_period": 20, "signal_period": 9},
            risk_per_trade=0.02,
            max_position_size=0.1,
            confidence_threshold=0.7,
            use_stop_loss=True,
            use_take_profit=True,
            trailing_stop=True,
            trailing_stop_activation=0.02,
            trailing_stop_distance=0.01,
            timeframes=["1h", "4h"],
            symbols=["BTC/USDT", "ETH/USDT"],
            market_regime_filter=[MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN],
            enabled=True,
            version="1.0.0",
            created_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 1),
        )

    def test_creation(self, sample_config):
        """Тест создания конфигурации."""
        assert sample_config.strategy_type == StrategyType.TREND
        assert sample_config.name == "Moving Average Crossover"
        assert sample_config.risk_per_trade == 0.02
        assert sample_config.parameters["fast_period"] == 10
        assert len(sample_config.timeframes) == 2
        assert len(sample_config.symbols) == 2

    def test_default_values(self):
        """Тест значений по умолчанию."""
        config = StrategyConfig(
            strategy_type=StrategyType.SCALPING, name="Test Strategy", description="Test Description", parameters={}
        )
        assert config.risk_per_trade == 0.02
        assert config.max_position_size == 0.1
        assert config.confidence_threshold == 0.7
        assert config.use_stop_loss is True
        assert config.enabled is True
        assert config.version == "1.0.0"
        assert isinstance(config.timeframes, list)
        assert isinstance(config.symbols, list)
        assert isinstance(config.market_regime_filter, list)


class TestStrategyAnalysis:
    """Тесты для StrategyAnalysis."""

    @pytest.fixture
    def sample_analysis(self) -> StrategyAnalysis:
        """Тестовый анализ."""
        market_data = pd.DataFrame({"close": [100, 101, 102, 103, 104], "volume": [1000, 1100, 1200, 1300, 1400]})

        indicators = {"sma_20": pd.Series([99, 99.5, 100, 100.5, 101]), "rsi": pd.Series([50, 55, 60, 65, 70])}

        signals = [Signal(direction=StrategyDirection.LONG, entry_price=102.0, confidence=0.8)]

        metrics = StrategyMetrics(total_signals=10, successful_signals=7, win_rate=0.7)

        return StrategyAnalysis(
            strategy_id="test_strategy_001",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            market_data=market_data,
            indicators=indicators,
            signals=signals,
            metrics=metrics,
            market_regime=MarketRegime.TRENDING_UP,
            confidence=0.75,
            risk_assessment={"var_95": 0.05, "max_drawdown": 0.1, "volatility": 0.2},
            recommendations=["Увеличить размер позиции", "Добавить дополнительные фильтры"],
            metadata={"analysis_version": "1.0"},
        )

    def test_creation(self, sample_analysis):
        """Тест создания анализа."""
        assert sample_analysis.strategy_id == "test_strategy_001"
        assert sample_analysis.market_regime == MarketRegime.TRENDING_UP
        assert sample_analysis.confidence == 0.75
        assert len(sample_analysis.signals) == 1
        assert len(sample_analysis.recommendations) == 2
        assert isinstance(sample_analysis.market_data, pd.DataFrame)
        assert isinstance(sample_analysis.indicators, dict)

    def test_market_data_shape(self, sample_analysis):
        """Тест формы рыночных данных."""
        assert sample_analysis.market_data.shape == (5, 2)
        assert "close" in sample_analysis.market_data.columns
        assert "volume" in sample_analysis.market_data.columns

    def test_indicators_structure(self, sample_analysis):
        """Тест структуры индикаторов."""
        assert "sma_20" in sample_analysis.indicators
        assert "rsi" in sample_analysis.indicators
        assert isinstance(sample_analysis.indicators["sma_20"], pd.Series)


class TestTypedDicts:
    """Тесты для TypedDict типов."""

    def test_strategy_validation_result(self):
        """Тест StrategyValidationResult."""
        result: StrategyValidationResult = {
            "errors": ["Invalid parameter: fast_period"],
            "warnings": ["Low confidence threshold"],
            "is_valid": False,
            "validation_score": 0.6,
            "recommendations": ["Increase fast_period", "Adjust confidence threshold"],
        }

        assert result["is_valid"] is False
        assert result["validation_score"] == 0.6
        assert len(result["errors"]) == 1
        assert len(result["warnings"]) == 1

    def test_strategy_optimization_result(self):
        """Тест StrategyOptimizationResult."""
        result: StrategyOptimizationResult = {
            "original_params": {"fast_period": 10, "slow_period": 20},
            "optimized_params": {"fast_period": 12, "slow_period": 22},
            "improvement_expected": True,
            "optimization_method": "genetic_algorithm",
            "performance_improvement": 0.15,
            "risk_adjustment": 0.05,
            "confidence_interval": (0.10, 0.20),
        }

        assert result["improvement_expected"] is True
        assert result["performance_improvement"] == 0.15
        assert result["confidence_interval"] == (0.10, 0.20)

    def test_strategy_performance_result(self):
        """Тест StrategyPerformanceResult."""
        result: StrategyPerformanceResult = {
            "analysis": {"total_trades": 100, "avg_return": 0.02},
            "metrics": StrategyMetrics(total_signals=100, win_rate=0.65),
            "backtest_results": {"sharpe_ratio": 1.2, "max_drawdown": 0.1},
            "risk_metrics": {"var_95": 0.05, "volatility": 0.2},
            "comparison_benchmark": {"benchmark_return": 0.01, "alpha": 0.01},
        }

        assert result["analysis"]["total_trades"] == 100
        assert result["metrics"].win_rate == 0.65
        assert result["risk_metrics"]["var_95"] == 0.05


class TestProtocols:
    """Тесты для протоколов."""

    def test_strategy_service_protocol_implementation(self):
        """Тест реализации StrategyServiceProtocol."""

        class MockStrategyService:
            async def create_strategy(self, config: StrategyConfig) -> Any:
                return Mock()

            async def validate_strategy(self, strategy: Any) -> StrategyValidationResult:
                return {"is_valid": True, "errors": [], "warnings": []}

            async def optimize_strategy(
                self, strategy: Any, historical_data: pd.DataFrame
            ) -> StrategyOptimizationResult:
                return {"improvement_expected": True, "performance_improvement": 0.1}

            async def analyze_performance(self, strategy: Any, period: Any) -> StrategyPerformanceResult:
                return {"analysis": {}, "metrics": StrategyMetrics()}

            async def backtest_strategy(self, strategy: Any, data: pd.DataFrame) -> Dict[str, Any]:
                return {"total_return": 0.15}

            async def get_strategy_metrics(self, strategy_id: str) -> StrategyMetrics:
                return StrategyMetrics()

            async def update_strategy_config(self, strategy_id: str, config: StrategyConfig) -> bool:
                return True

        service = MockStrategyService()
        assert isinstance(service, StrategyServiceProtocol)

    def test_strategy_protocol_implementation(self):
        """Тест реализации StrategyProtocol."""

        class MockStrategy:
            def analyze(self, data: pd.DataFrame) -> StrategyAnalysis:
                return StrategyAnalysis(
                    strategy_id="test",
                    timestamp=datetime.now(),
                    market_data=data,
                    indicators={},
                    signals=[],
                    metrics=StrategyMetrics(),
                    market_regime=MarketRegime.SIDEWAYS,
                    confidence=0.5,
                    risk_assessment={},
                    recommendations=[],
                )

            def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
                return Signal(direction=StrategyDirection.LONG, entry_price=100.0)

            def validate_data(self, data: pd.DataFrame) -> tuple[bool, Optional[str]]:
                return True, None

            def calculate_position_size(self, signal: Signal, account_balance: float) -> float:
                return 0.1

            def calculate_risk_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
                return {"volatility": 0.2}

            def update_metrics(self, signal: Signal, result: Dict[str, Any]) -> None:
                pass

            def get_metrics(self) -> Dict[str, Any]:
                return {"total_signals": 10}

            def save_state(self) -> None:
                pass

            def load_state(self) -> None:
                pass

        strategy = MockStrategy()
        assert isinstance(strategy, StrategyProtocol)

    def test_strategy_factory_protocol_implementation(self):
        """Тест реализации StrategyFactoryProtocol."""

        class MockStrategyFactory:
            def create_strategy(self, strategy_type: StrategyType, config: StrategyConfig) -> StrategyProtocol:
                return Mock()

            def get_available_strategies(self) -> List[StrategyType]:
                return [StrategyType.TREND, StrategyType.SCALPING]

            def validate_strategy_config(self, config: StrategyConfig) -> StrategyValidationResult:
                return {"is_valid": True, "errors": [], "warnings": []}

        factory = MockStrategyFactory()
        assert isinstance(factory, StrategyFactoryProtocol)


class TestEvolutionTypes:
    """Тесты для эволюционных типов."""

    def test_evolution_config(self):
        """Тест EvolutionConfig."""
        config = EvolutionConfig(
            learning_rate=1e-3,
            adaptation_rate=0.01,
            evolution_threshold=0.5,
            mutation_rate=0.1,
            crossover_rate=0.8,
            population_size=50,
            generations=100,
            elite_size=5,
        )

        assert config.learning_rate == 1e-3
        assert config.population_size == 50
        assert config.generations == 100

    def test_evolution_metrics(self):
        """Тест EvolutionMetrics."""
        metrics = EvolutionMetrics(generation=10, best_fitness=0.85, avg_fitness=0.75)

        assert metrics.generation == 10
        assert metrics.best_fitness == 0.85
        assert metrics.adaptation_success == 0.9


class TestAdaptationTypes:
    """Тесты для адаптивных типов."""

    def test_adaptation_config(self):
        """Тест AdaptationConfig."""
        config = AdaptationConfig(
            adaptation_threshold=0.7,
            learning_rate=0.01,
            memory_size=1000,
            adaptation_frequency=100,
            regime_detection_sensitivity=0.8,
        )

        assert config.adaptation_threshold == 0.7
        assert config.memory_size == 1000
        assert config.regime_detection_sensitivity == 0.8

    def test_market_context(self):
        """Тест MarketContext."""
        correlation_matrix = pd.DataFrame({"BTC": [1.0, 0.5], "ETH": [0.5, 1.0]}, index=["BTC", "ETH"])

        context = MarketContext(
            regime=MarketRegime.TRENDING_UP,
            volatility=0.25,
            trend_strength=0.8,
            volume_profile={"BTC": 0.6, "ETH": 0.4},
            liquidity_conditions={"BTC": 0.9, "ETH": 0.7},
            market_sentiment=0.75,
            correlation_matrix=correlation_matrix,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
        )

        assert context.regime == MarketRegime.TRENDING_UP
        assert context.volatility == 0.25
        assert context.trend_strength == 0.8
        assert context.market_sentiment == 0.75
        assert isinstance(context.correlation_matrix, pd.DataFrame)


class TestMLTypes:
    """Тесты для ML типов."""

    def test_ml_model_config(self):
        """Тест MLModelConfig."""
        config = MLModelConfig(
            model_type="random_forest",
            input_features=["price", "volume", "rsi"],
            output_features=["signal"],
            hyperparameters={"n_estimators": 100, "max_depth": 10},
            training_config={"test_size": 0.2, "random_state": 42},
            validation_config={"cv_folds": 5},
            model_path="/models/rf_model.pkl",
        )

        assert config.model_type == "random_forest"
        assert len(config.input_features) == 3
        assert config.hyperparameters["n_estimators"] == 100
        assert config.model_path == "/models/rf_model.pkl"

    def test_ml_prediction(self):
        """Тест MLPrediction."""
        prediction = MLPrediction(
            model_id="rf_model_001",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            input_features={"price": 50000.0, "volume": 1000.0, "rsi": 65.0},
            predictions={"signal": 0.8, "confidence": 0.85},
            confidence=0.85,
            uncertainty=0.15,
            model_version="1.0.0",
            metadata={"feature_importance": {"price": 0.4, "volume": 0.3, "rsi": 0.3}},
        )

        assert prediction.model_id == "rf_model_001"
        assert prediction.confidence == 0.85
        assert prediction.uncertainty == 0.15
        assert prediction.predictions["signal"] == 0.8


class TestRiskTypes:
    """Тесты для риск-менеджмент типов."""

    def test_risk_config(self):
        """Тест RiskConfig."""
        config = RiskConfig(
            stress_test_scenarios=[
                {"name": "market_crash", "price_shock": -0.3},
                {"name": "liquidity_crisis", "volume_reduction": 0.5},
            ],
            risk_budget={"BTC": 0.4, "ETH": 0.3, "cash": 0.3},
            max_drawdown=0.2,
            max_position_size=0.1,
            max_correlation=0.7,
            var_confidence=0.95,
            stop_loss_multiplier=2.0,
            take_profit_multiplier=3.0,
        )

        assert config.max_drawdown == 0.2
        assert config.max_position_size == 0.1
        assert len(config.stress_test_scenarios) == 2
        assert config.risk_budget["BTC"] == 0.4

    def test_risk_assessment(self):
        """Тест RiskAssessment."""
        assessment = RiskAssessment(
            portfolio_var=0.05,
            position_var=0.03,
            correlation_risk=0.2,
            liquidity_risk=0.15,
            concentration_risk=0.25,
            model_risk=0.1,
            total_risk_score=0.75,
            risk_decomposition={"market_risk": 0.4, "credit_risk": 0.2, "operational_risk": 0.15},
            stress_test_results={"market_crash": 0.12, "liquidity_crisis": 0.08},
            recommendations=["Уменьшить концентрацию в BTC", "Добавить хеджирование"],
        )

        assert assessment.portfolio_var == 0.05
        assert assessment.total_risk_score == 0.75
        assert len(assessment.recommendations) == 2
        assert assessment.stress_test_results["market_crash"] == 0.12


class TestIntegration:
    """Интеграционные тесты."""

    def test_strategy_workflow(self):
        """Тест полного workflow стратегии."""
        # Создание конфигурации
        config = StrategyConfig(
            strategy_type=StrategyType.TREND,
            name="Test Strategy",
            description="Test Description",
            parameters={"period": 20},
        )

        # Создание сигнала
        signal = Signal(direction=StrategyDirection.LONG, entry_price=100.0, confidence=0.8)

        # Создание метрик
        metrics = StrategyMetrics(total_signals=1, successful_signals=1, win_rate=1.0)

        # Создание анализа
        market_data = pd.DataFrame({"close": [100, 101, 102]})
        analysis = StrategyAnalysis(
            strategy_id="test",
            timestamp=datetime.now(),
            market_data=market_data,
            indicators={},
            signals=[signal],
            metrics=metrics,
            market_regime=MarketRegime.TRENDING_UP,
            confidence=0.8,
            risk_assessment={},
            recommendations=[],
        )

        assert analysis.strategy_id == "test"
        assert len(analysis.signals) == 1
        assert analysis.signals[0].direction == StrategyDirection.LONG
        assert analysis.metrics.win_rate == 1.0

    def test_evolution_workflow(self):
        """Тест workflow эволюции."""
        # Конфигурация эволюции
        evolution_config = EvolutionConfig(population_size=50, generations=100)

        # Метрики эволюции
        evolution_metrics = EvolutionMetrics(generation=10, best_fitness=0.85, avg_fitness=0.75)

        # Конфигурация адаптации
        adaptation_config = AdaptationConfig(adaptation_threshold=0.7, learning_rate=0.01)

        # Контекст рынка
        market_context = MarketContext(
            regime=MarketRegime.TRENDING_UP,
            volatility=0.25,
            trend_strength=0.8,
            volume_profile={},
            liquidity_conditions={},
            market_sentiment=0.75,
            correlation_matrix=pd.DataFrame(),
        )

        assert evolution_config.population_size == 50
        assert evolution_metrics.best_fitness == 0.85
        assert adaptation_config.adaptation_threshold == 0.7
        assert market_context.regime == MarketRegime.TRENDING_UP
