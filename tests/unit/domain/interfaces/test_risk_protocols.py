"""
Unit тесты для risk_protocols.

Покрывает:
- Валидацию результатов оценки рисков
- Тестирование протоколов риск-менеджмента
- Базовые классы анализаторов
- Обработку ошибок
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock

from domain.interfaces.risk_protocols import (
    RiskAssessmentResult,
    LiquidityGravityMetrics,
    RiskAnalyzerProtocol,
    LiquidityAnalyzerProtocol,
    StressTesterProtocol,
    PortfolioOptimizerProtocol,
    BaseRiskAnalyzer,
    BaseLiquidityAnalyzer,
    BaseStressTester,
    BasePortfolioOptimizer
)
from domain.types.risk_types import (
    PortfolioOptimizationMethod,
    PortfolioRisk,
    PositionRisk,
    RiskLevel,
    RiskMetrics,
    StressTestResult
)
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money


class TestRiskAssessmentResult:
    """Тесты для RiskAssessmentResult."""

    @pytest.fixture
    def valid_risk_data(self) -> Dict[str, Any]:
        """Валидные данные для оценки рисков."""
        return {
            "risk_level": RiskLevel.MEDIUM,
            "risk_score": 0.65,
            "portfolio_risk": PortfolioRisk(
                total_risk=0.65,
                systematic_risk=0.45,
                unsystematic_risk=0.20,
                var_95=Money(amount=Decimal("1000.00"), currency=Currency.USD),
                expected_shortfall=Money(amount=Decimal("1500.00"), currency=Currency.USD)
            ),
            "position_risks": [
                PositionRisk(
                    symbol="BTC/USD",
                    risk_score=0.7,
                    var_95=Money(amount=Decimal("500.00"), currency=Currency.USD),
                    beta=1.2,
                    correlation=0.8
                )
            ],
            "stress_test_results": [
                StressTestResult(
                    scenario_name="Market Crash",
                    portfolio_loss=Money(amount=Decimal("2000.00"), currency=Currency.USD),
                    risk_metrics=RiskMetrics(var=Decimal("1500.00"), es=Decimal("2000.00")),
                    passed=False
                )
            ],
            "recommendations": ["Reduce BTC exposure", "Increase diversification"],
            "timestamp": datetime.now(),
            "metadata": {"volatility": 0.025, "correlation": 0.75}
        }

    def test_valid_risk_assessment_creation(self, valid_risk_data):
        """Тест создания валидной оценки рисков."""
        result = RiskAssessmentResult(**valid_risk_data)
        
        assert result.risk_level == RiskLevel.MEDIUM
        assert result.risk_score == 0.65
        assert len(result.position_risks) == 1
        assert len(result.stress_test_results) == 1
        assert len(result.recommendations) == 2

    def test_invalid_risk_score_too_high(self, valid_risk_data):
        """Тест валидации слишком высокого риска."""
        valid_risk_data["risk_score"] = 1.5
        
        with pytest.raises(ValueError, match="Risk score must be between 0.0 and 1.0"):
            RiskAssessmentResult(**valid_risk_data)

    def test_invalid_risk_score_too_low(self, valid_risk_data):
        """Тест валидации слишком низкого риска."""
        valid_risk_data["risk_score"] = -0.1
        
        with pytest.raises(ValueError, match="Risk score must be between 0.0 and 1.0"):
            RiskAssessmentResult(**valid_risk_data)

    def test_future_timestamp_validation(self, valid_risk_data):
        """Тест валидации временной метки в будущем."""
        valid_risk_data["timestamp"] = datetime.now() + timedelta(days=1)
        
        with pytest.raises(ValueError, match="Timestamp cannot be in the future"):
            RiskAssessmentResult(**valid_risk_data)

    def test_boundary_values(self, valid_risk_data):
        """Тест граничных значений."""
        # Минимальный риск
        valid_risk_data["risk_score"] = 0.0
        result = RiskAssessmentResult(**valid_risk_data)
        assert result.risk_score == 0.0

        # Максимальный риск
        valid_risk_data["risk_score"] = 1.0
        result = RiskAssessmentResult(**valid_risk_data)
        assert result.risk_score == 1.0


class TestLiquidityGravityMetrics:
    """Тесты для LiquidityGravityMetrics."""

    @pytest.fixture
    def valid_liquidity_data(self) -> Dict[str, Any]:
        """Валидные данные для метрик ликвидности."""
        return {
            "liquidity_score": 0.75,
            "gravity_center": 50000.0,
            "pressure_zones": [
                {"price": 49900.0, "pressure": 0.8},
                {"price": 50100.0, "pressure": 0.6}
            ],
            "flow_direction": "bullish",
            "concentration_level": 0.65,
            "timestamp": datetime.now()
        }

    def test_valid_liquidity_metrics_creation(self, valid_liquidity_data):
        """Тест создания валидных метрик ликвидности."""
        metrics = LiquidityGravityMetrics(**valid_liquidity_data)
        
        assert metrics.liquidity_score == 0.75
        assert metrics.gravity_center == 50000.0
        assert len(metrics.pressure_zones) == 2
        assert metrics.flow_direction == "bullish"
        assert metrics.concentration_level == 0.65

    def test_invalid_liquidity_score_too_high(self, valid_liquidity_data):
        """Тест валидации слишком высокого показателя ликвидности."""
        valid_liquidity_data["liquidity_score"] = 1.5
        
        with pytest.raises(ValueError, match="Liquidity score must be between 0.0 and 1.0"):
            LiquidityGravityMetrics(**valid_liquidity_data)

    def test_invalid_liquidity_score_too_low(self, valid_liquidity_data):
        """Тест валидации слишком низкого показателя ликвидности."""
        valid_liquidity_data["liquidity_score"] = -0.1
        
        with pytest.raises(ValueError, match="Liquidity score must be between 0.0 and 1.0"):
            LiquidityGravityMetrics(**valid_liquidity_data)

    def test_invalid_concentration_level_too_high(self, valid_liquidity_data):
        """Тест валидации слишком высокого уровня концентрации."""
        valid_liquidity_data["concentration_level"] = 1.2
        
        with pytest.raises(ValueError, match="Concentration level must be between 0.0 and 1.0"):
            LiquidityGravityMetrics(**valid_liquidity_data)

    def test_invalid_concentration_level_too_low(self, valid_liquidity_data):
        """Тест валидации слишком низкого уровня концентрации."""
        valid_liquidity_data["concentration_level"] = -0.1
        
        with pytest.raises(ValueError, match="Concentration level must be between 0.0 and 1.0"):
            LiquidityGravityMetrics(**valid_liquidity_data)

    def test_liquidity_boundary_values(self, valid_liquidity_data):
        """Тест граничных значений для метрик ликвидности."""
        # Минимальные значения
        valid_liquidity_data["liquidity_score"] = 0.0
        valid_liquidity_data["concentration_level"] = 0.0
        metrics = LiquidityGravityMetrics(**valid_liquidity_data)
        assert metrics.liquidity_score == 0.0
        assert metrics.concentration_level == 0.0

        # Максимальные значения
        valid_liquidity_data["liquidity_score"] = 1.0
        valid_liquidity_data["concentration_level"] = 1.0
        metrics = LiquidityGravityMetrics(**valid_liquidity_data)
        assert metrics.liquidity_score == 1.0
        assert metrics.concentration_level == 1.0


class TestRiskAnalyzerProtocol:
    """Тесты для RiskAnalyzerProtocol."""

    @pytest.fixture
    def mock_risk_analyzer(self) -> RiskAnalyzerProtocol:
        """Мок реализации RiskAnalyzerProtocol."""
        analyzer = Mock(spec=RiskAnalyzerProtocol)
        analyzer.analyze_portfolio_risk = AsyncMock()
        analyzer.calculate_var = Mock()
        analyzer.calculate_expected_shortfall = Mock()
        return analyzer

    @pytest.fixture
    def sample_portfolio_data(self) -> Dict[str, Any]:
        """Тестовые данные портфеля."""
        return {
            "positions": [
                {"symbol": "BTC/USD", "value": 10000.0, "quantity": 0.2},
                {"symbol": "ETH/USD", "value": 5000.0, "quantity": 2.0}
            ],
            "total_value": 15000.0
        }

    @pytest.fixture
    def sample_market_data(self) -> Dict[str, Any]:
        """Тестовые рыночные данные."""
        return {
            "volatility": 0.025,
            "correlation_matrix": [[1.0, 0.7], [0.7, 1.0]],
            "risk_free_rate": 0.02
        }

    def test_protocol_definition(self):
        """Тест определения протокола."""
        assert hasattr(RiskAnalyzerProtocol, 'analyze_portfolio_risk')
        assert hasattr(RiskAnalyzerProtocol, 'calculate_var')
        assert hasattr(RiskAnalyzerProtocol, 'calculate_expected_shortfall')

    @pytest.mark.asyncio
    async def test_analyze_portfolio_risk(self, mock_risk_analyzer, sample_portfolio_data, sample_market_data):
        """Тест анализа рисков портфеля."""
        expected_result = RiskAssessmentResult(
            risk_level=RiskLevel.MEDIUM,
            risk_score=0.65,
            portfolio_risk=PortfolioRisk(
                total_risk=0.65,
                systematic_risk=0.45,
                unsystematic_risk=0.20,
                var_95=Money(amount=Decimal("1000.00"), currency=Currency.USD),
                expected_shortfall=Money(amount=Decimal("1500.00"), currency=Currency.USD)
            ),
            position_risks=[],
            stress_test_results=[],
            recommendations=["Reduce exposure"],
            timestamp=datetime.now(),
            metadata={}
        )
        mock_risk_analyzer.analyze_portfolio_risk.return_value = expected_result

        result = await mock_risk_analyzer.analyze_portfolio_risk(sample_portfolio_data, sample_market_data)

        assert result == expected_result
        mock_risk_analyzer.analyze_portfolio_risk.assert_called_once_with(sample_portfolio_data, sample_market_data)

    def test_calculate_var(self, mock_risk_analyzer):
        """Тест расчета Value at Risk."""
        positions = [
            {"symbol": "BTC/USD", "value": 10000.0},
            {"symbol": "ETH/USD", "value": 5000.0}
        ]
        confidence = 0.95
        expected_var = Money(amount=Decimal("750.00"), currency=Currency.USD)
        mock_risk_analyzer.calculate_var.return_value = expected_var

        result = mock_risk_analyzer.calculate_var(positions, confidence)

        assert result == expected_var
        mock_risk_analyzer.calculate_var.assert_called_once_with(positions, confidence)

    def test_calculate_expected_shortfall(self, mock_risk_analyzer):
        """Тест расчета Expected Shortfall."""
        positions = [
            {"symbol": "BTC/USD", "value": 10000.0},
            {"symbol": "ETH/USD", "value": 5000.0}
        ]
        confidence = 0.95
        expected_es = Money(amount=Decimal("1125.00"), currency=Currency.USD)
        mock_risk_analyzer.calculate_expected_shortfall.return_value = expected_es

        result = mock_risk_analyzer.calculate_expected_shortfall(positions, confidence)

        assert result == expected_es
        mock_risk_analyzer.calculate_expected_shortfall.assert_called_once_with(positions, confidence)


class TestBaseRiskAnalyzer:
    """Тесты для BaseRiskAnalyzer."""

    @pytest.fixture
    def base_risk_analyzer(self) -> BaseRiskAnalyzer:
        """Экземпляр базового анализатора рисков."""
        return BaseRiskAnalyzer({"test": "config"})

    @pytest.fixture
    def sample_positions(self) -> List[Dict[str, Any]]:
        """Тестовые позиции."""
        return [
            {"symbol": "BTC/USD", "value": 10000.0, "quantity": 0.2},
            {"symbol": "ETH/USD", "value": 5000.0, "quantity": 2.0}
        ]

    def test_initialization(self):
        """Тест инициализации."""
        analyzer = BaseRiskAnalyzer()
        assert analyzer.config == {}
        assert analyzer._risk_history == []
        assert analyzer._var_history == []

        analyzer_with_config = BaseRiskAnalyzer({"test": "value"})
        assert analyzer_with_config.config == {"test": "value"}

    def test_calculate_var(self, base_risk_analyzer, sample_positions):
        """Тест расчета Value at Risk."""
        confidence = 0.95
        result = base_risk_analyzer.calculate_var(sample_positions, confidence)
        
        assert isinstance(result, Money)
        assert result.currency == Currency.USD
        assert result.amount > 0

    def test_calculate_expected_shortfall(self, base_risk_analyzer, sample_positions):
        """Тест расчета Expected Shortfall."""
        confidence = 0.95
        result = base_risk_analyzer.calculate_expected_shortfall(sample_positions, confidence)
        
        assert isinstance(result, Money)
        assert result.currency == Currency.USD
        assert result.amount > 0

    def test_get_risk_history(self, base_risk_analyzer):
        """Тест получения истории рисков."""
        # Добавляем тестовые результаты
        result1 = RiskAssessmentResult(
            risk_level=RiskLevel.LOW,
            risk_score=0.3,
            portfolio_risk=PortfolioRisk(
                total_risk=0.3,
                systematic_risk=0.2,
                unsystematic_risk=0.1,
                var_95=Money(amount=Decimal("500.00"), currency=Currency.USD),
                expected_shortfall=Money(amount=Decimal("750.00"), currency=Currency.USD)
            ),
            position_risks=[],
            stress_test_results=[],
            recommendations=[],
            timestamp=datetime.now(),
            metadata={}
        )
        result2 = RiskAssessmentResult(
            risk_level=RiskLevel.HIGH,
            risk_score=0.8,
            portfolio_risk=PortfolioRisk(
                total_risk=0.8,
                systematic_risk=0.6,
                unsystematic_risk=0.2,
                var_95=Money(amount=Decimal("2000.00"), currency=Currency.USD),
                expected_shortfall=Money(amount=Decimal("3000.00"), currency=Currency.USD)
            ),
            position_risks=[],
            stress_test_results=[],
            recommendations=[],
            timestamp=datetime.now(),
            metadata={}
        )
        
        base_risk_analyzer._risk_history = [result1, result2]
        
        # Тест получения всей истории
        history = base_risk_analyzer.get_risk_history()
        assert len(history) == 2
        assert history[0] == result1
        assert history[1] == result2
        
        # Тест ограничения истории
        limited_history = base_risk_analyzer.get_risk_history(limit=1)
        assert len(limited_history) == 1
        assert limited_history[0] == result2  # Последний результат

    def test_get_var_history(self, base_risk_analyzer):
        """Тест получения истории VaR."""
        var_data1 = {"timestamp": datetime.now(), "var": 1000.0, "confidence": 0.95}
        var_data2 = {"timestamp": datetime.now(), "var": 1200.0, "confidence": 0.95}
        
        base_risk_analyzer._var_history = [var_data1, var_data2]
        
        # Тест получения всей истории
        history = base_risk_analyzer.get_var_history()
        assert len(history) == 2
        assert history[0] == var_data1
        assert history[1] == var_data2
        
        # Тест ограничения истории
        limited_history = base_risk_analyzer.get_var_history(limit=1)
        assert len(limited_history) == 1
        assert limited_history[0] == var_data2  # Последние данные

    @pytest.mark.asyncio
    async def test_analyze_portfolio_risk_abstract(self, base_risk_analyzer):
        """Тест что analyze_portfolio_risk является абстрактным методом."""
        with pytest.raises(TypeError):
            await base_risk_analyzer.analyze_portfolio_risk({}, {})


class TestBaseLiquidityAnalyzer:
    """Тесты для BaseLiquidityAnalyzer."""

    @pytest.fixture
    def base_liquidity_analyzer(self) -> BaseLiquidityAnalyzer:
        """Экземпляр базового анализатора ликвидности."""
        return BaseLiquidityAnalyzer({"test": "config"})

    @pytest.fixture
    def sample_orderbook_data(self) -> Dict[str, Any]:
        """Тестовые данные ордербука."""
        return {
            "bids": [[50000.0, 1.0], [49999.0, 2.0]],
            "asks": [[50001.0, 1.5], [50002.0, 2.5]]
        }

    def test_initialization(self):
        """Тест инициализации."""
        analyzer = BaseLiquidityAnalyzer()
        assert analyzer.config == {}
        assert analyzer._liquidity_history == []

        analyzer_with_config = BaseLiquidityAnalyzer({"test": "value"})
        assert analyzer_with_config.config == {"test": "value"}

    def test_detect_liquidity_clusters(self, base_liquidity_analyzer, sample_orderbook_data):
        """Тест обнаружения кластеров ликвидности."""
        result = base_liquidity_analyzer.detect_liquidity_clusters(sample_orderbook_data)
        
        assert isinstance(result, list)

    def test_calculate_liquidity_pressure(self, base_liquidity_analyzer, sample_orderbook_data):
        """Тест расчета давления ликвидности."""
        result = base_liquidity_analyzer.calculate_liquidity_pressure(sample_orderbook_data)
        
        assert isinstance(result, float)
        assert result == 0.0  # Базовая реализация возвращает 0.0

    def test_get_liquidity_history(self, base_liquidity_analyzer):
        """Тест получения истории ликвидности."""
        metrics1 = LiquidityGravityMetrics(
            liquidity_score=0.6,
            gravity_center=50000.0,
            pressure_zones=[],
            flow_direction="neutral",
            concentration_level=0.5,
            timestamp=datetime.now()
        )
        metrics2 = LiquidityGravityMetrics(
            liquidity_score=0.8,
            gravity_center=50100.0,
            pressure_zones=[],
            flow_direction="bullish",
            concentration_level=0.7,
            timestamp=datetime.now()
        )
        
        base_liquidity_analyzer._liquidity_history = [metrics1, metrics2]
        
        # Тест получения всей истории
        history = base_liquidity_analyzer.get_liquidity_history()
        assert len(history) == 2
        assert history[0] == metrics1
        assert history[1] == metrics2
        
        # Тест ограничения истории
        limited_history = base_liquidity_analyzer.get_liquidity_history(limit=1)
        assert len(limited_history) == 1
        assert limited_history[0] == metrics2  # Последние метрики

    @pytest.mark.asyncio
    async def test_analyze_liquidity_gravity_abstract(self, base_liquidity_analyzer):
        """Тест что analyze_liquidity_gravity является абстрактным методом."""
        with pytest.raises(TypeError):
            await base_liquidity_analyzer.analyze_liquidity_gravity({}, {})


class TestBaseStressTester:
    """Тесты для BaseStressTester."""

    @pytest.fixture
    def base_stress_tester(self) -> BaseStressTester:
        """Экземпляр базового стресс-тестера."""
        return BaseStressTester({"test": "config"})

    def test_initialization(self):
        """Тест инициализации."""
        tester = BaseStressTester()
        assert tester.config == {}
        assert tester._stress_history == []

        tester_with_config = BaseStressTester({"test": "value"})
        assert tester_with_config.config == {"test": "value"}

    def test_generate_stress_scenarios(self, base_stress_tester):
        """Тест генерации сценариев стресс-тестирования."""
        market_conditions = {
            "volatility": 0.025,
            "correlation": 0.75,
            "liquidity": "high"
        }
        result = base_stress_tester.generate_stress_scenarios(market_conditions)
        
        assert isinstance(result, list)

    def test_get_stress_history(self, base_stress_tester):
        """Тест получения истории стресс-тестов."""
        result1 = StressTestResult(
            scenario_name="Market Crash",
            portfolio_loss=Money(amount=Decimal("2000.00"), currency=Currency.USD),
            risk_metrics=RiskMetrics(var=Decimal("1500.00"), es=Decimal("2000.00")),
            passed=False
        )
        result2 = StressTestResult(
            scenario_name="Liquidity Crisis",
            portfolio_loss=Money(amount=Decimal("1000.00"), currency=Currency.USD),
            risk_metrics=RiskMetrics(var=Decimal("800.00"), es=Decimal("1200.00")),
            passed=True
        )
        
        base_stress_tester._stress_history = [result1, result2]
        
        # Тест получения всей истории
        history = base_stress_tester.get_stress_history()
        assert len(history) == 2
        assert history[0] == result1
        assert history[1] == result2
        
        # Тест ограничения истории
        limited_history = base_stress_tester.get_stress_history(limit=1)
        assert len(limited_history) == 1
        assert limited_history[0] == result2  # Последний результат

    @pytest.mark.asyncio
    async def test_run_stress_tests_abstract(self, base_stress_tester):
        """Тест что run_stress_tests является абстрактным методом."""
        with pytest.raises(TypeError):
            await base_stress_tester.run_stress_tests({}, [])


class TestBasePortfolioOptimizer:
    """Тесты для BasePortfolioOptimizer."""

    @pytest.fixture
    def base_portfolio_optimizer(self) -> BasePortfolioOptimizer:
        """Экземпляр базового оптимизатора портфеля."""
        return BasePortfolioOptimizer({"test": "config"})

    def test_initialization(self):
        """Тест инициализации."""
        optimizer = BasePortfolioOptimizer()
        assert optimizer.config == {}
        assert optimizer._optimization_history == []

        optimizer_with_config = BasePortfolioOptimizer({"test": "value"})
        assert optimizer_with_config.config == {"test": "value"}

    def test_calculate_optimal_weights(self, base_portfolio_optimizer):
        """Тест расчета оптимальных весов."""
        assets = ["BTC/USD", "ETH/USD", "ADA/USD"]
        returns = [0.20, 0.15, 0.10]
        risks = [0.30, 0.25, 0.20]
        
        result = base_portfolio_optimizer.calculate_optimal_weights(assets, returns, risks)
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(w, float) for w in result)
        assert abs(sum(result) - 1.0) < 1e-6  # Сумма весов должна быть равна 1

    def test_calculate_optimal_weights_empty_assets(self, base_portfolio_optimizer):
        """Тест расчета оптимальных весов для пустого списка активов."""
        result = base_portfolio_optimizer.calculate_optimal_weights([], [], [])
        
        assert isinstance(result, list)
        assert len(result) == 0

    def test_get_optimization_history(self, base_portfolio_optimizer):
        """Тест получения истории оптимизаций."""
        optimization1 = {
            "timestamp": datetime.now(),
            "method": PortfolioOptimizationMethod.MARKOWITZ,
            "expected_return": 0.18,
            "expected_risk": 0.22
        }
        optimization2 = {
            "timestamp": datetime.now(),
            "method": PortfolioOptimizationMethod.BLACK_LITTERMAN,
            "expected_return": 0.20,
            "expected_risk": 0.25
        }
        
        base_portfolio_optimizer._optimization_history = [optimization1, optimization2]
        
        # Тест получения всей истории
        history = base_portfolio_optimizer.get_optimization_history()
        assert len(history) == 2
        assert history[0] == optimization1
        assert history[1] == optimization2
        
        # Тест ограничения истории
        limited_history = base_portfolio_optimizer.get_optimization_history(limit=1)
        assert len(limited_history) == 1
        assert limited_history[0] == optimization2  # Последняя оптимизация

    @pytest.mark.asyncio
    async def test_optimize_portfolio_abstract(self, base_portfolio_optimizer):
        """Тест что optimize_portfolio является абстрактным методом."""
        with pytest.raises(TypeError):
            await base_portfolio_optimizer.optimize_portfolio({}, {}, PortfolioOptimizationMethod.MARKOWITZ) 