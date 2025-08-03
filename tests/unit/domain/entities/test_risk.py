"""
Unit тесты для Risk.

Покрывает:
- Основной функционал
- Валидацию данных
- Бизнес-логику
- Обработку ошибок
- Сериализацию/десериализацию
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, patch
from uuid import uuid4

from domain.entities.risk import (
    RiskProfile, RiskMetrics, RiskManager,
    RiskLevel, RiskType, RiskMetricsProtocol
)
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money


class TestRiskProfile:
    """Тесты для RiskProfile."""
    
    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "id": uuid4(),
            "name": "Conservative Profile",
            "risk_level": RiskLevel.LOW,
            "max_risk_per_trade": Decimal("0.01"),  # 1%
            "max_daily_loss": Decimal("0.03"),  # 3%
            "max_weekly_loss": Decimal("0.10"),  # 10%
            "max_portfolio_risk": Decimal("0.05"),  # 5%
            "max_correlation": Decimal("0.5"),  # 50%
            "max_leverage": Decimal("2.0"),  # 2x
            "min_risk_reward_ratio": Decimal("2.0"),  # 2:1
            "max_drawdown": Decimal("0.15"),  # 15%
            "position_sizing_method": "kelly",
            "stop_loss_method": "atr",
            "metadata": {"strategy": "conservative"}
        }
    
    @pytest.fixture
    def risk_profile(self, sample_data) -> RiskProfile:
        """Создает тестовый профиль риска."""
        return RiskProfile(**sample_data)
    
    def test_creation(self, sample_data):
        """Тест создания профиля риска."""
        profile = RiskProfile(**sample_data)
        
        assert profile.id == sample_data["id"]
        assert profile.name == sample_data["name"]
        assert profile.risk_level == sample_data["risk_level"]
        assert profile.max_risk_per_trade == sample_data["max_risk_per_trade"]
        assert profile.max_daily_loss == sample_data["max_daily_loss"]
        assert profile.max_weekly_loss == sample_data["max_weekly_loss"]
        assert profile.max_portfolio_risk == sample_data["max_portfolio_risk"]
        assert profile.max_correlation == sample_data["max_correlation"]
        assert profile.max_leverage == sample_data["max_leverage"]
        assert profile.min_risk_reward_ratio == sample_data["min_risk_reward_ratio"]
        assert profile.max_drawdown == sample_data["max_drawdown"]
        assert profile.position_sizing_method == sample_data["position_sizing_method"]
        assert profile.stop_loss_method == sample_data["stop_loss_method"]
        assert profile.metadata == sample_data["metadata"]
    
    def test_default_creation(self):
        """Тест создания профиля риска с дефолтными значениями."""
        profile = RiskProfile()
        
        assert isinstance(profile.id, uuid4().__class__)
        assert profile.name == ""
        assert profile.risk_level == RiskLevel.MEDIUM
        assert profile.max_risk_per_trade == Decimal("0.02")
        assert profile.max_daily_loss == Decimal("0.05")
        assert profile.max_weekly_loss == Decimal("0.15")
        assert profile.max_portfolio_risk == Decimal("0.10")
        assert profile.max_correlation == Decimal("0.7")
        assert profile.max_leverage == Decimal("3.0")
        assert profile.min_risk_reward_ratio == Decimal("1.5")
        assert profile.max_drawdown == Decimal("0.20")
        assert profile.position_sizing_method == "kelly"
        assert profile.stop_loss_method == "atr"
        assert profile.metadata == {}
    
    def test_is_risk_level_acceptable(self, risk_profile):
        """Тест проверки приемлемости уровня риска."""
        # Приемлемый риск
        assert risk_profile.is_risk_level_acceptable(Decimal("0.005")) is True
        assert risk_profile.is_risk_level_acceptable(Decimal("0.01")) is True
        
        # Неприемлемый риск
        assert risk_profile.is_risk_level_acceptable(Decimal("0.015")) is False
        assert risk_profile.is_risk_level_acceptable(Decimal("0.02")) is False
    
    def test_is_daily_loss_acceptable(self, risk_profile):
        """Тест проверки приемлемости дневного убытка."""
        # Приемлемый убыток
        assert risk_profile.is_daily_loss_acceptable(Decimal("0.02")) is True
        assert risk_profile.is_daily_loss_acceptable(Decimal("0.03")) is True
        
        # Неприемлемый убыток
        assert risk_profile.is_daily_loss_acceptable(Decimal("0.04")) is False
        assert risk_profile.is_daily_loss_acceptable(Decimal("0.05")) is False
    
    def test_is_weekly_loss_acceptable(self, risk_profile):
        """Тест проверки приемлемости недельного убытка."""
        # Приемлемый убыток
        assert risk_profile.is_weekly_loss_acceptable(Decimal("0.08")) is True
        assert risk_profile.is_weekly_loss_acceptable(Decimal("0.10")) is True
        
        # Неприемлемый убыток
        assert risk_profile.is_weekly_loss_acceptable(Decimal("0.12")) is False
        assert risk_profile.is_weekly_loss_acceptable(Decimal("0.15")) is False
    
    def test_is_portfolio_risk_acceptable(self, risk_profile):
        """Тест проверки приемлемости риска портфеля."""
        # Приемлемый риск
        assert risk_profile.is_portfolio_risk_acceptable(Decimal("0.03")) is True
        assert risk_profile.is_portfolio_risk_acceptable(Decimal("0.05")) is True
        
        # Неприемлемый риск
        assert risk_profile.is_portfolio_risk_acceptable(Decimal("0.06")) is False
        assert risk_profile.is_portfolio_risk_acceptable(Decimal("0.08")) is False
    
    def test_is_correlation_acceptable(self, risk_profile):
        """Тест проверки приемлемости корреляции."""
        # Приемлемая корреляция
        assert risk_profile.is_correlation_acceptable(Decimal("0.3")) is True
        assert risk_profile.is_correlation_acceptable(Decimal("0.5")) is True
        assert risk_profile.is_correlation_acceptable(Decimal("-0.3")) is True
        assert risk_profile.is_correlation_acceptable(Decimal("-0.5")) is True
        
        # Неприемлемая корреляция
        assert risk_profile.is_correlation_acceptable(Decimal("0.6")) is False
        assert risk_profile.is_correlation_acceptable(Decimal("-0.6")) is False
    
    def test_is_leverage_acceptable(self, risk_profile):
        """Тест проверки приемлемости кредитного плеча."""
        # Приемлемое плечо
        assert risk_profile.is_leverage_acceptable(Decimal("1.5")) is True
        assert risk_profile.is_leverage_acceptable(Decimal("2.0")) is True
        
        # Неприемлемое плечо
        assert risk_profile.is_leverage_acceptable(Decimal("2.5")) is False
        assert risk_profile.is_leverage_acceptable(Decimal("3.0")) is False
    
    def test_is_risk_reward_acceptable(self, risk_profile):
        """Тест проверки приемлемости соотношения риск/доходность."""
        # Приемлемое соотношение
        assert risk_profile.is_risk_reward_acceptable(Decimal("2.5")) is True
        assert risk_profile.is_risk_reward_acceptable(Decimal("3.0")) is True
        
        # Неприемлемое соотношение
        assert risk_profile.is_risk_reward_acceptable(Decimal("1.0")) is False
        assert risk_profile.is_risk_reward_acceptable(Decimal("1.5")) is False
    
    def test_is_drawdown_acceptable(self, risk_profile):
        """Тест проверки приемлемости просадки."""
        # Приемлемая просадка
        assert risk_profile.is_drawdown_acceptable(Decimal("0.10")) is True
        assert risk_profile.is_drawdown_acceptable(Decimal("0.15")) is True
        
        # Неприемлемая просадка
        assert risk_profile.is_drawdown_acceptable(Decimal("0.18")) is False
        assert risk_profile.is_drawdown_acceptable(Decimal("0.20")) is False
    
    def test_to_dict(self, risk_profile):
        """Тест сериализации в словарь."""
        data = risk_profile.to_dict()
        
        assert data["id"] == str(risk_profile.id)
        assert data["name"] == risk_profile.name
        assert data["risk_level"] == risk_profile.risk_level.value
        assert data["max_risk_per_trade"] == str(risk_profile.max_risk_per_trade)
        assert data["max_daily_loss"] == str(risk_profile.max_daily_loss)
        assert data["max_weekly_loss"] == str(risk_profile.max_weekly_loss)
        assert data["max_portfolio_risk"] == str(risk_profile.max_portfolio_risk)
        assert data["max_correlation"] == str(risk_profile.max_correlation)
        assert data["max_leverage"] == str(risk_profile.max_leverage)
        assert data["min_risk_reward_ratio"] == str(risk_profile.min_risk_reward_ratio)
        assert data["max_drawdown"] == str(risk_profile.max_drawdown)
        assert data["position_sizing_method"] == risk_profile.position_sizing_method
        assert data["stop_loss_method"] == risk_profile.stop_loss_method
        assert "created_at" in data
        assert "updated_at" in data
        assert data["metadata"] == str(risk_profile.metadata)
    
    def test_from_dict(self, risk_profile):
        """Тест десериализации из словаря."""
        data = risk_profile.to_dict()
        new_profile = RiskProfile.from_dict(data)
        
        assert new_profile.id == risk_profile.id
        assert new_profile.name == risk_profile.name
        assert new_profile.risk_level == risk_profile.risk_level
        assert new_profile.max_risk_per_trade == risk_profile.max_risk_per_trade
        assert new_profile.max_daily_loss == risk_profile.max_daily_loss
        assert new_profile.max_weekly_loss == risk_profile.max_weekly_loss
        assert new_profile.max_portfolio_risk == risk_profile.max_portfolio_risk
        assert new_profile.max_correlation == risk_profile.max_correlation
        assert new_profile.max_leverage == risk_profile.max_leverage
        assert new_profile.min_risk_reward_ratio == risk_profile.min_risk_reward_ratio
        assert new_profile.max_drawdown == risk_profile.max_drawdown
        assert new_profile.position_sizing_method == risk_profile.position_sizing_method
        assert new_profile.stop_loss_method == risk_profile.stop_loss_method
        assert new_profile.metadata == risk_profile.metadata
    
    def test_risk_level_enum(self):
        """Тест enum RiskLevel."""
        assert RiskLevel.VERY_LOW.value == "very_low"
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.VERY_HIGH.value == "very_high"
        assert RiskLevel.EXTREME.value == "extreme"
    
    def test_risk_type_enum(self):
        """Тест enum RiskType."""
        assert RiskType.MARKET_RISK.value == "market_risk"
        assert RiskType.LIQUIDITY_RISK.value == "liquidity_risk"
        assert RiskType.CREDIT_RISK.value == "credit_risk"
        assert RiskType.OPERATIONAL_RISK.value == "operational_risk"
        assert RiskType.SYSTEMATIC_RISK.value == "systematic_risk"
        assert RiskType.SPECIFIC_RISK.value == "specific_risk"


class TestRiskMetrics:
    """Тесты для RiskMetrics."""
    
    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "id": uuid4(),
            "timestamp": datetime.now(),
            "portfolio_value": Money(Decimal("100000"), Currency.USD),
            "portfolio_risk": Decimal("0.05"),
            "portfolio_beta": Decimal("0.8"),
            "portfolio_volatility": Decimal("0.15"),
            "portfolio_var_95": Money(Decimal("5000"), Currency.USD),
            "portfolio_cvar_95": Money(Decimal("6000"), Currency.USD),
            "current_drawdown": Decimal("0.08"),
            "max_drawdown": Decimal("0.12"),
            "drawdown_duration": 5,
            "total_return": Decimal("0.15"),
            "sharpe_ratio": Decimal("1.2"),
            "sortino_ratio": Decimal("1.5"),
            "calmar_ratio": Decimal("1.25"),
            "total_positions": 10,
            "long_positions": 6,
            "short_positions": 4,
            "total_exposure": Money(Decimal("80000"), Currency.USD),
            "net_exposure": Money(Decimal("20000"), Currency.USD),
            "avg_correlation": Decimal("0.3"),
            "max_correlation": Decimal("0.6"),
            "liquidity_ratio": Decimal("0.8"),
            "bid_ask_spread": Decimal("0.001"),
            "concentration_ratio": Decimal("0.4"),
            "herfindahl_index": Decimal("0.25"),
            "stress_test_score": Decimal("0.7"),
            "scenario_analysis_score": Decimal("0.8"),
            "metadata": {"source": "risk_engine"}
        }
    
    @pytest.fixture
    def risk_metrics(self, sample_data) -> RiskMetrics:
        """Создает тестовые метрики риска."""
        return RiskMetrics(**sample_data)
    
    @pytest.fixture
    def risk_profile(self) -> RiskProfile:
        """Создает тестовый профиль риска."""
        return RiskProfile(
            max_risk_per_trade=Decimal("0.02"),
            max_daily_loss=Decimal("0.05"),
            max_weekly_loss=Decimal("0.15"),
            max_portfolio_risk=Decimal("0.10"),
            max_correlation=Decimal("0.7"),
            max_leverage=Decimal("3.0"),
            min_risk_reward_ratio=Decimal("1.5"),
            max_drawdown=Decimal("0.20")
        )
    
    def test_creation(self, sample_data):
        """Тест создания метрик риска."""
        metrics = RiskMetrics(**sample_data)
        
        assert metrics.id == sample_data["id"]
        assert metrics.timestamp == sample_data["timestamp"]
        assert metrics.portfolio_value == sample_data["portfolio_value"]
        assert metrics.portfolio_risk == sample_data["portfolio_risk"]
        assert metrics.portfolio_beta == sample_data["portfolio_beta"]
        assert metrics.portfolio_volatility == sample_data["portfolio_volatility"]
        assert metrics.portfolio_var_95 == sample_data["portfolio_var_95"]
        assert metrics.portfolio_cvar_95 == sample_data["portfolio_cvar_95"]
        assert metrics.current_drawdown == sample_data["current_drawdown"]
        assert metrics.max_drawdown == sample_data["max_drawdown"]
        assert metrics.drawdown_duration == sample_data["drawdown_duration"]
        assert metrics.total_return == sample_data["total_return"]
        assert metrics.sharpe_ratio == sample_data["sharpe_ratio"]
        assert metrics.sortino_ratio == sample_data["sortino_ratio"]
        assert metrics.calmar_ratio == sample_data["calmar_ratio"]
        assert metrics.total_positions == sample_data["total_positions"]
        assert metrics.long_positions == sample_data["long_positions"]
        assert metrics.short_positions == sample_data["short_positions"]
        assert metrics.total_exposure == sample_data["total_exposure"]
        assert metrics.net_exposure == sample_data["net_exposure"]
        assert metrics.avg_correlation == sample_data["avg_correlation"]
        assert metrics.max_correlation == sample_data["max_correlation"]
        assert metrics.liquidity_ratio == sample_data["liquidity_ratio"]
        assert metrics.bid_ask_spread == sample_data["bid_ask_spread"]
        assert metrics.concentration_ratio == sample_data["concentration_ratio"]
        assert metrics.herfindahl_index == sample_data["herfindahl_index"]
        assert metrics.stress_test_score == sample_data["stress_test_score"]
        assert metrics.scenario_analysis_score == sample_data["scenario_analysis_score"]
        assert metrics.metadata == sample_data["metadata"]
    
    def test_default_creation(self):
        """Тест создания метрик риска с дефолтными значениями."""
        metrics = RiskMetrics()
        
        assert isinstance(metrics.id, uuid4().__class__)
        assert isinstance(metrics.timestamp, datetime)
        assert metrics.portfolio_value.amount == Decimal("0")
        assert metrics.portfolio_risk == Decimal("0")
        assert metrics.portfolio_beta == Decimal("0")
        assert metrics.portfolio_volatility == Decimal("0")
        assert metrics.portfolio_var_95.amount == Decimal("0")
        assert metrics.portfolio_cvar_95.amount == Decimal("0")
        assert metrics.current_drawdown == Decimal("0")
        assert metrics.max_drawdown == Decimal("0")
        assert metrics.drawdown_duration == 0
        assert metrics.total_return == Decimal("0")
        assert metrics.sharpe_ratio == Decimal("0")
        assert metrics.sortino_ratio == Decimal("0")
        assert metrics.calmar_ratio == Decimal("0")
        assert metrics.total_positions == 0
        assert metrics.long_positions == 0
        assert metrics.short_positions == 0
        assert metrics.total_exposure.amount == Decimal("0")
        assert metrics.net_exposure.amount == Decimal("0")
        assert metrics.avg_correlation == Decimal("0")
        assert metrics.max_correlation == Decimal("0")
        assert metrics.liquidity_ratio == Decimal("0")
        assert metrics.bid_ask_spread == Decimal("0")
        assert metrics.concentration_ratio == Decimal("0")
        assert metrics.herfindahl_index == Decimal("0")
        assert metrics.stress_test_score == Decimal("0")
        assert metrics.scenario_analysis_score == Decimal("0")
        assert metrics.metadata == {}
    
    def test_is_portfolio_healthy(self, risk_metrics, risk_profile):
        """Тест проверки здоровья портфеля."""
        # Здоровый портфель
        assert risk_metrics.is_portfolio_healthy(risk_profile) is True
        
        # Нездоровый портфель - высокий риск
        risk_metrics.portfolio_risk = Decimal("0.15")
        assert risk_metrics.is_portfolio_healthy(risk_profile) is False
        
        # Нездоровый портфель - высокая просадка
        risk_metrics.portfolio_risk = Decimal("0.05")
        risk_metrics.current_drawdown = Decimal("0.25")
        assert risk_metrics.is_portfolio_healthy(risk_profile) is False
    
    def test_get_risk_score(self, risk_metrics):
        """Тест расчета оценки риска."""
        risk_score = risk_metrics.get_risk_score()
        
        # Проверяем, что оценка риска в разумных пределах
        assert isinstance(risk_score, Decimal)
        assert Decimal("0") <= risk_score <= Decimal("1")
        
        # Тест с высокими рисками
        risk_metrics.portfolio_risk = Decimal("0.3")
        risk_metrics.current_drawdown = Decimal("0.4")
        risk_metrics.portfolio_volatility = Decimal("0.5")
        
        high_risk_score = risk_metrics.get_risk_score()
        assert high_risk_score > risk_score
    
    def test_get_risk_level(self, risk_metrics):
        """Тест определения уровня риска."""
        # Низкий риск
        risk_metrics.portfolio_risk = Decimal("0.02")
        risk_metrics.current_drawdown = Decimal("0.05")
        risk_level = risk_metrics.get_risk_level()
        assert risk_level in [RiskLevel.VERY_LOW, RiskLevel.LOW]
        
        # Средний риск
        risk_metrics.portfolio_risk = Decimal("0.08")
        risk_metrics.current_drawdown = Decimal("0.12")
        risk_level = risk_metrics.get_risk_level()
        assert risk_level == RiskLevel.MEDIUM
        
        # Высокий риск
        risk_metrics.portfolio_risk = Decimal("0.15")
        risk_metrics.current_drawdown = Decimal("0.25")
        risk_level = risk_metrics.get_risk_level()
        assert risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.EXTREME]
    
    def test_to_dict(self, risk_metrics):
        """Тест сериализации в словарь."""
        data = risk_metrics.to_dict()
        
        assert data["id"] == str(risk_metrics.id)
        assert "timestamp" in data
        assert data["portfolio_value"] == str(risk_metrics.portfolio_value.amount)
        assert data["portfolio_risk"] == str(risk_metrics.portfolio_risk)
        assert data["portfolio_beta"] == str(risk_metrics.portfolio_beta)
        assert data["portfolio_volatility"] == str(risk_metrics.portfolio_volatility)
        assert data["portfolio_var_95"] == str(risk_metrics.portfolio_var_95.amount)
        assert data["portfolio_cvar_95"] == str(risk_metrics.portfolio_cvar_95.amount)
        assert data["current_drawdown"] == str(risk_metrics.current_drawdown)
        assert data["max_drawdown"] == str(risk_metrics.max_drawdown)
        assert data["drawdown_duration"] == str(risk_metrics.drawdown_duration)
        assert data["total_return"] == str(risk_metrics.total_return)
        assert data["sharpe_ratio"] == str(risk_metrics.sharpe_ratio)
        assert data["sortino_ratio"] == str(risk_metrics.sortino_ratio)
        assert data["calmar_ratio"] == str(risk_metrics.calmar_ratio)
        assert data["total_positions"] == str(risk_metrics.total_positions)
        assert data["long_positions"] == str(risk_metrics.long_positions)
        assert data["short_positions"] == str(risk_metrics.short_positions)
        assert data["total_exposure"] == str(risk_metrics.total_exposure.amount)
        assert data["net_exposure"] == str(risk_metrics.net_exposure.amount)
        assert data["avg_correlation"] == str(risk_metrics.avg_correlation)
        assert data["max_correlation"] == str(risk_metrics.max_correlation)
        assert data["liquidity_ratio"] == str(risk_metrics.liquidity_ratio)
        assert data["bid_ask_spread"] == str(risk_metrics.bid_ask_spread)
        assert data["concentration_ratio"] == str(risk_metrics.concentration_ratio)
        assert data["herfindahl_index"] == str(risk_metrics.herfindahl_index)
        assert data["stress_test_score"] == str(risk_metrics.stress_test_score)
        assert data["scenario_analysis_score"] == str(risk_metrics.scenario_analysis_score)
        assert data["metadata"] == str(risk_metrics.metadata)
    
    def test_from_dict(self, risk_metrics):
        """Тест десериализации из словаря."""
        data = risk_metrics.to_dict()
        new_metrics = RiskMetrics.from_dict(data)
        
        assert new_metrics.id == risk_metrics.id
        assert new_metrics.portfolio_value.amount == risk_metrics.portfolio_value.amount
        assert new_metrics.portfolio_risk == risk_metrics.portfolio_risk
        assert new_metrics.portfolio_beta == risk_metrics.portfolio_beta
        assert new_metrics.portfolio_volatility == risk_metrics.portfolio_volatility
        assert new_metrics.portfolio_var_95.amount == risk_metrics.portfolio_var_95.amount
        assert new_metrics.portfolio_cvar_95.amount == risk_metrics.portfolio_cvar_95.amount
        assert new_metrics.current_drawdown == risk_metrics.current_drawdown
        assert new_metrics.max_drawdown == risk_metrics.max_drawdown
        assert new_metrics.drawdown_duration == risk_metrics.drawdown_duration
        assert new_metrics.total_return == risk_metrics.total_return
        assert new_metrics.sharpe_ratio == risk_metrics.sharpe_ratio
        assert new_metrics.sortino_ratio == risk_metrics.sortino_ratio
        assert new_metrics.calmar_ratio == risk_metrics.calmar_ratio
        assert new_metrics.total_positions == risk_metrics.total_positions
        assert new_metrics.long_positions == risk_metrics.long_positions
        assert new_metrics.short_positions == risk_metrics.short_positions
        assert new_metrics.total_exposure.amount == risk_metrics.total_exposure.amount
        assert new_metrics.net_exposure.amount == risk_metrics.net_exposure.amount
        assert new_metrics.avg_correlation == risk_metrics.avg_correlation
        assert new_metrics.max_correlation == risk_metrics.max_correlation
        assert new_metrics.liquidity_ratio == risk_metrics.liquidity_ratio
        assert new_metrics.bid_ask_spread == risk_metrics.bid_ask_spread
        assert new_metrics.concentration_ratio == risk_metrics.concentration_ratio
        assert new_metrics.herfindahl_index == risk_metrics.herfindahl_index
        assert new_metrics.stress_test_score == risk_metrics.stress_test_score
        assert new_metrics.scenario_analysis_score == risk_metrics.scenario_analysis_score
        assert new_metrics.metadata == risk_metrics.metadata
    
    def test_risk_metrics_protocol_compliance(self, risk_metrics):
        """Тест соответствия протоколу RiskMetricsProtocol."""
        assert isinstance(risk_metrics, RiskMetricsProtocol)
        
        risk_score = risk_metrics.get_risk_score()
        assert isinstance(risk_score, Decimal)
        
        risk_profile = RiskProfile()
        is_healthy = risk_metrics.is_portfolio_healthy(risk_profile)
        assert isinstance(is_healthy, bool)


class TestRiskManager:
    """Тесты для RiskManager."""
    
    @pytest.fixture
    def risk_profile(self) -> RiskProfile:
        """Создает тестовый профиль риска."""
        return RiskProfile(
            name="Test Profile",
            risk_level=RiskLevel.MEDIUM,
            max_risk_per_trade=Decimal("0.02"),
            max_daily_loss=Decimal("0.05"),
            max_weekly_loss=Decimal("0.15"),
            max_portfolio_risk=Decimal("0.10"),
            max_correlation=Decimal("0.7"),
            max_leverage=Decimal("3.0"),
            min_risk_reward_ratio=Decimal("1.5"),
            max_drawdown=Decimal("0.20")
        )
    
    @pytest.fixture
    def risk_manager(self, risk_profile) -> RiskManager:
        """Создает тестовый менеджер рисков."""
        return RiskManager(risk_profile, "Test Risk Manager")
    
    @pytest.fixture
    def risk_metrics(self) -> RiskMetrics:
        """Создает тестовые метрики риска."""
        return RiskMetrics(
            portfolio_value=Money(Decimal("100000"), Currency.USD),
            portfolio_risk=Decimal("0.05"),
            current_drawdown=Decimal("0.08"),
            max_drawdown=Decimal("0.12"),
            total_return=Decimal("0.15"),
            sharpe_ratio=Decimal("1.2"),
            total_positions=10,
            long_positions=6,
            short_positions=4,
            total_exposure=Money(Decimal("80000"), Currency.USD),
            net_exposure=Money(Decimal("20000"), Currency.USD),
            avg_correlation=Decimal("0.3"),
            max_correlation=Decimal("0.6"),
            liquidity_ratio=Decimal("0.8"),
            bid_ask_spread=Decimal("0.001"),
            concentration_ratio=Decimal("0.4"),
            herfindahl_index=Decimal("0.25"),
            stress_test_score=Decimal("0.7"),
            scenario_analysis_score=Decimal("0.8")
        )
    
    def test_creation(self, risk_profile):
        """Тест создания менеджера рисков."""
        manager = RiskManager(risk_profile, "Test Manager")
        
        assert manager.risk_profile == risk_profile
        assert manager.name == "Test Manager"
        assert manager.current_metrics is None
        assert manager.risk_alerts == []
        assert manager.is_active is True
    
    def test_update_metrics(self, risk_manager, risk_metrics):
        """Тест обновления метрик."""
        risk_manager.update_metrics(risk_metrics)
        
        assert risk_manager.current_metrics == risk_metrics
        assert risk_manager.last_update is not None
    
    def test_validate_trade_acceptable(self, risk_manager, risk_metrics):
        """Тест валидации приемлемой сделки."""
        risk_manager.update_metrics(risk_metrics)
        
        trade_value = Money(Decimal("1000"), Currency.USD)
        portfolio_value = Money(Decimal("100000"), Currency.USD)
        
        is_valid = risk_manager.validate_trade(trade_value, portfolio_value)
        assert is_valid is True
    
    def test_validate_trade_unacceptable(self, risk_manager, risk_metrics):
        """Тест валидации неприемлемой сделки."""
        risk_manager.update_metrics(risk_metrics)
        
        # Слишком большая сделка (5% от портфеля)
        trade_value = Money(Decimal("5000"), Currency.USD)
        portfolio_value = Money(Decimal("100000"), Currency.USD)
        
        is_valid = risk_manager.validate_trade(trade_value, portfolio_value)
        assert is_valid is False
    
    def test_validate_trade_no_metrics(self, risk_manager):
        """Тест валидации сделки без метрик."""
        trade_value = Money(Decimal("1000"), Currency.USD)
        portfolio_value = Money(Decimal("100000"), Currency.USD)
        
        is_valid = risk_manager.validate_trade(trade_value, portfolio_value)
        assert is_valid is False
    
    def test_get_position_size_kelly(self, risk_manager):
        """Тест расчета размера позиции методом Келли."""
        risk_manager.risk_profile.position_sizing_method = "kelly"
        
        available_capital = Money(Decimal("100000"), Currency.USD)
        risk_per_trade = Decimal("0.02")
        
        position_size = risk_manager.get_position_size(available_capital, risk_per_trade)
        
        assert isinstance(position_size, Money)
        assert position_size.amount > Decimal("0")
        assert position_size.amount <= available_capital.amount
    
    def test_get_position_size_fixed(self, risk_manager):
        """Тест расчета размера позиции фиксированным методом."""
        risk_manager.risk_profile.position_sizing_method = "fixed"
        
        available_capital = Money(Decimal("100000"), Currency.USD)
        risk_per_trade = Decimal("0.02")
        
        position_size = risk_manager.get_position_size(available_capital, risk_per_trade)
        
        assert isinstance(position_size, Money)
        assert position_size.amount > Decimal("0")
        assert position_size.amount <= available_capital.amount
    
    def test_get_position_size_volatility(self, risk_manager):
        """Тест расчета размера позиции методом волатильности."""
        risk_manager.risk_profile.position_sizing_method = "volatility"
        
        available_capital = Money(Decimal("100000"), Currency.USD)
        risk_per_trade = Decimal("0.02")
        
        position_size = risk_manager.get_position_size(available_capital, risk_per_trade)
        
        assert isinstance(position_size, Money)
        assert position_size.amount > Decimal("0")
        assert position_size.amount <= available_capital.amount
    
    def test_should_stop_trading_healthy(self, risk_manager, risk_metrics):
        """Тест проверки необходимости остановки торговли - здоровый портфель."""
        risk_manager.update_metrics(risk_metrics)
        
        should_stop = risk_manager.should_stop_trading()
        assert should_stop is False
    
    def test_should_stop_trading_unhealthy(self, risk_manager, risk_metrics):
        """Тест проверки необходимости остановки торговли - нездоровый портфель."""
        # Устанавливаем высокие риски
        risk_metrics.portfolio_risk = Decimal("0.3")
        risk_metrics.current_drawdown = Decimal("0.4")
        risk_manager.update_metrics(risk_metrics)
        
        should_stop = risk_manager.should_stop_trading()
        assert should_stop is True
    
    def test_should_stop_trading_no_metrics(self, risk_manager):
        """Тест проверки необходимости остановки торговли без метрик."""
        should_stop = risk_manager.should_stop_trading()
        assert should_stop is True
    
    def test_get_risk_alerts_empty(self, risk_manager):
        """Тест получения предупреждений о рисках - пустой список."""
        alerts = risk_manager.get_risk_alerts()
        assert alerts == []
    
    def test_get_risk_alerts_with_alerts(self, risk_manager, risk_metrics):
        """Тест получения предупреждений о рисках - с предупреждениями."""
        # Устанавливаем высокие риски для генерации предупреждений
        risk_metrics.portfolio_risk = Decimal("0.3")
        risk_metrics.current_drawdown = Decimal("0.4")
        risk_metrics.max_correlation = Decimal("0.8")
        risk_manager.update_metrics(risk_metrics)
        
        alerts = risk_manager.get_risk_alerts()
        assert len(alerts) > 0
        assert all(isinstance(alert, str) for alert in alerts)
    
    def test_to_dict(self, risk_manager):
        """Тест сериализации в словарь."""
        data = risk_manager.to_dict()
        
        assert data["name"] == risk_manager.name
        assert data["is_active"] == risk_manager.is_active
        assert "risk_profile" in data
        assert "current_metrics" in data
        assert "risk_alerts" in data
        assert "last_update" in data
    
    def test_from_dict(self, risk_manager):
        """Тест десериализации из словаря."""
        data = risk_manager.to_dict()
        new_manager = RiskManager.from_dict(data)
        
        assert new_manager.name == risk_manager.name
        assert new_manager.is_active == risk_manager.is_active
        assert new_manager.risk_profile.name == risk_manager.risk_profile.name
        assert new_manager.risk_profile.risk_level == risk_manager.risk_profile.risk_level 