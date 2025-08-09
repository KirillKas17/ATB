"""
Unit тесты для StrategyService.

Покрывает:
- Основной функционал управления стратегиями
- Создание стратегий
- Валидацию стратегий
- Оптимизацию стратегий
- Анализ производительности
- Обработку ошибок
"""

import pytest
import pandas as pd
from shared.numpy_utils import np
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from domain.services.strategy_service import StrategyService
from domain.entities.strategy import Strategy, StrategyStatus, StrategyType
from domain.exceptions import StrategyExecutionError
from domain.type_definitions.strategy_types import (
    StrategyValidationResult,
    StrategyOptimizationResult,
    StrategyPerformanceResult,
)


class TestStrategyService:
    """Тесты для StrategyService."""

    @pytest.fixture
    def service(self) -> StrategyService:
        """Экземпляр StrategyService."""
        return StrategyService()

    @pytest.fixture
    def valid_config(self) -> Dict[str, Any]:
        """Валидная конфигурация стратегии."""
        return {
            "name": "test_strategy",
            "description": "Test strategy for unit testing",
            "strategy_type": "trend_following",
            "trading_pairs": ["BTC/USD", "ETH/USD"],
            "parameters": {"trend_strength": 0.5, "trend_period": 20, "stop_loss": 0.05, "take_profit": 0.1},
            "metadata": {"author": "test_user", "version": "1.0.0"},
        }

    @pytest.fixture
    def historical_data(self) -> pd.DataFrame:
        """Тестовые исторические данные."""
        dates = pd.date_range("2023-01-01", periods=100, freq="1H")
        data = pd.DataFrame(
            {
                "open": np.random.uniform(100, 200, 100),
                "high": np.random.uniform(200, 300, 100),
                "low": np.random.uniform(50, 100, 100),
                "close": np.random.uniform(100, 200, 100),
                "volume": np.random.uniform(1000, 10000, 100),
            },
            index=dates,
        )

        # Создаем тренд
        data["close"] = data["close"] + np.arange(100) * 0.1
        return data

    def test_creation(self):
        """Тест создания сервиса."""
        service = StrategyService()

        assert service._validation_rules is not None
        assert len(service._validation_rules) > 0
        assert "trend_following" in service._validation_rules
        assert "mean_reversion" in service._validation_rules
        assert "breakout" in service._validation_rules
        assert "scalping" in service._validation_rules

    def test_setup_validation_rules(self, service):
        """Тест настройки правил валидации."""
        rules = service._setup_validation_rules()

        expected_strategies = ["trend_following", "mean_reversion", "breakout", "scalping"]

        for strategy_type in expected_strategies:
            assert strategy_type in rules
            assert "required_params" in rules[strategy_type]
            assert "param_ranges" in rules[strategy_type]
            assert isinstance(rules[strategy_type]["required_params"], list)
            assert isinstance(rules[strategy_type]["param_ranges"], dict)

    @pytest.mark.asyncio
    async def test_create_strategy_valid_config(self, service, valid_config):
        """Тест создания стратегии с валидной конфигурацией."""
        strategy = await service.create_strategy(valid_config)

        assert isinstance(strategy, Strategy)
        assert strategy.name == "test_strategy"
        assert strategy.description == "Test strategy for unit testing"
        assert strategy.strategy_type == StrategyType.TREND_FOLLOWING
        assert strategy.trading_pairs == ["BTC/USD", "ETH/USD"]
        assert strategy.status == StrategyStatus.ACTIVE
        assert strategy.parameters.get("trend_strength") == 0.5
        assert strategy.metadata["author"] == "test_user"

    @pytest.mark.asyncio
    async def test_create_strategy_invalid_config(self, service):
        """Тест создания стратегии с невалидной конфигурацией."""
        invalid_config = {
            "name": "test_strategy",
            "strategy_type": "invalid_type",  # Невалидный тип
            "trading_pairs": [],
        }

        with pytest.raises(StrategyExecutionError, match="Invalid strategy config"):
            await service.create_strategy(invalid_config)

    @pytest.mark.asyncio
    async def test_create_strategy_missing_required_params(self, service):
        """Тест создания стратегии с отсутствующими обязательными параметрами."""
        config = {
            "name": "test_strategy",
            "strategy_type": "trend_following",
            "trading_pairs": ["BTC/USD"],
            "parameters": {
                "trend_strength": 0.5
                # Отсутствуют другие обязательные параметры
            },
        }

        with pytest.raises(StrategyExecutionError, match="Invalid strategy config"):
            await service.create_strategy(config)

    @pytest.mark.asyncio
    async def test_create_strategy_invalid_parameter_ranges(self, service):
        """Тест создания стратегии с невалидными диапазонами параметров."""
        config = {
            "name": "test_strategy",
            "strategy_type": "trend_following",
            "trading_pairs": ["BTC/USD"],
            "parameters": {
                "trend_strength": 2.0,  # Вне допустимого диапазона
                "trend_period": 20,
                "stop_loss": 0.05,
                "take_profit": 0.1,
            },
        }

        with pytest.raises(StrategyExecutionError, match="Invalid strategy config"):
            await service.create_strategy(config)

    @pytest.mark.asyncio
    async def test_validate_strategy_valid(self, service, valid_config):
        """Тест валидации валидной стратегии."""
        strategy = await service.create_strategy(valid_config)

        result = await service.validate_strategy(strategy)

        assert isinstance(result, StrategyValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) >= 0

    @pytest.mark.asyncio
    async def test_validate_strategy_invalid(self, service):
        """Тест валидации невалидной стратегии."""
        strategy = Strategy(
            name="invalid_strategy",
            description="Invalid strategy",
            strategy_type=StrategyType.TREND_FOLLOWING,
            trading_pairs=[],  # Пустой список торговых пар
        )

        result = await service.validate_strategy(strategy)

        assert isinstance(result, StrategyValidationResult)
        assert result.is_valid is False
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_validate_config_valid(self, service, valid_config):
        """Тест валидации валидной конфигурации."""
        errors = await service._validate_config(valid_config)

        assert isinstance(errors, list)
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_validate_config_missing_name(self, service):
        """Тест валидации конфигурации без имени."""
        config = {"strategy_type": "trend_following", "trading_pairs": ["BTC/USD"]}

        errors = await service._validate_config(config)

        assert isinstance(errors, list)
        assert len(errors) > 0
        assert any("name" in error.lower() for error in errors)

    @pytest.mark.asyncio
    async def test_validate_config_missing_strategy_type(self, service):
        """Тест валидации конфигурации без типа стратегии."""
        config = {"name": "test_strategy", "trading_pairs": ["BTC/USD"]}

        errors = await service._validate_config(config)

        assert isinstance(errors, list)
        assert len(errors) > 0
        assert any("strategy_type" in error.lower() for error in errors)

    @pytest.mark.asyncio
    async def test_validate_config_invalid_strategy_type(self, service):
        """Тест валидации конфигурации с невалидным типом стратегии."""
        config = {"name": "test_strategy", "strategy_type": "invalid_type", "trading_pairs": ["BTC/USD"]}

        errors = await service._validate_config(config)

        assert isinstance(errors, list)
        assert len(errors) > 0
        assert any("strategy_type" in error.lower() for error in errors)

    @pytest.mark.asyncio
    async def test_validate_parameters_valid(self, service, valid_config):
        """Тест валидации валидных параметров."""
        strategy = await service.create_strategy(valid_config)

        errors = await service._validate_parameters(strategy)

        assert isinstance(errors, list)
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_validate_parameters_missing_required(self, service):
        """Тест валидации параметров с отсутствующими обязательными."""
        strategy = Strategy(
            name="test_strategy",
            description="Test strategy",
            strategy_type=StrategyType.TREND_FOLLOWING,
            trading_pairs=["BTC/USD"],
        )
        # Не устанавливаем параметры

        errors = await service._validate_parameters(strategy)

        assert isinstance(errors, list)
        assert len(errors) > 0
        assert any("trend_strength" in error.lower() for error in errors)

    @pytest.mark.asyncio
    async def test_validate_parameters_out_of_range(self, service):
        """Тест валидации параметров вне допустимого диапазона."""
        strategy = Strategy(
            name="test_strategy",
            description="Test strategy",
            strategy_type=StrategyType.TREND_FOLLOWING,
            trading_pairs=["BTC/USD"],
        )
        strategy.parameters.update_parameters(
            {
                "trend_strength": 2.0,  # Вне диапазона (0.1, 1.0)
                "trend_period": 20,
                "stop_loss": 0.05,
                "take_profit": 0.1,
            }
        )

        errors = await service._validate_parameters(strategy)

        assert isinstance(errors, list)
        assert len(errors) > 0
        assert any("trend_strength" in error.lower() for error in errors)

    @pytest.mark.asyncio
    async def test_optimize_strategy(self, service, valid_config, historical_data):
        """Тест оптимизации стратегии."""
        strategy = await service.create_strategy(valid_config)

        result = await service.optimize_strategy(strategy, historical_data)

        assert isinstance(result, StrategyOptimizationResult)
        assert result.optimized_parameters is not None
        assert result.performance_improvement >= 0.0
        assert result.optimization_confidence >= 0.0
        assert result.optimization_confidence <= 1.0

    @pytest.mark.asyncio
    async def test_optimize_strategy_empty_data(self, service, valid_config):
        """Тест оптимизации стратегии с пустыми данными."""
        strategy = await service.create_strategy(valid_config)
        empty_data = pd.DataFrame()

        result = await service.optimize_strategy(strategy, empty_data)

        assert isinstance(result, StrategyOptimizationResult)
        assert result.optimized_parameters is not None
        assert result.performance_improvement == 0.0
        assert result.optimization_confidence == 0.0

    @pytest.mark.asyncio
    async def test_analyze_performance(self, service, valid_config):
        """Тест анализа производительности стратегии."""
        strategy = await service.create_strategy(valid_config)
        period = timedelta(days=30)

        result = await service.analyze_performance(strategy, period)

        assert isinstance(result, StrategyPerformanceResult)
        assert result.total_return >= 0.0
        assert result.sharpe_ratio is not None
        assert result.max_drawdown >= 0.0
        assert result.win_rate >= 0.0
        assert result.win_rate <= 1.0
        assert result.quality_score >= 0.0
        assert result.quality_score <= 1.0
        assert isinstance(result.risk_level, str)
        assert isinstance(result.recommendations, list)

    @pytest.mark.asyncio
    async def test_analyze_performance_zero_period(self, service, valid_config):
        """Тест анализа производительности с нулевым периодом."""
        strategy = await service.create_strategy(valid_config)
        period = timedelta(0)

        result = await service.analyze_performance(strategy, period)

        assert isinstance(result, StrategyPerformanceResult)
        assert result.total_return == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.max_drawdown == 0.0
        assert result.win_rate == 0.0

    def test_calculate_quality_score(self, service):
        """Тест расчета оценки качества."""
        analysis = {
            "total_return": 0.15,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.05,
            "win_rate": 0.65,
            "volatility": 0.12,
        }

        score = service._calculate_quality_score(analysis)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_calculate_quality_score_negative_returns(self, service):
        """Тест расчета оценки качества с отрицательной доходностью."""
        analysis = {
            "total_return": -0.10,
            "sharpe_ratio": -0.5,
            "max_drawdown": 0.15,
            "win_rate": 0.35,
            "volatility": 0.20,
        }

        score = service._calculate_quality_score(analysis)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # Низкая оценка для плохой стратегии

    def test_assess_risk_level(self, service):
        """Тест оценки уровня риска."""
        # Низкий риск
        low_risk_analysis = {"max_drawdown": 0.02, "volatility": 0.08, "var_95": -0.03}

        risk_level = service._assess_risk_level(low_risk_analysis)

        assert isinstance(risk_level, str)
        assert risk_level in ["low", "medium", "high"]

        # Высокий риск
        high_risk_analysis = {"max_drawdown": 0.25, "volatility": 0.35, "var_95": -0.15}

        risk_level = service._assess_risk_level(high_risk_analysis)

        assert isinstance(risk_level, str)
        assert risk_level in ["low", "medium", "high"]

    @pytest.mark.asyncio
    async def test_generate_recommendations(self, service, valid_config):
        """Тест генерации рекомендаций."""
        strategy = await service.create_strategy(valid_config)

        recommendations = await service._generate_recommendations(strategy)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)

    @pytest.mark.asyncio
    async def test_create_strategy_different_types(self, service):
        """Тест создания стратегий разных типов."""
        strategy_types = ["trend_following", "mean_reversion", "breakout", "scalping"]

        for strategy_type in strategy_types:
            config = {
                "name": f"test_{strategy_type}",
                "description": f"Test {strategy_type} strategy",
                "strategy_type": strategy_type,
                "trading_pairs": ["BTC/USD"],
                "parameters": self._get_parameters_for_type(strategy_type),
            }

            strategy = await service.create_strategy(config)

            assert isinstance(strategy, Strategy)
            assert strategy.name == f"test_{strategy_type}"
            assert strategy.strategy_type.value == strategy_type

    def _get_parameters_for_type(self, strategy_type: str) -> Dict[str, Any]:
        """Получить параметры для конкретного типа стратегии."""
        parameters = {
            "trend_following": {"trend_strength": 0.5, "trend_period": 20, "stop_loss": 0.05, "take_profit": 0.1},
            "mean_reversion": {
                "mean_reversion_threshold": 2.0,
                "lookback_period": 50,
                "stop_loss": 0.05,
                "take_profit": 0.1,
            },
            "breakout": {"breakout_threshold": 1.5, "volume_multiplier": 2.0, "stop_loss": 0.05, "take_profit": 0.1},
            "scalping": {"scalping_threshold": 0.1, "max_hold_time": 300, "stop_loss": 0.01, "take_profit": 0.02},
        }
        return parameters.get(strategy_type, {})

    @pytest.mark.asyncio
    async def test_error_handling_invalid_config_type(self, service):
        """Тест обработки ошибок с невалидным типом конфигурации."""
        invalid_config = "not a dict"

        with pytest.raises(StrategyExecutionError, match="Invalid strategy config"):
            await service.create_strategy(invalid_config)

    @pytest.mark.asyncio
    async def test_error_handling_none_strategy(self, service):
        """Тест обработки ошибок с None стратегией."""
        with pytest.raises(Exception):  # Должно вызывать исключение
            await service.validate_strategy(None)

    @pytest.mark.asyncio
    async def test_error_handling_none_historical_data(self, service, valid_config):
        """Тест обработки ошибок с None историческими данными."""
        strategy = await service.create_strategy(valid_config)

        result = await service.optimize_strategy(strategy, None)

        assert isinstance(result, StrategyOptimizationResult)
        assert result.optimized_parameters is not None
        assert result.performance_improvement == 0.0

    @pytest.mark.asyncio
    async def test_concurrent_strategy_operations(self, service, valid_config, historical_data):
        """Тест конкурентных операций со стратегиями."""
        import asyncio

        strategy = await service.create_strategy(valid_config)

        async def validate_strategy():
            return await service.validate_strategy(strategy)

        async def optimize_strategy():
            return await service.optimize_strategy(strategy, historical_data)

        async def analyze_performance():
            return await service.analyze_performance(strategy, timedelta(days=30))

        # Запускаем операции одновременно
        tasks = [validate_strategy(), optimize_strategy(), analyze_performance()]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert isinstance(results[0], StrategyValidationResult)
        assert isinstance(results[1], StrategyOptimizationResult)
        assert isinstance(results[2], StrategyPerformanceResult)

    @pytest.mark.asyncio
    async def test_strategy_parameter_validation_edge_cases(self, service):
        """Тест валидации параметров стратегии на граничных случаях."""
        strategy = Strategy(
            name="edge_case_strategy",
            description="Edge case strategy",
            strategy_type=StrategyType.TREND_FOLLOWING,
            trading_pairs=["BTC/USD"],
        )

        # Граничные значения параметров
        edge_case_parameters = {
            "trend_strength": 0.1,  # Минимальное значение
            "trend_period": 5,  # Минимальное значение
            "stop_loss": 0.01,  # Минимальное значение
            "take_profit": 0.02,  # Минимальное значение
        }

        strategy.parameters.update_parameters(edge_case_parameters)

        errors = await service._validate_parameters(strategy)

        assert isinstance(errors, list)
        assert len(errors) == 0  # Граничные значения должны быть валидными

    @pytest.mark.asyncio
    async def test_strategy_metadata_handling(self, service):
        """Тест обработки метаданных стратегии."""
        config = {
            "name": "metadata_test_strategy",
            "strategy_type": "trend_following",
            "trading_pairs": ["BTC/USD"],
            "parameters": {"trend_strength": 0.5, "trend_period": 20, "stop_loss": 0.05, "take_profit": 0.1},
            "metadata": {
                "author": "test_user",
                "version": "1.0.0",
                "tags": ["trend", "momentum"],
                "description": "Detailed description",
                "risk_level": "medium",
            },
        }

        strategy = await service.create_strategy(config)

        assert strategy.metadata["author"] == "test_user"
        assert strategy.metadata["version"] == "1.0.0"
        assert strategy.metadata["tags"] == ["trend", "momentum"]
        assert strategy.metadata["risk_level"] == "medium"
