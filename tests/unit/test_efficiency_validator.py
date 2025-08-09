"""
Unit тесты для EfficiencyValidator.
Тестирует валидацию эффективности стратегий, анализ метрик производительности
и генерацию рекомендаций по оптимизации.
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from shared.numpy_utils import np
from infrastructure.core.efficiency_validator import EfficiencyValidator


class TestEfficiencyValidator:
    """Тесты для EfficiencyValidator."""

    @pytest.fixture
    def efficiency_validator(self) -> EfficiencyValidator:
        """Фикстура для EfficiencyValidator."""
        return EfficiencyValidator()

    @pytest.fixture
    def sample_strategy_data(self) -> dict:
        """Фикстура с тестовыми данными стратегии."""
        return {
            "strategy_name": "test_strategy",
            "total_trades": 150,
            "winning_trades": 95,
            "losing_trades": 55,
            "total_profit": 2500.0,
            "total_loss": -800.0,
            "max_drawdown": -0.12,
            "sharpe_ratio": 1.8,
            "profit_factor": 3.125,
            "avg_win": 26.32,
            "avg_loss": -14.55,
            "win_rate": 0.633,
            "total_return": 0.17,
            "volatility": 0.08,
            "max_consecutive_losses": 5,
            "avg_trade_duration": 3600,
            "commission_rate": 0.001,
            "slippage": 0.0005,
        }

    @pytest.fixture
    def sample_performance_metrics(self) -> dict:
        """Фикстура с метриками производительности."""
        return {
            "returns": np.random.normal(0.001, 0.02, 1000),
            "drawdowns": np.random.uniform(-0.15, 0, 1000),
            "volatility": 0.08,
            "sharpe_ratio": 1.8,
            "sortino_ratio": 2.1,
            "calmar_ratio": 1.4,
            "max_drawdown": -0.12,
            "var_95": -0.025,
            "cvar_95": -0.035,
            "profit_factor": 3.125,
            "win_rate": 0.633,
            "avg_win_loss_ratio": 1.81,
        }

    def test_initialization(self, efficiency_validator: EfficiencyValidator) -> None:
        """Тест инициализации валидатора."""
        assert efficiency_validator is not None
        assert hasattr(efficiency_validator, "config")
        assert hasattr(efficiency_validator, "testing_candidates")
        assert hasattr(efficiency_validator, "confirmed_evolutions")
        assert hasattr(efficiency_validator, "validation_stats")

    def test_validation_thresholds_structure(self, efficiency_validator: EfficiencyValidator) -> None:
        """Тест структуры порогов валидации."""
        config = efficiency_validator.config
        # Проверка наличия всех порогов
        assert hasattr(config, "efficiency_improvement_threshold")
        assert hasattr(config, "confidence_threshold")
        assert hasattr(config, "stability_threshold")
        assert hasattr(config, "statistical_significance_level")
        # Проверка валидности значений
        assert config.efficiency_improvement_threshold > 0
        assert 0 < config.confidence_threshold < 1
        assert 0 < config.stability_threshold < 1
        assert 0 < config.statistical_significance_level < 1

    @pytest.mark.asyncio
    async def test_validate_evolution_candidate(self, efficiency_validator: EfficiencyValidator) -> None:
        """Тест валидации кандидата на эволюцию."""

        # Мок компонента
        class MockComponent:
            def __init__(self) -> Any:
                self.name = "test_component"

            def get_performance(self) -> Any:
                return 0.75

            def save_state(self, path) -> Any:
                return True

            async def evolve(self, data) -> Any:
                return True

        component = MockComponent()
        proposed_changes = {"param1": 0.1, "param2": 0.2}

        # Валидация кандидата
        validation_result = await efficiency_validator.validate_evolution_candidate(component, proposed_changes)

        # Проверки результата
        assert isinstance(validation_result, bool)

    def test_get_validation_stats(self, efficiency_validator: EfficiencyValidator) -> None:
        """Тест получения статистики валидации."""
        # Получение статистики
        stats = efficiency_validator.get_validation_stats()

        # Проверки
        assert stats is not None
        assert isinstance(stats, dict)
        assert "total_validations" in stats
        assert "successful_validations" in stats
        assert "failed_validations" in stats
        assert "success_rate" in stats
        assert "avg_improvement" in stats
        assert "avg_confidence" in stats
        # Проверка типов данных
        assert isinstance(stats["total_validations"], int)
        assert isinstance(stats["successful_validations"], int)
        assert isinstance(stats["failed_validations"], int)
        assert isinstance(stats["success_rate"], float)
        assert isinstance(stats["avg_improvement"], float)
        assert isinstance(stats["avg_confidence"], float)
        # Проверка диапазонов
        assert 0 <= stats["success_rate"] <= 1
        assert 0 <= stats["avg_confidence"] <= 1

    def test_get_evolution_history(self, efficiency_validator: EfficiencyValidator) -> None:
        """Тест получения истории эволюции."""
        # Получение истории
        history = efficiency_validator.get_evolution_history()

        # Проверки
        assert history is not None
        assert isinstance(history, list)
        # Каждая запись должна быть словарем
        for entry in history:
            assert isinstance(entry, dict)
            assert "component_name" in entry
            assert "timestamp" in entry
            assert "status" in entry
            assert "improvement" in entry
            assert "confidence" in entry

    def test_get_rollback_history(self, efficiency_validator: EfficiencyValidator) -> None:
        """Тест получения истории откатов."""
        # Получение истории откатов
        rollback_history = efficiency_validator.get_rollback_history()

        # Проверки
        assert rollback_history is not None
        assert isinstance(rollback_history, list)
        # Каждая запись должна быть словарем
        for entry in rollback_history:
            assert isinstance(entry, dict)
            assert "component_name" in entry
            assert "timestamp" in entry
            assert "reason" in entry
            assert "backup_path" in entry

    def test_save_validation_log(self, efficiency_validator: EfficiencyValidator) -> None:
        """Тест сохранения лога валидации."""
        # Сохранение лога
        efficiency_validator.save_validation_log()

        # Проверяем, что метод выполнился без ошибок
        assert True

    def test_load_validation_log(self, efficiency_validator: EfficiencyValidator) -> None:
        """Тест загрузки лога валидации."""
        # Создаем тестовый файл лога
        test_log_file = "test_validation_log.json"

        # Загрузка лога
        log_data = efficiency_validator.load_validation_log(test_log_file)

        # Проверки
        assert log_data is not None
        assert isinstance(log_data, dict)

    @pytest.mark.asyncio
    async def test_error_handling(self, efficiency_validator: EfficiencyValidator) -> None:
        """Тест обработки ошибок."""
        # Тест с некорректными данными
        invalid_component = None
        invalid_changes: dict = {}

        # Валидация должна обработать ошибку
        result = await efficiency_validator.validate_evolution_candidate(invalid_component, invalid_changes)
        assert result is False

    @pytest.mark.asyncio
    async def test_edge_cases(self, efficiency_validator: EfficiencyValidator) -> None:
        """Тест граничных случаев."""

        # Тест с пустыми изменениями
        class MockComponent:
            def __init__(self) -> Any:
                self.name = "test_component"

            def get_performance(self) -> Any:
                return 0.5

            def save_state(self, path) -> Any:
                return True

            async def evolve(self, data) -> Any:
                return True

        component = MockComponent()
        empty_changes: dict = {}

        # Валидация с пустыми изменениями
        result = await efficiency_validator.validate_evolution_candidate(component, empty_changes)
        assert isinstance(result, bool)

        # Тест с очень большими изменениями
        large_changes = {"param": 1000000.0}
        result = await efficiency_validator.validate_evolution_candidate(component, large_changes)
        assert isinstance(result, bool)

        # Тест с отрицательными значениями
        negative_changes = {"param": -1.0}
        result = await efficiency_validator.validate_evolution_candidate(component, negative_changes)
        assert isinstance(result, bool)

    def test_cleanup(self, efficiency_validator: EfficiencyValidator) -> None:
        """Тест очистки ресурсов."""
        # Проверяем, что валидатор остается функциональным
        stats = efficiency_validator.get_validation_stats()
        assert stats is not None
        assert isinstance(stats, dict)

        # Проверяем, что история эволюции доступна
        history = efficiency_validator.get_evolution_history()
        assert history is not None
        assert isinstance(history, list)
