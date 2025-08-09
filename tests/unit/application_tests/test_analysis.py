"""
Тесты для анализа application слоя.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from decimal import Decimal
from typing import Any

from application.analysis.entanglement_monitor import EntanglementMonitor


class TestEntanglementMonitor:
    """Тесты для EntanglementMonitor."""

    @pytest.fixture
    def mock_repositories(self) -> Mock:
        """Создает mock репозитории."""
        market_repo = Mock()
        market_repo.get_market_data = AsyncMock()
        market_repo.get_correlation_data = AsyncMock()
        return market_repo

    @pytest.fixture
    def monitor(self, mock_repositories: Mock) -> EntanglementMonitor:
        """Создает экземпляр монитора."""
        return EntanglementMonitor(mock_repositories)

    @pytest.fixture
    def sample_market_data(self) -> list[dict[str, Any]]:
        """Создает образец рыночных данных."""
        return [
            {"timestamp": "2024-01-01T00:00:00", "close": "50000", "volume": "1000"},
            {"timestamp": "2024-01-01T01:00:00", "close": "51000", "volume": "1200"},
            {"timestamp": "2024-01-01T02:00:00", "close": "52000", "volume": "1100"},
            {"timestamp": "2024-01-01T03:00:00", "close": "51500", "volume": "1300"},
            {"timestamp": "2024-01-01T04:00:00", "close": "53000", "volume": "1400"},
        ]

    @pytest.mark.asyncio
    async def test_detect_entanglement(
        self, monitor: EntanglementMonitor, mock_repositories: Mock, sample_market_data: list[dict[str, Any]]
    ) -> None:
        """Тест обнаружения запутанности."""
        symbol1 = "BTC/USD"
        symbol2 = "ETH/USD"
        timeframe = "1h"

        mock_repositories.get_market_data.return_value = sample_market_data

        result = await monitor.analyze_entanglement(symbol1, symbol2, timeframe)

        assert "entanglement_score" in result
        assert "correlation" in result
        assert "phase_shift" in result
        assert "confidence" in result
        assert isinstance(result["entanglement_score"], (int, float))
        assert isinstance(result["correlation"], (int, float))
        assert isinstance(result["phase_shift"], (int, float))
        assert isinstance(result["confidence"], (int, float))

        # Проверяем вызовы
        assert mock_repositories.get_market_data.call_count == 2

    @pytest.mark.asyncio
    async def test_analyze_market_correlations(self, monitor: EntanglementMonitor, mock_repositories: Mock) -> None:
        """Тест анализа рыночных корреляций."""
        symbols = ["BTC/USD", "ETH/USD", "ADA/USD"]
        timeframe = "1h"

        mock_correlation_data = {
            "BTC/USD": {"ETH/USD": 0.8, "ADA/USD": 0.6},
            "ETH/USD": {"BTC/USD": 0.8, "ADA/USD": 0.7},
            "ADA/USD": {"BTC/USD": 0.6, "ETH/USD": 0.7},
        }

        mock_repositories.get_correlation_data.return_value = mock_correlation_data

        result = await monitor.analyze_correlations(symbols, timeframe)

        assert "correlation_matrix" in result
        assert "strong_correlations" in result
        assert "weak_correlations" in result
        assert "correlation_clusters" in result

        assert isinstance(result["correlation_matrix"], dict)
        assert isinstance(result["strong_correlations"], list)
        assert isinstance(result["weak_correlations"], list)
        assert isinstance(result["correlation_clusters"], list)

        mock_repositories.get_correlation_data.assert_called_once_with(symbols, timeframe)

    def test_calculate_correlation(self, monitor: EntanglementMonitor) -> None:
        """Тест расчета корреляции."""
        prices1 = [Decimal("50000"), Decimal("51000"), Decimal("52000")]
        prices2 = [Decimal("3000"), Decimal("3100"), Decimal("3200")]

        correlation = monitor.calculate_correlation(prices1, prices2)

        assert isinstance(correlation, (int, float))
        assert -1 <= correlation <= 1

    def test_calculate_phase_shift(self, monitor: EntanglementMonitor) -> None:
        """Тест расчета фазового сдвига."""
        prices1 = [Decimal("50000"), Decimal("51000"), Decimal("52000")]
        prices2 = [Decimal("3000"), Decimal("3100"), Decimal("3200")]

        phase_shift = monitor.calculate_phase_shift(prices1, prices2)

        assert isinstance(phase_shift, (int, float))

    def test_calculate_entanglement_score(self, monitor: EntanglementMonitor) -> None:
        """Тест расчета оценки запутанности."""
        correlation = 0.8
        phase_shift = 0.1
        volatility_ratio = 1.2

        score = monitor.calculate_entanglement_score(correlation, phase_shift, volatility_ratio)

        assert isinstance(score, (int, float))
        assert 0 <= score <= 1

    def test_detect_correlation_clusters(self, monitor: EntanglementMonitor) -> None:
        """Тест обнаружения кластеров корреляции."""
        correlation_matrix = {
            "BTC/USD": {"ETH/USD": 0.8, "ADA/USD": 0.6, "DOT/USD": 0.3},
            "ETH/USD": {"BTC/USD": 0.8, "ADA/USD": 0.7, "DOT/USD": 0.4},
            "ADA/USD": {"BTC/USD": 0.6, "ETH/USD": 0.7, "DOT/USD": 0.5},
            "DOT/USD": {"BTC/USD": 0.3, "ETH/USD": 0.4, "ADA/USD": 0.5},
        }

        clusters = monitor.detect_correlation_clusters(correlation_matrix, threshold=0.6)

        assert isinstance(clusters, list)
        assert len(clusters) > 0

        for cluster in clusters:
            assert isinstance(cluster, list)
            assert len(cluster) > 1

    def test_calculate_volatility_ratio(self, monitor: EntanglementMonitor) -> None:
        """Тест расчета отношения волатильности."""
        prices1 = [Decimal("50000"), Decimal("51000"), Decimal("52000")]
        prices2 = [Decimal("3000"), Decimal("3100"), Decimal("3200")]

        ratio = monitor.calculate_volatility_ratio(prices1, prices2)

        assert isinstance(ratio, (int, float))
        assert ratio > 0

    @pytest.mark.asyncio
    async def test_monitor_entanglement_changes(self, monitor: EntanglementMonitor, mock_repositories: Mock) -> None:
        """Тест мониторинга изменений запутанности."""
        symbol1 = "BTC/USD"
        symbol2 = "ETH/USD"
        timeframe = "1h"
        window_size = 10

        mock_repositories.get_market_data.return_value = []

        result = await monitor.monitor_changes(symbol1, symbol2, timeframe, window_size)

        assert "current_entanglement" in result
        assert "entanglement_trend" in result
        assert "change_detected" in result
        assert "change_magnitude" in result

        assert isinstance(result["current_entanglement"], (int, float))
        assert isinstance(result["entanglement_trend"], str)
        assert isinstance(result["change_detected"], bool)
        assert isinstance(result["change_magnitude"], (int, float))

    def test_detect_entanglement_breakdown(self, monitor: EntanglementMonitor) -> None:
        """Тест обнаружения разрыва запутанности."""
        historical_scores = [0.8, 0.7, 0.6, 0.3, 0.2]  # Резкое падение

        breakdown_detected = monitor.detect_breakdown(historical_scores, threshold=0.5)

        assert isinstance(breakdown_detected, bool)
        assert breakdown_detected is True

    def test_calculate_entanglement_trend(self, monitor: EntanglementMonitor) -> None:
        """Тест расчета тренда запутанности."""
        historical_scores = [0.5, 0.6, 0.7, 0.8, 0.9]  # Растущий тренд

        trend = monitor.calculate_trend(historical_scores)

        assert isinstance(trend, str)
        assert trend in ["increasing", "decreasing", "stable"]

    def test_validate_input_data(self, monitor: EntanglementMonitor) -> None:
        """Тест валидации входных данных."""
        # Корректные данные
        valid_prices = [Decimal("50000"), Decimal("51000"), Decimal("52000")]
        assert monitor.validate_data(valid_prices) is True

        # Некорректные данные - пустой список
        empty_prices: list = []
        assert monitor.validate_data(empty_prices) is False

        # Некорректные данные - None
        assert monitor.validate_data(None) is False

        # Некорректные данные - не список
        assert monitor.validate_data("invalid") is False

    def test_calculate_confidence_interval(self, monitor: EntanglementMonitor) -> None:
        """Тест расчета доверительного интервала."""
        prices1 = [Decimal("50000"), Decimal("51000"), Decimal("52000")]
        prices2 = [Decimal("3000"), Decimal("3100"), Decimal("3200")]
        confidence_level = 0.95

        interval = monitor.calculate_confidence_interval(prices1, prices2, confidence_level)

        assert isinstance(interval, dict)
        assert "lower_bound" in interval
        assert "upper_bound" in interval
        assert interval["lower_bound"] <= interval["upper_bound"]

    @pytest.mark.asyncio
    async def test_get_entanglement_history(self, monitor: EntanglementMonitor, mock_repositories: Mock) -> None:
        """Тест получения истории запутанности."""
        symbol1 = "BTC/USD"
        symbol2 = "ETH/USD"
        timeframe = "1h"
        limit = 100

        mock_history = [
            {"timestamp": "2024-01-01T00:00:00", "entanglement_score": 0.8},
            {"timestamp": "2024-01-01T01:00:00", "entanglement_score": 0.7},
            {"timestamp": "2024-01-01T02:00:00", "entanglement_score": 0.9},
        ]

        mock_repositories.get_market_data.return_value = mock_history

        result = monitor.get_entanglement_history(limit)

        assert isinstance(result, list)
        assert len(result) > 0

        for entry in result:
            assert "timestamp" in entry
            assert "entanglement_score" in entry
            assert isinstance(entry["entanglement_score"], (int, float))
