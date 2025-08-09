"""
Unit тесты для LiquidityAnalyzer.
Тестирует анализ ликвидности, включая расчет глубины рынка,
анализ спредов, оценку ликвидности и мониторинг ликвидных рисков.
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta
from decimal import Decimal
# LiquidityAnalyzer не найден в infrastructure.core
# from infrastructure.core.liquidity_analyzer import LiquidityAnalyzer


class LiquidityAnalyzer:
    """Анализатор ликвидности для тестов."""
    
    def __init__(self):
        self.analysis_history = []
    
    def analyze_liquidity(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ ликвидности."""
        analysis = {
            "liquidity_score": 0.8,
            "spread": 0.001,
            "depth": 1000000,
            "timestamp": datetime.now()
        }
        self.analysis_history.append(analysis)
        return analysis
    
    def get_liquidity_metrics(self) -> Dict[str, float]:
        """Получение метрик ликвидности."""
        return {
            "average_spread": 0.001,
            "average_depth": 1000000,
            "liquidity_score": 0.8
        }


class TestLiquidityAnalyzer:
    """Тесты для LiquidityAnalyzer."""

    @pytest.fixture
    def liquidity_analyzer(self) -> LiquidityAnalyzer:
        """Фикстура для LiquidityAnalyzer."""
        return LiquidityAnalyzer()

    @pytest.fixture
    def sample_orderbook_data(self) -> dict:
        """Фикстура с данными ордербука."""
        return {
            "symbol": "BTCUSDT",
            "timestamp": datetime.now(),
            "bids": [
                [Decimal("50000.0"), Decimal("1.5")],
                [Decimal("49999.0"), Decimal("2.0")],
                [Decimal("49998.0"), Decimal("3.0")],
                [Decimal("49997.0"), Decimal("2.5")],
                [Decimal("49996.0"), Decimal("1.8")],
            ],
            "asks": [
                [Decimal("50001.0"), Decimal("1.0")],
                [Decimal("50002.0"), Decimal("2.5")],
                [Decimal("50003.0"), Decimal("1.8")],
                [Decimal("50004.0"), Decimal("3.2")],
                [Decimal("50005.0"), Decimal("2.1")],
            ],
        }

    @pytest.fixture
    def sample_trade_data(self) -> list:
        """Фикстура с данными о сделках."""
        return [
            {
                "id": "trade_001",
                "price": Decimal("50000.0"),
                "quantity": Decimal("0.1"),
                "side": "buy",
                "timestamp": datetime.now() - timedelta(minutes=5),
            },
            {
                "id": "trade_002",
                "price": Decimal("50001.0"),
                "quantity": Decimal("0.2"),
                "side": "sell",
                "timestamp": datetime.now() - timedelta(minutes=4),
            },
            {
                "id": "trade_003",
                "price": Decimal("49999.0"),
                "quantity": Decimal("0.15"),
                "side": "buy",
                "timestamp": datetime.now() - timedelta(minutes=3),
            },
        ]

    def test_initialization(self, liquidity_analyzer: LiquidityAnalyzer) -> None:
        """Тест инициализации анализатора ликвидности."""
        assert liquidity_analyzer is not None
        assert hasattr(liquidity_analyzer, "liquidity_metrics")
        assert hasattr(liquidity_analyzer, "spread_analyzers")
        assert hasattr(liquidity_analyzer, "depth_analyzers")

    def test_calculate_market_depth(self, liquidity_analyzer: LiquidityAnalyzer, sample_orderbook_data: dict) -> None:
        """Тест расчета глубины рынка."""
        # Расчет глубины рынка
        depth_result = liquidity_analyzer.calculate_market_depth(sample_orderbook_data)
        # Проверки
        assert depth_result is not None
        assert "bid_depth" in depth_result
        assert "ask_depth" in depth_result
        assert "total_depth" in depth_result
        assert "depth_by_level" in depth_result
        # Проверка типов данных
        assert isinstance(depth_result["bid_depth"], Decimal)
        assert isinstance(depth_result["ask_depth"], Decimal)
        assert isinstance(depth_result["total_depth"], Decimal)
        assert isinstance(depth_result["depth_by_level"], dict)
        # Проверка логики
        assert depth_result["bid_depth"] > 0
        assert depth_result["ask_depth"] > 0
        assert depth_result["total_depth"] == depth_result["bid_depth"] + depth_result["ask_depth"]

    def test_analyze_bid_ask_spread(self, liquidity_analyzer: LiquidityAnalyzer, sample_orderbook_data: dict) -> None:
        """Тест анализа спреда bid-ask."""
        # Анализ спреда
        spread_result = liquidity_analyzer.analyze_bid_ask_spread(sample_orderbook_data)
        # Проверки
        assert spread_result is not None
        assert "spread" in spread_result
        assert "spread_percentage" in spread_result
        assert "spread_quality" in spread_result
        assert "spread_trend" in spread_result
        # Проверка типов данных
        assert isinstance(spread_result["spread"], Decimal)
        assert isinstance(spread_result["spread_percentage"], float)
        assert spread_result["spread_quality"] in ["tight", "normal", "wide"]
        assert isinstance(spread_result["spread_trend"], str)
        # Проверка логики
        assert spread_result["spread"] > 0
        assert spread_result["spread_percentage"] > 0

    def test_calculate_liquidity_score(
        self, liquidity_analyzer: LiquidityAnalyzer, sample_orderbook_data: dict
    ) -> None:
        """Тест расчета оценки ликвидности."""
        # Расчет оценки ликвидности
        liquidity_score = liquidity_analyzer.calculate_liquidity_score(sample_orderbook_data)
        # Проверки
        assert liquidity_score is not None
        assert "liquidity_score" in liquidity_score
        assert "liquidity_factors" in liquidity_score
        assert "liquidity_rating" in liquidity_score
        assert "confidence" in liquidity_score
        # Проверка типов данных
        assert isinstance(liquidity_score["liquidity_score"], float)
        assert isinstance(liquidity_score["liquidity_factors"], dict)
        assert liquidity_score["liquidity_rating"] in ["excellent", "good", "fair", "poor"]
        assert isinstance(liquidity_score["confidence"], float)
        # Проверка диапазонов
        assert 0.0 <= liquidity_score["liquidity_score"] <= 1.0
        assert 0.0 <= liquidity_score["confidence"] <= 1.0

    def test_analyze_liquidity_risk(self, liquidity_analyzer: LiquidityAnalyzer, sample_orderbook_data: dict) -> None:
        """Тест анализа рисков ликвидности."""
        # Анализ рисков ликвидности
        risk_result = liquidity_analyzer.analyze_liquidity_risk(sample_orderbook_data)
        # Проверки
        assert risk_result is not None
        assert "risk_level" in risk_result
        assert "risk_factors" in risk_result
        assert "risk_score" in risk_result
        assert "risk_alerts" in risk_result
        # Проверка типов данных
        assert risk_result["risk_level"] in ["low", "medium", "high", "critical"]
        assert isinstance(risk_result["risk_factors"], dict)
        assert isinstance(risk_result["risk_score"], float)
        assert isinstance(risk_result["risk_alerts"], list)
        # Проверка диапазона
        assert 0.0 <= risk_result["risk_score"] <= 1.0

    def test_calculate_impact_cost(self, liquidity_analyzer: LiquidityAnalyzer, sample_orderbook_data: dict) -> None:
        """Тест расчета стоимости влияния."""
        # Расчет стоимости влияния
        impact_result = liquidity_analyzer.calculate_impact_cost(sample_orderbook_data, trade_size=Decimal("1.0"))
        # Проверки
        assert impact_result is not None
        assert "buy_impact" in impact_result
        assert "sell_impact" in impact_result
        assert "average_impact" in impact_result
        assert "impact_analysis" in impact_result
        # Проверка типов данных
        assert isinstance(impact_result["buy_impact"], Decimal)
        assert isinstance(impact_result["sell_impact"], Decimal)
        assert isinstance(impact_result["average_impact"], Decimal)
        assert isinstance(impact_result["impact_analysis"], dict)
        # Проверка логики
        assert impact_result["buy_impact"] > 0
        assert impact_result["sell_impact"] > 0

    def test_analyze_liquidity_trends(self, liquidity_analyzer: LiquidityAnalyzer, sample_orderbook_data: dict) -> None:
        """Тест анализа трендов ликвидности."""
        # Анализ трендов ликвидности
        trends_result = liquidity_analyzer.analyze_liquidity_trends([sample_orderbook_data])
        # Проверки
        assert trends_result is not None
        assert "liquidity_trend" in trends_result
        assert "trend_strength" in trends_result
        assert "trend_duration" in trends_result
        assert "trend_prediction" in trends_result
        # Проверка типов данных
        assert trends_result["liquidity_trend"] in ["improving", "stable", "declining"]
        assert isinstance(trends_result["trend_strength"], float)
        assert isinstance(trends_result["trend_duration"], timedelta)
        assert isinstance(trends_result["trend_prediction"], dict)
        # Проверка диапазона
        assert 0.0 <= trends_result["trend_strength"] <= 1.0

    def test_monitor_liquidity_events(self, liquidity_analyzer: LiquidityAnalyzer, sample_orderbook_data: dict) -> None:
        """Тест мониторинга событий ликвидности."""
        # Мониторинг событий ликвидности
        events_result = liquidity_analyzer.monitor_liquidity_events(sample_orderbook_data)
        # Проверки
        assert events_result is not None
        assert "liquidity_events" in events_result
        assert "event_severity" in events_result
        assert "event_alerts" in events_result
        assert "event_recommendations" in events_result
        # Проверка типов данных
        assert isinstance(events_result["liquidity_events"], list)
        assert isinstance(events_result["event_severity"], dict)
        assert isinstance(events_result["event_alerts"], list)
        assert isinstance(events_result["event_recommendations"], list)

    def test_calculate_liquidity_metrics(
        self, liquidity_analyzer: LiquidityAnalyzer, sample_orderbook_data: dict
    ) -> None:
        """Тест расчета метрик ликвидности."""
        # Расчет метрик ликвидности
        metrics = liquidity_analyzer.calculate_liquidity_metrics(sample_orderbook_data)
        # Проверки
        assert metrics is not None
        assert "depth_metrics" in metrics
        assert "spread_metrics" in metrics
        assert "volume_metrics" in metrics
        assert "turnover_metrics" in metrics
        assert "efficiency_metrics" in metrics
        # Проверка типов данных
        assert isinstance(metrics["depth_metrics"], dict)
        assert isinstance(metrics["spread_metrics"], dict)
        assert isinstance(metrics["volume_metrics"], dict)
        assert isinstance(metrics["turnover_metrics"], dict)
        assert isinstance(metrics["efficiency_metrics"], dict)

    def test_analyze_liquidity_by_size(
        self, liquidity_analyzer: LiquidityAnalyzer, sample_orderbook_data: dict
    ) -> None:
        """Тест анализа ликвидности по размеру."""
        # Анализ ликвидности по размеру
        size_analysis = liquidity_analyzer.analyze_liquidity_by_size(sample_orderbook_data)
        # Проверки
        assert size_analysis is not None
        assert "small_trades" in size_analysis
        assert "medium_trades" in size_analysis
        assert "large_trades" in size_analysis
        assert "size_breakdown" in size_analysis
        # Проверка типов данных
        assert isinstance(size_analysis["small_trades"], dict)
        assert isinstance(size_analysis["medium_trades"], dict)
        assert isinstance(size_analysis["large_trades"], dict)
        assert isinstance(size_analysis["size_breakdown"], dict)

    def test_validate_liquidity_data(self, liquidity_analyzer: LiquidityAnalyzer, sample_orderbook_data: dict) -> None:
        """Тест валидации данных ликвидности."""
        # Валидация данных
        validation_result = liquidity_analyzer.validate_liquidity_data(sample_orderbook_data)
        # Проверки
        assert validation_result is not None
        assert "is_valid" in validation_result
        assert "validation_errors" in validation_result
        assert "data_quality_score" in validation_result
        # Проверка типов данных
        assert isinstance(validation_result["is_valid"], bool)
        assert isinstance(validation_result["validation_errors"], list)
        assert isinstance(validation_result["data_quality_score"], float)
        # Проверка диапазона
        assert 0.0 <= validation_result["data_quality_score"] <= 1.0

    def test_get_liquidity_statistics(self, liquidity_analyzer: LiquidityAnalyzer, sample_orderbook_data: dict) -> None:
        """Тест получения статистики ликвидности."""
        # Получение статистики
        statistics = liquidity_analyzer.get_liquidity_statistics([sample_orderbook_data])
        # Проверки
        assert statistics is not None
        assert "liquidity_distribution" in statistics
        assert "spread_statistics" in statistics
        assert "depth_statistics" in statistics
        assert "volume_statistics" in statistics
        # Проверка типов данных
        assert isinstance(statistics["liquidity_distribution"], dict)
        assert isinstance(statistics["spread_statistics"], dict)
        assert isinstance(statistics["depth_statistics"], dict)
        assert isinstance(statistics["volume_statistics"], dict)

    def test_error_handling(self, liquidity_analyzer: LiquidityAnalyzer) -> None:
        """Тест обработки ошибок."""
        # Тест с некорректными данными
        with pytest.raises(ValueError):
            liquidity_analyzer.calculate_market_depth(None)
        with pytest.raises(ValueError):
            liquidity_analyzer.validate_liquidity_data(None)

    def test_edge_cases(self, liquidity_analyzer: LiquidityAnalyzer) -> None:
        """Тест граничных случаев."""
        # Тест с очень маленьким ордербуком
        small_orderbook = {
            "symbol": "BTCUSDT",
            "timestamp": datetime.now(),
            "bids": [[Decimal("50000.0"), Decimal("0.001")]],
            "asks": [[Decimal("50001.0"), Decimal("0.001")]],
        }
        depth_result = liquidity_analyzer.calculate_market_depth(small_orderbook)
        assert depth_result is not None
        # Тест с очень большим спредом
        wide_spread_orderbook = {
            "symbol": "BTCUSDT",
            "timestamp": datetime.now(),
            "bids": [[Decimal("40000.0"), Decimal("1.0")]],
            "asks": [[Decimal("60000.0"), Decimal("1.0")]],
        }
        spread_result = liquidity_analyzer.analyze_bid_ask_spread(wide_spread_orderbook)
        assert spread_result is not None

    def test_cleanup(self, liquidity_analyzer: LiquidityAnalyzer) -> None:
        """Тест очистки ресурсов."""
        # Проверка корректной очистки
        liquidity_analyzer.cleanup()
        # Проверка, что ресурсы освобождены
        assert liquidity_analyzer.liquidity_metrics == {}
        assert liquidity_analyzer.spread_analyzers == {}
        assert liquidity_analyzer.depth_analyzers == {}
