"""
Unit тесты для WhaleTracker.
Тестирует отслеживание крупных игроков рынка, включая мониторинг
транзакций, анализ поведения китов и прогнозирование их действий.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta
from decimal import Decimal
from infrastructure.core.whale_tracker import WhaleTracker
class TestWhaleTracker:
    """Тесты для WhaleTracker."""
    @pytest.fixture
    def whale_tracker(self) -> WhaleTracker:
        """Фикстура для WhaleTracker."""
        return WhaleTracker()
    @pytest.fixture
    def sample_whale_data(self) -> list:
        """Фикстура с тестовыми данными о китах."""
        return [
            {
                "id": "whale_001",
                "address": "0x1234567890abcdef",
                "balance": Decimal("1000000.0"),
                "transactions": [
                    {
                        "tx_hash": "0xabc123",
                        "type": "buy",
                        "amount": Decimal("100000.0"),
                        "price": Decimal("50000.0"),
                        "timestamp": datetime.now() - timedelta(hours=2)
                    }
                ],
                "last_activity": datetime.now() - timedelta(hours=1),
                "whale_score": 0.9,
                "category": "institutional"
            },
            {
                "id": "whale_002",
                "address": "0xfedcba0987654321",
                "balance": Decimal("500000.0"),
                "transactions": [
                    {
                        "tx_hash": "0xdef456",
                        "type": "sell",
                        "amount": Decimal("50000.0"),
                        "price": Decimal("52000.0"),
                        "timestamp": datetime.now() - timedelta(hours=3)
                    }
                ],
                "last_activity": datetime.now() - timedelta(hours=2),
                "whale_score": 0.7,
                "category": "exchange"
            }
        ]
    @pytest.fixture
    def sample_transaction_data(self) -> list:
        """Фикстура с данными о транзакциях."""
        return [
            {
                "tx_hash": "0x123abc",
                "from_address": "0x1111111111111111",
                "to_address": "0x2222222222222222",
                "amount": Decimal("100000.0"),
                "token": "BTC",
                "timestamp": datetime.now() - timedelta(hours=1),
                "gas_price": Decimal("50.0"),
                "block_number": 12345678
            },
            {
                "tx_hash": "0x456def",
                "from_address": "0x3333333333333333",
                "to_address": "0x4444444444444444",
                "amount": Decimal("200000.0"),
                "token": "ETH",
                "timestamp": datetime.now() - timedelta(hours=2),
                "gas_price": Decimal("45.0"),
                "block_number": 12345677
            }
        ]
    def test_initialization(self, whale_tracker: WhaleTracker) -> None:
        """Тест инициализации трекера китов."""
        assert whale_tracker is not None
        assert hasattr(whale_tracker, 'whale_database')
        assert hasattr(whale_tracker, 'transaction_monitors')
        assert hasattr(whale_tracker, 'behavior_analyzers')
    def test_identify_whales(self, whale_tracker: WhaleTracker, sample_transaction_data: list) -> None:
        """Тест идентификации китов."""
        # Идентификация китов
        identification_result = whale_tracker.identify_whales(sample_transaction_data)
        # Проверки
        assert identification_result is not None
        assert "identified_whales" in identification_result
        assert "whale_criteria" in identification_result
        assert "identification_confidence" in identification_result
        assert "whale_categories" in identification_result
        # Проверка типов данных
        assert isinstance(identification_result["identified_whales"], list)
        assert isinstance(identification_result["whale_criteria"], dict)
        assert isinstance(identification_result["identification_confidence"], float)
        assert isinstance(identification_result["whale_categories"], dict)
        # Проверка диапазона
        assert 0.0 <= identification_result["identification_confidence"] <= 1.0
    def test_track_whale_transactions(self, whale_tracker: WhaleTracker, sample_whale_data: list) -> None:
        """Тест отслеживания транзакций китов."""
        # Отслеживание транзакций
        tracking_result = whale_tracker.track_whale_transactions(sample_whale_data[0]["address"])
        # Проверки
        assert tracking_result is not None
        assert "whale_address" in tracking_result
        assert "transactions" in tracking_result
        assert "transaction_patterns" in tracking_result
        assert "tracking_period" in tracking_result
        # Проверка типов данных
        assert isinstance(tracking_result["whale_address"], str)
        assert isinstance(tracking_result["transactions"], list)
        assert isinstance(tracking_result["transaction_patterns"], dict)
        assert isinstance(tracking_result["tracking_period"], timedelta)
    def test_analyze_whale_behavior(self, whale_tracker: WhaleTracker, sample_whale_data: list) -> None:
        """Тест анализа поведения китов."""
        # Анализ поведения
        behavior_result = whale_tracker.analyze_whale_behavior(sample_whale_data[0])
        # Проверки
        assert behavior_result is not None
        assert "behavior_patterns" in behavior_result
        assert "trading_style" in behavior_result
        assert "risk_profile" in behavior_result
        assert "predictability_score" in behavior_result
        # Проверка типов данных
        assert isinstance(behavior_result["behavior_patterns"], dict)
        assert isinstance(behavior_result["trading_style"], str)
        assert isinstance(behavior_result["risk_profile"], str)
        assert isinstance(behavior_result["predictability_score"], float)
        # Проверка диапазона
        assert 0.0 <= behavior_result["predictability_score"] <= 1.0
    def test_detect_whale_movements(self, whale_tracker: WhaleTracker, sample_transaction_data: list) -> None:
        """Тест обнаружения движений китов."""
        # Обнаружение движений
        movements_result = whale_tracker.detect_whale_movements(sample_transaction_data)
        # Проверки
        assert movements_result is not None
        assert "whale_movements" in movements_result
        assert "movement_significance" in movements_result
        assert "market_impact" in movements_result
        assert "movement_alerts" in movements_result
        # Проверка типов данных
        assert isinstance(movements_result["whale_movements"], list)
        assert isinstance(movements_result["movement_significance"], dict)
        assert isinstance(movements_result["market_impact"], dict)
        assert isinstance(movements_result["movement_alerts"], list)
    def test_predict_whale_actions(self, whale_tracker: WhaleTracker, sample_whale_data: list) -> None:
        """Тест прогнозирования действий китов."""
        # Прогнозирование действий
        prediction_result = whale_tracker.predict_whale_actions(sample_whale_data[0])
        # Проверки
        assert prediction_result is not None
        assert "predicted_actions" in prediction_result
        assert "prediction_confidence" in prediction_result
        assert "time_horizon" in prediction_result
        assert "action_probabilities" in prediction_result
        # Проверка типов данных
        assert isinstance(prediction_result["predicted_actions"], list)
        assert isinstance(prediction_result["prediction_confidence"], float)
        assert isinstance(prediction_result["time_horizon"], timedelta)
        assert isinstance(prediction_result["action_probabilities"], dict)
        # Проверка диапазона
        assert 0.0 <= prediction_result["prediction_confidence"] <= 1.0
    def test_calculate_whale_metrics(self, whale_tracker: WhaleTracker, sample_whale_data: list) -> None:
        """Тест расчета метрик китов."""
        # Расчет метрик
        metrics = whale_tracker.calculate_whale_metrics(sample_whale_data)
        # Проверки
        assert metrics is not None
        assert "total_whales" in metrics
        assert "total_balance" in metrics
        assert "avg_whale_score" in metrics
        assert "whale_distribution" in metrics
        assert "activity_metrics" in metrics
        # Проверка типов данных
        assert isinstance(metrics["total_whales"], int)
        assert isinstance(metrics["total_balance"], Decimal)
        assert isinstance(metrics["avg_whale_score"], float)
        assert isinstance(metrics["whale_distribution"], dict)
        assert isinstance(metrics["activity_metrics"], dict)
        # Проверка логики
        assert metrics["total_whales"] == len(sample_whale_data)
    def test_monitor_whale_clusters(self, whale_tracker: WhaleTracker, sample_whale_data: list) -> None:
        """Тест мониторинга кластеров китов."""
        # Мониторинг кластеров
        clusters_result = whale_tracker.monitor_whale_clusters(sample_whale_data)
        # Проверки
        assert clusters_result is not None
        assert "whale_clusters" in clusters_result
        assert "cluster_behavior" in clusters_result
        assert "cluster_significance" in clusters_result
        assert "cluster_alerts" in clusters_result
        # Проверка типов данных
        assert isinstance(clusters_result["whale_clusters"], list)
        assert isinstance(clusters_result["cluster_behavior"], dict)
        assert isinstance(clusters_result["cluster_significance"], dict)
        assert isinstance(clusters_result["cluster_alerts"], list)
    def test_analyze_whale_sentiment(self, whale_tracker: WhaleTracker, sample_whale_data: list) -> None:
        """Тест анализа настроений китов."""
        # Анализ настроений
        sentiment_result = whale_tracker.analyze_whale_sentiment(sample_whale_data)
        # Проверки
        assert sentiment_result is not None
        assert "overall_sentiment" in sentiment_result
        assert "sentiment_by_category" in sentiment_result
        assert "sentiment_trends" in sentiment_result
        assert "sentiment_confidence" in sentiment_result
        # Проверка типов данных
        assert isinstance(sentiment_result["overall_sentiment"], float)
        assert isinstance(sentiment_result["sentiment_by_category"], dict)
        assert isinstance(sentiment_result["sentiment_trends"], dict)
        assert isinstance(sentiment_result["sentiment_confidence"], float)
        # Проверка диапазонов
        assert -1.0 <= sentiment_result["overall_sentiment"] <= 1.0
        assert 0.0 <= sentiment_result["sentiment_confidence"] <= 1.0
    def test_generate_whale_alerts(self, whale_tracker: WhaleTracker, sample_whale_data: list) -> None:
        """Тест генерации алертов о китах."""
        # Генерация алертов
        alerts_result = whale_tracker.generate_whale_alerts(sample_whale_data)
        # Проверки
        assert alerts_result is not None
        assert "whale_alerts" in alerts_result
        assert "alert_severity" in alerts_result
        assert "alert_recommendations" in alerts_result
        assert "alert_time" in alerts_result
        # Проверка типов данных
        assert isinstance(alerts_result["whale_alerts"], list)
        assert isinstance(alerts_result["alert_severity"], dict)
        assert isinstance(alerts_result["alert_recommendations"], list)
        assert isinstance(alerts_result["alert_time"], datetime)
    def test_validate_whale_data(self, whale_tracker: WhaleTracker, sample_whale_data: list) -> None:
        """Тест валидации данных о китах."""
        # Валидация данных
        validation_result = whale_tracker.validate_whale_data(sample_whale_data[0])
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
    def test_get_whale_statistics(self, whale_tracker: WhaleTracker, sample_whale_data: list) -> None:
        """Тест получения статистики китов."""
        # Получение статистики
        statistics = whale_tracker.get_whale_statistics(sample_whale_data)
        # Проверки
        assert statistics is not None
        assert "whale_distribution" in statistics
        assert "balance_distribution" in statistics
        assert "activity_statistics" in statistics
        assert "category_statistics" in statistics
        # Проверка типов данных
        assert isinstance(statistics["whale_distribution"], dict)
        assert isinstance(statistics["balance_distribution"], dict)
        assert isinstance(statistics["activity_statistics"], dict)
        assert isinstance(statistics["category_statistics"], dict)
    def test_error_handling(self, whale_tracker: WhaleTracker) -> None:
        """Тест обработки ошибок."""
        # Тест с некорректными данными
        with pytest.raises(ValueError):
            whale_tracker.analyze_whale_behavior(None)
        with pytest.raises(ValueError):
            whale_tracker.validate_whale_data(None)
    def test_edge_cases(self, whale_tracker: WhaleTracker) -> None:
        """Тест граничных случаев."""
        # Тест с очень маленькими балансами
        small_whale = {
            "id": "small_whale",
            "address": "0x1111111111111111",
            "balance": Decimal("100.0"),
            "transactions": [],
            "whale_score": 0.1
        }
        behavior_result = whale_tracker.analyze_whale_behavior(small_whale)
        assert behavior_result is not None
        # Тест с очень большими балансами
        large_whale = {
            "id": "large_whale",
            "address": "0x2222222222222222",
            "balance": Decimal("1000000000.0"),
            "transactions": [],
            "whale_score": 1.0
        }
        behavior_result = whale_tracker.analyze_whale_behavior(large_whale)
        assert behavior_result is not None
    def test_cleanup(self, whale_tracker: WhaleTracker) -> None:
        """Тест очистки ресурсов."""
        # Проверка корректной очистки
        whale_tracker.cleanup()
        # Проверка, что ресурсы освобождены
        assert whale_tracker.whale_database == {}
        assert whale_tracker.transaction_monitors == {}
        assert whale_tracker.behavior_analyzers == {} 
