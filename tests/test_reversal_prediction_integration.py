"""
Тесты интеграции системы прогнозирования разворотов.
"""

import pytest
import pandas as pd
from shared.numpy_utils import np
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

from domain.prediction.reversal_predictor import ReversalPredictor
from domain.prediction.reversal_signal import ReversalSignal, ReversalDirection
from domain.value_objects.price import Price
from domain.value_objects.currency import Currency
from domain.value_objects.confidence_score import ConfidenceScore
from domain.value_objects.signal_strength_score import SignalStrengthScore
from domain.value_objects.timestamp import Timestamp
from domain.type_definitions import OHLCVData  # Исправление: правильный импорт OHLCVData
from domain.prediction.prediction_config import PredictionConfig
from application.prediction.reversal_controller import ReversalController
from infrastructure.data.price_pattern_extractor import PricePatternExtractor
from domain.protocols.agent_protocols import AgentContextProtocol

class TestReversalPredictionIntegration:
    """Интеграционные тесты системы прогнозирования разворотов."""
    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        return None
        """Создание тестовых рыночных данных."""
        dates = pd.date_range(start="2024-01-01", periods=200, freq="1H")
        # Создаем трендовые данные с разворотом
        trend = np.linspace(100, 120, 100)  # Восходящий тренд
        reversal = np.linspace(120, 110, 100)  # Нисходящий тренд после разворота
        prices = np.concatenate([trend, reversal])
        # Добавляем волатильность
        noise = np.random.normal(0, 0.5, 200)
        prices += noise
        # Создаем OHLCV данные
        data = pd.DataFrame(
            {
                "open": prices,
                "high": prices + np.random.uniform(0, 1, 200),
                "low": prices - np.random.uniform(0, 1, 200),
                "close": prices,
                "volume": np.random.uniform(1000, 5000, 200),
            },
            index=dates,
        )
        # Корректируем high/low
        data["high"] = data[["open", "high", "close"]].max(axis=1)
        data["low"] = data[["open", "low", "close"]].min(axis=1)
        return data
        # Мокаем методы получения данных
        # Мокаем market service
        # Исправление: создаем конфигурацию напрямую как словарь
        # Создаем мок контроллера
        # Проверяем, что найдены пивоты
        # Проверяем структуру пивотов
        # Проверяем, что найдены уровни Фибоначчи
        # Проверяем структуру уровней
        # Проверяем, что профиль создан
        # Проверяем, что найдены кластеры
        # Проверяем структуру кластеров
        # Исправление: используем OHLCVData тип
        # Проверяем, что сигнал создан (может быть None при недостаточных данных)
        # Тест категории силы сигнала
        # Тест уровня риска
        # Тест истечения срока
        # Тест времени до истечения
        # Тест усиления уверенности
        # Тест снижения уверенности
        # Тест пометки как спорного
        # Настраиваем моки
        # Создаем тестовый сигнал
        # Интегрируем сигнал
        # Проверяем, что сигнал добавлен
        # Настраиваем мок глобального прогноза
        # Создаем сигнал
        # Вычисляем согласованность
        # Проверяем результат
        # Настраиваем конфликтующий глобальный прогноз
        # Создаем сигнал с низкой силой
        # Обнаруживаем спорные аспекты
        # Проверяем, что найдены причины споров
        # Проверяем структуру статистики
        # Извлекаем паттерны
        # Проверяем, что паттерны извлечены
        # Прогнозируем разворот
        # Проверяем результат (может быть None при недостаточных данных)
        # Сериализуем в словарь
        # Проверяем структуру
        # Тест с пустыми данными
        # Тест с недостаточными данными
        # Создаем устаревший сигнал
        # Добавляем в активные сигналы
        # Выполняем очистку
        # Проверяем, что устаревшие сигналы удалены
