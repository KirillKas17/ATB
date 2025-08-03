from datetime import datetime, timedelta
import pytest
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from ml.live_adaptation import (AdaptationConfig, LiveAdaptation,
                                LiveAdaptationModel)
@pytest.fixture
def sample_data() -> Any:
    """Фикстура с тестовыми данными"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="H")
    data = pd.DataFrame(
        {
            "open": np.random.normal(100, 1, 100),
            "high": np.random.normal(101, 1, 100),
            "low": np.random.normal(99, 1, 100),
            "close": np.random.normal(100, 1, 100),
            "volume": np.random.normal(1000, 100, 100),
        },
        index=dates,
    )
    data["high"] = data[["open", "close"]].max(axis=1) + abs(
        np.random.normal(0, 0.5, 100)
    )
    data["low"] = data[["open", "close"]].min(axis=1) - abs(
        np.random.normal(0, 0.5, 100)
    )
    return data
@pytest.fixture
def sample_trades() -> Any:
    """Фикстура с тестовыми сделками"""
    return [
        {
            "symbol": "BTC/USD",
            "side": "buy",
            "price": 100.0,
            "size": 1.0,
            "timestamp": datetime.now() - timedelta(hours=i),
        }
        for i in range(10)
    ]
@pytest.fixture
def adaptation() -> Any:
    """Фикстура с экземпляром LiveAdaptation"""
    return LiveAdaptation()
@pytest.fixture
def adaptation_model() -> Any:
    """Фикстура с экземпляром LiveAdaptationModel"""
    config = AdaptationConfig(
        min_samples=10,
        max_samples=100,
        update_interval=1,
        retrain_threshold=0.1,
        confidence_threshold=0.7,
    )
    return LiveAdaptationModel(config)
@pytest.fixture
def mock_market_data() -> Any:
    """Фикстура с тестовыми рыночными данными"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="h")
    data = pd.DataFrame(
        {
            "open": np.random.normal(100, 1, 100),
            "high": np.random.normal(101, 1, 100),
            "low": np.random.normal(99, 1, 100),
            "close": np.random.normal(100, 1, 100),
            "volume": np.random.normal(1000, 100, 100),
        },
        index=dates,
    )
    return data
class TestLiveAdaptation:
    """Тесты для LiveAdaptation"""
    def test_initialization(self, adaptation) -> None:
        """Тест инициализации"""
        assert adaptation.config is not None
        assert adaptation.adaptation_state == {}
        assert adaptation._last_update is None
    def test_update_state(self, adaptation, sample_data, sample_trades) -> None:
        """Тест обновления состояния"""
        state = adaptation.update_state(sample_data, sample_trades)
        assert isinstance(state, dict)
        assert "volatility" in state
        assert "trend" in state
        assert "market_regime" in state
        assert "position_size_factor" in state
        assert "stop_loss_factor" in state
        assert "take_profit_factor" in state
        assert "commission_factor" in state
        assert "slippage_factor" in state
        assert "last_update" in state
        assert isinstance(state["volatility"], float)
        assert isinstance(state["trend"], float)
        assert isinstance(state["market_regime"], str)
        assert isinstance(state["position_size_factor"], float)
        assert isinstance(state["stop_loss_factor"], float)
        assert isinstance(state["take_profit_factor"], float)
        assert isinstance(state["commission_factor"], float)
        assert isinstance(state["slippage_factor"], float)
        assert isinstance(state["last_update"], datetime)
    def test_calculate_trend(self, adaptation, sample_data) -> None:
        """Тест расчета тренда"""
        trend = adaptation._calculate_trend(sample_data)
        assert isinstance(trend, float)
    def test_detect_regime(self, adaptation, sample_data) -> None:
        """Тест определения режима рынка"""
        regime = adaptation._detect_regime(sample_data)
        assert isinstance(regime, str)
        assert regime in ["trend", "volatile", "sideways", "normal"]
    def test_calculate_position_size_factor(self, adaptation) -> None:
        """Тест расчета фактора размера позиции"""
        factor = adaptation._calculate_position_size_factor(0.01, 0.2)
        assert isinstance(factor, float)
        assert 0.5 <= factor <= 1.5
    def test_calculate_stop_loss_factor(self, adaptation) -> None:
        """Тест расчета фактора стоп-лосса"""
        factor = adaptation._calculate_stop_loss_factor(0.01)
        assert isinstance(factor, float)
        assert 0.8 <= factor <= 1.5
    def test_calculate_take_profit_factor(self, adaptation) -> None:
        """Тест расчета фактора тейк-профита"""
        factor = adaptation._calculate_take_profit_factor(0.2)
        assert isinstance(factor, float)
        assert 0.8 <= factor <= 1.5
    def test_calculate_commission_factor(self, adaptation) -> None:
        """Тест расчета фактора комиссии"""
        factor = adaptation._calculate_commission_factor(0.01)
        assert isinstance(factor, float)
        assert 1.0 <= factor <= 1.2
    def test_calculate_slippage_factor(self, adaptation) -> None:
        """Тест расчета фактора проскальзывания"""
        factor = adaptation._calculate_slippage_factor(0.01)
        assert isinstance(factor, float)
        assert 0.8 <= factor <= 1.5
class TestLiveAdaptationModel:
    """Тесты для LiveAdaptationModel"""
    def test_initialization(self, adaptation_model) -> None:
        """Тест инициализации"""
        assert adaptation_model.config is not None
        assert isinstance(adaptation_model.config, AdaptationConfig)
        assert adaptation_model.data_buffer.empty
        assert adaptation_model.metrics_history == []
        assert adaptation_model.models == {}
        assert adaptation_model.scalers == {}
        assert adaptation_model.metrics == {}
    def test_update(self, adaptation_model, sample_data) -> None:
        """Тест обновления модели"""
        model_id = "test_model"
        model = adaptation_model.models.get(model_id)
        adaptation_model.update(sample_data, model_id, model)
        assert not adaptation_model.data_buffer.empty
        assert len(adaptation_model.metrics_history) > 0
        assert model_id in adaptation_model.metrics
    def test_predict(self, adaptation_model, sample_data) -> None:
        """Тест предсказания"""
        model_id = "test_model"
        model = adaptation_model.models.get(model_id)
        # Сначала обновляем модель
        adaptation_model.update(sample_data, model_id, model)
        # Затем делаем предсказание
        predictions, confidence = adaptation_model.predict(sample_data, model_id)
        assert isinstance(predictions, np.ndarray)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
    def test_get_metrics(self, adaptation_model, sample_data) -> None:
        """Тест получения метрик"""
        model_id = "test_model"
        model = adaptation_model.models.get(model_id)
        # Сначала обновляем модель
        adaptation_model.update(sample_data, model_id, model)
        # Получаем метрики
        metrics = adaptation_model.get_metrics(model_id)
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "confidence" in metrics
        assert "drift_score" in metrics
        assert "last_update" in metrics
        assert "samples_count" in metrics
        assert "retrain_count" in metrics
        assert "error_count" in metrics
    def test_get_history(self, adaptation_model, sample_data) -> None:
        """Тест получения истории"""
        model_id = "test_model"
        model = adaptation_model.models.get(model_id)
        # Сначала обновляем модель
        adaptation_model.update(sample_data, model_id, model)
        # Получаем историю
        history = adaptation_model.get_history(model_id)
        assert isinstance(history, list)
        assert len(history) > 0
        assert all(isinstance(item, dict) for item in history)
    def test_reset(self, adaptation_model, sample_data) -> None:
        """Тест сброса состояния"""
        model_id = "test_model"
        model = adaptation_model.models.get(model_id)
        # Сначала обновляем модель
        adaptation_model.update(sample_data, model_id, model)
        # Затем сбрасываем состояние
        adaptation_model.reset()
        assert adaptation_model.data_buffer.empty
        assert adaptation_model.metrics_history == []
        assert adaptation_model.models == {}
        assert adaptation_model.scalers == {}
        assert adaptation_model.metrics == {}
        assert len(adaptation_model._prediction_cache) == 0
        assert len(adaptation_model._feature_cache) == 0
