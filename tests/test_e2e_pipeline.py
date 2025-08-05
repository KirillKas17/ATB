import json
import pytest
import pandas as pd
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from core.market_state import MarketStateManager
from core.ml_integration import MLIntegration
    @pytest.fixture
def sample_candles() -> Any:
    with open("tests/data/sample_candles.json", "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)
    @pytest.fixture
def market_state_manager() -> Any:
    config = {"atr_period": 14, "buffer_size": 10, "min_candles": 5}
    return MarketStateManager(config)
    @pytest.fixture
def ml_integration() -> Any:
    """Фикстура с ML интеграцией"""
    return MLIntegration(
        config={
            "model_path": "models/",
            "data_path": "data/",
            "features": ["close", "volume", "rsi", "macd"],
            "target": "returns",
            "train_size": 0.8,
            "validation_size": 0.1,
            "test_size": 0.1,
            "random_state": 42,
        }
    )
    @pytest.fixture
def mock_dashboard() -> Any:
    # Мокаем API/dashboard
    class Dashboard:
        def __init__(self) -> Any:
            self.last_update = None
        def update(self, snapshot) -> Any:
            self.last_update = snapshot
    return Dashboard()
    def test_e2e_trading_pipeline(
    sample_candles, market_state_manager, ml_integration, mock_dashboard
) -> None:
    pair = "BTCUSDT"
    # 1. Прогоняем свечи через MarketStateManager
    snapshot = None
    for _, candle in sample_candles.iterrows():
        snap = market_state_manager.update(pair, candle.to_dict())
        if snap:
            snapshot = snap
    assert snapshot is not None
    # 2. Генерируем признаки через feature_engineering
    features = snapshot["features"]
    assert isinstance(features, dict)
    # 3. ML-инференс (transformer или fallback)
    pred_result = ml_integration.predict(features, pair)
    assert "prediction" in pred_result
    assert "confidence" in pred_result
    assert pred_result["model_source"] in ["transformer", "window_optimizer", "none"]
    # 4. Генерируем сигнал (мокаем)
    signal = {
        "side": "buy",
        "confidence": pred_result["confidence"],
        "price": features["close"],
    }
    assert signal["side"] in ["buy", "sell"]
    # 5. Передаём результат на dashboard/api
    mock_dashboard.update({"pair": pair, "signal": signal, "snapshot": snapshot})
    assert mock_dashboard.last_update is not None
    # Проверяем, что все ключевые этапы сработали
    assert "regime" in snapshot
    assert "optimal_window" in snapshot
    assert signal["confidence"] >= 0
