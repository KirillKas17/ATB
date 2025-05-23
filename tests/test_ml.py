from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from core.market_state import MarketState
from ml.decision_reasoner import DecisionReasoner
from ml.live_adaptation import LiveAdaptationModel
from ml.meta_learning import MetaLearning
from ml.model_selector import ModelSelector
from ml.pattern_discovery import PatternDiscovery
from ml.window_optimizer import WindowSizeOptimizer
from utils.logger import setup_logger

logger = setup_logger(__name__)


# Фикстуры
@pytest.fixture
def mock_market_data():
    """Фикстура с тестовыми рыночными данными"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1H")
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


@pytest.fixture
def mock_model():
    """Фикстура с тестовой моделью"""
    model = Mock()
    model.predict.return_value = np.array([0.8, 0.2])
    model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
    return model


@pytest.fixture
def model_selector():
    """Фикстура с селектором моделей"""
    return ModelSelector(min_accuracy=0.7, max_models=3)


@pytest.fixture
def pattern_discovery():
    """Фикстура с обнаружением паттернов"""
    return PatternDiscovery(min_pattern_length=5, max_pattern_length=20)


@pytest.fixture
def meta_learning():
    """Фикстура с мета-обучением"""
    return MetaLearning(update_interval=60, min_samples=100)


@pytest.fixture
def live_adaptation():
    """Фикстура с адаптацией в реальном времени"""
    return LiveAdaptationModel()


@pytest.fixture
def decision_reasoner():
    """Фикстура с обоснованием решений"""
    return DecisionReasoner(min_confidence=0.7, max_uncertainty=0.3)


# Тесты для ModelSelector
class TestModelSelector:
    def test_select_best_model(self, model_selector, mock_market_data, mock_model):
        """Тест выбора лучшей модели"""
        models = {"model1": mock_model, "model2": mock_model}
        validation_data = mock_market_data

        best_model = model_selector.select_best_model(
            models=models, validation_data=validation_data
        )

        assert isinstance(best_model, str)
        assert best_model in models

    def test_evaluate_model(self, model_selector, mock_model, mock_market_data):
        """Тест оценки модели"""
        metrics = model_selector.evaluate_model(model=mock_model, data=mock_market_data)

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics


# Тесты для PatternDiscovery
class TestPatternDiscovery:
    def test_discover_patterns(self, pattern_discovery, mock_market_data):
        """Тест обнаружения паттернов"""
        patterns = pattern_discovery.discover_patterns(mock_market_data)

        assert isinstance(patterns, list)
        assert len(patterns) > 0

    def test_validate_pattern(self, pattern_discovery, mock_market_data):
        """Тест валидации паттерна"""
        pattern = mock_market_data.iloc[:10]

        is_valid = pattern_discovery.validate_pattern(
            pattern=pattern, data=mock_market_data
        )

        assert isinstance(is_valid, bool)


# Тесты для MetaLearning
class TestMetaLearning:
    def test_update_meta_features(self, meta_learning, mock_market_data):
        """Тест обновления мета-признаков"""
        features = meta_learning.update_meta_features(mock_market_data)

        assert isinstance(features, dict)
        assert "volatility" in features
        assert "trend" in features
        assert "volume" in features

    def test_select_meta_strategy(self, meta_learning, mock_market_data):
        """Тест выбора мета-стратегии"""
        strategy = meta_learning.select_meta_strategy(mock_market_data)

        assert isinstance(strategy, str)
        assert strategy in ["trend", "volatility", "manipulation"]


# Тесты для LiveAdaptation
class TestLiveAdaptation:
    def test_detect_concept_drift(self, live_adaptation, mock_market_data):
        """Тест обнаружения дрейфа концепции"""
        drift = live_adaptation.detect_concept_drift(mock_market_data)

        assert isinstance(drift, bool)

    def test_adapt_model(self, live_adaptation, mock_model, mock_market_data):
        """Тест адаптации модели"""
        adapted_model = live_adaptation.adapt_model(
            model=mock_model, new_data=mock_market_data
        )

        assert adapted_model is not None


# Тесты для DecisionReasoner
class TestDecisionReasoner:
    def test_explain_decision(self, decision_reasoner, mock_model, mock_market_data):
        """Тест объяснения решения"""
        explanation = decision_reasoner.explain_decision(
            model=mock_model, data=mock_market_data
        )

        assert isinstance(explanation, dict)
        assert "confidence" in explanation
        assert "features" in explanation
        assert "reasoning" in explanation

    def test_validate_decision(self, decision_reasoner, mock_model, mock_market_data):
        """Тест валидации решения"""
        decision = {"action": "buy", "confidence": 0.8}

        is_valid = decision_reasoner.validate_decision(
            decision=decision, model=mock_model, data=mock_market_data
        )

        assert isinstance(is_valid, bool)


def create_sample_data() -> pd.DataFrame:
    """Создание тестовых данных"""
    return pd.DataFrame(
        {
            "open": np.random.uniform(100, 200, 100),
            "high": np.random.uniform(200, 300, 100),
            "low": np.random.uniform(50, 100, 100),
            "close": np.random.uniform(100, 200, 100),
            "volume": np.random.uniform(1000, 5000, 100),
            "atr": np.random.uniform(1, 5, 100),
            "adx": np.random.uniform(10, 50, 100),
            "rsi": np.random.uniform(20, 80, 100),
            "bb_upper": np.random.uniform(200, 300, 100),
            "bb_lower": np.random.uniform(50, 100, 100),
        }
    )


def test_window_optimizer_prediction():
    """Тест предсказания размера окна"""
    # Создание тестовых данных
    df = create_sample_data()

    # Инициализация оптимизатора
    optimizer = WindowSizeOptimizer("models/test_window_model.pkl")

    # Тестовые метаданные
    meta = {"volatility": 0.3, "trend_strength": 0.5, "regime": "трендовый"}

    # Извлечение признаков
    features = optimizer.extract_features(df, meta)

    # Проверка структуры признаков
    assert all(
        key in features
        for key in [
            "volatility",
            "trend_strength",
            "regime_encoded",
            "atr",
            "adx",
            "rsi",
            "bollinger_width",
        ]
    )

    # Проверка кодирования режима
    assert optimizer.encode_regime("трендовый") == 0
    assert optimizer.encode_regime("боковой") == 1
    assert optimizer.encode_regime("неизвестный") == 5

    # Проверка предсказания
    prediction = optimizer.predict(features)
    assert isinstance(prediction, int)
    assert 150 <= prediction <= 2000


def test_window_optimizer_error_handling():
    """Тест обработки ошибок"""
    optimizer = WindowSizeOptimizer()

    # Тест с пустыми данными
    empty_df = pd.DataFrame()
    meta = {"volatility": 0.3, "trend_strength": 0.5, "regime": "трендовый"}

    features = optimizer.extract_features(empty_df, meta)
    assert all(
        key in features
        for key in [
            "volatility",
            "trend_strength",
            "regime_encoded",
            "atr",
            "adx",
            "rsi",
            "bollinger_width",
        ]
    )

    # Проверка предсказания при ошибке
    prediction = optimizer.predict(features)
    assert isinstance(prediction, int)
    assert prediction == 300  # Базовое значение при ошибке


def test_market_state_window_optimizer_integration():
    """Проверка интеграции WindowSizeOptimizer в MarketState"""
    ms = MarketState()
    pair = "BTCUSDT"
    timeframe = "1h"
    # Генерируем 60 свечей (достаточно для анализа)
    candles = [
        {
            "open": 100 + i,
            "high": 101 + i,
            "low": 99 + i,
            "close": 100 + i + np.random.uniform(-1, 1),
            "volume": 1000 + np.random.uniform(-100, 100),
        }
        for i in range(60)
    ]
    for candle in candles:
        ms.update_state(pair, timeframe, candle)
    # После обновления состояния должен быть рассчитан optimal_window
    optimal_window = ms.get_optimal_window(pair)
    assert isinstance(optimal_window, int)
    assert 150 <= optimal_window <= 2000


def test_live_adaptation_initialization(live_adaptation):
    """Тест инициализации адаптации"""
    assert live_adaptation is not None
    assert live_adaptation.config is not None
    assert live_adaptation.market_state is not None
    assert live_adaptation.model_selector is not None
    assert live_adaptation.pattern_discovery is not None


def test_adaptation_process(live_adaptation, mock_market_data):
    """Тест процесса адаптации"""
    # Создаем тестовую модель
    model = Mock()
    model.predict.return_value = np.random.normal(0, 1, len(mock_market_data))

    # Тестируем адаптацию
    adapted_model, confidence = live_adaptation.adapt(
        pair="BTC/USDT", timeframe="1h", data=mock_market_data, current_model=model
    )

    assert adapted_model is not None
    assert 0 <= confidence <= 1


def test_adaptation_metrics(live_adaptation, mock_market_data):
    """Тест метрик адаптации"""
    # Создаем тестовую модель
    model = Mock()
    model.predict.return_value = np.random.normal(0, 1, len(mock_market_data))

    # Выполняем адаптацию
    live_adaptation.adapt(
        pair="BTC/USDT", timeframe="1h", data=mock_market_data, current_model=model
    )

    # Проверяем метрики
    metrics = live_adaptation.get_adaptation_metrics("BTC/USDT", "1h")
    assert metrics is not None
    assert hasattr(metrics, "accuracy_change")
    assert hasattr(metrics, "win_rate_change")
    assert hasattr(metrics, "adaptation_time")
    assert hasattr(metrics, "confidence")
    assert hasattr(metrics, "last_update")


def test_adaptation_with_market_regime(live_adaptation, mock_market_data):
    """Тест адаптации с учетом рыночного режима"""
    # Создаем тестовую модель
    model = Mock()
    model.predict.return_value = np.random.normal(0, 1, len(mock_market_data))

    # Мокаем состояние рынка
    with patch.object(
        live_adaptation.market_state, "get_current_regime"
    ) as mock_regime:
        mock_regime.return_value = "trend"

        # Тестируем адаптацию
        adapted_model, confidence = live_adaptation.adapt(
            pair="BTC/USDT", timeframe="1h", data=mock_market_data, current_model=model
        )

        assert adapted_model is not None
        assert 0 <= confidence <= 1
        mock_regime.assert_called_once_with("BTC/USDT", "1h")


def test_adaptation_with_low_accuracy(live_adaptation, mock_market_data):
    """Тест адаптации при низкой точности"""
    # Создаем тестовую модель с низкой точностью
    model = Mock()
    model.predict.return_value = np.random.normal(
        0, 10, len(mock_market_data)
    )  # Большой разброс

    # Тестируем адаптацию
    adapted_model, confidence = live_adaptation.adapt(
        pair="BTC/USDT", timeframe="1h", data=mock_market_data, current_model=model
    )

    assert adapted_model is not None
    assert confidence < 1.0  # Уверенность должна быть ниже при низкой точности


def test_adaptation_with_insufficient_data(live_adaptation):
    """Тест адаптации при недостаточном количестве данных"""
    # Создаем небольшой набор данных
    small_data = pd.DataFrame(
        {"open": [100], "high": [101], "low": [99], "close": [100], "volume": [1000]}
    )

    # Создаем тестовую модель
    model = Mock()
    model.predict.return_value = [0]

    # Тестируем адаптацию
    adapted_model, confidence = live_adaptation.adapt(
        pair="BTC/USDT", timeframe="1h", data=small_data, current_model=model
    )

    assert adapted_model is not None
    assert confidence < 1.0  # Уверенность должна быть ниже при недостаточных данных
