import json
import os
from datetime import datetime
import pytest
import pandas as pd
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from infrastructure.ml_services.decision_reasoner import (DecisionReasoner, DecisionReport,
                                  TradeDecision)
from sklearn.ensemble import RandomForestClassifier
@pytest.fixture
def sample_data() -> Any:
    """Фикстура с тестовыми данными"""
    # Создание тестовых данных
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
    # Добавление тренда
    data["close"] = data["close"] + np.linspace(0, 10, 100)
    data["high"] = data["high"] + np.linspace(0, 10, 100)
    data["low"] = data["low"] + np.linspace(0, 10, 100)
    return data


@pytest.fixture
def sample_model() -> Any:
    """Фикстура с тестовой моделью"""
    model = RandomForestClassifier(n_estimators=10)
    # Создание тестовых данных для обучения
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 3, 100)
    model.fit(X, y)
    return model


@pytest.fixture
def decision_reasoner() -> Any:
    """Фикстура с экземпляром DecisionReasoner"""
    return DecisionReasoner()


@pytest.fixture
def reasoner(tmp_path) -> Any:
    """Фикстура для DecisionReasoner"""
    return DecisionReasoner(log_dir=str(tmp_path))


@pytest.fixture
def trade_decision() -> Any:
    """Фикстура для торгового решения"""
    return TradeDecision(
        symbol="BTCUSDT",
        action="open",
        direction="long",
        volume=0.1,
        confidence=0.85,
        stop_loss=49000.0,
        take_profit=52000.0,
        timestamp=datetime.now(),
        metadata={},
    )
    @pytest.fixture
    def market_data() -> Any:
        """Фикстура для рыночных данных"""
        return {
            "strategy": {"name": "trend_following", "confirmations": 3},
            "regime": "uptrend",
            "indicators": {
                "RSI": {"value": 65.0},
                "MACD": {"value": 0.5},
                "BB": {"value": -0.2},
            },
            "whale_activity": {
                "active": True,
                "volume": 100.0,
                "side": "buy",
                "price": 50000.0,
            },
            "volume_data": {"current": 1000.0, "change": 0.2},
        }

    def test_initialization(decision_reasoner) -> None:
        """Тест инициализации"""
        assert decision_reasoner.config["min_confidence"] == 0.7
        assert decision_reasoner.config["report_dir"] == "decision_reports"
        assert decision_reasoner.config["visualization_dir"] == "decision_visualizations"
        assert isinstance(decision_reasoner.reports, dict)
        assert decision_reasoner.explainer is None
        assert decision_reasoner.lime_explainer is None

    def test_explain_decision(decision_reasoner, sample_data, sample_model) -> None:
        """Тест объяснения решения"""
        signal = {"type": "long", "confidence": 0.85}
        report = decision_reasoner.explain_decision(
            pair="BTC/USD",
            timeframe="1h",
            data=sample_data,
            model=sample_model,
            signal=signal,
        )
        assert isinstance(report, DecisionReport)
        assert report.signal_type == "long"
        assert report.confidence == 0.85
        assert isinstance(report.timestamp, datetime)
        assert isinstance(report.features_importance, dict)
        assert isinstance(report.technical_indicators, dict)
        assert isinstance(report.market_context, dict)
        assert isinstance(report.explanation, str)
        assert isinstance(report.visualization_path, str)
        # Проверяем, что все необходимые поля присутствуют
        assert "confidence" in report.__dict__
        assert "features_importance" in report.__dict__  # factors
        assert "technical_indicators" in report.__dict__  # raw_output
    def test_feature_importance(decision_reasoner, sample_data, sample_model) -> None:
        """Тест расчета важности признаков"""
        features = decision_reasoner._prepare_features(sample_data)
        importance = decision_reasoner._get_feature_importance(features, sample_model)
        assert isinstance(importance, dict)
        assert len(importance) > 0
        assert all(0 <= v <= 1 for v in importance.values())
        assert abs(sum(importance.values()) - 1.0) < 1e-6

    def test_technical_indicators(decision_reasoner, sample_data) -> None:
        """Тест расчета технических индикаторов"""
        indicators = decision_reasoner._get_technical_indicators(sample_data)
        assert isinstance(indicators, dict)
        assert "rsi" in indicators
        assert "macd" in indicators
        assert "macd_signal" in indicators
        assert "bb_upper" in indicators
        assert "bb_middle" in indicators
        assert "bb_lower" in indicators
        assert "atr" in indicators

    def test_market_context(decision_reasoner, sample_data) -> None:
        """Тест получения контекста рынка"""
        context = decision_reasoner._get_market_context(sample_data)
        assert isinstance(context, dict)
        assert "trend" in context
        assert context["trend"] in ["up", "down"]
        assert "volatility" in context
        assert "volume" in context
        assert "candle_patterns" in context
        assert isinstance(context["candle_patterns"], list)

    def test_explanation_generation(decision_reasoner) -> None:
        """Тест генерации объяснения"""
        signal = {"type": "long", "confidence": 0.85}
        feature_importance = {"rsi": 0.3, "macd": 0.4, "volume": 0.3}
        indicators = {"rsi": 65.0, "macd": 0.5, "macd_signal": 0.3}
        market_context = {
            "trend": "up",
            "volatility": 0.02,
            "volume": 1000,
            "candle_patterns": ["doji"],
        }
        explanation = decision_reasoner._generate_explanation(
            signal, feature_importance, indicators, market_context
        )
        assert isinstance(explanation, str)
        assert "Signal: LONG" in explanation
        assert "Key factors:" in explanation
        assert "Technical indicators:" in explanation
        assert "Market context:" in explanation

    def test_visualization_creation(decision_reasoner) -> None:
        """Тест создания визуализации"""
        feature_importance = {"rsi": 0.3, "macd": 0.4, "volume": 0.3}
        indicators = {"rsi": 65.0, "macd": 0.5, "macd_signal": 0.3}
        viz_path = decision_reasoner._create_visualization(
            pair="BTC/USD",
            timeframe="1h",
            feature_importance=feature_importance,
            indicators=indicators,
        )
        assert isinstance(viz_path, str)
        assert viz_path.endswith(".png")
        assert os.path.exists(viz_path)

    def test_report_saving(decision_reasoner) -> None:
        """Тест сохранения отчета"""
        report = DecisionReport(
            signal_type="long",
            confidence=0.85,
            timestamp=datetime.now(),
            features_importance={"rsi": 0.3},
            technical_indicators={"rsi": 65.0},
            market_context={"trend": "up"},
            explanation="Test explanation",
            visualization_path="test.png",
        )
        decision_reasoner._save_report(pair="BTC/USD", timeframe="1h", report=report)
        assert "BTC/USD" in decision_reasoner.reports
        assert len(decision_reasoner.reports["BTC/USD"]) > 0

    def test_report_retrieval(decision_reasoner) -> None:
        """Тест получения отчетов"""
        # Добавление тестового отчета
        report = DecisionReport(
            signal_type="long",
            confidence=0.85,
            timestamp=datetime.now(),
            features_importance={"rsi": 0.3},
            technical_indicators={"rsi": 65.0},
            market_context={"trend": "up"},
            explanation="Test explanation",
            visualization_path="test.png",
        )
        decision_reasoner.reports["BTC/USD"] = [report]
        # Получение отчетов
        reports = decision_reasoner.get_reports("BTC/USD")
        assert len(reports) == 1
        assert reports[0].signal_type == "long"
        # Получение отчетов с фильтрацией по таймфрейму
    reports = decision_reasoner.get_reports("BTC/USD", timeframe="1h")
    assert len(reports) == 1
    # Получение отчетов для несуществующей пары
    reports = decision_reasoner.get_reports("ETH/USD")
    assert len(reports) == 0
    def test_error_handling(decision_reasoner) -> None:
        """Тест обработки ошибок"""
        # Тест с некорректными данными
        report = decision_reasoner.explain_decision(
            pair="BTC/USD",
            timeframe="1h",
            data=pd.DataFrame(),  # Пустой DataFrame
            model=None,
            signal={"type": "long", "confidence": 0.85},
        )
        assert report is None
        # Тест с некорректным сигналом
        report = decision_reasoner.explain_decision(
            pair="BTC/USD",
            timeframe="1h",
            data=pd.DataFrame({"close": [1, 2, 3]}),
            model=RandomForestClassifier(),
            signal={"type": "invalid", "confidence": 1.5},  # Некорректный тип и уверенность
        )
        assert report is None

    def test_feature_preparation(decision_reasoner, sample_data) -> None:
        """Тест подготовки признаков"""
        features = decision_reasoner._prepare_features(sample_data)
        assert isinstance(features, pd.DataFrame)
        assert "rsi" in features.columns
        assert "macd" in features.columns
        assert "macd_signal" in features.columns
        assert "bb_upper" in features.columns
        assert "bb_middle" in features.columns
        assert "bb_lower" in features.columns
        assert "atr" in features.columns
        assert "body_size" in features.columns
        assert "upper_shadow" in features.columns
        assert "lower_shadow" in features.columns
        assert "is_bullish" in features.columns
        assert "volume_ma" in features.columns
        assert "volume_ratio" in features.columns
        assert "volatility" in features.columns
        assert "trend" in features.columns
class TestDecisionReasoner:
    """Тесты для DecisionReasoner"""
    def test_initialization(self, reasoner, tmp_path) -> None:
        """Тест инициализации"""
        assert reasoner.log_dir == tmp_path
        assert (tmp_path / "decision_reason.log").exists()
    def test_explain_hold_decision(self, reasoner, tmp_path) -> None:
        """Тест объяснения решения о воздержании"""
        decision = TradeDecision(
            symbol="BTCUSDT",
            action="hold",
            direction="none",
            volume=0.0,
            confidence=0.4,
            stop_loss=0.0,
            take_profit=0.0,
            timestamp=datetime.now(),
            metadata={},
        )
        explanation = reasoner.explain(decision, {})
        assert "воздержаться от торговли" in explanation
        assert "низкой уверенности" in explanation
        assert "40.00%" in explanation
    def test_explain_open_decision(self, reasoner, trade_decision, market_data) -> None:
        """Тест объяснения решения об открытии позиции"""
        explanation = reasoner.explain(trade_decision, market_data)
        # Проверка основных компонентов объяснения
        assert "длинную позицию" in explanation
        assert "BTCUSDT" in explanation
        assert "0.1000" in explanation
        assert "85.00%" in explanation
        assert "49000.00" in explanation
        assert "52000.00" in explanation
        # Проверка стратегии и режима
        assert "стратегия trend_following" in explanation
        assert "режима рынка 'uptrend'" in explanation
        assert "3 индикаторов" in explanation
        # Проверка индикаторов
        assert "RSI (65.00)" in explanation
        assert "MACD (0.50)" in explanation
        # Проверка активности китов
        assert "активность китов" in explanation
        assert "100.00 buy" in explanation
        assert "50000.00" in explanation
        # Проверка объемов
        assert "Объемы: 1000.00" in explanation
        assert "+20.00%" in explanation
    def test_explain_without_optional_data(self, reasoner, trade_decision) -> None:
        """Тест объяснения без опциональных данных"""
        explanation = reasoner.explain(trade_decision, {})
        # Проверка, что объяснение содержит только основную информацию
        assert "длинную позицию" in explanation
        assert "BTCUSDT" in explanation
        assert "0.1000" in explanation
        assert "85.00%" in explanation
        # Проверка отсутствия опциональных компонентов
        assert "стратегия" not in explanation
        assert "индикаторов" not in explanation
        assert "китов" not in explanation
        assert "Объемы" not in explanation
    def test_log_explanation(self, reasoner, trade_decision, market_data, tmp_path) -> None:
        """Тест логирования объяснения"""
        # Генерация объяснения
        reasoner.explain(trade_decision, market_data)
        # Проверка наличия файла лога
        log_file = tmp_path / "decision_reason.log"
        assert log_file.exists()
        # Проверка содержимого лога
        with open(log_file) as f:
            log_entry = json.loads(f.readline())
            assert log_entry["symbol"] == "BTCUSDT"
            assert log_entry["decision"]["action"] == "open"
            assert log_entry["decision"]["direction"] == "long"
            assert log_entry["decision"]["volume"] == 0.1
            assert log_entry["decision"]["confidence"] == 0.85
            assert log_entry["decision"]["stop_loss"] == 49000.0
            assert log_entry["decision"]["take_profit"] == 52000.0
            assert "explanation" in log_entry
            assert "data" in log_entry
    def test_error_handling(self, reasoner) -> None:
        """Тест обработки ошибок"""
        # Тест с некорректными данными
        explanation = reasoner.explain(None, {})
        assert explanation == "Ошибка при формировании объяснения"
        # Тест с некорректными индикаторами
        decision = TradeDecision(
            symbol="BTCUSDT",
            action="open",
            direction="long",
            volume=0.1,
            confidence=0.85,
            stop_loss=49000.0,
            take_profit=52000.0,
            timestamp=datetime.now(),
            metadata={},
        )
        data = {"indicators": {"RSI": None}}
        explanation = reasoner.explain(decision, data)
        assert "длинную позицию" in explanation  # Основное объяснение должно быть
