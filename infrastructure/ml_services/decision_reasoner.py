"""
Основной модуль для принятия торговых решений.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Правильные type aliases для mypy
DataFrameType = pd.DataFrame
SeriesType = pd.Series

# Импорт доменных типов
from domain.type_definitions.ml_types import ActionType, AggregatedSignal, SignalSource, SignalType, TradingSignal

from .candle_patterns import CandlePatternAnalyzer
from .explanation import DecisionExplainer

# Импорт созданных модулей
from .market_regime_detector import MarketRegimeDetector
from .ml_models import MLModelManager
from .online_learning_reasoner import OnlineLearningReasoner
from .signal_aggregator import SignalAggregator
from .technical_indicators import TechnicalIndicators
from .visualization import TradingVisualizer

# Настройка логирования
logger = logging.getLogger(__name__)
# Импорт библиотек для технического анализа (опционально)
try:
    import ta
except ImportError:
    ta = None
    logger.warning("Библиотека 'ta' не установлена. Некоторые индикаторы будут недоступны.")
# Импорт библиотек для визуализации (опционально)
try:
    pass
except ImportError:
    plt = None
    sns = None
    logger.warning("Библиотеки matplotlib/seaborn не установлены. Визуализация будет недоступна.")
# Импорт библиотек для объяснений (опционально)
try:
    import lime
    import shap
except ImportError:
    shap = None
    lime = None
    logger.warning("Библиотеки SHAP/LIME не установлены. Объяснения будут упрощенными.")


class DecisionReasoner:
    """
    Основной класс для принятия торговых решений на основе ML моделей.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Инициализация системы принятия решений."""
        self.config = config or {}
        self.signal_aggregator = SignalAggregator()
        self.market_regime_detector = MarketRegimeDetector()
        self.ml_models = MLModelManager()
        self.pattern_analyzer = CandlePatternAnalyzer()
        self.technical_indicators = TechnicalIndicators()
        self.explainer = DecisionExplainer()
        self.visualizer = TradingVisualizer()
        self.online_learner = OnlineLearningReasoner()

        logger.info("DecisionReasoner инициализирован")

    def analyze_and_decide(self, market_data: DataFrameType, symbol: str) -> AggregatedSignal:
        """
        Главный метод анализа и принятия решений.
        """
        try:
            # Получаем сигналы от разных источников
            signals = self._collect_signals(market_data, symbol)

            # Агрегируем сигналы
            if hasattr(self.signal_aggregator, 'aggregate_ensemble_signals'):
                aggregated = self.signal_aggregator.aggregate_ensemble_signals(signals)
            else:
                # Создаем базовый агрегированный сигнал
                aggregated = self._create_basic_aggregated_signal(signals, symbol)

            return aggregated

        except Exception as e:
            logger.error(f"Ошибка в анализе решений: {e}")
            # Возвращаем нейтральный сигнал в случае ошибки
            return AggregatedSignal(
                action=ActionType.HOLD,
                confidence=0.0,
                timestamp=datetime.now(),
                symbol=symbol,
                component_signals=[],
                weights={},
                consensus_score=0.0
            )

    def _collect_signals(self, market_data: DataFrameType, symbol: str) -> List[TradingSignal]:
        """Собирает сигналы от различных источников."""
        signals = []

        # Технические индикаторы
        signals.extend(self._get_technical_signals(market_data, symbol))

        # Паттерны свечей
        signals.extend(self._get_pattern_signals(market_data, symbol))

        # ML модели
        signals.extend(self._get_ml_signals(market_data, symbol))

        # Режим рынка
        signals.extend(self._get_regime_signals(market_data, symbol))

        return signals

    def _get_technical_signals(self, market_data: DataFrameType, symbol: str) -> List[TradingSignal]:
        """Получает сигналы технического анализа."""
        signals = []

        try:
            # Проверяем, что данные доступны
            if market_data is None or market_data.empty:
                return signals

            # RSI сигнал
            rsi = self.technical_indicators.calculate_rsi(market_data['close'])
            if rsi.iloc[-1] > 70:
                signals.append(TradingSignal(
                    action=ActionType.SELL,
                    confidence=0.7,
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type=SignalType.MOMENTUM,
                    source=SignalSource.TECHNICAL,
                    metadata={"indicator": "rsi", "value": float(rsi.iloc[-1])}
                ))
            elif rsi.iloc[-1] < 30:
                signals.append(TradingSignal(
                    action=ActionType.BUY,
                    confidence=0.7,
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type=SignalType.MOMENTUM,
                    source=SignalSource.TECHNICAL,
                    metadata={"indicator": "rsi", "value": float(rsi.iloc[-1])}
                ))

            # MACD сигнал
            macd = self.technical_indicators.calculate_macd(market_data['close'])
            if macd['histogram'].iloc[-1] > 0:
                signals.append(TradingSignal(
                    action=ActionType.BUY,
                    confidence=0.6,
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type=SignalType.TREND,
                    source=SignalSource.TECHNICAL,
                    metadata={"indicator": "macd", "histogram": float(macd['histogram'].iloc[-1])}
                ))
            else:
                signals.append(TradingSignal(
                    action=ActionType.SELL,
                    confidence=0.6,
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type=SignalType.TREND,
                    source=SignalSource.TECHNICAL,
                    metadata={"indicator": "macd", "histogram": float(macd['histogram'].iloc[-1])}
                ))

            # Bollinger Bands сигнал
            bb = self.technical_indicators.calculate_bollinger_bands(market_data['close'])
            current_price = market_data['close'].iloc[-1]
            if current_price > bb['upper'].iloc[-1]:
                signals.append(TradingSignal(
                    action=ActionType.SELL,
                    confidence=0.8,
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type=SignalType.REVERSAL,
                    source=SignalSource.TECHNICAL,
                    metadata={"indicator": "bollinger", "position": "above_upper"}
                ))
            elif current_price < bb['lower'].iloc[-1]:
                signals.append(TradingSignal(
                    action=ActionType.BUY,
                    confidence=0.8,
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type=SignalType.REVERSAL,
                    source=SignalSource.TECHNICAL,
                    metadata={"indicator": "bollinger", "position": "below_lower"}
                ))

        except Exception as e:
            logger.error(f"Ошибка в получении технических сигналов: {e}")

        return signals

    def _get_pattern_signals(self, market_data: DataFrameType, symbol: str) -> List[TradingSignal]:
        """Получает сигналы паттернов свечей."""
        signals = []

        try:
            patterns = self.pattern_analyzer.detect_patterns(market_data)

            for pattern in patterns:
                if pattern['bullish']:
                    signals.append(TradingSignal(
                        action=ActionType.BUY,
                        confidence=pattern['confidence'],
                        timestamp=datetime.now(),
                        symbol=symbol,
                        signal_type=SignalType.REVERSAL,
                        source=SignalSource.TECHNICAL,
                        metadata={"pattern": pattern['name'], "type": "bullish"}
                    ))
                else:
                    signals.append(TradingSignal(
                        action=ActionType.SELL,
                        confidence=pattern['confidence'],
                        timestamp=datetime.now(),
                        symbol=symbol,
                        signal_type=SignalType.REVERSAL,
                        source=SignalSource.TECHNICAL,
                        metadata={"pattern": pattern['name'], "type": "bearish"}
                    ))

        except Exception as e:
            logger.error(f"Ошибка в получении сигналов паттернов: {e}")

        return signals

    def _get_ml_signals(self, market_data: DataFrameType, symbol: str) -> List[TradingSignal]:
        """Получает сигналы от ML моделей."""
        signals = []

        try:
            predictions = self.ml_models.predict(market_data)

            for model_name, prediction in predictions.items():
                action = ActionType.HOLD
                if prediction['action'] == 'buy':
                    action = ActionType.BUY
                elif prediction['action'] == 'sell':
                    action = ActionType.SELL

                signals.append(TradingSignal(
                    action=action,
                    confidence=prediction['confidence'],
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type=SignalType.TREND,
                    source=SignalSource.TECHNICAL,
                    metadata={"model": model_name, "features": prediction.get('features', {})}
                ))

        except Exception as e:
            logger.error(f"Ошибка в получении ML сигналов: {e}")

        return signals

    def _get_regime_signals(self, market_data: DataFrameType, symbol: str) -> List[TradingSignal]:
        """Получает сигналы режима рынка."""
        signals = []

        try:
            regime = self.market_regime_detector.detect_regime(market_data)

            if regime['trend'] == 'bullish':
                signals.append(TradingSignal(
                    action=ActionType.BUY,
                    confidence=regime['confidence'],
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type=SignalType.TREND,
                    source=SignalSource.TECHNICAL,
                    metadata={"regime": "bullish", "volatility": regime.get('volatility', 0)}
                ))
            elif regime['trend'] == 'bearish':
                signals.append(TradingSignal(
                    action=ActionType.SELL,
                    confidence=regime['confidence'],
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type=SignalType.TREND,
                    source=SignalSource.TECHNICAL,
                    metadata={"regime": "bearish", "volatility": regime.get('volatility', 0)}
                ))
            else:
                signals.append(TradingSignal(
                    action=ActionType.HOLD,
                    confidence=regime['confidence'],
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type=SignalType.TREND,
                    source=SignalSource.TECHNICAL,
                    metadata={"regime": "sideways", "volatility": regime.get('volatility', 0)}
                ))

        except Exception as e:
            logger.error(f"Ошибка в получении сигналов режима: {e}")

        return signals

    def _create_basic_aggregated_signal(self, signals: List[TradingSignal], symbol: str) -> AggregatedSignal:
        """Создает базовый агрегированный сигнал."""
        if not signals:
            return AggregatedSignal(
                action=ActionType.HOLD,
                confidence=0.0,
                timestamp=datetime.now(),
                symbol=symbol,
                component_signals=[],
                weights={},
                consensus_score=0.0
            )

        # Простое голосование
        buy_votes = sum(1 for s in signals if s.action == ActionType.BUY)
        sell_votes = sum(1 for s in signals if s.action == ActionType.SELL)
        hold_votes = sum(1 for s in signals if s.action == ActionType.HOLD)

        if buy_votes > sell_votes and buy_votes > hold_votes:
            action = ActionType.BUY
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            action = ActionType.SELL
        else:
            action = ActionType.HOLD

        # Средняя уверенность
        avg_confidence = sum(s.confidence for s in signals) / len(signals)

        # Подсчет весов источников
        weights = {}
        for signal in signals:
            if signal.source:
                weights[signal.source] = weights.get(signal.source, 0) + 1

        # Нормализация весов
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        consensus_score = max(buy_votes, sell_votes, hold_votes) / len(signals)

        return AggregatedSignal(
            action=action,
            confidence=avg_confidence,
            timestamp=datetime.now(),
            symbol=symbol,
            component_signals=signals,
            weights=weights,
            consensus_score=consensus_score
        )

    def analyze_risk(self, signal: AggregatedSignal, market_data: DataFrameType) -> float:
        """Анализирует риск торгового сигнала."""
        try:
            # Базовый анализ риска
            volatility = self._calculate_volatility(market_data)
            volume_risk = self._calculate_volume_risk(market_data)
            
            # Комбинированный риск
            risk_score = (volatility + volume_risk) / 2
            
            # Учитываем уверенность в сигнале
            adjusted_risk = risk_score * (1 - signal.confidence)
            
            return min(max(adjusted_risk, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Ошибка в анализе риска: {e}")
            return 0.5  # Средний риск по умолчанию

    def _calculate_volatility(self, market_data: DataFrameType) -> float:
        """Рассчитывает волатильность."""
        try:
            if market_data is None or market_data.empty:
                return 0.5

            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Нормализуем к диапазону [0, 1]
            normalized_vol = min(volatility * 100, 1.0)
            return normalized_vol

        except Exception as e:
            logger.error(f"Ошибка расчета волатильности: {e}")
            return 0.5

    def _calculate_volume_risk(self, market_data: DataFrameType) -> float:
        """Рассчитывает риск по объему."""
        try:
            if market_data is None or market_data.empty or 'volume' not in market_data.columns:
                return 0.5

            # Анализ объемов
            volume_ma = market_data['volume'].rolling(window=20).mean()
            current_volume = market_data['volume'].iloc[-1]
            avg_volume = volume_ma.iloc[-1]

            if pd.isna(avg_volume) or avg_volume == 0:
                return 0.5

            volume_ratio = current_volume / avg_volume
            
            # Низкий объем = высокий риск
            if volume_ratio < 0.5:
                return 0.8
            elif volume_ratio > 2.0:
                return 0.3
            else:
                return 0.5

        except Exception as e:
            logger.error(f"Ошибка расчета риска по объему: {e}")
            return 0.5

    def get_explanation(self, signal: AggregatedSignal, market_data: DataFrameType) -> Dict[str, Any]:
        """Получает объяснение торгового решения."""
        try:
            explanation = {
                "decision": signal.action.value,
                "confidence": signal.confidence,
                "consensus_score": signal.consensus_score,
                "component_count": len(signal.component_signals),
                "sources": list(signal.weights.keys()) if hasattr(signal, 'weights') else [],
                "technical_factors": self._get_technical_factors(market_data),
                "risk_assessment": self.analyze_risk(signal, market_data)
            }

            # Добавляем детали компонентных сигналов
            if hasattr(signal, 'component_signals') and signal.component_signals:
                explanation["signal_breakdown"] = []
                for component in signal.component_signals:
                    explanation["signal_breakdown"].append({
                        "action": component.action.value,
                        "confidence": component.confidence,
                        "source": component.source.value if component.source else "unknown",
                        "type": component.signal_type.value if component.signal_type else "unknown",
                        "metadata": component.metadata
                    })

            return explanation

        except Exception as e:
            logger.error(f"Ошибка создания объяснения: {e}")
            return {
                "decision": signal.action.value,
                "confidence": signal.confidence,
                "error": str(e)
            }

    def _get_technical_factors(self, market_data: DataFrameType) -> Dict[str, Any]:
        """Получает технические факторы для объяснения."""
        try:
            if market_data is None or market_data.empty:
                return {}

            factors = {}

            # RSI
            if len(market_data) >= 14:
                rsi = self.technical_indicators.calculate_rsi(market_data['close'])
                if not rsi.empty:
                    factors["rsi"] = float(rsi.iloc[-1])

            # MACD
            if len(market_data) >= 26:
                macd = self.technical_indicators.calculate_macd(market_data['close'])
                if 'histogram' in macd and not macd['histogram'].empty:
                    factors["macd_histogram"] = float(macd['histogram'].iloc[-1])

            # Цена относительно скользящих средних
            if len(market_data) >= 20:
                ma20 = market_data['close'].rolling(20).mean()
                if not ma20.empty:
                    current_price = market_data['close'].iloc[-1]
                    factors["price_vs_ma20"] = float((current_price - ma20.iloc[-1]) / ma20.iloc[-1] * 100)

            return factors

        except Exception as e:
            logger.error(f"Ошибка получения технических факторов: {e}")
            return {}

    def update_models(self, market_data: DataFrameType, actual_outcome: float) -> None:
        """Обновляет модели на основе фактических результатов."""
        try:
            self.online_learner.update(market_data, actual_outcome)
            logger.info("Модели обновлены на основе фактических результатов")

        except Exception as e:
            logger.error(f"Ошибка обновления моделей: {e}")

    def get_model_performance(self) -> Dict[str, Any]:
        """Возвращает метрики производительности моделей."""
        try:
            return self.ml_models.get_model_metrics()

        except Exception as e:
            logger.error(f"Ошибка получения метрик производительности: {e}")
            return {} 