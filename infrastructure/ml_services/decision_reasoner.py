"""
Основной модуль для принятия торговых решений.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Type aliases for better mypy support
Series = pd.Series
DataFrame = pd.DataFrame

# Импорт доменных типов
from domain.types.ml_types import ActionType, AggregatedSignal, SignalSource, SignalType

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
    """Основной класс для принятия торговых решений"""
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        # Инициализация компонентов
        self.market_regime_detector = MarketRegimeDetector()
        self.signal_aggregator = SignalAggregator()
        self.online_learner = OnlineLearningReasoner()
        self.ml_model_manager = MLModelManager()
        self.visualizer = TradingVisualizer()
        self.explainer = DecisionExplainer()
        self.candle_pattern_analyzer = CandlePatternAnalyzer()  # Добавляем анализатор паттернов
        # Состояние
        self.last_decision: Optional[AggregatedSignal] = None
        self.decision_history: List[AggregatedSignal] = []
        self.performance_metrics: Dict[str, List[float]] = {
            'accuracy': [],
            'profit': [],
            'risk': []
        }
    def make_decision(self, market_data: DataFrame, 
                     additional_signals: Optional[List[SignalSource]] = None) -> AggregatedSignal:
        """Принятие торгового решения"""
        try:
            if len(market_data) < 50:
                return self._create_default_signal("Недостаточно данных")
            # 1. Анализ рыночного режима
            market_regime = self._detect_market_regime(market_data)
            # 2. Расчет технических индикаторов
            indicators = TechnicalIndicators.get_all_indicators(market_data)
            # 3. Поиск паттернов свечей
            patterns = self._get_candle_patterns(market_data)
            # 4. Генерация сигналов
            signals = self._generate_signals(market_data, indicators, patterns, market_regime)
            # 5. Добавление внешних сигналов
            if additional_signals:
                signals.extend(additional_signals)
            # 6. Агрегация сигналов
            aggregated_signal = self.signal_aggregator.aggregate_signals(signals)
            # 7. Обновление онлайн-обучения
            self._update_online_learning(market_data, indicators, aggregated_signal)
            # 8. Сохранение решения
            self.last_decision = aggregated_signal
            self.decision_history.append(aggregated_signal)
            # 9. Обновление метрик
            self._update_performance_metrics(aggregated_signal)
            logger.info(f"Принято решение: {aggregated_signal.action} "
                       f"(уверенность: {aggregated_signal.confidence:.3f})")
            return aggregated_signal
        except Exception as e:
            logger.error(f"Ошибка при принятии решения: {e}")
            return self._create_default_signal(f"Ошибка: {str(e)}")

    def _detect_market_regime(self, market_data: DataFrame) -> str:
        """Детекция рыночного режима."""
        try:
            # Используем метод predict из MarketRegimeDetector
            if not self.market_regime_detector.is_fitted:
                self.market_regime_detector.fit(market_data)
            regimes = self.market_regime_detector.predict(market_data)
            # Возвращаем последний режим как строку
            return f"regime_{regimes[-1] if len(regimes) > 0 else 0}"
        except Exception as e:
            logger.warning(f"Ошибка при детекции режима: {e}")
            return "unknown"

    def _get_candle_patterns(self, market_data: DataFrame) -> List[str]:
        """Получение паттернов свечей."""
        try:
            patterns_data = self.candle_pattern_analyzer.detect_patterns(market_data)
            # Извлекаем названия паттернов
            pattern_names = []
            for pattern_name, pattern_list in patterns_data.items():
                if pattern_list:  # Если найдены паттерны
                    pattern_names.append(pattern_name)
            return pattern_names
        except Exception as e:
            logger.warning(f"Ошибка при анализе паттернов: {e}")
            return []

    def _generate_signals(self, market_data: DataFrame, 
                         indicators: Dict[str, float],
                         patterns: List[str],
                         market_regime: str) -> List[SignalSource]:
        """Генерация сигналов на основе различных источников"""
        signals: List[SignalSource] = []
        # Сигнал на основе RSI
        rsi_signal = self._generate_rsi_signal(indicators.get('rsi', 50))
        if rsi_signal:
            signals.append(rsi_signal)
        # Сигнал на основе MACD
        macd_signal = self._generate_macd_signal(
            indicators.get('macd', 0),
            indicators.get('macd_signal', 0)
        )
        if macd_signal:
            signals.append(macd_signal)
        # Сигнал на основе Bollinger Bands
        # Исправляем использование .iloc на .values
        close_values = market_data['close'].values if hasattr(market_data['close'], 'values') else list(market_data['close'])
        bb_signal = self._generate_bb_signal(
            close_values[-1] if len(close_values) > 0 else 0,
            indicators.get('bb_upper', 0),
            indicators.get('bb_lower', 0)
        )
        if bb_signal:
            signals.append(bb_signal)
        # Сигнал на основе паттернов
        pattern_signal = self._generate_pattern_signal(patterns)
        if pattern_signal:
            signals.append(pattern_signal)
        # Сигнал на основе рыночного режима
        regime_signal = self._generate_regime_signal(market_regime)
        if regime_signal:
            signals.append(regime_signal)
        return signals
    def _generate_rsi_signal(self, rsi: float) -> Optional[SignalSource]:
        """Генерация сигнала на основе RSI"""
        if rsi < 30:
            return SignalSource(
                name="RSI",
                signal_type=SignalType.BUY,
                confidence=min((30 - rsi) / 30, 1.0),
                weight=1.0,
                metadata={"rsi_value": rsi}
            )
        elif rsi > 70:
            return SignalSource(
                name="RSI",
                signal_type=SignalType.SELL,
                confidence=min((rsi - 70) / 30, 1.0),
                weight=1.0,
                metadata={"rsi_value": rsi}
            )
        return None
    def _generate_macd_signal(self, macd: float, macd_signal: float) -> Optional[SignalSource]:
        """Генерация сигнала на основе MACD"""
        diff = macd - macd_signal
        if abs(diff) < 0.001:
            return None
        if diff > 0:
            return SignalSource(
                name="MACD",
                signal_type=SignalType.BUY,
                confidence=min(abs(diff) * 10, 1.0),
                weight=1.0,
                metadata={"macd": macd, "signal": macd_signal}
            )
        else:
            return SignalSource(
                name="MACD",
                signal_type=SignalType.SELL,
                confidence=min(abs(diff) * 10, 1.0),
                weight=1.0,
                metadata={"macd": macd, "signal": macd_signal}
            )
    def _generate_bb_signal(self, close: float, bb_upper: float, bb_lower: float) -> Optional[SignalSource]:
        """Генерация сигнала на основе полос Боллинджера"""
        if bb_upper == 0 or bb_lower == 0:
            return None
        if close <= bb_lower:
            return SignalSource(
                name="Bollinger_Bands",
                signal_type=SignalType.BUY,
                confidence=0.8,
                weight=1.0,
                metadata={"close": close, "bb_lower": bb_lower}
            )
        elif close >= bb_upper:
            return SignalSource(
                name="Bollinger_Bands",
                signal_type=SignalType.SELL,
                confidence=0.8,
                weight=1.0,
                metadata={"close": close, "bb_upper": bb_upper}
            )
        return None
    def _generate_pattern_signal(self, patterns: List[str]) -> Optional[SignalSource]:
        """Генерация сигнала на основе паттернов"""
        if not patterns:
            return None
        # Определение направления паттернов
        bullish_patterns = ['three_white_soldiers', 'dragonfly_doji', 'belt_hold']
        bearish_patterns = ['three_black_crows', 'gravestone_doji', 'kicking']
        bullish_count = sum(1 for p in patterns if p in bullish_patterns)
        bearish_count = sum(1 for p in patterns if p in bearish_patterns)
        if bullish_count > bearish_count:
            return SignalSource(
                name="Candle_Patterns",
                signal_type=SignalType.BUY,
                confidence=min(bullish_count * 0.3, 1.0),
                weight=1.0,
                metadata={"patterns": patterns}
            )
        elif bearish_count > bullish_count:
            return SignalSource(
                name="Candle_Patterns",
                signal_type=SignalType.SELL,
                confidence=min(bearish_count * 0.3, 1.0),
                weight=1.0,
                metadata={"patterns": patterns}
            )
        return None
    def _generate_regime_signal(self, regime: str) -> Optional[SignalSource]:
        """Генерация сигнала на основе рыночного режима"""
        if regime == "trend":
            return SignalSource(
                name="Market_Regime",
                signal_type=SignalType.BUY,
                confidence=0.6,
                weight=1.0,
                metadata={"regime": regime}
            )
        elif regime == "sideways":
            return SignalSource(
                name="Market_Regime",
                signal_type=SignalType.HOLD,
                confidence=0.7,
                weight=1.0,
                metadata={"regime": regime}
            )
        return None
    def _update_online_learning(self, market_data: DataFrame, 
                               indicators: Dict[str, float],
                               decision: AggregatedSignal) -> None:
        """Обновление онлайн-обучения"""
        try:
            # Подготовка признаков
            features = indicators.copy()
            # Определение целевой переменной (упрощенно)
            target = 1 if decision.action == ActionType.OPEN else (0 if decision.action == ActionType.CLOSE else 0.5)
            # Предсказание (упрощенно)
            prediction = 1 if decision.confidence > 0.5 else 0
            # Обновление модели
            self.online_learner.update(features, int(target), int(prediction))
        except Exception as e:
            logger.error(f"Ошибка при обновлении онлайн-обучения: {e}")
    def _update_performance_metrics(self, decision: AggregatedSignal) -> None:
        """Обновление метрик производительности"""
        try:
            # Получение метрик онлайн-обучения
            online_metrics = self.online_learner.get_metrics()
            if online_metrics:
                # Правильный доступ к метрикам
                accuracy = online_metrics.get('accuracy', 0.0) if isinstance(online_metrics, dict) else 0.0
                self.performance_metrics['accuracy'].append(accuracy)
                self.performance_metrics['profit'].append(decision.confidence)
                self.performance_metrics['risk'].append(decision.risk_score)
        except Exception as e:
            logger.error(f"Ошибка при обновлении метрик: {e}")
    def _create_default_signal(self, explanation: str) -> AggregatedSignal:
        """Создание сигнала по умолчанию"""
        return AggregatedSignal(
            action=ActionType.HOLD,
            confidence=0.0,
            risk_score=1.0,
            sources=[],
            timestamp=datetime.now(),
            explanation=explanation
        )
    def get_explanation(self, market_data: DataFrame, 
                       indicators: Dict[str, float]) -> str:
        """Получение объяснения последнего решения"""
        if not self.last_decision:
            return "Нет последнего решения для объяснения"
        return self.explainer.explain_decision(
            {
                'action': self.last_decision.action.value,
                'confidence': self.last_decision.confidence,
                'risk_score': self.last_decision.risk_score,
                'sources': self.last_decision.sources
            },
            market_data,
            indicators
        )
    def get_performance_summary(self) -> Dict[str, Any]:
        """Получение сводки производительности"""
        try:
            summary: Dict[str, Any] = {
                'total_decisions': len(self.decision_history),
                'last_decision': None,
                'online_learning_metrics': self.online_learner.get_metrics(),
                'performance_metrics': self.performance_metrics.copy()
            }
            if self.last_decision:
                summary['last_decision'] = {
                    'action': self.last_decision.action.value,
                    'confidence': self.last_decision.confidence,
                    'risk_score': self.last_decision.risk_score,
                    'timestamp': self.last_decision.timestamp.isoformat()
                }
            return summary
        except Exception as e:
            logger.error(f"Ошибка при получении сводки производительности: {e}")
            return {}
    def create_visualization(self, market_data: DataFrame, 
                           save_path: Optional[str] = None) -> None:
        """Создание визуализации"""
        try:
            if not self.last_decision:
                logger.warning("Нет данных для визуализации")
                return
            # Подготовка сигналов для визуализации
            signals = []
            for source in self.last_decision.sources:
                # Безопасное получение последней цены
                if len(market_data) > 0:
                    close_series = market_data['close']
                    if hasattr(close_series, 'iloc'):
                        current_price = close_series.iloc[-1]
                    else:
                        current_price = close_series[-1] if len(close_series) > 0 else 0
                else:
                    current_price = 0
                
                signals.append({
                    'timestamp': datetime.now(),
                    'action': source.signal_type.value,
                    'price': current_price
                })
            # Создание визуализации
            # Исправление: правильное использование .values для pandas
            if hasattr(market_data['close'], 'to_numpy'):
                close_values = market_data['close'].to_numpy()
            elif hasattr(market_data['close'], 'values'):
                close_values = market_data['close'].values
            else:
                close_values = list(market_data['close'])
            current_price = close_values[-1] if len(close_values) > 0 else 0
            
            # Обновляем сигналы с правильной ценой
            for signal in signals:
                signal['price'] = current_price
                
            self.visualizer.plot_candlestick_with_signals(
                market_data, signals, save_path
            )
        except Exception as e:
            logger.error(f"Ошибка при создании визуализации: {e}")
    def detect_drift(self) -> bool:
        """Обнаружение дрейфа данных"""
        return self.online_learner.detect_drift()
    def retrain_models(self, new_data: DataFrame, new_labels: Series) -> bool:
        """Переобучение моделей"""
        try:
            # Подготовка признаков
            features = []
            for i in range(len(new_data)):
                if i >= 50:  # Минимум 50 свечей для расчета индикаторов
                    # Исправление: правильное использование .iloc для pandas
                    if hasattr(new_data, 'iloc'):
                        window_data = new_data.iloc[i-50:i+1]
                    else:
                        window_data = new_data[i-50:i+1]
                    indicators = TechnicalIndicators.get_all_indicators(window_data)
                    features.append(list(indicators.values()))
            if len(features) < 10:
                logger.warning("Недостаточно данных для переобучения")
                return False
            # Создание DataFrame признаков
            # Исправление: правильное использование .iloc для pandas
            if hasattr(new_data, 'iloc'):
                initial_data = new_data.iloc[:50]
            else:
                initial_data = new_data[:50]
            feature_names = list(TechnicalIndicators.get_all_indicators(initial_data).keys())
            X: pd.DataFrame = DataFrame(features, columns=feature_names)
            # Исправление: правильное использование .iloc для pandas
            if hasattr(new_labels, 'iloc'):
                y_current = new_labels.iloc[50:len(features)+50]  # Соответствие индексов
            else:
                y_current = new_labels[50:len(features)+50]  # Соответствие индексов
            # Переобучение
            self.ml_model_manager.train_models(X, y_current)
            logger.info("Модели успешно переобучены")
            return True
        except Exception as e:
            logger.error(f"Ошибка при переобучении моделей: {e}")
            return False 