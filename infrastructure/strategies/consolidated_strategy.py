"""
Консолидированная стратегия - объединяет функциональность дублирующихся стратегий.
Этот модуль решает проблему дублирования кода между различными стратегиями,
предоставляя единый интерфейс с модульной архитектурой.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

import pandas as pd
from loguru import logger

from .base_strategy import BaseStrategy, Signal, StrategyMetrics


class StrategyType(Enum):
    """Типы стратегий."""

    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    ARBITRAGE = "arbitrage"
    PAIRS_TRADING = "pairs_trading"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    GRID = "grid"
    HEDGING = "hedging"
    MARTINGALE = "martingale"
    ADAPTIVE = "adaptive"
    EVOLVABLE = "evolvable"
    MANIPULATION = "manipulation"
    REGIME_ADAPTIVE = "regime_adaptive"


class MarketRegime(Enum):
    """Рыночные режимы."""

    TRENDING = "trending"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    RANGING = "ranging"
    BREAKOUT = "breakout"


@dataclass
class StrategyConfig:
    """Конфигурация стратегии."""

    strategy_type: StrategyType
    timeframes: List[str] = field(default_factory=lambda: ["1h"])
    symbols: List[str] = field(default_factory=list)
    risk_per_trade: float = 0.02
    max_position_size: float = 0.1
    confidence_threshold: float = 0.7
    use_stop_loss: bool = True
    use_take_profit: bool = True
    trailing_stop: bool = False
    trailing_stop_activation: float = 0.02
    trailing_stop_distance: float = 0.01
    adaptive_enabled: bool = False
    evolution_enabled: bool = False
    regime_detection_enabled: bool = False
    parameters: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AnalysisModule(Protocol):
    """Протокол для модулей анализа."""

    def analyze(
        self, data: pd.DataFrame, config: StrategyConfig
    ) -> Dict[str, Any]: ...


@runtime_checkable
class SignalGenerator(Protocol):
    """Протокол для генераторов сигналов."""

    def generate_signal(
        self, data: pd.DataFrame, analysis: Dict[str, Any], config: StrategyConfig
    ) -> Optional[Signal]: ...


@runtime_checkable
class RiskManager(Protocol):
    """Протокол для управления рисками."""

    def calculate_position_size(
        self, signal: Signal, account_balance: float, config: StrategyConfig
    ) -> float: ...

    def validate_signal(self, signal: Signal, config: StrategyConfig) -> bool: ...


@runtime_checkable
class RegimeDetector(Protocol):
    """Протокол для детекции рыночного режима."""

    def detect_regime(self, data: pd.DataFrame) -> MarketRegime: ...


class TechnicalAnalysisModule:
    """Модуль технического анализа."""

    def analyze(
        self, data: pd.DataFrame, config: StrategyConfig
    ) -> Dict[str, Any]:
        """Анализирует рыночные данные."""
        analysis: Dict[str, Any] = {}
        # Базовые индикаторы
        analysis["sma"] = self._calculate_sma(
            data, config.parameters.get("sma_period", 20)
        )
        analysis["ema"] = self._calculate_ema(
            data, config.parameters.get("ema_period", 20)
        )
        analysis["rsi"] = self._calculate_rsi(
            data, config.parameters.get("rsi_period", 14)
        )
        analysis["macd"] = self._calculate_macd(data)
        analysis["bollinger_bands"] = self._calculate_bollinger_bands(data)
        # Анализ тренда
        analysis["trend"] = self._analyze_trend(data, analysis)
        analysis["volatility"] = self._calculate_volatility(data)
        analysis["momentum"] = self._calculate_momentum(data)
        return analysis

    def _calculate_sma(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Расчет простой скользящей средней."""
        return data["close"].rolling(window=period).mean()

    def _calculate_ema(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Расчет экспоненциальной скользящей средней."""
        return data["close"].ewm(span=period).mean()

    def _calculate_rsi(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Расчет RSI."""
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Расчет MACD."""
        ema12 = data["close"].ewm(span=12).mean()
        ema26 = data["close"].ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        return {
            "macd_line": macd_line,
            "signal_line": signal_line,
            "histogram": histogram,
        }

    def _calculate_bollinger_bands(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Расчет полос Боллинджера."""
        sma = data["close"].rolling(window=20).mean()
        std = data["close"].rolling(window=20).std()
        return {"upper": sma + (std * 2), "middle": sma, "lower": sma - (std * 2)}

    def _analyze_trend(self, data: pd.DataFrame, analysis: Dict[str, Any]) -> str:
        """Анализ тренда."""
        if len(data) < 20:
            return "neutral"
        current_price = data["close"].iloc[-1]
        sma = analysis["sma"].iloc[-1]
        # Используем адаптивный порог на основе волатильности
        from shared.adaptive_thresholds import AdaptiveThresholds
        adaptive_thresholds = AdaptiveThresholds()
        volatility = analysis.get("volatility", 0.02)
        tolerance = max(0.005, min(0.02, volatility))  # От 0.5% до 2%
        
        if current_price > sma * (1 + tolerance):
            return "uptrend"
        elif current_price < sma * (1 - tolerance):
            return "downtrend"
        else:
            return "neutral"

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Расчет волатильности."""
        returns = data["close"].pct_change().dropna()
        return float(returns.std())

    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """Расчет моментума."""
        if len(data) < 10:
            return 0.0
        return float((data["close"].iloc[-1] / data["close"].iloc[-10]) - 1)


class TrendSignalGenerator:
    """Генератор сигналов для трендовых стратегий."""

    def generate_signal(
        self, data: pd.DataFrame, analysis: Dict[str, Any], config: StrategyConfig
    ) -> Optional[Signal]:
        """Генерирует сигнал для трендовой стратегии."""
        if len(data) < 20:
            return None
        current_price = data["close"].iloc[-1]
        sma = analysis["sma"].iloc[-1]
        rsi = analysis["rsi"].iloc[-1]
        trend = analysis["trend"]
        # Сигналы на покупку
        if trend == "uptrend" and current_price > sma and rsi < 70:
            stop_loss = current_price * (1 - config.risk_per_trade)
            take_profit = current_price * (1 + config.risk_per_trade * 2)
            return Signal(
                direction="buy",
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume=config.max_position_size,
                confidence=0.8,
                timestamp=data.index[-1],
                metadata={"strategy": "trend_following", "trend": trend},
            )
        # Сигналы на продажу
        elif trend == "downtrend" and current_price < sma and rsi > 30:
            stop_loss = current_price * (1 + config.risk_per_trade)
            take_profit = current_price * (1 - config.risk_per_trade * 2)
            return Signal(
                direction="sell",
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume=config.max_position_size,
                confidence=0.8,
                timestamp=data.index[-1],
                metadata={"strategy": "trend_following", "trend": trend},
            )
        return None


class MeanReversionSignalGenerator:
    """Генератор сигналов для стратегий возврата к среднему."""

    def generate_signal(
        self, data: pd.DataFrame, analysis: Dict[str, Any], config: StrategyConfig
    ) -> Optional[Signal]:
        """Генерирует сигнал для стратегии возврата к среднему."""
        if len(data) < 20:
            return None
        current_price = data["close"].iloc[-1]
        rsi = analysis["rsi"].iloc[-1]
        bollinger = analysis["bollinger_bands"]
        # Сигналы на покупку (перепроданность)
        if rsi < 30 and current_price < bollinger["lower"].iloc[-1]:
            stop_loss = current_price * (1 - config.risk_per_trade)
            take_profit = current_price * (1 + config.risk_per_trade)
            return Signal(
                direction="buy",
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume=config.max_position_size,
                confidence=0.7,
                timestamp=data.index[-1],
                metadata={"strategy": "mean_reversion", "rsi": rsi},
            )
        # Сигналы на продажу (перекупленность)
        elif rsi > 70 and current_price > bollinger["upper"].iloc[-1]:
            stop_loss = current_price * (1 + config.risk_per_trade)
            take_profit = current_price * (1 - config.risk_per_trade)
            return Signal(
                direction="sell",
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume=config.max_position_size,
                confidence=0.7,
                timestamp=data.index[-1],
                metadata={"strategy": "mean_reversion", "rsi": rsi},
            )
        return None


class RiskManagementModule:
    """Модуль управления рисками."""

    def calculate_position_size(
        self, signal: Signal, account_balance: float, config: StrategyConfig
    ) -> float:
        """Рассчитывает размер позиции."""
        risk_amount = account_balance * config.risk_per_trade
        if signal.stop_loss is None:
            return config.max_position_size
        risk_per_share = abs(signal.entry_price - signal.stop_loss)
        position_size = risk_amount / risk_per_share
        # Ограничиваем размер позиции
        max_position = account_balance * config.max_position_size
        return min(position_size, max_position)

    def validate_signal(self, signal: Signal, config: StrategyConfig) -> bool:
        """Валидирует сигнал."""
        if signal.confidence < config.confidence_threshold:
            return False
        if signal.entry_price <= 0:
            return False
        if signal.stop_loss is not None and signal.stop_loss <= 0:
            return False
        if signal.take_profit is not None and signal.take_profit <= 0:
            return False
        return True


class RegimeDetectionModule:
    """Модуль детекции рыночного режима."""

    def detect_regime(self, data: pd.DataFrame) -> MarketRegime:
        """Определяет рыночный режим."""
        if len(data) < 50:
            return MarketRegime.SIDEWAYS
        # Рассчитываем волатильность
        returns = data["close"].pct_change().dropna()
        volatility = returns.std()
        # Рассчитываем тренд
        sma_short = data["close"].rolling(window=10).mean()
        sma_long = data["close"].rolling(window=50).mean()
        trend_strength = abs(sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
        # Определяем режим
        if volatility > 0.03:  # Высокая волатильность
            return MarketRegime.VOLATILE
        elif trend_strength > 0.02:  # Сильный тренд
            return MarketRegime.TRENDING
        elif volatility < 0.01:  # Низкая волатильность
            return MarketRegime.RANGING
        else:
            return MarketRegime.SIDEWAYS


class ConsolidatedStrategy(BaseStrategy):
    """Консолидированная стратегия."""

    def __init__(self, config: Union[Dict[str, Any], StrategyConfig]):
        """
        Инициализация консолидированной стратегии.
        Args:
            config: Конфигурация стратегии
        """
        # Преобразуем конфигурацию в словарь для базового класса
        if isinstance(config, StrategyConfig):
            config_dict = {
                "strategy_type": config.strategy_type.value,
                "timeframes": config.timeframes,
                "symbols": config.symbols,
                "risk_per_trade": config.risk_per_trade,
                "max_position_size": config.max_position_size,
                "confidence_threshold": config.confidence_threshold,
                "use_stop_loss": config.use_stop_loss,
                "use_take_profit": config.use_take_profit,
                "trailing_stop": config.trailing_stop,
                "trailing_stop_activation": config.trailing_stop_activation,
                "trailing_stop_distance": config.trailing_stop_distance,
                "adaptive_enabled": config.adaptive_enabled,
                "evolution_enabled": config.evolution_enabled,
                "regime_detection_enabled": config.regime_detection_enabled,
                "parameters": config.parameters,
            }
        else:
            config_dict = config
        
        super().__init__(config_dict)
        
        # Устанавливаем конфигурацию
        if isinstance(config, StrategyConfig):
            self._config = config
        else:
            # Если config - это словарь, создаем объект StrategyConfig
            if isinstance(config, dict):
                # Извлекаем strategy_type из словаря
                strategy_type_value = config.get("strategy_type")
                if isinstance(strategy_type_value, str):
                    # Преобразуем строку в StrategyType
                    try:
                        strategy_type = StrategyType(strategy_type_value)
                    except ValueError:
                        strategy_type = StrategyType.TREND_FOLLOWING  # Значение по умолчанию
                else:
                    strategy_type = StrategyType.TREND_FOLLOWING
                
                # Создаем новый словарь с правильным strategy_type
                config_with_type = config.copy()
                config_with_type["strategy_type"] = strategy_type
                self._config = StrategyConfig(**config_with_type)
            else:
                # Fallback на значения по умолчанию
                self._config = StrategyConfig(strategy_type=StrategyType.TREND_FOLLOWING)
        
        # Инициализация модулей
        self.analysis_module = TechnicalAnalysisModule()
        self.risk_manager = RiskManagementModule()
        self.regime_detector = RegimeDetectionModule()
        
        # Получение генератора сигналов
        if isinstance(self._config, dict):
            strategy_type = StrategyType(self._config['strategy_type'])
        else:
            strategy_type = self._config.strategy_type
        self.signal_generator = self._get_signal_generator(strategy_type)
        
        # Состояние стратегии
        self.current_regime = MarketRegime.TRENDING
        self.adaptive_parameters = self._config.parameters.copy()
        if hasattr(config, 'strategy_type'):
            logger.info(f"Initialized ConsolidatedStrategy: {config.strategy_type.value}")
        else:
            logger.info("Initialized ConsolidatedStrategy with default config")

    def _get_signal_generator(self, strategy_type: StrategyType) -> SignalGenerator:
        """Возвращает подходящий генератор сигналов."""
        generators: Dict[StrategyType, SignalGenerator] = {
            StrategyType.TREND_FOLLOWING: TrendSignalGenerator(),
            StrategyType.MEAN_REVERSION: MeanReversionSignalGenerator(),
            StrategyType.BREAKOUT: TrendSignalGenerator(),  # Используем трендовый для брейкаутов
            StrategyType.SCALPING: TrendSignalGenerator(),  # Упрощенная версия для скальпинга
            StrategyType.ARBITRAGE: MeanReversionSignalGenerator(),  # Для арбитража
            StrategyType.PAIRS_TRADING: MeanReversionSignalGenerator(),  # Для парного трейдинга
            StrategyType.VOLATILITY: MeanReversionSignalGenerator(),  # Для волатильности
            StrategyType.MOMENTUM: TrendSignalGenerator(),  # Для моментума
            StrategyType.GRID: MeanReversionSignalGenerator(),  # Для сетки
            StrategyType.HEDGING: TrendSignalGenerator(),  # Для хеджирования
            StrategyType.MARTINGALE: MeanReversionSignalGenerator(),  # Для мартингейла
            StrategyType.ADAPTIVE: TrendSignalGenerator(),  # Адаптивный
            StrategyType.EVOLVABLE: TrendSignalGenerator(),  # Эволюционный
            StrategyType.MANIPULATION: MeanReversionSignalGenerator(),  # Для манипуляций
            StrategyType.REGIME_ADAPTIVE: TrendSignalGenerator(),  # Адаптивный к режиму
        }
        return generators.get(strategy_type, TrendSignalGenerator())

    def analyze(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Анализ рыночных данных.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Dict с результатами анализа
        """
        # Валидация данных
        is_valid, error_msg = self.validate_data(data)
        if not is_valid:
            logger.error(f"Data validation failed: {error_msg}")
            return {}
        # Технический анализ
        analysis = self.analysis_module.analyze(data, self._config)
        # Детекция рыночного режима
        if self._config.regime_detection_enabled:
            self.current_regime = self.regime_detector.detect_regime(data)
            analysis["market_regime"] = self.current_regime.value
        # Адаптивная настройка параметров
        if self._config.adaptive_enabled:
            analysis = self._adapt_parameters(analysis, data)
        self.last_analysis = analysis
        return analysis

    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """
        Генерация торгового сигнала.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Optional[Signal] с сигналом или None
        """
        # Анализ данных
        analysis = self.analyze(data)
        if not analysis:
            return None
        # Генерация сигнала
        signal = self.signal_generator.generate_signal(
            data, analysis, self._config
        )
        if signal:
            # Валидация сигнала
            if not self.risk_manager.validate_signal(signal, self._config):
                logger.warning("Signal validation failed")
                return None
            # Обновление метрик
            self.metrics.total_signals += 1
            logger.info(f"Generated signal: {signal.direction} at {signal.entry_price}")
        return signal

    def _adapt_parameters(
        self, analysis: Dict[str, Any], data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Адаптивная настройка параметров стратегии."""
        if self.current_regime == MarketRegime.VOLATILE:
            # Увеличиваем стоп-лосс в волатильном рынке
            self._config.risk_per_trade *= 1.5
            analysis["adapted_risk"] = self._config.risk_per_trade
        elif self.current_regime == MarketRegime.RANGING:
            # Уменьшаем риск в боковом рынке
            self._config.risk_per_trade *= 0.7
            analysis["adapted_risk"] = self._config.risk_per_trade
        return analysis

    def get_strategy_info(self) -> Dict[str, Any]:
        """Возвращает информацию о стратегии."""
        return {
            "strategy_type": self._config.strategy_type.value,
            "current_regime": self.current_regime.value,
            "parameters": self._config.parameters,
            "metrics": self.metrics.__dict__,
            "modules": {
                "analysis": type(self.analysis_module).__name__,
                "signal_generator": type(self.signal_generator).__name__,
                "risk_manager": type(self.risk_manager).__name__,
                "regime_detector": type(self.regime_detector).__name__,
            },
        }

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Обновляет конфигурацию стратегии."""
        for key, value in new_config.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            elif key in self._config.parameters:
                self._config.parameters[key] = value
        logger.info(f"Updated strategy config: {new_config}")

    def __str__(self) -> str:
        """Строковое представление стратегии."""
        return f"ConsolidatedStrategy({self._config.strategy_type.value})"

    def __repr__(self) -> str:
        """Представление стратегии для отладки."""
        return (
            f"ConsolidatedStrategy(config={self._config}, regime={self.current_regime})"
        )
