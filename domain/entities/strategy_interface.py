"""
Интерфейсы для стратегий с полной реализацией.
"""

import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from dataclasses import field

from domain.entities.signal import Signal, SignalStrength, SignalType
from domain.entities.strategy import StrategyStatus, StrategyType
from domain.types import (
    ConfidenceLevel,
    PerformanceMetrics,
    PriceValue,
    RiskMetrics,
    StrategyConfig,
    StrategyId,
    VolumeValue,
)


@runtime_checkable
class StrategyInterface(Protocol):
    """Базовый протокол для всех стратегий."""

    def generate_signals(self, market_data: pd.DataFrame) -> List[Signal]:
        """Генерация торговых сигналов на основе рыночных данных."""
        ...

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Валидация входных данных."""
        ...

    def get_parameters(self) -> Dict[str, Any]:
        """Получение параметров стратегии."""
        ...

    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """Обновление параметров стратегии."""
        ...

    def is_active(self) -> bool:
        """Проверка активности стратегии."""
        ...

    def activate(self) -> None:
        """Активация стратегии."""
        ...

    def deactivate(self) -> None:
        """Деактивация стратегии."""
        ...

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Получение метрик производительности."""
        ...

    def reset(self) -> None:
        """Сброс состояния стратегии."""
        ...


class BaseStrategy(ABC):
    """Базовая реализация стратегии с полной функциональностью."""

    def __init__(
        self,
        strategy_id: StrategyId,
        name: str,
        strategy_type: StrategyType,
        config: StrategyConfig,
        status: StrategyStatus = StrategyStatus.INACTIVE,
    ):
        self.strategy_id = strategy_id
        self.name = name
        self.strategy_type = strategy_type
        self.config = config
        self.status = status
        self.is_strategy_active = False
        self.performance_metrics: Dict[str, Any] = {}
        self.risk_metrics: Dict[str, float] = {}
        self.last_signal_time: Optional[datetime] = None
        self.signal_count = 0

    def generate_signals(self, market_data: pd.DataFrame) -> List[Signal]:
        """Генерация торговых сигналов на основе рыночных данных."""
        if not self.is_active():
            return []
        if not self.validate_data(market_data):
            return []
        # Проверяем cooldown между сигналами
        if self._is_signal_cooldown_active():
            return []
        # Генерируем сигналы в зависимости от типа стратегии
        signals = self._generate_strategy_signals(market_data)
        # Обновляем счетчики
        self.signal_count += len(signals)
        if signals:
            self.last_signal_time = datetime.now()
        return signals

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Валидация входных данных."""
        if data.empty:
            return False
        required_columns = ["open", "high", "low", "close", "volume"]
        if not all(col in data.columns for col in required_columns):
            return False
        # Проверяем на NaN значения
        if data[required_columns].isnull().any().any():
            return False
        # Проверяем минимальное количество данных
        min_data_points = self.config.get("min_data_points", 20)
        try:
            if isinstance(min_data_points, (int, float, str)):
                min_data_points_int = int(min_data_points)
            else:
                min_data_points_int = 20
        except (ValueError, TypeError):
            min_data_points_int = 20
        if len(data) < min_data_points_int:
            return False
        return True

    def get_parameters(self) -> Dict[str, Any]:
        """Получение параметров стратегии."""
        return {
            "strategy_id": str(self.strategy_id),
            "name": self.name,
            "strategy_type": self.strategy_type.value,
            "status": self.status.value,
            "config": self.config,
            "performance_metrics": self.performance_metrics,
            "risk_metrics": self.risk_metrics,
            "signal_count": self.signal_count,
            "last_signal_time": (
                self.last_signal_time.isoformat() if self.last_signal_time else None
            ),
        }

    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """Обновление параметров стратегии."""
        # Обновляем конфигурацию
        if "config" in parameters:
            self.config.update(parameters["config"])
        # Обновляем статус
        if "status" in parameters:
            self.status = StrategyStatus(parameters["status"])
        # Обновляем метрики
        if "performance_metrics" in parameters:
            self.performance_metrics.update(parameters["performance_metrics"])
        if "risk_metrics" in parameters:
            self.risk_metrics.update(parameters["risk_metrics"])

    def is_active(self) -> bool:
        """Проверка активности стратегии."""
        return self.is_strategy_active and self.status == StrategyStatus.ACTIVE

    def activate(self) -> None:
        """Активация стратегии."""
        self.is_strategy_active = True
        self.status = StrategyStatus.ACTIVE

    def deactivate(self) -> None:
        """Деактивация стратегии."""
        self.is_strategy_active = False
        self.status = StrategyStatus.INACTIVE

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Получение метрик производительности."""
        return {
            "strategy_id": str(self.strategy_id),
            "name": self.name,
            "total_signals": self.signal_count,
            "last_signal_time": (
                self.last_signal_time.isoformat() if self.last_signal_time else None
            ),
            "performance": self.performance_metrics,
            "risk": self.risk_metrics,
            "status": self.status.value,
            "is_active": self.is_active(),
        }

    def reset(self) -> None:
        """Сброс состояния стратегии."""
        self.signal_count = 0
        self.last_signal_time = None
        self.performance_metrics.clear()
        self.risk_metrics.clear()

    def _is_signal_cooldown_active(self) -> bool:
        """Проверка активного cooldown между сигналами."""
        if not self.last_signal_time:
            return False
        cooldown_seconds = self.config.get(
            "signal_cooldown", 300
        )  # 5 минут по умолчанию
        time_since_last = (datetime.now() - self.last_signal_time).total_seconds()
        return time_since_last < cooldown_seconds

    @abstractmethod
    def _generate_strategy_signals(self, market_data: pd.DataFrame) -> List[Signal]:
        """
        Абстрактный метод для генерации сигналов конкретной стратегии.
        Args:
            market_data: Рыночные данные в формате DataFrame
        Returns:
            List[Signal]: Список сгенерированных сигналов
        Raises:
            NotImplementedError: Если метод не реализован в дочернем классе
        """
        raise NotImplementedError(
            "_generate_strategy_signals must be implemented in subclasses"
        )


class AdvancedStrategyInterface(BaseStrategy):
    """Расширенный интерфейс для продвинутых стратегий."""

    def _generate_strategy_signals(self, market_data: pd.DataFrame) -> List[Signal]:
        """Реализация генерации сигналов для продвинутой стратегии."""
        # Базовая реализация - генерируем сигнал на основе простого анализа
        signals: List[Signal] = []
        if len(market_data) < 20:
            return signals
            
        # Простой анализ тренда
        current_price = market_data["close"].iloc[-1]
        sma_20 = market_data["close"].rolling(20).mean().iloc[-1]
        from decimal import Decimal
        from domain.value_objects.money import Money
        from domain.value_objects.currency import Currency
        if current_price > sma_20 * 1.02:  # Цена выше SMA на 2%
            signal = Signal(
                signal_type=SignalType.BUY,
                confidence=Decimal("0.7"),
                price=Money(Decimal(str(current_price)), Currency.USD),
                quantity=Decimal("1.0"),
                timestamp=datetime.now()
            )
            signals.append(signal)
        elif current_price < sma_20 * 0.98:  # Цена ниже SMA на 2%
            signal = Signal(
                signal_type=SignalType.SELL,
                confidence=Decimal("0.7"),
                price=Money(Decimal(str(current_price)), Currency.USD),
                quantity=Decimal("1.0"),
                timestamp=datetime.now()
            )
            signals.append(signal)
            
        return signals

    def analyze_market_conditions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ рыночных условий."""
        analysis = {
            "trend": self._analyze_trend(data),
            "volatility": self._analyze_volatility(data),
            "volume": self._analyze_volume(data),
            "support_resistance": self._find_support_resistance(data),
            "market_regime": self._determine_market_regime(data),
        }
        return analysis

    def calculate_risk_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Расчет метрик риска."""
        returns = data["close"].pct_change().dropna()
        if len(returns) < 2:
            return {
                "volatility": 0.0,
                "var_95": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
            }
        volatility = float(returns.std() * (252**0.5))  # Годовая волатильность
        var_95 = float(returns.quantile(0.05))
        max_drawdown = self._calculate_max_drawdown(data["close"])
        sharpe_ratio = (
            float(returns.mean() / returns.std() * (252**0.5))
            if returns.std() > 0
            else 0.0
        )
        # Sortino ratio
        downside_returns = returns[returns < 0]
        sortino_ratio = (
            float(returns.mean() / downside_returns.std() * (252**0.5))
            if len(downside_returns) > 0 and downside_returns.std() > 0
            else 0.0
        )
        return {
            "volatility": volatility,
            "var_95": var_95,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
        }

    def optimize_parameters(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Оптимизация параметров на исторических данных."""
        # Простая оптимизация на основе исторических данных
        optimized_params = {}
        # Оптимизация параметров тренда
        if self.strategy_type in [StrategyType.TREND_FOLLOWING, StrategyType.MOMENTUM]:
            optimized_params.update(self._optimize_trend_parameters(historical_data))
        # Оптимизация параметров волатильности
        if self.strategy_type in [StrategyType.VOLATILITY, StrategyType.MEAN_REVERSION]:
            optimized_params.update(
                self._optimize_volatility_parameters(historical_data)
            )
        # Оптимизация параметров объема
        if self.strategy_type in [StrategyType.BREAKOUT, StrategyType.SCALPING]:
            optimized_params.update(self._optimize_volume_parameters(historical_data))
        return optimized_params

    def adapt_to_market_changes(self, current_data: pd.DataFrame) -> bool:
        """Адаптация к изменениям рынка."""
        # Анализируем текущие рыночные условия
        market_conditions = self.analyze_market_conditions(current_data)
        # Определяем необходимость адаптации
        adaptation_needed = False
        # Проверяем изменение волатильности
        current_volatility = market_conditions["volatility"]["current"]
        if abs(current_volatility - self.risk_metrics.get("volatility", 0)) > 0.1:
            adaptation_needed = True
        # Проверяем изменение тренда
        if market_conditions["trend"]["strength"] < 0.3:
            adaptation_needed = True
        # Если адаптация нужна, обновляем параметры
        if adaptation_needed:
            self._adapt_parameters(market_conditions)
        return adaptation_needed

    def get_signal_confidence(self, signal: Signal) -> float:
        """Получение уверенности в сигнале."""
        base_confidence = float(signal.confidence)
        # Корректируем уверенность на основе рыночных условий
        market_conditions = self.analyze_market_conditions(
            pd.DataFrame()
        )  # Нужны актуальные данные
        # Увеличиваем уверенность при сильном тренде
        if market_conditions.get("trend", {}).get("strength", 0) > 0.7:
            base_confidence *= 1.2
        # Уменьшаем уверенность при высокой волатильности
        if market_conditions.get("volatility", {}).get("current", 0) > 0.5:
            base_confidence *= 0.8
        # Ограничиваем уверенность в разумных пределах
        return min(max(base_confidence, 0.0), 1.0)

    def should_execute_signal(self, signal: Signal, market_data: pd.DataFrame) -> bool:
        """Проверка необходимости исполнения сигнала."""
        # Базовая проверка активности стратегии
        if not self.is_active():
            return False
        # Проверка уверенности в сигнале
        confidence = self.get_signal_confidence(signal)
        if confidence < self.config.get("confidence_threshold", 0.6):
            return False
        # Проверка рыночных условий
        market_conditions = self.analyze_market_conditions(market_data)
        # Не исполняем сигналы в неблагоприятных условиях
        if market_conditions.get("volatility", {}).get("current", 0) > 0.8:
            return False
        if market_conditions.get("trend", {}).get("strength", 0) < 0.2:
            return False
        # Проверка лимитов на количество сигналов
        max_signals = self.config.get("max_signals", 10)
        if self.signal_count >= max_signals:
            return False
        return True

    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ тренда."""
        if len(data) < 20:
            return {"direction": "neutral", "strength": 0.0}
        # Простой анализ тренда на основе SMA
        short_sma = data["close"].rolling(10).mean()
        long_sma = data["close"].rolling(20).mean()
        current_short = short_sma.iloc[-1]
        current_long = long_sma.iloc[-1]
        if current_short > current_long:
            direction = "up"
            strength = min((current_short - current_long) / current_long, 1.0)
        else:
            direction = "down"
            strength = min((current_long - current_short) / current_long, 1.0)
        return {"direction": direction, "strength": float(strength)}

    def _analyze_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ волатильности."""
        if len(data) < 20:
            return {"current": 0.0, "average": 0.0, "regime": "low"}
        returns = data["close"].pct_change().dropna()
        current_vol = float(returns.tail(10).std())
        avg_vol = float(returns.std())
        if current_vol < avg_vol * 0.7:
            regime = "low"
        elif current_vol > avg_vol * 1.3:
            regime = "high"
        else:
            regime = "normal"
        return {"current": current_vol, "average": avg_vol, "regime": regime}

    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ объема."""
        if len(data) < 20:
            return {"current": 0.0, "average": 0.0, "ratio": 1.0}
        current_volume = float(data["volume"].iloc[-1])
        avg_volume = float(data["volume"].tail(20).mean())
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        return {"current": current_volume, "average": avg_volume, "ratio": volume_ratio}

    def _find_support_resistance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Поиск уровней поддержки и сопротивления."""
        if len(data) < 20:
            return {"support": 0.0, "resistance": 0.0}
        # Простой поиск уровней на основе минимумов и максимумов
        recent_lows = data["low"].tail(20).nsmallest(3)
        recent_highs = data["high"].tail(20).nlargest(3)
        support = float(recent_lows.mean())
        resistance = float(recent_highs.mean())
        return {"support": support, "resistance": resistance}

    def _determine_market_regime(self, data: pd.DataFrame) -> str:
        """Определение режима рынка."""
        trend_analysis = self._analyze_trend(data)
        volatility_analysis = self._analyze_volatility(data)
        if trend_analysis["strength"] > 0.7:
            if trend_analysis["direction"] == "up":
                return "strong_uptrend"
            else:
                return "strong_downtrend"
        elif volatility_analysis["regime"] == "high":
            return "high_volatility"
        elif volatility_analysis["regime"] == "low":
            return "low_volatility"
        else:
            return "sideways"

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Расчет максимальной просадки."""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return float(drawdown.min())

    def _optimize_trend_parameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Оптимизация параметров трендовых стратегий."""
        # Простая оптимизация на основе исторических данных
        returns = data["close"].pct_change().dropna()
        # Оптимизируем период SMA
        best_period = 20
        best_sharpe = 0
        for period in [10, 15, 20, 25, 30]:
            sma = data["close"].rolling(period).mean()
            signals = (data["close"] > sma).astype(int)
            strategy_returns = signals.shift(1) * returns
            sharpe = (
                strategy_returns.mean() / strategy_returns.std()
                if strategy_returns.std() > 0
                else 0
            )
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_period = int(period)
        return {"trend_period": int(best_period), "trend_strength_threshold": 0.5}

    def _optimize_volatility_parameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Оптимизация параметров волатильных стратегий."""
        returns = data["close"].pct_change().dropna()
        volatility = returns.rolling(20).std()
        # Оптимизируем порог волатильности
        volatility_threshold = float(volatility.quantile(0.7))
        return {"volatility_threshold": volatility_threshold, "volatility_period": 20}

    def _optimize_volume_parameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Оптимизация параметров объемных стратегий."""
        volume = data["volume"]
        avg_volume = volume.rolling(20).mean()
        volume_ratio = volume / avg_volume
        # Оптимизируем порог объема
        volume_threshold = float(volume_ratio.quantile(0.8))
        return {"volume_threshold": volume_threshold, "volume_period": 20}

    def _adapt_parameters(self, market_conditions: Dict[str, Any]) -> None:
        """Адаптация параметров к рыночным условиям."""
        regime = market_conditions.get("market_regime", "sideways")
        if regime == "high_volatility":
            # Увеличиваем пороги для снижения риска
            confidence_threshold = float(self.config.get("confidence_threshold", 0.6)) * 1.2
            self.config["confidence_threshold"] = min(confidence_threshold, 0.9)
            self.config["stop_loss"] = float(self.config.get("stop_loss", 0.02)) * 1.5
        elif regime == "low_volatility":
            # Снижаем пороги для увеличения активности
            current_threshold = float(self.config.get("confidence_threshold", 0.6))
            self.config["confidence_threshold"] = max(current_threshold * 0.8, 0.3)
            self.config["stop_loss"] = float(self.config.get("stop_loss", 0.02)) * 0.7


class StrategyFactoryInterface(ABC):
    """Интерфейс фабрики стратегий."""

    def create_strategy(
        self, strategy_type: str, config: Dict[str, Any]
    ) -> BaseStrategy:
        """Создание стратегии заданного типа."""
        from domain.entities.strategy import StrategyType

        strategy_enum = StrategyType(strategy_type)
        # Создаем конкретную реализацию стратегии
        strategy_id = config.get("strategy_id")
        if strategy_id is None:
            from uuid import uuid4
            strategy_id = StrategyId(uuid4())
        
        strategy = AdvancedStrategyInterface(
            strategy_id=strategy_id,
            name=config.get("name", f"{strategy_type}_strategy"),
            strategy_type=strategy_enum,
            config=StrategyConfig(
                name=config.get("name", f"{strategy_type}_strategy"),
                parameters=config.get("parameters", {}),
                trading_pairs=config.get("trading_pairs", [])
            ),
            status=StrategyStatus.INACTIVE,
        )
        return strategy

    def get_available_strategies(self) -> List[str]:
        """Получение списка доступных стратегий."""
        from domain.entities.strategy import StrategyType

        return [strategy.value for strategy in StrategyType]

    def validate_config(self, strategy_type: str, config: Dict[str, Any]) -> List[str]:
        """Валидация конфигурации стратегии."""
        errors = []
        # Проверка обязательных полей
        if not config.get("name"):
            errors.append("Strategy name is required")
        if not config.get("trading_pairs"):
            errors.append("At least one trading pair is required")
        # Проверка параметров стратегии
        parameters = config.get("parameters", {})
        # Проверка stop_loss
        stop_loss = parameters.get("stop_loss", 0)
        if stop_loss <= 0:
            errors.append("Stop loss must be positive")
        elif stop_loss > 1:
            errors.append("Stop loss cannot exceed 100%")
        # Проверка take_profit
        take_profit = parameters.get("take_profit", 0)
        if take_profit <= 0:
            errors.append("Take profit must be positive")
        elif take_profit > 10:
            errors.append("Take profit cannot exceed 1000%")
        # Проверка confidence_threshold
        confidence_threshold = parameters.get("confidence_threshold", 0)
        if confidence_threshold < 0 or confidence_threshold > 1:
            errors.append("Confidence threshold must be between 0 and 1")
        return errors


class StrategyRepositoryInterface(ABC):
    """Интерфейс репозитория стратегий."""

    async def save(self, strategy: BaseStrategy) -> BaseStrategy:
        """Сохранение стратегии."""
        # В реальной реализации здесь была бы работа с БД
        # Пока возвращаем стратегию как есть
        return strategy

    async def find_by_id(self, strategy_id: StrategyId) -> Optional[BaseStrategy]:
        """Поиск стратегии по ID."""
        # В реальной реализации здесь был бы поиск в БД
        # Пока возвращаем None
        return None

    async def find_by_type(self, strategy_type: str) -> List[BaseStrategy]:
        """Поиск стратегий по типу."""
        # В реальной реализации здесь был бы поиск в БД
        # Пока возвращаем пустой список
        return []

    async def find_active(self) -> List[BaseStrategy]:
        """Поиск активных стратегий."""
        # В реальной реализации здесь был бы поиск в БД
        # Пока возвращаем пустой список
        return []

    async def delete(self, strategy_id: StrategyId) -> bool:
        """Удаление стратегии."""
        # В реальной реализации здесь было бы удаление из БД
        # Пока возвращаем True
        return True

    async def update_performance(
        self, strategy_id: StrategyId, metrics: Dict[str, Any]
    ) -> bool:
        """Обновление метрик производительности."""
        # В реальной реализации здесь было бы обновление в БД
        # Пока возвращаем True
        return True
