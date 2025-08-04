"""
Типы и конфигурации стратегий.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Final, List, Literal, Optional, Union
from uuid import UUID

from domain.type_definitions import (
    ConfidenceLevel,
    MetadataDict,
    PerformanceScore,
    RiskLevel,
    StrategyConfig,
    StrategyId,
    TradingPair,
)


class StrategyCategory(Enum):
    """Категории стратегий."""

    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    ARBITRAGE = "arbitrage"
    GRID = "grid"
    MARTINGALE = "martingale"
    HEDGING = "hedging"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    MACHINE_LEARNING = "machine_learning"
    QUANTITATIVE = "quantitative"


class RiskProfile(Enum):
    """Профили риска."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"


class TimeHorizon(Enum):
    """Временные горизонты."""

    ULTRA_SHORT = "ultra_short"  # секунды-минуты
    SHORT = "short"  # минуты-часы
    MEDIUM = "medium"  # часы-дни
    LONG = "long"  # дни-недели
    VERY_LONG = "very_long"  # недели-месяцы


class MarketCondition(Enum):
    """Рыночные условия."""

    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    TRENDING = "trending"
    RANGING = "ranging"
    BREAKOUT = "breakout"


class Timeframe(Enum):
    """Временные фреймы."""

    TICK = "tick"
    SECOND_1 = "1s"
    SECOND_5 = "5s"
    SECOND_15 = "15s"
    SECOND_30 = "30s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


class MarketRegime(Enum):
    """Режимы рынка."""

    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"


@dataclass
class StrategyParameters:
    """Базовые параметры стратегии."""

    # Основные параметры
    confidence_threshold: Decimal = Decimal("0.6")
    risk_level: RiskLevel = RiskLevel(Decimal("0.5"))
    max_position_size: Decimal = Decimal("0.1")
    stop_loss: Decimal = Decimal("0.02")
    take_profit: Decimal = Decimal("0.04")

    # Параметры исполнения
    max_signals: int = 5
    signal_cooldown: int = 300  # секунды
    execution_timeout: int = 30  # секунды
    max_slippage: Decimal = Decimal("0.001")

    # Параметры мониторинга
    enable_logging: bool = True
    enable_metrics: bool = True
    enable_alerts: bool = True

    # Метаданные
    version: str = "1.0.0"
    author: str = "Unknown"
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            "confidence_threshold": str(self.confidence_threshold),
            "risk_level": str(self.risk_level),
            "max_position_size": str(self.max_position_size),
            "stop_loss": str(self.stop_loss),
            "take_profit": str(self.take_profit),
            "max_signals": self.max_signals,
            "signal_cooldown": self.signal_cooldown,
            "execution_timeout": self.execution_timeout,
            "max_slippage": str(self.max_slippage),
            "enable_logging": self.enable_logging,
            "enable_metrics": self.enable_metrics,
            "enable_alerts": self.enable_alerts,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyParameters":
        """Создать из словаря."""
        return cls(
            confidence_threshold=Decimal(str(data.get("confidence_threshold", "0.6"))),
            risk_level=RiskLevel(Decimal(str(data.get("risk_level", "0.5")))),
            max_position_size=Decimal(str(data.get("max_position_size", "0.1"))),
            stop_loss=Decimal(str(data.get("stop_loss", "0.02"))),
            take_profit=Decimal(str(data.get("take_profit", "0.04"))),
            max_signals=data.get("max_signals", 5),
            signal_cooldown=data.get("signal_cooldown", 300),
            execution_timeout=data.get("execution_timeout", 30),
            max_slippage=Decimal(str(data.get("max_slippage", "0.001"))),
            enable_logging=data.get("enable_logging", True),
            enable_metrics=data.get("enable_metrics", True),
            enable_alerts=data.get("enable_alerts", True),
            version=data.get("version", "1.0.0"),
            author=data.get("author", "Unknown"),
            description=data.get("description", ""),
            tags=data.get("tags", []),
        )


@dataclass
class TrendFollowingParams(StrategyParameters):
    """Параметры стратегии следования за трендом."""

    # Периоды для скользящих средних
    short_period: int = 10
    long_period: int = 20

    # Параметры RSI
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70

    # Параметры MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Параметры тренда
    trend_strength_threshold: Decimal = Decimal("0.7")
    trend_confirmation_period: int = 3

    # Параметры объема
    volume_confirmation: bool = True
    volume_threshold: Decimal = Decimal("1.5")

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "short_period": self.short_period,
                "long_period": self.long_period,
                "rsi_period": self.rsi_period,
                "rsi_oversold": self.rsi_oversold,
                "rsi_overbought": self.rsi_overbought,
                "macd_fast": self.macd_fast,
                "macd_slow": self.macd_slow,
                "macd_signal": self.macd_signal,
                "trend_strength_threshold": str(self.trend_strength_threshold),
                "trend_confirmation_period": self.trend_confirmation_period,
                "volume_confirmation": self.volume_confirmation,
                "volume_threshold": str(self.volume_threshold),
            }
        )
        return base_dict


@dataclass
class MeanReversionParams(StrategyParameters):
    """Параметры стратегии возврата к среднему."""

    # Период для расчета среднего
    lookback_period: int = 50

    # Порог отклонения
    deviation_threshold: Decimal = Decimal("2.0")

    # Параметры RSI
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70

    # Параметры Bollinger Bands
    bb_period: int = 20
    bb_std_dev: Decimal = Decimal("2.0")

    # Параметры фильтрации
    min_reversion_probability: Decimal = Decimal("0.6")
    max_holding_period: int = 24  # часы

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "lookback_period": self.lookback_period,
                "deviation_threshold": str(self.deviation_threshold),
                "rsi_period": self.rsi_period,
                "rsi_oversold": self.rsi_oversold,
                "rsi_overbought": self.rsi_overbought,
                "bb_period": self.bb_period,
                "bb_std_dev": str(self.bb_std_dev),
                "min_reversion_probability": str(self.min_reversion_probability),
                "max_holding_period": self.max_holding_period,
            }
        )
        return base_dict


@dataclass
class BreakoutParams(StrategyParameters):
    """Параметры стратегии пробоя."""

    # Порог пробоя
    breakout_threshold: Decimal = Decimal("1.5")

    # Параметры объема
    volume_multiplier: Decimal = Decimal("2.0")
    volume_confirmation_period: int = 3

    # Параметры уровней
    support_resistance_period: int = 20
    level_tolerance: Decimal = Decimal("0.001")

    # Параметры подтверждения
    confirmation_period: int = 2
    false_breakout_filter: bool = True

    # Параметры волатильности
    min_volatility: Decimal = Decimal("0.01")
    max_volatility: Decimal = Decimal("0.1")

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "breakout_threshold": str(self.breakout_threshold),
                "volume_multiplier": str(self.volume_multiplier),
                "volume_confirmation_period": self.volume_confirmation_period,
                "support_resistance_period": self.support_resistance_period,
                "level_tolerance": str(self.level_tolerance),
                "confirmation_period": self.confirmation_period,
                "false_breakout_filter": self.false_breakout_filter,
                "min_volatility": str(self.min_volatility),
                "max_volatility": str(self.max_volatility),
            }
        )
        return base_dict


@dataclass
class ScalpingParams(StrategyParameters):
    """Параметры скальпинг стратегии."""

    # Пороги прибыли и убытка
    profit_threshold: Decimal = Decimal("0.001")
    stop_loss: Decimal = Decimal("0.0005")

    # Временные параметры
    max_hold_time: int = 300  # секунды
    min_hold_time: int = 10  # секунды

    # Параметры волатильности
    min_volatility: Decimal = Decimal("0.0001")
    max_volatility: Decimal = Decimal("0.01")

    # Параметры объема
    min_volume: Decimal = Decimal("1000")
    volume_spike_threshold: Decimal = Decimal("2.0")

    # Параметры исполнения
    execution_timeout: int = 5  # секунды
    max_slippage: Decimal = Decimal("0.0001")

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "profit_threshold": str(self.profit_threshold),
                "stop_loss": str(self.stop_loss),
                "max_hold_time": self.max_hold_time,
                "min_hold_time": self.min_hold_time,
                "min_volatility": str(self.min_volatility),
                "max_volatility": str(self.max_volatility),
                "min_volume": str(self.min_volume),
                "volume_spike_threshold": str(self.volume_spike_threshold),
                "execution_timeout": self.execution_timeout,
                "max_slippage": str(self.max_slippage),
            }
        )
        return base_dict


@dataclass
class ArbitrageParams(StrategyParameters):
    """Параметры арбитражной стратегии."""

    # Минимальный спред
    min_spread: Decimal = Decimal("0.001")

    # Максимальный слиппаж
    max_slippage: Decimal = Decimal("0.0005")

    # Временные параметры
    execution_timeout: int = 10  # секунды
    max_hold_time: int = 60  # секунды

    # Параметры ликвидности
    min_liquidity: Decimal = Decimal("10000")
    max_order_size: Decimal = Decimal("1000")

    # Параметры комиссий
    max_total_fees: Decimal = Decimal("0.002")
    include_fees_in_calculation: bool = True

    # Параметры мониторинга
    exchange_monitoring_interval: int = 1  # секунды
    price_update_frequency: int = 1  # секунды

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "min_spread": str(self.min_spread),
                "max_slippage": str(self.max_slippage),
                "execution_timeout": self.execution_timeout,
                "max_hold_time": self.max_hold_time,
                "min_liquidity": str(self.min_liquidity),
                "max_order_size": str(self.max_order_size),
                "max_total_fees": str(self.max_total_fees),
                "include_fees_in_calculation": self.include_fees_in_calculation,
                "exchange_monitoring_interval": self.exchange_monitoring_interval,
                "price_update_frequency": self.price_update_frequency,
            }
        )
        return base_dict


@dataclass
class StrategyConfigData:
    """Конфигурация стратегии."""

    # Основные параметры
    strategy_id: StrategyId
    name: str
    category: StrategyCategory
    risk_profile: RiskProfile
    timeframe: Timeframe

    # Торговые параметры
    trading_pairs: List[TradingPair]
    parameters: StrategyParameters

    # Параметры исполнения
    is_active: bool = True
    auto_start: bool = False
    max_concurrent_positions: int = 5

    # Параметры мониторинга
    enable_monitoring: bool = True
    monitoring_interval: int = 60  # секунды
    alert_thresholds: Dict[str, Decimal] = field(default_factory=dict)

    # Метаданные
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            "strategy_id": str(self.strategy_id),
            "name": self.name,
            "category": self.category.value,
            "risk_profile": self.risk_profile.value,
            "timeframe": self.timeframe.value,
            "trading_pairs": [str(pair) for pair in self.trading_pairs],
            "parameters": self.parameters.to_dict(),
            "is_active": self.is_active,
            "auto_start": self.auto_start,
            "max_concurrent_positions": self.max_concurrent_positions,
            "enable_monitoring": self.enable_monitoring,
            "monitoring_interval": self.monitoring_interval,
            "alert_thresholds": {k: str(v) for k, v in self.alert_thresholds.items()},
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "description": self.description,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyConfigData":
        """Создать из словаря."""
        from uuid import UUID

        return cls(
            strategy_id=StrategyId(UUID(data["strategy_id"])),
            name=data["name"],
            category=StrategyCategory(data["category"]),
            risk_profile=RiskProfile(data["risk_profile"]),
            timeframe=Timeframe(data["timeframe"]),
            trading_pairs=[TradingPair(pair) for pair in data["trading_pairs"]],
            parameters=StrategyParameters.from_dict(data["parameters"]),
            is_active=data.get("is_active", True),
            auto_start=data.get("auto_start", False),
            max_concurrent_positions=data.get("max_concurrent_positions", 5),
            enable_monitoring=data.get("enable_monitoring", True),
            monitoring_interval=data.get("monitoring_interval", 60),
            alert_thresholds={
                k: Decimal(v) for k, v in data.get("alert_thresholds", {}).items()
            },
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            tags=data.get("tags", []),
        )


@dataclass
class StrategyMetrics:
    """Метрики стратегии."""

    # Основные метрики
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: Decimal = Decimal("0")

    # Финансовые метрики
    total_pnl: Decimal = Decimal("0")
    total_commission: Decimal = Decimal("0")
    net_pnl: Decimal = Decimal("0")
    profit_factor: Decimal = Decimal("0")

    # Риск-метрики
    max_drawdown: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    sortino_ratio: Decimal = Decimal("0")
    calmar_ratio: Decimal = Decimal("0")

    # Временные метрики
    avg_trade_duration: float = 0.0
    avg_execution_time: float = 0.0
    total_runtime: float = 0.0

    # Статистические метрики
    avg_trade_size: Decimal = Decimal("0")
    largest_win: Decimal = Decimal("0")
    largest_loss: Decimal = Decimal("0")
    consecutive_wins: int = 0
    consecutive_losses: int = 0

    # Временные метки
    first_trade: Optional[datetime] = None
    last_trade: Optional[datetime] = None
    last_update: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": str(self.win_rate),
            "total_pnl": str(self.total_pnl),
            "total_commission": str(self.total_commission),
            "net_pnl": str(self.net_pnl),
            "profit_factor": str(self.profit_factor),
            "max_drawdown": str(self.max_drawdown),
            "sharpe_ratio": str(self.sharpe_ratio),
            "sortino_ratio": str(self.sortino_ratio),
            "calmar_ratio": str(self.calmar_ratio),
            "avg_trade_duration": self.avg_trade_duration,
            "avg_execution_time": self.avg_execution_time,
            "total_runtime": self.total_runtime,
            "avg_trade_size": str(self.avg_trade_size),
            "largest_win": str(self.largest_win),
            "largest_loss": str(self.largest_loss),
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "first_trade": self.first_trade.isoformat() if self.first_trade else None,
            "last_trade": self.last_trade.isoformat() if self.last_trade else None,
            "last_update": self.last_update.isoformat(),
        }

    def update_metrics(
        self,
        trade_pnl: Decimal,
        trade_commission: Decimal,
        trade_duration: float,
        execution_time: float,
    ) -> None:
        """Обновить метрики на основе новой сделки."""
        self.total_trades += 1
        self.total_pnl += trade_pnl
        self.total_commission += trade_commission
        self.net_pnl = self.total_pnl - self.total_commission

        if trade_pnl > 0:
            self.winning_trades += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            if trade_pnl > self.largest_win:
                self.largest_win = trade_pnl
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            if trade_pnl < self.largest_loss:
                self.largest_loss = trade_pnl

        # Обновляем win rate
        if self.total_trades > 0:
            self.win_rate = Decimal(str(self.winning_trades / self.total_trades))

        # Обновляем profit factor
        if self.losing_trades > 0:
            total_wins = sum(1 for _ in range(self.winning_trades))  # Упрощенно
            total_losses = sum(1 for _ in range(self.losing_trades))  # Упрощенно
            if total_losses > 0:
                self.profit_factor = Decimal(str(total_wins / total_losses))

        # Обновляем средние значения
        if self.total_trades == 1:
            self.avg_trade_duration = trade_duration
            self.avg_execution_time = execution_time
            self.avg_trade_size = abs(trade_pnl)
        else:
            self.avg_trade_duration = (
                self.avg_trade_duration * (self.total_trades - 1) + trade_duration
            ) / self.total_trades
            self.avg_execution_time = (
                self.avg_execution_time * (self.total_trades - 1) + execution_time
            ) / self.total_trades
            self.avg_trade_size = (
                self.avg_trade_size * (self.total_trades - 1) + abs(trade_pnl)
            ) / self.total_trades

        # Обновляем временные метки
        if not self.first_trade:
            self.first_trade = datetime.now()
        self.last_trade = datetime.now()
        self.last_update = datetime.now()


# Алиасы для обратной совместимости
TrendFollowingParameters = TrendFollowingParams
MeanReversionParameters = MeanReversionParams
BreakoutParameters = BreakoutParams
ScalpingParameters = ScalpingParams
ArbitrageParameters = ArbitrageParams
StrategyConfiguration = StrategyConfigData

# Константы по умолчанию
DEFAULT_TREND_FOLLOWING_CONFIG = TrendFollowingParams()
DEFAULT_MEAN_REVERSION_CONFIG = MeanReversionParams()
DEFAULT_BREAKOUT_CONFIG = BreakoutParams()
DEFAULT_SCALPING_CONFIG = ScalpingParams()
DEFAULT_ARBITRAGE_CONFIG = ArbitrageParams()
