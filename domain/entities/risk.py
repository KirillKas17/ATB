"""
Доменные сущности управления рисками.
"""

import ast
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from uuid import UUID, uuid4

from domain.type_definitions import RiskLevel as RiskLevelType
from domain.type_definitions import RiskMetrics as RiskMetricsTypedDict
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money


class RiskLevel(Enum):
    """Уровни риска."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


class RiskType(Enum):
    """Типы рисков."""

    MARKET_RISK = "market_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    CREDIT_RISK = "credit_risk"
    OPERATIONAL_RISK = "operational_risk"
    SYSTEMATIC_RISK = "systematic_risk"
    SPECIFIC_RISK = "specific_risk"


@dataclass(frozen=True)
class RiskProfile:
    """Профиль риска."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    risk_level: RiskLevel = RiskLevel.MEDIUM
    max_risk_per_trade: Decimal = Decimal("0.02")  # 2%
    max_daily_loss: Decimal = Decimal("0.05")  # 5%
    max_weekly_loss: Decimal = Decimal("0.15")  # 15%
    max_portfolio_risk: Decimal = Decimal("0.10")  # 10%
    max_correlation: Decimal = Decimal("0.7")  # 70%
    max_leverage: Decimal = Decimal("3.0")  # 3x
    min_risk_reward_ratio: Decimal = Decimal("1.5")  # 1.5:1
    max_drawdown: Decimal = Decimal("0.20")  # 20%
    position_sizing_method: str = "kelly"  # kelly, fixed, volatility
    stop_loss_method: str = "atr"  # atr, percentage, fixed
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, str] = field(default_factory=dict)

    def is_risk_level_acceptable(self, current_risk: Decimal) -> bool:
        """Проверка приемлемости уровня риска."""
        return current_risk <= self.max_risk_per_trade

    def is_daily_loss_acceptable(self, current_loss: Decimal) -> bool:
        """Проверка приемлемости дневного убытка."""
        return current_loss <= self.max_daily_loss

    def is_weekly_loss_acceptable(self, current_loss: Decimal) -> bool:
        """Проверка приемлемости недельного убытка."""
        return current_loss <= self.max_weekly_loss

    def is_portfolio_risk_acceptable(self, current_risk: Decimal) -> bool:
        """Проверка приемлемости риска портфеля."""
        return current_risk <= self.max_portfolio_risk

    def is_correlation_acceptable(self, correlation: Decimal) -> bool:
        """Проверка приемлемости корреляции."""
        return abs(correlation) <= self.max_correlation

    def is_leverage_acceptable(self, current_leverage: Decimal) -> bool:
        """Проверка приемлемости кредитного плеча."""
        return current_leverage <= self.max_leverage

    def is_risk_reward_acceptable(self, risk_reward_ratio: Decimal) -> bool:
        """Проверка приемлемости соотношения риск/доходность."""
        return risk_reward_ratio >= self.min_risk_reward_ratio

    def is_drawdown_acceptable(self, current_drawdown: Decimal) -> bool:
        """Проверка приемлемости просадки."""
        return current_drawdown <= self.max_drawdown

    def to_dict(self) -> Dict[str, str]:
        """Преобразование в словарь."""
        return {
            "id": str(self.id),
            "name": self.name,
            "risk_level": self.risk_level.value,
            "max_risk_per_trade": str(self.max_risk_per_trade),
            "max_daily_loss": str(self.max_daily_loss),
            "max_weekly_loss": str(self.max_weekly_loss),
            "max_portfolio_risk": str(self.max_portfolio_risk),
            "max_correlation": str(self.max_correlation),
            "max_leverage": str(self.max_leverage),
            "min_risk_reward_ratio": str(self.min_risk_reward_ratio),
            "max_drawdown": str(self.max_drawdown),
            "position_sizing_method": self.position_sizing_method,
            "stop_loss_method": self.stop_loss_method,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": str(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "RiskProfile":
        """Создание из словаря."""
        return cls(
            id=UUID(data["id"]),
            name=data["name"],
            risk_level=RiskLevel(data["risk_level"]),
            max_risk_per_trade=Decimal(data["max_risk_per_trade"]),
            max_daily_loss=Decimal(data["max_daily_loss"]),
            max_weekly_loss=Decimal(data["max_weekly_loss"]),
            max_portfolio_risk=Decimal(data["max_portfolio_risk"]),
            max_correlation=Decimal(data["max_correlation"]),
            max_leverage=Decimal(data["max_leverage"]),
            min_risk_reward_ratio=Decimal(data["min_risk_reward_ratio"]),
            max_drawdown=Decimal(data["max_drawdown"]),
            position_sizing_method=data["position_sizing_method"],
            stop_loss_method=data["stop_loss_method"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=ast.literal_eval(data.get("metadata", "{}")),
        )


@runtime_checkable
class RiskMetricsProtocol(Protocol):
    def get_risk_score(self) -> Decimal: ...
    def is_portfolio_healthy(self, risk_profile: "RiskProfile") -> bool: ...


@dataclass(frozen=True)
class RiskMetrics:
    """Метрики риска."""

    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)

    # Портфельные метрики
    portfolio_value: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USD)
    )
    portfolio_risk: Decimal = Decimal("0")
    portfolio_beta: Decimal = Decimal("0")
    portfolio_volatility: Decimal = Decimal("0")
    portfolio_var_95: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USD)
    )
    portfolio_cvar_95: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USD)
    )

    # Метрики просадки
    current_drawdown: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    drawdown_duration: int = 0

    # Метрики доходности
    total_return: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    sortino_ratio: Decimal = Decimal("0")
    calmar_ratio: Decimal = Decimal("0")

    # Метрики позиций
    total_positions: int = 0
    long_positions: int = 0
    short_positions: int = 0
    total_exposure: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USD)
    )
    net_exposure: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USD)
    )

    # Метрики корреляции
    avg_correlation: Decimal = Decimal("0")
    max_correlation: Decimal = Decimal("0")

    # Метрики ликвидности
    liquidity_ratio: Decimal = Decimal("0")
    bid_ask_spread: Decimal = Decimal("0")

    # Метрики концентрации
    concentration_ratio: Decimal = Decimal("0")
    herfindahl_index: Decimal = Decimal("0")

    # Метрики стресс-тестирования
    stress_test_score: Decimal = Decimal("0")
    scenario_analysis_score: Decimal = Decimal("0")

    metadata: Dict[str, str] = field(default_factory=dict)

    def is_portfolio_healthy(self, risk_profile: RiskProfile) -> bool:
        """Проверка здоровья портфеля."""
        return (
            self.portfolio_risk <= risk_profile.max_portfolio_risk
            and self.current_drawdown <= risk_profile.max_drawdown
            and self.avg_correlation <= risk_profile.max_correlation
        )

    def get_risk_score(self) -> Decimal:
        """Получение общего рискового скора."""
        # Простая формула рискового скора
        risk_score = (
            self.portfolio_risk * Decimal("0.3")
            + self.current_drawdown * Decimal("0.3")
            + self.portfolio_volatility * Decimal("0.2")
            + (Decimal("1") - self.liquidity_ratio) * Decimal("0.1")
            + self.concentration_ratio * Decimal("0.1")
        )
        return min(risk_score, Decimal("1"))

    def get_risk_level(self) -> RiskLevel:
        """Получение уровня риска на основе метрик."""
        risk_score = self.get_risk_score()

        if risk_score <= Decimal("0.2"):
            return RiskLevel.VERY_LOW
        elif risk_score <= Decimal("0.4"):
            return RiskLevel.LOW
        elif risk_score <= Decimal("0.6"):
            return RiskLevel.MEDIUM
        elif risk_score <= Decimal("0.8"):
            return RiskLevel.HIGH
        elif risk_score <= Decimal("0.9"):
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.EXTREME

    def to_dict(self) -> Dict[str, str]:
        """Преобразование в словарь."""
        return {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "portfolio_value": str(self.portfolio_value.amount),
            "portfolio_risk": str(self.portfolio_risk),
            "portfolio_beta": str(self.portfolio_beta),
            "portfolio_volatility": str(self.portfolio_volatility),
            "portfolio_var_95": str(self.portfolio_var_95.amount),
            "portfolio_cvar_95": str(self.portfolio_cvar_95.amount),
            "current_drawdown": str(self.current_drawdown),
            "max_drawdown": str(self.max_drawdown),
            "drawdown_duration": str(self.drawdown_duration),
            "total_return": str(self.total_return),
            "sharpe_ratio": str(self.sharpe_ratio),
            "sortino_ratio": str(self.sortino_ratio),
            "calmar_ratio": str(self.calmar_ratio),
            "total_positions": str(self.total_positions),
            "long_positions": str(self.long_positions),
            "short_positions": str(self.short_positions),
            "total_exposure": str(self.total_exposure.amount),
            "net_exposure": str(self.net_exposure.amount),
            "avg_correlation": str(self.avg_correlation),
            "max_correlation": str(self.max_correlation),
            "liquidity_ratio": str(self.liquidity_ratio),
            "bid_ask_spread": str(self.bid_ask_spread),
            "concentration_ratio": str(self.concentration_ratio),
            "herfindahl_index": str(self.herfindahl_index),
            "stress_test_score": str(self.stress_test_score),
            "scenario_analysis_score": str(self.scenario_analysis_score),
            "metadata": str(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "RiskMetrics":
        """Создание из словаря."""
        return cls(
            id=UUID(data["id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            portfolio_value=Money(Decimal(data["portfolio_value"]), Currency.USD),
            portfolio_risk=Decimal(data["portfolio_risk"]),
            portfolio_beta=Decimal(data["portfolio_beta"]),
            portfolio_volatility=Decimal(data["portfolio_volatility"]),
            portfolio_var_95=Money(Decimal(data["portfolio_var_95"]), Currency.USD),
            portfolio_cvar_95=Money(Decimal(data["portfolio_cvar_95"]), Currency.USD),
            current_drawdown=Decimal(data["current_drawdown"]),
            max_drawdown=Decimal(data["max_drawdown"]),
            drawdown_duration=int(data["drawdown_duration"]),
            total_return=Decimal(data["total_return"]),
            sharpe_ratio=Decimal(data["sharpe_ratio"]),
            sortino_ratio=Decimal(data["sortino_ratio"]),
            calmar_ratio=Decimal(data["calmar_ratio"]),
            total_positions=int(data["total_positions"]),
            long_positions=int(data["long_positions"]),
            short_positions=int(data["short_positions"]),
            total_exposure=Money(Decimal(data["total_exposure"]), Currency.USD),
            net_exposure=Money(Decimal(data["net_exposure"]), Currency.USD),
            avg_correlation=Decimal(data["avg_correlation"]),
            max_correlation=Decimal(data["max_correlation"]),
            liquidity_ratio=Decimal(data["liquidity_ratio"]),
            bid_ask_spread=Decimal(data["bid_ask_spread"]),
            concentration_ratio=Decimal(data["concentration_ratio"]),
            herfindahl_index=Decimal(data["herfindahl_index"]),
            stress_test_score=Decimal(data["stress_test_score"]),
            scenario_analysis_score=Decimal(data["scenario_analysis_score"]),
            metadata=ast.literal_eval(data.get("metadata", "{}")),
        )


class RiskManager:
    """Менеджер рисков - основной агрегат управления рисками."""

    def __init__(self, risk_profile: RiskProfile, name: str = ""):
        self.id = str(uuid4())
        self.name = name
        self.risk_profile = risk_profile
        self.current_metrics: Optional[RiskMetrics] = None
        self.risk_history: List[RiskMetrics] = []
        self.permissions: List[str] = []

    def update_metrics(self, metrics: RiskMetrics) -> None:
        """Обновление метрик риска."""
        self.current_metrics = metrics
        self.risk_history.append(metrics)

        # Ограничиваем историю последними 100 записями
        if len(self.risk_history) > 100:
            self.risk_history = self.risk_history[-100:]

    def validate_trade(self, trade_value: Money, portfolio_value: Money) -> bool:
        """Валидация торговой операции."""
        if not self.current_metrics:
            return True

        # Проверка риска на сделку
        trade_risk = trade_value.value / portfolio_value.value
        if trade_risk > self.risk_profile.max_risk_per_trade:
            return False

        # Проверка общего риска портфеля
        if self.current_metrics.portfolio_risk > self.risk_profile.max_portfolio_risk:
            return False

        return True

    def get_position_size(
        self, available_capital: Money, risk_per_trade: Decimal
    ) -> Money:
        """Расчет размера позиции на основе риск-профиля."""
        if self.risk_profile.position_sizing_method == "kelly":
            # Формула Келли
            if self.current_metrics and self.current_metrics.sharpe_ratio > 0:
                kelly_fraction = self.current_metrics.sharpe_ratio / 100
                return Money(available_capital.value * kelly_fraction, Currency.USD)

        # Фиксированный размер
        return Money(available_capital.value * risk_per_trade, Currency.USD)

    def should_stop_trading(self) -> bool:
        """Проверка необходимости остановки торговли."""
        if not self.current_metrics:
            return False

        # Проверка просадки
        if self.current_metrics.current_drawdown > self.risk_profile.max_drawdown:
            return True

        # Проверка дневного убытка
        if self.current_metrics.total_return < -self.risk_profile.max_daily_loss:
            return True

        return False

    def get_risk_alerts(self) -> List[str]:
        """Получение предупреждений о рисках."""
        alerts: List[str] = []

        if not self.current_metrics:
            return alerts

        # Проверка просадки
        if (
            self.current_metrics.current_drawdown
            > self.risk_profile.max_drawdown * Decimal("0.8")
        ):
            alerts.append(f"High drawdown: {self.current_metrics.current_drawdown:.2%}")

        # Проверка волатильности
        if self.current_metrics.portfolio_volatility > Decimal("0.3"):
            alerts.append(
                f"High volatility: {self.current_metrics.portfolio_volatility:.2%}"
            )

        # Проверка корреляции
        if (
            self.current_metrics.avg_correlation
            > self.risk_profile.max_correlation * Decimal("0.8")
        ):
            alerts.append(
                f"High correlation: {self.current_metrics.avg_correlation:.2%}"
            )

        return alerts

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "risk_profile": self.risk_profile.to_dict(),
            "current_metrics": (
                self.current_metrics.to_dict() if self.current_metrics else None
            ),
            "risk_history_count": len(self.risk_history),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskManager":
        """Создание из словаря."""
        risk_profile = RiskProfile.from_dict(data["risk_profile"])
        risk_manager = cls(risk_profile)

        if data.get("current_metrics"):
            risk_manager.current_metrics = RiskMetrics.from_dict(
                data["current_metrics"]
            )

        return risk_manager
