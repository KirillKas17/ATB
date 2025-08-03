"""
Доменная сущность RiskMetrics.
Представляет метрики риска портфеля.
"""

import ast
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Optional, Protocol, runtime_checkable
from uuid import UUID, uuid4

from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.timestamp import Timestamp


@dataclass
class RiskMetrics:
    """
    Метрики риска портфеля.
    Представляет собой совокупность показателей риска
    для анализа и управления портфелем.
    """

    id: UUID = field(default_factory=uuid4)
    portfolio_id: UUID = field(default_factory=uuid4)
    var_95: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USD))
    var_99: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USD))
    volatility: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    sortino_ratio: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    beta: Decimal = Decimal("1")
    alpha: Decimal = Decimal("0")
    skewness: Decimal = Decimal("0")
    kurtosis: Decimal = Decimal("0")
    correlation: Decimal = Decimal("0")
    created_at: Timestamp = field(default_factory=Timestamp.now)
    updated_at: Timestamp = field(default_factory=Timestamp.now)
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Инициализация после создания объекта.
        Валидация и нормализация полей.
        """
        # Базовая валидация на уровне типов обеспечивается TypeScript-стилем типизации
        # Дополнительная валидация может быть добавлена здесь по необходимости
        pass

    @property
    def risk_level(self) -> str:
        """Уровень риска на основе VaR."""
        if self.var_95.amount > 0.1:  # 10% от портфеля
            return "high"
        elif self.var_95.amount > 0.05:  # 5% от портфеля
            return "medium"
        else:
            return "low"

    @property
    def is_high_risk(self) -> bool:
        """Проверка высокого риска."""
        return self.risk_level == "high"

    @property
    def is_medium_risk(self) -> bool:
        """Проверка среднего риска."""
        return self.risk_level == "medium"

    @property
    def is_low_risk(self) -> bool:
        """Проверка низкого риска."""
        return self.risk_level == "low"

    def update_metrics(
        self,
        var_95: Optional[Money] = None,
        var_99: Optional[Money] = None,
        volatility: Optional[Decimal] = None,
        sharpe_ratio: Optional[Decimal] = None,
        sortino_ratio: Optional[Decimal] = None,
        max_drawdown: Optional[Decimal] = None,
        beta: Optional[Decimal] = None,
        alpha: Optional[Decimal] = None,
        skewness: Optional[Decimal] = None,
        kurtosis: Optional[Decimal] = None,
        correlation: Optional[Decimal] = None,
    ) -> None:
        """
        Обновление метрик риска.
        Args:
            var_95: Value at Risk 95%
            var_99: Value at Risk 99%
            volatility: Волатильность
            sharpe_ratio: Коэффициент Шарпа
            sortino_ratio: Коэффициент Сортино
            max_drawdown: Максимальная просадка
            beta: Бета
            alpha: Альфа
            skewness: Асимметрия
            kurtosis: Эксцесс
            correlation: Корреляция
        """
        if var_95 is not None:
            self.var_95 = var_95
        if var_99 is not None:
            self.var_99 = var_99
        if volatility is not None:
            self.volatility = volatility
        if sharpe_ratio is not None:
            self.sharpe_ratio = sharpe_ratio
        if sortino_ratio is not None:
            self.sortino_ratio = sortino_ratio
        if max_drawdown is not None:
            self.max_drawdown = max_drawdown
        if beta is not None:
            self.beta = beta
        if alpha is not None:
            self.alpha = alpha
        if skewness is not None:
            self.skewness = skewness
        if kurtosis is not None:
            self.kurtosis = kurtosis
        if correlation is not None:
            self.correlation = correlation
        self.updated_at = Timestamp.now()

    def to_dict(self) -> Dict[str, str]:
        """
        Преобразование в словарь.
        Returns:
            Dict[str, str]: Словарь с данными метрик
        """
        return {
            "id": str(self.id),
            "portfolio_id": str(self.portfolio_id),
            "var_95": str(self.var_95.amount),
            "var_99": str(self.var_99.amount),
            "volatility": str(self.volatility),
            "sharpe_ratio": str(self.sharpe_ratio),
            "sortino_ratio": str(self.sortino_ratio),
            "max_drawdown": str(self.max_drawdown),
            "beta": str(self.beta),
            "alpha": str(self.alpha),
            "skewness": str(self.skewness),
            "kurtosis": str(self.kurtosis),
            "correlation": str(self.correlation),
            "risk_level": self.risk_level,
            "created_at": str(self.created_at),
            "updated_at": str(self.updated_at),
            "metadata": str(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "RiskMetrics":
        """
        Создание из словаря.
        Args:
            data: Словарь с данными метрик
        Returns:
            RiskMetrics: Объект метрик
        """
        return cls(
            id=UUID(data["id"]),
            portfolio_id=UUID(data["portfolio_id"]),
            var_95=Money(Decimal(data["var_95"]), Currency.USD),
            var_99=Money(Decimal(data["var_99"]), Currency.USD),
            volatility=Decimal(data["volatility"]),
            sharpe_ratio=Decimal(data["sharpe_ratio"]),
            sortino_ratio=Decimal(data["sortino_ratio"]),
            max_drawdown=Decimal(data["max_drawdown"]),
            beta=Decimal(data["beta"]),
            alpha=Decimal(data["alpha"]),
            skewness=Decimal(data["skewness"]),
            kurtosis=Decimal(data["kurtosis"]),
            correlation=Decimal(data["correlation"]),
            created_at=Timestamp.from_iso(data["created_at"]),
            updated_at=Timestamp.from_iso(data["updated_at"]),
            metadata=ast.literal_eval(data.get("metadata", "{}")),
        )

    def __str__(self) -> str:
        """Строковое представление метрик."""
        return f"RiskMetrics(portfolio={self.portfolio_id}, var_95={self.var_95})"

    def __repr__(self) -> str:
        """Представление для отладки."""
        return (
            f"RiskMetrics(id={self.id}, portfolio_id={self.portfolio_id}, "
            f"var_95={self.var_95}, volatility={self.volatility})"
        )


@runtime_checkable
class RiskMetricsProtocol(Protocol):
    """Протокол для метрик риска."""
    
    @property
    def risk_level(self) -> str:
        """Уровень риска."""
        ...
    
    @property
    def is_high_risk(self) -> bool:
        """Проверка высокого риска."""
        ...
    
    @property
    def is_medium_risk(self) -> bool:
        """Проверка среднего риска."""
        ...
    
    @property
    def is_low_risk(self) -> bool:
        """Проверка низкого риска."""
        ...
    
    def update_metrics(self, **kwargs: object) -> None:
        """Обновление метрик."""
        ...
    
    def to_dict(self) -> Dict[str, str]:
        """Преобразование в словарь."""
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "RiskMetrics":
        """Создание из словаря."""
        ...
