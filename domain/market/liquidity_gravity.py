# -*- coding: utf-8 -*-
"""Liquidity Gravity Field Model for Market Analysis."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from .market_types import MarketMetadataDict


# ============================================================================
# TYPES AND PROTOCOLS
# ============================================================================
@runtime_checkable
class OrderBookProtocol(Protocol):
    """Протокол для ордербука."""

    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    timestamp: datetime
    symbol: str

    def get_bid_volume(self) -> float: ...
    def get_ask_volume(self) -> float: ...
    def get_mid_price(self) -> float: ...
    def get_spread(self) -> float: ...
    def get_spread_percentage(self) -> float: ...
@runtime_checkable
class LiquidityGravityProtocol(Protocol):
    """Протокол для модели гравитации ликвидности."""

    def compute_liquidity_gravity(self, order_book: OrderBookProtocol) -> float: ...
    def analyze_liquidity_gravity(
        self, order_book: OrderBookProtocol
    ) -> "LiquidityGravityResult": ...
    def compute_gravity_gradient(
        self, order_book: OrderBookProtocol
    ) -> Dict[str, float]: ...
    def get_model_statistics(self) -> Dict[str, Any]: ...
    def update_config(self, new_config: "LiquidityGravityConfig") -> None: ...


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class LiquidityGravityConfig:
    """Конфигурация модели гравитации ликвидности."""

    gravitational_constant: float = 1e-6
    min_volume_threshold: float = 0.001
    max_price_distance: float = 0.1  # 10% от цены
    volume_weight: float = 1.0
    price_weight: float = 1.0
    decay_factor: float = 0.95
    normalization_factor: float = 1e6
    # Дополнительные параметры для расширенной аналитики
    risk_thresholds: Dict[str, float] = field(
        default_factory=lambda: {"low": 0.1, "medium": 0.5, "high": 1.0, "extreme": 2.0}
    )
    volume_imbalance_threshold: float = 0.5
    spread_threshold: float = 1.0  # 1% спред
    momentum_weight: float = 0.3
    volatility_weight: float = 0.2


# ============================================================================
# DATA STRUCTURES
# ============================================================================
@dataclass
class LiquidityGravityResult:
    """Результат вычисления гравитации ликвидности."""

    total_gravity: float
    bid_ask_forces: List[Tuple[float, float, float]]  # (bid_volume, ask_volume, force)
    gravity_distribution: Dict[str, float]
    risk_level: str
    timestamp: datetime
    metadata: MarketMetadataDict
    # Расширенные метрики
    volume_imbalance: float
    price_momentum: float
    volatility_score: float
    liquidity_score: float
    market_efficiency: float

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "total_gravity": self.total_gravity,
            "bid_ask_forces": self.bid_ask_forces,
            "gravity_distribution": self.gravity_distribution,
            "risk_level": self.risk_level,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "volume_imbalance": self.volume_imbalance,
            "price_momentum": self.price_momentum,
            "volatility_score": self.volatility_score,
            "liquidity_score": self.liquidity_score,
            "market_efficiency": self.market_efficiency,
        }


@dataclass
class OrderBookSnapshot(OrderBookProtocol):
    """Снимок ордербука для анализа гравитации ликвидности."""

    bids: List[Tuple[float, float]] = field(default_factory=list)  # (price, volume)
    asks: List[Tuple[float, float]] = field(default_factory=list)  # (price, volume)
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: str = ""

    def get_bid_volume(self) -> float:
        """Получение общего объема бидов."""
        return sum(volume for _, volume in self.bids)

    def get_ask_volume(self) -> float:
        """Получение общего объема асков."""
        return sum(volume for _, volume in self.asks)

    def get_mid_price(self) -> float:
        """Получение средней цены."""
        if not self.bids or not self.asks:
            return 0.0
        best_bid = self.bids[0][0] if self.bids else 0.0
        best_ask = self.asks[0][0] if self.asks else 0.0
        return (best_bid + best_ask) / 2

    def get_spread(self) -> float:
        """Получение спреда."""
        if not self.bids or not self.asks:
            return 0.0
        best_bid = self.bids[0][0] if self.bids else 0.0
        best_ask = self.asks[0][0] if self.asks else 0.0
        return best_ask - best_bid

    def get_spread_percentage(self) -> float:
        """Получение спреда в процентах."""
        mid_price = self.get_mid_price()
        if mid_price == 0.0:
            return 0.0
        spread = self.get_spread()
        return (spread / mid_price) * 100


# ============================================================================
# CORE MODEL
# ============================================================================
class LiquidityGravityModel(LiquidityGravityProtocol):
    """Физическая модель гравитации ликвидности."""

    def __init__(self, config: Optional[LiquidityGravityConfig] = None):
        self.config = config or LiquidityGravityConfig()
        self._statistics = {
            "total_calculations": 0,
            "average_gravity": 0.0,
            "max_gravity": 0.0,
            "min_gravity": float("inf"),
            "last_update": datetime.now(),
        }
        logger.info(
            f"LiquidityGravityModel initialized with G={self.config.gravitational_constant}"
        )

    def compute_liquidity_gravity(self, order_book: OrderBookProtocol) -> float:
        """
        Вычисление силы гравитации ликвидности.
        Args:
            order_book: Снимок ордербука
        Returns:
            float: Общая сила гравитации ликвидности
        """
        try:
            if not order_book.bids or not order_book.asks:
                logger.warning("Empty order book for gravity calculation")
                return 0.0
            total_gravity = 0.0
            mid_price = order_book.get_mid_price()
            if mid_price == 0.0:
                logger.warning("Invalid mid price for gravity calculation")
                return 0.0
            # Вычисляем силу гравитации между всеми парами bid-ask
            for bid_price, bid_volume in order_book.bids:
                for ask_price, ask_volume in order_book.asks:
                    # Пропускаем слишком малые объемы
                    if (
                        bid_volume < self.config.min_volume_threshold
                        or ask_volume < self.config.min_volume_threshold
                    ):
                        continue
                    # Вычисляем расстояние между ценами
                    price_distance = abs(ask_price - bid_price)
                    # Пропускаем слишком большие расстояния
                    if price_distance > mid_price * self.config.max_price_distance:
                        continue
                    # Вычисляем силу гравитации: F = G * v₁ * v₂ / (Δp)²
                    force = self._compute_gravitational_force(
                        bid_volume, ask_volume, price_distance
                    )
                    total_gravity += force
            # Нормализация результата
            normalized_gravity = total_gravity / self.config.normalization_factor
            # Обновляем статистику
            self._update_statistics(normalized_gravity)
            logger.debug(f"Liquidity gravity computed: {normalized_gravity:.6f}")
            return normalized_gravity
        except Exception as e:
            logger.error(f"Error computing liquidity gravity: {e}")
            return 0.0

    def _compute_gravitational_force(
        self, volume1: float, volume2: float, price_distance: float
    ) -> float:
        """
        Вычисление силы гравитации между двумя объемами.
        Args:
            volume1: Первый объем
            volume2: Второй объем
            price_distance: Расстояние между ценами
        Returns:
            float: Сила гравитации
        """
        try:
            if price_distance == 0.0:
                # Избегаем деления на ноль
                price_distance = 1e-10
            # Базовая формула: F = G * v₁ * v₂ / (Δp)²
            force = (
                self.config.gravitational_constant
                * volume1
                * volume2
                * self.config.volume_weight
            ) / (price_distance**2 * self.config.price_weight)
            return force
        except Exception as e:
            logger.error(f"Error computing gravitational force: {e}")
            return 0.0

    def analyze_liquidity_gravity(
        self, order_book: OrderBookProtocol
    ) -> LiquidityGravityResult:
        """
        Полный анализ гравитации ликвидности.
        Args:
            order_book: Снимок ордербука
        Returns:
            LiquidityGravityResult: Результат анализа
        """
        try:
            if not order_book.bids or not order_book.asks:
                return LiquidityGravityResult(
                    total_gravity=0.0,
                    bid_ask_forces=[],
                    gravity_distribution={},
                    risk_level="low",
                    timestamp=order_book.timestamp,
                    metadata={"source": "", "exchange": "", "error": "Empty order book"},
                    volume_imbalance=0.0,
                    price_momentum=0.0,
                    volatility_score=0.0,
                    liquidity_score=0.0,
                    market_efficiency=0.0,
                )
            total_gravity = 0.0
            bid_ask_forces = []
            gravity_distribution = {
                "bid_volume": order_book.get_bid_volume(),
                "ask_volume": order_book.get_ask_volume(),
                "mid_price": order_book.get_mid_price(),
                "spread": order_book.get_spread(),
                "spread_percentage": order_book.get_spread_percentage(),
            }
            mid_price = order_book.get_mid_price()
            # Анализируем каждую пару bid-ask
            for bid_price, bid_volume in order_book.bids:
                for ask_price, ask_volume in order_book.asks:
                    if (
                        bid_volume < self.config.min_volume_threshold
                        or ask_volume < self.config.min_volume_threshold
                    ):
                        continue
                    price_distance = abs(ask_price - bid_price)
                    if price_distance > mid_price * self.config.max_price_distance:
                        continue
                    force = self._compute_gravitational_force(
                        bid_volume, ask_volume, price_distance
                    )
                    total_gravity += force
                    bid_ask_forces.append((bid_volume, ask_volume, force))
            # Нормализация
            normalized_gravity = total_gravity / self.config.normalization_factor
            # Расширенная аналитика
            volume_imbalance = self._calculate_volume_imbalance(order_book)
            price_momentum = self._calculate_price_momentum(order_book)
            volatility_score = self._calculate_volatility_score(order_book)
            liquidity_score = self._calculate_liquidity_score(order_book)
            market_efficiency = self._calculate_market_efficiency(order_book)
            # Определение уровня риска
            risk_level = self._determine_risk_level(normalized_gravity, order_book)
            # Метаданные
            metadata: MarketMetadataDict = {
                "source": "liquidity_gravity",
                "exchange": "",
                "extra": {
                    "gravitational_constant": str(self.config.gravitational_constant),
                    "min_volume_threshold": str(self.config.min_volume_threshold),
                    "max_price_distance": str(self.config.max_price_distance),
                    "volume_weight": str(self.config.volume_weight),
                    "price_weight": str(self.config.price_weight),
                    "normalization_factor": str(self.config.normalization_factor),
                    "total_forces": str(len(bid_ask_forces)),
                    "analysis_version": "2.0",
                },
            }
            result = LiquidityGravityResult(
                total_gravity=normalized_gravity,
                bid_ask_forces=bid_ask_forces,
                gravity_distribution=gravity_distribution,
                risk_level=risk_level,
                timestamp=order_book.timestamp,
                metadata=metadata,
                volume_imbalance=volume_imbalance,
                price_momentum=price_momentum,
                volatility_score=volatility_score,
                liquidity_score=liquidity_score,
                market_efficiency=market_efficiency,
            )
            logger.info(
                f"Liquidity gravity analysis completed: gravity={normalized_gravity:.6f}, risk={risk_level}"
            )
            return result
        except Exception as e:
            logger.error(f"Error analyzing liquidity gravity: {e}")
            return LiquidityGravityResult(
                total_gravity=0.0,
                bid_ask_forces=[],
                gravity_distribution={},
                risk_level="unknown",
                timestamp=order_book.timestamp,
                metadata={"source": "", "exchange": "", "error": str(e)},
                volume_imbalance=0.0,
                price_momentum=0.0,
                volatility_score=0.0,
                liquidity_score=0.0,
                market_efficiency=0.0,
            )

    def _determine_risk_level(
        self, gravity: float, order_book: OrderBookProtocol
    ) -> str:
        """
        Определение уровня риска на основе гравитации ликвидности.
        Args:
            gravity: Сила гравитации
            order_book: Снимок ордербука
        Returns:
            str: Уровень риска (low, medium, high, extreme)
        """
        try:
            thresholds = self.config.risk_thresholds
            # Дополнительные факторы
            spread_percentage = order_book.get_spread_percentage()
            volume_imbalance = self._calculate_volume_imbalance(order_book)
            # Корректировка порогов на основе рыночных условий
            if spread_percentage > self.config.spread_threshold:  # Широкий спред
                thresholds = {k: v * 0.5 for k, v in thresholds.items()}
            if (
                volume_imbalance > self.config.volume_imbalance_threshold
            ):  # Дисбаланс объемов
                thresholds = {k: v * 0.7 for k, v in thresholds.items()}
            # Определение уровня риска
            if gravity < thresholds["low"]:
                return "low"
            elif gravity < thresholds["medium"]:
                return "medium"
            elif gravity < thresholds["high"]:
                return "high"
            else:
                return "extreme"
        except Exception as e:
            logger.error(f"Error determining risk level: {e}")
            return "unknown"

    def compute_gravity_gradient(
        self, order_book: OrderBookProtocol
    ) -> Dict[str, float]:
        """
        Вычисление градиента гравитации по уровням цен.
        Args:
            order_book: Снимок ордербука
        Returns:
            Dict[str, float]: Градиент гравитации
        """
        try:
            if not order_book.bids or not order_book.asks:
                return {}
            mid_price = order_book.get_mid_price()
            if mid_price == 0.0:
                return {}
            gradient = {}
            # Анализируем гравитацию на разных уровнях
            for i, (bid_price, bid_volume) in enumerate(
                order_book.bids[:5]
            ):  # Топ-5 бидов
                level_gravity = 0.0
                for ask_price, ask_volume in order_book.asks[:5]:  # Топ-5 асков
                    price_distance = abs(ask_price - bid_price)
                    if price_distance > mid_price * self.config.max_price_distance:
                        continue
                    force = self._compute_gravitational_force(
                        bid_volume, ask_volume, price_distance
                    )
                    level_gravity += force
                gradient[f"bid_level_{i+1}"] = (
                    level_gravity / self.config.normalization_factor
                )
            return gradient
        except Exception as e:
            logger.error(f"Error computing gravity gradient: {e}")
            return {}

    def _calculate_volume_imbalance(self, order_book: OrderBookProtocol) -> float:
        """Вычисление дисбаланса объемов."""
        bid_volume = order_book.get_bid_volume()
        ask_volume = order_book.get_ask_volume()
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
        return abs(bid_volume - ask_volume) / total_volume

    def _calculate_price_momentum(self, order_book: OrderBookProtocol) -> float:
        """Вычисление ценового импульса."""
        if len(order_book.bids) < 2 or len(order_book.asks) < 2:
            return 0.0
        # Простой расчет импульса на основе лучших цен
        best_bid = order_book.bids[0][0]
        second_best_bid = (
            order_book.bids[1][0] if len(order_book.bids) > 1 else best_bid
        )
        best_ask = order_book.asks[0][0]
        second_best_ask = (
            order_book.asks[1][0] if len(order_book.asks) > 1 else best_ask
        )
        bid_momentum = (best_bid - second_best_bid) / best_bid if best_bid != 0 else 0
        ask_momentum = (second_best_ask - best_ask) / best_ask if best_ask != 0 else 0
        return (bid_momentum + ask_momentum) / 2

    def _calculate_volatility_score(self, order_book: OrderBookProtocol) -> float:
        """Вычисление оценки волатильности."""
        if not order_book.bids or not order_book.asks:
            return 0.0
        # Используем спред как индикатор волатильности
        spread_pct = order_book.get_spread_percentage()
        return min(spread_pct / 10.0, 1.0)  # Нормализуем к [0, 1]

    def _calculate_liquidity_score(self, order_book: OrderBookProtocol) -> float:
        """Вычисление оценки ликвидности."""
        total_volume = order_book.get_bid_volume() + order_book.get_ask_volume()
        spread_pct = order_book.get_spread_percentage()
        if total_volume == 0 or spread_pct == 0:
            return 0.0
        # Ликвидность = объем / спред
        return min(total_volume / (spread_pct * 1000), 1.0)

    def _calculate_market_efficiency(self, order_book: OrderBookProtocol) -> float:
        """Вычисление эффективности рынка."""
        spread_pct = order_book.get_spread_percentage()
        volume_imbalance = self._calculate_volume_imbalance(order_book)
        # Эффективность = 1 - (спред + дисбаланс)
        efficiency = 1.0 - (spread_pct / 100.0 + volume_imbalance)
        return max(efficiency, 0.0)

    def _update_statistics(self, gravity: float) -> None:
        """Обновление статистики модели."""
        total_calc = self._statistics["total_calculations"]
        if isinstance(total_calc, int):
            self._statistics["total_calculations"] = total_calc + 1
        avg_gravity = self._statistics["average_gravity"]
        if isinstance(avg_gravity, (int, float)):
            new_total_calc = self._statistics["total_calculations"]
            if isinstance(new_total_calc, int):
                self._statistics["average_gravity"] = (
                    avg_gravity * (new_total_calc - 1) + gravity
                ) / new_total_calc
        max_gravity = self._statistics["max_gravity"]
        if isinstance(max_gravity, (int, float)):
            self._statistics["max_gravity"] = max(max_gravity, gravity)
        min_gravity = self._statistics["min_gravity"]
        if isinstance(min_gravity, (int, float)):
            self._statistics["min_gravity"] = min(min_gravity, gravity)
        self._statistics["last_update"] = datetime.now()

    def get_model_statistics(self) -> Dict[str, Any]:
        """Получение статистики модели."""
        return {
            "config": {
                "gravitational_constant": self.config.gravitational_constant,
                "min_volume_threshold": self.config.min_volume_threshold,
                "max_price_distance": self.config.max_price_distance,
                "volume_weight": self.config.volume_weight,
                "price_weight": self.config.price_weight,
                "decay_factor": self.config.decay_factor,
                "normalization_factor": self.config.normalization_factor,
            },
            "statistics": self._statistics,
        }

    def update_config(self, new_config: LiquidityGravityConfig) -> None:
        """Обновление конфигурации модели."""
        try:
            self.config = new_config
            logger.info(f"LiquidityGravityModel config updated: {self.config}")
        except Exception as e:
            logger.error(f"Error updating config: {e}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def compute_liquidity_gravity(order_book: OrderBookProtocol) -> float:
    """
    Удобная функция для быстрого вычисления гравитации ликвидности.
    Args:
        order_book: Снимок ордербука
    Returns:
        float: Сила гравитации ликвидности
    """
    model = LiquidityGravityModel()
    return model.compute_liquidity_gravity(order_book)
