from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import setup_logger

logger = setup_logger(__name__)


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"


class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"
    CANCELED = "canceled"


@dataclass
class Position:
    """Класс для представления позиции"""

    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN
    leverage: float = 1.0
    margin: float = 0.0
    fees: float = 0.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.margin = self.calculate_margin()

    def calculate_margin(self) -> float:
        """Расчет требуемой маржи"""
        return (self.size * self.entry_price) / self.leverage

    def calculate_pnl(self, current_price: float) -> float:
        """Расчет текущего P&L"""
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) * self.size
        return (self.entry_price - current_price) * self.size

    def calculate_roi(self) -> float:
        """Расчет ROI"""
        if self.pnl is None or self.margin == 0:
            return 0.0
        return (self.pnl / self.margin) * 100

    def is_profitable(self) -> bool:
        """Проверка прибыльности позиции"""
        return self.pnl is not None and self.pnl > 0

    def get_duration(self) -> float:
        """Получение длительности позиции в часах"""
        if self.exit_time is None:
            return 0.0
        return (self.exit_time - self.entry_time).total_seconds() / 3600


class PositionManager:
    """Класс для управления позициями"""

    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация менеджера.

        Args:
            config: Конфигурация
        """
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.pairs: List[str] = []
        self.max_positions = config.get("max_positions", 10)
        self.max_position_size = config.get("max_position_size", 1.0)
        self.min_position_size = config.get("min_position_size", 0.01)
        self.stop_loss = config.get("stop_loss", 0.02)
        self.take_profit = config.get("take_profit", 0.04)
        self.position_history: List[Position] = []
        self.risk_metrics: Dict[str, float] = {}

    def add_position(self, position: Position) -> None:
        """
        Добавление позиции.

        Args:
            position: Позиция
        """
        if position.symbol not in self.positions:
            self.positions[position.symbol] = position
            if position.symbol not in self.pairs:
                self.pairs.append(position.symbol)

    def remove_position(self, symbol: str) -> None:
        """
        Удаление позиции.

        Args:
            symbol: Торговая пара
        """
        if symbol in self.positions:
            del self.positions[symbol]
            if symbol in self.pairs:
                self.pairs.remove(symbol)

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Получение позиции.

        Args:
            symbol: Торговая пара

        Returns:
            Optional[Position]: Позиция или None
        """
        return self.positions.get(symbol)

    def get_positions(self) -> List[Position]:
        """
        Получение всех позиций.

        Returns:
            List[Position]: Список позиций
        """
        return list(self.positions.values())

    def get_pairs(self) -> List[str]:
        """
        Получение всех пар.

        Returns:
            List[str]: Список пар
        """
        return self.pairs.copy()

    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        size: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: Optional[float] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Position]:
        """
        Открытие новой позиции

        Args:
            symbol: Торговая пара
            side: Сторона (LONG или SHORT)
            size: Размер позиции
            entry_price: Цена входа
            stop_loss: Уровень стоп-лосса
            take_profit: Уровень тейк-профита
            leverage: Плечо
            tags: Теги позиции
            metadata: Дополнительные метаданные

        Returns:
            Optional[Position]: Созданная позиция или None в случае ошибки
        """
        try:
            # Проверка лимитов
            if not self._check_position_limits(symbol, size):
                return None

            # Проверка рисков
            if not self._check_risk_limits(symbol, size, entry_price, leverage or 1.0):
                return None

            # Создание позиции
            position = Position(
                symbol=symbol,
                side=side,
                size=size,
                entry_price=entry_price,
                stop_loss=stop_loss or self._calculate_default_stop_loss(entry_price, side),
                take_profit=take_profit or self._calculate_default_take_profit(entry_price, side),
                entry_time=datetime.now(),
                leverage=leverage or self.config["default_leverage"],
                tags=tags or [],
                metadata=metadata or {},
            )

            # Сохранение позиции
            position_id = self._generate_position_id(position)
            self.positions[position_id] = position
            logger.info(f"Opened {side.value} position for {symbol} with size {size}")

            # Обновление метрик
            self._update_risk_metrics()

            return position

        except Exception as e:
            logger.error(f"Error opening position: {str(e)}")
            return None

    def close_position(
        self, position_id: str, exit_price: float, fees: float = 0.0, reason: Optional[str] = None
    ) -> Optional[Position]:
        """
        Закрытие позиции

        Args:
            position_id: Идентификатор позиции
            exit_price: Цена выхода
            fees: Комиссии
            reason: Причина закрытия

        Returns:
            Optional[Position]: Закрытая позиция или None в случае ошибки
        """
        try:
            if position_id not in self.positions:
                logger.warning(f"Position {position_id} not found")
                return None

            position = self.positions[position_id]
            position.exit_time = datetime.now()
            position.exit_price = exit_price
            position.fees = fees
            position.pnl = position.calculate_pnl(exit_price) - fees
            position.status = PositionStatus.CLOSED

            if reason:
                position.metadata["close_reason"] = reason

            # Перемещение в историю
            self.position_history.append(position)
            del self.positions[position_id]

            logger.info(f"Closed position {position_id} with P&L {position.pnl}")

            # Обновление метрик
            self._update_risk_metrics()

            return position

        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            return None

    def update_position(self, position_id: str, current_price: float) -> Optional[Position]:
        """
        Обновление состояния позиции

        Args:
            position_id: Идентификатор позиции
            current_price: Текущая цена

        Returns:
            Optional[Position]: Обновленная позиция или None в случае ошибки
        """
        try:
            if position_id not in self.positions:
                return None

            position = self.positions[position_id]

            # Проверка стоп-лосса и тейк-профита
            if self._check_stop_loss(position, current_price):
                return self.close_position(position_id, current_price, reason="stop_loss")
            elif self._check_take_profit(position, current_price):
                return self.close_position(position_id, current_price, reason="take_profit")

            # Обновление P&L
            position.pnl = position.calculate_pnl(current_price)

            return position

        except Exception as e:
            logger.error(f"Error updating position: {str(e)}")
            return None

    def get_open_positions(self) -> List[Position]:
        """Получение списка открытых позиций"""
        return [p for p in self.positions.values() if p.status == PositionStatus.OPEN]

    def get_closed_positions(self) -> List[Position]:
        """Получение списка закрытых позиций"""
        return [p for p in self.position_history if p.status == PositionStatus.CLOSED]

    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Получение позиций по торговой паре"""
        return [p for p in self.positions.values() if p.symbol == symbol]

    def get_total_pnl(self) -> float:
        """Получение общего P&L"""
        return sum(p.pnl or 0 for p in self.positions.values()) + sum(
            p.pnl or 0 for p in self.position_history
        )

    def get_open_pnl(self) -> float:
        """Получение P&L открытых позиций"""
        return sum(p.pnl or 0 for p in self.positions.values() if p.status == PositionStatus.OPEN)

    def get_closed_pnl(self) -> float:
        """Получение P&L закрытых позиций"""
        return sum(p.pnl or 0 for p in self.position_history if p.status == PositionStatus.CLOSED)

    def get_win_rate(self) -> float:
        """Получение процента прибыльных сделок"""
        closed_positions = self.get_closed_positions()
        if not closed_positions:
            return 0.0
        return sum(1 for p in closed_positions if p.is_profitable()) / len(closed_positions)

    def get_average_roi(self) -> float:
        """Получение среднего ROI"""
        closed_positions = self.get_closed_positions()
        if not closed_positions:
            return 0.0
        rois = [p.calculate_roi() for p in closed_positions if p.calculate_roi() is not None]
        return float(np.mean(rois)) if rois else 0.0

    def get_max_drawdown(self) -> float:
        """Получение максимальной просадки"""
        if not self.position_history:
            return 0.0

        cumulative_pnl = np.cumsum([p.pnl or 0 for p in self.position_history])
        max_drawdown = 0.0
        peak = cumulative_pnl[0]

        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            drawdown = (peak - pnl) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def get_position_correlation(self, symbol1: str, symbol2: str) -> float:
        """Расчет корреляции между позициями"""
        positions1 = self.get_positions_by_symbol(symbol1)
        positions2 = self.get_positions_by_symbol(symbol2)

        if not positions1 or not positions2:
            return 0.0

        pnl1 = [p.pnl or 0 for p in positions1]
        pnl2 = [p.pnl or 0 for p in positions2]

        return np.corrcoef(pnl1, pnl2)[0, 1]

    def _check_position_limits(self, symbol: str, size: float) -> bool:
        """Проверка лимитов позиции"""
        if len(self.positions) >= self.max_positions:
            logger.warning(f"Maximum number of positions reached ({self.max_positions})")
            return False

        if size > self.max_position_size:
            logger.warning(f"Position size {size} exceeds maximum {self.max_position_size}")
            return False

        if size < self.min_position_size:
            logger.warning(f"Position size {size} below minimum {self.min_position_size}")
            return False

        return True

    def _check_risk_limits(self, symbol: str, size: float, price: float, leverage: float) -> bool:
        """Проверка лимитов риска"""
        if leverage > self.config["max_leverage"]:
            logger.warning(f"Leverage {leverage} exceeds maximum {self.config['max_leverage']}")
            return False

        if leverage < self.config["min_leverage"]:
            logger.warning(f"Leverage {leverage} below minimum {self.config['min_leverage']}")
            return False

        # Проверка диверсификации
        if len(self.get_positions_by_symbol(symbol)) >= self.config["min_diversification"]:
            logger.warning(f"Maximum positions for {symbol} reached")
            return False

        # Проверка корреляции
        for other_symbol in set(p.symbol for p in self.positions.values()):
            if other_symbol != symbol:
                correlation = self.get_position_correlation(symbol, other_symbol)
                if abs(correlation) > self.config["max_correlation"]:
                    logger.warning(
                        f"High correlation {correlation} between {symbol} and {other_symbol}"
                    )
                    return False

        return True

    def _calculate_default_stop_loss(self, entry_price: float, side: PositionSide) -> float:
        """Расчет стоп-лосса по умолчанию"""
        if side == PositionSide.LONG:
            return entry_price * (1 - self.stop_loss)
        return entry_price * (1 + self.stop_loss)

    def _calculate_default_take_profit(self, entry_price: float, side: PositionSide) -> float:
        """Расчет тейк-профита по умолчанию"""
        if side == PositionSide.LONG:
            return entry_price * (1 + self.take_profit)
        return entry_price * (1 - self.take_profit)

    def _generate_position_id(self, position: Position) -> str:
        """Генерация ID позиции"""
        return f"{position.symbol}_{position.side.value}_{int(position.entry_time.timestamp())}"

    def _check_stop_loss(self, position: Position, current_price: float) -> bool:
        """Проверка срабатывания стоп-лосса"""
        if position.side == PositionSide.LONG:
            return float(current_price) <= float(position.stop_loss)
        return float(current_price) >= float(position.stop_loss)

    def _check_take_profit(self, position: Position, current_price: float) -> bool:
        """Проверка срабатывания тейк-профита"""
        if position.side == PositionSide.LONG:
            return float(current_price) >= float(position.take_profit)
        return float(current_price) <= float(position.take_profit)

    def _update_risk_metrics(self) -> None:
        """Обновление метрик риска"""
        self.risk_metrics = {
            "total_pnl": self.get_total_pnl(),
            "open_pnl": self.get_open_pnl(),
            "closed_pnl": self.get_closed_pnl(),
            "win_rate": self.get_win_rate(),
            "average_roi": self.get_average_roi(),
            "max_drawdown": self.get_max_drawdown(),
            "position_count": len(self.positions),
            "closed_position_count": len(self.position_history),
        }

    def get_all_positions(self) -> List[Position]:
        """
        Получение всех позиций (открытых и закрытых).

        Returns:
            List[Position]: Список всех позиций
        """
        return list(self.positions.values()) + self.position_history

    def _convert_to_str(self, value: Any) -> str:
        """Безопасное преобразование в строку."""
        if value is None:
            return ""
        try:
            return str(value)
        except Exception as e:
            logger.error(f"Error converting to string: {e}")
            return ""

    def _convert_to_float(self, value: Any) -> float:
        """Безопасное преобразование в float."""
        if value is None:
            return 0.0
        try:
            result = float(str(value))
            return result if isinstance(result, float) else 0.0
        except Exception as e:
            logger.error(f"Error converting to float: {e}")
            return 0.0
