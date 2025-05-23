from typing import Any, Dict, List

from core.logger import Logger

from ..models import Position

logger = Logger()


class RiskController:
    """Контроллер для управления рисками"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_metrics: Dict[str, float] = {}

    def calculate_position_size(
        self, pair: str, price: float, risk_per_trade: float
    ) -> float:
        """
        Расчет размера позиции.

        Args:
            pair: Торговая пара
            price: Цена
            risk_per_trade: Риск на сделку

        Returns:
            float: Размер позиции
        """
        try:
            # Получение баланса
            balance = self.config.get("balance", 0.0)

            # Расчет риска в деньгах
            risk_amount = balance * risk_per_trade

            # Расчет размера позиции
            position_size = risk_amount / price

            # Округление
            position_size = round(position_size, 8)

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            raise

    def calculate_stop_loss(self, position: Position, atr: float) -> float:
        """
        Расчет стоп-лосса.

        Args:
            position: Позиция
            atr: Average True Range

        Returns:
            float: Цена стоп-лосса
        """
        try:
            # Получение множителя
            multiplier = self.config.get("stop_loss_multiplier", 2.0)

            # Расчет стоп-лосса
            if position.side == "long":
                stop_loss = position.entry_price - (atr * multiplier)
            else:
                stop_loss = position.entry_price + (atr * multiplier)

            return stop_loss

        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            raise

    def calculate_take_profit(self, position: Position, atr: float) -> float:
        """
        Расчет тейк-профита.

        Args:
            position: Позиция
            atr: Average True Range

        Returns:
            float: Цена тейк-профита
        """
        try:
            # Получение множителя
            multiplier = self.config.get("take_profit_multiplier", 3.0)

            # Расчет тейк-профита
            if position.side == "long":
                take_profit = position.entry_price + (atr * multiplier)
            else:
                take_profit = position.entry_price - (atr * multiplier)

            return take_profit

        except Exception as e:
            logger.error(f"Error calculating take profit: {e}")
            raise

    def check_risk_limits(self, position: Position) -> bool:
        """
        Проверка лимитов риска.

        Args:
            position: Позиция

        Returns:
            bool: True если лимиты не превышены
        """
        try:
            # Получение лимитов
            max_position_size = self.config.get("max_position_size", float("inf"))
            max_daily_loss = self.config.get("max_daily_loss", float("inf"))

            # Проверка размера позиции
            if position.size > max_position_size:
                return False

            # Проверка дневного убытка
            daily_pnl = self.risk_metrics.get("daily_pnl", 0.0)
            if daily_pnl < -max_daily_loss:
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False

    def update_risk_metrics(self, positions: List[Position]) -> None:
        """
        Обновление метрик риска.

        Args:
            positions: Список позиций
        """
        try:
            # Расчет метрик
            total_pnl = sum(p.pnl for p in positions)
            win_rate = (
                sum(1 for p in positions if p.pnl > 0) / len(positions)
                if positions
                else 0.0
            )

            # Обновление
            self.risk_metrics.update(
                {
                    "total_pnl": total_pnl,
                    "win_rate": win_rate,
                    "position_count": len(positions),
                }
            )

        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
            raise

    def validate_signal(self, signal: Any) -> bool:
        """
        Проверка сигнала на соответствие правилам риск-менеджмента.

        Args:
            signal: Сигнал для проверки

        Returns:
            bool: True если сигнал валиден
        """
        try:
            # Проверка лимитов
            if self.risk_metrics.get("position_count", 0) >= self.config.get(
                "max_positions", 10
            ):
                logger.warning("Maximum number of positions reached")
                return False

            # Проверка дневного убытка
            daily_pnl = self.risk_metrics.get("daily_pnl", 0.0)
            max_daily_loss = self.config.get("max_daily_loss", float("inf"))
            if daily_pnl < -max_daily_loss:
                logger.warning("Maximum daily loss reached")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False
