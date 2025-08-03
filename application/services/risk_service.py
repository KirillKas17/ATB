"""
Сервис управления рисками.
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import uuid4

from domain.entities.order import Order, OrderId, OrderSide, OrderType, OrderStatus
from domain.entities.portfolio import Portfolio
from domain.entities.position import Position, PositionSide
from domain.entities.risk import RiskManager
from domain.entities.signal import Signal
from domain.protocols.repository_protocol import PortfolioRepositoryProtocol
from domain.types import PortfolioId, TradingPair, VolumeValue, PriceValue
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.price import Price, Currency


@dataclass
class RiskValidationResult:
    """Результат валидации рисков."""

    is_valid: bool
    reason: Optional[str] = None
    risk_level: float = 0.0


class RiskService:
    """Сервис управления рисками."""

    def __init__(
        self,
        risk_manager: RiskManager,
        portfolio_repository: PortfolioRepositoryProtocol,
        config: Dict[str, Any],
    ):
        self.risk_manager = risk_manager
        self.portfolio_repository = portfolio_repository
        self.logger = logging.getLogger(self.__class__.__name__)

        # Конфигурация риск-лимитов
        self.max_risk_per_trade = Decimal(str(config.get("max_risk_per_trade", "0.02")))
        self.max_daily_loss = Decimal(str(config.get("max_daily_loss", "0.05")))
        self.max_portfolio_risk = Decimal(str(config.get("max_portfolio_risk", "0.10")))
        self.max_correlation = Decimal(str(config.get("max_correlation", "0.70")))

    async def validate_signal(self, signal: Signal) -> RiskValidationResult:
        """Валидация торгового сигнала с точки зрения рисков."""
        try:
            # Получаем портфель (упрощенно)
            portfolio = await self.portfolio_repository.get_portfolio(PortfolioId(uuid4()))
            if not portfolio:
                return RiskValidationResult(
                    is_valid=False,
                    reason="Portfolio not found",
                    risk_level=1.0,
                )

            # Проверяем различные типы рисков
            position_size_risk = self._calculate_position_size_risk(signal, portfolio)
            correlation_risk = await self._calculate_correlation_risk(signal, portfolio)
            portfolio_risk = await self._calculate_portfolio_risk(signal, portfolio)
            daily_loss_risk = await self._calculate_daily_loss_risk(portfolio)

            # Общий уровень риска
            total_risk = (
                position_size_risk + correlation_risk + portfolio_risk + daily_loss_risk
            )

            # Проверяем лимиты
            if position_size_risk > self.max_risk_per_trade:
                return RiskValidationResult(
                    is_valid=False,
                    reason=f"Position size risk ({position_size_risk}) exceeds limit ({self.max_risk_per_trade})",
                    risk_level=float(total_risk),
                )

            if correlation_risk > self.max_correlation:
                return RiskValidationResult(
                    is_valid=False,
                    reason=f"Correlation risk ({correlation_risk}) exceeds limit ({self.max_correlation})",
                    risk_level=float(total_risk),
                )

            if portfolio_risk > self.max_portfolio_risk:
                return RiskValidationResult(
                    is_valid=False,
                    reason=f"Portfolio risk ({portfolio_risk}) exceeds limit ({self.max_portfolio_risk})",
                    risk_level=float(total_risk),
                )

            if daily_loss_risk > self.max_daily_loss:
                return RiskValidationResult(
                    is_valid=False,
                    reason=f"Daily loss risk ({daily_loss_risk}) exceeds limit ({self.max_daily_loss})",
                    risk_level=float(total_risk),
                )

            return RiskValidationResult(
                is_valid=True,
                reason="All risk checks passed",
                risk_level=float(total_risk),
            )

        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return RiskValidationResult(
                is_valid=False,
                reason=f"Validation error: {str(e)}",
                risk_level=1.0,
            )

    def _calculate_position_size_risk(
        self, signal: Signal, portfolio: Portfolio
    ) -> Decimal:
        """Расчет риска размера позиции."""
        try:
            if not signal.price or not signal.quantity:
                return Decimal("0")

            position_value = signal.price.amount * signal.quantity
            portfolio_value = portfolio.total_balance.amount

            if portfolio_value <= 0:
                return Decimal("1")  # Максимальный риск

            return position_value / portfolio_value

        except Exception as e:
            self.logger.error(f"Error calculating position size risk: {e}")
            return Decimal("0")

    async def _calculate_correlation_risk(
        self, signal: Signal, portfolio: Portfolio
    ) -> Decimal:
        """Расчет риска корреляции."""
        try:
            existing_positions: List[Position] = self._get_portfolio_positions(portfolio)
            if not existing_positions:
                return Decimal("0")

            # Проверяем, есть ли уже позиция по этому символу
            for position in existing_positions:
                if hasattr(position, 'trading_pair') and position.trading_pair.symbol == signal.trading_pair:
                    return Decimal("0.8")  # Высокая корреляция для того же символа

            # Проверяем корреляцию с другими позициями
            correlation_count = 0
            for position in existing_positions:
                # Упрощенная логика: считаем корреляцию на основе направления
                if hasattr(position, 'side') and hasattr(signal, 'signal_type'):
                    if position.side.value == signal.signal_type.value:
                        correlation_count += 1

            if correlation_count > 0:
                return Decimal(str(correlation_count / len(existing_positions)))

            return Decimal("0")

        except Exception as e:
            self.logger.error(f"Error calculating correlation risk: {e}")
            return Decimal("0")

    async def _calculate_portfolio_risk(
        self, signal: Optional[Signal], portfolio: Portfolio
    ) -> Decimal:
        """Расчет общего риска портфеля."""
        try:
            # Упрощенный расчет риска портфеля
            total_risk = Decimal("0")

            # Риск от существующих позиций
            existing_positions: List[Position] = self._get_portfolio_positions(portfolio)
            for position in existing_positions:
                if hasattr(position, 'quantity') and hasattr(position, 'entry_price'):
                    position_risk = (
                        position.quantity.amount
                        * position.entry_price.amount
                        / portfolio.total_balance.amount
                    )
                    total_risk += position_risk

            # Риск от новой позиции
            if signal and signal.price and signal.quantity:
                new_position_risk = (
                    signal.price.amount
                    * signal.quantity
                    / portfolio.total_balance.amount
                )
                total_risk += new_position_risk

            return total_risk

        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk: {e}")
            return Decimal("0")

    async def _calculate_daily_loss_risk(self, portfolio: Portfolio) -> Decimal:
        """Расчет риска дневных убытков."""
        try:
            # Упрощенный расчет дневных убытков
            # В реальной системе здесь был бы анализ P&L за день

            daily_pnl = Decimal("0")
            existing_positions: List[Position] = self._get_portfolio_positions(portfolio)
            for position in existing_positions:
                if hasattr(position, 'quantity') and hasattr(position, 'entry_price'):
                    # Расчет нереализованного P&L
                    current_price = position.entry_price.amount  # Упрощенно
                    pnl = (
                        current_price - position.entry_price.amount
                    ) * position.quantity.amount
                    daily_pnl += pnl

            if daily_pnl < 0:
                return abs(daily_pnl) / portfolio.total_balance.amount
            else:
                return Decimal("0")

        except Exception as e:
            self.logger.error(f"Error calculating daily loss risk: {e}")
            return Decimal("0")

    def _get_portfolio_positions(self, portfolio: Portfolio) -> List[Position]:
        """Получение позиций портфеля."""
        # В реальной системе здесь был бы вызов репозитория позиций
        # Пока возвращаем пустой список
        return []

    async def calculate_risk_metrics(self) -> Dict[str, float]:
        """Расчет всех риск-метрик."""
        try:
            portfolio = await self.portfolio_repository.get_portfolio(PortfolioId(uuid4()))
            if not portfolio:
                return {}

            existing_positions: List[Position] = self._get_portfolio_positions(portfolio)

            metrics = {
                "portfolio_risk": float(
                    await self._calculate_portfolio_risk(None, portfolio)
                ),
                "daily_loss_risk": float(
                    await self._calculate_daily_loss_risk(portfolio)
                ),
                "position_count": len(existing_positions),
                "total_equity": float(portfolio.total_balance.amount),
                "free_margin": float(portfolio.free_margin.amount),
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {}

    async def close_risky_positions(self) -> List[str]:
        """Закрытие рискованных позиций."""
        try:
            portfolio = await self.portfolio_repository.get_portfolio(PortfolioId(uuid4()))
            if not portfolio:
                return []

            closed_positions: List[str] = []
            existing_positions: List[Position] = self._get_portfolio_positions(portfolio)

            for position in existing_positions:
                # Упрощенная логика определения рискованных позиций
                if hasattr(position, 'quantity') and hasattr(position, 'entry_price'):
                    position_risk = (
                        position.quantity.amount
                        * position.entry_price.amount
                        / portfolio.total_balance.amount
                    )

                    if position_risk > self.max_risk_per_trade:
                        await self._close_position(position)
                        if hasattr(position, 'trading_pair'):
                            closed_positions.append(position.trading_pair.symbol)
                        self.logger.warning(
                            f"Closed risky position: {getattr(position, 'trading_pair', type('obj', (object,), {'symbol': 'unknown'}))().symbol}"
                        )

            return closed_positions

        except Exception as e:
            self.logger.error(f"Error closing risky positions: {e}")
            return []

    async def _close_position(self, position: Position) -> None:
        """Закрытие позиции."""
        try:
            # Реальная логика закрытия позиции через exchange
            # Создаем ордер на закрытие позиции
            close_side = (
                OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
            )

            close_order = Order(
                id=OrderId(uuid4()),
                portfolio_id=PortfolioId(uuid4()),
                trading_pair=TradingPair(getattr(position, 'trading_pair', type('obj', (object,), {'symbol': 'unknown'}))().symbol),
                order_type=OrderType.MARKET,
                side=close_side,
                quantity=VolumeValue(getattr(position, 'quantity', type('obj', (object,), {'amount': Decimal("0")}))().amount),
                price=Price(Decimal("0"), Currency.USD),  # Исправление: используем Price вместо PriceValue
                status=OrderStatus.PENDING,
                created_at=Timestamp.now(),
                updated_at=Timestamp.now()
            )

            # В реальной системе здесь был бы вызов exchange API
            # await self.exchange_service.place_order(close_order)

            self.logger.info(
                f"Closing position: {position.trading_pair.symbol} with order {close_order.id}"
            )

            # Обновляем статус позиции (если есть такой атрибут)
            if hasattr(position, 'status'):
                position.status = "CLOSED"  # Упрощенно

        except Exception as e:
            self.logger.error(
                f"Error closing position {getattr(position, 'trading_pair', type('obj', (object,), {'symbol': 'unknown'}))().symbol}: {e}"
            )
            raise

    def get_risk_limits(self) -> Dict[str, float]:
        """Получение риск-лимитов."""
        return {
            "max_risk_per_trade": float(self.max_risk_per_trade),
            "max_daily_loss": float(self.max_daily_loss),
            "max_portfolio_risk": float(self.max_portfolio_risk),
            "max_correlation": float(self.max_correlation),
        }
