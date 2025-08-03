"""
Сервис для расчета параметров ордеров.
"""

from decimal import Decimal
from typing import Any, Dict, Tuple

from domain.entities.order import Order, OrderSide, OrderType


class OrderCalculator:
    """Сервис для расчета параметров ордеров."""

    def calculate_order_value(self, quantity: Decimal, price: Decimal) -> Decimal:
        """Расчет стоимости ордера."""
        return quantity * price

    def calculate_commission(
        self, order_value: Decimal, commission_rate: Decimal = Decimal("0.001")
    ) -> Decimal:
        """Расчет комиссии."""
        return order_value * commission_rate

    def calculate_total_cost(
        self, order_value: Decimal, commission_rate: Decimal = Decimal("0.001")
    ) -> Tuple[Decimal, Decimal]:
        """Расчет общей стоимости с комиссией."""
        commission = self.calculate_commission(order_value, commission_rate)
        total_cost = order_value + commission
        return total_cost, commission

    def calculate_optimal_order_size(
        self,
        available_funds: Decimal,
        current_price: Decimal,
        risk_percentage: Decimal = Decimal("0.02"),
        commission_rate: Decimal = Decimal("0.001"),
    ) -> Tuple[Decimal, Decimal]:
        """Расчет оптимального размера ордера."""
        # Учитываем комиссию при расчете
        max_order_value = available_funds * risk_percentage
        commission_factor = Decimal("1") + commission_rate
        optimal_order_value = max_order_value / commission_factor
        optimal_quantity = optimal_order_value / current_price
        return optimal_quantity, optimal_order_value

    def calculate_stop_loss_price(
        self,
        entry_price: Decimal,
        side: OrderSide,
        stop_loss_percentage: Decimal = Decimal("0.02"),
    ) -> Decimal:
        """Расчет цены стоп-лосса."""
        if side == OrderSide.BUY:
            # Для длинной позиции стоп-лосс ниже цены входа
            stop_loss_price = entry_price * (Decimal("1") - stop_loss_percentage)
        else:
            # Для короткой позиции стоп-лосс выше цены входа
            stop_loss_price = entry_price * (Decimal("1") + stop_loss_percentage)
        return stop_loss_price

    def calculate_take_profit_price(
        self,
        entry_price: Decimal,
        side: OrderSide,
        take_profit_percentage: Decimal = Decimal("0.04"),
    ) -> Decimal:
        """Расчет цены тейк-профита."""
        if side == OrderSide.BUY:
            # Для длинной позиции тейк-профит выше цены входа
            take_profit_price = entry_price * (Decimal("1") + take_profit_percentage)
        else:
            # Для короткой позиции тейк-профит ниже цены входа
            take_profit_price = entry_price * (Decimal("1") - take_profit_percentage)
        return take_profit_price

    def calculate_risk_reward_ratio(
        self,
        entry_price: Decimal,
        stop_loss_price: Decimal,
        take_profit_price: Decimal,
        side: OrderSide,
    ) -> Decimal:
        """Расчет соотношения риск/прибыль."""
        if side == OrderSide.BUY:
            risk = entry_price - stop_loss_price
            reward = take_profit_price - entry_price
        else:
            risk = stop_loss_price - entry_price
            reward = entry_price - take_profit_price
        if risk == 0:
            return Decimal("0")
        return reward / risk

    def calculate_position_size_for_risk(
        self,
        account_balance: Decimal,
        risk_amount: Decimal,
        entry_price: Decimal,
        stop_loss_price: Decimal,
        side: OrderSide,
    ) -> Tuple[Decimal, Decimal]:
        """Расчет размера позиции на основе риска."""
        if side == OrderSide.BUY:
            price_difference = entry_price - stop_loss_price
        else:
            price_difference = stop_loss_price - entry_price
        if price_difference <= 0:
            return Decimal("0"), Decimal("0")
        # Размер позиции = риск / разница в цене
        position_size = risk_amount / price_difference
        position_value = position_size * entry_price
        return position_size, position_value

    def calculate_martingale_order_size(
        self,
        base_order_size: Decimal,
        martingale_multiplier: Decimal = Decimal("2"),
        max_orders: int = 5,
    ) -> Dict[int, Decimal]:
        """Расчет размеров ордеров для стратегии Мартингейла."""
        order_sizes = {}
        for i in range(max_orders):
            order_sizes[i + 1] = base_order_size * (martingale_multiplier**i)
        return order_sizes

    def calculate_grid_order_prices(
        self,
        base_price: Decimal,
        grid_spacing: Decimal = Decimal("0.01"),
        grid_levels: int = 10,
        side: OrderSide = OrderSide.BUY,
    ) -> Dict[int, Decimal]:
        """Расчет цен для сетки ордеров."""
        grid_prices = {}
        for i in range(grid_levels):
            if side == OrderSide.BUY:
                # Для покупок цена уменьшается с каждым уровнем
                price = base_price * (Decimal("1") - grid_spacing * i)
            else:
                # Для продаж цена увеличивается с каждым уровнем
                price = base_price * (Decimal("1") + grid_spacing * i)
            grid_prices[i + 1] = price
        return grid_prices

    def calculate_dca_order_sizes(
        self,
        total_investment: Decimal,
        dca_levels: int = 5,
        dca_percentage: Decimal = Decimal("0.2"),
    ) -> Dict[int, Decimal]:
        """Расчет размеров ордеров для стратегии DCA."""
        order_sizes = {}
        remaining_investment = total_investment
        for i in range(dca_levels):
            if i == dca_levels - 1:
                # Последний ордер использует оставшиеся средства
                order_size = remaining_investment
            else:
                # Обычный ордер
                order_size = total_investment * dca_percentage
                remaining_investment -= order_size
            order_sizes[i + 1] = order_size
        return order_sizes

    def calculate_break_even_price(
        self,
        entry_price: Decimal,
        side: OrderSide,
        commission_rate: Decimal = Decimal("0.001"),
    ) -> Decimal:
        """Расчет цены безубыточности."""
        commission_cost = entry_price * commission_rate
        if side == OrderSide.BUY:
            # Для длинной позиции нужно покрыть комиссию
            break_even_price = entry_price + commission_cost
        else:
            # Для короткой позиции нужно покрыть комиссию
            break_even_price = entry_price - commission_cost
        return break_even_price

    def calculate_order_metrics(
        self,
        order: Order,
        current_price: Decimal,
        commission_rate: Decimal = Decimal("0.001"),
    ) -> Dict[str, Any]:
        """Расчет метрик ордера."""
        # Проверяем, что order.price не None
        if order.price is None:
            raise ValueError("Order price cannot be None")
            
        # Получаем значения цены и количества
        if hasattr(order.price, 'amount'):
            price_value = order.price.amount
        else:
            price_value = order.price  # type: ignore
            
        if hasattr(order.quantity, 'amount'):
            quantity_value = order.quantity.amount
        else:
            quantity_value = order.quantity  # type: ignore
            
        # Расчет стоимости ордера
        order_value = self.calculate_order_value(quantity_value, price_value)
        
        # Расчет комиссии
        commission = self.calculate_commission(order_value, commission_rate)
        
        # Расчет потенциального P&L
        if order.side == OrderSide.BUY:
            potential_pnl = (current_price - price_value) * quantity_value
        else:
            potential_pnl = (price_value - current_price) * quantity_value
            
        # Расчет ROI
        roi = (
            potential_pnl / order_value if order_value > 0 else Decimal("0")
        )
        
        # Получаем стоп-цену
        if hasattr(order, 'stop_price') and order.stop_price:
            if hasattr(order.stop_price, 'amount'):
                stop_price_value = order.stop_price.amount
            else:
                stop_price_value = order.stop_price  # type: ignore
        else:
            stop_price_value = price_value
            
        # Расчет рисков
        risk_metrics = {
            "entry_price": price_value,
            "stop_price": stop_price_value,
            "take_profit": price_value * Decimal("1.02"),  # Упрощенно
        }
        
        return {
            "order_value": order_value,
            "commission": commission,
            "total_cost": order_value + commission,
            "potential_pnl": potential_pnl,
            "roi": roi,
            "risk_reward_ratio": self.calculate_risk_reward_ratio(
                price_value,
                stop_price_value,
                price_value * Decimal("1.02"),  # Упрощенно
                order.side,
            ),
        }
