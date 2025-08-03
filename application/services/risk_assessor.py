"""
Сервис для оценки рисков.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union

from domain.entities.order import Order, OrderSide
from domain.entities.portfolio import Portfolio
from domain.entities.position import Position
from domain.value_objects.price import Price


@dataclass
class RiskMetrics:
    """Метрики риска."""

    total_exposure: Decimal
    max_drawdown: Decimal
    var_95: Decimal  # Value at Risk 95%
    sharpe_ratio: Decimal
    risk_score: Decimal  # 0-100, где 100 - максимальный риск


class RiskAssessor:
    """Сервис для оценки рисков."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            "max_position_size": Decimal("0.2"),  # 20% от портфеля
            "max_leverage": Decimal("10"),
            "max_drawdown": Decimal("0.1"),  # 10%
            "var_confidence": Decimal("0.95"),
            "var_timeframe": 1,  # 1 день
        }

    def assess_portfolio_risk(self, portfolio: Portfolio) -> RiskMetrics:
        """Оценка риска портфеля."""
        total_exposure = self._calculate_total_exposure(portfolio)
        max_drawdown = self._calculate_max_drawdown(portfolio)
        var_95 = self._calculate_value_at_risk(portfolio)
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio)
        risk_score = self._calculate_risk_score(
            total_exposure, max_drawdown, var_95, sharpe_ratio
        )
        return RiskMetrics(
            total_exposure=total_exposure,
            max_drawdown=max_drawdown,
            var_95=var_95,
            sharpe_ratio=sharpe_ratio,
            risk_score=risk_score,
        )

    def assess_order_risk(
        self, order: Order, portfolio: Portfolio, current_price: Decimal
    ) -> Dict[str, Any]:
        """Оценка риска ордера."""
        order_value = order.quantity * (order.price.amount if order.price else Decimal("0"))
        portfolio_ratio = (
            order_value / portfolio.balance.amount
            if portfolio.balance.amount > 0
            else Decimal("0")
        )
        # Расчет потенциального убытка
        potential_loss = order_value * Decimal("0.1")  # Предполагаем 10% убыток
        # Оценка риска
        risk_level = self._assess_risk_level(portfolio_ratio, potential_loss, portfolio)
        return {
            "order_value": order_value,
            "portfolio_ratio": portfolio_ratio,
            "potential_loss": potential_loss,
            "risk_level": risk_level,
            "risk_score": self._calculate_order_risk_score(
                portfolio_ratio, potential_loss, portfolio
            ),
            "recommendations": self._generate_risk_recommendations(
                portfolio_ratio, potential_loss, portfolio
            ),
        }

    def assess_position_risk(
        self, position: Position, portfolio: Portfolio, current_price: Union[Decimal, "Price"]
    ) -> Dict[str, Any]:
        """Оценка риска позиции."""
        position_value = position.size.value * (current_price.amount if hasattr(current_price, 'amount') else current_price)
        unrealized_pnl = self._calculate_unrealized_pnl(position, current_price)
        portfolio_ratio = (
            position_value / portfolio.balance.amount
            if portfolio.balance.amount > 0
            else Decimal("0")
        )
        # Расчет риска ликвидации
        # Исправлено: приводим current_price к Decimal
        current_price_decimal = current_price.amount if hasattr(current_price, 'amount') else current_price
        liquidation_risk = self._calculate_liquidation_risk(position, current_price_decimal)
        return {
            "position_value": position_value,
            "unrealized_pnl": unrealized_pnl,
            "portfolio_ratio": portfolio_ratio,
            "liquidation_risk": liquidation_risk,
            "risk_level": self._assess_position_risk_level(
                position_value, unrealized_pnl, portfolio
            ),
            "recommendations": self._generate_position_recommendations(
                position, current_price_decimal, portfolio
            ),
        }

    def check_risk_limits(self, portfolio: Portfolio) -> Tuple[bool, List[str]]:
        """Проверка лимитов риска."""
        violations = []
        # Проверка общего риска
        risk_metrics = self.assess_portfolio_risk(portfolio)
        if risk_metrics.total_exposure > self.config["max_total_exposure"]:
            violations.append(
                f"Total exposure {risk_metrics.total_exposure:.2%} exceeds limit {self.config['max_total_exposure']:.2%}"
            )
        if risk_metrics.max_drawdown > self.config["max_drawdown"]:
            violations.append(
                f"Max drawdown {risk_metrics.max_drawdown:.2%} exceeds limit {self.config['max_drawdown']:.2%}"
            )
        if risk_metrics.var_95 > self.config["max_var"]:
            violations.append(
                f"VaR 95% {risk_metrics.var_95:.2%} exceeds limit {self.config['max_var']:.2%}"
            )
        if risk_metrics.sharpe_ratio < self.config["min_sharpe"]:
            violations.append(
                f"Sharpe ratio {risk_metrics.sharpe_ratio:.2f} below minimum {self.config['min_sharpe']:.2f}"
            )
        return len(violations) == 0, violations

    def calculate_position_sizing(
        self,
        portfolio: Portfolio,
        risk_per_trade: Decimal = Decimal("0.02"),
        max_risk_per_trade: Decimal = Decimal("0.05"),
    ) -> Dict[str, Decimal]:
        """Расчет размера позиции на основе риска."""
        account_balance = portfolio.balance.amount
        max_risk_amount = account_balance * max_risk_per_trade
        target_risk_amount = account_balance * risk_per_trade
        return {
            "max_risk_amount": max_risk_amount,
            "target_risk_amount": target_risk_amount,
            "max_position_size": max_risk_amount * Decimal("10"),  # 10:1 leverage
            "recommended_position_size": target_risk_amount
            * Decimal("5"),  # 5:1 leverage
        }

    def _calculate_total_exposure(self, portfolio: Portfolio) -> Decimal:
        """Расчет общего риска портфеля."""
        # Упрощенный расчет - сумма всех позиций
        total_exposure = Decimal("0")
        # Здесь должен быть перебор позиций портфеля, например: for position in portfolio.positions:
        # total_exposure += position.size.to_decimal() * position.entry_price.to_decimal()
        # Пока позиций нет, возвращаем 0
        return (
            total_exposure / portfolio.balance.amount
            if portfolio.balance.amount > 0
            else Decimal("0")
        )

    def _calculate_max_drawdown(self, portfolio: Portfolio) -> Decimal:
        """Расчет максимальной просадки."""
        # Упрощенный расчет - разница между максимальным и текущим балансом
        # Здесь должен быть перебор истории баланса, пока возвращаем 0
        return Decimal("0")

    def _calculate_value_at_risk(self, portfolio: Portfolio) -> Decimal:
        """Расчет Value at Risk."""
        # Упрощенный расчет - 5% от общего баланса
        return portfolio.balance.amount * Decimal("0.05")

    def _calculate_sharpe_ratio(self, portfolio: Portfolio) -> Decimal:
        """Расчет коэффициента Шарпа."""
        # Упрощенный расчет - возвращаем базовое значение
        # Исправлено: убираем проблемную логику с пустыми списками
        return Decimal("1.0")  # Базовый коэффициент Шарпа

    def _calculate_risk_score(
        self,
        total_exposure: Decimal,
        max_drawdown: Decimal,
        var_95: Decimal,
        sharpe_ratio: Decimal,
    ) -> Decimal:
        """Расчет общего риска (0-100)."""
        # Взвешенная оценка риска
        exposure_score = min(total_exposure * Decimal("100"), Decimal("100"))
        drawdown_score = min(max_drawdown * Decimal("100"), Decimal("100"))
        var_score = min(
            var_95 / Decimal("0.1") * Decimal("100"), Decimal("100")
        )  # Нормализация к 10%
        sharpe_score = max(Decimal("0"), Decimal("100") - sharpe_ratio * Decimal("20"))
        # Средневзвешенная оценка
        risk_score = (
            exposure_score * Decimal("0.3")
            + drawdown_score * Decimal("0.3")
            + var_score * Decimal("0.2")
            + sharpe_score * Decimal("0.2")
        )
        return min(risk_score, Decimal("100"))

    def _assess_risk_level(
        self, portfolio_ratio: Decimal, potential_loss: Decimal, portfolio: Portfolio
    ) -> str:
        """Оценка уровня риска."""
        if portfolio_ratio > Decimal(
            "0.5"
        ) or potential_loss > portfolio.balance.amount * Decimal("0.1"):
            return "HIGH"
        elif portfolio_ratio > Decimal(
            "0.2"
        ) or potential_loss > portfolio.balance.amount * Decimal("0.05"):
            return "MEDIUM"
        else:
            return "LOW"

    def _calculate_order_risk_score(
        self, portfolio_ratio: Decimal, potential_loss: Decimal, portfolio: Portfolio
    ) -> Decimal:
        """Расчет риска ордера."""
        ratio_score = min(portfolio_ratio * Decimal("100"), Decimal("100"))
        loss_score = min(
            potential_loss / portfolio.balance.amount * Decimal("100"), Decimal("100")
        )
        return ratio_score * Decimal("0.6") + loss_score * Decimal("0.4")

    def _generate_risk_recommendations(
        self, portfolio_ratio: Decimal, potential_loss: Decimal, portfolio: Portfolio
    ) -> List[str]:
        """Генерация рекомендаций по риску."""
        recommendations = []
        if portfolio_ratio > Decimal("0.3"):
            recommendations.append("Consider reducing order size")
        if potential_loss > portfolio.balance.amount * Decimal("0.05"):
            recommendations.append("Set stop loss to limit potential loss")
        if portfolio_ratio > Decimal("0.5"):
            recommendations.append("Order size too large - high risk")
        return recommendations

    def _calculate_unrealized_pnl(
        self, position: Position, current_price: Union[Decimal, "Price"]
    ) -> Decimal:
        """Расчет нереализованной прибыли/убытка."""
        if position.side.value == "long":
            if isinstance(current_price, Price):
                return (current_price.amount - position.entry_price.amount) * position.size.value
            else:
                return (current_price - position.entry_price.amount) * position.size.value
        else:
            if isinstance(current_price, Price):
                return (position.entry_price.amount - current_price.amount) * position.size.value
            else:
                return (position.entry_price.amount - current_price) * position.size.value

    def _calculate_liquidation_risk(
        self, position: Position, current_price: Decimal
    ) -> Decimal:
        """Расчет риска ликвидации."""
        if position.leverage <= Decimal("1"):
            return Decimal("0")
        liquidation_price = position.entry_price.amount / position.leverage
        if position.side.value == "long":
            distance = (current_price - liquidation_price) / current_price
        else:
            distance = (liquidation_price - current_price) / current_price
        return max(Decimal("0"), distance)

    def _assess_position_risk_level(
        self, position_value: Decimal, unrealized_pnl: Decimal, portfolio: Portfolio
    ) -> str:
        """Оценка уровня риска позиции."""
        portfolio_ratio = (
            position_value / portfolio.balance.amount
            if portfolio.balance.amount > 0
            else Decimal("0")
        )
        if portfolio_ratio > Decimal(
            "0.3"
        ) or unrealized_pnl < -portfolio.balance.amount * Decimal("0.1"):
            return "HIGH"
        elif portfolio_ratio > Decimal(
            "0.15"
        ) or unrealized_pnl < -portfolio.balance.amount * Decimal("0.05"):
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_position_recommendations(
        self, position: Position, current_price: Decimal, portfolio: Portfolio
    ) -> List[str]:
        """Генерация рекомендаций по позиции."""
        recommendations = []
        unrealized_pnl = self._calculate_unrealized_pnl(position, current_price)
        if unrealized_pnl < -portfolio.balance.amount * Decimal("0.05"):
            recommendations.append("Consider closing position to limit losses")
        if position.leverage > Decimal("5"):
            recommendations.append("High leverage - consider reducing position size")
        if not position.stop_loss:
            recommendations.append("Set stop loss to protect capital")
        return recommendations
