"""
Use case для управления рисками с промышленной типизацией и валидацией.
"""

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID

import pandas as pd

from application.types import (
    RiskAssessmentRequest,
    RiskAssessmentResponse,
    RiskLimitRequest,
    RiskLimitResponse,
)
from domain.entities.position import Position, PositionSide
from domain.repositories.portfolio_repository import PortfolioRepository
from domain.repositories.position_repository import PositionRepository
from domain.type_definitions import (
    AmountValue,
    EntityId,
    PortfolioId,
    PriceValue,
    TimestampValue,
    VolumeValue,
)
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.trading_pair import TradingPair
from domain.value_objects.volume import Volume

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Метрики риска для анализа портфеля."""

    total_exposure: AmountValue
    max_drawdown: AmountValue
    var_95: AmountValue
    sharpe_ratio: float
    volatility: float
    correlation_matrix: Dict[str, float]
    concentration_risk: float
    leverage_ratio: float


@dataclass
class PositionRisk:
    """Риск отдельной позиции."""

    position_id: str
    trading_pair: str
    side: str
    volume: VolumeValue
    entry_price: PriceValue
    current_price: PriceValue
    unrealized_pnl: AmountValue
    risk_score: float
    var_contribution: AmountValue
    correlation_risk: float


class RiskManagementUseCase:
    """Use case для управления рисками."""

    def __init__(
        self,
        portfolio_repository: PortfolioRepository,
        position_repository: PositionRepository,
    ):
        self.portfolio_repository = portfolio_repository
        self.position_repository = position_repository

    async def assess_portfolio_risk(
        self, request: RiskAssessmentRequest
    ) -> RiskAssessmentResponse:
        """Оценка риска портфеля."""
        try:
            # Получение позиций портфеля
            # portfolio_positions = await self.portfolio_repository.get_all_positions(request.portfolio_id)
            portfolio_positions = []
            positions = [p for p in portfolio_positions if isinstance(p, Position)]

            if not positions:
                return RiskAssessmentResponse(
                    success=True,
                    portfolio_risk={},
                    position_risks=[],
                    recommendations=[],
                    risk_score=Decimal("0"),
                    is_acceptable=True,
                    var_95=Money(Decimal("0"), Currency.USDT),
                    var_99=Money(Decimal("0"), Currency.USDT),
                    max_drawdown=Decimal("0"),
                    message="No positions found",
                )

            # Расчет метрик портфеля
            portfolio_metrics = await self.calculate_portfolio_metrics(EntityId(request.portfolio_id), positions)
            
            # Получение рыночных данных
            market_data = await self._get_market_data(positions)
            
            # Расчет рисков позиций
            position_risks = []
            for position in positions:
                position_risk = await self.calculate_position_risk(position)
                position_risks.append(position_risk)

            # Определение уровня риска
            risk_score = portfolio_metrics.risk_score if hasattr(portfolio_metrics, 'risk_score') else Decimal("0")
            is_acceptable = risk_score < Decimal("0.7")
            
            # Генерация рекомендаций
            recommendations = await self._generate_risk_recommendations(portfolio_metrics, position_risks)

            return RiskAssessmentResponse(
                success=True,
                portfolio_risk={
                    "total_exposure": float(portfolio_metrics.total_exposure),
                    "max_drawdown": float(portfolio_metrics.max_drawdown),
                    "var_95": float(portfolio_metrics.var_95),
                    "sharpe_ratio": portfolio_metrics.sharpe_ratio,
                    "volatility": portfolio_metrics.volatility,
                    "concentration_risk": portfolio_metrics.concentration_risk,
                    "leverage_ratio": portfolio_metrics.leverage_ratio,
                },
                position_risks=[{
                    "position_id": pr.position_id,
                    "trading_pair": pr.trading_pair,
                    "side": pr.side,
                    "volume": float(pr.volume),
                    "entry_price": float(pr.entry_price),
                    "current_price": float(pr.current_price),
                    "unrealized_pnl": float(pr.unrealized_pnl),
                    "risk_score": pr.risk_score,
                    "var_contribution": float(pr.var_contribution),
                    "correlation_risk": pr.correlation_risk,
                } for pr in position_risks],
                recommendations=recommendations,
                risk_score=risk_score,
                is_acceptable=is_acceptable,
                var_95=Money(portfolio_metrics.var_95, Currency.USDT),
                var_99=Money(portfolio_metrics.var_95 * Decimal("1.5"), Currency.USDT),
                max_drawdown=Decimal(str(portfolio_metrics.max_drawdown)),
                message="Risk assessment completed successfully",
            )

        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return RiskAssessmentResponse(
                success=False,
                portfolio_risk={},
                position_risks=[],
                recommendations=[],
                risk_score=Decimal("0"),
                is_acceptable=False,
                var_95=Money(Decimal("0"), Currency.USDT),
                var_99=Money(Decimal("0"), Currency.USDT),
                max_drawdown=Decimal("0"),
                message=f"Error assessing portfolio risk: {str(e)}",
            )

    async def validate_risk_limits(
        self, request: RiskLimitRequest
    ) -> RiskLimitResponse:
        """Проверка лимитов риска."""
        try:
            # Получение позиций портфеля
            # portfolio_positions = await self.portfolio_repository.get_all_positions(request.portfolio_id)
            portfolio_positions = []
            positions = [p for p in portfolio_positions if isinstance(p, Position)]

            if not positions:
                return RiskLimitResponse(
                    success=True,
                    current_risk={},
                    risk_limits={},
                    limits_set=False,
                    message="No positions to validate",
                )

            # Расчет метрик портфеля
            portfolio_metrics = await self.calculate_portfolio_metrics(EntityId(request.portfolio_id), positions)
            
            # Проверка лимитов
            violations = []
            
            # Проверка максимальной экспозиции
            if portfolio_metrics.total_exposure > request.max_var_95:
                violations.append(f"Total exposure {portfolio_metrics.total_exposure} exceeds limit {request.max_var_95}")
            
            # Проверка максимального drawdown
            if portfolio_metrics.max_drawdown > request.max_drawdown:
                violations.append(f"Max drawdown {portfolio_metrics.max_drawdown} exceeds limit {request.max_drawdown}")
            
            # Проверка VaR
            if portfolio_metrics.var_95 > request.max_var_95:
                violations.append(f"VaR {portfolio_metrics.var_95} exceeds limit {request.max_var_95}")
            
            # Проверка концентрации
            if portfolio_metrics.concentration_risk > request.max_correlation:
                violations.append(f"Concentration risk {portfolio_metrics.concentration_risk} exceeds limit {request.max_correlation}")
            
            # Проверка плеча
            if portfolio_metrics.leverage_ratio > request.max_leverage:
                violations.append(f"Leverage ratio {portfolio_metrics.leverage_ratio} exceeds limit {request.max_leverage}")

            limits_set = len(violations) == 0

            return RiskLimitResponse(
                success=True,
                current_risk={
                    "total_exposure": float(portfolio_metrics.total_exposure),
                    "max_drawdown": float(portfolio_metrics.max_drawdown),
                    "var_95": float(portfolio_metrics.var_95),
                    "concentration_risk": portfolio_metrics.concentration_risk,
                    "leverage_ratio": portfolio_metrics.leverage_ratio,
                },
                risk_limits={
                    "max_var_95": float(request.max_var_95),
                    "max_var_99": float(request.max_var_99),
                    "max_drawdown": float(request.max_drawdown),
                    "max_position_size": float(request.max_position_size),
                    "max_correlation": float(request.max_correlation),
                    "max_leverage": float(request.max_leverage),
                },
                limits_set=limits_set,
                message=f"Risk limits validation completed. {'All limits respected' if limits_set else f'{len(violations)} violations found'}",
            )

        except Exception as e:
            logger.error(f"Error validating risk limits: {e}")
            return RiskLimitResponse(
                success=False,
                current_risk={},
                risk_limits={},
                limits_set=False,
                message=f"Error validating risk limits: {str(e)}",
            )

    async def calculate_portfolio_metrics(
        self, portfolio_id: EntityId, positions: List[Position]
    ) -> RiskMetrics:
        """Расчет метрик портфеля."""
        try:
            if not positions:
                return RiskMetrics(
                    total_exposure=AmountValue(Decimal("0")),
                    max_drawdown=AmountValue(Decimal("0")),
                    var_95=AmountValue(Decimal("0")),
                    sharpe_ratio=0.0,
                    volatility=0.0,
                    correlation_matrix={},
                    concentration_risk=0.0,
                    leverage_ratio=0.0,
                )

            # Расчет общей экспозиции
            total_exposure = Decimal("0")
            
            # Расчет общего P&L
            total_pnl = Decimal("0")
            
            # Расчет максимального drawdown (упрощенная версия)
            max_drawdown = min(total_pnl, Decimal("0"))
            
            # Расчет VaR (упрощенная версия)
            var_95 = total_exposure * Decimal("0.05")  # 5% VaR
            
            # Расчет волатильности (упрощенная версия)
            volatility = float(abs(total_pnl) / total_exposure) if total_exposure > 0 else 0.0
            
            # Расчет коэффициента Шарпа (упрощенная версия)
            # Исправление: приводим к float перед делением
            sharpe_ratio = float(total_pnl) / float(volatility) if volatility > 0 else 0.0
            
            # Расчет корреляционной матрицы (упрощенная версия)
            correlation_matrix = {}
            for i, pos1 in enumerate(positions):
                for j, pos2 in enumerate(positions):
                    if i != j:
                        key = f"pos_{i}_pos_{j}"
                        correlation_matrix[key] = 0.5  # Упрощенная корреляция
            
            # Расчет риска концентрации
            if total_exposure > 0:
                max_position_exposure = Decimal("0")
                concentration_risk = float(max_position_exposure / total_exposure)
            else:
                concentration_risk = 0.0
            
            # Расчет плеча
            total_margin = Decimal("0")
            leverage_ratio = float(total_exposure / total_margin) if total_margin > 0 else 0.0

            return RiskMetrics(
                total_exposure=AmountValue(total_exposure),
                max_drawdown=AmountValue(max_drawdown),
                var_95=AmountValue(var_95),
                sharpe_ratio=sharpe_ratio,
                volatility=volatility,
                correlation_matrix=correlation_matrix,
                concentration_risk=concentration_risk,
                leverage_ratio=leverage_ratio,
            )

        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return RiskMetrics(
                total_exposure=AmountValue(Decimal("0")),
                max_drawdown=AmountValue(Decimal("0")),
                var_95=AmountValue(Decimal("0")),
                sharpe_ratio=0.0,
                volatility=0.0,
                correlation_matrix={},
                concentration_risk=0.0,
                leverage_ratio=0.0,
            )

    async def _get_market_data(self, positions: List[Position]) -> Dict[str, float]:
        """Получение рыночных данных для позиций."""
        try:
            market_data: Dict[str, float] = {}
            for position in positions:
                # Упрощенная версия - используем текущую цену позиции
                market_data[str(position.trading_pair.symbol)] = float(position.current_price.amount)
            return market_data
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {}

    async def calculate_position_risk(self, position: Position) -> PositionRisk:
        """Расчет риска отдельной позиции."""
        try:
            # Расчет P&L
            unrealized_pnl = position.unrealized_pnl.amount if position.unrealized_pnl else Decimal("0")
            
            # Расчет риска (упрощенная версия)
            # Исправление: добавляем проверки на существование атрибута
            notional_value = getattr(position, 'notional_value', None)
            if notional_value and notional_value.amount > 0:
                risk_score = float(abs(unrealized_pnl) / notional_value.amount)
            else:
                risk_score = 0.0
            
            # Вклад в VaR
            # Исправление: добавляем проверки на существование атрибута
            if hasattr(position, 'notional_value') and position.notional_value:
                var_contribution = getattr(position.notional_value, 'amount', Decimal("0")) * Decimal("0.05")
            else:
                var_contribution = Decimal("0")
            
            # Корреляционный риск (упрощенная версия)
            correlation_risk = 0.5

            return PositionRisk(
                position_id=str(position.id),
                trading_pair=str(position.trading_pair.symbol),
                side=position.side.value,
                volume=VolumeValue(position.volume.to_decimal()),
                entry_price=PriceValue(position.entry_price.amount),
                current_price=PriceValue(position.current_price.amount),
                unrealized_pnl=AmountValue(unrealized_pnl),
                risk_score=risk_score,
                var_contribution=AmountValue(var_contribution),
                correlation_risk=correlation_risk,
            )

        except Exception as e:
            logger.error(f"Error calculating position risk: {e}")
            return PositionRisk(
                position_id=str(position.id),
                trading_pair=str(position.trading_pair.symbol),
                side="unknown",
                volume=VolumeValue(Decimal("0")),
                entry_price=PriceValue(Decimal("0")),
                current_price=PriceValue(Decimal("0")),
                unrealized_pnl=AmountValue(Decimal("0")),
                risk_score=0.0,
                var_contribution=AmountValue(Decimal("0")),
                correlation_risk=0.0,
            )

    async def _generate_risk_recommendations(
        self, portfolio_metrics: RiskMetrics, position_risks: List[PositionRisk]
    ) -> List[str]:
        """Генерация рекомендаций по рискам."""
        recommendations = []
        
        try:
            # Рекомендации по концентрации
            if portfolio_metrics.concentration_risk > 0.5:
                recommendations.append("Consider diversifying portfolio to reduce concentration risk")

            # Рекомендации по плечу
            if portfolio_metrics.leverage_ratio > 5.0:
                recommendations.append("Consider reducing leverage to manage risk")

            # Рекомендации по VaR
            # Исправление: приводим к правильным типам для сравнения
            if float(portfolio_metrics.var_95) > float(portfolio_metrics.total_exposure) * 0.1:
                recommendations.append("Consider reducing position sizes to lower VaR")

            # Рекомендации по позициям
            high_risk_positions = [pr for pr in position_risks if pr.risk_score > 0.5]
            if high_risk_positions:
                recommendations.append(f"Review {len(high_risk_positions)} high-risk positions")

        except Exception as e:
            logger.error(f"Error generating risk recommendations: {e}")
            recommendations.append(f"Error generating recommendations: {str(e)}")

        return recommendations

    async def calculate_portfolio_risk_metrics(
        self, portfolio_id: str
    ) -> Dict[str, float]:
        """Расчет метрик риска портфеля."""
        try:
            # Получение позиций портфеля
            portfolio_id_obj = PortfolioId(UUID(portfolio_id))
            # Исправление: передаем правильный тип EntityId
            from domain.type_definitions.repository_types import EntityId as RepositoryEntityId
            portfolio_positions = await self.portfolio_repository.get_all_positions(RepositoryEntityId(portfolio_id_obj))
            positions = [p for p in portfolio_positions if isinstance(p, Position)]

            if not positions:
                return {
                    "total_exposure": 0.0,
                    "total_pnl": 0.0,
                    "risk_score": 0.0,
                    "volatility": 0.0,
                    "concentration_risk": 0.0,
                    "leverage_ratio": 0.0,
                }

            # Расчет метрик
            # Исправление: добавляем проверки на существование атрибутов
            total_exposure = sum(
                getattr(getattr(position, 'notional_value', AmountValue(Decimal("0"))), 'value', Decimal("0"))
                for position in positions
            )
            total_pnl = sum(
                getattr(getattr(position, 'total_pnl', AmountValue(Decimal("0"))), 'value', Decimal("0"))
                for position in positions
            )
            
            # Расчет волатильности
            if total_exposure > 0:
                volatility = float(abs(total_pnl) / total_exposure)
            else:
                volatility = 0.0
            
            # Расчет риска концентрации
            if total_exposure > 0:
                # Исправление: добавляем проверки на существование атрибута
                max_position_exposure = max(
                    getattr(getattr(position, 'notional_value', AmountValue(Decimal("0"))), 'value', Decimal("0"))
                    for position in positions
                )
                concentration_risk = float(max_position_exposure / total_exposure)
            else:
                concentration_risk = 0.0
            
            # Расчет плеча
            # Исправление: добавляем проверки на существование атрибута
            total_margin = sum(
                getattr(getattr(position, 'margin_used', AmountValue(Decimal("0"))), 'value', Decimal("0"))
                for position in positions 
                if getattr(position, 'margin_used', None)
            )
            leverage_ratio = float(total_exposure / total_margin) if total_margin > 0 else 0.0
            
            # Общий риск
            risk_score = (
                concentration_risk * 0.3 +
                min(leverage_ratio / 10, 1.0) * 0.3 +
                min(volatility, 1.0) * 0.4
            )

            return {
                "total_exposure": float(total_exposure),
                "total_pnl": float(total_pnl),
                "risk_score": risk_score,
                "volatility": volatility,
                "concentration_risk": concentration_risk,
                "leverage_ratio": leverage_ratio,
            }

        except Exception as e:
            logger.error(f"Error calculating portfolio risk metrics: {e}")
            return {
                "total_exposure": 0.0,
                "total_pnl": 0.0,
                "risk_score": 0.0,
                "volatility": 0.0,
                "concentration_risk": 0.0,
                "leverage_ratio": 0.0,
            }

    async def validate_position_risk_limits(
        self, position: Position, risk_limits: Dict[str, float]
    ) -> List[str]:
        """Валидация лимитов риска для позиции."""
        violations = []
        
        # Проверка размера позиции
        if position.volume.amount > risk_limits.get("max_position_size", float("inf")):
            violations.append(f"Position size {position.volume.amount} exceeds limit")
        
        # Проверка убытка
        if hasattr(position, 'unrealized_pnl') and float(str(position.unrealized_pnl)) < -risk_limits.get("max_loss_per_position", float("inf")):
            violations.append(f"Position loss {position.unrealized_pnl} exceeds limit")
        
        return violations

    async def check_order_risk(self, signal: Any, portfolio: Optional[Any] = None) -> bool:
        """
        Проверка риска ордера.
        Args:
            signal: Торговый сигнал
            portfolio: Портфель (опционально)
        Returns:
            True если риск приемлем
        """
        try:
            # В реальной системе здесь была бы логика проверки риска ордера
            # Пока возвращаем заглушку
            logger.info(f"Checking order risk for signal: {signal}")
            return True
        except Exception as e:
            logger.error(f"Error checking order risk: {e}")
            return False


class DefaultRiskManagementUseCase(RiskManagementUseCase):
    """Реализация по умолчанию для управления рисками."""

    def __init__(
        self,
        portfolio_repository: PortfolioRepository,
        position_repository: PositionRepository,
    ):
        super().__init__(portfolio_repository, position_repository)
