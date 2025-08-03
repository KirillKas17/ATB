"""
Контроллер для управления рисками.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class RiskController:
    """Контроллер для управления рисками."""

    def __init__(self, config: Dict[str, Any]):
        """Инициализация контроллера.
        
        Args:
            config: Конфигурация рисков
        """
        self.config = config
        self.max_position_size = config.get("max_position_size", 1000.0)
        self.max_daily_loss = config.get("max_daily_loss", 100.0)
        self.max_leverage = config.get("max_leverage", 10)
        self.stop_loss_pct = config.get("stop_loss_pct", 5.0)
        self.take_profit_pct = config.get("take_profit_pct", 10.0)
        self.max_open_positions = config.get("max_open_positions", 5)

    async def validate_order(
        self, 
        symbol: str, 
        side: str, 
        amount: float, 
        price: float,
        current_positions: List[Dict[str, Any]],
        account_balance: float
    ) -> Dict[str, Any]:
        """Валидация ордера с точки зрения рисков.
        
        Args:
            symbol: Торговая пара
            side: Сторона (buy/sell)
            amount: Количество
            price: Цена
            current_positions: Текущие позиции
            account_balance: Баланс аккаунта
            
        Returns:
            Результат валидации
        """
        try:
            validation_result: Dict[str, Any] = {
                "is_valid": True,
                "warnings": [],
                "errors": []
            }
            
            # Проверка размера позиции
            position_value = amount * price
            if position_value > self.max_position_size:
                validation_result["is_valid"] = False
                validation_result["errors"].append(
                    f"Position size {position_value} exceeds maximum {self.max_position_size}"
                )
                
            # Проверка баланса
            if position_value > account_balance:
                validation_result["is_valid"] = False
                validation_result["errors"].append(
                    f"Insufficient balance: {account_balance} < {position_value}"
                )
                
            # Проверка количества открытых позиций
            if len(current_positions) >= self.max_open_positions:
                validation_result["is_valid"] = False
                validation_result["errors"].append(
                    f"Maximum open positions reached: {len(current_positions)} >= {self.max_open_positions}"
                )
                
            # Проверка дневных потерь
            daily_pnl = sum(pos.get("unrealized_pnl", 0) for pos in current_positions)
            if daily_pnl < -self.max_daily_loss:
                validation_result["is_valid"] = False
                validation_result["errors"].append(
                    f"Daily loss limit exceeded: {daily_pnl} < -{self.max_daily_loss}"
                )
                
            return validation_result
        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return {
                "is_valid": False,
                "warnings": [],
                "errors": [f"Validation error: {str(e)}"]
            }

    async def calculate_position_risk(
        self, 
        symbol: str, 
        position: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Расчет риска позиции.
        
        Args:
            symbol: Торговая пара
            position: Информация о позиции
            market_data: Рыночные данные
            
        Returns:
            Оценка риска
        """
        try:
            entry_price = position["entry_price"]
            current_price = market_data.get("last_price", entry_price)
            size = position["size"]
            leverage = position.get("leverage", 1)
            
            # Рассчитываем PnL
            if position["side"] == "buy":
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
                
            # Рассчитываем риск
            risk_score = abs(pnl_pct) * leverage / 100
            
            # Определяем уровень риска
            if risk_score > 0.5:
                risk_level = "high"
            elif risk_score > 0.2:
                risk_level = "medium"
            else:
                risk_level = "low"
                
            return {
                "symbol": symbol,
                "risk_level": risk_level,
                "risk_score": risk_score,
                "pnl_pct": pnl_pct,
                "leverage": leverage,
                "position_size": size,
                "entry_price": entry_price,
                "current_price": current_price,
                "stop_loss_price": entry_price * (1 - self.stop_loss_pct / 100),
                "take_profit_price": entry_price * (1 + self.take_profit_pct / 100)
            }
        except Exception as e:
            logger.error(f"Error calculating position risk: {e}")
            return {
                "symbol": symbol,
                "risk_level": "unknown",
                "risk_score": 0.0,
                "error": str(e)
            }

    async def get_portfolio_risk(
        self, 
        positions: List[Dict[str, Any]],
        account_balance: float
    ) -> Dict[str, Any]:
        """Расчет риска портфеля.
        
        Args:
            positions: Список позиций
            account_balance: Баланс аккаунта
            
        Returns:
            Оценка риска портфеля
        """
        try:
            total_pnl = sum(pos.get("unrealized_pnl", 0) for pos in positions)
            total_exposure = sum(pos.get("size", 0) * pos.get("entry_price", 0) for pos in positions)
            
            # Рассчитываем метрики риска
            exposure_ratio = total_exposure / account_balance if account_balance > 0 else 0
            pnl_ratio = total_pnl / account_balance if account_balance > 0 else 0
            
            # Определяем уровень риска портфеля
            if exposure_ratio > 2.0 or pnl_ratio < -0.1:
                portfolio_risk = "high"
            elif exposure_ratio > 1.0 or pnl_ratio < -0.05:
                portfolio_risk = "medium"
            else:
                portfolio_risk = "low"
                
            return {
                "portfolio_risk": portfolio_risk,
                "total_exposure": total_exposure,
                "total_pnl": total_pnl,
                "exposure_ratio": exposure_ratio,
                "pnl_ratio": pnl_ratio,
                "position_count": len(positions),
                "account_balance": account_balance
            }
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return {
                "portfolio_risk": "unknown",
                "error": str(e)
            }

    async def should_close_position(
        self, 
        position: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Проверка необходимости закрытия позиции.
        
        Args:
            position: Информация о позиции
            market_data: Рыночные данные
            
        Returns:
            Рекомендация по закрытию
        """
        try:
            entry_price = position["entry_price"]
            current_price = market_data.get("last_price", entry_price)
            
            # Рассчитываем процент изменения
            if position["side"] == "buy":
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
                
            should_close = False
            reason = ""
            
            # Проверяем stop loss
            if pnl_pct < -self.stop_loss_pct:
                should_close = True
                reason = f"Stop loss triggered: {pnl_pct:.2f}% < -{self.stop_loss_pct}%"
                
            # Проверяем take profit
            elif pnl_pct > self.take_profit_pct:
                should_close = True
                reason = f"Take profit triggered: {pnl_pct:.2f}% > {self.take_profit_pct}%"
                
            return {
                "should_close": should_close,
                "reason": reason,
                "pnl_pct": pnl_pct,
                "entry_price": entry_price,
                "current_price": current_price
            }
        except Exception as e:
            logger.error(f"Error checking position closure: {e}")
            return {
                "should_close": False,
                "reason": f"Error: {str(e)}",
                "pnl_pct": 0.0
            }

    async def get_risk_alerts(
        self, 
        positions: List[Dict[str, Any]],
        account_balance: float
    ) -> List[Dict[str, Any]]:
        """Получение предупреждений о рисках.
        
        Args:
            positions: Список позиций
            account_balance: Баланс аккаунта
            
        Returns:
            Список предупреждений
        """
        try:
            alerts = []
            
            # Проверяем общий риск портфеля
            portfolio_risk = await self.get_portfolio_risk(positions, account_balance)
            if portfolio_risk["portfolio_risk"] == "high":
                alerts.append({
                    "type": "portfolio_risk",
                    "severity": "high",
                    "message": f"High portfolio risk: {portfolio_risk['exposure_ratio']:.2f}x exposure",
                    "timestamp": datetime.now().isoformat()
                })
                
            # Проверяем дневные потери
            total_pnl = sum(pos.get("unrealized_pnl", 0) for pos in positions)
            if total_pnl < -self.max_daily_loss:
                alerts.append({
                    "type": "daily_loss",
                    "severity": "high",
                    "message": f"Daily loss limit exceeded: {total_pnl:.2f}",
                    "timestamp": datetime.now().isoformat()
                })
                
            # Проверяем количество позиций
            if len(positions) >= self.max_open_positions:
                alerts.append({
                    "type": "max_positions",
                    "severity": "medium",
                    "message": f"Maximum positions reached: {len(positions)}",
                    "timestamp": datetime.now().isoformat()
                })
                
            return alerts
        except Exception as e:
            logger.error(f"Error getting risk alerts: {e}")
            return [{
                "type": "error",
                "severity": "high",
                "message": f"Error calculating risk alerts: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }] 