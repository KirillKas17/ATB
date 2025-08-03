"""
Главный контроллер для управления торговыми операциями.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import asyncio
import logging

from infrastructure.core.exchange import Exchange
from infrastructure.core.controllers.market_controller import MarketController
from infrastructure.core.controllers.order_controller import OrderController
from infrastructure.core.controllers.position_controller import PositionController
from infrastructure.core.controllers.risk_controller import RiskController

logger = logging.getLogger(__name__)


class TradingController:
    """Главный контроллер для управления торговыми операциями."""

    def __init__(self, exchange: Exchange, config: Dict[str, Any]):
        """Инициализация контроллера.
        
        Args:
            exchange: Экземпляр биржи
            config: Конфигурация торговли
        """
        self.exchange = exchange
        self.config = config
        
        # Инициализируем подконтроллеры
        self.market_controller = MarketController(exchange)
        self.order_controller = OrderController(exchange, config.get("order_config", {}))
        self.position_controller = PositionController(exchange, self.order_controller)
        self.risk_controller = RiskController(config.get("risk_config", {}))
        
        # Состояние торговли
        self.is_trading = False
        self.trading_mode = "manual"  # manual, automated, semi_automated

    async def start_trading(self, mode: str = "manual") -> bool:
        """Запуск торговли.
        
        Args:
            mode: Режим торговли
            
        Returns:
            True если торговля запущена успешно
        """
        try:
            self.trading_mode = mode
            self.is_trading = True
            logger.info(f"Trading started in {mode} mode")
            return True
        except Exception as e:
            logger.error(f"Error starting trading: {e}")
            return False

    async def stop_trading(self) -> bool:
        """Остановка торговли.
        
        Returns:
            True если торговля остановлена успешно
        """
        try:
            self.is_trading = False
            logger.info("Trading stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping trading: {e}")
            return False

    async def place_trade(
        self, 
        symbol: str, 
        side: str, 
        amount: float, 
        order_type: str = "market",
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Размещение торговой операции.
        
        Args:
            symbol: Торговая пара
            side: Сторона (buy/sell)
            amount: Количество
            order_type: Тип ордера
            price: Цена (для лимитных ордеров)
            
        Returns:
            Результат операции
        """
        try:
            # Получаем текущие данные
            account_balance = self.exchange.get_balance()
            positions = await self.position_controller.get_positions()
            
            # Валидируем ордер - исправляем тип account_balance
            balance_value = float(account_balance.get("total", 0.0)) if isinstance(account_balance, dict) else float(account_balance or 0.0)
            validation = await self.risk_controller.validate_order(
                symbol, side, amount, price or 0.0, positions, balance_value
            )
            
            if not validation["is_valid"]:
                return {
                    "success": False,
                    "errors": validation["errors"],
                    "warnings": validation["warnings"]
                }
                
            # Размещаем ордер
            order = await self.order_controller.create_order(
                symbol, side, order_type, amount, price
            )
            
            if order:
                return {
                    "success": True,
                    "order": order,
                    "warnings": validation["warnings"]
                }
            else:
                return {
                    "success": False,
                    "errors": ["Failed to create order"]
                }
        except Exception as e:
            logger.error(f"Error placing trade: {e}")
            return {
                "success": False,
                "errors": [f"Trade error: {str(e)}"]
            }

    async def close_position(self, symbol: str) -> Dict[str, Any]:
        """Закрытие позиции.
        
        Args:
            symbol: Торговая пара
            
        Returns:
            Результат операции
        """
        try:
            # Проверяем наличие позиции
            position = await self.position_controller.get_position(symbol)
            if not position:
                return {
                    "success": False,
                    "errors": [f"No position found for {symbol}"]
                }
                
            # Закрываем позицию
            order = await self.position_controller.close_position(symbol)
            
            if order:
                return {
                    "success": True,
                    "order": order
                }
            else:
                return {
                    "success": False,
                    "errors": ["Failed to close position"]
                }
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {
                "success": False,
                "errors": [f"Close position error: {str(e)}"]
            }

    async def get_trading_status(self) -> Dict[str, Any]:
        """Получение статуса торговли.
        
        Returns:
            Статус торговли
        """
        try:
            positions = await self.position_controller.get_positions()
            account_balance = self.exchange.get_balance()
            
            return {
                "is_trading": self.is_trading,
                "trading_mode": self.trading_mode,
                "open_positions": len(positions),
                "account_balance": account_balance,
                "total_pnl": sum(pos.get("unrealized_pnl", 0) for pos in positions),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting trading status: {e}")
            return {
                "is_trading": self.is_trading,
                "trading_mode": self.trading_mode,
                "error": str(e)
            }

    async def get_risk_report(self) -> Dict[str, Any]:
        """Получение отчета о рисках.
        
        Returns:
            Отчет о рисках
        """
        try:
            positions = await self.position_controller.get_positions()
            account_balance = self.exchange.get_balance()
            
            # Получаем риски портфеля
            portfolio_risk = await self.risk_controller.get_portfolio_risk(
                positions, 0.0
            )
            
            # Получаем предупреждения
            alerts = await self.risk_controller.get_risk_alerts(
                positions, 0.0
            )
            
            return {
                "portfolio_risk": portfolio_risk,
                "alerts": alerts,
                "positions_count": len(positions),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting risk report: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def execute_strategy(
        self, 
        strategy_name: str, 
        symbol: str, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Выполнение торговой стратегии.
        
        Args:
            strategy_name: Название стратегии
            symbol: Торговая пара
            parameters: Параметры стратегии
            
        Returns:
            Результат выполнения стратегии
        """
        try:
            if not self.is_trading:
                return {
                    "success": False,
                    "errors": ["Trading is not active"]
                }
                
            # Получаем рыночные данные
            market_data = await self.market_controller.get_market_data(symbol)
            
            # Здесь должна быть логика выполнения конкретной стратегии
            # Пока возвращаем заглушку
            return {
                "success": True,
                "strategy": strategy_name,
                "symbol": symbol,
                "action": "no_action",
                "reason": "Strategy not implemented",
                "market_data": market_data
            }
        except Exception as e:
            logger.error(f"Error executing strategy: {e}")
            return {
                "success": False,
                "errors": [f"Strategy error: {str(e)}"]
            }

    async def monitor_positions(self) -> List[Dict[str, Any]]:
        """Мониторинг позиций.
        
        Returns:
            Список действий для позиций
        """
        try:
            positions = await self.position_controller.get_positions()
            actions = []
            
            for position in positions:
                symbol = position["symbol"]
                
                # Получаем рыночные данные
                market_data = await self.market_controller.get_ticker(symbol)
                
                # Проверяем необходимость закрытия
                close_check = await self.risk_controller.should_close_position(
                    position, market_data
                )
                
                if close_check["should_close"]:
                    actions.append({
                        "action": "close_position",
                        "symbol": symbol,
                        "reason": close_check["reason"],
                        "position": position
                    })
                    
            return actions
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
            return [] 