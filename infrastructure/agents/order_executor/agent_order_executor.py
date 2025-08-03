"""
Агент исполнения ордеров.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from domain.types.agent_types import AgentConfig, AgentStatus, ProcessingResult, AgentType
from infrastructure.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# Типы для ордеров
OrderType = str
OrderStatus = str
OrderRequestDict = Dict[str, Any]
ExecutionResultDict = Dict[str, Any]


class OrderExecutorAgent(BaseAgent):
    """Агент исполнения ордеров."""

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        if config is None:
            config = {
                "name": "OrderExecutorAgent",
                "agent_type": "order_executor",  # [1] правильный AgentType
                "max_position_size": 1000.0,
                "max_portfolio_risk": 0.02,
                "max_risk_per_trade": 0.01,
                "confidence_threshold": 0.8,  # [1] обязательное поле
                "risk_threshold": 0.3,  # [1] обязательное поле
                "performance_threshold": 0.7,  # [1] обязательное поле
                "rebalance_interval": 60,  # [1] обязательное поле
                "processing_timeout_ms": 30000,  # [1] обязательное поле
                "retry_attempts": 3,  # [1] обязательное поле
                "enable_evolution": False,  # [1] обязательное поле
                "enable_learning": True,  # [1] обязательное поле
                "metadata": {  # [1] обязательное поле
                    "max_retries": 3,
                    "retry_delay": 1.0,
                    "timeout": 30.0,
                    "enable_slippage_protection": True,
                    "max_slippage": 0.001,
                    "enable_partial_fills": True,
                    "min_order_size": 0.0,
                    "max_order_size": float("inf"),
                    "enable_logging": True,
                }
            }
        super().__init__("OrderExecutorAgent", "order_executor", config.__dict__ if config else {})  # Исправление: преобразуем AgentConfig в dict

        self.active_orders: Dict[str, Dict[str, Any]] = {}
        self.order_history: Dict[str, Dict[str, Any]] = {}
        self.executions: Dict[str, List[Dict[str, Any]]] = {}

        self.stats = {
            "total_orders": 0,
            "successful_orders": 0,
            "failed_orders": 0,
            "total_volume": 0.0,
            "total_fees": 0.0,
            "average_execution_time": 0.0,
        }

        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False

    async def initialize(self) -> bool:
        """Инициализация агента."""
        try:
            if not self.validate_config():
                return False

            self._update_state(AgentStatus.HEALTHY, is_healthy=True)  # [2] правильный метод
            self.update_confidence(0.8)

            logger.info("OrderExecutorAgent initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize OrderExecutorAgent: {e}")
            self.record_error(f"Initialization failed: {e}")
            return False

    async def process(self, data: Any) -> ProcessingResult:
        """Обработка данных."""
        start_time = datetime.now()

        try:
            if isinstance(data, dict):
                order_params = data.get("order_params")
                if not order_params:
                    raise ValueError("Order parameters are required")

                result = await self.execute_order(order_params)

                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                self.record_success(processing_time)

                return ProcessingResult(
                    success=True,
                    data={
                        "execution_result": result,
                        "order_id": result.get("order_id"),
                        "success": result.get("success"),
                        "timestamp": datetime.now().isoformat(),
                    },
                    confidence=0.8,  # [3] добавляю confidence
                    risk_score=0.2,  # [3] добавляю risk_score
                    processing_time_ms=int(processing_time),
                    errors=[]
                )
            else:
                raise ValueError("Invalid data format")

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.record_error(f"Processing failed: {e}", processing_time)
            return ProcessingResult(
                success=False,
                data={},
                confidence=0.0,  # [3] добавляю confidence
                risk_score=1.0,  # [3] добавляю risk_score
                processing_time_ms=int(processing_time),
                errors=[str(e)]
            )

    async def cleanup(self) -> None:
        """Очистка ресурсов."""
        try:
            await self.cancel_all_orders()
            self.active_orders.clear()
            self.order_history.clear()
            self.executions.clear()
            logger.info("OrderExecutorAgent cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def validate_config(self) -> bool:
        """Валидация конфигурации."""
        try:
            # [4] правильный доступ к TypedDict через metadata
            metadata = self.config.get("metadata", {})
            required_keys = [
                "max_retries",
                "retry_delay",
                "timeout",
                "max_slippage",
            ]

            for key in required_keys:
                if key not in metadata:
                    logger.error(f"Missing required config key: {key}")
                    return False

                value = metadata[key]
                if not isinstance(value, (int, float)) or value < 0:
                    logger.error(f"Invalid config value for {key}: {value}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Config validation error: {e}")
            return False

    async def execute_order(self, order_params: OrderRequestDict) -> ExecutionResultDict:
        """Исполнение ордера."""
        try:
            if not self._validate_order_params(order_params):
                return ExecutionResultDict(
                    success=False, error_message="Invalid order parameters"
                )

            order = self._create_order(order_params)

            if not self._check_order_limits(order):
                return ExecutionResultDict(
                    success=False, error_message="Order exceeds limits"
                )

            order_type = order.get("order_type")
            if order_type == "market":
                result = await self._execute_market_order(order)
            elif order_type == "limit":
                result = await self._execute_limit_order(order)
            elif order_type == "stop":
                result = await self._execute_stop_order(order)
            else:
                result = await self._execute_generic_order(order)

            self._update_statistics(result)
            return result

        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return ExecutionResultDict(success=False, error_message=str(e))

    async def cancel_order(self, order_id: str) -> bool:
        """Отмена ордера."""
        try:
            if order_id not in self.active_orders:
                logger.warning(f"Order {order_id} not found")
                return False

            order = self.active_orders[order_id]
            order["status"] = "cancelled"
            order["updated_at"] = datetime.now()

            self.order_history[order_id] = order
            del self.active_orders[order_id]

            logger.info(f"Order {order_id} canceled")
            return True

        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return False

    async def cancel_all_orders(self) -> bool:
        """Отмена всех ордеров."""
        try:
            order_ids = list(self.active_orders.keys())
            results = await asyncio.gather(
                *[self.cancel_order(order_id) for order_id in order_ids],
                return_exceptions=True,
            )

            success_count = sum(1 for result in results if result is True)
            logger.info(f"Canceled {success_count}/{len(order_ids)} orders")

            return success_count == len(order_ids)

        except Exception as e:
            logger.error(f"Error canceling all orders: {e}")
            return False

    def _validate_order_params(self, order_params: OrderRequestDict) -> bool:
        """Валидация параметров."""
        try:
            required_fields = ["symbol", "side", "quantity"]
            for field in required_fields:
                if field not in order_params:
                    logger.error(f"Missing required field: {field}")
                    return False

            # Проверка типов
            if not isinstance(order_params.get("symbol"), str):
                logger.error("Symbol must be a string")
                return False

            if not isinstance(order_params.get("quantity"), (int, float)):
                logger.error("Quantity must be a number")
                return False

            if order_params.get("quantity", 0) <= 0:
                logger.error("Quantity must be positive")
                return False

            return True

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

    def _create_order(self, order_params: OrderRequestDict) -> Dict[str, Any]:
        """Создание объекта ордера."""
        try:
            order = {
                "id": f"order_{datetime.now().timestamp()}",
                "symbol": order_params.get("symbol"),
                "side": order_params.get("side"),
                "quantity": order_params.get("quantity"),
                "order_type": order_params.get("order_type", "market"),
                "price": order_params.get("price"),
                "stop_price": order_params.get("stop_price"),
                "status": "pending",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "metadata": order_params.get("metadata", {}),
            }

            # Добавляем в активные ордера
            self.active_orders[order["id"]] = order
            return order

        except Exception as e:
            logger.error(f"Error creating order: {e}")
            raise

    def _check_order_limits(self, order: Dict[str, Any]) -> bool:
        """Проверка лимитов ордера."""
        try:
            quantity = order.get("quantity", 0)
            min_size = self.config.get("min_order_size", 0.0)
            max_size = self.config.get("max_order_size", float("inf"))

            if quantity < min_size:
                logger.error(f"Order quantity {quantity} below minimum {min_size}")
                return False

            if quantity > max_size:
                logger.error(f"Order quantity {quantity} above maximum {max_size}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking order limits: {e}")
            return False

    async def _execute_market_order(self, order: Dict[str, Any]) -> ExecutionResultDict:
        """Исполнение рыночного ордера."""
        try:
            logger.info(f"Executing market order: {order['id']}")

            # Симуляция исполнения
            await asyncio.sleep(0.1)

            # Получаем текущую цену
            current_price = await self._get_current_price(order["symbol"])

            # Обновляем статус
            order["status"] = "filled"
            order["executed_price"] = current_price
            order["executed_quantity"] = order["quantity"]
            order["updated_at"] = datetime.now()

            # Перемещаем в историю
            self.order_history[order["id"]] = order
            del self.active_orders[order["id"]]

            result = {
                "success": True,
                "order_id": order["id"],
                "executed_price": current_price,
                "executed_quantity": order["quantity"],
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(f"Market order executed: {order['id']}")
            return result

        except Exception as e:
            logger.error(f"Error executing market order: {e}")
            return {"success": False, "error_message": str(e)}

    async def _execute_limit_order(self, order: Dict[str, Any]) -> ExecutionResultDict:
        """Исполнение лимитного ордера."""
        try:
            logger.info(f"Placing limit order: {order['id']}")

            # Симуляция размещения
            await asyncio.sleep(0.1)

            # Обновляем статус
            order["status"] = "pending"
            order["updated_at"] = datetime.now()

            result = {
                "success": True,
                "order_id": order["id"],
                "status": "pending",
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(f"Limit order placed: {order['id']}")
            return result

        except Exception as e:
            logger.error(f"Error executing limit order: {e}")
            return {"success": False, "error_message": str(e)}

    async def _execute_stop_order(self, order: Dict[str, Any]) -> ExecutionResultDict:
        """Исполнение стоп-ордера."""
        try:
            logger.info(f"Placing stop order: {order['id']}")

            # Симуляция размещения
            await asyncio.sleep(0.1)

            # Обновляем статус
            order["status"] = "pending"
            order["updated_at"] = datetime.now()

            result = {
                "success": True,
                "order_id": order["id"],
                "status": "pending",
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(f"Stop order placed: {order['id']}")
            return result

        except Exception as e:
            logger.error(f"Error executing stop order: {e}")
            return {"success": False, "error_message": str(e)}

    async def _execute_generic_order(self, order: Dict[str, Any]) -> ExecutionResultDict:
        """Исполнение общего ордера."""
        try:
            logger.info(f"Executing generic order: {order['id']}")

            # Симуляция исполнения
            await asyncio.sleep(0.1)

            result = {
                "success": True,
                "order_id": order["id"],
                "status": "executed",
                "timestamp": datetime.now().isoformat(),
            }

            return result

        except Exception as e:
            logger.error(f"Error executing generic order: {e}")
            return {"success": False, "error_message": str(e)}

    async def _get_current_price(self, symbol: str) -> float:
        """Получение текущей цены."""
        # Симуляция получения цены
        return 100.0

    def _update_statistics(self, result: ExecutionResultDict) -> None:
        """Обновление статистики."""
        try:
            self.stats["total_orders"] += 1

            if result.get("success"):
                self.stats["successful_orders"] += 1
                if "executed_quantity" in result:
                    self.stats["total_volume"] += result["executed_quantity"]
            else:
                self.stats["failed_orders"] += 1

        except Exception as e:
            logger.error(f"Error updating statistics: {e}")

    def get_executor_summary(self) -> Dict[str, Any]:
        """Получение сводки исполнителя."""
        try:
            return {
                "active_orders": len(self.active_orders),
                "total_orders": self.stats["total_orders"],
                "successful_orders": self.stats["successful_orders"],
                "failed_orders": self.stats["failed_orders"],
                "success_rate": (
                    self.stats["successful_orders"] / self.stats["total_orders"]
                    if self.stats["total_orders"] > 0
                    else 0.0
                ),
                "total_volume": self.stats["total_volume"],
                "total_fees": self.stats["total_fees"],
                "average_execution_time": self.stats["average_execution_time"],
            }

        except Exception as e:
            logger.error(f"Error getting executor summary: {e}")
            return {}
