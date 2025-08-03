"""
Сервисы для исполнения ордеров.

Включает:
- Управление исполнением ордеров
- Отслеживание статусов
- Обработку событий
- Логирование операций
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from loguru import logger

from domain.value_objects.timestamp import Timestamp

from .brokers import (
    BrokerProtocol,
    OrderRequest,
    OrderResponse,
    OrderStatus,
    OrderUpdate,
    Trade,
)


@dataclass
class ExecutionConfig:
    """Конфигурация исполнения."""

    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    enable_logging: bool = True
    enable_metrics: bool = True


@dataclass
class ExecutionResult:
    """Результат исполнения ордера."""

    success: bool
    order_id: Optional[str] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderExecutionState:
    """Состояние исполнения ордера."""

    order_id: str
    request: OrderRequest
    response: Optional[OrderResponse]
    status: OrderStatus
    created_at: Timestamp
    updated_at: Timestamp
    retry_count: int = 0
    trades: List[Trade] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ExecutionCallback(Protocol):
    """Протокол для callback'ов исполнения."""

    async def on_order_placed(self, order_id: str, response: OrderResponse) -> None:
        """Вызывается при размещении ордера."""
        ...

    async def on_order_updated(self, order_id: str, update: OrderUpdate) -> None:
        """Вызывается при обновлении ордера."""
        ...

    async def on_order_filled(self, order_id: str, trades: List[Trade]) -> None:
        """Вызывается при исполнении ордера."""
        ...

    async def on_order_cancelled(self, order_id: str) -> None:
        """Вызывается при отмене ордера."""
        ...

    async def on_order_failed(self, order_id: str, error: str) -> None:
        """Вызывается при ошибке ордера."""
        ...


class OrderExecutor:
    """Исполнитель ордеров."""

    def __init__(
        self, broker: BrokerProtocol, config: Optional[ExecutionConfig] = None
    ):
        self.broker = broker
        self.config = config or ExecutionConfig()
        self.active_orders: Dict[str, OrderExecutionState] = {}
        self.callbacks: List[ExecutionCallback] = []
        self._execution_lock = asyncio.Lock()

    def add_callback(self, callback: ExecutionCallback) -> None:
        """Добавить callback для событий исполнения."""
        self.callbacks.append(callback)

    def remove_callback(self, callback: ExecutionCallback) -> None:
        """Удалить callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    async def execute_order(self, request: OrderRequest) -> ExecutionResult:
        """Исполнить ордер."""
        start_time = asyncio.get_event_loop().time()

        try:
            async with self._execution_lock:
                # Размещение ордера
                response = await self._place_order_with_retry(request)

                # Создание состояния исполнения
                execution_state = OrderExecutionState(
                    order_id=response.order_id,
                    request=request,
                    response=response,
                    status=response.status,
                    created_at=response.timestamp,
                    updated_at=response.timestamp,
                )

                self.active_orders[response.order_id] = execution_state

                # Уведомление callback'ов
                await self._notify_order_placed(response.order_id, response)

                # Ожидание исполнения
                result = await self._wait_for_execution(response.order_id)

                execution_time = asyncio.get_event_loop().time() - start_time
                result.execution_time = execution_time

                return result

        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            execution_time = asyncio.get_event_loop().time() - start_time

            return ExecutionResult(
                success=False, error_message=str(e), execution_time=execution_time
            )

    async def _place_order_with_retry(self, request: OrderRequest) -> OrderResponse:
        """Разместить ордер с повторными попытками."""
        last_error: Optional[Exception] = None

        for attempt in range(self.config.max_retries):
            try:
                return await self.broker.place_order(request)
            except Exception as e:
                last_error = e
                logger.warning(f"Order placement attempt {attempt + 1} failed: {e}")

                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2**attempt))

        if last_error is not None:
            raise last_error
        else:
            raise RuntimeError("Order placement failed after all retries")

    async def _wait_for_execution(self, order_id: str) -> ExecutionResult:
        """Ожидать исполнения ордера."""
        timeout = self.config.timeout
        start_time = asyncio.get_event_loop().time()

        while True:
            try:
                # Проверка статуса
                response = await self.broker.get_order_status(
                    order_id, self.active_orders[order_id].request.symbol
                )

                # Обновление состояния
                self.active_orders[order_id].status = response.status
                self.active_orders[order_id].updated_at = response.timestamp

                # Проверка завершения
                if response.status in [
                    OrderStatus.FILLED,
                    OrderStatus.CANCELLED,
                    OrderStatus.REJECTED,
                    OrderStatus.EXPIRED,
                ]:
                    # Получение сделок
                    trades = await self.broker.get_trades(order_id)
                    self.active_orders[order_id].trades = trades

                    # Уведомление callback'ов
                    if response.status == OrderStatus.FILLED:
                        await self._notify_order_filled(order_id, trades)
                    elif response.status == OrderStatus.CANCELLED:
                        await self._notify_order_cancelled(order_id)
                    else:
                        await self._notify_order_failed(
                            order_id, f"Order {response.status.value}"
                        )

                    # Удаление из активных ордеров
                    if order_id in self.active_orders:
                        del self.active_orders[order_id]

                    return ExecutionResult(
                        success=response.status == OrderStatus.FILLED,
                        order_id=order_id,
                        trades=trades,
                        metadata={"final_status": response.status.value},
                    )

                # Проверка таймаута
                if asyncio.get_event_loop().time() - start_time > timeout:
                    await self._notify_order_failed(order_id, "Execution timeout")
                    return ExecutionResult(
                        success=False,
                        order_id=order_id,
                        error_message="Execution timeout",
                    )

                # Ожидание перед следующей проверкой
                await asyncio.sleep(1.0)

            except Exception as e:
                logger.error(f"Error checking order status: {e}")
                await asyncio.sleep(1.0)

    async def cancel_order(self, order_id: str) -> bool:
        """Отменить ордер."""
        if order_id not in self.active_orders:
            return False

        try:
            state = self.active_orders[order_id]
            success = await self.broker.cancel_order(order_id, state.request.symbol)

            if success:
                await self._notify_order_cancelled(order_id)
                if order_id in self.active_orders:
                    del self.active_orders[order_id]

            return success

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    async def get_active_orders(self) -> List[OrderExecutionState]:
        """Получить активные ордера."""
        return list(self.active_orders.values())

    async def _notify_order_placed(
        self, order_id: str, response: OrderResponse
    ) -> None:
        """Уведомить о размещении ордера."""
        for callback in self.callbacks:
            try:
                await callback.on_order_placed(order_id, response)
            except Exception as e:
                logger.error(f"Error in order placed callback: {e}")

    async def _notify_order_updated(self, order_id: str, update: OrderUpdate) -> None:
        """Уведомить об обновлении ордера."""
        for callback in self.callbacks:
            try:
                await callback.on_order_updated(order_id, update)
            except Exception as e:
                logger.error(f"Error in order updated callback: {e}")

    async def _notify_order_filled(self, order_id: str, trades: List[Trade]) -> None:
        """Уведомить об исполнении ордера."""
        for callback in self.callbacks:
            try:
                await callback.on_order_filled(order_id, trades)
            except Exception as e:
                logger.error(f"Error in order filled callback: {e}")

    async def _notify_order_cancelled(self, order_id: str) -> None:
        """Уведомить об отмене ордера."""
        for callback in self.callbacks:
            try:
                await callback.on_order_cancelled(order_id)
            except Exception as e:
                logger.error(f"Error in order cancelled callback: {e}")

    async def _notify_order_failed(self, order_id: str, error: str) -> None:
        """Уведомить об ошибке ордера."""
        for callback in self.callbacks:
            try:
                await callback.on_order_failed(order_id, error)
            except Exception as e:
                logger.error(f"Error in order failed callback: {e}")


class ExecutionManager:
    """Менеджер исполнения ордеров."""

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.executors: Dict[str, OrderExecutor] = {}
        self.execution_history: List[ExecutionResult] = []

    def add_executor(self, name: str, executor: OrderExecutor) -> None:
        """Добавить исполнителя."""
        self.executors[name] = executor

    def get_executor(self, name: str) -> Optional[OrderExecutor]:
        """Получить исполнителя."""
        return self.executors.get(name)

    async def execute_order(
        self, executor_name: str, request: OrderRequest
    ) -> ExecutionResult:
        """Исполнить ордер через указанного исполнителя."""
        executor = self.get_executor(executor_name)
        if not executor:
            return ExecutionResult(
                success=False, error_message=f"Executor not found: {executor_name}"
            )

        result = await executor.execute_order(request)
        self.execution_history.append(result)

        return result

    async def cancel_all_orders(self, executor_name: str) -> List[bool]:
        """Отменить все ордера у исполнителя."""
        executor = self.get_executor(executor_name)
        if not executor:
            return []

        active_orders = await executor.get_active_orders()
        results = []

        for order_state in active_orders:
            result = await executor.cancel_order(order_state.order_id)
            results.append(result)

        return results

    def get_execution_stats(self) -> Dict[str, Any]:
        """Получить статистику исполнения."""
        total_orders = len(self.execution_history)
        successful_orders = sum(
            1 for result in self.execution_history if result.success
        )
        failed_orders = total_orders - successful_orders

        avg_execution_time = 0.0
        if total_orders > 0:
            avg_execution_time = (
                sum(result.execution_time for result in self.execution_history)
                / total_orders
            )

        return {
            "total_orders": total_orders,
            "successful_orders": successful_orders,
            "failed_orders": failed_orders,
            "success_rate": (
                successful_orders / total_orders if total_orders > 0 else 0.0
            ),
            "average_execution_time": avg_execution_time,
            "active_executors": len(self.executors),
        }


class OrderTracker:
    """Отслеживатель ордеров."""

    def __init__(self) -> None:
        self.tracked_orders: Dict[str, Dict[str, Any]] = {}

    def track_order(self, order_id: str, metadata: Dict[str, Any]) -> None:
        """Начать отслеживание ордера."""
        self.tracked_orders[order_id] = {
            "metadata": metadata,
            "created_at": Timestamp.now(),
            "status_updates": [],
            "trades": [],
        }

    def update_order_status(
        self, order_id: str, status: OrderStatus, metadata: Dict[str, Any]
    ) -> None:
        """Обновить статус отслеживаемого ордера."""
        if order_id in self.tracked_orders:
            self.tracked_orders[order_id]["status_updates"].append(
                {"status": status, "timestamp": Timestamp.now(), "metadata": metadata}
            )

    def add_trade(self, order_id: str, trade: Trade) -> None:
        """Добавить сделку к отслеживаемому ордеру."""
        if order_id in self.tracked_orders:
            self.tracked_orders[order_id]["trades"].append(trade)

    def get_order_info(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Получить информацию об отслеживаемом ордере."""
        return self.tracked_orders.get(order_id)

    def get_all_tracked_orders(self) -> Dict[str, Dict[str, Any]]:
        """Получить все отслеживаемые ордера."""
        return self.tracked_orders.copy()

    def stop_tracking(self, order_id: str) -> None:
        """Прекратить отслеживание ордера."""
        if order_id in self.tracked_orders:
            del self.tracked_orders[order_id]


# Фабричные функции
def create_execution_config(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    timeout: float = 30.0,
    enable_logging: bool = True,
    enable_metrics: bool = True,
) -> ExecutionConfig:
    """Создать конфигурацию исполнения."""
    return ExecutionConfig(
        max_retries=max_retries,
        retry_delay=retry_delay,
        timeout=timeout,
        enable_logging=enable_logging,
        enable_metrics=enable_metrics,
    )


def create_order_executor(
    broker: BrokerProtocol, config: Optional[ExecutionConfig] = None
) -> OrderExecutor:
    """Создать исполнителя ордеров."""
    return OrderExecutor(broker, config)


def create_execution_manager(
    config: Optional[ExecutionConfig] = None,
) -> ExecutionManager:
    """Создать менеджер исполнения."""
    return ExecutionManager(config)


def create_order_tracker() -> OrderTracker:
    """Создать отслеживатель ордеров."""
    return OrderTracker()
