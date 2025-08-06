"""
Протоколы для WebSocket клиентов и коннекторов бирж.
"""

from typing import Any, Callable, Dict, List, Optional, Protocol


class WebSocketClientProtocol(Protocol):
    """Протокол для WebSocket клиента биржи."""

    async def connect(self) -> bool:
        """Подключение к WebSocket."""
        ...

    async def disconnect(self) -> None:
        """Отключение от WebSocket."""
        ...

    async def subscribe_symbol(self, symbol: str) -> bool:
        """Подписка на символ."""
        ...

    async def unsubscribe_symbol(self, symbol: str) -> bool:
        """Отписка от символа."""
        ...

    def add_callback(self, callback: Callable[[Any], None]) -> None:
        """Добавление callback для обработки данных."""
        ...

    def remove_callback(self, callback: Callable[[Any], None]) -> None:
        """Удаление callback."""
        ...

    def is_connected(self) -> bool:
        """Проверка подключения."""
        ...

    def get_status(self) -> Dict[str, Any]:
        """Получение статуса клиента."""
        ...


class MarketDataConnectorProtocol(Protocol):
    """Протокол для коннектора рыночных данных."""

    async def get_ohlcv_data(
        self, symbol: str, interval: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Получение OHLCV данных."""
        ...

    async def get_order_book(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """Получение данных ордербука."""
        ...

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Получение тикера."""
        ...

    async def get_recent_trades(
        self, symbol: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Получение последних сделок."""
        ...

    def is_connected(self) -> bool:
        """Проверка подключения."""
        ...

    def get_status(self) -> Dict[str, Any]:
        """Получение статуса коннектора."""
        ...


class MarketStreamAggregatorProtocol(Protocol):
    """Протокол для агрегатора потоков данных."""

    def add_source(self, name: str, client: WebSocketClientProtocol) -> bool:
        """Добавление источника данных."""
        ...

    def remove_source(self, name: str) -> bool:
        """Удаление источника данных."""
        ...

    async def subscribe_symbol(self, symbol: str) -> bool:
        """Подписка на символ во всех источниках."""
        ...

    async def unsubscribe_symbol(self, symbol: str) -> bool:
        """Отписка от символа во всех источниках."""
        ...

    def add_callback(self, callback: Callable[[Any], None]) -> None:
        """Добавление callback для обработки данных."""
        ...

    def remove_callback(self, callback: Callable[[Any], None]) -> None:
        """Удаление callback."""
        ...

    async def start(self) -> None:
        """Запуск агрегатора."""
        ...

    async def stop(self) -> None:
        """Остановка агрегатора."""
        ...

    def get_synchronized_updates(self) -> List[Any]:
        """Получение синхронизированных обновлений."""
        ...

    def get_status(self) -> Dict[str, Any]:
        """Получение статуса агрегатора."""
        ...

    def get_aggregator_stats(self) -> Dict[str, Any]:
        """Получение статистики агрегатора."""
        ...

    def get_source_status(self) -> Dict[str, Any]:
        """Получение статуса источников."""
        ...

    def reset_stats(self) -> None:
        """Сброс статистики."""
        ...

    def update_buffer(self, symbol: str, data: Any) -> None:
        """Обновление буфера для символа."""
        ...

    def clear_buffer(self, symbol: str) -> None:
        """Очистка буфера для символа."""
        ...


class SymbolMetricsProviderProtocol(Protocol):
    """Протокол для провайдера метрик символов."""

    async def get_symbol_metrics(self, symbol: str) -> Dict[str, Any]:
        """Получение метрик для символа."""
        ...

    async def get_multiple_symbol_metrics(
        self, symbols: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Получение метрик для нескольких символов."""
        ...

    async def get_market_data(
        self, symbol: str, interval: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Получение рыночных данных для символа."""
        ...

    async def get_order_book_data(self, symbol: str, depth: int) -> Dict[str, Any]:
        """Получение данных ордербука для символа."""
        ...

    def is_available(self, symbol: str) -> bool:
        """Проверка доступности символа."""
        ...

    def get_status(self) -> Dict[str, Any]:
        """Получение статуса провайдера."""
        ...
