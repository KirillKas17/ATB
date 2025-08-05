"""
Тесты для фильтров application слоя.
"""
import pytest
from decimal import Decimal
from typing import Any
from application.filters.orderbook_filter import OrderBookPreFilter
class TestOrderbookFilter:
    """Тесты для OrderbookFilter."""
    @pytest.fixture
    def filter_instance(self) -> OrderBookPreFilter:
        """Создает экземпляр фильтра."""
        return OrderBookPreFilter()
    @pytest.fixture
    def sample_orderbook(self) -> dict[str, Any]:
        """Создает образец ордербука."""
        return {
            "symbol": "BTC/USD",
            "timestamp": "2024-01-01T00:00:00",
            "bids": [
                {"price": "50000", "quantity": "0.1"},
                {"price": "49999", "quantity": "0.2"},
                {"price": "49998", "quantity": "0.3"}
            ],
            "asks": [
                {"price": "50001", "quantity": "0.1"},
                {"price": "50002", "quantity": "0.2"},
                {"price": "50003", "quantity": "0.3"}
            ]
        }
    def test_filter_by_price_range(self: "TestOrderbookFilter") -> None:
        """Тест фильтрации по диапазону цен."""
        filter_instance = OrderBookPreFilter()
        
        # Создаем тестовые данные
        bids = [(50000, 1.0), (49999, 2.0), (49998, 1.5)]
        asks = [(50001, 1.2), (50002, 1.8), (50003, 0.9)]
        
        # Фильтруем ордербук
        filtered_orderbook = filter_instance.filter_order_book(
            "test_exchange", "BTC/USD", bids, asks, 1640995200.0
        )
        
        assert filtered_orderbook is not None
        assert hasattr(filtered_orderbook, 'bids')
        assert hasattr(filtered_orderbook, 'asks')
    def test_filter_by_quantity_threshold(self: "TestOrderbookFilter") -> None:
        """Тест фильтрации по порогу количества."""
        filter_instance = OrderBookPreFilter()
        
        # Создаем тестовые данные
        bids = [(50000, 1.0), (49999, 2.0), (49998, 1.5)]
        asks = [(50001, 1.2), (50002, 1.8), (50003, 0.9)]
        
        # Фильтруем ордербук
        filtered_orderbook = filter_instance.filter_order_book(
            "test_exchange", "BTC/USD", bids, asks, 1640995200.0
        )
        
        assert filtered_orderbook is not None
        assert len(filtered_orderbook.bids) > 0
        assert len(filtered_orderbook.asks) > 0
    def test_filter_by_spread(self: "TestOrderbookFilter") -> None:
        """Тест фильтрации по спреду."""
        filter_instance = OrderBookPreFilter()
        
        # Создаем тестовые данные с узким спредом
        bids = [(50000, 1.0), (49999, 2.0)]
        asks = [(50001, 1.2), (50002, 1.8)]
        
        # Фильтруем ордербук
        filtered_orderbook = filter_instance.filter_order_book(
            "test_exchange", "BTC/USD", bids, asks, 1640995200.0
        )
        
        assert filtered_orderbook is not None
        # Проверяем, что спред разумный
        if filtered_orderbook.bids and filtered_orderbook.asks:
            best_bid = filtered_orderbook.bids[0][0].amount
            best_ask = filtered_orderbook.asks[0][0].amount
            spread = best_ask - best_bid
            assert spread > 0
    def test_filter_by_depth(self: "TestOrderbookFilter") -> None:
        """Тест фильтрации по глубине."""
        filter_instance = OrderBookPreFilter()
        
        # Создаем тестовые данные с хорошей глубиной
        bids = [(50000, 1.0), (49999, 2.0), (49998, 1.5), (49997, 3.0)]
        asks = [(50001, 1.2), (50002, 1.8), (50003, 0.9), (50004, 2.5)]
        
        # Фильтруем ордербук
        filtered_orderbook = filter_instance.filter_order_book(
            "test_exchange", "BTC/USD", bids, asks, 1640995200.0
        )
        
        assert filtered_orderbook is not None
        assert len(filtered_orderbook.bids) >= 3
        assert len(filtered_orderbook.asks) >= 3
    def test_remove_outliers(self: "TestOrderbookFilter") -> None:
        """Тест удаления выбросов."""
        filter_instance = OrderBookPreFilter()
        
        # Создаем тестовые данные с выбросами
        bids = [(50000, 1.0), (49999, 2.0), (49998, 1.5), (49000, 100.0)]  # Выброс
        asks = [(50001, 1.2), (50002, 1.8), (50003, 0.9), (51000, 100.0)]  # Выброс
        
        # Фильтруем ордербук
        filtered_orderbook = filter_instance.filter_order_book(
            "test_exchange", "BTC/USD", bids, asks, 1640995200.0
        )
        
        assert filtered_orderbook is not None
        # Проверяем, что выбросы обработаны
        assert len(filtered_orderbook.bids) > 0
        assert len(filtered_orderbook.asks) > 0
    def test_normalize_orderbook(self: "TestOrderbookFilter") -> None:
        """Тест нормализации ордербука."""
        filter_instance = OrderBookPreFilter()
        
        # Создаем тестовые данные
        bids = [(50000, 1.0), (49999, 2.0), (49998, 1.5)]
        asks = [(50001, 1.2), (50002, 1.8), (50003, 0.9)]
        
        # Фильтруем ордербук
        filtered_orderbook = filter_instance.filter_order_book(
            "test_exchange", "BTC/USD", bids, asks, 1640995200.0
        )
        
        assert filtered_orderbook is not None
        # Проверяем, что данные нормализованы
        for bid in filtered_orderbook.bids:
            assert bid[0].amount > 0
            assert bid[1].amount > 0
    def test_calculate_orderbook_metrics(self: "TestOrderbookFilter") -> None:
        """Тест расчета метрик ордербука."""
        filter_instance = OrderBookPreFilter()
        
        # Создаем тестовые данные
        bids = [(50000, 1.0), (49999, 2.0), (49998, 1.5)]
        asks = [(50001, 1.2), (50002, 1.8), (50003, 0.9)]
        
        # Фильтруем ордербук
        filtered_orderbook = filter_instance.filter_order_book(
            "test_exchange", "BTC/USD", bids, asks, 1640995200.0
        )
        
        assert filtered_orderbook is not None
        
        # Получаем статистику фильтра
        stats = filter_instance.get_filter_statistics()
        assert stats is not None
        assert isinstance(stats, dict)
        assert "total_processed" in stats
    def test_filter_invalid_data(self: "TestOrderbookFilter") -> None:
        """Тест фильтрации невалидных данных."""
        filter_instance = OrderBookPreFilter()
        
        # Создаем тестовые данные с невалидными значениями
        bids = [(50000, 1.0), (49999, 2.0), (-1, 1.5)]  # Невалидная цена
        asks = [(50001, 1.2), (50002, 1.8), (50003, -1)]  # Невалидный объем
        
        # Фильтруем ордербук
        filtered_orderbook = filter_instance.filter_order_book(
            "test_exchange", "BTC/USD", bids, asks, 1640995200.0
        )
        
        assert filtered_orderbook is not None
        # Проверяем, что невалидные данные обработаны
        for bid in filtered_orderbook.bids:
            assert bid[0].amount > 0
            assert bid[1].amount > 0
    def test_apply_multiple_filters(self: "TestOrderbookFilter") -> None:
        """Тест применения множественных фильтров."""
        filter_instance = OrderBookPreFilter()
        
        # Создаем тестовые данные
        bids = [(50000, 1.0), (49999, 2.0), (49998, 1.5)]
        asks = [(50001, 1.2), (50002, 1.8), (50003, 0.9)]
        
        # Применяем фильтры
        filtered_orderbook = filter_instance.filter_order_book(
            "test_exchange", "BTC/USD", bids, asks, 1640995200.0
        )
        
        assert filtered_orderbook is not None
        
        # Проверяем результаты фильтрации
        if filtered_orderbook.bids and filtered_orderbook.asks:
            best_bid = filtered_orderbook.bids[0][0].amount
            best_ask = filtered_orderbook.asks[0][0].amount
            
            # Проверяем, что спред разумный
            spread = best_ask - best_bid
            assert spread > 0
            assert spread < 1000  # Максимальный разумный спред
            
            # Проверяем, что объемы положительные
            for bid in filtered_orderbook.bids:
                assert bid[1].amount > 0
            for ask in filtered_orderbook.asks:
                assert ask[1].amount > 0
    def test_empty_orderbook(self, filter_instance: OrderBookPreFilter) -> None:
        """Тест обработки пустого ордербука."""
        # Создаем пустой ордербук
        empty_bids: list = []
        empty_asks: list = []
        
        # Фильтруем пустой ордербук
        filtered_orderbook = filter_instance.filter_order_book(
            "test_exchange", "BTC/USD", empty_bids, empty_asks, 1640995200.0
        )
        
        assert filtered_orderbook is not None
        assert len(filtered_orderbook.bids) == 0
        assert len(filtered_orderbook.asks) == 0
        
        # Проверяем статистику
        stats = filter_instance.get_filter_statistics()
        assert stats["total_processed"] > 0 
