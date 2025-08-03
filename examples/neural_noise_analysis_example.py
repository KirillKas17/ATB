# -*- coding: utf-8 -*-
"""
Пример анализа нейронного шума в ордербуке.
Демонстрирует использование NoiseAnalyzer и OrderBookPreFilter.
"""

import random
import time
from decimal import Decimal
from typing import List

from loguru import logger

from domain.value_objects.currency import Currency
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.timestamp import Timestamp
from domain.entities.orderbook import OrderBookSnapshot
from application.filters.orderbook_filter import (
    FilterConfig,
    OrderBookPreFilter,
)
from infrastructure.agents.analytical.noise_analyzer import (
    NoiseAnalysisResult,
    NoiseAnalyzer,
)


def create_synthetic_order_book() -> OrderBookSnapshot:
    """Создание синтетического ордербука с искусственным шумом."""
    # Базовые параметры
    base_price = 50000.0
    spread = 10.0

    # Создаем bids с синтетическим шумом
    bids = []
    for i in range(10):
        # Синтетический шум - регулярные паттерны
        noise = (i % 3) * 2 - 2  # Циклический шум: -2, 0, 1, -2, 0, 1...
        price = base_price - i * 10 + noise
        volume = random.uniform(0.1, 2.0) + (i % 2) * 0.5  # Регулярные объемы
        bids.append((Price(Decimal(str(price)), Currency.USDT), Volume(Decimal(str(volume)))))

    return OrderBookSnapshot(
        exchange="synthetic",
        symbol="BTCUSDT",
        bids=bids,
        asks=[],
        timestamp=Timestamp.from_unix(int(time.time())),
    )


def create_natural_order_book() -> OrderBookSnapshot:
    """Создание естественного ордербука без синтетического шума."""
    # Базовые параметры
    base_price = 50000.0
    spread = 10.0

    # Создаем bids с естественным шумом
    bids = []
    for i in range(10):
        # Естественный случайный шум
        noise = random.gauss(0, 2)  # Нормальное распределение
        price = base_price - i * 10 + noise
        volume = random.uniform(0.1, 2.0)  # Случайные объемы
        bids.append((Price(Decimal(str(price)), Currency.USDT), Volume(Decimal(str(volume)))))

    # Создаем asks с естественным шумом
    asks = []
    for i in range(10):
        # Естественный случайный шум
        noise = random.gauss(0, 2)  # Нормальное распределение
        price = base_price + spread + i * 10 + noise
        volume = random.uniform(0.1, 2.0)  # Случайные объемы
        asks.append((Price(Decimal(str(price)), Currency.USDT), Volume(Decimal(str(volume)))))

    return OrderBookSnapshot(
        exchange="natural",
        symbol="BTCUSDT",
        bids=bids,
        asks=asks,
        timestamp=Timestamp.from_unix(int(time.time())),
    )


def test_noise_analyzer():
    """Тестирование NoiseAnalyzer."""
    logger.info("=== Testing NoiseAnalyzer ===")

    # Создаем анализатор
    analyzer = NoiseAnalyzer(
        fractal_dimension_lower=1.2,
        fractal_dimension_upper=1.4,
        entropy_threshold=0.7,
        min_data_points=20,
        window_size=50,
    )

    # Тестируем с синтетическим ордербуком
    logger.info("Testing with synthetic order book...")
    synthetic_ob = create_synthetic_order_book()

    # Анализируем несколько раз для накопления истории
    for i in range(30):
        result = analyzer.analyze_noise(synthetic_ob)
        logger.debug(
            f"Iteration {i+1}: FD={result.fractal_dimension:.3f}, Entropy={result.entropy:.3f}"
        )

    # Финальный анализ
    final_result = analyzer.analyze_noise(synthetic_ob)
    logger.info(f"Synthetic order book analysis:")
    logger.info(f"  Fractal Dimension: {final_result.fractal_dimension:.3f}")
    logger.info(f"  Entropy: {final_result.entropy:.3f}")
    logger.info(f"  Is Synthetic Noise: {final_result.is_synthetic_noise}")
    logger.info(f"  Confidence: {final_result.confidence:.3f}")

    # Тестируем с естественным ордербуком
    logger.info("\nTesting with natural order book...")
    natural_ob = create_natural_order_book()

    # Сбрасываем историю
    analyzer.reset_history()

    # Анализируем несколько раз
    for i in range(30):
        result = analyzer.analyze_noise(natural_ob)
        logger.debug(
            f"Iteration {i+1}: FD={result.fractal_dimension:.3f}, Entropy={result.entropy:.3f}"
        )

    # Финальный анализ
    final_result = analyzer.analyze_noise(natural_ob)
    logger.info(f"Natural order book analysis:")
    logger.info(f"  Fractal Dimension: {final_result.fractal_dimension:.3f}")
    logger.info(f"  Entropy: {final_result.entropy:.3f}")
    logger.info(f"  Is Synthetic Noise: {final_result.is_synthetic_noise}")
    logger.info(f"  Confidence: {final_result.confidence:.3f}")

    # Статистика анализатора
    stats = analyzer.get_analysis_statistics()
    logger.info(f"\nAnalyzer statistics: {stats}")


def test_order_book_filter():
    """Тестирование OrderBookPreFilter."""
    logger.info("\n=== Testing OrderBookPreFilter ===")

    # Создаем конфигурацию фильтра
    config = FilterConfig(
        enabled=True,
        fractal_dimension_lower=1.2,
        fractal_dimension_upper=1.4,
        entropy_threshold=0.7,
        min_data_points=20,
        window_size=50,
        log_filtered=True,
        log_analysis=True,
    )

    # Создаем фильтр
    filter_obj = OrderBookPreFilter(config)

    # Тестируем с синтетическими данными
    logger.info("Testing with synthetic data...")
    synthetic_bids = [(Price(Decimal(str(50000 - j * 10)), Currency.USDT), Volume(Decimal(str(1.0 + j * 0.1)))) for j in range(10)]
    synthetic_asks = [(Price(Decimal(str(50010 + j * 10)), Currency.USDT), Volume(Decimal(str(1.0 + j * 0.1)))) for j in range(10)]

    # Фильтруем несколько раз для накопления истории
    for i in range(30):
        filtered_ob = filter_obj.filter_order_book(
            exchange="synthetic",
            symbol="BTCUSDT",
            bids=synthetic_bids,
            asks=synthetic_asks,
            timestamp=Timestamp.from_unix(int(time.time())),
            sequence_id=i,
        )

        if filtered_ob.meta.get("synthetic_noise"):
            logger.info(f"Synthetic noise detected at iteration {i+1}")

    # Финальная фильтрация
    final_filtered_ob = filter_obj.filter_order_book(
        exchange="synthetic",
        symbol="BTCUSDT",
        bids=synthetic_bids,
        asks=synthetic_asks,
        timestamp=Timestamp.from_unix(int(time.time())),
        sequence_id=999,
    )

    logger.info(f"Final synthetic order book filtering:")
    logger.info(f"  Filtered: {final_filtered_ob.meta.get('filtered', False)}")
    logger.info(
        f"  Synthetic Noise: {final_filtered_ob.meta.get('synthetic_noise', False)}"
    )
    logger.info(
        f"  Confidence: {final_filtered_ob.meta.get('filter_confidence', 0):.3f}"
    )

    # Тестируем с естественными данными
    logger.info("\nTesting with natural data...")
    natural_bids = [
        (Price(Decimal(str(50000 - i * 10 + random.gauss(0, 2))), Currency.USDT), Volume(Decimal(str(random.uniform(0.1, 2.0)))))
        for i in range(10)
    ]
    natural_asks = [
        (Price(Decimal(str(50010 + i * 10 + random.gauss(0, 2))), Currency.USDT), Volume(Decimal(str(random.uniform(0.1, 2.0)))))
        for i in range(10)
    ]

    # Сбрасываем статистику
    filter_obj.reset_statistics()

    # Фильтруем несколько раз
    for i in range(30):
        filtered_ob = filter_obj.filter_order_book(
            exchange="natural",
            symbol="BTCUSDT",
            bids=natural_bids,
            asks=natural_asks,
            timestamp=Timestamp.from_unix(int(time.time())),
            sequence_id=i,
        )

    # Финальная фильтрация
    final_filtered_ob = filter_obj.filter_order_book(
        exchange="natural",
        symbol="BTCUSDT",
        bids=natural_bids,
        asks=natural_asks,
        timestamp=Timestamp.from_unix(int(time.time())),
        sequence_id=999,
    )

    logger.info(f"Final natural order book filtering:")
    logger.info(f"  Filtered: {final_filtered_ob.meta.get('filtered', False)}")
    logger.info(
        f"  Synthetic Noise: {final_filtered_ob.meta.get('synthetic_noise', False)}"
    )
    logger.info(
        f"  Confidence: {final_filtered_ob.meta.get('filter_confidence', 0):.3f}"
    )

    # Статистика фильтра
    stats = filter_obj.get_statistics()
    logger.info(f"\nFilter statistics: {stats}")


async def test_async_filtering():
    """Тестирование асинхронной фильтрации."""
    logger.info("\n=== Testing Async Filtering ===")

    # Создаем фильтр
    config = FilterConfig(
        enabled=True,
        fractal_dimension_lower=1.2,
        fractal_dimension_upper=1.4,
        entropy_threshold=0.7,
        min_data_points=20,
        window_size=50,
    )
    filter_obj = OrderBookPreFilter(config)

    # Симуляция потока данных
    async def order_book_stream():
        for i in range(10):
            # Создаем случайный ордербук
            bids = [
                (Price(Decimal(str(50000 - j * 10 + random.gauss(0, 1))), Currency.USDT), 
                Volume(Decimal(str(random.uniform(0.1, 2.0)))))
                for j in range(10)
            ]
            asks = [
                (Price(Decimal(str(50010 + j * 10 + random.gauss(0, 1))), Currency.USDT), 
                Volume(Decimal(str(random.uniform(0.1, 2.0)))))
                for j in range(10)
            ]
            
            yield {
                "exchange": "test",
                "symbol": "BTCUSDT",
                "bids": bids,
                "asks": asks,
                "timestamp": Timestamp.from_unix(int(time.time())),
                "sequence_id": i,
            }
            await asyncio.sleep(0.1)

    # Обрабатываем поток
    async for order_book_data in order_book_stream():
        filtered_ob = filter_obj.filter_order_book(**order_book_data)
        logger.info(f"Filtered order book {order_book_data['sequence_id']}: {filtered_ob.meta.get('filtered', False)}")


def test_fractal_dimension_calculation():
    """Тестирование расчета фрактальной размерности."""
    logger.info("\n=== Testing Fractal Dimension Calculation ===")

    # Создаем анализатор
    analyzer = NoiseAnalyzer()

    # Тестовые данные с известной фрактальной размерностью
    test_data = [
        # Линейные данные (размерность ~1.0)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        # Случайные данные (размерность ~1.5)
        [random.random() for _ in range(20)],
        # Фрактальные данные (размерность ~1.26)
        [random.gauss(0, 1) for _ in range(20)],
    ]

    for i, data in enumerate(test_data):
        # Создаем временной ряд из данных
        prices = [float(x) for x in data]
        
        # Создаем ордербук на основе цен
        base_price = sum(prices) / len(prices)
        bids = [(Price(Decimal(str(base_price - i * 0.1)), Currency.USDT), Volume(Decimal("1.0"))) for i in range(10)]
        asks = [(Price(Decimal(str(base_price + i * 0.1)), Currency.USDT), Volume(Decimal("1.0"))) for i in range(10)]
        
        order_book = OrderBookSnapshot(
            exchange="test",
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            timestamp=Timestamp.from_unix(int(time.time())),
        )
        
        result = analyzer.analyze_noise(order_book)
        logger.info(f"Test data {i+1}: FD={result.fractal_dimension:.3f}")


def test_entropy_calculation():
    """Тестирование расчета энтропии."""
    logger.info("\n=== Testing Entropy Calculation ===")

    # Создаем анализатор
    analyzer = NoiseAnalyzer()

    # Тестовые данные с разной энтропией
    test_cases = [
        ("Low entropy (ordered)", [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        ("Medium entropy (mixed)", [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]),
        ("High entropy (random)", [random.randint(1, 10) for _ in range(10)]),
    ]

    for name, data in test_cases:
        # Создаем ордербук на основе данных
        base_price = 50000.0
        bids = [(Price(Decimal(str(base_price - i * 10)), Currency.USDT), Volume(Decimal(str(data[i % len(data)])))) for i in range(10)]
        asks = [(Price(Decimal(str(base_price + i * 10)), Currency.USDT), Volume(Decimal(str(data[i % len(data)])))) for i in range(10)]
        
        order_book = OrderBookSnapshot(
            exchange="test",
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            timestamp=Timestamp.from_unix(int(time.time())),
        )
        
        result = analyzer.analyze_noise(order_book)
        logger.info(f"{name}: Entropy={result.entropy:.3f}")


async def main():
    """Основная функция для запуска всех тестов."""
    logger.info("Starting Neural Noise Analysis Example")
    
    try:
        # Тестируем анализатор шума
        test_noise_analyzer()
        
        # Тестируем фильтр ордербука
        test_order_book_filter()
        
        # Тестируем асинхронную фильтрацию
        await test_async_filtering()
        
        # Тестируем расчет фрактальной размерности
        test_fractal_dimension_calculation()
        
        # Тестируем расчет энтропии
        test_entropy_calculation()
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
