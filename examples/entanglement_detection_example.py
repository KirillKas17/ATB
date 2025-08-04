# -*- coding: utf-8 -*-
"""
Пример использования системы Quantum Order Entanglement Detection.

Этот пример демонстрирует:
1. Создание и настройку EntanglementMonitor
2. Запуск мониторинга запутанности между биржами
3. Анализ результатов и статистики
"""

import asyncio
import time

from loguru import logger

from application.analysis.entanglement_monitor import EntanglementMonitor
from domain.intelligence.entanglement_detector import EntanglementDetector


async def main():
    """Основная функция примера."""
    logger.info("=== Quantum Order Entanglement Detection Example ===")

    # Создаем мониторинг запутанности
    monitor = EntanglementMonitor(
        log_file_path="logs/entanglement_events.json",
        detection_interval=0.5,  # Проверка каждые 500мс
        max_lag_ms=3.0,  # Максимальный lag 3мс
        correlation_threshold=0.95,  # Порог корреляции 95%
    )

    logger.info("EntanglementMonitor created successfully")

    # Запускаем мониторинг в фоновом режиме
    monitor_task = asyncio.create_task(monitor.start_monitoring())

    try:
        # Мониторим в течение 30 секунд
        logger.info("Starting monitoring for 30 seconds...")

        for i in range(30):
            await asyncio.sleep(1)

            # Каждые 5 секунд выводим статус
            if (i + 1) % 5 == 0:
                status = monitor.get_status()
                logger.info(f"Status at {i+1}s: {status}")

        # Останавливаем мониторинг
        monitor.stop_monitoring()
        await monitor_task

        # Выводим финальную статистику
        final_status = monitor.get_status()
        logger.info("=== Final Statistics ===")
        logger.info(f"Total detections: {final_status['stats']['total_detections']}")
        logger.info(
            f"Entangled detections: {final_status['stats']['entangled_detections']}"
        )
        logger.info(
            f"Detection rate: {final_status['stats']['entangled_detections'] / max(1, final_status['stats']['total_detections']) * 100:.2f}%"
        )

        # Получаем историю обнаружений
        history = monitor.get_entanglement_history(limit=10)
        logger.info(f"Recent entanglement events: {len(history)}")

        for event in history[-5:]:  # Последние 5 событий
            data = event["data"]
            logger.info(
                f"Entanglement: {data['exchange_pair'][0]} ↔ {data['exchange_pair'][1]} "
                f"({data['symbol']}) - Lag: {data['lag_ms']:.2f}ms, "
                f"Correlation: {data['correlation_score']:.3f}"
            )

    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
        monitor.stop_monitoring()
        await monitor_task

    except Exception as e:
        logger.error(f"Error during monitoring: {e}")
        monitor.stop_monitoring()
        await monitor_task

    logger.info("=== Example completed ===")


def test_entanglement_detector():
    """Тестирование EntanglementDetector напрямую."""
    logger.info("=== Testing EntanglementDetector ===")

    detector = EntanglementDetector(
        max_lag_ms=3.0, correlation_threshold=0.95, window_size=50, min_data_points=20
    )

    # Создаем тестовые данные

    # Симулируем данные с высокой корреляцией
    base_prices = np.linspace(50000, 51000, 100)
    noise1 = np.random.normal(0, 10, 100)
    noise2 = np.random.normal(0, 10, 100)

    # Данные с задержкой и корреляцией
    prices1 = base_prices + noise1
    prices2 = np.roll(base_prices + noise2, 2)  # Задержка на 2 тика

    # Создаем обновления ордербуков
    updates = []

    for i in range(100):
        # Обновление для первой биржи
        update1 = OrderBookUpdate(
            exchange="binance",
            symbol="BTCUSDT",
            bids=[(Price(prices1[i] - 10), Volume(1.0))],
            asks=[(Price(prices1[i] + 10), Volume(1.0))],
            timestamp=Timestamp(time.time() + i * 0.001),  # 1мс интервалы
        )
        updates.append(update1)

        # Обновление для второй биржи (с задержкой)
        if i >= 2:
            update2 = OrderBookUpdate(
                exchange="coinbase",
                symbol="BTCUSDT",
                bids=[(Price(prices2[i] - 10), Volume(1.0))],
                asks=[(Price(prices2[i] + 10), Volume(1.0))],
                timestamp=Timestamp(time.time() + i * 0.001),
            )
            updates.append(update2)

    # Обрабатываем обновления
    results = detector.process_order_book_updates(updates)

    logger.info(f"Processed {len(updates)} updates, got {len(results)} results")

    for result in results:
        logger.info(
            f"Detection: {result.exchange_pair[0]} ↔ {result.exchange_pair[1]} "
            f"- Entangled: {result.is_entangled}, "
            f"Lag: {result.lag_ms:.2f}ms, "
            f"Correlation: {result.correlation_score:.3f}, "
            f"Confidence: {result.confidence:.3f}"
        )

    logger.info("=== EntanglementDetector test completed ===")


if __name__ == "__main__":
    # Запускаем тест детектора
    test_entanglement_detector()

    # Запускаем основной пример
    asyncio.run(main())
