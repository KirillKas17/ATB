#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для запуска мониторинга Quantum Order Entanglement Detection.

Использование:
    python scripts/start_entanglement_monitoring.py [--config config.yaml] [--duration 3600]
"""

import argparse
import asyncio
import signal
import sys
import time
from pathlib import Path

from loguru import logger

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from application.analysis.entanglement_monitor import EntanglementMonitor
from shared.config import reload_config as load_config


def setup_logging():
    """Настройка логирования."""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        "logs/entanglement_monitoring.log",
        rotation="1 day",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
    )


def signal_handler(signum, frame):
    """Обработчик сигналов для graceful shutdown."""
    logger.info(f"Получен сигнал {signum}, завершение работы...")
    sys.exit(0)


async def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(
        description="Quantum Order Entanglement Detection Monitor"
    )
    parser.add_argument(
        "--config", default="config/config.yaml", help="Путь к конфигурационному файлу"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Длительность мониторинга в секундах (0 = бесконечно)",
    )
    parser.add_argument(
        "--log-file", default="logs/entanglement_events.json", help="Путь к файлу логов"
    )
    parser.add_argument(
        "--detection-interval",
        type=float,
        default=1.0,
        help="Интервал обнаружения в секундах",
    )
    parser.add_argument(
        "--max-lag", type=float, default=3.0, help="Максимальный lag в миллисекундах"
    )
    parser.add_argument(
        "--correlation-threshold", type=float, default=0.95, help="Порог корреляции"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод")

    args = parser.parse_args()

    # Настройка логирования
    setup_logging()
    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")

    # Обработка сигналов
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=== Quantum Order Entanglement Detection Monitor ===")
    logger.info(f"Конфигурация: {args.config}")
    logger.info(
        f"Длительность: {args.duration if args.duration > 0 else 'бесконечно'} секунд"
    )
    logger.info(f"Интервал обнаружения: {args.detection_interval}с")
    logger.info(f"Максимальный lag: {args.max_lag}мс")
    logger.info(f"Порог корреляции: {args.correlation_threshold}")

    try:
        # Загрузка конфигурации
        config = load_config(args.config)
        logger.info("Конфигурация загружена успешно")

        # Создание монитора
        monitor = EntanglementMonitor(
            log_file_path=args.log_file,
            detection_interval=args.detection_interval,
            max_lag_ms=args.max_lag,
            correlation_threshold=args.correlation_threshold,
        )

        logger.info("EntanglementMonitor создан успешно")

        # Вывод начальной статистики
        initial_status = monitor.get_status()
        logger.info(f"Активных пар бирж: {initial_status['active_pairs']}")
        logger.info(f"Всего пар: {initial_status['total_pairs']}")

        # Запуск мониторинга
        start_time = time.time()
        monitor_task = asyncio.create_task(monitor.start_monitoring())

        logger.info("Мониторинг запущен...")

        # Основной цикл
        while True:
            await asyncio.sleep(5)  # Проверка каждые 5 секунд

            # Вывод статистики
            status = monitor.get_status()
            elapsed = time.time() - start_time

            logger.info(
                f"Время работы: {elapsed:.1f}с | "
                f"Проверок: {status['stats']['total_detections']} | "
                f"Запутанностей: {status['stats']['entangled_detections']} | "
                f"Активных пар: {status['active_pairs']}"
            )

            # Проверка длительности
            if args.duration > 0 and elapsed >= args.duration:
                logger.info(f"Достигнута максимальная длительность ({args.duration}с)")
                break

        # Остановка мониторинга
        logger.info("Остановка мониторинга...")
        monitor.stop_monitoring()
        await monitor_task

        # Финальная статистика
        final_status = monitor.get_status()
        total_time = time.time() - start_time

        logger.info("=== Финальная статистика ===")
        logger.info(f"Общее время работы: {total_time:.1f}с")
        logger.info(f"Всего проверок: {final_status['stats']['total_detections']}")
        logger.info(
            f"Обнаружено запутанностей: {final_status['stats']['entangled_detections']}"
        )

        if final_status["stats"]["total_detections"] > 0:
            detection_rate = (
                final_status["stats"]["entangled_detections"]
                / final_status["stats"]["total_detections"]
                * 100
            )
            logger.info(f"Частота обнаружения: {detection_rate:.2f}%")

        # Размеры буферов
        logger.info("Размеры буферов:")
        for exchange, size in final_status["buffer_sizes"].items():
            logger.info(f"  {exchange}: {size}")

        logger.info("=== Мониторинг завершен ===")

    except KeyboardInterrupt:
        logger.info("Мониторинг прерван пользователем")
        if "monitor" in locals():
            monitor.stop_monitoring()
            await monitor_task

    except Exception as e:
        logger.error(f"Ошибка в мониторинге: {e}")
        if "monitor" in locals():
            monitor.stop_monitoring()
            await monitor_task
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
