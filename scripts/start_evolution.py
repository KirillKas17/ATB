#!/usr/bin/env python3
"""
Скрипт для запуска модуля эволюции торговых стратегий.
"""

import asyncio
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from application.di_container_refactored import ContainerConfig, get_service_locator
from shared.config import get_config
from shared.logging import setup_logging


async def main() -> None:
    """Основная функция запуска эволюции стратегий."""
    logger = setup_logging()

    try:
        logger.info("Запуск модуля эволюции стратегий...")

        # Получаем конфигурацию
        config = get_config()
        evolution_config = config.get("evolution", {})

        if not evolution_config.get("enabled", False):
            logger.warning("Модуль эволюции стратегий отключен в конфигурации")
            return

        # Создаем конфигурацию контейнера
        container_config = ContainerConfig(
            evolution_enabled=True,
            cache_enabled=True,
            risk_management_enabled=True,
            technical_analysis_enabled=True,
        )

        # Получаем DI контейнер
        container = get_service_locator(container_config)

        # Получаем оркестратор эволюции
        evolution_orchestrator = container.get("evolution_orchestrator")

        logger.info("Модуль эволюции стратегий инициализирован")
        logger.info(f"Размер популяции: {evolution_config.get('population_size', 50)}")
        logger.info(f"Количество поколений: {evolution_config.get('generations', 100)}")
        logger.info(
            f"Минимальная точность: {evolution_config.get('min_accuracy', 0.82)}"
        )
        logger.info(
            f"Минимальная прибыльность: {evolution_config.get('min_profitability', 0.05)}"
        )

        # Запускаем эволюцию
        logger.info("Запуск процесса эволюции стратегий...")
        await evolution_orchestrator.start_evolution()

    except KeyboardInterrupt:
        logger.info("Получен сигнал прерывания, остановка эволюции...")
    except Exception as e:
        logger.error(f"Ошибка при запуске эволюции стратегий: {e}")
        raise
    finally:
        logger.info("Завершение работы модуля эволюции стратегий")


if __name__ == "__main__":
    asyncio.run(main())
