#!/usr/bin/env python3
"""
Скрипт запуска торговой системы Syntra
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Добавление корневой директории в путь
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger

from main import main as main_function


def setup_environment() -> None:
    """Настройка переменных окружения"""
    # Загрузка переменных из .env файла
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value
        logger.info("Environment variables loaded from .env file")


def check_dependencies() -> None:
    """Проверка зависимостей"""
    required_dirs = ["logs", "config", "data", "models", "backups"]

    for dir_name in required_dirs:
        dir_path = Path(__file__).parent.parent / dir_name
        dir_path.mkdir(exist_ok=True)
        logger.info(f"Directory {dir_name} ready")


def parse_arguments() -> None:
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description="Syntra Trading System")

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["live", "paper", "backtest"],
        default="paper",
        help="Trading mode",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    parser.add_argument(
        "--dry-run", action="store_true", help="Run without executing trades"
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check system health without starting trading",
    )

    return parser.parse_args()


async def main() -> None:
    """Основная функция"""
    # Парсинг аргументов
    args = parse_arguments()

    # Настройка окружения
    setup_environment()

    # Проверка зависимостей
    check_dependencies()

    # Установка переменных окружения из аргументов
    if args.debug:
        os.environ["ATB_DEBUG"] = "true"

    if args.dry_run:
        os.environ["ATB_DRY_RUN"] = "true"

    os.environ["ATB_MODE"] = args.mode

    try:
        if args.check_only:
            # Только проверка системы
            logger.info("Running system health check...")
            # Здесь можно добавить проверку здоровья системы
            logger.info("System health check completed")
            return

        # Полный запуск системы
        logger.info(f"Starting Syntra Trading System in {args.mode} mode")

        # Запуск основной функции
        await main_function()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"System error: {e}")
        raise


if __name__ == "__main__":
    # Запуск системы
    asyncio.run(main())
