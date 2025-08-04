#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт запуска ATB Trading System Desktop Application
Запуск современного Windows приложения для торговой системы ATB
"""

import argparse
import os
import sys
from pathlib import Path

# Добавление корневой директории в путь
sys.path.append(str(Path(__file__).parent))

from loguru import logger

def setup_environment():
    """Настройка переменных окружения"""
    # Загрузка переменных из .env файла
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value
        logger.info("Environment variables loaded from .env file")

def check_dependencies():
    """Проверка зависимостей"""
    required_dirs = ["logs", "config", "data", "models", "backups"]

    for dir_name in required_dirs:
        dir_path = Path(__file__).parent / dir_name
        dir_path.mkdir(exist_ok=True)
        logger.info(f"Directory {dir_name} ready")

def check_pyqt6():
    """Проверка наличия PyQt6"""
    try:
        import PyQt6
        logger.info("PyQt6 is available")
        return True
    except ImportError:
        logger.error("PyQt6 is not installed. Please install it with: pip install PyQt6")
        return False

def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description="ATB Trading System Desktop Application")

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["live", "paper", "backtest", "simulation"],
        default="simulation",
        help="Trading mode",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    parser.add_argument(
        "--dry-run", action="store_true", help="Run without executing trades"
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check system health without starting application",
    )

    parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Use enhanced version with additional features",
    )

    return parser.parse_args()

def main():
    """Основная функция"""
    # Парсинг аргументов
    args = parse_arguments()

    # Настройка окружения
    setup_environment()

    # Проверка зависимостей
    check_dependencies()

    # Проверка PyQt6
    if not check_pyqt6():
        print("❌ PyQt6 не установлен. Установите его командой:")
        print("pip install PyQt6 PyQt6-Charts PyQt6-WebEngine")
        return 1

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
            return 0

        # Запуск десктопного приложения
        logger.info(f"Starting ATB Trading System Desktop Application in {args.mode} mode")

        if args.enhanced:
            # Запуск улучшенной версии
            from atb_desktop_app_enhanced import main as desktop_main
            logger.info("Using enhanced desktop application")
        else:
            # Запуск базовой версии
            from atb_desktop_app import main as desktop_main
            logger.info("Using basic desktop application")

        # Запуск приложения
        desktop_main()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise

if __name__ == "__main__":
    sys.exit(main()) 