#!/usr/bin/env python3
"""
Скрипт для запуска линтера Ruff
"""

import subprocess
import sys
from pathlib import Path


def run_ruff():
    """Запуск Ruff для проверки кода"""
    try:
        # Проверяем наличие Ruff
        subprocess.run(["ruff", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Установка Ruff...")
        subprocess.run([sys.executable, "-m", "pip", "install", "ruff"], check=True)

    # Запускаем проверку кода
    print("Запуск проверки кода...")
    result = subprocess.run(["ruff", "check", "."], capture_output=True, text=True)

    if result.returncode != 0:
        print("Найдены ошибки:")
        print(result.stdout)
        print(result.stderr)
        return False

    print("Проверка кода завершена успешно!")
    return True


def format_code():
    """Форматирование кода с помощью Ruff"""
    try:
        print("Форматирование кода...")
        result = subprocess.run(["ruff", "format", "."], capture_output=True, text=True)

        if result.returncode != 0:
            print("Ошибка при форматировании:")
            print(result.stdout)
            print(result.stderr)
            return False

        print("Код отформатирован успешно!")
        return True
    except Exception as e:
        print(f"Ошибка при форматировании: {e}")
        return False


if __name__ == "__main__":
    # Создаем директорию для скриптов, если её нет
    Path("scripts").mkdir(exist_ok=True)

    # Запускаем проверку и форматирование
    if run_ruff():
        format_code()
