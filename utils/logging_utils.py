import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import aiofiles

# Создание директории для логов
Path("logs").mkdir(exist_ok=True)

# Очередь для асинхронной записи логов
log_queue = asyncio.Queue()


async def _write_log(message: str, level: str, context: str = ""):
    """Асинхронная запись лога в файл"""
    try:
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "context": context,
            "message": message,
        }

        # Запись в соответствующий файл
        filename = f"logs/{level.lower()}.log"
        async with aiofiles.open(filename, mode="a", encoding="utf-8") as f:
            await f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"Ошибка записи лога: {str(e)}", file=sys.stderr)


async def _log_worker():
    """Фоновый процесс для записи логов"""
    while True:
        try:
            log_entry = await log_queue.get()
            await _write_log(**log_entry)
            log_queue.task_done()
        except Exception as e:
            print(f"Ошибка в лог-воркере: {str(e)}", file=sys.stderr)


def log_error(message: str, context: str = ""):
    """Логирование ошибок"""
    try:
        # Вывод в консоль
        print(f"[ERROR] {context}: {message}", file=sys.stderr)

        # Добавление в очередь для асинхронной записи
        asyncio.create_task(
            log_queue.put({"message": message, "level": "ERROR", "context": context})
        )

    except Exception as e:
        print(f"Ошибка логирования: {str(e)}", file=sys.stderr)


def log_info(message: str, context: str = ""):
    """Логирование информационных сообщений"""
    try:
        # Вывод в консоль
        print(f"[INFO] {context}: {message}")

        # Добавление в очередь для асинхронной записи
        asyncio.create_task(
            log_queue.put({"message": message, "level": "INFO", "context": context})
        )

    except Exception as e:
        print(f"Ошибка логирования: {str(e)}", file=sys.stderr)


def log_warning(message: str, context: str = ""):
    """Логирование предупреждений"""
    try:
        # Вывод в консоль
        print(f"[WARNING] {context}: {message}")

        # Добавление в очередь для асинхронной записи
        asyncio.create_task(
            log_queue.put({"message": message, "level": "WARNING", "context": context})
        )

    except Exception as e:
        print(f"Ошибка логирования: {str(e)}", file=sys.stderr)


def log_debug(message: str, context: str = ""):
    """Логирование отладочных сообщений"""
    try:
        # Вывод в консоль
        print(f"[DEBUG] {context}: {message}")

        # Добавление в очередь для асинхронной записи
        asyncio.create_task(
            log_queue.put({"message": message, "level": "DEBUG", "context": context})
        )

    except Exception as e:
        print(f"Ошибка логирования: {str(e)}", file=sys.stderr)


async def start_logging():
    """Запуск фонового процесса логирования"""
    asyncio.create_task(_log_worker())


async def stop_logging():
    """Остановка фонового процесса логирования"""
    await log_queue.join()
