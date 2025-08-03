import os
import subprocess
from typing import List

# Папки, которые нужно обработать
INCLUDE_DIRS: List[str] = [
    "agents",
    "analysis",
    "backtest",
    "config",
    "core",
    "dashboard",
    "data",
    "exchange",
    "ml",
    "patterns",
    "scripts",
    "simulation",
    "state",
    "strategies",
    "utils",
]

# Папки, которые нужно пропустить
EXCLUDE_DIRS: List[str] = ["venv", "logs", "models", "datasets", "__pycache__"]

# Параметры black
LINE_LENGTH: int = 100


def should_exclude(path: str) -> bool:
    """Проверяет, следует ли исключить путь из обработки"""
    return any(exclude in path for exclude in EXCLUDE_DIRS)


def run_black() -> None:
    """Запускает форматирование кода с помощью black"""
    for directory in INCLUDE_DIRS:
        if os.path.exists(directory) and not should_exclude(directory):
            print(f"\n📂 Форматируем: {directory}")
            subprocess.run(["black", directory, f"--line-length={LINE_LENGTH}"])
        else:
            print(f"⏭ Пропущено: {directory}")


if __name__ == "__main__":
    print("🚀 Запуск автоформатирования black...\n")
    run_black()
    print("\n✅ Готово!")
