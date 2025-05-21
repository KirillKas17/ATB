import os
import subprocess

# Папки, которые нужно обработать
INCLUDE_DIRS = [
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
EXCLUDE_DIRS = ["venv", "logs", "models", "datasets", "__pycache__"]

# Параметры black
LINE_LENGTH = 100


def should_exclude(path):
    return any(exclude in path for exclude in EXCLUDE_DIRS)


def run_black():
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
