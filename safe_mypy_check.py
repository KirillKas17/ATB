import os
import subprocess

CHECK_DIRS = [
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


def contains_python_files(path):
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".py"):
                return True
    return False


def run_mypy():
    for path in CHECK_DIRS:
        if os.path.exists(path) and contains_python_files(path):
            print(f"\n🔍 Проверка типов: {path}")
            subprocess.run(["mypy", path, "--ignore-missing-imports"])
        else:
            print(f"⏭ Пропущено (нет .py): {path}")


if __name__ == "__main__":
    print("🚀 Запуск безопасной проверки mypy...\n")
    run_mypy()
    print("\n✅ Готово.")
