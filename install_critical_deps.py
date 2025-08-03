#!/usr/bin/env python3
"""
Скрипт установки критических зависимостей для ATB Trading System
"""

import subprocess
import sys
import os
from typing import List, Tuple

# Критические зависимости в порядке приоритета
CRITICAL_DEPENDENCIES = [
    # Основа Python
    "typing_extensions",
    "pydantic",
    
    # Обработка данных
    "pandas",
    "numpy", 
    "scipy",
    
    # Веб и сеть
    "requests",
    "aiohttp",
    "websockets",
    
    # База данных
    "sqlalchemy",
    "psycopg2-binary",
    "redis",
    "alembic",
    
    # Конфигурация
    "python-dotenv",
    "pyyaml",
    
    # Машинное обучение (базовое)
    "scikit-learn",
    "joblib",
    
    # Утилиты
    "tqdm",
    "pytest",
    "black",
    "isort",
]

# Зависимости для торговли (могут отсутствовать в некоторых системах)
TRADING_DEPENDENCIES = [
    "ccxt",
    "yfinance",
]

# ML зависимости (тяжелые, устанавливаем по желанию)
HEAVY_ML_DEPENDENCIES = [
    "tensorflow",
    "torch", 
    "transformers",
    "xgboost",
    "catboost",
]

def run_command(command: List[str]) -> Tuple[bool, str]:
    """Выполнить команду и вернуть результат."""
    try:
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 минут на пакет
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout: установка заняла слишком много времени"
    except Exception as e:
        return False, f"Ошибка: {e}"

def install_package(package: str, use_break_system: bool = True) -> bool:
    """Установить один пакет."""
    print(f"📦 Устанавливаю {package}...")
    
    command = ["pip3", "install"]
    if use_break_system:
        command.append("--break-system-packages")
    command.append(package)
    
    success, output = run_command(command)
    
    if success:
        print(f"✅ {package} установлен успешно")
        return True
    else:
        print(f"❌ Ошибка установки {package}")
        print(f"   Вывод: {output[:200]}...")
        return False

def check_package(package: str) -> bool:
    """Проверить, установлен ли пакет."""
    try:
        __import__(package.replace("-", "_"))
        return True
    except ImportError:
        return False

def install_dependencies(deps: List[str], category: str) -> Tuple[int, int]:
    """Установить список зависимостей."""
    print(f"\n🔧 Установка {category}...")
    
    installed = 0
    failed = 0
    
    for package in deps:
        # Проверяем, установлен ли уже
        package_import_name = package.replace("-", "_")
        if check_package(package_import_name):
            print(f"✅ {package} уже установлен")
            installed += 1
            continue
            
        # Пытаемся установить
        if install_package(package):
            installed += 1
        else:
            failed += 1
            # Пытаемся альтернативные способы
            print(f"🔄 Пробую альтернативный метод для {package}...")
            if install_package(package, use_break_system=False):
                installed += 1
                failed -= 1
    
    return installed, failed

def main() -> None:
    """Основная функция."""
    print("🚀 Установка критических зависимостей ATB Trading System")
    print("=" * 60)
    
    # Проверяем Python и pip
    try:
        python_version = sys.version_info
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    except Exception as e:
        print(f"❌ Ошибка проверки Python: {e}")
        return False
    
    # Создаем директории
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    total_installed = 0
    total_failed = 0
    
    # Устанавливаем критические зависимости
    installed, failed = install_dependencies(CRITICAL_DEPENDENCIES, "критических зависимостей")
    total_installed += installed
    total_failed += failed
    
    # Устанавливаем торговые зависимости
    print(f"\n🤔 Устанавливать торговые зависимости? (могут быть проблемы)")
    install_trading = input("y/N: ").lower().startswith('y')
    
    if install_trading:
        installed, failed = install_dependencies(TRADING_DEPENDENCIES, "торговых зависимостей")
        total_installed += installed
        total_failed += failed
    
    # ML зависимости только по запросу
    print(f"\n🤔 Устанавливать ML зависимости? (требуют много времени и места)")
    install_ml = input("y/N: ").lower().startswith('y')
    
    if install_ml:
        installed, failed = install_dependencies(HEAVY_ML_DEPENDENCIES, "ML зависимостей")
        total_installed += installed
        total_failed += failed
    
    # Финальный отчет
    print("\n" + "=" * 60)
    print("📊 ИТОГОВЫЙ ОТЧЕТ:")
    print(f"✅ Успешно установлено: {total_installed}")
    print(f"❌ Не удалось установить: {total_failed}")
    
    if total_failed == 0:
        print("🎉 Все зависимости установлены успешно!")
        
        # Тестируем импорт
        print("\n🔍 Тестирование критических импортов...")
        test_imports = ["pandas", "numpy", "requests", "pydantic", "sqlalchemy"]
        
        for package in test_imports:
            try:
                __import__(package)
                print(f"✅ {package}")
            except ImportError as e:
                print(f"❌ {package}: {e}")
                
        return True
    else:
        print("⚠️  Некоторые зависимости не установились")
        print("💡 Попробуйте:")
        print("   1. Обновить pip: pip3 install --upgrade pip")
        print("   2. Использовать virtual environment")
        print("   3. Установить системные пакеты: apt install python3-dev")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)