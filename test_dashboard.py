#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы современного дашборда
"""

import sys
import os

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports() -> None:
    """Тестирование импортов"""
    try:
        from PyQt6.QtWidgets import QApplication
        print("✅ PyQt6 импортирован успешно")
        
        import matplotlib
        print("✅ matplotlib импортирован успешно")
        
        import numpy as np
        print("✅ numpy импортирован успешно")
        
        from modern_dashboard_app import MainWindow
        print("✅ MainWindow импортирован успешно")
        
        return True
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return False

def test_application() -> None:
    """Тестирование запуска приложения"""
    try:
        from PyQt6.QtWidgets import QApplication
        from modern_dashboard_app import MainWindow
        
        # Создаем приложение
        app = QApplication(sys.argv)
        
        # Создаем главное окно
        window = MainWindow()
        
        print("✅ Приложение создано успешно")
        print("✅ Главное окно создано успешно")
        
        # Показываем окно на короткое время
        window.show()
        print("✅ Окно отображено успешно")
        
        # Закрываем приложение
        app.quit()
        print("✅ Приложение закрыто успешно")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования приложения: {e}")
        return False

def main() -> None:
    """Главная функция тестирования"""
    print("🧪 Тестирование современного дашборда ATB")
    print("=" * 50)
    
    # Тест импортов
    print("\n1. Тестирование импортов...")
    if not test_imports():
        print("❌ Тест импортов не пройден")
        return False
    
    # Тест приложения
    print("\n2. Тестирование приложения...")
    if not test_application():
        print("❌ Тест приложения не пройден")
        return False
    
    print("\n✅ Все тесты пройдены успешно!")
    print("🎉 Приложение готово к использованию")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 