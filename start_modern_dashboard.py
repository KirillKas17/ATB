#!/usr/bin/env python3
"""
Запуск современного дашборда ATB
"""

import sys
import os

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from modern_dashboard_app import MainWindow
    from PyQt6.QtWidgets import QApplication
    
    def main() -> None:
        """Главная функция запуска приложения"""
        app = QApplication(sys.argv)
        
        # Устанавливаем информацию о приложении
        app.setApplicationName("ATB Modern Dashboard")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("ATB Trading System")
        
        # Создаем и показываем главное окно
        window = MainWindow()
        window.show()
        
        # Запускаем главный цикл приложения
        sys.exit(app.exec())
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("Убедитесь, что установлены все зависимости:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Ошибка запуска: {e}")
    sys.exit(1) 