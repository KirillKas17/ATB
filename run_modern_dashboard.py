#!/usr/bin/env python3
"""
Универсальный лаунчер современного торгового дашборда ATB.
Выбор между Desktop (Tkinter) и Web (Flask) версиями.
"""

import sys
import os
import subprocess
from pathlib import Path
import argparse

def check_dependencies():
    """Проверка зависимостей"""
    print("🔍 Проверка зависимостей...")
    
    # Базовые зависимости
    required_packages = {
        'tkinter': 'tkinter (встроен в Python)',
        'matplotlib': 'matplotlib',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'flask': 'flask',
        'flask_socketio': 'flask-socketio',
    }
    
    missing_packages = []
    
    for package, install_name in required_packages.items():
        try:
            if package == 'tkinter':
                import tkinter
            elif package == 'matplotlib':
                import matplotlib
            elif package == 'numpy':
                import numpy
            elif package == 'pandas':
                import pandas
            elif package == 'flask':
                import flask
            elif package == 'flask_socketio':
                import flask_socketio
                
            print(f"✅ {package}")
            
        except ImportError:
            print(f"❌ {package} - НЕ установлен")
            missing_packages.append(install_name)
    
    if missing_packages:
        print(f"\n❌ Найдены отсутствующие зависимости: {', '.join(missing_packages)}")
        print("📦 Установите их командой:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ Все зависимости установлены!")
    return True

def show_banner():
    """Показать баннер"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🚀 ATB Trading Dashboard                   ║
║                     Modern Apple-style UI                    ║
║                                                              ║
║  💫 Dark Theme  📊 Live Data  🎯 AI Signals  📺 Twitch Ready ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def get_dashboard_choice():
    """Выбор типа дашборда"""
    print("📊 Выберите тип дашборда:\n")
    print("1. 🖥️  Desktop версия (Tkinter)")
    print("   ├─ Максимальная производительность")
    print("   ├─ Нативный интерфейс Windows")
    print("   └─ Локальный запуск\n")
    
    print("2. 🌐 Web версия (Flask + WebSocket)")
    print("   ├─ Современный веб-интерфейс")
    print("   ├─ Доступ через браузер")
    print("   ├─ Лучше для демонстрации")
    print("   └─ Поддержка мультимедиа\n")
    
    while True:
        try:
            choice = input("Ваш выбор (1 или 2): ").strip()
            if choice in ['1', '2']:
                return int(choice)
            else:
                print("❌ Введите 1 или 2")
        except KeyboardInterrupt:
            print("\n👋 Выход из программы")
            sys.exit(0)

def run_desktop_dashboard():
    """Запуск desktop версии"""
    print("🖥️ Запуск Desktop дашборда...")
    print("💫 Apple-style темная тема")
    print("📊 Live метрики и графики")
    print("🎯 AI сигналы в реальном времени")
    print("\n🔄 Инициализация...")
    
    try:
        # Проверка tkinter (требуется для desktop версии)
        try:
            import tkinter
        except ImportError:
            print("❌ tkinter не установлен. Для Linux установите:")
            print("sudo apt-get install python3-tk")
            print("💡 Используйте Web версию вместо Desktop")
            return
            
        # Импорт и запуск desktop версии
        sys.path.append(str(Path(__file__).parent))
        from interfaces.presentation.dashboard.modern_trading_dashboard import ModernTradingDashboard
        
        dashboard = ModernTradingDashboard()
        dashboard.run()
        
    except Exception as e:
        print(f"❌ Ошибка запуска Desktop дашборда: {e}")
        print("💡 Попробуйте Web версию")

def run_web_dashboard():
    """Запуск web версии"""
    print("🌐 Запуск Web дашборда...")
    print("💫 Apple-style веб-интерфейс")
    print("📡 WebSocket для live-данных")
    print("🎯 Оптимизирован для демонстрации")
    print("\n🔄 Запуск сервера...")
    
    try:
        # Импорт и запуск web версии
        sys.path.append(str(Path(__file__).parent))
        from interfaces.presentation.dashboard.web_dashboard import app, socketio
        
        print("🌍 Dashboard URL: http://localhost:5000")
        print("📺 Готов для Twitch демонстрации!")
        print("🔴 Live Data Streaming...")
        print("\n💡 Откройте браузер и перейдите по адресу выше")
        print("⚠️  Для остановки нажмите Ctrl+C\n")
        
        socketio.run(app, debug=False, host='0.0.0.0', port=5000)
        
    except Exception as e:
        print(f"❌ Ошибка запуска Web дашборда: {e}")
        print("💡 Попробуйте Desktop версию")

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='ATB Trading Dashboard Launcher')
    parser.add_argument('--type', choices=['desktop', 'web'], 
                       help='Тип дашборда: desktop или web')
    parser.add_argument('--skip-check', action='store_true',
                       help='Пропустить проверку зависимостей')
    
    args = parser.parse_args()
    
    # Показать баннер
    show_banner()
    
    # Проверка зависимостей
    if not args.skip_check:
        if not check_dependencies():
            sys.exit(1)
        print()
    
    # Выбор типа дашборда
    if args.type:
        if args.type == 'desktop':
            dashboard_type = 1
        else:
            dashboard_type = 2
    else:
        dashboard_type = get_dashboard_choice()
    
    print()
    
    # Запуск выбранного дашборда
    try:
        if dashboard_type == 1:
            run_desktop_dashboard()
        else:
            run_web_dashboard()
            
    except KeyboardInterrupt:
        print("\n\n👋 Дашборд остановлен пользователем")
        print("💾 Данные сохранены")
        print("🚀 Спасибо за использование ATB Trading Dashboard!")
        
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        print("🔧 Попробуйте перезапустить программу")
        sys.exit(1)

if __name__ == "__main__":
    main()