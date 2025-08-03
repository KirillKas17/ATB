#!/usr/bin/env python3
"""
Запуск интегрированного ATB Trading System с демонстрационным дашбордом
и полным управлением настройками через GUI.
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Проверка версии Python"""
    if sys.version_info < (3, 8):
        print("❌ Требуется Python 3.8 или выше")
        print(f"Текущая версия: {sys.version}")
        return False
    return True

def check_dependencies():
    """Проверка основных зависимостей"""
    print("🔍 Проверка зависимостей...")
    
    required_packages = {
        'tkinter': 'tkinter (GUI)',
        'matplotlib': 'matplotlib (графики)',
        'numpy': 'numpy (вычисления)',
        'pandas': 'pandas (данные)',
    }
    
    missing_packages = []
    
    for package, description in required_packages.items():
        try:
            if package == 'tkinter':
                import tkinter
            elif package == 'matplotlib':
                import matplotlib
            elif package == 'numpy':
                import numpy
            elif package == 'pandas':
                import pandas
                
            print(f"✅ {description}")
            
        except ImportError:
            print(f"❌ {description} - НЕ установлен")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Отсутствующие зависимости: {', '.join(missing_packages)}")
        if 'tkinter' in missing_packages:
            print("💡 Для установки tkinter на Linux:")
            print("sudo apt-get install python3-tk")
        print("💡 Для установки остальных зависимостей:")
        print("pip install matplotlib numpy pandas")
        return False
    
    print("✅ Все зависимости установлены!")
    return True

def show_banner():
    """Показать баннер приложения"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════╗
║                        🚀 ATB Trading System v3.0                         ║
║                          Apple-style Dark Theme                          ║
║                                                                          ║
║  💰 Trading  📺 Live Demo  📊 Analytics  ⚙️ Settings  🔑 API Management  ║
║                                                                          ║
║         Интегрированная Windows платформа для трейдинга                  ║
║              с демонстрационным дашбордом для Twitch                     ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def main():
    """Главная функция запуска"""
    # Проверка Python версии
    if not check_python_version():
        input("Нажмите Enter для выхода...")
        return False
    
    # Показать баннер
    show_banner()
    
    # Проверка зависимостей
    if not check_dependencies():
        input("Нажмите Enter для выхода...")
        return False
    
    print("\n🔄 Инициализация приложения...")
    
    try:
        # Добавляем путь к проекту
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        # Импорт и запуск главного приложения
        from interfaces.desktop.dashboard_with_settings import MainApplication
        
        print("🎨 Загрузка Apple-style интерфейса...")
        print("📺 Подготовка Live Demo дашборда...")
        print("⚙️ Инициализация системы настроек...")
        
        app = MainApplication()
        
        print("\n🎯 Возможности приложения:")
        print("📺 Live Demo - демонстрационный дашборд для Twitch")
        print("💰 Торговля - основная торговая панель")
        print("📊 Аналитика - анализ производительности")
        print("⏮ Бэктест - тестирование стратегий")
        print("⚙️ Настройки - полное управление конфигурацией")
        print("🔑 API ключи - настройка подключений к биржам")
        
        print("\n🎮 Горячие клавиши:")
        print("Ctrl+, - Настройки")
        print("Ctrl+S - Сохранить конфигурацию")
        print("Ctrl+O - Загрузить конфигурацию")
        print("F5 - Обновить данные")
        print("F9 - Быстрый старт")
        print("F10 - Остановить торговлю")
        
        print("\n🚀 Запуск ATB Trading System...")
        print("💡 Для демонстрации на Twitch используйте вкладку '📺 Live Demo'")
        print("⚙️ Настройки API ключей доступны через меню 'Настройки'")
        
        app.run()
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("💡 Убедитесь, что все файлы проекта на месте")
        input("Нажмите Enter для выхода...")
        return False
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        print("🔧 Попробуйте перезапустить приложение")
        input("Нажмите Enter для выхода...")
        return False
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Приложение остановлено пользователем")
        print("💾 Настройки сохранены")
        print("🚀 Спасибо за использование ATB Trading System!")
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        input("Нажмите Enter для выхода...")
        sys.exit(1)