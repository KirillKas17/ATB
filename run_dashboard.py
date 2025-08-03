#!/usr/bin/env python3
"""
Запуск ATB Trading Dashboard
Современный дашборд управления торговлей
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
    """Проверка зависимостей"""
    required_packages = {
        'tkinter': 'tkinter (встроен в Python)',
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'matplotlib': 'matplotlib',
        'decimal': 'decimal (встроен в Python)'
    }
    
    missing_packages = []
    
    for package, install_name in required_packages.items():
        try:
            if package == 'tkinter':
                import tkinter
            elif package == 'numpy':
                import numpy
            elif package == 'pandas':
                import pandas
            elif package == 'matplotlib':
                import matplotlib
            elif package == 'decimal':
                import decimal
                
            print(f"✅ {package} - установлен")
            
        except ImportError:
            print(f"❌ {package} - НЕ установлен")
            missing_packages.append(install_name)
    
    if missing_packages:
        print(f"\n🔧 Для установки недостающих пакетов выполните:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def setup_environment():
    """Настройка окружения"""
    # Добавление пути к проекту
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Установка переменных окружения
    os.environ['PYTHONPATH'] = str(project_root)
    
    return True

def run_dashboard():
    """Запуск дашборда"""
    try:
        print("🚀 Запуск ATB Trading Dashboard...")
        
        # Проверяем, какую версию запускать
        dashboard_type = "integrated"  # integrated, basic, или simple
        
        if dashboard_type == "integrated":
            # Попытка запуска интегрированного дашборда
            try:
                from interfaces.desktop.integrated_dashboard import IntegratedTradingDashboard
                print("🎯 Запуск интегрированного дашборда...")
                dashboard = IntegratedTradingDashboard()
                dashboard.run()
                
            except ImportError as e:
                print(f"⚠️ Не удалось загрузить интегрированный дашборд: {e}")
                print("🔄 Переключение на базовую версию...")
                dashboard_type = "basic"
        
        if dashboard_type == "basic":
            # Запуск базового дашборда
            try:
                from interfaces.desktop.trading_dashboard import ModernTradingDashboard
                print("📊 Запуск базового дашборда...")
                dashboard = ModernTradingDashboard()
                dashboard.run()
                
            except ImportError as e:
                print(f"⚠️ Не удалось загрузить базовый дашборд: {e}")
                print("🔄 Переключение на простую версию...")
                dashboard_type = "simple"
        
        if dashboard_type == "simple":
            # Запуск простой версии
            run_simple_dashboard()
    
    except KeyboardInterrupt:
        print("\n⏹️ Дашборд остановлен пользователем")
    except Exception as e:
        print(f"❌ Ошибка запуска дашборда: {e}")
        import traceback
        traceback.print_exc()

def run_simple_dashboard():
    """Запуск упрощенной версии дашборда"""
    import tkinter as tk
    from tkinter import ttk, messagebox
    
    class SimpleDashboard:
        def __init__(self):
            self.root = tk.Tk()
            self.root.title("ATB Trading Dashboard - Простая версия")
            self.root.geometry("800x600")
            self.root.configure(bg='#2d2d2d')
            
            self.create_ui()
        
        def create_ui(self):
            # Заголовок
            title_label = tk.Label(self.root, text="⚡ ATB Trading Dashboard",
                                 font=('Arial', 20, 'bold'),
                                 fg='#3742fa', bg='#2d2d2d')
            title_label.pack(pady=20)
            
            # Информационное сообщение
            info_text = """
Добро пожаловать в ATB Trading Dashboard!

Это упрощенная версия дашборда.
Для полной функциональности необходимы дополнительные компоненты.

Основные возможности:
• Мониторинг торговых пар
• Базовая аналитика  
• Управление настройками
• Симуляция торговли

Для получения полной версии убедитесь, что установлены
все зависимости проекта.
            """
            
            info_label = tk.Label(self.root, text=info_text.strip(),
                                font=('Arial', 11), fg='white', bg='#2d2d2d',
                                justify='left')
            info_label.pack(pady=20, padx=20)
            
            # Кнопки
            button_frame = tk.Frame(self.root, bg='#2d2d2d')
            button_frame.pack(pady=20)
            
            tk.Button(button_frame, text="📊 Тестовая аналитика",
                     bg='#3742fa', fg='white', font=('Arial', 12),
                     command=self.show_analytics).pack(side='left', padx=10)
            
            tk.Button(button_frame, text="⚙️ Настройки", 
                     bg='#2d2d2d', fg='white', font=('Arial', 12),
                     command=self.show_settings).pack(side='left', padx=10)
            
            tk.Button(button_frame, text="❓ Справка",
                     bg='#2d2d2d', fg='white', font=('Arial', 12), 
                     command=self.show_help).pack(side='left', padx=10)
            
            # Статус
            status_label = tk.Label(self.root, text="Статус: Готов к работе",
                                  font=('Arial', 10), fg='#00ff88', bg='#2d2d2d')
            status_label.pack(side='bottom', pady=10)
        
        def show_analytics(self):
            messagebox.showinfo("Аналитика", 
                              "Функция аналитики будет доступна в полной версии.\n"
                              "Установите все зависимости для получения полного функционала.")
        
        def show_settings(self):
            messagebox.showinfo("Настройки",
                              "Настройки будут доступны в полной версии.\n"
                              "Текущая версия работает с базовыми параметрами.")
        
        def show_help(self):
            help_text = """
ATB Trading Dashboard - Справка

Горячие клавиши:
F1 - Справка
F5 - Обновить
Ctrl+Q - Выход

Для полной функциональности установите:
pip install numpy pandas matplotlib

Поддержка: support@atb-trading.com
            """
            messagebox.showinfo("Справка", help_text.strip())
        
        def run(self):
            self.root.mainloop()
    
    print("📱 Запуск простой версии дашборда...")
    dashboard = SimpleDashboard()
    dashboard.run()

def main():
    """Главная функция"""
    print("=" * 60)
    print("🎯 ATB Trading Dashboard - Система запуска")
    print("=" * 60)
    
    # Проверка версии Python
    if not check_python_version():
        input("Нажмите Enter для выхода...")
        return
    
    print("\n🔍 Проверка зависимостей...")
    dependencies_ok = check_dependencies()
    
    print("\n⚙️ Настройка окружения...")
    if not setup_environment():
        print("❌ Ошибка настройки окружения")
        return
    
    if dependencies_ok:
        print("✅ Все зависимости установлены")
    else:
        print("⚠️ Некоторые зависимости отсутствуют")
        print("Будет запущена упрощенная версия")
    
    print("\n" + "=" * 60)
    
    # Запуск дашборда
    run_dashboard()

if __name__ == "__main__":
    main()