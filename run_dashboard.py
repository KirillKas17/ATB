#!/usr/bin/env python3
"""
Запуск ATB Trading Dashboard
Современный дашборд управления торговлей
"""

import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def check_python_version() -> bool:
    """Проверка версии Python"""
    if sys.version_info < (3, 8):
        print("❌ Требуется Python 3.8 или выше")
        print(f"Текущая версия: {sys.version}")
        return False
    return True

def check_dependencies() -> bool:
    """Проверка зависимостей"""
    required_packages: Dict[str, str] = {
        'tkinter': 'tkinter (встроен в Python)',
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'matplotlib': 'matplotlib',
        'decimal': 'decimal (встроен в Python)'
    }
    
    missing_packages: List[str] = []
    
    for package, install_name in required_packages.items():
        try:
            if package == 'tkinter':
                import tkinter
            elif package == 'numpy':
                from shared.numpy_utils import np
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
        print("\n❌ Отсутствующие зависимости:")
        for package in missing_packages:
            print(f"   pip install {package}")
        return False
    
    return True

def setup_environment() -> None:
    """Настройка окружения"""
    # Добавляем текущую директорию в Python path
    current_dir = Path(__file__).parent.absolute()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    print(f"📁 Рабочая директория: {current_dir}")
    
    # Создаем необходимые директории
    for directory in ['logs', 'data', 'temp']:
        dir_path: Path = current_dir / directory
        dir_path.mkdir(exist_ok=True)

def run_dashboard() -> None:
    """Запуск основного дашборда"""
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from datetime import datetime, timedelta
        import threading
        import queue
        import time
        
        print("🚀 Запуск ATB Trading Dashboard...")
        
        class TradingDashboard:
            def __init__(self) -> None:
                self.root: tk.Tk = tk.Tk()
                self.root.title("ATB Trading Dashboard")
                self.root.geometry("1200x800")
                self.root.configure(bg='#1e1e1e')
                
                # Очередь для межпоточного взаимодействия
                self.data_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
                self.running: bool = True
                
                # Данные
                self.market_data: List[Dict[str, Any]] = []
                self.positions: List[Dict[str, Any]] = []
                
                self.create_ui()
                self.start_data_simulation()
            
            def create_ui(self) -> None:
                """Создание пользовательского интерфейса"""
                # Стили
                style = ttk.Style()
                style.theme_use('default')
                
                # Конфигурация стилей
                style.configure('Title.TLabel', 
                              background='#1e1e1e', 
                              foreground='#ffffff',
                              font=('Helvetica', 16, 'bold'))
                
                style.configure('Heading.TLabel',
                              background='#1e1e1e',
                              foreground='#00ff88',
                              font=('Helvetica', 12, 'bold'))
                
                style.configure('Info.TLabel',
                              background='#1e1e1e',
                              foreground='#ffffff',
                              font=('Helvetica', 10))
                
                # Главный заголовок
                title_frame = tk.Frame(self.root, bg='#1e1e1e', height=60)
                title_frame.pack(fill='x', padx=10, pady=5)
                title_frame.pack_propagate(False)
                
                title_label = ttk.Label(title_frame, 
                                      text="🤖 ATB Trading Dashboard", 
                                      style='Title.TLabel')
                title_label.pack(side='left', pady=10)
                
                status_label = ttk.Label(title_frame, 
                                       text="🟢 ONLINE", 
                                       style='Heading.TLabel')
                status_label.pack(side='right', pady=10)
                
                # Создание вкладок
                notebook = ttk.Notebook(self.root)
                notebook.pack(fill='both', expand=True, padx=10, pady=5)
                
                # Вкладка аналитики
                analytics_frame = ttk.Frame(notebook)
                notebook.add(analytics_frame, text="📊 Аналитика")
                self.create_analytics_tab(analytics_frame)
                
                # Вкладка позиций
                positions_frame = ttk.Frame(notebook)
                notebook.add(positions_frame, text="💼 Позиции")
                self.create_positions_tab(positions_frame)
                
                # Вкладка настроек
                settings_frame = ttk.Frame(notebook)
                notebook.add(settings_frame, text="⚙️ Настройки")
                self.create_settings_tab(settings_frame)
            
            def create_analytics_tab(self, parent: ttk.Frame) -> None:
                """Создание вкладки аналитики"""
                # График цены
                self.create_price_chart(parent)
                
            def create_positions_tab(self, parent: ttk.Frame) -> None:
                """Создание вкладки позиций"""
                # Таблица позиций
                columns = ('Symbol', 'Side', 'Size', 'Entry Price', 'Current Price', 'PnL')
                tree = ttk.Treeview(parent, columns=columns, show='headings', height=10)
                
                for col in columns:
                    tree.heading(col, text=col)
                    tree.column(col, width=120)
                
                tree.pack(fill='both', expand=True, padx=10, pady=10)
                
                # Пример данных
                sample_positions = [
                    ('BTC/USDT', 'LONG', '0.1', '45000', '46500', '+150.00'),
                    ('ETH/USDT', 'SHORT', '1.0', '3200', '3150', '+50.00'),
                ]
                
                for position in sample_positions:
                    tree.insert('', 'end', values=position)
            
            def create_settings_tab(self, parent: ttk.Frame) -> None:
                """Создание вкладки настроек"""
                settings_label = ttk.Label(parent, text="Настройки системы", style='Heading.TLabel')
                settings_label.pack(pady=20)
                
                # Настройки риска
                risk_frame = ttk.LabelFrame(parent, text="Управление рисками", padding=10)
                risk_frame.pack(fill='x', padx=20, pady=10)
                
                ttk.Label(risk_frame, text="Максимальный риск на сделку (%):").pack(anchor='w')
                risk_scale = ttk.Scale(risk_frame, from_=0.1, to=5.0, orient='horizontal')
                risk_scale.set(2.0)
                risk_scale.pack(fill='x', pady=5)
                
            def create_price_chart(self, parent: ttk.Frame) -> None:
                """Создание графика цены"""
                # Создание matplotlib фигуры
                fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1e1e1e')
                ax.set_facecolor('#1e1e1e')
                
                # Генерация примерных данных
                dates = [datetime.now() - timedelta(hours=i) for i in range(100, 0, -1)]
                prices = np.cumsum(np.random.randn(100) * 0.01) + 100
                
                ax.plot(dates, prices, color='#00ff88', linewidth=2)
                ax.set_title('BTC/USDT Price Chart', color='white', fontsize=14, fontweight='bold')
                ax.set_xlabel('Time', color='white')
                ax.set_ylabel('Price (USDT)', color='white')
                ax.tick_params(colors='white')
                ax.grid(True, alpha=0.3)
                
                # Встраивание в tkinter
                canvas = FigureCanvasTkAgg(fig, parent)
                canvas.draw()
                canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
            
            def start_data_simulation(self) -> None:
                """Запуск симуляции данных"""
                def data_worker() -> None:
                    while self.running:
                        # Симуляция новых данных
                        new_data = {
                            'timestamp': datetime.now(),
                            'price': np.random.uniform(45000, 47000),
                            'volume': np.random.uniform(100, 1000)
                        }
                        self.data_queue.put(new_data)
                        time.sleep(1)
                
                # Запуск в отдельном потоке
                data_thread = threading.Thread(target=data_worker, daemon=True)
                data_thread.start()
            
            def run(self) -> None:
                """Запуск дашборда"""
                try:
                    self.root.mainloop()
                finally:
                    self.running = False
        
        # Создание и запуск дашборда
        dashboard = TradingDashboard()
        dashboard.run()
        
    except Exception as e:
        print(f"❌ Ошибка запуска дашборда: {e}")
        run_simple_dashboard()

def run_simple_dashboard() -> None:
    """Запуск простого дашборда без matplotlib"""
    print("🔄 Запуск упрощенной версии дашборда...")
    
    import tkinter as tk
    from tkinter import ttk
    
    class SimpleDashboard:
        def __init__(self) -> None:
            self.root: tk.Tk = tk.Tk()
            self.root.title("ATB Simple Dashboard")
            self.root.geometry("800x600")
            self.root.configure(bg='#2b2b2b')
            
            self.create_ui()
        
        def create_ui(self) -> None:
            """Создание простого интерфейса"""
            # Заголовок
            title = tk.Label(self.root, 
                           text="🤖 ATB Trading System", 
                           font=('Helvetica', 18, 'bold'),
                           bg='#2b2b2b', 
                           fg='#00ff88')
            title.pack(pady=20)
            
            # Статус
            status = tk.Label(self.root, 
                            text="✅ Система запущена", 
                            font=('Helvetica', 12),
                            bg='#2b2b2b', 
                            fg='white')
            status.pack(pady=10)
            
            # Кнопки управления
            button_frame = tk.Frame(self.root, bg='#2b2b2b')
            button_frame.pack(pady=20)
            
            tk.Button(button_frame, 
                     text="📊 Показать аналитику", 
                     command=self.show_analytics,
                     bg='#0066cc', 
                     fg='white',
                     font=('Helvetica', 10, 'bold'),
                     padx=20, 
                     pady=10).pack(side='left', padx=10)
            
            tk.Button(button_frame, 
                     text="⚙️ Настройки", 
                     command=self.show_settings,
                     bg='#666666', 
                     fg='white',
                     font=('Helvetica', 10, 'bold'),
                     padx=20, 
                     pady=10).pack(side='left', padx=10)
            
            tk.Button(button_frame, 
                     text="❓ Помощь", 
                     command=self.show_help,
                     bg='#009900', 
                     fg='white',
                     font=('Helvetica', 10, 'bold'),
                     padx=20, 
                     pady=10).pack(side='left', padx=10)
            
            # Текстовая область для логов
            log_frame = tk.LabelFrame(self.root, 
                                    text="Системные логи", 
                                    bg='#2b2b2b', 
                                    fg='white',
                                    font=('Helvetica', 10, 'bold'))
            log_frame.pack(fill='both', expand=True, padx=20, pady=20)
            
            self.log_text = tk.Text(log_frame, 
                                  bg='#1a1a1a', 
                                  fg='#00ff88',
                                  font=('Courier', 9),
                                  wrap='word')
            self.log_text.pack(fill='both', expand=True, padx=5, pady=5)
            
            # Добавляем начальные логи
            self.log_text.insert('end', "[СИСТЕМА] ATB Trading System запущена\n")
            self.log_text.insert('end', "[КОНФИГ] Загружена конфигурация по умолчанию\n")
            self.log_text.insert('end', "[СЕТЬ] Подключение к биржам: В ОЖИДАНИИ\n")
            self.log_text.insert('end', "[СТРАТЕГИИ] Загружено стратегий: 0\n")
        
        def show_analytics(self) -> None:
            """Показать аналитику"""
            self.log_text.insert('end', "[АНАЛИТИКА] Открыта панель аналитики\n")
            self.log_text.see('end')
        
        def show_settings(self) -> None:
            """Показать настройки"""
            self.log_text.insert('end', "[НАСТРОЙКИ] Открыта панель настроек\n")
            self.log_text.see('end')
        
        def show_help(self) -> None:
            """Показать помощь"""
            help_text = """
ATB Trading System - Автоматизированная торговая система

Основные функции:
- Автоматическая торговля на криптобиржах
- Анализ рыночных данных
- Управление рисками
- Мониторинг производительности

Для получения подробной документации см. файлы в папке docs/
            """
            self.log_text.insert('end', f"[ПОМОЩЬ] {help_text}\n")
            self.log_text.see('end')
        
        def run(self) -> None:
            """Запуск простого дашборда"""
            self.root.mainloop()
    
    dashboard = SimpleDashboard()
    dashboard.run()

def main() -> None:
    """Главная функция"""
    print("🤖 ATB Trading Dashboard")
    print("=" * 50)
    
    # Проверяем версию Python
    if not check_python_version():
        input("\nНажмите Enter для выхода...")
        return
    
    # Настраиваем окружение
    setup_environment()
    
    # Проверяем зависимости
    if not check_dependencies():
        print("\n⚠️ Запуск в режиме ограниченной функциональности...")
        input("Нажмите Enter для продолжения...")
    
    # Запускаем дашборд
    try:
        run_dashboard()
    except KeyboardInterrupt:
        print("\n👋 Завершение работы...")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        input("Нажмите Enter для выхода...")

if __name__ == "__main__":
    main()