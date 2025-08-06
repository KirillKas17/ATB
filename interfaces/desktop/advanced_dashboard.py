"""
ATB Trading Dashboard v2.0 - Продвинутый интерфейс
Современный дашборд с расширенной функциональностью
"""

import sys
import asyncio
import threading
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import tkinter.font as tkFont

# Добавляем путь к проекту
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    ADVANCED_FEATURES = True
except ImportError:
    ADVANCED_FEATURES = False

# Импорт компонентов системы
from interfaces.desktop.dashboard_controller import DashboardController

class AdvancedTradingDashboard:
    """Продвинутый торговый дашборд с полным функционалом"""
    
    def __init__(self):
        # Настройка логирования
        self.setup_logging()
        
        # Инициализация контроллера
        self.controller = DashboardController()
        
        # Состояние приложения
        self.is_trading_active = False
        self.current_session = None
        self.selected_pairs = []
        self.active_strategies = []
        
        # Данные для графиков
        self.price_data = {}
        self.performance_data = []
        
        # Настройки интерфейса
        self.theme = "dark"
        self.update_interval = 1000  # ms
        
        # Создание основного окна
        self.create_main_window()
        
        # Создание интерфейса
        self.create_interface()
        
        # Настройка стилей
        self.setup_styles()
        
        # Настройка горячих клавиш
        self.setup_hotkeys()
        
        # Запуск циклов обновления
        self.start_update_cycles()
        
        self.logger.info("Продвинутый дашборд инициализирован")
    
    def setup_logging(self):
        """Настройка логирования"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "dashboard.log", encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("AdvancedDashboard")
    
    def create_main_window(self):
        """Создание основного окна"""
        self.root = tk.Tk()
        self.root.title("⚡ ATB Trading Dashboard v2.0 - Advanced Edition")
        self.root.geometry("1600x1000")
        self.root.minsize(1200, 800)
        
        # Иконка и настройки окна
        try:
            self.root.iconbitmap(default="assets/icon.ico")
        except:
            pass
        
        # Настройка темной темы
        self.root.configure(bg='#1e1e1e')
        
        # Обработка закрытия
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_interface(self):
        """Создание интерфейса"""
        # Главное меню
        self.create_menu_bar()
        
        # Панель инструментов
        self.create_toolbar()
        
        # Основная область с вкладками
        self.create_main_notebook()
        
        # Статусная строка
        self.create_status_bar()
        
        # Панель уведомлений
        self.create_notifications_panel()
    
    def create_menu_bar(self):
        """Создание строки меню"""
        menubar = tk.Menu(self.root, bg='#2d2d2d', fg='white', 
                         activebackground='#3742fa', activeforeground='white')
        self.root.config(menu=menubar)
        
        # Файл
        file_menu = tk.Menu(menubar, tearoff=0, bg='#2d2d2d', fg='white')
        menubar.add_cascade(label="📁 Файл", menu=file_menu)
        file_menu.add_command(label="🆕 Новая сессия", 
                             command=self.new_session, accelerator="Ctrl+N")
        file_menu.add_command(label="📂 Загрузить конфигурацию", 
                             command=self.load_config, accelerator="Ctrl+O")
        file_menu.add_command(label="💾 Сохранить конфигурацию", 
                             command=self.save_config, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="📤 Экспорт данных", command=self.export_data)
        file_menu.add_command(label="📥 Импорт стратегий", command=self.import_strategies)
        file_menu.add_separator()
        file_menu.add_command(label="🚪 Выход", command=self.root.quit, accelerator="Alt+F4")
        
        # Торговля
        trading_menu = tk.Menu(menubar, tearoff=0, bg='#2d2d2d', fg='white')
        menubar.add_cascade(label="💼 Торговля", menu=trading_menu)
        trading_menu.add_command(label="🚀 Запустить торговлю", 
                               command=self.start_trading, accelerator="F9")
        trading_menu.add_command(label="⏹️ Остановить торговлю", 
                               command=self.stop_trading, accelerator="F10")
        trading_menu.add_command(label="⚡ Быстрый старт", command=self.quick_start)
        trading_menu.add_separator()
        trading_menu.add_command(label="⚙️ Настройки стратегий", 
                               command=self.strategy_settings)
        trading_menu.add_command(label="🛡️ Управление рисками", 
                               command=self.risk_settings)
        trading_menu.add_separator()
        trading_menu.add_command(label="❌ Закрыть все позиции", 
                               command=self.close_all_positions)
        trading_menu.add_command(label="⛔ Отменить все ордера", 
                               command=self.cancel_all_orders)
        
        # Аналитика
        analytics_menu = tk.Menu(menubar, tearoff=0, bg='#2d2d2d', fg='white')
        menubar.add_cascade(label="📊 Аналитика", menu=analytics_menu)
        analytics_menu.add_command(label="📈 Отчет производительности", 
                                  command=self.performance_report)
        analytics_menu.add_command(label="⚠️ Анализ рисков", command=self.risk_analysis)
        analytics_menu.add_command(label="🎯 Оптимизация стратегий", 
                                  command=self.strategy_optimization)
        analytics_menu.add_separator()
        analytics_menu.add_command(label="📋 Экспорт в Excel", command=self.export_excel)
        analytics_menu.add_command(label="📄 PDF отчет", command=self.generate_pdf_report)
        
        # Инструменты
        tools_menu = tk.Menu(menubar, tearoff=0, bg='#2d2d2d', fg='white')
        menubar.add_cascade(label="🛠️ Инструменты", menu=tools_menu)
        tools_menu.add_command(label="🧮 Калькулятор позиций", 
                              command=self.position_calculator)
        tools_menu.add_command(label="💱 Конвертер валют", 
                              command=self.currency_converter)
        tools_menu.add_command(label="📅 Экономический календарь", 
                              command=self.economic_calendar)
        tools_menu.add_separator()
        tools_menu.add_command(label="⚙️ Настройки", command=self.open_settings)
        
        # Помощь
        help_menu = tk.Menu(menubar, tearoff=0, bg='#2d2d2d', fg='white')
        menubar.add_cascade(label="❓ Помощь", menu=help_menu)
        help_menu.add_command(label="📚 Справка", command=self.show_help, accelerator="F1")
        help_menu.add_command(label="🔥 Горячие клавиши", command=self.show_hotkeys)
        help_menu.add_command(label="🆔 О программе", command=self.show_about)
    
    def create_toolbar(self):
        """Создание панели инструментов"""
        toolbar_frame = tk.Frame(self.root, bg='#2d2d2d', height=60)
        toolbar_frame.pack(fill='x', pady=(0, 5))
        toolbar_frame.pack_propagate(False)
        
        # Левая группа - управление торговлей
        left_group = tk.Frame(toolbar_frame, bg='#2d2d2d')
        left_group.pack(side='left', padx=10, pady=5)
        
        # Кнопка запуска/остановки
        self.trade_button = tk.Button(left_group, text="▶ Запустить", 
                                     bg='#00ff88', fg='black', font=('Arial', 12, 'bold'),
                                     command=self.toggle_trading, width=12)
        self.trade_button.pack(side='left', padx=5)
        
        # Кнопка экстренной остановки
        emergency_button = tk.Button(left_group, text="🛑 СТОП", 
                                   bg='#ff4757', fg='white', font=('Arial', 10, 'bold'),
                                   command=self.emergency_stop, width=8)
        emergency_button.pack(side='left', padx=5)
        
        # Выбор режима
        tk.Label(left_group, text="Режим:", bg='#2d2d2d', fg='white', 
                font=('Arial', 10)).pack(side='left', padx=(20, 5))
        
        self.mode_var = tk.StringVar(value="simulation")
        mode_combo = ttk.Combobox(left_group, textvariable=self.mode_var, 
                                 values=["simulation", "live", "backtest"], 
                                 state="readonly", width=12)
        mode_combo.pack(side='left', padx=5)
        
        # Центральная группа - статистика
        center_group = tk.Frame(toolbar_frame, bg='#2d2d2d')
        center_group.pack(side='left', expand=True, padx=20)
        
        # Статистика в реальном времени
        stats_frame = tk.Frame(center_group, bg='#2d2d2d')
        stats_frame.pack(fill='x')
        
        # Баланс
        self.balance_label = tk.Label(stats_frame, text="💰 Баланс: $10,000.00", 
                                     bg='#2d2d2d', fg='#00ff88', font=('Arial', 12, 'bold'))
        self.balance_label.pack(side='left', padx=10)
        
        # P&L
        self.pnl_label = tk.Label(stats_frame, text="📊 P&L: +$0.00 (0.00%)", 
                                 bg='#2d2d2d', fg='#00ff88', font=('Arial', 12, 'bold'))
        self.pnl_label.pack(side='left', padx=10)
        
        # Активные позиции
        self.positions_label = tk.Label(stats_frame, text="📈 Позиции: 0", 
                                       bg='#2d2d2d', fg='white', font=('Arial', 11))
        self.positions_label.pack(side='left', padx=10)
        
        # Активные ордера
        self.orders_label = tk.Label(stats_frame, text="📋 Ордера: 0", 
                                    bg='#2d2d2d', fg='white', font=('Arial', 11))
        self.orders_label.pack(side='left', padx=10)
        
        # Правая группа - быстрые действия
        right_group = tk.Frame(toolbar_frame, bg='#2d2d2d')
        right_group.pack(side='right', padx=10, pady=5)
        
        # Обновить данные
        refresh_button = tk.Button(right_group, text="🔄", bg='#3742fa', fg='white', 
                                  font=('Arial', 12), command=self.refresh_data, width=3)
        refresh_button.pack(side='right', padx=2)
        
        # Настройки
        settings_button = tk.Button(right_group, text="⚙️", bg='#2d2d2d', fg='white', 
                                   font=('Arial', 12), command=self.open_settings, width=3)
        settings_button.pack(side='right', padx=2)
        
        # Полноэкранный режим
        fullscreen_button = tk.Button(right_group, text="🔳", bg='#2d2d2d', fg='white', 
                                     font=('Arial', 12), command=self.toggle_fullscreen, width=3)
        fullscreen_button.pack(side='right', padx=2)
    
    def create_main_notebook(self):
        """Создание основной области с вкладками"""
        # Создание notebook с кастомными стилями
        style = ttk.Style()
        style.theme_use('clam')
        
        # Настройка стилей для темной темы
        style.configure('TNotebook', background='#1e1e1e', borderwidth=0)
        style.configure('TNotebook.Tab', 
                       background='#2d2d2d', foreground='white', 
                       padding=[12, 8], font=('Arial', 10, 'bold'))
        style.map('TNotebook.Tab',
                 background=[('selected', '#3742fa'), ('active', '#454545')])
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Создание вкладок
        self.create_main_tab()
        self.create_analytics_tab()
        self.create_backtest_tab()
        self.create_positions_tab()
        self.create_settings_tab()
    
    def create_main_tab(self):
        """Создание главной вкладки"""
        main_frame = tk.Frame(self.notebook, bg='#1e1e1e')
        self.notebook.add(main_frame, text='📊 Основное')
        
        # Разделение на панели
        main_paned = tk.PanedWindow(main_frame, orient='horizontal', 
                                   bg='#1e1e1e', sashwidth=5, relief='flat')
        main_paned.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Левая панель - управление
        self.create_control_panel(main_paned)
        
        # Центральная панель - графики
        self.create_charts_panel(main_paned)
        
        # Правая панель - информация
        self.create_info_panel(main_paned)
    
    def create_control_panel(self, parent):
        """Создание панели управления"""
        control_frame = tk.Frame(parent, bg='#1e1e1e', width=300)
        parent.add(control_frame)
        
        # Заголовок
        title_label = tk.Label(control_frame, text="🎛️ Панель управления", 
                              bg='#1e1e1e', fg='#3742fa', font=('Arial', 14, 'bold'))
        title_label.pack(pady=(10, 20))
        
        # Торговые пары
        self.create_pairs_section(control_frame)
        
        # Стратегии
        self.create_strategies_section(control_frame)
        
        # Управление рисками
        self.create_risk_section(control_frame)
        
        # Новости и сигналы
        self.create_news_section(control_frame)
    
    def create_pairs_section(self, parent):
        """Секция торговых пар"""
        pairs_frame = tk.LabelFrame(parent, text="💱 Торговые пары", 
                                   bg='#1e1e1e', fg='white', font=('Arial', 11, 'bold'),
                                   borderwidth=2, relief='groove')
        pairs_frame.pack(fill='x', padx=10, pady=5)
        
        # Поиск пар
        search_frame = tk.Frame(pairs_frame, bg='#1e1e1e')
        search_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Label(search_frame, text="🔍", bg='#1e1e1e', fg='white').pack(side='left')
        self.pairs_search = tk.Entry(search_frame, bg='#2d2d2d', fg='white', 
                                    insertbackground='white', font=('Arial', 10))
        self.pairs_search.pack(side='left', fill='x', expand=True, padx=(5, 0))
        self.pairs_search.bind('<KeyRelease>', self.filter_pairs)
        
        # Список пар
        pairs_list_frame = tk.Frame(pairs_frame, bg='#1e1e1e')
        pairs_list_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Scrollbar для списка
        pairs_scrollbar = tk.Scrollbar(pairs_list_frame)
        pairs_scrollbar.pack(side='right', fill='y')
        
        self.pairs_listbox = tk.Listbox(pairs_list_frame, 
                                       yscrollcommand=pairs_scrollbar.set,
                                       bg='#2d2d2d', fg='white', 
                                       selectbackground='#3742fa',
                                       font=('Arial', 10), height=8,
                                       selectmode='multiple')
        self.pairs_listbox.pack(side='left', fill='both', expand=True)
        pairs_scrollbar.config(command=self.pairs_listbox.yview)
        
        # Заполнение списка пар
        self.populate_pairs_list()
        
        # Кнопки управления парами
        pairs_buttons = tk.Frame(pairs_frame, bg='#1e1e1e')
        pairs_buttons.pack(fill='x', padx=5, pady=5)
        
        tk.Button(pairs_buttons, text="✅ Выбрать все", bg='#00ff88', fg='black',
                 command=self.select_all_pairs, font=('Arial', 9)).pack(side='left', padx=2)
        tk.Button(pairs_buttons, text="❌ Очистить", bg='#ff4757', fg='white',
                 command=self.clear_pairs_selection, font=('Arial', 9)).pack(side='left', padx=2)
    
    def create_strategies_section(self, parent):
        """Секция стратегий"""
        strategies_frame = tk.LabelFrame(parent, text="🧠 Торговые стратегии", 
                                        bg='#1e1e1e', fg='white', font=('Arial', 11, 'bold'),
                                        borderwidth=2, relief='groove')
        strategies_frame.pack(fill='x', padx=10, pady=5)
        
        # Список стратегий с чекбоксами
        self.strategy_vars = {}
        strategies = [
            ("RSI Bounce", "Торговля на отскоках RSI", "medium"),
            ("MACD Cross", "Пересечения MACD", "medium"),
            ("Bollinger Squeeze", "Сжатие полос Боллинджера", "high"),
            ("Mean Reversion", "Возврат к среднему", "low"),
            ("Momentum", "Следование тренду", "high"),
            ("Grid Trading", "Сеточная торговля", "medium")
        ]
        
        for strategy, description, risk_level in strategies:
            strategy_frame = tk.Frame(strategies_frame, bg='#1e1e1e')
            strategy_frame.pack(fill='x', padx=5, pady=2)
            
            var = tk.BooleanVar()
            self.strategy_vars[strategy] = var
            
            checkbox = tk.Checkbutton(strategy_frame, text=strategy, variable=var,
                                     bg='#1e1e1e', fg='white', selectcolor='#2d2d2d',
                                     font=('Arial', 10), anchor='w')
            checkbox.pack(side='left')
            
            # Индикатор риска
            risk_colors = {"low": "#00ff88", "medium": "#ffa500", "high": "#ff4757"}
            risk_label = tk.Label(strategy_frame, text=f"({risk_level})",
                                 bg='#1e1e1e', fg=risk_colors[risk_level], 
                                 font=('Arial', 8))
            risk_label.pack(side='right')
        
        # Кнопка настроек стратегий
        strategy_settings_btn = tk.Button(strategies_frame, text="⚙️ Настроить параметры",
                                         bg='#3742fa', fg='white', 
                                         command=self.strategy_settings)
        strategy_settings_btn.pack(pady=5)
    
    def create_risk_section(self, parent):
        """Секция управления рисками"""
        risk_frame = tk.LabelFrame(parent, text="🛡️ Управление рисками", 
                                  bg='#1e1e1e', fg='white', font=('Arial', 11, 'bold'),
                                  borderwidth=2, relief='groove')
        risk_frame.pack(fill='x', padx=10, pady=5)
        
        # Размер позиции
        position_frame = tk.Frame(risk_frame, bg='#1e1e1e')
        position_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Label(position_frame, text="💰 Размер позиции:", 
                bg='#1e1e1e', fg='white', font=('Arial', 10)).pack(anchor='w')
        
        self.position_size_var = tk.StringVar(value="2.0")
        position_scale = tk.Scale(position_frame, from_=0.1, to=10.0, resolution=0.1,
                                 variable=self.position_size_var, orient='horizontal',
                                 bg='#2d2d2d', fg='white', highlightthickness=0,
                                 troughcolor='#3742fa', activebackground='#454545')
        position_scale.pack(fill='x', pady=2)
        
        position_label = tk.Label(position_frame, text="", bg='#1e1e1e', fg='#00ff88')
        position_label.pack()
        
        def update_position_label(val):
            percentage = float(val)
            if percentage <= 1.0:
                risk_level = "Низкий"
                color = "#00ff88"
            elif percentage <= 3.0:
                risk_level = "Средний"
                color = "#ffa500"
            else:
                risk_level = "Высокий"
                color = "#ff4757"
            
            position_label.config(text=f"{percentage}% от депозита ({risk_level})", fg=color)
        
        position_scale.config(command=update_position_label)
        update_position_label("2.0")
        
        # Стоп-лосс
        sl_frame = tk.Frame(risk_frame, bg='#1e1e1e')
        sl_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Label(sl_frame, text="🛑 Стоп-лосс:", 
                bg='#1e1e1e', fg='white', font=('Arial', 10)).pack(anchor='w')
        
        self.stop_loss_var = tk.StringVar(value="2.0")
        sl_scale = tk.Scale(sl_frame, from_=0.5, to=10.0, resolution=0.1,
                           variable=self.stop_loss_var, orient='horizontal',
                           bg='#2d2d2d', fg='white', highlightthickness=0,
                           troughcolor='#ff4757', activebackground='#454545')
        sl_scale.pack(fill='x', pady=2)
        
        sl_label = tk.Label(sl_frame, text="2.0% от входа", bg='#1e1e1e', fg='#ff4757')
        sl_label.pack()
        
        def update_sl_label(val):
            sl_label.config(text=f"{float(val)}% от входа")
        
        sl_scale.config(command=update_sl_label)
        
        # Тейк-профит
        tp_frame = tk.Frame(risk_frame, bg='#1e1e1e')
        tp_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Label(tp_frame, text="🎯 Тейк-профит:", 
                bg='#1e1e1e', fg='white', font=('Arial', 10)).pack(anchor='w')
        
        self.take_profit_var = tk.StringVar(value="4.0")
        tp_scale = tk.Scale(tp_frame, from_=1.0, to=20.0, resolution=0.5,
                           variable=self.take_profit_var, orient='horizontal',
                           bg='#2d2d2d', fg='white', highlightthickness=0,
                           troughcolor='#00ff88', activebackground='#454545')
        tp_scale.pack(fill='x', pady=2)
        
        tp_label = tk.Label(tp_frame, text="4.0% от входа", bg='#1e1e1e', fg='#00ff88')
        tp_label.pack()
        
        def update_tp_label(val):
            tp_label.config(text=f"{float(val)}% от входа")
        
        tp_scale.config(command=update_tp_label)
    
    def create_news_section(self, parent):
        """Секция новостей и сигналов"""
        news_frame = tk.LabelFrame(parent, text="📰 Новости и сигналы", 
                                  bg='#1e1e1e', fg='white', font=('Arial', 11, 'bold'),
                                  borderwidth=2, relief='groove')
        news_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Новости
        news_text = tk.Text(news_frame, bg='#2d2d2d', fg='white', 
                           font=('Arial', 9), height=6, wrap='word',
                           insertbackground='white')
        news_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Заполнение примерами новостей
        sample_news = [
            "🔴 BTC: Прорыв уровня сопротивления $45,000",
            "🟡 ETH: Консолидация в диапазоне $3,000-$3,200",
            "🟢 Market: Увеличение объемов торгов на 15%",
            "🔵 Signal: RSI oversold на BTC/USDT 1H",
            "⚪ News: Fed meeting результаты в 16:00 UTC"
        ]
        
        for news in sample_news:
            news_text.insert('end', f"{datetime.now().strftime('%H:%M')} {news}\n")
        
        news_text.config(state='disabled')
    
    def create_charts_panel(self, parent):
        """Создание панели графиков"""
        charts_frame = tk.Frame(parent, bg='#1e1e1e')
        parent.add(charts_frame)
        
        if ADVANCED_FEATURES:
            self.create_advanced_charts(charts_frame)
        else:
            self.create_simple_charts(charts_frame)
    
    def create_advanced_charts(self, parent):
        """Создание продвинутых графиков с matplotlib"""
        # Notebook для разных типов графиков
        charts_notebook = ttk.Notebook(parent)
        charts_notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # График цены
        self.create_price_chart(charts_notebook)
        
        # График P&L
        self.create_pnl_chart(charts_notebook)
        
        # Распределения
        self.create_distribution_chart(charts_notebook)
    
    def create_price_chart(self, parent):
        """График цены"""
        price_frame = tk.Frame(parent, bg='#1e1e1e')
        parent.add(price_frame, text='📈 Цена')
        
        # Создание matplotlib фигуры
        self.price_fig = Figure(figsize=(12, 6), facecolor='#1e1e1e')
        self.price_ax = self.price_fig.add_subplot(111, facecolor='#2d2d2d')
        
        # Настройка стиля
        self.price_ax.tick_params(colors='white')
        self.price_ax.spines['bottom'].set_color('white')
        self.price_ax.spines['top'].set_color('white')
        self.price_ax.spines['right'].set_color('white')
        self.price_ax.spines['left'].set_color('white')
        
        # Canvas для отображения
        self.price_canvas = FigureCanvasTkAgg(self.price_fig, price_frame)
        self.price_canvas.draw()
        self.price_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Инициализация данных
        self.update_price_chart()
    
    def create_pnl_chart(self, parent):
        """График P&L"""
        pnl_frame = tk.Frame(parent, bg='#1e1e1e')
        parent.add(pnl_frame, text='💰 P&L')
        
        self.pnl_fig = Figure(figsize=(12, 6), facecolor='#1e1e1e')
        self.pnl_ax = self.pnl_fig.add_subplot(111, facecolor='#2d2d2d')
        
        # Настройка стиля
        self.pnl_ax.tick_params(colors='white')
        self.pnl_ax.spines['bottom'].set_color('white')
        self.pnl_ax.spines['top'].set_color('white')
        self.pnl_ax.spines['right'].set_color('white')
        self.pnl_ax.spines['left'].set_color('white')
        
        self.pnl_canvas = FigureCanvasTkAgg(self.pnl_fig, pnl_frame)
        self.pnl_canvas.draw()
        self.pnl_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.update_pnl_chart()
    
    def create_distribution_chart(self, parent):
        """График распределений"""
        dist_frame = tk.Frame(parent, bg='#1e1e1e')
        parent.add(dist_frame, text='📊 Аналитика')
        
        self.dist_fig = Figure(figsize=(12, 6), facecolor='#1e1e1e')
        self.dist_ax = self.dist_fig.add_subplot(111, facecolor='#2d2d2d')
        
        # Настройка стиля
        self.dist_ax.tick_params(colors='white')
        self.dist_ax.spines['bottom'].set_color('white')
        self.dist_ax.spines['top'].set_color('white')
        self.dist_ax.spines['right'].set_color('white')
        self.dist_ax.spines['left'].set_color('white')
        
        self.dist_canvas = FigureCanvasTkAgg(self.dist_fig, dist_frame)
        self.dist_canvas.draw()
        self.dist_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.update_distribution_chart()
    
    def create_simple_charts(self, parent):
        """Простые графики без matplotlib"""
        simple_frame = tk.Frame(parent, bg='#1e1e1e')
        simple_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        tk.Label(simple_frame, 
                text="📊 ГРАФИК ЦЕНЫ\n\nДля полнофункциональных графиков\nустановите matplotlib:\n\npip install matplotlib",
                bg='#1e1e1e', fg='white', font=('Arial', 14),
                justify='center').pack(expand=True)
    
    def create_info_panel(self, parent):
        """Создание информационной панели"""
        info_frame = tk.Frame(parent, bg='#1e1e1e', width=300)
        parent.add(info_frame)
        
        # Заголовок
        title_label = tk.Label(info_frame, text="📋 Информация", 
                              bg='#1e1e1e', fg='#3742fa', font=('Arial', 14, 'bold'))
        title_label.pack(pady=(10, 20))
        
        # Позиции
        self.create_positions_section(info_frame)
        
        # Ордера
        self.create_orders_section(info_frame)
        
        # Производительность
        self.create_performance_section(info_frame)
    
    def create_positions_section(self, parent):
        """Секция позиций"""
        pos_frame = tk.LabelFrame(parent, text="📈 Открытые позиции", 
                                 bg='#1e1e1e', fg='white', font=('Arial', 11, 'bold'))
        pos_frame.pack(fill='x', padx=10, pady=5)
        
        # Таблица позиций
        pos_tree_frame = tk.Frame(pos_frame, bg='#1e1e1e')
        pos_tree_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        pos_scrollbar = tk.Scrollbar(pos_tree_frame)
        pos_scrollbar.pack(side='right', fill='y')
        
        self.positions_tree = ttk.Treeview(pos_tree_frame, 
                                          yscrollcommand=pos_scrollbar.set,
                                          columns=('Symbol', 'Side', 'Size', 'PnL'),
                                          show='headings', height=6)
        
        self.positions_tree.heading('Symbol', text='Пара')
        self.positions_tree.heading('Side', text='Сторона')
        self.positions_tree.heading('Size', text='Размер')
        self.positions_tree.heading('PnL', text='P&L')
        
        self.positions_tree.column('Symbol', width=80)
        self.positions_tree.column('Side', width=60)
        self.positions_tree.column('Size', width=80)
        self.positions_tree.column('PnL', width=80)
        
        self.positions_tree.pack(side='left', fill='both', expand=True)
        pos_scrollbar.config(command=self.positions_tree.yview)
    
    def create_orders_section(self, parent):
        """Секция ордеров"""
        orders_frame = tk.LabelFrame(parent, text="📋 Активные ордера", 
                                    bg='#1e1e1e', fg='white', font=('Arial', 11, 'bold'))
        orders_frame.pack(fill='x', padx=10, pady=5)
        
        # Таблица ордеров
        orders_tree_frame = tk.Frame(orders_frame, bg='#1e1e1e')
        orders_tree_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        orders_scrollbar = tk.Scrollbar(orders_tree_frame)
        orders_scrollbar.pack(side='right', fill='y')
        
        self.orders_tree = ttk.Treeview(orders_tree_frame, 
                                       yscrollcommand=orders_scrollbar.set,
                                       columns=('Symbol', 'Type', 'Side', 'Amount'),
                                       show='headings', height=6)
        
        self.orders_tree.heading('Symbol', text='Пара')
        self.orders_tree.heading('Type', text='Тип')
        self.orders_tree.heading('Side', text='Сторона')
        self.orders_tree.heading('Amount', text='Количество')
        
        self.orders_tree.column('Symbol', width=80)
        self.orders_tree.column('Type', width=60)
        self.orders_tree.column('Side', width=60)
        self.orders_tree.column('Amount', width=80)
        
        self.orders_tree.pack(side='left', fill='both', expand=True)
        orders_scrollbar.config(command=self.orders_tree.yview)
    
    def create_performance_section(self, parent):
        """Секция производительности"""
        perf_frame = tk.LabelFrame(parent, text="📊 Производительность", 
                                  bg='#1e1e1e', fg='white', font=('Arial', 11, 'bold'))
        perf_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Метрики производительности
        metrics_frame = tk.Frame(perf_frame, bg='#1e1e1e')
        metrics_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Создание меток для метрик
        self.metrics_labels = {}
        metrics = [
            ('Total P&L', '$0.00', '#00ff88'),
            ('Win Rate', '0%', '#ffa500'),
            ('Sharpe Ratio', '0.00', '#3742fa'),
            ('Max Drawdown', '0%', '#ff4757'),
            ('Total Trades', '0', 'white')
        ]
        
        for i, (name, value, color) in enumerate(metrics):
            metric_frame = tk.Frame(metrics_frame, bg='#1e1e1e')
            metric_frame.pack(fill='x', pady=2)
            
            label = tk.Label(metric_frame, text=f"{name}:", 
                           bg='#1e1e1e', fg='white', font=('Arial', 10))
            label.pack(side='left')
            
            value_label = tk.Label(metric_frame, text=value, 
                                 bg='#1e1e1e', fg=color, font=('Arial', 10, 'bold'))
            value_label.pack(side='right')
            
            self.metrics_labels[name.lower().replace(' ', '_')] = value_label
    
    def create_analytics_tab(self):
        """Создание вкладки аналитики"""
        analytics_frame = tk.Frame(self.notebook, bg='#1e1e1e')
        self.notebook.add(analytics_frame, text='📊 Аналитика')
        
        tk.Label(analytics_frame, text="📊 Детальная аналитика", 
                bg='#1e1e1e', fg='#3742fa', font=('Arial', 16, 'bold')).pack(pady=20)
        
        # Здесь будет детальная аналитика
        tk.Label(analytics_frame, text="Функция в разработке...", 
                bg='#1e1e1e', fg='white', font=('Arial', 12)).pack()
    
    def create_backtest_tab(self):
        """Создание вкладки бэктестинга"""
        backtest_frame = tk.Frame(self.notebook, bg='#1e1e1e')
        self.notebook.add(backtest_frame, text='🧪 Бэктест')
        
        tk.Label(backtest_frame, text="🧪 Бэктестинг стратегий", 
                bg='#1e1e1e', fg='#3742fa', font=('Arial', 16, 'bold')).pack(pady=20)
        
        # Здесь будет интерфейс бэктестинга
        tk.Label(backtest_frame, text="Функция в разработке...", 
                bg='#1e1e1e', fg='white', font=('Arial', 12)).pack()
    
    def create_positions_tab(self):
        """Создание вкладки позиций"""
        positions_frame = tk.Frame(self.notebook, bg='#1e1e1e')
        self.notebook.add(positions_frame, text='📈 Портфель')
        
        tk.Label(positions_frame, text="📈 Управление портфелем", 
                bg='#1e1e1e', fg='#3742fa', font=('Arial', 16, 'bold')).pack(pady=20)
        
        # Здесь будет детальное управление портфелем
        tk.Label(positions_frame, text="Функция в разработке...", 
                bg='#1e1e1e', fg='white', font=('Arial', 12)).pack()
    
    def create_settings_tab(self):
        """Создание вкладки настроек"""
        settings_frame = tk.Frame(self.notebook, bg='#1e1e1e')
        self.notebook.add(settings_frame, text='⚙️ Настройки')
        
        tk.Label(settings_frame, text="⚙️ Настройки системы", 
                bg='#1e1e1e', fg='#3742fa', font=('Arial', 16, 'bold')).pack(pady=20)
        
        # Здесь будут настройки
        tk.Label(settings_frame, text="Функция в разработке...", 
                bg='#1e1e1e', fg='white', font=('Arial', 12)).pack()
    
    def create_status_bar(self):
        """Создание статусной строки"""
        status_frame = tk.Frame(self.root, bg='#2d2d2d', height=25)
        status_frame.pack(fill='x', side='bottom')
        status_frame.pack_propagate(False)
        
        # Статус подключения
        self.connection_status = tk.Label(status_frame, text="🔴 Отключен", 
                                         bg='#2d2d2d', fg='#ff4757', font=('Arial', 9))
        self.connection_status.pack(side='left', padx=10)
        
        # Время
        self.time_label = tk.Label(status_frame, text="", 
                                  bg='#2d2d2d', fg='white', font=('Arial', 9))
        self.time_label.pack(side='right', padx=10)
        
        # Обновление времени
        self.update_time()
    
    def create_notifications_panel(self):
        """Создание панели уведомлений"""
        # Это будет всплывающая панель для уведомлений
        pass
    
    def setup_styles(self):
        """Настройка стилей"""
        style = ttk.Style()
        
        # Настройка стилей для Treeview
        style.configure('Treeview', 
                       background='#2d2d2d', 
                       foreground='white',
                       fieldbackground='#2d2d2d',
                       borderwidth=0)
        style.configure('Treeview.Heading', 
                       background='#3742fa', 
                       foreground='white',
                       borderwidth=1)
        style.map('Treeview', 
                 background=[('selected', '#3742fa')])
    
    def setup_hotkeys(self):
        """Настройка горячих клавиш"""
        self.root.bind('<Control-n>', lambda e: self.new_session())
        self.root.bind('<Control-s>', lambda e: self.save_config())
        self.root.bind('<Control-o>', lambda e: self.load_config())
        self.root.bind('<F5>', lambda e: self.refresh_data())
        self.root.bind('<F9>', lambda e: self.start_trading())
        self.root.bind('<F10>', lambda e: self.stop_trading())
        self.root.bind('<F1>', lambda e: self.show_help())
        self.root.bind('<F11>', lambda e: self.toggle_fullscreen())
        self.root.bind('<Escape>', lambda e: self.emergency_stop())
    
    def start_update_cycles(self):
        """Запуск циклов обновления"""
        self.update_data_cycle()
        self.update_time()
    
    def update_data_cycle(self):
        """Цикл обновления данных"""
        if self.is_trading_active:
            self.update_positions()
            self.update_orders()
            self.update_performance_metrics()
            if ADVANCED_FEATURES:
                self.update_charts()
        
        # Планирование следующего обновления
        self.root.after(self.update_interval, self.update_data_cycle)
    
    def update_time(self):
        """Обновление времени"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time)
    
    # Методы обновления данных
    def update_positions(self):
        """Обновление позиций"""
        # Очистка таблицы
        for item in self.positions_tree.get_children():
            self.positions_tree.delete(item)
        
        # Получение позиций (мок-данные)
        positions = [
            ("BTC/USDT", "Long", "0.1", "+$150.00"),
            ("ETH/USDT", "Short", "2.5", "-$75.00")
        ]
        
        for position in positions:
            self.positions_tree.insert('', 'end', values=position)
    
    def update_orders(self):
        """Обновление ордеров"""
        # Очистка таблицы
        for item in self.orders_tree.get_children():
            self.orders_tree.delete(item)
        
        # Получение ордеров (мок-данные)
        orders = [
            ("BTC/USDT", "Limit", "Buy", "0.05"),
            ("ETH/USDT", "Stop", "Sell", "1.0")
        ]
        
        for order in orders:
            self.orders_tree.insert('', 'end', values=order)
    
    def update_performance_metrics(self):
        """Обновление метрик производительности"""
        # Мок-данные
        metrics = {
            'total_p&l': '+$150.75',
            'win_rate': '65%',
            'sharpe_ratio': '1.25',
            'max_drawdown': '-5.2%',
            'total_trades': '8'
        }
        
        for key, value in metrics.items():
            if key in self.metrics_labels:
                self.metrics_labels[key].config(text=value)
    
    def update_charts(self):
        """Обновление графиков"""
        self.update_price_chart()
        self.update_pnl_chart()
        self.update_distribution_chart()
    
    def update_price_chart(self):
        """Обновление графика цены"""
        if not ADVANCED_FEATURES:
            return
        
        # Генерация мок-данных
        x = np.linspace(0, 100, 100)
        y = 45000 + np.cumsum(np.random.randn(100) * 50)
        
        self.price_ax.clear()
        self.price_ax.plot(x, y, color='#00ff88', linewidth=2)
        self.price_ax.set_title('BTC/USDT Price', color='white', fontsize=12)
        self.price_ax.set_facecolor('#2d2d2d')
        self.price_ax.tick_params(colors='white')
        
        # Обновление спайнов
        for spine in self.price_ax.spines.values():
            spine.set_color('white')
        
        self.price_canvas.draw()
    
    def update_pnl_chart(self):
        """Обновление графика P&L"""
        if not ADVANCED_FEATURES:
            return
        
        # Генерация мок-данных
        x = np.linspace(0, 100, 100)
        y = np.cumsum(np.random.randn(100) * 10)
        
        self.pnl_ax.clear()
        colors = ['#ff4757' if val < 0 else '#00ff88' for val in y]
        self.pnl_ax.plot(x, y, color='#3742fa', linewidth=2)
        self.pnl_ax.fill_between(x, y, 0, alpha=0.3, color='#3742fa')
        self.pnl_ax.set_title('P&L History', color='white', fontsize=12)
        self.pnl_ax.set_facecolor('#2d2d2d')
        self.pnl_ax.tick_params(colors='white')
        
        # Обновление спайнов
        for spine in self.pnl_ax.spines.values():
            spine.set_color('white')
        
        self.pnl_canvas.draw()
    
    def update_distribution_chart(self):
        """Обновление графика распределений"""
        if not ADVANCED_FEATURES:
            return
        
        # Генерация мок-данных
        returns = np.random.normal(0, 1, 1000)
        
        self.dist_ax.clear()
        self.dist_ax.hist(returns, bins=50, color='#3742fa', alpha=0.7, edgecolor='white')
        self.dist_ax.set_title('Returns Distribution', color='white', fontsize=12)
        self.dist_ax.set_facecolor('#2d2d2d')
        self.dist_ax.tick_params(colors='white')
        
        # Обновление спайнов
        for spine in self.dist_ax.spines.values():
            spine.set_color('white')
        
        self.dist_canvas.draw()
    
    # Методы управления торговлей
    def toggle_trading(self):
        """Переключение торговли"""
        if self.is_trading_active:
            self.stop_trading()
        else:
            self.start_trading()
    
    def start_trading(self):
        """Запуск торговли"""
        if not self.is_trading_active:
            self.is_trading_active = True
            self.trade_button.config(text="⏸️ Остановить", bg='#ff4757')
            self.connection_status.config(text="🟢 Активен", fg='#00ff88')
            self.logger.info("Торговля запущена")
            
            # Здесь будет запуск торговых стратегий
            messagebox.showinfo("Торговля", "Торговля запущена успешно!")
    
    def stop_trading(self):
        """Остановка торговли"""
        if self.is_trading_active:
            self.is_trading_active = False
            self.trade_button.config(text="▶ Запустить", bg='#00ff88')
            self.connection_status.config(text="🔴 Остановлен", fg='#ff4757')
            self.logger.info("Торговля остановлена")
            
            messagebox.showinfo("Торговля", "Торговля остановлена успешно!")
    
    def emergency_stop(self):
        """Экстренная остановка"""
        self.stop_trading()
        # Здесь будет закрытие всех позиций
        messagebox.showwarning("Экстренная остановка", 
                              "Выполнена экстренная остановка торговли!\nВсе активные операции прерваны.")
    
    def quick_start(self):
        """Быстрый старт"""
        # Выбор популярных настроек
        self.pairs_listbox.selection_set(0, 2)  # BTC, ETH, BNB
        
        # Активация рекомендуемых стратегий
        self.strategy_vars["RSI Bounce"].set(True)
        self.strategy_vars["MACD Cross"].set(True)
        
        # Запуск торговли
        self.start_trading()
    
    # Методы меню
    def new_session(self):
        """Новая сессия"""
        if messagebox.askyesno("Новая сессия", "Создать новую торговую сессию?\nТекущие данные будут очищены."):
            # Сброс данных
            self.stop_trading()
            self.clear_pairs_selection()
            for var in self.strategy_vars.values():
                var.set(False)
            messagebox.showinfo("Новая сессия", "Новая сессия создана успешно!")
    
    def save_config(self):
        """Сохранение конфигурации"""
        config = {
            'selected_pairs': [self.pairs_listbox.get(i) for i in self.pairs_listbox.curselection()],
            'active_strategies': [name for name, var in self.strategy_vars.items() if var.get()],
            'position_size': float(self.position_size_var.get()),
            'stop_loss': float(self.stop_loss_var.get()),
            'take_profit': float(self.take_profit_var.get()),
            'mode': self.mode_var.get()
        }
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Сохранить конфигурацию"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                messagebox.showinfo("Сохранение", f"Конфигурация сохранена:\n{filename}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить конфигурацию:\n{e}")
    
    def load_config(self):
        """Загрузка конфигурации"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Загрузить конфигурацию"
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Применение конфигурации
                if 'mode' in config:
                    self.mode_var.set(config['mode'])
                
                if 'position_size' in config:
                    self.position_size_var.set(str(config['position_size']))
                
                if 'stop_loss' in config:
                    self.stop_loss_var.set(str(config['stop_loss']))
                
                if 'take_profit' in config:
                    self.take_profit_var.set(str(config['take_profit']))
                
                messagebox.showinfo("Загрузка", f"Конфигурация загружена:\n{filename}")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить конфигурацию:\n{e}")
    
    def export_data(self):
        """Экспорт данных"""
        messagebox.showinfo("Экспорт", "Функция экспорта в разработке")
    
    def import_strategies(self):
        """Импорт стратегий"""
        messagebox.showinfo("Импорт", "Функция импорта в разработке")
    
    def strategy_settings(self):
        """Настройки стратегий"""
        messagebox.showinfo("Настройки стратегий", "Детальные настройки стратегий в разработке")
    
    def risk_settings(self):
        """Настройки управления рисками"""
        messagebox.showinfo("Управление рисками", "Расширенные настройки рисков в разработке")
    
    def close_all_positions(self):
        """Закрытие всех позиций"""
        if messagebox.askyesno("Закрытие позиций", "Закрыть все открытые позиции?"):
            messagebox.showinfo("Позиции", "Все позиции закрыты")
    
    def cancel_all_orders(self):
        """Отмена всех ордеров"""
        if messagebox.askyesno("Отмена ордеров", "Отменить все активные ордера?"):
            messagebox.showinfo("Ордера", "Все ордера отменены")
    
    def performance_report(self):
        """Отчет производительности"""
        messagebox.showinfo("Отчет", "Детальный отчет производительности в разработке")
    
    def risk_analysis(self):
        """Анализ рисков"""
        messagebox.showinfo("Анализ рисков", "Анализ рисков в разработке")
    
    def strategy_optimization(self):
        """Оптимизация стратегий"""
        messagebox.showinfo("Оптимизация", "Оптимизация стратегий в разработке")
    
    def export_excel(self):
        """Экспорт в Excel"""
        messagebox.showinfo("Excel", "Экспорт в Excel в разработке")
    
    def generate_pdf_report(self):
        """Генерация PDF отчета"""
        messagebox.showinfo("PDF", "Генерация PDF отчета в разработке")
    
    def position_calculator(self):
        """Калькулятор позиций"""
        calc_window = tk.Toplevel(self.root)
        calc_window.title("🧮 Калькулятор позиций")
        calc_window.geometry("400x300")
        calc_window.configure(bg='#1e1e1e')
        
        tk.Label(calc_window, text="🧮 Калькулятор позиций", 
                bg='#1e1e1e', fg='#3742fa', font=('Arial', 14, 'bold')).pack(pady=20)
        
        tk.Label(calc_window, text="Функция в разработке...", 
                bg='#1e1e1e', fg='white', font=('Arial', 12)).pack()
    
    def currency_converter(self):
        """Конвертер валют"""
        messagebox.showinfo("Конвертер", "Конвертер валют в разработке")
    
    def economic_calendar(self):
        """Экономический календарь"""
        messagebox.showinfo("Календарь", "Экономический календарь в разработке")
    
    def open_settings(self):
        """Открытие настроек"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("⚙️ Настройки")
        settings_window.geometry("500x400")
        settings_window.configure(bg='#1e1e1e')
        
        tk.Label(settings_window, text="⚙️ Настройки системы", 
                bg='#1e1e1e', fg='#3742fa', font=('Arial', 14, 'bold')).pack(pady=20)
        
        tk.Label(settings_window, text="Функция в разработке...", 
                bg='#1e1e1e', fg='white', font=('Arial', 12)).pack()
    
    def show_help(self):
        """Показать справку"""
        help_window = tk.Toplevel(self.root)
        help_window.title("📚 Справка ATB Trading Dashboard")
        help_window.geometry("600x500")
        help_window.configure(bg='#1e1e1e')
        
        # Создание текста справки
        help_text = tk.Text(help_window, bg='#2d2d2d', fg='white', 
                           font=('Arial', 11), wrap='word',
                           insertbackground='white')
        help_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        help_content = """
📚 ATB Trading Dashboard v2.0 - Справка

🎯 ОСНОВНЫЕ ФУНКЦИИ:
• Торговля в режимах: симуляция, реальная торговля, бэктест
• 6 встроенных торговых стратегий
• Управление рисками и портфелем
• Аналитика и отчеты в реальном времени

🎮 УПРАВЛЕНИЕ:
• F9 - Запустить торговлю
• F10 - Остановить торговлю
• F5 - Обновить данные
• F1 - Справка
• F11 - Полноэкранный режим
• Esc - Экстренная остановка

⚙️ НАСТРОЙКА:
1. Выберите торговые пары в левой панели
2. Активируйте нужные стратегии
3. Настройте параметры риска
4. Выберите режим торговли
5. Нажмите "Запустить"

🛡️ БЕЗОПАСНОСТЬ:
• Всегда начинайте с режима "simulation"
• Используйте разумные размеры позиций (1-3%)
• Устанавливайте стоп-лоссы
• Регулярно анализируйте результаты

📞 ПОДДЕРЖКА:
• Email: support@atb-trading.com
• Документация: README.md
• Логи: logs/dashboard.log
        """
        
        help_text.insert('1.0', help_content.strip())
        help_text.config(state='disabled')
    
    def show_hotkeys(self):
        """Показать горячие клавиши"""
        hotkeys_window = tk.Toplevel(self.root)
        hotkeys_window.title("🔥 Горячие клавиши")
        hotkeys_window.geometry("400x300")
        hotkeys_window.configure(bg='#1e1e1e')
        
        tk.Label(hotkeys_window, text="🔥 Горячие клавиши", 
                bg='#1e1e1e', fg='#3742fa', font=('Arial', 14, 'bold')).pack(pady=20)
        
        hotkeys_text = """
F1 - Справка
F5 - Обновить данные
F9 - Запустить торговлю
F10 - Остановить торговлю
F11 - Полноэкранный режим
Esc - Экстренная остановка

Ctrl+N - Новая сессия
Ctrl+S - Сохранить конфигурацию
Ctrl+O - Загрузить конфигурацию

Alt+F4 - Выход
        """
        
        tk.Label(hotkeys_window, text=hotkeys_text.strip(), 
                bg='#1e1e1e', fg='white', font=('Arial', 11), justify='left').pack(pady=10)
    
    def show_about(self):
        """О программе"""
        about_window = tk.Toplevel(self.root)
        about_window.title("🆔 О программе")
        about_window.geometry("400x300")
        about_window.configure(bg='#1e1e1e')
        
        tk.Label(about_window, text="⚡ ATB Trading Dashboard", 
                bg='#1e1e1e', fg='#3742fa', font=('Arial', 16, 'bold')).pack(pady=20)
        
        about_text = """
Версия: 2.0 Advanced Edition

Современная платформа для торговли
криптовалютами с полным функционалом
управления рисками и аналитики.

© 2024 ATB Trading Systems
Все права защищены.

Разработано с использованием:
• Python 3.8+
• Tkinter GUI
• Matplotlib (опционально)
• NumPy, Pandas (опционально)
        """
        
        tk.Label(about_window, text=about_text.strip(), 
                bg='#1e1e1e', fg='white', font=('Arial', 11), justify='center').pack(pady=10)
    
    def refresh_data(self):
        """Обновление данных"""
        self.update_positions()
        self.update_orders()
        self.update_performance_metrics()
        if ADVANCED_FEATURES:
            self.update_charts()
        
        # Показать уведомление об обновлении
        self.show_notification("🔄 Данные обновлены", "success")
    
    def toggle_fullscreen(self):
        """Переключение полноэкранного режима"""
        current_state = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not current_state)
    
    def show_notification(self, message, type_="info"):
        """Показать уведомление"""
        # Простая реализация через статусную строку
        colors = {
            "info": "#3742fa",
            "success": "#00ff88",
            "warning": "#ffa500",
            "error": "#ff4757"
        }
        
        # Временно изменяем статус
        original_text = self.connection_status.cget('text')
        original_color = self.connection_status.cget('fg')
        
        self.connection_status.config(text=message, fg=colors.get(type_, "#3742fa"))
        
        # Возвращаем через 3 секунды
        self.root.after(3000, lambda: self.connection_status.config(text=original_text, fg=original_color))
    
    # Вспомогательные методы
    def populate_pairs_list(self):
        """Заполнение списка торговых пар"""
        pairs = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT",
            "SOL/USDT", "DOT/USDT", "MATIC/USDT", "LINK/USDT",
            "AVAX/USDT", "UNI/USDT", "ATOM/USDT", "FTM/USDT",
            "NEAR/USDT", "ALGO/USDT", "XRP/USDT", "LTC/USDT"
        ]
        
        for pair in pairs:
            self.pairs_listbox.insert('end', pair)
    
    def filter_pairs(self, event=None):
        """Фильтрация торговых пар"""
        search_term = self.pairs_search.get().upper()
        
        # Очистка списка
        self.pairs_listbox.delete(0, 'end')
        
        # Заполнение отфильтрованным списком
        all_pairs = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT",
            "SOL/USDT", "DOT/USDT", "MATIC/USDT", "LINK/USDT",
            "AVAX/USDT", "UNI/USDT", "ATOM/USDT", "FTM/USDT",
            "NEAR/USDT", "ALGO/USDT", "XRP/USDT", "LTC/USDT"
        ]
        
        for pair in all_pairs:
            if search_term in pair.upper():
                self.pairs_listbox.insert('end', pair)
    
    def select_all_pairs(self):
        """Выбрать все пары"""
        self.pairs_listbox.selection_set(0, 'end')
    
    def clear_pairs_selection(self):
        """Очистить выбор пар"""
        self.pairs_listbox.selection_clear(0, 'end')
    
    def on_closing(self):
        """Обработка закрытия окна"""
        if messagebox.askokcancel("Выход", "Закрыть ATB Trading Dashboard?"):
            if self.is_trading_active:
                self.stop_trading()
            self.logger.info("Дашборд закрыт пользователем")
            self.root.destroy()
    
    def run(self):
        """Запуск дашборда"""
        try:
            self.logger.info("Запуск продвинутого дашборда")
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"Критическая ошибка дашборда: {e}")
            messagebox.showerror("Критическая ошибка", f"Произошла критическая ошибка:\n{e}")

def main() -> None:
    """Главная функция"""
    try:
        dashboard = AdvancedTradingDashboard()
        dashboard.run()
    except Exception as e:
        print(f"Ошибка запуска дашборда: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()