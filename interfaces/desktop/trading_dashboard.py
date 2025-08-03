"""
Современный дашборд управления торговлей для Windows
Включает управление торговлей, аналитику, бэктестинг и обучение
"""

import sys
import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import tkinter.font as tkFont
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd

# Импорты из основного проекта
from domain.entities.trading import Trade
from domain.entities.strategy import Strategy
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.types import Symbol, TradingPair
from application.services.trading_service import TradingService
from infrastructure.external_services.exchanges.base_exchange_service import BaseExchangeService

@dataclass
class DashboardState:
    """Состояние дашборда"""
    selected_pairs: List[str]
    active_strategies: List[str]
    trading_mode: str  # 'live', 'simulation', 'backtest'
    total_balance: Decimal
    current_pnl: Decimal
    daily_pnl: Decimal
    active_positions: int
    is_trading_active: bool
    last_update: datetime

class ModernTradingDashboard:
    """Современный дашборд торговли"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.setup_window()
        self.setup_styles()
        
        # Состояние дашборда
        self.state = DashboardState(
            selected_pairs=[],
            active_strategies=[],
            trading_mode='simulation',
            total_balance=Decimal('10000'),
            current_pnl=Decimal('0'),
            daily_pnl=Decimal('0'),
            active_positions=0,
            is_trading_active=False,
            last_update=datetime.now()
        )
        
        # Данные для графиков
        self.price_data = {}
        self.pnl_history = []
        self.equity_curve = []
        
        # Инициализация компонентов
        self.create_main_layout()
        self.create_toolbar()
        self.create_side_panels()
        self.create_main_content()
        self.create_status_bar()
        
        # Запуск обновления данных
        self.start_data_updates()
    
    def setup_window(self):
        """Настройка главного окна"""
        self.root.title("ATB Trading Dashboard v2.0")
        self.root.geometry("1600x1000")
        self.root.minsize(1200, 800)
        
        # Иконка и настройки окна
        self.root.configure(bg='#1e1e1e')
        
        # Центрирование окна
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1600 // 2)
        y = (self.root.winfo_screenheight() // 2) - (1000 // 2)
        self.root.geometry(f"1600x1000+{x}+{y}")
    
    def setup_styles(self):
        """Настройка современных стилей"""
        self.style = ttk.Style()
        
        # Темная тема
        self.colors = {
            'bg_primary': '#1e1e1e',
            'bg_secondary': '#2d2d2d', 
            'bg_tertiary': '#3e3e3e',
            'text_primary': '#ffffff',
            'text_secondary': '#b0b0b0',
            'accent_green': '#00ff88',
            'accent_red': '#ff4757',
            'accent_blue': '#3742fa',
            'accent_orange': '#ffa726',
            'border': '#404040'
        }
        
        # Настройка стилей ttk
        self.style.theme_use('clam')
        self.style.configure('Dark.TFrame', background=self.colors['bg_secondary'])
        self.style.configure('Dark.TLabel', background=self.colors['bg_secondary'], 
                           foreground=self.colors['text_primary'])
        self.style.configure('Dark.TButton', background=self.colors['bg_tertiary'],
                           foreground=self.colors['text_primary'])
        
        # Шрифты
        self.fonts = {
            'title': tkFont.Font(family='Segoe UI', size=16, weight='bold'),
            'subtitle': tkFont.Font(family='Segoe UI', size=12, weight='bold'),
            'body': tkFont.Font(family='Segoe UI', size=10),
            'small': tkFont.Font(family='Segoe UI', size=8),
            'mono': tkFont.Font(family='Consolas', size=10)
        }
    
    def create_main_layout(self):
        """Создание основной компоновки"""
        # Главный контейнер
        self.main_container = tk.Frame(self.root, bg=self.colors['bg_primary'])
        self.main_container.pack(fill='both', expand=True)
        
        # Панель инструментов (сверху)
        self.toolbar_frame = tk.Frame(self.main_container, 
                                    bg=self.colors['bg_secondary'], height=60)
        self.toolbar_frame.pack(fill='x', pady=(0, 2))
        self.toolbar_frame.pack_propagate(False)
        
        # Основной контент (по центру)
        self.content_frame = tk.Frame(self.main_container, bg=self.colors['bg_primary'])
        self.content_frame.pack(fill='both', expand=True, padx=5, pady=2)
        
        # Боковые панели
        self.left_panel = tk.Frame(self.content_frame, bg=self.colors['bg_secondary'], width=300)
        self.left_panel.pack(side='left', fill='y', padx=(0, 5))
        self.left_panel.pack_propagate(False)
        
        self.right_panel = tk.Frame(self.content_frame, bg=self.colors['bg_secondary'], width=300)
        self.right_panel.pack(side='right', fill='y', padx=(5, 0))
        self.right_panel.pack_propagate(False)
        
        # Центральная область для графиков
        self.center_panel = tk.Frame(self.content_frame, bg=self.colors['bg_primary'])
        self.center_panel.pack(side='left', fill='both', expand=True, padx=5)
        
        # Статус-бар (снизу)
        self.status_frame = tk.Frame(self.main_container, 
                                   bg=self.colors['bg_secondary'], height=30)
        self.status_frame.pack(fill='x', pady=(2, 0))
        self.status_frame.pack_propagate(False)
    
    def create_toolbar(self):
        """Создание панели инструментов"""
        # Логотип и название
        logo_frame = tk.Frame(self.toolbar_frame, bg=self.colors['bg_secondary'])
        logo_frame.pack(side='left', padx=10, pady=5)
        
        tk.Label(logo_frame, text="⚡ ATB Trading", 
               font=self.fonts['title'], 
               fg=self.colors['accent_blue'],
               bg=self.colors['bg_secondary']).pack(side='left')
        
        # Режимы торговли
        mode_frame = tk.Frame(self.toolbar_frame, bg=self.colors['bg_secondary'])
        mode_frame.pack(side='left', padx=20, pady=5)
        
        tk.Label(mode_frame, text="Режим:", 
               font=self.fonts['body'],
               fg=self.colors['text_secondary'],
               bg=self.colors['bg_secondary']).pack(side='left')
        
        self.trading_mode_var = tk.StringVar(value=self.state.trading_mode)
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.trading_mode_var,
                                values=['simulation', 'live', 'backtest'],
                                state='readonly', width=12)
        mode_combo.pack(side='left', padx=(5, 0))
        mode_combo.bind('<<ComboboxSelected>>', self.on_mode_change)
        
        # Кнопки управления
        control_frame = tk.Frame(self.toolbar_frame, bg=self.colors['bg_secondary'])
        control_frame.pack(side='left', padx=20, pady=5)
        
        self.start_btn = tk.Button(control_frame, text="▶ Запустить",
                                 bg=self.colors['accent_green'], fg='white',
                                 font=self.fonts['body'], relief='flat',
                                 command=self.start_trading)
        self.start_btn.pack(side='left', padx=(0, 5))
        
        self.stop_btn = tk.Button(control_frame, text="⏹ Остановить",
                                bg=self.colors['accent_red'], fg='white',
                                font=self.fonts['body'], relief='flat',
                                command=self.stop_trading, state='disabled')
        self.stop_btn.pack(side='left', padx=(0, 5))
        
        self.reset_btn = tk.Button(control_frame, text="🔄 Сброс",
                                 bg=self.colors['bg_tertiary'], fg=self.colors['text_primary'],
                                 font=self.fonts['body'], relief='flat',
                                 command=self.reset_trading)
        self.reset_btn.pack(side='left')
        
        # Статистика в toolbar
        stats_frame = tk.Frame(self.toolbar_frame, bg=self.colors['bg_secondary'])
        stats_frame.pack(side='right', padx=10, pady=5)
        
        self.balance_label = tk.Label(stats_frame, text="Баланс: $10,000.00",
                                    font=self.fonts['subtitle'],
                                    fg=self.colors['text_primary'],
                                    bg=self.colors['bg_secondary'])
        self.balance_label.pack(side='right', padx=(0, 20))
        
        self.pnl_label = tk.Label(stats_frame, text="P&L: $0.00",
                                font=self.fonts['subtitle'],
                                fg=self.colors['accent_green'],
                                bg=self.colors['bg_secondary'])
        self.pnl_label.pack(side='right', padx=(0, 20))
    
    def create_side_panels(self):
        """Создание боковых панелей"""
        # Левая панель - торговые пары и стратегии
        self.create_trading_pairs_panel()
        self.create_strategies_panel()
        self.create_risk_panel()
        
        # Правая панель - позиции и ордера
        self.create_positions_panel()
        self.create_orders_panel()
        self.create_performance_panel()
    
    def create_trading_pairs_panel(self):
        """Панель выбора торговых пар"""
        frame = tk.LabelFrame(self.left_panel, text="Торговые пары",
                            bg=self.colors['bg_secondary'],
                            fg=self.colors['text_primary'],
                            font=self.fonts['subtitle'])
        frame.pack(fill='x', padx=5, pady=5)
        
        # Поиск пар
        search_frame = tk.Frame(frame, bg=self.colors['bg_secondary'])
        search_frame.pack(fill='x', padx=5, pady=5)
        
        self.pair_search_var = tk.StringVar()
        search_entry = tk.Entry(search_frame, textvariable=self.pair_search_var,
                              bg=self.colors['bg_tertiary'], fg=self.colors['text_primary'],
                              insertbackground=self.colors['text_primary'])
        search_entry.pack(fill='x')
        search_entry.bind('<KeyRelease>', self.filter_pairs)
        
        # Список пар с чекбоксами
        list_frame = tk.Frame(frame, bg=self.colors['bg_secondary'])
        list_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Скроллбар
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.pairs_listbox = tk.Listbox(list_frame, 
                                      bg=self.colors['bg_tertiary'],
                                      fg=self.colors['text_primary'],
                                      selectbackground=self.colors['accent_blue'],
                                      yscrollcommand=scrollbar.set,
                                      selectmode='multiple')
        self.pairs_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.pairs_listbox.yview)
        
        # Популярные пары
        popular_pairs = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 
                        'SOL/USDT', 'DOT/USDT', 'MATIC/USDT', 'LINK/USDT']
        
        for pair in popular_pairs:
            self.pairs_listbox.insert('end', pair)
        
        self.pairs_listbox.bind('<<ListboxSelect>>', self.on_pair_select)
    
    def create_strategies_panel(self):
        """Панель стратегий"""
        frame = tk.LabelFrame(self.left_panel, text="Стратегии",
                            bg=self.colors['bg_secondary'],
                            fg=self.colors['text_primary'],
                            font=self.fonts['subtitle'])
        frame.pack(fill='x', padx=5, pady=5)
        
        # Доступные стратегии
        strategies = [
            'RSI Bounce', 'MACD Cross', 'Bollinger Squeeze',
            'Mean Reversion', 'Momentum', 'Grid Trading'
        ]
        
        self.strategy_vars = {}
        for strategy in strategies:
            var = tk.BooleanVar()
            self.strategy_vars[strategy] = var
            
            cb = tk.Checkbutton(frame, text=strategy, variable=var,
                              bg=self.colors['bg_secondary'],
                              fg=self.colors['text_primary'],
                              selectcolor=self.colors['bg_tertiary'],
                              command=self.update_active_strategies)
            cb.pack(anchor='w', padx=5, pady=2)
    
    def create_risk_panel(self):
        """Панель управления рисками"""
        frame = tk.LabelFrame(self.left_panel, text="Управление рисками",
                            bg=self.colors['bg_secondary'],
                            fg=self.colors['text_primary'],
                            font=self.fonts['subtitle'])
        frame.pack(fill='x', padx=5, pady=5)
        
        # Размер позиции
        tk.Label(frame, text="Размер позиции (%)", 
               bg=self.colors['bg_secondary'],
               fg=self.colors['text_secondary']).pack(anchor='w', padx=5)
        
        self.position_size_var = tk.DoubleVar(value=2.0)
        position_scale = tk.Scale(frame, from_=0.1, to=10.0, resolution=0.1,
                                orient='horizontal', variable=self.position_size_var,
                                bg=self.colors['bg_secondary'],
                                fg=self.colors['text_primary'],
                                troughcolor=self.colors['accent_blue'])
        position_scale.pack(fill='x', padx=5, pady=2)
        
        # Стоп-лосс
        tk.Label(frame, text="Стоп-лосс (%)", 
               bg=self.colors['bg_secondary'],
               fg=self.colors['text_secondary']).pack(anchor='w', padx=5)
        
        self.stop_loss_var = tk.DoubleVar(value=2.0)
        stop_scale = tk.Scale(frame, from_=0.5, to=10.0, resolution=0.1,
                            orient='horizontal', variable=self.stop_loss_var,
                            bg=self.colors['bg_secondary'],
                            fg=self.colors['text_primary'],
                            troughcolor=self.colors['accent_red'])
        stop_scale.pack(fill='x', padx=5, pady=2)
        
        # Тейк-профит
        tk.Label(frame, text="Тейк-профит (%)", 
               bg=self.colors['bg_secondary'],
               fg=self.colors['text_secondary']).pack(anchor='w', padx=5)
        
        self.take_profit_var = tk.DoubleVar(value=4.0)
        tp_scale = tk.Scale(frame, from_=1.0, to=20.0, resolution=0.1,
                          orient='horizontal', variable=self.take_profit_var,
                          bg=self.colors['bg_secondary'],
                          fg=self.colors['text_primary'],
                          troughcolor=self.colors['accent_green'])
        tp_scale.pack(fill='x', padx=5, pady=2)
    
    def create_positions_panel(self):
        """Панель текущих позиций"""
        frame = tk.LabelFrame(self.right_panel, text="Открытые позиции",
                            bg=self.colors['bg_secondary'],
                            fg=self.colors['text_primary'],
                            font=self.fonts['subtitle'])
        frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Таблица позиций
        columns = ('Symbol', 'Side', 'Size', 'Entry', 'Current', 'P&L')
        self.positions_tree = ttk.Treeview(frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.positions_tree.heading(col, text=col)
            self.positions_tree.column(col, width=70)
        
        # Скроллбар для таблицы
        pos_scrollbar = ttk.Scrollbar(frame, orient='vertical', command=self.positions_tree.yview)
        self.positions_tree.configure(yscrollcommand=pos_scrollbar.set)
        
        self.positions_tree.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        pos_scrollbar.pack(side='right', fill='y')
    
    def create_orders_panel(self):
        """Панель ордеров"""
        frame = tk.LabelFrame(self.right_panel, text="Активные ордера",
                            bg=self.colors['bg_secondary'],
                            fg=self.colors['text_primary'],
                            font=self.fonts['subtitle'])
        frame.pack(fill='x', padx=5, pady=5)
        
        # Таблица ордеров
        columns = ('Symbol', 'Type', 'Side', 'Amount', 'Price', 'Status')
        self.orders_tree = ttk.Treeview(frame, columns=columns, show='headings', height=6)
        
        for col in columns:
            self.orders_tree.heading(col, text=col)
            self.orders_tree.column(col, width=60)
        
        self.orders_tree.pack(fill='x', padx=5, pady=5)
    
    def create_performance_panel(self):
        """Панель производительности"""
        frame = tk.LabelFrame(self.right_panel, text="Производительность",
                            bg=self.colors['bg_secondary'],
                            fg=self.colors['text_primary'],
                            font=self.fonts['subtitle'])
        frame.pack(fill='x', padx=5, pady=5)
        
        # Метрики
        metrics_frame = tk.Frame(frame, bg=self.colors['bg_secondary'])
        metrics_frame.pack(fill='x', padx=5, pady=5)
        
        # Сетка метрик 2x2
        tk.Label(metrics_frame, text="Общий P&L:", 
               bg=self.colors['bg_secondary'], fg=self.colors['text_secondary']).grid(row=0, column=0, sticky='w')
        self.total_pnl_label = tk.Label(metrics_frame, text="$0.00",
                                      bg=self.colors['bg_secondary'], fg=self.colors['accent_green'])
        self.total_pnl_label.grid(row=0, column=1, sticky='e')
        
        tk.Label(metrics_frame, text="Дневной P&L:", 
               bg=self.colors['bg_secondary'], fg=self.colors['text_secondary']).grid(row=1, column=0, sticky='w')
        self.daily_pnl_label = tk.Label(metrics_frame, text="$0.00",
                                      bg=self.colors['bg_secondary'], fg=self.colors['accent_green'])
        self.daily_pnl_label.grid(row=1, column=1, sticky='e')
        
        tk.Label(metrics_frame, text="Win Rate:", 
               bg=self.colors['bg_secondary'], fg=self.colors['text_secondary']).grid(row=2, column=0, sticky='w')
        self.win_rate_label = tk.Label(metrics_frame, text="0%",
                                     bg=self.colors['bg_secondary'], fg=self.colors['text_primary'])
        self.win_rate_label.grid(row=2, column=1, sticky='e')
        
        tk.Label(metrics_frame, text="Sharpe Ratio:", 
               bg=self.colors['bg_secondary'], fg=self.colors['text_secondary']).grid(row=3, column=0, sticky='w')
        self.sharpe_label = tk.Label(metrics_frame, text="0.00",
                                   bg=self.colors['bg_secondary'], fg=self.colors['text_primary'])
        self.sharpe_label.grid(row=3, column=1, sticky='e')
        
        metrics_frame.columnconfigure(1, weight=1)
    
    def create_main_content(self):
        """Создание основного контента с графиками"""
        # Верхняя панель графиков
        self.charts_notebook = ttk.Notebook(self.center_panel)
        self.charts_notebook.pack(fill='both', expand=True)
        
        # Вкладка основного графика цены
        self.create_price_chart_tab()
        
        # Вкладка P&L графика
        self.create_pnl_chart_tab()
        
        # Вкладка бэктестинга
        self.create_backtest_tab()
        
        # Вкладка аналитики
        self.create_analytics_tab()
    
    def create_price_chart_tab(self):
        """Вкладка графика цен"""
        price_frame = ttk.Frame(self.charts_notebook)
        self.charts_notebook.add(price_frame, text="📈 График цен")
        
        # Matplotlib график
        self.price_figure = Figure(figsize=(12, 8), facecolor='#1e1e1e')
        self.price_ax = self.price_figure.add_subplot(111, facecolor='#2d2d2d')
        
        # Стилизация графика
        self.price_ax.tick_params(colors='white')
        self.price_ax.spines['bottom'].set_color('white')
        self.price_ax.spines['top'].set_color('white')
        self.price_ax.spines['right'].set_color('white')
        self.price_ax.spines['left'].set_color('white')
        
        self.price_canvas = FigureCanvasTkAgg(self.price_figure, price_frame)
        self.price_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Инициализация данных
        self.update_price_chart()
    
    def create_pnl_chart_tab(self):
        """Вкладка P&L графика"""
        pnl_frame = ttk.Frame(self.charts_notebook)
        self.charts_notebook.add(pnl_frame, text="💰 P&L График")
        
        self.pnl_figure = Figure(figsize=(12, 8), facecolor='#1e1e1e')
        self.pnl_ax = self.pnl_figure.add_subplot(111, facecolor='#2d2d2d')
        
        # Стилизация
        self.pnl_ax.tick_params(colors='white')
        self.pnl_ax.spines['bottom'].set_color('white')
        self.pnl_ax.spines['top'].set_color('white')
        self.pnl_ax.spines['right'].set_color('white')
        self.pnl_ax.spines['left'].set_color('white')
        
        self.pnl_canvas = FigureCanvasTkAgg(self.pnl_figure, pnl_frame)
        self.pnl_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.update_pnl_chart()
    
    def create_backtest_tab(self):
        """Вкладка бэктестинга"""
        backtest_frame = ttk.Frame(self.charts_notebook)
        self.charts_notebook.add(backtest_frame, text="📊 Бэктестинг")
        
        # Панель управления бэктестом
        control_frame = tk.Frame(backtest_frame, bg=self.colors['bg_secondary'], height=60)
        control_frame.pack(fill='x', padx=5, pady=5)
        control_frame.pack_propagate(False)
        
        # Период бэктеста
        tk.Label(control_frame, text="Период:", 
               bg=self.colors['bg_secondary'], fg=self.colors['text_primary']).pack(side='left', padx=5)
        
        self.backtest_period = ttk.Combobox(control_frame, values=['1M', '3M', '6M', '1Y', '2Y'],
                                          state='readonly', width=10)
        self.backtest_period.set('3M')
        self.backtest_period.pack(side='left', padx=5)
        
        # Кнопка запуска
        tk.Button(control_frame, text="🚀 Запустить бэктест",
                bg=self.colors['accent_blue'], fg='white',
                relief='flat', command=self.run_backtest).pack(side='left', padx=10)
        
        # Результаты бэктеста
        results_frame = tk.Frame(backtest_frame, bg=self.colors['bg_primary'])
        results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # График результатов
        self.backtest_figure = Figure(figsize=(12, 6), facecolor='#1e1e1e')
        self.backtest_ax = self.backtest_figure.add_subplot(111, facecolor='#2d2d2d')
        
        self.backtest_canvas = FigureCanvasTkAgg(self.backtest_figure, results_frame)
        self.backtest_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def create_analytics_tab(self):
        """Вкладка аналитики"""
        analytics_frame = ttk.Frame(self.charts_notebook)
        self.charts_notebook.add(analytics_frame, text="📈 Аналитика")
        
        # Разделение на секции
        top_frame = tk.Frame(analytics_frame, bg=self.colors['bg_primary'])
        top_frame.pack(fill='x', padx=5, pady=5)
        
        bottom_frame = tk.Frame(analytics_frame, bg=self.colors['bg_primary'])
        bottom_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Ключевые метрики
        metrics_panel = tk.LabelFrame(top_frame, text="Ключевые метрики",
                                    bg=self.colors['bg_secondary'],
                                    fg=self.colors['text_primary'])
        metrics_panel.pack(fill='x', padx=5, pady=5)
        
        # Создание сетки метрик
        self.create_analytics_metrics(metrics_panel)
        
        # Графики распределения
        charts_panel = tk.Frame(bottom_frame, bg=self.colors['bg_primary'])
        charts_panel.pack(fill='both', expand=True)
        
        self.analytics_figure = Figure(figsize=(12, 6), facecolor='#1e1e1e')
        self.analytics_canvas = FigureCanvasTkAgg(self.analytics_figure, charts_panel)
        self.analytics_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.update_analytics_charts()
    
    def create_analytics_metrics(self, parent):
        """Создание панели аналитических метрик"""
        # Сетка 4x4 для метрик
        metrics = [
            ("Общая доходность", "0.00%", self.colors['text_primary']),
            ("Максимальная просадка", "0.00%", self.colors['accent_red']),
            ("Коэффициент Шарпа", "0.00", self.colors['text_primary']),
            ("Коэффициент Сортино", "0.00", self.colors['text_primary']),
            ("Количество сделок", "0", self.colors['text_primary']),
            ("Винрейт", "0.00%", self.colors['accent_green']),
            ("Средняя прибыль", "$0.00", self.colors['accent_green']),
            ("Средний убыток", "$0.00", self.colors['accent_red']),
            ("Коэффициент прибыли", "0.00", self.colors['text_primary']),
            ("Фактор восстановления", "0.00", self.colors['text_primary']),
            ("VaR (95%)", "$0.00", self.colors['accent_orange']),
            ("Калмар-коэффициент", "0.00", self.colors['text_primary'])
        ]
        
        for i, (label, value, color) in enumerate(metrics):
            row = i // 4
            col = (i % 4) * 2
            
            tk.Label(parent, text=f"{label}:",
                   bg=self.colors['bg_secondary'],
                   fg=self.colors['text_secondary'],
                   font=self.fonts['small']).grid(row=row, column=col, sticky='w', padx=5, pady=2)
            
            tk.Label(parent, text=value,
                   bg=self.colors['bg_secondary'],
                   fg=color,
                   font=self.fonts['body']).grid(row=row, column=col+1, sticky='e', padx=5, pady=2)
    
    def create_status_bar(self):
        """Создание статус-бара"""
        # Статус подключения
        self.connection_status = tk.Label(self.status_frame, text="⚫ Отключено",
                                        bg=self.colors['bg_secondary'],
                                        fg=self.colors['accent_red'],
                                        font=self.fonts['small'])
        self.connection_status.pack(side='left', padx=5)
        
        # Последнее обновление
        self.last_update_label = tk.Label(self.status_frame, 
                                        text=f"Обновлено: {datetime.now().strftime('%H:%M:%S')}",
                                        bg=self.colors['bg_secondary'],
                                        fg=self.colors['text_secondary'],
                                        font=self.fonts['small'])
        self.last_update_label.pack(side='right', padx=5)
        
        # Версия
        tk.Label(self.status_frame, text="v2.0",
               bg=self.colors['bg_secondary'],
               fg=self.colors['text_secondary'],
               font=self.fonts['small']).pack(side='right', padx=10)
    
    # Методы обработки событий
    def on_mode_change(self, event=None):
        """Обработка смены режима торговли"""
        mode = self.trading_mode_var.get()
        self.state.trading_mode = mode
        
        if mode == 'backtest':
            self.charts_notebook.select(2)  # Переключиться на вкладку бэктестинга
        
        messagebox.showinfo("Режим изменен", f"Активирован режим: {mode}")
    
    def on_pair_select(self, event=None):
        """Обработка выбора торговых пар"""
        selection = self.pairs_listbox.curselection()
        self.state.selected_pairs = [self.pairs_listbox.get(i) for i in selection]
        
        if self.state.selected_pairs:
            # Обновить график для первой выбранной пары
            self.update_price_chart(self.state.selected_pairs[0])
    
    def filter_pairs(self, event=None):
        """Фильтрация торговых пар"""
        search_term = self.pair_search_var.get().upper()
        
        # Очистить список
        self.pairs_listbox.delete(0, 'end')
        
        # Заполнить отфильтрованными парами
        all_pairs = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 
                    'SOL/USDT', 'DOT/USDT', 'MATIC/USDT', 'LINK/USDT',
                    'AVAX/USDT', 'UNI/USDT', 'ATOM/USDT', 'FTM/USDT']
        
        for pair in all_pairs:
            if search_term in pair:
                self.pairs_listbox.insert('end', pair)
    
    def update_active_strategies(self):
        """Обновление активных стратегий"""
        active = []
        for strategy, var in self.strategy_vars.items():
            if var.get():
                active.append(strategy)
        
        self.state.active_strategies = active
        print(f"Активные стратегии: {active}")
    
    def start_trading(self):
        """Запуск торговли"""
        if not self.state.selected_pairs:
            messagebox.showwarning("Предупреждение", "Выберите торговые пары")
            return
        
        if not self.state.active_strategies:
            messagebox.showwarning("Предупреждение", "Выберите стратегии")
            return
        
        self.state.is_trading_active = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.connection_status.config(text="🟢 Подключено", fg=self.colors['accent_green'])
        
        messagebox.showinfo("Торговля запущена", 
                          f"Режим: {self.state.trading_mode}\n"
                          f"Пары: {', '.join(self.state.selected_pairs)}\n"
                          f"Стратегии: {', '.join(self.state.active_strategies)}")
    
    def stop_trading(self):
        """Остановка торговли"""
        self.state.is_trading_active = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.connection_status.config(text="⚫ Отключено", fg=self.colors['accent_red'])
        
        messagebox.showinfo("Торговля остановлена", "Все активные операции завершены")
    
    def reset_trading(self):
        """Сброс торговых данных"""
        if self.state.is_trading_active:
            messagebox.showwarning("Предупреждение", "Сначала остановите торговлю")
            return
        
        # Сброс состояния
        self.state.current_pnl = Decimal('0')
        self.state.daily_pnl = Decimal('0')
        self.state.active_positions = 0
        
        # Очистка данных
        self.pnl_history.clear()
        self.equity_curve.clear()
        
        # Очистка таблиц
        for item in self.positions_tree.get_children():
            self.positions_tree.delete(item)
        
        for item in self.orders_tree.get_children():
            self.orders_tree.delete(item)
        
        # Обновление интерфейса
        self.update_displays()
        
        messagebox.showinfo("Сброс выполнен", "Все данные сброшены")
    
    def run_backtest(self):
        """Запуск бэктестинга"""
        if not self.state.selected_pairs:
            messagebox.showwarning("Предупреждение", "Выберите торговые пары для бэктеста")
            return
        
        period = self.backtest_period.get()
        
        # Симуляция бэктеста
        messagebox.showinfo("Бэктест запущен", f"Запуск бэктеста на период {period}")
        
        # Генерация тестовых данных
        self.generate_backtest_results()
    
    def generate_backtest_results(self):
        """Генерация результатов бэктеста"""
        # Создание синтетических данных
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)
        
        # Генерация кривой доходности
        returns = np.random.normal(0.001, 0.02, len(dates))
        equity_curve = (1 + pd.Series(returns).cumsum())
        
        # Очистка и построение графика
        self.backtest_ax.clear()
        self.backtest_ax.plot(dates, equity_curve, color=self.colors['accent_green'], linewidth=2)
        self.backtest_ax.set_title('Результаты бэктестинга', color='white', fontsize=14)
        self.backtest_ax.set_ylabel('Кумулятивная доходность', color='white')
        self.backtest_ax.tick_params(colors='white')
        self.backtest_ax.grid(True, alpha=0.3)
        
        # Стилизация
        self.backtest_ax.spines['bottom'].set_color('white')
        self.backtest_ax.spines['top'].set_color('white')
        self.backtest_ax.spines['right'].set_color('white')
        self.backtest_ax.spines['left'].set_color('white')
        
        self.backtest_canvas.draw()
    
    def update_price_chart(self, symbol='BTC/USDT'):
        """Обновление графика цен"""
        # Генерация тестовых данных
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                            end=datetime.now(), freq='H')
        
        # Симуляция цены
        np.random.seed(42)
        base_price = 45000 if symbol == 'BTC/USDT' else 3000
        price_changes = np.random.normal(0, base_price * 0.01, len(dates))
        prices = base_price + np.cumsum(price_changes)
        
        # Очистка и построение
        self.price_ax.clear()
        self.price_ax.plot(dates, prices, color=self.colors['accent_blue'], linewidth=1.5)
        self.price_ax.set_title(f'График цены {symbol}', color='white', fontsize=14)
        self.price_ax.set_ylabel('Цена (USDT)', color='white')
        self.price_ax.tick_params(colors='white')
        self.price_ax.grid(True, alpha=0.3)
        
        # Форматирование дат
        self.price_ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        self.price_ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        
        # Стилизация
        for spine in self.price_ax.spines.values():
            spine.set_color('white')
        
        self.price_canvas.draw()
    
    def update_pnl_chart(self):
        """Обновление P&L графика"""
        if not self.pnl_history:
            # Генерация тестовых данных
            dates = pd.date_range(start=datetime.now() - timedelta(days=7), 
                                end=datetime.now(), freq='15min')
            
            np.random.seed(123)
            pnl_changes = np.random.normal(0, 50, len(dates))
            pnl_curve = np.cumsum(pnl_changes)
            
            self.pnl_history = list(zip(dates, pnl_curve))
        
        if self.pnl_history:
            dates, pnl_values = zip(*self.pnl_history)
            
            self.pnl_ax.clear()
            
            # Цвет линии в зависимости от P&L
            colors = [self.colors['accent_green'] if pnl >= 0 else self.colors['accent_red'] 
                     for pnl in pnl_values]
            
            self.pnl_ax.plot(dates, pnl_values, color=self.colors['accent_green'], linewidth=2)
            self.pnl_ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
            self.pnl_ax.set_title('P&L График', color='white', fontsize=14)
            self.pnl_ax.set_ylabel('P&L (USDT)', color='white')
            self.pnl_ax.tick_params(colors='white')
            self.pnl_ax.grid(True, alpha=0.3)
            
            # Заливка области
            self.pnl_ax.fill_between(dates, pnl_values, 0, 
                                   where=np.array(pnl_values) >= 0,
                                   color=self.colors['accent_green'], alpha=0.3)
            self.pnl_ax.fill_between(dates, pnl_values, 0,
                                   where=np.array(pnl_values) < 0,
                                   color=self.colors['accent_red'], alpha=0.3)
            
            # Стилизация
            for spine in self.pnl_ax.spines.values():
                spine.set_color('white')
            
            self.pnl_canvas.draw()
    
    def update_analytics_charts(self):
        """Обновление аналитических графиков"""
        # Создание subplot'ов для разных графиков
        self.analytics_figure.clear()
        
        # 2x2 сетка графиков
        ax1 = self.analytics_figure.add_subplot(221, facecolor='#2d2d2d')
        ax2 = self.analytics_figure.add_subplot(222, facecolor='#2d2d2d')
        ax3 = self.analytics_figure.add_subplot(223, facecolor='#2d2d2d')
        ax4 = self.analytics_figure.add_subplot(224, facecolor='#2d2d2d')
        
        # График 1: Распределение доходности
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)
        ax1.hist(returns, bins=50, color=self.colors['accent_blue'], alpha=0.7)
        ax1.set_title('Распределение доходности', color='white', fontsize=10)
        ax1.tick_params(colors='white')
        
        # График 2: Просадки
        equity = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        
        ax2.fill_between(range(len(drawdown)), drawdown, 0, 
                        color=self.colors['accent_red'], alpha=0.7)
        ax2.set_title('Просадки (%)', color='white', fontsize=10)
        ax2.tick_params(colors='white')
        
        # График 3: Месячная доходность
        monthly_returns = np.random.normal(0.02, 0.05, 12)
        months = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
                 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
        
        colors = [self.colors['accent_green'] if r >= 0 else self.colors['accent_red'] 
                 for r in monthly_returns]
        ax3.bar(months, monthly_returns * 100, color=colors, alpha=0.7)
        ax3.set_title('Месячная доходность (%)', color='white', fontsize=10)
        ax3.tick_params(colors='white')
        ax3.tick_params(axis='x', rotation=45)
        
        # График 4: Risk-Return scatter
        risks = np.random.uniform(0.1, 0.3, 20)
        returns_annual = np.random.uniform(-0.1, 0.4, 20)
        
        ax4.scatter(risks * 100, returns_annual * 100, 
                   color=self.colors['accent_orange'], alpha=0.7, s=50)
        ax4.set_xlabel('Риск (%)', color='white')
        ax4.set_ylabel('Доходность (%)', color='white')
        ax4.set_title('Риск-Доходность', color='white', fontsize=10)
        ax4.tick_params(colors='white')
        
        # Стилизация всех осей
        for ax in [ax1, ax2, ax3, ax4]:
            for spine in ax.spines.values():
                spine.set_color('white')
            ax.grid(True, alpha=0.3)
        
        self.analytics_figure.tight_layout()
        self.analytics_canvas.draw()
    
    def update_displays(self):
        """Обновление всех дисплеев"""
        # Обновление лейблов в toolbar
        self.balance_label.config(text=f"Баланс: ${self.state.total_balance:,.2f}")
        
        pnl_color = self.colors['accent_green'] if self.state.current_pnl >= 0 else self.colors['accent_red']
        self.pnl_label.config(text=f"P&L: ${self.state.current_pnl:,.2f}", fg=pnl_color)
        
        # Обновление метрик производительности
        self.total_pnl_label.config(text=f"${self.state.current_pnl:,.2f}")
        self.daily_pnl_label.config(text=f"${self.state.daily_pnl:,.2f}")
        
        # Обновление времени
        self.last_update_label.config(text=f"Обновлено: {datetime.now().strftime('%H:%M:%S')}")
    
    def start_data_updates(self):
        """Запуск периодических обновлений данных"""
        def update_loop():
            if self.state.is_trading_active:
                # Симуляция изменения P&L
                change = np.random.normal(0, 10)
                self.state.current_pnl += Decimal(str(change))
                self.state.daily_pnl += Decimal(str(change))
                
                # Добавление точки в P&L историю
                self.pnl_history.append((datetime.now(), float(self.state.current_pnl)))
                
                # Ограничение истории
                if len(self.pnl_history) > 500:
                    self.pnl_history = self.pnl_history[-400:]
                
                # Обновление графика P&L
                self.update_pnl_chart()
            
            # Обновление дисплеев
            self.update_displays()
            
            # Планирование следующего обновления
            self.root.after(1000, update_loop)
        
        # Запуск цикла обновлений
        self.root.after(1000, update_loop)
    
    def run(self):
        """Запуск дашборда"""
        self.root.mainloop()

if __name__ == "__main__":
    # Создание и запуск дашборда
    dashboard = ModernTradingDashboard()
    dashboard.run()