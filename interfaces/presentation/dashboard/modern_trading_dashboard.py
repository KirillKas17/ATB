"""
Современный торговый дашборд в стиле Apple для live-демонстрации на Twitch.
Темная тема, плавные анимации, максимум live-данных.
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
from decimal import Decimal

# Apple-style темная палитра
COLORS = {
    'bg_primary': '#000000',        # Черный фон
    'bg_secondary': '#1C1C1E',      # Темно-серый
    'bg_tertiary': '#2C2C2E',       # Серый
    'accent_blue': '#007AFF',       # Apple Blue
    'accent_green': '#30D158',      # Apple Green (прибыль)
    'accent_red': '#FF453A',        # Apple Red (убыток)
    'accent_orange': '#FF9F0A',     # Apple Orange (предупреждения)
    'accent_purple': '#AF52DE',     # Apple Purple (AI/ML)
    'text_primary': '#FFFFFF',      # Белый текст
    'text_secondary': '#8E8E93',    # Серый текст
    'text_tertiary': '#48484A',     # Темно-серый текст
    'separator': '#38383A',         # Разделители
}

@dataclass
class LiveMetric:
    """Живая метрика для отображения"""
    name: str
    value: float
    change: float = 0.0
    color: str = COLORS['accent_blue']
    format_str: str = "{:.2f}"
    suffix: str = ""
    
class AnimatedLabel(tk.Label):
    """Лэйбл с анимацией изменения значений"""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.target_value = 0.0
        self.current_value = 0.0
        self.animation_speed = 0.1
        
    def animate_to(self, value: float, color: str = None):
        """Анимированное изменение значения"""
        self.target_value = value
        if color:
            self.configure(fg=color)
        self._animate()
        
    def _animate(self):
        """Внутренняя анимация"""
        if abs(self.current_value - self.target_value) > 0.01:
            diff = self.target_value - self.current_value
            self.current_value += diff * self.animation_speed
            self.configure(text=f"{self.current_value:.2f}")
            self.after(16, self._animate)  # ~60 FPS
        else:
            self.current_value = self.target_value
            self.configure(text=f"{self.current_value:.2f}")

class ModernTradingDashboard:
    """Современный торговый дашборд"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.setup_window()
        self.setup_styles()
        
        # Live данные
        self.live_data = {
            'portfolio_value': 50000.0,
            'daily_pnl': 1250.0,
            'total_trades': 127,
            'win_rate': 68.5,
            'active_positions': 8,
            'ai_confidence': 87.3,
            'market_sentiment': 65.2,
            'volatility': 23.4,
            'prices': {
                'BTCUSDT': 42150.0,
                'ETHUSDT': 2580.0,
                'ADAUSDT': 0.485,
                'SOLUSDT': 98.45,
            },
            'orderbook_data': [],
            'trade_history': [],
            'ai_signals': [],
        }
        
        # Графики
        self.price_history = {symbol: [] for symbol in self.live_data['prices']}
        self.pnl_history = []
        
        self.create_layout()
        self.start_live_updates()
        
    def setup_window(self):
        """Настройка основного окна"""
        self.root.title("ATB Trading Dashboard - Live Demo")
        self.root.configure(bg=COLORS['bg_primary'])
        
        # Полноэкранный режим для демонстрации
        self.root.state('zoomed')  # Windows
        # self.root.attributes('-zoomed', True)  # Linux
        
        # Стиль окна
        self.root.configure(borderwidth=0, highlightthickness=0)
        
    def setup_styles(self):
        """Настройка стилей в стиле Apple"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Настройка темной темы
        style.configure('Dark.TFrame', background=COLORS['bg_secondary'])
        style.configure('Card.TFrame', background=COLORS['bg_tertiary'], 
                       relief='flat', borderwidth=1)
        style.configure('Header.TLabel', background=COLORS['bg_primary'],
                       foreground=COLORS['text_primary'], font=('SF Pro Display', 24, 'bold'))
        style.configure('Metric.TLabel', background=COLORS['bg_tertiary'],
                       foreground=COLORS['text_primary'], font=('SF Pro Display', 14))
        style.configure('Value.TLabel', background=COLORS['bg_tertiary'],
                       foreground=COLORS['accent_blue'], font=('SF Pro Display', 18, 'bold'))
        
    def create_layout(self):
        """Создание основного макета"""
        # Главный контейнер
        main_frame = tk.Frame(self.root, bg=COLORS['bg_primary'])
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Заголовок с live-статусом
        self.create_header(main_frame)
        
        # Основной контент в 3 колонки
        content_frame = tk.Frame(main_frame, bg=COLORS['bg_primary'])
        content_frame.pack(fill='both', expand=True, pady=(20, 0))
        
        # Левая колонка - метрики и портфель
        left_frame = tk.Frame(content_frame, bg=COLORS['bg_primary'])
        left_frame.pack(side='left', fill='both', expand=False, padx=(0, 10))
        
        # Центральная колонка - графики
        center_frame = tk.Frame(content_frame, bg=COLORS['bg_primary'])
        center_frame.pack(side='left', fill='both', expand=True, padx=10)
        
        # Правая колонка - ордербук и торги
        right_frame = tk.Frame(content_frame, bg=COLORS['bg_primary'])
        right_frame.pack(side='right', fill='both', expand=False, padx=(10, 0))
        
        # Создание секций
        self.create_metrics_section(left_frame)
        self.create_portfolio_section(left_frame)
        self.create_ai_section(left_frame)
        
        self.create_charts_section(center_frame)
        self.create_market_analysis_section(center_frame)
        
        self.create_orderbook_section(right_frame)
        self.create_trades_section(right_frame)
        self.create_signals_section(right_frame)
        
    def create_header(self, parent):
        """Создание заголовка с live-индикаторами"""
        header_frame = tk.Frame(parent, bg=COLORS['bg_primary'], height=80)
        header_frame.pack(fill='x', pady=(0, 20))
        header_frame.pack_propagate(False)
        
        # Заголовок
        title_label = tk.Label(header_frame, text="ATB Trading System", 
                              bg=COLORS['bg_primary'], fg=COLORS['text_primary'],
                              font=('SF Pro Display', 28, 'bold'))
        title_label.pack(side='left', pady=20)
        
        # Live индикаторы
        live_frame = tk.Frame(header_frame, bg=COLORS['bg_primary'])
        live_frame.pack(side='right', pady=20)
        
        # Live статус
        self.live_indicator = tk.Label(live_frame, text="● LIVE", 
                                     bg=COLORS['bg_primary'], fg=COLORS['accent_red'],
                                     font=('SF Pro Display', 14, 'bold'))
        self.live_indicator.pack(side='right', padx=(0, 20))
        
        # Время
        self.time_label = tk.Label(live_frame, text="", 
                                 bg=COLORS['bg_primary'], fg=COLORS['text_secondary'],
                                 font=('SF Pro Mono', 12))
        self.time_label.pack(side='right', padx=(0, 20))
        
        # Статус подключения
        self.connection_label = tk.Label(live_frame, text="🟢 Connected", 
                                       bg=COLORS['bg_primary'], fg=COLORS['accent_green'],
                                       font=('SF Pro Display', 12))
        self.connection_label.pack(side='right', padx=(0, 20))
        
    def create_metrics_section(self, parent):
        """Секция основных метрик"""
        metrics_frame = self.create_card(parent, "Performance Metrics", height=200)
        
        metrics = [
            ("Portfolio Value", self.live_data['portfolio_value'], "$", COLORS['accent_blue']),
            ("Daily P&L", self.live_data['daily_pnl'], "$", COLORS['accent_green']),
            ("Win Rate", self.live_data['win_rate'], "%", COLORS['accent_green']),
            ("Active Trades", self.live_data['active_positions'], "", COLORS['accent_orange']),
        ]
        
        self.metric_labels = {}
        
        for i, (name, value, suffix, color) in enumerate(metrics):
            row = i // 2
            col = i % 2
            
            metric_container = tk.Frame(metrics_frame, bg=COLORS['bg_tertiary'])
            metric_container.grid(row=row, column=col, padx=5, pady=5, sticky='ew')
            
            name_label = tk.Label(metric_container, text=name,
                                bg=COLORS['bg_tertiary'], fg=COLORS['text_secondary'],
                                font=('SF Pro Display', 10))
            name_label.pack()
            
            value_label = AnimatedLabel(metric_container,
                                      bg=COLORS['bg_tertiary'], fg=color,
                                      font=('SF Pro Display', 16, 'bold'))
            value_label.pack()
            value_label.animate_to(value)
            
            self.metric_labels[name] = value_label
            
        # Настройка grid
        metrics_frame.grid_columnconfigure(0, weight=1)
        metrics_frame.grid_columnconfigure(1, weight=1)
        
    def create_portfolio_section(self, parent):
        """Секция портфеля"""
        portfolio_frame = self.create_card(parent, "Portfolio Holdings", height=250)
        
        # Заголовки
        headers = ["Asset", "Amount", "Value", "P&L"]
        for i, header in enumerate(headers):
            label = tk.Label(portfolio_frame, text=header,
                           bg=COLORS['bg_tertiary'], fg=COLORS['text_secondary'],
                           font=('SF Pro Display', 10, 'bold'))
            label.grid(row=0, column=i, padx=5, pady=5, sticky='w')
            
        # Данные портфеля
        holdings = [
            ("BTC", "0.5432", "$22,890", "+$1,250"),
            ("ETH", "8.7654", "$21,456", "+$890"),
            ("ADA", "15,432", "$7,456", "-$123"),
            ("SOL", "45.23", "$4,456", "+$234"),
        ]
        
        for i, (asset, amount, value, pnl) in enumerate(holdings, 1):
            tk.Label(portfolio_frame, text=asset,
                    bg=COLORS['bg_tertiary'], fg=COLORS['text_primary'],
                    font=('SF Pro Display', 11, 'bold')).grid(row=i, column=0, padx=5, pady=2, sticky='w')
                    
            tk.Label(portfolio_frame, text=amount,
                    bg=COLORS['bg_tertiary'], fg=COLORS['text_primary'],
                    font=('SF Pro Mono', 10)).grid(row=i, column=1, padx=5, pady=2, sticky='w')
                    
            tk.Label(portfolio_frame, text=value,
                    bg=COLORS['bg_tertiary'], fg=COLORS['text_primary'],
                    font=('SF Pro Mono', 10)).grid(row=i, column=2, padx=5, pady=2, sticky='w')
                    
            pnl_color = COLORS['accent_green'] if '+' in pnl else COLORS['accent_red']
            tk.Label(portfolio_frame, text=pnl,
                    bg=COLORS['bg_tertiary'], fg=pnl_color,
                    font=('SF Pro Mono', 10, 'bold')).grid(row=i, column=3, padx=5, pady=2, sticky='w')
                    
    def create_ai_section(self, parent):
        """Секция AI/ML метрик"""
        ai_frame = self.create_card(parent, "AI Intelligence", height=150)
        
        # AI метрики с анимированными прогресс-барами
        ai_metrics = [
            ("AI Confidence", self.live_data['ai_confidence'], COLORS['accent_purple']),
            ("Market Sentiment", self.live_data['market_sentiment'], COLORS['accent_blue']),
            ("Volatility", self.live_data['volatility'], COLORS['accent_orange']),
        ]
        
        self.ai_progress_bars = {}
        
        for i, (name, value, color) in enumerate(ai_metrics):
            # Контейнер для метрики
            container = tk.Frame(ai_frame, bg=COLORS['bg_tertiary'])
            container.pack(fill='x', padx=5, pady=5)
            
            # Заголовок и значение
            header_frame = tk.Frame(container, bg=COLORS['bg_tertiary'])
            header_frame.pack(fill='x')
            
            tk.Label(header_frame, text=name,
                    bg=COLORS['bg_tertiary'], fg=COLORS['text_secondary'],
                    font=('SF Pro Display', 10)).pack(side='left')
                    
            value_label = tk.Label(header_frame, text=f"{value:.1f}%",
                                 bg=COLORS['bg_tertiary'], fg=color,
                                 font=('SF Pro Mono', 10, 'bold'))
            value_label.pack(side='right')
            
            # Прогресс-бар
            progress_bg = tk.Frame(container, bg=COLORS['separator'], height=4)
            progress_bg.pack(fill='x', pady=(2, 0))
            
            progress_fill = tk.Frame(progress_bg, bg=color, height=4)
            progress_fill.place(relwidth=value/100)
            
            self.ai_progress_bars[name] = (value_label, progress_fill)
            
    def create_charts_section(self, parent):
        """Секция графиков"""
        charts_frame = self.create_card(parent, "Live Market Data", height=400)
        
        # Создание matplotlib figure
        self.fig = Figure(figsize=(12, 6), facecolor=COLORS['bg_tertiary'])
        self.fig.patch.set_facecolor(COLORS['bg_tertiary'])
        
        # Создание субплотов
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # График цен
        self.price_ax = self.fig.add_subplot(gs[0, :])
        self.price_ax.set_facecolor(COLORS['bg_secondary'])
        self.price_ax.set_title('BTC/USDT Price', color=COLORS['text_primary'], fontsize=12)
        
        # График P&L
        self.pnl_ax = self.fig.add_subplot(gs[1, 0])
        self.pnl_ax.set_facecolor(COLORS['bg_secondary'])
        self.pnl_ax.set_title('Portfolio P&L', color=COLORS['text_primary'], fontsize=10)
        
        # График объемов
        self.volume_ax = self.fig.add_subplot(gs[1, 1])
        self.volume_ax.set_facecolor(COLORS['bg_secondary'])
        self.volume_ax.set_title('Trading Volume', color=COLORS['text_primary'], fontsize=10)
        
        # Настройка стилей графиков
        for ax in [self.price_ax, self.pnl_ax, self.volume_ax]:
            ax.tick_params(colors=COLORS['text_secondary'], labelsize=8)
            ax.spines['bottom'].set_color(COLORS['separator'])
            ax.spines['top'].set_color(COLORS['separator'])
            ax.spines['right'].set_color(COLORS['separator'])
            ax.spines['left'].set_color(COLORS['separator'])
            ax.grid(True, alpha=0.1, color=COLORS['text_tertiary'])
            
        # Интеграция с tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def create_market_analysis_section(self, parent):
        """Секция анализа рынка"""
        analysis_frame = self.create_card(parent, "Market Analysis", height=150)
        
        # Создание текстового поля для live-анализа
        text_frame = tk.Frame(analysis_frame, bg=COLORS['bg_tertiary'])
        text_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.analysis_text = tk.Text(text_frame, 
                                   bg=COLORS['bg_secondary'], 
                                   fg=COLORS['text_primary'],
                                   font=('SF Pro Mono', 9),
                                   wrap=tk.WORD,
                                   state='disabled')
        self.analysis_text.pack(fill='both', expand=True)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(text_frame, command=self.analysis_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.analysis_text.configure(yscrollcommand=scrollbar.set)
        
    def create_orderbook_section(self, parent):
        """Секция ордербука"""
        orderbook_frame = self.create_card(parent, "Order Book", height=300, width=250)
        
        # Заголовки
        headers_frame = tk.Frame(orderbook_frame, bg=COLORS['bg_tertiary'])
        headers_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Label(headers_frame, text="Size", bg=COLORS['bg_tertiary'], 
                fg=COLORS['text_secondary'], font=('SF Pro Display', 9)).pack(side='left')
        tk.Label(headers_frame, text="Price", bg=COLORS['bg_tertiary'], 
                fg=COLORS['text_secondary'], font=('SF Pro Display', 9)).pack()
        tk.Label(headers_frame, text="Size", bg=COLORS['bg_tertiary'], 
                fg=COLORS['text_secondary'], font=('SF Pro Display', 9)).pack(side='right')
                
        # Контейнер для ордербука
        self.orderbook_container = tk.Frame(orderbook_frame, bg=COLORS['bg_tertiary'])
        self.orderbook_container.pack(fill='both', expand=True, padx=5, pady=5)
        
    def create_trades_section(self, parent):
        """Секция последних сделок"""
        trades_frame = self.create_card(parent, "Recent Trades", height=200, width=250)
        
        # Заголовки
        headers = ["Time", "Side", "Price", "Size"]
        headers_frame = tk.Frame(trades_frame, bg=COLORS['bg_tertiary'])
        headers_frame.pack(fill='x', padx=5, pady=5)
        
        for header in headers:
            tk.Label(headers_frame, text=header, bg=COLORS['bg_tertiary'],
                    fg=COLORS['text_secondary'], font=('SF Pro Display', 8)).pack(side='left', expand=True)
                    
        # Контейнер для сделок
        self.trades_container = tk.Frame(trades_frame, bg=COLORS['bg_tertiary'])
        self.trades_container.pack(fill='both', expand=True, padx=5, pady=5)
        
    def create_signals_section(self, parent):
        """Секция AI сигналов"""
        signals_frame = self.create_card(parent, "AI Signals", height=200, width=250)
        
        # Контейнер для сигналов
        self.signals_container = tk.Frame(signals_frame, bg=COLORS['bg_tertiary'])
        self.signals_container.pack(fill='both', expand=True, padx=5, pady=5)
        
    def create_card(self, parent, title: str, height: int = None, width: int = None):
        """Создание карточки в стиле Apple"""
        card_frame = tk.Frame(parent, bg=COLORS['bg_tertiary'], 
                             relief='flat', bd=1, highlightbackground=COLORS['separator'])
        card_frame.pack(fill='x', pady=(0, 10))
        
        if height:
            card_frame.configure(height=height)
            card_frame.pack_propagate(False)
            
        if width:
            card_frame.configure(width=width)
            
        # Заголовок карточки
        title_frame = tk.Frame(card_frame, bg=COLORS['bg_tertiary'], height=30)
        title_frame.pack(fill='x', padx=10, pady=(10, 5))
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text=title,
                              bg=COLORS['bg_tertiary'], fg=COLORS['text_primary'],
                              font=('SF Pro Display', 12, 'bold'))
        title_label.pack(side='left', anchor='w')
        
        # Контент карточки
        content_frame = tk.Frame(card_frame, bg=COLORS['bg_tertiary'])
        content_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        return content_frame
        
    def start_live_updates(self):
        """Запуск live-обновлений"""
        self.update_thread = threading.Thread(target=self._live_update_loop, daemon=True)
        self.update_thread.start()
        
        # Запуск обновления UI
        self.update_ui()
        
    def _live_update_loop(self):
        """Основной цикл обновления live-данных"""
        while True:
            # Симуляция live-данных
            self._simulate_market_data()
            self._simulate_trading_activity()
            self._simulate_ai_analysis()
            
            time.sleep(0.5)  # Обновление каждые 500мс
            
    def _simulate_market_data(self):
        """Симуляция рыночных данных"""
        # Обновление цен с реалистичными изменениями
        for symbol in self.live_data['prices']:
            current_price = self.live_data['prices'][symbol]
            change = random.uniform(-0.02, 0.02)  # ±2% изменение
            new_price = current_price * (1 + change)
            self.live_data['prices'][symbol] = new_price
            
            # Сохранение истории
            if len(self.price_history[symbol]) > 100:
                self.price_history[symbol].pop(0)
            self.price_history[symbol].append(new_price)
            
        # Обновление метрик
        self.live_data['daily_pnl'] += random.uniform(-50, 100)
        self.live_data['portfolio_value'] = 50000 + self.live_data['daily_pnl']
        self.live_data['ai_confidence'] = max(0, min(100, 
            self.live_data['ai_confidence'] + random.uniform(-2, 2)))
        self.live_data['market_sentiment'] = max(0, min(100,
            self.live_data['market_sentiment'] + random.uniform(-3, 3)))
        self.live_data['volatility'] = max(0, min(100,
            self.live_data['volatility'] + random.uniform(-1, 1)))
            
    def _simulate_trading_activity(self):
        """Симуляция торговой активности"""
        # Генерация новых сделок
        if random.random() < 0.3:  # 30% шанс новой сделки
            trade = {
                'time': datetime.now().strftime('%H:%M:%S'),
                'side': random.choice(['BUY', 'SELL']),
                'price': self.live_data['prices']['BTCUSDT'] + random.uniform(-10, 10),
                'size': random.uniform(0.001, 0.1)
            }
            
            if len(self.live_data['trade_history']) > 20:
                self.live_data['trade_history'].pop(0)
            self.live_data['trade_history'].append(trade)
            
    def _simulate_ai_analysis(self):
        """Симуляция AI анализа"""
        if random.random() < 0.1:  # 10% шанс нового сигнала
            signals = [
                "🔴 Strong resistance at $42,500",
                "🟢 Bullish divergence detected",
                "🟡 High volatility period starting",
                "🔵 Volume spike in BTC",
                "🟢 Long signal: BTC/USDT",
                "🔴 Risk management: Reduce position",
                "🟡 Market sentiment turning bearish",
                "🔵 Whale activity detected",
            ]
            
            signal = {
                'time': datetime.now().strftime('%H:%M:%S'),
                'message': random.choice(signals),
                'confidence': random.uniform(60, 95)
            }
            
            if len(self.live_data['ai_signals']) > 10:
                self.live_data['ai_signals'].pop(0)
            self.live_data['ai_signals'].append(signal)
            
    def update_ui(self):
        """Обновление UI элементов"""
        try:
            # Обновление времени
            current_time = datetime.now().strftime('%H:%M:%S')
            self.time_label.configure(text=current_time)
            
            # Мигание live индикатора
            current_color = self.live_indicator.cget('fg')
            new_color = COLORS['accent_red'] if current_color == COLORS['bg_primary'] else COLORS['bg_primary']
            self.live_indicator.configure(fg=new_color)
            
            # Обновление метрик
            self._update_metrics()
            
            # Обновление графиков
            self._update_charts()
            
            # Обновление ордербука
            self._update_orderbook()
            
            # Обновление сделок
            self._update_trades()
            
            # Обновление сигналов
            self._update_signals()
            
            # Обновление анализа
            self._update_analysis()
            
            # Обновление AI метрик
            self._update_ai_metrics()
            
        except Exception as e:
            print(f"UI Update Error: {e}")
            
        # Планирование следующего обновления
        self.root.after(250, self.update_ui)  # 4 FPS для UI
        
    def _update_metrics(self):
        """Обновление основных метрик"""
        metrics_data = {
            "Portfolio Value": self.live_data['portfolio_value'],
            "Daily P&L": self.live_data['daily_pnl'],
            "Win Rate": self.live_data['win_rate'],
            "Active Trades": self.live_data['active_positions'],
        }
        
        for name, value in metrics_data.items():
            if name in self.metric_labels:
                color = COLORS['accent_green'] if value > 0 else COLORS['accent_red']
                if name in ["Portfolio Value", "Active Trades"]:
                    color = COLORS['accent_blue']
                elif name == "Win Rate":
                    color = COLORS['accent_green']
                    
                self.metric_labels[name].animate_to(value, color)
                
    def _update_charts(self):
        """Обновление графиков"""
        try:
            # Очистка графиков
            self.price_ax.clear()
            self.pnl_ax.clear()
            self.volume_ax.clear()
            
            # График цены BTC
            if self.price_history['BTCUSDT']:
                x_data = range(len(self.price_history['BTCUSDT']))
                y_data = self.price_history['BTCUSDT']
                
                self.price_ax.plot(x_data, y_data, color=COLORS['accent_blue'], linewidth=2)
                self.price_ax.set_title('BTC/USDT Price', color=COLORS['text_primary'], fontsize=12)
                self.price_ax.tick_params(colors=COLORS['text_secondary'], labelsize=8)
                self.price_ax.set_facecolor(COLORS['bg_secondary'])
                
            # График P&L
            if len(self.pnl_history) > 50:
                self.pnl_history.pop(0)
            self.pnl_history.append(self.live_data['daily_pnl'])
            
            if self.pnl_history:
                x_pnl = range(len(self.pnl_history))
                colors = [COLORS['accent_green'] if p > 0 else COLORS['accent_red'] for p in self.pnl_history]
                self.pnl_ax.bar(x_pnl, self.pnl_history, color=colors, alpha=0.7)
                self.pnl_ax.set_title('Portfolio P&L', color=COLORS['text_primary'], fontsize=10)
                self.pnl_ax.tick_params(colors=COLORS['text_secondary'], labelsize=8)
                self.pnl_ax.set_facecolor(COLORS['bg_secondary'])
                
            # График объемов (симуляция)
            volumes = [random.uniform(100, 1000) for _ in range(20)]
            x_vol = range(len(volumes))
            self.volume_ax.bar(x_vol, volumes, color=COLORS['accent_purple'], alpha=0.7)
            self.volume_ax.set_title('Trading Volume', color=COLORS['text_primary'], fontsize=10)
            self.volume_ax.tick_params(colors=COLORS['text_secondary'], labelsize=8)
            self.volume_ax.set_facecolor(COLORS['bg_secondary'])
            
            # Обновление канваса
            self.canvas.draw()
            
        except Exception as e:
            print(f"Chart update error: {e}")
            
    def _update_orderbook(self):
        """Обновление ордербука"""
        # Очистка контейнера
        for widget in self.orderbook_container.winfo_children():
            widget.destroy()
            
        # Генерация ордербука
        base_price = self.live_data['prices']['BTCUSDT']
        
        # Asks (продажи) - сверху
        for i in range(5):
            price = base_price + (i + 1) * random.uniform(1, 5)
            size = random.uniform(0.1, 2.0)
            
            row = tk.Frame(self.orderbook_container, bg=COLORS['bg_tertiary'])
            row.pack(fill='x', pady=1)
            
            tk.Label(row, text="", bg=COLORS['bg_tertiary']).pack(side='left', expand=True)
            tk.Label(row, text=f"{price:.1f}", bg=COLORS['bg_tertiary'],
                    fg=COLORS['accent_red'], font=('SF Pro Mono', 8)).pack()
            tk.Label(row, text=f"{size:.3f}", bg=COLORS['bg_tertiary'],
                    fg=COLORS['text_primary'], font=('SF Pro Mono', 8)).pack(side='right', expand=True)
                    
        # Разделитель
        separator = tk.Frame(self.orderbook_container, bg=COLORS['separator'], height=2)
        separator.pack(fill='x', pady=5)
        
        # Bids (покупки) - снизу
        for i in range(5):
            price = base_price - (i + 1) * random.uniform(1, 5)
            size = random.uniform(0.1, 2.0)
            
            row = tk.Frame(self.orderbook_container, bg=COLORS['bg_tertiary'])
            row.pack(fill='x', pady=1)
            
            tk.Label(row, text=f"{size:.3f}", bg=COLORS['bg_tertiary'],
                    fg=COLORS['text_primary'], font=('SF Pro Mono', 8)).pack(side='left', expand=True)
            tk.Label(row, text=f"{price:.1f}", bg=COLORS['bg_tertiary'],
                    fg=COLORS['accent_green'], font=('SF Pro Mono', 8)).pack()
            tk.Label(row, text="", bg=COLORS['bg_tertiary']).pack(side='right', expand=True)
            
    def _update_trades(self):
        """Обновление списка сделок"""
        # Очистка контейнера
        for widget in self.trades_container.winfo_children():
            widget.destroy()
            
        # Отображение последних сделок
        for trade in self.live_data['trade_history'][-10:]:
            row = tk.Frame(self.trades_container, bg=COLORS['bg_tertiary'])
            row.pack(fill='x', pady=1)
            
            side_color = COLORS['accent_green'] if trade['side'] == 'BUY' else COLORS['accent_red']
            
            tk.Label(row, text=trade['time'], bg=COLORS['bg_tertiary'],
                    fg=COLORS['text_secondary'], font=('SF Pro Mono', 7)).pack(side='left', expand=True)
            tk.Label(row, text=trade['side'], bg=COLORS['bg_tertiary'],
                    fg=side_color, font=('SF Pro Mono', 7, 'bold')).pack(expand=True)
            tk.Label(row, text=f"{trade['price']:.1f}", bg=COLORS['bg_tertiary'],
                    fg=COLORS['text_primary'], font=('SF Pro Mono', 7)).pack(expand=True)
            tk.Label(row, text=f"{trade['size']:.3f}", bg=COLORS['bg_tertiary'],
                    fg=COLORS['text_primary'], font=('SF Pro Mono', 7)).pack(side='right', expand=True)
                    
    def _update_signals(self):
        """Обновление AI сигналов"""
        # Очистка контейнера
        for widget in self.signals_container.winfo_children():
            widget.destroy()
            
        # Отображение последних сигналов
        for signal in self.live_data['ai_signals'][-8:]:
            row = tk.Frame(self.signals_container, bg=COLORS['bg_secondary'])
            row.pack(fill='x', pady=2, padx=2)
            
            # Время и уверенность
            header = tk.Frame(row, bg=COLORS['bg_secondary'])
            header.pack(fill='x')
            
            tk.Label(header, text=signal['time'], bg=COLORS['bg_secondary'],
                    fg=COLORS['text_secondary'], font=('SF Pro Mono', 7)).pack(side='left')
            tk.Label(header, text=f"{signal['confidence']:.0f}%", bg=COLORS['bg_secondary'],
                    fg=COLORS['accent_purple'], font=('SF Pro Mono', 7, 'bold')).pack(side='right')
                    
            # Сообщение
            tk.Label(row, text=signal['message'], bg=COLORS['bg_secondary'],
                    fg=COLORS['text_primary'], font=('SF Pro Display', 8),
                    wraplength=200, justify='left').pack(fill='x')
                    
    def _update_analysis(self):
        """Обновление анализа рынка"""
        if random.random() < 0.05:  # 5% шанс нового анализа
            analyses = [
                "🔍 Detecting bullish pattern formation...",
                "📊 Volume analysis shows strong buying pressure",
                "⚡ High-frequency trading detected",
                "🎯 Target price: $43,200 (R/R: 2.5)",
                "🛡️ Stop loss adjusted to $41,800",
                "📈 RSI approaching oversold levels",
                "🌊 Fibonacci retracement at 61.8%",
                "🔮 Neural network confidence: 94.2%",
                "⚖️ Risk management: Position size reduced",
                "🏦 Institutional flow: Net buying $2.3M",
            ]
            
            timestamp = datetime.now().strftime('%H:%M:%S')
            new_text = f"[{timestamp}] {random.choice(analyses)}\n"
            
            self.analysis_text.configure(state='normal')
            self.analysis_text.insert(tk.END, new_text)
            
            # Ограничение количества строк
            lines = self.analysis_text.get('1.0', tk.END).split('\n')
            if len(lines) > 20:
                self.analysis_text.delete('1.0', '2.0')
                
            self.analysis_text.configure(state='disabled')
            self.analysis_text.see(tk.END)
            
    def _update_ai_metrics(self):
        """Обновление AI метрик с прогресс-барами"""
        ai_data = {
            "AI Confidence": self.live_data['ai_confidence'],
            "Market Sentiment": self.live_data['market_sentiment'],
            "Volatility": self.live_data['volatility'],
        }
        
        for name, value in ai_data.items():
            if name in self.ai_progress_bars:
                value_label, progress_fill = self.ai_progress_bars[name]
                value_label.configure(text=f"{value:.1f}%")
                progress_fill.place(relwidth=value/100)
                
    def run(self):
        """Запуск дашборда"""
        print("🚀 Launching Modern Trading Dashboard...")
        print("💫 Apple-style Dark Theme Activated")
        print("📺 Twitch Demo Mode Ready")
        print("🔴 Live Data Streaming...")
        
        self.root.mainloop()

if __name__ == "__main__":
    try:
        dashboard = ModernTradingDashboard()
        dashboard.run()
    except Exception as e:
        print(f"❌ Dashboard Error: {e}")
        input("Press Enter to exit...")