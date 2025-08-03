"""
Интегрированное Windows приложение ATB Trading с вкладкой демонстрационного дашборда
и формой настроек для API ключей и переменных.
"""

import sys
import os
import json
import asyncio
import threading
import time
import random
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Any, Optional
import configparser

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

# Добавляем путь к проекту
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

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

class ConfigManager:
    """Менеджер конфигурации для хранения настроек и API ключей"""
    
    def __init__(self):
        self.config_file = Path("config/atb_settings.json")
        self.config_file.parent.mkdir(exist_ok=True)
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации из файла"""
        default_config = {
            "api_settings": {
                "bybit_api_key": "",
                "bybit_secret_key": "",
                "binance_api_key": "",
                "binance_secret_key": "",
                "coinbase_api_key": "",
                "coinbase_secret_key": "",
                "telegram_bot_token": "",
                "telegram_chat_id": "",
                "discord_webhook_url": ""
            },
            "trading_settings": {
                "default_risk_percentage": 2.0,
                "max_positions": 5,
                "stop_loss_percentage": 2.5,
                "take_profit_percentage": 5.0,
                "enable_trailing_stop": True,
                "trailing_stop_percentage": 1.0
            },
            "dashboard_settings": {
                "update_interval_ms": 500,
                "show_animations": True,
                "theme": "dark",
                "auto_save_charts": True,
                "chart_history_length": 100
            },
            "notification_settings": {
                "enable_sound": True,
                "enable_popup": True,
                "enable_telegram": False,
                "enable_discord": False,
                "sound_file": "sounds/notification.wav"
            },
            "demo_data": {
                "portfolio_value": 50000.0,
                "daily_pnl": 1250.0,
                "win_rate": 68.5,
                "active_positions": 8,
                "ai_confidence": 87.3,
                "market_sentiment": 65.2,
                "volatility": 23.4
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # Merge with default config to add missing keys
                    for section, values in default_config.items():
                        if section not in loaded_config:
                            loaded_config[section] = values
                        else:
                            for key, value in values.items():
                                if key not in loaded_config[section]:
                                    loaded_config[section][key] = value
                    return loaded_config
            except Exception as e:
                print(f"Ошибка загрузки конфигурации: {e}")
        
        return default_config
        
    def save_config(self):
        """Сохранение конфигурации в файл"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Ошибка сохранения конфигурации: {e}")
            return False
            
    def get(self, section: str, key: str, default=None):
        """Получение значения из конфигурации"""
        return self.config.get(section, {}).get(key, default)
        
    def set(self, section: str, key: str, value):
        """Установка значения в конфигурации"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        
    def get_section(self, section: str) -> Dict[str, Any]:
        """Получение всей секции конфигурации"""
        return self.config.get(section, {})

class SettingsWindow:
    """Окно настроек для конфигурации API ключей и параметров"""
    
    def __init__(self, parent, config_manager: ConfigManager):
        self.parent = parent
        self.config_manager = config_manager
        self.window = None
        self.widgets = {}
        
    def show(self):
        """Показать окно настроек"""
        if self.window is not None:
            self.window.lift()
            return
            
        self.window = tk.Toplevel(self.parent)
        self.window.title("Настройки ATB Trading")
        self.window.geometry("800x700")
        self.window.configure(bg=COLORS['bg_secondary'])
        
        # Делаем окно модальным
        self.window.transient(self.parent)
        self.window.grab_set()
        
        # Обработчик закрытия окна
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.create_widgets()
        
    def create_widgets(self):
        """Создание элементов интерфейса настроек"""
        # Главный контейнер
        main_frame = tk.Frame(self.window, bg=COLORS['bg_secondary'])
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Заголовок
        title_label = tk.Label(main_frame, text="⚙️ Настройки ATB Trading System",
                             bg=COLORS['bg_secondary'], fg=COLORS['text_primary'],
                             font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Notebook для вкладок
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Настройка стилей для темной темы
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=COLORS['bg_secondary'])
        style.configure('TNotebook.Tab', background=COLORS['bg_tertiary'], 
                       foreground=COLORS['text_primary'])
        
        # Вкладки настроек
        self.create_api_tab()
        self.create_trading_tab()
        self.create_dashboard_tab()
        self.create_notifications_tab()
        
        # Кнопки управления
        buttons_frame = tk.Frame(main_frame, bg=COLORS['bg_secondary'])
        buttons_frame.pack(fill='x', pady=(20, 0))
        
        tk.Button(buttons_frame, text="💾 Сохранить", command=self.save_settings,
                 bg=COLORS['accent_blue'], fg='white', font=('Arial', 10, 'bold'),
                 padx=20, pady=5).pack(side='left', padx=(0, 10))
                 
        tk.Button(buttons_frame, text="🔄 Сбросить", command=self.reset_settings,
                 bg=COLORS['accent_orange'], fg='white', font=('Arial', 10, 'bold'),
                 padx=20, pady=5).pack(side='left', padx=(0, 10))
                 
        tk.Button(buttons_frame, text="❌ Отмена", command=self.on_close,
                 bg=COLORS['accent_red'], fg='white', font=('Arial', 10, 'bold'),
                 padx=20, pady=5).pack(side='right')
        
    def create_api_tab(self):
        """Вкладка настроек API"""
        api_frame = tk.Frame(self.notebook, bg=COLORS['bg_secondary'])
        self.notebook.add(api_frame, text="🔑 API Ключи")
        
        # Scrollable frame
        canvas = tk.Canvas(api_frame, bg=COLORS['bg_secondary'])
        scrollbar = ttk.Scrollbar(api_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=COLORS['bg_secondary'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # API настройки
        api_settings = self.config_manager.get_section("api_settings")
        
        # Bybit
        self.create_api_section(scrollable_frame, "Bybit", "bybit", {
            "api_key": api_settings.get("bybit_api_key", ""),
            "secret_key": api_settings.get("bybit_secret_key", "")
        })
        
        # Binance
        self.create_api_section(scrollable_frame, "Binance", "binance", {
            "api_key": api_settings.get("binance_api_key", ""),
            "secret_key": api_settings.get("binance_secret_key", "")
        })
        
        # Coinbase
        self.create_api_section(scrollable_frame, "Coinbase Pro", "coinbase", {
            "api_key": api_settings.get("coinbase_api_key", ""),
            "secret_key": api_settings.get("coinbase_secret_key", "")
        })
        
        # Уведомления
        self.create_notification_section(scrollable_frame, api_settings)
        
    def create_api_section(self, parent, exchange_name, key_prefix, values):
        """Создание секции настроек для биржи"""
        # Рамка для биржи
        exchange_frame = tk.LabelFrame(parent, text=f"📈 {exchange_name}",
                                      bg=COLORS['bg_tertiary'], fg=COLORS['text_primary'],
                                      font=('Arial', 11, 'bold'), pady=10)
        exchange_frame.pack(fill='x', padx=10, pady=10)
        
        # API Key
        tk.Label(exchange_frame, text="API Key:", bg=COLORS['bg_tertiary'],
                fg=COLORS['text_secondary'], font=('Arial', 9)).grid(row=0, column=0, sticky='w', padx=10)
        
        api_key_var = tk.StringVar(value=values.get("api_key", ""))
        api_key_entry = tk.Entry(exchange_frame, textvariable=api_key_var, width=50,
                               show="*" if values.get("api_key") else None)
        api_key_entry.grid(row=0, column=1, padx=10, pady=5, sticky='ew')
        self.widgets[f"{key_prefix}_api_key"] = api_key_var
        
        # Secret Key
        tk.Label(exchange_frame, text="Secret Key:", bg=COLORS['bg_tertiary'],
                fg=COLORS['text_secondary'], font=('Arial', 9)).grid(row=1, column=0, sticky='w', padx=10)
        
        secret_key_var = tk.StringVar(value=values.get("secret_key", ""))
        secret_key_entry = tk.Entry(exchange_frame, textvariable=secret_key_var, width=50,
                                  show="*" if values.get("secret_key") else None)
        secret_key_entry.grid(row=1, column=1, padx=10, pady=5, sticky='ew')
        self.widgets[f"{key_prefix}_secret_key"] = secret_key_var
        
        # Кнопка тестирования
        tk.Button(exchange_frame, text="🧪 Тест подключения",
                 command=lambda: self.test_api_connection(exchange_name),
                 bg=COLORS['accent_blue'], fg='white', font=('Arial', 8)).grid(row=2, column=1, pady=5, sticky='e')
        
        exchange_frame.grid_columnconfigure(1, weight=1)
        
    def create_notification_section(self, parent, api_settings):
        """Создание секции уведомлений"""
        notif_frame = tk.LabelFrame(parent, text="📬 Уведомления",
                                   bg=COLORS['bg_tertiary'], fg=COLORS['text_primary'],
                                   font=('Arial', 11, 'bold'), pady=10)
        notif_frame.pack(fill='x', padx=10, pady=10)
        
        # Telegram
        tk.Label(notif_frame, text="Telegram Bot Token:", bg=COLORS['bg_tertiary'],
                fg=COLORS['text_secondary']).grid(row=0, column=0, sticky='w', padx=10)
        
        telegram_token_var = tk.StringVar(value=api_settings.get("telegram_bot_token", ""))
        tk.Entry(notif_frame, textvariable=telegram_token_var, width=50).grid(row=0, column=1, padx=10, pady=2, sticky='ew')
        self.widgets["telegram_bot_token"] = telegram_token_var
        
        tk.Label(notif_frame, text="Telegram Chat ID:", bg=COLORS['bg_tertiary'],
                fg=COLORS['text_secondary']).grid(row=1, column=0, sticky='w', padx=10)
        
        telegram_chat_var = tk.StringVar(value=api_settings.get("telegram_chat_id", ""))
        tk.Entry(notif_frame, textvariable=telegram_chat_var, width=50).grid(row=1, column=1, padx=10, pady=2, sticky='ew')
        self.widgets["telegram_chat_id"] = telegram_chat_var
        
        # Discord
        tk.Label(notif_frame, text="Discord Webhook URL:", bg=COLORS['bg_tertiary'],
                fg=COLORS['text_secondary']).grid(row=2, column=0, sticky='w', padx=10)
        
        discord_webhook_var = tk.StringVar(value=api_settings.get("discord_webhook_url", ""))
        tk.Entry(notif_frame, textvariable=discord_webhook_var, width=50).grid(row=2, column=1, padx=10, pady=2, sticky='ew')
        self.widgets["discord_webhook_url"] = discord_webhook_var
        
        notif_frame.grid_columnconfigure(1, weight=1)
        
    def create_trading_tab(self):
        """Вкладка торговых настроек"""
        trading_frame = tk.Frame(self.notebook, bg=COLORS['bg_secondary'])
        self.notebook.add(trading_frame, text="💰 Торговля")
        
        trading_settings = self.config_manager.get_section("trading_settings")
        
        # Риск-менеджмент
        risk_frame = tk.LabelFrame(trading_frame, text="⚖️ Риск-менеджмент",
                                  bg=COLORS['bg_tertiary'], fg=COLORS['text_primary'],
                                  font=('Arial', 11, 'bold'))
        risk_frame.pack(fill='x', padx=10, pady=10)
        
        # Размер риска
        tk.Label(risk_frame, text="Риск на сделку (%):", bg=COLORS['bg_tertiary'],
                fg=COLORS['text_secondary']).grid(row=0, column=0, sticky='w', padx=10, pady=5)
        
        risk_var = tk.DoubleVar(value=trading_settings.get("default_risk_percentage", 2.0))
        risk_spinbox = tk.Spinbox(risk_frame, from_=0.1, to=10.0, increment=0.1, 
                                 textvariable=risk_var, width=10)
        risk_spinbox.grid(row=0, column=1, padx=10, pady=5, sticky='w')
        self.widgets["default_risk_percentage"] = risk_var
        
        # Максимум позиций
        tk.Label(risk_frame, text="Макс. позиций:", bg=COLORS['bg_tertiary'],
                fg=COLORS['text_secondary']).grid(row=1, column=0, sticky='w', padx=10, pady=5)
        
        max_positions_var = tk.IntVar(value=trading_settings.get("max_positions", 5))
        tk.Spinbox(risk_frame, from_=1, to=20, textvariable=max_positions_var, width=10).grid(row=1, column=1, padx=10, pady=5, sticky='w')
        self.widgets["max_positions"] = max_positions_var
        
        # Стоп-лосс
        tk.Label(risk_frame, text="Стоп-лосс (%):", bg=COLORS['bg_tertiary'],
                fg=COLORS['text_secondary']).grid(row=2, column=0, sticky='w', padx=10, pady=5)
        
        stop_loss_var = tk.DoubleVar(value=trading_settings.get("stop_loss_percentage", 2.5))
        tk.Spinbox(risk_frame, from_=0.5, to=10.0, increment=0.1, 
                  textvariable=stop_loss_var, width=10).grid(row=2, column=1, padx=10, pady=5, sticky='w')
        self.widgets["stop_loss_percentage"] = stop_loss_var
        
        # Тейк-профит
        tk.Label(risk_frame, text="Тейк-профит (%):", bg=COLORS['bg_tertiary'],
                fg=COLORS['text_secondary']).grid(row=3, column=0, sticky='w', padx=10, pady=5)
        
        take_profit_var = tk.DoubleVar(value=trading_settings.get("take_profit_percentage", 5.0))
        tk.Spinbox(risk_frame, from_=1.0, to=20.0, increment=0.1, 
                  textvariable=take_profit_var, width=10).grid(row=3, column=1, padx=10, pady=5, sticky='w')
        self.widgets["take_profit_percentage"] = take_profit_var
        
        # Трейлинг стоп
        trailing_var = tk.BooleanVar(value=trading_settings.get("enable_trailing_stop", True))
        tk.Checkbutton(risk_frame, text="Включить трейлинг стоп", variable=trailing_var,
                      bg=COLORS['bg_tertiary'], fg=COLORS['text_primary'],
                      selectcolor=COLORS['bg_secondary']).grid(row=4, column=0, columnspan=2, sticky='w', padx=10, pady=5)
        self.widgets["enable_trailing_stop"] = trailing_var
        
    def create_dashboard_tab(self):
        """Вкладка настроек дашборда"""
        dashboard_frame = tk.Frame(self.notebook, bg=COLORS['bg_secondary'])
        self.notebook.add(dashboard_frame, text="📊 Дашборд")
        
        dashboard_settings = self.config_manager.get_section("dashboard_settings")
        
        # Интерфейс
        ui_frame = tk.LabelFrame(dashboard_frame, text="🎨 Интерфейс",
                                bg=COLORS['bg_tertiary'], fg=COLORS['text_primary'],
                                font=('Arial', 11, 'bold'))
        ui_frame.pack(fill='x', padx=10, pady=10)
        
        # Интервал обновления
        tk.Label(ui_frame, text="Интервал обновления (мс):", bg=COLORS['bg_tertiary'],
                fg=COLORS['text_secondary']).grid(row=0, column=0, sticky='w', padx=10, pady=5)
        
        update_interval_var = tk.IntVar(value=dashboard_settings.get("update_interval_ms", 500))
        tk.Spinbox(ui_frame, from_=100, to=5000, increment=100, 
                  textvariable=update_interval_var, width=10).grid(row=0, column=1, padx=10, pady=5, sticky='w')
        self.widgets["update_interval_ms"] = update_interval_var
        
        # Анимации
        animations_var = tk.BooleanVar(value=dashboard_settings.get("show_animations", True))
        tk.Checkbutton(ui_frame, text="Показывать анимации", variable=animations_var,
                      bg=COLORS['bg_tertiary'], fg=COLORS['text_primary'],
                      selectcolor=COLORS['bg_secondary']).grid(row=1, column=0, columnspan=2, sticky='w', padx=10, pady=5)
        self.widgets["show_animations"] = animations_var
        
        # Автосохранение графиков
        auto_save_var = tk.BooleanVar(value=dashboard_settings.get("auto_save_charts", True))
        tk.Checkbutton(ui_frame, text="Автосохранение графиков", variable=auto_save_var,
                      bg=COLORS['bg_tertiary'], fg=COLORS['text_primary'],
                      selectcolor=COLORS['bg_secondary']).grid(row=2, column=0, columnspan=2, sticky='w', padx=10, pady=5)
        self.widgets["auto_save_charts"] = auto_save_var
        
    def create_notifications_tab(self):
        """Вкладка настроек уведомлений"""
        notif_frame = tk.Frame(self.notebook, bg=COLORS['bg_secondary'])
        self.notebook.add(notif_frame, text="🔔 Уведомления")
        
        notif_settings = self.config_manager.get_section("notification_settings")
        
        # Типы уведомлений
        types_frame = tk.LabelFrame(notif_frame, text="🔊 Типы уведомлений",
                                   bg=COLORS['bg_tertiary'], fg=COLORS['text_primary'],
                                   font=('Arial', 11, 'bold'))
        types_frame.pack(fill='x', padx=10, pady=10)
        
        # Звук
        sound_var = tk.BooleanVar(value=notif_settings.get("enable_sound", True))
        tk.Checkbutton(types_frame, text="🔊 Звуковые уведомления", variable=sound_var,
                      bg=COLORS['bg_tertiary'], fg=COLORS['text_primary'],
                      selectcolor=COLORS['bg_secondary']).pack(anchor='w', padx=10, pady=2)
        self.widgets["enable_sound"] = sound_var
        
        # Popup
        popup_var = tk.BooleanVar(value=notif_settings.get("enable_popup", True))
        tk.Checkbutton(types_frame, text="💬 Всплывающие окна", variable=popup_var,
                      bg=COLORS['bg_tertiary'], fg=COLORS['text_primary'],
                      selectcolor=COLORS['bg_secondary']).pack(anchor='w', padx=10, pady=2)
        self.widgets["enable_popup"] = popup_var
        
        # Telegram
        telegram_var = tk.BooleanVar(value=notif_settings.get("enable_telegram", False))
        tk.Checkbutton(types_frame, text="📱 Telegram уведомления", variable=telegram_var,
                      bg=COLORS['bg_tertiary'], fg=COLORS['text_primary'],
                      selectcolor=COLORS['bg_secondary']).pack(anchor='w', padx=10, pady=2)
        self.widgets["enable_telegram"] = telegram_var
        
        # Discord
        discord_var = tk.BooleanVar(value=notif_settings.get("enable_discord", False))
        tk.Checkbutton(types_frame, text="🎮 Discord уведомления", variable=discord_var,
                      bg=COLORS['bg_tertiary'], fg=COLORS['text_primary'],
                      selectcolor=COLORS['bg_secondary']).pack(anchor='w', padx=10, pady=2)
        self.widgets["enable_discord"] = discord_var
        
    def test_api_connection(self, exchange_name):
        """Тестирование подключения к API"""
        messagebox.showinfo("Тест API", f"Тестирование подключения к {exchange_name}...\n(Функция в разработке)")
        
    def save_settings(self):
        """Сохранение настроек"""
        try:
            # API настройки
            self.config_manager.set("api_settings", "bybit_api_key", self.widgets["bybit_api_key"].get())
            self.config_manager.set("api_settings", "bybit_secret_key", self.widgets["bybit_secret_key"].get())
            self.config_manager.set("api_settings", "binance_api_key", self.widgets["binance_api_key"].get())
            self.config_manager.set("api_settings", "binance_secret_key", self.widgets["binance_secret_key"].get())
            self.config_manager.set("api_settings", "coinbase_api_key", self.widgets["coinbase_api_key"].get())
            self.config_manager.set("api_settings", "coinbase_secret_key", self.widgets["coinbase_secret_key"].get())
            self.config_manager.set("api_settings", "telegram_bot_token", self.widgets["telegram_bot_token"].get())
            self.config_manager.set("api_settings", "telegram_chat_id", self.widgets["telegram_chat_id"].get())
            self.config_manager.set("api_settings", "discord_webhook_url", self.widgets["discord_webhook_url"].get())
            
            # Торговые настройки
            if "default_risk_percentage" in self.widgets:
                self.config_manager.set("trading_settings", "default_risk_percentage", self.widgets["default_risk_percentage"].get())
                self.config_manager.set("trading_settings", "max_positions", self.widgets["max_positions"].get())
                self.config_manager.set("trading_settings", "stop_loss_percentage", self.widgets["stop_loss_percentage"].get())
                self.config_manager.set("trading_settings", "take_profit_percentage", self.widgets["take_profit_percentage"].get())
                self.config_manager.set("trading_settings", "enable_trailing_stop", self.widgets["enable_trailing_stop"].get())
            
            # Настройки дашборда
            if "update_interval_ms" in self.widgets:
                self.config_manager.set("dashboard_settings", "update_interval_ms", self.widgets["update_interval_ms"].get())
                self.config_manager.set("dashboard_settings", "show_animations", self.widgets["show_animations"].get())
                self.config_manager.set("dashboard_settings", "auto_save_charts", self.widgets["auto_save_charts"].get())
            
            # Настройки уведомлений
            if "enable_sound" in self.widgets:
                self.config_manager.set("notification_settings", "enable_sound", self.widgets["enable_sound"].get())
                self.config_manager.set("notification_settings", "enable_popup", self.widgets["enable_popup"].get())
                self.config_manager.set("notification_settings", "enable_telegram", self.widgets["enable_telegram"].get())
                self.config_manager.set("notification_settings", "enable_discord", self.widgets["enable_discord"].get())
            
            # Сохранение в файл
            if self.config_manager.save_config():
                messagebox.showinfo("Успех", "✅ Настройки успешно сохранены!")
                self.on_close()
            else:
                messagebox.showerror("Ошибка", "❌ Ошибка сохранения настроек")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"❌ Ошибка сохранения: {e}")
            
    def reset_settings(self):
        """Сброс настроек к значениям по умолчанию"""
        if messagebox.askyesno("Подтверждение", "Сбросить все настройки к значениям по умолчанию?"):
            # Удаляем файл конфигурации и перезагружаем
            if self.config_manager.config_file.exists():
                self.config_manager.config_file.unlink()
            self.config_manager.config = self.config_manager.load_config()
            messagebox.showinfo("Успех", "✅ Настройки сброшены к значениям по умолчанию")
            self.on_close()
            
    def on_close(self):
        """Закрытие окна настроек"""
        if self.window:
            self.window.grab_release()
            self.window.destroy()
            self.window = None

class LiveDemoTab:
    """Вкладка с live-демонстрацией дашборда для Twitch"""
    
    def __init__(self, parent_notebook, config_manager: ConfigManager):
        self.parent_notebook = parent_notebook
        self.config_manager = config_manager
        self.is_running = False
        self.update_thread = None
        
        # Live данные
        demo_data = config_manager.get_section("demo_data")
        self.live_data = {
            'portfolio_value': demo_data.get('portfolio_value', 50000.0),
            'daily_pnl': demo_data.get('daily_pnl', 1250.0),
            'total_trades': 127,
            'win_rate': demo_data.get('win_rate', 68.5),
            'active_positions': demo_data.get('active_positions', 8),
            'ai_confidence': demo_data.get('ai_confidence', 87.3),
            'market_sentiment': demo_data.get('market_sentiment', 65.2),
            'volatility': demo_data.get('volatility', 23.4),
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
        
        # История для графиков
        self.price_history = {symbol: [] for symbol in self.live_data['prices']}
        self.pnl_history = []
        
        self.create_tab()
        
    def create_tab(self):
        """Создание вкладки demo дашборда"""
        # Создаем фрейм для вкладки
        self.demo_frame = tk.Frame(self.parent_notebook, bg=COLORS['bg_primary'])
        self.parent_notebook.add(self.demo_frame, text="📺 Live Demo")
        
        # Заголовок с controls
        self.create_header()
        
        # Главный контент
        self.create_main_content()
        
    def create_header(self):
        """Создание заголовка с элементами управления"""
        header_frame = tk.Frame(self.demo_frame, bg=COLORS['bg_primary'], height=60)
        header_frame.pack(fill='x', padx=20, pady=(20, 0))
        header_frame.pack_propagate(False)
        
        # Заголовок
        title_label = tk.Label(header_frame, text="📺 ATB Trading System - Live Demo",
                              bg=COLORS['bg_primary'], fg=COLORS['text_primary'],
                              font=('Arial', 18, 'bold'))
        title_label.pack(side='left', pady=15)
        
        # Элементы управления
        controls_frame = tk.Frame(header_frame, bg=COLORS['bg_primary'])
        controls_frame.pack(side='right', pady=15)
        
        # Кнопка запуска/остановки
        self.start_stop_btn = tk.Button(controls_frame, text="▶ Запустить Demo",
                                       command=self.toggle_demo, bg=COLORS['accent_green'],
                                       fg='white', font=('Arial', 10, 'bold'), padx=15)
        self.start_stop_btn.pack(side='left', padx=(0, 10))
        
        # Live индикатор
        self.live_indicator = tk.Label(controls_frame, text="⚫ OFFLINE",
                                     bg=COLORS['bg_primary'], fg=COLORS['text_secondary'],
                                     font=('Arial', 10, 'bold'))
        self.live_indicator.pack(side='left', padx=(0, 10))
        
        # Время
        self.time_label = tk.Label(controls_frame, text="",
                                 bg=COLORS['bg_primary'], fg=COLORS['text_secondary'],
                                 font=('Courier', 10))
        self.time_label.pack(side='left')
        
        self.update_time()
        
    def create_main_content(self):
        """Создание основного контента дашборда"""
        # Основной контейнер
        main_frame = tk.Frame(self.demo_frame, bg=COLORS['bg_primary'])
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Разделение на 3 колонки
        # Левая колонка
        left_frame = tk.Frame(main_frame, bg=COLORS['bg_primary'], width=280)
        left_frame.pack(side='left', fill='y', padx=(0, 10))
        left_frame.pack_propagate(False)
        
        # Центральная колонка
        center_frame = tk.Frame(main_frame, bg=COLORS['bg_primary'])
        center_frame.pack(side='left', fill='both', expand=True, padx=10)
        
        # Правая колонка
        right_frame = tk.Frame(main_frame, bg=COLORS['bg_primary'], width=280)
        right_frame.pack(side='right', fill='y', padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # Создание секций
        self.create_metrics_section(left_frame)
        self.create_portfolio_section(left_frame)
        self.create_ai_section(left_frame)
        
        self.create_charts_section(center_frame)
        self.create_analysis_section(center_frame)
        
        self.create_orderbook_section(right_frame)
        self.create_trades_section(right_frame)
        self.create_signals_section(right_frame)
        
    def create_card(self, parent, title: str, height: int = None):
        """Создание карточки в стиле Apple"""
        card_frame = tk.Frame(parent, bg=COLORS['bg_secondary'], relief='flat', bd=1)
        card_frame.pack(fill='x', pady=(0, 10))
        
        if height:
            card_frame.configure(height=height)
            card_frame.pack_propagate(False)
        
        # Заголовок
        title_frame = tk.Frame(card_frame, bg=COLORS['bg_secondary'], height=30)
        title_frame.pack(fill='x', padx=15, pady=(15, 5))
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text=title, bg=COLORS['bg_secondary'],
                              fg=COLORS['text_primary'], font=('Arial', 12, 'bold'))
        title_label.pack(side='left', anchor='w')
        
        # Контент
        content_frame = tk.Frame(card_frame, bg=COLORS['bg_secondary'])
        content_frame.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        
        return content_frame
        
    def create_metrics_section(self, parent):
        """Секция основных метрик"""
        metrics_frame = self.create_card(parent, "📊 Performance Metrics", height=180)
        
        # Grid для метрик 2x2
        metrics = [
            ("Portfolio Value", "portfolio_value", "$", COLORS['accent_blue']),
            ("Daily P&L", "daily_pnl", "$", COLORS['accent_green']),
            ("Win Rate", "win_rate", "%", COLORS['accent_green']),
            ("Active Trades", "active_positions", "", COLORS['accent_orange']),
        ]
        
        self.metric_labels = {}
        
        for i, (name, key, suffix, color) in enumerate(metrics):
            row = i // 2
            col = i % 2
            
            # Контейнер для метрики
            metric_container = tk.Frame(metrics_frame, bg=COLORS['bg_tertiary'])
            metric_container.grid(row=row, column=col, padx=5, pady=5, sticky='ew')
            
            # Название
            name_label = tk.Label(metric_container, text=name, bg=COLORS['bg_tertiary'],
                                fg=COLORS['text_secondary'], font=('Arial', 8))
            name_label.pack()
            
            # Значение
            value_label = tk.Label(metric_container, text=f"{suffix}{self.live_data[key]:.0f}",
                                 bg=COLORS['bg_tertiary'], fg=color,
                                 font=('Arial', 14, 'bold'))
            value_label.pack()
            
            self.metric_labels[key] = (value_label, suffix, color)
            
        # Настройка grid
        metrics_frame.grid_columnconfigure(0, weight=1)
        metrics_frame.grid_columnconfigure(1, weight=1)
        
    def create_portfolio_section(self, parent):
        """Секция портфеля"""
        portfolio_frame = self.create_card(parent, "💼 Portfolio Holdings", height=200)
        
        # Scrollable область
        canvas = tk.Canvas(portfolio_frame, bg=COLORS['bg_secondary'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(portfolio_frame, orient="vertical", command=canvas.yview)
        self.portfolio_content = tk.Frame(canvas, bg=COLORS['bg_secondary'])
        
        self.portfolio_content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.portfolio_content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.update_portfolio()
        
    def create_ai_section(self, parent):
        """Секция AI метрик"""
        ai_frame = self.create_card(parent, "🤖 AI Intelligence", height=150)
        
        ai_metrics = [
            ("AI Confidence", "ai_confidence", COLORS['accent_purple']),
            ("Market Sentiment", "market_sentiment", COLORS['accent_blue']),
            ("Volatility", "volatility", COLORS['accent_orange']),
        ]
        
        self.ai_progress_bars = {}
        
        for i, (name, key, color) in enumerate(ai_metrics):
            # Container для метрики
            container = tk.Frame(ai_frame, bg=COLORS['bg_secondary'])
            container.pack(fill='x', padx=5, pady=8)
            
            # Header с названием и значением
            header_frame = tk.Frame(container, bg=COLORS['bg_secondary'])
            header_frame.pack(fill='x')
            
            tk.Label(header_frame, text=name, bg=COLORS['bg_secondary'],
                    fg=COLORS['text_secondary'], font=('Arial', 9)).pack(side='left')
            
            value_label = tk.Label(header_frame, text=f"{self.live_data[key]:.1f}%",
                                 bg=COLORS['bg_secondary'], fg=color,
                                 font=('Arial', 9, 'bold'))
            value_label.pack(side='right')
            
            # Progress bar
            progress_bg = tk.Frame(container, bg=COLORS['separator'], height=4)
            progress_bg.pack(fill='x', pady=(3, 0))
            
            progress_fill = tk.Frame(progress_bg, bg=color, height=4)
            progress_fill.place(relwidth=self.live_data[key]/100)
            
            self.ai_progress_bars[key] = (value_label, progress_fill, color)
            
    def create_charts_section(self, parent):
        """Секция графиков"""
        charts_frame = self.create_card(parent, "📈 Live Market Data", height=350)
        
        # Создание matplotlib figure
        self.fig = Figure(figsize=(10, 6), facecolor=COLORS['bg_secondary'])
        self.fig.patch.set_facecolor(COLORS['bg_secondary'])
        
        # График цены
        self.price_ax = self.fig.add_subplot(111)
        self.price_ax.set_facecolor(COLORS['bg_primary'])
        self.price_ax.set_title('BTC/USDT Price', color=COLORS['text_primary'], fontsize=12)
        
        # Настройка стилей
        self.price_ax.tick_params(colors=COLORS['text_secondary'], labelsize=8)
        for spine in self.price_ax.spines.values():
            spine.set_color(COLORS['separator'])
        self.price_ax.grid(True, alpha=0.1, color=COLORS['text_tertiary'])
        
        # Интеграция с tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def create_analysis_section(self, parent):
        """Секция анализа рынка"""
        analysis_frame = self.create_card(parent, "📊 Market Analysis", height=120)
        
        # Текстовое поле для анализа
        self.analysis_text = tk.Text(analysis_frame, bg=COLORS['bg_primary'],
                                   fg=COLORS['text_primary'], font=('Courier', 8),
                                   wrap=tk.WORD, state='disabled', height=6)
        self.analysis_text.pack(fill='both', expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(analysis_frame, command=self.analysis_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.analysis_text.configure(yscrollcommand=scrollbar.set)
        
    def create_orderbook_section(self, parent):
        """Секция ордербука"""
        orderbook_frame = self.create_card(parent, "📋 Order Book", height=250)
        
        # Headers
        headers_frame = tk.Frame(orderbook_frame, bg=COLORS['bg_secondary'])
        headers_frame.pack(fill='x', pady=(0, 5))
        
        tk.Label(headers_frame, text="Size", bg=COLORS['bg_secondary'],
                fg=COLORS['text_secondary'], font=('Arial', 8)).pack(side='left')
        tk.Label(headers_frame, text="Price", bg=COLORS['bg_secondary'],
                fg=COLORS['text_secondary'], font=('Arial', 8)).pack()
        tk.Label(headers_frame, text="Size", bg=COLORS['bg_secondary'],
                fg=COLORS['text_secondary'], font=('Arial', 8)).pack(side='right')
        
        # Scrollable content
        canvas = tk.Canvas(orderbook_frame, bg=COLORS['bg_secondary'], highlightthickness=0)
        self.orderbook_content = tk.Frame(canvas, bg=COLORS['bg_secondary'])
        
        canvas.create_window((0, 0), window=self.orderbook_content, anchor="nw")
        canvas.pack(fill='both', expand=True)
        
        self.orderbook_canvas = canvas
        
    def create_trades_section(self, parent):
        """Секция последних сделок"""
        trades_frame = self.create_card(parent, "🔄 Recent Trades", height=180)
        
        # Headers
        headers = ["Time", "Side", "Price", "Size"]
        headers_frame = tk.Frame(trades_frame, bg=COLORS['bg_secondary'])
        headers_frame.pack(fill='x', pady=(0, 5))
        
        for header in headers:
            tk.Label(headers_frame, text=header, bg=COLORS['bg_secondary'],
                    fg=COLORS['text_secondary'], font=('Arial', 8)).pack(side='left', expand=True)
        
        # Scrollable content
        canvas = tk.Canvas(trades_frame, bg=COLORS['bg_secondary'], highlightthickness=0)
        self.trades_content = tk.Frame(canvas, bg=COLORS['bg_secondary'])
        
        canvas.create_window((0, 0), window=self.trades_content, anchor="nw")
        canvas.pack(fill='both', expand=True)
        
        self.trades_canvas = canvas
        
    def create_signals_section(self, parent):
        """Секция AI сигналов"""
        signals_frame = self.create_card(parent, "🤖 AI Signals", height=150)
        
        # Scrollable content
        canvas = tk.Canvas(signals_frame, bg=COLORS['bg_secondary'], highlightthickness=0)
        self.signals_content = tk.Frame(canvas, bg=COLORS['bg_secondary'])
        
        canvas.create_window((0, 0), window=self.signals_content, anchor="nw")
        canvas.pack(fill='both', expand=True)
        
        self.signals_canvas = canvas
        
    def toggle_demo(self):
        """Переключение состояния демонстрации"""
        if self.is_running:
            self.stop_demo()
        else:
            self.start_demo()
            
    def start_demo(self):
        """Запуск live демонстрации"""
        self.is_running = True
        self.start_stop_btn.configure(text="⏸ Остановить Demo", bg=COLORS['accent_red'])
        self.live_indicator.configure(text="🔴 LIVE", fg=COLORS['accent_red'])
        
        # Запуск потока обновлений
        self.update_thread = threading.Thread(target=self._live_update_loop, daemon=True)
        self.update_thread.start()
        
        # Запуск обновления UI
        self.update_ui()
        
    def stop_demo(self):
        """Остановка live демонстрации"""
        self.is_running = False
        self.start_stop_btn.configure(text="▶ Запустить Demo", bg=COLORS['accent_green'])
        self.live_indicator.configure(text="⚫ OFFLINE", fg=COLORS['text_secondary'])
        
    def _live_update_loop(self):
        """Основной цикл обновления live данных"""
        while self.is_running:
            self._simulate_market_data()
            self._simulate_trading_activity()
            self._simulate_ai_analysis()
            
            # Интервал обновления из настроек
            interval = self.config_manager.get("dashboard_settings", "update_interval_ms", 500)
            time.sleep(interval / 1000.0)
            
    def _simulate_market_data(self):
        """Симуляция рыночных данных"""
        # Обновление цен
        for symbol in self.live_data['prices']:
            current_price = self.live_data['prices'][symbol]
            change = random.uniform(-0.02, 0.02)
            new_price = current_price * (1 + change)
            self.live_data['prices'][symbol] = new_price
            
            # История цен
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
        if random.random() < 0.3:
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
        if random.random() < 0.1:
            signals = [
                "🔴 Strong resistance at $42,500",
                "🟢 Bullish divergence detected",
                "🟡 High volatility period starting",
                "🔵 Volume spike in BTC",
                "🟢 Long signal: BTC/USDT",
                "🔴 Risk management: Reduce position",
            ]
            
            signal = {
                'time': datetime.now().strftime('%H:%M:%S'),
                'message': random.choice(signals),
                'confidence': random.uniform(60, 95)
            }
            
            if len(self.live_data['ai_signals']) > 10:
                self.live_data['ai_signals'].pop(0)
            self.live_data['ai_signals'].append(signal)
            
    def update_time(self):
        """Обновление времени"""
        current_time = datetime.now().strftime('%H:%M:%S')
        self.time_label.configure(text=current_time)
        self.demo_frame.after(1000, self.update_time)
        
    def update_ui(self):
        """Обновление UI элементов"""
        if not self.is_running:
            return
            
        try:
            # Обновление метрик
            self._update_metrics()
            
            # Обновление AI метрик
            self._update_ai_metrics()
            
            # Обновление графика
            self._update_chart()
            
            # Обновление ордербука
            self._update_orderbook()
            
            # Обновление сделок
            self._update_trades()
            
            # Обновление сигналов
            self._update_signals()
            
            # Обновление анализа
            self._update_analysis()
            
        except Exception as e:
            print(f"UI Update Error: {e}")
        
        # Планирование следующего обновления
        interval = self.config_manager.get("dashboard_settings", "update_interval_ms", 500)
        self.demo_frame.after(interval, self.update_ui)
        
    def _update_metrics(self):
        """Обновление основных метрик"""
        for key, (label, suffix, color) in self.metric_labels.items():
            value = self.live_data[key]
            if key == 'daily_pnl':
                text = f"{'+' if value >= 0 else ''}{suffix}{value:.0f}"
                color = COLORS['accent_green'] if value >= 0 else COLORS['accent_red']
            else:
                text = f"{suffix}{value:.1f}" if key == 'win_rate' else f"{suffix}{value:.0f}"
            
            label.configure(text=text, fg=color)
            
    def _update_ai_metrics(self):
        """Обновление AI метрик"""
        for key, (label, progress_fill, color) in self.ai_progress_bars.items():
            value = self.live_data[key]
            label.configure(text=f"{value:.1f}%")
            progress_fill.place(relwidth=value/100)
            
    def _update_chart(self):
        """Обновление графика"""
        try:
            if self.price_history['BTCUSDT']:
                self.price_ax.clear()
                
                x_data = range(len(self.price_history['BTCUSDT']))
                y_data = self.price_history['BTCUSDT']
                
                self.price_ax.plot(x_data, y_data, color=COLORS['accent_blue'], linewidth=2)
                self.price_ax.set_title('BTC/USDT Price', color=COLORS['text_primary'], fontsize=12)
                self.price_ax.set_facecolor(COLORS['bg_primary'])
                
                # Стилизация
                self.price_ax.tick_params(colors=COLORS['text_secondary'], labelsize=8)
                for spine in self.price_ax.spines.values():
                    spine.set_color(COLORS['separator'])
                self.price_ax.grid(True, alpha=0.1, color=COLORS['text_tertiary'])
                
                self.canvas.draw()
                
        except Exception as e:
            print(f"Chart update error: {e}")
            
    def _update_orderbook(self):
        """Обновление ордербука"""
        # Очистка
        for widget in self.orderbook_content.winfo_children():
            widget.destroy()
            
        base_price = self.live_data['prices']['BTCUSDT']
        
        # Asks (продажи)
        for i in range(5):
            price = base_price + (i + 1) * random.uniform(1, 5)
            size = random.uniform(0.1, 2.0)
            
            row = tk.Frame(self.orderbook_content, bg=COLORS['bg_secondary'])
            row.pack(fill='x', pady=1)
            
            tk.Label(row, text="", bg=COLORS['bg_secondary'], width=8).pack(side='left')
            tk.Label(row, text=f"{price:.1f}", bg=COLORS['bg_secondary'],
                    fg=COLORS['accent_red'], font=('Courier', 8)).pack()
            tk.Label(row, text=f"{size:.3f}", bg=COLORS['bg_secondary'],
                    fg=COLORS['text_primary'], font=('Courier', 8), width=8).pack(side='right')
        
        # Spread
        separator = tk.Frame(self.orderbook_content, bg=COLORS['accent_orange'], height=2)
        separator.pack(fill='x', pady=2)
        
        # Bids (покупки)
        for i in range(5):
            price = base_price - (i + 1) * random.uniform(1, 5)
            size = random.uniform(0.1, 2.0)
            
            row = tk.Frame(self.orderbook_content, bg=COLORS['bg_secondary'])
            row.pack(fill='x', pady=1)
            
            tk.Label(row, text=f"{size:.3f}", bg=COLORS['bg_secondary'],
                    fg=COLORS['text_primary'], font=('Courier', 8), width=8).pack(side='left')
            tk.Label(row, text=f"{price:.1f}", bg=COLORS['bg_secondary'],
                    fg=COLORS['accent_green'], font=('Courier', 8)).pack()
            tk.Label(row, text="", bg=COLORS['bg_secondary'], width=8).pack(side='right')
            
    def _update_trades(self):
        """Обновление списка сделок"""
        # Очистка
        for widget in self.trades_content.winfo_children():
            widget.destroy()
            
        # Последние сделки
        for trade in self.live_data['trade_history'][-8:]:
            row = tk.Frame(self.trades_content, bg=COLORS['bg_secondary'])
            row.pack(fill='x', pady=1)
            
            side_color = COLORS['accent_green'] if trade['side'] == 'BUY' else COLORS['accent_red']
            
            tk.Label(row, text=trade['time'], bg=COLORS['bg_secondary'],
                    fg=COLORS['text_secondary'], font=('Courier', 7), width=8).pack(side='left')
            tk.Label(row, text=trade['side'], bg=COLORS['bg_secondary'],
                    fg=side_color, font=('Courier', 7, 'bold'), width=4).pack(side='left')
            tk.Label(row, text=f"{trade['price']:.1f}", bg=COLORS['bg_secondary'],
                    fg=COLORS['text_primary'], font=('Courier', 7), width=8).pack(side='left')
            tk.Label(row, text=f"{trade['size']:.3f}", bg=COLORS['bg_secondary'],
                    fg=COLORS['text_primary'], font=('Courier', 7), width=8).pack(side='left')
                    
    def _update_signals(self):
        """Обновление AI сигналов"""
        # Очистка
        for widget in self.signals_content.winfo_children():
            widget.destroy()
            
        # Последние сигналы
        for signal in self.live_data['ai_signals'][-5:]:
            row = tk.Frame(self.signals_content, bg=COLORS['bg_tertiary'], pady=5)
            row.pack(fill='x', pady=2, padx=2)
            
            # Время и confidence
            header = tk.Frame(row, bg=COLORS['bg_tertiary'])
            header.pack(fill='x')
            
            tk.Label(header, text=signal['time'], bg=COLORS['bg_tertiary'],
                    fg=COLORS['text_secondary'], font=('Courier', 7)).pack(side='left')
            tk.Label(header, text=f"{signal['confidence']:.0f}%", bg=COLORS['bg_tertiary'],
                    fg=COLORS['accent_purple'], font=('Courier', 7, 'bold')).pack(side='right')
            
            # Сообщение
            tk.Label(row, text=signal['message'], bg=COLORS['bg_tertiary'],
                    fg=COLORS['text_primary'], font=('Arial', 8), wraplength=200, justify='left').pack(fill='x')
                    
    def _update_analysis(self):
        """Обновление анализа рынка"""
        if random.random() < 0.05:
            analyses = [
                "🔍 Detecting bullish pattern formation...",
                "📊 Volume analysis shows strong buying pressure",
                "⚡ High-frequency trading detected",
                "🎯 Target price: $43,200 (R/R: 2.5)",
                "🛡️ Stop loss adjusted to $41,800",
                "📈 RSI approaching oversold levels",
            ]
            
            timestamp = datetime.now().strftime('%H:%M:%S')
            new_text = f"[{timestamp}] {random.choice(analyses)}\n"
            
            self.analysis_text.configure(state='normal')
            self.analysis_text.insert(tk.END, new_text)
            
            # Ограничение строк
            lines = self.analysis_text.get('1.0', tk.END).split('\n')
            if len(lines) > 15:
                self.analysis_text.delete('1.0', '2.0')
                
            self.analysis_text.configure(state='disabled')
            self.analysis_text.see(tk.END)
            
    def update_portfolio(self):
        """Обновление портфеля"""
        # Очистка
        for widget in self.portfolio_content.winfo_children():
            widget.destroy()
            
        holdings = [
            ("BTC", "0.5432", "$22,890", "+$1,250"),
            ("ETH", "8.7654", "$21,456", "+$890"),
            ("ADA", "15,432", "$7,456", "-$123"),
            ("SOL", "45.23", "$4,456", "+$234"),
        ]
        
        for asset, amount, value, pnl in holdings:
            item = tk.Frame(self.portfolio_content, bg=COLORS['bg_secondary'])
            item.pack(fill='x', pady=2)
            
            # Asset info
            info_frame = tk.Frame(item, bg=COLORS['bg_secondary'])
            info_frame.pack(side='left', fill='x', expand=True)
            
            tk.Label(info_frame, text=asset, bg=COLORS['bg_secondary'],
                    fg=COLORS['text_primary'], font=('Arial', 9, 'bold')).pack(anchor='w')
            tk.Label(info_frame, text=amount, bg=COLORS['bg_secondary'],
                    fg=COLORS['text_secondary'], font=('Arial', 8)).pack(anchor='w')
            
            # Value info
            value_frame = tk.Frame(item, bg=COLORS['bg_secondary'])
            value_frame.pack(side='right')
            
            tk.Label(value_frame, text=value, bg=COLORS['bg_secondary'],
                    fg=COLORS['text_primary'], font=('Arial', 9)).pack(anchor='e')
            
            pnl_color = COLORS['accent_green'] if '+' in pnl else COLORS['accent_red']
            tk.Label(value_frame, text=pnl, bg=COLORS['bg_secondary'],
                    fg=pnl_color, font=('Arial', 8, 'bold')).pack(anchor='e')

class MainApplication:
    """Главное приложение ATB Trading с интегрированным дашбордом"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ATB Trading System v3.0 - Apple Style")
        self.root.geometry("1400x900")
        self.root.configure(bg=COLORS['bg_primary'])
        
        # Менеджер конфигурации
        self.config_manager = ConfigManager()
        
        # Настройка стилей
        self.setup_styles()
        
        # Создание интерфейса
        self.create_interface()
        
        # Настройка горячих клавиш
        self.setup_hotkeys()
        
    def setup_styles(self):
        """Настройка стилей для темной темы"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Настройка темных стилей
        style.configure('TNotebook', background=COLORS['bg_primary'], borderwidth=0)
        style.configure('TNotebook.Tab', background=COLORS['bg_tertiary'],
                       foreground=COLORS['text_primary'], padding=[12, 8])
        style.map('TNotebook.Tab', background=[('selected', COLORS['accent_blue'])])
        
    def create_interface(self):
        """Создание основного интерфейса"""
        # Меню
        self.create_menu()
        
        # Основной контейнер
        main_frame = tk.Frame(self.root, bg=COLORS['bg_primary'])
        main_frame.pack(fill='both', expand=True)
        
        # Notebook для вкладок
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Создание вкладок
        self.create_tabs()
        
    def create_menu(self):
        """Создание меню приложения"""
        menubar = tk.Menu(self.root, bg=COLORS['bg_secondary'], fg=COLORS['text_primary'])
        self.root.config(menu=menubar)
        
        # Файл
        file_menu = tk.Menu(menubar, tearoff=0, bg=COLORS['bg_secondary'], fg=COLORS['text_primary'])
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Новая сессия", command=self.new_session)
        file_menu.add_command(label="Загрузить конфигурацию", command=self.load_config)
        file_menu.add_command(label="Сохранить конфигурацию", command=self.save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)
        
        # Торговля
        trading_menu = tk.Menu(menubar, tearoff=0, bg=COLORS['bg_secondary'], fg=COLORS['text_primary'])
        menubar.add_cascade(label="Торговля", menu=trading_menu)
        trading_menu.add_command(label="Быстрый старт", command=self.quick_start)
        trading_menu.add_command(label="Остановить торговлю", command=self.stop_trading)
        
        # Настройки
        settings_menu = tk.Menu(menubar, tearoff=0, bg=COLORS['bg_secondary'], fg=COLORS['text_primary'])
        menubar.add_cascade(label="Настройки", menu=settings_menu)
        settings_menu.add_command(label="⚙️ Настройки системы", command=self.open_settings)
        settings_menu.add_command(label="🔑 API ключи", command=self.open_api_settings)
        
        # Помощь
        help_menu = tk.Menu(menubar, tearoff=0, bg=COLORS['bg_secondary'], fg=COLORS['text_primary'])
        menubar.add_cascade(label="Помощь", menu=help_menu)
        help_menu.add_command(label="📚 Руководство", command=self.show_help)
        help_menu.add_command(label="ℹ️ О программе", command=self.show_about)
        
    def create_tabs(self):
        """Создание всех вкладок приложения"""
        # Основная торговая вкладка
        self.create_main_trading_tab()
        
        # Live Demo вкладка
        self.live_demo_tab = LiveDemoTab(self.notebook, self.config_manager)
        
        # Аналитика
        self.create_analytics_tab()
        
        # Бэктестинг
        self.create_backtest_tab()
        
    def create_main_trading_tab(self):
        """Основная торговая вкладка"""
        trading_frame = tk.Frame(self.notebook, bg=COLORS['bg_primary'])
        self.notebook.add(trading_frame, text="💰 Торговля")
        
        # Заглушка для основной торговли
        tk.Label(trading_frame, text="🚧 Основная торговая панель\n(В разработке)",
                bg=COLORS['bg_primary'], fg=COLORS['text_secondary'],
                font=('Arial', 16), pady=50).pack(expand=True)
        
    def create_analytics_tab(self):
        """Вкладка аналитики"""
        analytics_frame = tk.Frame(self.notebook, bg=COLORS['bg_primary'])
        self.notebook.add(analytics_frame, text="📊 Аналитика")
        
        tk.Label(analytics_frame, text="📈 Аналитическая панель\n(В разработке)",
                bg=COLORS['bg_primary'], fg=COLORS['text_secondary'],
                font=('Arial', 16), pady=50).pack(expand=True)
        
    def create_backtest_tab(self):
        """Вкладка бэктестинга"""
        backtest_frame = tk.Frame(self.notebook, bg=COLORS['bg_primary'])
        self.notebook.add(backtest_frame, text="⏮ Бэктест")
        
        tk.Label(backtest_frame, text="🧪 Панель бэктестинга\n(В разработке)",
                bg=COLORS['bg_primary'], fg=COLORS['text_secondary'],
                font=('Arial', 16), pady=50).pack(expand=True)
        
    def setup_hotkeys(self):
        """Настройка горячих клавиш"""
        self.root.bind('<Control-s>', lambda e: self.save_config())
        self.root.bind('<Control-o>', lambda e: self.load_config())
        self.root.bind('<Control-comma>', lambda e: self.open_settings())
        self.root.bind('<F5>', lambda e: self.refresh_data())
        self.root.bind('<F9>', lambda e: self.quick_start())
        self.root.bind('<F10>', lambda e: self.stop_trading())
        
    def open_settings(self):
        """Открытие окна настроек"""
        settings_window = SettingsWindow(self.root, self.config_manager)
        settings_window.show()
        
    def open_api_settings(self):
        """Открытие настроек API (быстрый доступ)"""
        settings_window = SettingsWindow(self.root, self.config_manager)
        settings_window.show()
        # Переключение на вкладку API ключей
        if hasattr(settings_window, 'notebook'):
            settings_window.notebook.select(0)
            
    def new_session(self):
        """Новая торговая сессия"""
        if messagebox.askyesno("Новая сессия", "Начать новую торговую сессию?\nТекущие данные будут очищены."):
            messagebox.showinfo("Успех", "✅ Новая сессия создана!")
            
    def load_config(self):
        """Загрузка конфигурации"""
        filename = filedialog.askopenfilename(
            title="Выберите файл конфигурации",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            messagebox.showinfo("Успех", f"✅ Конфигурация загружена из {filename}")
            
    def save_config(self):
        """Сохранение конфигурации"""
        filename = filedialog.asksaveasfilename(
            title="Сохранить конфигурацию",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            messagebox.showinfo("Успех", f"✅ Конфигурация сохранена в {filename}")
            
    def quick_start(self):
        """Быстрый старт торговли"""
        messagebox.showinfo("Быстрый старт", "🚀 Быстрый старт торговли\n(Функция в разработке)")
        
    def stop_trading(self):
        """Остановка торговли"""
        messagebox.showinfo("Остановка", "⏹ Торговля остановлена\n(Функция в разработке)")
        
    def refresh_data(self):
        """Обновление данных"""
        messagebox.showinfo("Обновление", "🔄 Данные обновлены\n(Функция в разработке)")
        
    def show_help(self):
        """Показать справку"""
        help_text = """
📚 ATB Trading System - Руководство пользователя

🎯 Основные функции:
• 📺 Live Demo - демонстрационный дашборд для стримов
• 💰 Торговля - основная торговая панель
• 📊 Аналитика - анализ производительности
• ⏮ Бэктест - тестирование стратегий

⚙️ Настройки:
• Ctrl+, - Открыть настройки
• 🔑 API ключи для бирж (Bybit, Binance, Coinbase)
• 📊 Параметры дашборда
• 🔔 Настройки уведомлений

🎮 Горячие клавиши:
• Ctrl+S - Сохранить конфигурацию
• Ctrl+O - Загрузить конфигурацию
• F5 - Обновить данные
• F9 - Быстрый старт
• F10 - Остановить торговлю

💡 Для демонстрации на Twitch используйте вкладку "📺 Live Demo"
        """
        
        messagebox.showinfo("Руководство", help_text)
        
    def show_about(self):
        """О программе"""
        about_text = """
🚀 ATB Trading System v3.0
Apple-style Dark Theme Edition

💫 Современный торговый дашборд
📺 Оптимизирован для Twitch демонстраций
🎨 Дизайн в стиле Apple
🔴 Live данные в реальном времени

Разработано для профессиональных трейдеров
и контент-мейкеров.

© 2024 ATB Trading Systems
        """
        
        messagebox.showinfo("О программе", about_text)
        
    def run(self):
        """Запуск приложения"""
        print("🚀 Запуск ATB Trading System...")
        print("💫 Apple-style Dark Theme")
        print("📺 Twitch Demo Ready")
        print("⚙️ Настройки доступны через меню")
        
        self.root.mainloop()

if __name__ == "__main__":
    try:
        app = MainApplication()
        app.run()
    except Exception as e:
        print(f"❌ Ошибка запуска приложения: {e}")
        messagebox.showerror("Критическая ошибка", f"Не удалось запустить приложение:\n{e}")