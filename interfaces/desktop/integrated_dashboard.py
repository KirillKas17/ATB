"""
Интегрированный дашборд торговли - полная версия с интеграцией
"""

import sys
import asyncio
import threading
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox

# Добавляем путь к проекту
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Импорт компонентов дашборда
from interfaces.desktop.trading_dashboard import ModernTradingDashboard
from interfaces.desktop.dashboard_controller import DashboardController

class IntegratedTradingDashboard(ModernTradingDashboard):
    """Интегрированный дашборд с полной функциональностью"""
    
    def __init__(self):
        # Инициализация контроллера
        self.controller = DashboardController()
        
        # Инициализация родительского класса
        super().__init__()
        
        # Добавление дополнительных элементов
        self.create_advanced_features()
        
        # Запуск асинхронного цикла
        self.setup_async_loop()
    
    def create_advanced_features(self):
        """Создание продвинутых функций"""
        
        # Меню
        self.create_menu_bar()
        
        # Дополнительные панели
        self.create_alerts_panel()
        self.create_news_panel()
        self.create_portfolio_panel()
        
        # Горячие клавиши
        self.setup_hotkeys()
    
    def create_menu_bar(self):
        """Создание строки меню"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Файл
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Новая сессия", command=self.new_session)
        file_menu.add_command(label="Загрузить конфигурацию", command=self.load_config)
        file_menu.add_command(label="Сохранить конфигурацию", command=self.save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Экспорт данных", command=self.export_data)
        file_menu.add_command(label="Импорт стратегий", command=self.import_strategies)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)
        
        # Торговля
        trading_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Торговля", menu=trading_menu)
        trading_menu.add_command(label="Быстрый старт", command=self.quick_start)
        trading_menu.add_command(label="Настройки стратегий", command=self.strategy_settings)
        trading_menu.add_command(label="Управление рисками", command=self.risk_settings)
        trading_menu.add_separator()
        trading_menu.add_command(label="Закрыть все позиции", command=self.close_all_positions)
        trading_menu.add_command(label="Отменить все ордера", command=self.cancel_all_orders)
        
        # Аналитика
        analytics_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Аналитика", menu=analytics_menu)
        analytics_menu.add_command(label="Отчет по производительности", command=self.performance_report)
        analytics_menu.add_command(label="Анализ рисков", command=self.risk_analysis)
        analytics_menu.add_command(label="Оптимизация стратегий", command=self.strategy_optimization)
        analytics_menu.add_separator()
        analytics_menu.add_command(label="Экспорт в Excel", command=self.export_excel)
        analytics_menu.add_command(label="Генерация PDF отчета", command=self.generate_pdf_report)
        
        # Инструменты
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Инструменты", menu=tools_menu)
        tools_menu.add_command(label="Калькулятор позиций", command=self.position_calculator)
        tools_menu.add_command(label="Конвертер валют", command=self.currency_converter)
        tools_menu.add_command(label="Экономический календарь", command=self.economic_calendar)
        tools_menu.add_separator()
        tools_menu.add_command(label="Настройки", command=self.open_settings)
        
        # Помощь
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Помощь", menu=help_menu)
        help_menu.add_command(label="Руководство пользователя", command=self.show_manual)
        help_menu.add_command(label="Горячие клавиши", command=self.show_hotkeys)
        help_menu.add_command(label="О программе", command=self.show_about)
    
    def create_alerts_panel(self):
        """Создание панели уведомлений"""
        alerts_frame = tk.LabelFrame(self.right_panel, text="Уведомления",
                                   bg=self.colors['bg_secondary'],
                                   fg=self.colors['text_primary'],
                                   font=self.fonts['subtitle'])
        alerts_frame.pack(fill='x', padx=5, pady=5)
        
        # Список уведомлений
        self.alerts_listbox = tk.Listbox(alerts_frame, height=4,
                                       bg=self.colors['bg_tertiary'],
                                       fg=self.colors['text_primary'],
                                       selectbackground=self.colors['accent_orange'])
        self.alerts_listbox.pack(fill='x', padx=5, pady=5)
        
        # Добавление тестовых уведомлений
        test_alerts = [
            "🟢 Позиция BTC/USDT закрыта с прибылью +$150",
            "⚠️ Высокая волатильность на ETH/USDT",
            "🔴 Стоп-лосс сработал на ADA/USDT -$75"
        ]
        
        for alert in test_alerts:
            self.alerts_listbox.insert('end', alert)
    
    def create_news_panel(self):
        """Создание панели новостей"""
        news_frame = tk.LabelFrame(self.left_panel, text="Рыночные новости",
                                 bg=self.colors['bg_secondary'],
                                 fg=self.colors['text_primary'],
                                 font=self.fonts['subtitle'])
        news_frame.pack(fill='x', padx=5, pady=5)
        
        # Текстовое поле для новостей
        self.news_text = tk.Text(news_frame, height=6, wrap='word',
                               bg=self.colors['bg_tertiary'],
                               fg=self.colors['text_primary'],
                               font=self.fonts['small'])
        self.news_text.pack(fill='x', padx=5, pady=5)
        
        # Добавление тестовых новостей
        test_news = """
🔥 Bitcoin поднялся выше $45,000 на фоне растущего институционального интереса

📈 Ethereum готовится к очередному обновлению сети, ожидается рост активности

⚡ Новые регулятивные меры в ЕС могут повлиять на рынок криптовалют

🏦 Крупные банки начинают предлагать сервисы хранения криптовалют
        """
        
        self.news_text.insert('1.0', test_news.strip())
        self.news_text.config(state='disabled')
    
    def create_portfolio_panel(self):
        """Создание панели портфеля"""
        portfolio_frame = tk.LabelFrame(self.right_panel, text="Портфель",
                                      bg=self.colors['bg_secondary'],
                                      fg=self.colors['text_primary'],
                                      font=self.fonts['subtitle'])
        portfolio_frame.pack(fill='x', padx=5, pady=5)
        
        # Распределение активов
        assets_frame = tk.Frame(portfolio_frame, bg=self.colors['bg_secondary'])
        assets_frame.pack(fill='x', padx=5, pady=5)
        
        # Пример активов
        assets = [
            ("USDT", "70%", self.colors['text_primary']),
            ("BTC", "20%", self.colors['accent_orange']),
            ("ETH", "8%", self.colors['accent_blue']),
            ("Другие", "2%", self.colors['text_secondary'])
        ]
        
        for i, (asset, percentage, color) in enumerate(assets):
            tk.Label(assets_frame, text=f"{asset}:",
                   bg=self.colors['bg_secondary'],
                   fg=self.colors['text_secondary']).grid(row=i, column=0, sticky='w')
            
            tk.Label(assets_frame, text=percentage,
                   bg=self.colors['bg_secondary'],
                   fg=color, font=self.fonts['body']).grid(row=i, column=1, sticky='e')
        
        assets_frame.columnconfigure(1, weight=1)
    
    def setup_hotkeys(self):
        """Настройка горячих клавиш"""
        self.root.bind('<Control-n>', lambda e: self.new_session())
        self.root.bind('<Control-s>', lambda e: self.save_config())
        self.root.bind('<Control-o>', lambda e: self.load_config())
        self.root.bind('<F5>', lambda e: self.refresh_data())
        self.root.bind('<F9>', lambda e: self.start_trading())
        self.root.bind('<F10>', lambda e: self.stop_trading())
        self.root.bind('<Escape>', lambda e: self.emergency_stop())
    
    def setup_async_loop(self):
        """Настройка асинхронного цикла"""
        self.loop = asyncio.new_event_loop()
        
        def run_loop():
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        
        self.async_thread = threading.Thread(target=run_loop, daemon=True)
        self.async_thread.start()
    
    def run_async(self, coro):
        """Запуск асинхронной корутины"""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout=10)
    
    # Переопределение методов родительского класса
    async def start_trading_async(self):
        """Асинхронный запуск торговли"""
        try:
            # Подготовка параметров
            risk_params = {
                'position_size': self.position_size_var.get(),
                'stop_loss': self.stop_loss_var.get(),
                'take_profit': self.take_profit_var.get()
            }
            
            # Запуск сессии через контроллер
            session = await self.controller.start_trading_session(
                mode=self.state.trading_mode,
                selected_pairs=self.state.selected_pairs,
                active_strategies=self.state.active_strategies,
                initial_balance=self.state.total_balance,
                risk_params=risk_params
            )
            
            # Обновление состояния
            self.state.is_trading_active = True
            
            # Запуск мониторинга данных
            self.start_real_time_updates()
            
            return True
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось запустить торговлю: {e}")
            return False
    
    def start_trading(self):
        """Переопределенный метод запуска торговли"""
        if not self.state.selected_pairs:
            messagebox.showwarning("Предупреждение", "Выберите торговые пары")
            return
        
        if not self.state.active_strategies:
            messagebox.showwarning("Предупреждение", "Выберите стратегии")
            return
        
        # Запуск асинхронно
        try:
            success = self.run_async(self.start_trading_async())
            
            if success:
                self.start_btn.config(state='disabled')
                self.stop_btn.config(state='normal')
                self.connection_status.config(text="🟢 Подключено", fg=self.colors['accent_green'])
                
                messagebox.showinfo("Торговля запущена", 
                                  f"Режим: {self.state.trading_mode}\n"
                                  f"Пары: {', '.join(self.state.selected_pairs)}\n"
                                  f"Стратегии: {', '.join(self.state.active_strategies)}")
        
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось запустить торговлю: {e}")
    
    async def stop_trading_async(self):
        """Асинхронная остановка торговли"""
        try:
            await self.controller.stop_trading_session()
            self.state.is_trading_active = False
            return True
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось остановить торговлю: {e}")
            return False
    
    def stop_trading(self):
        """Переопределенный метод остановки торговли"""
        try:
            success = self.run_async(self.stop_trading_async())
            
            if success:
                self.start_btn.config(state='normal')
                self.stop_btn.config(state='disabled')
                self.connection_status.config(text="⚫ Отключено", fg=self.colors['accent_red'])
                
                messagebox.showinfo("Торговля остановлена", "Все активные операции завершены")
        
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось остановить торговлю: {e}")
    
    def start_real_time_updates(self):
        """Запуск обновлений в реальном времени"""
        def update_data():
            if self.state.is_trading_active:
                try:
                    # Обновление позиций
                    positions = self.run_async(self.controller.get_positions())
                    self.update_positions_display(positions)
                    
                    # Обновление ордеров
                    orders = self.run_async(self.controller.get_orders())
                    self.update_orders_display(orders)
                    
                    # Обновление метрик
                    metrics = self.run_async(self.controller.get_performance_metrics())
                    self.update_metrics_display(metrics)
                    
                except Exception as e:
                    print(f"Ошибка обновления данных: {e}")
            
            # Планирование следующего обновления
            self.root.after(5000, update_data)  # Каждые 5 секунд
        
        # Запуск цикла обновлений
        self.root.after(1000, update_data)
    
    def update_positions_display(self, positions):
        """Обновление отображения позиций"""
        # Очистка таблицы
        for item in self.positions_tree.get_children():
            self.positions_tree.delete(item)
        
        # Добавление новых данных
        for position in positions:
            pnl_color = 'green' if position['pnl'] >= 0 else 'red'
            
            self.positions_tree.insert('', 'end', values=(
                position['symbol'],
                position['side'],
                f"{position['size']:.4f}",
                f"${position['entry_price']:.2f}",
                f"${position['current_price']:.2f}",
                f"${position['pnl']:.2f}"
            ), tags=(pnl_color,))
        
        # Настройка цветов
        self.positions_tree.tag_configure('green', foreground=self.colors['accent_green'])
        self.positions_tree.tag_configure('red', foreground=self.colors['accent_red'])
    
    def update_orders_display(self, orders):
        """Обновление отображения ордеров"""
        # Очистка таблицы
        for item in self.orders_tree.get_children():
            self.orders_tree.delete(item)
        
        # Добавление новых данных
        for order in orders:
            self.orders_tree.insert('', 'end', values=(
                order['symbol'],
                order['type'],
                order['side'],
                f"{order['amount']:.4f}",
                f"${order['price']:.2f}",
                order['status']
            ))
    
    def update_metrics_display(self, metrics):
        """Обновление отображения метрик"""
        # Обновление основных метрик
        pnl_color = self.colors['accent_green'] if metrics['total_pnl'] >= 0 else self.colors['accent_red']
        
        self.total_pnl_label.config(text=f"${metrics['total_pnl']:.2f}", fg=pnl_color)
        self.win_rate_label.config(text=f"{metrics['win_rate_pct']:.1f}%")
        self.sharpe_label.config(text=f"{metrics['sharpe_ratio']:.2f}")
        
        # Обновление состояния
        self.state.current_pnl = Decimal(str(metrics['total_pnl']))
        self.state.active_positions = metrics['total_trades']
    
    # Обработчики событий меню
    def new_session(self):
        """Создание новой сессии"""
        if self.state.is_trading_active:
            if messagebox.askyesno("Подтверждение", "Остановить текущую торговлю и создать новую сессию?"):
                self.stop_trading()
            else:
                return
        
        self.reset_trading()
        messagebox.showinfo("Новая сессия", "Создана новая торговая сессия")
    
    def load_config(self):
        """Загрузка конфигурации"""
        from tkinter import filedialog
        
        filename = filedialog.askopenfilename(
            title="Выберите файл конфигурации",
            filetypes=[("JSON файлы", "*.json"), ("Все файлы", "*.*")]
        )
        
        if filename:
            try:
                import json
                with open(filename, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Применение конфигурации
                self.apply_config(config)
                messagebox.showinfo("Успех", "Конфигурация загружена")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить конфигурацию: {e}")
    
    def save_config(self):
        """Сохранение конфигурации"""
        from tkinter import filedialog
        
        filename = filedialog.asksaveasfilename(
            title="Сохранить конфигурацию",
            defaultextension=".json",
            filetypes=[("JSON файлы", "*.json"), ("Все файлы", "*.*")]
        )
        
        if filename:
            try:
                config = self.get_current_config()
                
                import json
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                messagebox.showinfo("Успех", "Конфигурация сохранена")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить конфигурацию: {e}")
    
    def apply_config(self, config):
        """Применение конфигурации"""
        try:
            # Торговые пары
            if 'selected_pairs' in config:
                self.state.selected_pairs = config['selected_pairs']
                # Обновление UI
                self.pairs_listbox.selection_clear(0, 'end')
                for pair in config['selected_pairs']:
                    for i in range(self.pairs_listbox.size()):
                        if self.pairs_listbox.get(i) == pair:
                            self.pairs_listbox.selection_set(i)
            
            # Стратегии
            if 'active_strategies' in config:
                for strategy, var in self.strategy_vars.items():
                    var.set(strategy in config['active_strategies'])
                self.update_active_strategies()
            
            # Параметры риска
            if 'risk_params' in config:
                risk = config['risk_params']
                self.position_size_var.set(risk.get('position_size', 2.0))
                self.stop_loss_var.set(risk.get('stop_loss', 2.0))
                self.take_profit_var.set(risk.get('take_profit', 4.0))
            
            # Режим торговли
            if 'trading_mode' in config:
                self.trading_mode_var.set(config['trading_mode'])
                self.state.trading_mode = config['trading_mode']
        
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось применить конфигурацию: {e}")
    
    def get_current_config(self):
        """Получение текущей конфигурации"""
        return {
            'selected_pairs': self.state.selected_pairs,
            'active_strategies': self.state.active_strategies,
            'trading_mode': self.state.trading_mode,
            'risk_params': {
                'position_size': self.position_size_var.get(),
                'stop_loss': self.stop_loss_var.get(),
                'take_profit': self.take_profit_var.get()
            },
            'created_at': datetime.now().isoformat()
        }
    
    def quick_start(self):
        """Быстрый старт с предустановленными настройками"""
        # Выбор популярных пар
        popular_pairs = ['BTC/USDT', 'ETH/USDT']
        for pair in popular_pairs:
            for i in range(self.pairs_listbox.size()):
                if self.pairs_listbox.get(i) == pair:
                    self.pairs_listbox.selection_set(i)
        
        self.on_pair_select()
        
        # Включение базовых стратегий
        self.strategy_vars['RSI Bounce'].set(True)
        self.strategy_vars['MACD Cross'].set(True)
        self.update_active_strategies()
        
        # Безопасные настройки риска
        self.position_size_var.set(1.0)
        self.stop_loss_var.set(2.0)
        self.take_profit_var.set(4.0)
        
        messagebox.showinfo("Быстрый старт", "Настройки для быстрого старта применены")
    
    def emergency_stop(self):
        """Экстренная остановка"""
        if self.state.is_trading_active:
            if messagebox.askyesno("ЭКСТРЕННАЯ ОСТАНОВКА", 
                                 "Немедленно остановить всю торговлю?\n"
                                 "Все позиции будут закрыты!"):
                self.stop_trading()
                messagebox.showwarning("Остановлено", "Торговля экстренно остановлена")
    
    def refresh_data(self):
        """Обновление всех данных"""
        if self.state.is_trading_active:
            try:
                # Принудительное обновление всех данных
                self.start_real_time_updates()
                messagebox.showinfo("Обновлено", "Данные обновлены")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось обновить данные: {e}")
    
    def show_about(self):
        """О программе"""
        about_text = """
ATB Trading Dashboard v2.0

Профессиональная торговая платформа для криптовалют

Возможности:
• Реальная и симуляционная торговля
• Продвинутая аналитика
• Множественные стратегии
• Управление рисками
• Бэктестинг

© 2024 ATB Trading Systems
        """
        
        messagebox.showinfo("О программе", about_text.strip())
    
    def show_hotkeys(self):
        """Горячие клавиши"""
        hotkeys_text = """
Горячие клавиши:

Ctrl+N - Новая сессия
Ctrl+S - Сохранить конфигурацию  
Ctrl+O - Загрузить конфигурацию
F5 - Обновить данные
F9 - Запустить торговлю
F10 - Остановить торговлю
Esc - Экстренная остановка
        """
        
        messagebox.showinfo("Горячие клавиши", hotkeys_text.strip())
    
    def run(self):
        """Запуск интегрированного дашборда"""
        try:
            self.root.mainloop()
        finally:
            # Очистка ресурсов
            if hasattr(self, 'loop'):
                self.loop.call_soon_threadsafe(self.loop.stop)

if __name__ == "__main__":
    # Создание и запуск интегрированного дашборда
    try:
        dashboard = IntegratedTradingDashboard()
        dashboard.run()
    except Exception as e:
        print(f"Ошибка запуска дашборда: {e}")
        import traceback
        traceback.print_exc()