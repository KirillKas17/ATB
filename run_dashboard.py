#!/usr/bin/env python3
"""
Запуск дашборда ATB с проверкой передачи данных.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
<<<<<<< HEAD
from typing import Dict, Any, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import websockets

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Создание FastAPI приложения
app = FastAPI(title="ATB Dashboard", version="1.0.0")

# Монтирование статических файлов
dashboard_path = Path("interfaces/presentation/dashboard")
if dashboard_path.exists():
    app.mount("/static", StaticFiles(directory=str(dashboard_path)), name="static")

# Хранилище для WebSocket соединений
active_connections: List[WebSocket] = []

# Моковые данные для тестирования
MOCK_DATA = {
    "system_status": {
        "status": "online",
        "uptime": "02:15:30",
        "cpu_usage": 23.5,
        "memory_usage": 1.2,
        "active_connections": 5
    },
    "trading_data": {
        "total_pnl": 1234.56,
        "daily_pnl": 123.45,
        "active_positions": 3,
        "win_rate": 78.5,
        "total_trades": 1247,
        "profitable_trades": 978,
        "losing_trades": 269
    },
    "positions": [
        {
            "symbol": "BTC/USDT",
            "side": "long",
            "size": 0.1,
            "entry_price": 45000,
            "current_price": 45200,
            "pnl": 200,
            "pnl_percent": 0.44
        },
        {
            "symbol": "ETH/USDT",
            "side": "short",
            "size": 1.5,
            "entry_price": 3200,
            "current_price": 3180,
            "pnl": 30,
            "pnl_percent": 0.94
        }
    ],
    "analytics": {
        "rsi": 45.2,
        "macd": 0.023,
        "bollinger_position": "middle",
        "ai_signals": [
            {"type": "buy", "strength": "strong", "symbol": "BTC", "message": "Сильный сигнал на покупку BTC"},
            {"type": "sell", "strength": "weak", "symbol": "ETH", "message": "Слабый сигнал на продажу ETH"}
        ]
=======
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
>>>>>>> e90116ec91962c532322996f100c96a0340b15b3
    }
}

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Главная страница дашборда."""
    html_file = dashboard_path / "index.html"
    if html_file.exists():
        with open(html_file, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head><title>ATB Dashboard</title></head>
        <body>
            <h1>ATB Dashboard</h1>
            <p>Файл index.html не найден в папке dashboard</p>
        </body>
        </html>
        """)

@app.get("/api/status")
async def get_system_status():
    """Получение статуса системы."""
    logger.info("Запрос статуса системы")
    return MOCK_DATA["system_status"]

@app.get("/api/trading")
async def get_trading_data():
    """Получение торговых данных."""
    logger.info("Запрос торговых данных")
    return MOCK_DATA["trading_data"]

@app.get("/api/positions")
async def get_positions():
    """Получение активных позиций."""
    logger.info("Запрос позиций")
    return MOCK_DATA["positions"]

@app.get("/api/analytics")
async def get_analytics():
    """Получение аналитических данных."""
    logger.info("Запрос аналитики")
    return MOCK_DATA["analytics"]

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint для real-time данных."""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WebSocket подключен. Всего соединений: {len(active_connections)}")
    
<<<<<<< HEAD
    try:
        while True:
            # Отправка данных каждые 5 секунд
            await asyncio.sleep(5)
=======
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
>>>>>>> e90116ec91962c532322996f100c96a0340b15b3
            
            # Обновление времени работы
            MOCK_DATA["system_status"]["uptime"] = "02:15:35"
            MOCK_DATA["system_status"]["active_connections"] = len(active_connections)
            
            # Отправка обновленных данных
            await websocket.send_text(json.dumps({
                "type": "data_update",
                "timestamp": datetime.now().isoformat(),
                "data": MOCK_DATA
            }))
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"WebSocket отключен. Осталось соединений: {len(active_connections)}")
    except Exception as e:
        logger.error(f"Ошибка WebSocket: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

@app.get("/api/health")
async def health_check():
    """Проверка здоровья API."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(active_connections),
        "dashboard_files": {
            "index_html": (dashboard_path / "index.html").exists(),
            "dashboard_js": (dashboard_path / "dashboard.js").exists(),
            "style_css": (dashboard_path / "style.css").exists()
        }
    }

async def broadcast_data():
    """Отправка данных всем подключенным клиентам."""
    while True:
        if active_connections:
            # Обновление данных
            MOCK_DATA["system_status"]["uptime"] = "02:15:40"
            MOCK_DATA["system_status"]["active_connections"] = len(active_connections)
            
            message = json.dumps({
                "type": "broadcast",
                "timestamp": datetime.now().isoformat(),
                "data": MOCK_DATA
            })
            
            # Отправка всем подключенным клиентам
            for connection in active_connections[:]:  # Копия списка для безопасного удаления
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Ошибка отправки данных: {e}")
                    if connection in active_connections:
                        active_connections.remove(connection)
        
        await asyncio.sleep(10)  # Обновление каждые 10 секунд

@app.on_event("startup")
async def startup_event():
    """Событие запуска приложения."""
    logger.info("Запуск ATB Dashboard...")
    logger.info(f"Дашборд доступен по адресу: http://localhost:8000")
    logger.info(f"API документация: http://localhost:8000/docs")
    
<<<<<<< HEAD
    # Запуск фоновой задачи для broadcast
    asyncio.create_task(broadcast_data())

def check_dashboard_files():
    """Проверка наличия файлов дашборда."""
    required_files = ["index.html", "dashboard.js", "style.css"]
    missing_files = []
    
    for file_name in required_files:
        file_path = dashboard_path / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        logger.warning(f"Отсутствуют файлы дашборда: {missing_files}")
=======
    if missing_packages:
        print("\n❌ Отсутствующие зависимости:")
        for package in missing_packages:
            print(f"   pip install {package}")
>>>>>>> e90116ec91962c532322996f100c96a0340b15b3
        return False
    
    logger.info("Все файлы дашборда найдены")
    return True

<<<<<<< HEAD
def main():
    """Главная функция."""
    print("=" * 60)
    print("           ATB Dashboard Launcher")
    print("=" * 60)
    print()
    
    # Проверка файлов дашборда
    if not check_dashboard_files():
        print("Предупреждение: Некоторые файлы дашборда отсутствуют")
        print("Дашборд может работать некорректно")
        print()
    
    # Проверка API
    print("Проверка API endpoints:")
    print("  - GET /api/status - Статус системы")
    print("  - GET /api/trading - Торговые данные")
    print("  - GET /api/positions - Активные позиции")
    print("  - GET /api/analytics - Аналитические данные")
    print("  - GET /api/health - Проверка здоровья")
    print("  - WS /ws - WebSocket для real-time данных")
    print()
    
    # Запуск сервера
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False
        )
    except KeyboardInterrupt:
        print("\nДашборд остановлен")
    except Exception as e:
        logger.error(f"Ошибка запуска сервера: {e}")
        sys.exit(1)
=======
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
>>>>>>> e90116ec91962c532322996f100c96a0340b15b3

if __name__ == "__main__":
    main()