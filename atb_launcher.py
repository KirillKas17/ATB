#!/usr/bin/env python3
"""
ATB Trading System Launcher v2.0
Единая точка входа для запуска всей торговой системы
"""

import sys
import os
import subprocess
import threading
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncio

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

class ATBSystemLauncher:
    """Главный launcher для всей системы ATB Trading"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.config_file = self.project_root / "launcher_config.json"
        self.log_file = self.project_root / "logs" / "launcher.log"
        
        # Создание директорий
        self.setup_directories()
        
        # Настройка логирования
        self.setup_logging()
        
        # Состояние системы
        self.system_status = {
            'database': False,
            'redis': False,
            'api_server': False,
            'trading_engine': False,
            'dashboard': False,
            'monitoring': False
        }
        
        # Процессы
        self.processes = {}
        
        # Конфигурация
        self.config = self.load_config()
        
        self.logger.info("🚀 ATB Trading System Launcher инициализирован")
    
    def setup_directories(self):
        """Создание необходимых директорий"""
        dirs = [
            "logs",
            "data",
            "config", 
            "temp",
            "backups"
        ]
        
        for dir_name in dirs:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(exist_ok=True)
    
    def setup_logging(self):
        """Настройка логирования"""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def load_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации launcher'а"""
        default_config = {
            "auto_start_components": [
                "database",
                "trading_engine", 
                "dashboard"
            ],
            "startup_delay": 2,
            "health_check_interval": 30,
            "auto_restart": True,
            "max_restart_attempts": 3,
            "dashboard_port": 8080,
            "api_port": 8000,
            "monitoring_port": 9090,
            "enable_monitoring": True,
            "enable_auto_backup": True,
            "backup_interval_hours": 24,
            "python_executable": "python",
            "environment": "development"
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # Объединяем с дефолтной конфигурацией
                    default_config.update(config)
                    return default_config
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки конфигурации: {e}")
        
        # Сохраняем дефолтную конфигурацию
        self.save_config(default_config)
        return default_config
    
    def save_config(self, config: Dict[str, Any]):
        """Сохранение конфигурации"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Ошибка сохранения конфигурации: {e}")
    
    def check_dependencies(self) -> bool:
        """Проверка зависимостей системы"""
        self.logger.info("🔍 Проверка зависимостей...")
        
        # Проверка Python
        python_version = sys.version_info
        if python_version < (3, 8):
            self.logger.error(f"❌ Требуется Python 3.8+, найден {python_version}")
            return False
        
        self.logger.info(f"✅ Python {python_version.major}.{python_version.minor}")
        
        # Проверка основных библиотек
        required_packages = {
            'asyncio': True,
            'threading': True,
            'decimal': True,
            'datetime': True,
            'pathlib': True,
            'json': True,
            'logging': True
        }
        
        optional_packages = {
            'tkinter': 'GUI дашборд',
            'numpy': 'Численные вычисления',
            'pandas': 'Анализ данных',
            'matplotlib': 'Графики',
            'fastapi': 'API сервер',
            'uvicorn': 'ASGI сервер',
            'redis': 'Кэширование',
            'psycopg2': 'PostgreSQL',
            'sqlalchemy': 'ORM'
        }
        
        # Проверка обязательных пакетов
        missing_required = []
        for package in required_packages:
            try:
                __import__(package)
                self.logger.info(f"✅ {package}")
            except ImportError:
                self.logger.error(f"❌ {package} - ОБЯЗАТЕЛЬНЫЙ")
                missing_required.append(package)
        
        # Проверка опциональных пакетов
        missing_optional = []
        for package, description in optional_packages.items():
            try:
                __import__(package)
                self.logger.info(f"✅ {package} - {description}")
            except ImportError:
                self.logger.warning(f"⚠️ {package} - {description} (опционально)")
                missing_optional.append(package)
        
        if missing_required:
            self.logger.error("❌ Отсутствуют обязательные зависимости!")
            return False
        
        if missing_optional:
            self.logger.info("ℹ️ Некоторые функции будут недоступны без опциональных пакетов")
            self.offer_install_optional(missing_optional)
        
        return True
    
    def offer_install_optional(self, packages: List[str]):
        """Предложение установки опциональных пакетов"""
        try:
            import tkinter as tk
            from tkinter import messagebox
            
            root = tk.Tk()
            root.withdraw()  # Скрываем главное окно
            
            message = f"""
Обнаружены отсутствующие опциональные пакеты:
{', '.join(packages)}

Установить их автоматически?
Это улучшит функциональность системы.
            """
            
            if messagebox.askyesno("Установка зависимостей", message.strip()):
                self.install_packages(packages)
            
            root.destroy()
            
        except ImportError:
            # Если tkinter недоступен, спрашиваем через консоль
            print(f"\nОбнаружены отсутствующие пакеты: {', '.join(packages)}")
            response = input("Установить их? (y/n): ").lower().strip()
            if response in ['y', 'yes', 'да']:
                self.install_packages(packages)
    
    def install_packages(self, packages: List[str]):
        """Автоматическая установка пакетов"""
        self.logger.info(f"📦 Установка пакетов: {', '.join(packages)}")
        
        for package in packages:
            try:
                self.logger.info(f"Установка {package}...")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.logger.info(f"✅ {package} установлен успешно")
                else:
                    self.logger.error(f"❌ Ошибка установки {package}: {result.stderr}")
                    
            except Exception as e:
                self.logger.error(f"❌ Исключение при установке {package}: {e}")
    
    def start_component(self, component: str) -> bool:
        """Запуск компонента системы"""
        self.logger.info(f"🚀 Запуск компонента: {component}")
        
        try:
            if component == "database":
                return self.start_database()
            elif component == "redis":
                return self.start_redis()
            elif component == "api_server":
                return self.start_api_server()
            elif component == "trading_engine":
                return self.start_trading_engine()
            elif component == "dashboard":
                return self.start_dashboard()
            elif component == "monitoring":
                return self.start_monitoring()
            else:
                self.logger.error(f"❌ Неизвестный компонент: {component}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка запуска {component}: {e}")
            return False
    
    def start_database(self) -> bool:
        """Запуск базы данных"""
        try:
            # Проверяем наличие PostgreSQL или SQLite
            db_file = self.project_root / "data" / "atb_trading.db"
            
            # Инициализация SQLite базы
            import sqlite3
            conn = sqlite3.connect(str(db_file))
            
            # Создание базовых таблиц
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_status (
                    id INTEGER PRIMARY KEY,
                    component TEXT,
                    status TEXT,
                    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trading_sessions (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT UNIQUE,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    mode TEXT,
                    status TEXT,
                    config TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            
            self.system_status['database'] = True
            self.logger.info("✅ База данных инициализирована")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации БД: {e}")
            return False
    
    def start_redis(self) -> bool:
        """Запуск Redis (опционально)"""
        try:
            import redis
            
            # Попытка подключения к Redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            
            self.system_status['redis'] = True
            self.logger.info("✅ Redis подключен")
            return True
            
        except ImportError:
            self.logger.info("ℹ️ Redis не установлен, используем локальный кэш")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ Redis недоступен: {e}")
            return True  # Не критично
    
    def start_api_server(self) -> bool:
        """Запуск API сервера"""
        try:
            # Создаем простой API сервер
            api_script = self.project_root / "api_server.py"
            
            if not api_script.exists():
                self.create_api_server_script()
            
            # Запуск в отдельном процессе
            process = subprocess.Popen([
                sys.executable, str(api_script)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['api_server'] = process
            
            # Даем время на запуск
            time.sleep(2)
            
            if process.poll() is None:  # Процесс запущен
                self.system_status['api_server'] = True
                self.logger.info(f"✅ API сервер запущен (PID: {process.pid})")
                return True
            else:
                self.logger.error("❌ API сервер не запустился")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка запуска API сервера: {e}")
            return False
    
    def start_trading_engine(self) -> bool:
        """Запуск торгового движка"""
        try:
            # Создаем скрипт торгового движка
            engine_script = self.project_root / "trading_engine.py"
            
            if not engine_script.exists():
                self.create_trading_engine_script()
            
            # Запуск в отдельном процессе
            process = subprocess.Popen([
                sys.executable, str(engine_script)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['trading_engine'] = process
            
            # Даем время на запуск
            time.sleep(2)
            
            if process.poll() is None:
                self.system_status['trading_engine'] = True
                self.logger.info(f"✅ Торговый движок запущен (PID: {process.pid})")
                return True
            else:
                self.logger.error("❌ Торговый движок не запустился")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка запуска торгового движка: {e}")
            return False
    
    def start_dashboard(self) -> bool:
        """Запуск дашборда"""
        try:
            # Запуск дашборда в отдельном потоке
            dashboard_thread = threading.Thread(
                target=self.run_dashboard,
                daemon=True
            )
            dashboard_thread.start()
            
            # Даем время на запуск
            time.sleep(3)
            
            self.system_status['dashboard'] = True
            self.logger.info("✅ Дашборд запущен")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка запуска дашборда: {e}")
            return False
    
    def run_dashboard(self):
        """Запуск дашборда в отдельном потоке"""
        try:
            # Проверяем наличие дашборда
            dashboard_files = [
                "interfaces/desktop/integrated_dashboard.py",
                "interfaces/desktop/trading_dashboard.py",
                "run_dashboard.py"
            ]
            
            dashboard_file = None
            for file_path in dashboard_files:
                full_path = self.project_root / file_path
                if full_path.exists():
                    dashboard_file = full_path
                    break
            
            if dashboard_file:
                self.logger.info(f"Запуск дашборда: {dashboard_file}")
                
                # Импорт и запуск
                if "integrated_dashboard" in str(dashboard_file):
                    from interfaces.desktop.integrated_dashboard import IntegratedTradingDashboard
                    dashboard = IntegratedTradingDashboard()
                    dashboard.run()
                elif "trading_dashboard" in str(dashboard_file):
                    from interfaces.desktop.trading_dashboard import ModernTradingDashboard
                    dashboard = ModernTradingDashboard()
                    dashboard.run()
                else:
                    # Запуск через subprocess
                    subprocess.run([sys.executable, str(dashboard_file)])
            else:
                self.logger.error("❌ Файлы дашборда не найдены")
                self.create_simple_dashboard()
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка в потоке дашборда: {e}")
            self.create_simple_dashboard()
    
    def start_monitoring(self) -> bool:
        """Запуск системы мониторинга"""
        try:
            # Запуск мониторинга в отдельном потоке
            monitoring_thread = threading.Thread(
                target=self.run_monitoring,
                daemon=True
            )
            monitoring_thread.start()
            
            self.system_status['monitoring'] = True
            self.logger.info("✅ Система мониторинга запущена")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка запуска мониторинга: {e}")
            return False
    
    def run_monitoring(self):
        """Мониторинг системы"""
        while True:
            try:
                time.sleep(self.config.get('health_check_interval', 30))
                
                # Проверка статуса компонентов
                for component, process in self.processes.items():
                    if process and process.poll() is not None:
                        # Процесс завершился
                        self.logger.warning(f"⚠️ Компонент {component} завершился")
                        self.system_status[component] = False
                        
                        if self.config.get('auto_restart', True):
                            self.logger.info(f"🔄 Перезапуск {component}")
                            self.start_component(component)
                
                # Логирование статуса
                active_components = sum(1 for status in self.system_status.values() if status)
                total_components = len(self.system_status)
                
                self.logger.info(f"📊 Статус системы: {active_components}/{total_components} компонентов активно")
                
            except Exception as e:
                self.logger.error(f"❌ Ошибка мониторинга: {e}")
    
    def create_api_server_script(self):
        """Создание скрипта API сервера"""
        api_content = '''#!/usr/bin/env python3
"""
ATB Trading API Server
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

try:
    from fastapi import FastAPI
    from uvicorn import run
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

if FASTAPI_AVAILABLE:
    app = FastAPI(title="ATB Trading API", version="2.0")
    
    @app.get("/")
    async def root():
        return {"message": "ATB Trading API Server", "status": "running", "time": datetime.now().isoformat()}
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    @app.get("/system/status")
    async def system_status():
        return {
            "database": True,
            "trading_engine": True,
            "dashboard": True,
            "timestamp": datetime.now().isoformat()
        }
    
    if __name__ == "__main__":
        run(app, host="127.0.0.1", port=8000, log_level="info")

else:
    # Простой HTTP сервер
    import http.server
    import socketserver
    
    class SimpleHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                "message": "ATB Trading API Server (Simple)",
                "status": "running",
                "time": datetime.now().isoformat()
            }
            
            self.wfile.write(json.dumps(response).encode())
    
    if __name__ == "__main__":
        PORT = 8000
        with socketserver.TCPServer(("", PORT), SimpleHandler) as httpd:
            print(f"API Server running on port {PORT}")
            httpd.serve_forever()
'''
        
        api_file = self.project_root / "api_server.py"
        with open(api_file, 'w', encoding='utf-8') as f:
            f.write(api_content)
        
        self.logger.info("✅ Создан скрипт API сервера")
    
    def create_trading_engine_script(self):
        """Создание скрипта торгового движка"""
        engine_content = '''#!/usr/bin/env python3
"""
ATB Trading Engine
"""

import asyncio
import logging
import time
from datetime import datetime
from decimal import Decimal

class TradingEngine:
    def __init__(self):
        self.is_running = False
        self.logger = logging.getLogger(__name__)
        
        # Настройка логирования
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | TRADING ENGINE | %(levelname)s | %(message)s'
        )
    
    async def start(self):
        """Запуск торгового движка"""
        self.is_running = True
        self.logger.info("🚀 Торговый движок запущен")
        
        while self.is_running:
            try:
                # Основной цикл торгового движка
                await self.process_trading_cycle()
                await asyncio.sleep(5)  # Цикл каждые 5 секунд
                
            except Exception as e:
                self.logger.error(f"❌ Ошибка в торговом цикле: {e}")
                await asyncio.sleep(10)
    
    async def process_trading_cycle(self):
        """Обработка торгового цикла"""
        # Симуляция торговых операций
        current_time = datetime.now()
        self.logger.info(f"📊 Торговый цикл: {current_time.strftime('%H:%M:%S')}")
        
        # Здесь будет реальная логика торговли
        # - Получение рыночных данных
        # - Анализ сигналов
        # - Выполнение ордеров
        # - Управление позициями
    
    def stop(self):
        """Остановка торгового движка"""
        self.is_running = False
        self.logger.info("⏹️ Торговый движок остановлен")

async def main():
    engine = TradingEngine()
    
    try:
        await engine.start()
    except KeyboardInterrupt:
        engine.stop()
    except Exception as e:
        engine.logger.error(f"❌ Критическая ошибка: {e}")

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        engine_file = self.project_root / "trading_engine.py"
        with open(engine_file, 'w', encoding='utf-8') as f:
            f.write(engine_content)
        
        self.logger.info("✅ Создан скрипт торгового движка")
    
    def create_simple_dashboard(self):
        """Создание простого дашборда как fallback"""
        try:
            import tkinter as tk
            from tkinter import ttk
            
            class SimpleDashboard:
                def __init__(self):
                    self.root = tk.Tk()
                    self.root.title("ATB Trading System - Статус")
                    self.root.geometry("600x400")
                    self.root.configure(bg='#1e1e1e')
                    
                    self.create_ui()
                
                def create_ui(self):
                    # Заголовок
                    title = tk.Label(self.root, text="⚡ ATB Trading System",
                                   font=('Arial', 18, 'bold'),
                                   fg='#00ff88', bg='#1e1e1e')
                    title.pack(pady=20)
                    
                    # Статус системы
                    status_frame = tk.Frame(self.root, bg='#1e1e1e')
                    status_frame.pack(pady=20)
                    
                    tk.Label(status_frame, text="Статус системы:",
                           font=('Arial', 14, 'bold'),
                           fg='white', bg='#1e1e1e').pack()
                    
                    # Показываем статус компонентов
                    for component, status in launcher.system_status.items():
                        color = '#00ff88' if status else '#ff4757'
                        status_text = "✅ Активен" if status else "❌ Неактивен"
                        
                        tk.Label(status_frame, 
                               text=f"{component.replace('_', ' ').title()}: {status_text}",
                               font=('Arial', 10),
                               fg=color, bg='#1e1e1e').pack(pady=2)
                    
                    # Кнопки управления
                    btn_frame = tk.Frame(self.root, bg='#1e1e1e')
                    btn_frame.pack(pady=20)
                    
                    tk.Button(btn_frame, text="🔄 Обновить статус",
                             bg='#3742fa', fg='white',
                             command=self.refresh_status).pack(side='left', padx=10)
                    
                    tk.Button(btn_frame, text="⚙️ Настройки",
                             bg='#2d2d2d', fg='white',
                             command=self.show_settings).pack(side='left', padx=10)
                
                def refresh_status(self):
                    # Обновление статуса
                    self.root.destroy()
                    self.__init__()
                
                def show_settings(self):
                    tk.messagebox.showinfo("Настройки", "Настройки доступны через файл launcher_config.json")
                
                def run(self):
                    self.root.mainloop()
            
            dashboard = SimpleDashboard()
            dashboard.run()
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания простого дашборда: {e}")
    
    def shutdown_system(self):
        """Остановка всей системы"""
        self.logger.info("🛑 Остановка системы...")
        
        # Остановка процессов
        for component, process in self.processes.items():
            if process and process.poll() is None:
                self.logger.info(f"Остановка {component}...")
                process.terminate()
                
                # Ждем завершения
                try:
                    process.wait(timeout=5)
                    self.logger.info(f"✅ {component} остановлен")
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"⚠️ Принудительная остановка {component}")
                    process.kill()
        
        # Обновление статуса
        for component in self.system_status:
            self.system_status[component] = False
        
        self.logger.info("✅ Система остановлена")
    
    def launch_system(self):
        """Запуск всей системы"""
        self.logger.info("=" * 60)
        self.logger.info("🎯 ATB Trading System Launcher v2.0")
        self.logger.info("=" * 60)
        
        # Проверка зависимостей
        if not self.check_dependencies():
            self.logger.error("❌ Критические зависимости отсутствуют!")
            return False
        
        self.logger.info("✅ Все зависимости проверены")
        
        # Последовательный запуск компонентов
        components_to_start = self.config.get('auto_start_components', [
            'database', 'trading_engine', 'dashboard'
        ])
        
        self.logger.info(f"🚀 Запуск компонентов: {', '.join(components_to_start)}")
        
        success_count = 0
        for component in components_to_start:
            self.logger.info(f"Запуск {component}...")
            
            if self.start_component(component):
                success_count += 1
                self.logger.info(f"✅ {component} запущен успешно")
            else:
                self.logger.error(f"❌ Ошибка запуска {component}")
            
            # Задержка между запусками
            time.sleep(self.config.get('startup_delay', 2))
        
        # Запуск мониторинга
        if self.config.get('enable_monitoring', True):
            self.start_component('monitoring')
        
        # Итоговый статус
        total_components = len(components_to_start)
        self.logger.info("=" * 60)
        self.logger.info(f"📊 ИТОГ: {success_count}/{total_components} компонентов запущено успешно")
        
        if success_count > 0:
            self.logger.info("🎉 Система запущена! Дашборд должен открыться автоматически.")
            
            # Показываем информацию о доступных сервисах
            if self.system_status.get('api_server'):
                self.logger.info(f"🌐 API доступен: http://localhost:{self.config.get('api_port', 8000)}")
            
            return True
        else:
            self.logger.error("❌ Не удалось запустить ни одного компонента!")
            return False

# Глобальная переменная для доступа из других модулей
launcher = None

def main():
    """Главная функция launcher'а"""
    global launcher
    
    try:
        launcher = ATBSystemLauncher()
        
        # Запуск системы
        success = launcher.launch_system()
        
        if success:
            # Ожидание завершения (система работает)
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                launcher.logger.info("\n🛑 Получен сигнал остановки...")
                launcher.shutdown_system()
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Критическая ошибка launcher'а: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)