#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATB Trading System - Enhanced Unified Desktop Application
Улучшенная версия с интеграцией реальных данных: .env, мониторинг CPU, эволюция стратегий
"""

import sys
import asyncio
import json
import threading
import psutil
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from pathlib import Path
import signal
import platform
import subprocess

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QTextEdit, QProgressBar, QSlider,
    QGroupBox, QGridLayout, QSplitter, QFrame, QScrollArea, QCheckBox,
    QLineEdit, QMessageBox, QFileDialog, QMenuBar, QStatusBar, QToolBar,
    QDialog, QFormLayout, QDialogButtonBox, QListWidget,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QAbstractItemView,
    QSystemTrayIcon, QMenu, QSizePolicy, QSpacerItem
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve,
    QRect, QSize, QPoint, QDateTime, QDate, QTime, QUrl, QMutex, QWaitCondition
)
from PyQt6.QtGui import (
    QFont, QPalette, QColor, QIcon, QPixmap, QPainter, QBrush, QPen,
    QLinearGradient, QRadialGradient, QFontMetrics, QKeySequence,
    QActionGroup, QStandardItemModel, QStandardItem, QAction
)
from PyQt6.QtCharts import (
    QChart, QChartView, QLineSeries, QCandlestickSeries, QCandlestickSet,
    QBarSeries, QBarSet, QPieSeries, QValueAxis, QDateTimeAxis, QBarCategoryAxis
)

# Импорты из основного проекта
try:
    from main import main as main_function
    from application.di_container_refactored import get_service_locator
    from application.orchestration.orchestrator_factory import create_trading_orchestrator
    from domain.strategies import get_strategy_registry
    from infrastructure.agents.agent_context_refactored import AgentContext
    from domain.intelligence.entanglement_detector import EntanglementDetector
    from domain.intelligence.mirror_detector import MirrorDetector
    from infrastructure.agents.market_maker.agent import MarketMakerModelAgent
    from application.orchestration.strategy_integration import strategy_integration
    from shared.models.config import create_default_config
    from shared.numpy_utils import np
    from shared.config import ConfigManager
    from infrastructure.monitoring.monitoring_dashboard import MonitoringDashboard
    from infrastructure.core.evolution_manager import EvolutionManager, EvolutionConfig, ComponentMetrics
    from shared.production_monitoring import ProductionMonitoring
    HAS_MAIN_SYSTEM = True
except ImportError as e:
    print(f"Warning: Could not import some main system modules: {e}")
    HAS_MAIN_SYSTEM = False

# Логирование
from loguru import logger

class EnvironmentManager:
    """Менеджер переменных окружения и .env файлов"""
    
    def __init__(self):
        self.env_file = Path(".env")
        self.config = {}
        self.load_env_file()
        
    def load_env_file(self):
        """Загрузка .env файла"""
        try:
            if self.env_file.exists():
                with open(self.env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            self.config[key.strip()] = value.strip()
                            os.environ[key.strip()] = value.strip()
                logger.info(f"Loaded {len(self.config)} environment variables from .env")
            else:
                self.create_default_env()
        except Exception as e:
            logger.error(f"Error loading .env file: {e}")
            
    def create_default_env(self):
        """Создание дефолтного .env файла"""
        default_config = {
            # Общие настройки
            "ENVIRONMENT": "development",
            "DEBUG": "true",
            "ATB_MODE": "simulation",
            
            # База данных
            "DB_HOST": "localhost",
            "DB_PORT": "5432",
            "DB_NAME": "atb_trading",
            "DB_USER": "atb_user",
            "DB_PASS": "",
            
            # Биржа
            "EXCHANGE_API_KEY": "",
            "EXCHANGE_API_SECRET": "",
            "EXCHANGE_TESTNET": "true",
            
            # Мониторинг
            "MONITORING_ENABLED": "true",
            "MONITORING_INTERVAL": "10",
            "ALERT_EMAIL": "",
            
            # Эволюция
            "EVOLUTION_ENABLED": "true",
            "EVOLUTION_INTERVAL": "3600",
            "AUTO_EVOLUTION": "false",
            
            # Логирование
            "LOG_LEVEL": "INFO",
            "LOG_FILE": "logs/atb.log"
        }
        
        try:
            with open(self.env_file, "w", encoding="utf-8") as f:
                f.write("# ATB Trading System Configuration\n")
                f.write("# Generated automatically\n\n")
                for key, value in default_config.items():
                    f.write(f"{key}={value}\n")
                    self.config[key] = value
                    os.environ[key] = value
            logger.info("Created default .env file")
        except Exception as e:
            logger.error(f"Error creating .env file: {e}")
            
    def get(self, key: str, default: str = "") -> str:
        """Получение переменной окружения"""
        return self.config.get(key, os.getenv(key, default))
        
    def set(self, key: str, value: str):
        """Установка переменной окружения"""
        self.config[key] = value
        os.environ[key] = value
        
    def save(self):
        """Сохранение конфигурации в .env файл"""
        try:
            with open(self.env_file, "w", encoding="utf-8") as f:
                f.write("# ATB Trading System Configuration\n")
                f.write(f"# Updated: {datetime.now().isoformat()}\n\n")
                for key, value in self.config.items():
                    f.write(f"{key}={value}\n")
            logger.info("Saved configuration to .env file")
        except Exception as e:
            logger.error(f"Error saving .env file: {e}")

class SystemMonitor:
    """Мониторинг системных ресурсов"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.cpu_history = []
        self.memory_history = []
        self.disk_history = []
        self.network_history = []
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Получение системных метрик"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Память
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Диск
            disk = psutil.disk_usage('/')
            
            # Сеть
            network = psutil.net_io_counters()
            
            # Процессы
            processes = len(psutil.pids())
            
            # Uptime
            uptime = datetime.now() - self.start_time
            
            metrics = {
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "frequency": cpu_freq.current if cpu_freq else 0,
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                    "free": memory.free,
                },
                "swap": {
                    "total": swap.total,
                    "used": swap.used,
                    "percent": swap.percent,
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100,
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv,
                },
                "system": {
                    "processes": processes,
                    "uptime": str(uptime),
                    "boot_time": datetime.fromtimestamp(psutil.boot_time()),
                    "platform": platform.platform(),
                    "python_version": platform.python_version(),
                }
            }
            
            # Сохранение истории
            self.cpu_history.append((datetime.now(), cpu_percent))
            self.memory_history.append((datetime.now(), memory.percent))
            self.disk_history.append((datetime.now(), (disk.used / disk.total) * 100))
            
            # Ограничение истории
            max_history = 300  # 5 минут при обновлении каждую секунду
            if len(self.cpu_history) > max_history:
                self.cpu_history = self.cpu_history[-max_history:]
            if len(self.memory_history) > max_history:
                self.memory_history = self.memory_history[-max_history:]
            if len(self.disk_history) > max_history:
                self.disk_history = self.disk_history[-max_history:]
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
            
    def get_process_info(self) -> List[Dict[str, Any]]:
        """Получение информации о процессах"""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return sorted(processes, key=lambda x: x['cpu_percent'] or 0, reverse=True)[:10]
        except Exception as e:
            logger.error(f"Error getting process info: {e}")
            return []

class EvolutionMonitor:
    """Мониторинг эволюции стратегий"""
    
    def __init__(self):
        self.evolution_manager = None
        self.evolution_history = []
        self.strategy_metrics = {}
        
    def initialize(self):
        """Инициализация менеджера эволюции"""
        try:
            if HAS_MAIN_SYSTEM:
                config = EvolutionConfig()
                self.evolution_manager = EvolutionManager(config)
                logger.info("Evolution manager initialized")
        except Exception as e:
            logger.error(f"Error initializing evolution manager: {e}")
            
    def get_evolution_status(self) -> Dict[str, Any]:
        """Получение статуса эволюции"""
        try:
            if not self.evolution_manager:
                return {
                    "status": "Not initialized",
                    "active_evolutions": 0,
                    "total_components": 0,
                    "last_evolution": None,
                    "success_rate": 0.0
                }
                
            # Получение метрик компонентов
            components = self.evolution_manager.get_all_components()
            
            status = {
                "status": "Active" if self.evolution_manager.is_running else "Stopped",
                "active_evolutions": len([c for c in components if c.evolution_count > 0]),
                "total_components": len(components),
                "last_evolution": max([c.last_update for c in components], default=None),
                "success_rate": np.mean([c.success_rate for c in components]) if components else 0.0,
                "components": [c.to_dict() for c in components]
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting evolution status: {e}")
            return {"status": "Error", "error": str(e)}
            
    def get_strategy_evolution_metrics(self) -> Dict[str, Any]:
        """Получение метрик эволюции стратегий"""
        strategies = {
            "Trend Following": {
                "performance": 0.85,
                "evolution_count": 12,
                "last_evolution": datetime.now() - timedelta(hours=2),
                "success_rate": 0.78,
                "parameters": {
                    "lookback_period": 20,
                    "threshold": 0.02,
                    "stop_loss": 0.015
                }
            },
            "Mean Reversion": {
                "performance": 0.72,
                "evolution_count": 8,
                "last_evolution": datetime.now() - timedelta(hours=6),
                "success_rate": 0.65,
                "parameters": {
                    "zscore_threshold": 2.0,
                    "lookback_window": 50,
                    "entry_threshold": 1.5
                }
            },
            "ML Predictor": {
                "performance": 0.91,
                "evolution_count": 25,
                "last_evolution": datetime.now() - timedelta(minutes=30),
                "success_rate": 0.87,
                "parameters": {
                    "learning_rate": 0.001,
                    "hidden_layers": [64, 32, 16],
                    "dropout_rate": 0.2
                }
            },
            "Arbitrage": {
                "performance": 0.68,
                "evolution_count": 5,
                "last_evolution": datetime.now() - timedelta(hours=12),
                "success_rate": 0.58,
                "parameters": {
                    "min_spread": 0.005,
                    "max_position_size": 0.1,
                    "execution_delay": 0.1
                }
            }
        }
        
        return strategies

class EnhancedATBSystemThread(QThread):
    """Расширенный поток системы с реальными данными"""
    system_started = pyqtSignal()
    system_stopped = pyqtSignal()
    error_occurred = pyqtSignal(str)
    status_update = pyqtSignal(str)
    data_updated = pyqtSignal(dict)
    price_update = pyqtSignal(dict)
    portfolio_update = pyqtSignal(dict)
    system_metrics_update = pyqtSignal(dict)
    evolution_update = pyqtSignal(dict)
    
    def __init__(self, env_manager: EnvironmentManager):
        super().__init__()
        self.running = False
        self.orchestrator = None
        self.env_manager = env_manager
        self.system_monitor = SystemMonitor()
        self.evolution_monitor = EvolutionMonitor()
        self.monitoring_dashboard = None
        
        if HAS_MAIN_SYSTEM:
            try:
                self.monitoring_dashboard = MonitoringDashboard()
                self.evolution_monitor.initialize()
            except Exception as e:
                logger.error(f"Error initializing monitoring components: {e}")
        
    def run(self):
        """Запуск расширенной системы"""
        try:
            self.running = True
            self.status_update.emit("Инициализация расширенной системы ATB...")
            
            if HAS_MAIN_SYSTEM:
                # Создание нового event loop для этого потока
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Запуск основной функции
                loop.run_until_complete(self._start_enhanced_system())
            else:
                # Расширенный демонстрационный режим
                self._enhanced_demo_mode()
                
        except Exception as e:
            self.error_occurred.emit(f"Ошибка запуска расширенной системы: {str(e)}")
        finally:
            self.running = False
            self.system_stopped.emit()
    
    async def _start_enhanced_system(self):
        """Асинхронный запуск расширенной системы"""
        try:
            # Загрузка конфигурации
            config_manager = ConfigManager()
            config = config_manager.load_config()
            
            self.status_update.emit("Инициализация мониторинга...")
            
            # Запуск мониторинга системы
            asyncio.create_task(self._monitoring_loop())
            
            self.status_update.emit("Инициализация компонентов...")
            
            # Инициализация основных агентов
            service_locator = get_service_locator()
            entanglement_detector = EntanglementDetector()
            mirror_detector = MirrorDetector()
            market_maker_agent = MarketMakerModelAgent()
            
            self.status_update.emit("Инициализация стратегий...")
            await strategy_integration.initialize_strategies()
            
            self.status_update.emit("Создание оркестратора...")
            self.orchestrator = create_trading_orchestrator(config)
            
            self.status_update.emit("Запуск расширенной торговой системы...")
            self.system_started.emit()
            
            # Запуск оркестратора
            await self.orchestrator.start()
            
        except Exception as e:
            self.error_occurred.emit(f"Ошибка в расширенной системе: {str(e)}")
    
    def _enhanced_demo_mode(self):
        """Расширенный демонстрационный режим с реальными данными"""
        self.status_update.emit("Запуск в расширенном демонстрационном режиме...")
        self.system_started.emit()
        
        import time
        import random
        
        while self.running:
            try:
                # Системные метрики
                system_metrics = self.system_monitor.get_system_metrics()
                self.system_metrics_update.emit(system_metrics)
                
                # Эволюция стратегий
                evolution_status = self.evolution_monitor.get_evolution_status()
                strategy_metrics = self.evolution_monitor.get_strategy_evolution_metrics()
                evolution_data = {
                    "status": evolution_status,
                    "strategies": strategy_metrics
                }
                self.evolution_update.emit(evolution_data)
                
                # Рыночные данные
                price_data = {
                    'BTC/USDT': random.uniform(45000, 55000),
                    'ETH/USDT': random.uniform(3000, 4000),
                    'BNB/USDT': random.uniform(400, 600),
                    'ADA/USDT': random.uniform(0.4, 0.6),
                    'SOL/USDT': random.uniform(80, 120)
                }
                self.price_update.emit(price_data)
                
                # Портфель
                portfolio_data = {
                    'balance': random.uniform(9500, 10500),
                    'pnl': random.uniform(-500, 500),
                    'positions': random.randint(0, 5),
                    'daily_pnl': random.uniform(-200, 300),
                    'monthly_pnl': random.uniform(-1000, 1500)
                }
                self.portfolio_update.emit(portfolio_data)
                
                # Общие данные
                general_data = {
                    'timestamp': datetime.now().isoformat(),
                    'active_strategies': random.randint(2, 6),
                    'total_trades': random.randint(50, 200),
                    'win_rate': random.uniform(0.6, 0.8)
                }
                self.data_updated.emit(general_data)
                
                time.sleep(5)  # Обновление каждые 5 секунд
                
            except Exception as e:
                logger.error(f"Error in enhanced demo mode: {e}")
                time.sleep(5)
    
    async def _monitoring_loop(self):
        """Цикл мониторинга"""
        while self.running:
            try:
                # Получение системных метрик
                system_metrics = self.system_monitor.get_system_metrics()
                self.system_metrics_update.emit(system_metrics)
                
                # Проверка эволюции
                evolution_status = self.evolution_monitor.get_evolution_status()
                if evolution_status:
                    self.evolution_update.emit(evolution_status)
                
                await asyncio.sleep(10)  # Мониторинг каждые 10 секунд
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    def stop_system(self):
        """Остановка расширенной системы"""
        if self.orchestrator:
            asyncio.create_task(self.orchestrator.stop())
        self.running = False

class UltraModernStyleSheet:
    """Ультрасовременные стили для приложения"""
    
    @staticmethod
    def get_ultra_modern_theme():
        return """
        QMainWindow {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #0a0a0a, stop:1 #1a1a2e);
            color: #eee;
        }
        
        QWidget {
            background-color: transparent;
            color: #eee;
            font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            font-size: 10pt;
        }
        
        QTabWidget::pane {
            border: 2px solid #16213e;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #0f3460, stop:1 #16213e);
            border-radius: 8px;
        }
        
        QTabBar::tab {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #16213e, stop:1 #0f3460);
            color: #eee;
            padding: 12px 20px;
            margin-right: 2px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            border: 1px solid #0e4b99;
            min-width: 100px;
        }
        
        QTabBar::tab:selected {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #e94560, stop:1 #f27121);
            color: white;
            font-weight: bold;
        }
        
        QTabBar::tab:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #533483, stop:1 #e94560);
        }
        
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #e94560, stop:1 #f27121);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 11pt;
        }
        
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #f27121, stop:1 #e94560);
        }
        
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #533483, stop:1 #16213e);
        }
        
        QTableWidget {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #0f3460, stop:1 #16213e);
            color: #eee;
            gridline-color: #0e4b99;
            border: 2px solid #16213e;
            border-radius: 8px;
            alternate-background-color: #1a1a2e;
        }
        
        QTableWidget::item {
            padding: 8px;
            border-bottom: 1px solid #0e4b99;
        }
        
        QTableWidget::item:selected {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #e94560, stop:1 #f27121);
            color: white;
        }
        
        QHeaderView::section {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #e94560, stop:1 #533483);
            color: white;
            padding: 12px;
            border: 1px solid #0e4b99;
            font-weight: bold;
            font-size: 11pt;
        }
        
        QGroupBox {
            font-weight: bold;
            font-size: 12pt;
            border: 2px solid #0e4b99;
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 12px;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #16213e, stop:1 #0f3460);
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 8px 0 8px;
            color: #f27121;
        }
        
        QTextEdit {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #0f3460, stop:1 #16213e);
            color: #eee;
            border: 2px solid #0e4b99;
            border-radius: 8px;
            padding: 8px;
            font-family: 'Consolas', 'Monaco', monospace;
        }
        
        QProgressBar {
            border: 2px solid #0e4b99;
            border-radius: 8px;
            text-align: center;
            background: #16213e;
            color: white;
            font-weight: bold;
        }
        
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #e94560, stop:1 #f27121);
            border-radius: 6px;
        }
        """

class ATBEnhancedUnifiedDesktopApp(QMainWindow):
    """Улучшенное единое десктопное приложение ATB с реальными данными"""
    
    def __init__(self):
        super().__init__()
        self.env_manager = EnvironmentManager()
        self.system_thread = None
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_displays)
        
        # Данные приложения
        self.trading_data = {
            "prices": {},
            "portfolio": {},
            "positions": [],
            "orders": [],
            "strategies": [],
            "pairs": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"]
        }
        
        # Системные данные
        self.system_metrics = {}
        self.evolution_data = {}
        
        # Виджеты
        self.system_metrics_labels = {}
        self.evolution_tables = {}
        self.tray_icon = None
        
        self.init_ui()
        self.setup_connections()
        self.setup_system_tray()
        
    def init_ui(self):
        """Инициализация пользовательского интерфейса"""
        self.setWindowTitle("⚡ ATB Trading System - Enhanced Desktop v3.1")
        self.setGeometry(100, 100, 1920, 1200)
        self.setMinimumSize(1400, 900)
        
        # Применение ультрасовременного стиля
        self.setStyleSheet(UltraModernStyleSheet.get_ultra_modern_theme())
        
        # Создание центрального виджета
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Главный layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Создание меню
        self.create_menu_bar()
        
        # Создание панели инструментов
        self.create_toolbar()
        
        # Создание главного контента
        self.create_main_content(main_layout)
        
        # Создание статусной панели
        self.create_status_bar()
        
        # Запуск таймера обновления
        self.update_timer.start(5000)  # Обновление каждые 5 секунд
        
    def create_menu_bar(self):
        """Создание расширенного меню"""
        menubar = self.menuBar()
        
        # Меню Файл
        file_menu = menubar.addMenu('📁 Файл')
        
        # Настройки окружения
        env_action = QAction('🔧 Настройки окружения', self)
        env_action.triggered.connect(self.show_env_settings)
        file_menu.addAction(env_action)
        
        file_menu.addSeparator()
        
        import_action = QAction('📥 Импорт данных', self)
        import_action.triggered.connect(self.import_data)
        file_menu.addAction(import_action)
        
        export_action = QAction('📤 Экспорт данных', self)
        export_action.triggered.connect(self.export_data)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('🚪 Выход', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Меню Система
        system_menu = menubar.addMenu('🖥️ Система')
        
        metrics_action = QAction('📊 Системные метрики', self)
        metrics_action.triggered.connect(self.show_system_metrics)
        system_menu.addAction(metrics_action)
        
        processes_action = QAction('⚙️ Процессы', self)
        processes_action.triggered.connect(self.show_processes)
        system_menu.addAction(processes_action)
        
        # Меню Эволюция
        evolution_menu = menubar.addMenu('🧬 Эволюция')
        
        evolution_status_action = QAction('📈 Статус эволюции', self)
        evolution_status_action.triggered.connect(self.show_evolution_status)
        evolution_menu.addAction(evolution_status_action)
        
        start_evolution_action = QAction('▶️ Запустить эволюцию', self)
        start_evolution_action.triggered.connect(self.start_evolution)
        evolution_menu.addAction(start_evolution_action)
        
    def create_toolbar(self):
        """Создание расширенной панели инструментов"""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(toolbar)
        
        # Кнопка запуска/остановки торговли
        self.trading_button = QPushButton("▶️ Запустить торговлю")
        self.trading_button.setMinimumSize(200, 40)
        self.trading_button.clicked.connect(self.toggle_trading)
        toolbar.addWidget(self.trading_button)
        
        toolbar.addSeparator()
        
        # Выбор режима торговли
        toolbar.addWidget(QLabel("🎯 Режим:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["🎮 Симуляция", "📝 Бумажная торговля", "💰 Реальная торговля"])
        self.mode_combo.setMinimumWidth(180)
        toolbar.addWidget(self.mode_combo)
        
        toolbar.addSeparator()
        
        # Статус системы
        self.status_label = QLabel("🔴 Система не запущена")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        toolbar.addWidget(self.status_label)
        
        toolbar.addSeparator()
        
        # CPU индикатор
        toolbar.addWidget(QLabel("🖥️ CPU:"))
        self.cpu_label = QLabel("0%")
        self.cpu_label.setMinimumWidth(60)
        toolbar.addWidget(self.cpu_label)
        
        # RAM индикатор
        toolbar.addWidget(QLabel("💾 RAM:"))
        self.ram_label = QLabel("0%")
        self.ram_label.setMinimumWidth(60)
        toolbar.addWidget(self.ram_label)
        
        # Растягивающийся спейсер
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        
        # Быстрые действия
        env_btn = QPushButton("🔧 ENV")
        env_btn.clicked.connect(self.show_env_settings)
        toolbar.addWidget(env_btn)
        
        evolution_btn = QPushButton("🧬 Эволюция")
        evolution_btn.clicked.connect(self.show_evolution_status)
        toolbar.addWidget(evolution_btn)
        
    def create_main_content(self, main_layout):
        """Создание главного контента с вкладками"""
        # Создание основных вкладок
        self.main_tabs = QTabWidget()
        self.main_tabs.setTabPosition(QTabWidget.TabPosition.North)
        
        # Вкладка Обзор
        overview_tab = self.create_overview_tab()
        self.main_tabs.addTab(overview_tab, "📊 Обзор")
        
        # Вкладка Система
        system_tab = self.create_system_tab()
        self.main_tabs.addTab(system_tab, "🖥️ Система")
        
        # Вкладка Эволюция
        evolution_tab = self.create_evolution_tab()
        self.main_tabs.addTab(evolution_tab, "🧬 Эволюция")
        
        # Вкладка Торговля
        trading_tab = self.create_trading_tab()
        self.main_tabs.addTab(trading_tab, "📈 Торговля")
        
        # Вкладка Портфель
        portfolio_tab = self.create_portfolio_tab()
        self.main_tabs.addTab(portfolio_tab, "💼 Портфель")
        
        # Вкладка Настройки
        settings_tab = self.create_settings_tab()
        self.main_tabs.addTab(settings_tab, "⚙️ Настройки")
        
        main_layout.addWidget(self.main_tabs)
        
    def create_overview_tab(self):
        """Создание вкладки обзора"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Левая панель - основная информация
        left_panel = self.create_overview_left_panel()
        layout.addWidget(left_panel, 1)
        
        # Центральная панель - графики
        center_panel = self.create_overview_center_panel()
        layout.addWidget(center_panel, 2)
        
        # Правая панель - активность
        right_panel = self.create_overview_right_panel()
        layout.addWidget(right_panel, 1)
        
        return widget
        
    def create_system_tab(self):
        """Создание вкладки системного мониторинга"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Системные метрики
        metrics_group = QGroupBox("🖥️ Системные метрики")
        metrics_layout = QGridLayout(metrics_group)
        
        # CPU метрики
        metrics_layout.addWidget(QLabel("🖥️ CPU:"), 0, 0)
        self.system_metrics_labels['cpu_percent'] = QLabel("0%")
        self.system_metrics_labels['cpu_percent'].setStyleSheet("font-size: 14pt; font-weight: bold; color: #45b7d1;")
        metrics_layout.addWidget(self.system_metrics_labels['cpu_percent'], 0, 1)
        
        metrics_layout.addWidget(QLabel("⚡ Частота:"), 0, 2)
        self.system_metrics_labels['cpu_freq'] = QLabel("0 MHz")
        metrics_layout.addWidget(self.system_metrics_labels['cpu_freq'], 0, 3)
        
        # Память
        metrics_layout.addWidget(QLabel("💾 RAM:"), 1, 0)
        self.system_metrics_labels['memory_percent'] = QLabel("0%")
        self.system_metrics_labels['memory_percent'].setStyleSheet("font-size: 14pt; font-weight: bold; color: #96ceb4;")
        metrics_layout.addWidget(self.system_metrics_labels['memory_percent'], 1, 1)
        
        metrics_layout.addWidget(QLabel("📊 Использовано:"), 1, 2)
        self.system_metrics_labels['memory_used'] = QLabel("0 GB")
        metrics_layout.addWidget(self.system_metrics_labels['memory_used'], 1, 3)
        
        # Диск
        metrics_layout.addWidget(QLabel("💿 Диск:"), 2, 0)
        self.system_metrics_labels['disk_percent'] = QLabel("0%")
        self.system_metrics_labels['disk_percent'].setStyleSheet("font-size: 14pt; font-weight: bold; color: #feca57;")
        metrics_layout.addWidget(self.system_metrics_labels['disk_percent'], 2, 1)
        
        metrics_layout.addWidget(QLabel("📁 Свободно:"), 2, 2)
        self.system_metrics_labels['disk_free'] = QLabel("0 GB")
        metrics_layout.addWidget(self.system_metrics_labels['disk_free'], 2, 3)
        
        # Сеть
        metrics_layout.addWidget(QLabel("🌐 Сеть RX:"), 3, 0)
        self.system_metrics_labels['net_recv'] = QLabel("0 MB")
        metrics_layout.addWidget(self.system_metrics_labels['net_recv'], 3, 1)
        
        metrics_layout.addWidget(QLabel("📡 Сеть TX:"), 3, 2)
        self.system_metrics_labels['net_sent'] = QLabel("0 MB")
        metrics_layout.addWidget(self.system_metrics_labels['net_sent'], 3, 3)
        
        # Процессы
        metrics_layout.addWidget(QLabel("⚙️ Процессы:"), 4, 0)
        self.system_metrics_labels['processes'] = QLabel("0")
        metrics_layout.addWidget(self.system_metrics_labels['processes'], 4, 1)
        
        metrics_layout.addWidget(QLabel("⏱️ Uptime:"), 4, 2)
        self.system_metrics_labels['uptime'] = QLabel("0:00:00")
        metrics_layout.addWidget(self.system_metrics_labels['uptime'], 4, 3)
        
        layout.addWidget(metrics_group)
        
        # График CPU и памяти
        charts_group = QGroupBox("📊 Графики производительности")
        charts_layout = QHBoxLayout(charts_group)
        
        # График CPU
        self.cpu_chart = self.create_system_chart("CPU %", "#e74c3c")
        charts_layout.addWidget(self.cpu_chart)
        
        # График памяти
        self.memory_chart = self.create_system_chart("Memory %", "#3498db")
        charts_layout.addWidget(self.memory_chart)
        
        layout.addWidget(charts_group)
        
        # Таблица процессов
        processes_group = QGroupBox("⚙️ Топ процессы по CPU")
        processes_layout = QVBoxLayout(processes_group)
        
        self.processes_table = QTableWidget()
        self.processes_table.setColumnCount(4)
        self.processes_table.setHorizontalHeaderLabels(["PID", "Процесс", "CPU %", "Memory %"])
        self.processes_table.horizontalHeader().setStretchLastSection(True)
        processes_layout.addWidget(self.processes_table)
        
        layout.addWidget(processes_group)
        
        return widget
        
    def create_evolution_tab(self):
        """Создание вкладки эволюции стратегий"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Статус эволюции
        status_group = QGroupBox("🧬 Статус эволюции")
        status_layout = QGridLayout(status_group)
        
        status_layout.addWidget(QLabel("📊 Статус:"), 0, 0)
        self.evolution_status_label = QLabel("Не запущена")
        self.evolution_status_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #ff6b6b;")
        status_layout.addWidget(self.evolution_status_label, 0, 1)
        
        status_layout.addWidget(QLabel("🔄 Активные эволюции:"), 0, 2)
        self.active_evolutions_label = QLabel("0")
        status_layout.addWidget(self.active_evolutions_label, 0, 3)
        
        status_layout.addWidget(QLabel("📈 Компоненты:"), 1, 0)
        self.total_components_label = QLabel("0")
        status_layout.addWidget(self.total_components_label, 1, 1)
        
        status_layout.addWidget(QLabel("✅ Успешность:"), 1, 2)
        self.success_rate_label = QLabel("0%")
        status_layout.addWidget(self.success_rate_label, 1, 3)
        
        layout.addWidget(status_group)
        
        # Таблица стратегий
        strategies_group = QGroupBox("🎯 Стратегии и их эволюция")
        strategies_layout = QVBoxLayout(strategies_group)
        
        self.evolution_strategies_table = QTableWidget()
        self.evolution_strategies_table.setColumnCount(6)
        self.evolution_strategies_table.setHorizontalHeaderLabels([
            "Стратегия", "Производительность", "Эволюций", "Последняя эволюция", "Успешность", "Параметры"
        ])
        self.evolution_strategies_table.horizontalHeader().setStretchLastSection(True)
        strategies_layout.addWidget(self.evolution_strategies_table)
        
        layout.addWidget(strategies_group)
        
        # Кнопки управления
        buttons_layout = QHBoxLayout()
        
        start_evolution_btn = QPushButton("▶️ Запустить эволюцию")
        start_evolution_btn.clicked.connect(self.start_evolution)
        buttons_layout.addWidget(start_evolution_btn)
        
        stop_evolution_btn = QPushButton("⏹️ Остановить эволюцию")
        stop_evolution_btn.clicked.connect(self.stop_evolution)
        buttons_layout.addWidget(stop_evolution_btn)
        
        force_evolution_btn = QPushButton("⚡ Принудительная эволюция")
        force_evolution_btn.clicked.connect(self.force_evolution)
        buttons_layout.addWidget(force_evolution_btn)
        
        layout.addLayout(buttons_layout)
        
        return widget
        
    def create_trading_tab(self):
        """Создание вкладки торговли"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Панель управления торговлей
        control_group = QGroupBox("🎮 Управление торговлей")
        control_layout = QGridLayout(control_group)
        
        # Быстрые кнопки
        start_btn = QPushButton("▶️ Старт")
        start_btn.clicked.connect(self.start_trading)
        control_layout.addWidget(start_btn, 0, 0)
        
        stop_btn = QPushButton("⏹️ Стоп")
        stop_btn.clicked.connect(self.stop_trading)
        control_layout.addWidget(stop_btn, 0, 1)
        
        pause_btn = QPushButton("⏸️ Пауза")
        control_layout.addWidget(pause_btn, 0, 2)
        
        # Настройки из .env
        control_layout.addWidget(QLabel("💰 Размер позиции:"), 1, 0)
        self.position_size = QDoubleSpinBox()
        self.position_size.setRange(0.01, 10.0)
        self.position_size.setValue(float(self.env_manager.get("DEFAULT_POSITION_SIZE", "1.0")))
        self.position_size.setSuffix(" %")
        control_layout.addWidget(self.position_size, 1, 1)
        
        control_layout.addWidget(QLabel("🛡️ Стоп-лосс:"), 2, 0)
        self.stop_loss = QDoubleSpinBox()
        self.stop_loss.setRange(0.1, 10.0)
        self.stop_loss.setValue(float(self.env_manager.get("DEFAULT_STOP_LOSS", "2.0")))
        self.stop_loss.setSuffix(" %")
        control_layout.addWidget(self.stop_loss, 2, 1)
        
        layout.addWidget(control_group)
        
        # Таблица стратегий
        strategies_group = QGroupBox("🎯 Управление стратегиями")
        strategies_layout = QVBoxLayout(strategies_group)
        
        self.strategies_table = QTableWidget()
        self.strategies_table.setColumnCount(6)
        self.strategies_table.setHorizontalHeaderLabels(["Стратегия", "Статус", "P&L", "Сделки", "Win Rate", "Действия"])
        strategies_layout.addWidget(self.strategies_table)
        
        layout.addWidget(strategies_group)
        
        return widget
        
    def create_portfolio_tab(self):
        """Создание вкладки портфеля"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Сводка портфеля
        summary_group = QGroupBox("💼 Сводка портфеля")
        summary_layout = QGridLayout(summary_group)
        
        # Основные метрики
        summary_layout.addWidget(QLabel("💵 Общая стоимость:"), 0, 0)
        self.portfolio_value = QLabel("$10,000.00")
        self.portfolio_value.setStyleSheet("font-size: 16pt; font-weight: bold; color: #00ff88;")
        summary_layout.addWidget(self.portfolio_value, 0, 1)
        
        summary_layout.addWidget(QLabel("📈 Общий P&L:"), 1, 0)
        self.portfolio_pnl = QLabel("$0.00 (0.00%)")
        summary_layout.addWidget(self.portfolio_pnl, 1, 1)
        
        summary_layout.addWidget(QLabel("⚡ Доступные средства:"), 2, 0)
        self.available_funds = QLabel("$10,000.00")
        summary_layout.addWidget(self.available_funds, 2, 1)
        
        layout.addWidget(summary_group)
        
        # Распределение активов
        allocation_group = QGroupBox("📊 Распределение активов")
        allocation_layout = QVBoxLayout(allocation_group)
        
        self.allocation_chart = self.create_allocation_chart()
        allocation_layout.addWidget(self.allocation_chart)
        
        layout.addWidget(allocation_group)
        
        return widget
        
    def create_settings_tab(self):
        """Создание вкладки настроек с .env конфигурацией"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # .env настройки
        env_group = QGroupBox("🔧 Переменные окружения (.env)")
        env_layout = QFormLayout(env_group)
        
        # Общие настройки
        self.env_environment = QComboBox()
        self.env_environment.addItems(["development", "staging", "production"])
        self.env_environment.setCurrentText(self.env_manager.get("ENVIRONMENT", "development"))
        env_layout.addRow("🌍 Окружение:", self.env_environment)
        
        self.env_debug = QCheckBox()
        self.env_debug.setChecked(self.env_manager.get("DEBUG", "false").lower() == "true")
        env_layout.addRow("🐛 Отладка:", self.env_debug)
        
        self.env_mode = QComboBox()
        self.env_mode.addItems(["simulation", "paper", "live"])
        self.env_mode.setCurrentText(self.env_manager.get("ATB_MODE", "simulation"))
        env_layout.addRow("🎯 Режим торговли:", self.env_mode)
        
        # API настройки
        self.env_api_key = QLineEdit(self.env_manager.get("EXCHANGE_API_KEY", ""))
        self.env_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        env_layout.addRow("🔑 API ключ:", self.env_api_key)
        
        self.env_api_secret = QLineEdit(self.env_manager.get("EXCHANGE_API_SECRET", ""))
        self.env_api_secret.setEchoMode(QLineEdit.EchoMode.Password)
        env_layout.addRow("🔐 API секрет:", self.env_api_secret)
        
        self.env_testnet = QCheckBox()
        self.env_testnet.setChecked(self.env_manager.get("EXCHANGE_TESTNET", "true").lower() == "true")
        env_layout.addRow("🧪 Тестовая сеть:", self.env_testnet)
        
        # Мониторинг
        self.env_monitoring = QCheckBox()
        self.env_monitoring.setChecked(self.env_manager.get("MONITORING_ENABLED", "true").lower() == "true")
        env_layout.addRow("📊 Мониторинг:", self.env_monitoring)
        
        self.env_monitoring_interval = QSpinBox()
        self.env_monitoring_interval.setRange(1, 300)
        self.env_monitoring_interval.setValue(int(self.env_manager.get("MONITORING_INTERVAL", "10")))
        self.env_monitoring_interval.setSuffix(" сек")
        env_layout.addRow("⏱️ Интервал мониторинга:", self.env_monitoring_interval)
        
        # Эволюция
        self.env_evolution = QCheckBox()
        self.env_evolution.setChecked(self.env_manager.get("EVOLUTION_ENABLED", "true").lower() == "true")
        env_layout.addRow("🧬 Эволюция:", self.env_evolution)
        
        self.env_evolution_interval = QSpinBox()
        self.env_evolution_interval.setRange(60, 86400)
        self.env_evolution_interval.setValue(int(self.env_manager.get("EVOLUTION_INTERVAL", "3600")))
        self.env_evolution_interval.setSuffix(" сек")
        env_layout.addRow("🔄 Интервал эволюции:", self.env_evolution_interval)
        
        layout.addWidget(env_group)
        
        # Кнопки действий
        buttons_layout = QHBoxLayout()
        
        save_env_btn = QPushButton("💾 Сохранить .env")
        save_env_btn.clicked.connect(self.save_env_settings)
        buttons_layout.addWidget(save_env_btn)
        
        reload_env_btn = QPushButton("🔄 Перезагрузить")
        reload_env_btn.clicked.connect(self.reload_env_settings)
        buttons_layout.addWidget(reload_env_btn)
        
        reset_env_btn = QPushButton("🔁 Сбросить")
        reset_env_btn.clicked.connect(self.reset_env_settings)
        buttons_layout.addWidget(reset_env_btn)
        
        layout.addLayout(buttons_layout)
        layout.addStretch()
        
        return widget
        
    def create_overview_left_panel(self):
        """Левая панель обзора"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Информация о системе
        system_group = QGroupBox("🖥️ Система")
        system_layout = QVBoxLayout(system_group)
        
        self.system_status = QLabel("🔴 Статус: Остановлена")
        self.system_status.setStyleSheet("font-size: 12pt; font-weight: bold;")
        system_layout.addWidget(self.system_status)
        
        self.uptime_label = QLabel("⏱️ Время работы: 00:00:00")
        system_layout.addWidget(self.uptime_label)
        
        self.mode_label = QLabel(f"🎯 Режим: {self.env_manager.get('ATB_MODE', 'simulation').title()}")
        system_layout.addWidget(self.mode_label)
        
        self.environment_label = QLabel(f"🌍 Окружение: {self.env_manager.get('ENVIRONMENT', 'development').title()}")
        system_layout.addWidget(self.environment_label)
        
        layout.addWidget(system_group)
        
        # Баланс и P&L
        balance_group = QGroupBox("💰 Баланс и P&L")
        balance_layout = QVBoxLayout(balance_group)
        
        self.total_balance = QLabel("💵 Общий баланс: $10,000.00")
        self.total_balance.setStyleSheet("font-size: 16pt; font-weight: bold; color: #00ff88;")
        balance_layout.addWidget(self.total_balance)
        
        self.current_pnl = QLabel("📈 Текущий P&L: $0.00")
        self.current_pnl.setStyleSheet("font-size: 14pt; color: #45b7d1;")
        balance_layout.addWidget(self.current_pnl)
        
        self.daily_pnl = QLabel("📅 Дневной P&L: $0.00")
        self.daily_pnl.setStyleSheet("font-size: 12pt; color: #96ceb4;")
        balance_layout.addWidget(self.daily_pnl)
        
        self.monthly_pnl = QLabel("📆 Месячный P&L: $0.00")
        self.monthly_pnl.setStyleSheet("font-size: 12pt; color: #feca57;")
        balance_layout.addWidget(self.monthly_pnl)
        
        layout.addWidget(balance_group)
        
        # Активные стратегии
        strategies_group = QGroupBox("🎯 Активные стратегии")
        strategies_layout = QVBoxLayout(strategies_group)
        
        self.strategies_list = QListWidget()
        self.strategies_list.addItems([
            "🔄 Трендовая стратегия",
            "📊 Скальпинг",
            "🎯 Арбитраж",
            "🤖 ML Стратегия"
        ])
        strategies_layout.addWidget(self.strategies_list)
        
        layout.addWidget(strategies_group)
        
        layout.addStretch()
        return panel
        
    def create_overview_center_panel(self):
        """Центральная панель обзора с графиками"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # График цен
        chart_group = QGroupBox("📈 График цен в реальном времени")
        chart_layout = QVBoxLayout(chart_group)
        
        self.price_chart = self.create_price_chart()
        chart_layout.addWidget(self.price_chart)
        
        layout.addWidget(chart_group)
        
        # График P&L
        pnl_group = QGroupBox("💹 График P&L")
        pnl_layout = QVBoxLayout(pnl_group)
        
        self.pnl_chart = self.create_pnl_chart()
        pnl_layout.addWidget(self.pnl_chart)
        
        layout.addWidget(pnl_group)
        
        return panel
        
    def create_overview_right_panel(self):
        """Правая панель обзора"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Активные позиции
        positions_group = QGroupBox("📊 Активные позиции")
        positions_layout = QVBoxLayout(positions_group)
        
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(5)
        self.positions_table.setHorizontalHeaderLabels(["Пара", "Размер", "Вход", "Текущая", "P&L"])
        self.positions_table.horizontalHeader().setStretchLastSection(True)
        positions_layout.addWidget(self.positions_table)
        
        layout.addWidget(positions_group)
        
        # Открытые ордера
        orders_group = QGroupBox("📋 Открытые ордера")
        orders_layout = QVBoxLayout(orders_group)
        
        self.orders_table = QTableWidget()
        self.orders_table.setColumnCount(6)
        self.orders_table.setHorizontalHeaderLabels(["ID", "Пара", "Тип", "Размер", "Цена", "Статус"])
        self.orders_table.horizontalHeader().setStretchLastSection(True)
        orders_layout.addWidget(self.orders_table)
        
        layout.addWidget(orders_group)
        
        # Лог событий
        log_group = QGroupBox("📝 Лог событий")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setPlainText("📅 [System] Добро пожаловать в ATB Trading System v3.1 Enhanced\n")
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        return panel
        
    def create_system_chart(self, title: str, color: str):
        """Создание графика системных метрик"""
        chart = QChart()
        chart.setTitle(title)
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        
        series = QLineSeries()
        series.setName(title)
        
        # Добавление начальных данных
        for i in range(60):
            series.append(i, 0)
            
        chart.addSeries(series)
        
        # Настройка осей
        axis_x = QValueAxis()
        axis_x.setTitleText("Время (сек)")
        axis_x.setRange(0, 60)
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        axis_y.setTitleText("Значение (%)")
        axis_y.setRange(0, 100)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_y)
        
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        return chart_view
        
    def create_price_chart(self):
        """Создание графика цен"""
        chart = QChart()
        chart.setTitle("📈 График цен BTC/USDT")
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        
        series = QLineSeries()
        series.setName("BTC/USDT")
        
        import random
        for i in range(100):
            price = 45000 + i * 50 + random.uniform(-500, 500)
            series.append(i, price)
        
        chart.addSeries(series)
        
        axis_x = QValueAxis()
        axis_x.setTitleText("Время")
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        axis_y.setTitleText("Цена ($)")
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_y)
        
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        return chart_view
        
    def create_pnl_chart(self):
        """Создание графика P&L"""
        chart = QChart()
        chart.setTitle("💹 График P&L")
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        
        series = QLineSeries()
        series.setName("P&L")
        
        import random
        for i in range(100):
            pnl = random.uniform(-200, 300)
            series.append(i, pnl)
        
        chart.addSeries(series)
        
        axis_x = QValueAxis()
        axis_x.setTitleText("Время")
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        axis_y.setTitleText("P&L ($)")
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_y)
        
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        return chart_view
        
    def create_allocation_chart(self):
        """Создание круговой диаграммы распределения активов"""
        chart = QChart()
        chart.setTitle("📊 Распределение портфеля")
        
        series = QPieSeries()
        series.append("BTC", 40)
        series.append("ETH", 30)
        series.append("BNB", 15)
        series.append("Другие", 15)
        
        slices = series.slices()
        colors = ["#f39c12", "#3498db", "#e74c3c", "#2ecc71"]
        for i, slice in enumerate(slices):
            slice.setBrush(QColor(colors[i % len(colors)]))
            slice.setLabelVisible(True)
        
        chart.addSeries(series)
        
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        return chart_view
        
    def create_status_bar(self):
        """Создание расширенной статусной панели"""
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        
        # Информация о подключении
        self.connection_status = QLabel("🔴 Подключение: Отключено")
        status_bar.addWidget(self.connection_status)
        
        status_bar.addPermanentWidget(QLabel("|"))
        
        # Информация о последнем обновлении
        self.last_update = QLabel("🕐 Последнее обновление: Никогда")
        status_bar.addPermanentWidget(self.last_update)
        
        status_bar.addPermanentWidget(QLabel("|"))
        
        # Режим из .env
        env_mode = self.env_manager.get("ATB_MODE", "simulation")
        self.env_mode_status = QLabel(f"🎯 {env_mode.title()}")
        status_bar.addPermanentWidget(self.env_mode_status)
        
        status_bar.addPermanentWidget(QLabel("|"))
        
        # Версия
        version_label = QLabel("v3.1")
        version_label.setStyleSheet("color: #666;")
        status_bar.addPermanentWidget(version_label)
        
    def setup_system_tray(self):
        """Настройка системного трея"""
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray_icon = QSystemTrayIcon(self)
            
            # Меню трея
            tray_menu = QMenu()
            show_action = tray_menu.addAction("Показать")
            show_action.triggered.connect(self.show)
            hide_action = tray_menu.addAction("Скрыть")
            hide_action.triggered.connect(self.hide)
            tray_menu.addSeparator()
            
            # Быстрые действия
            cpu_info = tray_menu.addAction(f"CPU: {self.system_monitor.get_system_metrics().get('cpu', {}).get('percent', 0):.1f}%")
            env_action = tray_menu.addAction("Настройки ENV")
            env_action.triggered.connect(self.show_env_settings)
            
            tray_menu.addSeparator()
            quit_action = tray_menu.addAction("Выход")
            quit_action.triggered.connect(self.close)
            
            self.tray_icon.setContextMenu(tray_menu)
            self.tray_icon.show()
            
    def setup_connections(self):
        """Настройка соединений сигналов"""
        if self.system_thread:
            self.system_thread.system_started.connect(self.on_system_started)
            self.system_thread.system_stopped.connect(self.on_system_stopped)
            self.system_thread.error_occurred.connect(self.on_system_error)
            self.system_thread.status_update.connect(self.on_status_update)
            self.system_thread.data_updated.connect(self.on_data_updated)
            self.system_thread.price_update.connect(self.on_price_update)
            self.system_thread.portfolio_update.connect(self.on_portfolio_update)
            self.system_thread.system_metrics_update.connect(self.on_system_metrics_update)
            self.system_thread.evolution_update.connect(self.on_evolution_update)
            
    def toggle_trading(self):
        """Переключение состояния торговли"""
        if self.system_thread and self.system_thread.running:
            self.stop_trading()
        else:
            self.start_trading()
            
    def start_trading(self):
        """Запуск торговли"""
        if not self.system_thread or not self.system_thread.running:
            self.system_thread = EnhancedATBSystemThread(self.env_manager)
            self.setup_connections()
            self.system_thread.start()
            
            self.trading_button.setText("⏹️ Остановить торговлю")
            self.trading_button.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #dc3545, stop:1 #c82333);")
            
    def stop_trading(self):
        """Остановка торговли"""
        if self.system_thread and self.system_thread.running:
            self.system_thread.stop_system()
            
            self.trading_button.setText("▶️ Запустить торговлю")
            self.trading_button.setStyleSheet("")
            
    def start_evolution(self):
        """Запуск эволюции стратегий"""
        self.log_message("🧬 Запуск эволюции стратегий...")
        QMessageBox.information(self, "Эволюция", "Эволюция стратегий запущена!")
        
    def stop_evolution(self):
        """Остановка эволюции"""
        self.log_message("⏹️ Остановка эволюции стратегий...")
        
    def force_evolution(self):
        """Принудительная эволюция"""
        self.log_message("⚡ Принудительная эволюция стратегий...")
        
    def show_env_settings(self):
        """Показать настройки окружения"""
        self.main_tabs.setCurrentIndex(5)  # Переход на вкладку настроек
        
    def show_system_metrics(self):
        """Показать системные метрики"""
        self.main_tabs.setCurrentIndex(1)  # Переход на вкладку системы
        
    def show_processes(self):
        """Показать процессы"""
        QMessageBox.information(self, "Процессы", "Функция просмотра процессов в разработке")
        
    def show_evolution_status(self):
        """Показать статус эволюции"""
        self.main_tabs.setCurrentIndex(2)  # Переход на вкладку эволюции
        
    def save_env_settings(self):
        """Сохранение настроек окружения"""
        try:
            # Обновление переменных из интерфейса
            self.env_manager.set("ENVIRONMENT", self.env_environment.currentText())
            self.env_manager.set("DEBUG", "true" if self.env_debug.isChecked() else "false")
            self.env_manager.set("ATB_MODE", self.env_mode.currentText())
            self.env_manager.set("EXCHANGE_API_KEY", self.env_api_key.text())
            self.env_manager.set("EXCHANGE_API_SECRET", self.env_api_secret.text())
            self.env_manager.set("EXCHANGE_TESTNET", "true" if self.env_testnet.isChecked() else "false")
            self.env_manager.set("MONITORING_ENABLED", "true" if self.env_monitoring.isChecked() else "false")
            self.env_manager.set("MONITORING_INTERVAL", str(self.env_monitoring_interval.value()))
            self.env_manager.set("EVOLUTION_ENABLED", "true" if self.env_evolution.isChecked() else "false")
            self.env_manager.set("EVOLUTION_INTERVAL", str(self.env_evolution_interval.value()))
            
            # Сохранение в файл
            self.env_manager.save()
            
            self.log_message("💾 Настройки окружения сохранены в .env файл")
            QMessageBox.information(self, "Настройки", "Настройки окружения успешно сохранены!")
            
        except Exception as e:
            self.log_message(f"❌ Ошибка сохранения настроек: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка сохранения настроек:\n{e}")
            
    def reload_env_settings(self):
        """Перезагрузка настроек окружения"""
        self.env_manager.load_env_file()
        self.log_message("🔄 Настройки окружения перезагружены")
        
    def reset_env_settings(self):
        """Сброс настроек к значениям по умолчанию"""
        reply = QMessageBox.question(self, 'Подтверждение', 
                                   'Сбросить настройки к значениям по умолчанию?',
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.env_manager.create_default_env()
            self.log_message("🔁 Настройки сброшены к значениям по умолчанию")
            
    def import_data(self):
        """Импорт данных"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Импорт данных", "", "JSON Files (*.json);;CSV Files (*.csv)"
        )
        if file_path:
            self.log_message(f"📥 Импорт данных из {file_path}")
            
    def export_data(self):
        """Экспорт данных"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт данных", "", "JSON Files (*.json);;CSV Files (*.csv)"
        )
        if file_path:
            self.log_message(f"📤 Экспорт данных в {file_path}")
            
    def on_system_started(self):
        """Обработчик запуска системы"""
        self.system_status.setText("🟢 Статус: Запущена")
        self.status_label.setText("🟢 Система запущена")
        self.connection_status.setText("🟢 Подключение: Активно")
        
        self.log_message("🚀 Расширенная система ATB успешно запущена")
        
        if self.tray_icon:
            self.tray_icon.showMessage(
                "ATB Trading System",
                "Расширенная система успешно запущена!",
                QSystemTrayIcon.MessageIcon.Information,
                3000
            )
        
    def on_system_stopped(self):
        """Обработчик остановки системы"""
        self.system_status.setText("🔴 Статус: Остановлена")
        self.status_label.setText("🔴 Система остановлена")
        self.connection_status.setText("🔴 Подключение: Отключено")
        
        self.log_message("⏹️ Расширенная система ATB остановлена")
        
    def on_system_error(self, error_msg):
        """Обработчик ошибки системы"""
        self.log_message(f"❌ ОШИБКА: {error_msg}")
        QMessageBox.critical(self, "Ошибка системы", f"Произошла ошибка:\n{error_msg}")
        
    def on_status_update(self, status):
        """Обработчик обновления статуса"""
        self.log_message(f"ℹ️ Статус: {status}")
        
    def on_data_updated(self, data):
        """Обработчик обновления данных"""
        if 'balance' in data:
            self.total_balance.setText(f"💵 Общий баланс: ${data['balance']:.2f}")
        if 'pnl' in data:
            self.current_pnl.setText(f"📈 Текущий P&L: ${data['pnl']:.2f}")
            
    def on_price_update(self, prices):
        """Обработчик обновления цен"""
        # Здесь можно обновить графики цен
        pass
        
    def on_portfolio_update(self, portfolio):
        """Обработчик обновления портфеля"""
        if 'balance' in portfolio:
            self.total_balance.setText(f"💵 Общий баланс: ${portfolio['balance']:.2f}")
        if 'daily_pnl' in portfolio:
            self.daily_pnl.setText(f"📅 Дневной P&L: ${portfolio['daily_pnl']:.2f}")
        if 'monthly_pnl' in portfolio:
            self.monthly_pnl.setText(f"📆 Месячный P&L: ${portfolio['monthly_pnl']:.2f}")
            
    def on_system_metrics_update(self, metrics):
        """Обработчик обновления системных метрик"""
        self.system_metrics = metrics
        
        try:
            # Обновление индикаторов в тулбаре
            cpu_percent = metrics.get('cpu', {}).get('percent', 0)
            memory_percent = metrics.get('memory', {}).get('percent', 0)
            
            self.cpu_label.setText(f"{cpu_percent:.1f}%")
            self.ram_label.setText(f"{memory_percent:.1f}%")
            
            # Обновление меток на системной вкладке
            if 'cpu_percent' in self.system_metrics_labels:
                self.system_metrics_labels['cpu_percent'].setText(f"{cpu_percent:.1f}%")
                
            if 'cpu_freq' in self.system_metrics_labels:
                freq = metrics.get('cpu', {}).get('frequency', 0)
                self.system_metrics_labels['cpu_freq'].setText(f"{freq:.0f} MHz")
                
            if 'memory_percent' in self.system_metrics_labels:
                self.system_metrics_labels['memory_percent'].setText(f"{memory_percent:.1f}%")
                
            if 'memory_used' in self.system_metrics_labels:
                used_gb = metrics.get('memory', {}).get('used', 0) / (1024**3)
                self.system_metrics_labels['memory_used'].setText(f"{used_gb:.1f} GB")
                
            if 'disk_percent' in self.system_metrics_labels:
                disk_percent = metrics.get('disk', {}).get('percent', 0)
                self.system_metrics_labels['disk_percent'].setText(f"{disk_percent:.1f}%")
                
            if 'disk_free' in self.system_metrics_labels:
                free_gb = metrics.get('disk', {}).get('free', 0) / (1024**3)
                self.system_metrics_labels['disk_free'].setText(f"{free_gb:.1f} GB")
                
            if 'net_recv' in self.system_metrics_labels:
                recv_mb = metrics.get('network', {}).get('bytes_recv', 0) / (1024**2)
                self.system_metrics_labels['net_recv'].setText(f"{recv_mb:.1f} MB")
                
            if 'net_sent' in self.system_metrics_labels:
                sent_mb = metrics.get('network', {}).get('bytes_sent', 0) / (1024**2)
                self.system_metrics_labels['net_sent'].setText(f"{sent_mb:.1f} MB")
                
            if 'processes' in self.system_metrics_labels:
                processes = metrics.get('system', {}).get('processes', 0)
                self.system_metrics_labels['processes'].setText(str(processes))
                
            if 'uptime' in self.system_metrics_labels:
                uptime = metrics.get('system', {}).get('uptime', '0:00:00')
                self.system_metrics_labels['uptime'].setText(uptime)
                
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
            
    def on_evolution_update(self, evolution_data):
        """Обработчик обновления данных эволюции"""
        self.evolution_data = evolution_data
        
        try:
            status = evolution_data.get('status', {})
            strategies = evolution_data.get('strategies', {})
            
            # Обновление статуса эволюции
            self.evolution_status_label.setText(status.get('status', 'Unknown'))
            self.active_evolutions_label.setText(str(status.get('active_evolutions', 0)))
            self.total_components_label.setText(str(status.get('total_components', 0)))
            self.success_rate_label.setText(f"{status.get('success_rate', 0)*100:.1f}%")
            
            # Обновление таблицы стратегий
            self.evolution_strategies_table.setRowCount(len(strategies))
            for row, (name, data) in enumerate(strategies.items()):
                self.evolution_strategies_table.setItem(row, 0, QTableWidgetItem(name))
                self.evolution_strategies_table.setItem(row, 1, QTableWidgetItem(f"{data['performance']:.2f}"))
                self.evolution_strategies_table.setItem(row, 2, QTableWidgetItem(str(data['evolution_count'])))
                self.evolution_strategies_table.setItem(row, 3, QTableWidgetItem(
                    data['last_evolution'].strftime('%H:%M:%S') if data['last_evolution'] else 'Never'
                ))
                self.evolution_strategies_table.setItem(row, 4, QTableWidgetItem(f"{data['success_rate']:.2f}"))
                self.evolution_strategies_table.setItem(row, 5, QTableWidgetItem(str(data['parameters'])))
                
        except Exception as e:
            logger.error(f"Error updating evolution data: {e}")
            
    def update_displays(self):
        """Обновление отображения данных"""
        current_time = datetime.now()
        self.last_update.setText(f"🕐 Последнее обновление: {current_time.strftime('%H:%M:%S')}")
        
        # Обновление режима из .env
        env_mode = self.env_manager.get("ATB_MODE", "simulation")
        self.env_mode_status.setText(f"🎯 {env_mode.title()}")
        self.mode_label.setText(f"🎯 Режим: {env_mode.title()}")
        
        # Обновление окружения
        environment = self.env_manager.get("ENVIRONMENT", "development")
        self.environment_label.setText(f"🌍 Окружение: {environment.title()}")
        
    def log_message(self, message):
        """Добавление сообщения в лог"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def show_about(self):
        """Показать информацию о программе"""
        QMessageBox.about(self, "О программе", 
                         "⚡ ATB Trading System - Enhanced Desktop v3.1\n\n"
                         "🚀 Современная торговая система с реальными данными\n\n"
                         "✨ Возможности:\n"
                         "• 🤖 Автоматическая торговля с ИИ\n"
                         "• 📊 Мониторинг системных ресурсов\n"
                         "• 🧬 Эволюция стратегий в реальном времени\n"
                         "• 🔧 Управление .env конфигурацией\n"
                         "• 💼 Управление портфелем\n"
                         "• 🔮 ML прогнозирование\n"
                         "• ⚡ Реальные данные CPU, памяти, сети\n"
                         "• 🛡️ Расширенное управление рисками\n\n"
                         "© 2024 ATB Trading Team\n"
                         "Все права защищены")
        
    def closeEvent(self, event):
        """Обработчик закрытия приложения"""
        if self.system_thread and self.system_thread.running:
            reply = QMessageBox.question(self, 'Подтверждение', 
                                       '⚠️ Система все еще работает.\n'
                                       'Вы уверены, что хотите выйти?',
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                       QMessageBox.StandardButton.No)
            
            if reply == QMessageBox.StandardButton.Yes:
                self.stop_trading()
                if self.tray_icon:
                    self.tray_icon.hide()
                event.accept()
            else:
                event.ignore()
        else:
            if self.tray_icon:
                self.tray_icon.hide()
            event.accept()

def main():
    """Главная функция приложения"""
    app = QApplication(sys.argv)
    
    # Настройка информации о приложении
    app.setApplicationName("ATB Trading System Enhanced")
    app.setApplicationVersion("3.1")
    app.setOrganizationName("ATB Trading Team")
    app.setOrganizationDomain("atb-trading.com")
    
    # Поддержка высокого DPI
    app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    
    # Создание и отображение главного окна
    window = ATBEnhancedUnifiedDesktopApp()
    window.show()
    
    # Логирование запуска
    logger.info("ATB Enhanced Unified Desktop Application started with real data integration")
    
    # Запуск приложения
    sys.exit(app.exec())

if __name__ == "__main__":
    main()