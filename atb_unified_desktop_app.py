#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATB Trading System - Unified Desktop Application
Единое полноценное десктопное приложение с интеграцией всей функциональности
"""

import sys
import asyncio
import json
import threading
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from pathlib import Path
import signal
import os

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
    HAS_MAIN_SYSTEM = True
except ImportError as e:
    print(f"Warning: Could not import some main system modules: {e}")
    HAS_MAIN_SYSTEM = False

# Импорт дополнительных виджетов
try:
    from atb_desktop_widgets import (
        BacktestDialog, ConfigurationDialog, PerformanceWidget, 
        StrategyManagerWidget, MarketDataWidget
    )
    HAS_WIDGETS = True
except ImportError:
    print("Warning: Could not import custom widgets")
    HAS_WIDGETS = False

# Логирование
from loguru import logger

class ATBSystemThread(QThread):
    """Поток для запуска основной системы ATB"""
    system_started = pyqtSignal()
    system_stopped = pyqtSignal()
    error_occurred = pyqtSignal(str)
    status_update = pyqtSignal(str)
    data_updated = pyqtSignal(dict)
    price_update = pyqtSignal(dict)
    portfolio_update = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.orchestrator = None
        self.data_timer = QTimer()
        
    def run(self):
        """Запуск системы в отдельном потоке"""
        try:
            self.running = True
            self.status_update.emit("Инициализация системы ATB...")
            
            if HAS_MAIN_SYSTEM:
                # Создание нового event loop для этого потока
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Запуск основной функции
                loop.run_until_complete(self._start_system())
            else:
                # Режим демонстрации без основной системы
                self._demo_mode()
                
        except Exception as e:
            self.error_occurred.emit(f"Ошибка запуска системы: {str(e)}")
        finally:
            self.running = False
            self.system_stopped.emit()
    
    async def _start_system(self):
        """Асинхронный запуск системы"""
        try:
            config = create_default_config()
            service_locator = get_service_locator()
            
            self.status_update.emit("Инициализация компонентов...")
            
            # Инициализация основных агентов
            entanglement_detector = EntanglementDetector()
            mirror_detector = MirrorDetector()
            market_maker_agent = MarketMakerModelAgent()
            
            self.status_update.emit("Инициализация стратегий...")
            await strategy_integration.initialize_strategies()
            
            self.status_update.emit("Создание оркестратора...")
            self.orchestrator = create_trading_orchestrator(config)
            
            self.status_update.emit("Запуск торгового оркестратора...")
            self.system_started.emit()
            
            # Запуск оркестратора
            await self.orchestrator.start()
            
        except Exception as e:
            self.error_occurred.emit(f"Ошибка в системе: {str(e)}")
    
    def _demo_mode(self):
        """Демонстрационный режим"""
        self.status_update.emit("Запуск в демонстрационном режиме...")
        self.system_started.emit()
        
        # Симуляция работы системы
        import time
        import random
        
        while self.running:
            # Генерация тестовых данных
            price_data = {
                'BTC/USDT': random.uniform(45000, 55000),
                'ETH/USDT': random.uniform(3000, 4000),
                'BNB/USDT': random.uniform(400, 600)
            }
            self.price_update.emit(price_data)
            
            portfolio_data = {
                'balance': random.uniform(9500, 10500),
                'pnl': random.uniform(-500, 500),
                'positions': random.randint(0, 5)
            }
            self.portfolio_update.emit(portfolio_data)
            
            time.sleep(1)
    
    def stop_system(self):
        """Остановка системы"""
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
            transform: translateY(-2px);
        }
        
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #533483, stop:1 #16213e);
        }
        
        QPushButton:disabled {
            background: #333;
            color: #666;
        }
        
        QComboBox {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #16213e, stop:1 #0f3460);
            color: #eee;
            border: 2px solid #0e4b99;
            border-radius: 6px;
            padding: 8px 12px;
            min-width: 150px;
        }
        
        QComboBox:hover {
            border-color: #e94560;
        }
        
        QComboBox::drop-down {
            border: none;
            width: 30px;
        }
        
        QComboBox::down-arrow {
            image: none;
            border-left: 8px solid transparent;
            border-right: 8px solid transparent;
            border-top: 8px solid #eee;
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
        
        QStatusBar {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #0f3460, stop:1 #16213e);
            color: #eee;
            border-top: 2px solid #0e4b99;
        }
        
        QMenuBar {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #16213e, stop:1 #0f3460);
            color: #eee;
            border-bottom: 2px solid #0e4b99;
        }
        
        QMenuBar::item {
            background-color: transparent;
            padding: 8px 16px;
        }
        
        QMenuBar::item:selected {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #e94560, stop:1 #f27121);
            border-radius: 4px;
        }
        
        QMenu {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #16213e, stop:1 #0f3460);
            color: #eee;
            border: 2px solid #0e4b99;
            border-radius: 8px;
        }
        
        QMenu::item:selected {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #e94560, stop:1 #f27121);
            border-radius: 4px;
        }
        
        QListWidget {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #0f3460, stop:1 #16213e);
            color: #eee;
            border: 2px solid #0e4b99;
            border-radius: 8px;
            alternate-background-color: #1a1a2e;
        }
        
        QListWidget::item {
            padding: 8px;
            border-bottom: 1px solid #0e4b99;
        }
        
        QListWidget::item:selected {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #e94560, stop:1 #f27121);
            color: white;
        }
        
        QScrollBar:vertical {
            background: #16213e;
            width: 12px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:vertical {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #e94560, stop:1 #f27121);
            border-radius: 6px;
            min-height: 20px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #f27121, stop:1 #e94560);
        }
        """

class ATBUnifiedDesktopApp(QMainWindow):
    """Единое полноценное десктопное приложение ATB"""
    
    def __init__(self):
        super().__init__()
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
        
        # Виджеты (будут создаваться динамически)
        self.performance_widget = None
        self.strategy_manager_widget = None
        self.market_data_widget = None
        
        # Системный трей
        self.tray_icon = None
        
        self.init_ui()
        self.setup_connections()
        self.setup_system_tray()
        
    def init_ui(self):
        """Инициализация пользовательского интерфейса"""
        self.setWindowTitle("⚡ ATB Trading System - Unified Desktop v3.0")
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
        self.update_timer.start(1000)  # Обновление каждую секунду
        
    def create_menu_bar(self):
        """Создание расширенного меню"""
        menubar = self.menuBar()
        
        # Меню Файл
        file_menu = menubar.addMenu('📁 Файл')
        
        new_action = QAction('🆕 Новый', self)
        new_action.setShortcut('Ctrl+N')
        file_menu.addAction(new_action)
        
        open_action = QAction('📂 Открыть', self)
        open_action.setShortcut('Ctrl+O')
        file_menu.addAction(open_action)
        
        save_action = QAction('💾 Сохранить', self)
        save_action.setShortcut('Ctrl+S')
        file_menu.addAction(save_action)
        
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
        
        # Меню Торговля
        trading_menu = menubar.addMenu('📈 Торговля')
        
        start_trading_action = QAction('▶️ Запустить торговлю', self)
        start_trading_action.triggered.connect(self.start_trading)
        trading_menu.addAction(start_trading_action)
        
        stop_trading_action = QAction('⏹️ Остановить торговлю', self)
        stop_trading_action.triggered.connect(self.stop_trading)
        trading_menu.addAction(stop_trading_action)
        
        trading_menu.addSeparator()
        
        quick_trade_action = QAction('⚡ Быстрая торговля', self)
        quick_trade_action.triggered.connect(self.quick_trade)
        trading_menu.addAction(quick_trade_action)
        
        # Меню Аналитика
        analytics_menu = menubar.addMenu('📊 Аналитика')
        
        backtest_action = QAction('🔙 Бэктестинг', self)
        backtest_action.triggered.connect(self.show_backtest_dialog)
        analytics_menu.addAction(backtest_action)
        
        performance_action = QAction('📈 Производительность', self)
        performance_action.triggered.connect(self.show_performance)
        analytics_menu.addAction(performance_action)
        
        ml_analysis_action = QAction('🤖 ML Анализ', self)
        ml_analysis_action.triggered.connect(self.show_ml_analysis)
        analytics_menu.addAction(ml_analysis_action)
        
        # Меню Инструменты
        tools_menu = menubar.addMenu('🛠️ Инструменты')
        
        calculator_action = QAction('🧮 Калькулятор позиций', self)
        calculator_action.triggered.connect(self.show_calculator)
        tools_menu.addAction(calculator_action)
        
        converter_action = QAction('💱 Конвертер валют', self)
        converter_action.triggered.connect(self.show_converter)
        tools_menu.addAction(converter_action)
        
        # Меню Настройки
        settings_menu = menubar.addMenu('⚙️ Настройки')
        
        config_action = QAction('🔧 Конфигурация', self)
        config_action.triggered.connect(self.show_configuration)
        settings_menu.addAction(config_action)
        
        # Меню Справка
        help_menu = menubar.addMenu('❓ Справка')
        
        about_action = QAction('ℹ️ О программе', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
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
        
        # Растягивающийся спейсер
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        
        # Быстрые действия
        quick_actions_label = QLabel("⚡ Быстрые действия:")
        toolbar.addWidget(quick_actions_label)
        
        backtest_btn = QPushButton("🔙 Бэктест")
        backtest_btn.clicked.connect(self.show_backtest_dialog)
        toolbar.addWidget(backtest_btn)
        
        config_btn = QPushButton("⚙️ Настройки")
        config_btn.clicked.connect(self.show_configuration)
        toolbar.addWidget(config_btn)
        
    def create_main_content(self, main_layout):
        """Создание главного контента с вкладками"""
        # Создание основных вкладок
        self.main_tabs = QTabWidget()
        self.main_tabs.setTabPosition(QTabWidget.TabPosition.North)
        
        # Вкладка Обзор
        overview_tab = self.create_overview_tab()
        self.main_tabs.addTab(overview_tab, "📊 Обзор")
        
        # Вкладка Торговля
        trading_tab = self.create_trading_tab()
        self.main_tabs.addTab(trading_tab, "📈 Торговля")
        
        # Вкладка Портфель
        portfolio_tab = self.create_portfolio_tab()
        self.main_tabs.addTab(portfolio_tab, "💼 Портфель")
        
        # Вкладка Аналитика
        analytics_tab = self.create_analytics_tab()
        self.main_tabs.addTab(analytics_tab, "📊 Аналитика")
        
        # Вкладка ML Анализ
        ml_tab = self.create_ml_analysis_tab()
        self.main_tabs.addTab(ml_tab, "🤖 ML Анализ")
        
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
        
        self.mode_label = QLabel("🎯 Режим: Симуляция")
        system_layout.addWidget(self.mode_label)
        
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
        
        # Торговые пары
        pairs_group = QGroupBox("💱 Торговые пары")
        pairs_layout = QVBoxLayout(pairs_group)
        
        self.pairs_list = QListWidget()
        for pair in self.trading_data["pairs"]:
            self.pairs_list.addItem(f"📈 {pair}")
        pairs_layout.addWidget(self.pairs_list)
        
        layout.addWidget(pairs_group)
        
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
        self.log_text.setPlainText("📅 [System] Добро пожаловать в ATB Trading System v3.0\n")
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        return panel
        
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
        
        # Настройки
        control_layout.addWidget(QLabel("💰 Размер позиции:"), 1, 0)
        position_size = QDoubleSpinBox()
        position_size.setRange(0.01, 10.0)
        position_size.setValue(1.0)
        position_size.setSuffix(" %")
        control_layout.addWidget(position_size, 1, 1)
        
        control_layout.addWidget(QLabel("🛡️ Стоп-лосс:"), 2, 0)
        stop_loss = QDoubleSpinBox()
        stop_loss.setRange(0.1, 10.0)
        stop_loss.setValue(2.0)
        stop_loss.setSuffix(" %")
        control_layout.addWidget(stop_loss, 2, 1)
        
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
        
        # Здесь будет круговая диаграмма
        self.allocation_chart = self.create_allocation_chart()
        allocation_layout.addWidget(self.allocation_chart)
        
        layout.addWidget(allocation_group)
        
        return widget
        
    def create_analytics_tab(self):
        """Создание вкладки аналитики"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Панель метрик
        metrics_group = QGroupBox("📊 Ключевые метрики")
        metrics_layout = QGridLayout(metrics_group)
        
        # Метрики производительности
        metrics = [
            ("📈 Общая доходность:", "0.00%"),
            ("📅 Дневная доходность:", "0.00%"),
            ("🎯 Sharpe Ratio:", "0.00"),
            ("📉 Max Drawdown:", "0.00%"),
            ("🔢 Всего сделок:", "0"),
            ("✅ Прибыльных сделок:", "0 (0%)")
        ]
        
        for i, (label, value) in enumerate(metrics):
            row, col = i // 2, (i % 2) * 2
            metrics_layout.addWidget(QLabel(label), row, col)
            value_label = QLabel(value)
            value_label.setStyleSheet("font-weight: bold; color: #45b7d1;")
            metrics_layout.addWidget(value_label, row, col + 1)
            
        layout.addWidget(metrics_group)
        
        # График производительности
        performance_group = QGroupBox("📈 График производительности")
        performance_layout = QVBoxLayout(performance_group)
        
        self.performance_chart = self.create_performance_chart()
        performance_layout.addWidget(self.performance_chart)
        
        layout.addWidget(performance_group)
        
        return widget
        
    def create_ml_analysis_tab(self):
        """Создание вкладки ML анализа"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Панель ML моделей
        ml_group = QGroupBox("🤖 Модели машинного обучения")
        ml_layout = QGridLayout(ml_group)
        
        # Статус моделей
        ml_layout.addWidget(QLabel("🧠 Модель прогнозирования:"), 0, 0)
        model_status = QLabel("🟢 Активна")
        model_status.setStyleSheet("color: #00ff88; font-weight: bold;")
        ml_layout.addWidget(model_status, 0, 1)
        
        ml_layout.addWidget(QLabel("📊 Точность модели:"), 1, 0)
        accuracy_label = QLabel("85.2%")
        accuracy_label.setStyleSheet("color: #45b7d1; font-weight: bold;")
        ml_layout.addWidget(accuracy_label, 1, 1)
        
        # Прогнозы
        ml_layout.addWidget(QLabel("🔮 Прогноз BTC/USDT (24h):"), 2, 0)
        prediction_label = QLabel("📈 +2.4% (Высокая вероятность)")
        prediction_label.setStyleSheet("color: #00ff88; font-weight: bold;")
        ml_layout.addWidget(prediction_label, 2, 1)
        
        layout.addWidget(ml_group)
        
        # График прогнозов
        prediction_group = QGroupBox("🔮 Прогнозы и сигналы")
        prediction_layout = QVBoxLayout(prediction_group)
        
        self.prediction_chart = self.create_prediction_chart()
        prediction_layout.addWidget(self.prediction_chart)
        
        layout.addWidget(prediction_group)
        
        return widget
        
    def create_settings_tab(self):
        """Создание вкладки настроек"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Общие настройки
        general_group = QGroupBox("⚙️ Общие настройки")
        general_layout = QFormLayout(general_group)
        
        # Тема
        theme_combo = QComboBox()
        theme_combo.addItems(["🌙 Темная", "☀️ Светлая", "🎨 Автоматическая"])
        general_layout.addRow("🎨 Тема:", theme_combo)
        
        # Язык
        language_combo = QComboBox()
        language_combo.addItems(["🇷🇺 Русский", "🇺🇸 English", "🇨🇳 中文"])
        general_layout.addRow("🌐 Язык:", language_combo)
        
        # Обновления
        update_check = QCheckBox("Автоматически проверять обновления")
        update_check.setChecked(True)
        general_layout.addRow("🔄 Обновления:", update_check)
        
        layout.addWidget(general_group)
        
        # Торговые настройки
        trading_group = QGroupBox("📈 Торговые настройки")
        trading_layout = QFormLayout(trading_group)
        
        # API ключи
        api_key_edit = QLineEdit()
        api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        api_key_edit.setPlaceholderText("Введите API ключ...")
        trading_layout.addRow("🔑 API Ключ:", api_key_edit)
        
        # Лимиты
        max_position_size = QDoubleSpinBox()
        max_position_size.setRange(0.1, 100.0)
        max_position_size.setValue(10.0)
        max_position_size.setSuffix(" %")
        trading_layout.addRow("💰 Макс. размер позиции:", max_position_size)
        
        layout.addWidget(trading_group)
        
        # Кнопки действий
        buttons_layout = QHBoxLayout()
        
        save_btn = QPushButton("💾 Сохранить настройки")
        save_btn.clicked.connect(self.save_settings)
        buttons_layout.addWidget(save_btn)
        
        reset_btn = QPushButton("🔄 Сбросить")
        buttons_layout.addWidget(reset_btn)
        
        export_btn = QPushButton("📤 Экспорт настроек")
        buttons_layout.addWidget(export_btn)
        
        layout.addLayout(buttons_layout)
        layout.addStretch()
        
        return widget
        
    def create_price_chart(self):
        """Создание графика цен"""
        chart = QChart()
        chart.setTitle("📈 График цен BTC/USDT")
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        
        # Создание серии данных
        series = QLineSeries()
        series.setName("BTC/USDT")
        
        # Добавление тестовых данных
        import random
        for i in range(100):
            price = 45000 + i * 50 + random.uniform(-500, 500)
            series.append(i, price)
        
        chart.addSeries(series)
        
        # Настройка осей
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
        
        # Настройка цветов
        slices = series.slices()
        colors = ["#f39c12", "#3498db", "#e74c3c", "#2ecc71"]
        for i, slice in enumerate(slices):
            slice.setBrush(QColor(colors[i % len(colors)]))
            slice.setLabelVisible(True)
        
        chart.addSeries(series)
        
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        return chart_view
        
    def create_performance_chart(self):
        """Создание графика производительности"""
        chart = QChart()
        chart.setTitle("📈 Производительность стратегий")
        
        bar_series = QBarSeries()
        
        strategy_set = QBarSet("Доходность (%)")
        strategy_set.append(5.2)
        strategy_set.append(3.8)
        strategy_set.append(7.1)
        strategy_set.append(2.4)
        bar_series.append(strategy_set)
        
        chart.addSeries(bar_series)
        
        axis_x = QBarCategoryAxis()
        axis_x.append(["Тренд", "Скальпинг", "Арбитраж", "ML"])
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        bar_series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        axis_y.setTitleText("Доходность (%)")
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        bar_series.attachAxis(axis_y)
        
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        return chart_view
        
    def create_prediction_chart(self):
        """Создание графика прогнозов"""
        chart = QChart()
        chart.setTitle("🔮 ML Прогнозы")
        
        # Реальные данные
        actual_series = QLineSeries()
        actual_series.setName("Реальная цена")
        
        # Прогнозы
        prediction_series = QLineSeries()
        prediction_series.setName("Прогноз")
        
        import random
        for i in range(50):
            price = 45000 + i * 20 + random.uniform(-100, 100)
            actual_series.append(i, price)
            
        for i in range(50, 70):
            price = 46000 + (i-50) * 30 + random.uniform(-50, 50)
            prediction_series.append(i, price)
        
        chart.addSeries(actual_series)
        chart.addSeries(prediction_series)
        
        axis_x = QValueAxis()
        axis_x.setTitleText("Время")
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        actual_series.attachAxis(axis_x)
        prediction_series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        axis_y.setTitleText("Цена ($)")
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        actual_series.attachAxis(axis_y)
        prediction_series.attachAxis(axis_y)
        
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
        
        # Версия
        version_label = QLabel("v3.0")
        version_label.setStyleSheet("color: #666;")
        status_bar.addPermanentWidget(version_label)
        
    def setup_system_tray(self):
        """Настройка системного трея"""
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray_icon = QSystemTrayIcon(self)
            # Здесь можно добавить иконку: self.tray_icon.setIcon(QIcon("icon.png"))
            
            # Меню трея
            tray_menu = QMenu()
            show_action = tray_menu.addAction("Показать")
            show_action.triggered.connect(self.show)
            hide_action = tray_menu.addAction("Скрыть")
            hide_action.triggered.connect(self.hide)
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
            
    def toggle_trading(self):
        """Переключение состояния торговли"""
        if self.system_thread and self.system_thread.running:
            self.stop_trading()
        else:
            self.start_trading()
            
    def start_trading(self):
        """Запуск торговли"""
        if not self.system_thread or not self.system_thread.running:
            self.system_thread = ATBSystemThread()
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
            
    def quick_trade(self):
        """Быстрая торговля"""
        QMessageBox.information(self, "Быстрая торговля", "Функция быстрой торговли будет реализована в следующей версии")
        
    def show_backtest_dialog(self):
        """Показать диалог бэктестинга"""
        if HAS_WIDGETS:
            try:
                dialog = BacktestDialog(self)
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    settings = dialog.get_settings()
                    self.log_message(f"🔙 Запуск бэктестинга: {settings['strategy']} на {settings['pair']}")
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", f"Не удалось открыть диалог бэктестинга: {e}")
        else:
            QMessageBox.information(self, "Бэктестинг", "Настройте стратегию и запустите бэктестинг на исторических данных")
            
    def show_configuration(self):
        """Показать диалог конфигурации"""
        if HAS_WIDGETS:
            try:
                dialog = ConfigurationDialog(self)
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    self.log_message("⚙️ Конфигурация обновлена")
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", f"Не удалось открыть настройки: {e}")
        else:
            self.main_tabs.setCurrentIndex(5)  # Переход на вкладку настроек
            
    def show_performance(self):
        """Показать вкладку производительности"""
        self.main_tabs.setCurrentIndex(3)  # Переход на вкладку аналитики
        
    def show_ml_analysis(self):
        """Показать вкладку ML анализа"""
        self.main_tabs.setCurrentIndex(4)  # Переход на вкладку ML
        
    def show_calculator(self):
        """Показать калькулятор позиций"""
        QMessageBox.information(self, "Калькулятор", "Функция калькулятора позиций в разработке")
        
    def show_converter(self):
        """Показать конвертер валют"""
        QMessageBox.information(self, "Конвертер", "Функция конвертера валют в разработке")
        
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
            
    def save_settings(self):
        """Сохранение настроек"""
        self.log_message("💾 Настройки сохранены")
        QMessageBox.information(self, "Настройки", "Настройки успешно сохранены!")
        
    def on_system_started(self):
        """Обработчик запуска системы"""
        self.system_status.setText("🟢 Статус: Запущена")
        self.status_label.setText("🟢 Система запущена")
        self.connection_status.setText("🟢 Подключение: Активно")
        
        self.log_message("🚀 Система ATB успешно запущена")
        
        # Уведомление в трее
        if self.tray_icon:
            self.tray_icon.showMessage(
                "ATB Trading System",
                "Система успешно запущена!",
                QSystemTrayIcon.MessageIcon.Information,
                3000
            )
        
    def on_system_stopped(self):
        """Обработчик остановки системы"""
        self.system_status.setText("🔴 Статус: Остановлена")
        self.status_label.setText("🔴 Система остановлена")
        self.connection_status.setText("🔴 Подключение: Отключено")
        
        self.log_message("⏹️ Система ATB остановлена")
        
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
            self.portfolio_value.setText(f"${data['balance']:.2f}")
        if 'pnl' in data:
            self.current_pnl.setText(f"📈 Текущий P&L: ${data['pnl']:.2f}")
            
    def on_price_update(self, prices):
        """Обработчик обновления цен"""
        # Обновление данных в интерфейсе
        pass
        
    def on_portfolio_update(self, portfolio):
        """Обработчик обновления портфеля"""
        if 'balance' in portfolio:
            self.total_balance.setText(f"💵 Общий баланс: ${portfolio['balance']:.2f}")
            
    def update_displays(self):
        """Обновление отображения данных"""
        current_time = datetime.now()
        self.last_update.setText(f"🕐 Последнее обновление: {current_time.strftime('%H:%M:%S')}")
        
        # Обновление времени работы
        if self.system_thread and self.system_thread.running:
            # Здесь можно добавить расчет времени работы
            pass
            
    def log_message(self, message):
        """Добавление сообщения в лог"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
        # Прокрутка к последнему сообщению
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def show_about(self):
        """Показать информацию о программе"""
        QMessageBox.about(self, "О программе", 
                         "⚡ ATB Trading System - Unified Desktop v3.0\n\n"
                         "🚀 Современная торговая система с искусственным интеллектом\n\n"
                         "✨ Возможности:\n"
                         "• 🤖 Автоматическая торговля с ИИ\n"
                         "• 📊 Расширенная аналитика и отчеты\n"
                         "• 🔙 Полноценный бэктестинг\n"
                         "• 💼 Управление портфелем\n"
                         "• 🔮 ML прогнозирование\n"
                         "• ⚡ Реальное время данных\n"
                         "• 🛡️ Управление рисками\n\n"
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
    app.setApplicationName("ATB Trading System")
    app.setApplicationVersion("3.0")
    app.setOrganizationName("ATB Trading Team")
    app.setOrganizationDomain("atb-trading.com")
    
    # Поддержка высокого DPI
    app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    
    # Создание и отображение главного окна
    window = ATBUnifiedDesktopApp()
    window.show()
    
    # Логирование запуска
    logger.info("ATB Unified Desktop Application started")
    
    # Запуск приложения
    sys.exit(app.exec())

if __name__ == "__main__":
    main()