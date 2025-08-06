#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATB Trading System - Enhanced Desktop Application
Улучшенная версия десктопного приложения с полной функциональностью
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
    QTreeWidget, QTreeWidgetItem, QHeaderView, QAbstractItemView
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

# Импорт дополнительных виджетов
from atb_desktop_widgets import (
    BacktestDialog, ConfigurationDialog, PerformanceWidget, 
    StrategyManagerWidget, MarketDataWidget
)

# Импорт ультрасовременной системы стилей и Twitch страницы
from modern_style_system import UltraModernStyleSystem, UltraModernCard, UltraModernButton, UltraModernLabel, UltraModernBadge
from twitch_demo_page import TwitchDemoPage

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
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")

# Логирование
from loguru import logger

class ATBSystemThread(QThread):
    """Поток для запуска основной системы ATB"""
    system_started = pyqtSignal()
    system_stopped = pyqtSignal()
    error_occurred = pyqtSignal(str)
    status_update = pyqtSignal(str)
    data_updated = pyqtSignal(dict)
    
    def __init__(self) -> None:
        super().__init__()
        self.running = False
        self.orchestrator = None
        
    def run(self) -> None:
        """Запуск системы в отдельном потоке"""
        try:
            self.running = True
            self.status_update.emit("Инициализация системы ATB...")
            
            # Создание нового event loop для этого потока
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Запуск основной функции
            loop.run_until_complete(self._start_system())
            
        except Exception as e:
            self.error_occurred.emit(f"Ошибка запуска системы: {str(e)}")
        finally:
            self.running = False
            self.system_stopped.emit()
    
    async def _start_system(self) -> None:
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
    
    def stop_system(self) -> None:
        """Остановка системы"""
        if self.orchestrator:
            asyncio.create_task(self.orchestrator.stop())
        self.running = False

class ModernStyleSheet:
    """Современные стили для приложения"""
    
    @staticmethod
    def get_dark_theme() -> Any:
        return """
        
        QWidget {
            background-color: #1e1e1e;
            color: #ffffff;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 10pt;
        }
        
        QTabWidget::pane {
            border: 1px solid #3c3c3c;
            background-color: #2d2d2d;
        }
        
        QTabBar::tab {
            background-color: #3c3c3c;
            color: #ffffff;
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        
        QTabBar::tab:selected {
            background-color: #0078d4;
            color: #ffffff;
        }
        
        QTabBar::tab:hover {
            background-color: #4c4c4c;
        }
        
        QPushButton {
            background-color: #0078d4;
            color: #ffffff;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: #106ebe;
        }
        
        QPushButton:pressed {
            background-color: #005a9e;
        }
        
        QPushButton:disabled {
            background-color: #3c3c3c;
            color: #666666;
        }
        
        QComboBox {
            background-color: #3c3c3c;
            color: #ffffff;
            border: 1px solid #5c5c5c;
            border-radius: 4px;
            padding: 4px 8px;
        }
        
        QComboBox::drop-down {
            border: none;
        }
        
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #ffffff;
        }
        
        QTableWidget {
            background-color: #2d2d2d;
            color: #ffffff;
            gridline-color: #3c3c3c;
            border: 1px solid #3c3c3c;
        }
        
        QTableWidget::item {
            padding: 4px;
        }
        
        QTableWidget::item:selected {
            background-color: #0078d4;
        }
        
        QHeaderView::section {
            background-color: #3c3c3c;
            color: #ffffff;
            padding: 4px;
            border: 1px solid #5c5c5c;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 8px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px 0 4px;
        }
        
        QTextEdit {
            background-color: #2d2d2d;
            color: #ffffff;
            border: 1px solid #3c3c3c;
            border-radius: 4px;
        }
        
        QProgressBar {
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            text-align: center;
        }
        
        QProgressBar::chunk {
            background-color: #0078d4;
            border-radius: 3px;
        }
        
        QStatusBar {
            background-color: #2d2d2d;
            color: #ffffff;
        }
        
        QMenuBar {
            background-color: #2d2d2d;
            color: #ffffff;
        }
        
        QMenuBar::item {
            background-color: transparent;
            padding: 4px 8px;
        }
        
        QMenuBar::item:selected {
            background-color: #0078d4;
        }
        
        QMenu {
            background-color: #2d2d2d;
            color: #ffffff;
            border: 1px solid #3c3c3c;
        }
        
        QMenu::item:selected {
            background-color: #0078d4;
        }
        
        QTreeWidget {
            background-color: #2d2d2d;
            color: #ffffff;
            border: 1px solid #3c3c3c;
        }
        
        QTreeWidget::item {
            padding: 4px;
        }
        
        QTreeWidget::item:selected {
            background-color: #0078d4;
        }
        """

class ATBEnhancedDesktopApp(QMainWindow):
    """Улучшенное главное окно приложения ATB"""
    
    def __init__(self) -> None:
        super().__init__()
        self.system_thread = None
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_displays)
        
        # Виджеты
        self.performance_widget = None
        self.strategy_manager_widget = None
        self.market_data_widget = None
        
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self) -> None:
        """Инициализация пользовательского интерфейса"""
        self.setWindowTitle("ATB Trading System - Professional Edition v2.0")
        self.setGeometry(100, 100, 1800, 1200)
        self.setMinimumSize(1400, 900)
        
        # Применение ультрасовременного стиля
        self.setStyleSheet(UltraModernStyleSystem.get_ultra_modern_stylesheet())
        
        # Создание центрального виджета
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Главный layout
        main_layout = QVBoxLayout(central_widget)
        
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
        
    def create_menu_bar(self) -> None:
        """Создание меню"""
        menubar = self.menuBar()
        
        # Меню Файл
        file_menu = menubar.addMenu('&Файл')
        
        new_action = QAction('&Новый', self)
        new_action.setShortcut('Ctrl+N')
        file_menu.addAction(new_action)
        
        open_action = QAction('&Открыть', self)
        open_action.setShortcut('Ctrl+O')
        file_menu.addAction(open_action)
        
        save_action = QAction('&Сохранить', self)
        save_action.setShortcut('Ctrl+S')
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        export_action = QAction('&Экспорт данных', self)
        export_action.triggered.connect(self.export_data)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('&Выход', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Меню Торговля
        trading_menu = menubar.addMenu('&Торговля')
        
        start_trading_action = QAction('&Запустить торговлю', self)
        start_trading_action.triggered.connect(self.start_trading)
        trading_menu.addAction(start_trading_action)
        
        stop_trading_action = QAction('&Остановить торговлю', self)
        stop_trading_action.triggered.connect(self.stop_trading)
        trading_menu.addAction(stop_trading_action)
        
        trading_menu.addSeparator()
        
        backtest_action = QAction('&Бэктестинг', self)
        backtest_action.triggered.connect(self.show_backtest_dialog)
        trading_menu.addAction(backtest_action)
        
        # Меню Аналитика
        analytics_menu = menubar.addMenu('&Аналитика')
        
        performance_action = QAction('&Производительность', self)
        performance_action.triggered.connect(self.show_performance)
        analytics_menu.addAction(performance_action)
        
        market_data_action = QAction('&Рыночные данные', self)
        market_data_action.triggered.connect(self.show_market_data)
        analytics_menu.addAction(market_data_action)
        
        # Меню Стратегии
        strategies_menu = menubar.addMenu('&Стратегии')
        
        manage_strategies_action = QAction('&Управление стратегиями', self)
        manage_strategies_action.triggered.connect(self.show_strategy_manager)
        strategies_menu.addAction(manage_strategies_action)
        
        # Меню Настройки
        settings_menu = menubar.addMenu('&Настройки')
        
        config_action = QAction('&Конфигурация', self)
        config_action.triggered.connect(self.show_configuration)
        settings_menu.addAction(config_action)
        
        # Меню Справка
        help_menu = menubar.addMenu('&Справка')
        
        about_action = QAction('&О программе', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def create_toolbar(self) -> None:
        """Создание панели инструментов"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Кнопка запуска/остановки торговли
        self.trading_button = QPushButton("Запустить торговлю")
        self.trading_button.clicked.connect(self.toggle_trading)
        toolbar.addWidget(self.trading_button)
        
        toolbar.addSeparator()
        
        # Выбор режима торговли
        toolbar.addWidget(QLabel("Режим:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Симуляция", "Бумажная торговля", "Реальная торговля"])
        toolbar.addWidget(self.mode_combo)
        
        toolbar.addSeparator()
        
        # Статус системы
        self.status_label = QLabel("Система не запущена")
        self.status_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        toolbar.addWidget(self.status_label)
        
        toolbar.addSeparator()
        
        # Быстрые действия
        quick_actions_label = QLabel("Быстрые действия:")
        toolbar.addWidget(quick_actions_label)
        
        backtest_btn = QPushButton("Бэктест")
        backtest_btn.clicked.connect(self.show_backtest_dialog)
        toolbar.addWidget(backtest_btn)
        
        config_btn = QPushButton("Настройки")
        config_btn.clicked.connect(self.show_configuration)
        toolbar.addWidget(config_btn)
        
        # Кнопка современных настроек
        modern_settings_btn = QPushButton("⚙️ Modern Settings")
        modern_settings_btn.clicked.connect(self.show_modern_settings)
        toolbar.addWidget(modern_settings_btn)
        
    def create_main_content(self, main_layout) -> None:
        """Создание главного контента"""
        # Создание сплиттера для разделения панелей
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Левая панель с информацией
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Центральная панель с графиками и вкладками
        center_panel = self.create_center_panel()
        splitter.addWidget(center_panel)
        
        # Правая панель с торговыми данными
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Установка пропорций
        splitter.setSizes([450, 900, 450])
        
    def create_left_panel(self) -> None:
        """Создание левой панели"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Информация о системе
        system_group = QGroupBox("Система")
        system_layout = QVBoxLayout(system_group)
        
        self.system_status = QLabel("Статус: Остановлена")
        self.system_status.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        system_layout.addWidget(self.system_status)
        
        self.uptime_label = QLabel("Время работы: 00:00:00")
        system_layout.addWidget(self.uptime_label)
        
        self.mode_label = QLabel("Режим: Симуляция")
        system_layout.addWidget(self.mode_label)
        
        layout.addWidget(system_group)
        
        # Баланс и P&L
        balance_group = QGroupBox("Баланс и P&L")
        balance_layout = QVBoxLayout(balance_group)
        
        self.total_balance = QLabel("Общий баланс: $0.00")
        self.total_balance.setStyleSheet("font-size: 14pt; font-weight: bold; color: #4ecdc4;")
        balance_layout.addWidget(self.total_balance)
        
        self.current_pnl = QLabel("Текущий P&L: $0.00")
        self.current_pnl.setStyleSheet("font-size: 12pt; color: #45b7d1;")
        balance_layout.addWidget(self.current_pnl)
        
        self.daily_pnl = QLabel("Дневной P&L: $0.00")
        self.daily_pnl.setStyleSheet("font-size: 12pt; color: #96ceb4;")
        balance_layout.addWidget(self.daily_pnl)
        
        layout.addWidget(balance_group)
        
        # Активные стратегии
        strategies_group = QGroupBox("Активные стратегии")
        strategies_layout = QVBoxLayout(strategies_group)
        
        self.strategies_list = QListWidget()
        strategies_layout.addWidget(self.strategies_list)
        
        layout.addWidget(strategies_group)
        
        # Торговые пары
        pairs_group = QGroupBox("Торговые пары")
        pairs_layout = QVBoxLayout(pairs_group)
        
        self.pairs_list = QListWidget()
        pairs_layout.addWidget(self.pairs_list)
        
        layout.addWidget(pairs_group)
        
        layout.addStretch()
        return panel
        
    def create_center_panel(self) -> None:
        """Создание центральной панели с графиками и вкладками"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Вкладки для разных типов графиков и функций
        self.main_tabs = QTabWidget()
        
        # Вкладка с графиком цен
        price_tab = self.create_price_chart()
        self.main_tabs.addTab(price_tab, "График цен")
        
        # Вкладка с P&L
        pnl_tab = self.create_pnl_chart()
        self.main_tabs.addTab(pnl_tab, "P&L")
        
        # Вкладка с аналитикой
        analytics_tab = self.create_analytics_chart()
        self.main_tabs.addTab(analytics_tab, "Аналитика")
        
        # Вкладка производительности
        self.performance_widget = PerformanceWidget()
        self.main_tabs.addTab(self.performance_widget, "Производительность")
        
        # Вкладка управления стратегиями
        self.strategy_manager_widget = StrategyManagerWidget()
        self.main_tabs.addTab(self.strategy_manager_widget, "Стратегии")
        
        # Вкладка рыночных данных
        self.market_data_widget = MarketDataWidget()
        self.main_tabs.addTab(self.market_data_widget, "Рыночные данные")
        
        # Twitch Demo страница
        self.twitch_demo_page = TwitchDemoPage()
        self.main_tabs.addTab(self.twitch_demo_page, "🎥 Twitch Demo")
        
        layout.addWidget(self.main_tabs)
        return panel
        
    def create_price_chart(self) -> None:
        """Создание графика цен"""
        chart = QChart()
        chart.setTitle("График цен BTC/USDT")
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        
        # Создание серии данных
        series = QLineSeries()
        series.setName("Цена BTC/USDT")
        
        # Добавление тестовых данных
        for i in range(100):
            series.append(i, 45000 + i * 10 + (i % 20) * 5)
        
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
        
    def create_pnl_chart(self) -> None:
        """Создание графика P&L"""
        chart = QChart()
        chart.setTitle("График P&L")
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        
        # Создание серии данных
        series = QLineSeries()
        series.setName("P&L")
        
        # Добавление тестовых данных
        for i in range(100):
            pnl = (i - 50) * 10 + (i % 10) * 5
            series.append(i, pnl)
        
        chart.addSeries(series)
        
        # Настройка осей
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
        
    def create_analytics_chart(self) -> None:
        """Создание графика аналитики"""
        chart = QChart()
        chart.setTitle("Аналитика торговли")
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        
        # Создание столбчатой диаграммы
        bar_series = QBarSeries()
        
        # Данные по стратегиям
        strategy_set = QBarSet("Стратегии")
        strategy_set.append(10)
        strategy_set.append(15)
        strategy_set.append(8)
        strategy_set.append(12)
        bar_series.append(strategy_set)
        
        chart.addSeries(bar_series)
        
        # Настройка осей
        axis_x = QBarCategoryAxis()
        axis_x.append(["Тренд", "Боковик", "Адаптив", "Волатильность"])
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        bar_series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        axis_y.setTitleText("Прибыльность")
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        bar_series.attachAxis(axis_y)
        
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        return chart_view
        
    def create_right_panel(self) -> None:
        """Создание правой панели"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Активные позиции
        positions_group = QGroupBox("Активные позиции")
        positions_layout = QVBoxLayout(positions_group)
        
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(5)
        self.positions_table.setHorizontalHeaderLabels(["Пара", "Размер", "Цена входа", "Текущая цена", "P&L"])
        self.positions_table.horizontalHeader().setStretchLastSection(True)
        positions_layout.addWidget(self.positions_table)
        
        layout.addWidget(positions_group)
        
        # Открытые ордера
        orders_group = QGroupBox("Открытые ордера")
        orders_layout = QVBoxLayout(orders_group)
        
        self.orders_table = QTableWidget()
        self.orders_table.setColumnCount(6)
        self.orders_table.setHorizontalHeaderLabels(["ID", "Пара", "Тип", "Размер", "Цена", "Статус"])
        self.orders_table.horizontalHeader().setStretchLastSection(True)
        orders_layout.addWidget(self.orders_table)
        
        layout.addWidget(orders_group)
        
        # Лог событий
        log_group = QGroupBox("Лог событий")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        layout.addStretch()
        return panel
        
    def create_status_bar(self) -> None:
        """Создание статусной панели"""
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        
        # Информация о подключении
        self.connection_status = QLabel("Подключение: Отключено")
        self.connection_status.setStyleSheet("color: #ff6b6b;")
        status_bar.addWidget(self.connection_status)
        
        status_bar.addPermanentWidget(QLabel("|"))
        
        # Информация о последнем обновлении
        self.last_update = QLabel("Последнее обновление: Никогда")
        status_bar.addPermanentWidget(self.last_update)
        
    def setup_connections(self) -> None:
        """Настройка соединений сигналов"""
        # Соединения с системным потоком
        if self.system_thread:
            self.system_thread.system_started.connect(self.on_system_started)
            self.system_thread.system_stopped.connect(self.on_system_stopped)
            self.system_thread.error_occurred.connect(self.on_system_error)
            self.system_thread.status_update.connect(self.on_status_update)
            self.system_thread.data_updated.connect(self.on_data_updated)
            
    def toggle_trading(self) -> None:
        """Переключение состояния торговли"""
        if self.system_thread and self.system_thread.running:
            self.stop_trading()
        else:
            self.start_trading()
            
    def start_trading(self) -> None:
        """Запуск торговли"""
        if not self.system_thread or not self.system_thread.running:
            self.system_thread = ATBSystemThread()
            self.setup_connections()
            self.system_thread.start()
            
            self.trading_button.setText("Остановить торговлю")
            self.trading_button.setStyleSheet("background-color: #dc3545;")
            
    def stop_trading(self) -> None:
        """Остановка торговли"""
        if self.system_thread and self.system_thread.running:
            self.system_thread.stop_system()
            
            self.trading_button.setText("Запустить торговлю")
            self.trading_button.setStyleSheet("")
            
    def show_backtest_dialog(self) -> None:
        """Показать диалог бэктестинга"""
        dialog = BacktestDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            settings = dialog.get_settings()
            self.log_message(f"Запуск бэктестинга: {settings['strategy']} на {settings['pair']}")
            # Здесь будет запуск бэктестинга
            
    def show_configuration(self) -> None:
        """Показать диалог конфигурации"""
        dialog = ConfigurationDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.log_message("Конфигурация обновлена")
            # Здесь будет сохранение конфигурации
    
    def show_modern_settings(self) -> None:
        """Показать современный диалог настроек"""
        try:
            from modern_settings_dialog import ModernSettingsDialog
            dialog = ModernSettingsDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                settings = dialog.get_settings()
                self.log_message(f"Современные настройки сохранены: {len(settings['selected_pairs'])} пар выбрано")
                # Здесь можно сохранить настройки в конфигурацию
        except ImportError as e:
            QMessageBox.warning(self, "Error", f"Could not load modern settings: {e}")
            
    def show_performance(self) -> None:
        """Показать вкладку производительности"""
        self.main_tabs.setCurrentIndex(3)  # Индекс вкладки производительности
        
    def show_market_data(self) -> None:
        """Показать вкладку рыночных данных"""
        self.main_tabs.setCurrentIndex(5)  # Индекс вкладки рыночных данных
        
    def show_strategy_manager(self) -> None:
        """Показать вкладку управления стратегиями"""
        self.main_tabs.setCurrentIndex(4)  # Индекс вкладки стратегий
        
    def export_data(self) -> None:
        """Экспорт данных"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт данных", "", "JSON Files (*.json);;CSV Files (*.csv)"
        )
        if file_path:
            self.log_message(f"Экспорт данных в {file_path}")
            # Здесь будет экспорт данных
            
    def on_system_started(self) -> None:
        """Обработчик запуска системы"""
        self.system_status.setText("Статус: Запущена")
        self.system_status.setStyleSheet("color: #28a745; font-weight: bold;")
        self.status_label.setText("Система запущена")
        self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        self.connection_status.setText("Подключение: Активно")
        self.connection_status.setStyleSheet("color: #28a745;")
        
        self.log_message("Система ATB успешно запущена")
        
    def on_system_stopped(self) -> None:
        """Обработчик остановки системы"""
        self.system_status.setText("Статус: Остановлена")
        self.system_status.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        self.status_label.setText("Система остановлена")
        self.status_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        self.connection_status.setText("Подключение: Отключено")
        self.connection_status.setStyleSheet("color: #ff6b6b;")
        
        self.log_message("Система ATB остановлена")
        
    def on_system_error(self, error_msg) -> None:
        """Обработчик ошибки системы"""
        self.log_message(f"ОШИБКА: {error_msg}")
        QMessageBox.critical(self, "Ошибка системы", f"Произошла ошибка:\n{error_msg}")
        
    def on_status_update(self, status) -> None:
        """Обработчик обновления статуса"""
        self.log_message(f"Статус: {status}")
        
    def on_data_updated(self, data) -> None:
        """Обработчик обновления данных"""
        # Обновление отображения данных
        if 'balance' in data:
            self.total_balance.setText(f"Общий баланс: ${data['balance']:.2f}")
        if 'pnl' in data:
            self.current_pnl.setText(f"Текущий P&L: ${data['pnl']:.2f}")
            
    def update_displays(self) -> None:
        """Обновление отображения данных"""
        current_time = datetime.now()
        self.last_update.setText(f"Последнее обновление: {current_time.strftime('%H:%M:%S')}")
        
        # Обновление времени работы
        if self.system_thread and self.system_thread.running:
            # Здесь можно добавить расчет времени работы
            pass
            
    def log_message(self, message) -> None:
        """Добавление сообщения в лог"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
        # Прокрутка к последнему сообщению
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def show_about(self) -> None:
        """Показать информацию о программе"""
        QMessageBox.about(self, "О программе", 
                         "ATB Trading System - Professional Edition v2.0\n"
                         "Современная торговая система с искусственным интеллектом\n\n"
                         "Возможности:\n"
                         "• Автоматическая торговля\n"
                         "• Бэктестинг стратегий\n"
                         "• Аналитика производительности\n"
                         "• Управление рисками\n"
                         "• Мониторинг рынка\n\n"
                         "© 2024 ATB Trading Team")
        
    def closeEvent(self, event) -> None:
        """Обработчик закрытия приложения"""
        if self.system_thread and self.system_thread.running:
            reply = QMessageBox.question(self, 'Подтверждение', 
                                       'Система все еще работает. Вы уверены, что хотите выйти?',
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                       QMessageBox.StandardButton.No)
            
            if reply == QMessageBox.StandardButton.Yes:
                self.stop_trading()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

def main() -> None:
    """Главная функция приложения"""
    app = QApplication(sys.argv)
    
    # Настройка информации о приложении
    app.setApplicationName("ATB Trading System")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("ATB Trading Team")
    
    # Создание и отображение главного окна
    window = ATBEnhancedDesktopApp()
    window.show()
    
    # Запуск приложения
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 