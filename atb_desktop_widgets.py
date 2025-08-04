#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Дополнительные виджеты для ATB Desktop Application
Расширенная функциональность и специализированные компоненты
"""

import sys
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
import json

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem, QTextEdit,
    QProgressBar, QSlider, QGroupBox, QGridLayout, QSplitter, QFrame,
    QScrollArea, QCheckBox, QLineEdit, QMessageBox, QDialog, QFormLayout,
    QDialogButtonBox, QListWidget, QTreeWidget, QTreeWidgetItem, QHeaderView,
    QAbstractItemView, QTabWidget, QApplication, QMainWindow, QMenu
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

class BacktestDialog(QDialog):
    """Диалог настройки бэктестинга"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройка бэктестинга")
        self.setModal(True)
        self.resize(500, 400)
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Форма настроек
        form_layout = QFormLayout()
        
        # Выбор стратегии
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "Трендовая стратегия",
            "Боковая стратегия", 
            "Адаптивная стратегия",
            "Волатильностная стратегия",
            "Парная торговля"
        ])
        form_layout.addRow("Стратегия:", self.strategy_combo)
        
        # Выбор торговой пары
        self.pair_combo = QComboBox()
        self.pair_combo.addItems([
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"
        ])
        form_layout.addRow("Торговая пара:", self.pair_combo)
        
        # Начальный капитал
        self.initial_capital = QDoubleSpinBox()
        self.initial_capital.setRange(100, 1000000)
        self.initial_capital.setValue(10000)
        self.initial_capital.setSuffix(" USDT")
        form_layout.addRow("Начальный капитал:", self.initial_capital)
        
        # Период бэктестинга
        self.period_combo = QComboBox()
        self.period_combo.addItems([
            "1 день", "1 неделя", "1 месяц", "3 месяца", "6 месяцев", "1 год"
        ])
        form_layout.addRow("Период:", self.period_combo)
        
        # Комиссия
        self.commission = QDoubleSpinBox()
        self.commission.setRange(0, 1)
        self.commission.setValue(0.001)
        self.commission.setSuffix(" %")
        self.commission.setSingleStep(0.001)
        form_layout.addRow("Комиссия:", self.commission)
        
        layout.addLayout(form_layout)
        
        # Кнопки
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def get_settings(self):
        """Получение настроек бэктестинга"""
        return {
            'strategy': self.strategy_combo.currentText(),
            'pair': self.pair_combo.currentText(),
            'initial_capital': self.initial_capital.value(),
            'period': self.period_combo.currentText(),
            'commission': self.commission.value()
        }

class ConfigurationDialog(QDialog):
    """Диалог конфигурации системы"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Конфигурация системы")
        self.setModal(True)
        self.resize(600, 500)
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Вкладки конфигурации
        tab_widget = QTabWidget()
        
        # Вкладка торговли
        trading_tab = self.create_trading_tab()
        tab_widget.addTab(trading_tab, "Торговля")
        
        # Вкладка рисков
        risk_tab = self.create_risk_tab()
        tab_widget.addTab(risk_tab, "Риски")
        
        # Вкладка подключений
        connection_tab = self.create_connection_tab()
        tab_widget.addTab(connection_tab, "Подключения")
        
        layout.addWidget(tab_widget)
        
        # Кнопки
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def create_trading_tab(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # Режим торговли
        self.trading_mode = QComboBox()
        self.trading_mode.addItems(["Симуляция", "Бумажная торговля", "Реальная торговля"])
        layout.addRow("Режим торговли:", self.trading_mode)
        
        # Максимальный размер позиции
        self.max_position_size = QDoubleSpinBox()
        self.max_position_size.setRange(0.001, 1.0)
        self.max_position_size.setValue(0.1)
        self.max_position_size.setSuffix(" % от капитала")
        layout.addRow("Макс. размер позиции:", self.max_position_size)
        
        # Количество одновременных позиций
        self.max_positions = QSpinBox()
        self.max_positions.setRange(1, 20)
        self.max_positions.setValue(5)
        layout.addRow("Макс. позиций:", self.max_positions)
        
        return widget
        
    def create_risk_tab(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # Максимальный дневной убыток
        self.max_daily_loss = QDoubleSpinBox()
        self.max_daily_loss.setRange(1, 50)
        self.max_daily_loss.setValue(5)
        self.max_daily_loss.setSuffix(" %")
        layout.addRow("Макс. дневной убыток:", self.max_daily_loss)
        
        # Максимальный убыток на сделку
        self.max_trade_loss = QDoubleSpinBox()
        self.max_trade_loss.setRange(0.1, 10)
        self.max_trade_loss.setValue(2)
        self.max_trade_loss.setSuffix(" %")
        layout.addRow("Макс. убыток на сделку:", self.max_trade_loss)
        
        # Stop Loss
        self.stop_loss = QDoubleSpinBox()
        self.stop_loss.setRange(1, 50)
        self.stop_loss.setValue(10)
        self.stop_loss.setSuffix(" %")
        layout.addRow("Stop Loss:", self.stop_loss)
        
        # Take Profit
        self.take_profit = QDoubleSpinBox()
        self.take_profit.setRange(1, 100)
        self.take_profit.setValue(20)
        self.take_profit.setSuffix(" %")
        layout.addRow("Take Profit:", self.take_profit)
        
        return widget
        
    def create_connection_tab(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # API ключ
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addRow("API Key:", self.api_key)
        
        # Секретный ключ
        self.secret_key = QLineEdit()
        self.secret_key.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addRow("Secret Key:", self.secret_key)
        
        # URL биржи
        self.exchange_url = QLineEdit()
        self.exchange_url.setText("https://api.binance.com")
        layout.addRow("URL биржи:", self.exchange_url)
        
        # Таймаут подключения
        self.timeout = QSpinBox()
        self.timeout.setRange(5, 60)
        self.timeout.setValue(30)
        self.timeout.setSuffix(" сек")
        layout.addRow("Таймаут:", self.timeout)
        
        return widget

class PerformanceWidget(QWidget):
    """Виджет отображения производительности"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Заголовок
        title = QLabel("Производительность системы")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #4ecdc4;")
        layout.addWidget(title)
        
        # Метрики производительности
        metrics_layout = QGridLayout()
        
        # CPU
        self.cpu_label = QLabel("CPU: 0%")
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setRange(0, 100)
        metrics_layout.addWidget(self.cpu_label, 0, 0)
        metrics_layout.addWidget(self.cpu_progress, 0, 1)
        
        # Память
        self.memory_label = QLabel("Память: 0 MB")
        self.memory_progress = QProgressBar()
        self.memory_progress.setRange(0, 100)
        metrics_layout.addWidget(self.memory_label, 1, 0)
        metrics_layout.addWidget(self.memory_progress, 1, 1)
        
        # Сеть
        self.network_label = QLabel("Сеть: 0 KB/s")
        metrics_layout.addWidget(self.network_label, 2, 0)
        
        # Диск
        self.disk_label = QLabel("Диск: 0 KB/s")
        metrics_layout.addWidget(self.disk_label, 2, 1)
        
        layout.addLayout(metrics_layout)
        
        # График производительности
        self.performance_chart = self.create_performance_chart()
        layout.addWidget(self.performance_chart)
        
        # Таймер обновления
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_metrics)
        self.update_timer.start(1000)  # Обновление каждую секунду
        
    def create_performance_chart(self):
        """Создание графика производительности"""
        chart = QChart()
        chart.setTitle("Использование ресурсов")
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        
        # Серия CPU
        self.cpu_series = QLineSeries()
        self.cpu_series.setName("CPU")
        
        # Серия памяти
        self.memory_series = QLineSeries()
        self.memory_series.setName("Память")
        
        chart.addSeries(self.cpu_series)
        chart.addSeries(self.memory_series)
        
        # Настройка осей
        axis_x = QValueAxis()
        axis_x.setTitleText("Время (сек)")
        axis_x.setRange(0, 60)
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        self.cpu_series.attachAxis(axis_x)
        self.memory_series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        axis_y.setTitleText("Использование (%)")
        axis_y.setRange(0, 100)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        self.cpu_series.attachAxis(axis_y)
        self.memory_series.attachAxis(axis_y)
        
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        chart_view.setMaximumHeight(200)
        
        return chart_view
        
    def update_metrics(self):
        """Обновление метрик производительности"""
        import psutil
        
        # CPU
        cpu_percent = psutil.cpu_percent()
        self.cpu_label.setText(f"CPU: {cpu_percent:.1f}%")
        self.cpu_progress.setValue(int(cpu_percent))
        
        # Память
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_mb = memory.used / 1024 / 1024
        self.memory_label.setText(f"Память: {memory_mb:.0f} MB")
        self.memory_progress.setValue(int(memory_percent))
        
        # Обновление графика
        current_time = len(self.cpu_series.points())
        if current_time > 60:
            # Удаление старых точек
            self.cpu_series.removePoints(0, 1)
            self.memory_series.removePoints(0, 1)
            # Сдвиг оси X
            self.cpu_series.chart().axes(Qt.Orientation.Horizontal)[0].setRange(current_time - 60, current_time)
        
        self.cpu_series.append(current_time, cpu_percent)
        self.memory_series.append(current_time, memory_percent)

class StrategyManagerWidget(QWidget):
    """Виджет управления стратегиями"""
    
    strategy_activated = pyqtSignal(str)
    strategy_deactivated = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Заголовок
        title = QLabel("Управление стратегиями")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #4ecdc4;")
        layout.addWidget(title)
        
        # Список стратегий
        self.strategies_tree = QTreeWidget()
        self.strategies_tree.setHeaderLabels(["Стратегия", "Статус", "P&L", "Действия"])
        self.strategies_tree.setColumnWidth(0, 200)
        self.strategies_tree.setColumnWidth(1, 100)
        self.strategies_tree.setColumnWidth(2, 100)
        layout.addWidget(self.strategies_tree)
        
        # Кнопки управления
        button_layout = QHBoxLayout()
        
        self.add_strategy_btn = QPushButton("Добавить стратегию")
        self.add_strategy_btn.clicked.connect(self.add_strategy)
        button_layout.addWidget(self.add_strategy_btn)
        
        self.remove_strategy_btn = QPushButton("Удалить стратегию")
        self.remove_strategy_btn.clicked.connect(self.remove_strategy)
        button_layout.addWidget(self.remove_strategy_btn)
        
        self.configure_strategy_btn = QPushButton("Настроить")
        self.configure_strategy_btn.clicked.connect(self.configure_strategy)
        button_layout.addWidget(self.configure_strategy_btn)
        
        layout.addLayout(button_layout)
        
        # Заполнение тестовыми данными
        self.populate_strategies()
        
    def populate_strategies(self):
        """Заполнение списка стратегий"""
        strategies = [
            {"name": "Трендовая стратегия", "status": "Активна", "pnl": "+15.2%"},
            {"name": "Боковая стратегия", "status": "Неактивна", "pnl": "-2.1%"},
            {"name": "Адаптивная стратегия", "status": "Активна", "pnl": "+8.7%"},
            {"name": "Волатильностная стратегия", "status": "Неактивна", "pnl": "+12.3%"},
            {"name": "Парная торговля", "status": "Активна", "pnl": "+5.9%"}
        ]
        
        for strategy in strategies:
            item = QTreeWidgetItem()
            item.setText(0, strategy["name"])
            item.setText(1, strategy["status"])
            item.setText(2, strategy["pnl"])
            
            # Кнопка активации/деактивации
            if strategy["status"] == "Активна":
                item.setText(3, "Деактивировать")
            else:
                item.setText(3, "Активировать")
                
            self.strategies_tree.addTopLevelItem(item)
            
    def add_strategy(self):
        """Добавление новой стратегии"""
        # Здесь можно добавить диалог выбора стратегии
        QMessageBox.information(self, "Информация", "Функция добавления стратегии будет реализована позже")
        
    def remove_strategy(self):
        """Удаление стратегии"""
        current_item = self.strategies_tree.currentItem()
        if current_item:
            reply = QMessageBox.question(self, "Подтверждение", 
                                       f"Удалить стратегию '{current_item.text(0)}'?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.strategies_tree.takeTopLevelItem(self.strategies_tree.indexOfTopLevelItem(current_item))
        else:
            QMessageBox.warning(self, "Предупреждение", "Выберите стратегию для удаления")
            
    def configure_strategy(self):
        """Настройка стратегии"""
        current_item = self.strategies_tree.currentItem()
        if current_item:
            QMessageBox.information(self, "Информация", 
                                  f"Настройка стратегии '{current_item.text(0)}' будет реализована позже")
        else:
            QMessageBox.warning(self, "Предупреждение", "Выберите стратегию для настройки")

class MarketDataWidget(QWidget):
    """Виджет отображения рыночных данных"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Заголовок
        title = QLabel("Рыночные данные")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #4ecdc4;")
        layout.addWidget(title)
        
        # Таблица рыночных данных
        self.market_table = QTableWidget()
        self.market_table.setColumnCount(6)
        self.market_table.setHorizontalHeaderLabels([
            "Пара", "Цена", "Изменение 24ч", "Объем 24ч", "Высшая", "Низшая"
        ])
        self.market_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.market_table)
        
        # Заполнение тестовыми данными
        self.populate_market_data()
        
        # Таймер обновления
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_market_data)
        self.update_timer.start(5000)  # Обновление каждые 5 секунд
        
    def populate_market_data(self):
        """Заполнение рыночных данных"""
        market_data = [
            {"pair": "BTC/USDT", "price": "45,234.56", "change": "+2.34%", "volume": "1.2B", "high": "46,123.45", "low": "44,567.89"},
            {"pair": "ETH/USDT", "price": "2,345.67", "change": "-1.23%", "volume": "856M", "high": "2,456.78", "low": "2,234.56"},
            {"pair": "BNB/USDT", "price": "312.45", "change": "+0.87%", "volume": "234M", "high": "318.90", "low": "308.12"},
            {"pair": "ADA/USDT", "price": "0.456", "change": "+3.45%", "volume": "123M", "high": "0.467", "low": "0.445"},
            {"pair": "SOL/USDT", "price": "98.76", "change": "-0.56%", "volume": "345M", "high": "101.23", "low": "97.45"}
        ]
        
        self.market_table.setRowCount(len(market_data))
        
        for i, data in enumerate(market_data):
            self.market_table.setItem(i, 0, QTableWidgetItem(data["pair"]))
            self.market_table.setItem(i, 1, QTableWidgetItem(data["price"]))
            
            # Цветовое кодирование изменений
            change_item = QTableWidgetItem(data["change"])
            if "+" in data["change"]:
                change_item.setForeground(QColor("#28a745"))
            else:
                change_item.setForeground(QColor("#dc3545"))
            self.market_table.setItem(i, 2, change_item)
            
            self.market_table.setItem(i, 3, QTableWidgetItem(data["volume"]))
            self.market_table.setItem(i, 4, QTableWidgetItem(data["high"]))
            self.market_table.setItem(i, 5, QTableWidgetItem(data["low"]))
            
    def update_market_data(self):
        """Обновление рыночных данных"""
        # Здесь будет реальное обновление данных с биржи
        # Пока просто обновляем время последнего обновления
        pass 