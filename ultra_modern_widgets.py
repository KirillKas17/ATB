#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra Modern Widgets for ATB Desktop Application
Ультрасовременные виджеты для ATB
"""

from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtCharts import *
import math
import random
from datetime import datetime, timedelta

from modern_style_system import (
    UltraModernStyleSystem, UltraModernCard, UltraModernButton, 
    UltraModernLabel, UltraModernBadge, UltraModernDivider,
    UltraModernProgressBar, UltraModernTable, UltraModernComboBox
)

class UltraModernMetricCard(QWidget):
    """Ультрасовременная карточка с метрикой"""
    
    def __init__(self, title="", value="", change="", trend="up", parent=None) -> None:
        super().__init__(parent)
        self.title = title
        self.value = value
        self.change = change
        self.trend = trend
        self.init_ui()
        
    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        
        # Заголовок
        title_label = UltraModernLabel(self.title, style="caption")
        layout.addWidget(title_label)
        
        # Значение
        value_label = UltraModernLabel(self.value, style="metric")
        layout.addWidget(value_label)
        
        # Изменение
        if self.change:
            change_layout = QHBoxLayout()
            change_layout.setSpacing(8)
            
            # Иконка тренда
            trend_icon = QLabel()
            if self.trend == "up":
                trend_icon.setText("↗")
                trend_icon.setStyleSheet("color: #10B981; font-size: 16px; font-weight: bold;")
            else:
                trend_icon.setText("↘")
                trend_icon.setStyleSheet("color: #EF4444; font-size: 16px; font-weight: bold;")
            
            change_layout.addWidget(trend_icon)
            
            # Текст изменения
            change_label = UltraModernLabel(self.change, style="body")
            change_label.setStyleSheet(change_label.styleSheet() + f"color: {'#10B981' if self.trend == 'up' else '#EF4444'};")
            change_layout.addWidget(change_label)
            
            change_layout.addStretch()
            layout.addLayout(change_layout)
        
        layout.addStretch()
        
        # Применение стиля карточки
        self.setProperty("class", "UltraModernCard")

class UltraModernChartCard(QWidget):
    """Ультрасовременная карточка с графиком"""
    
    def __init__(self, title="", chart_type="line", parent=None) -> None:
        super().__init__(parent)
        self.title = title
        self.chart_type = chart_type
        self.init_ui()
        
    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        # Заголовок
        title_label = UltraModernLabel(self.title, style="subheader")
        layout.addWidget(title_label)
        
        # График
        self.chart_view = self.create_chart()
        layout.addWidget(self.chart_view)
        
        # Применение стиля карточки
        self.setProperty("class", "UltraModernCard")
        
    def create_chart(self) -> None:
        chart = QChart()
        chart.setTheme(QChart.ChartTheme.ChartThemeDark)
        chart.setBackgroundBrush(QBrush(QColor(UltraModernStyleSystem.COLORS['surface'])))
        chart.setPlotAreaBackgroundBrush(QBrush(QColor(UltraModernStyleSystem.COLORS['surface'])))
        chart.legend().hide()
        
        # Создание данных
        series = QLineSeries()
        series.setPen(QPen(QColor(UltraModernStyleSystem.COLORS['primary']), 3))
        
        # Генерация случайных данных
        for i in range(50):
            value = random.uniform(100, 200)
            series.append(i, value)
        
        chart.addSeries(series)
        chart.createDefaultAxes()
        
        # Настройка осей
        chart.axes(Qt.Orientation.Horizontal)[0].setLabelsColor(QColor(UltraModernStyleSystem.COLORS['text_secondary']))
        chart.axes(Qt.Orientation.Vertical)[0].setLabelsColor(QColor(UltraModernStyleSystem.COLORS['text_secondary']))
        
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        chart_view.setMinimumHeight(200)
        
        return chart_view

class UltraModernStatusCard(QWidget):
    """Ультрасовременная карточка статуса"""
    
    def __init__(self, title="", status="active", description="", parent=None) -> None:
        super().__init__(parent)
        self.title = title
        self.status = status
        self.description = description
        self.init_ui()
        
    def init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        # Иконка статуса
        status_icon = QLabel()
        if self.status == "active":
            status_icon.setText("●")
            status_icon.setStyleSheet("color: #10B981; font-size: 24px; font-weight: bold;")
        elif self.status == "warning":
            status_icon.setText("●")
            status_icon.setStyleSheet("color: #F59E0B; font-size: 24px; font-weight: bold;")
        else:
            status_icon.setText("●")
            status_icon.setStyleSheet("color: #EF4444; font-size: 24px; font-weight: bold;")
        
        layout.addWidget(status_icon)
        
        # Контент
        content_layout = QVBoxLayout()
        content_layout.setSpacing(4)
        
        title_label = UltraModernLabel(self.title, style="body")
        title_label.setStyleSheet(title_label.styleSheet() + "font-weight: 600;")
        content_layout.addWidget(title_label)
        
        if self.description:
            desc_label = UltraModernLabel(self.description, style="caption")
            content_layout.addWidget(desc_label)
        
        layout.addLayout(content_layout)
        layout.addStretch()
        
        # Бейдж статуса
        status_badge = UltraModernBadge(self.status.upper(), self.status)
        layout.addWidget(status_badge)
        
        # Применение стиля карточки
        self.setProperty("class", "UltraModernCard")

class UltraModernActivityFeed(QWidget):
    """Ультрасовременная лента активности"""
    
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        # Заголовок
        title_label = UltraModernLabel("Активность системы", style="subheader")
        layout.addWidget(title_label)
        
        # Список активности
        self.activity_list = QListWidget()
        self.activity_list.setStyleSheet("""
            QListWidget {
                background: transparent;
                border: none;
                outline: none;
            }
            QListWidget::item {
                background: transparent;
                border: none;
                padding: 12px 0;
            }
        """)
        
        # Добавление примеров активности
        activities = [
            ("Покупка BTC", "2.5 BTC по цене $45,000", "success", "2 мин назад"),
            ("Продажа ETH", "10.0 ETH по цене $3,200", "danger", "5 мин назад"),
            ("Обновление стратегии", "Стратегия MA_Cross обновлена", "info", "10 мин назад"),
            ("Риск-менеджмент", "Стоп-лосс активирован", "warning", "15 мин назад"),
            ("Анализ рынка", "Новый паттерн обнаружен", "primary", "20 мин назад")
        ]
        
        for title, desc, type_, time in activities:
            item_widget = self.create_activity_item(title, desc, type_, time)
            list_item = QListWidgetItem()
            self.activity_list.addItem(list_item)
            self.activity_list.setItemWidget(list_item, item_widget)
            list_item.setSizeHint(item_widget.sizeHint())
        
        layout.addWidget(self.activity_list)
        
        # Применение стиля карточки
        self.setProperty("class", "UltraModernCard")
        
    def create_activity_item(self, title, description, type_, time) -> None:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # Иконка типа
        type_icon = QLabel()
        icons = {
            "success": "✓",
            "danger": "✗", 
            "warning": "⚠",
            "info": "ℹ",
            "primary": "●"
        }
        colors = {
            "success": "#10B981",
            "danger": "#EF4444",
            "warning": "#F59E0B", 
            "info": "#3B82F6",
            "primary": "#6366F1"
        }
        
        type_icon.setText(icons.get(type_, "●"))
        type_icon.setStyleSheet(f"color: {colors.get(type_, '#6366F1')}; font-size: 16px; font-weight: bold;")
        layout.addWidget(type_icon)
        
        # Контент
        content_layout = QVBoxLayout()
        content_layout.setSpacing(4)
        
        title_label = UltraModernLabel(title, style="body")
        title_label.setStyleSheet(title_label.styleSheet() + "font-weight: 500;")
        content_layout.addWidget(title_label)
        
        desc_label = UltraModernLabel(description, style="caption")
        content_layout.addWidget(desc_label)
        
        layout.addLayout(content_layout)
        layout.addStretch()
        
        # Время
        time_label = UltraModernLabel(time, style="caption")
        layout.addWidget(time_label)
        
        return widget

class UltraModernQuickActions(QWidget):
    """Ультрасовременные быстрые действия"""
    
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        # Заголовок
        title_label = UltraModernLabel("Быстрые действия", style="subheader")
        layout.addWidget(title_label)
        
        # Кнопки действий
        actions_layout = QGridLayout()
        actions_layout.setSpacing(12)
        
        actions = [
            ("Запустить торговлю", "success"),
            ("Остановить торговлю", "danger"),
            ("Бэктестинг", "info"),
            ("Настройки", "primary"),
            ("Экспорт данных", "secondary"),
            ("Аналитика", "accent")
        ]
        
        for i, (text, style) in enumerate(actions):
            btn = UltraModernButton(text, style=style, size="small")
            actions_layout.addWidget(btn, i // 2, i % 2)
        
        layout.addLayout(actions_layout)
        
        # Применение стиля карточки
        self.setProperty("class", "UltraModernCard")

class UltraModernPerformanceWidget(QWidget):
    """Ультрасовременный виджет производительности"""
    
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)
        
        # Заголовок
        title_label = UltraModernLabel("Производительность системы", style="header")
        layout.addWidget(title_label)
        
        # Метрики
        metrics_layout = QGridLayout()
        metrics_layout.setSpacing(16)
        
        metrics = [
            ("Общий P&L", "$12,450.50", "+8.5%", "up"),
            ("ROI", "24.5%", "+2.1%", "up"),
            ("Винрейт", "68.2%", "-1.3%", "down"),
            ("Макс. просадка", "12.4%", "0.0%", "neutral")
        ]
        
        for i, (title, value, change, trend) in enumerate(metrics):
            metric_card = UltraModernMetricCard(title, value, change, trend)
            metrics_layout.addWidget(metric_card, i // 2, i % 2)
        
        layout.addLayout(metrics_layout)
        
        # График производительности
        chart_card = UltraModernChartCard("График P&L", "line")
        layout.addWidget(chart_card)
        
        # Применение стиля карточки
        self.setProperty("class", "UltraModernCard")

class UltraModernDashboard(QWidget):
    """Ультрасовременная панель управления"""
    
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(24)
        
        # Заголовок панели
        header_layout = QHBoxLayout()
        header_layout.setSpacing(16)
        
        title_label = UltraModernLabel("ATB Trading System", style="header")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Статус системы
        status_badge = UltraModernBadge("АКТИВЕН", "success")
        header_layout.addWidget(status_badge)
        
        layout.addLayout(header_layout)
        
        # Основной контент
        content_layout = QHBoxLayout()
        content_layout.setSpacing(24)
        
        # Левая колонка
        left_column = QVBoxLayout()
        left_column.setSpacing(24)
        
        # Быстрые действия
        quick_actions = UltraModernQuickActions()
        left_column.addWidget(quick_actions)
        
        # Статусы компонентов
        statuses = [
            ("Торговый движок", "active", "Все системы работают нормально"),
            ("Риск-менеджмент", "active", "Мониторинг активен"),
            ("Аналитика", "warning", "Обновление данных"),
            ("API соединение", "active", "Стабильное соединение")
        ]
        
        for title, status, desc in statuses:
            status_card = UltraModernStatusCard(title, status, desc)
            left_column.addWidget(status_card)
        
        content_layout.addLayout(left_column)
        
        # Правая колонка
        right_column = QVBoxLayout()
        right_column.setSpacing(24)
        
        # Лента активности
        activity_feed = UltraModernActivityFeed()
        right_column.addWidget(activity_feed)
        
        # Прогресс-бары
        progress_card = UltraModernCard("Прогресс операций")
        progress_layout = QVBoxLayout(progress_card)
        progress_layout.setSpacing(16)
        
        progress_items = [
            ("Обработка ордеров", 75),
            ("Анализ рынка", 45),
            ("Управление рисками", 90),
            ("Синхронизация данных", 30)
        ]
        
        for label, value in progress_items:
            item_layout = QVBoxLayout()
            item_layout.setSpacing(8)
            
            label_layout = QHBoxLayout()
            label_layout.setSpacing(8)
            
            progress_label = UltraModernLabel(label, style="body")
            label_layout.addWidget(progress_label)
            
            label_layout.addStretch()
            
            percent_label = UltraModernLabel(f"{value}%", style="caption")
            label_layout.addWidget(percent_label)
            
            item_layout.addLayout(label_layout)
            
            progress_bar = UltraModernProgressBar()
            progress_bar.setValue(value)
            item_layout.addWidget(progress_bar)
            
            progress_layout.addLayout(item_layout)
        
        right_column.addWidget(progress_card)
        
        content_layout.addLayout(right_column)
        
        layout.addLayout(content_layout)
        
        # Нижняя панель с метриками
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(24)
        
        bottom_metrics = [
            ("Объем торгов", "$2.4M", "+12.5%", "up"),
            ("Количество сделок", "1,247", "+8.2%", "up"),
            ("Средняя прибыль", "$1,850", "+5.7%", "up"),
            ("Время работы", "24ч 32м", "0.0%", "neutral")
        ]
        
        for title, value, change, trend in bottom_metrics:
            metric_card = UltraModernMetricCard(title, value, change, trend)
            metrics_layout.addWidget(metric_card)
        
        layout.addLayout(metrics_layout) 