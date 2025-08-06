#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra Modern Design Demo for ATB Desktop Application
Демонстрация ультрасовременного дизайна для ATB
"""

import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

from modern_style_system import UltraModernStyleSystem
from ultra_modern_widgets import (
    UltraModernDashboard, UltraModernPerformanceWidget,
    UltraModernMetricCard, UltraModernChartCard, UltraModernStatusCard,
    UltraModernActivityFeed, UltraModernQuickActions
)

class UltraModernDemoApp(QMainWindow):
    """Демонстрационное приложение с ультрасовременным дизайном"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Инициализация пользовательского интерфейса"""
        self.setWindowTitle("ATB Trading System - Ultra Modern Design Demo")
        self.setGeometry(100, 100, 1920, 1080)
        self.setMinimumSize(1600, 900)
        
        # Применение ультрасовременного стиля
        self.setStyleSheet(UltraModernStyleSystem.get_ultra_modern_stylesheet())
        
        # Установка шрифта Inter (если доступен)
        try:
            font = QFont("Inter", 10)
            self.setFont(font)
        except (AttributeError, TypeError) as e:
            # Fallback на системный шрифт при проблемах с QFont
            print(f"Warning: Failed to set Inter font, using system default: {e}")
            pass
        
        # Создание центрального виджета
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Главный layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Создание вкладок
        self.create_tabs(main_layout)
        
        # Запуск таймера для обновления данных
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_data)
        self.update_timer.start(5000)  # Обновление каждые 5 секунд
        
    def create_tabs(self, main_layout):
        """Создание вкладок приложения"""
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background: transparent;
            }
        """)
        
        # Вкладка 1: Главная панель
        dashboard_tab = UltraModernDashboard()
        self.tab_widget.addTab(dashboard_tab, "Главная панель")
        
        # Вкладка 2: Производительность
        performance_tab = UltraModernPerformanceWidget()
        self.tab_widget.addTab(performance_tab, "Производительность")
        
        # Вкладка 3: Компоненты
        components_tab = self.create_components_tab()
        self.tab_widget.addTab(components_tab, "Компоненты")
        
        # Вкладка 4: Аналитика
        analytics_tab = self.create_analytics_tab()
        self.tab_widget.addTab(analytics_tab, "Аналитика")
        
        main_layout.addWidget(self.tab_widget)
        
    def create_components_tab(self):
        """Создание вкладки с компонентами"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(24)
        
        # Заголовок
        from modern_style_system import UltraModernLabel
        title_label = UltraModernLabel("Компоненты системы", style="header")
        layout.addWidget(title_label)
        
        # Сетка компонентов
        from PyQt6.QtWidgets import QGridLayout
        grid_layout = QGridLayout()
        grid_layout.setSpacing(24)
        
        # Метрики
        metrics = [
            ("Цена BTC", "$45,250.00", "+2.5%", "up"),
            ("Цена ETH", "$3,200.50", "-1.2%", "down"),
            ("Объем 24ч", "$2.4B", "+8.7%", "up"),
            ("Изменение 24ч", "+$1,250", "+5.2%", "up")
        ]
        
        for i, (title, value, change, trend) in enumerate(metrics):
            metric_card = UltraModernMetricCard(title, value, change, trend)
            grid_layout.addWidget(metric_card, i // 2, i % 2)
        
        layout.addLayout(grid_layout)
        
        # Статусы компонентов
        statuses = [
            ("Торговый движок", "active", "Все системы работают нормально"),
            ("Риск-менеджмент", "active", "Мониторинг активен"),
            ("Аналитика", "warning", "Обновление данных"),
            ("API соединение", "active", "Стабильное соединение"),
            ("База данных", "active", "Синхронизирована"),
            ("Веб-сокеты", "active", "Подключены")
        ]
        
        status_grid = QGridLayout()
        status_grid.setSpacing(16)
        
        for i, (title, status, desc) in enumerate(statuses):
            status_card = UltraModernStatusCard(title, status, desc)
            status_grid.addWidget(status_card, i // 2, i % 2)
        
        layout.addLayout(status_grid)
        
        return widget
        
    def create_analytics_tab(self):
        """Создание вкладки аналитики"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(24)
        
        # Заголовок
        from modern_style_system import UltraModernLabel
        title_label = UltraModernLabel("Аналитика и графики", style="header")
        layout.addWidget(title_label)
        
        # Графики
        from PyQt6.QtWidgets import QHBoxLayout
        charts_layout = QHBoxLayout()
        charts_layout.setSpacing(24)
        
        # График цены
        price_chart = UltraModernChartCard("График цены BTC/USD", "line")
        charts_layout.addWidget(price_chart)
        
        # График объема
        volume_chart = UltraModernChartCard("Объем торгов", "bar")
        charts_layout.addWidget(volume_chart)
        
        layout.addLayout(charts_layout)
        
        # Лента активности
        activity_feed = UltraModernActivityFeed()
        layout.addWidget(activity_feed)
        
        return widget
        
    def update_data(self):
        """Обновление данных для демонстрации"""
        # Здесь можно добавить логику обновления данных
        pass

def main():
    """Главная функция приложения"""
    app = QApplication(sys.argv)
    
    # Настройка приложения
    app.setApplicationName("ATB Ultra Modern Demo")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("ATB Trading System")
    
    # Создание и отображение главного окна
    window = UltraModernDemoApp()
    window.show()
    
    # Запуск приложения
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 