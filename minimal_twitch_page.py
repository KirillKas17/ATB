#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimalistic Twitch Demo Page for ATB Trading System
Минималистичная страница для демонстрации на Twitch
"""

from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtCharts import *
import random
import time
from datetime import datetime, timedelta
from minimal_apple_style import *

class MinimalTwitchPage(QWidget):
    """Минималистичная страница для демонстрации на Twitch"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.setup_timers()
        self.generate_demo_data()
    
    def init_ui(self):
        """Инициализация интерфейса"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Заголовок страницы
        header = self.create_header()
        layout.addWidget(header)
        
        # Основной контент
        content = self.create_content()
        layout.addWidget(content)
    
    def create_header(self):
        """Создание минималистичного заголовка"""
        header_widget = QWidget()
        header_widget.setFixedHeight(60)
        header_widget.setStyleSheet(f"""
            background: {MinimalAppleStyle.COLORS['surface']};
            border-bottom: 1px solid {MinimalAppleStyle.COLORS['subtle_border']};
        """)
        
        layout = QHBoxLayout(header_widget)
        layout.setContentsMargins(24, 12, 24, 12)
        
        # Логотип и название
        title_layout = QHBoxLayout()
        
        # Иконка торговли
        icon_label = QLabel("📈")
        icon_label.setStyleSheet("font-size: 24px; margin-right: 12px;")
        title_layout.addWidget(icon_label)
        
        title_label = QLabel("ATB Trading System")
        title_label.setStyleSheet("""
            font-size: 18px;
            font-weight: 500;
            color: white;
        """)
        title_layout.addWidget(title_label)
        
        layout.addLayout(title_layout)
        layout.addStretch()
        
        # Статус системы
        self.status_label = QLabel("🟢 Active")
        self.status_label.setStyleSheet("""
            font-size: 14px;
            font-weight: 500;
            color: white;
            padding: 6px 12px;
            background: rgba(52, 199, 89, 0.2);
            border-radius: 12px;
        """)
        layout.addWidget(self.status_label)
        
        return header_widget
    
    def create_content(self):
        """Создание основного контента"""
        content_widget = QWidget()
        layout = QHBoxLayout(content_widget)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(24)
        
        # Левая панель - Основная статистика
        left_panel = self.create_left_panel()
        layout.addWidget(left_panel, 2)
        
        # Центральная панель - Графики
        center_panel = self.create_center_panel()
        layout.addWidget(center_panel, 3)
        
        # Правая панель - Активные позиции
        right_panel = self.create_right_panel()
        layout.addWidget(right_panel, 2)
        
        return content_widget
    
    def create_left_panel(self):
        """Создание левой панели с основной статистикой"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(16)
        
        # Заголовок
        title = MinimalLabel("Statistics", "subtitle")
        layout.addWidget(title)
        
        # Карточки статистики
        self.pnl_card = self.create_stat_card("P&L Today", "$0.00", "success")
        layout.addWidget(self.pnl_card)
        
        self.trades_card = self.create_stat_card("Total Trades", "0", "body")
        layout.addWidget(self.trades_card)
        
        self.winrate_card = self.create_stat_card("Win Rate", "0%", "body")
        layout.addWidget(self.winrate_card)
        
        self.volume_card = self.create_stat_card("Volume 24h", "$0.00", "body")
        layout.addWidget(self.volume_card)
        
        # Временные интервалы
        time_stats = self.create_time_stats()
        layout.addWidget(time_stats)
        
        layout.addStretch()
        return panel
    
    def create_stat_card(self, title, value, color_type):
        """Создание минималистичной карточки статистики"""
        card = MinimalCard()
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(8)
        
        # Заголовок
        title_label = MinimalLabel(title, "caption")
        card_layout.addWidget(title_label)
        
        # Значение
        value_label = QLabel(value)
        if color_type == "success":
            value_label.setStyleSheet(f"""
                font-size: 24px;
                font-weight: 500;
                color: {MinimalAppleStyle.COLORS['success']};
                margin: 4px 0;
            """)
        else:
            value_label.setStyleSheet(f"""
                font-size: 24px;
                font-weight: 500;
                color: {MinimalAppleStyle.COLORS['text_primary']};
                margin: 4px 0;
            """)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(value_label)
        
        # Сохраняем ссылку для обновления
        card.value_label = value_label
        card.color_type = color_type
        
        return card
    
    def create_time_stats(self):
        """Создание статистики по временным интервалам"""
        card = MinimalCard("Time Intervals")
        layout = QGridLayout(card)
        layout.setSpacing(12)
        
        # Заголовки
        headers = ["Period", "Trades", "P&L", "Volume"]
        for i, header in enumerate(headers):
            header_label = MinimalLabel(header, "caption")
            header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(header_label, 0, i)
        
        # Данные
        periods = ["1 Hour", "1 Day", "1 Week", "1 Month"]
        self.time_stats_labels = {}
        
        for i, period in enumerate(periods):
            # Период
            period_label = MinimalLabel(period, "body")
            layout.addWidget(period_label, i+1, 0)
            
            # Торговли
            trades_label = MinimalLabel("0", "body")
            trades_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(trades_label, i+1, 1)
            
            # P&L
            pnl_label = MinimalLabel("$0.00", "body")
            pnl_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(pnl_label, i+1, 2)
            
            # Объем
            volume_label = MinimalLabel("$0.00", "body")
            volume_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(volume_label, i+1, 3)
            
            self.time_stats_labels[period] = {
                'trades': trades_label,
                'pnl': pnl_label,
                'volume': volume_label
            }
        
        return card
    
    def create_center_panel(self):
        """Создание центральной панели с графиками"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(16)
        
        # Заголовок
        title = MinimalLabel("Charts", "subtitle")
        layout.addWidget(title)
        
        # График P&L
        pnl_chart = self.create_pnl_chart()
        layout.addWidget(pnl_chart)
        
        # График активности
        activity_chart = self.create_activity_chart()
        layout.addWidget(activity_chart)
        
        return panel
    
    def create_pnl_chart(self):
        """Создание минималистичного графика P&L"""
        card = MinimalCard("P&L")
        
        chart_view = MinimalChart()
        chart = chart_view.chart()
        chart.setTitle("Real-time P&L")
        
        # Создаем серию данных
        self.pnl_series = QLineSeries()
        self.pnl_series.setName("P&L")
        self.pnl_series.setPen(QPen(QColor(MinimalAppleStyle.COLORS['primary']), 2))
        
        chart.addSeries(self.pnl_series)
        
        # Оси
        axis_x = QValueAxis()
        axis_x.setRange(0, 24)
        axis_x.setTitleText("Hours")
        axis_x.setLabelFormat("%d")
        
        axis_y = QValueAxis()
        axis_y.setRange(-1000, 1000)
        axis_y.setTitleText("P&L ($)")
        axis_y.setLabelFormat("%.0f")
        
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        
        self.pnl_series.attachAxis(axis_x)
        self.pnl_series.attachAxis(axis_y)
        
        card_layout = QVBoxLayout(card)
        card_layout.addWidget(chart_view)
        
        return card
    
    def create_activity_chart(self):
        """Создание минималистичного графика активности"""
        card = MinimalCard("Trading Activity")
        
        chart_view = MinimalChart()
        chart = chart_view.chart()
        chart.setTitle("Trades per Hour")
        
        # Создаем серию данных
        self.activity_series = QBarSeries()
        
        # Создаем набор данных
        self.activity_set = QBarSet("Trades")
        self.activity_set.setColor(QColor(MinimalAppleStyle.COLORS['primary']))
        
        # Инициализируем данные
        for i in range(24):
            self.activity_set.append(random.randint(0, 10))
        
        self.activity_series.append(self.activity_set)
        chart.addSeries(self.activity_series)
        
        # Оси
        axis_x = QBarCategoryAxis()
        hours = [str(i) for i in range(24)]
        axis_x.append(hours)
        axis_x.setTitleText("Hour")
        
        axis_y = QValueAxis()
        axis_y.setRange(0, 15)
        axis_y.setTitleText("Number of Trades")
        
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        
        self.activity_series.attachAxis(axis_x)
        self.activity_series.attachAxis(axis_y)
        
        card_layout = QVBoxLayout(card)
        card_layout.addWidget(chart_view)
        
        return card
    
    def create_right_panel(self):
        """Создание правой панели с активными позициями"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(16)
        
        # Заголовок
        title = MinimalLabel("Active Positions", "subtitle")
        layout.addWidget(title)
        
        # Таблица активных позиций
        positions_table = self.create_positions_table()
        layout.addWidget(positions_table)
        
        # Последние сделки
        recent_trades = self.create_recent_trades()
        layout.addWidget(recent_trades)
        
        return panel
    
    def create_positions_table(self):
        """Создание минималистичной таблицы активных позиций"""
        card = MinimalCard("Open Positions")
        
        table = MinimalTable()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Symbol", "Side", "Size", "Entry", "P&L"])
        table.setRowCount(0)
        
        # Настройка таблицы
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        
        self.positions_table = table
        
        card_layout = QVBoxLayout(card)
        card_layout.addWidget(table)
        
        return card
    
    def create_recent_trades(self):
        """Создание минималистичного списка последних сделок"""
        card = MinimalCard("Recent Trades")
        
        table = MinimalTable()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(["Time", "Symbol", "Side", "Size", "Price", "P&L"])
        table.setRowCount(0)
        
        # Настройка таблицы
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        
        self.trades_table = table
        
        card_layout = QVBoxLayout(card)
        card_layout.addWidget(table)
        
        return card
    
    def setup_timers(self):
        """Настройка таймеров для обновления данных"""
        # Таймер для обновления статистики
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_statistics)
        self.stats_timer.start(3000)  # Обновление каждые 3 секунды
        
        # Таймер для обновления графиков
        self.chart_timer = QTimer()
        self.chart_timer.timeout.connect(self.update_charts)
        self.chart_timer.start(5000)  # Обновление каждые 5 секунд
        
        # Таймер для обновления позиций
        self.positions_timer = QTimer()
        self.positions_timer.timeout.connect(self.update_positions)
        self.positions_timer.start(4000)  # Обновление каждые 4 секунды
    
    def generate_demo_data(self):
        """Генерация демонстрационных данных"""
        self.demo_data = {
            'total_pnl': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'volume_24h': 0.0,
            'positions': [],
            'trades': [],
            'hourly_pnl': [0] * 24,
            'hourly_trades': [0] * 24
        }
        
        # Генерируем начальные позиции
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"]
        sides = ["LONG", "SHORT"]
        
        for i in range(3):
            position = {
                'symbol': random.choice(symbols),
                'side': random.choice(sides),
                'size': round(random.uniform(0.1, 2.0), 2),
                'entry_price': round(random.uniform(100, 50000), 2),
                'current_price': round(random.uniform(100, 50000), 2),
                'pnl': round(random.uniform(-500, 500), 2)
            }
            self.demo_data['positions'].append(position)
        
        # Генерируем начальные сделки
        for i in range(10):
            trade = {
                'time': datetime.now() - timedelta(minutes=random.randint(1, 60)),
                'symbol': random.choice(symbols),
                'side': random.choice(sides),
                'size': round(random.uniform(0.1, 1.0), 2),
                'price': round(random.uniform(100, 50000), 2),
                'pnl': round(random.uniform(-100, 100), 2)
            }
            self.demo_data['trades'].append(trade)
    
    def update_statistics(self):
        """Обновление статистики"""
        # Обновляем основные показатели
        self.demo_data['total_pnl'] += random.uniform(-30, 30)
        self.demo_data['total_trades'] += random.randint(0, 1)
        self.demo_data['win_rate'] = random.uniform(60, 85)
        self.demo_data['volume_24h'] += random.uniform(50, 500)
        
        # Обновляем карточки
        self.pnl_card.value_label.setText(f"${self.demo_data['total_pnl']:,.2f}")
        self.trades_card.value_label.setText(f"{self.demo_data['total_trades']}")
        self.winrate_card.value_label.setText(f"{self.demo_data['win_rate']:.1f}%")
        self.volume_card.value_label.setText(f"${self.demo_data['volume_24h']:,.0f}")
        
        # Обновляем временную статистику
        for period in ["1 Hour", "1 Day", "1 Week", "1 Month"]:
            stats = self.time_stats_labels[period]
            stats['trades'].setText(str(random.randint(5, 50)))
            stats['pnl'].setText(f"${random.uniform(-1000, 2000):,.2f}")
            stats['volume'].setText(f"${random.uniform(1000, 10000):,.0f}")
        
        # Обновляем статус
        if random.random() > 0.97:  # 3% шанс изменения статуса
            if self.status_label.text() == "🟢 Active":
                self.status_label.setText("🟡 Processing")
                self.status_label.setStyleSheet("""
                    font-size: 14px;
                    font-weight: 500;
                    color: white;
                    padding: 6px 12px;
                    background: rgba(255, 149, 0, 0.2);
                    border-radius: 12px;
                """)
            else:
                self.status_label.setText("🟢 Active")
                self.status_label.setStyleSheet("""
                    font-size: 14px;
                    font-weight: 500;
                    color: white;
                    padding: 6px 12px;
                    background: rgba(52, 199, 89, 0.2);
                    border-radius: 12px;
                """)
    
    def update_charts(self):
        """Обновление графиков"""
        # Обновляем график P&L
        current_time = datetime.now().hour
        self.demo_data['hourly_pnl'][current_time] = self.demo_data['total_pnl']
        
        self.pnl_series.clear()
        for i, pnl in enumerate(self.demo_data['hourly_pnl']):
            self.pnl_series.append(i, pnl)
        
        # Обновляем график активности
        self.demo_data['hourly_trades'][current_time] += random.randint(0, 2)
        
        self.activity_set.remove(0, 24)
        for trades in self.demo_data['hourly_trades']:
            self.activity_set.append(trades)
    
    def update_positions(self):
        """Обновление позиций и сделок"""
        # Обновляем позиции
        for position in self.demo_data['positions']:
            position['current_price'] += random.uniform(-50, 50)
            position['pnl'] = (position['current_price'] - position['entry_price']) * position['size']
            if position['side'] == 'SHORT':
                position['pnl'] = -position['pnl']
        
        # Обновляем таблицу позиций
        self.positions_table.setRowCount(len(self.demo_data['positions']))
        for i, position in enumerate(self.demo_data['positions']):
            self.positions_table.setItem(i, 0, QTableWidgetItem(position['symbol']))
            self.positions_table.setItem(i, 1, QTableWidgetItem(position['side']))
            self.positions_table.setItem(i, 2, QTableWidgetItem(str(position['size'])))
            self.positions_table.setItem(i, 3, QTableWidgetItem(f"${position['entry_price']:,.2f}"))
            
            pnl_item = QTableWidgetItem(f"${position['pnl']:,.2f}")
            if position['pnl'] > 0:
                pnl_item.setForeground(QBrush(QColor(MinimalAppleStyle.COLORS['success'])))
            else:
                pnl_item.setForeground(QBrush(QColor(MinimalAppleStyle.COLORS['danger'])))
            self.positions_table.setItem(i, 4, pnl_item)
        
        # Добавляем новые сделки
        if random.random() > 0.8:  # 20% шанс новой сделки
            symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"]
            sides = ["LONG", "SHORT"]
            
            trade = {
                'time': datetime.now(),
                'symbol': random.choice(symbols),
                'side': random.choice(sides),
                'size': round(random.uniform(0.1, 1.0), 2),
                'price': round(random.uniform(100, 50000), 2),
                'pnl': round(random.uniform(-100, 100), 2)
            }
            self.demo_data['trades'].insert(0, trade)
            
            # Ограничиваем количество сделок
            if len(self.demo_data['trades']) > 15:
                self.demo_data['trades'] = self.demo_data['trades'][:15]
        
        # Обновляем таблицу сделок
        self.trades_table.setRowCount(len(self.demo_data['trades']))
        for i, trade in enumerate(self.demo_data['trades']):
            self.trades_table.setItem(i, 0, QTableWidgetItem(trade['time'].strftime("%H:%M:%S")))
            self.trades_table.setItem(i, 1, QTableWidgetItem(trade['symbol']))
            self.trades_table.setItem(i, 2, QTableWidgetItem(trade['side']))
            self.trades_table.setItem(i, 3, QTableWidgetItem(str(trade['size'])))
            self.trades_table.setItem(i, 4, QTableWidgetItem(f"${trade['price']:,.2f}"))
            
            pnl_item = QTableWidgetItem(f"${trade['pnl']:,.2f}")
            if trade['pnl'] > 0:
                pnl_item.setForeground(QBrush(QColor(MinimalAppleStyle.COLORS['success'])))
            else:
                pnl_item.setForeground(QBrush(QColor(MinimalAppleStyle.COLORS['danger'])))
            self.trades_table.setItem(i, 5, pnl_item) 