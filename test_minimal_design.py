#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for minimalistic Apple-style design system
"""

import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget
from PyQt6.QtCore import Qt
from minimal_apple_style import *

class MinimalDesignTest(QMainWindow):
    """Тестовое окно для проверки минималистичного дизайна"""
    
    def __init__(self) -> None:
        super().__init__()
        self.init_ui()
    
    def init_ui(self) -> None:
        """Инициализация интерфейса"""
        self.setWindowTitle("Minimal Design Test - ATB Trading System")
        self.setGeometry(100, 100, 1000, 700)
        
        # Применение минималистичного стиля
        self.setStyleSheet(MinimalAppleStyle.get_minimal_stylesheet())
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основной layout
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)
        
        # Заголовок
        title = MinimalLabel("Minimalistic Apple-Style Design", "title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        subtitle = MinimalLabel("Clean, elegant, and sophisticated design system", "body")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)
        
        # Разделитель
        divider = MinimalDivider()
        layout.addWidget(divider)
        
        # Секция кнопок
        buttons_section = self.create_buttons_section()
        layout.addWidget(buttons_section)
        
        # Секция карточек
        cards_section = self.create_cards_section()
        layout.addWidget(cards_section)
        
        # Секция форм
        forms_section = self.create_forms_section()
        layout.addWidget(forms_section)
        
        # Секция таблиц
        tables_section = self.create_tables_section()
        layout.addWidget(tables_section)
        
        layout.addStretch()
    
    def create_buttons_section(self) -> None:
        """Создание секции с кнопками"""
        card = MinimalCard("Buttons")
        layout = QHBoxLayout(card)
        layout.setSpacing(16)
        
        # Кнопки разных стилей
        primary_btn = MinimalButton("Primary", style="primary")
        layout.addWidget(primary_btn)
        
        default_btn = MinimalButton("Default", style="default")
        layout.addWidget(default_btn)
        
        success_btn = MinimalButton("Success", style="success")
        layout.addWidget(success_btn)
        
        danger_btn = MinimalButton("Danger", style="danger")
        layout.addWidget(danger_btn)
        
        layout.addStretch()
        return card
    
    def create_cards_section(self) -> None:
        """Создание секции с карточками"""
        card = MinimalCard("Cards")
        layout = QHBoxLayout(card)
        layout.setSpacing(16)
        
        # Обычная карточка
        normal_card = MinimalCard("Statistics")
        normal_layout = QVBoxLayout(normal_card)
        normal_layout.addWidget(MinimalLabel("Total Trades: 1,234", "body"))
        normal_layout.addWidget(MinimalLabel("Win Rate: 67.8%", "success"))
        normal_layout.addWidget(MinimalLabel("P&L: +$12,345", "success"))
        layout.addWidget(normal_card)
        
        # Еще одна карточка
        second_card = MinimalCard("Performance")
        second_layout = QVBoxLayout(second_card)
        second_layout.addWidget(MinimalLabel("Active Positions: 5", "body"))
        second_layout.addWidget(MinimalLabel("Daily Volume: $45,678", "body"))
        second_layout.addWidget(MinimalLabel("Risk Level: Medium", "warning"))
        layout.addWidget(second_card)
        
        layout.addStretch()
        return card
    
    def create_forms_section(self) -> None:
        """Создание секции с формами"""
        card = MinimalCard("Forms")
        layout = QVBoxLayout(card)
        layout.setSpacing(16)
        
        # Строка с элементами формы
        form_row = QHBoxLayout()
        
        # Комбобокс
        combo_label = MinimalLabel("Exchange:", "body")
        form_row.addWidget(combo_label)
        
        combo = MinimalComboBox()
        combo.addItems(["Binance", "Bybit", "OKX", "KuCoin"])
        form_row.addWidget(combo)
        
        # Спинбокс
        spin_label = MinimalLabel("Position Size (%):", "body")
        form_row.addWidget(spin_label)
        
        spin = MinimalSpinBox()
        spin.setRange(1, 100)
        spin.setValue(10)
        form_row.addWidget(spin)
        
        # Чекбокс
        checkbox = MinimalCheckBox("Use Testnet")
        checkbox.setChecked(True)
        form_row.addWidget(checkbox)
        
        form_row.addStretch()
        layout.addLayout(form_row)
        
        # Прогресс-бар
        progress_label = MinimalLabel("System Load:", "body")
        layout.addWidget(progress_label)
        
        progress = MinimalProgressBar()
        progress.setValue(75)
        layout.addWidget(progress)
        
        return card
    
    def create_tables_section(self) -> None:
        """Создание секции с таблицами"""
        card = MinimalCard("Tables")
        layout = QVBoxLayout(card)
        
        # Таблица
        table = MinimalTable()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Symbol", "Side", "Size", "P&L"])
        table.setRowCount(3)
        
        # Данные
        data = [
            ("BTC/USDT", "LONG", "0.5", "+$234.56"),
            ("ETH/USDT", "SHORT", "2.0", "-$123.45"),
            ("SOL/USDT", "LONG", "10.0", "+$567.89")
        ]
        
        for i, row_data in enumerate(data):
            for j, cell_data in enumerate(row_data):
                item = QTableWidgetItem(cell_data)
                if j == 3:  # P&L колонка
                    if cell_data.startswith('+'):
                        item.setForeground(QBrush(QColor(MinimalAppleStyle.COLORS['success'])))
                    else:
                        item.setForeground(QBrush(QColor(MinimalAppleStyle.COLORS['danger'])))
                table.setItem(i, j, item)
        
        layout.addWidget(table)
        return card

def main() -> None:
    """Главная функция"""
    app = QApplication(sys.argv)
    
    # Создание тестового окна
    window = MinimalDesignTest()
    window.show()
    
    print("🎨 Minimal Design Test Window opened!")
    print("✅ Testing minimalistic Apple-style design system...")
    print("📱 Features tested:")
    print("   - Clean color palette")
    print("   - Subtle borders and shadows")
    print("   - Minimalistic buttons")
    print("   - Clean cards and forms")
    print("   - Professional tables")
    print("   - Smooth transitions")
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 