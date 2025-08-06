#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for modern Apple-style design system
"""

import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget
from PyQt6.QtCore import Qt
from modern_style_system import *

class ModernDesignTest(QMainWindow):
    """Тестовое окно для проверки современного дизайна"""
    
    def __init__(self) -> None:
        super().__init__()
        self.init_ui()
    
    def init_ui(self) -> None:
        """Инициализация интерфейса"""
        self.setWindowTitle("🎨 Modern Design Test - ATB Trading System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Применение современного стиля
        self.setStyleSheet(ModernStyleSystem.get_modern_stylesheet())
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основной layout
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)
        
        # Заголовок
        title = ModernLabel("🎨 Modern Apple-Style Design System", "title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        subtitle = ModernLabel("Современная система дизайна в стиле Apple для ATB Trading System", "body")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)
        
        # Разделитель
        divider = ModernDivider()
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
        card = GlassCard("🎯 Modern Buttons")
        layout = QHBoxLayout(card)
        layout.setSpacing(16)
        
        # Кнопки разных стилей
        primary_btn = ModernButton("Primary Button", style="primary")
        layout.addWidget(primary_btn)
        
        success_btn = ModernButton("Success Button", style="success")
        layout.addWidget(success_btn)
        
        danger_btn = ModernButton("Danger Button", style="danger")
        layout.addWidget(danger_btn)
        
        warning_btn = ModernButton("Warning Button", style="warning")
        layout.addWidget(warning_btn)
        
        layout.addStretch()
        return card
    
    def create_cards_section(self) -> None:
        """Создание секции с карточками"""
        card = GlassCard("📱 Modern Cards")
        layout = QHBoxLayout(card)
        layout.setSpacing(16)
        
        # Обычная карточка
        normal_card = ModernCard("📊 Statistics Card")
        normal_layout = QVBoxLayout(normal_card)
        normal_layout.addWidget(ModernLabel("Total Trades: 1,234", "body"))
        normal_layout.addWidget(ModernLabel("Win Rate: 67.8%", "success"))
        normal_layout.addWidget(ModernLabel("P&L: +$12,345", "success"))
        layout.addWidget(normal_card)
        
        # Карточка с эффектом стекла
        glass_card = GlassCard("💎 Glass Effect Card")
        glass_layout = QVBoxLayout(glass_card)
        glass_layout.addWidget(ModernLabel("Active Positions: 5", "body"))
        glass_layout.addWidget(ModernLabel("Daily Volume: $45,678", "info"))
        glass_layout.addWidget(ModernLabel("Risk Level: Medium", "warning"))
        layout.addWidget(glass_card)
        
        layout.addStretch()
        return card
    
    def create_forms_section(self) -> None:
        """Создание секции с формами"""
        card = GlassCard("📝 Modern Forms")
        layout = QVBoxLayout(card)
        layout.setSpacing(16)
        
        # Строка с элементами формы
        form_row = QHBoxLayout()
        
        # Комбобокс
        combo_label = ModernLabel("Exchange:", "body")
        form_row.addWidget(combo_label)
        
        combo = ModernComboBox()
        combo.addItems(["Binance", "Bybit", "OKX", "KuCoin"])
        form_row.addWidget(combo)
        
        # Спинбокс
        spin_label = ModernLabel("Position Size (%):", "body")
        form_row.addWidget(spin_label)
        
        spin = ModernSpinBox()
        spin.setRange(1, 100)
        spin.setValue(10)
        form_row.addWidget(spin)
        
        # Чекбокс
        checkbox = ModernCheckBox("Use Testnet")
        checkbox.setChecked(True)
        form_row.addWidget(checkbox)
        
        form_row.addStretch()
        layout.addLayout(form_row)
        
        # Прогресс-бар
        progress_label = ModernLabel("System Load:", "body")
        layout.addWidget(progress_label)
        
        progress = ModernProgressBar()
        progress.setValue(75)
        layout.addWidget(progress)
        
        return card
    
    def create_tables_section(self) -> None:
        """Создание секции с таблицами"""
        card = GlassCard("📊 Modern Tables")
        layout = QVBoxLayout(card)
        
        # Таблица
        table = ModernTable()
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
                        item.setForeground(QBrush(QColor(ModernStyleSystem.COLORS['success'])))
                    else:
                        item.setForeground(QBrush(QColor(ModernStyleSystem.COLORS['danger'])))
                table.setItem(i, j, item)
        
        layout.addWidget(table)
        return card

def main() -> None:
    """Главная функция"""
    app = QApplication(sys.argv)
    
    # Создание тестового окна
    window = ModernDesignTest()
    window.show()
    
    print("🎨 Modern Design Test Window opened!")
    print("✅ Testing Apple-style design system...")
    print("📱 Features tested:")
    print("   - Modern color palette")
    print("   - Glassmorphism effects")
    print("   - Apple-style buttons")
    print("   - Modern cards and forms")
    print("   - Professional tables")
    print("   - Smooth animations")
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 