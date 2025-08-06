#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimalistic Apple-Style Design System for ATB Trading System
Минималистичная система дизайна в стиле Apple
"""

from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtCharts import *

class MinimalAppleStyle:
    """Минималистичная система стилей в стиле Apple"""
    
    # Минималистичная цветовая палитра Apple
    COLORS = {
        # Основные цвета - очень сдержанные
        'primary': '#007AFF',           # Apple Blue - единственный акцентный цвет
        'success': '#34C759',           # Apple Green - только для успеха
        'danger': '#FF3B30',            # Apple Red - только для ошибок
        'warning': '#FF9500',           # Apple Orange - только для предупреждений
        
        # Нейтральные цвета - минималистичные
        'background': '#000000',        # Pure Black
        'surface': '#1C1C1E',           # Dark Gray
        'surface_secondary': '#2C2C2E', # Lighter Dark Gray
        'surface_tertiary': '#3A3A3C',  # Even Lighter
        
        # Текст - минималистичный
        'text_primary': '#FFFFFF',      # White
        'text_secondary': '#EBEBF5',    # Light Gray
        'text_tertiary': '#EBEBF599',   # Semi-transparent
        
        # Минималистичные градиенты
        'gradient_primary': 'qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #007AFF, stop:1 #0056CC)',
        'gradient_success': 'qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #34C759, stop:1 #28A745)',
        'gradient_danger': 'qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FF3B30, stop:1 #DC3545)',
        
        # Минималистичные эффекты
        'subtle_glow': 'rgba(0, 122, 255, 0.1)',
        'subtle_border': 'rgba(255, 255, 255, 0.1)',
        'subtle_shadow': 'rgba(0, 0, 0, 0.2)',
    }
    
    @staticmethod
    def get_minimal_stylesheet() -> Any:
        """Получить минималистичную таблицу стилей"""
        return f"""
        
        /* Минималистичные кнопки */
        QPushButton {{
            background: {MinimalAppleStyle.COLORS['surface']};
            border: 1px solid {MinimalAppleStyle.COLORS['subtle_border']};
            border-radius: 8px;
            padding: 12px 20px;
            font-size: 14px;
            font-weight: 500;
            color: {MinimalAppleStyle.COLORS['text_primary']};
            min-height: 40px;
        }}
        
        QPushButton:hover {{
            background: {MinimalAppleStyle.COLORS['surface_secondary']};
            border: 1px solid {MinimalAppleStyle.COLORS['primary']};
        }}
        
        QPushButton:pressed {{
            background: {MinimalAppleStyle.COLORS['surface_tertiary']};
            transform: scale(0.98);
        }}
        
        /* Минималистичные кнопки состояний */
        QPushButton.primary {{
            background: {MinimalAppleStyle.COLORS['primary']};
            border: none;
            color: white;
        }}
        
        QPushButton.primary:hover {{
            background: #0056CC;
        }}
        
        QPushButton.success {{
            background: {MinimalAppleStyle.COLORS['success']};
            border: none;
            color: white;
        }}
        
        QPushButton.danger {{
            background: {MinimalAppleStyle.COLORS['danger']};
            border: none;
            color: white;
        }}
        
        /* Минималистичные карточки */
        .MinimalCard {{
            background: {MinimalAppleStyle.COLORS['surface']};
            border: 1px solid {MinimalAppleStyle.COLORS['subtle_border']};
            border-radius: 12px;
            padding: 20px;
            margin: 8px;
        }}
        
        .MinimalCard:hover {{
            border: 1px solid {MinimalAppleStyle.COLORS['primary']};
            box-shadow: 0 4px 12px {MinimalAppleStyle.COLORS['subtle_shadow']};
        }}
        
        /* Минималистичные панели */
        QTabWidget::pane {{
            border: none;
            background: {MinimalAppleStyle.COLORS['background']};
        }}
        
        QTabBar::tab {{
            background: transparent;
            color: {MinimalAppleStyle.COLORS['text_secondary']};
            padding: 12px 20px;
            margin-right: 2px;
            border: none;
            font-weight: 500;
            border-bottom: 2px solid transparent;
        }}
        
        QTabBar::tab:selected {{
            color: {MinimalAppleStyle.COLORS['text_primary']};
            border-bottom: 2px solid {MinimalAppleStyle.COLORS['primary']};
        }}
        
        QTabBar::tab:hover {{
            color: {MinimalAppleStyle.COLORS['text_primary']};
        }}
        
        /* Минималистичные таблицы */
        QTableWidget {{
            background: {MinimalAppleStyle.COLORS['surface']};
            border: 1px solid {MinimalAppleStyle.COLORS['subtle_border']};
            border-radius: 8px;
            gridline-color: {MinimalAppleStyle.COLORS['surface_secondary']};
            selection-background-color: {MinimalAppleStyle.COLORS['subtle_glow']};
        }}
        
        QTableWidget::item {{
            padding: 12px;
            border: none;
        }}
        
        QHeaderView::section {{
            background: {MinimalAppleStyle.COLORS['surface_secondary']};
            color: {MinimalAppleStyle.COLORS['text_primary']};
            padding: 12px;
            border: none;
            font-weight: 500;
        }}
        
        /* Минималистичные поля ввода */
        QLineEdit, QTextEdit, QComboBox {{
            background: {MinimalAppleStyle.COLORS['surface_secondary']};
            border: 1px solid {MinimalAppleStyle.COLORS['subtle_border']};
            border-radius: 6px;
            padding: 10px 12px;
            color: {MinimalAppleStyle.COLORS['text_primary']};
            font-size: 14px;
        }}
        
        QLineEdit:focus, QTextEdit:focus, QComboBox:focus {{
            border: 1px solid {MinimalAppleStyle.COLORS['primary']};
            background: {MinimalAppleStyle.COLORS['surface']};
        }}
        
        /* Минималистичные чекбоксы */
        QCheckBox {{
            spacing: 8px;
            color: {MinimalAppleStyle.COLORS['text_primary']};
        }}
        
        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border-radius: 3px;
            border: 1px solid {MinimalAppleStyle.COLORS['subtle_border']};
            background: {MinimalAppleStyle.COLORS['surface']};
        }}
        
        QCheckBox::indicator:checked {{
            background: {MinimalAppleStyle.COLORS['primary']};
            border: 1px solid {MinimalAppleStyle.COLORS['primary']};
        }}
        
        /* Минималистичные слайдеры */
        QSlider::groove:horizontal {{
            border: none;
            height: 3px;
            background: {MinimalAppleStyle.COLORS['surface_secondary']};
            border-radius: 2px;
        }}
        
        QSlider::handle:horizontal {{
            background: {MinimalAppleStyle.COLORS['primary']};
            border: none;
            width: 16px;
            height: 16px;
            border-radius: 8px;
            margin: -6px 0;
        }}
        
        /* Минималистичные прогресс-бары */
        QProgressBar {{
            border: none;
            border-radius: 4px;
            background: {MinimalAppleStyle.COLORS['surface_secondary']};
            height: 6px;
            text-align: center;
        }}
        
        QProgressBar::chunk {{
            background: {MinimalAppleStyle.COLORS['primary']};
            border-radius: 4px;
        }}
        
        /* Минималистичные меню */
        QMenuBar {{
            background: {MinimalAppleStyle.COLORS['surface']};
            border-bottom: 1px solid {MinimalAppleStyle.COLORS['subtle_border']};
            color: {MinimalAppleStyle.COLORS['text_primary']};
        }}
        
        QMenuBar::item {{
            padding: 8px 12px;
            background: transparent;
        }}
        
        QMenuBar::item:selected {{
            background: {MinimalAppleStyle.COLORS['surface_secondary']};
            border-radius: 6px;
        }}
        
        QMenu {{
            background: {MinimalAppleStyle.COLORS['surface']};
            border: 1px solid {MinimalAppleStyle.COLORS['subtle_border']};
            border-radius: 8px;
            padding: 6px;
        }}
        
        QMenu::item {{
            padding: 8px 12px;
            border-radius: 4px;
        }}
        
        QMenu::item:selected {{
            background: {MinimalAppleStyle.COLORS['surface_secondary']};
        }}
        
        /* Минималистичные скроллбары */
        QScrollBar:vertical {{
            background: transparent;
            width: 6px;
            border-radius: 3px;
        }}
        
        QScrollBar::handle:vertical {{
            background: {MinimalAppleStyle.COLORS['surface_secondary']};
            border-radius: 3px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background: {MinimalAppleStyle.COLORS['surface_tertiary']};
        }}
        
        /* Минималистичные групповые панели */
        QGroupBox {{
            font-weight: 500;
            color: {MinimalAppleStyle.COLORS['text_primary']};
            border: 1px solid {MinimalAppleStyle.COLORS['subtle_border']};
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 8px;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 8px 0 8px;
        }}
        
        /* Минималистичные разделители */
        QFrame[frameShape="4"] {{
            color: {MinimalAppleStyle.COLORS['subtle_border']};
        }}
        
        /* Минималистичные анимации */
        * {{
            transition: all 0.15s ease-out;
        }}
        """

class MinimalCard(QWidget):
    """Минималистичная карточка"""
    
    def __init__(self, title="", content_widget=None, parent=None) -> None:
        super().__init__(parent)
        self.title = title
        self.content_widget = content_widget
        self.init_ui()
    
    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        if self.title:
            title_label = QLabel(self.title)
            title_label.setStyleSheet(f"""
                font-size: 16px;
                font-weight: 500;
                color: {MinimalAppleStyle.COLORS['text_primary']};
                margin-bottom: 4px;
            """)
            layout.addWidget(title_label)
        
        if self.content_widget:
            layout.addWidget(self.content_widget)
        
        self.setProperty("class", "MinimalCard")

class MinimalButton(QPushButton):
    """Минималистичная кнопка"""
    
    def __init__(self, text="", style="default", parent=None) -> None:
        super().__init__(text, parent)
        self.style_type = style
        self.setup_style()
    
    def setup_style(self) -> None:
        self.setProperty("class", self.style_type)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

class MinimalProgressBar(QProgressBar):
    """Минималистичный прогресс-бар"""
    
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setup_style()
    
    def setup_style(self) -> None:
        self.setTextVisible(False)
        self.setFixedHeight(6)

class MinimalChart(QChartView):
    """Минималистичный график"""
    
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setup_style()
    
    def setup_style(self) -> None:
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.chart().setTheme(QChart.ChartTheme.ChartThemeDark)
        self.chart().setBackgroundBrush(QBrush(QColor(MinimalAppleStyle.COLORS['surface'])))
        self.chart().setPlotAreaBackgroundBrush(QBrush(QColor(MinimalAppleStyle.COLORS['surface'])))
        self.chart().setTitleBrush(QBrush(QColor(MinimalAppleStyle.COLORS['text_primary'])))
        self.chart().setTitleFont(QFont("SF Pro Display", 14, QFont.Weight.Medium))
        
        # Минималистичные цвета для графиков
        self.chart().setMargins(QMargins(10, 10, 10, 10))

class MinimalTable(QTableWidget):
    """Минималистичная таблица"""
    
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setup_style()
    
    def setup_style(self) -> None:
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)
        self.setShowGrid(False)
        self.setFrameShape(QFrame.Shape.NoFrame)

class MinimalComboBox(QComboBox):
    """Минималистичный комбобокс"""
    
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setup_style()
    
    def setup_style(self) -> None:
        self.setFixedHeight(40)

class MinimalSpinBox(QSpinBox):
    """Минималистичный спинбокс"""
    
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setup_style()
    
    def setup_style(self) -> None:
        self.setFixedHeight(40)
        self.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)

class MinimalCheckBox(QCheckBox):
    """Минималистичный чекбокс"""
    
    def __init__(self, text="", parent=None) -> None:
        super().__init__(text, parent)
        self.setup_style()
    
    def setup_style(self) -> None:
        self.setCursor(Qt.CursorShape.PointingHandCursor)

class MinimalLabel(QLabel):
    """Минималистичная метка"""
    
    def __init__(self, text="", style="body", parent=None) -> None:
        super().__init__(text, parent)
        self.style_type = style
        self.setup_style()
    
    def setup_style(self) -> None:
        styles = {
            "title": f"font-size: 20px; font-weight: 500; color: {MinimalAppleStyle.COLORS['text_primary']};",
            "subtitle": f"font-size: 16px; font-weight: 500; color: {MinimalAppleStyle.COLORS['text_primary']};",
            "body": f"font-size: 14px; font-weight: 400; color: {MinimalAppleStyle.COLORS['text_secondary']};",
            "caption": f"font-size: 12px; font-weight: 400; color: {MinimalAppleStyle.COLORS['text_tertiary']};",
            "success": f"font-size: 14px; font-weight: 500; color: {MinimalAppleStyle.COLORS['success']};",
            "danger": f"font-size: 14px; font-weight: 500; color: {MinimalAppleStyle.COLORS['danger']};",
            "warning": f"font-size: 14px; font-weight: 500; color: {MinimalAppleStyle.COLORS['warning']};",
        }
        self.setStyleSheet(styles.get(self.style_type, styles["body"]))

class MinimalDivider(QFrame):
    """Минималистичный разделитель"""
    
    def __init__(self, orientation="horizontal", parent=None) -> None:
        super().__init__(parent)
        self.orientation = orientation
        self.setup_style()
    
    def setup_style(self) -> None:
        if self.orientation == "horizontal":
            self.setFrameShape(QFrame.Shape.HLine)
            self.setFixedHeight(1)
        else:
            self.setFrameShape(QFrame.Shape.VLine)
            self.setFixedWidth(1)
        
        self.setStyleSheet(f"background-color: {MinimalAppleStyle.COLORS['subtle_border']}; border: none;")

class MinimalBadge(QLabel):
    """Минималистичный бейдж"""
    
    def __init__(self, text="", color="primary", parent=None) -> None:
        super().__init__(text, parent)
        self.color = color
        self.setup_style()
    
    def setup_style(self) -> None:
        colors = {
            "primary": MinimalAppleStyle.COLORS['primary'],
            "success": MinimalAppleStyle.COLORS['success'],
            "danger": MinimalAppleStyle.COLORS['danger'],
            "warning": MinimalAppleStyle.COLORS['warning'],
        }
        
        self.setStyleSheet(f"""
            background: {colors.get(self.color, colors['primary'])};
            color: white;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
        """)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter) 