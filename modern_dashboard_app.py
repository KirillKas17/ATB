import sys
import random
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QLineEdit, QTableWidget, QTableWidgetItem, 
    QFrame, QScrollArea, QGridLayout, QSpacerItem, QSizePolicy,
    QGraphicsDropShadowEffect, QComboBox, QProgressBar, QSlider, QStackedWidget
)
from PyQt6.QtGui import QIcon, QColor, QBrush, QFont, QPainter, QLinearGradient, QPalette
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty, QTranslator, QLocale, pyqtSignal
import matplotlib
matplotlib.use('Qt5Agg')  # Используем Qt5Agg backend для PyQt6
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

# Мультиязычность
TRANSLATIONS = {
    'ru': {
        'dashboard': 'Дашборд',
        'analytics': 'Аналитика',
        'trading': 'Торговля',
        'portfolio': 'Портфель',
        'settings': 'Настройки',
        'profile': 'Профиль',
        'trading_volume': 'Объем торгов',
        'daily_pnl': 'Дневная прибыль',
        'active_trades': 'Активные сделки',
        'win_rate': 'Процент выигрышей',
        'total_profit': 'Общая прибыль',
        'risk_ratio': 'Соотношение риска',
        'open_positions': 'Открытые позиции',
        'closed_trades': 'Закрытые сделки',
        'avg_trade': 'Средняя сделка',
        'max_drawdown': 'Макс. просадка',
        'profit_factor': 'Фактор прибыли',
        'sharpe_ratio': 'Коэффициент Шарпа',
        'total_trades': 'Всего сделок',
        'successful_trades': 'Успешные сделки',
        'failed_trades': 'Неудачные сделки',
        'recent_trades': 'Последние сделки',
        'symbol': 'Символ',
        'type': 'Тип',
        'entry_price': 'Цена входа',
        'exit_price': 'Цена выхода',
        'profit_loss': 'Прибыль/Убыток',
        'status': 'Статус',
        'vs_yesterday': 'vs вчера',
        'vs_last_week': 'vs прошлая неделя',
        'vs_last_month': 'vs прошлый месяц',
        'trades': 'сделок',
        'profit': 'прибыль',
        'loss': 'убыток',
        'language': 'Язык',
        'buy': 'Покупка',
        'sell': 'Продажа',
        'open': 'Открыта',
        'closed': 'Закрыта',
        'pending': 'Ожидает'
    },
    'en': {
        'dashboard': 'Dashboard',
        'analytics': 'Analytics',
        'trading': 'Trading',
        'portfolio': 'Portfolio',
        'settings': 'Settings',
        'profile': 'Profile',
        'trading_volume': 'Trading Volume',
        'daily_pnl': 'Daily P&L',
        'active_trades': 'Active Trades',
        'win_rate': 'Win Rate',
        'total_profit': 'Total Profit',
        'risk_ratio': 'Risk Ratio',
        'open_positions': 'Open Positions',
        'closed_trades': 'Closed Trades',
        'avg_trade': 'Avg Trade',
        'max_drawdown': 'Max Drawdown',
        'profit_factor': 'Profit Factor',
        'sharpe_ratio': 'Sharpe Ratio',
        'total_trades': 'Total Trades',
        'successful_trades': 'Successful Trades',
        'failed_trades': 'Failed Trades',
        'recent_trades': 'Recent Trades',
        'symbol': 'Symbol',
        'type': 'Type',
        'entry_price': 'Entry Price',
        'exit_price': 'Exit Price',
        'profit_loss': 'P&L',
        'status': 'Status',
        'vs_yesterday': 'vs yesterday',
        'vs_last_week': 'vs last week',
        'vs_last_month': 'vs last month',
        'trades': 'trades',
        'profit': 'profit',
        'loss': 'loss',
        'language': 'Language',
        'buy': 'Buy',
        'sell': 'Sell',
        'open': 'Open',
        'closed': 'Closed',
        'pending': 'Pending'
    },
    'zh': {
        'dashboard': '仪表板',
        'analytics': '分析',
        'trading': '交易',
        'portfolio': '投资组合',
        'settings': '设置',
        'profile': '个人资料',
        'trading_volume': '交易量',
        'daily_pnl': '日盈亏',
        'active_trades': '活跃交易',
        'win_rate': '胜率',
        'total_profit': '总利润',
        'risk_ratio': '风险比率',
        'open_positions': '未平仓',
        'closed_trades': '已平仓',
        'avg_trade': '平均交易',
        'max_drawdown': '最大回撤',
        'profit_factor': '利润因子',
        'sharpe_ratio': '夏普比率',
        'total_trades': '总交易',
        'successful_trades': '成功交易',
        'failed_trades': '失败交易',
        'recent_trades': '最近交易',
        'symbol': '符号',
        'type': '类型',
        'entry_price': '入场价',
        'exit_price': '出场价',
        'profit_loss': '盈亏',
        'status': '状态',
        'vs_yesterday': 'vs 昨天',
        'vs_last_week': 'vs 上周',
        'vs_last_month': 'vs 上月',
        'trades': '交易',
        'profit': '利润',
        'loss': '亏损',
        'language': '语言',
        'buy': '买入',
        'sell': '卖出',
        'open': '开放',
        'closed': '关闭',
        'pending': '待定'
    }
}

class GlassmorphismWidget(QWidget):
    """Виджет с эффектом жидкого стекла"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Создаем градиент для эффекта стекла
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(255, 255, 255, 30))
        gradient.setColorAt(1, QColor(255, 255, 255, 10))
        
        # Рисуем фон с эффектом стекла
        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 12, 12)

class CustomScrollBar(QWidget):
    """Кастомный скроллбар"""
    def __init__(self, orientation=Qt.Orientation.Vertical, parent=None):
        super().__init__(parent)
        self.orientation = orientation
        self.setFixedSize(8, 100) if orientation == Qt.Orientation.Vertical else self.setFixedSize(100, 8)
        self.setStyleSheet("""
            QWidget {
                background-color: transparent;
                border: none;
            }
        """)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Рисуем трек скроллбара
        track_color = QColor(58, 61, 90, 100)
        painter.setBrush(track_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 4, 4)
        
        # Рисуем ползунок
        thumb_color = QColor(255, 69, 96, 150)
        painter.setBrush(thumb_color)
        thumb_rect = self.rect().adjusted(1, 1, -1, -1)
        painter.drawRoundedRect(thumb_rect, 3, 3)

class CustomTitleBar(QWidget):
    """Кастомная панель заголовка окна"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setFixedHeight(50)
        self.current_language = 'ru'
        self.setup_ui()
        
    def setup_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(12)
        
        # Логотип
        logo = QLabel("ATB Trading Dashboard")
        logo.setStyleSheet("""
            QLabel {
                color: #FF4560;
                font-family: 'Segoe UI', sans-serif;
                font-size: 18px;
                font-weight: 700;
            }
        """)
        
        # Селектор языка
        self.language_combo = QComboBox()
        self.language_combo.addItems(['🇷🇺 Русский', '🇺🇸 English', '🇨🇳 中文'])
        self.language_combo.setCurrentText('🇷🇺 Русский')
        self.language_combo.currentTextChanged.connect(self.change_language)
        self.language_combo.setStyleSheet("""
            QComboBox {
                background-color: #2A2D4A;
                border: 1px solid #3A3D5A;
                border-radius: 8px;
                padding: 8px 12px;
                color: #FFFFFF;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
                min-width: 120px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #BFBFBF;
            }
            QComboBox QAbstractItemView {
                background-color: #2A2D4A;
                border: 1px solid #3A3D5A;
                border-radius: 8px;
                color: #FFFFFF;
                selection-background-color: #FF4560;
            }
        """)
        
        # Кнопки управления окном
        minimize_btn = QPushButton("−")
        minimize_btn.setFixedSize(32, 32)
        minimize_btn.clicked.connect(self.parent.showMinimized)
        
        maximize_btn = QPushButton("□")
        maximize_btn.setFixedSize(32, 32)
        maximize_btn.clicked.connect(self.toggle_maximize)
        
        close_btn = QPushButton("×")
        close_btn.setFixedSize(32, 32)
        close_btn.clicked.connect(self.parent.close)
        
        # Стили для кнопок управления окном
        window_btn_style = """
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 6px;
                color: #BFBFBF;
                font-family: 'Segoe UI', sans-serif;
                font-size: 16px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #3A3D5A;
                color: #FFFFFF;
            }
        """
        
        close_btn_style = """
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 6px;
                color: #BFBFBF;
                font-family: 'Segoe UI', sans-serif;
                font-size: 18px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #FF4560;
                color: #FFFFFF;
            }
        """
        
        minimize_btn.setStyleSheet(window_btn_style)
        maximize_btn.setStyleSheet(window_btn_style)
        close_btn.setStyleSheet(close_btn_style)
        
        # Добавление элементов в layout
        layout.addWidget(logo)
        layout.addStretch()
        layout.addWidget(self.language_combo)
        layout.addWidget(minimize_btn)
        layout.addWidget(maximize_btn)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
        
    def toggle_maximize(self):
        if self.parent.isMaximized():
            self.parent.showNormal()
        else:
            self.parent.showMaximized()
            
    def change_language(self, text):
        if 'Русский' in text:
            self.current_language = 'ru'
        elif 'English' in text:
            self.current_language = 'en'
        elif '中文' in text:
            self.current_language = 'zh'
        
        # Обновляем язык во всем приложении
        if hasattr(self.parent, 'update_language'):
            self.parent.update_language(self.current_language)

class Sidebar(QWidget):
    """Боковое меню с навигацией"""
    page_changed = pyqtSignal(str)  # Сигнал для смены страницы
    
    def __init__(self, language='ru'):
        super().__init__()
        self.setFixedWidth(260)
        self.active_item = "dashboard"
        self.language = language
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        # Элементы меню
        menu_items = [
            ("dashboard", ""),
            ("analytics", ""),
            ("trading", ""),
            ("portfolio", ""),
            ("settings", ""),
            ("profile", "")
        ]
        
        for item_id, icon in menu_items:
            btn = self.create_menu_button(icon, item_id)
            layout.addWidget(btn)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def create_menu_button(self, icon, item_id):
        btn = QPushButton(f"{TRANSLATIONS[self.language].get(item_id, item_id)}")
        btn.setFixedHeight(52)
        btn.setCheckable(True)
        btn.setChecked(item_id == self.active_item)
        btn.clicked.connect(lambda: self.set_active_item(item_id))
        btn.setObjectName(item_id)  # Для идентификации кнопки
        
        # Стили для кнопок меню
        if item_id == self.active_item:
            btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                        stop:0 #FF4560, stop:1 #FF6B7A);
                    border: none;
                    border-radius: 12px;
                    color: white;
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 15px;
                    font-weight: 500;
                    text-align: left;
                    padding-left: 20px;
                }
            """)
        else:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    border: none;
                    border-radius: 12px;
                    color: #BFBFBF;
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 15px;
                    font-weight: 500;
                    text-align: left;
                    padding-left: 20px;
                }
                QPushButton:hover {
                    background-color: #2A2D4A;
                    color: #FFFFFF;
                }
            """)
        
        return btn
    
    def set_active_item(self, item_id):
        self.active_item = item_id
        # Обновляем стили всех кнопок
        for i in range(self.layout().count() - 1):  # -1 для addStretch
            btn = self.layout().itemAt(i).widget()
            if isinstance(btn, QPushButton):
                btn.setChecked(btn.text() == TRANSLATIONS[self.language].get(item_id, item_id))
        
        # Отправляем сигнал о смене страницы
        self.page_changed.emit(item_id)
                
    def update_language(self, language):
        self.language = language
        # Обновляем текст кнопок
        menu_items = [
            ("dashboard", ""),
            ("analytics", ""),
            ("trading", ""),
            ("portfolio", ""),
            ("settings", ""),
            ("profile", "")
        ]
        
        for i, (item_id, icon) in enumerate(menu_items):
            if i < self.layout().count() - 1:
                btn = self.layout().itemAt(i).widget()
                if isinstance(btn, QPushButton):
                    btn.setText(f"{TRANSLATIONS[language].get(item_id, item_id)}")

class MetricCard(GlassmorphismWidget):
    """Карточка с метрикой с эффектом стекла"""
    def __init__(self, title, value, subtitle="", trend="", color="#FF4560", icon="📊"):
        super().__init__()
        self.title = title
        self.value = value
        self.subtitle = subtitle
        self.trend = trend
        self.color = color
        self.icon = icon
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        
        # Иконка и заголовок
        header_layout = QHBoxLayout()
        icon_label = QLabel(self.icon)
        icon_label.setStyleSheet(f"""
            QLabel {{
                color: {self.color};
                font-size: 24px;
            }}
        """)
        
        title_label = QLabel(self.title)
        title_label.setStyleSheet(f"""
            QLabel {{
                color: #BFBFBF;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
                font-weight: 400;
            }}
        """)
        
        header_layout.addWidget(icon_label)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # Значение
        value_label = QLabel(self.value)
        value_label.setStyleSheet(f"""
            QLabel {{
                color: #FFFFFF;
                font-family: 'Segoe UI', sans-serif;
                font-size: 28px;
                font-weight: 700;
            }}
        """)
        
        # Подзаголовок
        subtitle_label = QLabel(self.subtitle)
        subtitle_label.setStyleSheet(f"""
            QLabel {{
                color: #8A8A8A;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
                font-weight: 400;
            }}
        """)
        
        # Тренд
        if self.trend:
            trend_label = QLabel(self.trend)
            trend_label.setStyleSheet(f"""
                QLabel {{
                    color: {self.color};
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 12px;
                    font-weight: 600;
                }}
            """)
            layout.addWidget(trend_label)
        
        layout.addLayout(header_layout)
        layout.addWidget(value_label)
        layout.addWidget(subtitle_label)
        
        self.setLayout(layout)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: #2A2D4A;
                border-radius: 16px;
                border: 1px solid #3A3D5A;
            }}
        """)

class ProgressMetricCard(GlassmorphismWidget):
    """Карточка с прогресс-баром"""
    def __init__(self, title, value, max_value, subtitle="", color="#FF4560", icon="📊"):
        super().__init__()
        self.title = title
        self.value = value
        self.max_value = max_value
        self.subtitle = subtitle
        self.color = color
        self.icon = icon
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        
        # Иконка и заголовок
        header_layout = QHBoxLayout()
        icon_label = QLabel(self.icon)
        icon_label.setStyleSheet(f"""
            QLabel {{
                color: {self.color};
                font-size: 20px;
            }}
        """)
        
        title_label = QLabel(self.title)
        title_label.setStyleSheet(f"""
            QLabel {{
                color: #BFBFBF;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
                font-weight: 400;
            }}
        """)
        
        header_layout.addWidget(icon_label)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # Значение
        value_label = QLabel(f"{self.value:,}")
        value_label.setStyleSheet(f"""
            QLabel {{
                color: #FFFFFF;
                font-family: 'Segoe UI', sans-serif;
                font-size: 24px;
                font-weight: 700;
            }}
        """)
        
        # Прогресс-бар
        progress = QProgressBar()
        progress.setMaximum(self.max_value)
        progress.setValue(self.value)
        progress.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 8px;
                background-color: rgba(58, 61, 90, 0.5);
                height: 8px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 {self.color}, stop:1 {self.color}88);
                border-radius: 8px;
            }}
        """)
        
        # Подзаголовок
        subtitle_label = QLabel(self.subtitle)
        subtitle_label.setStyleSheet(f"""
            QLabel {{
                color: #8A8A8A;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
                font-weight: 400;
            }}
        """)
        
        layout.addLayout(header_layout)
        layout.addWidget(value_label)
        layout.addWidget(progress)
        layout.addWidget(subtitle_label)
        
        self.setLayout(layout)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: #2A2D4A;
                border-radius: 16px;
                border: 1px solid #3A3D5A;
            }}
        """)

class GraphWidget(GlassmorphismWidget):
    """Виджет с графиком"""
    def __init__(self, title="Visitors Today", language='ru'):
        super().__init__()
        self.title = title
        self.language = language
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        # Заголовок
        title_label = QLabel(self.title)
        title_label.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-family: 'Segoe UI', sans-serif;
                font-size: 18px;
                font-weight: 600;
            }
        """)
        
        # График
        self.figure = Figure(figsize=(8, 4), facecolor='#2A2D4A')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: transparent; border: none;")
        
        self.create_graph()
        
        layout.addWidget(title_label)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(42, 45, 74, 0.8);
                border-radius: 16px;
                border: 1px solid rgba(58, 61, 90, 0.5);
            }
        """)
        
    def create_graph(self):
        ax = self.figure.add_subplot(111)
        
        # Данные для графика
        x = np.linspace(0, 24, 25)
        y = 1000 + 200 * np.sin(x * np.pi / 12) + np.random.normal(0, 50, 25)
        
        # Настройка графика
        ax.plot(x, y, color='#FF4560', linewidth=3, marker='o', markersize=6)
        ax.fill_between(x, y, alpha=0.3, color='#FF4560')
        
        ax.set_facecolor('#2A2D4A')
        ax.grid(True, alpha=0.2, color='#3A3D5A')
        ax.tick_params(colors='#BFBFBF')
        
        # Настройка осей
        ax.set_xlabel('Hour', color='#BFBFBF', fontsize=10)
        ax.set_ylabel('Visitors', color='#BFBFBF', fontsize=10)
        
        self.figure.tight_layout()
        self.canvas.draw()

class DataTable(GlassmorphismWidget):
    """Таблица с данными"""
    def __init__(self, language='ru'):
        super().__init__()
        self.language = language
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        # Заголовок
        title_label = QLabel(TRANSLATIONS[self.language].get('recent_trades', 'Recent Trades'))
        title_label.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-family: 'Segoe UI', sans-serif;
                font-size: 18px;
                font-weight: 600;
            }
        """)
        
        # Таблица
        self.table = QTableWidget(8, 5)
        self.table.setHorizontalHeaderLabels([
            TRANSLATIONS[self.language].get('symbol', 'Symbol'),
            TRANSLATIONS[self.language].get('type', 'Type'),
            TRANSLATIONS[self.language].get('entry_price', 'Entry Price'),
            TRANSLATIONS[self.language].get('exit_price', 'Exit Price'),
            TRANSLATIONS[self.language].get('profit_loss', 'P&L'),
            TRANSLATIONS[self.language].get('status', 'Status'),
            TRANSLATIONS[self.language].get('date', 'Date')
        ])
        
        # Данные таблицы
        data = [
            ["BTC/USD", "Buy", "$45,000", "$46,500", "$1,500", "Closed"],
            ["ETH/USD", "Sell", "$3,200", "$3,100", "$100", "Closed"],
            ["XRP/USD", "Buy", "$0.80", "$0.85", "$0.05", "Open"],
            ["ADA/USD", "Sell", "$0.50", "$0.45", "$0.05", "Closed"],
            ["DOT/USD", "Buy", "$20", "$22", "$2", "Open"],
            ["BTC/USD", "Sell", "$46,000", "$45,500", "$500", "Closed"],
            ["ETH/USD", "Buy", "$3,150", "$3,200", "$50", "Open"],
            ["XRP/USD", "Sell", "$0.82", "$0.80", "$0.02", "Closed"]
        ]
        
        for i, row in enumerate(data):
            for j, cell in enumerate(row):
                item = QTableWidgetItem(cell)
                self.table.setItem(i, j, item)
        
        # Стилизация таблицы
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #2A2D4A;
                border: 1px solid #3A3D5A;
                border-radius: 12px;
                gridline-color: #3A3D5A;
                color: #BFBFBF;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
            }
            QTableWidget::item {
                padding: 12px;
                border-bottom: 1px solid #3A3D5A;
            }
            QTableWidget::item:selected {
                background-color: #FF4560;
                color: white;
            }
            QHeaderView::section {
                background-color: #1E213A;
                color: #FFFFFF;
                padding: 16px 12px;
                border: none;
                border-bottom: 1px solid #3A3D5A;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
                font-weight: 600;
            }
        """)
        
        layout.addWidget(title_label)
        layout.addWidget(self.table)
        
        self.setLayout(layout)
        self.setStyleSheet("""
            QWidget {
                background-color: #2A2D4A;
                border-radius: 16px;
                border: 1px solid #3A3D5A;
            }
        """)

class Dashboard(QWidget):
    """Основной дашборд"""
    def __init__(self, language='ru'):
        super().__init__()
        self.language = language
        self.setup_ui()
        
    def setup_ui(self):
        # Создаем скроллируемую область
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Кастомный скроллбар
        scroll_area.verticalScrollBar().setStyleSheet("""
            QScrollBar:vertical {
                background-color: transparent;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #FF4560, stop:1 #FF6B7A);
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #FF3350, stop:1 #FF5A6A);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)
        
        # Основной контент
        content_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(24)
        
        # Первый ряд метрик
        metrics_row1 = QHBoxLayout()
        metrics_row1.setSpacing(20)
        
        metric_cards1 = [
            MetricCard(
                TRANSLATIONS[self.language].get('trading_volume', 'Trading Volume'),
                "$123,456",
                TRANSLATIONS[self.language].get('vs_yesterday', 'vs yesterday'),
                "+15.2%",
                "#FF4560",
                ""
            ),
            MetricCard(
                TRANSLATIONS[self.language].get('daily_pnl', 'Daily P&L'),
                "$1,234",
                TRANSLATIONS[self.language].get('vs_last_month', 'vs last month'),
                "+20.5%",
                "#00D4AA",
                ""
            ),
            MetricCard(
                TRANSLATIONS[self.language].get('active_trades', 'Active Trades'),
                "12",
                TRANSLATIONS[self.language].get('trades', 'trades'),
                "+5",
                "#775DD0",
                ""
            ),
            MetricCard(
                TRANSLATIONS[self.language].get('win_rate', 'Win Rate'),
                "65%",
                "vs target 60%",
                "+5%",
                "#FF6B7A",
                ""
            )
        ]
        
        for card in metric_cards1:
            metrics_row1.addWidget(card)
        
        # Второй ряд метрик
        metrics_row2 = QHBoxLayout()
        metrics_row2.setSpacing(20)
        
        metric_cards2 = [
            ProgressMetricCard(
                TRANSLATIONS[self.language].get('total_profit', 'Total Profit'),
                12345,
                20000,
                f"{61.7}% of target",
                "#FF4560",
                ""
            ),
            ProgressMetricCard(
                TRANSLATIONS[self.language].get('risk_ratio', 'Risk Ratio'),
                150,
                200,
                f"{75.0}% of target",
                "#00D4AA",
                ""
            ),
            ProgressMetricCard(
                TRANSLATIONS[self.language].get('open_positions', 'Open Positions'),
                8,
                15,
                f"{53.3}% of target",
                "#775DD0",
                ""
            ),
            MetricCard(
                TRANSLATIONS[self.language].get('avg_trade', 'Avg Trade'),
                "$1,234",
                TRANSLATIONS[self.language].get('vs_last_week', 'vs last week'),
                "+10%",
                "#FF6B7A",
                ""
            )
        ]
        
        for card in metric_cards2:
            metrics_row2.addWidget(card)
        
        # Третий ряд метрик
        metrics_row3 = QHBoxLayout()
        metrics_row3.setSpacing(20)
        
        metric_cards3 = [
            MetricCard(
                TRANSLATIONS[self.language].get('max_drawdown', 'Max Drawdown'),
                "15%",
                "vs target 10%",
                "-5%",
                "#00D4AA",
                ""
            ),
            MetricCard(
                TRANSLATIONS[self.language].get('profit_factor', 'Profit Factor'),
                "1.8",
                "vs target 1.5",
                "+0.3",
                "#775DD0",
                ""
            ),
            MetricCard(
                TRANSLATIONS[self.language].get('sharpe_ratio', 'Sharpe Ratio'),
                "0.8",
                "vs target 0.7",
                "+0.1",
                "#FF6B7A",
                ""
            ),
            MetricCard(
                TRANSLATIONS[self.language].get('total_trades', 'Total Trades'),
                "123",
                TRANSLATIONS[self.language].get('trades', 'trades'),
                "+10",
                "#FF4560",
                ""
            )
        ]
        
        for card in metric_cards3:
            metrics_row3.addWidget(card)
        
        # График и таблица
        content_layout = QHBoxLayout()
        content_layout.setSpacing(24)
        
        # График
        graph_widget = GraphWidget(TRANSLATIONS[self.language].get('trading_volume', 'Trading Volume'), self.language)
        content_layout.addWidget(graph_widget, 2)
        
        # Таблица
        table_widget = DataTable(self.language)
        content_layout.addWidget(table_widget, 1)
        
        layout.addLayout(metrics_row1)
        layout.addLayout(metrics_row2)
        layout.addLayout(metrics_row3)
        layout.addLayout(content_layout)
        
        content_widget.setLayout(layout)
        scroll_area.setWidget(content_widget)
        
        # Главный layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)


class AnalyticsPage(QWidget):
    """Страница аналитики"""
    def __init__(self, language='ru'):
        super().__init__()
        self.language = language
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(24)
        
        # Заголовок
        title = QLabel("Аналитика торговли")
        title.setStyleSheet("""
            QLabel {
                font-family: 'Segoe UI', sans-serif;
                font-size: 28px;
                font-weight: 600;
                color: #FFFFFF;
                margin-bottom: 20px;
            }
        """)
        layout.addWidget(title)
        
        # Контент аналитики
        content = QLabel("Детальная аналитика торговых операций, графики производительности, анализ рисков и другие метрики будут отображаться здесь.")
        content.setStyleSheet("""
            QLabel {
                font-family: 'Segoe UI', sans-serif;
                font-size: 16px;
                color: #BFBFBF;
                line-height: 1.6;
            }
        """)
        content.setWordWrap(True)
        layout.addWidget(content)
        
        layout.addStretch()
        self.setLayout(layout)


class TradingPage(QWidget):
    """Страница торговли"""
    def __init__(self, language='ru'):
        super().__init__()
        self.language = language
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(24)
        
        # Заголовок
        title = QLabel("Торговля")
        title.setStyleSheet("""
            QLabel {
                font-family: 'Segoe UI', sans-serif;
                font-size: 28px;
                font-weight: 600;
                color: #FFFFFF;
                margin-bottom: 20px;
            }
        """)
        layout.addWidget(title)
        
        # Контент торговли
        content = QLabel("Интерфейс для совершения торговых операций, управление позициями, установка стоп-лоссов и тейк-профитов.")
        content.setStyleSheet("""
            QLabel {
                font-family: 'Segoe UI', sans-serif;
                font-size: 16px;
                color: #BFBFBF;
                line-height: 1.6;
            }
        """)
        content.setWordWrap(True)
        layout.addWidget(content)
        
        layout.addStretch()
        self.setLayout(layout)


class PortfolioPage(QWidget):
    """Страница портфеля"""
    def __init__(self, language='ru'):
        super().__init__()
        self.language = language
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(24)
        
        # Заголовок
        title = QLabel("Портфель")
        title.setStyleSheet("""
            QLabel {
                font-family: 'Segoe UI', sans-serif;
                font-size: 28px;
                font-weight: 600;
                color: #FFFFFF;
                margin-bottom: 20px;
            }
        """)
        layout.addWidget(title)
        
        # Контент портфеля
        content = QLabel("Обзор портфеля, распределение активов, история операций и текущие позиции.")
        content.setStyleSheet("""
            QLabel {
                font-family: 'Segoe UI', sans-serif;
                font-size: 16px;
                color: #BFBFBF;
                line-height: 1.6;
            }
        """)
        content.setWordWrap(True)
        layout.addWidget(content)
        
        layout.addStretch()
        self.setLayout(layout)


class SettingsPage(QWidget):
    """Страница настроек"""
    def __init__(self, language='ru'):
        super().__init__()
        self.language = language
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(24)
        
        # Заголовок
        title = QLabel("Настройки")
        title.setStyleSheet("""
            QLabel {
                font-family: 'Segoe UI', sans-serif;
                font-size: 28px;
                font-weight: 600;
                color: #FFFFFF;
                margin-bottom: 20px;
            }
        """)
        layout.addWidget(title)
        
        # Контент настроек
        content = QLabel("Настройки системы, параметры торговли, уведомления и другие конфигурации.")
        content.setStyleSheet("""
            QLabel {
                font-family: 'Segoe UI', sans-serif;
                font-size: 16px;
                color: #BFBFBF;
                line-height: 1.6;
            }
        """)
        content.setWordWrap(True)
        layout.addWidget(content)
        
        layout.addStretch()
        self.setLayout(layout)


class ProfilePage(QWidget):
    """Страница профиля"""
    def __init__(self, language='ru'):
        super().__init__()
        self.language = language
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(24)
        
        # Заголовок
        title = QLabel("Профиль")
        title.setStyleSheet("""
            QLabel {
                font-family: 'Segoe UI', sans-serif;
                font-size: 28px;
                font-weight: 600;
                color: #FFFFFF;
                margin-bottom: 20px;
            }
        """)
        layout.addWidget(title)
        
        # Контент профиля
        content = QLabel("Информация о пользователе, статистика торговли, достижения и личные данные.")
        content.setStyleSheet("""
            QLabel {
                font-family: 'Segoe UI', sans-serif;
                font-size: 16px;
                color: #BFBFBF;
                line-height: 1.6;
            }
        """)
        content.setWordWrap(True)
        layout.addWidget(content)
        
        layout.addStretch()
        self.setLayout(layout)

class MainWindow(QMainWindow):
    """Главное окно приложения"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ATB Trading Dashboard")
        self.setGeometry(100, 100, 1600, 1000)
        self.setMinimumSize(1400, 900)
        self.current_language = 'ru'
        
        # Убираем стандартную рамку окна
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        
        # Устанавливаем стили
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #1E213A, stop:1 #2A2D4A);
                color: #BFBFBF;
                font-family: 'Segoe UI', sans-serif;
            }
        """)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Главный layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Кастомная панель заголовка
        self.title_bar = CustomTitleBar(self)
        main_layout.addWidget(self.title_bar)
        
        # Контент
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Боковое меню
        self.sidebar = Sidebar(self.current_language)
        self.sidebar.page_changed.connect(self.switch_page)
        content_layout.addWidget(self.sidebar)
        
        # Стек виджетов для страниц
        self.page_stack = QStackedWidget()
        
        # Создаем страницы
        self.pages = {
            'dashboard': Dashboard(self.current_language),
            'analytics': AnalyticsPage(self.current_language),
            'trading': TradingPage(self.current_language),
            'portfolio': PortfolioPage(self.current_language),
            'settings': SettingsPage(self.current_language),
            'profile': ProfilePage(self.current_language)
        }
        
        # Добавляем все страницы в стек
        for page in self.pages.values():
            self.page_stack.addWidget(page)
        
        # Устанавливаем дашборд как текущую страницу
        self.page_stack.setCurrentWidget(self.pages['dashboard'])
        self.current_page = 'dashboard'
        
        content_layout.addWidget(self.page_stack)
        
        main_layout.addLayout(content_layout)
        central_widget.setLayout(main_layout)
        
    def switch_page(self, page_name):
        """Переключение между страницами"""
        if page_name in self.pages:
            self.page_stack.setCurrentWidget(self.pages[page_name])
            self.current_page = page_name
    
    def update_language(self, language):
        self.current_language = language
        # Обновляем язык в сайдбаре
        self.sidebar.update_language(language)
        
        # Пересоздаем все страницы с новым языком
        old_pages = self.pages
        self.pages = {
            'dashboard': Dashboard(language),
            'analytics': AnalyticsPage(language),
            'trading': TradingPage(language),
            'portfolio': PortfolioPage(language),
            'settings': SettingsPage(language),
            'profile': ProfilePage(language)
        }
        
        # Очищаем стек и добавляем новые страницы
        self.page_stack.clear()
        for page in self.pages.values():
            self.page_stack.addWidget(page)
        
        # Устанавливаем текущую страницу
        if self.current_page in self.pages:
            self.page_stack.setCurrentWidget(self.pages[self.current_page])
        
        # Удаляем старые страницы
        for old_page in old_pages.values():
            old_page.deleteLater()
        
    def mousePressEvent(self, event):
        """Обработка нажатия мыши для перетаскивания окна"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            
    def mouseMoveEvent(self, event):
        """Обработка движения мыши для перетаскивания окна"""
        if event.buttons() == Qt.MouseButton.LeftButton and hasattr(self, 'drag_position'):
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Устанавливаем глобальные стили
    app.setStyleSheet("""
        QApplication {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                stop:0 #1E213A, stop:1 #2A2D4A);
        }
    """)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec()) 