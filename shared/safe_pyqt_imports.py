"""
Безопасная система импортов PyQt с контролем пространства имен.
"""

import sys
from typing import Any, Dict, List, Optional

# Кэш для импортированных модулей
_import_cache: Dict[str, Any] = {}


class SafeQtImporter:
    """Безопасный импортер PyQt модулей."""
    
    def __init__(self):
        self.imported_modules = {}
        self.namespace_conflicts = []
    
    def safe_import_widgets(self) -> Dict[str, Any]:
        """Безопасный импорт PyQt6.QtWidgets."""
        if 'widgets' in _import_cache:
            cached = _import_cache['widgets']
            return cached if isinstance(cached, dict) else {}
        
        try:
            from PyQt6.QtWidgets import (
                QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox,
                QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem,
                QTabWidget, QSplitter, QFrame, QScrollArea, QGroupBox,
                QCheckBox, QRadioButton, QSlider, QProgressBar, QSpinBox,
                QDialog, QDialogButtonBox, QMessageBox, QFileDialog,
                QMenuBar, QMenu, QAction, QToolBar, QStatusBar,
                QGridLayout, QFormLayout, QStackedWidget, QListWidget,
                QListWidgetItem, QSizePolicy, QSpacerItem
            )
            
            widgets = {
                'QApplication': QApplication,
                'QMainWindow': QMainWindow,
                'QWidget': QWidget,
                'QVBoxLayout': QVBoxLayout,
                'QHBoxLayout': QHBoxLayout,
                'QLabel': QLabel,
                'QPushButton': QPushButton,
                'QLineEdit': QLineEdit,
                'QTextEdit': QTextEdit,
                'QComboBox': QComboBox,
                'QTableWidget': QTableWidget,
                'QTableWidgetItem': QTableWidgetItem,
                'QTreeWidget': QTreeWidget,
                'QTreeWidgetItem': QTreeWidgetItem,
                'QTabWidget': QTabWidget,
                'QSplitter': QSplitter,
                'QFrame': QFrame,
                'QScrollArea': QScrollArea,
                'QGroupBox': QGroupBox,
                'QCheckBox': QCheckBox,
                'QRadioButton': QRadioButton,
                'QSlider': QSlider,
                'QProgressBar': QProgressBar,
                'QSpinBox': QSpinBox,
                'QDialog': QDialog,
                'QDialogButtonBox': QDialogButtonBox,
                'QMessageBox': QMessageBox,
                'QFileDialog': QFileDialog,
                'QMenuBar': QMenuBar,
                'QMenu': QMenu,
                'QAction': QAction,
                'QToolBar': QToolBar,
                'QStatusBar': QStatusBar,
                'QGridLayout': QGridLayout,
                'QFormLayout': QFormLayout,
                'QStackedWidget': QStackedWidget,
                'QListWidget': QListWidget,
                'QListWidgetItem': QListWidgetItem,
                'QSizePolicy': QSizePolicy,
                'QSpacerItem': QSpacerItem,
            }
            
            _import_cache['widgets'] = widgets
            return widgets
            
        except ImportError as e:
            print(f"Warning: Failed to import PyQt6.QtWidgets: {e}")
            return {}
    
    def safe_import_core(self) -> Dict[str, Any]:
        """Безопасный импорт PyQt6.QtCore."""
        if 'core' in _import_cache:
            cached = _import_cache['core']
            return cached if isinstance(cached, dict) else {}
        
        try:
            from PyQt6.QtCore import (
                QObject, QThread, QTimer, QEvent, QEventLoop,
                QSize, QRect, QPoint, QPointF, QSizeF, QRectF,
                Qt, QUrl, QDateTime, QDate, QTime,
                pyqtSignal, pyqtSlot, QVariant, QModelIndex,
                QAbstractItemModel, QSortFilterProxyModel,
                QSettings, QStandardPaths, QDir, QFileInfo
            )
            
            core = {
                'QObject': QObject,
                'QThread': QThread,
                'QTimer': QTimer,
                'QEvent': QEvent,
                'QEventLoop': QEventLoop,
                'QSize': QSize,
                'QRect': QRect,
                'QPoint': QPoint,
                'QPointF': QPointF,
                'QSizeF': QSizeF,
                'QRectF': QRectF,
                'Qt': Qt,
                'QUrl': QUrl,
                'QDateTime': QDateTime,
                'QDate': QDate,
                'QTime': QTime,
                'pyqtSignal': pyqtSignal,
                'pyqtSlot': pyqtSlot,
                'QVariant': QVariant,
                'QModelIndex': QModelIndex,
                'QAbstractItemModel': QAbstractItemModel,
                'QSortFilterProxyModel': QSortFilterProxyModel,
                'QSettings': QSettings,
                'QStandardPaths': QStandardPaths,
                'QDir': QDir,
                'QFileInfo': QFileInfo,
            }
            
            _import_cache['core'] = core
            return core
            
        except ImportError as e:
            print(f"Warning: Failed to import PyQt6.QtCore: {e}")
            return {}
    
    def safe_import_gui(self) -> Dict[str, Any]:
        """Безопасный импорт PyQt6.QtGui."""
        if 'gui' in _import_cache:
            cached = _import_cache['gui']
            return cached if isinstance(cached, dict) else {}
        
        try:
            from PyQt6.QtGui import (
                QFont, QFontMetrics, QColor, QPalette, QBrush, QPen,
                QPixmap, QIcon, QImage, QPainter, QPaintEvent,
                QKeyEvent, QMouseEvent, QWheelEvent, QResizeEvent,
                QAction, QShortcut, QKeySequence, QCursor,
                QLinearGradient, QRadialGradient, QConicalGradient
            )
            
            gui = {
                'QFont': QFont,
                'QFontMetrics': QFontMetrics,
                'QColor': QColor,
                'QPalette': QPalette,
                'QBrush': QBrush,
                'QPen': QPen,
                'QPixmap': QPixmap,
                'QIcon': QIcon,
                'QImage': QImage,
                'QPainter': QPainter,
                'QPaintEvent': QPaintEvent,
                'QKeyEvent': QKeyEvent,
                'QMouseEvent': QMouseEvent,
                'QWheelEvent': QWheelEvent,
                'QResizeEvent': QResizeEvent,
                'QAction': QAction,
                'QShortcut': QShortcut,
                'QKeySequence': QKeySequence,
                'QCursor': QCursor,
                'QLinearGradient': QLinearGradient,
                'QRadialGradient': QRadialGradient,
                'QConicalGradient': QConicalGradient,
            }
            
            _import_cache['gui'] = gui
            return gui
            
        except ImportError as e:
            print(f"Warning: Failed to import PyQt6.QtGui: {e}")
            return {}
    
    def safe_import_charts(self) -> Dict[str, Any]:
        """Безопасный импорт PyQt6.QtCharts."""
        if 'charts' in _import_cache:
            cached = _import_cache['charts']
            return cached if isinstance(cached, dict) else {}
        
        try:
            from PyQt6.QtCharts import (
                QChart, QChartView, QLineSeries, QAreaSeries,
                QBarSeries, QBarSet, QStackedBarSeries,
                QPieSeries, QPieSlice, QScatterSeries,
                QValueAxis, QCategoryAxis, QDateTimeAxis,
                QLegend, QAbstractSeries
            )
            
            charts = {
                'QChart': QChart,
                'QChartView': QChartView,
                'QLineSeries': QLineSeries,
                'QAreaSeries': QAreaSeries,
                'QBarSeries': QBarSeries,
                'QBarSet': QBarSet,
                'QStackedBarSeries': QStackedBarSeries,
                'QPieSeries': QPieSeries,
                'QPieSlice': QPieSlice,
                'QScatterSeries': QScatterSeries,
                'QValueAxis': QValueAxis,
                'QCategoryAxis': QCategoryAxis,
                'QDateTimeAxis': QDateTimeAxis,
                'QLegend': QLegend,
                'QAbstractSeries': QAbstractSeries,
            }
            
            _import_cache['charts'] = charts
            return charts
            
        except ImportError as e:
            print(f"Warning: Failed to import PyQt6.QtCharts: {e}")
            return {}
    
    def get_all_safe_imports(self) -> Dict[str, Any]:
        """Получение всех безопасных импортов."""
        all_imports = {}
        
        all_imports.update(self.safe_import_widgets())
        all_imports.update(self.safe_import_core())
        all_imports.update(self.safe_import_gui())
        all_imports.update(self.safe_import_charts())
        
        return all_imports
    
    def check_conflicts(self, imports: Dict[str, Any]) -> List[str]:
        """Проверка конфликтов имен."""
        conflicts = []
        
        # Проверяем конфликты с встроенными функциями Python
        builtins = dir(__builtins__)
        for name in imports.keys():
            if name in builtins:
                conflicts.append(f"Conflict with builtin: {name}")
        
        return conflicts


# Глобальный экземпляр импортера
_safe_importer = SafeQtImporter()

# Удобные функции для использования
def get_qt_widgets():
    """Получение безопасных виджетов PyQt."""
    return _safe_importer.safe_import_widgets()

def get_qt_core():
    """Получение безопасных классов PyQt Core."""
    return _safe_importer.safe_import_core()

def get_qt_gui():
    """Получение безопасных классов PyQt GUI."""
    return _safe_importer.safe_import_gui()

def get_qt_charts():
    """Получение безопасных классов PyQt Charts."""
    return _safe_importer.safe_import_charts()

def get_all_qt():
    """Получение всех безопасных классов PyQt."""
    return _safe_importer.get_all_safe_imports()


# Проверка безопасности при импорте модуля
def validate_import_safety():
    """Валидация безопасности импортов."""
    all_imports = get_all_qt()
    conflicts = _safe_importer.check_conflicts(all_imports)
    
    if conflicts:
        print("Warning: Found import conflicts:")
        for conflict in conflicts:
            print(f"  - {conflict}")
    
    return len(conflicts) == 0


# Автоматическая проверка при импорте
if __name__ != "__main__":
    validate_import_safety()