#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra Modern Dark Design System for ATB Desktop Application
Ультрасовременная система дизайна в темном стиле для ATB
"""

from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtCharts import *
import math

class UltraModernStyleSystem:
    """Ультрасовременная система стилей в темном стиле"""
    
    # Расширенная цветовая палитра
    COLORS = {
        # Основные цвета
        'primary': '#6366F1',           # Modern Indigo
        'primary_dark': '#4F46E5',      # Darker Indigo
        'primary_light': '#818CF8',     # Lighter Indigo
        'secondary': '#8B5CF6',         # Modern Purple
        'accent': '#06B6D4',            # Modern Cyan
        'success': '#10B981',           # Modern Emerald
        'warning': '#F59E0B',           # Modern Amber
        'danger': '#EF4444',            # Modern Red
        'info': '#3B82F6',              # Modern Blue
        
        # Темная палитра
        'background': '#0A0A0A',        # Pure Black
        'surface': '#111111',           # Dark Surface
        'surface_secondary': '#1A1A1A', # Secondary Surface
        'surface_tertiary': '#262626',  # Tertiary Surface
        'surface_elevated': '#1F1F1F',  # Elevated Surface
        'surface_overlay': '#2A2A2A',   # Overlay Surface
        
        # Границы и разделители
        'border': '#2A2A2A',            # Border Color
        'border_secondary': '#404040',  # Secondary Border
        'divider': '#333333',           # Divider Color
        
        # Текст
        'text_primary': '#FFFFFF',      # Primary Text
        'text_secondary': '#E5E5E5',    # Secondary Text
        'text_tertiary': '#A3A3A3',     # Tertiary Text
        'text_disabled': '#737373',     # Disabled Text
        'text_inverse': '#000000',      # Inverse Text
        
        # Градиенты
        'gradient_primary': 'qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #6366F1, stop:1 #8B5CF6)',
        'gradient_secondary': 'qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #8B5CF6, stop:1 #EC4899)',
        'gradient_success': 'qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #10B981, stop:1 #059669)',
        'gradient_danger': 'qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #EF4444, stop:1 #DC2626)',
        'gradient_warning': 'qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #F59E0B, stop:1 #D97706)',
        'gradient_info': 'qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3B82F6, stop:1 #2563EB)',
        'gradient_accent': 'qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #06B6D4, stop:1 #0891B2)',
        
        # Glassmorphism эффекты
        'glass_bg': 'rgba(255, 255, 255, 0.05)',
        'glass_border': 'rgba(255, 255, 255, 0.1)',
        'glass_shadow': 'rgba(0, 0, 0, 0.4)',
        'glass_blur': 'blur(20px)',
        
        # Тени
        'shadow_small': '0 1px 3px rgba(0, 0, 0, 0.5)',
        'shadow_medium': '0 4px 6px rgba(0, 0, 0, 0.6)',
        'shadow_large': '0 10px 25px rgba(0, 0, 0, 0.7)',
        'shadow_xlarge': '0 20px 40px rgba(0, 0, 0, 0.8)',
        
        # Состояния
        'hover_bg': 'rgba(255, 255, 255, 0.05)',
        'active_bg': 'rgba(255, 255, 255, 0.1)',
        'focus_ring': 'rgba(99, 102, 241, 0.3)',
    }
    
    # Типографика
    FONTS = {
        'font_family': "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif",
        'font_size_small': '12px',
        'font_size_base': '14px',
        'font_size_large': '16px',
        'font_size_xlarge': '18px',
        'font_size_2xl': '20px',
        'font_size_3xl': '24px',
        'font_size_4xl': '32px',
        'font_weight_light': '300',
        'font_weight_normal': '400',
        'font_weight_medium': '500',
        'font_weight_semibold': '600',
        'font_weight_bold': '700',
        'line_height_tight': '1.25',
        'line_height_normal': '1.5',
        'line_height_relaxed': '1.75',
    }
    
    # Размеры и отступы
    SPACING = {
        'xs': '4px',
        'sm': '8px',
        'md': '12px',
        'lg': '16px',
        'xl': '20px',
        '2xl': '24px',
        '3xl': '32px',
        '4xl': '40px',
        '5xl': '48px',
    }
    
    # Радиусы скругления
    RADIUS = {
        'none': '0px',
        'sm': '4px',
        'md': '8px',
        'lg': '12px',
        'xl': '16px',
        '2xl': '20px',
        '3xl': '24px',
        'full': '9999px',
    }
    
    @staticmethod
    def get_ultra_modern_stylesheet():
        """Получить ультрасовременную таблицу стилей"""
        return f"""
        /* Основные стили приложения */
        QMainWindow {{
            background: {UltraModernStyleSystem.COLORS['background']};
            color: {UltraModernStyleSystem.COLORS['text_primary']};
            font-family: {UltraModernStyleSystem.FONTS['font_family']};
            font-size: {UltraModernStyleSystem.FONTS['font_size_base']};
            font-weight: {UltraModernStyleSystem.FONTS['font_weight_normal']};
        }}
        
        QWidget {{
            background: transparent;
            color: {UltraModernStyleSystem.COLORS['text_primary']};
            font-family: {UltraModernStyleSystem.FONTS['font_family']};
        }}
        
        /* Ультрасовременные кнопки */
        QPushButton {{
            background: {UltraModernStyleSystem.COLORS['gradient_primary']};
            border: none;
            border-radius: {UltraModernStyleSystem.RADIUS['lg']};
            padding: {UltraModernStyleSystem.SPACING['lg']} {UltraModernStyleSystem.SPACING['2xl']};
            font-size: {UltraModernStyleSystem.FONTS['font_size_base']};
            font-weight: {UltraModernStyleSystem.FONTS['font_weight_semibold']};
            color: {UltraModernStyleSystem.COLORS['text_inverse']};
            min-height: 48px;
            box-shadow: {UltraModernStyleSystem.COLORS['shadow_medium']};
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        QPushButton:hover {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4F46E5, stop:1 #7C3AED);
            /* transform: translateY(-2px); */
            box-shadow: {UltraModernStyleSystem.COLORS['shadow_large']};
        }}
        
        QPushButton:pressed {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4338CA, stop:1 #6D28D9);
            transform: translateY(0px);
            box-shadow: {UltraModernStyleSystem.COLORS['shadow_small']};
        }}
        
        QPushButton:disabled {{
            background: {UltraModernStyleSystem.COLORS['surface_tertiary']};
            color: {UltraModernStyleSystem.COLORS['text_disabled']};
            box-shadow: none;
        }}
        
        /* Кнопки состояний */
        QPushButton.success {{
            background: {UltraModernStyleSystem.COLORS['gradient_success']};
        }}
        
        QPushButton.danger {{
            background: {UltraModernStyleSystem.COLORS['gradient_danger']};
        }}
        
        QPushButton.warning {{
            background: {UltraModernStyleSystem.COLORS['gradient_warning']};
        }}
        
        QPushButton.info {{
            background: {UltraModernStyleSystem.COLORS['gradient_info']};
        }}
        
        QPushButton.accent {{
            background: {UltraModernStyleSystem.COLORS['gradient_accent']};
        }}
        
        /* Вторичные кнопки */
        QPushButton.secondary {{
            background: {UltraModernStyleSystem.COLORS['surface_secondary']};
            border: 1px solid {UltraModernStyleSystem.COLORS['border']};
            color: {UltraModernStyleSystem.COLORS['text_primary']};
        }}
        
        QPushButton.secondary:hover {{
            background: {UltraModernStyleSystem.COLORS['surface_tertiary']};
            border-color: {UltraModernStyleSystem.COLORS['border_secondary']};
        }}
        
        /* Ультрасовременные карточки */
        .UltraModernCard {{
            background: {UltraModernStyleSystem.COLORS['surface']};
            border: 1px solid {UltraModernStyleSystem.COLORS['border']};
            border-radius: {UltraModernStyleSystem.RADIUS['xl']};
            padding: {UltraModernStyleSystem.SPACING['2xl']};
            margin: {UltraModernStyleSystem.SPACING['md']};
            box-shadow: {UltraModernStyleSystem.COLORS['shadow_medium']};
        }}
        
        .GlassCard {{
            background: {UltraModernStyleSystem.COLORS['glass_bg']};
            border: 1px solid {UltraModernStyleSystem.COLORS['glass_border']};
            border-radius: {UltraModernStyleSystem.RADIUS['2xl']};
            backdrop-filter: {UltraModernStyleSystem.COLORS['glass_blur']};
            box-shadow: {UltraModernStyleSystem.COLORS['shadow_large']};
            padding: {UltraModernStyleSystem.SPACING['3xl']};
        }}
        
        .ElevatedCard {{
            background: {UltraModernStyleSystem.COLORS['surface_elevated']};
            border: 1px solid {UltraModernStyleSystem.COLORS['border']};
            border-radius: {UltraModernStyleSystem.RADIUS['xl']};
            box-shadow: {UltraModernStyleSystem.COLORS['shadow_large']};
            padding: {UltraModernStyleSystem.SPACING['2xl']};
        }}
        
        /* Ультрасовременные панели */
        QTabWidget::pane {{
            border: none;
            background: {UltraModernStyleSystem.COLORS['background']};
        }}
        
        QTabBar::tab {{
            background: {UltraModernStyleSystem.COLORS['surface_secondary']};
            color: {UltraModernStyleSystem.COLORS['text_secondary']};
            padding: {UltraModernStyleSystem.SPACING['lg']} {UltraModernStyleSystem.SPACING['2xl']};
            margin-right: {UltraModernStyleSystem.SPACING['sm']};
            border-radius: {UltraModernStyleSystem.RADIUS['lg']} {UltraModernStyleSystem.RADIUS['lg']} 0 0;
            font-weight: {UltraModernStyleSystem.FONTS['font_weight_medium']};
            font-size: {UltraModernStyleSystem.FONTS['font_size_base']};
            border: 1px solid {UltraModernStyleSystem.COLORS['border']};
            border-bottom: none;
        }}
        
        QTabBar::tab:selected {{
            background: {UltraModernStyleSystem.COLORS['gradient_primary']};
            color: {UltraModernStyleSystem.COLORS['text_inverse']};
            border-color: {UltraModernStyleSystem.COLORS['primary']};
        }}
        
        QTabBar::tab:hover:!selected {{
            background: {UltraModernStyleSystem.COLORS['surface_tertiary']};
            color: {UltraModernStyleSystem.COLORS['text_primary']};
        }}
        
        /* Ультрасовременные таблицы */
        QTableWidget {{
            background: {UltraModernStyleSystem.COLORS['surface']};
            border: 1px solid {UltraModernStyleSystem.COLORS['border']};
            border-radius: {UltraModernStyleSystem.RADIUS['lg']};
            gridline-color: {UltraModernStyleSystem.COLORS['divider']};
            selection-background-color: {UltraModernStyleSystem.COLORS['primary']};
            alternate-background-color: {UltraModernStyleSystem.COLORS['surface_secondary']};
        }}
        
        QTableWidget::item {{
            padding: {UltraModernStyleSystem.SPACING['lg']};
            border: none;
            font-size: {UltraModernStyleSystem.FONTS['font_size_base']};
        }}
        
        QTableWidget::item:selected {{
            background: {UltraModernStyleSystem.COLORS['primary']};
            color: {UltraModernStyleSystem.COLORS['text_inverse']};
        }}
        
        QHeaderView::section {{
            background: {UltraModernStyleSystem.COLORS['surface_secondary']};
            color: {UltraModernStyleSystem.COLORS['text_primary']};
            padding: {UltraModernStyleSystem.SPACING['lg']};
            border: none;
            border-bottom: 1px solid {UltraModernStyleSystem.COLORS['divider']};
            font-weight: {UltraModernStyleSystem.FONTS['font_weight_semibold']};
            font-size: {UltraModernStyleSystem.FONTS['font_size_base']};
        }}
        
        /* Ультрасовременные поля ввода */
        QLineEdit, QTextEdit, QComboBox {{
            background: {UltraModernStyleSystem.COLORS['surface_secondary']};
            border: 1px solid {UltraModernStyleSystem.COLORS['border']};
            border-radius: {UltraModernStyleSystem.RADIUS['md']};
            padding: {UltraModernStyleSystem.SPACING['lg']};
            color: {UltraModernStyleSystem.COLORS['text_primary']};
            font-size: {UltraModernStyleSystem.FONTS['font_size_base']};
            font-weight: {UltraModernStyleSystem.FONTS['font_weight_normal']};
        }}
        
        QLineEdit:focus, QTextEdit:focus, QComboBox:focus {{
            border: 2px solid {UltraModernStyleSystem.COLORS['primary']};
            background: {UltraModernStyleSystem.COLORS['surface']};
            box-shadow: 0 0 0 3px {UltraModernStyleSystem.COLORS['focus_ring']};
        }}
        
        QLineEdit:hover, QTextEdit:hover, QComboBox:hover {{
            border-color: {UltraModernStyleSystem.COLORS['border_secondary']};
        }}
        
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        
        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid {UltraModernStyleSystem.COLORS['text_secondary']};
        }}
        
        QComboBox QAbstractItemView {{
            background: {UltraModernStyleSystem.COLORS['surface']};
            border: 1px solid {UltraModernStyleSystem.COLORS['border']};
            border-radius: {UltraModernStyleSystem.RADIUS['md']};
            selection-background-color: {UltraModernStyleSystem.COLORS['primary']};
        }}
        
        /* Ультрасовременные чекбоксы */
        QCheckBox {{
            spacing: {UltraModernStyleSystem.SPACING['md']};
            color: {UltraModernStyleSystem.COLORS['text_primary']};
            font-size: {UltraModernStyleSystem.FONTS['font_size_base']};
        }}
        
        QCheckBox::indicator {{
            width: 20px;
            height: 20px;
            border-radius: {UltraModernStyleSystem.RADIUS['sm']};
            border: 2px solid {UltraModernStyleSystem.COLORS['border']};
            background: {UltraModernStyleSystem.COLORS['surface']};
        }}
        
        QCheckBox::indicator:checked {{
            background: {UltraModernStyleSystem.COLORS['gradient_primary']};
            border: 2px solid {UltraModernStyleSystem.COLORS['primary']};
        }}
        
        QCheckBox::indicator:hover {{
            border-color: {UltraModernStyleSystem.COLORS['primary']};
        }}
        
        /* Ультрасовременные слайдеры */
        QSlider::groove:horizontal {{
            border: none;
            height: 6px;
            background: {UltraModernStyleSystem.COLORS['surface_secondary']};
            border-radius: {UltraModernStyleSystem.RADIUS['sm']};
        }}
        
        QSlider::handle:horizontal {{
            background: {UltraModernStyleSystem.COLORS['gradient_primary']};
            border: none;
            width: 24px;
            height: 24px;
            border-radius: {UltraModernStyleSystem.RADIUS['full']};
            margin: -9px 0;
            box-shadow: {UltraModernStyleSystem.COLORS['shadow_medium']};
        }}
        
        QSlider::handle:horizontal:hover {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4F46E5, stop:1 #7C3AED);
            transform: scale(1.1);
        }}
        
        /* Ультрасовременные прогресс-бары */
        QProgressBar {{
            border: none;
            border-radius: {UltraModernStyleSystem.RADIUS['md']};
            background: {UltraModernStyleSystem.COLORS['surface_secondary']};
            height: 12px;
            text-align: center;
            font-weight: {UltraModernStyleSystem.FONTS['font_weight_medium']};
        }}
        
        QProgressBar::chunk {{
            background: {UltraModernStyleSystem.COLORS['gradient_primary']};
            border-radius: {UltraModernStyleSystem.RADIUS['md']};
        }}
        
        /* Ультрасовременные меню */
        QMenuBar {{
            background: {UltraModernStyleSystem.COLORS['surface']};
            border-bottom: 1px solid {UltraModernStyleSystem.COLORS['border']};
            color: {UltraModernStyleSystem.COLORS['text_primary']};
            font-weight: {UltraModernStyleSystem.FONTS['font_weight_medium']};
        }}
        
        QMenuBar::item {{
            padding: {UltraModernStyleSystem.SPACING['md']} {UltraModernStyleSystem.SPACING['lg']};
            background: transparent;
            border-radius: {UltraModernStyleSystem.RADIUS['sm']};
        }}
        
        QMenuBar::item:selected {{
            background: {UltraModernStyleSystem.COLORS['hover_bg']};
        }}
        
        QMenu {{
            background: {UltraModernStyleSystem.COLORS['surface']};
            border: 1px solid {UltraModernStyleSystem.COLORS['border']};
            border-radius: {UltraModernStyleSystem.RADIUS['lg']};
            padding: {UltraModernStyleSystem.SPACING['md']};
            box-shadow: {UltraModernStyleSystem.COLORS['shadow_large']};
        }}
        
        QMenu::item {{
            padding: {UltraModernStyleSystem.SPACING['md']} {UltraModernStyleSystem.SPACING['lg']};
            border-radius: {UltraModernStyleSystem.RADIUS['sm']};
            font-size: {UltraModernStyleSystem.FONTS['font_size_base']};
        }}
        
        QMenu::item:selected {{
            background: {UltraModernStyleSystem.COLORS['hover_bg']};
        }}
        
        QMenu::separator {{
            height: 1px;
            background: {UltraModernStyleSystem.COLORS['divider']};
            margin: {UltraModernStyleSystem.SPACING['sm']} 0;
        }}
        
        /* Ультрасовременные скроллбары */
        QScrollBar:vertical {{
            background: {UltraModernStyleSystem.COLORS['surface']};
            width: 12px;
            border-radius: {UltraModernStyleSystem.RADIUS['md']};
            margin: 0;
        }}
        
        QScrollBar::handle:vertical {{
            background: {UltraModernStyleSystem.COLORS['surface_secondary']};
            border-radius: {UltraModernStyleSystem.RADIUS['md']};
            min-height: 40px;
            margin: 2px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background: {UltraModernStyleSystem.COLORS['surface_tertiary']};
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        
        QScrollBar:horizontal {{
            background: {UltraModernStyleSystem.COLORS['surface']};
            height: 12px;
            border-radius: {UltraModernStyleSystem.RADIUS['md']};
            margin: 0;
        }}
        
        QScrollBar::handle:horizontal {{
            background: {UltraModernStyleSystem.COLORS['surface_secondary']};
            border-radius: {UltraModernStyleSystem.RADIUS['md']};
            min-width: 40px;
            margin: 2px;
        }}
        
        QScrollBar::handle:horizontal:hover {{
            background: {UltraModernStyleSystem.COLORS['surface_tertiary']};
        }}
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}
        
        /* Ультрасовременные групповые панели */
        QGroupBox {{
            font-weight: {UltraModernStyleSystem.FONTS['font_weight_semibold']};
            font-size: {UltraModernStyleSystem.FONTS['font_size_large']};
            border: 1px solid {UltraModernStyleSystem.COLORS['border']};
            border-radius: {UltraModernStyleSystem.RADIUS['lg']};
            margin-top: {UltraModernStyleSystem.SPACING['xl']};
            padding-top: {UltraModernStyleSystem.SPACING['lg']};
            background: {UltraModernStyleSystem.COLORS['surface']};
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: {UltraModernStyleSystem.SPACING['lg']};
            padding: 0 {UltraModernStyleSystem.SPACING['md']} 0 {UltraModernStyleSystem.SPACING['md']};
            color: {UltraModernStyleSystem.COLORS['text_primary']};
        }}
        
        /* Ультрасовременные спинбоксы */
        QSpinBox, QDoubleSpinBox {{
            background: {UltraModernStyleSystem.COLORS['surface_secondary']};
            border: 1px solid {UltraModernStyleSystem.COLORS['border']};
            border-radius: {UltraModernStyleSystem.RADIUS['md']};
            padding: {UltraModernStyleSystem.SPACING['lg']};
            color: {UltraModernStyleSystem.COLORS['text_primary']};
            font-size: {UltraModernStyleSystem.FONTS['font_size_base']};
        }}
        
        QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 2px solid {UltraModernStyleSystem.COLORS['primary']};
            box-shadow: 0 0 0 3px {UltraModernStyleSystem.COLORS['focus_ring']};
        }}
        
        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            background: {UltraModernStyleSystem.COLORS['surface_tertiary']};
            border: none;
            border-radius: {UltraModernStyleSystem.RADIUS['sm']};
            width: 20px;
            height: 20px;
        }}
        
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
            background: {UltraModernStyleSystem.COLORS['hover_bg']};
        }}
        
        /* Ультрасовременные деревья */
        QTreeWidget {{
            background: {UltraModernStyleSystem.COLORS['surface']};
            border: 1px solid {UltraModernStyleSystem.COLORS['border']};
            border-radius: {UltraModernStyleSystem.RADIUS['lg']};
            color: {UltraModernStyleSystem.COLORS['text_primary']};
            font-size: {UltraModernStyleSystem.FONTS['font_size_base']};
        }}
        
        QTreeWidget::item {{
            padding: {UltraModernStyleSystem.SPACING['md']};
            border-radius: {UltraModernStyleSystem.RADIUS['sm']};
        }}
        
        QTreeWidget::item:selected {{
            background: {UltraModernStyleSystem.COLORS['primary']};
            color: {UltraModernStyleSystem.COLORS['text_inverse']};
        }}
        
        QTreeWidget::item:hover:!selected {{
            background: {UltraModernStyleSystem.COLORS['hover_bg']};
        }}
        
        /* Ультрасовременные списки */
        QListWidget {{
            background: {UltraModernStyleSystem.COLORS['surface']};
            border: 1px solid {UltraModernStyleSystem.COLORS['border']};
            border-radius: {UltraModernStyleSystem.RADIUS['lg']};
            color: {UltraModernStyleSystem.COLORS['text_primary']};
            font-size: {UltraModernStyleSystem.FONTS['font_size_base']};
        }}
        
        QListWidget::item {{
            padding: {UltraModernStyleSystem.SPACING['lg']};
            border-radius: {UltraModernStyleSystem.RADIUS['sm']};
            margin: 1px;
        }}
        
        QListWidget::item:selected {{
            background: {UltraModernStyleSystem.COLORS['primary']};
            color: {UltraModernStyleSystem.COLORS['text_inverse']};
        }}
        
        QListWidget::item:hover:!selected {{
            background: {UltraModernStyleSystem.COLORS['hover_bg']};
        }}
        
        /* Ультрасовременные статусные панели */
        QStatusBar {{
            background: {UltraModernStyleSystem.COLORS['surface']};
            color: {UltraModernStyleSystem.COLORS['text_secondary']};
            border-top: 1px solid {UltraModernStyleSystem.COLORS['border']};
            font-size: {UltraModernStyleSystem.FONTS['font_size_small']};
        }}
        
        QStatusBar::item {{
            border: none;
        }}
        
        /* Ультрасовременные панели инструментов */
        QToolBar {{
            background: {UltraModernStyleSystem.COLORS['surface']};
            border: none;
            border-bottom: 1px solid {UltraModernStyleSystem.COLORS['border']};
            spacing: {UltraModernStyleSystem.SPACING['md']};
            padding: {UltraModernStyleSystem.SPACING['md']};
        }}
        
        QToolBar::handle {{
            background: {UltraModernStyleSystem.COLORS['divider']};
            width: 1px;
            margin: {UltraModernStyleSystem.SPACING['sm']} 0;
        }}
        
        /* Ультрасовременные диалоги */
        QDialog {{
            background: {UltraModernStyleSystem.COLORS['surface']};
            border: 1px solid {UltraModernStyleSystem.COLORS['border']};
            border-radius: {UltraModernStyleSystem.RADIUS['xl']};
        }}
        
        QDialogButtonBox {{
            background: transparent;
        }}
        
        /* Анимации и эффекты */
        * {{
            /* transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1); */
        }}
        
        /* Специальные классы для компонентов */
        .HeaderLabel {{
            font-size: {UltraModernStyleSystem.FONTS['font_size_3xl']};
            font-weight: {UltraModernStyleSystem.FONTS['font_weight_bold']};
            color: {UltraModernStyleSystem.COLORS['text_primary']};
            margin-bottom: {UltraModernStyleSystem.SPACING['lg']};
        }}
        
        .SubHeaderLabel {{
            font-size: {UltraModernStyleSystem.FONTS['font_size_2xl']};
            font-weight: {UltraModernStyleSystem.FONTS['font_weight_semibold']};
            color: {UltraModernStyleSystem.COLORS['text_primary']};
            margin-bottom: {UltraModernStyleSystem.SPACING['md']};
        }}
        
        .BodyLabel {{
            font-size: {UltraModernStyleSystem.FONTS['font_size_base']};
            font-weight: {UltraModernStyleSystem.FONTS['font_weight_normal']};
            color: {UltraModernStyleSystem.COLORS['text_secondary']};
            line-height: {UltraModernStyleSystem.FONTS['line_height_normal']};
        }}
        
        .CaptionLabel {{
            font-size: {UltraModernStyleSystem.FONTS['font_size_small']};
            font-weight: {UltraModernStyleSystem.FONTS['font_weight_medium']};
            color: {UltraModernStyleSystem.COLORS['text_tertiary']};
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .MetricValue {{
            font-size: {UltraModernStyleSystem.FONTS['font_size_4xl']};
            font-weight: {UltraModernStyleSystem.FONTS['font_weight_bold']};
            color: {UltraModernStyleSystem.COLORS['text_primary']};
            line-height: 1;
        }}
        
        .MetricLabel {{
            font-size: {UltraModernStyleSystem.FONTS['font_size_small']};
            font-weight: {UltraModernStyleSystem.FONTS['font_weight_medium']};
            color: {UltraModernStyleSystem.COLORS['text_tertiary']};
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}
        """

# Обновленные классы компонентов
class UltraModernCard(QWidget):
    """Ультрасовременная карточка"""
    
    def __init__(self, title="", content_widget=None, style="default", parent=None):
        super().__init__(parent)
        self.title = title
        self.content_widget = content_widget
        self.style = style
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        if self.title:
            title_label = QLabel(self.title)
            title_label.setProperty("class", "SubHeaderLabel")
            layout.addWidget(title_label)
        
        if self.content_widget:
            layout.addWidget(self.content_widget)
        
        # Применение стиля
        if self.style == "glass":
            self.setProperty("class", "GlassCard")
        elif self.style == "elevated":
            self.setProperty("class", "ElevatedCard")
        else:
            self.setProperty("class", "UltraModernCard")

class UltraModernButton(QPushButton):
    """Ультрасовременная кнопка"""
    
    def __init__(self, text="", icon=None, style="primary", size="normal", parent=None):
        super().__init__(text, parent)
        self.style = style
        self.size = size
        self.setup_style()
        
    def setup_style(self):
        if self.style != "primary":
            self.setProperty("class", self.style)
        
        if self.size == "small":
            self.setMaximumHeight(36)
            self.setStyleSheet(self.styleSheet() + "font-size: 12px; padding: 8px 16px;")
        elif self.size == "large":
            self.setMinimumHeight(56)
            self.setStyleSheet(self.styleSheet() + "font-size: 16px; padding: 16px 32px;")

class UltraModernProgressBar(QProgressBar):
    """Ультрасовременный прогресс-бар"""
    
    def __init__(self, style="primary", parent=None):
        super().__init__(parent)
        self.style = style
        self.setup_style()
        
    def setup_style(self):
        if self.style != "primary":
            # Применение различных стилей прогресс-бара
            pass

class UltraModernChart(QChartView):
    """Ультрасовременный график"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_style()
        
    def setup_style(self):
        # Настройка стиля графика
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QBrush(QColor(UltraModernStyleSystem.COLORS['surface'])))

class UltraModernTable(QTableWidget):
    """Ультрасовременная таблица"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_style()
        
    def setup_style(self):
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)

class UltraModernComboBox(QComboBox):
    """Ультрасовременный комбобокс"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_style()
        
    def setup_style(self):
        self.setMinimumHeight(48)

class UltraModernSpinBox(QSpinBox):
    """Ультрасовременный спинбокс"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_style()
        
    def setup_style(self):
        self.setMinimumHeight(48)

class UltraModernCheckBox(QCheckBox):
    """Ультрасовременный чекбокс"""
    
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setup_style()
        
    def setup_style(self):
        pass

class UltraModernLabel(QLabel):
    """Ультрасовременная метка"""
    
    def __init__(self, text="", style="body", parent=None):
        super().__init__(text, parent)
        self.style = style
        self.setup_style()
        
    def setup_style(self):
        if self.style == "header":
            self.setProperty("class", "HeaderLabel")
        elif self.style == "subheader":
            self.setProperty("class", "SubHeaderLabel")
        elif self.style == "body":
            self.setProperty("class", "BodyLabel")
        elif self.style == "caption":
            self.setProperty("class", "CaptionLabel")
        elif self.style == "metric":
            self.setProperty("class", "MetricValue")
        elif self.style == "metric_label":
            self.setProperty("class", "MetricLabel")

class UltraModernDivider(QFrame):
    """Ультрасовременный разделитель"""
    
    def __init__(self, orientation="horizontal", parent=None):
        super().__init__(parent)
        self.orientation = orientation
        self.setup_style()
        
    def setup_style(self):
        if self.orientation == "horizontal":
            self.setFrameShape(QFrame.Shape.HLine)
            self.setMaximumHeight(1)
        else:
            self.setFrameShape(QFrame.Shape.VLine)
            self.setMaximumWidth(1)
        
        self.setStyleSheet(f"background-color: {UltraModernStyleSystem.COLORS['divider']}; border: none;")

class UltraModernBadge(QLabel):
    """Ультрасовременный бейдж"""
    
    def __init__(self, text="", color="primary", parent=None):
        super().__init__(text, parent)
        self.color = color
        self.setup_style()
        
    def setup_style(self):
        gradient_map = {
            'primary': UltraModernStyleSystem.COLORS['gradient_primary'],
            'secondary': UltraModernStyleSystem.COLORS['gradient_secondary'],
            'success': UltraModernStyleSystem.COLORS['gradient_success'],
            'danger': UltraModernStyleSystem.COLORS['gradient_danger'],
            'warning': UltraModernStyleSystem.COLORS['gradient_warning'],
            'info': UltraModernStyleSystem.COLORS['gradient_info'],
            'accent': UltraModernStyleSystem.COLORS['gradient_accent']
        }
        
        gradient = gradient_map.get(self.color, UltraModernStyleSystem.COLORS['gradient_primary'])
        
        self.setStyleSheet(f"""
            background: {gradient};
            color: {UltraModernStyleSystem.COLORS['text_inverse']};
            border-radius: {UltraModernStyleSystem.RADIUS['full']};
            padding: 4px 12px;
            font-size: {UltraModernStyleSystem.FONTS['font_size_small']};
            font-weight: {UltraModernStyleSystem.FONTS['font_weight_semibold']};
        """)

# Для обратной совместимости
ModernStyleSystem = UltraModernStyleSystem
ModernCard = UltraModernCard
ModernButton = UltraModernButton
ModernProgressBar = UltraModernProgressBar
ModernChart = UltraModernChart
ModernTable = UltraModernTable
ModernComboBox = UltraModernComboBox
ModernSpinBox = UltraModernSpinBox
ModernCheckBox = UltraModernCheckBox
ModernLabel = UltraModernLabel
ModernDivider = UltraModernDivider
ModernBadge = UltraModernBadge 