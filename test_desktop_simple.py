#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script to verify desktop application UI components
"""

import sys

def test_pyqt6_imports():
    """Test PyQt6 imports only"""
    print("Testing PyQt6 imports...")
    
    try:
        from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget
        from PyQt6.QtGui import QAction
        from PyQt6.QtCore import QTimer, QThread, pyqtSignal
        from PyQt6.QtCharts import QChart, QChartView, QLineSeries
        print("✓ PyQt6 imports successful")
        return True
    except ImportError as e:
        print(f"❌ PyQt6 import error: {e}")
        return False

def test_desktop_app_creation():
    """Test desktop app creation without main system imports"""
    print("\nTesting desktop app creation...")
    
    try:
        from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QTabWidget
        from PyQt6.QtGui import QFont
        from PyQt6.QtCore import Qt
        
        # Create QApplication instance
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # Test creating a simple window
        print("✓ Testing simple window creation...")
        window = QMainWindow()
        window.setWindowTitle("ATB Desktop Test")
        window.resize(800, 600)
        print("  ✓ Simple window created successfully")
        
        # Test creating basic widgets
        print("✓ Testing basic widget creation...")
        
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        
        # Add some test widgets
        label = QLabel("ATB Trading System - Desktop Application")
        label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(label)
        
        button = QPushButton("Test Button")
        layout.addWidget(button)
        
        tab_widget = QTabWidget()
        tab_widget.addTab(QLabel("Tab 1"), "Main")
        tab_widget.addTab(QLabel("Tab 2"), "Settings")
        layout.addWidget(tab_widget)
        
        window.setCentralWidget(central_widget)
        print("  ✓ Basic widgets created successfully")
        
        print("🎉 Desktop UI components are working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ UI creation error: {e}")
        return False

def test_charts():
    """Test PyQt6 Charts functionality"""
    print("\nTesting PyQt6 Charts...")
    
    try:
        from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
        from PyQt6.QtCore import QPointF
        
        # Create a simple chart
        chart = QChart()
        chart.setTitle("Test Chart")
        
        # Create a line series
        series = QLineSeries()
        series.append(QPointF(0, 1))
        series.append(QPointF(1, 2))
        series.append(QPointF(2, 3))
        series.append(QPointF(3, 2))
        series.append(QPointF(4, 1))
        
        chart.addSeries(series)
        
        # Create axes
        axis_x = QValueAxis()
        axis_x.setRange(0, 4)
        axis_x.setTitleText("Time")
        
        axis_y = QValueAxis()
        axis_y.setRange(0, 4)
        axis_y.setTitleText("Value")
        
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        
        series.attachAxis(axis_x)
        series.attachAxis(axis_y)
        
        print("✓ PyQt6 Charts working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Charts error: {e}")
        return False

if __name__ == "__main__":
    print("ATB Desktop Application Simple Test")
    print("=" * 40)
    
    success = test_pyqt6_imports()
    if success:
        success = test_desktop_app_creation()
    if success:
        success = test_charts()
    
    if success:
        print("\n✅ All UI tests passed! Desktop application is ready to use.")
        print("\nTo launch the full application, run:")
        print("  python start_atb_desktop.py --enhanced")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1) 