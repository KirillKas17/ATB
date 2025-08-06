#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify desktop application imports
"""

import sys
from pathlib import Path

def test_imports() -> None:
    """Test all desktop application imports"""
    print("Testing desktop application imports...")
    
    try:
        # Test PyQt6 imports
        print("✓ Testing PyQt6 imports...")
        from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget
        from PyQt6.QtGui import QAction
        from PyQt6.QtCore import QTimer, QThread, pyqtSignal
        from PyQt6.QtCharts import QChart, QChartView, QLineSeries
        print("  ✓ PyQt6 imports successful")
        
        # Test desktop application imports
        print("✓ Testing desktop application imports...")
        from atb_desktop_app import ATBDesktopApp, ATBSystemThread
        print("  ✓ Basic desktop app imports successful")
        
        # Test enhanced desktop application imports
        print("✓ Testing enhanced desktop application imports...")
        from atb_desktop_app_enhanced import ATBDesktopApp as EnhancedATBDesktopApp
        print("  ✓ Enhanced desktop app imports successful")
        
        # Test widgets imports
        print("✓ Testing widgets imports...")
        from atb_desktop_widgets import (
            BacktestDialog, ConfigurationDialog, PerformanceWidget,
            StrategyManagerWidget, MarketDataWidget
        )
        print("  ✓ Widgets imports successful")
        
        # Test main system imports
        print("✓ Testing main system imports...")
        try:
            from main import main as main_function
            print("  ✓ Main system imports successful")
        except ImportError as e:
            print(f"  ⚠ Main system imports warning: {e}")
        
        print("\n🎉 All desktop application imports are working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_ui_creation() -> None:
    """Test UI creation without showing window"""
    print("\nTesting UI creation...")
    
    try:
        # Create QApplication instance
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # Test creating main window
        print("✓ Testing main window creation...")
        window = ATBDesktopApp()
        print("  ✓ Main window created successfully")
        
        # Test creating enhanced window
        print("✓ Testing enhanced window creation...")
        enhanced_window = EnhancedATBDesktopApp()
        print("  ✓ Enhanced window created successfully")
        
        # Test creating widgets
        print("✓ Testing widget creation...")
        backtest_dialog = BacktestDialog()
        config_dialog = ConfigurationDialog()
        performance_widget = PerformanceWidget()
        strategy_widget = StrategyManagerWidget()
        market_widget = MarketDataWidget()
        print("  ✓ All widgets created successfully")
        
        print("🎉 All UI components can be created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ UI creation error: {e}")
        return False

if __name__ == "__main__":
    print("ATB Desktop Application Import Test")
    print("=" * 40)
    
    success = test_imports()
    if success:
        success = test_ui_creation()
    
    if success:
        print("\n✅ All tests passed! Desktop application is ready to use.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1) 