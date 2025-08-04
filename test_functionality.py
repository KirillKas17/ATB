#!/usr/bin/env python3
"""
Test script to verify the trading dashboard functionality
"""

import sys
from PyQt6.QtWidgets import QApplication
from modern_dashboard_app import MainWindow

def test_application():
    """Test the main application functionality"""
    app = QApplication(sys.argv)
    
    # Create main window
    window = MainWindow()
    
    # Test that the window was created successfully
    print("✓ Main window created successfully")
    
    # Test that pages exist
    expected_pages = ['dashboard', 'analytics', 'trading', 'portfolio', 'settings', 'profile']
    for page in expected_pages:
        if page in window.pages:
            print(f"✓ {page} page exists")
        else:
            print(f"✗ {page} page missing")
    
    # Test sidebar connection
    if hasattr(window.sidebar, 'page_changed'):
        print("✓ Sidebar has page_changed signal")
    else:
        print("✗ Sidebar missing page_changed signal")
    
    # Test current page
    if window.current_page == 'dashboard':
        print("✓ Dashboard is the default page")
    else:
        print(f"✗ Default page is {window.current_page}, expected dashboard")
    
    print("\nAll tests completed!")
    print("The application should now be ready to run with:")
    print("python start_modern_dashboard.py")

if __name__ == "__main__":
    test_application() 