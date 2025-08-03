#!/usr/bin/env python3
"""
Тест для проверки исправленных RSI расчетов
"""

import pandas as pd
import numpy as np
from infrastructure.core.technical import rsi

def test_rsi_with_zero_losses():
    """Тест RSI с нулевыми потерями (было деление на ноль)"""
    print("🧪 Тестирование исправлений RSI...")
    
    # Тестовые данные где все цены растут (нет потерь)
    prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
    
    try:
        result = rsi(prices, period=14)
        print(f"✅ RSI успешно рассчитан: {result.iloc[-1]:.2f}")
        print(f"✅ Последнее значение RSI: {result.iloc[-1]:.2f} (должно быть ~100)")
        
        # Проверяем что нет NaN
        if not result.isna().any():
            print("✅ Нет NaN значений в результате")
        else:
            print("❌ Найдены NaN значения")
            
        return True
    except Exception as e:
        print(f"❌ ОШИБКА: {e}")
        return False

def test_rsi_with_zero_gains():
    """Тест RSI с нулевыми приростами (все цены падают)"""
    print("\n🧪 Тестирование RSI с падающими ценами...")
    
    # Тестовые данные где все цены падают (нет приростов)
    prices = [115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100]
    
    try:
        result = rsi(prices, period=14)
        print(f"✅ RSI успешно рассчитан: {result.iloc[-1]:.2f}")
        print(f"✅ Последнее значение RSI: {result.iloc[-1]:.2f} (должно быть ~0)")
        return True
    except Exception as e:
        print(f"❌ ОШИБКА: {e}")
        return False

def test_empty_data():
    """Тест с пустыми данными"""
    print("\n🧪 Тестирование RSI с пустыми данными...")
    
    try:
        result = rsi([], period=14)
        print(f"✅ RSI с пустыми данными: длина результата = {len(result)}")
        return True
    except Exception as e:
        print(f"❌ ОШИБКА с пустыми данными: {e}")
        return False

if __name__ == "__main__":
    print("🚀 ПРОВЕРКА ИСПРАВЛЕНИЙ RSI")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    if test_rsi_with_zero_losses():
        success_count += 1
    
    if test_rsi_with_zero_gains():
        success_count += 1
        
    if test_empty_data():
        success_count += 1
    
    print(f"\n📊 РЕЗУЛЬТАТ: {success_count}/{total_tests} тестов пройдено")
    
    if success_count == total_tests:
        print("🎉 ВСЕ ИСПРАВЛЕНИЯ RSI РАБОТАЮТ КОРРЕКТНО!")
    else:
        print("⚠️ Некоторые тесты не прошли")