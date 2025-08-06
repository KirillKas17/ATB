#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Запуск демонстрации ультрасовременного дизайна ATB
"""

import sys
import os
from pathlib import Path

# Добавление текущей директории в путь
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main() -> None:
    """Главная функция запуска демонстрации"""
    try:
        # Проверка наличия необходимых модулей
        from ultra_modern_demo import main as demo_main
        
        print("🚀 Запуск демонстрации ультрасовременного дизайна ATB...")
        print("📱 Применены современные дизайнерские решения:")
        print("   • Темная тема с продвинутыми цветами")
        print("   • Современная типографика (Inter)")
        print("   • Glassmorphism эффекты")
        print("   • Плавные анимации и переходы")
        print("   • Адаптивная сетка компонентов")
        print("   • Улучшенные отступы и пропорции")
        print()
        
        # Запуск демонстрации
        demo_main()
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("Убедитесь, что все необходимые файлы находятся в текущей директории")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 