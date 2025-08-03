#!/usr/bin/env python3
"""
Скрипт для создания exe файла ATB Trading System
"""

import os
import sys
import subprocess
from pathlib import Path

def check_pyinstaller():
    """Проверка наличия PyInstaller"""
    try:
        import PyInstaller
        print("✅ PyInstaller найден")
        return True
    except ImportError:
        print("❌ PyInstaller не найден")
        print("📦 Установка PyInstaller...")
        
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], 
                         check=True)
            print("✅ PyInstaller установлен")
            return True
        except subprocess.CalledProcessError:
            print("❌ Ошибка установки PyInstaller")
            return False

def create_exe():
    """Создание exe файла"""
    project_root = Path(__file__).parent
    launcher_file = project_root / "atb_launcher.py"
    
    if not launcher_file.exists():
        print(f"❌ Файл {launcher_file} не найден!")
        return False
    
    print("🔨 Создание exe файла...")
    
    # Параметры для PyInstaller
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",                    # Один exe файл
        "--windowed",                   # Без консоли (можно убрать если нужна консоль)
        "--name=ATB_Trading_System",    # Имя exe файла
        "--icon=icon.ico",              # Иконка (если есть)
        "--add-data=interfaces;interfaces",  # Добавляем папку interfaces
        "--add-data=domain;domain",     # Добавляем папку domain
        "--add-data=application;application",  # Добавляем папку application
        "--add-data=infrastructure;infrastructure",  # Добавляем папку infrastructure
        "--hidden-import=tkinter",      # Явный импорт tkinter
        "--hidden-import=numpy",        # Явный импорт numpy
        "--hidden-import=pandas",       # Явный импорт pandas
        "--hidden-import=matplotlib",   # Явный импорт matplotlib
        "--clean",                      # Очистка временных файлов
        str(launcher_file)
    ]
    
    try:
        # Удаляем иконку если её нет
        if not (project_root / "icon.ico").exists():
            cmd.remove("--icon=icon.ico")
        
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ exe файл создан успешно!")
            
            # Проверяем результат
            exe_file = project_root / "dist" / "ATB_Trading_System.exe"
            if exe_file.exists():
                print(f"📁 Файл создан: {exe_file}")
                print(f"📏 Размер: {exe_file.stat().st_size / (1024*1024):.1f} MB")
                
                # Создаем простой bat файл для запуска
                create_launcher_bat(project_root)
                
                return True
            else:
                print("❌ exe файл не найден после сборки")
                return False
        else:
            print("❌ Ошибка создания exe файла:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Исключение при создании exe: {e}")
        return False

def create_launcher_bat(project_root):
    """Создание bat файла для запуска exe"""
    bat_content = """@echo off
title ATB Trading System v2.0
color 0A

echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                        ATB TRADING SYSTEM v2.0                              ║
echo ║                           EXE LAUNCHER                                      ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.

set EXE_PATH=%~dp0dist\\ATB_Trading_System.exe

if not exist "%EXE_PATH%" (
    echo [ERROR] ATB_Trading_System.exe не найден в папке dist\\
    echo [INFO] Убедитесь, что exe файл был создан успешно
    echo.
    pause
    exit /b 1
)

echo [INFO] Запуск ATB Trading System...
echo [INFO] EXE файл: %EXE_PATH%
echo.

"%EXE_PATH%"

echo.
echo [INFO] Система завершена
pause"""

    bat_file = project_root / "START_ATB_EXE.bat"
    with open(bat_file, 'w', encoding='utf-8') as f:
        f.write(bat_content)
    
    print(f"✅ Создан launcher: {bat_file}")

def create_icon():
    """Создание простой иконки"""
    try:
        from PIL import Image, ImageDraw
        
        # Создаем простую иконку
        img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Рисуем простой символ
        draw.rectangle([10, 10, 54, 54], fill=(55, 66, 250, 255))
        draw.text((25, 25), "ATB", fill=(255, 255, 255, 255))
        
        # Сохраняем как ico
        icon_path = Path(__file__).parent / "icon.ico"
        img.save(icon_path, format='ICO')
        
        print(f"✅ Иконка создана: {icon_path}")
        return True
        
    except ImportError:
        print("⚠️ Pillow не установлен, иконка не будет создана")
        return False
    except Exception as e:
        print(f"⚠️ Ошибка создания иконки: {e}")
        return False

def main():
    """Главная функция"""
    print("🚀 ATB Trading System - Создание EXE файла")
    print("=" * 60)
    
    # Проверка PyInstaller
    if not check_pyinstaller():
        print("❌ Невозможно создать exe без PyInstaller")
        return False
    
    # Создание иконки
    create_icon()
    
    # Создание exe
    success = create_exe()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 EXE файл создан успешно!")
        print("\n📋 Файлы для запуска:")
        print("   • dist/ATB_Trading_System.exe - Основной exe файл")
        print("   • START_ATB_EXE.bat - Bat файл для запуска exe")
        print("   • ATB_START.bat - Bat файл для запуска из исходников")
        print("\n💡 Рекомендации:")
        print("   • Для разработки используйте ATB_START.bat")
        print("   • Для продакшена используйте START_ATB_EXE.bat")
        print("   • Скопируйте всю папку dist/ на целевой компьютер")
        
        return True
    else:
        print("\n❌ Не удалось создать exe файл")
        print("\n🔧 Возможные решения:")
        print("   • pip install pyinstaller")
        print("   • pip install pillow (для иконки)")
        print("   • Запустить от имени администратора")
        print("   • Проверить антивирус")
        
        return False

if __name__ == "__main__":
    try:
        success = main()
        input("\nНажмите Enter для выхода...")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        input("\nНажмите Enter для выхода...")
        sys.exit(1)