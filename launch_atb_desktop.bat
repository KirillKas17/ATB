@echo off
chcp 65001 >nul
title ⚡ ATB Trading System - Desktop Launcher v3.0

echo.
echo ===============================================================================
echo                ⚡ ATB Trading System - Desktop Launcher v3.0
echo ===============================================================================
echo.
echo 🚀 Запуск полноценного десктопного приложения для торговли...
echo.

:: Проверка наличия Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Ошибка: Python не найден!
    echo 📥 Пожалуйста, установите Python 3.8+ с https://python.org
    pause
    exit /b 1
)

echo ✅ Python найден
echo.

:: Проверка наличия pip
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Ошибка: pip не найден!
    echo 📥 Пожалуйста, переустановите Python с включенным pip
    pause
    exit /b 1
)

echo ✅ pip найден
echo.

:: Проверка и установка зависимостей
echo 🔍 Проверка зависимостей...

:: Проверка PyQt6
python -c "import PyQt6" >nul 2>&1
if errorlevel 1 (
    echo 📥 Установка PyQt6...
    pip install PyQt6 PyQt6-Charts
    if errorlevel 1 (
        echo ❌ Ошибка установки PyQt6
        pause
        exit /b 1
    )
)

echo ✅ PyQt6 доступен

:: Проверка loguru
python -c "import loguru" >nul 2>&1
if errorlevel 1 (
    echo 📥 Установка loguru...
    pip install loguru
)

echo ✅ loguru доступен

:: Проверка других зависимостей
python -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo 📥 Установка numpy...
    pip install numpy
)

python -c "import pandas" >nul 2>&1
if errorlevel 1 (
    echo 📥 Установка pandas...
    pip install pandas
)

python -c "import matplotlib" >nul 2>&1
if errorlevel 1 (
    echo 📥 Установка matplotlib...
    pip install matplotlib
)

echo.
echo ✅ Все зависимости проверены и установлены
echo.

:: Создание необходимых директорий
if not exist "logs" mkdir logs
if not exist "config" mkdir config
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "backups" mkdir backups

echo ✅ Рабочие директории подготовлены
echo.

:: Проверка наличия основного файла приложения
if not exist "atb_unified_desktop_app.py" (
    echo ❌ Ошибка: Файл atb_unified_desktop_app.py не найден!
    echo 📂 Убедитесь, что вы находитесь в правильной директории
    pause
    exit /b 1
)

echo ✅ Основной файл приложения найден
echo.

:: Установка переменных окружения
set ATB_MODE=simulation
set ATB_DEBUG=false
set ATB_LOG_LEVEL=INFO

echo 🎯 Режим работы: Симуляция
echo 🐛 Режим отладки: Отключен
echo 📝 Уровень логирования: INFO
echo.

echo ===============================================================================
echo                        🚀 ЗАПУСК ПРИЛОЖЕНИЯ
echo ===============================================================================
echo.
echo 🎮 Готовимся к запуску ATB Trading System...
echo.
echo 💡 Возможности приложения:
echo   • 📊 Полноценный торговый дашборд
echo   • 🤖 Автоматическая торговля с ИИ
echo   • 📈 Реальная аналитика и графики
echo   • 🔙 Бэктестинг стратегий
echo   • 💼 Управление портфелем
echo   • 🔮 ML прогнозирование
echo   • ⚙️ Гибкие настройки
echo.
echo ⚡ Запуск через 3 секунды...
timeout /t 3 /nobreak >nul

:: Запуск основного приложения
echo.
echo 🚀 Запуск ATB Trading System Desktop...
echo.

python atb_unified_desktop_app.py

:: Обработка завершения
if errorlevel 1 (
    echo.
    echo ❌ Приложение завершилось с ошибкой!
    echo 📝 Проверьте файл logs/dashboard.log для деталей
    echo.
) else (
    echo.
    echo ✅ Приложение успешно завершено
    echo.
)

echo ===============================================================================
echo                             ЗАВЕРШЕНИЕ
echo ===============================================================================
echo.
echo 💫 Спасибо за использование ATB Trading System!
echo 🌐 Сайт: https://atb-trading.com
echo 📧 Поддержка: support@atb-trading.com
echo.
echo 📊 Статистика сессии:
echo   • Время запуска: %time%
echo   • Дата: %date%
echo   • Режим: %ATB_MODE%
echo.

pause