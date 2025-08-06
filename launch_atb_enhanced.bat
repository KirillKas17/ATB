@echo off
chcp 65001 >nul
title ⚡ ATB Trading System - Enhanced Launcher v3.1

echo.
echo ===============================================================================
echo                ⚡ ATB Trading System - Enhanced Launcher v3.1
echo ===============================================================================
echo.
echo 🚀 Запуск улучшенного десктопного приложения с реальными данными...
echo.
echo ✨ Новые возможности:
echo • 📊 Реальный мониторинг системных ресурсов (CPU, RAM, Диск, Сеть)
echo • 🧬 Эволюция стратегий в реальном времени
echo • 🔧 Полное управление .env конфигурацией
echo • 💼 Расширенный анализ портфеля
echo • 📈 Интеграция всех торговых данных
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

:: Установка дополнительных зависимостей для мониторинга
echo 📦 Установка дополнительных пакетов для мониторинга...
pip install --quiet --upgrade psutil loguru
if errorlevel 1 (
    echo ⚠️ Не удалось установить psutil, продолжаем...
)

:: Проверка наличия PyQt6
python -c "import PyQt6" >nul 2>&1
if errorlevel 1 (
    echo 📦 Установка PyQt6...
    pip install --quiet PyQt6 PyQt6-Charts
    if errorlevel 1 (
        echo ❌ Ошибка: Не удалось установить PyQt6!
        echo 💡 Попробуйте установить вручную: pip install PyQt6 PyQt6-Charts
        pause
        exit /b 1
    )
)

echo ✅ PyQt6 готов
echo.

:: Создание .env файла если его нет
if not exist ".env" (
    echo 🔧 Создание конфигурационного файла .env...
    (
        echo # ATB Trading System Enhanced Configuration
        echo # Общие настройки
        echo ENVIRONMENT=development
        echo DEBUG=true
        echo ATB_MODE=simulation
        echo.
        echo # База данных
        echo DB_HOST=localhost
        echo DB_PORT=5432
        echo DB_NAME=atb_trading
        echo DB_USER=atb_user
        echo DB_PASS=
        echo.
        echo # Биржа
        echo EXCHANGE_API_KEY=
        echo EXCHANGE_API_SECRET=
        echo EXCHANGE_TESTNET=true
        echo.
        echo # Мониторинг
        echo MONITORING_ENABLED=true
        echo MONITORING_INTERVAL=10
        echo ALERT_EMAIL=
        echo.
        echo # Эволюция
        echo EVOLUTION_ENABLED=true
        echo EVOLUTION_INTERVAL=3600
        echo AUTO_EVOLUTION=false
        echo.
        echo # Торговля
        echo DEFAULT_POSITION_SIZE=1.0
        echo DEFAULT_STOP_LOSS=2.0
        echo MAX_DRAWDOWN=5.0
        echo.
        echo # Логирование
        echo LOG_LEVEL=INFO
        echo LOG_FILE=logs/atb.log
    ) > .env
    echo ✅ Конфигурационный файл создан
)

:: Создание директории для логов
if not exist "logs" mkdir logs

echo.
echo 🎯 Запуск ATB Trading System Enhanced Desktop v3.1...
echo.

:: Установка переменных окружения
set PYTHONPATH=%cd%
set ATB_ENHANCED=true

:: Запуск улучшенного приложения
python atb_unified_desktop_app_enhanced.py

if errorlevel 1 (
    echo.
    echo ❌ Ошибка запуска улучшенного приложения!
    echo.
    echo 💡 Возможные решения:
    echo • Убедитесь, что установлены все зависимости
    echo • Проверьте настройки в файле .env
    echo • Запустите: pip install -r requirements.txt
    echo • Проверьте логи в папке logs/
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ ATB Trading System Enhanced завершен успешно
echo.
pause