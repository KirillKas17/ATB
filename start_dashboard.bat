@echo off
title ATB Trading Dashboard - Launcher
color 0A

echo.
echo ===============================================================
echo          ATB Trading Dashboard v2.0 - Windows Launcher
echo ===============================================================
echo.

REM Проверка наличия Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python не найден в системе!
    echo Пожалуйста, установите Python 3.8+ с https://python.org
    echo.
    pause
    exit /b 1
)

echo [INFO] Python найден...

REM Проверка зависимостей
echo [INFO] Проверка зависимостей...
python -c "import tkinter, decimal" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Отсутствуют базовые модули Python!
    pause
    exit /b 1
)

echo [INFO] Базовые модули найдены...

REM Попытка установки дополнительных зависимостей
echo [INFO] Проверка дополнительных библиотек...
python -c "import numpy, pandas, matplotlib" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Дополнительные библиотеки не найдены.
    echo [INFO] Попытка автоматической установки...
    pip install numpy pandas matplotlib
    if %errorlevel% neq 0 (
        echo [WARNING] Не удалось установить некоторые библиотеки.
        echo [INFO] Запуск в упрощенном режиме...
    )
)

echo [INFO] Все проверки завершены.
echo.

REM Запуск дашборда
echo ===============================================================
echo                    Запуск дашборда...
echo ===============================================================
echo.

python run_dashboard.py

REM Обработка завершения
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Дашборд завершился с ошибкой (код: %errorlevel%)
    echo [INFO] Проверьте сообщения выше для диагностики.
) else (
    echo.
    echo [INFO] Дашборд завершен успешно.
)

echo.
echo Нажмите любую клавишу для выхода...
pause >nul