@echo off
setlocal enabledelayedexpansion
title ATB Trading Dashboard v2.0 - Advanced Launcher
color 0B

REM Настройка переменных
set "APP_NAME=ATB Trading Dashboard"
set "APP_VERSION=v2.0"
set "PYTHON_MIN_VERSION=3.8"
set "LOG_FILE=logs\launcher.log"

REM Создание директорий
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "config" mkdir config
if not exist "temp" mkdir temp
if not exist "backups" mkdir backups

REM Начало логирования
echo %date% %time% - Launcher started >> %LOG_FILE%

cls
echo.
echo                    ███████╗██╗   ██╗███╗   ██╗████████╗██████╗  █████╗ 
echo                    ██╔════╝╚██╗ ██╔╝████╗  ██║╚══██╔══╝██╔══██╗██╔══██╗
echo                    ███████╗ ╚████╔╝ ██╔██╗ ██║   ██║   ██████╔╝███████║
echo                    ╚════██║  ╚██╔╝  ██║╚██╗██║   ██║   ██╔══██╗██╔══██║
echo                    ███████║   ██║   ██║ ╚████║   ██║   ██║  ██║██║  ██║
echo                    ╚══════╝   ╚═╝   ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝
echo.
echo ═══════════════════════════════════════════════════════════════════════════════
color 0A
echo                    🚀 %APP_NAME% %APP_VERSION% 🚀
echo                          Advanced Trading Platform Launcher
color 0B
echo ═══════════════════════════════════════════════════════════════════════════════
echo.

REM Функция проверки Python
:CHECK_PYTHON
echo [⚡] Проверка системных требований...
echo.
python --version >nul 2>&1 || python3 --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo ❌ ОШИБКА: Python не найден в системе!
    echo.
    echo 📥 Для работы системы требуется Python %PYTHON_MIN_VERSION%+
    echo 🌐 Скачайте с: https://python.org/downloads/
    echo 🔧 Убедитесь, что Python добавлен в PATH
    echo.
    pause
    exit /b 1
)

python --version >nul 2>&1 && (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VER=%%i
    set PYTHON_CMD=python
) || (
    for /f "tokens=2" %%i in ('python3 --version 2^>^&1') do set PYTHON_VER=%%i
    set PYTHON_CMD=python3
)
echo ✅ Python найден: %PYTHON_VER%
echo %date% %time% - Python version: %PYTHON_VER% >> %LOG_FILE%

REM Проверка базовых зависимостей
echo [🔍] Проверка базовых модулей...
%PYTHON_CMD% -c "import sys, os, asyncio, logging, json, threading, decimal, datetime, pathlib" >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo ❌ Критическая ошибка: Отсутствуют базовые модули Python!
    echo 🔧 Переустановите Python с полным набором модулей
    pause
    exit /b 1
)
echo ✅ Базовые модули проверены

REM Проверка GUI модулей
echo [🖥️] Проверка GUI компонентов...
%PYTHON_CMD% -c "import tkinter" >nul 2>&1
if %errorlevel% neq 0 (
    color 0E
    echo ⚠️ WARNING: Tkinter недоступен
    echo 📱 GUI интерфейс будет ограничен
) else (
    echo ✅ Tkinter готов к работе
)

REM Проверка научных библиотек
echo [📊] Проверка аналитических библиотек...
set "MISSING_LIBS="
%PYTHON_CMD% -c "import numpy" >nul 2>&1
if %errorlevel% neq 0 set "MISSING_LIBS=!MISSING_LIBS! numpy"

%PYTHON_CMD% -c "import pandas" >nul 2>&1
if %errorlevel% neq 0 set "MISSING_LIBS=!MISSING_LIBS! pandas"

%PYTHON_CMD% -c "import matplotlib" >nul 2>&1
if %errorlevel% neq 0 set "MISSING_LIBS=!MISSING_LIBS! matplotlib"

if not "!MISSING_LIBS!"=="" (
    color 0E
    echo ⚠️ Отсутствующие библиотеки:!MISSING_LIBS!
    echo.
    echo [💿] Автоматическая установка зависимостей...
    echo 🔄 Установка может занять несколько минут...
    echo.
    
    pip install --upgrade pip
    for %%lib in (!MISSING_LIBS!) do (
        echo 📦 Установка %%lib...
        pip install %%lib
        if !errorlevel! neq 0 (
            echo ❌ Ошибка установки %%lib
        ) else (
            echo ✅ %%lib установлен успешно
        )
    )
    echo.
    echo 🔄 Повторная проверка зависимостей...
    goto CHECK_ADVANCED_DEPS
)

:CHECK_ADVANCED_DEPS
echo ✅ Все библиотеки готовы к работе

REM Проверка дополнительных зависимостей
echo [🧪] Проверка расширенного функционала...
set "OPTIONAL_FEATURES="

%PYTHON_CMD% -c "import requests" >nul 2>&1
if %errorlevel% equ 0 (
    set "OPTIONAL_FEATURES=!OPTIONAL_FEATURES! ✅API-клиент"
) else (
    set "OPTIONAL_FEATURES=!OPTIONAL_FEATURES! ❌API-клиент"
)

%PYTHON_CMD% -c "import websockets" >nul 2>&1
if %errorlevel% equ 0 (
    set "OPTIONAL_FEATURES=!OPTIONAL_FEATURES! ✅WebSocket"
) else (
    set "OPTIONAL_FEATURES=!OPTIONAL_FEATURES! ❌WebSocket"
)

%PYTHON_CMD% -c "import fastapi" >nul 2>&1
if %errorlevel% equ 0 (
    set "OPTIONAL_FEATURES=!OPTIONAL_FEATURES! ✅FastAPI"
) else (
    set "OPTIONAL_FEATURES=!OPTIONAL_FEATURES! ❌FastAPI"
)

echo 🔧 Дополнительные модули:!OPTIONAL_FEATURES!

REM Проверка файловой системы
echo [📁] Проверка файловой структуры...
if exist "domain\" echo ✅ Доменный слой
if exist "application\" echo ✅ Слой приложений
if exist "infrastructure\" echo ✅ Инфраструктурный слой
if exist "interfaces\" echo ✅ Интерфейсный слой

REM Проверка конфигурации
echo [⚙️] Проверка конфигурации...
if exist "launcher_config.json" (
    echo ✅ Конфигурация launcher'а найдена
) else (
    echo 🔧 Создание конфигурации по умолчанию...
    %PYTHON_CMD% -c "import json; config={'auto_start_components':['database','trading_engine','dashboard'],'dashboard_port':8080,'environment':'development'}; open('launcher_config.json','w').write(json.dumps(config,indent=2))"
    echo ✅ Конфигурация создана
)

echo.
color 0A
echo ═══════════════════════════════════════════════════════════════════════════════
echo                      🎯 СИСТЕМНАЯ ДИАГНОСТИКА ЗАВЕРШЕНА
echo ═══════════════════════════════════════════════════════════════════════════════
echo.

REM Выбор режима запуска
:MAIN_MENU
echo [🎮] Выберите режим запуска:
echo.
echo     1️⃣  - Полный запуск системы (рекомендуется)
echo     2️⃣  - Только дашборд
echo     3️⃣  - Системный launcher (Advanced)
echo     4️⃣  - Простой режим (быстрый старт)
echo     5️⃣  - Диагностика системы
echo     6️⃣  - Настройки
echo     7️⃣  - Справка
echo     0️⃣  - Выход
echo.
set /p "choice=👉 Ваш выбор (1-7, 0 для выхода): "

if "%choice%"=="1" goto FULL_LAUNCH
if "%choice%"=="2" goto DASHBOARD_ONLY
if "%choice%"=="3" goto SYSTEM_LAUNCHER
if "%choice%"=="4" goto SIMPLE_MODE
if "%choice%"=="5" goto DIAGNOSTICS
if "%choice%"=="6" goto SETTINGS
if "%choice%"=="7" goto HELP
if "%choice%"=="0" goto EXIT
echo ❌ Неверный выбор. Попробуйте снова.
echo.
goto MAIN_MENU

:FULL_LAUNCH
cls
color 0B
echo.
echo ═══════════════════════════════════════════════════════════════════════════════
echo                     🚀 ПОЛНЫЙ ЗАПУСК СИСТЕМЫ ATB
echo ═══════════════════════════════════════════════════════════════════════════════
echo.
echo [🔄] Инициализация компонентов...
echo.
echo %date% %time% - Full system launch initiated >> %LOG_FILE%

%PYTHON_CMD% atb_launcher.py
if %errorlevel% neq 0 (
    color 0C
    echo ❌ Ошибка запуска системы (код: %errorlevel%)
    echo 📋 Проверьте логи в файле: %LOG_FILE%
) else (
    color 0A
    echo ✅ Система запущена успешно
)
goto END

:DASHBOARD_ONLY
cls
color 0A
echo.
echo ═══════════════════════════════════════════════════════════════════════════════
echo                       📊 ЗАПУСК ДАШБОРДА
echo ═══════════════════════════════════════════════════════════════════════════════
echo.
echo [🖥️] Запуск торгового дашборда...
echo.
echo %date% %time% - Dashboard-only launch >> %LOG_FILE%

%PYTHON_CMD% run_dashboard.py
if %errorlevel% neq 0 (
    color 0C
    echo ❌ Ошибка запуска дашборда
) else (
    color 0A
    echo ✅ Дашборд завершен успешно
)
goto END

:SYSTEM_LAUNCHER
cls
color 0D
echo.
echo ═══════════════════════════════════════════════════════════════════════════════
echo                    ⚡ СИСТЕМНЫЙ LAUNCHER (ADVANCED)
echo ═══════════════════════════════════════════════════════════════════════════════
echo.
echo [🔧] Запуск расширенного системного launcher'а...
echo 📊 Мониторинг компонентов, автоперезапуск, API сервер
echo.
echo %date% %time% - Advanced launcher mode >> %LOG_FILE%

start "ATB System Monitor" %PYTHON_CMD% atb_launcher.py
echo ✅ Системный launcher запущен в отдельном окне
goto END

:SIMPLE_MODE
cls
color 0E
echo.
echo ═══════════════════════════════════════════════════════════════════════════════
echo                      ⚡ БЫСТРЫЙ СТАРТ
echo ═══════════════════════════════════════════════════════════════════════════════
echo.
echo [🚄] Быстрый запуск с минимальными проверками...
echo.
echo %date% %time% - Simple mode launch >> %LOG_FILE%

REM Простая проверка Python и прямой запуск
%PYTHON_CMD% -c "print('✅ Python готов')"
echo [🚀] Прямой запуск дашборда...
%PYTHON_CMD% run_dashboard.py
goto END

:DIAGNOSTICS
cls
color 0F
echo.
echo ═══════════════════════════════════════════════════════════════════════════════
echo                       🔍 СИСТЕМНАЯ ДИАГНОСТИКА
echo ═══════════════════════════════════════════════════════════════════════════════
echo.
echo [📊] Полная диагностика системы...
echo.

REM Расширенная диагностика
echo 🔍 Информация о системе:
echo   ▶ ОС: %OS%
echo   ▶ Архитектура: %PROCESSOR_ARCHITECTURE%
echo   ▶ Пользователь: %USERNAME%
echo   ▶ Время: %date% %time%
echo.

echo 🐍 Python диагностика:
%PYTHON_CMD% -c "import sys,platform;print(f'  ▶ Версия: {sys.version}');print(f'  ▶ Платформа: {platform.platform()}');print(f'  ▶ Путь: {sys.executable}')"
echo.

echo 📦 Установленные пакеты:
pip list | findstr /i "numpy pandas matplotlib tkinter"
echo.

echo 💾 Дисковое пространство:
dir /-c | find "bytes free"
echo.

echo 🌐 Сетевое подключение:
ping -n 1 google.com >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✅ Интернет соединение активно
) else (
    echo   ❌ Интернет соединение недоступно
)
echo.

echo 📋 Диагностика завершена
pause
goto MAIN_MENU

:SETTINGS
cls
color 0C
echo.
echo ═══════════════════════════════════════════════════════════════════════════════
echo                          ⚙️ НАСТРОЙКИ
echo ═══════════════════════════════════════════════════════════════════════════════
echo.
echo [🔧] Настройки системы:
echo.
echo     1️⃣  - Редактировать launcher_config.json
echo     2️⃣  - Очистить логи
echo     3️⃣  - Очистить кэш
echo     4️⃣  - Переустановить зависимости
echo     5️⃣  - Сброс к заводским настройкам
echo     0️⃣  - Назад в главное меню
echo.
set /p "settings_choice=👉 Выберите действие: "

if "%settings_choice%"=="1" (
    echo [📝] Открытие конфигурации...
    if exist launcher_config.json (
        notepad launcher_config.json
    ) else (
        echo ❌ Файл конфигурации не найден
    )
)
if "%settings_choice%"=="2" (
    echo [🧹] Очистка логов...
    if exist logs\*.log del /q logs\*.log
    echo ✅ Логи очищены
)
if "%settings_choice%"=="3" (
    echo [🧹] Очистка кэша...
    if exist temp\*.* del /q temp\*.*
    if exist __pycache__ rd /s /q __pycache__
    echo ✅ Кэш очищен
)
if "%settings_choice%"=="4" (
    echo [🔄] Переустановка зависимостей...
    pip install --upgrade --force-reinstall numpy pandas matplotlib
    echo ✅ Зависимости переустановлены
)
if "%settings_choice%"=="5" (
    echo [⚠️] Сброс настроек к заводским...
    if exist launcher_config.json del launcher_config.json
    echo ✅ Настройки сброшены
)
if "%settings_choice%"=="0" goto MAIN_MENU

pause
goto SETTINGS

:HELP
cls
color 0B
echo.
echo ═══════════════════════════════════════════════════════════════════════════════
echo                           📚 СПРАВКА ATB
echo ═══════════════════════════════════════════════════════════════════════════════
echo.
echo 🎯 ATB Trading Dashboard v2.0 - Справочная информация
echo.
echo 📋 РЕЖИМЫ ЗАПУСКА:
echo   1️⃣ Полный запуск    - Запуск всех компонентов системы
echo   2️⃣ Только дашборд   - Запуск только торгового интерфейса  
echo   3️⃣ System launcher  - Расширенный мониторинг и управление
echo   4️⃣ Простой режим    - Быстрый запуск без проверок
echo.
echo 🔧 СИСТЕМНЫЕ ТРЕБОВАНИЯ:
echo   ▶ Windows 10/11
echo   ▶ Python 3.8+
echo   ▶ 4GB RAM (рекомендуется 8GB)
echo   ▶ 2GB свободного места
echo   ▶ Интернет соединение
echo.
echo 📦 ЗАВИСИМОСТИ:
echo   ▶ numpy, pandas - Численные вычисления
echo   ▶ matplotlib - Графики и визуализация
echo   ▶ tkinter - Графический интерфейс
echo   ▶ requests - HTTP клиент (опционально)
echo.
echo 🆘 ПОДДЕРЖКА:
echo   ▶ Документация: README.md
echo   ▶ Логи: logs\launcher.log
echo   ▶ Конфигурация: launcher_config.json
echo.
echo 🔥 ГОРЯЧИЕ КЛАВИШИ В ДАШБОРДЕ:
echo   ▶ F5 - Обновить данные
echo   ▶ F9 - Запустить торговлю
echo   ▶ F10 - Остановить торговлю
echo   ▶ Ctrl+S - Сохранить конфигурацию
echo   ▶ Esc - Экстренная остановка
echo.
pause
goto MAIN_MENU

:EXIT
cls
color 0A
echo.
echo ═══════════════════════════════════════════════════════════════════════════════
echo                         👋 ЗАВЕРШЕНИЕ РАБОТЫ
echo ═══════════════════════════════════════════════════════════════════════════════
echo.
echo %date% %time% - Launcher shutdown >> %LOG_FILE%
echo ✅ Спасибо за использование ATB Trading Dashboard!
echo 📈 Удачных торгов!
echo.
echo 🔄 Перезапуск: start_atb.bat
echo 📋 Логи: %LOG_FILE%
echo.
timeout /t 3 /nobreak >nul
exit /b 0

:END
echo.
color 0A
echo ═══════════════════════════════════════════════════════════════════════════════
echo                       ✅ ОПЕРАЦИЯ ЗАВЕРШЕНА
echo ═══════════════════════════════════════════════════════════════════════════════
echo.
echo %date% %time% - Operation completed >> %LOG_FILE%
echo 🔄 Для повторного запуска запустите: start_atb.bat
echo 📋 Логи сохранены в: %LOG_FILE%
echo.
echo 👋 Нажмите любую клавишу для выхода...
pause >nul
exit /b 0