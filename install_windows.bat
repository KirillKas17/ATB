@echo off
chcp 65001 >nul
echo.
echo ███████╗████████╗██████╗     ████████╗██████╗  █████╗ ██████╗ ██╗███╗   ██╗ ██████╗ 
echo ██╔══██║╚══██╔══╝██╔══██╗    ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝ 
echo ███████║   ██║   ██████╔╝       ██║   ██████╔╝███████║██║  ██║██║██╔██╗ ██║██║  ███╗
echo ██╔══██║   ██║   ██╔══██╗       ██║   ██╔══██╗██╔══██║██║  ██║██║██║╚██╗██║██║   ██║
echo ██║  ██║   ██║   ██████╔╝       ██║   ██║  ██║██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝
echo ╚═╝  ╚═╝   ╚═╝   ╚═════╝        ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝ 
echo.
echo =============================================================================
echo    Автоматическая установка ATB Trading System для Windows
echo    Python 3.10.10 + специальная обработка TA-Lib
echo =============================================================================
echo.

:: Проверка версии Python
echo [1/10] Проверка версии Python...
python --version | findstr "3.10" >nul
if errorlevel 1 (
    echo ❌ ОШИБКА: Требуется Python 3.10.x!
    echo Скачайте Python 3.10.10 с https://www.python.org/downloads/release/python-31010/
    echo.
    pause
    exit /b 1
)
echo ✅ Python версия корректная

:: Проверка pip
echo.
echo [2/10] Проверка pip...
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ОШИБКА: pip не найден!
    echo Переустановите Python с включением pip
    pause
    exit /b 1
)
echo ✅ pip найден

:: Обновление pip
echo.
echo [3/10] Обновление pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ⚠️ Предупреждение: Не удалось обновить pip, продолжаем...
)

:: Создание виртуального окружения
echo.
echo [4/10] Создание виртуального окружения...
if exist "atb_venv" (
    echo Виртуальное окружение уже существует, пропускаем...
) else (
    python -m venv atb_venv
    if errorlevel 1 (
        echo ❌ ОШИБКА: Не удалось создать виртуальное окружение!
        pause
        exit /b 1
    )
)
echo ✅ Виртуальное окружение готово

:: Активация виртуального окружения
echo.
echo [5/10] Активация виртуального окружения...
call atb_venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ ОШИБКА: Не удалось активировать виртуальное окружение!
    pause
    exit /b 1
)
echo ✅ Виртуальное окружение активировано

:: Установка wheel и setuptools
echo.
echo [6/10] Установка базовых инструментов...
pip install --upgrade setuptools wheel
if errorlevel 1 (
    echo ⚠️ Предупреждение: Проблемы с установкой базовых инструментов
)

:: Попытка установки TA-Lib
echo.
echo [7/10] Специальная установка TA-Lib для Windows...
echo Пробуем метод 1: Предкомпилированные колеса...
pip install TA-Lib --find-links https://github.com/cgohlke/windows-binaries/releases
if errorlevel 1 (
    echo Метод 1 не сработал, пробуем метод 2...
    pip install TA-Lib --no-cache-dir
    if errorlevel 1 (
        echo ⚠️ ВНИМАНИЕ: TA-Lib не установлен автоматически!
        echo Скачайте .whl файл вручную с https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
        echo Для Python 3.10 Windows 64-bit: TA_Lib-0.4.28-cp310-cp310-win_amd64.whl
        echo Затем выполните: pip install путь_к_файлу.whl
        echo.
        echo Нажмите любую клавишу для продолжения установки остальных пакетов...
        pause >nul
    ) else (
        echo ✅ TA-Lib установлен методом 2
    )
) else (
    echo ✅ TA-Lib установлен методом 1
)

:: Выбор типа установки
echo.
echo [8/10] Выбор типа установки...
echo.
echo Выберите тип установки:
echo 1) Полная установка (все зависимости)
echo 2) Минимальная установка (только критические пакеты)
echo 3) Пропустить установку зависимостей
echo.
set /p choice="Введите номер (1-3): "

if "%choice%"=="1" (
    echo Устанавливаем полные зависимости...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ⚠️ Ошибки при полной установке, попробуем минимальную...
        pip install -r requirements_minimal.txt
    ) else (
        echo ✅ Полная установка завершена
    )
) else if "%choice%"=="2" (
    echo Устанавливаем минимальные зависимости...
    pip install -r requirements_minimal.txt
    if errorlevel 1 (
        echo ❌ ОШИБКА: Не удалось установить даже минимальные зависимости!
        echo Проверьте подключение к интернету и права доступа
        pause
        exit /b 1
    ) else (
        echo ✅ Минимальная установка завершена
    )
) else if "%choice%"=="3" (
    echo Пропускаем установку зависимостей...
) else (
    echo Некорректный выбор, используем минимальную установку...
    pip install -r requirements_minimal.txt
)

:: Проверка установки
echo.
echo [9/10] Проверка ключевых компонентов...
echo Проверяем pandas...
python -c "import pandas; print('✅ Pandas OK')" 2>nul || echo "❌ Pandas FAILED"

echo Проверяем numpy...
python -c "import numpy; print('✅ NumPy OK')" 2>nul || echo "❌ NumPy FAILED"

echo Проверяем loguru...
python -c "import loguru; print('✅ Loguru OK')" 2>nul || echo "❌ Loguru FAILED"

echo Проверяем PyQt6...
python -c "import PyQt6; print('✅ PyQt6 OK')" 2>nul || echo "❌ PyQt6 FAILED"

echo Проверяем TA-Lib...
python -c "import talib; print('✅ TA-Lib OK')" 2>nul || echo "⚠️ TA-Lib не установлен (см. инструкции выше)"

echo Проверяем FastAPI...
python -c "import fastapi; print('✅ FastAPI OK')" 2>nul || echo "❌ FastAPI FAILED"

:: Создание .env файла
echo.
echo [10/10] Создание файла конфигурации...
if not exist ".env" (
    echo # ATB Trading System Configuration > .env
    echo DATABASE_URL=postgresql://postgres:password@localhost:5432/atb_trading >> .env
    echo REDIS_URL=redis://localhost:6379 >> .env
    echo LOG_LEVEL=INFO >> .env
    echo LOG_DIR=logs >> .env
    echo PYTHONIOENCODING=utf-8 >> .env
    echo PYTHONPATH=. >> .env
    echo ✅ Файл .env создан
) else (
    echo ✅ Файл .env уже существует
)

:: Завершение
echo.
echo =============================================================================
echo                          🎉 УСТАНОВКА ЗАВЕРШЕНА!
echo =============================================================================
echo.
echo Что дальше:
echo.
echo 1. Активируйте виртуальное окружение:
echo    atb_venv\Scripts\activate
echo.
echo 2. Запустите приложение:
echo    python start_atb_desktop.py     (десктопное приложение)
echo    python run_dashboard.py         (веб-дашборд)
echo    python main.py                  (полная торговая система)
echo.
echo 3. Если TA-Lib не установился:
echo    - Скачайте .whl файл с https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
echo    - Установите: pip install путь_к_файлу.whl
echo.
echo 4. Настройте базу данных PostgreSQL и Redis (см. WINDOWS_INSTALL.md)
echo.
echo 📖 Подробные инструкции: WINDOWS_INSTALL.md
echo.
echo Нажмите любую клавишу для выхода...
pause >nul