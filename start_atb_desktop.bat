@echo off
REM ATB Trading System Desktop Application Launcher
REM Запуск современного Windows приложения для торговой системы ATB

echo.
echo ========================================
echo   ATB Trading System Desktop App
echo   Professional Edition v2.0
echo ========================================
echo.

REM Проверка наличия Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python не найден в системе
    echo Установите Python с https://python.org
    pause
    exit /b 1
)

REM Проверка виртуального окружения
if not exist "venv\Scripts\activate.bat" (
    echo Создание виртуального окружения...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Не удалось создать виртуальное окружение
        pause
        exit /b 1
    )
)

REM Активация виртуального окружения
echo Активация виртуального окружения...
call venv\Scripts\activate.bat

REM Проверка и установка зависимостей
echo Проверка зависимостей...
pip show PyQt6 >nul 2>&1
if errorlevel 1 (
    echo Установка PyQt6...
    pip install PyQt6 PyQt6-Charts PyQt6-WebEngine
    if errorlevel 1 (
        echo ERROR: Не удалось установить PyQt6
        pause
        exit /b 1
    )
)

REM Установка остальных зависимостей
echo Установка зависимостей...
pip install -r requirements.txt

REM Запуск приложения
echo.
echo Запуск ATB Trading System Desktop Application...
echo.

REM Запуск с параметрами по умолчанию (симуляция)
python start_atb_desktop.py --mode simulation

REM Если произошла ошибка
if errorlevel 1 (
    echo.
    echo ERROR: Приложение завершилось с ошибкой
    echo Проверьте логи в папке logs/
    pause
    exit /b 1
)

echo.
echo Приложение завершено
pause 