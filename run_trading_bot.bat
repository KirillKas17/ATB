@echo off
echo Запуск ATB Trading System с эволюцией стратегий...
echo.

REM Активация виртуального окружения
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo Виртуальное окружение активировано
) else (
    echo Виртуальное окружение не найдено, используем системный Python
)

REM Проверка наличия Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Ошибка: Python не найден в системе
    pause
    exit /b 1
)

REM Запуск основной системы (включает эволюцию в фоне)
echo Запуск основной торговой системы с эволюцией...
python main.py

if errorlevel 1 (
    echo Ошибка при запуске торговой системы
    pause
    exit /b 1
)

echo Торговая система завершена
pause 