@echo off
chcp 65001 >nul
echo ========================================
echo    ATB Dashboard Launcher
echo ========================================
echo.

:: Проверка наличия Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: Python не найден в системе!
    echo Установите Python с https://python.org
    pause
    exit /b 1
)

:: Проверка наличия pip
pip --version >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: pip не найден!
    pause
    exit /b 1
)

:: Переход в корневую папку проекта
cd /d "%~dp0"

:: Проверка наличия виртуального окружения
if not exist "venv\Scripts\activate.bat" (
    echo Создание виртуального окружения...
    python -m venv venv
    if errorlevel 1 (
        echo ОШИБКА: Не удалось создать виртуальное окружение!
        pause
        exit /b 1
    )
)

:: Активация виртуального окружения
echo Активация виртуального окружения...
call venv\Scripts\activate.bat

:: Установка зависимостей
echo Установка зависимостей...
pip install -r requirements.txt >nul 2>&1
if errorlevel 1 (
    echo Предупреждение: Не удалось установить зависимости из requirements.txt
    echo Устанавливаем основные пакеты...
    pip install dash plotly fastapi uvicorn websockets
)

:: Проверка наличия файлов дашборда
if not exist "interfaces\presentation\dashboard\index.html" (
    echo ОШИБКА: Файлы дашборда не найдены!
    echo Проверьте структуру проекта
    pause
    exit /b 1
)

:: Запуск HTTP сервера для статического дашборда
echo.
echo Запуск дашборда...
echo URL: http://localhost:8080
echo.
echo Нажмите Ctrl+C для остановки
echo.

:: Запуск Python HTTP сервера
cd interfaces\presentation\dashboard
python -m http.server 8080

pause