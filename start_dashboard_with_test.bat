@echo off
chcp 65001 >nul
title ATB Dashboard Launcher with Data Testing

echo ========================================
echo    ATB Dashboard Launcher v2.0
echo    С проверкой передачи данных
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
pip install fastapi uvicorn aiohttp websockets >nul 2>&1
if errorlevel 1 (
    echo Предупреждение: Не удалось установить некоторые зависимости
    echo Продолжаем с базовыми пакетами...
)

:: Проверка наличия файлов дашборда
if not exist "interfaces\presentation\dashboard\index.html" (
    echo ОШИБКА: Файлы дашборда не найдены!
    echo Проверьте структуру проекта
    pause
    exit /b 1
)

:: Проверка наличия скрипта запуска
if not exist "run_dashboard.py" (
    echo ОШИБКА: Скрипт run_dashboard.py не найден!
    pause
    exit /b 1
)

:: Запуск дашборда в фоновом режиме
echo.
echo Запуск дашборда...
echo URL: http://localhost:8000
echo API документация: http://localhost:8000/docs
echo.
echo Ожидание запуска сервера...
start /B python run_dashboard.py

:: Ожидание запуска сервера
timeout /t 5 /nobreak >nul

:: Проверка доступности сервера
echo Проверка доступности сервера...
for /l %%i in (1,1,10) do (
    curl -s http://localhost:8000/api/health >nul 2>&1
    if not errorlevel 1 (
        echo ✅ Сервер запущен и доступен
        goto :test_data
    )
    timeout /t 1 /nobreak >nul
)

echo ❌ Сервер не отвечает
echo Проверьте логи выше
pause
exit /b 1

:test_data
echo.
echo ========================================
echo    ПРОВЕРКА ПЕРЕДАЧИ ДАННЫХ
echo ========================================
echo.

:: Проверка наличия скрипта тестирования
if exist "test_dashboard_data.py" (
    echo Запуск тестирования данных...
    python test_dashboard_data.py
    echo.
    echo Тестирование завершено
    echo Результаты сохранены в файл dashboard_test_results_*.json
) else (
    echo Предупреждение: Скрипт тестирования не найден
    echo Создаем базовую проверку...
    
    :: Простая проверка API
    echo Проверка API endpoints:
    curl -s http://localhost:8000/api/status | python -m json.tool
    echo.
    curl -s http://localhost:8000/api/trading | python -m json.tool
    echo.
    curl -s http://localhost:8000/api/positions | python -m json.tool
    echo.
    curl -s http://localhost:8000/api/analytics | python -m json.tool
)

echo.
echo ========================================
echo    ДАШБОРД ГОТОВ К РАБОТЕ
echo ========================================
echo.
echo Откройте браузер и перейдите по адресу:
echo http://localhost:8000
echo.
echo Для остановки сервера нажмите Ctrl+C
echo.
echo Нажмите любую клавишу для открытия в браузере...
pause >nul

:: Открытие в браузере
start http://localhost:8000

echo.
echo Дашборд открыт в браузере
echo Сервер продолжает работать в фоновом режиме
echo.
echo Для полной остановки закройте это окно
pause 