@echo off
title ATB Trading System v2.0 - System Launcher
color 0A
chcp 65001 >nul

echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                        ATB TRADING SYSTEM v2.0                              ║
echo ║                         Система запуска                                     ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.

REM Установка переменных
set PYTHON_CMD=python
set SCRIPT_NAME=atb_launcher.py
set PROJECT_DIR=%~dp0

echo [INFO] Рабочая директория: %PROJECT_DIR%
echo [INFO] Скрипт запуска: %SCRIPT_NAME%
echo.

REM Проверка Python
echo [CHECK] Проверка наличия Python...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python не найден в PATH!
    echo.
    echo Пожалуйста, установите Python 3.8+ с официального сайта:
    echo https://www.python.org/downloads/
    echo.
    echo Убедитесь, что при установке выбрана опция "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

REM Получение версии Python
for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo [OK] Python найден: %PYTHON_VERSION%

REM Проверка наличия launcher скрипта
if not exist "%PROJECT_DIR%%SCRIPT_NAME%" (
    echo [ERROR] Файл %SCRIPT_NAME% не найден!
    echo [INFO] Убедитесь, что вы запускаете из корня проекта ATB Trading
    echo.
    pause
    exit /b 1
)

echo [OK] Launcher скрипт найден
echo.

REM Проверка и создание виртуального окружения (опционально)
if exist "%PROJECT_DIR%venv" (
    echo [INFO] Найдено виртуальное окружение, активация...
    call "%PROJECT_DIR%venv\Scripts\activate.bat"
    if %errorlevel% neq 0 (
        echo [WARNING] Не удалось активировать виртуальное окружение
        echo [INFO] Продолжаем с системным Python
    ) else (
        echo [OK] Виртуальное окружение активировано
    )
)

REM Обновление pip (тихо)
echo [INFO] Проверка pip...
python -m pip install --upgrade pip --quiet --no-warn-script-location

REM Проверка основных зависимостей
echo [INFO] Проверка основных зависимостей...
python -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"

REM Попытка установки основных библиотек
echo [INFO] Установка/обновление основных библиотек...
python -m pip install --quiet --no-warn-script-location ^
    numpy ^
    pandas ^
    matplotlib ^
    requests ^
    aiohttp ^
    websockets

if %errorlevel% neq 0 (
    echo [WARNING] Некоторые библиотеки не удалось установить
    echo [INFO] Система попытается работать с доступными компонентами
)

echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                            ЗАПУСК СИСТЕМЫ                                   ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.

REM Создание папки логов
if not exist "%PROJECT_DIR%logs" mkdir "%PROJECT_DIR%logs"

REM Сохранение времени запуска
echo %date% %time% - Запуск ATB Trading System >> "%PROJECT_DIR%logs\startup.log"

echo [LAUNCH] Запуск ATB Trading System Launcher...
echo [INFO] Логи сохраняются в папку logs/
echo [INFO] Для остановки используйте Ctrl+C или закройте окно
echo.

REM Запуск основного launcher'а
cd /d "%PROJECT_DIR%"
python "%SCRIPT_NAME%"

REM Обработка завершения
set LAUNCHER_EXIT_CODE=%errorlevel%

echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                          ЗАВЕРШЕНИЕ РАБОТЫ                                  ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝

if %LAUNCHER_EXIT_CODE% equ 0 (
    echo [SUCCESS] Система завершена успешно
    echo %date% %time% - Успешное завершение >> "%PROJECT_DIR%logs\startup.log"
) else (
    echo [ERROR] Система завершена с ошибкой (код: %LAUNCHER_EXIT_CODE%)
    echo %date% %time% - Завершение с ошибкой %LAUNCHER_EXIT_CODE% >> "%PROJECT_DIR%logs\startup.log"
    echo.
    echo [HELP] Возможные причины ошибок:
    echo   - Отсутствуют зависимости Python (numpy, pandas, matplotlib)
    echo   - Заняты порты системы (8000, 8080)
    echo   - Недостаточно прав доступа
    echo   - Антивирус блокирует выполнение
    echo.
    echo [SOLUTION] Попробуйте:
    echo   1. Запустить от имени администратора
    echo   2. Проверить логи в папке logs/
    echo   3. Установить зависимости: pip install numpy pandas matplotlib
    echo   4. Перезагрузить компьютер
)

echo.
echo [INFO] Время работы системы записано в logs\startup.log
echo [INFO] Подробные логи системы в logs\launcher.log
echo.

REM Предложение просмотра логов при ошибке
if %LAUNCHER_EXIT_CODE% neq 0 (
    set /p view_logs="Показать последние логи? (y/n): "
    if /i "!view_logs!"=="y" (
        echo.
        echo ===== ПОСЛЕДНИЕ ЛОГИ =====
        if exist "%PROJECT_DIR%logs\launcher.log" (
            powershell -command "Get-Content '%PROJECT_DIR%logs\launcher.log' | Select-Object -Last 20"
        ) else (
            echo Файл логов не найден
        )
        echo ===========================
    )
)

echo.
echo Нажмите любую клавишу для выхода...
pause >nul

REM Очистка переменных
set PYTHON_CMD=
set SCRIPT_NAME=
set PROJECT_DIR=
set PYTHON_VERSION=
set LAUNCHER_EXIT_CODE=

exit /b %LAUNCHER_EXIT_CODE%