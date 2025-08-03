@echo off
echo ========================================
echo    Очистка проекта ATB
echo ========================================
echo.

REM Активация виртуального окружения
if exist "venv\Scripts\activate.bat" (
    echo Активация виртуального окружения...
    call venv\Scripts\activate.bat
) else (
    echo Виртуальное окружение не найдено
    pause
    exit /b 1
)

echo.
echo Выберите действие:
echo 1. Полная очистка проекта
echo 2. Только удаление неиспользуемых импортов
echo 3. Только форматирование кода
echo 4. Только проверка типов
echo 5. Выход
echo.

set /p choice="Введите номер (1-5): "

if "%choice%"=="1" (
    echo.
    echo Запуск полной очистки...
    python scripts\clean_project.py
) else if "%choice%"=="2" (
    echo.
    echo Удаление неиспользуемых импортов...
    python scripts\clean_project.py --imports-only
) else if "%choice%"=="3" (
    echo.
    echo Форматирование кода...
    python scripts\clean_project.py --format-only
) else if "%choice%"=="4" (
    echo.
    echo Проверка типов...
    python scripts\clean_project.py --types-only
) else if "%choice%"=="5" (
    echo Выход...
    exit /b 0
) else (
    echo Неверный выбор!
    pause
    exit /b 1
)

echo.
echo Очистка завершена!
pause 