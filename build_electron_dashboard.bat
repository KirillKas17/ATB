@echo off
chcp 65001 >nul
title ATB Electron Dashboard Builder

echo ========================================
echo    ATB Electron Dashboard Builder
echo ========================================
echo.

:: Проверка наличия Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: Node.js не найден в системе!
    echo Установите Node.js с https://nodejs.org
    pause
    exit /b 1
)

:: Проверка наличия npm
npm --version >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: npm не найден!
    pause
    exit /b 1
)

:: Переход в папку Electron приложения
cd /d "%~dp0electron_dashboard"

:: Проверка наличия package.json
if not exist "package.json" (
    echo ОШИБКА: package.json не найден!
    echo Проверьте структуру проекта
    pause
    exit /b 1
)

:: Установка зависимостей
echo Установка зависимостей...
call npm install
if errorlevel 1 (
    echo ОШИБКА: Не удалось установить зависимости!
    pause
    exit /b 1
)

:: Создание иконки если её нет
if not exist "assets\icon.ico" (
    echo Создание папки assets...
    mkdir assets 2>nul
    
    echo Создание заглушки иконки...
    echo ^<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256"^> > assets\icon.svg
    echo   ^<rect width="256" height="256" fill="#007AFF"/^> >> assets\icon.svg
    echo   ^<text x="128" y="140" font-family="Arial" font-size="80" fill="white" text-anchor="middle"^>ATB^</text^> >> assets\icon.svg
    echo ^</svg^> >> assets\icon.svg
    
    echo Конвертация SVG в ICO...
    echo Для полной функциональности установите ImageMagick и конвертируйте icon.svg в icon.ico
)

:: Создание папок если их нет
if not exist "renderer\styles" mkdir renderer\styles
if not exist "renderer\js" mkdir renderer\js
if not exist "renderer\assets" mkdir renderer\assets

:: Копирование файлов из веб-версии если их нет
if not exist "renderer\styles\main.css" (
    echo Копирование стилей...
    copy "..\interfaces\presentation\dashboard\style.css" "renderer\styles\main.css" >nul 2>&1
)

:: Создание базовых стилей если их нет
if not exist "renderer\styles\main.css" (
    echo Создание базовых стилей...
    echo /* ATB Dashboard Styles */ > renderer\styles\main.css
    echo body { margin: 0; padding: 0; font-family: 'SF Pro Display', sans-serif; } >> renderer\styles\main.css
    echo .app-container { display: flex; flex-direction: column; height: 100vh; } >> renderer\styles\main.css
)

if not exist "renderer\styles\components.css" (
    echo Создание стилей компонентов...
    echo /* Component Styles */ > renderer\styles\components.css
    echo .btn { padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; } >> renderer\styles\components.css
    echo .btn.primary { background: #007AFF; color: white; } >> renderer\styles\components.css
)

if not exist "renderer\styles\charts.css" (
    echo Создание стилей графиков...
    echo /* Chart Styles */ > renderer\styles\charts.css
    echo .chart-container { position: relative; height: 300px; } >> renderer\styles\charts.css
)

:: Создание JavaScript файлов если их нет
if not exist "renderer\js\app.js" (
    echo Создание основного JavaScript...
    echo // ATB Dashboard App > renderer\js\app.js
    echo console.log('ATB Dashboard loaded'); >> renderer\js\app.js
)

if not exist "renderer\js\charts.js" (
    echo Создание JavaScript для графиков...
    echo // Charts functionality > renderer\js\charts.js
    echo console.log('Charts module loaded'); >> renderer\js\charts.js
)

if not exist "renderer\js\data.js" (
    echo Создание JavaScript для данных...
    echo // Data handling > renderer\js\data.js
    echo console.log('Data module loaded'); >> renderer\js\data.js
)

if not exist "renderer\js\ui.js" (
    echo Создание JavaScript для UI...
    echo // UI interactions > renderer\js\ui.js
    echo console.log('UI module loaded'); >> renderer\js\ui.js
)

echo.
echo ========================================
echo    ВЫБОР ДЕЙСТВИЯ
echo ========================================
echo.
echo 1. Запустить в режиме разработки
echo 2. Собрать приложение для Windows
echo 3. Запустить собранное приложение
echo 4. Очистить сборку
echo.
set /p choice="Выберите действие (1-4): "

if "%choice%"=="1" goto :dev
if "%choice%"=="2" goto :build
if "%choice%"=="3" goto :run
if "%choice%"=="4" goto :clean
goto :invalid

:dev
echo.
echo Запуск в режиме разработки...
call npm run dev
goto :end

:build
echo.
echo Сборка приложения для Windows...
call npm run build-win
if errorlevel 1 (
    echo ОШИБКА: Сборка не удалась!
    pause
    exit /b 1
)
echo.
echo Сборка завершена успешно!
echo Файл установщика: dist\ATB Trading Dashboard Setup.exe
goto :end

:run
echo.
echo Запуск собранного приложения...
if exist "dist\win-unpacked\ATB Trading Dashboard.exe" (
    start "" "dist\win-unpacked\ATB Trading Dashboard.exe"
) else (
    echo Приложение не собрано. Сначала выполните сборку.
    pause
)
goto :end

:clean
echo.
echo Очистка сборки...
if exist "dist" rmdir /s /q dist
if exist "node_modules" rmdir /s /q node_modules
echo Очистка завершена.
goto :end

:invalid
echo Неверный выбор!
goto :end

:end
echo.
echo Нажмите любую клавишу для выхода...
pause >nul 