@echo off
chcp 65001 >nul
title ⚡ ATB Trading System - Electron Desktop Launcher v3.1

echo.
echo ===============================================================================
echo                ⚡ ATB Trading System - Electron Desktop v3.1
echo ===============================================================================
echo.
echo 🚀 Запуск современного Electron приложения для торговли...
echo.
echo ✨ Технологии:
echo • ⚛️ Electron - Кроссплатформенное десктопное приложение
echo • 🟨 Node.js - Серверная часть с реальными данными
echo • 🌐 HTML/CSS/JS - Современный веб интерфейс
echo • 📊 Chart.js - Интерактивные графики в реальном времени
echo • 🖥️ systeminformation - Мониторинг системных ресурсов
echo • 🧬 Интеграция с Python системой эволюции
echo.

:: Проверка наличия Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Ошибка: Node.js не найден!
    echo 📥 Пожалуйста, установите Node.js LTS с https://nodejs.org
    pause
    exit /b 1
)

echo ✅ Node.js найден
node --version

:: Проверка наличия npm
npm --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Ошибка: npm не найден!
    echo 📥 Пожалуйста, переустановите Node.js
    pause
    exit /b 1
)

echo ✅ npm найден
npm --version
echo.

:: Проверка package.json
if not exist "package.json" (
    echo ❌ Ошибка: package.json не найден!
    echo 📝 Убедитесь, что вы запускаете из корневой папки проекта
    pause
    exit /b 1
)

echo ✅ package.json найден
echo.

:: Установка зависимостей если node_modules не существует
if not exist "node_modules" (
    echo 📦 Установка зависимостей...
    npm install
    if errorlevel 1 (
        echo ❌ Ошибка: Не удалось установить зависимости!
        echo 💡 Попробуйте: npm install --force
        pause
        exit /b 1
    )
    echo ✅ Зависимости установлены
) else (
    echo ✅ Зависимости уже установлены
)
echo.

:: Проверка Electron
npm list electron --depth=0 >nul 2>&1
if errorlevel 1 (
    echo 📦 Установка Electron...
    npm install electron --save-dev
    if errorlevel 1 (
        echo ❌ Ошибка: Не удалось установить Electron!
        pause
        exit /b 1
    )
)

echo ✅ Electron готов
echo.

:: Создание .env файла если его нет
if not exist ".env" (
    echo 🔧 Создание конфигурационного файла .env...
    (
        echo # ATB Trading System Electron Configuration
        echo NODE_ENV=production
        echo.
        echo # Общие настройки
        echo ENVIRONMENT=development
        echo DEBUG=true
        echo ATB_MODE=simulation
        echo.
        echo # Сервер
        echo BACKEND_PORT=3001
        echo FRONTEND_PORT=3000
        echo.
        echo # База данных
        echo DB_HOST=localhost
        echo DB_PORT=5432
        echo DB_NAME=atb_trading
        echo DB_USER=atb_user
        echo DB_PASS=
        echo.
        echo # Биржа
        echo EXCHANGE_API_KEY=
        echo EXCHANGE_API_SECRET=
        echo EXCHANGE_TESTNET=true
        echo.
        echo # Мониторинг
        echo MONITORING_ENABLED=true
        echo MONITORING_INTERVAL=10000
        echo ALERT_EMAIL=
        echo.
        echo # Эволюция
        echo EVOLUTION_ENABLED=true
        echo EVOLUTION_INTERVAL=3600000
        echo AUTO_EVOLUTION=false
        echo.
        echo # Торговля
        echo DEFAULT_POSITION_SIZE=1.0
        echo DEFAULT_STOP_LOSS=2.0
        echo MAX_DRAWDOWN=5.0
        echo.
        echo # Логирование
        echo LOG_LEVEL=info
        echo LOG_FILE=logs/atb.log
        echo.
        echo # Безопасность
        echo ENABLE_CORS=true
        echo JWT_SECRET=your-secret-key
    ) > .env
    echo ✅ Конфигурационный файл создан
)

:: Создание необходимых директорий
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "assets" mkdir assets
if not exist "renderer" mkdir renderer
if not exist "backend" mkdir backend

:: Создание простых иконок если их нет
if not exist "assets\icon.png" (
    echo 🎨 Создание базовых ресурсов...
    :: Создаем простой текстовый файл вместо иконки для тестирования
    echo ATB > assets\icon.png
    echo ATB > assets\tray-icon.png
)

echo.
echo 🎯 Запуск ATB Trading System Electron Desktop v3.1...
echo.

:: Установка переменных окружения
set NODE_ENV=production
set ELECTRON_IS_DEV=0

:: Запуск Electron приложения
npm start

if errorlevel 1 (
    echo.
    echo ❌ Ошибка запуска Electron приложения!
    echo.
    echo 💡 Возможные решения:
    echo • Убедитесь, что установлены все зависимости: npm install
    echo • Проверьте файл main.js на наличие ошибок
    echo • Установите Electron глобально: npm install -g electron
    echo • Проверьте настройки в файле .env
    echo • Запустите в режиме разработки: npm run dev
    echo.
    echo 🔧 Для отладки запустите:
    echo npm run dev
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ ATB Trading System Electron завершен успешно
echo.
pause