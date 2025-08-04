#!/bin/bash

# ATB Trading Dashboard v2.0 - Linux Launcher
set -e

# Настройка переменных
APP_NAME="ATB Trading Dashboard"
APP_VERSION="v2.0"
PYTHON_MIN_VERSION="3.8"
LOG_FILE="logs/launcher.log"

# Создание директорий
mkdir -p logs data config temp backups

# Начало логирования
echo "$(date) - Launcher started" >> "$LOG_FILE"

# Очистка экрана и заголовок
clear
echo
echo "                    ███████╗██╗   ██╗███╗   ██╗████████╗██████╗  █████╗ "
echo "                    ██╔════╝╚██╗ ██╔╝████╗  ██║╚══██╔══╝██╔══██╗██╔══██╗"
echo "                    ███████╗ ╚████╔╝ ██╔██╗ ██║   ██║   ██████╔╝███████║"
echo "                    ╚════██║  ╚██╔╝  ██║╚██╗██║   ██║   ██╔══██╗██╔══██║"
echo "                    ███████║   ██║   ██║ ╚████║   ██║   ██║  ██║██║  ██║"
echo "                    ╚══════╝   ╚═╝   ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝"
echo
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "                    🚀 $APP_NAME $APP_VERSION 🚀"
echo "                         Advanced Trading Platform Launcher"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo

# Функция проверки Python
check_python() {
    echo "[⚡] Проверка системных требований..."
    echo
    
    # Поиск Python
    PYTHON_CMD=""
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo "❌ ОШИБКА: Python не найден в системе!"
        echo
        echo "📥 Для работы системы требуется Python $PYTHON_MIN_VERSION+"
        echo "🌐 Установите Python: sudo apt install python3 python3-pip"
        echo "🔧 Или используйте менеджер пакетов вашей системы"
        echo
        exit 1
    fi
    
    PYTHON_VER=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    echo "✅ Python найден: $PYTHON_VER (команда: $PYTHON_CMD)"
    echo "$(date) - Python version: $PYTHON_VER" >> "$LOG_FILE"
    
    # Экспорт команды Python для глобального использования
    export PYTHON_CMD
}

# Проверка базовых зависимостей
check_base_modules() {
    echo "[🔍] Проверка базовых модулей..."
    if ! $PYTHON_CMD -c "import sys, os, asyncio, logging, json, threading, decimal, datetime, pathlib" 2>/dev/null; then
        echo "❌ Критическая ошибка: Отсутствуют базовые модули Python!"
        echo "🔧 Переустановите Python с полным набором модулей"
        exit 1
    fi
    echo "✅ Базовые модули проверены"
}

# Проверка GUI модулей
check_gui() {
    echo "[🖥️] Проверка GUI компонентов..."
    if ! $PYTHON_CMD -c "import tkinter" 2>/dev/null; then
        echo "⚠️ WARNING: Tkinter недоступен"
        echo "📱 GUI интерфейс будет ограничен"
        echo "💡 Установите: sudo apt install python3-tk"
    else
        echo "✅ Tkinter готов к работе"
    fi
}

# Проверка научных библиотек
check_scientific_libs() {
    echo "[📊] Проверка аналитических библиотек..."
    MISSING_LIBS=""
    
    if ! $PYTHON_CMD -c "import numpy" 2>/dev/null; then
        MISSING_LIBS="$MISSING_LIBS numpy"
    fi
    
    if ! $PYTHON_CMD -c "import pandas" 2>/dev/null; then
        MISSING_LIBS="$MISSING_LIBS pandas"
    fi
    
    if ! $PYTHON_CMD -c "import matplotlib" 2>/dev/null; then
        MISSING_LIBS="$MISSING_LIBS matplotlib"
    fi
    
    if [ -n "$MISSING_LIBS" ]; then
        echo "⚠️ Отсутствующие библиотеки:$MISSING_LIBS"
        echo
        echo "[💿] Автоматическая установка зависимостей..."
        echo "🔄 Установка может занять несколько минут..."
        echo
        
        $PYTHON_CMD -m pip install --upgrade pip
        for lib in $MISSING_LIBS; do
            echo "📦 Установка $lib..."
            if $PYTHON_CMD -m pip install "$lib"; then
                echo "✅ $lib установлен успешно"
            else
                echo "❌ Ошибка установки $lib"
            fi
        done
        echo
        echo "🔄 Повторная проверка зависимостей..."
        check_scientific_libs
        return
    fi
    
    echo "✅ Все библиотеки готовы к работе"
}

# Проверка дополнительных зависимостей
check_optional_features() {
    echo "[🧪] Проверка расширенного функционала..."
    OPTIONAL_FEATURES=""
    
    if $PYTHON_CMD -c "import requests" 2>/dev/null; then
        OPTIONAL_FEATURES="$OPTIONAL_FEATURES ✅API-клиент"
    else
        OPTIONAL_FEATURES="$OPTIONAL_FEATURES ❌API-клиент"
    fi
    
    if $PYTHON_CMD -c "import websocket" 2>/dev/null; then
        OPTIONAL_FEATURES="$OPTIONAL_FEATURES ✅WebSocket"
    else
        OPTIONAL_FEATURES="$OPTIONAL_FEATURES ❌WebSocket"
    fi
    
    if $PYTHON_CMD -c "import fastapi" 2>/dev/null; then
        OPTIONAL_FEATURES="$OPTIONAL_FEATURES ✅FastAPI"
    else
        OPTIONAL_FEATURES="$OPTIONAL_FEATURES ❌FastAPI"
    fi
    
    echo "🔧 Дополнительные модули:$OPTIONAL_FEATURES"
}

# Проверка файловой структуры
check_file_structure() {
    echo "[📁] Проверка файловой структуры..."
    [ -d "domain" ] && echo "✅ Доменный слой"
    [ -d "application" ] && echo "✅ Слой приложений"
    [ -d "infrastructure" ] && echo "✅ Инфраструктурный слой"
    [ -d "interfaces" ] && echo "✅ Интерфейсный слой"
}

# Проверка конфигурации
check_config() {
    echo "[⚙️] Проверка конфигурации..."
    if [ -f "launcher_config.json" ]; then
        echo "✅ Конфигурация launcher'а найдена"
    else
        echo "🔧 Создание конфигурации по умолчанию..."
        $PYTHON_CMD -c "
import json
config = {
    'auto_start_components': ['database', 'trading_engine', 'dashboard'],
    'dashboard_port': 8080,
    'environment': 'development'
}
with open('launcher_config.json', 'w') as f:
    json.dump(config, f, indent=2)
"
        echo "✅ Конфигурация создана"
    fi
}

# Главное меню
main_menu() {
    while true; do
        echo
        echo "═══════════════════════════════════════════════════════════════════════════════"
        echo "                     🎯 СИСТЕМНАЯ ДИАГНОСТИКА ЗАВЕРШЕНА"
        echo "═══════════════════════════════════════════════════════════════════════════════"
        echo
        echo "[🎮] Выберите режим запуска:"
        echo
        echo "    1️⃣  - Полный запуск системы (рекомендуется)"
        echo "    2️⃣  - Только дашборд"
        echo "    3️⃣  - Системный launcher (Advanced)"
        echo "    4️⃣  - Простой режим (быстрый старт)"
        echo "    5️⃣  - Диагностика системы"
        echo "    6️⃣  - Настройки"
        echo "    7️⃣  - Справка"
        echo "    0️⃣  - Выход"
        echo
        read -p "👉 Ваш выбор (1-7, 0 для выхода): " choice
        
        case $choice in
            1) full_launch ;;
            2) dashboard_only ;;
            3) system_launcher ;;
            4) simple_mode ;;
            5) diagnostics ;;
            6) settings ;;
            7) help ;;
            0) exit_launcher ;;
            *) echo "❌ Неверный выбор. Попробуйте снова." ;;
        esac
    done
}

# Полный запуск системы
full_launch() {
    clear
    echo
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "                    🚀 ПОЛНЫЙ ЗАПУСК СИСТЕМЫ ATB"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo
    echo "[🔄] Инициализация компонентов..."
    echo
    echo "$(date) - Full system launch initiated" >> "$LOG_FILE"
    
    if $PYTHON_CMD atb_launcher.py; then
        echo "✅ Система запущена успешно"
    else
        echo "❌ Ошибка запуска системы (код: $?)"
        echo "📋 Проверьте логи в файле: $LOG_FILE"
    fi
    end_operation
}

# Только дашборд
dashboard_only() {
    clear
    echo
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "                      📊 ЗАПУСК ДАШБОРДА"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo
    echo "[🖥️] Запуск торгового дашборда..."
    echo
    echo "$(date) - Dashboard-only launch" >> "$LOG_FILE"
    
    if $PYTHON_CMD run_dashboard.py; then
        echo "✅ Дашборд завершен успешно"
    else
        echo "❌ Ошибка запуска дашборда"
    fi
    end_operation
}

# Системный launcher
system_launcher() {
    clear
    echo
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "                   ⚡ СИСТЕМНЫЙ LAUNCHER (ADVANCED)"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo
    echo "[🔧] Запуск расширенного системного launcher'а..."
    echo "📊 Мониторинг компонентов, автоперезапуск, API сервер"
    echo
    echo "$(date) - Advanced launcher mode" >> "$LOG_FILE"
    
    # Запуск в фоне
    $PYTHON_CMD atb_launcher.py &
    echo "✅ Системный launcher запущен в фоне (PID: $!)"
    end_operation
}

# Простой режим
simple_mode() {
    clear
    echo
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "                     ⚡ БЫСТРЫЙ СТАРТ"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo
    echo "[🚄] Быстрый запуск с минимальными проверками..."
    echo
    echo "$(date) - Simple mode launch" >> "$LOG_FILE"
    
    # Простая проверка Python и прямой запуск
    $PYTHON_CMD -c "print('✅ Python готов')"
    echo "[🚀] Прямой запуск дашборда..."
    $PYTHON_CMD run_dashboard.py
    end_operation
}

# Диагностика
diagnostics() {
    clear
    echo
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "                      🔍 СИСТЕМНАЯ ДИАГНОСТИКА"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo
    echo "[📊] Полная диагностика системы..."
    echo
    
    # Расширенная диагностика
    echo "🔍 Информация о системе:"
    echo "  ▶ ОС: $(uname -s) $(uname -r)"
    echo "  ▶ Архитектура: $(uname -m)"
    echo "  ▶ Пользователь: $(whoami)"
    echo "  ▶ Время: $(date)"
    echo
    
    echo "🐍 Python диагностика:"
    $PYTHON_CMD -c "
import sys, platform
print(f'  ▶ Версия: {sys.version}')
print(f'  ▶ Платформа: {platform.platform()}')
print(f'  ▶ Путь: {sys.executable}')
"
    echo
    
    echo "📦 Установленные пакеты:"
    $PYTHON_CMD -m pip list | grep -E "(numpy|pandas|matplotlib|tkinter)" || echo "  Основные пакеты не найдены"
    echo
    
    echo "💾 Дисковое пространство:"
    df -h . | tail -1
    echo
    
    echo "🌐 Сетевое подключение:"
    if ping -c 1 google.com >/dev/null 2>&1; then
        echo "  ✅ Интернет соединение активно"
    else
        echo "  ❌ Интернет соединение недоступно"
    fi
    echo
    
    echo "📋 Диагностика завершена"
    read -p "Нажмите Enter для продолжения..."
}

# Настройки
settings() {
    while true; do
        clear
        echo
        echo "═══════════════════════════════════════════════════════════════════════════════"
        echo "                         ⚙️ НАСТРОЙКИ"
        echo "═══════════════════════════════════════════════════════════════════════════════"
        echo
        echo "[🔧] Настройки системы:"
        echo
        echo "    1️⃣  - Редактировать launcher_config.json"
        echo "    2️⃣  - Очистить логи"
        echo "    3️⃣  - Очистить кэш"
        echo "    4️⃣  - Переустановить зависимости"
        echo "    5️⃣  - Сброс к заводским настройкам"
        echo "    0️⃣  - Назад в главное меню"
        echo
        read -p "👉 Выберите действие: " settings_choice
        
        case $settings_choice in
            1)
                echo "[📝] Открытие конфигурации..."
                if [ -f "launcher_config.json" ]; then
                    ${EDITOR:-nano} launcher_config.json
                else
                    echo "❌ Файл конфигурации не найден"
                fi
                ;;
            2)
                echo "[🧹] Очистка логов..."
                rm -f logs/*.log
                echo "✅ Логи очищены"
                ;;
            3)
                echo "[🧹] Очистка кэша..."
                rm -rf temp/* __pycache__ .pytest_cache
                find . -name "*.pyc" -delete
                echo "✅ Кэш очищен"
                ;;
            4)
                echo "[🔄] Переустановка зависимостей..."
                $PYTHON_CMD -m pip install --upgrade --force-reinstall numpy pandas matplotlib
                echo "✅ Зависимости переустановлены"
                ;;
            5)
                echo "[⚠️] Сброс настроек к заводским..."
                rm -f launcher_config.json
                echo "✅ Настройки сброшены"
                ;;
            0)
                return
                ;;
            *)
                echo "❌ Неверный выбор"
                ;;
        esac
        
        if [ "$settings_choice" != "0" ]; then
            read -p "Нажмите Enter для продолжения..."
        fi
    done
}

# Справка
help() {
    clear
    echo
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "                          📚 СПРАВКА ATB"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo
    echo "🎯 ATB Trading Dashboard v2.0 - Справочная информация"
    echo
    echo "📋 РЕЖИМЫ ЗАПУСКА:"
    echo "  1️⃣ Полный запуск    - Запуск всех компонентов системы"
    echo "  2️⃣ Только дашборд   - Запуск только торгового интерфейса"
    echo "  3️⃣ System launcher  - Расширенный мониторинг и управление"
    echo "  4️⃣ Простой режим    - Быстрый запуск без проверок"
    echo
    echo "🔧 СИСТЕМНЫЕ ТРЕБОВАНИЯ:"
    echo "  ▶ Linux/Unix система"
    echo "  ▶ Python 3.8+"
    echo "  ▶ 4GB RAM (рекомендуется 8GB)"
    echo "  ▶ 2GB свободного места"
    echo "  ▶ Интернет соединение"
    echo
    echo "📦 ЗАВИСИМОСТИ:"
    echo "  ▶ numpy, pandas - Численные вычисления"
    echo "  ▶ matplotlib - Графики и визуализация"
    echo "  ▶ tkinter - Графический интерфейс"
    echo "  ▶ requests - HTTP клиент (опционально)"
    echo
    echo "🆘 ПОДДЕРЖКА:"
    echo "  ▶ Документация: README.md"
    echo "  ▶ Логи: logs/launcher.log"
    echo "  ▶ Конфигурация: launcher_config.json"
    echo
    echo "🔥 ГОРЯЧИЕ КЛАВИШИ В ДАШБОРДЕ:"
    echo "  ▶ F5 - Обновить данные"
    echo "  ▶ F9 - Запустить торговлю"
    echo "  ▶ F10 - Остановить торговлю"
    echo "  ▶ Ctrl+S - Сохранить конфигурацию"
    echo "  ▶ Esc - Экстренная остановка"
    echo
    read -p "Нажмите Enter для продолжения..."
}

# Завершение операции
end_operation() {
    echo
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "                      ✅ ОПЕРАЦИЯ ЗАВЕРШЕНА"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo
    echo "$(date) - Operation completed" >> "$LOG_FILE"
    echo "🔄 Для повторного запуска запустите: ./start_atb.sh"
    echo "📋 Логи сохранены в: $LOG_FILE"
    echo
    read -p "👋 Нажмите Enter для возврата в главное меню..."
}

# Выход
exit_launcher() {
    clear
    echo
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "                        👋 ЗАВЕРШЕНИЕ РАБОТЫ"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo
    echo "$(date) - Launcher shutdown" >> "$LOG_FILE"
    echo "✅ Спасибо за использование ATB Trading Dashboard!"
    echo "📈 Удачных торгов!"
    echo
    echo "🔄 Перезапуск: ./start_atb.sh"
    echo "📋 Логи: $LOG_FILE"
    echo
    sleep 3
    exit 0
}

# Главная функция
main() {
    # Проверки системы
    check_python
    check_base_modules
    check_gui
    check_scientific_libs
    check_optional_features
    check_file_structure
    check_config
    
    # Главное меню
    main_menu
}

# Запуск
main "$@"