#!/bin/bash

# ATB Trading Dashboard - Quick Start
# Простой запуск без меню и проверок

echo "🚀 ATB Trading Dashboard - Quick Start"
echo "═══════════════════════════════════════"

# Поиск Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python не найден!"
    echo "💡 Установите: sudo apt install python3"
    exit 1
fi

echo "✅ Используется: $PYTHON_CMD"

# Попытка запуска
echo "🔄 Запуск системы..."
if [ -f "atb_launcher.py" ]; then
    echo "📁 Найден atb_launcher.py"
    $PYTHON_CMD atb_launcher.py
elif [ -f "run_dashboard.py" ]; then
    echo "📁 Найден run_dashboard.py"
    $PYTHON_CMD run_dashboard.py
else
    echo "❌ Файлы запуска не найдены!"
    echo "💡 Используйте полный launcher: ./start_atb_venv.sh"
    exit 1
fi

echo "✅ Завершено"