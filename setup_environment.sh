#!/bin/bash

# ============================================================================
# ATB Trading System - Environment Setup Script
# ============================================================================
# Скрипт для настройки рабочего окружения

set -e  # Прекратить выполнение при ошибке

echo "🚀 Настройка окружения ATB Trading System..."

# ============================================================================
# 1. ПРОВЕРКА СИСТЕМНЫХ ТРЕБОВАНИЙ
# ============================================================================
echo "📋 Проверка системных требований..."

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 не найден. Установите Python 3.8+ и повторите."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python $PYTHON_VERSION найден"

# Проверка pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 не найден. Установите pip и повторите."
    exit 1
fi

echo "✅ pip3 найден"

# ============================================================================
# 2. УСТАНОВКА ЗАВИСИМОСТЕЙ
# ============================================================================
echo "📦 Установка Python зависимостей..."

# Создание виртуального окружения (если возможно)
if python3 -m venv --help &> /dev/null; then
    echo "🔧 Создание виртуального окружения..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo "✅ Виртуальное окружение создано"
    else
        echo "✅ Виртуальное окружение уже существует"
    fi
    
    # Активация виртуального окружения
    source venv/bin/activate
    echo "✅ Виртуальное окружение активировано"
    
    # Установка зависимостей в виртуальном окружении
    pip install --upgrade pip
    pip install pandas numpy requests python-dotenv
else
    echo "⚠️  Устанавливаю в системные пакеты (не рекомендуется для продакшена)"
    pip3 install --break-system-packages pandas numpy requests python-dotenv || {
        echo "❌ Не удалось установить зависимости. Попробуйте:"
        echo "   sudo apt update && sudo apt install python3-pandas python3-numpy python3-requests"
        exit 1
    }
fi

echo "✅ Основные зависимости установлены"

# ============================================================================
# 3. НАСТРОЙКА КОНФИГУРАЦИИ
# ============================================================================
echo "⚙️  Настройка конфигурационных файлов..."

# Создание .env файла
if [ ! -f ".env" ]; then
    if [ -f "env.example" ]; then
        cp env.example .env
        echo "✅ Создан .env файл из env.example"
        echo "⚠️  ВНИМАНИЕ: Отредактируйте .env файл перед запуском!"
    else
        echo "❌ Файл env.example не найден"
        exit 1
    fi
else
    echo "✅ .env файл уже существует"
fi

# Создание директорий для логов
mkdir -p logs
mkdir -p models
mkdir -p data/training
echo "✅ Созданы необходимые директории"

# ============================================================================
# 4. ПРОВЕРКА СИНТАКСИСА
# ============================================================================
echo "🔍 Проверка синтаксиса критических файлов..."

SYNTAX_ERRORS=0

# Список файлов для проверки
FILES_TO_CHECK=(
    "main.py"
    "infrastructure/core/feature_engineering.py"
    "infrastructure/core/technical.py"
    "infrastructure/core/technical_analysis.py"
)

for file in "${FILES_TO_CHECK[@]}"; do
    if [ -f "$file" ]; then
        if python3 -m py_compile "$file" 2>/dev/null; then
            echo "✅ $file - синтаксис OK"
        else
            echo "❌ $file - есть синтаксические ошибки"
            SYNTAX_ERRORS=$((SYNTAX_ERRORS + 1))
        fi
    else
        echo "⚠️  $file - файл не найден"
    fi
done

# ============================================================================
# 5. СОЗДАНИЕ СКРИПТА ЗАПУСКА
# ============================================================================
echo "📝 Создание скрипта запуска..."

cat > run_trading_system.sh << 'EOF'
#!/bin/bash

# Скрипт запуска торговой системы
echo "🚀 Запуск ATB Trading System..."

# Активация виртуального окружения (если существует)
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Виртуальное окружение активировано"
fi

# Проверка .env файла
if [ ! -f ".env" ]; then
    echo "❌ .env файл не найден! Запустите setup_environment.sh"
    exit 1
fi

# Запуск системы
echo "🔄 Запуск торговой системы..."
python3 main.py

EOF

chmod +x run_trading_system.sh
echo "✅ Создан скрипт запуска: run_trading_system.sh"

# ============================================================================
# 6. ФИНАЛЬНАЯ ПРОВЕРКА
# ============================================================================
echo "🎯 Финальная проверка настройки..."

if [ $SYNTAX_ERRORS -eq 0 ]; then
    echo "✅ Все проверки пройдены успешно!"
    echo ""
    echo "📋 СЛЕДУЮЩИЕ ШАГИ:"
    echo "1. Отредактируйте .env файл с вашими API ключами"
    echo "2. Установите PostgreSQL и Redis (если нужно)"
    echo "3. Запустите: ./run_trading_system.sh"
    echo ""
    echo "⚠️  ВАЖНО: Используйте тестовые API ключи для начала!"
else
    echo "❌ Найдено $SYNTAX_ERRORS синтаксических ошибок"
    echo "🔧 Исправьте ошибки перед запуском системы"
    exit 1
fi

echo "🎉 Настройка окружения завершена!"