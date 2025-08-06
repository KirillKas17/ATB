# 📦 Зависимости ATB Trading System

## 🚀 Быстрая установка для Windows + Python 3.10.10

### Автоматическая установка
```cmd
# Запустите автоматический установщик
install_windows.bat
```

### Ручная установка
```cmd
# 1. Создайте виртуальное окружение
python -m venv atb_venv
atb_venv\Scripts\activate

# 2. Обновите pip
python -m pip install --upgrade pip

# 3. Полная установка
pip install -r requirements.txt

# ИЛИ минимальная установка при проблемах
pip install -r requirements_minimal.txt
```

## 📋 Основные компоненты

### Критически важные
- **pandas** - обработка данных
- **numpy** - математические вычисления  
- **loguru** - логирование
- **pydantic** - валидация данных
- **python-dotenv** - конфигурация

### GUI (десктопное приложение)
- **PyQt6** + **PyQt6-Charts** - графический интерфейс

### Веб API (дашборд)
- **FastAPI** + **uvicorn** - веб-сервер
- **aiohttp** - HTTP клиент
- **websockets** - WebSocket соединения

### Торговля и анализ
- **ccxt** - подключение к биржам
- **yfinance** - финансовые данные
- **TA-Lib** - технический анализ ⚠️
- **ta** - дополнительные индикаторы

### Машинное обучение
- **scikit-learn** - базовые алгоритмы
- **torch** - нейронные сети
- **xgboost** - градиентный бустинг
- **optuna** - оптимизация гиперпараметров

### Базы данных
- **SQLAlchemy** - ORM
- **psycopg2-binary** - PostgreSQL драйвер
- **redis** - кэширование

## ⚠️ Особые случаи

### TA-Lib на Windows
```cmd
# Метод 1: Предкомпилированные колеса
pip install TA-Lib --find-links https://github.com/cgohlke/windows-binaries/releases

# Метод 2: Вручную скачать .whl
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# TA_Lib‑0.4.28‑cp310‑cp310‑win_amd64.whl
pip install путь_к_файлу.whl
```

### PyTorch для разных систем
```cmd
# CPU версия
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA версия (если есть NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🔧 Системные требования

### Минимальные
- Python 3.10.10
- 8GB RAM
- 50GB свободного места
- Windows 10 1903+

### Рекомендуемые  
- Python 3.10.10
- 16GB+ RAM
- 100GB+ SSD
- Windows 11
- Visual Studio Build Tools

## 📝 Файлы зависимостей

- `requirements.txt` - полная установка (все компоненты)
- `requirements_minimal.txt` - минимальная установка (только критические)
- `requirements_enhanced.txt` - расширенная версия (устаревшая)
- `config/requirements.txt` - фиксированные версии

## 🆘 Проблемы и решения

### Ошибки компиляции
1. Установите Visual Studio Build Tools
2. Обновите pip и setuptools
3. Используйте предкомпилированные колеса

### Проблемы с PyQt6
```cmd
pip install PyQt6 --no-cache-dir --force-reinstall
set QT_QPA_PLATFORM_PLUGIN_PATH=%VIRTUAL_ENV%\Lib\site-packages\PyQt6\Qt6\plugins\platforms
```

### Конфликты версий
```cmd
pip install --upgrade --force-reinstall -r requirements_minimal.txt
```

## 📖 Подробные инструкции

- `WINDOWS_INSTALL.md` - полная инструкция для Windows
- `SETUP.md` - инструкции для всех ОС
- `TA_LIB_INSTALL.md` - специально для TA-Lib

## 🎯 Проверка установки

```cmd
# Активируйте окружение
atb_venv\Scripts\activate

# Проверьте ключевые компоненты
python -c "import pandas, numpy, loguru, PyQt6, fastapi; print('✅ Основные пакеты OK')"
python -c "import talib; print('✅ TA-Lib OK')"

# Запустите тест
python test_simple_coverage.py
```