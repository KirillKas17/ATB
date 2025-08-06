# 🪟 Установка ATB Trading System на Windows

## 📋 Системные требования

- **OS**: Windows 10 версии 1903 или выше (рекомендуется Windows 11)
- **Python**: 3.10.10 (точно эта версия для максимальной совместимости)
- **RAM**: 16GB+ (минимум 8GB)
- **Storage**: 100GB+ свободного места на SSD
- **Visual Studio Build Tools**: Обязательно для компиляции некоторых пакетов

## 🔧 Пошаговая установка

### 1. Подготовка системы

#### Установка Python 3.10.10
```cmd
# Скачайте точно версию 3.10.10 с официального сайта
# https://www.python.org/downloads/release/python-31010/
# Выберите "Windows installer (64-bit)" для 64-битной системы

# ВАЖНО: При установке обязательно отметьте:
# ✅ Add Python to PATH
# ✅ Install for all users (если нужно)
# ✅ pip
# ✅ tcl/tk and IDLE
```

#### Установка Visual Studio Build Tools (ОБЯЗАТЕЛЬНО!)
```cmd
# Скачайте Build Tools for Visual Studio 2022
# https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

# При установке выберите:
# ✅ C++ build tools
# ✅ Windows 10/11 SDK (последняя версия)
# ✅ CMake tools for Visual Studio
```

#### Установка Git
```cmd
# Скачайте с https://git-scm.com/download/win
# При установке оставьте настройки по умолчанию
```

### 2. Установка TA-Lib (КРИТИЧЕСКИ ВАЖНО!)

TA-Lib требует специальной установки на Windows:

```cmd
# Метод 1: Предкомпилированные колеса (РЕКОМЕНДУЕТСЯ)
pip install TA-Lib --find-links https://github.com/cgohlke/windows-binaries/releases

# Метод 2: Если метод 1 не работает, скачайте .whl файл вручную
# Перейдите на https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# Скачайте файл для Python 3.10 и Windows 64-bit:
# TA_Lib‑0.4.28‑cp310‑cp310‑win_amd64.whl
# Затем установите:
pip install путь_к_файлу/TA_Lib‑0.4.28‑cp310‑cp310‑win_amd64.whl

# Метод 3: Компиляция из исходников (ТОЛЬКО если методы 1-2 не работают)
# Скачайте исходники TA-Lib C library с http://ta-lib.org/hdr_dw.html
# Распакуйте в C:\ta-lib
# Затем:
pip install TA-Lib
```

### 3. Установка основных зависимостей

```cmd
# Создайте виртуальное окружение
python -m venv atb_venv

# Активируйте окружение
atb_venv\Scripts\activate

# Обновите pip до последней версии
python -m pip install --upgrade pip

# Установите wheel для предкомпилированных пакетов
pip install wheel

# Установите зависимости (с обработкой ошибок)
pip install --upgrade setuptools
pip install -r requirements.txt
```

### 4. Установка PostgreSQL и Redis

#### PostgreSQL
```cmd
# Скачайте PostgreSQL 15+ с https://www.postgresql.org/download/windows/
# При установке:
# - Запомните пароль для пользователя postgres
# - Порт оставьте 5432
# - Локаль выберите "C"

# После установки создайте базу данных:
# Откройте pgAdmin или используйте psql
createdb -U postgres atb_trading
```

#### Redis
```cmd
# Для Windows используйте WSL или Docker:

# Вариант 1: Docker (РЕКОМЕНДУЕТСЯ)
docker run -d -p 6379:6379 redis:latest

# Вариант 2: WSL2
wsl --install
# После перезагрузки:
wsl
sudo apt update
sudo apt install redis-server
redis-server

# Вариант 3: Memurai (Redis-совместимый для Windows)
# Скачайте с https://www.memurai.com/
```

### 5. Проверка установки

```cmd
# Активируйте окружение
atb_venv\Scripts\activate

# Проверьте Python и ключевые пакеты
python -c "import sys; print(f'Python {sys.version}')"
python -c "import talib; print('TA-Lib OK')"
python -c "import pandas; print('Pandas OK')"
python -c "import numpy; print('NumPy OK')"
python -c "import torch; print('PyTorch OK')"
python -c "import PyQt6; print('PyQt6 OK')"

# Запустите базовый тест
python test_simple_coverage.py
```

### 6. Возможные проблемы и решения

#### Ошибка при установке TA-Lib
```cmd
# Если получаете ошибки компиляции:
# 1. Убедитесь, что установлены Visual Studio Build Tools
# 2. Попробуйте установить Microsoft C++ Redistributable:
#    https://aka.ms/vs/17/release/vc_redist.x64.exe
# 3. Используйте предкомпилированные колеса (см. выше)
```

#### Ошибки с PyQt6
```cmd
# Если PyQt6 не устанавливается:
pip install PyQt6 --no-cache-dir --force-reinstall

# Если есть проблемы с Qt платформой:
set QT_QPA_PLATFORM_PLUGIN_PATH=%VIRTUAL_ENV%\Lib\site-packages\PyQt6\Qt6\plugins\platforms
```

#### Проблемы с torch
```cmd
# Для CPU-версии PyTorch:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Для CUDA (если есть NVIDIA GPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Ошибки с компиляцией других пакетов
```cmd
# Установите Microsoft Visual C++ 14.0:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Или используйте conda для проблемных пакетов:
conda install -c conda-forge package_name
```

### 7. Переменные окружения

Создайте файл `.env` в корне проекта:
```env
# База данных
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/atb_trading

# Redis
REDIS_URL=redis://localhost:6379

# Логирование
LOG_LEVEL=INFO
LOG_DIR=logs

# API ключи (замените на реальные)
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key

# Windows-специфичные настройки
PYTHONIOENCODING=utf-8
PYTHONPATH=.
```

### 8. Запуск приложения

```cmd
# Активируйте окружение
atb_venv\Scripts\activate

# Запустите десктопное приложение
python start_atb_desktop.py

# Или запустите веб-дашборд
python run_dashboard.py

# Или запустите полную торговую систему
python main.py
```

## 🔧 Автоматическая установка

Используйте предоставленные батники:

```cmd
# Полная автоматическая установка
start_atb.bat

# Только десктопное приложение
start_atb_desktop.bat

# Только дашборд
start_dashboard.bat
```

## 📝 Важные заметки

1. **Всегда используйте виртуальное окружение** - это предотвратит конфликты пакетов
2. **TA-Lib на Windows сложна** - следуйте инструкциям выше точно
3. **Visual Studio Build Tools обязательны** для компиляции некоторых пакетов
4. **Используйте 64-битную версию Python** для лучшей производительности
5. **Регулярно обновляйте pip и setuptools** перед установкой пакетов

## 🆘 Получение помощи

Если возникают проблемы:
1. Проверьте версию Python: `python --version` (должно быть 3.10.10)
2. Проверьте pip: `pip --version`
3. Проверьте виртуальное окружение: `which python` (должно указывать на venv)
4. Сохраните лог ошибок и обратитесь за помощью