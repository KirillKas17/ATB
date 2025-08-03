# 🚀 Установка и настройка Advanced Trading Bot

## 📋 Требования

### Системные требования
- **OS**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.10 или выше
- **RAM**: 8GB+ (рекомендуется 16GB)
- **Storage**: 50GB+ свободного места на SSD
- **Network**: Стабильное интернет-соединение

### Программное обеспечение
- **PostgreSQL**: 12+ 
- **Redis**: 6+
- **Git**: Последняя версия

## 🔧 Установка

### 1. Подготовка системы

#### Windows
```powershell
# Установка Python 3.10+
# Скачайте с https://www.python.org/downloads/

# Установка Git
# Скачайте с https://git-scm.com/download/win

# Установка PostgreSQL
# Скачайте с https://www.postgresql.org/download/windows/

# Установка Redis
# Скачайте с https://github.com/microsoftarchive/redis/releases
```

#### macOS
```bash
# Установка Homebrew (если не установлен)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Установка Python
brew install python@3.10

# Установка Git
brew install git

# Установка PostgreSQL
brew install postgresql
brew services start postgresql

# Установка Redis
brew install redis
brew services start redis
```

#### Ubuntu/Debian
```bash
# Обновление системы
sudo apt update && sudo apt upgrade -y

# Установка Python 3.10
sudo apt install python3.10 python3.10-venv python3.10-dev

# Установка Git
sudo apt install git

# Установка PostgreSQL
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Установка Redis
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

### 2. Клонирование репозитория

```bash
# Клонирование
git clone https://github.com/your-repo/advanced-trading-bot.git
cd advanced-trading-bot

# Проверка структуры
ls -la
```

### 3. Настройка Python окружения

```bash
# Создание виртуального окружения
python3.10 -m venv venv

# Активация окружения
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

# Проверка версии Python
python --version  # Должно быть 3.10+
```

### 4. Установка зависимостей

```bash
# Обновление pip
pip install --upgrade pip

# Установка зависимостей
pip install -r requirements.txt

# Установка TA-Lib (Windows)
# Скачайте соответствующий .whl файл и установите:
pip install TA_Lib-0.4.24-cp310-cp310-win_amd64.whl

# Установка TA-Lib (macOS/Linux)
# macOS:
brew install ta-lib
pip install TA-Lib

# Ubuntu:
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

### 5. Настройка базы данных

```bash
# Подключение к PostgreSQL
sudo -u postgres psql

# Создание пользователя и базы данных
CREATE USER trading_bot WITH PASSWORD 'your_secure_password';
CREATE DATABASE trading_bot OWNER trading_bot;
GRANT ALL PRIVILEGES ON DATABASE trading_bot TO trading_bot;
\q

# Инициализация базы данных
python scripts/init_db.py
```

### 6. Настройка Redis

```bash
# Проверка работы Redis
redis-cli ping  # Должен ответить PONG

# Настройка Redis (опционально)
sudo nano /etc/redis/redis.conf

# Основные настройки:
# maxmemory 2gb
# maxmemory-policy allkeys-lru
# save 900 1
# save 300 10
# save 60 10000
```

### 7. Настройка конфигурации

```bash
# Копирование примера конфигурации
cp config/config.example.yaml config/config.yaml

# Редактирование конфигурации
nano config/config.yaml
```

#### Основные настройки config.yaml:

```yaml
# Настройки биржи
exchange:
  name: "bybit"  # bybit, binance, okx, etc.
  api_key: "your_api_key_here"
  secret: "your_secret_here"
  testnet: true  # Использовать тестовую сеть

# Настройки базы данных
database:
  host: "localhost"
  port: 5432
  user: "trading_bot"
  password: "your_secure_password"
  database: "trading_bot"

# Настройки Redis
redis:
  host: "localhost"
  port: 6379
  password: ""
  database: 0

# Управление рисками
risk:
  max_risk_per_trade: 0.02      # 2% на сделку
  max_daily_loss: 0.05          # 5% в день
  max_weekly_loss: 0.15         # 15% в неделю
```

### 8. Проверка установки

```bash
# Проверка системы
python scripts/check_system.py

# Запуск тестов
pytest tests/ -v

# Проверка линтера
black .
isort .
flake8 .
mypy .
```

## 🚀 Запуск системы

### 1. Первый запуск

```bash
# Проверка системы перед запуском
python scripts/check_system.py

# Запуск системы
python scripts/main.py
```

### 2. Запуск в фоновом режиме

#### Linux/macOS
```bash
# Использование nohup
nohup python scripts/main.py > logs/trading_bot.log 2>&1 &

# Использование screen
screen -S trading_bot
python scripts/main.py
# Ctrl+A, D для отключения

# Использование tmux
tmux new-session -d -s trading_bot 'python scripts/main.py'
```

#### Windows
```powershell
# Запуск как служба Windows
# Создайте .bat файл:
# @echo off
# cd /d "C:\path\to\advanced-trading-bot"
# python scripts\main.py

# Или используйте Task Scheduler
```

### 3. Запуск с Docker

```bash
# Сборка образа
docker build -t advanced-trading-bot .

# Запуск контейнера
docker run -d \
  --name trading_bot \
  -p 8000:8000 \
  -p 8001:8001 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  advanced-trading-bot
```

## 📊 Мониторинг

### 1. Веб-дашборд

Откройте браузер и перейдите по адресу:
```
http://localhost:8000
```

### 2. Prometheus метрики

```
http://localhost:8001/metrics
```

### 3. Логи

```bash
# Просмотр логов в реальном времени
tail -f logs/trading_bot.log

# Поиск ошибок
grep "ERROR" logs/trading_bot.log

# Поиск торговых операций
grep "trade_executed" logs/trading_bot.log
```

### 4. API

```bash
# Проверка статуса системы
curl http://localhost:8000/api/v1/status

# Получение метрик
curl http://localhost:8000/api/v1/metrics

# Управление стратегиями
curl -X POST http://localhost:8000/api/v1/strategies/enable \
  -H "Content-Type: application/json" \
  -d '{"strategy": "trend_strategy"}'
```

## 🔧 Устранение неполадок

### Частые проблемы

#### 1. Ошибка подключения к базе данных
```bash
# Проверка статуса PostgreSQL
sudo systemctl status postgresql

# Проверка подключения
psql -h localhost -U trading_bot -d trading_bot

# Перезапуск PostgreSQL
sudo systemctl restart postgresql
```

#### 2. Ошибка подключения к Redis
```bash
# Проверка статуса Redis
sudo systemctl status redis-server

# Проверка подключения
redis-cli ping

# Перезапуск Redis
sudo systemctl restart redis-server
```

#### 3. Ошибки с TA-Lib
```bash
# Переустановка TA-Lib
pip uninstall TA-Lib
pip install TA-Lib

# Или установка из исходников
# См. инструкции выше
```

#### 4. Ошибки с зависимостями
```bash
# Очистка кэша pip
pip cache purge

# Переустановка зависимостей
pip install -r requirements.txt --force-reinstall
```

#### 5. Проблемы с правами доступа
```bash
# Проверка прав на директории
ls -la

# Установка правильных прав
chmod 755 scripts/
chmod 644 config/*.yaml
chmod 755 logs/
```

### Логи и отладка

```bash
# Включение отладочного режима
# В config.yaml установите:
development:
  debug: true
  verbose: true

# Просмотр детальных логов
tail -f logs/trading_bot.log | grep DEBUG

# Проверка использования ресурсов
htop
df -h
free -h
```

## 🔒 Безопасность

### 1. Защита API ключей

```bash
# Шифрование ключей
# Используйте переменные окружения:
export EXCHANGE_API_KEY="your_api_key"
export EXCHANGE_SECRET="your_secret"

# Или используйте .env файл:
echo "EXCHANGE_API_KEY=your_api_key" > .env
echo "EXCHANGE_SECRET=your_secret" >> .env
```

### 2. Настройка файрвола

```bash
# Ubuntu/Debian
sudo ufw allow 8000/tcp  # Веб-интерфейс
sudo ufw allow 8001/tcp  # Prometheus метрики
sudo ufw enable

# macOS
sudo pfctl -e
# Настройте правила в /etc/pf.conf
```

### 3. SSL/TLS сертификаты

```bash
# Получение сертификата Let's Encrypt
sudo apt install certbot
sudo certbot certonly --standalone -d your-domain.com

# Настройка в config.yaml:
dashboard:
  ssl:
    enabled: true
    cert_file: "/etc/letsencrypt/live/your-domain.com/fullchain.pem"
    key_file: "/etc/letsencrypt/live/your-domain.com/privkey.pem"
```

## 📈 Оптимизация производительности

### 1. Настройка PostgreSQL

```sql
-- Подключение к PostgreSQL
sudo -u postgres psql

-- Настройка производительности
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- Перезапуск PostgreSQL
sudo systemctl restart postgresql
```

### 2. Настройка Redis

```bash
# Редактирование конфигурации Redis
sudo nano /etc/redis/redis.conf

# Основные настройки производительности:
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
tcp-keepalive 300
```

### 3. Настройка системы

```bash
# Увеличение лимитов файлов
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Настройка swappiness
echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## 🔄 Обновление системы

```bash
# Остановка системы
python scripts/main.py --stop

# Обновление кода
git pull origin main

# Обновление зависимостей
pip install -r requirements.txt --upgrade

# Обновление базы данных (если необходимо)
python scripts/init_db.py

# Перезапуск системы
python scripts/main.py
```

## 📞 Поддержка

Если у вас возникли проблемы:

1. **Проверьте логи**: `tail -f logs/trading_bot.log`
2. **Запустите проверку системы**: `python scripts/check_system.py`
3. **Создайте issue** на GitHub с подробным описанием проблемы
4. **Обратитесь в Telegram**: [@atb_support](https://t.me/atb_support)

---

**🎉 Поздравляем! Система установлена и готова к работе!** 