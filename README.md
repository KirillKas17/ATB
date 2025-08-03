# Syntra 🚀

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/status-active-success.svg)](https://github.com/your-repo/syntra)
[![Architecture: DDD](https://img.shields.io/badge/architecture-DDD-green.svg)](https://martinfowler.com/bliki/DomainDrivenDesign.html)
[![Type Safety: MyPy](https://img.shields.io/badge/type%20safety-mypy-blue.svg)](https://mypy.readthedocs.io/)

**Продвинутый автономный торговый бот с Domain-Driven Design архитектурой, ИИ, машинным обучением и эволюционной системой для криптовалютных бирж**

## 🌟 Ключевые особенности

### 🏗️ Domain-Driven Design Архитектура
- **Четкое разделение слоев**: Domain, Application, Infrastructure, Interfaces
- **Бизнес-логика в центре**: Все торговые решения основаны на доменной модели
- **Чистая архитектура**: Независимость от внешних зависимостей
- **Типобезопасность**: Полная поддержка MyPy и статической типизации
- **SOLID принципы**: Следование лучшим практикам разработки

### 🧠 Искусственный интеллект и ML
- **Локальный ИИ-контроллер** - принятие решений на основе всех данных
- **Transformer модели** - прогнозирование цен и паттернов
- **Эволюционные агенты** - самообучающиеся компоненты
- **Pattern Discovery** - автоматическое обнаружение новых паттернов
- **Neural Noise Analysis** - анализ нейронного шума рынка
- **Entanglement Detection** - обнаружение квантовой запутанности в рынке

### 🔄 Эволюционная система
- **Непрерывная эволюция** - все компоненты постоянно улучшаются
- **Валидация эффективности** - изменения применяются только при подтвержденном улучшении
- **Автоматическая миграция** - плавный переход между версиями
- **Статистическая значимость** - научный подход к подтверждению улучшений
- **Backup и восстановление** - сохранение состояния системы

### 🛡️ Продвинутое управление рисками
- **Circuit Breaker** - защита от каскадных сбоев
- **Динамический риск-менеджмент** - адаптивные лимиты
- **Корреляционный анализ** - управление портфелем
- **Liquidity Gravity** - анализ ликвидности и гравитации рынка
- **Health Checker** - мониторинг состояния системы

### 📊 Мультистратегийность
- **20+ торговых стратегий** - от трендовых до арбитражных
- **Адаптивные режимы** - автоматическое переключение стратегий
- **Market Maker стратегии** - создание ликвидности
- **Manipulation Detection** - обнаружение манипуляций
- **Pairs Trading** - статистический арбитраж

## 🏗️ Архитектура

```
Syntra/
├── 🧠 domain/                          # Доменный слой (бизнес-логика)
│   ├── entities/                       # Доменные сущности
│   ├── value_objects/                  # Объекты-значения
│   ├── services/                       # Доменные сервисы
│   ├── repositories/                   # Интерфейсы репозиториев
│   ├── strategies/                     # Торговые стратегии
│   ├── sessions/                       # Торговые сессии
│   ├── intelligence/                   # ИИ компоненты
│   ├── evolution/                      # Эволюционная система
│   ├── memory/                         # Система памяти
│   ├── market/                         # Рыночные модели
│   ├── prediction/                     # Прогнозирование
│   └── types/                          # Типы данных
├── 🔧 application/                     # Прикладной слой
│   ├── use_cases/                      # Сценарии использования
│   ├── services/                       # Прикладные сервисы
│   ├── orchestration/                  # Оркестрация
│   ├── di_container_refactored.py      # DI контейнер
│   ├── monitoring/                     # Мониторинг
│   ├── analysis/                       # Аналитика
│   ├── signal/                         # Обработка сигналов
│   └── filters/                        # Фильтры
├── 🔌 infrastructure/                  # Инфраструктурный слой
│   ├── agents/                         # Эволюционные агенты
│   ├── core/                           # Основные компоненты
│   ├── exchange/                       # Адаптеры бирж
│   ├── strategies/                     # Реализации стратегий
│   ├── ml_services/                    # ML сервисы
│   ├── repositories/                   # Реализации репозиториев
│   ├── monitoring/                     # Мониторинг
│   ├── simulation/                     # Симуляция и бэктестинг
│   └── external_services/              # Внешние сервисы
├── 🖥️ interfaces/                      # Интерфейсный слой
│   ├── presentation/                   # Веб-интерфейс
│   └── api/                            # REST API
├── 🔧 shared/                          # Общие компоненты
│   ├── models/                         # Общие модели
│   ├── abstractions/                   # Абстракции
│   └── utils/                          # Утилиты
├── 📚 examples/                        # Примеры использования
├── 🧪 tests/                           # Тесты
├── ⚙️ config/                          # Конфигурация
└── 📊 dashboard/                       # Веб-дашборд
```

## 🚀 Быстрый старт

### Требования
- **Python 3.10+**
- **PostgreSQL 12+** (опционально)
- **Redis 6+** (опционально)
- **8GB+ RAM**
- **SSD диск**

### Установка

```bash
# 1. Клонирование репозитория
git clone https://github.com/your-repo/syntra.git
cd syntra

# 2. Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# 3. Установка зависимостей
pip install -r requirements.txt

# 4. Установка TA-Lib (Windows)
pip install TA_Lib-0.4.24-cp310-cp310-win_amd64.whl

# 5. Настройка конфигурации
cp env.example .env
# Отредактируйте .env файл

# 6. Запуск системы
python main.py
```

### Конфигурация

```yaml
# config/application.yaml
trading:
  exchange: "bybit"
  testnet: true
  pairs:
    - "BTC/USDT"
    - "ETH/USDT"
    - "BNB/USDT"

risk:
  max_risk_per_trade: 0.02
  max_daily_loss: 0.05
  max_weekly_loss: 0.15
  circuit_breaker_enabled: true

evolution:
  enabled: true
  efficiency_threshold: 0.05
  min_test_samples: 50
  statistical_significance: 0.05

strategies:
  enabled:
    - "trend_strategy"
    - "mean_reversion_strategy"
    - "scalping_strategy"
    - "arbitrage_strategy"
    - "volatility_strategy"
    - "market_maker_strategy"
```

## 📊 Торговые стратегии

### Доступные стратегии

| Стратегия | Тип | Описание |
|-----------|-----|----------|
| `TrendStrategy` | Трендовая | Moving Average Crossover с адаптивными параметрами |
| `SidewaysStrategy` | Боковая | RSI Mean Reversion для консолидации |
| `VolatilityStrategy` | Волатильность | ATR-Based Strategy с динамическими стопами |
| `PairsTradingStrategy` | Арбитраж | Statistical Arbitrage между коррелированными парами |
| `ManipulationStrategy` | Манипуляции | Обнаружение и противодействие манипуляциям |
| `AdaptiveStrategyGenerator` | Адаптивная | Автоматическая генерация стратегий |
| `MarketMakerStrategy` | Маркет-мейкинг | Создание ликвидности с управлением рисками |

### Эволюционные стратегии
- **Автоматическая оптимизация** параметров
- **Адаптация к рыночным условиям**
- **Генерация новых стратегий**
- **Валидация эффективности**

## 🧠 Искусственный интеллект

### Компоненты ИИ
- **Local AI Controller** - локальный контроллер принятия решений
- **Pattern Recognizer** - распознавание рыночных паттернов
- **Entanglement Detector** - обнаружение квантовой запутанности
- **Neural Noise Analyzer** - анализ нейронного шума
- **Mirror Neuron Signal** - зеркальные нейронные сигналы

### Машинное обучение
- **Transformer модели** для прогнозирования
- **Live Adaptation** - адаптация в реальном времени
- **Feature Engineering** - автоматическая инженерия признаков
- **Model Optimization** - оптимизация моделей

## 🔄 Эволюционная система

### Принципы эволюции
- **Подтверждение эффективности** - эволюция только при улучшении
- **Автоматическая валидация** - тестирование на исторических данных
- **Плавная миграция** - гибридный подход с обратной совместимостью
- **Непрерывное улучшение** - все компоненты эволюционируют параллельно

### Эволюционные агенты
- **Market Maker Agent** - эволюционный маркет-мейкер
- **Risk Agent** - эволюционный риск-агент
- **Portfolio Agent** - эволюционный портфельный агент
- **Strategy Agent** - эволюционный агент стратегий
- **News Agent** - эволюционный новостной агент
- **Order Executor Agent** - эволюционный исполнитель ордеров

## 🛡️ Управление рисками

### Risk Management
- **Circuit Breaker** - автоматическое отключение при ошибках
- **Dynamic Risk Limits** - адаптивные лимиты риска
- **Correlation Analysis** - анализ корреляций
- **Liquidity Gravity Monitor** - мониторинг ликвидности
- **Portfolio Optimization** - оптимизация портфеля

### Health Monitoring
- **System Health Checker** - проверка состояния системы
- **Performance Monitor** - мониторинг производительности
- **Alert Manager** - управление уведомлениями
- **Metrics Collector** - сбор метрик

## 📈 Мониторинг и аналитика

### Веб-дашборд
- **Real-time Trading** - торговля в реальном времени
- **Performance Charts** - графики производительности
- **Strategy Management** - управление стратегиями
- **Risk Monitoring** - мониторинг рисков
- **Evolution Statistics** - статистика эволюции

### API Endpoints
```python
# Статус системы
GET /api/v1/status

# Метрики производительности
GET /api/v1/metrics

# Управление стратегиями
POST /api/v1/strategies/enable
POST /api/v1/strategies/disable

# Статистика эволюции
GET /api/v1/evolution/stats
GET /api/v1/evolution/history

# Экстренная остановка
POST /api/v1/emergency/stop
```

## 🧪 Тестирование

### Типы тестов
- **Unit Tests** - модульные тесты
- **Integration Tests** - интеграционные тесты
- **E2E Tests** - end-to-end тесты
- **Performance Tests** - тесты производительности
- **Security Tests** - тесты безопасности

### Запуск тестов
```bash
# Все тесты
pytest

# С покрытием
pytest --cov=.

# Конкретная категория
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/
```

## 📚 Документация

- [API Documentation](docs/API_DOCUMENTATION.md) - Документация API
- [Architecture Guide](docs/ARCHITECTURE_GUIDE.md) - Руководство по архитектуре
- [Strategy Development](docs/STRATEGY_DEVELOPMENT.md) - Разработка стратегий
- [Evolution System](docs/EVOLUTION_SYSTEM.md) - Эволюционная система
- [Risk Management](docs/RISK_MANAGEMENT.md) - Управление рисками

## 🔧 Разработка

### Установка для разработки
```bash
# Установка dev зависимостей
pip install -r requirements-dev.txt

# Настройка pre-commit hooks
pre-commit install

# Форматирование кода
black .
isort .

# Проверка типов
mypy .

# Линтинг
flake8 .
```

### Структура разработки
- **Domain Layer** - бизнес-логика
- **Application Layer** - сценарии использования
- **Infrastructure Layer** - внешние зависимости
- **Interface Layer** - пользовательские интерфейсы

## 🤝 Вклад в проект

1. Fork репозитория
2. Создайте feature branch (`git checkout -b feature/amazing-feature`)
3. Commit изменения (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

### Стандарты кода
- **Type Hints** - обязательная типизация
- **Docstrings** - документация функций
- **SOLID Principles** - следование принципам
- **DDD Patterns** - паттерны DDD
- **Test Coverage** - покрытие тестами

## 📄 Лицензия

Этот проект лицензирован под MIT License - см. файл [LICENSE](LICENSE) для деталей.

## ⚠️ Отказ от ответственности

Это программное обеспечение предназначено только для образовательных целей. Торговля криптовалютами связана с высокими рисками. Авторы не несут ответственности за любые финансовые потери.

## 📞 Поддержка

- **Issues**: [GitHub Issues](https://github.com/your-repo/syntra/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/syntra/discussions)
- **Wiki**: [GitHub Wiki](https://github.com/your-repo/syntra/wiki)

---

**Сделано с ❤️ для сообщества трейдеров и разработчиков**

*Syntra - Advanced Trading Bot с Domain-Driven Design архитектурой* 