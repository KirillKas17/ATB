# 🧪 Руководство по достижению 90% покрытия тестами ATB

## 📋 Содержание
1. [Обзор текущего состояния](#обзор-текущего-состояния)
2. [Стратегия тестирования](#стратегия-тестирования)
3. [Чек-лист покрытия тестами](#чек-лист-покрытия-тестами)
4. [Инструкции по реализации](#инструкции-по-реализации)
5. [Шаблоны тестов](#шаблоны-тестов)
6. [Инструменты и метрики](#инструменты-и-метрики)

---

## 📊 Обзор текущего состояния

### Статистика покрытия
- **Текущее покрытие**: ~16.8% (48/286 файлов)
- **Целевое покрытие**: 90%
- **Файлов для покрытия**: ~209 файлов

### Структура проекта
```
ATB/
├── application/     (38 файлов) - Use cases, сервисы приложения
├── domain/         (66 файлов)  - Бизнес-логика, сущности
├── infrastructure/ (153 файла)  - Внешние адаптеры, ML, агенты
├── interfaces/     (16 файлов)  - API, CLI, UI
└── shared/         (13 файлов)  - Утилиты, конфиги
```

---

## 🎯 Стратегия тестирования

### Принципы тестирования
1. **Пирамида тестирования**: 70% unit, 20% integration, 10% e2e
2. **DDD подход**: Тестирование по слоям архитектуры
3. **Изоляция**: Каждый тест независим
4. **Читаемость**: Понятные названия и структура
5. **Поддержка**: Легко поддерживаемые тесты

### Типы тестов
- **Unit тесты**: Отдельные функции/классы
- **Integration тесты**: Взаимодействие компонентов
- **E2E тесты**: Полные пользовательские сценарии

---

## ✅ Чек-лист покрытия тестами

### 🏗️ Domain Layer (66 файлов)

#### Value Objects (domain/value_objects/)
- [ ] `currency.py` - Тестирование валют
- [ ] `money.py` - Тестирование денежных операций
- [ ] `percentage.py` - Тестирование процентов
- [ ] `price.py` - Тестирование цен
- [ ] `timestamp.py` - Тестирование временных меток
- [ ] `volume.py` - Тестирование объемов

#### Entities (domain/entities/)
- [ ] `market.py` - Тестирование рыночных данных
- [ ] `ml.py` - Тестирование ML моделей
- [ ] `models.py` - Тестирование доменных моделей
- [ ] `noise_analyzer.py` - Тестирование анализатора шума
- [ ] `order.py` - Тестирование ордеров
- [ ] `order_repository.py` - Тестирование репозитория ордеров
- [ ] `portfolio.py` - Тестирование портфеля
- [ ] `portfolio_fixed.py` - Тестирование фиксированного портфеля
- [ ] `portfolio_repository.py` - Тестирование репозитория портфеля
- [ ] `position.py` - Тестирование позиций
- [ ] `position_repository.py` - Тестирование репозитория позиций
- [ ] `project_structure.py` - Тестирование структуры проекта
- [ ] `risk.py` - Тестирование рисков
- [ ] `risk_analysis.py` - Тестирование анализа рисков
- [ ] `session_profile.py` - Тестирование профилей сессий
- [ ] `strategy.py` - Тестирование стратегий
- [ ] `strategy_model.py` - Тестирование моделей стратегий
- [ ] `trading.py` - Тестирование торговли

#### Intelligence (domain/intelligence/)
- [ ] `entanglement_detector.py` - ✅ Уже покрыт
- [ ] `market_pattern_recognizer.py` - Тестирование распознавания паттернов
- [ ] `mirror_detector.py` - Тестирование зеркального детектора
- [ ] `noise_analyzer.py` - Тестирование анализатора шума

#### Market (domain/market/)
- [ ] `liquidity_gravity.py` - ✅ Уже покрыт

#### Memory (domain/memory/)
- [ ] `pattern_memory.py` - Тестирование памяти паттернов

#### Prediction (domain/prediction/)
- [ ] `reversal_predictor.py` - Тестирование предиктора разворотов
- [ ] `reversal_signal.py` - Тестирование сигналов разворотов

#### Protocols (domain/protocols/)
- [ ] `exchange_protocol.py` - Тестирование протокола биржи
- [ ] `ml_protocol.py` - Тестирование ML протокола
- [ ] `repository_protocol.py` - Тестирование протокола репозитория
- [ ] `strategy_protocol.py` - Тестирование протокола стратегии

#### Repositories (domain/repositories/)
- [ ] `market_repository.py` - Тестирование репозитория рынка
- [ ] `ml_repository.py` - Тестирование ML репозитория
- [ ] `portfolio_repository.py` - Тестирование репозитория портфеля
- [ ] `position_repository.py` - Тестирование репозитория позиций
- [ ] `strategy_repository.py` - Тестирование репозитория стратегий
- [ ] `trading_repository.py` - Тестирование торгового репозитория

#### Services (domain/services/)
- [ ] `correlation_chain.py` - ✅ Уже покрыт
- [ ] `pattern_discovery.py` - ✅ Уже покрыт
- [ ] `strategy_optimizer.py` - Тестирование оптимизатора стратегий

#### Sessions (domain/sessions/)
- [ ] `session_influence_analyzer.py` - Тестирование анализатора влияния сессий
- [ ] `session_marker.py` - Тестирование маркера сессий

#### Strategies (domain/strategies/)
- [ ] `strategy_interface.py` - Тестирование интерфейса стратегии

#### Superfixer (domain/superfixer/)
- [ ] `error_report.py` - Тестирование отчета об ошибках
- [ ] `error_types.py` - Тестирование типов ошибок

### 🚀 Application Layer (38 файлов)

#### Analysis (application/analysis/)
- [ ] `entanglement_monitor.py` - ✅ Уже покрыт

#### Evolution (application/evolution/)
- [ ] `evolution_orchestrator.py` - ✅ Уже покрыт

#### Filters (application/filters/)
- [ ] `orderbook_filter.py` - ✅ Уже покрыт

#### Monitoring (application/monitoring/)
- [ ] `pattern_observer.py` - ✅ Уже покрыт

#### Prediction (application/prediction/)
- [ ] `combined_predictor.py` - ✅ Уже покрыт
- [ ] `pattern_predictor.py` - ✅ Уже покрыт
- [ ] `reversal_controller.py` - ✅ Уже покрыт

#### Risk (application/risk/)
- [ ] `liquidity_gravity_monitor.py` - ✅ Уже покрыт

#### Services (application/services/)
- [ ] `market_service.py` - ✅ Уже покрыт
- [ ] `ml_service.py` - ✅ Уже покрыт
- [ ] `portfolio_service.py` - ✅ Уже покрыт
- [ ] `report_writer.py` - ✅ Уже покрыт
- [ ] `risk_service.py` - ✅ Уже покрыт
- [ ] `session_signal_engine.py` - ✅ Уже покрыт
- [ ] `strategy_service.py` - ✅ Уже покрыт
- [ ] `summary_generator.py` - ✅ Уже покрыт
- [ ] `trading_orchestrator.py` - ✅ Уже покрыт
- [ ] `trading_service.py` - ✅ Уже покрыт
- [ ] `type_checker.py` - ✅ Уже покрыт

#### Signal (application/signal/)
- [ ] `session_signal_engine.py` - ✅ Уже покрыт

#### Strategy Advisor (application/strategy_advisor/)
- [ ] `mirror_map_builder.py` - ✅ Уже покрыт

#### Superfixer (application/superfixer/)
- [ ] `context_linker.py` - ✅ Уже покрыт
- [ ] `error_classifier.py` - ✅ Уже покрыт
- [ ] `error_scanner.py` - ✅ Уже покрыт
- [ ] `filesystem_writer.py` - ✅ Уже покрыт
- [ ] `graph_visualizer.py` - ✅ Уже покрыт

#### Use Cases (application/use_cases/)
- [ ] `manage_orders.py` - ✅ Уже покрыт
- [ ] `manage_positions.py` - ✅ Уже покрыт
- [ ] `manage_risk.py` - ✅ Уже покрыт
- [ ] `manage_trading_pairs.py` - ✅ Уже покрыт

### 🔧 Infrastructure Layer (153 файла)

#### Agents (infrastructure/agents/)
- [ ] `advanced_market_maker.py` - ✅ Уже покрыт
- [ ] `agent_context.py` - ✅ Уже покрыт
- [ ] `agent_market_maker_model.py` - ✅ Уже покрыт
- [ ] `agent_risk_model.py` - Тестирование модели риска агента
- [ ] `analytical_integration.py` - ✅ Уже покрыт
- [ ] `arbitrage_agent.py` - Тестирование арбитражного агента
- [ ] `base_agent.py` - Тестирование базового агента
- [ ] `correlation_agent.py` - Тестирование корреляционного агента
- [ ] `decision_agent.py` - Тестирование агента принятия решений
- [ ] `entanglement_agent.py` - Тестирование агента запутанности
- [ ] `evolution_agent.py` - Тестирование эволюционного агента
- [ ] `liquidity_agent.py` - Тестирование агента ликвидности
- [ ] `market_maker_agent.py` - Тестирование маркет-мейкер агента
- [ ] `mirror_agent.py` - Тестирование зеркального агента
- [ ] `noise_agent.py` - Тестирование агента шума
- [ ] `order_executor_agent.py` - Тестирование агента исполнения ордеров
- [ ] `portfolio_agent.py` - ✅ Уже покрыт
- [ ] `position_agent.py` - Тестирование агента позиций
- [ ] `risk_agent.py` - ✅ Уже покрыт
- [ ] `signal_agent.py` - Тестирование сигнального агента
- [ ] `strategy_agent.py` - Тестирование агента стратегий
- [ ] `trading_agent.py` - Тестирование торгового агента

#### Core (infrastructure/core/)
- [ ] `analysis/` - Тестирование аналитических модулей
- [ ] `auto_migration_manager.py` - Тестирование менеджера миграций
- [ ] `autonomous_controller.py` - Тестирование автономного контроллера
- [ ] `controllers/` - Тестирование контроллеров
- [ ] `ml/` - Тестирование ML модулей

#### Exchange (infrastructure/exchange/)
- [ ] `bybit_adapter.py` - Тестирование адаптера Bybit

#### External Services (infrastructure/external_services/)
- [ ] `account_manager.py` - ✅ Уже покрыт
- [ ] `bybit_client.py` - ✅ Уже покрыт
- [ ] `order_manager.py` - ✅ Уже покрыт
- [ ] `portfolio_manager.py` - Тестирование менеджера портфеля
- [ ] `position_manager.py` - ✅ Уже покрыт
- [ ] `risk_manager.py` - Тестирование менеджера рисков

#### Market Data (infrastructure/market_data/)
- [ ] `base_connector.py` - Тестирование базового коннектора
- [ ] `binance_connector.py` - Тестирование коннектора Binance
- [ ] `market_data_manager.py` - Тестирование менеджера рыночных данных

#### ML Services (infrastructure/ml_services/)
- [ ] `advanced_price_predictor.py` - ✅ Уже покрыт
- [ ] `dataset_manager.py` - ✅ Уже покрыт
- [ ] `feature_engineering.py` - Тестирование инженерии признаков
- [ ] `model_manager.py` - Тестирование менеджера моделей
- [ ] `model_optimizer.py` - Тестирование оптимизатора моделей
- [ ] `model_selector.py` - ✅ Уже покрыт
- [ ] `pattern_discovery.py` - ✅ Уже покрыт
- [ ] `prediction_engine.py` - Тестирование движка предсказаний
- [ ] `regime_discovery.py` - ✅ Уже покрыт
- [ ] `signal_processor.py` - ✅ Уже покрыт

#### Repositories (infrastructure/repositories/)
- [ ] `market_repository.py` - ✅ Уже покрыт
- [ ] `ml_repository.py` - ✅ Уже покрыт
- [ ] `portfolio_repository.py` - ✅ Уже покрыт
- [ ] `position_repository.py` - ✅ Уже покрыт
- [ ] `strategy_repository.py` - ✅ Уже покрыт

#### Strategies (infrastructure/strategies/)
- [ ] `adaptive_strategy_generator.py` - ✅ Уже покрыт
- [ ] `arbitrage_strategy.py` - ✅ Уже покрыт
- [ ] `backtest.py` - ✅ Уже покрыт
- [ ] `base_strategy.py` - ✅ Уже покрыт
- [ ] `manipulation_strategy.py` - Тестирование стратегии манипуляций
- [ ] `market_regime_strategy.py` - Тестирование стратегии рыночного режима
- [ ] `optimizer.py` - ✅ Уже покрыт
- [ ] `trend_strategy.py` - ✅ Уже покрыт
- [ ] `volatility_strategy.py` - ✅ Уже покрыт

### 🖥️ Interfaces Layer (16 файлов)

#### API (interfaces/presentation/api/)
- [ ] `trading_api.py` - Тестирование торгового API
- [ ] `analytics_api.py` - Тестирование аналитического API
- [ ] `risk_api.py` - Тестирование API рисков

#### CLI (interfaces/presentation/cli/)
- [ ] `trading_cli.py` - Тестирование торгового CLI
- [ ] `analytics_cli.py` - Тестирование аналитического CLI

#### Dashboard (interfaces/presentation/dashboard/)
- [ ] `dashboard.py` - ✅ Уже покрыт
- [ ] `widgets/` - Тестирование виджетов

### 🔧 Shared Layer (13 файлов)

- [ ] `config.py` - Тестирование конфигурации
- [ ] `data_loader.py` - ✅ Уже покрыт
- [ ] `event_bus.py` - Тестирование шины событий
- [ ] `exceptions.py` - Тестирование исключений
- [ ] `health_checker.py` - Тестирование проверки здоровья
- [ ] `logger.py` - Тестирование логгера
- [ ] `metrics.py` - Тестирование метрик
- [ ] `unified_cache.py` - Тестирование унифицированного кэша
- [ ] `utils.py` - ✅ Уже покрыт

---

## 📝 Инструкции по реализации

### 1. Подготовка окружения

```bash
# Установка зависимостей для тестирования
pip install pytest pytest-asyncio pytest-cov pytest-mock
pip install pytest-xdist  # для параллельного запуска
pip install coverage       # для детального анализа покрытия

# Создание конфигурации pytest
```

### 2. Структура тестового файла

```python
# tests/domain/value_objects/test_money.py
import pytest
from decimal import Decimal
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency

class TestMoney:
    """Тесты для value object Money."""
    
    def test_creation(self):
        """Тест создания объекта Money."""
        money = Money(Decimal("100.50"), Currency.USD)
        assert money.value == Decimal("100.50")
        assert money.currency == Currency.USD
    
    def test_addition(self):
        """Тест сложения денежных сумм."""
        money1 = Money(Decimal("100"), Currency.USD)
        money2 = Money(Decimal("50"), Currency.USD)
        result = money1 + money2
        assert result.value == Decimal("150")
        assert result.currency == Currency.USD
    
    def test_different_currencies_error(self):
        """Тест ошибки при операциях с разными валютами."""
        money1 = Money(Decimal("100"), Currency.USD)
        money2 = Money(Decimal("50"), Currency.EUR)
        with pytest.raises(ValueError):
            money1 + money2
```

### 3. Фикстуры и моки

```python
# tests/conftest.py
import pytest
from unittest.mock import Mock, AsyncMock

@pytest.fixture
def mock_exchange():
    """Фикстура для мока биржи."""
    exchange = Mock()
    exchange.create_order = AsyncMock(return_value={"id": "test_order"})
    exchange.cancel_order = AsyncMock(return_value=True)
    return exchange

@pytest.fixture
def sample_market_data():
    """Фикстура с тестовыми рыночными данными."""
    return {
        "symbol": "BTCUSDT",
        "price": 50000.0,
        "volume": 1000.0,
        "timestamp": "2024-01-01T00:00:00Z"
    }
```

### 4. Интеграционные тесты

```python
# tests/integration/test_trading_flow.py
import pytest
from application.use_cases.manage_orders import ManageOrdersUseCase
from domain.entities.order import Order, OrderSide, OrderType
from infrastructure.external_services.bybit_client import BybitClient

class TestTradingFlow:
    """Интеграционные тесты торгового потока."""
    
    @pytest.mark.asyncio
    async def test_complete_trading_flow(self, mock_exchange):
        """Тест полного торгового потока."""
        # Arrange
        use_case = ManageOrdersUseCase(mock_exchange)
        order = Order(
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000")
        )
        
        # Act
        result = await use_case.create_order(order)
        
        # Assert
        assert result["id"] == "test_order"
        mock_exchange.create_order.assert_called_once()
```

### 5. E2E тесты

```python
# tests/e2e/test_trading_session.py
import pytest
from application.di_container import DIContainer
from infrastructure.agents.trading_agent import TradingAgent

class TestTradingSession:
    """E2E тесты торговой сессии."""
    
    @pytest.mark.asyncio
    async def test_complete_trading_session(self):
        """Тест полной торговой сессии."""
        # Arrange
        container = DIContainer()
        agent = container.get(TradingAgent)
        
        # Act
        await agent.start()
        await agent.process_market_data(sample_data)
        await agent.stop()
        
        # Assert
        assert agent.is_running is False
        assert len(agent.trades) > 0
```

---

## 📋 Шаблоны тестов

### Unit тест для Value Object

```python
def test_value_object_creation():
    """Тест создания value object."""
    # Arrange & Act
    obj = ValueObject(value)
    
    # Assert
    assert obj.value == expected_value

def test_value_object_validation():
    """Тест валидации value object."""
    # Arrange & Act & Assert
    with pytest.raises(ValueError):
        ValueObject(invalid_value)

def test_value_object_equality():
    """Тест равенства value objects."""
    # Arrange
    obj1 = ValueObject(value)
    obj2 = ValueObject(value)
    obj3 = ValueObject(different_value)
    
    # Assert
    assert obj1 == obj2
    assert obj1 != obj3
```

### Unit тест для Entity

```python
def test_entity_creation():
    """Тест создания entity."""
    # Arrange & Act
    entity = Entity(id, attributes)
    
    # Assert
    assert entity.id == id
    assert entity.attributes == attributes

def test_entity_business_logic():
    """Тест бизнес-логики entity."""
    # Arrange
    entity = Entity(id, attributes)
    
    # Act
    result = entity.business_method()
    
    # Assert
    assert result == expected_result
```

### Unit тест для Service

```python
def test_service_method(mock_dependencies):
    """Тест метода сервиса."""
    # Arrange
    service = Service(mock_dependencies)
    
    # Act
    result = service.method(input_data)
    
    # Assert
    assert result == expected_result
    mock_dependencies.method.assert_called_once_with(input_data)
```

### Integration тест

```python
@pytest.mark.asyncio
async def test_component_integration():
    """Тест интеграции компонентов."""
    # Arrange
    component1 = Component1()
    component2 = Component2()
    
    # Act
    result = await component1.process(component2.get_data())
    
    # Assert
    assert result.status == "success"
    assert component2.was_called
```

---

## 🛠️ Инструменты и метрики

### 1. Запуск тестов

```bash
# Запуск всех тестов
pytest tests/ -v

# Запуск с покрытием
pytest tests/ --cov=. --cov-report=html --cov-report=term

# Запуск параллельно
pytest tests/ -n auto

# Запуск конкретного файла
pytest tests/domain/value_objects/test_money.py -v

# Запуск конкретного теста
pytest tests/domain/value_objects/test_money.py::TestMoney::test_creation -v
```

### 2. Анализ покрытия

```bash
# Генерация отчета покрытия
coverage run -m pytest tests/
coverage report
coverage html

# Проверка покрытия конкретного модуля
coverage run -m pytest tests/domain/value_objects/
coverage report --include="domain/value_objects/*"
```

### 3. Метрики качества

```python
# pytest.ini
[tool:pytest]
minversion = 6.0
addopts = 
    -v
    --strict-markers
    --disable-warnings
    --cov=.
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=90
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow running tests
```

### 4. Автоматизация

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.10
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio
    - name: Run tests
      run: |
        pytest tests/ --cov=. --cov-report=xml --cov-fail-under=90
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

---

## 📈 План реализации

### Фаза 1 (Неделя 1-2): Value Objects и Entities
- [ ] Покрыть все value objects (6 файлов)
- [ ] Покрыть основные entities (10 файлов)
- [ ] Настроить CI/CD

### Фаза 2 (Неделя 3-4): Domain Services
- [ ] Покрыть domain services (8 файлов)
- [ ] Покрыть repositories (6 файлов)
- [ ] Покрыть protocols (4 файла)

### Фаза 3 (Неделя 5-6): Application Layer
- [ ] Покрыть use cases (4 файла)
- [ ] Покрыть application services (15 файлов)
- [ ] Добавить интеграционные тесты

### Фаза 4 (Неделя 7-8): Infrastructure Layer
- [ ] Покрыть agents (20 файлов)
- [ ] Покрыть external services (6 файлов)
- [ ] Покрыть ML services (10 файлов)

### Фаза 5 (Неделя 9-10): Interfaces и Shared
- [ ] Покрыть interfaces (16 файлов)
- [ ] Покрыть shared (13 файлов)
- [ ] Добавить E2E тесты

### Фаза 6 (Неделя 11-12): Оптимизация
- [ ] Оптимизировать медленные тесты
- [ ] Добавить тесты производительности
- [ ] Финальная проверка покрытия

---

## 🎯 Критерии успеха

### Количественные метрики
- [ ] Покрытие кода ≥ 90%
- [ ] Покрытие веток ≥ 85%
- [ ] Время выполнения тестов < 5 минут
- [ ] 0 критических багов в продакшене

### Качественные метрики
- [ ] Все тесты проходят
- [ ] Тесты читаемы и поддерживаемы
- [ ] Хорошая изоляция тестов
- [ ] Понятные сообщения об ошибках

### Автоматизация
- [ ] CI/CD pipeline настроен
- [ ] Автоматические проверки покрытия
- [ ] Уведомления о падении тестов
- [ ] Автоматическое обновление отчетов

---

## 📞 Поддержка

### Полезные команды

```bash
# Поиск непокрытых файлов
find . -name "*.py" -not -path "./tests/*" -not -path "./venv/*" | xargs -I {} sh -c 'if ! grep -q "{}" coverage_report.txt; then echo "{}"; fi'

# Подсчет строк кода
find . -name "*.py" -not -path "./tests/*" -not -path "./venv/*" | xargs wc -l

# Проверка дублирования кода
pylint --duplicate-code-threshold=5 .

# Проверка сложности
radon cc . -a
```

### Контакты
- **Ответственный**: Команда разработки
- **Ревью**: Senior разработчики
- **Документация**: Технический писатель

---

*Документ обновлен: 2024-01-01*
*Версия: 1.0* 