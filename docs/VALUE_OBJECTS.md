# Value Objects - Объекты-значения домена

## Обзор

Value Objects в домене алготрейдинга представляют собой неизменяемые объекты, которые инкапсулируют бизнес-логику и обеспечивают типобезопасность для финансовых операций.

## Архитектура

### Базовые типы

```python
from domain.types.value_object_types import (
    AmountType, CurrencyCode, TimestampValue, PercentageValue,
    NumericType, PositiveNumeric, NonNegativeNumeric,
    CurrencyPair, ExchangeRate, PriceLevel, VolumeAmount, MoneyAmount
)
```

### Базовые классы

- `ValueObject` - базовый абстрактный класс для всех value objects
- `NumericValueObject` - базовый класс для числовых value objects
- `ValueObjectProtocol` - протокол для всех value objects

## Основные Value Objects

### Currency - Валюты

```python
from domain.value_objects import Currency

# Создание валюты
btc = Currency.BTC
usdt = Currency.USDT

# Проверка свойств
assert btc.is_crypto
assert usdt.is_stablecoin
assert btc.is_major_crypto

# Получение торговых пар
pairs = Currency.get_trading_pairs(btc)
print(f"Торговые пары для BTC: {[p.value for p in pairs[:5]]}")

# Проверка возможности торговли
assert btc.can_trade_with(usdt)
```

### Money - Денежные суммы

```python
from domain.value_objects import Money, Currency

# Создание денежной суммы
money = Money(100.50, Currency.USD)
print(f"Сумма: {money}")  # 100.50 USD

# Арифметические операции
money1 = Money(100, Currency.USD)
money2 = Money(50, Currency.USD)
result = money1 + money2  # 150 USD

# Процентные операции
percentage = Decimal("10")
increased = money1.increase_by_percentage(percentage)  # 110 USD
decreased = money1.decrease_by_percentage(percentage)  # 90 USD

# Сравнение
assert money1 > money2
assert money1.is_positive()
```

### Price - Цены

```python
from domain.value_objects import Price, Currency

# Создание цены
price = Price(50000, Currency.USD)

# Расчет изменений
previous_price = Price(48000, Currency.USD)
change_percent = price.percentage_change_from(previous_price)  # 4.17%
price_change = price.price_change_from(previous_price)  # 2000 USD

# Расчет спреда
bid_price = Price(49900, Currency.USD)
ask_price = Price(50100, Currency.USD)
spread = bid_price.spread_with(ask_price)  # 200 USD
spread_percent = bid_price.spread_percentage_with(ask_price)  # 0.4%

# Применение проскальзывания
slippage = Decimal("0.1")  # 0.1%
buy_price, sell_price = price.apply_slippage(slippage)
```

### Volume - Объемы

```python
from domain.value_objects import Volume

# Создание объема
volume = Volume(1000.5)

# Арифметические операции
volume1 = Volume(1000)
volume2 = Volume(500)
total = volume1 + volume2  # 1500

# Процентные операции
percentage = volume1.percentage_of(total)  # 66.67%

# Округление
rounded = volume.round(2)  # Округление до 2 знаков
```

### Percentage - Проценты

```python
from domain.value_objects import Percentage

# Создание процента
percentage = Percentage(5.5)
print(f"Процент: {percentage}")  # 5.50%

# Преобразование в долю
fraction = percentage.to_fraction()  # 0.055

# Применение к значению
value = 1000
result = percentage.apply_to(value)  # 55

# Сложный процент
p1 = Percentage(10)
p2 = Percentage(20)
compound = p1.compound_with(p2)  # 32%

# Годовой процент
annual = percentage.annualize(30)  # Годовой эквивалент для 30 дней
```

### Timestamp - Временные метки

```python
from domain.value_objects import Timestamp

# Создание временной метки
now = Timestamp.now()
timestamp = Timestamp.from_iso("2024-01-15T10:00:00Z")

# Арифметические операции
future = now.add_hours(1)
past = now.subtract_days(1)

# Разность времени
diff_seconds = future.time_difference(now)  # 3600
diff_hours = future.time_difference_hours(now)  # 1.0

# Проверки времени
assert future.is_future()
assert past.is_past()
assert now.is_now()

# Торговые часы
assert now.is_trading_hours(9, 17)  # Проверка торговых часов

# Округление
rounded_minute = now.round_to_minute()
rounded_hour = now.round_to_hour()
```

### Signal - Торговые сигналы

```python
from domain.value_objects import Signal, SignalType, SignalStrength, Timestamp

# Создание сигнала
timestamp = Timestamp.now()
signal = Signal(
    signal_type=SignalType.BUY,
    timestamp=timestamp,
    strength=SignalStrength.STRONG,
    confidence=Percentage(85),
    price=Decimal("50000"),
    volume=Decimal("1.5"),
    source="strategy_1",
    description="Breakout signal"
)

# Проверка свойств
assert signal.is_buy_signal
assert signal.is_strong_signal
assert signal.is_trading_signal

# Скоринг
score = signal.get_combined_score()  # Комбинированный скор

# Фабричные методы
buy_signal = Signal.create_buy_signal(timestamp, SignalStrength.STRONG)
sell_signal = Signal.create_sell_signal(timestamp, SignalStrength.MODERATE)
hold_signal = Signal.create_hold_signal(timestamp)
```

### TradingPair - Торговые пары

```python
from domain.value_objects import TradingPair, Currency

# Создание торговой пары
pair = TradingPair(Currency.BTC, Currency.USDT)
print(f"Пара: {pair}")  # BTCUSDT

# Проверка свойств
assert pair.is_crypto_pair
assert pair.is_stablecoin_pair
assert pair.is_major_pair

# Создание из символа
pair = TradingPair.from_symbol("ETHUSDT")

# Обратная пара
reverse = pair.get_reverse_pair()  # USDTETH

# Популярные пары
common_pairs = pair.get_common_pairs()
```

## Фабрика Value Objects

```python
from domain.value_objects.factory import factory

# Создание через фабрику
money = factory.create_money(100, Currency.USD)
price = factory.create_price(50000, Currency.USD)
volume = factory.create_volume(1000)
percentage = factory.create_percentage("10.5%")
timestamp = factory.create_timestamp("2024-01-15T10:00:00Z")
pair = factory.create_trading_pair("BTC", "USDT")
signal = factory.create_signal("BUY", timestamp, "STRONG")

# Сериализация
data = factory.to_dict(money)
reconstructed = factory.from_dict(data)
assert money == reconstructed

# Валидация
assert factory.validate(money)
```

## Сериализация

Все value objects поддерживают сериализацию в словари:

```python
# Сериализация
money = Money(100, Currency.USD)
data = money.to_dict()
# {
#     "amount": "100",
#     "currency": "USD",
#     "type": "Money"
# }

# Десериализация
reconstructed = Money.from_dict(data)
assert money == reconstructed
```

## Валидация

Все value objects включают встроенную валидацию:

```python
# Валидация валюты
with pytest.raises(ValueError):
    Money(100, "INVALID")

# Валидация цены
with pytest.raises(ValueError):
    Price(-100, Currency.USD)

# Валидация объема
with pytest.raises(ValueError):
    Volume(-100)

# Валидация торговой пары
with pytest.raises(ValueError):
    TradingPair(Currency.BTC, Currency.BTC)
```

## Типизация

Все value objects используют строгую типизацию:

```python
from domain.types.value_object_types import (
    MoneyAmount, PriceLevel, VolumeAmount, PercentageValue
)

# Типизированные свойства
money = Money(100, Currency.USD)
assert isinstance(money.amount, MoneyAmount)

price = Price(50000, Currency.USD)
assert isinstance(price.amount, PriceLevel)

volume = Volume(1000)
assert isinstance(volume.value, VolumeAmount)

percentage = Percentage(5.5)
assert isinstance(percentage.value, PercentageValue)
```

## Лучшие практики

### 1. Использование фабрики

```python
# Хорошо
money = factory.create_money(100, Currency.USD)

# Плохо
money = Money(100, Currency.USD)  # Напрямую
```

### 2. Проверка типов

```python
# Хорошо
if isinstance(value, Money):
    process_money(value)

# Плохо
if hasattr(value, 'amount'):
    process_money(value)
```

### 3. Использование протоколов

```python
from domain.types.value_object_types import NumericValueObjectProtocol

def process_numeric_value(obj: NumericValueObjectProtocol) -> None:
    result = obj.value * 2
    print(f"Результат: {result}")
```

### 4. Валидация входных данных

```python
def create_trade(amount: Union[int, float, str], currency: str) -> Money:
    # Валидация валюты
    currency_obj = Currency.from_string(currency)
    if not currency_obj:
        raise ValueError(f"Invalid currency: {currency}")
    
    # Создание через фабрику
    return factory.create_money(amount, currency_obj)
```

## Производительность

Value objects оптимизированы для производительности:

- Неизменяемые объекты (frozen dataclasses)
- Кэширование в фабрике
- Эффективные операции сравнения
- Минимальное использование памяти

## Расширение

Для добавления новых value objects:

1. Создайте класс, наследующий от `ValueObject`
2. Реализуйте абстрактные методы
3. Зарегистрируйте в фабрике
4. Добавьте тесты
5. Обновите документацию

```python
@dataclass(frozen=True)
class NewValueObject(ValueObject):
    value: NewType
    
    def _get_equality_components(self) -> tuple:
        return (self.value,)
    
    def to_dict(self) -> ValueObjectDict:
        return {"value": str(self.value), "type": "NewValueObject"}
    
    @classmethod
    def from_dict(cls, data: ValueObjectDict) -> "NewValueObject":
        return cls(NewType(data["value"]))

# Регистрация в фабрике
factory.register("NewValueObject", NewValueObject)
``` 