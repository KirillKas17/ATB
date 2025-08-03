# Правила импортов для промышленного DDD Python-проекта

## 1. Общие принципы
- **Никогда не импортируй из `__init__.py` слоёв domain, entities, value_objects, intelligence, types.**
- **Импорты между слоями (domain, application, infrastructure, interfaces, shared) — только точечные, внутри функций или под `if TYPE_CHECKING`.**
- **Везде, где возможно, используй строковые аннотации типов (`'Money'`, `'Price'`, `'Strategy'`).**
- **Импорты для типизации (`from ... import ...`) — только под `if TYPE_CHECKING`.**

---

## 2. Value Objects (например, `money.py`, `price.py`, `volume.py`)
- Импорты других value objects, кэшей, конфигов — только внутри функций/методов, где они реально нужны.
- Импорты для типизации — только под `if TYPE_CHECKING`.
- Не импортировать ничего из domain/entities, domain/intelligence, domain/types напрямую.

---

## 3. Кэши и конфиги (например, `money_cache.py`, `price_config.py`)
- Импортировать основной value object только внутри функций, а не на уровне модуля.
- Не импортировать ничего из других кэшей/конфигов на уровне модуля.

---

## 4. Domain Entities и Types
- Импорты value objects и стратегий — только через `if TYPE_CHECKING` или внутри функций.
- Везде, где возможно, использовать строковые аннотации типов.
- Не импортировать ничего из value_objects на уровне модуля.

---

## 5. __init__.py всех слоёв
- Не импортировать ничего, что может вызвать цикл (особенно из value_objects, intelligence, entities).
- Экспортировать только то, что не зависит от других крупных слоёв.

---

## 6. Фабрики, кэши, конфиги
- Все импорты между фабриками, кэшами, конфигами — только внутри функций.
- Не импортировать основной value object на уровне модуля.

---

## 7. Тесты и conftest.py
- Импорты value objects только внутри фикстур/тестов.
- Не импортировать ничего из domain/entities, domain/intelligence, domain/types на уровне модуля.

---

## 8. Пример (value object)
```python
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .money_cache import MoneyCache
    from .money_config import MoneyConfig

class Money:
    ...
    def some_method(self):
        from .money_cache import MoneyCache
        ...
```

---

## 9. Пример (entity)
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from domain.value_objects.money import Money

class Portfolio:
    ...
    def get_balance(self) -> 'Money':
        from domain.value_objects.money import Money
        ...
```

---

## 10. Пример (тест)
```python
import pytest

def test_money_arithmetic():
    from domain.value_objects.money import Money
    ...
```

---

**Соблюдение этих правил гарантирует отсутствие циклических зависимостей, высокую модульность и промышленную масштабируемость архитектуры.** 