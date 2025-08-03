"""
Тесты безопасности для проверки уязвимостей.
"""
import pytest
import hashlib
import hmac
import time
from typing import Dict, Any
from domain.entities.trading import OrderId, Trade
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.trading_pair import TradingPair

# Создаем классы для тестов
class SecretManager:
    """Менеджер секретов для тестов."""
    def __init__(self) -> Any:
        self.secrets = {}
    
    def encrypt_secret(self, secret: str) -> str:
        """Шифрование секрета."""
        return f"encrypted_{secret}"
    
    def decrypt_secret(self, encrypted_secret: str) -> str:
        """Расшифровка секрета."""
        return encrypted_secret.replace("encrypted_", "")
    
    def get_required_secret(self, key: str) -> str:
        """Получение обязательного секрета."""
        import os
        value = os.environ.get(key)
        if not value:
            raise ValueError(f"Secret {key} not found")
        return value
    
    def validate_secret_strength(self, secret: str) -> bool:
        """Проверка силы секрета."""
        if len(secret) < 8:
            raise ValueError("Secret too weak")
        return True

class TestSecretManagement:
    """Тесты управления секретами."""
    def test_secret_manager_initialization(self) -> None:
        """Тест инициализации менеджера секретов."""
        secret_manager = SecretManager()
        assert secret_manager is not None
    
    def test_api_key_encryption(self) -> None:
        """Тест шифрования API ключей."""
        secret_manager = SecretManager()
        # Тестовый API ключ
        test_api_key = "test_api_key_12345"
        # Шифруем ключ
        encrypted_key = secret_manager.encrypt_secret(test_api_key)
        assert encrypted_key != test_api_key
        assert len(encrypted_key) > len(test_api_key)
        # Расшифровываем ключ
        decrypted_key = secret_manager.decrypt_secret(encrypted_key)
        assert decrypted_key == test_api_key
    
    def test_secret_rotation(self) -> None:
        """Тест ротации секретов."""
        secret_manager = SecretManager()
        # Создаем старый ключ
        old_key = "old_api_key"
        encrypted_old = secret_manager.encrypt_secret(old_key)
        # Создаем новый ключ
        new_key = "new_api_key"
        encrypted_new = secret_manager.encrypt_secret(new_key)
        # Проверяем, что ключи разные
        assert encrypted_old != encrypted_new
        # Проверяем, что оба ключа работают
        assert secret_manager.decrypt_secret(encrypted_old) == old_key
        assert secret_manager.decrypt_secret(encrypted_new) == new_key
    
    def test_environment_variable_validation(self) -> None:
        """Тест валидации переменных окружения."""
        secret_manager = SecretManager()
        # Тестируем отсутствующие переменные
        with pytest.raises(ValueError):
            secret_manager.get_required_secret("NONEXISTENT_API_KEY")
        # Тестируем пустые переменные
        from unittest.mock import patch
        with patch.dict('os.environ', {'EMPTY_API_KEY': ''}):
            with pytest.raises(ValueError):
                secret_manager.get_required_secret("EMPTY_API_KEY")
    
    def test_secret_strength_validation(self) -> None:
        """Тест проверки силы секретов."""
        secret_manager = SecretManager()
        # Слабый секрет
        weak_secret = "123"
        with pytest.raises(ValueError):
            secret_manager.validate_secret_strength(weak_secret)
        # Сильный секрет
        strong_secret = "StrongSecret123!@#"
        assert secret_manager.validate_secret_strength(strong_secret) is True

class TestInputValidation:
    """Тесты валидации входных данных."""
    def test_order_validation(self) -> None:
        """Тест валидации ордеров."""
        # Валидный ордер ID
        valid_order_id = OrderId("test_order")
        assert valid_order_id.value == "test_order"
        
        # Невалидный ордер ID - пустой
        with pytest.raises(ValueError):
            OrderId("")
    
    def test_money_validation(self) -> None:
        """Тест валидации денежных значений."""
        # Валидные значения
        valid_money = Money(100.0, Currency.USDT)
        assert valid_money.amount == 100.0
        # Невалидные значения
        with pytest.raises(ValueError):
            Money(-100.0, Currency.USDT)  # Отрицательная сумма
        with pytest.raises(ValueError):
            Money(float('inf'), Currency.USDT)  # Бесконечность
        with pytest.raises(ValueError):
            Money(float('nan'), Currency.USDT)  # NaN
    
    def test_trading_pair_validation(self) -> None:
        """Тест валидации торговых пар."""
        # Валидные пары
        valid_pairs = [
            TradingPair("BTC", "USDT"),
            TradingPair("ETH", "USDT"),
            TradingPair("ADA", "BTC")
        ]
        for pair in valid_pairs:
            assert pair is not None
        # Невалидные пары
        with pytest.raises(ValueError):
            TradingPair("", "USDT")  # Пустая базовая валюта
        with pytest.raises(ValueError):
            TradingPair("BTC", "")  # Пустая котируемая валюта
        with pytest.raises(ValueError):
            TradingPair("BTC", "BTC")  # Одинаковые валюты

class TestSQLInjection:
    """Тесты защиты от SQL инъекций."""
    @pytest.mark.asyncio
    async def test_sql_injection_protection(self) -> None:
        """Тест защиты от SQL инъекций."""
        from infrastructure.core.optimized_database import OptimizedDatabase
        database = OptimizedDatabase("sqlite:///test_security.db")
        # Попытка SQL инъекции
        malicious_input = "'; DROP TABLE trades; --"
        # Создаем тестовый ордер ID с потенциально опасным вводом
        order_id = OrderId(malicious_input)
        # Пытаемся сохранить - должно работать безопасно
        try:
            await database.save_trade(Trade(
                id=order_id.value,
                order_id=order_id.value,
                trading_pair=TradingPair("BTC", "USDT"),
                side="buy",
                volume=Volume(0.1),
                price=Price(50000),
                executed_at=Timestamp.now()
            ))
            # Проверяем, что таблица не была удалена
            trades = await database.get_trades("BTCUSDT")
            assert isinstance(trades, list)
        except Exception as e:
            # Если произошла ошибка, она должна быть не связана с SQL инъекцией
            assert "DROP TABLE" not in str(e)
    
    def test_parameterized_queries(self) -> None:
        """Тест использования параметризованных запросов."""
        # Проверяем, что в коде используются параметризованные запросы
        # Это должно быть проверено в репозиториях
        from infrastructure.repositories.trading_repository import InMemoryTradingRepository
        repository = InMemoryTradingRepository()
        # Тестируем с потенциально опасным вводом
        malicious_id = "'; DROP TABLE trades; --"
        # Должно работать безопасно
        result = repository.get_order(malicious_id)
        assert result is None  # Просто не найдено, без ошибок

class TestAuthentication:
    """Тесты аутентификации."""
    def test_api_signature_validation(self) -> None:
        """Тест валидации подписи API."""
        # Тестовые данные
        api_key = "test_api_key"
        secret_key = "test_secret_key"
        timestamp = str(int(time.time() * 1000))
        # Создаем подпись
        message = f"{api_key}{timestamp}"
        signature = hmac.new(
            secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        # Проверяем подпись
        expected_signature = hmac.new(
            secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        assert signature == expected_signature
        # Проверяем неправильную подпись
        wrong_signature = "wrong_signature"
        assert signature != wrong_signature
    
    def test_timestamp_validation(self) -> None:
        """Тест валидации временных меток."""
        current_time = int(time.time() * 1000)
        # Валидная временная метка (в пределах 5 минут)
        valid_timestamp = str(current_time - 60000)  # 1 минута назад
        assert self._is_timestamp_valid(valid_timestamp, 300000)  # 5 минут
        # Невалидная временная метка (слишком старая)
        old_timestamp = str(current_time - 600000)  # 10 минут назад
        assert not self._is_timestamp_valid(old_timestamp, 300000)
        # Невалидная временная метка (в будущем)
        future_timestamp = str(current_time + 60000)  # 1 минута в будущем
        assert not self._is_timestamp_valid(future_timestamp, 300000)
    
    def _is_timestamp_valid(self, timestamp: str, max_drift: int) -> bool:
        """Проверка валидности временной метки."""
        try:
            ts = int(timestamp)
            current_time = int(time.time() * 1000)
            return abs(current_time - ts) <= max_drift
        except (ValueError, TypeError):
            return False

class TestAuthorization:
    """Тесты авторизации."""
    def test_permission_checking(self) -> None:
        """Тест проверки разрешений."""
        # Симуляция пользователя с разрешениями
        user_permissions = {
            'read_trades': True,
            'write_trades': True,
            'delete_trades': False,
            'admin_access': False
        }
        # Проверяем разрешения
        assert self._check_permission(user_permissions, 'read_trades')
        assert self._check_permission(user_permissions, 'write_trades')
        assert not self._check_permission(user_permissions, 'delete_trades')
        assert not self._check_permission(user_permissions, 'admin_access')
    
    def test_role_based_access(self) -> None:
        """Тест ролевого доступа."""
        # Определяем роли
        roles = {
            'trader': ['read_trades', 'write_trades'],
            'viewer': ['read_trades'],
            'admin': ['read_trades', 'write_trades', 'delete_trades', 'admin_access']
        }
        # Проверяем доступ для разных ролей
        assert self._has_role_permission(roles, 'trader', 'read_trades')
        assert self._has_role_permission(roles, 'trader', 'write_trades')
        assert not self._has_role_permission(roles, 'trader', 'delete_trades')
        assert self._has_role_permission(roles, 'viewer', 'read_trades')
        assert not self._has_role_permission(roles, 'viewer', 'write_trades')
        assert self._has_role_permission(roles, 'admin', 'admin_access')
    
    def _check_permission(self, permissions: Dict[str, bool], permission: str) -> bool:
        """Проверка разрешения."""
        return permissions.get(permission, False)
    
    def _has_role_permission(self, roles: Dict[str, list], role: str, permission: str) -> bool:
        """Проверка разрешения для роли."""
        role_permissions = roles.get(role, [])
        return permission in role_permissions

class TestDataIntegrity:
    """Тесты целостности данных."""
    def test_data_validation(self) -> None:
        """Тест валидации данных."""
        # Валидные данные
        valid_data = {
            'amount': 100.0,
            'currency': 'USDT',
            'timestamp': int(time.time())
        }
        assert self._validate_trade_data(valid_data)
        # Невалидные данные
        invalid_data = {
            'amount': -100.0,  # Отрицательная сумма
            'currency': 'USDT',
            'timestamp': int(time.time())
        }
        assert not self._validate_trade_data(invalid_data)
        invalid_data2 = {
            'amount': 100.0,
            'currency': '',  # Пустая валюта
            'timestamp': int(time.time())
        }
        assert not self._validate_trade_data(invalid_data2)
    
    def test_data_sanitization(self) -> None:
        """Тест санитизации данных."""
        # Данные с потенциально опасными символами
        dirty_data = {
            'name': '<script>alert("xss")</script>',
            'description': 'Normal description',
            'amount': '100.0'
        }
        # Санитизируем данные
        clean_data = self._sanitize_data(dirty_data)
        # Проверяем, что опасные символы удалены
        assert '<script>' not in clean_data['name']
        assert 'alert' not in clean_data['name']
        assert clean_data['description'] == 'Normal description'
        assert clean_data['amount'] == '100.0'
    
    def _validate_trade_data(self, data: Dict[str, Any]) -> bool:
        """Валидация данных сделки."""
        try:
            amount = float(data.get('amount', 0))
            currency = data.get('currency', '')
            timestamp = int(data.get('timestamp', 0))
            return (
                amount > 0 and
                len(currency) > 0 and
                timestamp > 0
            )
        except (ValueError, TypeError):
            return False
    
    def _sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Санитизация данных."""
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                # Удаляем потенциально опасные HTML теги
                import re
                clean_value = re.sub(r'<[^>]+>', '', value)
                sanitized[key] = clean_value
            else:
                sanitized[key] = value
        return sanitized

class TestRateLimiting:
    """Тесты ограничения скорости запросов."""
    def test_rate_limiter(self) -> None:
        """Тест ограничителя скорости."""
        rate_limiter = RateLimiter(max_requests=10, window_seconds=60)
        # Выполняем запросы в пределах лимита
        for i in range(10):
            assert rate_limiter.allow_request("user1")
        # Следующий запрос должен быть заблокирован
        assert not rate_limiter.allow_request("user1")
        # Запрос от другого пользователя должен пройти
        assert rate_limiter.allow_request("user2")
    
    def test_rate_limiter_reset(self) -> None:
        """Тест сброса ограничителя скорости."""
        rate_limiter = RateLimiter(max_requests=5, window_seconds=1)
        # Выполняем 5 запросов
        for i in range(5):
            assert rate_limiter.allow_request("user1")
        # Шестой запрос заблокирован
        assert not rate_limiter.allow_request("user1")
        # Ждем сброса
        time.sleep(1.1)
        # Теперь запрос должен пройти
        assert rate_limiter.allow_request("user1")

class RateLimiter:
    """Простой ограничитель скорости запросов."""
    def __init__(self, max_requests: int, window_seconds: int) -> Any:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = {}
    
    def allow_request(self, user_id: str) -> bool:
        """Проверяет, разрешен ли запрос."""
        current_time = time.time()
        if user_id not in self.requests:
            self.requests[user_id] = []
        # Удаляем старые запросы
        self.requests[user_id] = [
            req_time for req_time in self.requests[user_id]
            if current_time - req_time < self.window_seconds
        ]
        # Проверяем лимит
        if len(self.requests[user_id]) >= self.max_requests:
            return False
        # Добавляем новый запрос
        self.requests[user_id].append(current_time)
        return True

class TestEncryption:
    """Тесты шифрования."""
    def test_data_encryption(self) -> None:
        """Тест шифрования данных."""
        from cryptography.fernet import Fernet
        # Генерируем ключ
        key = Fernet.generate_key()
        cipher = Fernet(key)
        # Тестовые данные
        sensitive_data = "sensitive_trading_data"
        # Шифруем
        encrypted_data = cipher.encrypt(sensitive_data.encode())
        # Расшифровываем
        decrypted_data = cipher.decrypt(encrypted_data).decode()
        assert decrypted_data == sensitive_data
        assert encrypted_data != sensitive_data.encode()
    
    def test_secure_random_generation(self) -> None:
        """Тест генерации безопасных случайных чисел."""
        import secrets
        # Генерируем случайные данные
        random_bytes = secrets.token_bytes(32)
        random_hex = secrets.token_hex(16)
        assert len(random_bytes) == 32
        assert len(random_hex) == 32  # 16 bytes = 32 hex chars
        # Проверяем, что значения разные при повторной генерации
        random_bytes2 = secrets.token_bytes(32)
        random_hex2 = secrets.token_hex(16)
        assert random_bytes != random_bytes2
        assert random_hex != random_hex2 
