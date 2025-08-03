"""
Unit тесты для domain/protocols/security.py.

Покрывает:
- CryptoManager
- AuthenticationManager
- AuthorizationManager
- AuditManager
- SecurityManager
- Декораторы безопасности
- Обработку ошибок
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List, Set, Optional
from datetime import datetime, timedelta
from uuid import uuid4

from domain.protocols.security import (
    SecurityLevel,
    Permission,
    AuditEvent,
    SecurityContext,
    AuditLog,
    SecurityViolation,
    CryptoManager,
    AuthenticationManager,
    AuthorizationManager,
    AuditManager,
    SecurityManager,
    require_authentication,
    encrypt_sensitive_fields,
    audit_security_events,
)
from domain.exceptions.protocol_exceptions import (
    ProtocolAuthenticationError,
    ProtocolAuthorizationError,
    ProtocolIntegrityError,
    ProtocolSecurityError,
)


class TestSecurityLevel:
    """Тесты для SecurityLevel."""

    def test_enum_values(self):
        """Тест значений enum."""
        assert SecurityLevel.LOW.value == "low"
        assert SecurityLevel.MEDIUM.value == "medium"
        assert SecurityLevel.HIGH.value == "high"
        assert SecurityLevel.CRITICAL.value == "critical"


class TestPermission:
    """Тесты для Permission."""

    def test_enum_values(self):
        """Тест значений enum."""
        assert Permission.READ.value == "read"
        assert Permission.WRITE.value == "write"
        assert Permission.DELETE.value == "delete"
        assert Permission.EXECUTE.value == "execute"
        assert Permission.ADMIN.value == "admin"


class TestAuditEvent:
    """Тесты для AuditEvent."""

    def test_enum_values(self):
        """Тест значений enum."""
        assert AuditEvent.LOGIN.value == "login"
        assert AuditEvent.LOGOUT.value == "logout"
        assert AuditEvent.ACCESS_DENIED.value == "access_denied"
        assert AuditEvent.DATA_ACCESS.value == "data_access"
        assert AuditEvent.DATA_MODIFY.value == "data_modify"
        assert AuditEvent.CONFIG_CHANGE.value == "config_change"
        assert AuditEvent.SECURITY_VIOLATION.value == "security_violation"


class TestSecurityContext:
    """Тесты для SecurityContext."""

    def test_creation(self):
        """Тест создания контекста безопасности."""
        timestamp = datetime.now()
        permissions = {Permission.READ, Permission.WRITE}
        
        context = SecurityContext(
            user_id="user123",
            session_id="session456",
            permissions=permissions,
            security_level=SecurityLevel.HIGH,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            timestamp=timestamp,
            metadata={"key": "value"}
        )

        assert context.user_id == "user123"
        assert context.session_id == "session456"
        assert context.permissions == permissions
        assert context.security_level == SecurityLevel.HIGH
        assert context.ip_address == "192.168.1.1"
        assert context.user_agent == "Mozilla/5.0"
        assert context.timestamp == timestamp
        assert context.metadata == {"key": "value"}


class TestAuditLog:
    """Тесты для AuditLog."""

    def test_creation(self):
        """Тест создания записи аудита."""
        log_id = uuid4()
        timestamp = datetime.now()
        
        log = AuditLog(
            id=log_id,
            event=AuditEvent.LOGIN,
            user_id="user123",
            session_id="session456",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            resource="/api/data",
            action="GET",
            result="success",
            timestamp=timestamp,
            metadata={"key": "value"}
        )

        assert log.id == log_id
        assert log.event == AuditEvent.LOGIN
        assert log.user_id == "user123"
        assert log.session_id == "session456"
        assert log.ip_address == "192.168.1.1"
        assert log.user_agent == "Mozilla/5.0"
        assert log.resource == "/api/data"
        assert log.action == "GET"
        assert log.result == "success"
        assert log.timestamp == timestamp
        assert log.metadata == {"key": "value"}


class TestSecurityViolation:
    """Тесты для SecurityViolation."""

    def test_creation(self):
        """Тест создания нарушения безопасности."""
        violation_id = uuid4()
        timestamp = datetime.now()
        
        violation = SecurityViolation(
            id=violation_id,
            violation_type="brute_force",
            severity=SecurityLevel.HIGH,
            description="Multiple failed login attempts",
            user_id="user123",
            ip_address="192.168.1.1",
            timestamp=timestamp,
            context={"attempts": 10}
        )

        assert violation.id == violation_id
        assert violation.violation_type == "brute_force"
        assert violation.severity == SecurityLevel.HIGH
        assert violation.description == "Multiple failed login attempts"
        assert violation.user_id == "user123"
        assert violation.ip_address == "192.168.1.1"
        assert violation.timestamp == timestamp
        assert violation.context == {"attempts": 10}


class TestCryptoManager:
    """Тесты для CryptoManager."""

    @pytest.fixture
    def crypto_manager(self):
        """Фикстура менеджера криптографии."""
        return CryptoManager()

    def test_initialization(self, crypto_manager):
        """Тест инициализации."""
        assert crypto_manager.master_key is not None
        assert len(crypto_manager.master_key) > 0

    def test_generate_key(self, crypto_manager):
        """Тест генерации ключа."""
        key = crypto_manager.generate_key()
        assert isinstance(key, bytes)
        assert len(key) > 0

    def test_derive_key(self, crypto_manager):
        """Тест получения ключа из пароля."""
        password = "test_password"
        salt = b"test_salt"
        
        key = crypto_manager.derive_key(password, salt)
        assert isinstance(key, bytes)
        assert len(key) > 0

    def test_encrypt_decrypt_data(self, crypto_manager):
        """Тест шифрования и дешифрования данных."""
        original_data = b"test_data"
        
        encrypted = crypto_manager.encrypt_data(original_data)
        assert isinstance(encrypted, bytes)
        assert encrypted != original_data
        
        decrypted = crypto_manager.decrypt_data(encrypted)
        assert decrypted == original_data

    def test_encrypt_decrypt_string(self, crypto_manager):
        """Тест шифрования и дешифрования строк."""
        original_text = "test_string"
        
        encrypted = crypto_manager.encrypt_string(original_text)
        assert isinstance(encrypted, str)
        assert encrypted != original_text
        
        decrypted = crypto_manager.decrypt_string(encrypted)
        assert decrypted == original_text

    def test_hash_password(self, crypto_manager):
        """Тест хеширования пароля."""
        password = "test_password"
        
        hashed = crypto_manager.hash_password(password)
        assert isinstance(hashed, str)
        assert hashed != password

    def test_verify_password(self, crypto_manager):
        """Тест проверки пароля."""
        password = "test_password"
        hashed = crypto_manager.hash_password(password)
        
        assert crypto_manager.verify_password(password, hashed) is True
        assert crypto_manager.verify_password("wrong_password", hashed) is False

    def test_generate_verify_hmac(self, crypto_manager):
        """Тест генерации и проверки HMAC."""
        data = b"test_data"
        key = crypto_manager.generate_key()
        
        signature = crypto_manager.generate_hmac(data, key)
        assert isinstance(signature, str)
        
        assert crypto_manager.verify_hmac(data, key, signature) is True
        assert crypto_manager.verify_hmac(data, key, "wrong_signature") is False


class TestAuthenticationManager:
    """Тесты для AuthenticationManager."""

    @pytest.fixture
    def auth_manager(self):
        """Фикстура менеджера аутентификации."""
        return AuthenticationManager()

    def test_initialization(self, auth_manager):
        """Тест инициализации."""
        assert auth_manager.users == {}
        assert auth_manager.sessions == {}
        assert auth_manager.failed_attempts == {}
        assert auth_manager.max_failed_attempts == 5
        assert auth_manager.lockout_duration == 300

    async def test_register_user(self, auth_manager):
        """Тест регистрации пользователя."""
        username = "test_user"
        password = "test_password"
        email = "test@example.com"
        permissions = {Permission.READ, Permission.WRITE}
        
        result = await auth_manager.register_user(username, password, email, permissions)
        assert result is True
        
        assert username in auth_manager.users
        user = auth_manager.users[username]
        assert user["email"] == email
        assert user["permissions"] == permissions
        assert "password_hash" in user

    async def test_authenticate_user_success(self, auth_manager):
        """Тест успешной аутентификации."""
        username = "test_user"
        password = "test_password"
        email = "test@example.com"
        permissions = {Permission.READ}
        
        await auth_manager.register_user(username, password, email, permissions)
        
        context = await auth_manager.authenticate_user(
            username, password, "192.168.1.1", "Mozilla/5.0"
        )
        
        assert context is not None
        assert context.user_id == username
        assert context.permissions == permissions
        assert context.ip_address == "192.168.1.1"
        assert context.user_agent == "Mozilla/5.0"

    async def test_authenticate_user_failure(self, auth_manager):
        """Тест неудачной аутентификации."""
        username = "test_user"
        password = "test_password"
        email = "test@example.com"
        permissions = {Permission.READ}
        
        await auth_manager.register_user(username, password, email, permissions)
        
        context = await auth_manager.authenticate_user(
            username, "wrong_password", "192.168.1.1", "Mozilla/5.0"
        )
        
        assert context is None

    async def test_validate_session(self, auth_manager):
        """Тест валидации сессии."""
        username = "test_user"
        password = "test_password"
        email = "test@example.com"
        permissions = {Permission.READ}
        
        await auth_manager.register_user(username, password, email, permissions)
        
        # Сначала аутентифицируемся
        context = await auth_manager.authenticate_user(
            username, password, "192.168.1.1", "Mozilla/5.0"
        )
        
        # Затем валидируем сессию
        validated_context = await auth_manager.validate_session(
            context.session_id, "192.168.1.1"
        )
        
        assert validated_context is not None
        assert validated_context.user_id == username

    async def test_logout(self, auth_manager):
        """Тест выхода из системы."""
        username = "test_user"
        password = "test_password"
        email = "test@example.com"
        permissions = {Permission.READ}
        
        await auth_manager.register_user(username, password, email, permissions)
        
        context = await auth_manager.authenticate_user(
            username, password, "192.168.1.1", "Mozilla/5.0"
        )
        
        result = await auth_manager.logout(context.session_id)
        assert result is True
        
        # Проверяем, что сессия удалена
        validated_context = await auth_manager.validate_session(
            context.session_id, "192.168.1.1"
        )
        assert validated_context is None

    def test_is_account_locked(self, auth_manager):
        """Тест проверки блокировки аккаунта."""
        username = "test_user"
        
        # Изначально аккаунт не заблокирован
        assert auth_manager._is_account_locked(username) is False
        
        # Симулируем блокировку
        auth_manager.failed_attempts[username] = {
            "count": 5,
            "last_attempt": datetime.now()
        }
        
        assert auth_manager._is_account_locked(username) is True

    async def test_record_failed_attempt(self, auth_manager):
        """Тест записи неудачной попытки."""
        username = "test_user"
        ip_address = "192.168.1.1"
        
        await auth_manager._record_failed_attempt(username, ip_address)
        
        assert username in auth_manager.failed_attempts
        assert auth_manager.failed_attempts[username]["count"] == 1


class TestAuthorizationManager:
    """Тесты для AuthorizationManager."""

    @pytest.fixture
    def auth_manager(self):
        """Фикстура менеджера авторизации."""
        return AuthorizationManager()

    def test_initialization(self, auth_manager):
        """Тест инициализации."""
        assert auth_manager.resource_permissions == {}
        assert auth_manager.role_permissions == {}
        assert auth_manager.user_roles == {}

    async def test_check_permission_allowed(self, auth_manager):
        """Тест проверки разрешения - разрешено."""
        context = SecurityContext(
            user_id="user123",
            session_id="session456",
            permissions={Permission.READ, Permission.WRITE},
            security_level=SecurityLevel.HIGH,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            timestamp=datetime.now()
        )
        
        result = await auth_manager.check_permission(context, "/api/data", Permission.READ)
        assert result is True

    async def test_check_permission_denied(self, auth_manager):
        """Тест проверки разрешения - запрещено."""
        context = SecurityContext(
            user_id="user123",
            session_id="session456",
            permissions={Permission.READ},
            security_level=SecurityLevel.HIGH,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            timestamp=datetime.now()
        )
        
        result = await auth_manager.check_permission(context, "/api/data", Permission.WRITE)
        assert result is False

    async def test_require_permission_success(self, auth_manager):
        """Тест требования разрешения - успех."""
        context = SecurityContext(
            user_id="user123",
            session_id="session456",
            permissions={Permission.READ},
            security_level=SecurityLevel.HIGH,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            timestamp=datetime.now()
        )
        
        # Не должно вызывать исключение
        await auth_manager.require_permission(context, "/api/data", Permission.READ)

    async def test_require_permission_failure(self, auth_manager):
        """Тест требования разрешения - неудача."""
        context = SecurityContext(
            user_id="user123",
            session_id="session456",
            permissions={Permission.READ},
            security_level=SecurityLevel.HIGH,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            timestamp=datetime.now()
        )
        
        with pytest.raises(ProtocolAuthorizationError):
            await auth_manager.require_permission(context, "/api/data", Permission.WRITE)

    async def test_add_resource_permission(self, auth_manager):
        """Тест добавления разрешения для ресурса."""
        await auth_manager.add_resource_permission("/api/data", Permission.READ)
        
        assert "/api/data" in auth_manager.resource_permissions
        assert Permission.READ in auth_manager.resource_permissions["/api/data"]

    async def test_add_role_permission(self, auth_manager):
        """Тест добавления разрешения для роли."""
        await auth_manager.add_role_permission("admin", Permission.ADMIN)
        
        assert "admin" in auth_manager.role_permissions
        assert Permission.ADMIN in auth_manager.role_permissions["admin"]

    async def test_assign_user_role(self, auth_manager):
        """Тест назначения роли пользователю."""
        await auth_manager.assign_user_role("user123", "admin")
        
        assert "user123" in auth_manager.user_roles
        assert auth_manager.user_roles["user123"] == "admin"


class TestAuditManager:
    """Тесты для AuditManager."""

    @pytest.fixture
    def audit_manager(self):
        """Фикстура менеджера аудита."""
        return AuditManager()

    def test_initialization(self, audit_manager):
        """Тест инициализации."""
        assert audit_manager.audit_logs == []
        assert audit_manager.security_violations == []

    async def test_log_event(self, audit_manager):
        """Тест записи события."""
        context = SecurityContext(
            user_id="user123",
            session_id="session456",
            permissions={Permission.READ},
            security_level=SecurityLevel.HIGH,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            timestamp=datetime.now()
        )
        
        await audit_manager.log_event(
            AuditEvent.LOGIN,
            context,
            "/api/auth",
            "POST",
            "success",
            {"key": "value"}
        )
        
        assert len(audit_manager.audit_logs) == 1
        log = audit_manager.audit_logs[0]
        assert log.event == AuditEvent.LOGIN
        assert log.user_id == "user123"
        assert log.resource == "/api/auth"
        assert log.action == "POST"
        assert log.result == "success"

    async def test_log_security_violation(self, audit_manager):
        """Тест записи нарушения безопасности."""
        context = SecurityContext(
            user_id="user123",
            session_id="session456",
            permissions={Permission.READ},
            security_level=SecurityLevel.HIGH,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            timestamp=datetime.now()
        )
        
        await audit_manager.log_security_violation(
            "brute_force",
            SecurityLevel.HIGH,
            "Multiple failed login attempts",
            context,
            {"attempts": 10}
        )
        
        assert len(audit_manager.security_violations) == 1
        violation = audit_manager.security_violations[0]
        assert violation.violation_type == "brute_force"
        assert violation.severity == SecurityLevel.HIGH
        assert violation.description == "Multiple failed login attempts"

    async def test_get_audit_logs(self, audit_manager):
        """Тест получения записей аудита."""
        context = SecurityContext(
            user_id="user123",
            session_id="session456",
            permissions={Permission.READ},
            security_level=SecurityLevel.HIGH,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            timestamp=datetime.now()
        )
        
        await audit_manager.log_event(
            AuditEvent.LOGIN,
            context,
            "/api/auth",
            "POST",
            "success"
        )
        
        logs = await audit_manager.get_audit_logs(user_id="user123", hours=24)
        assert len(logs) == 1
        assert logs[0].user_id == "user123"

    async def test_get_security_violations(self, audit_manager):
        """Тест получения нарушений безопасности."""
        context = SecurityContext(
            user_id="user123",
            session_id="session456",
            permissions={Permission.READ},
            security_level=SecurityLevel.HIGH,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            timestamp=datetime.now()
        )
        
        await audit_manager.log_security_violation(
            "brute_force",
            SecurityLevel.HIGH,
            "Multiple failed login attempts",
            context
        )
        
        violations = await audit_manager.get_security_violations(
            severity=SecurityLevel.HIGH, hours=24
        )
        assert len(violations) == 1
        assert violations[0].severity == SecurityLevel.HIGH


class TestSecurityManager:
    """Тесты для SecurityManager."""

    @pytest.fixture
    def security_manager(self):
        """Фикстура менеджера безопасности."""
        return SecurityManager()

    def test_initialization(self, security_manager):
        """Тест инициализации."""
        assert security_manager.crypto_manager is not None
        assert security_manager.auth_manager is not None
        assert security_manager.authz_manager is not None
        assert security_manager.audit_manager is not None
        assert security_manager.rate_limits == {}

    async def test_authenticate_and_authorize(self, security_manager):
        """Тест аутентификации и авторизации."""
        # Регистрируем пользователя
        await security_manager.auth_manager.register_user(
            "test_user",
            "test_password",
            "test@example.com",
            {Permission.READ, Permission.WRITE}
        )
        
        # Добавляем разрешение для ресурса
        await security_manager.authz_manager.add_resource_permission(
            "/api/data", Permission.READ
        )
        
        context = await security_manager.authenticate_and_authorize(
            "test_user",
            "test_password",
            "192.168.1.1",
            "Mozilla/5.0",
            "/api/data",
            Permission.READ
        )
        
        assert context is not None
        assert context.user_id == "test_user"
        assert Permission.READ in context.permissions

    async def test_validate_session_and_authorize(self, security_manager):
        """Тест валидации сессии и авторизации."""
        # Регистрируем пользователя
        await security_manager.auth_manager.register_user(
            "test_user",
            "test_password",
            "test@example.com",
            {Permission.READ}
        )
        
        # Аутентифицируемся
        auth_context = await security_manager.auth_manager.authenticate_user(
            "test_user",
            "test_password",
            "192.168.1.1",
            "Mozilla/5.0"
        )
        
        # Добавляем разрешение
        await security_manager.authz_manager.add_resource_permission(
            "/api/data", Permission.READ
        )
        
        context = await security_manager.validate_session_and_authorize(
            auth_context.session_id,
            "192.168.1.1",
            "/api/data",
            Permission.READ
        )
        
        assert context is not None
        assert context.user_id == "test_user"

    async def test_check_rate_limit(self, security_manager):
        """Тест проверки ограничения скорости."""
        key = "user123"
        limit = 5
        window = 60
        
        # Первые 5 запросов должны быть разрешены
        for i in range(5):
            result = await security_manager.check_rate_limit(key, limit, window)
            assert result is True
        
        # 6-й запрос должен быть заблокирован
        result = await security_manager.check_rate_limit(key, limit, window)
        assert result is False

    async def test_encrypt_decrypt_sensitive_data(self, security_manager):
        """Тест шифрования и дешифрования чувствительных данных."""
        original_data = {
            "username": "test_user",
            "password": "secret_password",
            "email": "test@example.com",
            "public_info": "not_secret"
        }
        
        encrypted = await security_manager.encrypt_sensitive_data(original_data)
        assert encrypted["username"] != original_data["username"]
        assert encrypted["password"] != original_data["password"]
        assert encrypted["email"] != original_data["email"]
        assert encrypted["public_info"] == original_data["public_info"]
        
        decrypted = await security_manager.decrypt_sensitive_data(encrypted)
        assert decrypted == original_data

    def test_is_sensitive_field(self, security_manager):
        """Тест проверки чувствительных полей."""
        assert security_manager._is_sensitive_field("password") is True
        assert security_manager._is_sensitive_field("email") is True
        assert security_manager._is_sensitive_field("username") is True
        assert security_manager._is_sensitive_field("public_info") is False

    async def test_get_security_report(self, security_manager):
        """Тест получения отчета о безопасности."""
        # Создаем тестовые данные
        context = SecurityContext(
            user_id="user123",
            session_id="session456",
            permissions={Permission.READ},
            security_level=SecurityLevel.HIGH,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            timestamp=datetime.now()
        )
        
        await security_manager.audit_manager.log_event(
            AuditEvent.LOGIN,
            context,
            "/api/auth",
            "POST",
            "success"
        )
        
        await security_manager.audit_manager.log_security_violation(
            "brute_force",
            SecurityLevel.HIGH,
            "Multiple failed login attempts",
            context
        )
        
        report = await security_manager.get_security_report(hours=24)
        assert isinstance(report, dict)
        assert "total_events" in report
        assert "security_violations" in report
        assert "active_sessions" in report


class TestSecurityDecorators:
    """Тесты для декораторов безопасности."""

    def test_require_authentication_decorator(self):
        """Тест декоратора требования аутентификации."""
        context = SecurityContext(
            user_id="user123",
            session_id="session456",
            permissions={Permission.READ},
            security_level=SecurityLevel.HIGH,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            timestamp=datetime.now()
        )
        
        @require_authentication("/api/data", Permission.READ)
        async def test_function(security_context: SecurityContext):
            return "success"
        
        result = await test_function(context)
        assert result == "success"

    def test_encrypt_sensitive_fields_decorator(self):
        """Тест декоратора шифрования чувствительных полей."""
        @encrypt_sensitive_fields(["password", "email"])
        async def test_function(data: Dict[str, Any]):
            return data
        
        original_data = {
            "username": "test_user",
            "password": "secret_password",
            "email": "test@example.com"
        }
        
        result = await test_function(original_data)
        assert result["username"] == original_data["username"]
        assert result["password"] != original_data["password"]
        assert result["email"] != original_data["email"]

    def test_audit_security_events_decorator(self):
        """Тест декоратора аудита событий безопасности."""
        @audit_security_events(AuditEvent.DATA_ACCESS, "/api/data", "GET")
        async def test_function():
            return "success"
        
        result = await test_function()
        assert result == "success"


class TestSecurityIntegration:
    """Интеграционные тесты безопасности."""

    async def test_full_security_workflow(self):
        """Тест полного рабочего процесса безопасности."""
        security_manager = SecurityManager()
        
        # 1. Регистрация пользователя
        await security_manager.auth_manager.register_user(
            "test_user",
            "test_password",
            "test@example.com",
            {Permission.READ, Permission.WRITE}
        )
        
        # 2. Добавление разрешений
        await security_manager.authz_manager.add_resource_permission(
            "/api/data", Permission.READ
        )
        await security_manager.authz_manager.add_resource_permission(
            "/api/data", Permission.WRITE
        )
        
        # 3. Аутентификация и авторизация
        context = await security_manager.authenticate_and_authorize(
            "test_user",
            "test_password",
            "192.168.1.1",
            "Mozilla/5.0",
            "/api/data",
            Permission.READ
        )
        
        assert context is not None
        assert context.user_id == "test_user"
        
        # 4. Проверка ограничения скорости
        for i in range(3):
            result = await security_manager.check_rate_limit("test_user", 5, 60)
            assert result is True
        
        # 5. Шифрование чувствительных данных
        sensitive_data = {
            "username": "test_user",
            "password": "secret_password",
            "email": "test@example.com"
        }
        
        encrypted = await security_manager.encrypt_sensitive_data(sensitive_data)
        decrypted = await security_manager.decrypt_sensitive_data(encrypted)
        
        assert decrypted == sensitive_data
        
        # 6. Получение отчета о безопасности
        report = await security_manager.get_security_report(hours=24)
        assert isinstance(report, dict)

    async def test_security_error_handling(self):
        """Тест обработки ошибок безопасности."""
        security_manager = SecurityManager()
        
        # Тест с неверными учетными данными
        context = await security_manager.authenticate_and_authorize(
            "nonexistent_user",
            "wrong_password",
            "192.168.1.1",
            "Mozilla/5.0",
            "/api/data",
            Permission.READ
        )
        
        assert context is None
        
        # Тест с недостаточными разрешениями
        await security_manager.auth_manager.register_user(
            "limited_user",
            "password",
            "limited@example.com",
            {Permission.READ}
        )
        
        with pytest.raises(ProtocolAuthorizationError):
            await security_manager.authenticate_and_authorize(
                "limited_user",
                "password",
                "192.168.1.1",
                "Mozilla/5.0",
                "/api/data",
                Permission.WRITE
            ) 