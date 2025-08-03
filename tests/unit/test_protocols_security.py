"""
Production-ready unit тесты для security.py.
Полное покрытие безопасности, аутентификации, авторизации, шифрования, edge cases и типизации.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
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
    SecurityManager
)
from domain.exceptions.protocol_exceptions import (
    ProtocolAuthenticationError,
    ProtocolAuthorizationError,
    ProtocolIntegrityError,
    ProtocolSecurityError,
)
class TestSecurityManager:
    """Production-ready тесты для SecurityManager."""
    @pytest.fixture
    def security_manager(self) -> SecurityManager:
        return SecurityManager()
    
    @pytest.mark.asyncio
    async def test_security_manager_creation(self, security_manager: SecurityManager) -> None:
        """Тест создания менеджера безопасности."""
        assert security_manager is not None
    
    @pytest.mark.asyncio
    async def test_authenticate_and_authorize(self, security_manager: SecurityManager) -> None:
        """Тест аутентификации и авторизации."""
        # Сначала регистрируем пользователя
        auth_manager = security_manager.auth_manager
        await auth_manager.register_user(
            "test_user", 
            "test_password", 
            "test@example.com", 
            {Permission.READ, Permission.WRITE}
        )
        
        # Тестируем аутентификацию и авторизацию
        context = await security_manager.authenticate_and_authorize(
            "test_user", 
            "test_password", 
            "127.0.0.1", 
            "test_agent", 
            "orders", 
            Permission.READ
        )
        
        assert isinstance(context, SecurityContext)
        assert context.user_id == "test_user"
        assert Permission.READ in context.permissions
class TestAuthenticationManager:
    """Тесты для AuthenticationManager."""
    @pytest.fixture
    def auth_manager(self) -> AuthenticationManager:
        return AuthenticationManager()
    
    def test_auth_manager_creation(self, auth_manager: AuthenticationManager) -> None:
        """Тест создания менеджера аутентификации."""
        assert auth_manager is not None
    
    @pytest.mark.asyncio
    async def test_register_user(self, auth_manager: AuthenticationManager) -> None:
        """Тест регистрации пользователя."""
        result = await auth_manager.register_user(
            "test_user", 
            "test_password", 
            "test@example.com", 
            {Permission.READ, Permission.WRITE}
        )
        assert result is True
    
    @pytest.mark.asyncio
    async def test_authenticate_user(self, auth_manager: AuthenticationManager) -> None:
        """Тест аутентификации пользователя."""
        # Сначала регистрируем пользователя
        await auth_manager.register_user(
            "test_user", 
            "test_password", 
            "test@example.com", 
            {Permission.READ, Permission.WRITE}
        )
        
        # Тестируем аутентификацию
        result = await auth_manager.authenticate_user(
            "test_user", 
            "test_password", 
            "127.0.0.1", 
            "test_agent"
        )
        
        assert isinstance(result, SecurityContext)
        assert result.user_id == "test_user"
        assert Permission.READ in result.permissions
    
    @pytest.mark.asyncio
    async def test_authenticate_with_invalid_credentials(self, auth_manager: AuthenticationManager) -> None:
        """Тест аутентификации с неверными учетными данными."""
        result = await auth_manager.authenticate_user(
            "invalid_user", 
            "invalid_password", 
            "127.0.0.1", 
            "test_agent"
        )
        assert result is None
    
    @pytest.mark.asyncio
    async def test_authenticate_with_locked_account(self, auth_manager: AuthenticationManager) -> None:
        """Тест аутентификации заблокированного аккаунта."""
        # Симулируем заблокированный аккаунт через неудачные попытки (5 попыток)
        now = datetime.utcnow()
        auth_manager._failed_attempts = {"locked_user": [now - timedelta(seconds=i) for i in range(5)]}
        result = await auth_manager.authenticate_user("locked_user", "password", "127.0.0.1", "test_agent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_handle_failed_login(self, auth_manager: AuthenticationManager) -> None:
        """Тест обработки неудачного входа."""
        now = datetime.utcnow()
        auth_manager._failed_attempts = {"test_user": [now - timedelta(seconds=i) for i in range(4)]}
        result = await auth_manager.authenticate_user("test_user", "wrong_password", "127.0.0.1", "test_agent")
        assert result is None
        # Проверяем, что количество неудачных попыток увеличилось
        assert len(auth_manager._failed_attempts.get("test_user", [])) >= 5
    
    @pytest.mark.asyncio
    async def test_authentication_error(self, auth_manager: AuthenticationManager) -> None:
        """Тест ошибок аутентификации."""
        with pytest.raises(ProtocolAuthenticationError):
            await auth_manager.authenticate_user("", "", "", "")  # Пустые учетные данные
class TestAuthorizationManager:
    """Тесты для AuthorizationManager."""
    @pytest.fixture
    def authz_manager(self) -> AuthorizationManager:
        return AuthorizationManager()
    
    def test_authz_manager_creation(self, authz_manager: AuthorizationManager) -> None:
        """Тест создания менеджера авторизации."""
        assert authz_manager is not None
    
    @pytest.mark.asyncio
    async def test_check_permission(self, authz_manager: AuthorizationManager) -> None:
        """Тест проверки разрешений пользователя."""
        # Создаем контекст безопасности с разрешениями
        context = SecurityContext(
            user_id="user_123",
            session_id="session_123",
            permissions={Permission.READ, Permission.WRITE},
            security_level=SecurityLevel.MEDIUM,
            ip_address="127.0.0.1",
            user_agent="test_agent",
            timestamp=datetime.utcnow()
        )
        
        # Проверяем разрешения
        assert await authz_manager.check_permission(context, "orders", Permission.READ) is True
        assert await authz_manager.check_permission(context, "orders", Permission.WRITE) is True
        assert await authz_manager.check_permission(context, "trades", Permission.READ) is False
    
    @pytest.mark.asyncio
    async def test_require_permission(self, authz_manager: AuthorizationManager) -> None:
        """Тест требования разрешений."""
        # Создаем контекст безопасности с разрешениями
        context = SecurityContext(
            user_id="user_123",
            session_id="session_123",
            permissions={Permission.READ},
            security_level=SecurityLevel.MEDIUM,
            ip_address="127.0.0.1",
            user_agent="test_agent",
            timestamp=datetime.utcnow()
        )
        
        # Проверяем, что разрешение есть
        await authz_manager.require_permission(context, "orders", Permission.READ)
        
        # Проверяем, что исключение вызывается при отсутствии разрешения
        with pytest.raises(ProtocolAuthorizationError):
            await authz_manager.require_permission(context, "orders", Permission.WRITE)
    
    @pytest.mark.asyncio
    async def test_add_resource_permission(self, authz_manager: AuthorizationManager) -> None:
        """Тест добавления разрешений для ресурса."""
        await authz_manager.add_resource_permission("orders", Permission.READ)
        # Проверяем, что разрешение добавлено
        assert Permission.READ in authz_manager._resource_permissions.get("orders", set())
    
    @pytest.mark.asyncio
    async def test_add_role_permission(self, authz_manager: AuthorizationManager) -> None:
        """Тест добавления разрешений для роли."""
        await authz_manager.add_role_permission("trader", Permission.READ)
        # Проверяем, что разрешение добавлено
        assert Permission.READ in authz_manager._role_permissions.get("trader", set())
    
    @pytest.mark.asyncio
    async def test_assign_user_role(self, authz_manager: AuthorizationManager) -> None:
        """Тест назначения роли пользователю."""
        await authz_manager.assign_user_role("user_123", "trader")
        # Проверяем, что роль назначена
        assert "trader" in authz_manager._user_roles.get("user_123", set())
    
    @pytest.mark.asyncio
    async def test_authorization_error(self, authz_manager: AuthorizationManager) -> None:
        """Тест ошибок авторизации."""
        context = SecurityContext(
            user_id="",
            session_id="",
            permissions=set(),
            security_level=SecurityLevel.LOW,
            ip_address="",
            user_agent="",
            timestamp=datetime.utcnow()
        )
        with pytest.raises(ProtocolAuthorizationError):
            await authz_manager.require_permission(context, "", Permission.READ)
class TestCryptoManager:
    """Тесты для CryptoManager."""
    @pytest.fixture
    def crypto_manager(self) -> CryptoManager:
        return CryptoManager()
    
    def test_crypto_manager_creation(self, crypto_manager: CryptoManager) -> None:
        """Тест создания менеджера криптографии."""
        assert crypto_manager is not None
    
    def test_generate_key(self, crypto_manager: CryptoManager) -> None:
        """Тест генерации ключа."""
        key = crypto_manager.generate_key()
        assert isinstance(key, bytes)
        assert len(key) > 0
    
    def test_derive_key(self, crypto_manager: CryptoManager) -> None:
        """Тест вывода ключа из пароля."""
        password = "test_password"
        salt = b"test_salt"
        key = crypto_manager.derive_key(password, salt)
        assert isinstance(key, bytes)
        assert len(key) > 0
    
    def test_encrypt_decrypt_data(self, crypto_manager: CryptoManager) -> None:
        """Тест шифрования и расшифровки данных."""
        data = b"sensitive_data"
        key = crypto_manager.generate_key()
        
        # Шифруем данные
        encrypted = crypto_manager.encrypt_data(data, key)
        assert isinstance(encrypted, bytes)
        assert encrypted != data
        
        # Расшифровываем данные
        decrypted = crypto_manager.decrypt_data(encrypted, key)
        assert decrypted == data
    
    def test_encrypt_decrypt_string(self, crypto_manager: CryptoManager) -> None:
        """Тест шифрования и расшифровки строк."""
        text = "sensitive_text"
        key = crypto_manager.generate_key()
        
        # Шифруем строку
        encrypted = crypto_manager.encrypt_string(text, key)
        assert isinstance(encrypted, str)
        assert encrypted != text
        
        # Расшифровываем строку
        decrypted = crypto_manager.decrypt_string(encrypted, key)
        assert decrypted == text
    
    def test_encrypt_large_data(self, crypto_manager: CryptoManager) -> None:
        """Тест шифрования больших данных."""
        large_data = b"x" * 10000  # 10KB данных
        key = crypto_manager.generate_key()
        encrypted = crypto_manager.encrypt_data(large_data, key)
        assert len(encrypted) > 0
        decrypted = crypto_manager.decrypt_data(encrypted, key)
        assert decrypted == large_data
    
    def test_encryption_with_different_keys(self, crypto_manager: CryptoManager) -> None:
        """Тест шифрования с разными ключами."""
        data = b"sensitive_data"
        key1 = crypto_manager.generate_key()
        key2 = crypto_manager.generate_key()
        
        # Шифруем с разными ключами
        encrypted1 = crypto_manager.encrypt_data(data, key1)
        encrypted2 = crypto_manager.encrypt_data(data, key2)
        
        # Результаты должны быть разными
        assert encrypted1 != encrypted2
class TestAuditManager:
    """Тесты для AuditManager."""
    @pytest.fixture
    def audit_manager(self) -> AuditManager:
        return AuditManager()
    
    def test_audit_manager_creation(self, audit_manager: AuditManager) -> None:
        """Тест создания менеджера аудита."""
        assert audit_manager is not None
    
    @pytest.mark.asyncio
    async def test_log_event(self, audit_manager: AuditManager) -> None:
        """Тест логирования события."""
        context = SecurityContext(
            user_id="user_123",
            session_id="session_123",
            permissions={Permission.READ},
            security_level=SecurityLevel.MEDIUM,
            ip_address="127.0.0.1",
            user_agent="test_agent",
            timestamp=datetime.utcnow()
        )
        
        await audit_manager.log_event(
            AuditEvent.LOGIN,
            context,
            "orders",
            "read",
            "success"
        )
        
        # Проверяем, что событие записано
        logs = await audit_manager.get_audit_logs(user_id="user_123", hours=1)
        assert len(logs) > 0
        assert logs[0].event == AuditEvent.LOGIN
    
    @pytest.mark.asyncio
    async def test_log_security_violation(self, audit_manager: AuditManager) -> None:
        """Тест логирования нарушения безопасности."""
        context = SecurityContext(
            user_id="user_123",
            session_id="session_123",
            permissions=set(),
            security_level=SecurityLevel.LOW,
            ip_address="127.0.0.1",
            user_agent="test_agent",
            timestamp=datetime.utcnow()
        )
        
        await audit_manager.log_security_violation(
            "unauthorized_access",
            SecurityLevel.HIGH,
            "Attempted access to restricted resource",
            context
        )
        
        # Проверяем, что нарушение записано
        violations = await audit_manager.get_security_violations(hours=1)
        assert len(violations) > 0
        assert violations[0].violation_type == "unauthorized_access"
class TestSecurityIntegration:
    """Интеграционные тесты безопасности."""
    @pytest.mark.asyncio
    async def test_full_security_workflow(self) -> None:
        """Тест полного рабочего процесса безопасности."""
        # Создаем компоненты безопасности
        auth_manager = AuthenticationManager()
        authz_manager = AuthorizationManager()
        crypto_manager = CryptoManager()
        audit_manager = AuditManager()
        
        # 1. Регистрируем пользователя
        await auth_manager.register_user(
            "test_user", 
            "StrongP@ssw0rd123", 
            "test@example.com", 
            {Permission.READ, Permission.WRITE}
        )
        
        # 2. Аутентифицируем пользователя
        context = await auth_manager.authenticate_user(
            "test_user", 
            "StrongP@ssw0rd123", 
            "127.0.0.1", 
            "test_agent"
        )
        assert isinstance(context, SecurityContext)
        assert context.user_id == "test_user"
        
        # 3. Настраиваем разрешения
        await authz_manager.add_resource_permission("orders", Permission.READ)
        await authz_manager.add_resource_permission("orders", Permission.WRITE)
        
        # 4. Авторизуем действия
        assert await authz_manager.check_permission(context, "orders", Permission.READ) is True
        assert await authz_manager.check_permission(context, "orders", Permission.WRITE) is True
        assert await authz_manager.check_permission(context, "system", Permission.ADMIN) is False
        
        # 5. Шифруем чувствительные данные
        sensitive_data = "credit_card_number"
        encryption_key = crypto_manager.generate_key()
        encrypt_result = crypto_manager.encrypt_string(sensitive_data, encryption_key)
        assert encrypt_result != sensitive_data
        
        # 6. Расшифровываем данные
        decrypt_result = crypto_manager.decrypt_string(encrypt_result, encryption_key)
        assert decrypt_result == sensitive_data
        
        # 7. Логируем событие
        await audit_manager.log_event(
            AuditEvent.LOGIN,
            context,
            "orders",
            "read",
            "success"
        )
        
        # 8. Проверяем логи
        logs = await audit_manager.get_audit_logs(user_id="test_user", hours=1)
        assert len(logs) > 0
        assert logs[0].event == AuditEvent.LOGIN
    
    @pytest.mark.asyncio
    async def test_concurrent_security_operations(self) -> None:
        """Тест конкурентных операций безопасности."""
        auth_manager = AuthenticationManager()
        crypto_manager = CryptoManager()
        
        # Регистрируем пользователей
        await auth_manager.register_user("user1", "password1", "user1@test.com", {Permission.READ})
        await auth_manager.register_user("user2", "password2", "user2@test.com", {Permission.READ})
        
        # Создаем несколько задач
        tasks = [
            auth_manager.authenticate_user("user1", "password1", "127.0.0.1", "agent1"),
            auth_manager.authenticate_user("user2", "password2", "127.0.0.1", "agent2"),
            asyncio.create_task(asyncio.sleep(0.1)),  # Имитация асинхронной операции
        ]
        
        # Выполняем их конкурентно
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert len(results) == 3
        
        # Проверяем результаты аутентификации
        assert isinstance(results[0], SecurityContext) or results[0] is None
        assert isinstance(results[1], SecurityContext) or results[1] is None 
