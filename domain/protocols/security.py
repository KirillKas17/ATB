"""
Система безопасности протоколов - промышленный уровень.
Этот модуль содержит систему безопасности для протоколов:
- Шифрование данных
- Аутентификация и авторизация
- Аудит безопасности
- Управление ключами
- Проверка целостности
- Защита от атак
"""

import asyncio
import base64
import functools
import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4

try:
    import bcrypt  # type: ignore

    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    bcrypt = None
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    Fernet = type('Fernet', (), {})  # type: ignore
    hashes = type('hashes', (), {})  # type: ignore
    serialization = type('serialization', (), {})  # type: ignore
    rsa = type('rsa', (), {})  # type: ignore
    padding = type('padding', (), {})  # type: ignore
    PBKDF2HMAC = type('PBKDF2HMAC', (), {})  # type: ignore
from domain.exceptions.protocol_exceptions import (
    ProtocolAuthenticationError,
    ProtocolAuthorizationError,
    ProtocolIntegrityError,
    ProtocolSecurityError,
)


class SecurityLevel(Enum):
    """Уровни безопасности."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Permission(Enum):
    """Разрешения."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"


class AuditEvent(Enum):
    """События аудита."""

    LOGIN = "login"
    LOGOUT = "logout"
    ACCESS_DENIED = "access_denied"
    DATA_ACCESS = "data_access"
    DATA_MODIFY = "data_modify"
    CONFIG_CHANGE = "config_change"
    SECURITY_VIOLATION = "security_violation"


@dataclass
class SecurityContext:
    """Контекст безопасности."""

    user_id: str
    session_id: str
    permissions: Set[Permission]
    security_level: SecurityLevel
    ip_address: str
    user_agent: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditLog:
    """Запись аудита."""

    id: UUID
    event: AuditEvent
    user_id: str
    session_id: str
    ip_address: str
    user_agent: str
    resource: str
    action: str
    result: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityViolation:
    """Нарушение безопасности."""

    id: UUID
    violation_type: str
    severity: SecurityLevel
    description: str
    user_id: Optional[str]
    ip_address: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


class CryptoManager:
    """Менеджер криптографии."""

    def __init__(self, master_key: Optional[bytes] = None):
        if not CRYPTOGRAPHY_AVAILABLE:
            # Fallback без cryptography
            self.master_key = master_key or os.urandom(32)
            self.fernet = None
        else:
            self.master_key = master_key or Fernet.generate_key()
            self.fernet = Fernet(self.master_key)
        self._key_cache: Dict[str, bytes] = {}

    def generate_key(self) -> bytes:
        """Сгенерировать новый ключ."""
        if not CRYPTOGRAPHY_AVAILABLE:
            return os.urandom(32)
        return Fernet.generate_key()

    def derive_key(self, password: str, salt: bytes) -> bytes:
        """Вывести ключ из пароля."""
        if not CRYPTOGRAPHY_AVAILABLE:
            # Fallback на hashlib
            return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def encrypt_data(self, data: bytes, key: Optional[bytes] = None) -> bytes:
        """Зашифровать данные."""
        if not CRYPTOGRAPHY_AVAILABLE:
            # Простое XOR шифрование как fallback
            key = key or self.master_key
            return bytes(a ^ b for a, b in zip(data, key * (len(data) // len(key) + 1)))
        if key is None:
            key = self.master_key
        fernet = Fernet(key)
        return fernet.encrypt(data)

    def decrypt_data(self, encrypted_data: bytes, key: Optional[bytes] = None) -> bytes:
        """Расшифровать данные."""
        if not CRYPTOGRAPHY_AVAILABLE:
            # Простое XOR дешифрование как fallback
            key = key or self.master_key
            return bytes(
                a ^ b
                for a, b in zip(
                    encrypted_data, key * (len(encrypted_data) // len(key) + 1)
                )
            )
        if key is None:
            key = self.master_key
        fernet = Fernet(key)
        return fernet.decrypt(encrypted_data)

    def encrypt_string(self, text: str, key: Optional[bytes] = None) -> str:
        """Зашифровать строку."""
        encrypted = self.encrypt_data(text.encode(), key)
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt_string(self, encrypted_text: str, key: Optional[bytes] = None) -> str:
        """Расшифровать строку."""
        encrypted_data = base64.urlsafe_b64decode(encrypted_text.encode())
        decrypted = self.decrypt_data(encrypted_data, key)
        return decrypted.decode()

    def hash_password(self, password: str) -> str:
        """Хешировать пароль."""
        if not BCRYPT_AVAILABLE:
            # Fallback на встроенный hashlib
            salt = secrets.token_hex(16)
            hashed = hashlib.pbkdf2_hmac(
                "sha256", password.encode(), salt.encode(), 100000
            )
            return f"{salt}${hashed.hex()}"
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode(), salt)
        return str(hashed.decode())

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Проверить пароль."""
        if not BCRYPT_AVAILABLE:
            # Fallback для hashlib
            try:
                salt, hash_hex = hashed_password.split("$")
                hashed = hashlib.pbkdf2_hmac(
                    "sha256", password.encode(), salt.encode(), 100000
                )
                return hmac.compare_digest(hashed.hex(), hash_hex)
            except (ValueError, AttributeError):
                return False
        return bool(bcrypt.checkpw(password.encode(), hashed_password.encode()))

    def generate_hmac(self, data: bytes, key: bytes) -> str:
        """Сгенерировать HMAC."""
        h = hmac.new(key, data, hashlib.sha256)
        return base64.b64encode(h.digest()).decode()

    def verify_hmac(self, data: bytes, key: bytes, signature: str) -> bool:
        """Проверить HMAC."""
        expected_signature = self.generate_hmac(data, key)
        return hmac.compare_digest(signature, expected_signature)


class AuthenticationManager:
    """Менеджер аутентификации."""

    def __init__(self) -> None:
        self._users: Dict[str, Dict[str, Any]] = {}
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._failed_attempts: Dict[str, List[datetime]] = {}
        self._lock = asyncio.Lock()
        # Настройки безопасности
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
        self.session_timeout = timedelta(hours=24)

    async def register_user(
        self, username: str, password: str, email: str, permissions: Set[Permission]
    ) -> bool:
        """Зарегистрировать пользователя."""
        async with self._lock:
            if username in self._users:
                return False
            crypto = CryptoManager()
            hashed_password = crypto.hash_password(password)
            self._users[username] = {
                "username": username,
                "password_hash": hashed_password,
                "email": email,
                "permissions": permissions,
                "created_at": datetime.now(),
                "last_login": None,
                "enabled": True,
            }
            return True

    async def authenticate_user(
        self, username: str, password: str, ip_address: str, user_agent: str
    ) -> Optional[SecurityContext]:
        """Аутентифицировать пользователя."""
        async with self._lock:
            # Проверяем блокировку
            if self._is_account_locked(username):
                raise ProtocolAuthenticationError(
                    "Account is locked due to too many failed attempts", auth_issue="Account is locked")
            if username not in self._users:
                await self._record_failed_attempt(username, ip_address)
                return None
            user = self._users[username]
            if not user["enabled"]:
                raise ProtocolAuthenticationError("Account is disabled", auth_issue="Account is disabled")
            crypto = CryptoManager()
            if not crypto.verify_password(password, user["password_hash"]):
                await self._record_failed_attempt(username, ip_address)
                return None
            # Создаем сессию
            session_id = str(uuid4())
            self._sessions[session_id] = {
                "username": username,
                "created_at": datetime.now(),
                "last_activity": datetime.now(),
                "ip_address": ip_address,
                "user_agent": user_agent,
            }
            # Обновляем время последнего входа
            user["last_login"] = datetime.now()
            return SecurityContext(
                user_id=username,
                session_id=session_id,
                permissions=user["permissions"],
                security_level=SecurityLevel.MEDIUM,
                ip_address=ip_address,
                user_agent=user_agent,
                timestamp=datetime.now(),
            )

    async def validate_session(
        self, session_id: str, ip_address: str
    ) -> Optional[SecurityContext]:
        """Проверить сессию."""
        async with self._lock:
            if session_id not in self._sessions:
                return None
            session = self._sessions[session_id]
            # Проверяем таймаут сессии
            if datetime.now() - session["last_activity"] > self.session_timeout:
                del self._sessions[session_id]
                return None
            # Проверяем IP адрес
            if session["ip_address"] != ip_address:
                del self._sessions[session_id]
                return None
            # Обновляем активность
            session["last_activity"] = datetime.now()
            username = session["username"]
            user = self._users[username]
            return SecurityContext(
                user_id=username,
                session_id=session_id,
                permissions=user["permissions"],
                security_level=SecurityLevel.MEDIUM,
                ip_address=ip_address,
                user_agent=session["user_agent"],
                timestamp=datetime.now(),
            )

    async def logout(self, session_id: str) -> bool:
        """Выйти из системы."""
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def _is_account_locked(self, username: str) -> bool:
        """Проверить, заблокирован ли аккаунт."""
        if username not in self._failed_attempts:
            return False
        attempts = self._failed_attempts[username]
        recent_attempts = [
            attempt
            for attempt in attempts
            if datetime.now() - attempt < self.lockout_duration
        ]
        return len(recent_attempts) >= self.max_failed_attempts

    async def _record_failed_attempt(self, username: str, ip_address: str) -> None:
        """Записать неудачную попытку входа."""
        if username not in self._failed_attempts:
            self._failed_attempts[username] = []
        self._failed_attempts[username].append(datetime.now())
        # Ограничиваем количество попыток
        if len(self._failed_attempts[username]) > 10:
            self._failed_attempts[username] = self._failed_attempts[username][-5:]


class AuthorizationManager:
    """Менеджер авторизации."""

    def __init__(self) -> None:
        self._resource_permissions: Dict[str, Set[Permission]] = {}
        self._role_permissions: Dict[str, Set[Permission]] = {}
        self._user_roles: Dict[str, Set[str]] = {}

    async def check_permission(
        self, context: SecurityContext, resource: str, action: Permission
    ) -> bool:
        """Проверить разрешение."""
        # Проверяем прямые разрешения пользователя
        if action in context.permissions:
            return True
        # Проверяем разрешения ролей
        user_roles = self._user_roles.get(context.user_id, set())
        for role in user_roles:
            role_permissions = self._role_permissions.get(role, set())
            if action in role_permissions:
                return True
        # Проверяем разрешения ресурса
        resource_permissions = self._resource_permissions.get(resource, set())
        if action in resource_permissions:
            return True
        return False

    async def require_permission(
        self, context: SecurityContext, resource: str, action: Permission
    ) -> None:
        """Требовать разрешение."""
        if not await self.check_permission(context, resource, action):
            raise ProtocolAuthorizationError(
                f"Permission denied: {action.value} on {resource}", auth_issue="Permission denied")

    async def add_resource_permission(
        self, resource: str, permission: Permission
    ) -> None:
        """Добавить разрешение для ресурса."""
        if resource not in self._resource_permissions:
            self._resource_permissions[resource] = set()
        self._resource_permissions[resource].add(permission)

    async def add_role_permission(self, role: str, permission: Permission) -> None:
        """Добавить разрешение для роли."""
        if role not in self._role_permissions:
            self._role_permissions[role] = set()
        self._role_permissions[role].add(permission)

    async def assign_user_role(self, user_id: str, role: str) -> None:
        """Назначить роль пользователю."""
        if user_id not in self._user_roles:
            self._user_roles[user_id] = set()
        self._user_roles[user_id].add(role)


class AuditManager:
    """Менеджер аудита."""

    def __init__(self) -> None:
        self._audit_logs: List[AuditLog] = []
        self._security_violations: List[SecurityViolation] = []
        self._lock = asyncio.Lock()

    async def log_event(
        self,
        event: AuditEvent,
        context: SecurityContext,
        resource: str,
        action: str,
        result: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Записать событие аудита."""
        audit_log = AuditLog(
            id=uuid4(),
            event=event,
            user_id=context.user_id,
            session_id=context.session_id,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            resource=resource,
            action=action,
            result=result,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )
        async with self._lock:
            self._audit_logs.append(audit_log)
            # Ограничиваем размер логов
            if len(self._audit_logs) > 10000:
                self._audit_logs = self._audit_logs[-5000:]

    async def log_security_violation(
        self,
        violation_type: str,
        severity: SecurityLevel,
        description: str,
        context: Optional[SecurityContext] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Записать нарушение безопасности."""
        violation = SecurityViolation(
            id=uuid4(),
            violation_type=violation_type,
            severity=severity,
            description=description,
            user_id=context.user_id if context else None,
            ip_address=context.ip_address if context else "unknown",
            timestamp=datetime.now(),
            context=metadata or {},
        )
        async with self._lock:
            self._security_violations.append(violation)
            # Ограничиваем размер нарушений
            if len(self._security_violations) > 1000:
                self._security_violations = self._security_violations[-500:]

    async def get_audit_logs(
        self,
        user_id: Optional[str] = None,
        event: Optional[AuditEvent] = None,
        hours: int = 24,
    ) -> List[AuditLog]:
        """Получить записи аудита."""
        async with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            logs = [log for log in self._audit_logs if log.timestamp >= cutoff_time]
            if user_id:
                logs = [log for log in logs if log.user_id == user_id]
            if event:
                logs = [log for log in logs if log.event == event]
            return logs

    async def get_security_violations(
        self, severity: Optional[SecurityLevel] = None, hours: int = 24
    ) -> List[SecurityViolation]:
        """Получить нарушения безопасности."""
        async with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            violations = [
                violation
                for violation in self._security_violations
                if violation.timestamp >= cutoff_time
            ]
            if severity:
                violations = [v for v in violations if v.severity == severity]
            return violations


class SecurityManager:
    """Главный менеджер безопасности."""

    def __init__(self) -> None:
        self.crypto_manager = CryptoManager()
        self.auth_manager = AuthenticationManager()
        self.authz_manager = AuthorizationManager()
        self.audit_manager = AuditManager()
        self._rate_limiters: Dict[str, Dict[str, Any]] = {}

    async def authenticate_and_authorize(
        self,
        username: str,
        password: str,
        ip_address: str,
        user_agent: str,
        resource: str,
        action: Permission,
    ) -> SecurityContext:
        """Аутентифицировать и авторизовать пользователя."""
        # Аутентификация
        context = await self.auth_manager.authenticate_user(
            username, password, ip_address, user_agent
        )
        if not context:
            await self.audit_manager.log_security_violation(
                "authentication_failed",
                SecurityLevel.MEDIUM,
                f"Failed login attempt for user {username}",
                metadata={"ip_address": ip_address, "user_agent": user_agent},
            )
            raise ProtocolAuthenticationError("Invalid credentials", auth_issue="Invalid credentials")
        # Авторизация
        await self.authz_manager.require_permission(context, resource, action)
        # Логируем успешный доступ
        await self.audit_manager.log_event(
            AuditEvent.LOGIN, context, resource, action.value, "success"
        )
        return context

    async def validate_session_and_authorize(
        self, session_id: str, ip_address: str, resource: str, action: Permission
    ) -> SecurityContext:
        """Проверить сессию и авторизовать."""
        context = await self.auth_manager.validate_session(session_id, ip_address)
        if not context:
            await self.audit_manager.log_security_violation(
                "invalid_session",
                SecurityLevel.MEDIUM,
                f"Invalid session attempt: {session_id}",
                metadata={"ip_address": ip_address},
            )
            raise ProtocolAuthenticationError("Invalid session", auth_issue="Invalid session")
        await self.authz_manager.require_permission(context, resource, action)
        return context

    async def check_rate_limit(self, key: str, limit: int, window: int) -> bool:
        """Проверить лимит запросов."""
        now = time.time()
        if key not in self._rate_limiters:
            self._rate_limiters[key] = {
                "requests": [],
                "limit": limit,
                "window": window,
            }
        rate_limiter = self._rate_limiters[key]
        # Удаляем старые запросы
        rate_limiter["requests"] = [
            req_time for req_time in rate_limiter["requests"] if now - req_time < window
        ]
        # Проверяем лимит
        if len(rate_limiter["requests"]) >= limit:
            return False
        # Добавляем текущий запрос
        rate_limiter["requests"].append(now)
        return True

    async def encrypt_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Зашифровать чувствительные данные."""
        encrypted_data = {}
        for key, value in data.items():
            if isinstance(value, str) and self._is_sensitive_field(key):
                encrypted_data[key] = self.crypto_manager.encrypt_string(value)
            else:
                encrypted_data[key] = value
        return encrypted_data

    async def decrypt_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Расшифровать чувствительные данные."""
        decrypted_data = {}
        for key, value in data.items():
            if isinstance(value, str) and self._is_sensitive_field(key):
                try:
                    decrypted_data[key] = self.crypto_manager.decrypt_string(value)
                except Exception:
                    decrypted_data[key] = (
                        value  # Возвращаем как есть если не удалось расшифровать
                    )
            else:
                decrypted_data[key] = value
        return decrypted_data

    def _is_sensitive_field(self, field_name: str) -> bool:
        """Проверить, является ли поле чувствительным."""
        sensitive_fields = {
            "password",
            "api_key",
            "api_secret",
            "passphrase",
            "token",
            "secret",
            "key",
            "credential",
        }
        field_lower = field_name.lower()
        return any(sensitive in field_lower for sensitive in sensitive_fields)

    async def get_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Получить отчет по безопасности."""
        audit_logs = await self.audit_manager.get_audit_logs(hours=hours)
        violations = await self.audit_manager.get_security_violations(hours=hours)
        return {
            "total_events": len(audit_logs),
            "security_violations": len(violations),
            "critical_violations": len(
                [v for v in violations if v.severity == SecurityLevel.CRITICAL]
            ),
            "failed_logins": len(
                [log for log in audit_logs if log.event == AuditEvent.ACCESS_DENIED]
            ),
            "active_sessions": len(self.auth_manager._sessions),
            "locked_accounts": len(
                [
                    user
                    for user in self.auth_manager._failed_attempts
                    if self.auth_manager._is_account_locked(user)
                ]
            ),
            "timestamp": datetime.now().isoformat(),
        }


# Глобальный экземпляр менеджера безопасности
security_manager = SecurityManager()


# Декораторы для безопасности
def require_authentication(resource: str, action: Permission) -> Callable:
    """Декоратор для требования аутентификации."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Извлекаем контекст безопасности из аргументов
            context = kwargs.get("security_context")
            if not context:
                raise ProtocolAuthenticationError("Security context required", auth_issue="Security context required")
            # Проверяем авторизацию
            await security_manager.authz_manager.require_permission(
                context, resource, action
            )
            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            context = kwargs.get("security_context")
            if not context:
                raise ProtocolAuthenticationError("Security context required", auth_issue="Security context required")
            # Для синхронных функций запускаем проверку асинхронно
            asyncio.run(
                security_manager.authz_manager.require_permission(
                    context, resource, action
                )
            )
            return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def encrypt_sensitive_fields(fields: List[str]) -> Callable:
    """Декоратор для шифрования чувствительных полей."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Шифруем входные данные
            for field in fields:
                if field in kwargs and isinstance(kwargs[field], str):
                    kwargs[field] = security_manager.crypto_manager.encrypt_string(
                        kwargs[field]
                    )
            result = await func(*args, **kwargs)
            # Расшифровываем выходные данные
            if isinstance(result, dict):
                for field in fields:
                    if field in result and isinstance(result[field], str):
                        try:
                            result[field] = (
                                security_manager.crypto_manager.decrypt_string(
                                    result[field]
                                )
                            )
                        except Exception:
                            pass  # Оставляем как есть если не удалось расшифровать
            return result

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Шифруем входные данные
            for field in fields:
                if field in kwargs and isinstance(kwargs[field], str):
                    kwargs[field] = security_manager.crypto_manager.encrypt_string(
                        kwargs[field]
                    )
            result = func(*args, **kwargs)
            # Расшифровываем выходные данные
            if isinstance(result, dict):
                for field in fields:
                    if field in result and isinstance(result[field], str):
                        try:
                            result[field] = (
                                security_manager.crypto_manager.decrypt_string(
                                    result[field]
                                )
                            )
                        except Exception:
                            pass
            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def audit_security_events(event: AuditEvent, resource: str, action: str) -> Callable:
    """Декоратор для аудита событий безопасности."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            context = kwargs.get("security_context")
            try:
                result = await func(*args, **kwargs)
                if context:
                    await security_manager.audit_manager.log_event(
                        event, context, resource, action, "success"
                    )
                return result
            except Exception as e:
                if context:
                    await security_manager.audit_manager.log_event(
                        event, context, resource, action, "failure", {"error": str(e)}
                    )
                raise

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            context = kwargs.get("security_context")
            try:
                result = func(*args, **kwargs)
                if context:
                    asyncio.run(
                        security_manager.audit_manager.log_event(
                            event, context, resource, action, "success"
                        )
                    )
                return result
            except Exception as e:
                if context:
                    asyncio.run(
                        security_manager.audit_manager.log_event(
                            event,
                            context,
                            resource,
                            action,
                            "failure",
                            {"error": str(e)},
                        )
                    )
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator
