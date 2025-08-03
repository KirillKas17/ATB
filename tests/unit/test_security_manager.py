"""
Unit тесты для SecurityManager.
Тестирует управление безопасностью, включая аутентификацию,
авторизацию, шифрование и мониторинг безопасности.
"""
import pytest
from datetime import datetime, timedelta
from infrastructure.core.security_manager import SecurityManager
from typing import List

class TestSecurityManager:
    """Тесты для SecurityManager."""
    @pytest.fixture
    def security_manager(self) -> SecurityManager:
        """Фикстура для SecurityManager."""
        return SecurityManager()
    @pytest.fixture
    def sample_user(self) -> dict:
        """Фикстура с тестовым пользователем."""
        return {
            "id": "user_001",
            "username": "test_user",
            "email": "test@example.com",
            "role": "trader",
            "permissions": ["read", "write", "trade"],
            "created_at": datetime.now(),
            "last_login": datetime.now() - timedelta(hours=1)
        }
    @pytest.fixture
    def sample_api_key(self) -> dict:
        """Фикстура с тестовым API ключом."""
        return {
            "id": "key_001",
            "user_id": "user_001",
            "api_key": "test_api_key_12345",
            "api_secret": "test_api_secret_67890",
            "permissions": ["read", "trade"],
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(days=30),
            "is_active": True
        }
    def test_initialization(self, security_manager: SecurityManager) -> None:
        """Тест инициализации менеджера безопасности."""
        assert security_manager is not None
        assert hasattr(security_manager, 'encryption_keys')
        assert hasattr(security_manager, 'access_controls')
        assert hasattr(security_manager, 'security_monitors')
    def test_authenticate_user(self, security_manager: SecurityManager, sample_user: dict) -> None:
        """Тест аутентификации пользователя."""
        # Мок пароля
        password = "secure_password_123"
        # Аутентификация пользователя
        auth_result = security_manager.authenticate_user(
            sample_user["username"], password
        )
        # Проверки
        assert auth_result is not None
        assert "success" in auth_result
        assert "user_id" in auth_result
        assert "session_token" in auth_result
        assert "auth_time" in auth_result
        # Проверка типов данных
        assert isinstance(auth_result["success"], bool)
        assert isinstance(auth_result["user_id"], str)
        assert isinstance(auth_result["session_token"], str)
        assert isinstance(auth_result["auth_time"], datetime)
    def test_authorize_user(self, security_manager: SecurityManager, sample_user: dict) -> None:
        """Тест авторизации пользователя."""
        # Авторизация пользователя
        auth_result = security_manager.authorize_user(
            sample_user["id"], "trade"
        )
        # Проверки
        assert auth_result is not None
        assert "authorized" in auth_result
        assert "permissions" in auth_result
        assert "authorization_time" in auth_result
        # Проверка типов данных
        assert isinstance(auth_result["authorized"], bool)
        assert isinstance(auth_result["permissions"], list)
        assert isinstance(auth_result["authorization_time"], datetime)
    def test_validate_api_key(self, security_manager: SecurityManager, sample_api_key: dict) -> None:
        """Тест валидации API ключа."""
        # Валидация API ключа
        validation_result = security_manager.validate_api_key(
            sample_api_key["api_key"], sample_api_key["api_secret"]
        )
        # Проверки
        assert validation_result is not None
        assert "is_valid" in validation_result
        assert "user_id" in validation_result
        assert "permissions" in validation_result
        assert "validation_time" in validation_result
        # Проверка типов данных
        assert isinstance(validation_result["is_valid"], bool)
        assert isinstance(validation_result["user_id"], str)
        assert isinstance(validation_result["permissions"], list)
        assert isinstance(validation_result["validation_time"], datetime)
    def test_encrypt_data(self, security_manager: SecurityManager) -> None:
        """Тест шифрования данных."""
        # Тестовые данные
        test_data = "sensitive_information_123"
        # Шифрование данных
        encrypted_result = security_manager.encrypt_data(test_data)
        # Проверки
        assert encrypted_result is not None
        assert "encrypted_data" in encrypted_result
        assert "encryption_key" in encrypted_result
        assert "encryption_time" in encrypted_result
        # Проверка типов данных
        assert isinstance(encrypted_result["encrypted_data"], str)
        assert isinstance(encrypted_result["encryption_key"], str)
        assert isinstance(encrypted_result["encryption_time"], datetime)
        # Проверка, что данные зашифрованы
        assert encrypted_result["encrypted_data"] != test_data
    def test_decrypt_data(self, security_manager: SecurityManager) -> None:
        """Тест расшифровки данных."""
        # Тестовые данные
        test_data = "sensitive_information_123"
        # Шифрование данных
        encrypted_result = security_manager.encrypt_data(test_data)
        # Расшифровка данных
        decrypted_result = security_manager.decrypt_data(
            encrypted_result["encrypted_data"],
            encrypted_result["encryption_key"]
        )
        # Проверки
        assert decrypted_result is not None
        assert "decrypted_data" in decrypted_result
        assert "decryption_time" in decrypted_result
        # Проверка типов данных
        assert isinstance(decrypted_result["decrypted_data"], str)
        assert isinstance(decrypted_result["decryption_time"], datetime)
        # Проверка, что данные расшифрованы правильно
        assert decrypted_result["decrypted_data"] == test_data
    def test_hash_password(self, security_manager: SecurityManager) -> None:
        """Тест хеширования пароля."""
        # Тестовый пароль
        password = "secure_password_123"
        # Хеширование пароля
        hash_result = security_manager.hash_password(password)
        # Проверки
        assert hash_result is not None
        assert "hashed_password" in hash_result
        assert "salt" in hash_result
        assert "hash_time" in hash_result
        # Проверка типов данных
        assert isinstance(hash_result["hashed_password"], str)
        assert isinstance(hash_result["salt"], str)
        assert isinstance(hash_result["hash_time"], datetime)
        # Проверка, что хеш отличается от исходного пароля
        assert hash_result["hashed_password"] != password
    def test_verify_password(self, security_manager: SecurityManager) -> None:
        """Тест проверки пароля."""
        # Тестовый пароль
        password = "secure_password_123"
        # Хеширование пароля
        hash_result = security_manager.hash_password(password)
        # Проверка пароля
        verify_result = security_manager.verify_password(
            password, hash_result["hashed_password"], hash_result["salt"]
        )
        # Проверки
        assert verify_result is not None
        assert "is_valid" in verify_result
        assert "verification_time" in verify_result
        # Проверка типов данных
        assert isinstance(verify_result["is_valid"], bool)
        assert isinstance(verify_result["verification_time"], datetime)
        # Проверка, что пароль верный
        assert verify_result["is_valid"] is True
    def test_generate_api_key(self, security_manager: SecurityManager, sample_user: dict) -> None:
        """Тест генерации API ключа."""
        # Генерация API ключа
        api_key_result = security_manager.generate_api_key(
            sample_user["id"], permissions=["read", "trade"]
        )
        # Проверки
        assert api_key_result is not None
        assert "api_key" in api_key_result
        assert "api_secret" in api_key_result
        assert "user_id" in api_key_result
        assert "permissions" in api_key_result
        assert "created_at" in api_key_result
        assert "expires_at" in api_key_result
        # Проверка типов данных
        assert isinstance(api_key_result["api_key"], str)
        assert isinstance(api_key_result["api_secret"], str)
        assert isinstance(api_key_result["user_id"], str)
        assert isinstance(api_key_result["permissions"], list)
        assert isinstance(api_key_result["created_at"], datetime)
        assert isinstance(api_key_result["expires_at"], datetime)
    def test_revoke_api_key(self, security_manager: SecurityManager, sample_api_key: dict) -> None:
        """Тест отзыва API ключа."""
        # Отзыв API ключа
        revoke_result = security_manager.revoke_api_key(sample_api_key["api_key"])
        # Проверки
        assert revoke_result is not None
        assert "success" in revoke_result
        assert "revocation_time" in revoke_result
        assert "revoked_key" in revoke_result
        # Проверка типов данных
        assert isinstance(revoke_result["success"], bool)
        assert isinstance(revoke_result["revocation_time"], datetime)
        assert isinstance(revoke_result["revoked_key"], str)
    def test_create_session(self, security_manager: SecurityManager, sample_user: dict) -> None:
        """Тест создания сессии."""
        # Создание сессии
        session_result = security_manager.create_session(sample_user["id"])
        # Проверки
        assert session_result is not None
        assert "session_id" in session_result
        assert "user_id" in session_result
        assert "session_token" in session_result
        assert "created_at" in session_result
        assert "expires_at" in session_result
        # Проверка типов данных
        assert isinstance(session_result["session_id"], str)
        assert isinstance(session_result["user_id"], str)
        assert isinstance(session_result["session_token"], str)
        assert isinstance(session_result["created_at"], datetime)
        assert isinstance(session_result["expires_at"], datetime)
    def test_validate_session(self, security_manager: SecurityManager, sample_user: dict) -> None:
        """Тест валидации сессии."""
        # Создание сессии
        session_result = security_manager.create_session(sample_user["id"])
        # Валидация сессии
        validation_result = security_manager.validate_session(session_result["session_token"])
        # Проверки
        assert validation_result is not None
        assert "is_valid" in validation_result
        assert "user_id" in validation_result
        assert "session_data" in validation_result
        assert "validation_time" in validation_result
        # Проверка типов данных
        assert isinstance(validation_result["is_valid"], bool)
        assert isinstance(validation_result["user_id"], str)
        assert isinstance(validation_result["session_data"], dict)
        assert isinstance(validation_result["validation_time"], datetime)
    def test_terminate_session(self, security_manager: SecurityManager, sample_user: dict) -> None:
        """Тест завершения сессии."""
        # Создание сессии
        session_result = security_manager.create_session(sample_user["id"])
        # Завершение сессии
        terminate_result = security_manager.terminate_session(session_result["session_token"])
        # Проверки
        assert terminate_result is not None
        assert "success" in terminate_result
        assert "termination_time" in terminate_result
        assert "terminated_session" in terminate_result
        # Проверка типов данных
        assert isinstance(terminate_result["success"], bool)
        assert isinstance(terminate_result["termination_time"], datetime)
        assert isinstance(terminate_result["terminated_session"], str)
    def test_monitor_security_events(self, security_manager: SecurityManager) -> None:
        """Тест мониторинга событий безопасности."""
        # Мониторинг событий безопасности
        monitoring_result = security_manager.monitor_security_events()
        # Проверки
        assert monitoring_result is not None
        assert "security_events" in monitoring_result
        assert "threat_level" in monitoring_result
        assert "alerts" in monitoring_result
        assert "monitoring_time" in monitoring_result
        # Проверка типов данных
        assert isinstance(monitoring_result["security_events"], list)
        assert monitoring_result["threat_level"] in ["low", "medium", "high", "critical"]
        assert isinstance(monitoring_result["alerts"], list)
        assert isinstance(monitoring_result["monitoring_time"], datetime)
    def test_detect_security_threats(self, security_manager: SecurityManager) -> None:
        """Тест обнаружения угроз безопасности."""
        # Обнаружение угроз
        threat_detection = security_manager.detect_security_threats()
        # Проверки
        assert threat_detection is not None
        assert "threats_detected" in threat_detection
        assert "threat_level" in threat_detection
        assert "threat_analysis" in threat_detection
        assert "recommendations" in threat_detection
        # Проверка типов данных
        assert isinstance(threat_detection["threats_detected"], list)
        assert threat_detection["threat_level"] in ["low", "medium", "high", "critical"]
        assert isinstance(threat_detection["threat_analysis"], dict)
        assert isinstance(threat_detection["recommendations"], list)
    def test_audit_security_logs(self, security_manager: SecurityManager) -> None:
        """Тест аудита логов безопасности."""
        # Аудит логов
        audit_result = security_manager.audit_security_logs(
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now()
        )
        # Проверки
        assert audit_result is not None
        assert "audit_logs" in audit_result
        assert "security_violations" in audit_result
        assert "compliance_status" in audit_result
        assert "audit_summary" in audit_result
        # Проверка типов данных
        assert isinstance(audit_result["audit_logs"], list)
        assert isinstance(audit_result["security_violations"], list)
        assert isinstance(audit_result["compliance_status"], dict)
        assert isinstance(audit_result["audit_summary"], dict)
    def test_validate_permissions(self, security_manager: SecurityManager, sample_user: dict) -> None:
        """Тест валидации разрешений."""
        # Валидация разрешений
        permission_result = security_manager.validate_permissions(
            sample_user["id"], ["read", "trade"]
        )
        # Проверки
        assert permission_result is not None
        assert "has_permissions" in permission_result
        assert "granted_permissions" in permission_result
        assert "denied_permissions" in permission_result
        assert "validation_time" in permission_result
        # Проверка типов данных
        assert isinstance(permission_result["has_permissions"], bool)
        assert isinstance(permission_result["granted_permissions"], list)
        assert isinstance(permission_result["denied_permissions"], list)
        assert isinstance(permission_result["validation_time"], datetime)
    def test_error_handling(self, security_manager: SecurityManager) -> None:
        """Тест обработки ошибок."""
        # Тест с некорректными данными
        with pytest.raises(ValueError):
            security_manager.authenticate_user(None, None)
        with pytest.raises(ValueError):
            security_manager.encrypt_data(None)
    def test_edge_cases(self, security_manager: SecurityManager) -> None:
        """Тест граничных случаев."""
        # Тест с очень длинным паролем
        long_password = "a" * 1000
        hash_result = security_manager.hash_password(long_password)
        assert hash_result is not None
        # Тест с пустыми разрешениями
        empty_permissions: List[str] = []
        permission_result = security_manager.validate_permissions("user_001", empty_permissions)
        assert permission_result["has_permissions"] is True
    def test_cleanup(self, security_manager: SecurityManager) -> None:
        """Тест очистки ресурсов."""
        # Проверка корректной очистки
        security_manager.cleanup()
        # Проверка, что ресурсы освобождены
        assert security_manager.encryption_keys == {}
        assert security_manager.access_controls == {}
        assert security_manager.security_monitors == {} 
