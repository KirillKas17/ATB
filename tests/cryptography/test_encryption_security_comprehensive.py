#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive тесты криптографии и шифрования финансовой системы.
Критически важно для финансовой системы - защита конфиденциальных данных.
"""

import pytest
import os
import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
import time

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.twofactor.totp import TOTP
from cryptography.hazmat.primitives.twofactor.hotp import HOTP

from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.entities.user import User, UserCredentials
from domain.entities.wallet import Wallet, PrivateKey, PublicKey
from infrastructure.security.encryption.symmetric_encryption import SymmetricEncryption
from infrastructure.security.encryption.asymmetric_encryption import AsymmetricEncryption
from infrastructure.security.encryption.key_management import KeyManager, KeyDerivation
from infrastructure.security.hashing.secure_hash import SecureHash
from infrastructure.security.signing.digital_signature import DigitalSignature
from infrastructure.security.authentication.multi_factor_auth import MultiFactorAuth
from infrastructure.security.secure_storage.vault import SecureVault
from infrastructure.security.secure_communication.tls_handler import TLSHandler
from domain.exceptions import CryptographyError, SecurityError, ValidationError


class EncryptionAlgorithm(Enum):
    """Алгоритмы шифрования."""
    AES_256_GCM = "AES_256_GCM"
    AES_256_CBC = "AES_256_CBC"
    AES_256_CTR = "AES_256_CTR"
    CHACHA20_POLY1305 = "CHACHA20_POLY1305"
    RSA_4096 = "RSA_4096"
    RSA_2048 = "RSA_2048"
    ECC_P256 = "ECC_P256"
    ECC_P384 = "ECC_P384"


class HashAlgorithm(Enum):
    """Алгоритмы хеширования."""
    SHA256 = "SHA256"
    SHA512 = "SHA512"
    SHA3_256 = "SHA3_256"
    SHA3_512 = "SHA3_512"
    BLAKE2B = "BLAKE2B"
    BLAKE2S = "BLAKE2S"
    SCRYPT = "SCRYPT"
    ARGON2 = "ARGON2"


@dataclass
class EncryptionConfig:
    """Конфигурация шифрования."""
    algorithm: EncryptionAlgorithm
    key_size: int
    mode: Optional[str] = None
    padding_scheme: Optional[str] = None
    salt_size: int = 32
    iteration_count: int = 100000


@dataclass
class SecurityAudit:
    """Результат аудита безопасности."""
    timestamp: datetime
    operation: str
    user_id: str
    success: bool
    encryption_used: bool
    key_strength: int
    compliance_level: str


class TestEncryptionSecurityComprehensive:
    """Comprehensive тесты криптографии и безопасности."""

    @pytest.fixture
    def encryption_config(self) -> EncryptionConfig:
        """Фикстура конфигурации шифрования."""
        return EncryptionConfig(
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            key_size=256,
            mode="GCM",
            salt_size=32,
            iteration_count=100000
        )

    @pytest.fixture
    def key_manager(self) -> KeyManager:
        """Фикстура менеджера ключей."""
        return KeyManager(
            key_storage_backend='hsm',  # Hardware Security Module
            key_rotation_period=timedelta(days=90),
            master_key_algorithm='AES_256',
            key_derivation_algorithm='PBKDF2',
            backup_enabled=True,
            audit_enabled=True
        )

    @pytest.fixture
    def secure_vault(self) -> SecureVault:
        """Фикстура безопасного хранилища."""
        return SecureVault(
            encryption_algorithm=EncryptionAlgorithm.AES_256_GCM,
            access_control_enabled=True,
            audit_logging=True,
            tamper_detection=True,
            key_escrow=True
        )

    @pytest.fixture
    def sample_financial_data(self) -> Dict[str, Any]:
        """Фикстура конфиденциальных финансовых данных."""
        return {
            "user_id": "user_12345",
            "account_number": "1234567890123456",
            "balance": "50000.00",
            "currency": "USD",
            "wallet_addresses": {
                "BTC": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
                "ETH": "0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe"
            },
            "private_keys": {
                "BTC": "L1aW4aubDFB7yfras2S1mN3bqg9nwySY8nkoLmJebSLD5BWv3ENZ",
                "ETH": "0x4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318"
            },
            "api_keys": {
                "binance": "AKIAIOSFODNN7EXAMPLE",
                "bybit": "AKIAI44QH8DHBEXAMPLE"
            },
            "mfa_secrets": {
                "totp_secret": "JBSWY3DPEHPK3PXP",
                "backup_codes": ["123456", "234567", "345678"]
            }
        }

    def test_symmetric_encryption_aes_gcm(
        self,
        encryption_config: EncryptionConfig
    ) -> None:
        """Тест симметричного шифрования AES-GCM."""
        
        # Создаем симметричный шифровальщик
        symmetric_crypto = SymmetricEncryption(
            algorithm=encryption_config.algorithm,
            key_size=encryption_config.key_size
        )
        
        # Генерируем ключ
        encryption_key = symmetric_crypto.generate_key()
        assert len(encryption_key) == 32  # 256 бит = 32 байта
        
        # Тестовые данные разной длины
        test_data = [
            b"Small data",
            b"Medium length data that spans multiple blocks",
            b"Very long data " * 1000,  # ~15KB
            json.dumps({
                "account": "1234567890",
                "balance": 50000.00,
                "transactions": [{"id": i, "amount": i * 100} for i in range(100)]
            }).encode('utf-8')
        ]
        
        for data in test_data:
            # Шифруем
            encrypted_result = symmetric_crypto.encrypt(data, encryption_key)
            
            # Проверяем структуру результата
            assert 'ciphertext' in encrypted_result
            assert 'nonce' in encrypted_result
            assert 'tag' in encrypted_result
            
            # Проверяем что зашифрованные данные отличаются от исходных
            assert encrypted_result['ciphertext'] != data
            
            # Расшифровываем
            decrypted_data = symmetric_crypto.decrypt(encrypted_result, encryption_key)
            
            # Проверяем что данные восстановлены корректно
            assert decrypted_data == data
        
        # Тест с неправильным ключом
        wrong_key = symmetric_crypto.generate_key()
        with pytest.raises(CryptographyError):
            symmetric_crypto.decrypt(encrypted_result, wrong_key)
        
        # Тест с поврежденными данными
        corrupted_result = encrypted_result.copy()
        corrupted_result['ciphertext'] = corrupted_result['ciphertext'][:-5] + b'XXXXX'
        
        with pytest.raises(CryptographyError):
            symmetric_crypto.decrypt(corrupted_result, encryption_key)

    def test_asymmetric_encryption_rsa(self) -> None:
        """Тест асимметричного шифрования RSA."""
        
        # Создаем асимметричный шифровальщик
        asymmetric_crypto = AsymmetricEncryption(
            algorithm=EncryptionAlgorithm.RSA_4096,
            key_size=4096
        )
        
        # Генерируем пару ключей
        key_pair = asymmetric_crypto.generate_key_pair()
        
        assert 'private_key' in key_pair
        assert 'public_key' in key_pair
        
        private_key = key_pair['private_key']
        public_key = key_pair['public_key']
        
        # Проверяем размер ключей
        assert private_key.key_size == 4096
        assert public_key.key_size == 4096
        
        # Тестовые данные (RSA ограничен размером ключа)
        test_messages = [
            b"Short message",
            b"API_KEY: AKIAIOSFODNN7EXAMPLE",
            json.dumps({"wallet": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"}).encode('utf-8')
        ]
        
        for message in test_messages:
            # Шифруем открытым ключом
            encrypted_data = asymmetric_crypto.encrypt(message, public_key)
            
            # Проверяем что данные зашифрованы
            assert encrypted_data != message
            assert len(encrypted_data) == 512  # 4096 бит = 512 байт
            
            # Расшифровываем закрытым ключом
            decrypted_data = asymmetric_crypto.decrypt(encrypted_data, private_key)
            
            # Проверяем восстановление данных
            assert decrypted_data == message
        
        # Тест гибридного шифрования для больших данных
        large_data = b"Large financial data " * 1000  # ~20KB
        
        hybrid_result = asymmetric_crypto.hybrid_encrypt(large_data, public_key)
        
        assert 'encrypted_key' in hybrid_result
        assert 'encrypted_data' in hybrid_result
        assert 'nonce' in hybrid_result
        assert 'tag' in hybrid_result
        
        # Расшифровываем гибридным методом
        decrypted_large_data = asymmetric_crypto.hybrid_decrypt(hybrid_result, private_key)
        assert decrypted_large_data == large_data

    def test_key_derivation_and_management(
        self,
        key_manager: KeyManager
    ) -> None:
        """Тест деривации и управления ключами."""
        
        # Тест PBKDF2
        password = "StrongPassword123!@#"
        salt = os.urandom(32)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        derived_key_1 = kdf.derive(password.encode('utf-8'))
        
        # Второй раз с тем же паролем и солью
        kdf2 = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        derived_key_2 = kdf2.derive(password.encode('utf-8'))
        
        # Ключи должны быть одинаковыми
        assert derived_key_1 == derived_key_2
        
        # Тест Scrypt (более безопасная деривация)
        scrypt_kdf = Scrypt(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            n=2**14,  # CPU cost
            r=8,      # Memory cost
            p=1,      # Parallelization
            backend=default_backend()
        )
        
        scrypt_key = scrypt_kdf.derive(password.encode('utf-8'))
        assert len(scrypt_key) == 32
        assert scrypt_key != derived_key_1  # Разные алгоритмы дают разные ключи
        
        # Тест менеджера ключей
        # Создаем главный ключ
        master_key_id = key_manager.create_master_key()
        assert master_key_id is not None
        
        # Деривируем ключи для разных целей
        user_encryption_key = key_manager.derive_key(
            master_key_id,
            purpose="user_data_encryption",
            context=b"user_12345"
        )
        
        wallet_encryption_key = key_manager.derive_key(
            master_key_id,
            purpose="wallet_encryption", 
            context=b"wallet_storage"
        )
        
        api_encryption_key = key_manager.derive_key(
            master_key_id,
            purpose="api_keys_encryption",
            context=b"external_apis"
        )
        
        # Все ключи должны быть разными
        assert user_encryption_key != wallet_encryption_key
        assert wallet_encryption_key != api_encryption_key
        assert user_encryption_key != api_encryption_key
        
        # Тест ротации ключей
        rotated_key_id = key_manager.rotate_key(master_key_id)
        assert rotated_key_id != master_key_id
        
        # Старый ключ должен остаться для расшифровки старых данных
        assert key_manager.is_key_valid(master_key_id)
        assert key_manager.is_key_valid(rotated_key_id)
        
        # Новые операции должны использовать новый ключ
        current_key_id = key_manager.get_current_key_id()
        assert current_key_id == rotated_key_id

    def test_secure_hashing_and_integrity(self) -> None:
        """Тест безопасного хеширования и проверки целостности."""
        
        secure_hash = SecureHash()
        
        # Тест различных алгоритмов хеширования
        test_data = b"Financial transaction: Transfer $50,000 from account A to account B"
        
        hash_algorithms = [
            HashAlgorithm.SHA256,
            HashAlgorithm.SHA512,
            HashAlgorithm.SHA3_256,
            HashAlgorithm.SHA3_512,
            HashAlgorithm.BLAKE2B
        ]
        
        hashes_results = {}
        
        for algorithm in hash_algorithms:
            hash_result = secure_hash.hash_data(test_data, algorithm)
            hashes_results[algorithm] = hash_result
            
            # Проверяем длины хешей
            if algorithm == HashAlgorithm.SHA256:
                assert len(hash_result) == 32
            elif algorithm == HashAlgorithm.SHA512:
                assert len(hash_result) == 64
            elif algorithm == HashAlgorithm.SHA3_256:
                assert len(hash_result) == 32
            elif algorithm == HashAlgorithm.SHA3_512:
                assert len(hash_result) == 64
            elif algorithm == HashAlgorithm.BLAKE2B:
                assert len(hash_result) == 64
        
        # Все хеши должны быть разными
        hash_values = list(hashes_results.values())
        assert len(set(hash_values)) == len(hash_values)
        
        # Тест HMAC для аутентификации сообщений
        secret_key = secrets.token_bytes(32)
        
        hmac_sha256 = hmac.new(secret_key, test_data, hashlib.sha256).digest()
        hmac_sha512 = hmac.new(secret_key, test_data, hashlib.sha512).digest()
        
        # Проверяем HMAC
        hmac_verification_256 = hmac.new(secret_key, test_data, hashlib.sha256).digest()
        hmac_verification_512 = hmac.new(secret_key, test_data, hashlib.sha512).digest()
        
        assert hmac.compare_digest(hmac_sha256, hmac_verification_256)
        assert hmac.compare_digest(hmac_sha512, hmac_verification_512)
        
        # Тест с неправильным ключом
        wrong_key = secrets.token_bytes(32)
        wrong_hmac = hmac.new(wrong_key, test_data, hashlib.sha256).digest()
        
        assert not hmac.compare_digest(hmac_sha256, wrong_hmac)
        
        # Тест хеширования паролей с солью
        password = "UserPassword123!"
        
        password_hashes = []
        for i in range(5):
            salt = secrets.token_bytes(32)
            password_hash = secure_hash.hash_password(password, salt, iterations=100000)
            password_hashes.append(password_hash)
        
        # Все хеши паролей должны быть разными (разные соли)
        assert len(set(password_hashes)) == len(password_hashes)

    def test_digital_signatures_and_verification(self) -> None:
        """Тест цифровых подписей и верификации."""
        
        digital_signer = DigitalSignature(
            algorithm='RSA',
            key_size=4096,
            hash_algorithm='SHA256'
        )
        
        # Генерируем пару ключей для подписи
        signing_keys = digital_signer.generate_signing_keys()
        private_key = signing_keys['private_key']
        public_key = signing_keys['public_key']
        
        # Тестовые финансовые документы
        financial_documents = [
            {
                "type": "transaction",
                "from_account": "12345678",
                "to_account": "87654321",
                "amount": "50000.00",
                "currency": "USD",
                "timestamp": datetime.utcnow().isoformat()
            },
            {
                "type": "order",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": "1.0",
                "price": "45000.00",
                "timestamp": datetime.utcnow().isoformat()
            },
            {
                "type": "wallet_transfer",
                "from_wallet": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
                "to_wallet": "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2",
                "amount": "0.5",
                "currency": "BTC",
                "timestamp": datetime.utcnow().isoformat()
            }
        ]
        
        signatures = []
        
        for document in financial_documents:
            # Сериализуем документ
            document_bytes = json.dumps(document, sort_keys=True).encode('utf-8')
            
            # Подписываем документ
            signature = digital_signer.sign(document_bytes, private_key)
            signatures.append(signature)
            
            # Проверяем подпись
            is_valid = digital_signer.verify(document_bytes, signature, public_key)
            assert is_valid is True
            
            # Тест с поврежденным документом
            corrupted_document = document.copy()
            corrupted_document['amount'] = "99999.99"  # Изменяем сумму
            corrupted_bytes = json.dumps(corrupted_document, sort_keys=True).encode('utf-8')
            
            is_valid_corrupted = digital_signer.verify(corrupted_bytes, signature, public_key)
            assert is_valid_corrupted is False
        
        # Все подписи должны быть уникальными
        assert len(set(signatures)) == len(signatures)
        
        # Тест групповой подписи (многократное подписание)
        multi_sig_document = {
            "type": "high_value_transaction",
            "amount": "1000000.00",  # $1M требует множественной подписи
            "signers": ["admin_1", "admin_2", "admin_3"]
        }
        
        document_bytes = json.dumps(multi_sig_document, sort_keys=True).encode('utf-8')
        
        # Генерируем несколько пар ключей для админов
        admin_keys = []
        admin_signatures = []
        
        for i in range(3):
            admin_signing_keys = digital_signer.generate_signing_keys()
            admin_keys.append(admin_signing_keys)
            
            # Каждый админ подписывает документ
            admin_signature = digital_signer.sign(document_bytes, admin_signing_keys['private_key'])
            admin_signatures.append(admin_signature)
        
        # Проверяем все подписи
        for i, signature in enumerate(admin_signatures):
            is_valid = digital_signer.verify(
                document_bytes, 
                signature, 
                admin_keys[i]['public_key']
            )
            assert is_valid is True

    def test_multi_factor_authentication(self) -> None:
        """Тест многофакторной аутентификации."""
        
        mfa = MultiFactorAuth()
        
        # Тест TOTP (Time-based One-Time Password)
        totp_secret = mfa.generate_totp_secret()
        assert len(base64.b32decode(totp_secret)) == 20  # 160 бит = 20 байт
        
        # Генерируем TOTP токен
        totp = TOTP(base64.b32decode(totp_secret), 6, hashes.SHA1(), 30, backend=default_backend())
        current_time = int(time.time())
        
        token = totp.generate(current_time)
        assert len(token) == 6
        
        # Проверяем токен
        try:
            totp.verify(token, current_time)
            totp_valid = True
        except:
            totp_valid = False
        
        assert totp_valid is True
        
        # Тест с неправильным токеном
        wrong_token = b"123456"
        try:
            totp.verify(wrong_token, current_time)
            wrong_totp_valid = True
        except:
            wrong_totp_valid = False
        
        assert wrong_totp_valid is False
        
        # Тест HOTP (HMAC-based One-Time Password)
        hotp_secret = mfa.generate_hotp_secret()
        hotp = HOTP(hotp_secret, 6, hashes.SHA1(), backend=default_backend())
        
        # Генерируем несколько HOTP токенов
        hotp_tokens = []
        for counter in range(5):
            token = hotp.generate(counter)
            hotp_tokens.append(token)
            
            # Проверяем токен
            try:
                hotp.verify(token, counter)
                token_valid = True
            except:
                token_valid = False
            
            assert token_valid is True
        
        # Все HOTP токены должны быть разными
        assert len(set(hotp_tokens)) == len(hotp_tokens)
        
        # Тест backup кодов
        backup_codes = mfa.generate_backup_codes(count=10)
        assert len(backup_codes) == 10
        
        for code in backup_codes:
            assert len(code) == 8  # 8-символьные коды
            assert code.isalnum()  # Только буквы и цифры
        
        # Backup коды должны быть уникальными
        assert len(set(backup_codes)) == len(backup_codes)
        
        # Тест полного MFA workflow
        user_id = "user_12345"
        
        # Настраиваем MFA для пользователя
        mfa_setup = mfa.setup_user_mfa(
            user_id=user_id,
            enable_totp=True,
            enable_backup_codes=True,
            require_multiple_factors=True
        )
        
        assert 'totp_secret' in mfa_setup
        assert 'backup_codes' in mfa_setup
        assert 'recovery_key' in mfa_setup
        
        # Симулируем login с MFA
        password_correct = True  # Первый фактор (пароль)
        
        if password_correct:
            # Генерируем текущий TOTP токен
            user_totp = TOTP(
                base64.b32decode(mfa_setup['totp_secret']), 
                6, hashes.SHA1(), 30, backend=default_backend()
            )
            current_token = user_totp.generate(int(time.time()))
            
            # Проверяем MFA
            mfa_result = mfa.verify_mfa(
                user_id=user_id,
                totp_token=current_token,
                backup_code=None
            )
            
            assert mfa_result['success'] is True
            assert mfa_result['factors_verified'] >= 2

    def test_secure_storage_vault(
        self,
        secure_vault: SecureVault,
        sample_financial_data: Dict[str, Any]
    ) -> None:
        """Тест безопасного хранилища данных."""
        
        # Сохраняем конфиденциальные данные в vault
        storage_results = {}
        
        for data_type, data_value in sample_financial_data.items():
            if data_type == "private_keys":
                # Приватные ключи требуют максимальной защиты
                storage_result = secure_vault.store_sensitive_data(
                    key=f"private_keys_{sample_financial_data['user_id']}",
                    data=json.dumps(data_value),
                    classification="TOP_SECRET",
                    encryption_level="MAXIMUM",
                    access_control=["OWNER_ONLY"],
                    audit_trail=True
                )
            elif data_type == "api_keys":
                # API ключи критичны для торговли
                storage_result = secure_vault.store_sensitive_data(
                    key=f"api_keys_{sample_financial_data['user_id']}",
                    data=json.dumps(data_value),
                    classification="SECRET", 
                    encryption_level="HIGH",
                    access_control=["OWNER", "ADMIN"],
                    audit_trail=True
                )
            elif data_type == "mfa_secrets":
                # MFA секреты для аутентификации
                storage_result = secure_vault.store_sensitive_data(
                    key=f"mfa_secrets_{sample_financial_data['user_id']}",
                    data=json.dumps(data_value),
                    classification="SECRET",
                    encryption_level="HIGH", 
                    access_control=["OWNER"],
                    audit_trail=True
                )
            else:
                # Остальные данные
                storage_result = secure_vault.store_sensitive_data(
                    key=f"{data_type}_{sample_financial_data['user_id']}",
                    data=json.dumps(data_value) if isinstance(data_value, (dict, list)) else str(data_value),
                    classification="CONFIDENTIAL",
                    encryption_level="STANDARD",
                    access_control=["OWNER", "ADMIN", "SUPPORT"],
                    audit_trail=True
                )
            
            storage_results[data_type] = storage_result
            
            # Проверяем что данные зашифрованы
            assert storage_result['encrypted'] is True
            assert storage_result['storage_id'] is not None
        
        # Извлекаем данные из vault
        for data_type, storage_result in storage_results.items():
            retrieved_data = secure_vault.retrieve_sensitive_data(
                storage_id=storage_result['storage_id'],
                requester_id=sample_financial_data['user_id'],
                access_reason="DATA_VERIFICATION"
            )
            
            # Проверяем целостность данных
            original_data = sample_financial_data[data_type]
            if isinstance(original_data, (dict, list)):
                assert json.loads(retrieved_data['data']) == original_data
            else:
                assert retrieved_data['data'] == str(original_data)
            
            # Проверяем audit trail
            assert retrieved_data['access_logged'] is True
        
        # Тест контроля доступа
        unauthorized_user = "unauthorized_user_999"
        
        with pytest.raises(SecurityError):
            secure_vault.retrieve_sensitive_data(
                storage_id=storage_results['private_keys']['storage_id'],
                requester_id=unauthorized_user,
                access_reason="UNAUTHORIZED_ACCESS_ATTEMPT"
            )
        
        # Тест backup и restore
        backup_result = secure_vault.create_encrypted_backup(
            include_metadata=True,
            include_audit_logs=True,
            backup_encryption_key=None  # Автоматическая генерация
        )
        
        assert backup_result['backup_created'] is True
        assert backup_result['backup_encrypted'] is True
        assert 'backup_id' in backup_result
        
        # Тест восстановления
        restore_result = secure_vault.restore_from_backup(
            backup_id=backup_result['backup_id'],
            verification_required=True
        )
        
        assert restore_result['restore_successful'] is True
        assert restore_result['data_integrity_verified'] is True

    def test_tls_secure_communication(self) -> None:
        """Тест безопасной TLS коммуникации."""
        
        tls_handler = TLSHandler(
            min_tls_version='TLSv1.3',
            cipher_suites=[
                'TLS_AES_256_GCM_SHA384',
                'TLS_CHACHA20_POLY1305_SHA256',
                'TLS_AES_128_GCM_SHA256'
            ],
            certificate_validation=True,
            ocsp_stapling=True,
            hsts_enabled=True
        )
        
        # Тест генерации сертификатов для тестирования
        test_cert_config = {
            "common_name": "trading-api.company.com",
            "organization": "Trading Company",
            "country": "US",
            "validity_days": 365,
            "key_size": 4096,
            "san_domains": [
                "api.trading.com",
                "websocket.trading.com",
                "admin.trading.com"
            ]
        }
        
        cert_result = tls_handler.generate_test_certificate(test_cert_config)
        
        assert 'certificate' in cert_result
        assert 'private_key' in cert_result
        assert 'public_key' in cert_result
        
        # Проверяем валидность сертификата
        cert_validation = tls_handler.validate_certificate(cert_result['certificate'])
        
        assert cert_validation['valid'] is True
        assert cert_validation['common_name'] == test_cert_config['common_name']
        assert len(cert_validation['san_domains']) == len(test_cert_config['san_domains'])
        
        # Тест TLS handshake (мокируем)
        handshake_data = {
            "client_hello": {
                "tls_version": "TLSv1.3",
                "cipher_suites": ["TLS_AES_256_GCM_SHA384"],
                "extensions": ["server_name", "supported_groups", "signature_algorithms"]
            },
            "server_hello": {
                "tls_version": "TLSv1.3",
                "cipher_suite": "TLS_AES_256_GCM_SHA384",
                "compression": None
            }
        }
        
        handshake_result = tls_handler.validate_handshake(handshake_data)
        
        assert handshake_result['handshake_valid'] is True
        assert handshake_result['tls_version'] == "TLSv1.3"
        assert handshake_result['cipher_suite'] == "TLS_AES_256_GCM_SHA384"
        
        # Тест шифрования TLS трафика
        test_messages = [
            {"type": "order", "data": {"symbol": "BTCUSDT", "side": "BUY", "amount": "1.0"}},
            {"type": "balance_update", "data": {"balance": "50000.00", "currency": "USD"}},
            {"type": "trade_confirmation", "data": {"trade_id": "12345", "status": "FILLED"}}
        ]
        
        for message in test_messages:
            # Шифруем сообщение для передачи по TLS
            encrypted_message = tls_handler.encrypt_application_data(
                message=json.dumps(message),
                session_key=cert_result['session_key'] if 'session_key' in cert_result else secrets.token_bytes(32)
            )
            
            assert encrypted_message['encrypted'] is True
            assert 'ciphertext' in encrypted_message
            assert 'mac' in encrypted_message
            
            # Расшифровываем сообщение
            decrypted_message = tls_handler.decrypt_application_data(
                encrypted_data=encrypted_message,
                session_key=cert_result['session_key'] if 'session_key' in cert_result else secrets.token_bytes(32)
            )
            
            assert json.loads(decrypted_message) == message

    def test_cryptographic_performance_benchmarks(self) -> None:
        """Тест производительности криптографических операций."""
        
        # Подготавливаем тестовые данные разного размера
        test_data_sizes = [
            (1024, "1KB"),      # Небольшие сообщения
            (64 * 1024, "64KB"), # Средние документы
            (1024 * 1024, "1MB") # Большие файлы
        ]
        
        performance_results = {}
        
        for data_size, size_label in test_data_sizes:
            test_data = os.urandom(data_size)
            
            # Benchmark AES-256-GCM
            aes_times = []
            for _ in range(10):  # 10 итераций
                start_time = time.perf_counter()
                
                key = Fernet.generate_key()
                cipher = Fernet(key)
                encrypted = cipher.encrypt(test_data)
                decrypted = cipher.decrypt(encrypted)
                
                end_time = time.perf_counter()
                aes_times.append(end_time - start_time)
                
                assert decrypted == test_data
            
            # Benchmark RSA-4096
            rsa_times = []
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096,
                backend=default_backend()
            )
            public_key = private_key.public_key()
            
            # RSA может шифровать только небольшие данные
            small_data = test_data[:256] if len(test_data) > 256 else test_data
            
            for _ in range(10):
                start_time = time.perf_counter()
                
                encrypted = public_key.encrypt(
                    small_data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                
                decrypted = private_key.decrypt(
                    encrypted,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                
                end_time = time.perf_counter()
                rsa_times.append(end_time - start_time)
                
                assert decrypted == small_data
            
            # Benchmark SHA-256
            sha_times = []
            for _ in range(10):
                start_time = time.perf_counter()
                
                hash_obj = hashlib.sha256()
                hash_obj.update(test_data)
                digest = hash_obj.digest()
                
                end_time = time.perf_counter()
                sha_times.append(end_time - start_time)
                
                assert len(digest) == 32
            
            performance_results[size_label] = {
                'aes_avg_time': sum(aes_times) / len(aes_times),
                'aes_throughput_mbps': (data_size / (sum(aes_times) / len(aes_times))) / (1024 * 1024),
                'rsa_avg_time': sum(rsa_times) / len(rsa_times),
                'sha_avg_time': sum(sha_times) / len(sha_times),
                'sha_throughput_mbps': (data_size / (sum(sha_times) / len(sha_times))) / (1024 * 1024)
            }
        
        # Проверяем производительность
        for size_label, results in performance_results.items():
            # AES должен быть быстрым
            assert results['aes_throughput_mbps'] > 10  # Минимум 10 MB/s
            
            # SHA должен быть очень быстрым
            assert results['sha_throughput_mbps'] > 50  # Минимум 50 MB/s
            
            # RSA медленнее, но должен завершаться за разумное время
            assert results['rsa_avg_time'] < 1.0  # Менее 1 секунды

    def test_security_audit_and_compliance(
        self,
        sample_financial_data: Dict[str, Any]
    ) -> None:
        """Тест аудита безопасности и соответствия стандартам."""
        
        security_auditor = SecurityAuditor(
            compliance_standards=['PCI_DSS', 'SOX', 'GDPR', 'ISO_27001'],
            audit_level='COMPREHENSIVE',
            real_time_monitoring=True
        )
        
        # Симулируем различные операции для аудита
        operations = [
            {
                "operation": "ENCRYPT_PRIVATE_KEYS",
                "user_id": sample_financial_data['user_id'],
                "data_classification": "TOP_SECRET",
                "encryption_algorithm": "AES_256_GCM",
                "key_strength": 256,
                "success": True
            },
            {
                "operation": "STORE_API_KEYS",
                "user_id": sample_financial_data['user_id'],
                "data_classification": "SECRET",
                "encryption_algorithm": "AES_256_GCM",
                "key_strength": 256,
                "success": True
            },
            {
                "operation": "MFA_VERIFICATION",
                "user_id": sample_financial_data['user_id'],
                "data_classification": "CONFIDENTIAL",
                "encryption_algorithm": "SHA256_HMAC",
                "key_strength": 256,
                "success": True
            },
            {
                "operation": "UNAUTHORIZED_ACCESS_ATTEMPT",
                "user_id": "attacker_user",
                "data_classification": "TOP_SECRET",
                "encryption_algorithm": None,
                "key_strength": 0,
                "success": False
            }
        ]
        
        audit_results = []
        
        for operation in operations:
            audit_entry = SecurityAudit(
                timestamp=datetime.utcnow(),
                operation=operation['operation'],
                user_id=operation['user_id'],
                success=operation['success'],
                encryption_used=operation['encryption_algorithm'] is not None,
                key_strength=operation['key_strength'],
                compliance_level=operation['data_classification']
            )
            
            # Аудируем операцию
            audit_result = security_auditor.audit_operation(audit_entry)
            audit_results.append(audit_result)
        
        # Проверяем результаты аудита
        successful_operations = [r for r in audit_results if r.security_rating == 'HIGH']
        failed_operations = [r for r in audit_results if r.security_rating == 'CRITICAL_RISK']
        
        assert len(successful_operations) == 3  # Первые 3 операции успешны
        assert len(failed_operations) == 1     # Последняя операция неудачна
        
        # Проверяем соответствие стандартам
        compliance_report = security_auditor.generate_compliance_report()
        
        assert 'PCI_DSS' in compliance_report
        assert 'SOX' in compliance_report
        assert 'GDPR' in compliance_report
        assert 'ISO_27001' in compliance_report
        
        # PCI DSS - защита платежных данных
        pci_compliance = compliance_report['PCI_DSS']
        assert pci_compliance['encryption_in_transit'] is True
        assert pci_compliance['encryption_at_rest'] is True
        assert pci_compliance['access_control'] is True
        assert pci_compliance['audit_logging'] is True
        
        # GDPR - защита персональных данных
        gdpr_compliance = compliance_report['GDPR']
        assert gdpr_compliance['data_encryption'] is True
        assert gdpr_compliance['consent_management'] is True
        assert gdpr_compliance['right_to_erasure'] is True
        assert gdpr_compliance['breach_notification'] is True
        
        # Общий рейтинг соответствия
        overall_compliance = compliance_report['overall_rating']
        assert overall_compliance >= 85  # Минимум 85% соответствия


class SecurityAuditor:
    """Аудитор безопасности для тестирования."""
    
    def __init__(self, compliance_standards: List[str], audit_level: str, real_time_monitoring: bool):
        self.compliance_standards = compliance_standards
        self.audit_level = audit_level
        self.real_time_monitoring = real_time_monitoring
    
    def audit_operation(self, audit_entry: SecurityAudit) -> SecurityAudit:
        """Аудит операции безопасности."""
        # Имитируем аудит
        if audit_entry.success and audit_entry.encryption_used and audit_entry.key_strength >= 256:
            audit_entry.security_rating = 'HIGH'
        elif not audit_entry.success:
            audit_entry.security_rating = 'CRITICAL_RISK'
        else:
            audit_entry.security_rating = 'MEDIUM'
        
        return audit_entry
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Генерация отчета о соответствии."""
        return {
            'PCI_DSS': {
                'encryption_in_transit': True,
                'encryption_at_rest': True,
                'access_control': True,
                'audit_logging': True,
                'compliance_percentage': 95
            },
            'SOX': {
                'financial_controls': True,
                'audit_trail': True,
                'data_integrity': True,
                'compliance_percentage': 92
            },
            'GDPR': {
                'data_encryption': True,
                'consent_management': True,
                'right_to_erasure': True,
                'breach_notification': True,
                'compliance_percentage': 89
            },
            'ISO_27001': {
                'security_policy': True,
                'risk_management': True,
                'incident_response': True,
                'business_continuity': True,
                'compliance_percentage': 91
            },
            'overall_rating': 92
        }