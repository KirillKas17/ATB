"""
Data encryption utilities for cache services.
"""

import base64
import hashlib
import logging
import os
import secrets
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class EncryptionError(Exception):
    """Exception raised for encryption errors."""

    pass


class EncryptionType(Enum):
    """Types of encryption algorithms."""

    NONE = "none"
    AES_256 = "aes_256"
    FERNET = "fernet"


class KeyManager:
    """Manages encryption keys securely."""

    def __init__(self, master_key: Optional[str] = None):
        self.master_key = master_key or os.getenv("CACHE_ENCRYPTION_KEY")
        if not self.master_key:
            self.master_key = self._generate_master_key()
            logger.warning(
                "No encryption key provided, using generated key. Set CACHE_ENCRYPTION_KEY env var for production."
            )

        self._key_cache: Dict[str, bytes] = {}

    def _generate_master_key(self) -> str:
        """Generate a secure master key."""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("utf-8")

    def derive_key(self, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from master key."""
        if salt is None:
            salt = secrets.token_bytes(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend(),
        )

        if self.master_key is None:
            raise EncryptionError("Master key is not set")
        key = base64.urlsafe_b64decode(self.master_key.encode("utf-8"))
        derived_key = kdf.derive(key)
        return bytes(derived_key)

    def get_key_for_namespace(self, namespace: str) -> bytes:
        """Get encryption key for specific namespace."""
        if namespace not in self._key_cache:
            salt = hashlib.sha256(namespace.encode()).digest()[:16]
            key = self.derive_key(salt)
            self._key_cache[namespace] = key

        return self._key_cache[namespace]


class AESEncryptor:
    """AES-256 encryption implementation."""

    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager

    def encrypt(self, data: bytes, namespace: str = "default") -> bytes:
        """Encrypt data using AES-256-GCM."""
        try:
            key = self.key_manager.get_key_for_namespace(namespace)

            # Generate random IV
            iv = secrets.token_bytes(12)

            # Create cipher
            cipher = Cipher(
                algorithms.AES(key), modes.GCM(iv), backend=default_backend()
            )
            encryptor = cipher.encryptor()

            # Encrypt data
            ciphertext = encryptor.update(data) + encryptor.finalize()

            # Combine IV, tag, and ciphertext
            encrypted_data = iv + encryptor.tag + ciphertext

            return bytes(encrypted_data)

        except Exception as e:
            logger.error(f"AES encryption failed: {e}")
            raise EncryptionError(f"AES encryption failed: {e}")

    def decrypt(self, encrypted_data: bytes, namespace: str = "default") -> bytes:
        """Decrypt data using AES-256-GCM."""
        try:
            key = self.key_manager.get_key_for_namespace(namespace)

            # Extract IV, tag, and ciphertext
            iv = encrypted_data[:12]
            tag = encrypted_data[12:28]
            ciphertext = encrypted_data[28:]

            # Create cipher
            cipher = Cipher(
                algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend()
            )
            decryptor = cipher.decryptor()

            # Decrypt data
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()

            return bytes(plaintext)

        except Exception as e:
            logger.error(f"AES decryption failed: {e}")
            raise EncryptionError(f"AES decryption failed: {e}")


class FernetEncryptor:
    """Fernet encryption implementation."""

    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager
        self._fernet_cache: Dict[str, Fernet] = {}

    def _get_fernet(self, namespace: str) -> Fernet:
        """Get Fernet instance for namespace."""
        if namespace not in self._fernet_cache:
            key = self.key_manager.get_key_for_namespace(namespace)
            fernet_key = base64.urlsafe_b64encode(key)
            self._fernet_cache[namespace] = Fernet(fernet_key)

        return self._fernet_cache[namespace]

    def encrypt(self, data: bytes, namespace: str = "default") -> bytes:
        """Encrypt data using Fernet."""
        try:
            fernet = self._get_fernet(namespace)
            return bytes(fernet.encrypt(data))
        except Exception as e:
            logger.error(f"Fernet encryption failed: {e}")
            raise EncryptionError(f"Fernet encryption failed: {e}")

    def decrypt(self, encrypted_data: bytes, namespace: str = "default") -> bytes:
        """Decrypt data using Fernet."""
        try:
            fernet = self._get_fernet(namespace)
            return bytes(fernet.decrypt(encrypted_data))
        except Exception as e:
            logger.error(f"Fernet decryption failed: {e}")
            raise EncryptionError(f"Fernet decryption failed: {e}")


class CacheEncryptor:
    """High-level encryptor for cache services."""

    def __init__(
        self,
        encryption_type: EncryptionType = EncryptionType.AES_256,
        master_key: Optional[str] = None,
    ):
        self.encryption_type = encryption_type
        self.key_manager = KeyManager(master_key)
        self._encryption_enabled = True
        self._encryptor: Optional[Union[AESEncryptor, FernetEncryptor]] = None

        if encryption_type == EncryptionType.AES_256:
            self._encryptor = AESEncryptor(self.key_manager)
        elif encryption_type == EncryptionType.FERNET:
            self._encryptor = FernetEncryptor(self.key_manager)

    def should_encrypt(self, key: str, value: Any) -> bool:
        """Determine if data should be encrypted."""
        if not self._encryption_enabled or self._encryptor is None:
            return False

        # Encrypt sensitive keys
        sensitive_patterns = [
            "password",
            "token",
            "secret",
            "key",
            "auth",
            "credential",
            "private",
            "api_key",
            "session",
            "jwt",
            "bearer",
        ]

        key_lower = key.lower()
        return any(pattern in key_lower for pattern in sensitive_patterns)

    def encrypt_value(self, key: str, value: Union[str, bytes, Any]) -> bytes:
        """Encrypt a value for storage."""
        if not self.should_encrypt(key, value) or self._encryptor is None:
            return value if isinstance(value, bytes) else str(value).encode("utf-8")

        try:
            # Serialize value if needed
            if isinstance(value, str):
                data = value.encode("utf-8")
            elif isinstance(value, bytes):
                data = value
            else:
                import pickle

                data = pickle.dumps(value)

            # Encrypt data
            encrypted_data = self._encryptor.encrypt(data, namespace="cache")

            # Add encryption header
            return self._add_header(encrypted_data, self.encryption_type)

        except Exception as e:
            logger.error(f"Encryption failed for key {key}: {e}")
            # Fallback to no encryption
            return value if isinstance(value, bytes) else str(value).encode("utf-8")

    def decrypt_value(self, key: str, data: bytes) -> Any:
        """Decrypt a value from storage."""
        try:
            # Check if data is encrypted
            if not data.startswith(b"ENC:"):
                return data

            # Extract encryption type and encrypted data
            encryption_type, encrypted_data = self._extract_header(data)

            if encryption_type == EncryptionType.NONE or self._encryptor is None:
                return encrypted_data

            # Decrypt data
            decrypted_data = self._encryptor.decrypt(encrypted_data, namespace="cache")

            # Try to deserialize
            try:
                import pickle

                return pickle.loads(decrypted_data)
            except (pickle.UnpicklingError, EOFError):
                # Return as string if pickle fails
                return decrypted_data.decode("utf-8")

        except Exception as e:
            logger.error(f"Decryption failed for key {key}: {e}")
            # Return original data if decryption fails
            return data

    def _add_header(self, data: bytes, encryption_type: EncryptionType) -> bytes:
        """Add encryption header to data."""
        header = f"ENC:{encryption_type.value}:".encode("utf-8")
        return header + data

    def _extract_header(self, data: bytes) -> Tuple[EncryptionType, bytes]:
        """Extract encryption header from data."""
        header_end = data.find(b":", 4)  # Skip "ENC:"
        if header_end == -1:
            return (EncryptionType.NONE, data)
        encryption_type_str = data[4:header_end].decode("utf-8")
        encryption_type = EncryptionType(encryption_type_str)
        encrypted_data = data[header_end + 1 :]
        return (encryption_type, encrypted_data)

    def enable_encryption(self, enabled: bool = True) -> None:
        """Enable or disable encryption."""
        self._encryption_enabled = enabled

    def get_encryption_stats(self) -> Dict[str, Any]:
        """Get encryption statistics."""
        return {
            "encryption_type": self.encryption_type.value,
            "encryption_enabled": self._encryption_enabled,
            "key_manager_initialized": self.key_manager.master_key is not None,
        }


# Utility functions
def is_sensitive_key(key: str) -> bool:
    """Check if a key contains sensitive information."""
    sensitive_patterns = [
        "password",
        "token",
        "secret",
        "key",
        "auth",
        "credential",
        "private",
        "api_key",
        "session",
        "jwt",
        "bearer",
        "signature",
    ]

    key_lower = key.lower()
    return any(pattern in key_lower for pattern in sensitive_patterns)


def sanitize_log_data(data: Any, sensitive_keys: Optional[list] = None) -> Any:
    """Sanitize data for logging by masking sensitive values."""
    if sensitive_keys is None:
        sensitive_keys = ["password", "token", "secret", "key", "auth"]

    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            if any(pattern in key.lower() for pattern in sensitive_keys):
                sanitized[key] = "***MASKED***"
            else:
                sanitized[key] = sanitize_log_data(value, sensitive_keys)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_log_data(item, sensitive_keys) for item in data]
    else:
        return data


# Performance monitoring
class EncryptionMonitor:
    """Monitor encryption performance."""

    def __init__(self) -> None:
        self._stats = {
            "total_encrypted": 0,
            "total_decrypted": 0,
            "total_encryption_time": 0.0,
            "total_decryption_time": 0.0,
            "encryption_errors": 0,
            "decryption_errors": 0,
            "sensitive_keys_encrypted": 0,
        }

    def record_encryption(
        self, encryption_time: float, success: bool = True, is_sensitive: bool = False
    ) -> None:
        """Record encryption statistics."""
        if success:
            self._stats["total_encrypted"] += 1
            self._stats["total_encryption_time"] += encryption_time
            if is_sensitive:
                self._stats["sensitive_keys_encrypted"] += 1
        else:
            self._stats["encryption_errors"] += 1

    def record_decryption(self, decryption_time: float, success: bool = True) -> None:
        """Record decryption statistics."""
        if success:
            self._stats["total_decrypted"] += 1
            self._stats["total_decryption_time"] += decryption_time
        else:
            self._stats["decryption_errors"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get encryption statistics."""
        stats = self._stats.copy()

        if stats["total_encrypted"] > 0:
            stats["avg_encryption_time"] = (
                stats["total_encryption_time"] / stats["total_encrypted"]
            )

        if stats["total_decrypted"] > 0:
            stats["avg_decryption_time"] = (
                stats["total_decryption_time"] / stats["total_decrypted"]
            )

        return stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = {
            "total_encrypted": 0,
            "total_decrypted": 0,
            "total_encryption_time": 0.0,
            "total_decryption_time": 0.0,
            "encryption_errors": 0,
            "decryption_errors": 0,
            "sensitive_keys_encrypted": 0,
        }


# Global encryption monitor
_encryption_monitor = EncryptionMonitor()


def get_encryption_monitor() -> EncryptionMonitor:
    """Get the global encryption monitor."""
    return _encryption_monitor
