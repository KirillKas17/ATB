"""Utils module."""

import hashlib
import json
import threading
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def generate_snapshot_id() -> str:
    """Generate snapshot ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    thread_id = threading.get_ident()
    random_suffix = hashlib.sha256(
        f"{timestamp}_{thread_id}_{time.time()}".encode()
    ).hexdigest()[:12]
    return f"snapshot_{timestamp}_{random_suffix}"


def calculate_checksum(data: bytes) -> str:
    """Calculate checksum."""
    return hashlib.sha256(data).hexdigest()


def serialize_snapshot(data: Dict[str, Any], format_type: str) -> bytes:
    """Serialize snapshot."""
    return json.dumps(data, default=str).encode("utf-8")


def deserialize_snapshot(data: bytes, format_type: str) -> Dict[str, Any]:
    """Deserialize snapshot."""
    return json.loads(data.decode("utf-8"))


def encrypt_data(data: bytes, key: str) -> bytes:
    """Encrypt data."""
    if not key:
        return data
    encrypted = bytearray()
    key_bytes = key.encode()
    for i, byte in enumerate(data):
        encrypted.append(byte ^ key_bytes[i % len(key_bytes)])
    return bytes(encrypted)


def decrypt_data(data: bytes, key: str) -> bytes:
    """Decrypt data."""
    if not key:
        return data
    decrypted = bytearray()
    key_bytes = key.encode()
    for i, byte in enumerate(data):
        decrypted.append(byte ^ key_bytes[i % len(key_bytes)])
    return bytes(decrypted)


def validate_snapshot_data(data: Dict[str, Any]) -> bool:
    """Validate snapshot data."""
    return isinstance(data, dict) and "id" in data


def estimate_object_size(obj: Any) -> int:
    """Estimate object size."""
    try:
        import sys

        return sys.getsizeof(obj)
    except Exception:
        return 1024


def cleanup_old_files(directory: str, max_age_days: int) -> int:
    """Cleanup old files."""
    return 0


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0) -> Callable[[F], F]:
    """Retry decorator."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
            if last_exception is not None:
                raise last_exception
            raise RuntimeError("Unknown error in retry_on_failure")

        return wrapper

    return decorator


def get_memory_usage() -> Dict[str, float]:
    """Get memory usage."""
    return {
        "process_memory_mb": 0.0,
        "process_memory_percent": 0.0,
        "system_memory_percent": 0.0,
        "system_memory_available_mb": 0.0,
        "system_memory_total_mb": 0.0,
    }


def clear_serialization_cache() -> None:
    """Clear serialization cache."""
    pass
