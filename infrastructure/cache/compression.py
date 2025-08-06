"""
Data compression utilities for cache services.
"""

import gzip
import logging
import pickle
import time
import zlib
from enum import Enum
from typing import Any, Optional, Union

try:
    import lz4.frame
except ImportError:
    lz4 = None
    logger = logging.getLogger(__name__)
    logger.warning("lz4 not available, LZ4 compression will be disabled")

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Types of compression algorithms."""

    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZLIB = "zlib"


class CompressionError(Exception):
    """Exception raised for compression errors."""

    pass


class Compressor:
    """Base class for data compressors."""

    def __init__(self, compression_type: CompressionType = CompressionType.LZ4) -> None:
        self.compression_type = compression_type
        self._compression_threshold = 1024  # Only compress data larger than 1KB

    def should_compress(self, data: bytes) -> bool:
        """Determine if data should be compressed."""
        return len(data) > self._compression_threshold

    def compress(self, data: Union[str, bytes, Any]) -> bytes:
        """Compress data."""
        try:
            # Serialize data if needed
            if isinstance(data, str):
                serialized_data = data.encode("utf-8")
            elif isinstance(data, bytes):
                serialized_data = data
            else:
                serialized_data = pickle.dumps(data)
            # Check if compression is needed
            if not self.should_compress(serialized_data):
                return self._add_header(serialized_data, CompressionType.NONE)
            # Compress based on type
            if self.compression_type == CompressionType.GZIP:
                compressed_data = gzip.compress(serialized_data, compresslevel=6)
            elif self.compression_type == CompressionType.LZ4:
                if lz4 is None:
                    raise CompressionError("LZ4 compression not available")
                compressed_data = lz4.frame.compress(
                    serialized_data, compression_level=1
                )
            elif self.compression_type == CompressionType.ZLIB:
                compressed_data = zlib.compress(serialized_data, level=6)
            else:
                compressed_data = serialized_data
            # Add compression header
            return self._add_header(compressed_data, self.compression_type)
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise CompressionError(f"Compression failed: {e}")

    def decompress(self, data: bytes) -> Any:
        """Decompress data."""
        try:
            # Extract compression type from header
            compression_type, compressed_data = self._extract_header(data)
            # Decompress based on type
            if compression_type == CompressionType.NONE:
                decompressed_data = compressed_data
            elif compression_type == CompressionType.GZIP:
                decompressed_data = gzip.decompress(compressed_data)
            elif compression_type == CompressionType.LZ4:
                if lz4 is None:
                    raise CompressionError("LZ4 decompression not available")
                decompressed_data = lz4.frame.decompress(compressed_data)
            elif compression_type == CompressionType.ZLIB:
                decompressed_data = zlib.decompress(compressed_data)
            else:
                raise CompressionError(f"Unknown compression type: {compression_type}")
            # Try to deserialize
            try:
                return pickle.loads(decompressed_data)
            except (pickle.UnpicklingError, EOFError):
                # Return as string if pickle fails
                return decompressed_data.decode("utf-8")
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise CompressionError(f"Decompression failed: {e}")

    def _add_header(self, data: bytes, compression_type: CompressionType) -> bytes:
        """Add compression header to data."""
        header = compression_type.value.encode("utf-8")
        header_length = len(header).to_bytes(1, byteorder="big")
        return header_length + header + data

    def _extract_header(self, data: bytes) -> tuple[CompressionType, bytes]:
        """Extract compression header from data."""
        header_length = int.from_bytes(data[:1], byteorder="big")
        header = data[1 : 1 + header_length].decode("utf-8")
        compression_type = CompressionType(header)
        compressed_data = data[1 + header_length :]
        return compression_type, compressed_data


class AdaptiveCompressor(Compressor):
    """Adaptive compressor that chooses the best compression algorithm."""

    def __init__(self) -> None:
        super().__init__()
        self._compression_stats = {
            CompressionType.GZIP: {"ratio": 0.0, "speed": 0.0, "count": 0},
            CompressionType.LZ4: {"ratio": 0.0, "speed": 0.0, "count": 0},
            CompressionType.ZLIB: {"ratio": 0.0, "speed": 0.0, "count": 0},
        }

    def compress(self, data: Union[str, bytes, Any]) -> bytes:
        """Compress data using the best available algorithm."""
        try:
            # Serialize data
            if isinstance(data, str):
                serialized_data = data.encode("utf-8")
            elif isinstance(data, bytes):
                serialized_data = data
            else:
                serialized_data = pickle.dumps(data)
            # Check if compression is needed
            if not self.should_compress(serialized_data):
                return self._add_header(serialized_data, CompressionType.NONE)
            # Try all compression algorithms and choose the best
            best_compression = CompressionType.NONE
            best_ratio = 1.0
            best_compressed_data = serialized_data
            for compression_type in [
                CompressionType.LZ4,
                CompressionType.GZIP,
                CompressionType.ZLIB,
            ]:
                try:
                    start_time = time.time()
                    if compression_type == CompressionType.GZIP:
                        compressed_data = gzip.compress(
                            serialized_data, compresslevel=6
                        )
                    elif compression_type == CompressionType.LZ4:
                        if lz4 is None:
                            continue
                        compressed_data = lz4.frame.compress(
                            serialized_data, compression_level=1
                        )
                    elif compression_type == CompressionType.ZLIB:
                        compressed_data = zlib.compress(serialized_data, level=6)
                    compression_time = time.time() - start_time
                    compression_ratio = len(compressed_data) / len(serialized_data)
                    # Update stats
                    self._update_stats(
                        compression_type, compression_ratio, compression_time
                    )
                    # Choose best compression (considering both ratio and speed)
                    if compression_ratio < best_ratio:
                        best_ratio = compression_ratio
                        best_compression = compression_type
                        best_compressed_data = compressed_data
                except Exception as e:
                    logger.warning(f"Compression {compression_type.value} failed: {e}")
                    continue
            return self._add_header(best_compressed_data, best_compression)
        except Exception as e:
            logger.error(f"Adaptive compression failed: {e}")
            raise CompressionError(f"Adaptive compression failed: {e}")

    def _update_stats(
        self, compression_type: CompressionType, ratio: float, speed: float
    ) -> None:
        """Update compression statistics."""
        stats = self._compression_stats[compression_type]
        stats["count"] += 1
        stats["ratio"] = (stats["ratio"] * (stats["count"] - 1) + ratio) / stats[
            "count"
        ]
        stats["speed"] = (stats["speed"] * (stats["count"] - 1) + speed) / stats[
            "count"
        ]

    def get_compression_stats(self) -> dict[str, Any]:
        """Get compression statistics."""
        return self._compression_stats.copy()


class CacheCompressor:
    """High-level compressor for cache services."""

    def __init__(
        self,
        compression_type: CompressionType = CompressionType.LZ4,
        enable_adaptive: bool = True,
    ):
        if enable_adaptive:
            self._compressor: Union[AdaptiveCompressor, Compressor] = AdaptiveCompressor()
        else:
            self._compressor = Compressor(compression_type)
        self._compression_enabled = True
        self._compression_threshold = 1024  # 1KB

    def compress_value(self, value: Any) -> bytes:
        """Compress a value for storage."""
        if not self._compression_enabled:
            return pickle.dumps(value)
        try:
            return self._compressor.compress(value)
        except CompressionError:
            # Fallback to no compression
            logger.warning("Compression failed, falling back to no compression")
            return pickle.dumps(value)

    def decompress_value(self, data: bytes) -> Any:
        """Decompress a value from storage."""
        if not self._compression_enabled:
            return pickle.loads(data)
        try:
            return self._compressor.decompress(data)
        except CompressionError:
            # Fallback to no compression
            logger.warning("Decompression failed, falling back to no compression")
            return pickle.loads(data)

    def enable_compression(self, enabled: bool = True) -> None:
        """Enable or disable compression."""
        self._compression_enabled = enabled

    def set_compression_threshold(self, threshold: int) -> None:
        """Set compression threshold in bytes."""
        self._compression_threshold = threshold
        self._compressor._compression_threshold = threshold

    def get_compression_stats(self) -> dict[str, Any]:
        """Get compression statistics."""
        if isinstance(self._compressor, AdaptiveCompressor):
            return self._compressor.get_compression_stats()
        return {}

    def get_compression_ratio(self, data: Any) -> float:
        """Get compression ratio for given data."""
        try:
            original_size = len(pickle.dumps(data))
            compressed_size = len(self.compress_value(data))
            return compressed_size / original_size
        except Exception:
            return 1.0


# Utility functions
def get_optimal_compression_type(data: Any) -> CompressionType:
    """Determine the optimal compression type for given data."""
    try:
        serialized_data = pickle.dumps(data)
        data_size = len(serialized_data)
        if data_size < 1024:
            return CompressionType.NONE
        elif data_size < 10000:
            return CompressionType.LZ4  # Fast for small data
        else:
            return CompressionType.GZIP  # Better compression for large data
    except Exception:
        return CompressionType.NONE


def estimate_compression_ratio(data: Any, compression_type: CompressionType) -> float:
    """Estimate compression ratio without actually compressing."""
    try:
        serialized_data = pickle.dumps(data)
        data_size = len(serialized_data)
        # Rough estimates based on data type and size
        if isinstance(data, str):
            if compression_type == CompressionType.LZ4:
                return 0.7  # ~30% compression
            elif compression_type == CompressionType.GZIP:
                return 0.5  # ~50% compression
            elif compression_type == CompressionType.ZLIB:
                return 0.6  # ~40% compression
        elif isinstance(data, dict) or isinstance(data, list):
            if compression_type == CompressionType.LZ4:
                return 0.8  # ~20% compression
            elif compression_type == CompressionType.GZIP:
                return 0.6  # ~40% compression
            elif compression_type == CompressionType.ZLIB:
                return 0.7  # ~30% compression
        else:
            return 0.9  # Minimal compression for other types
    except Exception:
        return 1.0
    
    # Fallback return for any uncovered cases
    return 1.0


# Performance monitoring
class CompressionMonitor:
    """Monitor compression performance."""

    def __init__(self) -> None:
        self._stats = {
            "total_compressed": 0,
            "total_uncompressed": 0,
            "total_compression_time": 0.0,
            "total_decompression_time": 0.0,
            "compression_errors": 0,
            "decompression_errors": 0,
        }

    def record_compression(
        self,
        original_size: int,
        compressed_size: int,
        compression_time: float,
        success: bool = True,
    ) -> None:
        """Record compression statistics."""
        if success:
            self._stats["total_compressed"] += 1
            self._stats["total_compression_time"] += compression_time
        else:
            self._stats["compression_errors"] += 1

    def record_decompression(self, decompression_time: float, success: bool = True) -> None:
        """Record decompression statistics."""
        if success:
            self._stats["total_uncompressed"] += 1
            self._stats["total_decompression_time"] += decompression_time
        else:
            self._stats["decompression_errors"] += 1

    def get_stats(self) -> dict[str, Any]:
        """Get compression statistics."""
        stats = self._stats.copy()
        if stats["total_compressed"] > 0:
            stats["avg_compression_time"] = (
                stats["total_compression_time"] / stats["total_compressed"]
            )
        if stats["total_uncompressed"] > 0:
            stats["avg_decompression_time"] = (
                stats["total_decompression_time"] / stats["total_uncompressed"]
            )
        return stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = {
            "total_compressed": 0,
            "total_uncompressed": 0,
            "total_compression_time": 0.0,
            "total_decompression_time": 0.0,
            "compression_errors": 0,
            "decompression_errors": 0,
        }


# Global compression monitor
_compression_monitor = CompressionMonitor()


def get_compression_monitor() -> CompressionMonitor:
    """Get the global compression monitor."""
    return _compression_monitor
