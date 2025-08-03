"""
Юнит-тесты для моделей данных market_profiles.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta
from pathlib import Path
from infrastructure.market_profiles.models.storage_config import StorageConfig
from infrastructure.market_profiles.models.analysis_config import AnalysisConfig
from infrastructure.market_profiles.models.storage_models import (
    StorageStatistics, PatternMetadata, BehaviorRecord, SuccessMapEntry,
    StorageStatus
)
from domain.types.market_maker_types import (
    Confidence, SimilarityScore, Accuracy
)
class TestStorageConfig:
    """Тесты для StorageConfig."""
    def test_storage_config_default_values(self) -> None:
        """Тест значений по умолчанию для StorageConfig."""
        config = StorageConfig()
        assert config.base_path == Path("market_profiles")
        assert config.compression_enabled is True
        assert config.compression_level == 6
        assert config.max_workers == 4
        assert config.cache_size == 1000
        assert config.backup_enabled is True
        assert config.backup_interval_hours == 24
        assert config.cleanup_enabled is True
        assert config.cleanup_interval_days == 30
    def test_storage_config_custom_values(self) -> None:
        """Тест кастомных значений для StorageConfig."""
        custom_path = Path("/custom/path")
        config = StorageConfig(
            base_path=custom_path,
            compression_enabled=False,
            compression_level=9,
            max_workers=8,
            cache_size=2000,
            backup_enabled=False,
            backup_interval_hours=12,
            cleanup_enabled=False,
            cleanup_interval_days=7
        )
        assert config.base_path == custom_path
        assert config.compression_enabled is False
        assert config.compression_level == 9
        assert config.max_workers == 8
        assert config.cache_size == 2000
        assert config.backup_enabled is False
        assert config.backup_interval_hours == 12
        assert config.cleanup_enabled is False
        assert config.cleanup_interval_days == 7
    def test_storage_config_paths(self) -> None:
        """Тест путей в StorageConfig."""
        config = StorageConfig(base_path=Path("/test/path"))
        assert config.patterns_directory == Path("/test/path/patterns")
        assert config.metadata_directory == Path("/test/path/metadata")
        assert config.behavior_directory == Path("/test/path/behavior")
        assert config.backup_directory == Path("/test/path/backup")
    def test_storage_config_validation(self) -> None:
        """Тест валидации StorageConfig."""
        # Тест с отрицательными значениями
        with pytest.raises(ValueError):
            StorageConfig(max_workers=-1)
        with pytest.raises(ValueError):
            StorageConfig(cache_size=-1)
        with pytest.raises(ValueError):
            StorageConfig(compression_level=0)
        with pytest.raises(ValueError):
            StorageConfig(compression_level=10)
    def test_storage_config_immutability(self) -> None:
        """Тест неизменяемости StorageConfig."""
        config = StorageConfig()
        # Попытка изменения должна вызвать ошибку
        with pytest.raises(Exception):
            config.base_path = Path("/new/path")
class TestAnalysisConfig:
    """Тесты для AnalysisConfig."""
    def test_analysis_config_default_values(self) -> None:
        """Тест значений по умолчанию для AnalysisConfig."""
        config = AnalysisConfig()
        assert config.min_confidence == Confidence(0.6)
        assert config.similarity_threshold == 0.8
        assert config.accuracy_threshold == 0.7
        assert config.volume_threshold == 1000.0
        assert config.spread_threshold == 0.001
        assert config.time_window_seconds == 300
        assert config.min_trades_count == 10
        assert config.max_history_size == 1000
    def test_analysis_config_custom_values(self) -> None:
        """Тест кастомных значений для AnalysisConfig."""
        config = AnalysisConfig(
            min_confidence=Confidence(0.8),
            similarity_threshold=0.9,
            accuracy_threshold=0.8,
            volume_threshold=2000.0,
            spread_threshold=0.002,
            time_window_seconds=600,
            min_trades_count=20,
            max_history_size=2000
        )
        assert config.min_confidence == Confidence(0.8)
        assert config.similarity_threshold == 0.9
        assert config.accuracy_threshold == 0.8
        assert config.volume_threshold == 2000.0
        assert config.spread_threshold == 0.002
        assert config.time_window_seconds == 600
        assert config.min_trades_count == 20
        assert config.max_history_size == 2000
    def test_analysis_config_feature_weights(self) -> None:
        """Тест весов признаков в AnalysisConfig."""
        config = AnalysisConfig()
        assert "book_pressure" in config.feature_weights
        assert "volume_delta" in config.feature_weights
        assert "price_reaction" in config.feature_weights
        assert "spread_change" in config.feature_weights
        assert "order_imbalance" in config.feature_weights
        assert "liquidity_depth" in config.feature_weights
        assert "volume_concentration" in config.feature_weights
        # Проверяем, что веса в диапазоне [0, 1]
        for weight in config.feature_weights.values():
            assert 0.0 <= weight <= 1.0
    def test_analysis_config_market_phase_weights(self) -> None:
        """Тест весов рыночных фаз в AnalysisConfig."""
        config = AnalysisConfig()
        assert "accumulation" in config.market_phase_weights
        assert "markup" in config.market_phase_weights
        assert "distribution" in config.market_phase_weights
        assert "markdown" in config.market_phase_weights
        assert "transition" in config.market_phase_weights
    def test_analysis_config_volatility_weights(self) -> None:
        """Тест весов волатильности в AnalysisConfig."""
        config = AnalysisConfig()
        assert "low" in config.volatility_regime_weights
        assert "medium" in config.volatility_regime_weights
        assert "high" in config.volatility_regime_weights
        assert "extreme" in config.volatility_regime_weights
    def test_analysis_config_liquidity_weights(self) -> None:
        """Тест весов ликвидности в AnalysisConfig."""
        config = AnalysisConfig()
        assert "high" in config.liquidity_regime_weights
        assert "medium" in config.liquidity_regime_weights
        assert "low" in config.liquidity_regime_weights
        assert "very_low" in config.liquidity_regime_weights
    def test_analysis_config_validation(self) -> None:
        """Тест валидации AnalysisConfig."""
        # Тест с некорректными порогами
        with pytest.raises(ValueError):
            AnalysisConfig(similarity_threshold=1.5)
        with pytest.raises(ValueError):
            AnalysisConfig(accuracy_threshold=-0.1)
        with pytest.raises(ValueError):
            AnalysisConfig(volume_threshold=-100.0)
        with pytest.raises(ValueError):
            AnalysisConfig(time_window_seconds=0)
    def test_analysis_config_immutability(self) -> None:
        """Тест неизменяемости AnalysisConfig."""
        config = AnalysisConfig()
        # Попытка изменения должна вызвать ошибку
        with pytest.raises(Exception):
            config.similarity_threshold = 0.9
class TestStorageStatistics:
    """Тесты для StorageStatistics."""
    def test_storage_statistics_creation(self) -> None:
        """Тест создания StorageStatistics."""
        stats = StorageStatistics(
            total_patterns=100,
            total_symbols=10,
            total_successful_patterns=80,
            total_storage_size_bytes=1024000,
            avg_pattern_size_bytes=1024,
            compression_ratio=0.7,
            cache_hit_ratio=0.85,
            avg_read_time_ms=5.0,
            avg_write_time_ms=10.0,
            error_count=2,
            warning_count=5
        )
        assert stats.total_patterns == 100
        assert stats.total_symbols == 10
        assert stats.total_successful_patterns == 80
        assert stats.total_storage_size_bytes == 1024000
        assert stats.avg_pattern_size_bytes == 1024
        assert stats.compression_ratio == 0.7
        assert stats.cache_hit_ratio == 0.85
        assert stats.avg_read_time_ms == 5.0
        assert stats.avg_write_time_ms == 10.0
        assert stats.error_count == 2
        assert stats.warning_count == 5
    def test_storage_statistics_default_values(self) -> None:
        """Тест значений по умолчанию для StorageStatistics."""
        stats = StorageStatistics()
        assert stats.total_patterns == 0
        assert stats.total_symbols == 0
        assert stats.total_successful_patterns == 0
        assert stats.total_storage_size_bytes == 0
        assert stats.avg_pattern_size_bytes == 0
        assert stats.compression_ratio == 1.0
        assert stats.cache_hit_ratio == 0.0
        assert stats.avg_read_time_ms == 0.0
        assert stats.avg_write_time_ms == 0.0
        assert stats.error_count == 0
        assert stats.warning_count == 0
    def test_storage_statistics_validation(self) -> None:
        """Тест валидации StorageStatistics."""
        # Тест с отрицательными значениями
        with pytest.raises(ValueError):
            StorageStatistics(total_patterns=-1)
        with pytest.raises(ValueError):
            StorageStatistics(total_symbols=-1)
        with pytest.raises(ValueError):
            StorageStatistics(compression_ratio=-0.1)
        with pytest.raises(ValueError):
            StorageStatistics(cache_hit_ratio=1.5)
    def test_storage_statistics_calculated_properties(self) -> None:
        """Тест вычисляемых свойств StorageStatistics."""
        stats = StorageStatistics(
            total_patterns=100,
            total_successful_patterns=80
        )
        # Вычисляем успешность
        success_rate = stats.total_successful_patterns / stats.total_patterns
        assert success_rate == 0.8
    def test_storage_statistics_to_dict(self) -> None:
        """Тест преобразования в словарь."""
        stats = StorageStatistics(
            total_patterns=100,
            total_symbols=10,
            total_successful_patterns=80,
            total_storage_size_bytes=1024000,
            avg_pattern_size_bytes=1024,
            compression_ratio=0.7,
            cache_hit_ratio=0.85,
            avg_read_time_ms=5.0,
            avg_write_time_ms=10.0,
            error_count=2,
            warning_count=5
        )
        stats_dict = stats.to_dict()
        assert isinstance(stats_dict, dict)
        assert stats_dict["total_patterns"] == 100
        assert stats_dict["total_symbols"] == 10
        assert stats_dict["total_successful_patterns"] == 80
        assert stats_dict["total_storage_size_bytes"] == 1024000
        assert stats_dict["avg_pattern_size_bytes"] == 1024
        assert stats_dict["compression_ratio"] == 0.7
        assert stats_dict["cache_hit_ratio"] == 0.85
        assert stats_dict["avg_read_time_ms"] == 5.0
        assert stats_dict["avg_write_time_ms"] == 10.0
        assert stats_dict["error_count"] == 2
        assert stats_dict["warning_count"] == 5
class TestPatternMetadata:
    """Тесты для PatternMetadata."""
    def test_pattern_metadata_creation(self) -> None:
        """Тест создания PatternMetadata."""
        metadata = PatternMetadata(
            symbol="BTCUSDT",
            pattern_type="accumulation",
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            total_count=100,
            success_count=80,
            avg_accuracy=0.85,
            avg_return=0.02,
            avg_confidence=0.8,
            success_rate=0.8,
            trend_direction="up",
            trend_strength=0.7,
            market_phase="trending",
            volatility_regime="medium",
            liquidity_regime="high"
        )
        assert metadata.symbol == "BTCUSDT"
        assert metadata.pattern_type == "accumulation"
        assert metadata.total_count == 100
        assert metadata.success_count == 80
        assert metadata.avg_accuracy == 0.85
        assert metadata.avg_return == 0.02
        assert metadata.avg_confidence == 0.8
        assert metadata.success_rate == 0.8
        assert metadata.trend_direction == "up"
        assert metadata.trend_strength == 0.7
        assert metadata.market_phase == "trending"
        assert metadata.volatility_regime == "medium"
        assert metadata.liquidity_regime == "high"
    def test_pattern_metadata_default_values(self) -> None:
        """Тест значений по умолчанию для PatternMetadata."""
        metadata = PatternMetadata(symbol="BTCUSDT", pattern_type="accumulation")
        assert metadata.symbol == "BTCUSDT"
        assert metadata.pattern_type == "accumulation"
        assert metadata.total_count == 0
        assert metadata.success_count == 0
        assert metadata.avg_accuracy == 0.0
        assert metadata.avg_return == 0.0
        assert metadata.avg_confidence == 0.0
        assert metadata.success_rate == 0.0
        assert metadata.trend_direction == "neutral"
        assert metadata.trend_strength == 0.0
        assert metadata.market_phase == "unknown"
        assert metadata.volatility_regime == "unknown"
        assert metadata.liquidity_regime == "unknown"
    def test_pattern_metadata_validation(self) -> None:
        """Тест валидации PatternMetadata."""
        # Тест с пустым символом
        with pytest.raises(ValueError):
            PatternMetadata(symbol="", pattern_type="accumulation")
        # Тест с пустым типом паттерна
        with pytest.raises(ValueError):
            PatternMetadata(symbol="BTCUSDT", pattern_type="")
        # Тест с некорректными значениями
        with pytest.raises(ValueError):
            PatternMetadata(
                symbol="BTCUSDT",
                pattern_type="accumulation",
                success_rate=1.5
            )
    def test_pattern_metadata_to_dict(self) -> None:
        """Тест преобразования в словарь."""
        metadata = PatternMetadata(
            symbol="BTCUSDT",
            pattern_type="accumulation",
            total_count=100,
            success_count=80,
            avg_accuracy=0.85,
            avg_return=0.02,
            avg_confidence=0.8,
            success_rate=0.8
        )
        metadata_dict = metadata.to_dict()
        assert isinstance(metadata_dict, dict)
        assert metadata_dict["symbol"] == "BTCUSDT"
        assert metadata_dict["pattern_type"] == "accumulation"
        assert metadata_dict["total_count"] == 100
        assert metadata_dict["success_count"] == 80
        assert metadata_dict["avg_accuracy"] == 0.85
        assert metadata_dict["avg_return"] == 0.02
        assert metadata_dict["avg_confidence"] == 0.8
        assert metadata_dict["success_rate"] == 0.8
class TestBehaviorRecord:
    """Тесты для BehaviorRecord."""
    def test_behavior_record_creation(self) -> None:
        """Тест создания BehaviorRecord."""
        record = BehaviorRecord(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            pattern_type="accumulation",
            volume=1000.0,
            spread=0.001,
            imbalance=0.3,
            pressure=0.4,
            confidence=0.8,
            market_phase="trending",
            volatility_regime="medium",
            liquidity_regime="high",
            context={"session": "asian", "market_regime": "trending"}
        )
        assert record.symbol == "BTCUSDT"
        assert record.pattern_type == "accumulation"
        assert record.volume == 1000.0
        assert record.spread == 0.001
        assert record.imbalance == 0.3
        assert record.pressure == 0.4
        assert record.confidence == 0.8
        assert record.market_phase == "trending"
        assert record.volatility_regime == "medium"
        assert record.liquidity_regime == "high"
        assert record.context["session"] == "asian"
    def test_behavior_record_default_values(self) -> None:
        """Тест значений по умолчанию для BehaviorRecord."""
        record = BehaviorRecord(symbol="BTCUSDT", timestamp=datetime.now())
        assert record.symbol == "BTCUSDT"
        assert record.pattern_type == "unknown"
        assert record.volume == 0.0
        assert record.spread == 0.0
        assert record.imbalance == 0.0
        assert record.pressure == 0.0
        assert record.confidence == 0.0
        assert record.market_phase == "unknown"
        assert record.volatility_regime == "unknown"
        assert record.liquidity_regime == "unknown"
        assert record.context == {}
    def test_behavior_record_validation(self) -> None:
        """Тест валидации BehaviorRecord."""
        # Тест с пустым символом
        with pytest.raises(ValueError):
            BehaviorRecord(symbol="", timestamp=datetime.now())
        # Тест с некорректными значениями
        with pytest.raises(ValueError):
            BehaviorRecord(
                symbol="BTCUSDT",
                timestamp=datetime.now(),
                confidence=1.5
            )
    def test_behavior_record_to_dict(self) -> None:
        """Тест преобразования в словарь."""
        record = BehaviorRecord(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            pattern_type="accumulation",
            volume=1000.0,
            spread=0.001,
            imbalance=0.3,
            pressure=0.4,
            confidence=0.8
        )
        record_dict = record.to_dict()
        assert isinstance(record_dict, dict)
        assert record_dict["symbol"] == "BTCUSDT"
        assert record_dict["pattern_type"] == "accumulation"
        assert record_dict["volume"] == 1000.0
        assert record_dict["spread"] == 0.001
        assert record_dict["imbalance"] == 0.3
        assert record_dict["pressure"] == 0.4
        assert record_dict["confidence"] == 0.8
class TestSuccessMapEntry:
    """Тесты для SuccessMapEntry."""
    def test_success_map_entry_creation(self) -> None:
        """Тест создания SuccessMapEntry."""
        entry = SuccessMapEntry(
            pattern_type="accumulation",
            success_rate=0.8,
            avg_return=0.02,
            avg_accuracy=0.85,
            confidence=0.9,
            sample_size=100,
            last_updated=datetime.now()
        )
        assert entry.pattern_type == "accumulation"
        assert entry.success_rate == 0.8
        assert entry.avg_return == 0.02
        assert entry.avg_accuracy == 0.85
        assert entry.confidence == 0.9
        assert entry.sample_size == 100
    def test_success_map_entry_default_values(self) -> None:
        """Тест значений по умолчанию для SuccessMapEntry."""
        entry = SuccessMapEntry(pattern_type="accumulation")
        assert entry.pattern_type == "accumulation"
        assert entry.success_rate == 0.0
        assert entry.avg_return == 0.0
        assert entry.avg_accuracy == 0.0
        assert entry.confidence == 0.0
        assert entry.sample_size == 0
    def test_success_map_entry_validation(self) -> None:
        """Тест валидации SuccessMapEntry."""
        # Тест с пустым типом паттерна
        with pytest.raises(ValueError):
            SuccessMapEntry(pattern_type="")
        # Тест с некорректными значениями
        with pytest.raises(ValueError):
            SuccessMapEntry(
                pattern_type="accumulation",
                success_rate=1.5
            )
    def test_success_map_entry_to_dict(self) -> None:
        """Тест преобразования в словарь."""
        entry = SuccessMapEntry(
            pattern_type="accumulation",
            success_rate=0.8,
            avg_return=0.02,
            avg_accuracy=0.85,
            confidence=0.9,
            sample_size=100
        )
        entry_dict = entry.to_dict()
        assert isinstance(entry_dict, dict)
        assert entry_dict["pattern_type"] == "accumulation"
        assert entry_dict["success_rate"] == 0.8
        assert entry_dict["avg_return"] == 0.02
        assert entry_dict["avg_accuracy"] == 0.85
        assert entry_dict["confidence"] == 0.9
        assert entry_dict["sample_size"] == 100
class TestStorageStatus:
    """Тесты для StorageStatus."""
    def test_storage_status_values(self) -> None:
        """Тест значений StorageStatus."""
        assert StorageStatus.ACTIVE.value == "active"
        assert StorageStatus.MAINTENANCE.value == "maintenance"
        assert StorageStatus.ERROR.value == "error"
        assert StorageStatus.READONLY.value == "readonly"
    def test_storage_status_enumeration(self) -> None:
        """Тест перечисления StorageStatus."""
        statuses = list(StorageStatus)
        assert len(statuses) == 4
        assert StorageStatus.ACTIVE in statuses
        assert StorageStatus.MAINTENANCE in statuses
        assert StorageStatus.ERROR in statuses
        assert StorageStatus.READONLY in statuses
if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
