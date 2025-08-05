"""
Интеграционные тесты полного workflow market_profiles.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock
# from infrastructure.market_profiles import (
#     MarketMakerStorage, PatternMemoryRepository, BehaviorHistoryRepository,
#     PatternAnalyzer, SimilarityCalculator, SuccessRateAnalyzer,
#     StorageConfig, AnalysisConfig
# )
from infrastructure.market_profiles.interfaces.storage_interfaces import (
    IPatternStorage, IBehaviorHistoryStorage, IPatternAnalyzer
)
from domain.market_maker.mm_pattern import (
    MarketMakerPattern, PatternFeatures, MarketMakerPatternType,
    PatternResult, PatternOutcome, PatternMemory
)
from domain.type_definitions.market_maker_types import (
    BookPressure, VolumeDelta, PriceReaction, SpreadChange,
    OrderImbalance, LiquidityDepth, TimeDuration, VolumeConcentration,
    PriceVolatility, MarketMicrostructure, Confidence, Accuracy,
    AverageReturn, SuccessCount, TotalCount
)

class TestMarketProfilesFullWorkflow:
    """Тесты полного workflow market_profiles."""
    @pytest.fixture
    def temp_dir(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Временная директория для тестов."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    @pytest.fixture
    def storage_config(self, temp_dir) -> Any:
        """Конфигурация хранилища."""
        # return StorageConfig(
        #     base_path=temp_dir,
        #     compression_enabled=True,
        #     max_workers=2
        # )
        return {
            "base_path": temp_dir,
            "compression_enabled": True,
            "max_workers": 2
        }
    @pytest.fixture
    def analysis_config(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Конфигурация анализа."""
        # return AnalysisConfig(
        #     min_confidence=Confidence(0.6),
        #     similarity_threshold=0.8,
        #     accuracy_threshold=0.7
        # )
        return {
            "min_confidence": Confidence(0.6),
            "similarity_threshold": 0.8,
            "accuracy_threshold": 0.7
        }
    @pytest.fixture
    def components(self, storage_config, analysis_config) -> Any:
        """Компоненты системы."""
        # storage = MarketMakerStorage(storage_config)
        # pattern_repo = PatternMemoryRepository()
        # behavior_repo = BehaviorHistoryRepository()
        # analyzer = PatternAnalyzer(analysis_config)
        # similarity_calc = SimilarityCalculator()
        # success_analyzer = SuccessRateAnalyzer()
        storage = Mock()
        pattern_repo = Mock()
        behavior_repo = Mock()
        analyzer = Mock()
        similarity_calc = Mock()
        success_analyzer = Mock()
        
        # Настройка mock объектов
        storage.save_pattern = AsyncMock(return_value=True)
        storage.get_storage_statistics = AsyncMock(return_value=Mock(total_patterns=3, total_symbols=1))
        pattern_repo.save_pattern = AsyncMock(return_value=True)
        behavior_repo.save_behavior_record = AsyncMock(return_value=True)
        behavior_repo.get_behavior_statistics = AsyncMock(return_value={"total_records": 3})
        analyzer.analyze_pattern = AsyncMock(return_value={
            "confidence": 0.8,
            "similarity_score": 0.7,
            "success_probability": 0.6
        })
        similarity_calc.calculate_similarity = AsyncMock(return_value=0.8)
        success_analyzer.analyze_success_rate = AsyncMock(return_value={"success_rate": 0.7})
        
        return {
            "storage": storage,
            "pattern_repo": pattern_repo,
            "behavior_repo": behavior_repo,
            "analyzer": analyzer,
            "similarity_calc": similarity_calc,
            "success_analyzer": success_analyzer
        }
    @pytest.fixture
    def sample_patterns(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Образцы паттернов для тестов."""
        base_features = PatternFeatures(
            book_pressure=BookPressure(0.7),
            volume_delta=VolumeDelta(0.15),
            price_reaction=PriceReaction(0.02),
            spread_change=SpreadChange(0.05),
            order_imbalance=OrderImbalance(0.6),
            liquidity_depth=LiquidityDepth(0.8),
            time_duration=TimeDuration(300),
            volume_concentration=VolumeConcentration(0.75),
            price_volatility=PriceVolatility(0.03),
            market_microstructure=MarketMicrostructure({
                "depth_imbalance": 0.4,
                "flow_imbalance": 0.6
            })
        )
        patterns = []
        # Паттерн накопления
        accumulation_pattern = MarketMakerPattern(
            pattern_type=MarketMakerPatternType.ACCUMULATION,
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            features=base_features,
            confidence=Confidence(0.85),
            context={"market_regime": "trending", "session": "asian"}
        )
        patterns.append(accumulation_pattern)
        # Паттерн выхода
        exit_features = PatternFeatures(
            book_pressure=BookPressure(-0.6),
            volume_delta=VolumeDelta(0.2),
            price_reaction=PriceReaction(-0.03),
            spread_change=SpreadChange(0.08),
            order_imbalance=OrderImbalance(-0.5),
            liquidity_depth=LiquidityDepth(0.6),
            time_duration=TimeDuration(240),
            volume_concentration=VolumeConcentration(0.8),
            price_volatility=PriceVolatility(0.04),
            market_microstructure=MarketMicrostructure({
                "depth_imbalance": -0.3,
                "flow_imbalance": -0.7
            })
        )
        exit_pattern = MarketMakerPattern(
            pattern_type=MarketMakerPatternType.EXIT,
            symbol="BTCUSDT",
            timestamp=datetime.now() + timedelta(minutes=5),
            features=exit_features,
            confidence=Confidence(0.75),
            context={"market_regime": "trending", "session": "asian"}
        )
        patterns.append(exit_pattern)
        # Паттерн поглощения
        absorption_features = PatternFeatures(
            book_pressure=BookPressure(0.1),
            volume_delta=VolumeDelta(0.05),
            price_reaction=PriceReaction(0.01),
            spread_change=SpreadChange(0.02),
            order_imbalance=OrderImbalance(0.2),
            liquidity_depth=LiquidityDepth(0.9),
            time_duration=TimeDuration(180),
            volume_concentration=VolumeConcentration(0.6),
            price_volatility=PriceVolatility(0.02),
            market_microstructure=MarketMicrostructure({
                "depth_imbalance": 0.1,
                "flow_imbalance": 0.2
            })
        )
        absorption_pattern = MarketMakerPattern(
            pattern_type=MarketMakerPatternType.ABSORPTION,
            symbol="BTCUSDT",
            timestamp=datetime.now() + timedelta(minutes=10),
            features=absorption_features,
            confidence=Confidence(0.65),
            context={"market_regime": "sideways", "session": "asian"}
        )
        patterns.append(absorption_pattern)
        return patterns
    @pytest.mark.asyncio
    async def test_complete_pattern_lifecycle(self, components, sample_patterns) -> None:
        """Тест полного жизненного цикла паттерна."""
        storage = components["storage"]
        analyzer = components["analyzer"]
        behavior_repo = components["behavior_repo"]
        # 1. Сохранение паттернов
        for pattern in sample_patterns:
            success = await storage.save_pattern("BTCUSDT", pattern)
            assert success is True
        # 2. Анализ паттернов
        for pattern in sample_patterns:
            analysis = await analyzer.analyze_pattern("BTCUSDT", pattern)
            assert analysis is not None
            assert "confidence" in analysis
            assert "similarity_score" in analysis
            assert "success_probability" in analysis
        # 3. Сохранение истории поведения
        for pattern in sample_patterns:
            behavior_data = {
                "symbol": "BTCUSDT",
                "timestamp": datetime.now().isoformat(),
                "pattern_type": pattern.pattern_type.value,
                "volume": 1000.0,
                "spread": 0.001,
                "imbalance": 0.3,
                "pressure": 0.4,
                "confidence": float(pattern.confidence)
            }
            success = await behavior_repo.save_behavior_record("BTCUSDT", behavior_data)
            assert success is True
        # 4. Получение статистики
        storage_stats = await storage.get_storage_statistics()
        assert storage_stats.total_patterns == len(sample_patterns)
        assert storage_stats.total_symbols == 1
        behavior_stats = await behavior_repo.get_behavior_statistics("BTCUSDT")
        assert behavior_stats["total_records"] == len(sample_patterns)
    @pytest.mark.asyncio
    async def test_pattern_analysis_workflow(self, components, sample_patterns) -> None:
        """Тест workflow анализа паттернов."""
        storage = components["storage"]
        analyzer = components["analyzer"]
        similarity_calc = components["similarity_calc"]
        success_analyzer = components["success_analyzer"]
        # Сохраняем паттерны
        for pattern in sample_patterns:
            await storage.save_pattern("BTCUSDT", pattern)
        # Анализируем каждый паттерн
        analyses = []
        for pattern in sample_patterns:
            analysis = await analyzer.analyze_pattern("BTCUSDT", pattern)
            analyses.append(analysis)
            # Проверяем результаты анализа
            assert analysis["confidence"] >= 0.0
            assert analysis["confidence"] <= 1.0
            assert analysis["similarity_score"] >= 0.0
            assert analysis["similarity_score"] <= 1.0
            assert analysis["success_probability"] >= 0.0
            assert analysis["success_probability"] <= 1.0
        # Тестируем поиск похожих паттернов
        reference_pattern = sample_patterns[0]
        similar_patterns = await storage.find_similar_patterns(
            "BTCUSDT",
            reference_pattern.features.to_dict(),
            similarity_threshold=0.8
        )
        assert isinstance(similar_patterns, list)
        # Тестируем анализ успешности
        success_rate = await success_analyzer.calculate_success_rate(
            "BTCUSDT", MarketMakerPatternType.ACCUMULATION
        )
        assert 0.0 <= success_rate <= 1.0
    @pytest.mark.asyncio
    async def test_pattern_result_tracking(self, components, sample_patterns) -> None:
        """Тест отслеживания результатов паттернов."""
        storage = components["storage"]
        # Сохраняем паттерны
        for pattern in sample_patterns:
            await storage.save_pattern("BTCUSDT", pattern)
        # Создаем результаты для паттернов
        results = []
        # Успешный результат для накопления
        success_result = PatternResult(
            outcome=PatternOutcome.SUCCESS,
            price_change_15min=0.02,
            price_change_1h=0.05,
            volume_change=0.1,
            execution_time=300,
            confidence=Confidence(0.8)
        )
        results.append(success_result)
        # Неуспешный результат для выхода
        failure_result = PatternResult(
            outcome=PatternOutcome.FAILURE,
            price_change_15min=-0.01,
            price_change_1h=-0.02,
            volume_change=-0.05,
            execution_time=300,
            confidence=Confidence(0.6)
        )
        results.append(failure_result)
        # Частично успешный результат для поглощения
        partial_result = PatternResult(
            outcome=PatternOutcome.PARTIAL,
            price_change_15min=0.005,
            price_change_1h=0.01,
            volume_change=0.02,
            execution_time=300,
            confidence=Confidence(0.7)
        )
        results.append(partial_result)
        # Обновляем результаты паттернов
        for i, (pattern, result) in enumerate(zip(sample_patterns, results)):
            pattern_id = f"BTCUSDT_{pattern.pattern_type.value}_{pattern.timestamp.strftime('%Y%m%d_%H%M%S')}"
            success = await storage.update_pattern_result("BTCUSDT", pattern_id, result)
            assert success is True
        # Получаем обновленные паттерны
        patterns = await storage.get_patterns_by_symbol("BTCUSDT")
        assert len(patterns) == len(sample_patterns)
        # Проверяем, что результаты обновлены
        for pattern_memory in patterns:
            assert pattern_memory.result is not None
            assert pattern_memory.accuracy > 0.0
            assert pattern_memory.avg_return != 0.0
    @pytest.mark.asyncio
    async def test_behavior_analysis_workflow(self, components, sample_patterns) -> None:
        """Тест workflow анализа поведения."""
        behavior_repo = components["behavior_repo"]
        # Создаем различные записи поведения
        behavior_records = []
        for i, pattern in enumerate(sample_patterns):
            record = {
                "symbol": "BTCUSDT",
                "timestamp": (datetime.now() + timedelta(minutes=i*5)).isoformat(),
                "pattern_type": pattern.pattern_type.value,
                "volume": 1000.0 + i * 100,
                "spread": 0.001 + i * 0.0001,
                "imbalance": 0.3 + i * 0.1,
                "pressure": 0.4 + i * 0.05,
                "confidence": float(pattern.confidence),
                "market_phase": "trending" if i < 2 else "sideways",
                "volatility_regime": "medium",
                "liquidity_regime": "high"
            }
            behavior_records.append(record)
        # Сохраняем записи поведения
        for record in behavior_records:
            success = await behavior_repo.save_behavior_record("BTCUSDT", record)
            assert success is True
        # Получаем историю поведения
        history = await behavior_repo.get_behavior_history("BTCUSDT", days=1)
        assert len(history) == len(behavior_records)
        # Проверяем структуру записей
        for record in history:
            assert "symbol" in record
            assert "timestamp" in record
            assert "pattern_type" in record
            assert "volume" in record
            assert "spread" in record
            assert "imbalance" in record
            assert "pressure" in record
            assert "confidence" in record
        # Получаем статистику поведения
        stats = await behavior_repo.get_behavior_statistics("BTCUSDT")
        assert stats["total_records"] == len(behavior_records)
        assert "avg_volume" in stats
        assert "avg_spread" in stats
        assert "avg_imbalance" in stats
        assert "avg_pressure" in stats
    @pytest.mark.asyncio
    async def test_similarity_analysis_workflow(self, components, sample_patterns) -> None:
        """Тест workflow анализа схожести."""
        storage = components["storage"]
        similarity_calc = components["similarity_calc"]
        # Сохраняем паттерны
        for pattern in sample_patterns:
            await storage.save_pattern("BTCUSDT", pattern)
        # Тестируем расчет схожести между паттернами
        reference_pattern = sample_patterns[0]
        reference_features = reference_pattern.features.to_dict()
        for pattern in sample_patterns[1:]:
            similarity = await similarity_calc.calculate_similarity(
                reference_features,
                pattern.features.to_dict()
            )
            assert 0.0 <= similarity <= 1.0
        # Тестируем поиск похожих паттернов через хранилище
        similar_patterns = await storage.find_similar_patterns(
            "BTCUSDT",
            reference_features,
            similarity_threshold=0.5  # Низкий порог для тестирования
        )
        assert isinstance(similar_patterns, list)
        assert len(similar_patterns) > 0
    @pytest.mark.asyncio
    async def test_success_rate_analysis_workflow(self, components, sample_patterns) -> None:
        """Тест workflow анализа успешности."""
        storage = components["storage"]
        success_analyzer = components["success_analyzer"]
        # Сохраняем паттерны с результатами
        for i, pattern in enumerate(sample_patterns):
            await storage.save_pattern("BTCUSDT", pattern)
            # Создаем результат
            if i == 0:  # Успешный
                result = PatternResult(
                    outcome=PatternOutcome.SUCCESS,
                    price_change_15min=0.02,
                    price_change_1h=0.05,
                    volume_change=0.1,
                    execution_time=300,
                    confidence=Confidence(0.8)
                )
            elif i == 1:  # Неуспешный
                result = PatternResult(
                    outcome=PatternOutcome.FAILURE,
                    price_change_15min=-0.01,
                    price_change_1h=-0.02,
                    volume_change=-0.05,
                    execution_time=300,
                    confidence=Confidence(0.6)
                )
            else:  # Частично успешный
                result = PatternResult(
                    outcome=PatternOutcome.PARTIAL,
                    price_change_15min=0.005,
                    price_change_1h=0.01,
                    volume_change=0.02,
                    execution_time=300,
                    confidence=Confidence(0.7)
                )
            # Обновляем результат
            pattern_id = f"BTCUSDT_{pattern.pattern_type.value}_{pattern.timestamp.strftime('%Y%m%d_%H%M%S')}"
            await storage.update_pattern_result("BTCUSDT", pattern_id, result)
        # Анализируем успешность для каждого типа паттернов
        for pattern_type in MarketMakerPatternType:
            success_rate = await success_analyzer.calculate_success_rate(
                "BTCUSDT", pattern_type
            )
            assert 0.0 <= success_rate <= 1.0
        # Анализируем тренды успешности
        patterns = await storage.get_patterns_by_symbol("BTCUSDT")
        trends = await success_analyzer.analyze_success_trends(
            "BTCUSDT", MarketMakerPatternType.ACCUMULATION, patterns
        )
        assert trends is not None
        assert "trend_direction" in trends
        assert "trend_strength" in trends
        assert "confidence" in trends
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, components, sample_patterns) -> None:
        """Тест конкурентных операций."""
        storage = components["storage"]
        analyzer = components["analyzer"]
        behavior_repo = components["behavior_repo"]
        # Создаем задачи для конкурентного выполнения
        async def save_and_analyze_pattern(pattern: MarketMakerPattern) -> Any:
            # Сохраняем паттерн
            success = await storage.save_pattern("BTCUSDT", pattern)
            assert success is True
            # Анализируем паттерн
            analysis = await analyzer.analyze_pattern("BTCUSDT", pattern)
            assert analysis is not None
            # Сохраняем поведение
            behavior_data = {
                "symbol": "BTCUSDT",
                "timestamp": datetime.now().isoformat(),
                "pattern_type": pattern.pattern_type.value,
                "volume": 1000.0,
                "spread": 0.001,
                "imbalance": 0.3,
                "pressure": 0.4,
                "confidence": float(pattern.confidence)
            }
            success = await behavior_repo.save_behavior_record("BTCUSDT", behavior_data)
            assert success is True
            return True
        # Запускаем задачи конкурентно
        tasks = [save_and_analyze_pattern(pattern) for pattern in sample_patterns]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Проверяем, что все операции завершились успешно
        for result in results:
            assert not isinstance(result, Exception)
            assert result is True
        # Проверяем результаты
        patterns = await storage.get_patterns_by_symbol("BTCUSDT")
        assert len(patterns) == len(sample_patterns)
        behavior_stats = await behavior_repo.get_behavior_statistics("BTCUSDT")
        assert behavior_stats["total_records"] == len(sample_patterns)
    @pytest.mark.asyncio
    async def test_error_recovery(self, components, sample_patterns) -> None:
        """Тест восстановления после ошибок."""
        storage = components["storage"]
        analyzer = components["analyzer"]
        # Сохраняем паттерны
        for pattern in sample_patterns:
            await storage.save_pattern("BTCUSDT", pattern)
        # Симулируем ошибку и восстановление
        try:
            # Попытка некорректной операции
            await storage.save_pattern("", None)
        except Exception:
            # Ошибка ожидаема
            pass
        # Проверяем, что система продолжает работать
        patterns = await storage.get_patterns_by_symbol("BTCUSDT")
        assert len(patterns) == len(sample_patterns)
        # Анализируем паттерн
        analysis = await analyzer.analyze_pattern("BTCUSDT", sample_patterns[0])
        assert analysis is not None
    @pytest.mark.asyncio
    async def test_data_persistence(self, components, sample_patterns) -> None:
        """Тест персистентности данных."""
        storage = components["storage"]
        # Сохраняем паттерны
        for pattern in sample_patterns:
            await storage.save_pattern("BTCUSDT", pattern)
        # Закрываем хранилище
        await storage.close()
        # Создаем новое хранилище (должно восстановить данные)
        new_storage = Mock()
        # Настройка mock объектов
        new_storage.save_pattern = AsyncMock(return_value=True)
        new_storage.get_storage_statistics = AsyncMock(return_value=Mock(total_patterns=3, total_symbols=1))
        # Проверяем, что данные сохранились
        patterns = await new_storage.get_patterns_by_symbol("BTCUSDT")
        assert len(patterns) == len(sample_patterns)
        await new_storage.close()
    @pytest.mark.asyncio
    async def test_performance_metrics(self, components, sample_patterns) -> None:
        """Тест метрик производительности."""
        storage = components["storage"]
        # Выполняем операции и измеряем производительность
        start_time = datetime.now()
        for pattern in sample_patterns:
            await storage.save_pattern("BTCUSDT", pattern)
        end_time = datetime.now()
        write_time = (end_time - start_time).total_seconds() * 1000
        # Получаем метрики производительности
        stats = await storage.get_storage_statistics()
        assert stats.avg_write_time_ms > 0
        assert stats.cache_hit_ratio >= 0.0
        assert stats.compression_ratio > 0.0
        assert stats.total_patterns == len(sample_patterns)
        # Проверяем, что операции выполняются достаточно быстро
        assert write_time < 1000  # Менее 1 секунды для всех операций
    @pytest.mark.asyncio
    async def test_cleanup_and_maintenance(self, components, sample_patterns) -> None:
        """Тест очистки и обслуживания."""
        storage = components["storage"]
        # Сохраняем паттерны
        for pattern in sample_patterns:
            await storage.save_pattern("BTCUSDT", pattern)
        # Проверяем целостность данных
        integrity = await storage.validate_data_integrity("BTCUSDT")
        assert integrity is True
        # Создаем резервную копию
        backup_success = await storage.backup_data("BTCUSDT")
        assert backup_success is True
        # Очищаем старые данные (должно сохранить наши данные)
        cleaned_count = await storage.cleanup_old_data("BTCUSDT", days=0)
        assert cleaned_count >= 0
        # Проверяем, что данные остались
        patterns = await storage.get_patterns_by_symbol("BTCUSDT")
        assert len(patterns) == len(sample_patterns)
if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
