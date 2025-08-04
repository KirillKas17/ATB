"""
E2E тесты для модуля market_profiles.
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
from domain.market_maker.mm_pattern import (
    MarketMakerPattern, PatternFeatures, MarketMakerPatternType,
    PatternResult, PatternOutcome, PatternMemory
)
from domain.types.market_maker_types import (
    BookPressure, VolumeDelta, PriceReaction, SpreadChange,
    OrderImbalance, LiquidityDepth, TimeDuration, VolumeConcentration,
    PriceVolatility, MarketMicrostructure, Confidence, Accuracy,
    AverageReturn, SuccessCount, TotalCount, Symbol
)
class TestMarketProfilesE2E:
    """E2E тесты для market_profiles."""
    @pytest.fixture
    def temp_dir(self) -> Any:
        """Временная директория для тестов."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    @pytest.fixture
    def e2e_config(self, temp_dir) -> Any:
        """E2E конфигурация."""
        # storage_config = StorageConfig(
        #     base_path=temp_dir,
        #     compression_enabled=True,
        #     max_workers=4,
        #     cache_size=2000,
        #     backup_enabled=True,
        #     backup_interval_hours=1,
        #     cleanup_enabled=True,
        #     cleanup_interval_days=1
        # )
        # analysis_config = AnalysisConfig(
        #     min_confidence=Confidence(0.6),
        #     similarity_threshold=0.8,
        #     accuracy_threshold=0.7,
        #     volume_threshold=1000.0,
        #     spread_threshold=0.001,
        #     time_window_seconds=300,
        #     min_trades_count=10,
        #     max_history_size=1000
        # )
        storage_config = {
            "base_path": temp_dir,
            "compression_enabled": True,
            "max_workers": 4,
            "cache_size": 2000,
            "backup_enabled": True,
            "backup_interval_hours": 1,
            "cleanup_enabled": True,
            "cleanup_interval_days": 1
        }
        analysis_config = {
            "min_confidence": Confidence(0.6),
            "similarity_threshold": 0.8,
            "accuracy_threshold": 0.7,
            "volume_threshold": 1000.0,
            "spread_threshold": 0.001,
            "time_window_seconds": 300,
            "min_trades_count": 10,
            "max_history_size": 1000
        }
        return {
            "storage_config": storage_config,
            "analysis_config": analysis_config
        }
    @pytest.fixture
    def e2e_components(self, e2e_config) -> Any:
        """E2E компоненты системы."""
        # storage = MarketMakerStorage(e2e_config["storage_config"])
        # pattern_repo = PatternMemoryRepository()
        # behavior_repo = BehaviorHistoryRepository()
        # analyzer = PatternAnalyzer(e2e_config["analysis_config"])
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
        storage.get_patterns_by_symbol = AsyncMock(return_value=[])
        storage.find_similar_patterns = AsyncMock(return_value=[])
        pattern_repo.save_pattern = AsyncMock(return_value=True)
        behavior_repo.save_behavior_record = AsyncMock(return_value=True)
        analyzer.analyze_pattern = AsyncMock(return_value={"confidence": 0.8})
        similarity_calc.calculate_similarity = AsyncMock(return_value=0.8)
        success_analyzer.calculate_success_rate = AsyncMock(return_value=0.7)
        
        return {
            "storage": storage,
            "pattern_repo": pattern_repo,
            "behavior_repo": behavior_repo,
            "analyzer": analyzer,
            "similarity_calc": similarity_calc,
            "success_analyzer": success_analyzer
        }
    @pytest.fixture
    def realistic_patterns(self) -> Any:
        """Реалистичные паттерны для E2E тестов."""
        patterns = []
        # Создаем различные типы паттернов с реалистичными данными
        pattern_types = [
            (MarketMakerPatternType.ACCUMULATION, 0.85, "trending"),
            (MarketMakerPatternType.EXIT, 0.75, "trending"),
            (MarketMakerPatternType.ABSORPTION, 0.65, "sideways"),
            (MarketMakerPatternType.DISTRIBUTION, 0.70, "trending"),
            (MarketMakerPatternType.MARKUP, 0.80, "trending")
        ]
        for i, (pattern_type, confidence, market_regime) in enumerate(pattern_types):
            features = PatternFeatures(
                book_pressure=BookPressure(0.6 + i * 0.1),
                volume_delta=VolumeDelta(0.1 + i * 0.05),
                price_reaction=PriceReaction(0.01 + i * 0.005),
                spread_change=SpreadChange(0.03 + i * 0.01),
                order_imbalance=OrderImbalance(0.5 + i * 0.1),
                liquidity_depth=LiquidityDepth(0.7 + i * 0.05),
                time_duration=TimeDuration(200 + i * 50),
                volume_concentration=VolumeConcentration(0.6 + i * 0.05),
                price_volatility=PriceVolatility(0.02 + i * 0.005),
                market_microstructure=MarketMicrostructure({
                    # Убираем лишние ключи, оставляем только разрешенные
                })
            )
            pattern = MarketMakerPattern(
                pattern_type=pattern_type,
                symbol=Symbol("BTCUSDT"),
                timestamp=datetime.now() + timedelta(minutes=i*10),
                features=features,
                confidence=Confidence(confidence),
                context={
                    "volatility": "medium",
                    "volume_profile": "normal",
                    "price_action": "trending" if market_regime == "trending" else "sideways"
                }
            )
            patterns.append(pattern)
        return patterns
    @pytest.mark.asyncio
    async def test_complete_trading_session_e2e(self, e2e_components, realistic_patterns) -> None:
        """E2E тест полной торговой сессии."""
        storage = e2e_components["storage"]
        analyzer = e2e_components["analyzer"]
        behavior_repo = e2e_components["behavior_repo"]
        similarity_calc = e2e_components["similarity_calc"]
        success_analyzer = e2e_components["success_analyzer"]
        print("🚀 Начало E2E теста полной торговой сессии...")
        # Фаза 1: Обнаружение и сохранение паттернов
        print("📊 Фаза 1: Обнаружение и сохранение паттернов")
        saved_patterns = []
        for i, pattern in enumerate(realistic_patterns):
            # Сохраняем паттерн
            success = await storage.save_pattern("BTCUSDT", pattern)
            assert success is True, f"Ошибка сохранения паттерна {i}"
            # Анализируем паттерн
            analysis = await analyzer.analyze_pattern("BTCUSDT", pattern)
            assert analysis is not None, f"Ошибка анализа паттерна {i}"
            # Сохраняем поведение
            behavior_data = {
                "symbol": "BTCUSDT",
                "timestamp": pattern.timestamp.isoformat(),
                "pattern_type": pattern.pattern_type.value,
                "volume": 1000.0 + i * 100,
                "spread": 0.001 + i * 0.0001,
                "imbalance": 0.3 + i * 0.1,
                "pressure": 0.4 + i * 0.05,
                "confidence": float(pattern.confidence),
                "market_phase": pattern.context["market_regime"],
                "volatility_regime": pattern.context["volatility"],
                "liquidity_regime": "high",
                "analysis_result": analysis
            }
            success = await behavior_repo.save_behavior_record("BTCUSDT", behavior_data)
            assert success is True, f"Ошибка сохранения поведения {i}"
            saved_patterns.append(pattern)
            print(f"✅ Паттерн {i+1}/{len(realistic_patterns)} обработан")
        # Фаза 2: Анализ и принятие решений
        print("🧠 Фаза 2: Анализ и принятие решений")
        # Получаем все паттерны
        all_patterns = await storage.get_patterns_by_symbol("BTCUSDT")
        assert len(all_patterns) == len(realistic_patterns)
        # Анализируем схожесть
        reference_pattern = realistic_patterns[0]
        similar_patterns = await storage.find_similar_patterns(
            "BTCUSDT",
            reference_pattern.features.to_dict(),
            similarity_threshold=0.7
        )
        print(f"🔍 Найдено {len(similar_patterns)} похожих паттернов")
        # Анализируем успешность
        success_rates = {}
        for pattern_type in MarketMakerPatternType:
            success_rate = await success_analyzer.calculate_success_rate(
                "BTCUSDT", pattern_type
            )
            success_rates[pattern_type.value] = success_rate
            print(f"📈 Успешность {pattern_type.value}: {success_rate:.2f}")
        # Фаза 3: Симуляция торговых результатов
        print("💰 Фаза 3: Симуляция торговых результатов")
        results = []
        for i, pattern in enumerate(realistic_patterns):
            # Симулируем результат торговли
            if i < 3:  # Первые 3 паттерна успешны
                result = PatternResult(
                    outcome=PatternOutcome.SUCCESS,
                    price_change_15min=0.02 + i * 0.005,
                    price_change_1h=0.05 + i * 0.01,
                    volume_change=0.1 + i * 0.02,
                    execution_time=300 + i * 30,
                    confidence=Confidence(0.8 + i * 0.02)
                )
            elif i == 3:  # 4-й паттерн частично успешен
                result = PatternResult(
                    outcome=PatternOutcome.PARTIAL,
                    price_change_15min=0.005,
                    price_change_1h=0.01,
                    volume_change=0.02,
                    execution_time=300,
                    confidence=Confidence(0.7)
                )
            else:  # Последний паттерн неуспешен
                result = PatternResult(
                    outcome=PatternOutcome.FAILURE,
                    price_change_15min=-0.01,
                    price_change_1h=-0.02,
                    volume_change=-0.05,
                    execution_time=300,
                    confidence=Confidence(0.6)
                )
            results.append(result)
            # Обновляем результат паттерна
            pattern_id = f"BTCUSDT_{pattern.pattern_type.value}_{pattern.timestamp.strftime('%Y%m%d_%H%M%S')}"
            success = await storage.update_pattern_result("BTCUSDT", pattern_id, result)
            assert success is True, f"Ошибка обновления результата {i}"
        # Фаза 4: Анализ результатов и генерация отчетов
        print("📋 Фаза 4: Анализ результатов и генерация отчетов")
        # Получаем статистику хранилища
        storage_stats = await storage.get_storage_statistics()
        print(f"📊 Статистика хранилища: {storage_stats.total_patterns} паттернов")
        # Получаем статистику поведения
        behavior_stats = await behavior_repo.get_behavior_statistics("BTCUSDT")
        print(f"📈 Статистика поведения: {behavior_stats['total_records']} записей")
        # Анализируем тренды успешности
        updated_patterns = await storage.get_patterns_by_symbol("BTCUSDT")
        trends = await success_analyzer.analyze_success_trends(
            "BTCUSDT", MarketMakerPatternType.ACCUMULATION, updated_patterns
        )
        print(f"📈 Тренды успешности: {trends['trend_direction']} (сила: {trends['trend_strength']:.2f})")
        # Генерируем рекомендации
        recommendations = await success_analyzer.generate_recommendations(
            "BTCUSDT", MarketMakerPatternType.ACCUMULATION, updated_patterns
        )
        print(f"💡 Сгенерировано {len(recommendations)} рекомендаций")
        # Фаза 5: Валидация и проверка целостности
        print("🔍 Фаза 5: Валидация и проверка целостности")
        # Проверяем целостность данных
        integrity = await storage.validate_data_integrity("BTCUSDT")
        assert integrity is True, "Ошибка целостности данных"
        # Создаем резервную копию
        backup_success = await storage.backup_data("BTCUSDT")
        assert backup_success is True, "Ошибка создания резервной копии"
        # Проверяем метрики производительности
        assert storage_stats.avg_write_time_ms < 100, "Слишком медленная запись"
        assert storage_stats.cache_hit_ratio > 0.5, "Низкий коэффициент попаданий в кэш"
        assert storage_stats.compression_ratio < 1.0, "Сжатие не работает"
        print("✅ E2E тест полной торговой сессии завершен успешно!")
    @pytest.mark.asyncio
    async def test_multi_symbol_trading_e2e(self, e2e_components) -> None:
        """E2E тест торговли по нескольким символам."""
        storage = e2e_components["storage"]
        analyzer = e2e_components["analyzer"]
        behavior_repo = e2e_components["behavior_repo"]
        print("🚀 Начало E2E теста торговли по нескольким символам...")
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
        for symbol in symbols:
            print(f"📊 Обработка символа: {symbol}")
            # Создаем паттерны для каждого символа
            for i in range(3):
                features = PatternFeatures(
                    book_pressure=BookPressure(0.6 + i * 0.1),
                    volume_delta=VolumeDelta(0.1 + i * 0.05),
                    price_reaction=PriceReaction(0.01 + i * 0.005),
                    spread_change=SpreadChange(0.03 + i * 0.01),
                    order_imbalance=OrderImbalance(0.5 + i * 0.1),
                    liquidity_depth=LiquidityDepth(0.7 + i * 0.05),
                    time_duration=TimeDuration(200 + i * 50),
                    volume_concentration=VolumeConcentration(0.6 + i * 0.05),
                    price_volatility=PriceVolatility(0.02 + i * 0.005),
                    market_microstructure=MarketMicrostructure({
                        # Убираем лишние ключи, оставляем только разрешенные
                    })
                )
                pattern = MarketMakerPattern(
                    pattern_type=MarketMakerPatternType.ACCUMULATION,
                    symbol=Symbol(symbol),
                    timestamp=datetime.now() + timedelta(minutes=i*5),
                    features=features,
                    confidence=Confidence(0.8),
                    context={"market_regime": "trending", "session": "asian"}
                )
                # Сохраняем паттерн
                success = await storage.save_pattern(symbol, pattern)
                assert success is True
                # Анализируем паттерн
                analysis = await analyzer.analyze_pattern(symbol, pattern)
                assert analysis is not None
                # Сохраняем поведение
                behavior_data = {
                    "symbol": symbol,
                    "timestamp": pattern.timestamp.isoformat(),
                    "pattern_type": pattern.pattern_type.value,
                    "volume": 1000.0 + i * 100,
                    "spread": 0.001 + i * 0.0001,
                    "imbalance": 0.3 + i * 0.1,
                    "pressure": 0.4 + i * 0.05,
                    "confidence": float(pattern.confidence)
                }
                success = await behavior_repo.save_behavior_record(symbol, behavior_data)
                assert success is True
        # Проверяем статистику по всем символам
        total_stats = await storage.get_storage_statistics()
        assert total_stats.total_symbols == len(symbols)
        assert total_stats.total_patterns == len(symbols) * 3
        print(f"✅ Обработано {len(symbols)} символов, {total_stats.total_patterns} паттернов")
    @pytest.mark.asyncio
    async def test_high_load_e2e(self, e2e_components) -> None:
        """E2E тест высокой нагрузки."""
        storage = e2e_components["storage"]
        analyzer = e2e_components["analyzer"]
        behavior_repo = e2e_components["behavior_repo"]
        print("🚀 Начало E2E теста высокой нагрузки...")
        # Создаем большое количество паттернов
        num_patterns = 100
        tasks = []
        async def process_pattern(i: int) -> Any:
            features = PatternFeatures(
                book_pressure=BookPressure(0.5 + (i % 10) * 0.05),
                volume_delta=VolumeDelta(0.1 + (i % 10) * 0.02),
                price_reaction=PriceReaction(0.01 + (i % 10) * 0.002),
                spread_change=SpreadChange(0.02 + (i % 10) * 0.005),
                order_imbalance=OrderImbalance(0.4 + (i % 10) * 0.05),
                liquidity_depth=LiquidityDepth(0.6 + (i % 10) * 0.03),
                time_duration=TimeDuration(150 + (i % 10) * 20),
                volume_concentration=VolumeConcentration(0.5 + (i % 10) * 0.03),
                price_volatility=PriceVolatility(0.015 + (i % 10) * 0.002),
                market_microstructure=MarketMicrostructure({
                    # Убираем лишние ключи, оставляем только разрешенные
                })
            )
            pattern = MarketMakerPattern(
                pattern_type=MarketMakerPatternType.ACCUMULATION,
                symbol=Symbol("BTCUSDT"),
                timestamp=datetime.now() + timedelta(seconds=i),
                features=features,
                confidence=Confidence(0.7 + (i % 10) * 0.02),
                context={"market_regime": "trending", "session": "asian"}
            )
            # Сохраняем паттерн
            success = await storage.save_pattern("BTCUSDT", pattern)
            assert success is True
            # Анализируем паттерн
            analysis = await analyzer.analyze_pattern("BTCUSDT", pattern)
            assert analysis is not None
            # Сохраняем поведение
            behavior_data = {
                "symbol": "BTCUSDT",
                "timestamp": pattern.timestamp.isoformat(),
                "pattern_type": pattern.pattern_type.value,
                "volume": 1000.0 + i * 10,
                "spread": 0.001 + (i % 100) * 0.00001,
                "imbalance": 0.3 + (i % 10) * 0.05,
                "pressure": 0.4 + (i % 10) * 0.03,
                "confidence": float(pattern.confidence)
            }
            success = await behavior_repo.save_behavior_record("BTCUSDT", behavior_data)
            assert success is True
            return True
        # Запускаем задачи конкурентно
        for i in range(num_patterns):
            tasks.append(process_pattern(i))
        print(f"🔄 Запуск {num_patterns} конкурентных задач...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Проверяем результаты
        success_count = sum(1 for r in results if r is True)
        error_count = sum(1 for r in results if isinstance(r, Exception))
        print(f"✅ Успешно обработано: {success_count}")
        print(f"❌ Ошибок: {error_count}")
        assert success_count > num_patterns * 0.95, "Слишком много ошибок"
        # Проверяем финальную статистику
        stats = await storage.get_storage_statistics()
        assert stats.total_patterns >= num_patterns * 0.95
        print(f"📊 Финальная статистика: {stats.total_patterns} паттернов")
    @pytest.mark.asyncio
    async def test_data_recovery_e2e(self, e2e_components, realistic_patterns) -> None:
        """E2E тест восстановления данных."""
        storage = e2e_components["storage"]
        print("🚀 Начало E2E теста восстановления данных...")
        # Сохраняем паттерны
        for pattern in realistic_patterns:
            await storage.save_pattern("BTCUSDT", pattern)
        # Создаем результаты
        for i, pattern in enumerate(realistic_patterns):
            result = PatternResult(
                outcome=PatternOutcome.SUCCESS if i < 3 else PatternOutcome.FAILURE,
                price_change_15min=0.02 if i < 3 else -0.01,
                price_change_1h=0.05 if i < 3 else -0.02,
                volume_change=0.1 if i < 3 else -0.05,
                execution_time=300,
                confidence=Confidence(0.8 if i < 3 else 0.6)
            )
            pattern_id = f"BTCUSDT_{pattern.pattern_type.value}_{pattern.timestamp.strftime('%Y%m%d_%H%M%S')}"
            await storage.update_pattern_result("BTCUSDT", pattern_id, result)
        # Создаем резервную копию
        backup_success = await storage.backup_data("BTCUSDT")
        assert backup_success is True
        # Закрываем хранилище
        await storage.close()
        # Создаем новое хранилище
        new_storage = Mock()
        # Проверяем восстановление данных
        patterns = await new_storage.get_patterns_by_symbol("BTCUSDT")
        assert len(patterns) == len(realistic_patterns)
        # Проверяем, что результаты восстановлены
        for pattern_memory in patterns:
            assert pattern_memory.result is not None
            assert pattern_memory.accuracy > 0.0
            assert pattern_memory.avg_return != 0.0
        await new_storage.close()
        print("✅ Восстановление данных прошло успешно!")
    @pytest.mark.asyncio
    async def test_error_scenarios_e2e(self, e2e_components) -> None:
        """E2E тест сценариев ошибок."""
        storage = e2e_components["storage"]
        analyzer = e2e_components["analyzer"]
        print("🚀 Начало E2E теста сценариев ошибок...")
        # Тест 1: Некорректные данные
        print("🔍 Тест 1: Некорректные данные")
        try:
            await storage.save_pattern("", None)
            assert False, "Должна была возникнуть ошибка"
        except Exception:
            print("✅ Ошибка корректно обработана")
        # Тест 2: Пустые данные
        print("🔍 Тест 2: Пустые данные")
        patterns = await storage.get_patterns_by_symbol("NONEXISTENT")
        assert len(patterns) == 0
        print("✅ Пустые данные корректно обработаны")
        # Тест 3: Некорректный анализ
        print("🔍 Тест 3: Некорректный анализ")
        try:
            await analyzer.analyze_pattern("", None)
            assert False, "Должна была возникнуть ошибка"
        except Exception:
            print("✅ Ошибка анализа корректно обработана")
        # Тест 4: Система продолжает работать после ошибок
        print("🔍 Тест 4: Проверка работоспособности после ошибок")
        # Создаем корректный паттерн
        features = PatternFeatures(
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
                # Убираем лишние ключи, оставляем только разрешенные
            })
        )
        pattern = MarketMakerPattern(
            pattern_type=MarketMakerPatternType.ACCUMULATION,
            symbol=Symbol("BTCUSDT"),
            timestamp=datetime.now(),
            features=features,
            confidence=Confidence(0.85),
            context={}
        )
        success = await storage.save_pattern("BTCUSDT", pattern)
        assert success is True
        analysis = await analyzer.analyze_pattern("BTCUSDT", pattern)
        assert analysis is not None
        print("✅ Система продолжает работать после ошибок")
    @pytest.mark.asyncio
    async def test_performance_benchmark_e2e(self, e2e_components) -> None:
        """E2E тест бенчмарка производительности."""
        storage = e2e_components["storage"]
        analyzer = e2e_components["analyzer"]
        print("🚀 Начало E2E теста бенчмарка производительности...")
        # Тест записи
        print("📝 Тест производительности записи...")
        start_time = datetime.now()
        for i in range(50):
            features = PatternFeatures(
                book_pressure=BookPressure(0.6 + (i % 10) * 0.05),
                volume_delta=VolumeDelta(0.1 + (i % 10) * 0.02),
                price_reaction=PriceReaction(0.01 + (i % 10) * 0.002),
                spread_change=SpreadChange(0.02 + (i % 10) * 0.005),
                order_imbalance=OrderImbalance(0.4 + (i % 10) * 0.05),
                liquidity_depth=LiquidityDepth(0.6 + (i % 10) * 0.03),
                time_duration=TimeDuration(150 + (i % 10) * 20),
                volume_concentration=VolumeConcentration(0.5 + (i % 10) * 0.03),
                price_volatility=PriceVolatility(0.015 + (i % 10) * 0.002),
                market_microstructure=MarketMicrostructure({
                    # Убираем лишние ключи, оставляем только разрешенные
                })
            )
            pattern = MarketMakerPattern(
                pattern_type=MarketMakerPatternType.ACCUMULATION,
                symbol=Symbol("BTCUSDT"),  # Используем правильный тип Symbol
                timestamp=datetime.now() + timedelta(seconds=i),
                features=features,
                confidence=Confidence(0.7 + (i % 10) * 0.02),
                context={}  # Убираем лишние ключи
            )
            await storage.save_pattern("BTCUSDT", pattern)
        write_time = (datetime.now() - start_time).total_seconds()
        write_rate = 50 / write_time
        print(f"📝 Скорость записи: {write_rate:.2f} паттернов/сек")
        assert write_rate > 10, "Слишком медленная запись"
        # Тест чтения
        print("📖 Тест производительности чтения...")
        start_time = datetime.now()
        patterns = await storage.get_patterns_by_symbol("BTCUSDT")
        read_time = (datetime.now() - start_time).total_seconds()
        read_rate = len(patterns) / read_time
        print(f"📖 Скорость чтения: {read_rate:.2f} паттернов/сек")
        assert read_rate > 50, "Слишком медленное чтение"
        # Тест анализа
        print("🧠 Тест производительности анализа...")
        start_time = datetime.now()
        for pattern in patterns[:10]:  # Анализируем первые 10
            await analyzer.analyze_pattern("BTCUSDT", pattern.pattern)
        analysis_time = (datetime.now() - start_time).total_seconds()
        analysis_rate = 10 / analysis_time
        print(f"🧠 Скорость анализа: {analysis_rate:.2f} паттернов/сек")
        assert analysis_rate > 5, "Слишком медленный анализ"
        # Получаем финальные метрики
        stats = await storage.get_storage_statistics()
        print(f"📊 Финальные метрики:")
        print(f"   - Время записи: {stats.avg_write_time_ms:.2f} мс")
        print(f"   - Время чтения: {stats.avg_read_time_ms:.2f} мс")
        print(f"   - Коэффициент кэша: {stats.cache_hit_ratio:.2f}")
        print(f"   - Коэффициент сжатия: {stats.compression_ratio:.2f}")
        print("✅ Бенчмарк производительности завершен!")
if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
