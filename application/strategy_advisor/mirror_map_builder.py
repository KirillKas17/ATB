# -*- coding: utf-8 -*-
"""Mirror Map Builder for Asset Dependency Analysis."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from shared.numpy_utils import np
import pandas as pd
from loguru import logger

from domain.intelligence.mirror_detector import (
    CorrelationMatrix,
    MirrorDetectionConfig,
    MirrorDetector,
    MirrorSignal,
)
from domain.type_definitions.intelligence_types import CorrelationMethod
from domain.value_objects.timestamp import Timestamp


@dataclass
class MirrorMap:
    """Карта зеркальных зависимостей между активами."""

    assets: List[str]
    mirror_map: Dict[str, List[str]]
    correlation_matrix: Optional[CorrelationMatrix] = None
    clusters: List[List[str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Timestamp = field(
        default_factory=Timestamp.now
    )

    def get_mirror_assets(self, asset: str) -> List[str]:
        """Получение зеркальных активов для заданного актива."""
        return self.mirror_map.get(asset, [])

    def is_mirror_pair(self, asset1: str, asset2: str) -> bool:
        """Проверка, являются ли активы зеркальной парой."""
        mirror_assets = self.get_mirror_assets(asset1)
        return asset2 in mirror_assets

    def get_correlation(self, asset1: str, asset2: str) -> float:
        """Получение корреляции между активами."""
        if self.correlation_matrix:
            return self.correlation_matrix.get_correlation(asset1, asset2)
        return 0.0

    def get_lag(self, asset1: str, asset2: str) -> int:
        """Получение лага между активами."""
        if self.correlation_matrix:
            return self.correlation_matrix.get_lag(asset1, asset2)
        return 0

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "assets": self.assets,
            "mirror_map": self.mirror_map,
            "clusters": self.clusters,
            "metadata": self.metadata,
            "timestamp": self.timestamp.value,
            "correlation_matrix": (
                {
                    "assets": (
                        self.correlation_matrix.assets
                        if self.correlation_matrix
                        else []
                    ),
                    "correlation_matrix": (
                        self.correlation_matrix.correlation_matrix.tolist()
                        if self.correlation_matrix
                        else []
                    ),
                    "lag_matrix": (
                        self.correlation_matrix.lag_matrix.tolist()
                        if self.correlation_matrix
                        else []
                    ),
                }
                if self.correlation_matrix
                else None
            ),
        }


@dataclass
class MirrorMapConfig:
    """Конфигурация для построения карты зеркальных зависимостей."""

    min_correlation: float = 0.3
    max_p_value: float = 0.05
    min_confidence: float = 0.7
    max_lag: int = 5
    correlation_method: str = "pearson"
    normalize_data: bool = True
    remove_trend: bool = True
    min_cluster_size: int = 2
    max_cluster_size: int = 10
    update_interval: int = 3600  # секунды
    parallel_processing: bool = True
    max_workers: int = 4


class MirrorMapBuilder:
    """Сервис для построения карты зеркальных зависимостей между активами."""

    def __init__(self, config: Optional[MirrorMapConfig] = None):
        self.config = config or MirrorMapConfig()
        # Создаем конфигурацию для MirrorDetector
        mirror_config = MirrorDetectionConfig(
            min_correlation=self.config.min_correlation,
            max_p_value=self.config.max_p_value,
            min_confidence=self.config.min_confidence,
            correlation_method=CorrelationMethod(self.config.correlation_method),
            normalize_data=self.config.normalize_data,
            remove_trend=self.config.remove_trend,
            max_lag=self.config.max_lag,
        )
        self.mirror_detector = MirrorDetector(config=mirror_config)
        # Кэш для хранения результатов
        self._mirror_map_cache: Optional[MirrorMap] = None
        self._last_update: Optional[float] = None
        logger.info(f"MirrorMapBuilder initialized with config: {self.config}")

    def _validate_price_data(
        self, assets: List[str], price_data: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """Валидация и очистка данных о ценах."""
        try:
            valid_data = {}
            for asset in assets:
                if asset not in price_data:
                    logger.warning(f"Missing price data for asset: {asset}")
                    continue
                series = price_data[asset]
                # Проверяем, что это pandas Series
                if not isinstance(series, pd.Series):
                    logger.warning(
                        f"Invalid data type for asset {asset}: {type(series)}"
                    )
                    continue
                # Проверяем минимальную длину
                if len(series) < 50:
                    logger.warning(
                        f"Insufficient data for asset {asset}: {len(series)} points"
                    )
                    continue
                # Проверяем наличие данных
                if hasattr(series, 'isnull') and series.isnull().all():
                    logger.warning(f"All NaN data for asset {asset}")
                    continue
                valid_data[asset] = series
            logger.info(f"Validated {len(valid_data)} assets out of {len(assets)}")
            return valid_data
        except Exception as e:
            logger.error(f"Error validating price data: {e}")
            return {}

    def _build_correlation_matrix_parallel(
        self, assets: List[str], price_data: Dict[str, pd.Series]
    ) -> CorrelationMatrix:
        """Построение матрицы корреляций с параллельной обработкой."""
        try:
            if not self.config.parallel_processing:
                return self.mirror_detector.build_correlation_matrix(
                    assets, price_data, self.config.max_lag
                )
            # Создаем матрицы
            n = len(assets)
            correlation_matrix = np.zeros((n, n))
            lag_matrix = np.zeros((n, n), dtype=int)
            p_value_matrix = np.ones((n, n))
            confidence_matrix = np.zeros((n, n))
            # Подготавливаем задачи для параллельной обработки
            tasks = []
            for i, asset1 in enumerate(assets):
                for j, asset2 in enumerate(assets):
                    if i < j and asset1 in price_data and asset2 in price_data:
                        tasks.append(
                            (
                                i,
                                j,
                                asset1,
                                asset2,
                                price_data[asset1],
                                price_data[asset2],
                            )
                        )
            # Обрабатываем задачи параллельно
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                for i, j, asset1, asset2, series1, series2 in tasks:
                    future = executor.submit(
                        self._process_asset_pair, asset1, asset2, series1, series2
                    )
                    futures.append((i, j, future))
                # Собираем результаты
                for i, j, future in futures:
                    try:
                        result = future.result(timeout=30)  # 30 секунд таймаут
                        if result:
                            signal = result
                            correlation_matrix[i, j] = signal.correlation
                            correlation_matrix[j, i] = signal.correlation
                            lag_matrix[i, j] = signal.best_lag
                            lag_matrix[j, i] = -signal.best_lag
                            p_value_matrix[i, j] = signal.p_value
                            p_value_matrix[j, i] = signal.p_value
                            confidence_matrix[i, j] = signal.confidence
                            confidence_matrix[j, i] = signal.confidence
                    except Exception as e:
                        logger.error(f"Error processing asset pair {i}-{j}: {e}")
            # Заполняем диагональ
            for i in range(n):
                correlation_matrix[i, i] = 1.0
                lag_matrix[i, i] = 0
                p_value_matrix[i, i] = 0.0
                confidence_matrix[i, i] = 1.0
            return CorrelationMatrix(
                assets=assets,
                correlation_matrix=correlation_matrix,
                lag_matrix=lag_matrix,
                p_value_matrix=p_value_matrix,
                confidence_matrix=confidence_matrix,
                timestamp=Timestamp.now(),
                metadata={
                    "data_points": len(assets),
                    "confidence": 0.8,
                    "processing_time_ms": 0.0,
                    "algorithm_version": "1.0",
                    "parameters": {
                        "max_lag": self.config.max_lag,
                        "parallel_processing": self.config.parallel_processing,
                    },
                    "quality_metrics": {"matrix_density": 1.0, "avg_correlation": 0.0},
                },
            )
        except Exception as e:
            logger.error(f"Error in parallel correlation matrix building: {e}")
            return self.mirror_detector.build_correlation_matrix(
                assets, price_data, self.config.max_lag
            )

    def _process_asset_pair(
        self, asset1: str, asset2: str, series1: pd.Series, series2: pd.Series
    ) -> Optional[MirrorSignal]:
        """Обработка пары активов для параллельного выполнения."""
        try:
            return self.mirror_detector.detect_mirror_signal(
                asset1, asset2, series1, series2, self.config.max_lag
            )
        except Exception as e:
            logger.error(f"Error processing asset pair {asset1}-{asset2}: {e}")
            return None

    def _build_mirror_map_from_matrix(
        self, correlation_matrix: CorrelationMatrix
    ) -> Dict[str, List[str]]:
        """Построение карты зеркальных зависимостей из матрицы корреляций."""
        try:
            mirror_map = {}
            for i, asset1 in enumerate(correlation_matrix.assets):
                mirror_assets = []
                for j, asset2 in enumerate(correlation_matrix.assets):
                    if i != j:
                        correlation = correlation_matrix.get_correlation(asset1, asset2)
                        p_value = correlation_matrix.get_p_value(asset1, asset2)
                        confidence = correlation_matrix.get_confidence(asset1, asset2)
                        # Проверяем условия для зеркальной зависимости
                        if (
                            abs(correlation) >= self.config.min_correlation
                            and p_value <= self.config.max_p_value
                            and confidence >= self.config.min_confidence
                        ):
                            mirror_assets.append(asset2)
                if mirror_assets:
                    mirror_map[asset1] = mirror_assets
            logger.info(
                f"Built mirror map with {len(mirror_map)} assets having mirror dependencies"
            )
            return mirror_map
        except Exception as e:
            logger.error(f"Error building mirror map from matrix: {e}")
            return {}

    def build_mirror_map(
        self,
        assets: List[str],
        price_data: Dict[str, pd.Series],
        force_rebuild: bool = False,
    ) -> MirrorMap:
        """
        Построение карты зеркальных зависимостей между активами.
        Args:
            assets: Список активов для анализа
            price_data: Словарь с временными рядами цен
            force_rebuild: Принудительное перестроение карты
        Returns:
            MirrorMap: Карта зеркальных зависимостей
        """
        try:
            # Проверяем кэш
            current_time = Timestamp.now()
            if (
                not force_rebuild
                and self._mirror_map_cache
                and self._last_update
                and float(str(current_time.value)) - float(str(self._last_update)) < self.config.update_interval
            ):
                logger.info("Using cached mirror map")
                return self._mirror_map_cache
            logger.info(f"Building mirror map for {len(assets)} assets")
            # Валидируем данные
            valid_price_data = self._validate_price_data(assets, price_data)
            if not valid_price_data:
                logger.error("No valid price data available")
                return MirrorMap(assets=assets, mirror_map={})
            # Строим матрицу корреляций
            correlation_matrix = self._build_correlation_matrix_parallel(
                list(valid_price_data.keys()), valid_price_data
            )
            # Строим карту зеркальных зависимостей
            mirror_map = self._build_mirror_map_from_matrix(correlation_matrix)
            # Находим кластеры
            clusters = self.mirror_detector.find_mirror_clusters(
                correlation_matrix, self.config.min_correlation
            )
            # Фильтруем кластеры по размеру
            filtered_clusters = [
                cluster
                for cluster in clusters
                if self.config.min_cluster_size
                <= len(cluster)
                <= self.config.max_cluster_size
            ]
            # Метаданные
            metadata = {
                "total_assets": len(assets),
                "valid_assets": len(valid_price_data),
                "mirror_pairs": sum(len(mirrors) for mirrors in mirror_map.values())
                // 2,
                "clusters_found": len(filtered_clusters),
                "config": self.config.__dict__,
                "detector_stats": self.mirror_detector.get_detector_statistics(),
            }
            # Создаем результат
            result = MirrorMap(
                assets=list(valid_price_data.keys()),
                mirror_map=mirror_map,
                correlation_matrix=correlation_matrix,
                clusters=filtered_clusters,
                metadata=metadata,
                timestamp=Timestamp(current_time.value),
            )
            # Обновляем кэш
            self._mirror_map_cache = result
            # Исправление: правильное преобразование TimestampValue в float
            self._last_update = float(str(current_time.value))
            logger.info(
                f"Mirror map built successfully: {len(mirror_map)} assets with dependencies, {len(filtered_clusters)} clusters"
            )
            return result
        except Exception as e:
            logger.error(f"Error building mirror map: {e}")
            return MirrorMap(assets=assets, mirror_map={})

    async def build_mirror_map_async(
        self,
        assets: List[str],
        price_data: Dict[str, pd.Series],
        force_rebuild: bool = False,
    ) -> MirrorMap:
        """
        Асинхронное построение карты зеркальных зависимостей.
        Args:
            assets: Список активов для анализа
            price_data: Словарь с временными рядами цен
            force_rebuild: Принудительное перестроение карты
        Returns:
            MirrorMap: Карта зеркальных зависимостей
        """
        try:
            # Выполняем построение в отдельном потоке
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.build_mirror_map, assets, price_data, force_rebuild
            )
            return result
        except Exception as e:
            logger.error(f"Error in async mirror map building: {e}")
            return MirrorMap(assets=assets, mirror_map={})

    def get_mirror_assets_for_strategy(
        self,
        mirror_map: MirrorMap,
        base_asset: str,
        min_correlation: Optional[float] = None,
    ) -> List[Tuple[str, float, int]]:
        """
        Получение зеркальных активов для торговой стратегии.
        Args:
            mirror_map: Карта зеркальных зависимостей
            base_asset: Базовый актив
            min_correlation: Минимальная корреляция
        Returns:
            List[Tuple[str, float, int]]: Список (актив, корреляция, лаг)
        """
        try:
            if min_correlation is None:
                min_correlation = self.config.min_correlation
            mirror_assets = mirror_map.get_mirror_assets(base_asset)
            result = []
            for asset in mirror_assets:
                if mirror_map.correlation_matrix:
                    correlation = mirror_map.get_correlation(base_asset, asset)
                    lag = mirror_map.get_lag(base_asset, asset)
                    if abs(correlation) >= min_correlation:
                        result.append((asset, correlation, lag))
            # Сортируем по силе корреляции
            result.sort(key=lambda x: abs(x[1]), reverse=True)
            return result
        except Exception as e:
            logger.error(f"Error getting mirror assets for strategy: {e}")
            return []

    def analyze_mirror_clusters(self, mirror_map: MirrorMap) -> Dict[str, Any]:
        """
        Анализ кластеров зеркальных активов.
        Args:
            mirror_map: Карта зеркальных зависимостей
        Returns:
            Dict[str, Any]: Результаты анализа
        """
        try:
            analysis: Dict[str, Any] = {
                "total_clusters": len(mirror_map.clusters),
                "cluster_sizes": [len(cluster) for cluster in mirror_map.clusters],
                "largest_cluster": (
                    max([len(cluster) for cluster in mirror_map.clusters])
                    if mirror_map.clusters
                    else 0
                ),
                "average_cluster_size": (
                    np.mean([len(cluster) for cluster in mirror_map.clusters])
                    if mirror_map.clusters
                    else 0
                ),
                "cluster_details": [],
            }
            for i, cluster in enumerate(mirror_map.clusters):
                cluster_analysis = {
                    "cluster_id": i,
                    "assets": cluster,
                    "size": len(cluster),
                    "average_correlation": 0.0,
                    "average_lag": 0.0,
                }
                # Вычисляем средние значения для кластера
                correlations = []
                lags = []
                for j, asset1 in enumerate(cluster):
                    for k, asset2 in enumerate(cluster):
                        if j < k and mirror_map.correlation_matrix:
                            corr = mirror_map.get_correlation(asset1, asset2)
                            lag = mirror_map.get_lag(asset1, asset2)
                            correlations.append(corr)
                            lags.append(lag)
                if correlations:
                    cluster_analysis["average_correlation"] = np.mean(correlations)
                    cluster_analysis["average_lag"] = np.mean(lags)
                analysis["cluster_details"].append(cluster_analysis)
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing mirror clusters: {e}")
            return {}

    def get_mirror_map_statistics(self) -> Dict[str, Any]:
        """Получение статистики построения карты зеркальных зависимостей."""
        if not self._mirror_map_cache:
            return {
                "total_assets": 0,
                "mirror_pairs": 0,
                "clusters": 0,
                "average_correlation": 0.0,
                "cache_hit_rate": 0.0,
                "last_update": None,
            }

        mirror_map = self._mirror_map_cache
        total_assets = len(mirror_map.assets)
        mirror_pairs = sum(len(mirrors) for mirrors in mirror_map.mirror_map.values())
        clusters = len(mirror_map.clusters)

        # Расчет средней корреляции
        avg_correlation = 0.0
        if mirror_map.correlation_matrix:
            correlation_values = []
            for i, asset1 in enumerate(mirror_map.assets):
                for j, asset2 in enumerate(mirror_map.assets):
                    if i < j:
                        corr = mirror_map.correlation_matrix.get_correlation(asset1, asset2)
                        if not np.isnan(corr):
                            correlation_values.append(corr)
            if correlation_values:
                avg_correlation = float(np.mean(correlation_values))

        # Расчет hit rate кэша
        cache_hit_rate = 0.0
        if self._last_update:
            # Исправление: правильное преобразование TimestampValue в float
            time_since_update = float(str(Timestamp.now().value)) - float(self._last_update)
            if time_since_update < self.config.update_interval:
                cache_hit_rate = 1.0

        return {
            "total_assets": total_assets,
            "mirror_pairs": mirror_pairs,
            "clusters": clusters,
            "average_correlation": avg_correlation,
            "cache_hit_rate": cache_hit_rate,
            "last_update": self._last_update,
            "config": {
                "min_correlation": self.config.min_correlation,
                "max_lag": self.config.max_lag,
                "correlation_method": self.config.correlation_method,
            },
        }

    def clear_cache(self) -> None:
        """Очистка кэша."""
        self._mirror_map_cache = None
        self._last_update = None
        logger.info("Mirror map cache cleared")

    def update_config(self, new_config: MirrorMapConfig) -> None:
        """Обновление конфигурации."""
        try:
            self.config = new_config
            # Создаем конфигурацию для MirrorDetector
            mirror_config = MirrorDetectionConfig(
                min_correlation=self.config.min_correlation,
                max_p_value=self.config.max_p_value,
                min_confidence=self.config.min_confidence,
                correlation_method=CorrelationMethod(self.config.correlation_method),
                normalize_data=self.config.normalize_data,
                remove_trend=self.config.remove_trend,
                max_lag=self.config.max_lag,
            )
            self.mirror_detector = MirrorDetector(config=mirror_config)
            # Очищаем кэш при изменении конфигурации
            self.clear_cache()
            logger.info(f"MirrorMapBuilder config updated: {self.config}")
        except Exception as e:
            logger.error(f"Error updating config: {e}")
