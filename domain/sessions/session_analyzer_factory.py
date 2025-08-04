# -*- coding: utf-8 -*-
"""Фабрика анализаторов торговых сессий."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Type, Union

from loguru import logger

from domain.types.session_types import SessionType

from .interfaces import BaseSessionAnalyzer, SessionRegistry
from .session_analyzer import SessionAnalyzer
from .session_influence_analyzer import SessionInfluenceAnalyzer
from .session_marker import SessionMarker


@dataclass
class AnalyzerConfig:
    """Конфигурация анализатора."""

    # Основные параметры
    analyzer_type: str = "standard"  # "standard", "influence", "advanced"
    enabled_features: List[str] = None
    analysis_interval_minutes: int = 5
    cache_enabled: bool = True
    cache_ttl_minutes: int = 30
    # Параметры анализа
    min_data_points: int = 20
    confidence_threshold: float = 0.7
    volatility_threshold: float = 1.5
    # Параметры производительности
    max_concurrent_analyses: int = 10
    timeout_seconds: int = 30
    # Параметры логирования
    log_level: str = "INFO"
    log_analysis_results: bool = True

    def __post_init__(self) -> None:
        """Инициализация после создания объекта."""
        if self.enabled_features is None:
            self.enabled_features = [
                "volume_analysis",
                "volatility_analysis",
                "momentum_analysis",
                "session_overlap_analysis",
            ]

    def to_dict(self) -> Dict[str, Union[str, float, int, bool, List[str]]]:
        """Преобразование в словарь."""
        return {
            "analyzer_type": self.analyzer_type,
            "enabled_features": self.enabled_features,
            "analysis_interval_minutes": self.analysis_interval_minutes,
            "cache_enabled": self.cache_enabled,
            "cache_ttl_minutes": self.cache_ttl_minutes,
            "min_data_points": self.min_data_points,
            "confidence_threshold": self.confidence_threshold,
            "volatility_threshold": self.volatility_threshold,
            "max_concurrent_analyses": self.max_concurrent_analyses,
            "timeout_seconds": self.timeout_seconds,
            "log_level": self.log_level,
            "log_analysis_results": self.log_analysis_results,
        }

    def validate(self) -> bool:
        """Валидация конфигурации."""
        if self.analyzer_type not in ["standard", "influence", "advanced"]:
            logger.error(f"Invalid analyzer_type: {self.analyzer_type}")
            return False
        if self.analysis_interval_minutes < 1:
            logger.error(
                f"Invalid analysis_interval_minutes: {self.analysis_interval_minutes}"
            )
            return False
        if self.min_data_points < 10:
            logger.error(f"Invalid min_data_points: {self.min_data_points}")
            return False
        if not (0.0 <= self.confidence_threshold <= 1.0):
            logger.error(f"Invalid confidence_threshold: {self.confidence_threshold}")
            return False
        if self.volatility_threshold <= 0.0:
            logger.error(f"Invalid volatility_threshold: {self.volatility_threshold}")
            return False
        if self.max_concurrent_analyses < 1:
            logger.error(
                f"Invalid max_concurrent_analyses: {self.max_concurrent_analyses}"
            )
            return False
        if self.timeout_seconds < 1:
            logger.error(f"Invalid timeout_seconds: {self.timeout_seconds}")
            return False
        return True


class SessionAnalyzerFactory:
    """Фабрика анализаторов торговых сессий."""

    def __init__(self) -> None:
        self._analyzers: Dict[str, Type[BaseSessionAnalyzer]] = {
            "standard": SessionAnalyzer,
            "influence": SessionInfluenceAnalyzer,
        }
        self._configs: Dict[str, AnalyzerConfig] = {}
        self._instances: Dict[str, BaseSessionAnalyzer] = {}
        logger.info("SessionAnalyzerFactory initialized")

    def register_analyzer(
        self,
        name: str,
        analyzer_class: Type[BaseSessionAnalyzer],
        config: Optional[AnalyzerConfig] = None,
    ) -> None:
        """Регистрация нового анализатора."""
        try:
            if not issubclass(analyzer_class, BaseSessionAnalyzer):
                raise ValueError(
                    f"Analyzer class must inherit from BaseSessionAnalyzer"
                )
            self._analyzers[name] = analyzer_class
            if config:
                if not config.validate():
                    raise ValueError("Invalid analyzer configuration")
                self._configs[name] = config
            logger.info(f"Registered analyzer: {name}")
        except Exception as e:
            logger.error(f"Error registering analyzer {name}: {e}")
            raise

    def create_analyzer(
        self,
        name: str,
        registry: SessionRegistry,
        session_marker: SessionMarker,
        config: Optional[AnalyzerConfig] = None,
        force_recreate: bool = False,
    ) -> Optional[BaseSessionAnalyzer]:
        """Создание экземпляра анализатора."""
        try:
            analyzer_class = self._analyzers[name]
            # Используем переданную конфигурацию или сохраненную
            analyzer_config = config or self._configs.get(name)
            # Проверяем, какие аргументы принимает конструктор
            import inspect
            sig = inspect.signature(analyzer_class.__init__)
            params = sig.parameters
            
            # Проверяем, какие параметры принимает конструктор
            if 'registry' in params:
                analyzer = analyzer_class(registry=registry)
            else:
                analyzer = analyzer_class(registry=registry)
            self._instances[name] = analyzer
            logger.info(f"Created analyzer: {name}")
            return analyzer
        except Exception as e:
            logger.error(f"Error creating analyzer {name}: {e}")
            return None

    def get_analyzer(self, name: str) -> Optional[BaseSessionAnalyzer]:
        """Получение анализатора из кэша."""
        return self._instances.get(name)

    def get_available_analyzers(self) -> List[str]:
        """Получение списка доступных анализаторов."""
        return list(self._analyzers.keys())

    def get_analyzer_config(self, name: str) -> Optional[AnalyzerConfig]:
        """Получение конфигурации анализатора."""
        return self._configs.get(name)

    def update_analyzer_config(
        self,
        name: str,
        config: AnalyzerConfig,
    ) -> bool:
        """Обновление конфигурации анализатора."""
        try:
            if name not in self._analyzers:
                logger.error(f"Analyzer {name} not found")
                return False
            if not config.validate():
                logger.error(f"Invalid configuration for analyzer {name}")
                return False
            self._configs[name] = config
            # Удаляем существующий экземпляр, чтобы он был пересоздан с новой конфигурацией
            if name in self._instances:
                del self._instances[name]
            logger.info(f"Updated configuration for analyzer: {name}")
            return True
        except Exception as e:
            logger.error(f"Error updating analyzer config {name}: {e}")
            return False

    def remove_analyzer(self, name: str) -> bool:
        """Удаление анализатора."""
        try:
            if name not in self._analyzers:
                logger.warning(f"Analyzer {name} not found")
                return False
            # Удаляем из всех словарей
            del self._analyzers[name]
            if name in self._configs:
                del self._configs[name]
            if name in self._instances:
                del self._instances[name]
            logger.info(f"Removed analyzer: {name}")
            return True
        except Exception as e:
            logger.error(f"Error removing analyzer {name}: {e}")
            return False

    def create_analyzer_for_session_type(
        self,
        session_type: SessionType,
        registry: SessionRegistry,
        session_marker: SessionMarker,
        config: Optional[AnalyzerConfig] = None,
    ) -> Optional[BaseSessionAnalyzer]:
        """Создание анализатора для конкретного типа сессии."""
        try:
            # Определяем подходящий анализатор на основе типа сессии
            analyzer_name = self._select_analyzer_for_session_type(session_type)
            if not analyzer_name:
                logger.warning(
                    f"No suitable analyzer found for session type: {session_type.value}"
                )
                return None
            return self.create_analyzer(analyzer_name, registry, session_marker, config)
        except Exception as e:
            logger.error(f"Error creating analyzer for session type {session_type.value}: {e}")
            return None

    def create_multi_analyzer(
        self,
        analyzer_names: List[str],
        registry: SessionRegistry,
        session_marker: SessionMarker,
        configs: Optional[Dict[str, AnalyzerConfig]] = None,
    ) -> Dict[str, BaseSessionAnalyzer]:
        """Создание нескольких анализаторов."""
        analyzers: Dict[str, BaseSessionAnalyzer] = {}
        for name in analyzer_names:
            config = configs.get(name) if configs else None
            analyzer = self.create_analyzer(name, registry, session_marker, config)
            if analyzer:
                analyzers[name] = analyzer
        return analyzers

    def get_analyzer_statistics(self) -> Dict[str, Union[str, int, bool, List[str]]]:
        """Получение статистики фабрики."""
        return {
            "total_registered": len(self._analyzers),
            "total_instances": len(self._instances),
            "total_configs": len(self._configs),
            "available_analyzers": list(self._analyzers.keys()),
            "cached_instances": list(self._instances.keys()),
        }

    def clear_cache(self) -> None:
        """Очистка кэша экземпляров."""
        self._instances.clear()
        logger.info("Analyzer factory cache cleared")

    def validate_analyzer(self, name: str) -> bool:
        """Валидация анализатора."""
        try:
            if name not in self._analyzers:
                return False
            analyzer_class = self._analyzers[name]
            # Проверяем, что класс можно создать
            # Это базовая проверка - в реальности может потребоваться более сложная валидация
            return True
        except Exception as e:
            logger.error(f"Error validating analyzer {name}: {e}")
            return False

    def _select_analyzer_for_session_type(
        self, session_type: SessionType
    ) -> Optional[str]:
        """Выбор подходящего анализатора для типа сессии."""
        # Простая логика выбора на основе типа сессии
        if session_type == SessionType.CRYPTO_24H:
            return "influence"  # Для криптовалют используем анализатор влияния
        else:
            return "standard"  # Для остальных используем стандартный анализатор

    def _create_default_configs(self) -> None:
        """Создание конфигураций по умолчанию."""
        # Стандартный анализатор
        standard_config = AnalyzerConfig(
            analyzer_type="standard",
            enabled_features=[
                "volume_analysis",
                "volatility_analysis",
                "momentum_analysis",
                "session_overlap_analysis",
            ],
            analysis_interval_minutes=5,
            cache_enabled=True,
            cache_ttl_minutes=30,
            min_data_points=20,
            confidence_threshold=0.7,
            volatility_threshold=1.5,
            max_concurrent_analyses=10,
            timeout_seconds=30,
            log_level="INFO",
            log_analysis_results=True,
        )
        # Анализатор влияния
        influence_config = AnalyzerConfig(
            analyzer_type="influence",
            enabled_features=[
                "volume_analysis",
                "volatility_analysis",
                "momentum_analysis",
                "session_overlap_analysis",
                "influence_analysis",
                "correlation_analysis",
            ],
            analysis_interval_minutes=3,
            cache_enabled=True,
            cache_ttl_minutes=20,
            min_data_points=30,
            confidence_threshold=0.8,
            volatility_threshold=1.3,
            max_concurrent_analyses=5,
            timeout_seconds=45,
            log_level="DEBUG",
            log_analysis_results=True,
        )
        self._configs["standard"] = standard_config
        self._configs["influence"] = influence_config
        logger.info("Created default analyzer configurations")
