"""
Репозитории для модуля торговых сессий.
"""

import numpy as np
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from domain.types.session_types import (
    SessionAnalysisResult,
    SessionProfile,
    SessionType,
)
from domain.value_objects.timestamp import Timestamp

from .interfaces import (
    SessionConfigurationRepository as SessionConfigurationRepositoryProtocol,
)


class SessionDataRepository:
    """Реализация репозитория данных сессий."""

    def __init__(self, storage_path: Optional[str] = None) -> None:
        self.storage_path: str = storage_path or "data/sessions"
        self._cache: Dict[str, List[SessionAnalysisResult]] = {}
        # Создаем директорию, если не существует
        import os

        os.makedirs(self.storage_path, exist_ok=True)

    def save_session_analysis(self, analysis: SessionAnalysisResult) -> None:
        """Сохранение анализа сессии."""
        try:
            # Создаем ключ для кэширования
            cache_key = (
                f"{analysis.session_type.value}_{analysis.timestamp.to_iso()[:10]}"
            )
            # Добавляем в кэш
            if cache_key not in self._cache:
                self._cache[cache_key] = []
            self._cache[cache_key].append(analysis)
            # Сохраняем в файл
            self._save_to_file(analysis)
            logger.debug(f"Saved session analysis for {analysis.session_type.value}")
        except Exception as e:
            logger.error(f"Failed to save session analysis: {e}")
            raise

    def get_session_analysis(
        self, session_type: SessionType, start_time: Timestamp, end_time: Timestamp
    ) -> List[SessionAnalysisResult]:
        """Получение анализа сессии."""
        try:
            # Проверяем кэш
            cache_key = f"{session_type.value}_{start_time.to_iso()[:10]}"
            if cache_key in self._cache:
                cached_analyses = self._cache[cache_key]
                return [
                    analysis
                    for analysis in cached_analyses
                    if start_time <= analysis.timestamp <= end_time
                ]
            # Загружаем из файла
            return self._load_from_file(session_type, start_time, end_time)
        except Exception as e:
            logger.error(f"Failed to get session analysis: {e}")
            return []

    def get_session_statistics(
        self,
        session_type: SessionType,
        lookback_days: int = 1,
        start_time: Optional[Timestamp] = None,
        end_time: Optional[Timestamp] = None,
    ) -> Dict[str, Union[str, float, int]]:
        """Получение статистики сессии за указанный диапазон или lookback_days."""
        try:
            if start_time is not None and end_time is not None:
                analyses = self.get_session_analysis(session_type, start_time, end_time)
            else:
                end_time = Timestamp.now()
                start_time = Timestamp(
                    end_time.to_datetime() - timedelta(days=lookback_days)
                )
                analyses = self.get_session_analysis(session_type, start_time, end_time)
            if not analyses:
                return {}
            confidences = [float(analysis.confidence) for analysis in analyses]
            volatilities = [
                analysis.predictions.get("volatility", 0.0) for analysis in analyses
            ]
            volumes = [analysis.predictions.get("volume", 0.0) for analysis in analyses]
            return {
                "total_analyses": len(analyses),
                "avg_confidence": (
                    sum(confidences) / len(confidences) if confidences else 0.0
                ),
                "avg_volatility": (
                    sum(volatilities) / len(volatilities) if volatilities else 0.0
                ),
                "avg_volume": sum(volumes) / len(volumes) if volumes else 0.0,
                "max_confidence": max(confidences) if confidences else 0.0,
                "min_confidence": min(confidences) if confidences else 0.0,
                "confidence_std": (
                    float(np.std(confidences)) if len(confidences) > 1 else 0.0
                ),
            }
        except Exception as e:
            logger.error(f"Failed to get session statistics: {e}")
            return {}

    def delete_session_analysis(
        self, session_type: SessionType, start_time: Timestamp, end_time: Timestamp
    ) -> int:
        """Удаление анализа сессии."""
        try:
            # Получаем анализы для удаления
            analyses = self.get_session_analysis(session_type, start_time, end_time)
            # Удаляем из кэша
            cache_key = f"{session_type.value}_{start_time.to_iso()[:10]}"
            if cache_key in self._cache:
                self._cache[cache_key] = [
                    analysis
                    for analysis in self._cache[cache_key]
                    if not (start_time <= analysis.timestamp <= end_time)
                ]
            # Удаляем из файла
            deleted_count = self._delete_from_file(session_type, start_time, end_time)
            logger.info(
                f"Deleted {deleted_count} session analyses for {session_type.value}"
            )
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to delete session analysis: {e}")
            return 0

    def get_session_analysis_summary(
        self, session_type: SessionType, lookback_days: int = 30
    ) -> Dict[str, Union[str, float, int, Dict[str, Union[str, float, int]]]]:
        """Получение сводки анализа сессии."""
        try:
            end_time = Timestamp.now()
            start_time = Timestamp(
                end_time.to_datetime() - timedelta(days=lookback_days)
            )
            analyses = self.get_session_analysis(session_type, start_time, end_time)
            if not analyses:
                return {
                    "session_type": session_type.value,
                    "period_days": lookback_days,
                    "total_analyses": 0,
                    "phases": {},
                    "risk_factors": {},
                }
            # Группируем по фазам
            phases: Dict[str, Dict[str, Union[str, float, int]]] = {}
            risk_factors: Dict[str, int] = {}
            for analysis in analyses:
                phase = analysis.session_phase.value
                if phase not in phases:
                    phases[phase] = {
                        "count": 0,
                        "avg_confidence": 0.0,
                        "avg_volatility": 0.0,
                    }
                phases[phase]["count"] = int(phases[phase]["count"]) + 1
                phases[phase]["avg_confidence"] = float(phases[phase]["avg_confidence"]) + float(analysis.confidence)
                phases[phase]["avg_volatility"] = float(phases[phase]["avg_volatility"]) + analysis.predictions.get(
                    "volatility", 0.0
                )
            # Нормализуем средние значения
            for phase_data in phases.values():
                count = int(phase_data["count"])
                if count > 0:
                    phase_data["avg_confidence"] = float(phase_data["avg_confidence"]) / count
                    phase_data["avg_volatility"] = float(phase_data["avg_volatility"]) / count
            # Анализируем факторы риска
            for analysis in analyses:
                for risk_factor in analysis.risk_factors:
                    if risk_factor not in risk_factors:
                        risk_factors[risk_factor] = 0
                    risk_factors[risk_factor] = int(risk_factors[risk_factor]) + 1
            return {
                "session_type": session_type.value,
                "period_days": lookback_days,
                "total_analyses": len(analyses),
                "phases": phases,  # type: ignore
                "risk_factors": risk_factors,  # type: ignore
            }
        except Exception as e:
            logger.error(f"Failed to get session analysis summary: {e}")
            return {}

    def _save_to_file(self, analysis: SessionAnalysisResult) -> None:
        """Сохранение анализа в файл."""
        try:
            import json
            import os

            # Создаем имя файла на основе даты
            date_str = analysis.timestamp.to_iso()[:10]
            filename = f"{analysis.session_type.value}_{date_str}.json"
            filepath = os.path.join(self.storage_path, filename)
            # Загружаем существующие данные или создаем новый список
            existing_data: List[Dict[str, Union[str, float, int]]] = []
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            # Добавляем новый анализ
            analysis_dict = analysis.to_dict()
            existing_data.append(analysis_dict)
            # Сохраняем обратно в файл
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save analysis to file: {e}")
            raise

    def _load_from_file(
        self, session_type: SessionType, start_time: Timestamp, end_time: Timestamp
    ) -> List[SessionAnalysisResult]:
        """Загрузка анализа из файла."""
        try:
            import json
            import os
            from glob import glob

            analyses: List[SessionAnalysisResult] = []
            # Ищем файлы для данного типа сессии
            pattern = os.path.join(self.storage_path, f"{session_type.value}_*.json")
            files = glob(pattern)
            for filepath in files:
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for analysis_dict in data:
                        analysis = self._dict_to_analysis_result(analysis_dict)
                        if analysis and start_time <= analysis.timestamp <= end_time:
                            analyses.append(analysis)
                except Exception as e:
                    logger.warning(f"Failed to load file {filepath}: {e}")
                    continue
            return analyses
        except Exception as e:
            logger.error(f"Failed to load analyses from file: {e}")
            return []

    def _delete_from_file(
        self, session_type: SessionType, start_time: Timestamp, end_time: Timestamp
    ) -> int:
        """Удаление анализа из файла."""
        try:
            import json
            import os
            from glob import glob

            deleted_count = 0
            # Ищем файлы для данного типа сессии
            pattern = os.path.join(self.storage_path, f"{session_type.value}_*.json")
            files = glob(pattern)
            for filepath in files:
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # Фильтруем анализы, которые нужно удалить
                    filtered_data = []
                    for analysis_dict in data:
                        analysis = self._dict_to_analysis_result(analysis_dict)
                        if analysis and not (
                            start_time <= analysis.timestamp <= end_time
                        ):
                            filtered_data.append(analysis_dict)
                        else:
                            deleted_count += 1
                    # Сохраняем отфильтрованные данные
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.warning(f"Failed to delete from file {filepath}: {e}")
                    continue
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to delete analyses from file: {e}")
            return 0

    def _dict_to_analysis_result(
        self, analysis_dict: Dict[str, Union[str, float, int]]
    ) -> Optional[SessionAnalysisResult]:
        """Преобразование словаря в SessionAnalysisResult."""
        try:
            # Здесь должна быть логика создания SessionAnalysisResult из словаря
            # Пока возвращаем None, так как требуется полная реализация
            return None
        except Exception as e:
            logger.error(f"Failed to convert dict to analysis result: {e}")
            return None


class SessionConfigurationRepository:
    """Реализация репозитория конфигурации сессий."""

    def __init__(self, storage_path: Optional[str] = None) -> None:
        self.storage_path: str = storage_path or "data/session_configs"
        self._cache: Dict[SessionType, SessionProfile] = {}
        # Создаем директорию, если не существует
        import os

        os.makedirs(self.storage_path, exist_ok=True)

    def save_session_profile(self, profile: SessionProfile) -> None:
        """Сохранение профиля сессии."""
        try:
            self._save_profile_to_file(profile)
            self._cache[profile.session_type] = profile
            logger.info(f"Saved session profile: {profile.session_type.value}")
        except Exception as e:
            logger.error(f"Failed to save session profile: {e}")
            raise

    def get_session_profile(
        self, session_type: SessionType
    ) -> Optional[SessionProfile]:
        """Получение профиля сессии."""
        try:
            # Проверяем кэш
            if session_type in self._cache:
                return self._cache[session_type]
            # Загружаем из файла
            return self._load_profile_from_file(session_type)
        except Exception as e:
            logger.error(f"Failed to get session profile: {e}")
            return None

    def get_all_session_profiles(self) -> Dict[SessionType, SessionProfile]:
        """Получение всех профилей сессий."""
        try:
            import os
            from glob import glob

            profiles: Dict[SessionType, SessionProfile] = {}
            # Ищем все файлы профилей
            pattern = os.path.join(self.storage_path, "*.json")
            files = glob(pattern)
            for filepath in files:
                try:
                    # Извлекаем тип сессии из имени файла
                    filename = os.path.basename(filepath)
                    session_type_str = filename.replace(".json", "")
                    # Пытаемся найти соответствующий SessionType
                    for session_type in SessionType:
                        if session_type.value == session_type_str:
                            profile = self._load_profile_from_file(session_type)
                            if profile:
                                profiles[session_type] = profile
                            break
                except Exception as e:
                    logger.warning(f"Failed to load profile from {filepath}: {e}")
                    continue
            return profiles
        except Exception as e:
            logger.error(f"Failed to get all session profiles: {e}")
            return {}

    def delete_session_profile(self, session_type: SessionType) -> bool:
        """Удаление профиля сессии."""
        try:
            import os

            # Удаляем из кэша
            if session_type in self._cache:
                del self._cache[session_type]
            # Удаляем файл
            filename = f"{session_type.value}.json"
            filepath = os.path.join(self.storage_path, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Deleted session profile: {session_type.value}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete session profile: {e}")
            return False

    def update_session_profile(
        self,
        session_type: SessionType,
        updates: Dict[str, Union[str, float, int, bool]],
    ) -> bool:
        """Обновление профиля сессии."""
        try:
            profile = self.get_session_profile(session_type)
            if not profile:
                return False
            # Создаем новый профиль с обновленными значениями
            updated_profile = profile.model_copy(update=updates)
            # Сохраняем обновленный профиль
            self.save_session_profile(updated_profile)
            return True
        except Exception as e:
            logger.error(f"Failed to update session profile: {e}")
            return False

    def _save_profile_to_file(self, profile: SessionProfile) -> None:
        """Сохранение профиля в файл."""
        try:
            import json
            import os

            filename = f"{profile.session_type.value}.json"
            filepath = os.path.join(self.storage_path, filename)

            # Преобразуем профиль в словарь
            def convert(obj: Any) -> Union[str, float, int, Dict[str, Any], List[Any]]:
                if hasattr(obj, "to_dict"):
                    result = obj.to_dict()
                    if isinstance(result, str):
                        return result
                    elif isinstance(result, (int, float)):
                        return result
                    elif isinstance(result, dict):
                        return result
                    elif isinstance(result, list):
                        return result
                    else:
                        return str(result)
                elif hasattr(obj, "value"):
                    value = obj.value
                    if isinstance(value, (str, int, float)):
                        return value
                    else:
                        return str(value)
                else:
                    return str(obj)

            profile_dict = profile.model_dump()
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(
                    profile_dict, f, indent=2, ensure_ascii=False, default=convert
                )
        except Exception as e:
            logger.error(f"Failed to save profile to file: {e}")
            raise

    def _load_profile_from_file(
        self, session_type: SessionType
    ) -> Optional[SessionProfile]:
        """Загрузка профиля из файла."""
        try:
            import json
            import os

            filename = f"{session_type.value}.json"
            filepath = os.path.join(self.storage_path, filename)
            if not os.path.exists(filepath):
                return None
            with open(filepath, "r", encoding="utf-8") as f:
                profile_dict = json.load(f)
            # Создаем профиль из словаря
            # Здесь должна быть логика создания SessionProfile из словаря
            # Пока возвращаем None, так как требуется полная реализация
            return None
        except Exception as e:
            logger.error(f"Failed to load profile from file: {e}")
            return None
