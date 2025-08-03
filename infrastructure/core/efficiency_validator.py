# -*- coding: utf-8 -*-
"""
Валидатор эффективности эволюции.
Обеспечивает подтверждение улучшений перед применением изменений
в компонентах системы торговли.
"""
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


@dataclass
class EfficiencyConfig:
    """
    Конфигурация валидации эффективности.
    Содержит параметры для настройки процесса валидации
    эволюционных изменений в компонентах системы.
    """

    # Критически важные параметры подтверждения эффективности
    efficiency_improvement_threshold: float = 0.05  # Минимальное улучшение 5%
    confirmation_period: int = 1800  # 30 минут для подтверждения
    min_test_samples: int = 50  # Минимум тестовых образцов
    statistical_significance_level: float = 0.05  # Уровень статистической значимости
    rollback_on_degradation: bool = True  # Откат при деградации
    # Дополнительные параметры
    confidence_threshold: float = 0.6  # Минимальная уверенность
    stability_threshold: float = 0.8  # Минимальная стабильность результатов
    max_rollback_attempts: int = 3  # Максимум попыток отката
    # Пути для сохранения
    backup_path: str = "models/backup"
    validation_log_path: str = "logs/validation"


@dataclass
class EvolutionCandidate:
    """
    Кандидат на эволюцию.
    Представляет компонент, который проходит процесс валидации
    для подтверждения эффективности предложенных изменений.
    """

    component_name: str
    current_performance: float
    proposed_performance: float
    improvement: float
    confidence: float
    test_results: List[float]
    statistical_significance: float
    timestamp: datetime
    status: str = "pending"  # pending, testing, confirmed, rejected, rolled_back
    backup_path: str = ""
    rollback_attempts: int = 0


class EfficiencyValidator:
    """
    Валидатор эффективности эволюции.
    Обеспечивает проверку и подтверждение эффективности
    эволюционных изменений в компонентах системы.
    """

    def __init__(self, config: Optional[EfficiencyConfig] = None):
        """
        Инициализация валидатора эффективности.
        Args:
            config: Конфигурация валидатора
        """
        self.config = config or EfficiencyConfig()
        self.testing_candidates: Dict[str, EvolutionCandidate] = {}
        self.confirmed_evolutions: List[EvolutionCandidate] = []
        self.rollback_history: List[Dict[str, Any]] = []
        self.validation_stats: Dict[str, Any] = {}
        # Создание директорий
        os.makedirs(self.config.backup_path, exist_ok=True)
        os.makedirs(self.config.validation_log_path, exist_ok=True)
        logger.info("Efficiency Validator initialized")

    async def validate_evolution_candidate(
        self, component: Any, proposed_changes: Dict[str, Any]
    ) -> bool:
        """
        Валидация кандидата на эволюцию.
        Возвращает True только если подтверждена эффективность
        предложенных изменений.
        Args:
            component: Компонент для валидации
            proposed_changes: Предлагаемые изменения
        Returns:
            True если валидация успешна, False в противном случае
        """
        try:
            component_name = component.name
            current_performance = component.get_performance()
            logger.info(f"Validating evolution candidate for {component_name}")
            logger.info(f"Current performance: {current_performance:.4f}")
            # Создание кандидата
            candidate = EvolutionCandidate(
                component_name=component_name,
                current_performance=current_performance,
                proposed_performance=0.0,
                improvement=0.0,
                confidence=0.0,
                test_results=[],
                statistical_significance=0.0,
                timestamp=datetime.now(),
            )
            # Сохранение текущего состояния
            backup_path = (
                f"{self.config.backup_path}/{component_name}_backup_"
                f"{int(time.time())}"
            )
            if not component.save_state(backup_path):
                logger.error(f"Failed to backup {component_name} state")
                return False
            candidate.backup_path = backup_path
            # Применение предложенных изменений
            success = await self._apply_proposed_changes(component, proposed_changes)
            if not success:
                logger.error(f"Failed to apply proposed changes to {component_name}")
                await self._rollback_component(component, backup_path)
                return False
            # Тестирование на исторических данных
            test_results = await self._test_evolution_candidate(component)
            if len(test_results) < self.config.min_test_samples:
                logger.warning(
                    f"Insufficient test samples for {component_name}: "
                    f"{len(test_results)}"
                )
                await self._rollback_component(component, backup_path)
                return False
            # Статистический анализ
            improvement, significance = self._statistical_analysis(
                current_performance, test_results
            )
            # Обновление кандидата
            candidate.proposed_performance = float(np.mean(test_results))
            candidate.improvement = improvement
            candidate.test_results = test_results
            candidate.statistical_significance = significance
            candidate.confidence = self._calculate_confidence(
                test_results, significance
            )
            logger.info(f"Evolution candidate results for {component_name}:")
            logger.info(f"  Improvement: {improvement:.4f} ({improvement*100:.2f}%)")
            logger.info(f"  Statistical significance: {significance:.4f}")
            logger.info(f"  Confidence: {candidate.confidence:.4f}")
            # Проверка критериев подтверждения
            if self._meets_confirmation_criteria(candidate):
                logger.info(
                    f"Evolution confirmed for {component_name} - " f"applying changes"
                )
                self.confirmed_evolutions.append(candidate)
                candidate.status = "confirmed"
                # Обновление статистики
                self._update_validation_stats(candidate, True)
                return True
            else:
                logger.warning(
                    f"Evolution rejected for {component_name} - "
                    f"insufficient improvement"
                )
                await self._rollback_component(component, backup_path)
                candidate.status = "rejected"
                # Обновление статистики
                self._update_validation_stats(candidate, False)
                return False
        except Exception as e:
            logger.error(f"Error validating evolution candidate: {e}")
            if "backup_path" in locals():
                await self._rollback_component(component, backup_path)
            return False

    async def _apply_proposed_changes(
        self, component: Any, changes: Dict[str, Any]
    ) -> bool:
        """
        Применение предложенных изменений.
        Args:
            component: Компонент для изменения
            changes: Изменения для применения
        Returns:
            True если изменения применены успешно, False в противном случае
        """
        try:
            # Здесь должна быть логика применения изменений
            # Например, изменение архитектуры модели, параметров и т.д.
            # Временная реализация - просто вызываем evolve
            test_data = {"proposed_changes": changes, "timestamp": datetime.now()}
            return await component.evolve(test_data)
        except Exception as e:
            logger.error(f"Error applying proposed changes: {e}")
            return False

    async def _test_evolution_candidate(self, component: Any) -> List[float]:
        """
        Тестирование кандидата на эволюцию.
        Args:
            component: Компонент для тестирования
        Returns:
            Список результатов тестирования
        """
        try:
            test_results = []
            test_data = await self._get_test_data(component.name)
            # Множественные тестовые сценарии
            for scenario_id in range(self.config.min_test_samples):
                result = await self._test_component_scenario(
                    component, test_data, scenario_id
                )
                test_results.append(result)
            return test_results
        except Exception as e:
            logger.error(f"Error testing evolution candidate: {e}")
            return []

    async def _get_test_data(self, component_name: str) -> Any:
        """
        Получение тестовых данных для компонента.
        Args:
            component_name: Имя компонента
        Returns:
            Тестовые данные
        """
        try:
            # Здесь должна быть логика получения тестовых данных
            # Временная реализация - возвращаем фиктивные данные
            return {
                "test_scenarios": [
                    {"id": i, "data": np.random.rand(100)} 
                    for i in range(self.config.min_test_samples)
                ],
                "component_name": component_name,
                "timestamp": datetime.now(),
            }
        except Exception as e:
            logger.error(f"Error getting test data: {e}")
            return {}

    async def _test_component_scenario(
        self, component: Any, test_data: Any, scenario_id: int
    ) -> float:
        """
        Тестирование компонента на конкретном сценарии.
        Args:
            component: Компонент для тестирования
            test_data: Тестовые данные
            scenario_id: ID сценария
        Returns:
            Результат тестирования (производительность)
        """
        try:
            # Здесь должна быть логика тестирования компонента
            # Временная реализация - симуляция тестирования
            scenario_data = test_data.get("test_scenarios", [])
            if scenario_id < len(scenario_data):
                scenario = scenario_data[scenario_id]
                # Симуляция обработки данных
                processed_data = pd.DataFrame(scenario["data"])
                # Симуляция расчета производительности
                if hasattr(processed_data, 'to_numpy'):
                    performance = float(np.mean(processed_data.to_numpy()) + np.random.normal(0, 0.01))
                elif hasattr(processed_data, 'values'):
                    # Исправление: безопасное получение numpy array из DataFrame
                    if hasattr(processed_data, 'to_numpy'):
                        data_array = processed_data.to_numpy()
                    else:
                        data_array = np.asarray(processed_data.values)
                    performance = float(np.mean(data_array) + np.random.normal(0, 0.01))
                else:
                    # Если это не pandas DataFrame, используем альтернативный способ
                    performance = float(np.mean(list(processed_data.values())) + np.random.normal(0, 0.01))
                return max(0.0, min(1.0, performance))
            else:
                # Fallback - случайная производительность
                return float(np.random.uniform(0.5, 0.9))
        except Exception as e:
            logger.warning(f"Error in test scenario {scenario_id}: {e}")
            return 0.0

    def _statistical_analysis(
        self, current_performance: float, test_results: List[float]
    ) -> Tuple[float, float]:
        """
        Статистический анализ улучшений.
        Args:
            current_performance: Текущая производительность
            test_results: Результаты тестирования
        Returns:
            Кортеж (улучшение, статистическая значимость)
        """
        try:
            if len(test_results) < 2:
                return 0.0, 1.0
            # Расчет улучшения
            mean_test_performance = float(np.mean(test_results))
            improvement: float = float(mean_test_performance - current_performance)
            # Статистический тест (t-test)
            if len(test_results) > 1:
                # Создаем контрольную группу с текущей производительностью
                control_group = [current_performance] * len(test_results)
                # t-test для сравнения
                t_stat, p_value = stats.ttest_ind(test_results, control_group)
                significance = 1.0 - p_value  # Инвертируем p-value
            else:
                significance = 0.0
            return improvement, float(significance)
        except Exception as e:
            logger.error(f"Error in statistical analysis: {e}")
            return 0.0, 0.0

    def _calculate_confidence(
        self, test_results: List[float], significance: float
    ) -> float:
        """
        Расчет уверенности в результатах.
        Args:
            test_results: Результаты тестирования
            significance: Статистическая значимость
        Returns:
            Уровень уверенности
        """
        try:
            if len(test_results) < 2:
                return 0.0
            # Стабильность результатов
            stability = float(1.0 - np.std(test_results))
            # Количество тестов
            test_coverage = min(1.0, len(test_results) / self.config.min_test_samples)
            # Комбинированная уверенность
            confidence = 0.4 * stability + 0.4 * significance + 0.2 * test_coverage
            return float(max(0.0, min(1.0, confidence)))
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0

    def _meets_confirmation_criteria(self, candidate: EvolutionCandidate) -> bool:
        """
        Проверка критериев подтверждения.
        Args:
            candidate: Кандидат на эволюцию
        Returns:
            True если критерии выполнены, False в противном случае
        """
        try:
            # Минимальное улучшение
            if candidate.improvement < self.config.efficiency_improvement_threshold:
                logger.info(
                    f"Insufficient improvement: {candidate.improvement:.4f} < "
                    f"{self.config.efficiency_improvement_threshold}"
                )
                return False
            # Статистическая значимость
            if candidate.statistical_significance < (
                1.0 - self.config.statistical_significance_level
            ):
                logger.info(
                    f"Insignificant improvement: "
                    f"{candidate.statistical_significance:.4f}"
                )
                return False
            # Минимальная уверенность
            if candidate.confidence < self.config.confidence_threshold:
                logger.info(f"Low confidence: {candidate.confidence:.4f}")
                return False
            # Минимальное количество тестов
            if len(candidate.test_results) < self.config.min_test_samples:
                logger.info(f"Insufficient test samples: {len(candidate.test_results)}")
                return False
            # Стабильность результатов
            stability = 1.0 - np.std(candidate.test_results)
            if stability < self.config.stability_threshold:
                logger.info(f"Unstable results: {stability:.4f}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking confirmation criteria: {e}")
            return False

    async def _rollback_component(self, component: Any, backup_path: str) -> bool:
        """
        Откат компонента к предыдущему состоянию.
        Args:
            component: Компонент для отката
            backup_path: Путь к резервной копии
        Returns:
            True если откат успешен, False в противном случае
        """
        try:
            logger.info(f"Rolling back {component.name} to previous state")
            success = component.load_state(backup_path)
            if success:
                logger.info(f"Successfully rolled back {component.name}")
                # Запись в историю откатов
                rollback_record = {
                    "component_name": component.name,
                    "timestamp": datetime.now(),
                    "reason": "Failed evolution validation",
                    "backup_path": backup_path,
                }
                self.rollback_history.append(rollback_record)
                return True
            else:
                logger.error(f"Failed to rollback {component.name}")
                return False
        except Exception as e:
            logger.error(f"Error rolling back component: {e}")
            return False

    def _update_validation_stats(self, candidate: EvolutionCandidate, success: bool) -> None:
        """
        Обновление статистики валидации.
        Args:
            candidate: Кандидат на эволюцию
            success: Успешность валидации
        """
        try:
            if "total_validations" not in self.validation_stats:
                self.validation_stats = {
                    "total_validations": 0,
                    "successful_validations": 0,
                    "failed_validations": 0,
                    "average_improvement": 0.0,
                    "average_confidence": 0.0,
                    "last_update": datetime.now(),
                }
            self.validation_stats["total_validations"] += 1
            if success:
                self.validation_stats["successful_validations"] += 1
            else:
                self.validation_stats["failed_validations"] += 1
            # Обновление средних значений
            if self.confirmed_evolutions:
                improvements = [c.improvement for c in self.confirmed_evolutions]
                confidences = [c.confidence for c in self.confirmed_evolutions]
                self.validation_stats["average_improvement"] = np.mean(improvements)
                self.validation_stats["average_confidence"] = np.mean(confidences)
            self.validation_stats["last_update"] = datetime.now()
        except Exception as e:
            logger.error(f"Error updating validation stats: {e}")

    def get_validation_stats(self) -> Dict[str, Any]:
        """
        Получение статистики валидации.
        Returns:
            Словарь со статистикой валидации
        """
        try:
            stats = {
                "testing_candidates": len(self.testing_candidates),
                "confirmed_evolutions": len(self.confirmed_evolutions),
                "rollback_count": len(self.rollback_history),
                "success_rate": len(self.confirmed_evolutions)
                / max(1, len(self.testing_candidates) + len(self.confirmed_evolutions)),
                "recent_rollbacks": (
                    self.rollback_history[-5:] if self.rollback_history else []
                ),
                "validation_stats": self.validation_stats,
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting validation stats: {e}")
            return {}

    def save_validation_log(self) -> None:
        """
        Сохранение лога валидации.
        Сохраняет результаты валидации в JSON файл для последующего анализа.
        """
        try:
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "confirmed_evolutions": [c.__dict__ for c in self.confirmed_evolutions],
                "rollback_history": self.rollback_history,
                "validation_stats": self.validation_stats,
            }
            log_file = (
                f"{self.config.validation_log_path}/validation_log_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, default=str)
            logger.info(f"Validation log saved to {log_file}")
        except Exception as e:
            logger.error(f"Error saving validation log: {e}")

    def load_validation_log(self, log_file: str) -> Dict[str, Any]:
        """
        Загрузка лога валидации.
        Args:
            log_file: Путь к файлу лога
        Returns:
            Загруженные данные лога
        """
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                log_data = json.load(f)
            logger.info(f"Validation log loaded from {log_file}")
            return log_data
        except Exception as e:
            logger.error(f"Error loading validation log: {e}")
            return {}

    def get_evolution_history(self) -> List[Dict[str, Any]]:
        """
        Получение истории эволюций.
        Returns:
            Список записей об эволюциях
        """
        try:
            history = []
            for evolution in self.confirmed_evolutions:
                history.append({
                    "component_name": evolution.component_name,
                    "improvement": evolution.improvement,
                    "confidence": evolution.confidence,
                    "timestamp": evolution.timestamp,
                    "status": evolution.status,
                })
            return history
        except Exception as e:
            logger.error(f"Error getting evolution history: {e}")
            return []

    def get_rollback_history(self) -> List[Dict[str, Any]]:
        """
        Получение истории откатов.
        Returns:
            List[Dict[str, Any]]: История откатов
        """
        return self.rollback_history

    async def validate_improvement(
        self, component: Any, improvement_threshold: float
    ) -> Dict[str, Any]:
        """
        Валидация улучшения компонента.
        Args:
            component: Компонент для валидации
            improvement_threshold: Порог улучшения
        Returns:
            Dict[str, Any]: Результат валидации
        """
        try:
            # Получение текущей производительности
            current_performance = component.get_performance()
            # Тестирование компонента
            test_results = await self._test_evolution_candidate(component)
            if not test_results:
                return {
                    "is_improved": False,
                    "improvement": 0.0,
                    "confidence": 0.0,
                    "reason": "No test results"
                }
            # Статистический анализ
            improvement, significance = self._statistical_analysis(
                current_performance, test_results
            )
            # Расчет уверенности
            confidence = self._calculate_confidence(test_results, significance)
            # Проверка улучшения
            is_improved = (
                improvement >= improvement_threshold and 
                confidence >= self.config.confidence_threshold
            )
            return {
                "is_improved": is_improved,
                "improvement": improvement,
                "confidence": confidence,
                "significance": significance,
                "test_results_count": len(test_results)
            }
        except Exception as e:
            logger.error(f"Error validating improvement: {e}")
            return {
                "is_improved": False,
                "improvement": 0.0,
                "confidence": 0.0,
                "reason": str(e)
            }


# Глобальный экземпляр валидатора
efficiency_validator = EfficiencyValidator()
