"""Запускатор экспериментов."""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from shared.numpy_utils import np
from loguru import logger
from scipy import stats

from domain.types.entity_system_types import ExperimentData, ExperimentResult


class ExperimentMetadata:
    """Метаданные эксперимента."""
    
    def __init__(self, hypothesis: Dict[str, Any]) -> None:
        self.id: str = str(uuid.uuid4())
        self.hypothesis: Dict[str, Any] = hypothesis
        self.status: str = "created"
        self.created_at: str = datetime.now().isoformat()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.control_group: Dict[str, List[float]] = {}
        self.treatment_group: Dict[str, List[float]] = {}
        self.metrics: Dict[str, List[float]] = {
            "performance": [],
            "accuracy": [],
            "latency": [],
            "memory_usage": [],
            "error_rate": [],
        }
        self.statistical_results: Dict[str, Any] = {}
        self.confidence_interval: Dict[str, Any] = {}
        self.p_value: Optional[float] = None
        self.effect_size: Optional[float] = None
        self.overall_result: Optional[Dict[str, Any]] = None


class ExperimentRunner:
    """Запускатор экспериментов."""

    def __init__(self, experiments_dir: str = "entity/experiments") -> None:
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.experiments: Dict[str, ExperimentMetadata] = {}
        self.results: Dict[str, ExperimentResult] = {}
        self.active_experiments: List[str] = []
        # Конфигурация экспериментов
        self.config: Dict[str, Any] = {
            "min_test_duration": 300,  # 5 минут
            "max_test_duration": 3600,  # 1 час
            "min_sample_size": 50,
            "confidence_level": 0.95,
            "improvement_threshold": 0.05,  # 5%
            "max_concurrent_experiments": 3,
        }
        logger.info("ExperimentRunner initialized")

    async def create_experiment(
        self, hypothesis: Dict[str, Any]
    ) -> Optional[ExperimentMetadata]:
        """Создание эксперимента на основе гипотезы."""
        experiment = ExperimentMetadata(hypothesis)
        self.experiments[experiment.id] = experiment
        logger.info(
            f"Создан эксперимент {experiment.id} для гипотезы: {hypothesis['title']}"
        )
        return experiment

    async def start_experiment(self, experiment_id: str) -> bool:
        """Запуск эксперимента."""
        if experiment_id not in self.experiments:
            logger.error(f"Эксперимент {experiment_id} не найден")
            return False
        experiment = self.experiments[experiment_id]
        experiment.status = "running"
        experiment.started_at = datetime.now().isoformat()
        self.active_experiments.append(experiment_id)
        logger.info(f"Запущен эксперимент {experiment_id}")
        return True

    async def stop_experiment(self, experiment_id: str) -> bool:
        """Остановка эксперимента."""
        if experiment_id not in self.experiments:
            logger.error(f"Эксперимент {experiment_id} не найден")
            return False
        experiment = self.experiments[experiment_id]
        experiment.status = "completed"
        experiment.completed_at = datetime.now().isoformat()
        if experiment_id in self.active_experiments:
            self.active_experiments.remove(experiment_id)
        # Анализ результатов
        await self._analyze_experiment_results(experiment_id)
        logger.info(f"Завершен эксперимент {experiment_id}")
        return True

    async def _analyze_experiment_results(self, experiment_id: str) -> None:
        """Анализ результатов эксперимента."""
        experiment = self.experiments[experiment_id]
        # Статистический анализ
        control_data = experiment.control_group
        treatment_data = experiment.treatment_group
        if not control_data or not treatment_data:
            logger.warning(
                f"Недостаточно данных для анализа эксперимента {experiment_id}"
            )
            return
        # Расчет статистических показателей
        for metric in experiment.metrics:
            control_values = control_data.get(metric, [])
            treatment_values = treatment_data.get(metric, [])
            if control_values and treatment_values:
                statistical_result = await self._calculate_statistical_significance(
                    control_values, treatment_values
                )
                experiment.statistical_results[metric] = statistical_result
        # Общий результат эксперимента
        experiment.overall_result = await self._calculate_overall_result(experiment)
        # Создаем ExperimentResult
        self.results[experiment_id] = self._create_experiment_result(experiment)

    def _create_experiment_result(self, experiment: ExperimentMetadata) -> ExperimentResult:
        """Создание результата эксперимента."""
        # Берем первый значимый результат для создания ExperimentResult
        first_result: Dict[str, Any] = next(iter(experiment.statistical_results.values()), {})
        
        return ExperimentResult(
            experiment_id=experiment.id,
            test_name=experiment.hypothesis.get("title", "Unknown"),
            status="completed" if experiment.status == "completed" else "running",
            control_sample_size=len(cast(list, next(iter(experiment.control_group.values()), []))),
            treatment_sample_size=len(cast(list, next(iter(experiment.treatment_group.values()), []))),
            control_mean=float(np.mean(cast(list, next(iter(experiment.control_group.values()), [0.0])))),
            treatment_mean=float(np.mean(cast(list, next(iter(experiment.treatment_group.values()), [0.0])))),
            improvement_percent=float(first_result.get("effect_size", 0.0) * 100),
            significant=first_result.get("significant", False),
            p_value=first_result.get("p_value"),
            confidence_interval=first_result.get("confidence_interval"),
            analysis_date=datetime.now(),
        )

    async def _calculate_statistical_significance(
        self, control: List[float], treatment: List[float]
    ) -> Dict[str, Any]:
        """Расчет статистической значимости."""
        try:
            # t-тест
            t_stat, p_value = stats.ttest_ind(control, treatment)
            # Эффект размера (Cohen's d)
            pooled_std = np.sqrt(
                (
                    (len(control) - 1) * np.var(control, ddof=1)
                    + (len(treatment) - 1) * np.var(treatment, ddof=1)
                )
                / (len(control) + len(treatment) - 2)
            )
            effect_size = (np.mean(treatment) - np.mean(control)) / pooled_std
            # Доверительный интервал
            confidence_interval = stats.t.interval(
                0.95,
                len(control) + len(treatment) - 2,
                loc=float(np.mean(treatment) - np.mean(control)),
                scale=float(pooled_std * np.sqrt(1 / len(control) + 1 / len(treatment))),
            )
            return {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "effect_size": float(effect_size),
                "confidence_interval": [float(ci) for ci in confidence_interval],
                "significant": p_value < 0.05,
                "control_mean": float(np.mean(control)),
                "treatment_mean": float(np.mean(treatment)),
                "control_std": float(np.std(control)),
                "treatment_std": float(np.std(treatment)),
            }
        except Exception as e:
            logger.error(f"Ошибка расчета статистической значимости: {e}")
            return {"error": str(e), "significant": False}

    async def _calculate_overall_result(
        self, experiment: ExperimentMetadata
    ) -> Dict[str, Any]:
        """Расчет общего результата эксперимента."""
        statistical_results = experiment.statistical_results
        if not statistical_results:
            return {"success": False, "reason": "Нет статистических результатов"}
        # Подсчет значимых улучшений
        significant_improvements = 0
        total_metrics = len(statistical_results)
        for metric, result in statistical_results.items():
            if result.get("significant", False) and result.get("effect_size", 0) > 0:
                significant_improvements += 1
        success_rate = float(significant_improvements / total_metrics) if total_metrics > 0 else 0.0
        # Определение успешности эксперимента
        hypothesis = experiment.hypothesis
        expected_improvement = hypothesis.get("expected_improvement", 0.1)
        # Проверка достижения ожидаемого улучшения
        achieved_improvement = 0.0
        for result in statistical_results.values():
            if result.get("effect_size", 0) > 0:
                achieved_improvement += result["effect_size"]
        achieved_improvement /= total_metrics if total_metrics > 0 else 1
        success = success_rate >= 0.6 and achieved_improvement >= expected_improvement
        return {
            "success": success,
            "success_rate": success_rate,
            "achieved_improvement": achieved_improvement,
            "expected_improvement": expected_improvement,
            "significant_improvements": significant_improvements,
            "total_metrics": total_metrics,
        }

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """Получение эксперимента по ID."""
        return self.experiments.get(experiment_id)

    def get_active_experiments(self) -> List[ExperimentMetadata]:
        """Получение активных экспериментов."""
        return [
            self.experiments[exp_id]
            for exp_id in self.active_experiments
            if exp_id in self.experiments
        ]

    def get_completed_experiments(self) -> List[ExperimentMetadata]:
        """Получение завершенных экспериментов."""
        return [
            exp for exp in self.experiments.values() if exp.status == "completed"
        ]
