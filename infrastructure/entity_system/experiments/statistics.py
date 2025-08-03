"""Статистический анализатор для экспериментов."""

from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import numpy as np
from loguru import logger
from scipy import stats

from domain.types.entity_system_types import StatisticalResult


class StatisticalAnalyzer:
    """Анализатор статистических данных экспериментов."""

    def __init__(self) -> None:
        self.confidence_level: float = 0.95
        self.significance_threshold: float = 0.05

    async def calculate_statistical_significance(
        self, control: List[float], treatment: List[float]
    ) -> StatisticalResult:
        """Расчет статистической значимости между группами."""
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
                self.confidence_level,
                len(control) + len(treatment) - 2,
                loc=float(np.mean(treatment) - np.mean(control)),
                scale=float(pooled_std * np.sqrt(1 / len(control) + 1 / len(treatment))),
            )
            confidence_interval_float = [float(ci) for ci in cast(tuple, confidence_interval)]
            return StatisticalResult(
                test_name="t_test",
                test_type="t_test",
                statistic=float(t_stat),
                p_value=float(p_value),
                effect_size=float(effect_size),
                confidence_interval=confidence_interval_float,
                significant=bool(p_value < self.significance_threshold),
                sample_sizes={"control": len(control), "treatment": len(treatment)},
                means={
                    "control": float(np.mean(control)),
                    "treatment": float(np.mean(treatment)),
                },
                standard_deviations={
                    "control": float(np.std(control)),
                    "treatment": float(np.std(treatment)),
                },
                analysis_date=datetime.now(),
            )
        except Exception as e:
            logger.error(f"Ошибка расчета статистической значимости: {e}")
            # Возвращаем пустой результат при ошибке
            return StatisticalResult(
                test_name="error",
                test_type="error",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                confidence_interval=[0.0, 0.0],
                significant=False,
                sample_sizes={"control": 0, "treatment": 0},
                means={"control": 0.0, "treatment": 0.0},
                standard_deviations={"control": 0.0, "treatment": 0.0},
                analysis_date=datetime.now(),
            )

    async def calculate_power_analysis(
        self, control: List[float], treatment: List[float], alpha: float = 0.05
    ) -> Dict[str, float]:
        """Расчет мощности теста."""
        try:
            # Расчет размера эффекта
            pooled_std = np.sqrt(
                (
                    (len(control) - 1) * np.var(control, ddof=1)
                    + (len(treatment) - 1) * np.var(treatment, ddof=1)
                )
                / (len(control) + len(treatment) - 2)
            )
            effect_size = abs(np.mean(treatment) - np.mean(control)) / pooled_std
            # Расчет мощности (упрощенный)
            n = len(control) + len(treatment)
            # Используем упрощенный расчет мощности
            power = float(self._calculate_simple_power(float(effect_size), n, alpha))
            return {
                "power": power,
                "effect_size": float(effect_size),
                "sample_size": float(n),
                "alpha": alpha,
            }
        except Exception as e:
            logger.error(f"Ошибка расчета мощности теста: {e}")
            return {
                "power": 0.0,
                "effect_size": 0.0,
                "sample_size": 0.0,
                "alpha": alpha,
            }

    def _calculate_simple_power(
        self, effect_size: float, sample_size: int, alpha: float
    ) -> float:
        """Упрощенный расчет мощности теста."""
        # Простая аппроксимация мощности
        z_alpha = 1.96  # Для alpha = 0.05 (используем alpha параметр)
        if alpha != 0.05:
            # Для других значений alpha можно использовать stats.norm.ppf(1 - alpha/2)
            z_alpha = stats.norm.ppf(1 - alpha / 2)  # type: ignore[assignment]
        z_beta = effect_size * np.sqrt(sample_size / 2) - z_alpha
        power = float(1 - stats.norm.cdf(z_beta))
        return float(max(0.0, min(1.0, power)))

    async def calculate_sample_size(
        self, effect_size: float, power: float = 0.8, alpha: float = 0.05
    ) -> Dict[str, float]:
        """Расчет необходимого размера выборки."""
        try:
            # Упрощенный расчет размера выборки
            z_alpha = 1.96  # Для alpha = 0.05
            z_beta = stats.norm.ppf(1 - power)
            n_per_group = 2 * ((z_alpha + z_beta) / effect_size) ** 2
            return {
                "sample_size_per_group": float(n_per_group),
                "total_sample_size": float(n_per_group * 2),
                "effect_size": effect_size,
                "power": power,
                "alpha": alpha,
            }
        except Exception as e:
            logger.error(f"Ошибка расчета размера выборки: {e}")
            return {
                "sample_size_per_group": 0.0,
                "total_sample_size": 0.0,
                "effect_size": effect_size,
                "power": power,
                "alpha": alpha,
            }

    async def calculate_bayesian_analysis(
        self, control: List[float], treatment: List[float]
    ) -> Dict[str, Any]:
        """Байесовский анализ результатов."""
        try:
            # Простой байесовский анализ с нормальными распределениями
            control_mean = float(np.mean(control))
            treatment_mean = float(np.mean(treatment))
            control_std = float(np.std(control))
            treatment_std = float(np.std(treatment))
            # Расчет байесовского фактора (упрощенный)
            pooled_std = np.sqrt((control_std**2 + treatment_std**2) / 2)
            z_score = (treatment_mean - control_mean) / pooled_std
            # Приближенный байесовский фактор
            bayes_factor = np.exp(z_score**2 / 2)
            return {
                "bayes_factor": float(bayes_factor),
                "z_score": float(z_score),
                "control_mean": float(control_mean),
                "treatment_mean": float(treatment_mean),
                "evidence_strength": self._interpret_bayes_factor(bayes_factor),
            }
        except Exception as e:
            logger.error(f"Ошибка байесовского анализа: {e}")
            return {"error": str(e)}

    def _interpret_bayes_factor(self, bayes_factor: float) -> str:
        """Интерпретация байесовского фактора."""
        if bayes_factor > 100:
            return "decisive"
        elif bayes_factor > 30:
            return "very_strong"
        elif bayes_factor > 10:
            return "strong"
        elif bayes_factor > 3:
            return "moderate"
        elif bayes_factor > 1:
            return "weak"
        else:
            return "negative"
