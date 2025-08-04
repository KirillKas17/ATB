"""
Анализатор метрик производительности.
"""

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from shared.numpy_utils import np
import pandas as pd
from loguru import logger
from scipy import stats


class MetricType(Enum):
    """Типы метрик."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class TrendDirection(Enum):
    """Направления тренда."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class AnomalyType(Enum):
    """Типы аномалий."""

    SPIKE = "spike"
    DROP = "drop"
    TREND_BREAK = "trend_break"
    OUTLIER = "outlier"
    PATTERN_CHANGE = "pattern_change"


@dataclass
class MetricPoint:
    """Точка метрики."""

    timestamp: datetime
    value: float
    component: str
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Анализ тренда."""

    direction: TrendDirection
    slope: float
    confidence: float
    start_value: float
    end_value: float
    duration: timedelta
    volatility: float
    r_squared: float


@dataclass
class AnomalyDetection:
    """Обнаружение аномалии."""

    anomaly_type: AnomalyType
    timestamp: datetime
    value: float
    expected_value: float
    deviation: float
    severity: float
    confidence: float
    description: str


@dataclass
class MetricSummary:
    """Сводка метрики."""

    name: str
    component: str
    count: int
    min_value: float
    max_value: float
    mean_value: float
    median_value: float
    std_dev: float
    variance: float
    skewness: float
    kurtosis: float
    trend: Optional[TrendAnalysis] = None
    anomalies: List[AnomalyDetection] = field(default_factory=list)


class MetricsAnalyzer:
    """
    Анализатор метрик производительности.
    
    Предоставляет функциональность для сбора, анализа, обнаружения
    аномалии и генерирует отчёты.
    """

    def __init__(self) -> None:
        self.metrics_data: Dict[str, List[MetricPoint]] = defaultdict(list)
        self.analysis_cache: Dict[str, MetricSummary] = {}
        self.anomaly_threshold = 2.0  # Стандартные отклонения для аномалий
        self.trend_confidence_threshold = 0.7
        self.cache_ttl = timedelta(hours=1)

    def add_metric_point(
        self,
        name: str,
        timestamp: datetime,
        value: float,
        component: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Добавить точку метрики.
        Args:
            name: Название метрики
            timestamp: Временная метка
            value: Значение
            component: Компонент
            tags: Дополнительные теги
        """
        point = MetricPoint(
            timestamp=timestamp, value=value, component=component, tags=tags or {}
        )
        self.metrics_data[name].append(point)
        # Очищаем кэш для этой метрики
        if name in self.analysis_cache:
            del self.analysis_cache[name]

    def get_metric_data(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[MetricPoint]:
        """
        Получить данные метрики.
        Args:
            name: Название метрики
            start_time: Начальное время
            end_time: Конечное время
        Returns:
            Список точек метрики
        """
        if name not in self.metrics_data:
            return []
        data = self.metrics_data[name]
        if start_time:
            data = [point for point in data if point.timestamp >= start_time]
        if end_time:
            data = [point for point in data if point.timestamp <= end_time]
        return sorted(data, key=lambda x: x.timestamp)

    def analyze_metric(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        force_recalculate: bool = False,
    ) -> MetricSummary:
        """
        Анализ метрики.
        Args:
            name: Название метрики
            start_time: Начальное время
            end_time: Конечное время
            force_recalculate: Принудительный пересчёт
        Returns:
            Сводка анализа метрики
        """
        # Проверяем кэш
        cache_key = f"{name}_{start_time}_{end_time}"
        if not force_recalculate and cache_key in self.analysis_cache:
            cached = self.analysis_cache[cache_key]
            # Убираем проверку timestamp, так как его нет в MetricSummary
            return cached
        
        data = self.get_metric_data(name, start_time, end_time)
        if not data:
            raise ValueError(f"No data found for metric: {name}")
        
        # Базовая статистика
        values = [point.value for point in data]
        summary = MetricSummary(
            name=name,
            component=data[0].component,
            count=len(values),
            min_value=min(values),
            max_value=max(values),
            mean_value=statistics.mean(values),
            median_value=statistics.median(values),
            std_dev=statistics.stdev(values) if len(values) > 1 else 0.0,
            variance=statistics.variance(values) if len(values) > 1 else 0.0,
            skewness=self._calculate_skewness(values),
            kurtosis=self._calculate_kurtosis(values),
        )
        
        # Анализ тренда
        if len(data) > 1:
            summary.trend = self._analyze_trend(data)
        
        # Обнаружение аномалий
        summary.anomalies = self._detect_anomalies(data, summary)
        
        # Кэшируем результат
        self.analysis_cache[cache_key] = summary
        
        return summary

    def _calculate_skewness(self, values: List[float]) -> float:
        """Вычислить асимметрию."""
        if len(values) < 3:
            return 0.0
        return float(stats.skew(values))

    def _calculate_kurtosis(self, values: List[float]) -> float:
        """Вычислить эксцесс."""
        if len(values) < 4:
            return 0.0
        return float(stats.kurtosis(values))

    def _analyze_trend(self, data: List[MetricPoint]) -> Optional[TrendAnalysis]:
        """Анализ тренда."""
        if len(data) < 2:
            return None
        
        # Подготовка данных
        timestamps = [(point.timestamp - data[0].timestamp).total_seconds() for point in data]
        values = [point.value for point in data]
        
        # Линейная регрессия
        slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, values)
        
        # Определение направления тренда
        if abs(slope) < 0.001:
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING
        
        # Вычисление волатильности
        volatility = float(np.std(values))
        
        # Вычисление R-squared
        r_squared = float(r_value ** 2)
        
        # Вычисление доверительного интервала
        confidence = float(1 - p_value) if p_value is not None else 0.0
        
        return TrendAnalysis(
            direction=direction,
            slope=float(slope),
            confidence=confidence,
            start_value=float(values[0]),
            end_value=float(values[-1]),
            duration=timedelta(seconds=timestamps[-1] - timestamps[0]),
            volatility=volatility,
            r_squared=r_squared,
        )

    def _detect_anomalies(
        self, data: List[MetricPoint], summary: MetricSummary
    ) -> List[AnomalyDetection]:
        """Обнаружение аномалий."""
        if len(data) < 3:
            return []
        anomalies = []
        values = [point.value for point in data]
        mean = summary.mean_value
        std = summary.std_dev
        if std == 0:
            return []
        # Z-score для каждой точки
        z_scores = [(value - mean) / std for value in values]
        # Обнаружение выбросов
        for i, (point, z_score) in enumerate(zip(data, z_scores)):
            if abs(z_score) > self.anomaly_threshold:
                anomaly_type = AnomalyType.SPIKE if z_score > 0 else AnomalyType.DROP
                severity = min(abs(z_score) / self.anomaly_threshold, 3.0)
                anomaly = AnomalyDetection(
                    anomaly_type=anomaly_type,
                    timestamp=point.timestamp,
                    value=point.value,
                    expected_value=mean,
                    deviation=z_score,
                    severity=severity,
                    confidence=min(abs(z_score) / 4.0, 1.0),
                    description=f"{anomaly_type.value} detected: z-score = {z_score:.2f}",
                )
                anomalies.append(anomaly)
        # Обнаружение изменений тренда
        if summary.trend and len(values) > 10:
            trend_anomalies = self._detect_trend_breaks(data, summary.trend)
            anomalies.extend(trend_anomalies)
        return anomalies

    def _detect_trend_breaks(
        self, data: List[MetricPoint], trend: TrendAnalysis
    ) -> List[AnomalyDetection]:
        """Обнаружение разрывов тренда."""
        anomalies: List[AnomalyDetection] = []
        values = [point.value for point in data]
        if len(values) < 10:
            return anomalies
        # Скользящее среднее для сглаживания
        window_size = min(5, len(values) // 2)
        smoothed_values = []
        for i in range(len(values)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(values), i + window_size // 2 + 1)
            smoothed_values.append(statistics.mean(values[start_idx:end_idx]))
        # Поиск отклонений от тренда
        for i, (point, smoothed_value) in enumerate(zip(data, smoothed_values)):
            expected_value = trend.start_value + trend.slope * i
            deviation = abs(smoothed_value - expected_value)
            if deviation > trend.volatility * 2:
                anomaly = AnomalyDetection(
                    anomaly_type=AnomalyType.TREND_BREAK,
                    timestamp=point.timestamp,
                    value=point.value,
                    expected_value=expected_value,
                    deviation=deviation,
                    severity=min(deviation / trend.volatility, 3.0),
                    confidence=min(deviation / (trend.volatility * 3), 1.0),
                    description=f"Trend break detected at {point.timestamp}",
                )
                anomalies.append(anomaly)
        return anomalies

    def compare_metrics(
        self,
        metric1: str,
        metric2: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Сравнение двух метрик.
        Args:
            metric1: Первая метрика
            metric2: Вторая метрика
            start_time: Начальное время
            end_time: Конечное время
        Returns:
            Результат сравнения
        """
        data1 = self.get_metric_data(metric1, start_time, end_time)
        data2 = self.get_metric_data(metric2, start_time, end_time)
        if not data1 or not data2:
            raise ValueError("Insufficient data for comparison")
        values1 = [point.value for point in data1]
        values2 = [point.value for point in data2]
        # Корреляция
        if len(values1) == len(values2):
            correlation = np.corrcoef(values1, values2)[0, 1]
        else:
            correlation = 0.0
        # Статистическое сравнение
        comparison = {
            "correlation": correlation,
            "metric1_summary": self.analyze_metric(metric1, start_time, end_time),
            "metric2_summary": self.analyze_metric(metric2, start_time, end_time),
            "mean_difference": statistics.mean(values1) - statistics.mean(values2),
            "variance_ratio": (
                statistics.variance(values1) / statistics.variance(values2)
                if statistics.variance(values2) > 0
                else float("inf")
            ),
        }
        return comparison

    def generate_report(
        self,
        metrics: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Генерация отчёта по метрикам.
        Args:
            metrics: Список метрик
            start_time: Начальное время
            end_time: Конечное время
        Returns:
            Отчёт
        """
        report: Dict[str, Any] = {
            "generated_at": datetime.now(),
            "time_range": {"start": start_time, "end": end_time},
            "metrics_analyzed": len(metrics),
            "summaries": {},
            "anomalies_summary": {
                "total": 0,
                "by_type": defaultdict(int),
                "by_severity": defaultdict(int),
            },
            "trends_summary": {
                "increasing": 0,
                "decreasing": 0,
                "stable": 0,
                "volatile": 0,
            },
        }
        total_anomalies = 0
        for metric in metrics:
            try:
                summary = self.analyze_metric(metric, start_time, end_time)
                report["summaries"][metric] = summary
                # Подсчитываем аномалии
                total_anomalies += len(summary.anomalies)
                for anomaly in summary.anomalies:
                    anomalies_summary = report["anomalies_summary"]
                    if isinstance(anomalies_summary, dict):
                        by_type = anomalies_summary.get("by_type", {})
                        by_severity = anomalies_summary.get("by_severity", {})
                        if isinstance(by_type, dict):
                            by_type[anomaly.anomaly_type.value] = by_type.get(anomaly.anomaly_type.value, 0) + 1
                        severity_level = (
                            "low"
                            if anomaly.severity < 1.5
                            else "medium" if anomaly.severity < 2.5 else "high"
                        )
                        if isinstance(by_severity, dict):
                            by_severity[severity_level] = by_severity.get(severity_level, 0) + 1
                # Подсчитываем тренды
                if summary.trend:
                    trends_summary = report["trends_summary"]
                    if isinstance(trends_summary, dict):
                        trends_summary[summary.trend.direction.value] = trends_summary.get(summary.trend.direction.value, 0) + 1
            except Exception as e:
                logger.error(f"Error analyzing metric {metric}: {e}")
                report["summaries"][metric] = {"error": str(e)}
        anomalies_summary = report["anomalies_summary"]
        if isinstance(anomalies_summary, dict):
            anomalies_summary["total"] = total_anomalies
        return report

    def export_data(
        self,
        metrics: List[str],
        format: str = "csv",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> str:
        """
        Экспорт данных метрик.
        Args:
            metrics: Список метрик
            format: Формат экспорта
            start_time: Начальное время
            end_time: Конечное время
        Returns:
            Экспортированные данные
        """
        if format == "csv":
            all_data: List[Dict[str, Any]] = []
            for metric in metrics:
                data = self.get_metric_data(metric, start_time, end_time)
                for point in data:
                    all_data.append(
                        {
                            "metric": metric,
                            "timestamp": point.timestamp,
                            "value": point.value,
                            "component": point.component,
                            "tags": str(point.tags),
                        }
                    )
            df = pd.DataFrame(all_data)
            if hasattr(df, 'to_csv'):
                csv_result = df.to_csv(index=False)
                return str(csv_result) if csv_result is not None else ""
            else:
                # Альтернативный способ экспорта
                import csv
                import io
                output = io.StringIO()
                if all_data:
                    writer = csv.DictWriter(output, fieldnames=all_data[0].keys())
                    writer.writeheader()
                    writer.writerows(all_data)
                return output.getvalue()
        elif format == "json":
            json_data: Dict[str, Any] = {
                "metrics": {
                    metric: [
                        {
                            "timestamp": point.timestamp.isoformat(),
                            "value": point.value,
                            "component": point.component,
                            "tags": point.tags,
                        }
                        for point in self.get_metric_data(metric, start_time, end_time)
                    ]
                    for metric in metrics
                }
            }
            return json.dumps(json_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Глобальный экземпляр анализатора
metrics_analyzer = MetricsAnalyzer()


def analyze_metric_performance(
    name: str, data: List[Tuple[datetime, float]], component: str = "unknown"
) -> MetricSummary:
    """
    Быстрый анализ производительности метрики.
    Args:
        name: Название метрики
        data: Список кортежей (timestamp, value)
        component: Компонент
    Returns:
        Сводка анализа
    """
    # Добавляем данные в анализатор
    for timestamp, value in data:
        metrics_analyzer.add_metric_point(name, timestamp, value, component)
    # Выполняем анализ
    return metrics_analyzer.analyze_metric(name)


def detect_performance_anomalies(
    values: List[float], timestamps: List[datetime], threshold: float = 2.0
) -> List[AnomalyDetection]:
    """
    Быстрое обнаружение аномалий в производительности.
    Args:
        values: Список значений
        timestamps: Список временных меток
        threshold: Порог для аномалий
    Returns:
        Список обнаруженных аномалий
    """
    if len(values) != len(timestamps):
        raise ValueError("Values and timestamps must have the same length")
    anomalies = []
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0
    if std == 0:
        return []
    for i, (value, timestamp) in enumerate(zip(values, timestamps)):
        z_score = abs(value - mean) / std
        if z_score > threshold:
            anomaly_type = AnomalyType.SPIKE if value > mean else AnomalyType.DROP
            anomaly = AnomalyDetection(
                anomaly_type=anomaly_type,
                timestamp=timestamp,
                value=value,
                expected_value=mean,
                deviation=z_score,
                severity=min(z_score / threshold, 3.0),
                confidence=min(z_score / 4.0, 1.0),
                description=f"{anomaly_type.value} detected: z-score = {z_score:.2f}",
            )
            anomalies.append(anomaly)
    return anomalies
