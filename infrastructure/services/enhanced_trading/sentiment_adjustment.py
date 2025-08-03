"""
Модуль анализа и корректировки настроений рынка.
Содержит функции для анализа настроений и корректировки торговых параметров.
"""

# ВАЖНО: Для корректной работы mypy с pandas используйте pandas-stubs: pip install pandas-stubs

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union

# Type aliases for better mypy support
# Удаляю алиасы Series = pd.Series, DataFrame = pd.DataFrame


# Простые определения для совместимости
class SentimentType:
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SentimentSource:
    NEWS = "news"
    SOCIAL = "social"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    VOLUME = "volume"
    VOLATILITY = "volatility"


class SentimentConfidence:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


__all__ = [
    "analyze_market_sentiment",
    "analyze_news_sentiment",
    "analyze_social_sentiment",
    "calculate_sentiment_score",
    "adjust_trading_parameters",
    "detect_sentiment_shifts",
    "validate_sentiment_data",
    "combine_sentiment_sources",
    "generate_sentiment_alerts",
]


def validate_sentiment_data(data: pd.DataFrame) -> bool:
    """Валидация данных настроений."""
    if data is None or data.empty:
        return False
    required_columns = ["timestamp", "sentiment_score"]
    if not all(col in data.columns for col in required_columns):
        return False
    # Проверяем диапазон sentiment_score
    if "sentiment_score" in data.columns:
        scores = data["sentiment_score"]
        if scores.min() < -1 or scores.max() > 1:
            return False
    return True


def analyze_market_sentiment(
    market_data: pd.DataFrame, sentiment_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Анализ настроений рынка."""
    sentiment_analysis: Dict[str, Any] = {
        "overall_sentiment": SentimentType.NEUTRAL,
        "sentiment_score": Decimal("0"),
        "confidence": SentimentConfidence.LOW,
        "sources": [],
        "trend": "stable",
        "timestamp": datetime.now(),
    }
    if market_data.empty:
        return sentiment_analysis
    # Анализируем технические индикаторы настроений
    technical_sentiment = analyze_technical_sentiment(market_data)
    sentiment_analysis["sources"].append(
        {
            "source": SentimentSource.TECHNICAL,
            "score": technical_sentiment["score"],
            "confidence": technical_sentiment["confidence"],
        }
    )
    # Анализируем объёмные индикаторы
    volume_sentiment = analyze_volume_sentiment(market_data)
    sentiment_analysis["sources"].append(
        {
            "source": SentimentSource.VOLUME,
            "score": volume_sentiment["score"],
            "confidence": volume_sentiment["confidence"],
        }
    )
    # Анализируем волатильность
    volatility_sentiment = analyze_volatility_sentiment(market_data)
    sentiment_analysis["sources"].append(
        {
            "source": SentimentSource.VOLATILITY,
            "score": volatility_sentiment["score"],
            "confidence": volatility_sentiment["confidence"],
        }
    )
    # Объединяем источники
    combined_sentiment = combine_sentiment_sources(sentiment_analysis["sources"])
    sentiment_analysis["overall_sentiment"] = combined_sentiment["sentiment_type"]
    sentiment_analysis["sentiment_score"] = combined_sentiment["score"]
    sentiment_analysis["confidence"] = combined_sentiment["confidence"]
    # Определяем тренд
    sentiment_analysis["trend"] = detect_sentiment_trend(sentiment_data)
    return sentiment_analysis


def analyze_technical_sentiment(market_data: pd.DataFrame) -> Dict[str, Any]:
    """Анализ технических настроений."""
    if market_data.empty or "close" not in market_data.columns:
        return {"score": Decimal("0"), "confidence": SentimentConfidence.LOW}
    # Рассчитываем технические индикаторы
    close = market_data["close"]
    # RSI
    rsi = calculate_rsi(close)
    rsi_sentiment = analyze_rsi_sentiment(rsi)
    # MACD
    macd_sentiment = analyze_macd_sentiment(close)
    # Скользящие средние
    ma_sentiment = analyze_moving_averages_sentiment(close)
    # Объединяем технические сигналы
    technical_score = (rsi_sentiment + macd_sentiment + ma_sentiment) / 3
    return {
        "score": convert_to_decimal(technical_score),
        "confidence": SentimentConfidence.MEDIUM,
    }


def analyze_rsi_sentiment(rsi: pd.Series) -> float:
    """Анализ настроений по RSI."""
    if rsi.empty:
        return 0.0
    if len(rsi) > 0:
        current_rsi = float(rsi.iloc[-1])
    else:
        return 0.0
    if current_rsi > 70.0:
        return -0.5  # Перекупленность - негативные настроения
    elif current_rsi < 30.0:
        return 0.5  # Перепроданность - позитивные настроения
    elif 40.0 <= current_rsi <= 60.0:
        return 0.0  # Нейтральные настроения
    else:
        return 0.1  # Слабые позитивные настроения


def analyze_macd_sentiment(close: pd.Series) -> float:
    """Анализ настроений по MACD."""
    if close.empty:
        return 0.0
    # Упрощённый MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    if len(macd) < 2:
        return 0.0
    current_macd = macd.iloc[-1]
    current_signal = signal.iloc[-1]
    prev_macd = macd.iloc[-2]
    prev_signal = signal.iloc[-2]
    # Сигнал на покупку
    if current_macd > current_signal and prev_macd <= prev_signal:
        return 0.3
    # Сигнал на продажу
    elif current_macd < current_signal and prev_macd >= prev_signal:
        return -0.3
    # Позитивный тренд
    elif current_macd > 0 and current_macd > current_signal:
        return 0.2
    # Негативный тренд
    elif current_macd < 0 and current_macd < current_signal:
        return -0.2
    else:
        return 0.0


def analyze_moving_averages_sentiment(close: pd.Series) -> float:
    """Анализ настроений по скользящим средним."""
    if close.empty:
        return 0.0
    sma20 = close.rolling(window=20).mean()
    sma50 = close.rolling(window=50).mean()
    if len(sma20) < 1 or len(sma50) < 1:
        return 0.0
    if len(close) > 0 and len(sma20) > 0 and len(sma50) > 0:
        current_price = float(close.iloc[-1])
        current_sma20 = float(sma20.iloc[-1])
        current_sma50 = float(sma50.iloc[-1])
    else:
        return 0.0
    # Золотой крест
    if current_sma20 > current_sma50 and current_price > current_sma20:
        return 0.4
    # Мёртвый крест
    elif current_sma20 < current_sma50 and current_price < current_sma20:
        return -0.4
    # Позитивный тренд
    elif current_price > current_sma20 and current_sma20 > current_sma50:
        return 0.2
    # Негативный тренд
    elif current_price < current_sma20 and current_sma20 < current_sma50:
        return -0.2
    else:
        return 0.0


def analyze_volume_sentiment(market_data: pd.DataFrame) -> Dict[str, Any]:
    """Анализ объёмных настроений."""
    if market_data.empty or "volume" not in market_data.columns:
        return {"score": Decimal("0"), "confidence": SentimentConfidence.LOW}
    volume = market_data["volume"]
    close = market_data["close"]
    # Анализируем объём
    avg_volume = volume.rolling(window=20).mean()
    volume_ratio = volume / avg_volume
    # Анализируем цену и объём
    price_change = close.pct_change()
    volume_price_correlation = price_change.rolling(window=10).corr(volume_ratio)
    if hasattr(volume_ratio, 'iloc') and hasattr(volume_price_correlation, 'iloc'):
        current_volume_ratio = float(volume_ratio.iloc[-1]) if not volume_ratio.empty else 1.0
        current_correlation = (
            float(volume_price_correlation.iloc[-1]) if not volume_price_correlation.empty else 0.0
        )
    else:
        current_volume_ratio = 1.0
        current_correlation = 0.0
    # Высокий объём с ростом цены - позитивные настроения
    if current_volume_ratio > 1.5 and current_correlation > 0.3:
        score = 0.3
    # Высокий объём с падением цены - негативные настроения
    elif current_volume_ratio > 1.5 and current_correlation < -0.3:
        score = -0.3
    # Низкий объём - нейтральные настроения
    elif current_volume_ratio < 0.5:
        score = 0.0
    else:
        score = 0.1
    return {
        "score": convert_to_decimal(score),
        "confidence": SentimentConfidence.MEDIUM,
    }


def analyze_volatility_sentiment(market_data: pd.DataFrame) -> Dict[str, Any]:
    """Анализ настроений по волатильности."""
    if market_data.empty or "close" not in market_data.columns:
        return {"score": Decimal("0"), "confidence": SentimentConfidence.LOW}
    close = market_data["close"]
    returns = close.pct_change().dropna()
    if returns.empty:
        return {"score": Decimal("0"), "confidence": SentimentConfidence.LOW}
    # Рассчитываем волатильность
    volatility = returns.rolling(window=20).std()
    avg_volatility = volatility.mean()
    if hasattr(volatility, 'iloc'):
        current_volatility = float(volatility.iloc[-1]) if not volatility.empty else float(avg_volatility)
    else:
        current_volatility = float(avg_volatility)
    # Низкая волатильность - стабильные настроения
    if hasattr(current_volatility, '__lt__') and hasattr(avg_volatility, '__mul__'):
        if current_volatility < avg_volatility * 0.7:
            score = 0.2
        # Высокая волатильность - нестабильные настроения
        elif current_volatility > avg_volatility * 1.5:
            score = -0.2
        else:
            score = 0.0
    else:
        score = 0.0
    return {
        "score": convert_to_decimal(score),
        "confidence": SentimentConfidence.MEDIUM,
    }


def analyze_news_sentiment(
    news_data: List[Dict[str, Any]], market_data: pd.DataFrame
) -> Dict[str, Any]:
    """Анализ настроений новостей."""
    if not news_data:
        return {
            "score": Decimal("0"),
            "confidence": SentimentConfidence.LOW,
            "sources": [],
        }
    # Анализируем каждую новость
    news_sentiments = []
    for news in news_data:
        sentiment = analyze_single_news(news)
        news_sentiments.append(sentiment)
    # Объединяем настроения новостей
    if news_sentiments:
        avg_score = sum(s["score"] for s in news_sentiments) / len(news_sentiments)
        confidence = (
            SentimentConfidence.HIGH
            if len(news_sentiments) > 5
            else SentimentConfidence.MEDIUM
        )
    else:
        avg_score = 0.0
        confidence = SentimentConfidence.LOW
    return {
        "score": convert_to_decimal(avg_score),
        "confidence": confidence,
        "sources": news_sentiments,
    }


def analyze_single_news(news: Dict[str, Any]) -> Dict[str, Any]:
    """Анализ настроений отдельной новости."""
    # Упрощённый анализ на основе ключевых слов
    text = news.get("title", "") + " " + news.get("content", "")
    text_lower = text.lower()
    # Позитивные ключевые слова
    positive_words = [
        "growth",
        "profit",
        "gain",
        "rise",
        "positive",
        "bullish",
        "recovery",
    ]
    # Негативные ключевые слова
    negative_words = [
        "loss",
        "decline",
        "fall",
        "negative",
        "bearish",
        "crash",
        "crisis",
    ]
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    # Рассчитываем настроения
    if positive_count > negative_count:
        score = min(1.0, positive_count / 10.0)
    elif negative_count > positive_count:
        score = max(-1.0, -negative_count / 10.0)
    else:
        score = 0.0
    return {
        "source": SentimentSource.NEWS,
        "score": convert_to_decimal(score),
        "confidence": SentimentConfidence.MEDIUM,
        "timestamp": news.get("timestamp", datetime.now()),
    }


def analyze_social_sentiment(social_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Анализ настроений социальных сетей."""
    if not social_data:
        return {
            "score": Decimal("0"),
            "confidence": SentimentConfidence.LOW,
            "sources": [],
        }
    # Анализируем каждое сообщение
    social_sentiments = []
    for post in social_data:
        sentiment = analyze_single_social_post(post)
        social_sentiments.append(sentiment)
    # Объединяем настроения
    if social_sentiments:
        avg_score = sum(s["score"] for s in social_sentiments) / len(social_sentiments)
        confidence = (
            SentimentConfidence.MEDIUM
            if len(social_sentiments) > 10
            else SentimentConfidence.LOW
        )
    else:
        avg_score = 0.0
        confidence = SentimentConfidence.LOW
    return {
        "score": convert_to_decimal(avg_score),
        "confidence": confidence,
        "sources": social_sentiments,
    }


def analyze_single_social_post(post: Dict[str, Any]) -> Dict[str, Any]:
    """Анализ настроений отдельного поста в соцсети."""
    text = post.get("text", "")
    text_lower = text.lower()
    # Эмодзи и символы настроений
    positive_emojis = ["😊", "😄", "👍", "🚀", "💪", "bull", "moon"]
    negative_emojis = ["😞", "😢", "👎", "💩", "bear", "dump"]
    positive_count = sum(1 for emoji in positive_emojis if emoji in text_lower)
    negative_count = sum(1 for emoji in negative_emojis if emoji in text_lower)
    # Рассчитываем настроения
    if positive_count > negative_count:
        score = min(1.0, positive_count / 5.0)
    elif negative_count > positive_count:
        score = max(-1.0, -negative_count / 5.0)
    else:
        score = 0.0
    return {
        "source": SentimentSource.SOCIAL,
        "score": convert_to_decimal(score),
        "confidence": SentimentConfidence.LOW,
        "timestamp": post.get("timestamp", datetime.now()),
    }


def combine_sentiment_sources(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Объединение источников настроений."""
    if not sources:
        return {
            "sentiment_type": SentimentType.NEUTRAL,
            "score": Decimal("0"),
            "confidence": SentimentConfidence.LOW,
        }
    # Взвешенное среднее на основе уверенности
    total_weight = 0.0
    weighted_score = 0.0
    for source in sources:
        confidence_weight = {
            SentimentConfidence.LOW: 0.5,
            SentimentConfidence.MEDIUM: 1.0,
            SentimentConfidence.HIGH: 1.5,
        }.get(source["confidence"], 1.0)
        weighted_score += float(source["score"]) * confidence_weight
        total_weight += confidence_weight
    if total_weight > 0:
        final_score = weighted_score / total_weight
    else:
        final_score = 0.0
    # Определяем тип настроений
    if final_score > 0.3:
        sentiment_type = SentimentType.POSITIVE
    elif final_score < -0.3:
        sentiment_type = SentimentType.NEGATIVE
    else:
        sentiment_type = SentimentType.NEUTRAL
    # Определяем общую уверенность
    avg_confidence = sum(
        (
            0.5
            if s["confidence"] == SentimentConfidence.LOW
            else (
                1.0
                if s["confidence"] == SentimentConfidence.MEDIUM
                else 1.5
            )
        )
        for s in sources
    ) / len(sources)
    if avg_confidence > 1.5:
        overall_confidence = SentimentConfidence.HIGH
    elif avg_confidence > 0.5:
        overall_confidence = SentimentConfidence.MEDIUM
    else:
        overall_confidence = SentimentConfidence.LOW
    return {
        "sentiment_type": sentiment_type,
        "score": convert_to_decimal(final_score),
        "confidence": overall_confidence,
    }


def detect_sentiment_trend(sentiment_data: Optional[pd.DataFrame]) -> str:
    """Определение тренда настроений."""
    if sentiment_data is None or sentiment_data.empty:
        return "stable"
    if "sentiment_score" not in sentiment_data.columns:
        return "stable"
    scores = sentiment_data["sentiment_score"].dropna()
    if len(scores) < 5:
        return "stable"
    # Рассчитываем тренд
    recent_scores = scores.tail(5)
    trend_slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
    if trend_slope > 0.1:
        return "improving"
    elif trend_slope < -0.1:
        return "deteriorating"
    else:
        return "stable"


def detect_sentiment_shifts(
    current_sentiment: Dict[str, Any],
    historical_sentiment: pd.DataFrame,
    threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """Обнаружение сдвигов настроений."""
    shifts: List[Dict[str, Any]] = []
    if (
        historical_sentiment.empty
        or "sentiment_score" not in historical_sentiment.columns
    ):
        return shifts
    current_score = float(current_sentiment.get("sentiment_score", 0))
    historical_scores = historical_sentiment["sentiment_score"].dropna()
    if len(historical_scores) < 10:
        return shifts
    # Рассчитываем среднее и стандартное отклонение
    mean_score = historical_scores.mean()
    std_score = historical_scores.std()
    # Проверяем значительные отклонения
    if abs(current_score - mean_score) > threshold * std_score:
        shift_type = "positive" if current_score > mean_score else "negative"
        magnitude = abs(current_score - mean_score) / std_score
        shifts.append(
            {
                "type": shift_type,
                "magnitude": convert_to_decimal(magnitude),
                "current_score": convert_to_decimal(current_score),
                "historical_mean": convert_to_decimal(mean_score),
                "timestamp": datetime.now(),
                "significance": "high" if magnitude > 2 else "medium",
            }
        )
    return shifts


def adjust_trading_parameters(
    base_parameters: Dict[str, Any], sentiment_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Корректировка торговых параметров на основе настроений."""
    adjusted_parameters = base_parameters.copy()
    sentiment_score = float(sentiment_analysis.get("sentiment_score", 0))
    sentiment_type = sentiment_analysis.get("overall_sentiment", SentimentType.NEUTRAL)
    confidence = sentiment_analysis.get("confidence", SentimentConfidence.LOW)
    # Корректируем размер позиции
    if "position_size" in adjusted_parameters:
        base_size = adjusted_parameters["position_size"]
        if sentiment_type == SentimentType.POSITIVE:
            size_multiplier = 1.0 + sentiment_score * 0.5
        elif sentiment_type == SentimentType.NEGATIVE:
            size_multiplier = 1.0 - abs(sentiment_score) * 0.3
        else:
            size_multiplier = 1.0
        adjusted_parameters["position_size"] = base_size * convert_to_decimal(
            size_multiplier
        )
    # Корректируем стоп-лосс
    if "stop_loss" in adjusted_parameters:
        base_stop_loss = adjusted_parameters["stop_loss"]
        if sentiment_type == SentimentType.POSITIVE:
            stop_loss_multiplier = 1.0 - sentiment_score * 0.2  # Уменьшаем стоп-лосс
        elif sentiment_type == SentimentType.NEGATIVE:
            stop_loss_multiplier = (
                1.0 + abs(sentiment_score) * 0.3
            )  # Увеличиваем стоп-лосс
        else:
            stop_loss_multiplier = 1.0
        adjusted_parameters["stop_loss"] = base_stop_loss * convert_to_decimal(
            stop_loss_multiplier
        )
    # Корректируем тейк-профит
    if "take_profit" in adjusted_parameters:
        base_take_profit = adjusted_parameters["take_profit"]
        if sentiment_type == SentimentType.POSITIVE:
            take_profit_multiplier = (
                1.0 + sentiment_score * 0.3
            )  # Увеличиваем тейк-профит
        elif sentiment_type == SentimentType.NEGATIVE:
            take_profit_multiplier = (
                1.0 - abs(sentiment_score) * 0.2
            )  # Уменьшаем тейк-профит
        else:
            take_profit_multiplier = 1.0
        adjusted_parameters["take_profit"] = base_take_profit * convert_to_decimal(
            take_profit_multiplier
        )
    # Добавляем метаданные о корректировке
    adjusted_parameters["sentiment_adjustment"] = {
        "original_sentiment_score": sentiment_score,
        "sentiment_type": sentiment_type,
        "confidence": confidence,
        "adjustment_timestamp": datetime.now(),
    }
    return adjusted_parameters


def generate_sentiment_alerts(
    sentiment_analysis: Dict[str, Any], sentiment_shifts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Генерация алертов на основе настроений."""
    alerts = []
    # Алерты на основе сдвигов настроений
    for shift in sentiment_shifts:
        if shift["significance"] == "high":
            alerts.append(
                {
                    "type": "sentiment_shift",
                    "severity": "high",
                    "message": f"Major {shift['type']} sentiment shift detected (magnitude: {shift['magnitude']:.2f})",
                    "timestamp": shift["timestamp"],
                    "data": shift,
                }
            )
    # Алерты на основе экстремальных настроений
    sentiment_score = float(sentiment_analysis.get("sentiment_score", 0))
    if sentiment_score > 0.7:
        alerts.append(
            {
                "type": "extreme_sentiment",
                "severity": "medium",
                "message": f"Extremely positive sentiment detected (score: {sentiment_score:.2f})",
                "timestamp": datetime.now(),
                "data": {"sentiment_score": sentiment_score},
            }
        )
    elif sentiment_score < -0.7:
        alerts.append(
            {
                "type": "extreme_sentiment",
                "severity": "medium",
                "message": f"Extremely negative sentiment detected (score: {sentiment_score:.2f})",
                "timestamp": datetime.now(),
                "data": {"sentiment_score": sentiment_score},
            }
        )
    # Алерты на основе тренда
    trend = sentiment_analysis.get("trend", "stable")
    if trend == "deteriorating":
        alerts.append(
            {
                "type": "sentiment_trend",
                "severity": "low",
                "message": "Sentiment trend is deteriorating",
                "timestamp": datetime.now(),
                "data": {"trend": trend},
            }
        )
    return alerts


def calculate_sentiment_score(
    market_sentiment: Dict[str, Any],
    news_sentiment: Optional[Dict[str, Any]] = None,
    social_sentiment: Optional[Dict[str, Any]] = None,
) -> Decimal:
    """Расчёт общего показателя настроений."""
    scores = []
    weights = []
    # Рыночные настроения (вес 0.6)
    if market_sentiment:
        scores.append(float(market_sentiment.get("sentiment_score", 0)))
        weights.append(0.6)
    # Новостные настроения (вес 0.3)
    if news_sentiment:
        scores.append(float(news_sentiment.get("score", 0)))
        weights.append(0.3)
    # Социальные настроения (вес 0.1)
    if social_sentiment:
        scores.append(float(social_sentiment.get("score", 0)))
        weights.append(0.1)
    if not scores:
        return Decimal("0")
    # Взвешенное среднее
    weighted_score = sum(score * weight for score, weight in zip(scores, weights))
    total_weight = sum(weights)
    return convert_to_decimal(weighted_score / total_weight)


# Вспомогательные функции
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Расчёт RSI."""
    if prices.empty:
        return pd.Series()
    delta = prices.diff()
    # Используем pandas методы для безопасного сравнения
    gain = delta.where(delta.gt(0), 0)
    loss = delta.where(delta.lt(0), 0).abs()
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def convert_to_decimal(value: float) -> Decimal:
    """Конвертация в Decimal."""
    return Decimal(str(value))
