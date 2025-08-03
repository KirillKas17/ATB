"""
Основной модуль интеграции новостной торговли.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

from shared.logging import setup_logger

from .types import (
    NewsAnalysis,
    NewsCategory,
    NewsImpact,
    NewsItem,
    NewsTradingConfig,
    TradingSignal,
)


class NewsTradingIntegration:
    """
    Интеграция новостной торговли для анализа и торговли на основе новостей.
    """

    def __init__(self, config: Optional[NewsTradingConfig] = None):
        """
        Инициализация интеграции новостной торговли.
        :param config: конфигурация новостной торговли
        """
        self.config = config or NewsTradingConfig()

        # Хранилище новостей и анализов
        self.news_items: Dict[str, NewsItem] = {}
        self.news_analyses: Dict[str, NewsAnalysis] = {}
        self.symbol_analyses: Dict[str, List[NewsAnalysis]] = {}

        # Статистика
        self.stats = {
            "total_news": 0,
            "total_analyses": 0,
            "trading_signals": 0,
            "auto_trades": 0,
        }

        # Асинхронные задачи
        self.analysis_task: Optional[asyncio.Task] = None
        self.trading_task: Optional[asyncio.Task] = None
        self.is_running = False

        logger.info("NewsTradingIntegration initialized")

    async def start(self) -> None:
        """Запуск интеграции новостной торговли."""
        try:
            if self.is_running:
                return

            self.is_running = True

            # Запуск анализа новостей
            self.analysis_task = asyncio.create_task(self._news_analysis_loop())

            # Запуск торговли на основе новостей
            if self.config.enable_auto_trading:
                self.trading_task = asyncio.create_task(self._news_trading_loop())

            logger.info("NewsTradingIntegration started")

        except Exception as e:
            logger.error(f"Error starting NewsTradingIntegration: {e}")
            self.is_running = False

    async def stop(self) -> None:
        """Остановка интеграции новостной торговли."""
        try:
            if not self.is_running:
                return

            self.is_running = False

            # Отмена задач
            if self.analysis_task:
                self.analysis_task.cancel()
                try:
                    await self.analysis_task
                except asyncio.CancelledError:
                    pass

            if self.trading_task:
                self.trading_task.cancel()
                try:
                    await self.trading_task
                except asyncio.CancelledError:
                    pass

            logger.info("NewsTradingIntegration stopped")

        except Exception as e:
            logger.error(f"Error stopping NewsTradingIntegration: {e}")

    async def add_news_item(self, news_item: NewsItem) -> bool:
        """Добавление новостного элемента."""
        try:
            if not self.config.enable_news_trading:
                return True

            # Проверяем возраст новости
            news_age = (datetime.now() - news_item.timestamp).total_seconds() / 3600
            if news_age > self.config.max_news_age_hours:
                logger.warning(
                    f"News item {news_item.news_id} is too old ({news_age:.1f} hours)"
                )
                return False

            # Сохраняем новость
            self.news_items[news_item.news_id] = news_item

            # Анализируем новость для каждого символа
            for symbol in news_item.symbols:
                analysis = await self._analyze_news_for_symbol(news_item, symbol)
                if analysis:
                    self.news_analyses[analysis.analysis_id] = analysis

                    if symbol not in self.symbol_analyses:
                        self.symbol_analyses[symbol] = []
                    self.symbol_analyses[symbol].append(analysis)

            self.stats["total_news"] += 1

            if self.config.log_news_analysis:
                logger.info(
                    f"Added news item: {news_item.title[:50]}... "
                    f"for symbols: {news_item.symbols}"
                )

            return True

        except Exception as e:
            logger.error(f"Error adding news item {news_item.news_id}: {e}")
            return False

    async def get_news_analysis(
        self, symbol: str, hours: int = 24
    ) -> List[NewsAnalysis]:
        """Получение анализа новостей для символа."""
        try:
            if symbol not in self.symbol_analyses:
                return []

            cutoff_time = datetime.now() - timedelta(hours=hours)

            analyses = [
                analysis
                for analysis in self.symbol_analyses[symbol]
                if analysis.timestamp >= cutoff_time
            ]

            # Сортируем по времени
            analyses.sort(key=lambda x: x.timestamp, reverse=True)

            return analyses

        except Exception as e:
            logger.error(f"Error getting news analysis for {symbol}: {e}")
            return []

    async def get_trading_signals(self, symbol: str) -> List[Dict[str, Any]]:
        """Получение торговых сигналов на основе новостей."""
        try:
            analyses = await self.get_news_analysis(
                symbol, hours=6
            )  # Последние 6 часов

            if not analyses:
                return []

            signals = []

            for analysis in analyses:
                # Проверяем пороги
                if (
                    abs(analysis.sentiment_score) >= self.config.sentiment_threshold
                    and analysis.impact_score >= self.config.impact_threshold
                ):

                    signal = {
                        "symbol": symbol,
                        "signal": analysis.trading_signal.value,
                        "confidence": analysis.confidence,
                        "sentiment_score": analysis.sentiment_score,
                        "impact_score": analysis.impact_score,
                        "reasoning": analysis.reasoning,
                        "timestamp": analysis.timestamp,
                        "news_id": analysis.news_id,
                    }

                    signals.append(signal)
                    self.stats["trading_signals"] += 1

            return signals

        except Exception as e:
            logger.error(f"Error getting trading signals for {symbol}: {e}")
            return []

    async def should_trade_on_news(self, symbol: str) -> Dict[str, Any]:
        """Определение, следует ли торговать на основе новостей."""
        try:
            signals = await self.get_trading_signals(symbol)

            if not signals:
                return {
                    "should_trade": False,
                    "reason": "no_signals",
                    "confidence": 0.0,
                }

            # Агрегируем сигналы
            total_sentiment = sum(s["sentiment_score"] for s in signals)
            avg_confidence = sum(s["confidence"] for s in signals) / len(signals)
            avg_impact = sum(s["impact_score"] for s in signals) / len(signals)

            # Определяем общий сигнал
            if total_sentiment > 0.5 and avg_confidence > 0.7:
                overall_signal = "buy"
                should_trade = True
            elif total_sentiment < -0.5 and avg_confidence > 0.7:
                overall_signal = "sell"
                should_trade = True
            else:
                overall_signal = "hold"
                should_trade = False

            return {
                "should_trade": should_trade,
                "signal": overall_signal,
                "confidence": avg_confidence,
                "sentiment": total_sentiment,
                "impact": avg_impact,
                "signal_count": len(signals),
            }

        except Exception as e:
            logger.error(f"Error checking should trade on news for {symbol}: {e}")
            return {"should_trade": False, "reason": "error", "confidence": 0.0}

    async def _analyze_news_for_symbol(
        self, news_item: NewsItem, symbol: str
    ) -> Optional[NewsAnalysis]:
        """Анализ новости для конкретного символа."""
        try:
            # Анализ настроений
            sentiment_score = self._analyze_sentiment(news_item, symbol)

            # Анализ влияния
            impact_score = self._analyze_impact(news_item, symbol)

            # Определение торгового сигнала
            trading_signal = self._determine_trading_signal(
                sentiment_score, impact_score
            )

            # Формирование обоснования
            reasoning = self._generate_reasoning(
                news_item, sentiment_score, impact_score
            )

            # Вычисление уверенности
            confidence = self._calculate_confidence(
                news_item, sentiment_score, impact_score
            )

            analysis = NewsAnalysis(
                analysis_id=f"{news_item.news_id}_{symbol}",
                news_id=news_item.news_id,
                symbol=symbol,
                sentiment_score=sentiment_score,
                impact_score=impact_score,
                trading_signal=trading_signal,
                confidence=confidence,
                reasoning=reasoning,
                timestamp=datetime.now(),
            )

            self.stats["total_analyses"] += 1

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing news for symbol {symbol}: {e}")
            return None

    def _analyze_sentiment(self, news_item: NewsItem, symbol: str) -> float:
        """Анализ настроений новости."""
        try:
            # Базовый скор настроений
            base_sentiment = news_item.sentiment_score

            # Корректировка на основе категории
            category_multipliers = {
                NewsCategory.REGULATORY: 1.5,
                NewsCategory.TECHNICAL: 1.2,
                NewsCategory.FUNDAMENTAL: 1.3,
                NewsCategory.SENTIMENT: 1.0,
                NewsCategory.MARKET: 1.1,
                NewsCategory.POLITICAL: 1.4,
                NewsCategory.ECONOMIC: 1.2,
            }

            multiplier = category_multipliers.get(news_item.category, 1.0)
            adjusted_sentiment = base_sentiment * multiplier

            # Ограничиваем в диапазоне [-1, 1]
            return max(-1.0, min(1.0, adjusted_sentiment))

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0

    def _analyze_impact(self, news_item: NewsItem, symbol: str) -> float:
        """Анализ влияния новости."""
        try:
            # Базовый скор влияния на основе уровня
            impact_scores = {
                NewsImpact.LOW: 0.2,
                NewsImpact.MEDIUM: 0.5,
                NewsImpact.HIGH: 0.8,
                NewsImpact.CRITICAL: 1.0,
            }

            base_impact = impact_scores.get(news_item.impact, 0.5)

            # Корректировка на основе уверенности
            confidence_adjustment = news_item.confidence * 0.3

            # Корректировка на основе категории
            category_impacts = {
                NewsCategory.REGULATORY: 0.3,
                NewsCategory.TECHNICAL: 0.1,
                NewsCategory.FUNDAMENTAL: 0.2,
                NewsCategory.SENTIMENT: 0.0,
                NewsCategory.MARKET: 0.1,
                NewsCategory.POLITICAL: 0.4,
                NewsCategory.ECONOMIC: 0.2,
            }

            category_adjustment = category_impacts.get(news_item.category, 0.0)

            total_impact = base_impact + confidence_adjustment + category_adjustment

            # Ограничиваем в диапазоне [0, 1]
            return max(0.0, min(1.0, total_impact))

        except Exception as e:
            logger.error(f"Error analyzing impact: {e}")
            return 0.5

    def _determine_trading_signal(
        self, sentiment_score: float, impact_score: float
    ) -> TradingSignal:
        """Определение торгового сигнала."""
        try:
            # Комбинированный скор
            combined_score = sentiment_score * impact_score

            if combined_score > 0.6:
                return TradingSignal.STRONG_BUY
            elif combined_score > 0.2:
                return TradingSignal.BUY
            elif combined_score < -0.6:
                return TradingSignal.STRONG_SELL
            elif combined_score < -0.2:
                return TradingSignal.SELL
            else:
                return TradingSignal.HOLD

        except Exception as e:
            logger.error(f"Error determining trading signal: {e}")
            return TradingSignal.HOLD

    def _generate_reasoning(
        self, news_item: NewsItem, sentiment_score: float, impact_score: float
    ) -> List[str]:
        """Генерация обоснования."""
        try:
            reasoning = []

            # Обоснование на основе настроений
            if sentiment_score > 0.5:
                reasoning.append("Strong positive sentiment")
            elif sentiment_score > 0.2:
                reasoning.append("Positive sentiment")
            elif sentiment_score < -0.5:
                reasoning.append("Strong negative sentiment")
            elif sentiment_score < -0.2:
                reasoning.append("Negative sentiment")
            else:
                reasoning.append("Neutral sentiment")

            # Обоснование на основе влияния
            if impact_score > 0.8:
                reasoning.append("High impact news")
            elif impact_score > 0.5:
                reasoning.append("Medium impact news")
            else:
                reasoning.append("Low impact news")

            # Обоснование на основе категории
            reasoning.append(f"Category: {news_item.category.value}")

            # Обоснование на основе источника
            if (
                "reuters" in news_item.source.lower()
                or "bloomberg" in news_item.source.lower()
            ):
                reasoning.append("Reliable source")

            return reasoning

        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            return ["Analysis error"]

    def _calculate_confidence(
        self, news_item: NewsItem, sentiment_score: float, impact_score: float
    ) -> float:
        """Вычисление уверенности в анализе."""
        try:
            # Базовая уверенность
            base_confidence = news_item.confidence

            # Корректировка на основе силы сигнала
            signal_strength = abs(sentiment_score) * impact_score
            strength_adjustment = signal_strength * 0.2

            # Корректировка на основе источника
            source_confidence = 0.0
            if "reuters" in news_item.source.lower():
                source_confidence = 0.1
            elif "bloomberg" in news_item.source.lower():
                source_confidence = 0.1
            elif "coindesk" in news_item.source.lower():
                source_confidence = 0.05

            total_confidence = base_confidence + strength_adjustment + source_confidence

            # Ограничиваем в диапазоне [0, 1]
            return max(0.0, min(1.0, total_confidence))

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5

    async def _news_analysis_loop(self) -> None:
        """Цикл анализа новостей."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Каждую минуту

                # Очистка старых новостей
                await self._cleanup_old_news()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in news analysis loop: {e}")

    async def _news_trading_loop(self) -> None:
        """Цикл торговли на основе новостей."""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Каждые 30 секунд

                # Получаем все символы
                symbols = list(self.symbol_analyses.keys())

                # Проверяем торговые сигналы
                for symbol in symbols:
                    try:
                        trade_decision = await self.should_trade_on_news(symbol)

                        if (
                            trade_decision["should_trade"]
                            and trade_decision["confidence"] > 0.8
                        ):
                            await self._execute_news_trade(symbol, trade_decision)

                    except Exception as e:
                        logger.error(f"Error in news trading loop for {symbol}: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in news trading loop: {e}")

    async def _execute_news_trade(
        self, symbol: str, trade_decision: Dict[str, Any]
    ) -> None:
        """Выполнение торговли на основе новостей."""
        try:
            # Здесь должна быть интеграция с торговой системой
            logger.info(
                f"News-based trade signal for {symbol}: "
                f"{trade_decision['signal']} (confidence: {trade_decision['confidence']:.2f})"
            )

            self.stats["auto_trades"] += 1

            # Ждем указанную задержку
            await asyncio.sleep(self.config.trading_delay_seconds)

        except Exception as e:
            logger.error(f"Error executing news trade for {symbol}: {e}")

    async def _cleanup_old_news(self) -> None:
        """Очистка старых новостей."""
        try:
            cutoff_time = datetime.now() - timedelta(
                hours=self.config.max_news_age_hours
            )

            # Очищаем старые новости
            old_news_ids = [
                news_id
                for news_id, news_item in self.news_items.items()
                if news_item.timestamp < cutoff_time
            ]

            for news_id in old_news_ids:
                del self.news_items[news_id]

            # Очищаем старые анализы
            old_analysis_ids = [
                analysis_id
                for analysis_id, analysis in self.news_analyses.items()
                if analysis.timestamp < cutoff_time
            ]

            for analysis_id in old_analysis_ids:
                del self.news_analyses[analysis_id]

            # Очищаем анализы по символам
            for symbol in self.symbol_analyses:
                self.symbol_analyses[symbol] = [
                    analysis
                    for analysis in self.symbol_analyses[symbol]
                    if analysis.timestamp >= cutoff_time
                ]

            if old_news_ids or old_analysis_ids:
                logger.info(
                    f"Cleaned up {len(old_news_ids)} old news and {len(old_analysis_ids)} old analyses"
                )

        except Exception as e:
            logger.error(f"Error cleaning up old news: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики новостной торговли."""
        try:
            return {
                "total_news": len(self.news_items),
                "total_analyses": len(self.news_analyses),
                "symbols_with_news": list(self.symbol_analyses.keys()),
                "news_categories": self._get_news_category_stats(),
                "stats": self.stats.copy(),
                "config": {
                    "enable_news_trading": self.config.enable_news_trading,
                    "sentiment_threshold": self.config.sentiment_threshold,
                    "impact_threshold": self.config.impact_threshold,
                    "max_news_age_hours": self.config.max_news_age_hours,
                },
            }

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}

    def _get_news_category_stats(self) -> Dict[str, int]:
        """Получение статистики по категориям новостей."""
        try:
            stats: Dict[str, int] = {}
            for news_item in self.news_items.values():
                category = news_item.category.value
                stats[category] = stats.get(category, 0) + 1
            return stats

        except Exception as e:
            logger.error(f"Error getting news category stats: {e}")
            return {}

    def clear_data(self) -> None:
        """Очистка всех данных."""
        try:
            self.news_items.clear()
            self.news_analyses.clear()
            self.symbol_analyses.clear()
            self.stats["total_news"] = 0
            self.stats["total_analyses"] = 0
            self.stats["trading_signals"] = 0
            self.stats["auto_trades"] = 0

            logger.info("News trading data cleared")

        except Exception as e:
            logger.error(f"Error clearing news trading data: {e}")
