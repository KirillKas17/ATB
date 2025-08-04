"""
Оркестратор торговой системы - Application Layer.

Отвечает за координацию всех компонентов торговой системы
согласно принципам Domain-Driven Design.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from application.use_cases.manage_orders import DefaultOrderManagementUseCase
from application.use_cases.manage_positions import PositionManagementUseCase
from application.use_cases.manage_risk import RiskManagementUseCase
from application.use_cases.manage_trading_pairs import TradingPairManagementUseCase
from domain.interfaces.evolution_manager import EvolutionManagerProtocol
from domain.interfaces.market_data import MarketDataProtocol
from domain.interfaces.portfolio_manager import PortfolioManagerProtocol
from domain.interfaces.risk_manager import RiskManagerProtocol
from domain.interfaces.sentiment_analyzer import SentimentAnalyzerProtocol
from domain.interfaces.strategy_registry import StrategyRegistryProtocol
from domain.interfaces.trading_orchestrator import TradingOrchestratorProtocol
from domain.type_definitions.trading_types import Signal, TradingConfig, TradingPair
from shared.models.config import ApplicationConfig
from application.use_cases.manage_trading_pairs import GetTradingPairsRequest


@dataclass
class OrchestrationMetrics:
    """Метрики оркестрации."""

    total_cycles: int = 0
    successful_signals: int = 0
    failed_signals: int = 0
    average_cycle_time: float = 0.0
    last_cycle_time: Optional[datetime] = None
    system_uptime: timedelta = timedelta(0)


class TradingOrchestrator(TradingOrchestratorProtocol):
    """
    Оркестратор торговой системы.

    Координирует работу всех компонентов системы:
    - Анализ рынка
    - Генерация сигналов
    - Управление рисками
    - Исполнение ордеров
    - Управление портфелем
    - Эволюционные агенты
    """

    def __init__(
        self,
        config: ApplicationConfig,
        strategy_registry: StrategyRegistryProtocol,
        risk_manager: RiskManagerProtocol,
        market_data: MarketDataProtocol,
        sentiment_analyzer: SentimentAnalyzerProtocol,
        portfolio_manager: PortfolioManagerProtocol,
        evolution_manager: EvolutionManagerProtocol,
        order_use_case: DefaultOrderManagementUseCase,
        position_use_case: PositionManagementUseCase,
        risk_use_case: RiskManagementUseCase,
        trading_pair_use_case: TradingPairManagementUseCase,
    ):
        self.config = config
        self.strategy_registry = strategy_registry
        self.risk_manager = risk_manager
        self.market_data = market_data
        self.sentiment_analyzer = sentiment_analyzer
        self.portfolio_manager = portfolio_manager
        self.evolution_manager = evolution_manager

        # Use Cases
        self.order_use_case = order_use_case
        self.position_use_case = position_use_case
        self.risk_use_case = risk_use_case
        self.trading_pair_use_case = trading_pair_use_case

        # Состояние системы
        self.is_running = False
        self.metrics = OrchestrationMetrics()
        self.logger = logging.getLogger(__name__)

        # Интервалы из конфигурации
        self.trading_interval = config.trading.trading_interval
        self.sentiment_analysis_interval = config.trading.sentiment_analysis_interval
        self.portfolio_rebalance_interval = config.trading.portfolio_rebalance_interval
        self.evolution_cycle_interval = config.trading.evolution_cycle_interval

        # Время последних операций
        self.last_sentiment_analysis: Dict[str, Dict[str, Any]] = {}
        self.last_rebalance_time: Optional[datetime] = None
        self.last_evolution_cycle: Optional[datetime] = None
        self.start_time = datetime.now()

    async def start(self) -> None:
        """Запуск оркестратора."""
        self.logger.info("Starting Trading Orchestrator")
        self.is_running = True
        self.start_time = datetime.now()

        # Инициализация компонентов
        await self._initialize_components()

        # Запуск основных циклов
        await asyncio.gather(
            self._trading_cycle(),
            self._sentiment_analysis_cycle(),
            self._portfolio_rebalancing_cycle(),
            self._evolution_cycle(),
            self._monitoring_cycle(),
        )

    async def stop(self) -> None:
        """Остановка оркестратора."""
        self.logger.info("Stopping Trading Orchestrator")
        self.is_running = False

        # Остановка компонентов
        await self._cleanup_components()

        # Генерация финального отчёта
        await self._generate_final_report()

    async def _initialize_components(self) -> None:
        """Инициализация всех компонентов."""
        try:
            # Инициализация эволюционной системы
            await self.evolution_manager.initialize()

            # Инициализация стратегий
            await self.strategy_registry.initialize()

            # Инициализация риск-менеджера
            await self.risk_manager.initialize()

            # Инициализация портфель-менеджера
            await self.portfolio_manager.initialize()

            self.logger.info("All components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise

    async def _cleanup_components(self) -> None:
        """Очистка ресурсов компонентов."""
        try:
            await self.evolution_manager.cleanup()
            await self.strategy_registry.cleanup()
            await self.risk_manager.cleanup()
            await self.portfolio_manager.cleanup()

            self.logger.info("All components cleaned up successfully")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    async def _trading_cycle(self) -> None:
        """Основной торговый цикл."""
        while self.is_running:
            try:
                cycle_start = datetime.now()

                # Получение торговых пар
                trading_pairs_response = (
                    await self.trading_pair_use_case.get_trading_pairs(
                        GetTradingPairsRequest(is_active=True)
                    )
                )
                trading_pairs = trading_pairs_response.trading_pairs if trading_pairs_response.success else []

                # Обработка каждой торговой пары
                for trading_pair in trading_pairs:
                    await self._process_trading_pair(trading_pair)

                # Обновление метрик
                cycle_time = (datetime.now() - cycle_start).total_seconds()
                self.metrics.total_cycles = int(self.metrics.total_cycles) + 1
                self.metrics.average_cycle_time = (
                    self.metrics.average_cycle_time * (self.metrics.total_cycles - 1)
                    + cycle_time
                ) / self.metrics.total_cycles
                self.metrics.last_cycle_time = datetime.now()

                # Ожидание следующего цикла
                await asyncio.sleep(self.trading_interval)

            except Exception as e:
                self.logger.error(f"Error in trading cycle: {e}")
                await asyncio.sleep(5)  # Краткая пауза при ошибке

    async def _process_trading_pair(self, trading_pair: TradingPair) -> None:
        """Обработка одной торговой пары."""
        try:
            # Получение рыночных данных
            market_data = await self.market_data.get_market_data(trading_pair.symbol)

            # Технический анализ
            technical_analysis = await self._perform_technical_analysis(
                trading_pair.symbol, market_data
            )

            # Анализ рисков
            risk_analysis = await self._perform_risk_analysis(
                trading_pair.symbol, market_data
            )

            # Анализ настроений
            sentiment_analysis = await self._get_sentiment_analysis(trading_pair.symbol)

            # Анализ с эволюционными агентами
            evolution_analysis = await self._analyze_with_evolution_agents(
                trading_pair.symbol, sentiment_analysis, technical_analysis
            )

            # Генерация торговых сигналов
            signals = await self._generate_trading_signals(
                trading_pair.symbol,
                sentiment_analysis,
                technical_analysis,
                risk_analysis,
                evolution_analysis,
            )

            # Обработка сигналов
            for signal in signals:
                await self._process_trading_signal(signal)

        except Exception as e:
            self.logger.error(f"Error processing trading pair {trading_pair.symbol}: {e}")

    async def _perform_technical_analysis(
        self, symbol: str, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Выполнение технического анализа."""
        try:
            # В реальной системе здесь был бы вызов технического анализа
            return {
                "rsi": 50.0,
                "macd": 0.0,
                "bollinger_bands": {"upper": 100.0, "middle": 95.0, "lower": 90.0},
                "support_levels": [90.0, 85.0],
                "resistance_levels": [100.0, 105.0],
            }
        except Exception as e:
            self.logger.error(f"Error performing technical analysis: {e}")
            return {}

    async def _perform_risk_analysis(
        self, symbol: str, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Выполнение анализа рисков."""
        try:
            # В реальной системе здесь был бы анализ рисков
            return {
                "volatility": 0.15,
                "var_95": 0.02,
                "max_drawdown": 0.05,
                "correlation": 0.3,
            }
        except Exception as e:
            self.logger.error(f"Error performing risk analysis: {e}")
            return {}

    async def _get_sentiment_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Получение анализа настроений."""
        try:
            # Проверяем кэш
            if symbol in self.last_sentiment_analysis:
                cache_time = self.last_sentiment_analysis[symbol]["timestamp"]
                if datetime.now() - cache_time < timedelta(minutes=30):
                    return self.last_sentiment_analysis[symbol]["data"]

            # Получаем новый анализ
            sentiment = await self.sentiment_analyzer.analyze_sentiment(symbol)
            self.last_sentiment_analysis[symbol] = {
                "data": sentiment,
                "timestamp": datetime.now(),
            }
            return sentiment

        except Exception as e:
            self.logger.error(f"Error getting sentiment analysis: {e}")
            return None

    async def _analyze_with_evolution_agents(
        self,
        symbol: str,
        sentiment: Optional[Dict[str, Any]],
        technical_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Анализ с использованием эволюционных агентов."""
        try:
            # В реальной системе здесь был бы анализ эволюционными агентами
            return {
                "evolution_score": 0.7,
                "adaptation_level": 0.8,
                "prediction_confidence": 0.6,
            }
        except Exception as e:
            self.logger.error(f"Error analyzing with evolution agents: {e}")
            return {}

    async def _generate_trading_signals(
        self,
        symbol: str,
        sentiment_analysis: Optional[Dict[str, Any]],
        technical_analysis: Dict[str, Any],
        risk_analysis: Dict[str, Any],
        evolution_analysis: Dict[str, Any],
    ) -> List[Signal]:
        """Генерация торговых сигналов."""
        try:
            signals: List[Signal] = []

            # В реальной системе здесь была бы логика генерации сигналов
            # на основе всех анализов
            # Пока возвращаем пустой список

            return signals

        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {e}")
            return []

    async def _validate_signal(self, signal: Signal, strategy: Any) -> bool:
        """Валидация торгового сигнала."""
        try:
            # Проверка риск-лимитов
            risk_valid = await self.risk_use_case.check_order_risk(signal, None)
            if not risk_valid:
                self.logger.warning(f"Signal {signal} failed risk validation")
                return False

            # Дополнительные проверки могут быть добавлены здесь
            return True

        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return False

    async def _process_trading_signal(self, signal: Signal) -> None:
        """Обработка торгового сигнала."""
        try:
            # Валидация сигнала
            if not await self._validate_signal(signal, None):
                self.logger.warning(f"Invalid signal rejected: {signal}")
                self.metrics.failed_signals = int(self.metrics.failed_signals) + 1
                return

            # Исполнение сигнала через use case
            success = await self.order_use_case.execute_signal(signal)

            if success:
                self.metrics.successful_signals = int(self.metrics.successful_signals) + 1
                self.logger.info(f"Signal executed successfully: {signal}")
            else:
                self.metrics.failed_signals = int(self.metrics.failed_signals) + 1
                self.logger.error(f"Signal execution failed: {signal}")

        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
            self.metrics.failed_signals = int(self.metrics.failed_signals) + 1

    async def _sentiment_analysis_cycle(self) -> None:
        """Цикл анализа настроений."""
        while self.is_running:
            try:
                # Получение активных торговых пар
                trading_pairs_response = (
                    await self.trading_pair_use_case.get_trading_pairs(
                        GetTradingPairsRequest(is_active=True)
                    )
                )
                trading_pairs = trading_pairs_response.trading_pairs if trading_pairs_response.success else []

                for trading_pair in trading_pairs:
                    await self._update_sentiment_analysis(trading_pair.symbol)

                await asyncio.sleep(self.sentiment_analysis_interval)

            except Exception as e:
                self.logger.error(f"Error in sentiment analysis cycle: {e}")
                await asyncio.sleep(30)

    async def _update_sentiment_analysis(self, symbol: str) -> None:
        """Обновление анализа настроений для символа."""
        try:
            sentiment = await self.sentiment_analyzer.analyze_sentiment(symbol)
            self.last_sentiment_analysis[symbol] = {
                "data": sentiment,
                "timestamp": datetime.now(),
            }
        except Exception as e:
            self.logger.error(f"Error updating sentiment for {symbol}: {e}")

    async def _portfolio_rebalancing_cycle(self) -> None:
        """Цикл ребалансировки портфеля."""
        while self.is_running:
            try:
                if (
                    self.last_rebalance_time is None
                    or datetime.now() - self.last_rebalance_time > timedelta(hours=1)
                ):

                    await self._perform_portfolio_rebalancing()
                    self.last_rebalance_time = datetime.now()

                await asyncio.sleep(self.portfolio_rebalance_interval)

            except Exception as e:
                self.logger.error(f"Error in portfolio rebalancing cycle: {e}")
                await asyncio.sleep(300)

    async def _perform_portfolio_rebalancing(self) -> None:
        """Выполнение ребалансировки портфеля."""
        try:
            await self.portfolio_manager.rebalance_portfolio()
            self.logger.info("Portfolio rebalancing completed")
        except Exception as e:
            self.logger.error(f"Error in portfolio rebalancing: {e}")

    async def _evolution_cycle(self) -> None:
        """Цикл эволюции."""
        while self.is_running:
            try:
                if (
                    self.last_evolution_cycle is None
                    or datetime.now() - self.last_evolution_cycle > timedelta(hours=6)
                ):

                    await self._perform_evolution_cycle()
                    self.last_evolution_cycle = datetime.now()

                await asyncio.sleep(self.evolution_cycle_interval)

            except Exception as e:
                self.logger.error(f"Error in evolution cycle: {e}")
                await asyncio.sleep(1800)

    async def _perform_evolution_cycle(self) -> None:
        """Выполнение цикла эволюции."""
        try:
            await self.evolution_manager.perform_evolution_cycle()
            self.logger.info("Evolution cycle completed")
        except Exception as e:
            self.logger.error(f"Error in evolution cycle: {e}")

    async def _monitoring_cycle(self) -> None:
        """Цикл мониторинга."""
        while self.is_running:
            try:
                await self._monitor_system_health()
                await asyncio.sleep(60)  # Мониторинг каждую минуту

            except Exception as e:
                self.logger.error(f"Error in monitoring cycle: {e}")
                await asyncio.sleep(60)

    async def _monitor_system_health(self) -> None:
        """Мониторинг здоровья системы."""
        try:
            # Проверка компонентов
            health_status = {
                "strategy_registry": await self.strategy_registry.get_health_status(),
                "risk_manager": await self.risk_manager.get_health_status(),
                "market_data": await self.market_data.get_health_status(),
                "portfolio_manager": await self.portfolio_manager.get_health_status(),
                "evolution_manager": await self.evolution_manager.get_health_status(),
            }

            # Логирование статуса
            for component, status in health_status.items():
                # Обработка различных форматов ответа health status
                if isinstance(status, dict):
                    if not status.get("healthy", True):
                        self.logger.warning(
                            f"Component {component} unhealthy: {status.get('message', 'Unknown issue')}"
                        )
                elif isinstance(status, str):
                    if status.lower() not in ["healthy", "ok", "active"]:
                        self.logger.warning(f"Component {component} status: {status}")
                else:
                    self.logger.info(f"Component {component} health check completed: {status}")

        except Exception as e:
            self.logger.error(f"Error monitoring system health: {e}")

    async def _generate_final_report(self) -> None:
        """Генерация финального отчёта."""
        try:
            uptime = datetime.now() - self.start_time
            self.metrics.system_uptime = uptime

            report = {
                "uptime": str(uptime),
                "total_cycles": self.metrics.total_cycles,
                "successful_signals": self.metrics.successful_signals,
                "failed_signals": self.metrics.failed_signals,
                "average_cycle_time": self.metrics.average_cycle_time,
                "success_rate": (
                    self.metrics.successful_signals
                    / (self.metrics.successful_signals + self.metrics.failed_signals)
                    if (self.metrics.successful_signals + self.metrics.failed_signals)
                    > 0
                    else 0
                ),
            }

            self.logger.info(f"Final report: {report}")

        except Exception as e:
            self.logger.error(f"Error generating final report: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Получение статуса оркестратора."""
        return {
            "is_running": self.is_running,
            "uptime": str(datetime.now() - self.start_time),
            "metrics": {
                "total_cycles": self.metrics.total_cycles,
                "successful_signals": self.metrics.successful_signals,
                "failed_signals": self.metrics.failed_signals,
                "average_cycle_time": self.metrics.average_cycle_time,
            },
            "last_operations": {
                "sentiment_analysis": self.last_sentiment_analysis,
                "rebalance_time": self.last_rebalance_time,
                "evolution_cycle": self.last_evolution_cycle,
            },
        }
