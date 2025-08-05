"""
Интеграция аналитических модулей с маркет-мейкером.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger

from .integrator import AnalyticalIntegrator
from .types import AnalyticalIntegrationConfig


class MarketMakerAnalyticalIntegration:
    """Интеграция аналитических модулей с MarketMakerModelAgent."""

    def __init__(
        self,
        market_maker_agent: Any,
        config: Optional[AnalyticalIntegrationConfig] = None,
    ):
        self.market_maker_agent = market_maker_agent
        self.config = config or AnalyticalIntegrationConfig()
        self.analytical_integrator = AnalyticalIntegrator()

        # Сохраняем оригинальные методы
        self._original_calculate = None
        self._original_learn = None
        self._original_evolve = None

        # Интеграция с агентом
        self._integrate_with_agent()

        logger.info("MarketMakerAnalyticalIntegration initialized")

    def _integrate_with_agent(self) -> None:
        """Интеграция с маркет-мейкер агентом."""
        try:
            # Сохраняем оригинальные методы
            self._original_calculate = getattr(
                self.market_maker_agent, "calculate", None
            )
            self._original_learn = getattr(self.market_maker_agent, "learn", None)
            self._original_evolve = getattr(self.market_maker_agent, "evolve", None)

            # Заменяем методы на версии с аналитикой
            if self._original_calculate:
                self.market_maker_agent.calculate = self._calculate_with_analytics

            if self._original_learn:
                self.market_maker_agent.learn = self._learn_with_analytics

            if self._original_evolve:
                self.market_maker_agent.evolve = self._evolve_with_analytics

            # Добавляем новые методы
            self.market_maker_agent.get_analytical_context = (
                self._get_analytical_context
            )
            self.market_maker_agent.get_trading_recommendations = (
                self._get_trading_recommendations
            )
            self.market_maker_agent.should_proceed_with_trade = (
                self._should_proceed_with_trade
            )

            logger.info(
                "Successfully integrated analytical modules with market maker agent"
            )

        except Exception as e:
            logger.error(f"Error integrating with market maker agent: {e}")

    async def _calculate_with_analytics(self, *args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """Расчет с учетом аналитических данных."""
        try:
            # Получаем символ из аргументов
            symbol = self._extract_symbol_from_args(args, kwargs)

            if not symbol:
                logger.warning("No symbol provided for analytical calculation")
                if self._original_calculate:
                    result = await self._original_calculate(*args, **kwargs)
                    return result if isinstance(result, dict) else None
                else:
                    return None

            # Проверяем, следует ли продолжать торговлю
            if not self.analytical_integrator.should_proceed_with_trade(symbol):
                logger.info(f"Trading suspended for {symbol} due to analytical alerts")
                return None

            # Получаем аналитический контекст
            analytical_context = self._get_analytical_context(symbol)

            # Корректируем параметры на основе аналитики
            adjusted_kwargs = self._adjust_parameters_with_analytics(symbol, kwargs)

            # Выполняем оригинальный расчет
            if self._original_calculate:
                result = await self._original_calculate(*args, **adjusted_kwargs)
                if isinstance(result, dict):
                    result["analytical_context"] = analytical_context
                    return result
                else:
                    return None
            else:
                return None

        except Exception as e:
            logger.error(f"Error in calculate with analytics: {e}")
            if self._original_calculate:
                result = await self._original_calculate(*args, **kwargs)
                return result if isinstance(result, dict) else None
            else:
                return None

    async def _learn_with_analytics(self, data: Any) -> bool:
        """Обучение с учетом аналитических данных."""
        try:
            # Получаем символ из данных
            symbol = self._extract_symbol_from_data(data)

            if symbol:
                # Получаем аналитический контекст
                analytical_context = self._get_analytical_context(symbol)

                # Добавляем аналитический контекст к данным обучения
                if isinstance(data, dict):
                    data["analytical_context"] = analytical_context
                elif hasattr(data, "__dict__"):
                    setattr(data, "analytical_context", analytical_context)

            # Выполняем оригинальное обучение
            if self._original_learn:
                result = await self._original_learn(data)
                return bool(result)
            else:
                return False

        except Exception as e:
            logger.error(f"Error in learn with analytics: {e}")
            if self._original_learn:
                result = await self._original_learn(data)
                return bool(result)
            else:
                return False

    async def _evolve_with_analytics(self, data: Any) -> bool:
        """Эволюция с учетом аналитических данных."""
        try:
            # Получаем символ из данных
            symbol = self._extract_symbol_from_data(data)

            if symbol:
                # Получаем аналитический контекст
                analytical_context = self._get_analytical_context(symbol)

                # Корректируем параметры эволюции на основе аналитики
                evolution_params = self._adjust_evolution_parameters(
                    symbol, analytical_context
                )

                # Добавляем параметры к данным
                if isinstance(data, dict):
                    data["evolution_params"] = evolution_params
                elif hasattr(data, "__dict__"):
                    setattr(data, "evolution_params", evolution_params)

            # Выполняем оригинальную эволюцию
            if self._original_evolve:
                result = await self._original_evolve(data)
                return bool(result)
            else:
                return False

        except Exception as e:
            logger.error(f"Error in evolve with analytics: {e}")
            if self._original_evolve:
                result = await self._original_evolve(data)
                return bool(result)
            else:
                return False

    def _get_analytical_context(self, symbol: str) -> Dict[str, Any]:
        """Получение аналитического контекста для символа."""
        try:
            return self.analytical_integrator.get_trading_recommendations(symbol)
        except Exception as e:
            logger.error(f"Error getting analytical context for {symbol}: {e}")
            return {}

    def _get_trading_recommendations(self, symbol: str) -> Dict[str, Any]:
        """Получение торговых рекомендаций."""
        try:
            return self.analytical_integrator.get_trading_recommendations(symbol)
        except Exception as e:
            logger.error(f"Error getting trading recommendations for {symbol}: {e}")
            return {}

    def _should_proceed_with_trade(
        self, symbol: str, trade_aggression: float = 1.0
    ) -> bool:
        """Определение, следует ли продолжать торговлю."""
        try:
            return self.analytical_integrator.should_proceed_with_trade(
                symbol, trade_aggression
            )
        except Exception as e:
            logger.error(f"Error checking trade proceed for {symbol}: {e}")
            return True

    def _adjust_parameters_with_analytics(
        self, symbol: str, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Корректировка параметров на основе аналитики."""
        try:
            adjusted_kwargs = kwargs.copy()

            # Корректируем агрессивность
            if "aggressiveness" in adjusted_kwargs:
                base_aggressiveness = adjusted_kwargs["aggressiveness"]
                adjusted_aggressiveness = (
                    self.analytical_integrator.get_adjusted_aggressiveness(
                        symbol, base_aggressiveness
                    )
                )
                adjusted_kwargs["aggressiveness"] = adjusted_aggressiveness

            # Корректируем размер позиции
            if "position_size" in adjusted_kwargs:
                base_size = adjusted_kwargs["position_size"]
                adjusted_size = self.analytical_integrator.get_adjusted_position_size(
                    symbol, base_size
                )
                adjusted_kwargs["position_size"] = adjusted_size

            # Корректируем уверенность
            if "confidence" in adjusted_kwargs:
                base_confidence = adjusted_kwargs["confidence"]
                adjusted_confidence = (
                    self.analytical_integrator.get_adjusted_confidence(
                        symbol, base_confidence
                    )
                )
                adjusted_kwargs["confidence"] = adjusted_confidence

            # Корректируем цены
            if "price" in adjusted_kwargs:
                base_price = adjusted_kwargs["price"]
                side = adjusted_kwargs.get("side", "buy")
                price_offset = self.analytical_integrator.get_price_offset(
                    symbol, base_price, side
                )
                adjusted_kwargs["price"] = base_price + price_offset

            return adjusted_kwargs

        except Exception as e:
            logger.error(f"Error adjusting parameters for {symbol}: {e}")
            return kwargs

    def _extract_symbol_from_args(self, args: tuple, kwargs: dict) -> Optional[str]:
        """Извлекает символ из аргументов функции."""
        try:
            # Сначала проверяем kwargs
            if "symbol" in kwargs:
                symbol = kwargs["symbol"]
                return str(symbol) if symbol is not None else None
            # Затем проверяем первый позиционный аргумент
            if args and len(args) > 0:
                arg = args[0]
                return str(arg) if arg is not None else None
            return None
        except Exception as e:
            logger.error(f"Error extracting symbol from args: {e}")
            return None

    def _extract_symbol_from_data(self, data: Any) -> Optional[str]:
        """Извлекает символ из данных."""
        try:
            if isinstance(data, dict):
                symbol = data.get("symbol")
                return str(symbol) if symbol is not None else None
            elif hasattr(data, "symbol"):
                symbol = getattr(data, "symbol")
                return str(symbol) if symbol is not None else None
            elif hasattr(data, "__dict__"):
                symbol = getattr(data, "symbol", None)
                return str(symbol) if symbol is not None else None
            return None
        except Exception as e:
            logger.error(f"Error extracting symbol from data: {e}")
            return None

    def get_analytical_context(self, symbol: str) -> Dict[str, Any]:
        """Получение аналитического контекста для символа."""
        try:
            return self.analytical_integrator.get_trading_recommendations(symbol)
        except Exception as e:
            logger.error(f"Error getting analytical context for {symbol}: {e}")
            return {}

    def _adjust_evolution_parameters(
        self, symbol: str, analytical_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Корректировка параметров эволюции на основе аналитики."""
        try:
            # Базовые параметры эволюции
            evolution_params = {
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "selection_pressure": 0.5,
                "population_size": 50,
            }
            # Корректировка на основе рыночных условий
            market_conditions = analytical_context.get("market_conditions", "normal")
            if market_conditions == "volatile":
                evolution_params["mutation_rate"] *= 1.5
                evolution_params["selection_pressure"] *= 0.8
            elif market_conditions == "trending":
                evolution_params["crossover_rate"] *= 1.2
                evolution_params["selection_pressure"] *= 1.2
            # Корректировка на основе уверенности
            confidence = analytical_context.get("confidence", 0.5)
            if confidence < 0.3:
                evolution_params["mutation_rate"] *= 2.0
                evolution_params["population_size"] *= 1.5
            return evolution_params
        except Exception as e:
            logger.error(f"Error adjusting evolution parameters for {symbol}: {e}")
            return {
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "selection_pressure": 0.5,
                "population_size": 50,
            }

    async def start(self) -> None:
        """Запуск интеграции."""
        try:
            logger.info("Starting market maker analytical integration")
            # Здесь можно добавить логику запуска аналитических модулей
        except Exception as e:
            logger.error(f"Error starting integration: {e}")

    async def stop(self) -> None:
        """Остановка интеграции."""
        try:
            logger.info("Stopping market maker analytical integration")
            # Здесь можно добавить логику остановки аналитических модулей
        except Exception as e:
            logger.error(f"Error stopping integration: {e}")

    def get_analytics_statistics(self) -> Dict[str, Any]:
        """Получение статистики аналитики."""
        try:
            return {
                "analytics_enabled": True,
                "integration_active": True,
                "analytical_integrator_stats": self.analytical_integrator.get_statistics(),
                "config": {
                    "entanglement_enabled": self.config.entanglement_enabled,
                    "noise_enabled": self.config.noise_enabled,
                    "mirror_enabled": self.config.mirror_enabled,
                    "gravity_enabled": self.config.gravity_enabled,
                }
            }
        except Exception as e:
            logger.error(f"Error getting analytics statistics: {e}")
            return {"analytics_enabled": False, "error": str(e)}
