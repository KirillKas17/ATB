"""
Основной агент управления портфелем.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from domain.type_definitions.agent_types import ProcessingResult, AgentType, AgentConfig
from infrastructure.agents.base_agent import AgentStatus, BaseAgent
from infrastructure.core.portfolio_manager import PortfolioManager
from infrastructure.core.position_manager import PositionManager
from infrastructure.core.risk_manager import RiskManager
from infrastructure.messaging.event_bus import EventBus

from .services import PortfolioMetricsService
from .types import (
    PortfolioConfig,
    PortfolioLimits,
    PortfolioMetrics,
    PortfolioStatus,
)


class PortfolioAgent(BaseAgent):
    """Агент управления портфелем."""

    def __init__(self, config: Optional[PortfolioConfig] = None) -> None:
        # [3] создаю AgentConfig для базового агента
        agent_config: AgentConfig = {
            "name": "PortfolioAgent",
            "agent_type": "portfolio_optimizer",  # [1] правильный AgentType
            "max_position_size": 1000.0,
            "max_portfolio_risk": 0.02,
            "max_risk_per_trade": 0.01,
            "confidence_threshold": 0.8,  # [3] обязательное поле
            "risk_threshold": 0.3,  # [3] обязательное поле
            "performance_threshold": 0.7,  # [3] обязательное поле
            "rebalance_interval": 3600,  # [3] обязательное поле
            "processing_timeout_ms": 30000,  # [3] обязательное поле
            "retry_attempts": 3,  # [3] обязательное поле
            "enable_evolution": False,  # [3] обязательное поле
            "enable_learning": True,  # [3] обязательное поле
            "metadata": {}  # [3] обязательное поле
        }
        # Преобразование конфигурации
        if hasattr(config, 'to_dict') and callable(config.to_dict):
            config_dict = config.to_dict()
        elif hasattr(config, '__annotations__') and isinstance(config, dict) is False:
            # Исправление: правильное преобразование TypedDict
            config_dict = dict(config) if hasattr(config, "__iter__") else {} if config else {}
        else:
            config_dict = config or {}
        super().__init__("PortfolioAgent", "portfolio_optimizer", config_dict)

        self.portfolio_config = config or PortfolioConfig()
        self.portfolio_manager = PortfolioManager(event_bus=EventBus(), config={})
        self.position_manager = PositionManager(config={})
        self.risk_manager = RiskManager(event_bus=EventBus(), config={})
        self.metrics_service = PortfolioMetricsService()

        self.portfolio_state: Optional[Dict[str, Any]] = None
        self.asset_metrics: Dict[str, Dict[str, Any]] = {}
        self.trade_suggestions: List[Dict[str, Any]] = []
        self.constraints = PortfolioLimits()

        self.stats = {
            "total_trades": 0,
            "successful_trades": 0,
            "total_pnl": 0.0,
            "rebalance_count": 0,
        }

    async def initialize(self) -> bool:
        """Инициализация агента."""
        try:
            if not self.validate_config():
                return False

            # Используем _update_state для изменения статуса
            self._update_state(AgentStatus.HEALTHY)
            self.update_confidence(0.8)

            logger.info("PortfolioAgent initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize PortfolioAgent: {e}")
            self.record_error(f"Initialization failed: {e}")
            return False

    async def process(self, data: Any) -> ProcessingResult:
        """Обработка данных."""
        start_time = datetime.now()

        try:
            if isinstance(data, dict):
                # Обновление состояния портфеля
                portfolio_data = data.get("portfolio_data", {})
                await self.update_portfolio_state(portfolio_data)

                # Анализ активов
                assets_data = data.get("assets_data", {})
                await self.analyze_assets(assets_data)

                # Генерация торговых предложений
                suggestions = await self.generate_trade_suggestions()

                # Проверка необходимости ребалансировки
                rebalance_needed = self.check_rebalance_needed()

                result_data = {
                    "portfolio_state": self.portfolio_state or {},
                    "trade_suggestions": suggestions,
                    "rebalance_needed": rebalance_needed,
                    "asset_metrics": self.asset_metrics,
                    "timestamp": datetime.now().isoformat(),
                }

                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                self.record_success(processing_time)

                return ProcessingResult(
                    success=True,
                    data=result_data,
                    confidence=0.8,
                    risk_score=0.2,  # [2] добавляю risk_score
                    processing_time_ms=processing_time,
                    timestamp=datetime.now(),
                    metadata={},
                    errors=[],
                    warnings=[]
                )
            else:
                raise ValueError("Invalid data format")

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.record_error(f"Processing failed: {e}", processing_time)
            return ProcessingResult(
                success=False,
                data={"error": str(e)},
                confidence=0.0,
                risk_score=1.0,  # [2] добавляю risk_score
                processing_time_ms=processing_time,
                timestamp=datetime.now(),
                metadata={},
                errors=[str(e)],
                warnings=[]
            )

    async def cleanup(self) -> None:
        """Очистка ресурсов."""
        try:
            self.portfolio_state = None
            self.asset_metrics.clear()
            self.trade_suggestions.clear()
            logger.info("PortfolioAgent cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def validate_config(self) -> bool:
        """Валидация конфигурации."""
        try:
            required_keys = [
                "rebalancing_threshold",
                "max_position_size",
                "max_sector_allocation",
                "target_volatility",
            ]

            for key in required_keys:
                # [4] правильный доступ к полям dataclass
                if not hasattr(self.portfolio_config, key):
                    logger.error(f"Missing required config key: {key}")
                    return False

                value = getattr(self.portfolio_config, key)
                if not isinstance(value, (int, float)) or value < 0:
                    logger.error(f"Invalid config value for {key}: {value}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Config validation error: {e}")
            return False

    async def update_portfolio_state(self, portfolio_data: Dict[str, Any]) -> None:
        """Обновление состояния портфеля."""
        try:
            if not portfolio_data:
                return

            total_value = portfolio_data.get("total_value", 0.0)
            total_pnl = portfolio_data.get("total_pnl", 0.0)
            allocation = portfolio_data.get("allocation", {})
            risk_metrics = portfolio_data.get("risk_metrics", {})

            self.portfolio_state = {
                "total_value": total_value,
                "total_pnl": total_pnl,
                "total_pnl_percent": (
                    (total_pnl / total_value * 100) if total_value > 0 else 0.0
                ),
                "risk_metrics": risk_metrics,
                "allocation": allocation,
                "last_updated": datetime.now().isoformat(),
                "performance_history": portfolio_data.get("performance_history", []),
            }

        except Exception as e:
            logger.error(f"Error updating portfolio state: {e}")

    async def analyze_assets(self, assets_data: Dict[str, Any]) -> None:
        """Анализ активов."""
        try:
            for symbol, data in assets_data.items():
                if not isinstance(data, dict):
                    continue

                metrics = {
                    "symbol": symbol,
                    "expected_return": data.get("expected_return", 0.0),
                    "risk_score": data.get("risk_score", 0.0),
                    "liquidity_score": data.get("liquidity_score", 0.0),
                    "trend_strength": data.get("trend_strength", 0.0),
                    "volume_score": data.get("volume_score", 0.0),
                    "correlation_with_btc": data.get("correlation_with_btc", 0.0),
                    "current_weight": data.get("current_weight", 0.0),
                    "target_weight": data.get("target_weight", 0.0),
                }

                self.asset_metrics[symbol] = metrics

        except Exception as e:
            logger.error(f"Error analyzing assets: {e}")

    async def generate_trade_suggestions(self) -> List[Dict[str, Any]]:
        """Генерация торговых предложений."""
        try:
            suggestions: List[Dict[str, Any]] = []
            
            if not self.portfolio_state or not self.asset_metrics:
                return suggestions

            # Анализ отклонений от целевых весов
            for symbol, metrics in self.asset_metrics.items():
                current_weight = metrics.get("current_weight", 0.0)
                target_weight = metrics.get("target_weight", 0.0)
                
                if abs(current_weight - target_weight) > self.portfolio_config.rebalancing_threshold:
                    # Определение типа операции
                    if current_weight < target_weight:
                        action = "buy"
                        quantity = (target_weight - current_weight) * self.portfolio_state["total_value"]
                    else:
                        action = "sell"
                        quantity = (current_weight - target_weight) * self.portfolio_state["total_value"]

                    suggestion = {
                        "symbol": symbol,
                        "action": action,
                        "quantity": quantity,
                        "reason": "rebalancing",
                        "confidence": 0.8,
                        "timestamp": datetime.now().isoformat(),
                    }
                    suggestions.append(suggestion)

            # Проверка лимитов риска
            for symbol, metrics in self.asset_metrics.items():
                risk_score = metrics.get("risk_score", 0.0)
                if risk_score > 0.8:  # Высокий риск
                    suggestion = {
                        "symbol": symbol,
                        "action": "sell",
                        "quantity": metrics.get("current_weight", 0.0) * self.portfolio_state["total_value"] * 0.5,
                        "reason": "risk_reduction",
                        "confidence": 0.9,
                        "timestamp": datetime.now().isoformat(),
                    }
                    suggestions.append(suggestion)

            return suggestions

        except Exception as e:
            logger.error(f"Error generating trade suggestions: {e}")
            return []

    def check_rebalance_needed(self) -> bool:
        """Проверка необходимости ребалансировки."""
        try:
            if not self.portfolio_state or not self.asset_metrics:
                return False

            # Проверка отклонений от целевых весов
            for symbol, metrics in self.asset_metrics.items():
                current_weight = metrics.get("current_weight", 0.0)
                target_weight = metrics.get("target_weight", 0.0)
                
                if abs(current_weight - target_weight) > self.portfolio_config.rebalancing_threshold:
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking rebalance need: {e}")
            return False

    def _get_current_price(self, symbol: str) -> float:
        """Получение текущей цены актива."""
        try:
            # Здесь должна быть логика получения цены
            # Пока возвращаем заглушку
            return 100.0
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return 0.0

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Получение сводки портфеля."""
        try:
            return {
                "portfolio_state": self.portfolio_state,
                "asset_count": len(self.asset_metrics),
                "total_value": self.portfolio_state.get("total_value", 0.0) if self.portfolio_state else 0.0,
                "total_pnl": self.portfolio_state.get("total_pnl", 0.0) if self.portfolio_state else 0.0,
                "stats": self.stats,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {"error": str(e)}

    def reset_portfolio(self) -> None:
        """Сброс портфеля."""
        try:
            self.portfolio_state = None
            self.asset_metrics.clear()
            self.trade_suggestions.clear()
            self.stats = {
                "total_trades": 0,
                "successful_trades": 0,
                "total_pnl": 0.0,
                "rebalance_count": 0,
            }
            logger.info("Portfolio reset completed")
        except Exception as e:
            logger.error(f"Error resetting portfolio: {e}")

    def analyze_portfolio(self, portfolio_data: Dict[str, Any]) -> PortfolioMetrics:
        """Анализ портфеля."""
        try:
            # Используем существующий PortfolioMetricsService
            metrics = self.metrics_service.calculate_portfolio_metrics(portfolio_data)
            return PortfolioMetrics(
                total_value=metrics.get("total_value", 0.0),
                total_pnl=metrics.get("total_pnl", 0.0),
                total_pnl_percent=metrics.get("total_pnl_percent", 0.0),
                allocation=portfolio_data.get("allocation", {}),
                risk_score=metrics.get("risk_score", 0.0),
                diversification_score=metrics.get("diversification_score", 0.0),
                sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
                volatility=metrics.get("volatility", 0.0),
                max_drawdown=metrics.get("max_drawdown", 0.0),
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error analyzing portfolio: {e}")
            return PortfolioMetrics()

    def get_rebalancing_recommendations(
        self, portfolio_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Получение рекомендаций по ребалансировке."""
        try:
            # Используем логику из generate_trade_suggestions
            recommendations: List[Dict[str, Any]] = []
            allocation = portfolio_data.get("allocation", {})
            target_allocation = portfolio_data.get("target_allocation", {})
            
            for symbol, current_weight in allocation.items():
                target_weight = target_allocation.get(symbol, 0.0)
                if abs(current_weight - target_weight) > self.portfolio_config.get("rebalancing_threshold", 0.05):
                    recommendations.append({
                        "symbol": symbol,
                        "current_weight": current_weight,
                        "target_weight": target_weight,
                        "action": "buy" if current_weight < target_weight else "sell",
                        "quantity": abs(current_weight - target_weight),
                        "priority": "high" if abs(current_weight - target_weight) > 0.1 else "medium"
                    })
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting rebalancing recommendations: {e}")
            return []

    def validate_portfolio(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Валидация портфеля."""
        try:
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            total_value = portfolio_data.get("total_value", 0.0)
            allocation = portfolio_data.get("allocation", {})
            
            # Проверка общей стоимости
            if total_value <= 0:
                validation_result["valid"] = False
                validation_result["errors"].append("Total portfolio value must be positive")
            
            # Проверка распределения
            total_allocation = sum(allocation.values())
            if abs(total_allocation - 1.0) > 0.01:
                validation_result["warnings"].append(f"Allocation sum is {total_allocation:.2f}, should be 1.0")
            
            # Проверка лимитов
            max_allocation = self.portfolio_config.get("max_allocation", 0.3)
            for symbol, weight in allocation.items():
                if weight > max_allocation:
                    validation_result["warnings"].append(f"High allocation for {symbol}: {weight:.2f}")
            
            return validation_result
        except Exception as e:
            logger.error(f"Error validating portfolio: {e}")
            return {
                "valid": False,
                "errors": [str(e)],
                "warnings": []
            }
