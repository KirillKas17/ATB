from datetime import datetime
from typing import Any, Dict, List, Optional

from domain.type_definitions.agent_types import AgentConfig, ProcessingResult, AgentType
from infrastructure.agents.base_agent import AgentStatus, BaseAgent
from shared.logging import setup_logger

from .components import (
    DefaultPerformanceAnalyzer,
    DefaultRiskManager,
    DefaultStrategyOrchestrator,
)
from .types import ControllerSignal, MetaControllerConfig, PortfolioState

logger = setup_logger(__name__)


class BayesianMetaController:
    """Байесовский мета-контроллер для принятия решений на основе вероятностных моделей."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.prior_beliefs: Dict[str, float] = {}
        self.observation_history: List[Dict[str, Any]] = []
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.learning_rate = self.config.get("learning_rate", 0.1)
        
    def update_beliefs(self, observations: Dict[str, Any]) -> Dict[str, float]:
        """Обновление байесовских убеждений на основе новых наблюдений."""
        updated_beliefs = {}
        
        for key, observation in observations.items():
            if key not in self.prior_beliefs:
                self.prior_beliefs[key] = 0.5  # Нейтральное убеждение
            
            # Простое байесовское обновление
            prior = self.prior_beliefs[key]
            likelihood = self._calculate_likelihood(observation)
            
            # Нормализация
            evidence = prior * likelihood + (1 - prior) * (1 - likelihood)
            if evidence > 0:
                posterior = (prior * likelihood) / evidence
            else:
                posterior = prior
            
            # Обновление с учетом скорости обучения
            updated_belief = prior + self.learning_rate * (posterior - prior)
            updated_beliefs[key] = max(0.0, min(1.0, updated_belief))
            self.prior_beliefs[key] = updated_beliefs[key]
        
        self.observation_history.append(observations)
        return updated_beliefs
    
    def _calculate_likelihood(self, observation: Any) -> float:
        """Вычисление правдоподобия наблюдения."""
        if isinstance(observation, (int, float)):
            # Нормализация числовых значений
            return max(0.0, min(1.0, abs(observation)))
        elif isinstance(observation, bool):
            return 1.0 if observation else 0.0
        elif isinstance(observation, str):
            # Простая эвристика для строковых значений
            positive_words = ["good", "positive", "up", "buy", "long", "profit"]
            negative_words = ["bad", "negative", "down", "sell", "short", "loss"]
            
            observation_lower = observation.lower()
            if any(word in observation_lower for word in positive_words):
                return 0.8
            elif any(word in observation_lower for word in negative_words):
                return 0.2
            else:
                return 0.5
        else:
            return 0.5
    
    def get_decision_confidence(self, decision_type: str) -> float:
        """Получение уверенности в решении."""
        return self.prior_beliefs.get(decision_type, 0.5)
    
    def should_act(self, decision_type: str) -> bool:
        """Определение необходимости действия."""
        confidence = self.get_decision_confidence(decision_type)
        return confidence >= self.confidence_threshold
    
    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Получение рекомендаций на основе текущих убеждений."""
        recommendations = []
        
        for decision_type, belief in self.prior_beliefs.items():
            if belief >= self.confidence_threshold:
                recommendations.append({
                    "type": decision_type,
                    "confidence": belief,
                    "action": "recommended"
                })
            elif belief <= (1 - self.confidence_threshold):
                recommendations.append({
                    "type": decision_type,
                    "confidence": 1 - belief,
                    "action": "avoid"
                })
        
        return recommendations


class PairManager:
    """Менеджер торговых пар для мета-контроллера."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.pairs: Dict[str, Dict[str, Any]] = {}
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        self.volatility_cache: Dict[str, float] = {}
        self.liquidity_cache: Dict[str, float] = {}
        
    def add_pair(self, symbol: str, config: Dict[str, Any]) -> None:
        """Добавление торговой пары."""
        self.pairs[symbol] = {
            "config": config,
            "status": "active",
            "added_at": datetime.now(),
            "last_update": datetime.now()
        }
        
    def remove_pair(self, symbol: str) -> bool:
        """Удаление торговой пары."""
        if symbol in self.pairs:
            del self.pairs[symbol]
            return True
        return False
    
    def get_pair_config(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Получение конфигурации пары."""
        pair = self.pairs.get(symbol)
        return pair["config"] if pair else None
    
    def update_correlation(self, symbol1: str, symbol2: str, correlation: float) -> None:
        """Обновление корреляции между парами."""
        if symbol1 not in self.correlation_matrix:
            self.correlation_matrix[symbol1] = {}
        if symbol2 not in self.correlation_matrix:
            self.correlation_matrix[symbol2] = {}
            
        self.correlation_matrix[symbol1][symbol2] = correlation
        self.correlation_matrix[symbol2][symbol1] = correlation
    
    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Получение корреляции между парами."""
        return self.correlation_matrix.get(symbol1, {}).get(symbol2, 0.0)
    
    def update_volatility(self, symbol: str, volatility: float) -> None:
        """Обновление волатильности пары."""
        self.volatility_cache[symbol] = volatility
    
    def get_volatility(self, symbol: str) -> float:
        """Получение волатильности пары."""
        return self.volatility_cache.get(symbol, 0.0)
    
    def update_liquidity(self, symbol: str, liquidity: float) -> None:
        """Обновление ликвидности пары."""
        self.liquidity_cache[symbol] = liquidity
    
    def get_liquidity(self, symbol: str) -> float:
        """Получение ликвидности пары."""
        return self.liquidity_cache.get(symbol, 0.0)
    
    def get_active_pairs(self) -> List[str]:
        """Получение активных пар."""
        return [symbol for symbol, pair in self.pairs.items() if pair["status"] == "active"]
    
    def get_pair_statistics(self) -> Dict[str, Any]:
        """Получение статистики по парам."""
        return {
            "total_pairs": len(self.pairs),
            "active_pairs": len(self.get_active_pairs()),
            "correlation_pairs": len(self.correlation_matrix),
            "volatility_cache_size": len(self.volatility_cache),
            "liquidity_cache_size": len(self.liquidity_cache)
        }


class MetaControllerAgent(BaseAgent):
    """
    Мета-контроллер для координации стратегий и управления рисками.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Инициализация мета-контроллера.
        :param config: конфигурация контроллера
        """
        # Преобразуем config в AgentConfig для базового класса
        agent_config: AgentConfig = {
            "name": "MetaControllerAgent",
            "agent_type": "meta_controller",
            "max_position_size": config.get("max_position_size", 0.1) if config else 0.1,
            "max_portfolio_risk": config.get("max_portfolio_risk", 0.1) if config else 0.1,
            "max_risk_per_trade": config.get("max_risk_per_trade", 0.02) if config else 0.02,
            "confidence_threshold": config.get("confidence_threshold", 0.7) if config else 0.7,
            "risk_threshold": config.get("risk_threshold", 0.8) if config else 0.8,
            "performance_threshold": config.get("performance_threshold", 0.05) if config else 0.05,
            "rebalance_interval": config.get("rebalance_interval", 3600) if config else 3600,
            "processing_timeout_ms": config.get("processing_timeout_ms", 5000) if config else 5000,
            "retry_attempts": config.get("retry_attempts", 3) if config else 3,
            "enable_evolution": config.get("enable_evolution", False) if config else False,
            "enable_learning": config.get("enable_learning", False) if config else False,
            "metadata": config.get("metadata", {}) if config else {},
        }
        # Преобразуем AgentConfig в dict для BaseAgent
        agent_config_dict = {
            "retry_attempts": getattr(agent_config, 'retry_attempts', 3),
            "enable_evolution": getattr(agent_config, 'enable_evolution', False),
            "enable_learning": getattr(agent_config, 'enable_learning', False),
            "metadata": getattr(agent_config, 'metadata', {}),
        }
        super().__init__("MetaControllerAgent", "meta_controller", agent_config_dict)

        # Преобразуем config в MetaControllerConfig
        controller_config = MetaControllerConfig(
            max_risk_per_trade=config.get("max_risk_per_trade", 0.02) if config else 0.02,
            max_portfolio_risk=config.get("max_portfolio_risk", 0.1) if config else 0.1,
            performance_threshold=config.get("performance_threshold", 0.05) if config else 0.05,
            rebalance_interval=config.get("rebalance_interval", 3600) if config else 3600,
            strategy_timeout=config.get("strategy_timeout", 300) if config else 300,
            enable_auto_rebalance=config.get("enable_auto_rebalance", True) if config else True,
            enable_risk_control=config.get("enable_risk_control", True) if config else True,
            enable_performance_monitoring=config.get("enable_performance_monitoring", True) if config else True,
        )

        # Сохраняем конфигурацию контроллера
        self._controller_config = controller_config

        # Компоненты контроллера
        self.strategy_orchestrator = DefaultStrategyOrchestrator(controller_config)
        self.risk_manager = DefaultRiskManager(controller_config)
        self.performance_analyzer = DefaultPerformanceAnalyzer(controller_config)

        # Состояние контроллера
        self.active_strategies: Dict[str, Any] = {}
        self.portfolio_state: Dict[str, Any] = {}
        self.risk_metrics: Dict[str, float] = {}
        self.performance_metrics: Dict[str, float] = {}

        # Временные метрики
        self.last_rebalance: Optional[datetime] = None
        self.last_performance_check: Optional[datetime] = None

    @property
    def controller_config(self) -> MetaControllerConfig:
        """Конфигурация контроллера."""
        return self._controller_config

    async def initialize(self) -> bool:
        """Инициализация мета-контроллера."""
        try:
            # Валидация конфигурации
            if not self.validate_config():
                return False

            # Инициализация компонентов (они уже инициализированы в конструкторе)
            self._update_state(AgentStatus.HEALTHY)
            self.update_confidence(0.8)

            logger.info("MetaControllerAgent initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize MetaControllerAgent: {e}")
            self.record_error(f"Initialization failed: {e}")
            return False

    async def process(self, data: Any) -> ProcessingResult:
        """Обработка данных мета-контроллером."""
        start_time = datetime.now()

        try:
            if isinstance(data, dict):
                # Анализ производительности стратегий
                performance_data = await self.analyze_performance(data)

                # Управление рисками
                risk_data = await self.manage_risks(data)

                # Оркестрация стратегий
                strategy_data = await self.orchestrate_strategies(data)

                # Принятие решений
                decisions = await self.make_decisions(
                    data, performance_data, risk_data, strategy_data
                )

                result_data = {
                    "decisions": decisions,
                    "performance": performance_data,
                    "risk": risk_data,
                    "strategies": strategy_data,
                    "timestamp": datetime.now().isoformat(),
                }

                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                self.record_success(processing_time)

                return ProcessingResult(
                    success=True,
                    data=result_data,
                    confidence=self.get_confidence(),
                    risk_score=self.get_risk_score(),
                    processing_time_ms=processing_time,
                )
            else:
                raise ValueError("Invalid data format for MetaControllerAgent")

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.record_error(f"Processing failed: {e}", processing_time)
            return ProcessingResult(
                success=False,
                data={"error": str(e)},
                confidence=0.0,
                risk_score=1.0,
                processing_time_ms=processing_time,
                errors=[str(e)],
            )

    async def cleanup(self) -> None:
        """Очистка ресурсов мета-контроллера."""
        try:
            # Остановка всех стратегий
            for strategy_id in list(self.active_strategies.keys()):
                await self.strategy_orchestrator.stop_strategy(strategy_id)

            # Очистка состояния
            self.active_strategies.clear()
            self.portfolio_state.clear()
            self.risk_metrics.clear()
            self.performance_metrics.clear()

            logger.info("MetaControllerAgent cleanup completed")

        except Exception as e:
            logger.error(f"Error during MetaControllerAgent cleanup: {e}")

    def validate_config(self) -> bool:
        """Валидация конфигурации мета-контроллера."""
        try:
            config = self.controller_config
            required_keys = [
                "max_risk_per_trade",
                "max_portfolio_risk",
                "performance_threshold",
                "rebalance_interval",
                "strategy_timeout",
            ]

            for key in required_keys:
                if not hasattr(config, key):
                    logger.error(f"Missing required config key: {key}")
                    return False

                value = getattr(config, key)
                if value is None or not isinstance(value, (int, float)) or value <= 0:
                    logger.error(f"Invalid config value for {key}: {value}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Config validation error: {e}")
            return False

    async def analyze_performance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ производительности стратегий."""
        try:
            # Создаем PortfolioState из данных
            portfolio_state = self._create_portfolio_state(data)
            strategy_statuses = list(self.strategy_orchestrator.active_strategies.values())
            
            performance_metrics = await self.performance_analyzer.analyze_performance(
                portfolio_state, strategy_statuses
            )
            
            # Обновляем локальные метрики
            self.performance_metrics.update({
                "overall_performance": performance_metrics.overall_performance,
                "win_rate": performance_metrics.win_rate,
                "profit_factor": performance_metrics.profit_factor,
            })

            return {
                "overall_performance": performance_metrics.overall_performance,
                "strategy_performance": performance_metrics.strategy_performance,
                "win_rate": performance_metrics.win_rate,
                "profit_factor": performance_metrics.profit_factor,
            }

        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return {"error": str(e)}

    async def manage_risks(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Управление рисками портфеля."""
        try:
            portfolio_state = self._create_portfolio_state(data)
            risk_metrics = await self.risk_manager.calculate_portfolio_risk(portfolio_state)
            self.risk_metrics.update({
                "portfolio_risk": risk_metrics.portfolio_risk,
                "position_risk": risk_metrics.position_risk,
                "correlation_risk": risk_metrics.correlation_risk,
            })

            # Проверка лимитов риска
            risk_alerts = []
            if risk_metrics.portfolio_risk > self.controller_config.max_portfolio_risk:
                risk_alerts.append("Portfolio risk exceeds limit")

            if risk_metrics.position_risk > self.controller_config.max_risk_per_trade:
                risk_alerts.append("Position risk exceeds limit")

            return {
                "portfolio_risk": risk_metrics.portfolio_risk,
                "position_risk": risk_metrics.position_risk,
                "correlation_risk": risk_metrics.correlation_risk,
                "alerts": risk_alerts,
            }

        except Exception as e:
            logger.error(f"Error managing risks: {e}")
            return {"error": str(e)}

    async def orchestrate_strategies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Оркестрация торговых стратегий."""
        try:
            # Получение статуса всех стратегий
            strategy_statuses = {}
            for strategy_id in self.active_strategies:
                status = await self.strategy_orchestrator.get_strategy_status(strategy_id)
                if status:
                    strategy_statuses[strategy_id] = {
                        "name": status.name,
                        "status": status.status,
                        "performance": status.performance,
                        "risk_level": status.risk_level,
                    }

            # Проверка необходимости ребалансировки
            should_rebalance = self._should_rebalance()
            if should_rebalance:
                await self._perform_rebalance()

            return {
                "active_strategies": strategy_statuses,
                "should_rebalance": should_rebalance,
                "last_rebalance": self.last_rebalance.isoformat() if self.last_rebalance else None,
            }

        except Exception as e:
            logger.error(f"Error orchestrating strategies: {e}")
            return {"error": str(e)}

    async def make_decisions(
        self,
        data: Dict[str, Any],
        performance_data: Dict[str, Any],
        risk_data: Dict[str, Any],
        strategy_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Принятие решений на основе анализа."""
        decisions = []

        try:
            # Решения на основе риска
            if "alerts" in risk_data and risk_data["alerts"]:
                for alert in risk_data["alerts"]:
                    decisions.append({
                        "type": "risk_management",
                        "action": "reduce_exposure",
                        "priority": "high",
                        "reason": alert,
                        "data": {"risk_level": risk_data.get("portfolio_risk", 0.0)},
                    })

            # Решения на основе производительности
            overall_performance = performance_data.get("overall_performance", 0.0)
            if overall_performance < self.controller_config.performance_threshold:
                decisions.append({
                    "type": "performance_optimization",
                    "action": "adjust_strategies",
                    "priority": "medium",
                    "reason": "Performance below threshold",
                    "data": {"performance": overall_performance},
                })

            # Решения на основе стратегий
            if strategy_data.get("should_rebalance", False):
                decisions.append({
                    "type": "portfolio_rebalancing",
                    "action": "rebalance",
                    "priority": "medium",
                    "reason": "Scheduled rebalancing",
                    "data": {"rebalance_interval": self.controller_config.rebalance_interval},
                })

            return decisions

        except Exception as e:
            logger.error(f"Error making decisions: {e}")
            return [{"type": "error", "action": "none", "priority": "low", "reason": str(e), "data": {}}]

    def _should_rebalance(self) -> bool:
        """Проверка необходимости ребалансировки."""
        if not self.controller_config.enable_auto_rebalance:
            return False

        if self.last_rebalance is None:
            return True

        rebalance_interval = self.controller_config.rebalance_interval
        time_since_rebalance = (datetime.now() - self.last_rebalance).total_seconds()

        return time_since_rebalance >= rebalance_interval

    async def _perform_rebalance(self) -> None:
        """Выполнение ребалансировки портфеля."""
        try:
            success = await self.strategy_orchestrator.rebalance()
            if success:
                self.last_rebalance = datetime.now()
                logger.info("Portfolio rebalancing completed")
            else:
                logger.warning("Portfolio rebalancing failed")
        except Exception as e:
            logger.error(f"Error during rebalancing: {e}")

    def _create_portfolio_state(self, data: Dict[str, Any]) -> PortfolioState:
        """Создание объекта PortfolioState из данных."""
        return PortfolioState(
            total_value=data.get("total_value", 0.0),
            cash_balance=data.get("cash_balance", 0.0),
            positions=data.get("positions", {}),
            unrealized_pnl=data.get("unrealized_pnl", 0.0),
            realized_pnl=data.get("realized_pnl", 0.0),
        )

    def get_controller_summary(self) -> Dict[str, Any]:
        """Получение сводки контроллера."""
        return {
            "agent_id": str(self.agent_id),
            "name": self.name,
            "status": self.state.status.value,
            "active_strategies": len(self.active_strategies),
            "portfolio_value": self.portfolio_state.get("total_value", 0.0),
            "risk_metrics": self.risk_metrics,
            "performance_metrics": self.performance_metrics,
            "last_rebalance": self.last_rebalance.isoformat() if self.last_rebalance else None,
            "last_performance_check": self.last_performance_check.isoformat() if self.last_performance_check else None,
        }

    def reset_controller(self) -> None:
        """Сброс состояния контроллера."""
        self.active_strategies.clear()
        self.portfolio_state.clear()
        self.risk_metrics.clear()
        self.performance_metrics.clear()
        self.last_rebalance = None
        self.last_performance_check = None
        logger.info("MetaControllerAgent state reset")

    async def get_signals(self) -> List[ControllerSignal]:
        """Получение сигналов от контроллера."""
        signals = []

        try:
            # Сигналы риска
            risk_alerts = await self.risk_manager.get_risk_alerts()
            signals.extend(risk_alerts)

            # Сигналы производительности
            performance_alerts = await self.performance_analyzer.get_performance_alerts()
            signals.extend(performance_alerts)

            # Сигналы ребалансировки
            if self._should_rebalance():
                signals.append(ControllerSignal(
                    type="rebalancing",
                    action="rebalance_portfolio",
                    priority="medium",
                    data={"reason": "Scheduled rebalancing"},
                ))

            return signals

        except Exception as e:
            logger.error(f"Error getting signals: {e}")
            return []

    def update_portfolio_state(self, portfolio_data: Dict[str, Any]) -> None:
        """Обновление состояния портфеля."""
        self.portfolio_state.update(portfolio_data)

    def get_risk_metrics(self) -> Dict[str, float]:
        """Получение метрик риска."""
        return self.risk_metrics.copy()

    def get_performance_metrics(self) -> Dict[str, float]:
        """Получение метрик производительности."""
        return self.performance_metrics.copy()
