"""
Фабрика для создания Core менеджеров с правильными зависимостями.
"""

from typing import Dict, Any, Optional
from loguru import logger

from shared.event_bus import EventBus
from infrastructure.core.portfolio_manager import PortfolioManager
from infrastructure.core.risk_manager import RiskManager


class ManagersFactory:
    """Фабрика для создания Portfolio и Risk менеджеров с правильными зависимостями."""
    
    @staticmethod
    def create_default_config() -> Dict[str, Any]:
        """Создаёт дефолтную конфигурацию для менеджеров."""
        return {
            # Portfolio Manager config
            "portfolio": {
                "max_positions": 10,
                "max_position_size": 0.1,
                "rebalance_threshold": 0.05,
                "base_currency": "USDT",
                "initial_balance": 10000.0,
                "tracking_enabled": True,
                "performance_calculation": True,
                "auto_rebalance": False,
                "diversification_min": 3,
                "diversification_max": 8
            },
            
            # Risk Manager config  
            "risk": {
                "max_daily_loss": 0.02,
                "max_drawdown": 0.05,
                "max_leverage": 3.0,
                "var_confidence": 0.95,
                "risk_free_rate": 0.02,
                "correlation_threshold": 0.8,
                "volatility_window": 30,
                "stress_test_enabled": True,
                "emergency_stop_loss": 0.10,
                "position_limit_percent": 0.15
            },
            
            # Event Bus config
            "event_bus": {
                "max_events": 1000,
                "batch_processing": True,
                "async_mode": True,
                "persistent_events": False
            }
        }
    
    @staticmethod
    def create_event_bus(config: Optional[Dict[str, Any]] = None) -> EventBus:
        """Создаёт EventBus с дефолтной конфигурацией."""
        try:
            if config is None:
                config = ManagersFactory.create_default_config()["event_bus"]
            
            event_bus = EventBus()
            logger.info("EventBus создан с дефолтной конфигурацией")
            return event_bus
            
        except Exception as e:
            logger.error(f"Ошибка создания EventBus: {e}")
            raise
    
    @staticmethod
    def create_portfolio_manager(
        event_bus: Optional[EventBus] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> PortfolioManager:
        """Создаёт PortfolioManager с правильными зависимостями."""
        try:
            # Создаём event_bus если не предоставлен
            if event_bus is None:
                event_bus = ManagersFactory.create_event_bus()
            
            # Создаём config если не предоставлен
            if config is None:
                full_config = ManagersFactory.create_default_config()
                config = full_config["portfolio"]
            
            portfolio_manager = PortfolioManager(event_bus=event_bus, config=config)
            logger.info("PortfolioManager создан с дефолтной конфигурацией")
            return portfolio_manager
            
        except Exception as e:
            logger.error(f"Ошибка создания PortfolioManager: {e}")
            raise
    
    @staticmethod
    def create_risk_manager(
        event_bus: Optional[EventBus] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> RiskManager:
        """Создаёт RiskManager с правильными зависимостями."""
        try:
            # Создаём event_bus если не предоставлен
            if event_bus is None:
                event_bus = ManagersFactory.create_event_bus()
            
            # Создаём config если не предоставлен
            if config is None:
                full_config = ManagersFactory.create_default_config()
                config = full_config["risk"]
            
            risk_manager = RiskManager(event_bus=event_bus, config=config)
            logger.info("RiskManager создан с дефолтной конфигурацией")
            return risk_manager
            
        except Exception as e:
            logger.error(f"Ошибка создания RiskManager: {e}")
            raise
    
    @staticmethod
    def create_managers_suite(
        shared_event_bus: bool = True,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Создаёт полный набор менеджеров с общим event_bus."""
        try:
            config = custom_config or ManagersFactory.create_default_config()
            
            # Создаём общий event_bus
            event_bus = ManagersFactory.create_event_bus(config.get("event_bus"))
            
            if shared_event_bus:
                # Используем общий event_bus для всех менеджеров
                portfolio_manager = ManagersFactory.create_portfolio_manager(
                    event_bus=event_bus,
                    config=config.get("portfolio")
                )
                risk_manager = ManagersFactory.create_risk_manager(
                    event_bus=event_bus, 
                    config=config.get("risk")
                )
            else:
                # Создаём отдельные event_bus для каждого менеджера
                portfolio_manager = ManagersFactory.create_portfolio_manager(
                    config=config.get("portfolio")
                )
                risk_manager = ManagersFactory.create_risk_manager(
                    config=config.get("risk")
                )
            
            managers_suite = {
                "event_bus": event_bus,
                "portfolio_manager": portfolio_manager,
                "risk_manager": risk_manager,
                "config": config
            }
            
            logger.info("Managers suite создан успешно")
            return managers_suite
            
        except Exception as e:
            logger.error(f"Ошибка создания managers suite: {e}")
            raise
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Валидирует конфигурацию менеджеров."""
        try:
            required_sections = ["portfolio", "risk", "event_bus"]
            
            for section in required_sections:
                if section not in config:
                    logger.warning(f"Отсутствует секция конфигурации: {section}")
                    return False
            
            # Валидация portfolio config
            portfolio_config = config["portfolio"]
            portfolio_required = ["max_positions", "base_currency", "initial_balance"]
            for field in portfolio_required:
                if field not in portfolio_config:
                    logger.warning(f"Отсутствует поле portfolio.{field}")
                    return False
            
            # Валидация risk config
            risk_config = config["risk"]
            risk_required = ["max_daily_loss", "max_drawdown", "max_leverage"]
            for field in risk_required:
                if field not in risk_config:
                    logger.warning(f"Отсутствует поле risk.{field}")
                    return False
            
            logger.info("Конфигурация менеджеров валидна")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка валидации конфигурации: {e}")
            return False


def create_default_portfolio_manager() -> PortfolioManager:
    """Быстрое создание PortfolioManager с дефолтными настройками."""
    return ManagersFactory.create_portfolio_manager()


def create_default_risk_manager() -> RiskManager:
    """Быстрое создание RiskManager с дефолтными настройками."""
    return ManagersFactory.create_risk_manager()


def create_integrated_managers() -> Dict[str, Any]:
    """Создаёт интегрированный набор менеджеров с общим event_bus."""
    return ManagersFactory.create_managers_suite(shared_event_bus=True)