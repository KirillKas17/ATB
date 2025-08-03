"""Анализатор торговых стратегий."""

from typing import Any, Dict, List

from loguru import logger


class StrategyAnalyzer:
    """Анализатор торговых стратегий."""

    def __init__(self) -> None:
        self.strategy_patterns: Dict[str, List[str]] = {
            "risk_management": [
                r"stop_loss",
                r"take_profit",
                r"position_size",
                r"risk_per_trade",
                r"max_drawdown",
            ],
            "indicators": [
                r"SMA|EMA|RSI|MACD|BB|ATR",
                r"moving_average",
                r"relative_strength",
                r"momentum",
            ],
            "execution": [r"order_type", r"slippage", r"execution_time", r"fill_rate"],
        }

    async def analyze_strategies(
        self, strategy_structure: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Анализ торговых стратегий."""
        logger.info("Начало анализа стратегий")

        issues: List[Dict[str, Any]] = []

        # Анализ стратегий
        for strategy_name, strategy_info in strategy_structure.get(
            "strategies", {}
        ).items():
            strategy_issues = await self._analyze_strategy(strategy_name, strategy_info)
            issues.extend(strategy_issues)

        # Анализ агентов
        for agent_name, agent_info in strategy_structure.get("agents", {}).items():
            agent_issues = await self._analyze_agent(agent_name, agent_info)
            issues.extend(agent_issues)

        logger.info(f"Анализ стратегий завершен, найдено {len(issues)} проблем")
        return issues

    async def _analyze_strategy(
        self, strategy_name: str, strategy_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Анализ отдельной стратегии."""
        issues: List[Dict[str, Any]] = []

        # Проверка наличия риск-менеджмента
        risk_management = strategy_info.get("risk_management", [])
        if not risk_management:
            issues.append(
                {
                    "type": "risk_management",
                    "severity": "high",
                    "strategy": strategy_name,
                    "description": "Отсутствует риск-менеджмент",
                    "suggestion": "Добавить stop-loss, take-profit и position sizing",
                    "priority": 0.9,
                }
            )

        # Проверка технических индикаторов
        indicators = strategy_info.get("indicators", [])
        if not indicators:
            issues.append(
                {
                    "type": "indicators",
                    "severity": "medium",
                    "strategy": strategy_name,
                    "description": "Отсутствуют технические индикаторы",
                    "suggestion": "Добавить базовые индикаторы (SMA, RSI, MACD)",
                    "priority": 0.6,
                }
            )

        # Проверка сложности
        complexity = strategy_info.get("complexity", {})
        if complexity.get("cyclomatic", 0) > 15:
            issues.append(
                {
                    "type": "complexity",
                    "severity": "medium",
                    "strategy": strategy_name,
                    "description": f"Высокая сложность стратегии: {complexity['cyclomatic']}",
                    "suggestion": "Упростить логику стратегии",
                    "priority": 0.7,
                }
            )

        # Проверка качества
        quality = strategy_info.get("quality", {})
        if quality.get("score", 100) < 80:
            issues.append(
                {
                    "type": "quality",
                    "severity": "medium",
                    "strategy": strategy_name,
                    "description": f"Низкое качество кода: {quality['score']}/100",
                    "suggestion": "Улучшить качество кода стратегии",
                    "priority": 0.6,
                }
            )

        # Проверка методов
        methods = strategy_info.get("methods", [])
        required_methods = ["execute", "calculate_signals", "update"]
        missing_methods = [
            method
            for method in required_methods
            if not any(m["name"] == method for m in methods)
        ]

        if missing_methods:
            issues.append(
                {
                    "type": "methods",
                    "severity": "medium",
                    "strategy": strategy_name,
                    "description": f"Отсутствуют методы: {', '.join(missing_methods)}",
                    "suggestion": "Добавить недостающие методы",
                    "priority": 0.5,
                }
            )

        return issues

    async def _analyze_agent(
        self, agent_name: str, agent_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Анализ агента."""
        issues: List[Dict[str, Any]] = []

        # Проверка возможностей
        capabilities = agent_info.get("capabilities", [])
        if not capabilities:
            issues.append(
                {
                    "type": "capabilities",
                    "severity": "medium",
                    "agent": agent_name,
                    "description": "Не определены возможности агента",
                    "suggestion": "Добавить описание возможностей агента",
                    "priority": 0.5,
                }
            )

        # Проверка методов
        methods = agent_info.get("methods", [])
        if len(methods) < 3:
            issues.append(
                {
                    "type": "methods",
                    "severity": "low",
                    "agent": agent_name,
                    "description": f"Мало методов: {len(methods)}",
                    "suggestion": "Добавить больше функциональности",
                    "priority": 0.3,
                }
            )

        # Проверка сложности
        complexity = agent_info.get("complexity", {})
        if complexity.get("cyclomatic", 0) > 20:
            issues.append(
                {
                    "type": "complexity",
                    "severity": "medium",
                    "agent": agent_name,
                    "description": f"Высокая сложность агента: {complexity['cyclomatic']}",
                    "suggestion": "Упростить логику агента",
                    "priority": 0.6,
                }
            )

        return issues
