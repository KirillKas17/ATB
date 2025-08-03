"""
Эволюционный decision reasoner
Расширяет функциональность decision_reasoner эволюционными возможностями
"""

import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch
import torch.nn as nn
from loguru import logger

from infrastructure.core.evolution_manager import (
    EvolvableComponent,
    register_for_evolution,
)
from infrastructure.ml_services.decision_reasoner import DecisionReasoner, AggregatedSignal

# Type aliases for better mypy support
Series = pd.Series
DataFrame = pd.DataFrame


@dataclass
class EvolvableDecisionConfig:
    """Конфигурация эволюционного decision reasoner"""

    # Базовые параметры
    confidence_threshold: float = 0.7
    reasoning_depth: int = 3
    # Эволюционные параметры
    learning_rate: float = 1e-3
    adaptation_rate: float = 0.01
    evolution_threshold: float = 0.6
    # Параметры модели
    model_hidden_dim: int = 128
    model_dropout: float = 0.2
    # Параметры обучения
    batch_size: int = 32
    max_history: int = 1000
    # Параметры эволюции
    architecture_mutation_rate: float = 0.1
    parameter_mutation_rate: float = 0.05
    # Общие параметры
    name: str = "evolvable_decision_reasoner"
    version: str = "1.0.0"
    log_dir: str = "logs"


class DecisionMLModel(nn.Module):
    """ML модель для оптимизации принятия решений"""

    def __init__(self, input_dim: int = 30, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5),  # 5 типов решений
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EvolvableDecisionReasoner(DecisionReasoner, EvolvableComponent):
    """Эволюционный decision reasoner с ML-оптимизацией"""

    def __init__(self, config: Optional[EvolvableDecisionConfig] = None) -> None:
        # Инициализация базового reasoner с совместимым типом конфигурации
        if isinstance(config, EvolvableDecisionConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = EvolvableDecisionConfig(**config)
        else:
            self.config = EvolvableDecisionConfig()
            
        # Для совместимости с базовым классом (если нужно dict)
        self._config_dict: dict[str, Any] = self.config.__dict__ if hasattr(self.config, '__dict__') else {}
        
        # ML модель для оптимизации решений
        self.model = DecisionMLModel()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        # Эволюционные метрики
        self.performance: float = 0.0
        self.confidence: float = 0.0
        self.training_data: List[Dict[str, Any]] = []
        self.decision_history: List[Dict[str, Any]] = []
        
        # Исправление: правильное присваивание сигналов
        self.aggregated_signals: List[AggregatedSignal] = []
        # Регистрация в эволюционном менеджере
        register_for_evolution(self)
        logger.info("EvolvableDecisionReasoner initialized")

    async def adapt(self, data: Dict[str, Any]) -> bool:
        """Быстрая адаптация к новым данным"""
        try:
            if "market_conditions" in data:
                # Адаптация порогов на основе рыночных условий
                volatility = data["market_conditions"].get("volatility", 0.1)
                complexity = data["market_conditions"].get("complexity", 0.5)
                # Адаптация порога уверенности - исправление: используем правильный атрибут
                if hasattr(self.config, 'confidence_threshold'):
                    self.config.confidence_threshold = max(
                        0.5,
                        min(0.9, self.config.confidence_threshold * (1 + volatility * 0.1)),
                    )
                # Адаптация глубины рассуждений - исправление: используем правильный атрибут
                if hasattr(self.config, 'reasoning_depth'):
                    if complexity > 0.7:
                        self.config.reasoning_depth = min(
                            5, self.config.reasoning_depth + 1
                        )
                    elif complexity < 0.3:
                        self.config.reasoning_depth = max(
                            2, self.config.reasoning_depth - 1
                        )
                logger.debug(
                    f"DecisionReasoner adapted: confidence_threshold={getattr(self.config, 'confidence_threshold', 0.7):.3f}, reasoning_depth={getattr(self.config, 'reasoning_depth', 3)}"
                )
                return True
        except Exception as e:
            logger.error(f"DecisionReasoner adapt error: {e}")
        return False

    async def learn(self, data: Dict[str, Any]) -> bool:
        """Обучение на новых данных"""
        try:
            if "market_data" in data and "decisions" in data:
                # Извлечение признаков
                features = self._extract_decision_features(data["market_data"])
                targets = self._extract_decision_targets(data["decisions"])
                if len(features) > 0 and len(targets) > 0:
                    # Обучение модели
                    features_tensor = torch.FloatTensor(list(features.values())).unsqueeze(0)
                    targets_tensor = torch.FloatTensor(targets)
                    self.optimizer.zero_grad()
                    predictions = self.model(features_tensor)
                    loss = nn.MSELoss()(predictions, targets_tensor)
                    loss.backward()
                    self.optimizer.step()
                    # Обновление метрик
                    self.performance = 1.0 / (1.0 + loss.item())
                    self.confidence = float(torch.max(predictions).item())
                    # Сохранение данных
                    self.training_data.append(
                        {
                            "features": features,
                            "targets": targets,
                            "timestamp": datetime.now(),
                        }
                    )
                    # Ограничение размера истории
                    if len(self.training_data) > 1000:
                        self.training_data = self.training_data[-1000:]
                    logger.debug(f"DecisionReasoner learned: loss={loss.item():.6f}")
                    return True
        except Exception as e:
            logger.error(f"DecisionReasoner learn error: {e}")
        return False

    async def evolve(self, data: Dict[str, Any]) -> bool:
        """Полная эволюция компонента"""
        try:
            # Эволюция архитектуры модели
            await self._evolve_model_architecture()
            # Эволюция параметров
            await self._evolve_parameters()
            # Переобучение на истории
            await self._retrain_on_history()
            logger.info("DecisionReasoner evolved successfully")
            return True
        except Exception as e:
            logger.error(f"DecisionReasoner evolve error: {e}")
            return False

    def get_performance(self) -> float:
        """Получение производительности"""
        return self.performance

    def get_confidence(self) -> float:
        """Получение уверенности"""
        return self.confidence

    def save_state(self, path: str) -> bool:
        """Сохранение состояния"""
        try:
            state = {
                "config": self.config.__dict__,  # сохраняем как dict
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "performance": self.performance,
                "confidence": self.confidence,
                "training_data": self.training_data[-500:],
                "decision_history": self.decision_history[-100:],
            }
            os.makedirs(path, exist_ok=True)
            torch.save(state, f"{path}/evolvable_decision_reasoner_state.pth")
            return True
        except Exception as e:
            logger.error(f"DecisionReasoner save_state error: {e}")
            return False

    def load_state(self, path: str) -> bool:
        """Загрузка состояния из файла"""
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    state = pickle.load(f)
                # Восстанавливаем конфигурацию
                if "config" in state:
                    self.config = EvolvableDecisionConfig(**state["config"])
                # Восстанавливаем модель
                if "model_state" in state:
                    self.model.load_state_dict(state["model_state"])
                # Восстанавливаем оптимизатор
                if "optimizer_state" in state:
                    self.optimizer.load_state_dict(state["optimizer_state"])
                # Восстанавливаем метрики
                self.performance = state.get("performance", 0.0)
                self.confidence = state.get("confidence", 0.0)
                self.training_data = state.get("training_data", [])
                self.decision_history = state.get("decision_history", [])
                logger.info(f"DecisionReasoner state loaded from {path}")
                return True
        except Exception as e:
            logger.error(f"DecisionReasoner load_state error: {e}")
        return False

    def _extract_decision_features(self, market_data: DataFrame) -> Dict[str, float]:
        """Извлечение признаков для принятия решений"""
        features = {}
        try:
            # Рыночные данные
            features.update(
                {
                    "price_change": market_data["close"].pct_change().iloc[-1],
                    "volume_change": market_data["volume"].pct_change().iloc[-1],
                    "price_ma_ratio": market_data["close"].rolling(20).mean().iloc[-1]
                    / market_data["close"].iloc[-1]
                    - 1,
                    "volume_ma_ratio": market_data["volume"].rolling(20).mean().iloc[-1]
                    / market_data["volume"].iloc[-1]
                    - 1,
                }
            )
            # Волатильность
            features.update(
                {
                    "price_volatility": market_data["close"]
                    .pct_change()
                    .rolling(10)
                    .std()
                    .iloc[-1],
                    "volume_volatility": market_data["volume"]
                    .pct_change()
                    .rolling(10)
                    .std()
                    .iloc[-1],
                }
            )
            # Тренды
            features.update(
                {
                    "price_trend": market_data["close"]
                    .pct_change()
                    .rolling(5)
                    .mean()
                    .iloc[-1],
                    "volume_trend": market_data["volume"]
                    .pct_change()
                    .rolling(5)
                    .mean()
                    .iloc[-1],
                }
            )
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}

    def _extract_decision_targets(self, decisions: List[Dict]) -> List[float]:
        """Извлечение целевых значений для обучения"""
        targets = []
        try:
            for decision in decisions:
                # Преобразование решения в числовое значение
                if decision.get("action") == "BUY":
                    targets.append(1.0)
                elif decision.get("action") == "SELL":
                    targets.append(-1.0)
                elif decision.get("action") == "HOLD":
                    targets.append(0.0)
                else:
                    targets.append(0.0)
            return targets
        except Exception as e:
            logger.error(f"Error extracting targets: {e}")
            return []

    async def _evolve_model_architecture(self) -> None:
        """Эволюция архитектуры модели"""
        try:
            # Анализ производительности - исправление: используем правильный атрибут
            if self.performance < self.config.evolution_threshold:
                # Увеличение сложности модели
                current_hidden_dim = self.model.net[0].out_features
                new_hidden_dim = min(current_hidden_dim * 2, 512)
                # Создание новой модели
                new_model = DecisionMLModel(input_dim=30, hidden_dim=new_hidden_dim)
                # Копирование весов
                with torch.no_grad():
                    new_model.net[0].weight[:current_hidden_dim] = self.model.net[
                        0
                    ].weight
                    new_model.net[0].bias[:current_hidden_dim] = self.model.net[0].bias
                self.model = new_model
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(), lr=self.config.learning_rate
                )
                logger.info(f"Model architecture evolved: hidden_dim={new_hidden_dim}")
        except Exception as e:
            logger.error(f"Error evolving model architecture: {e}")

    async def _evolve_parameters(self) -> None:
        """Эволюция параметров конфигурации"""
        try:
            # Адаптация learning rate - исправление: используем правильный атрибут
            if self.performance < 0.5:
                self.config.learning_rate *= 1.1
            elif self.performance > 0.8:
                self.config.learning_rate *= 0.9
            # Адаптация confidence threshold - исправление: используем правильный атрибут
            if self.confidence < 0.6:
                self.config.confidence_threshold *= 0.95
            elif self.confidence > 0.9:
                self.config.confidence_threshold *= 1.05
            # Ограничения
            self.config.learning_rate = max(1e-5, min(1e-2, self.config.learning_rate))
            self.config.confidence_threshold = max(
                0.3, min(0.95, self.config.confidence_threshold)
            )
            logger.info(
                f"Parameters evolved: lr={self.config.learning_rate:.6f}, confidence_threshold={self.config.confidence_threshold:.3f}"
            )
        except Exception as e:
            logger.error(f"Error evolving parameters: {e}")

    async def _retrain_on_history(self) -> None:
        """Переобучение на исторических данных"""
        try:
            if len(self.training_data) < 10:
                return
            # Подготовка данных
            features_list = [item["features"] for item in self.training_data]
            targets_list = [item["targets"] for item in self.training_data]
            # Объединение признаков
            all_features = []
            for features in features_list:
                if isinstance(features, dict):
                    all_features.extend(list(features.values()))
                else:
                    all_features.extend(features)
            # Объединение целей
            all_targets = []
            for targets in targets_list:
                all_targets.extend(targets)
            if len(all_features) > 0 and len(all_targets) > 0:
                # Обучение
                features_tensor = torch.FloatTensor(all_features).unsqueeze(0)
                targets_tensor = torch.FloatTensor(all_targets)
                self.optimizer.zero_grad()
                predictions = self.model(features_tensor)
                loss = nn.MSELoss()(predictions, targets_tensor)
                loss.backward()
                self.optimizer.step()
                logger.info(f"Retrained on history: loss={loss.item():.6f}")
        except Exception as e:
            logger.error(f"Error retraining on history: {e}")

    async def _reason_at_level(
        self, 
        market_data: DataFrame, 
        signals: List[Any], 
        features: Dict[str, float], 
        level: int, 
        ml_prediction: float
    ) -> Dict[str, Any]:
        """Рассуждение на определенном уровне"""
        try:
            # Базовая логика рассуждения
            confidence = min(0.9, 0.5 + level * 0.1 + abs(ml_prediction) * 0.2)
            performance = max(0.3, 0.5 + abs(ml_prediction) * 0.3)
            
            return {
                "level": level,
                "confidence": confidence,
                "performance": performance,
                "reasoning": f"Level {level} reasoning with ML prediction {ml_prediction:.3f}",
                "decision": "hold" if abs(ml_prediction) < 0.3 else ("buy" if ml_prediction > 0 else "sell")
            }
        except Exception as e:
            logger.error(f"Error in _reason_at_level: {e}")
            return {
                "level": level,
                "confidence": 0.0,
                "performance": 0.0,
                "reasoning": f"Error: {str(e)}",
                "decision": "hold"
            }

    def _aggregate_reasoning_results(self, reasoning_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Агрегация результатов рассуждений"""
        try:
            if not reasoning_results:
                return {
                    "decision": "hold",
                    "confidence": 0.0,
                    "reasoning": "No reasoning results available",
                    "performance": 0.0
                }
            
            # Взвешенная агрегация по уровням
            total_confidence = 0.0
            total_performance = 0.0
            weighted_decision = 0.0
            total_weight = 0.0
            
            for result in reasoning_results:
                level = result.get("level", 0)
                confidence = result.get("confidence", 0.0)
                performance = result.get("performance", 0.0)
                decision = result.get("decision", "hold")
                
                # Вес по уровню (более глубокие уровни имеют больший вес)
                weight = level + 1
                
                total_confidence += confidence * weight
                total_performance += performance * weight
                
                # Преобразование решения в числовое значение
                decision_value = 0.0
                if decision == "buy":
                    decision_value = 1.0
                elif decision == "sell":
                    decision_value = -1.0
                
                weighted_decision += decision_value * weight
                total_weight += weight
            
            if total_weight > 0:
                avg_confidence = total_confidence / total_weight
                avg_performance = total_performance / total_weight
                final_decision_value = weighted_decision / total_weight
                
                # Определение финального решения
                if abs(final_decision_value) < 0.2:
                    final_decision = "hold"
                elif final_decision_value > 0:
                    final_decision = "buy"
                else:
                    final_decision = "sell"
                
                return {
                    "decision": final_decision,
                    "confidence": avg_confidence,
                    "reasoning": f"Aggregated from {len(reasoning_results)} reasoning levels",
                    "performance": avg_performance
                }
            else:
                return {
                    "decision": "hold",
                    "confidence": 0.0,
                    "reasoning": "No valid reasoning results",
                    "performance": 0.0
                }
        except Exception as e:
            logger.error(f"Error in _aggregate_reasoning_results: {e}")
            return {
                "decision": "hold",
                "confidence": 0.0,
                "reasoning": f"Error in aggregation: {str(e)}",
                "performance": 0.0
            }

    async def reason_about_market(
        self, market_data: DataFrame, signals: List[Any]
    ) -> Dict[str, Any]:
        """Рассуждение о рыночной ситуации с эволюционными улучшениями"""
        try:
            # Извлечение признаков
            features = self._extract_decision_features(market_data)
            
            # ML предсказание
            features_tensor = torch.FloatTensor(list(features.values())).unsqueeze(0)
            with torch.no_grad():
                ml_prediction = self.model(features_tensor).item()
            
            # Адаптивная глубина рассуждений
            reasoning_depth = getattr(self.config, 'reasoning_depth', 3)
            
            # Многоуровневое рассуждение
            reasoning_results = []
            for level in range(reasoning_depth):
                level_result = await self._reason_at_level(
                    market_data, signals, features, level, ml_prediction
                )
                reasoning_results.append(level_result)
                
                # Адаптивная остановка
                if level_result.get("confidence", 0.0) > getattr(self.config, 'confidence_threshold', 0.7):
                    break
            
            # Агрегация результатов
            final_decision = self._aggregate_reasoning_results(reasoning_results)
            
            # Обновление метрик
            self.confidence = final_decision.get("confidence", 0.0)
            self.performance = final_decision.get("performance", 0.0)
            
            # Сохранение в историю
            self.decision_history.append({
                "timestamp": datetime.now(),
                "decision": final_decision,
                "features": features,
                "ml_prediction": ml_prediction
            })
            
            # Ограничение размера истории
            if len(self.decision_history) > getattr(self.config, 'max_history', 1000):
                self.decision_history = self.decision_history[-getattr(self.config, 'max_history', 1000):]
            
            return final_decision
            
        except Exception as e:
            logger.error(f"DecisionReasoner reason_about_market error: {e}")
            return {
                "decision": "hold",
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "performance": 0.0
            }

    def get_evolution_stats(self) -> Dict[str, Any]:
        """Получение статистики эволюции"""
        return {
            "performance": self.performance,
            "confidence": self.confidence,
            "learning_rate": self.config.learning_rate,
            "confidence_threshold": self.config.confidence_threshold,
            "reasoning_depth": self.config.reasoning_depth,
            "training_data_size": len(self.training_data),
            "decision_history_size": len(self.decision_history),
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
        }
