"""
Эволюционный агент новостей
Автоматически адаптируется к изменениям рынка и анализирует новостные данные
"""

import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger

from infrastructure.core.evolution_manager import (
    EvolvableComponent,
    register_for_evolution,
)


class NewsML(nn.Module):
    """ML модель для анализа новостей"""

    def __init__(
        self, input_dim: int = 30, hidden_dim: int = 64, output_dim: int = 3
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),  # sentiment, impact, confidence
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


class EvolvableNewsAgent(EvolvableComponent):
    """Эволюционный агент новостей"""

    def __init__(self, name: str = "evolvable_news"):
        super().__init__(name)
        # ML модель для эволюции
        self.ml_model = NewsML()
        self.optimizer = torch.optim.Adam(self.ml_model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        # Метрики производительности
        self.performance_metric = 0.5
        self.confidence_metric = 0.5
        self.successful_predictions = 0
        self.total_predictions = 0
        # История данных для обучения
        self.training_data: List[Dict[str, Any]] = []
        self.max_training_samples: int = 10000
        # Параметры анализа новостей
        self.sentiment_threshold = 0.3
        self.impact_threshold = 0.5
        self.confidence_threshold = 0.7
        # Регистрация в эволюционном менеджере
        register_for_evolution(self)
        logger.info(f"EvolvableNewsAgent initialized: {name}")

    async def adapt(self, data: Any) -> bool:
        """Быстрая адаптация к новым данным"""
        try:
            if isinstance(data, dict) and "news_data" in data:
                news_data = data["news_data"]
                # Адаптация на основе новостных данных
                if news_data:
                    # Анализ тональности новостей
                    sentiment_scores = [item.get("sentiment", 0.0) for item in news_data]
                    if sentiment_scores:
                        avg_sentiment = np.mean(sentiment_scores)
                        # Адаптация порогов
                        if avg_sentiment > 0.5:  # Позитивные новости
                            self.sentiment_threshold *= 1.1
                        elif avg_sentiment < -0.5:  # Негативные новости
                            self.sentiment_threshold *= 0.9
                        # Ограничения
                        self.sentiment_threshold = max(0.1, min(0.8, self.sentiment_threshold))
                        logger.debug(f"NewsAgent adapted: sentiment_threshold={self.sentiment_threshold:.3f}")
                        return True
        except Exception as e:
            logger.error(f"Error in NewsAgent adaptation: {e}")
        return False

    async def learn(self, data: Any) -> bool:
        """Обучение на новых данных"""
        try:
            if isinstance(data, dict) and "news_data" in data:
                news_data = data["news_data"]
                market_impact = data.get("market_impact", {})
                # Подготовка данных для обучения
                features = self._extract_features(news_data, market_impact)
                targets = self._extract_targets(market_impact)
                if len(features) > 0 and len(targets) > 0:
                    # Обучение модели
                    features_tensor = torch.FloatTensor(features).unsqueeze(0)
                    targets_tensor = torch.FloatTensor(targets).unsqueeze(0)
                    self.optimizer.zero_grad()
                    predictions = self.ml_model(features_tensor)
                    loss = self.criterion(predictions, targets_tensor)
                    loss.backward()
                    self.optimizer.step()
                    # Обновление метрик
                    self._update_metrics(
                        loss.item(), predictions.detach().numpy()[0], targets
                    )
                    # Сохранение данных для обучения
                    self.training_data.append(
                        {
                            "features": features,
                            "targets": targets,
                            "timestamp": datetime.now(),
                        }
                    )
                    # Ограничение размера истории
                    if len(self.training_data) > self.max_training_samples:
                        self.training_data = self.training_data[-self.max_training_samples:]
                    logger.debug(f"NewsAgent learned: loss={loss.item():.6f}")
                    return True
        except Exception as e:
            logger.error(f"Error in NewsAgent learning: {e}")
        return False

    async def evolve(self, data: Any) -> bool:
        """Полная эволюция компонента"""
        try:
            # Эволюция архитектуры модели
            await self._evolve_model_architecture()
            # Эволюция параметров анализа
            await self._evolve_analysis_parameters()
            # Переобучение на истории
            await self._retrain_on_history()
            logger.info("NewsAgent evolved successfully")
            return True
        except Exception as e:
            logger.error(f"Error in NewsAgent evolution: {e}")
            return False

    def get_performance(self) -> float:
        """Получение производительности"""
        return self.performance_metric

    def get_confidence(self) -> float:
        """Получение уверенности"""
        return self.confidence_metric

    def save_state(self, path: str) -> bool:
        """Сохранение состояния"""
        try:
            state = {
                "name": self.name,
                "model_state": self.ml_model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "performance_metric": self.performance_metric,
                "confidence_metric": self.confidence_metric,
                "successful_predictions": self.successful_predictions,
                "total_predictions": self.total_predictions,
                "training_data": self.training_data[-1000:],
                "sentiment_threshold": self.sentiment_threshold,
                "impact_threshold": self.impact_threshold,
                "confidence_threshold": self.confidence_threshold,
            }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state, f"{path}/evolvable_news_agent_state.pth")
            return True
        except Exception as e:
            logger.error(f"NewsAgent save_state error: {e}")
            return False

    def load_state(self, path: str) -> bool:
        """Загрузка состояния"""
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    state = pickle.load(f)
                # Восстанавливаем модель
                if "model_state" in state:
                    self.ml_model.load_state_dict(state["model_state"])
                # Восстанавливаем оптимизатор
                if "optimizer_state" in state:
                    self.optimizer.load_state_dict(state["optimizer_state"])
                # Восстанавливаем метрики
                self.performance_metric = state.get("performance_metric", 0.5)
                self.confidence_metric = state.get("confidence_metric", 0.5)
                self.successful_predictions = state.get("successful_predictions", 0)
                self.total_predictions = state.get("total_predictions", 0)
                self.training_data = state.get("training_data", [])
                self.sentiment_threshold = state.get("sentiment_threshold", 0.3)
                self.impact_threshold = state.get("impact_threshold", 0.5)
                self.confidence_threshold = state.get("confidence_threshold", 0.7)
                logger.info(f"NewsAgent state loaded from {path}")
                return True
        except Exception as e:
            logger.error(f"NewsAgent load_state error: {e}")
        return False

    def _extract_features(
        self, news_data: List[Dict], market_impact: Dict
    ) -> List[float]:
        """Извлечение признаков из новостных данных"""
        try:
            if not news_data:
                return []
            # Анализ новостных данных
            sentiment_scores = []
            impact_scores = []
            confidence_scores = []
            volume_scores = []
            for news in news_data:
                sentiment_scores.append(news.get("sentiment", 0.0))
                impact_scores.append(news.get("impact", 0.0))
                confidence_scores.append(news.get("confidence", 0.0))
                volume_scores.append(news.get("volume", 0.0))
            # Статистики
            features = [
                np.mean(sentiment_scores),
                np.std(sentiment_scores),
                np.mean(impact_scores),
                np.std(impact_scores),
                np.mean(confidence_scores),
                np.std(confidence_scores),
                np.mean(volume_scores),
                np.std(volume_scores),
                len(news_data),  # Количество новостей
            ]
            # Нормализация
            features = [float(f) if not np.isnan(f) else 0.0 for f in features]
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return []

    def _extract_targets(self, market_impact: Dict) -> List[float]:
        """Извлечение целевых значений для обучения"""
        try:
            sentiment = market_impact.get("sentiment", 0.0)
            impact = market_impact.get("impact", 0.0)
            confidence = market_impact.get("confidence", 0.0)
            return [sentiment, impact, confidence]
        except Exception as e:
            logger.error(f"Error extracting targets: {e}")
            return [0.0, 0.0, 0.0]

    def _update_metrics(
        self, loss: float, predictions: np.ndarray, targets: List[float]
    ) -> None:
        """Обновление метрик производительности"""
        try:
            self.total_predictions += 1
            # Оценка точности предсказаний
            mse = np.mean((predictions - targets) ** 2)
            if mse < 0.1:  # Хорошее предсказание
                self.successful_predictions += 1
            # Обновление уверенности
            confidence = 1.0 - mse
            self.confidence_metric = 0.9 * self.confidence_metric + 0.1 * confidence
            # Обновление производительности
            accuracy = self.successful_predictions / self.total_predictions
            self.performance_metric = 0.9 * self.performance_metric + 0.1 * accuracy
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    async def _evolve_model_architecture(self) -> None:
        """Эволюция архитектуры модели"""
        try:
            # Простая эволюция - изменение размеров слоев
            current_hidden_dim = self.ml_model.net[0].out_features
            new_hidden_dim = current_hidden_dim + np.random.randint(-10, 11)
            new_hidden_dim = max(32, min(128, new_hidden_dim))
            # Создание новой модели
            new_model = NewsML(
                input_dim=30,
                hidden_dim=new_hidden_dim,
                output_dim=3
            )
            # Копирование весов где возможно
            try:
                new_model.net[0].weight.data[:current_hidden_dim, :] = self.ml_model.net[0].weight.data
                new_model.net[0].bias.data[:current_hidden_dim] = self.ml_model.net[0].bias.data
            except:
                pass
            self.ml_model = new_model
            self.optimizer = torch.optim.Adam(self.ml_model.parameters(), lr=1e-3)
            logger.info(f"Model architecture evolved: hidden_dim={new_hidden_dim}")
        except Exception as e:
            logger.error(f"Error evolving model architecture: {e}")

    async def _evolve_analysis_parameters(self) -> None:
        """Эволюция параметров анализа"""
        try:
            # Адаптация параметров на основе производительности
            if self.performance_metric < 0.5:
                self.sentiment_threshold *= 0.9  # Более чувствительно
                self.impact_threshold *= 0.9
            elif self.performance_metric > 0.8:
                self.sentiment_threshold *= 1.1  # Менее чувствительно
                self.impact_threshold *= 1.1
            # Ограничения
            self.sentiment_threshold = max(0.1, min(0.8, self.sentiment_threshold))
            self.impact_threshold = max(0.2, min(0.9, self.impact_threshold))
            logger.info(f"Analysis parameters evolved: sentiment_threshold={self.sentiment_threshold:.3f}, impact_threshold={self.impact_threshold:.3f}")
        except Exception as e:
            logger.error(f"Error evolving analysis parameters: {e}")

    async def _retrain_on_history(self) -> None:
        """Переобучение на исторических данных"""
        try:
            if len(self.training_data) < 10:
                return
            # Подготовка данных
            features_list = [item["features"] for item in self.training_data]
            targets_list = [item["targets"] for item in self.training_data]
            if len(features_list) > 0 and len(targets_list) > 0:
                # Объединение признаков
                all_features = []
                for features in features_list:
                    all_features.extend(features)
                # Объединение целей
                all_targets = []
                for targets in targets_list:
                    all_targets.extend(targets)
                if len(all_features) > 0 and len(all_targets) > 0:
                    # Обучение
                    features_tensor = torch.FloatTensor(all_features).unsqueeze(0)
                    targets_tensor = torch.FloatTensor(all_targets).unsqueeze(0)
                    self.optimizer.zero_grad()
                    predictions = self.ml_model(features_tensor)
                    loss = self.criterion(predictions, targets_tensor)
                    loss.backward()
                    self.optimizer.step()
                    logger.info(f"Retrained on history: loss={loss.item():.6f}")
        except Exception as e:
            logger.error(f"Error retraining on history: {e}")

    async def analyze_news(
        self, news_data: List[Dict], market_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Анализ новостных данных"""
        try:
            # Извлечение признаков
            market_impact = market_context or {}
            features = self._extract_features(news_data, market_impact)
            if len(features) == 0:
                return {
                    "sentiment": 0.0,
                    "impact": 0.0,
                    "confidence": 0.0,
                    "reasoning": "No features available"
                }
            # ML предсказание
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            with torch.no_grad():
                predictions = self.ml_model(features_tensor)
                sentiment = predictions[0][0].item()
                impact = predictions[0][1].item()
                confidence = predictions[0][2].item()
            # Применение порогов
            if abs(sentiment) < self.sentiment_threshold:
                sentiment = 0.0
            if impact < self.impact_threshold:
                impact = 0.0
            if confidence < self.confidence_threshold:
                confidence = 0.0
            return {
                "sentiment": sentiment,
                "impact": impact,
                "confidence": confidence,
                "reasoning": f"ML model analyzed {len(news_data)} news items with confidence {confidence:.3f}",
                "features": features
            }
        except Exception as e:
            logger.error(f"Error analyzing news: {e}")
            return {
                "sentiment": 0.0,
                "impact": 0.0,
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}"
            }

    async def get_analysis_parameters(self) -> Dict[str, float]:
        """Получение текущих параметров анализа"""
        return {
            "sentiment_threshold": self.sentiment_threshold,
            "impact_threshold": self.impact_threshold,
            "confidence_threshold": self.confidence_threshold,
            "performance": self.performance_metric,
            "confidence": self.confidence_metric,
        }

    async def update_analysis_metrics(self, metrics: Dict[str, float]) -> None:
        """Обновление метрик анализа"""
        try:
            # Обновление метрик на основе производительности
            if "performance" in metrics:
                performance = metrics["performance"]
                if performance > 0.7:
                    self.performance_metric = 0.9 * self.performance_metric + 0.1 * performance
                elif performance < 0.3:
                    self.performance_metric = 0.9 * self.performance_metric + 0.1 * performance
        except Exception as e:
            logger.error(f"Error updating analysis metrics: {e}")
