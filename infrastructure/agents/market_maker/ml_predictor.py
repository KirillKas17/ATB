"""
ML-предиктор для маркет-мейкера
"""

import os
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
from loguru import logger


class MLPredictor:
    """ML-предиктор для анализа рынка и принятия решений."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {
            "input_dim": 15,
            "learning_rate": 0.001,
            "batch_size": 32,
        }
        self.spread_predictor = self._create_spread_predictor()
        self.liquidity_analyzer = self._create_liquidity_analyzer()
        self.spread_optimizer = torch.optim.Adam(
            self.spread_predictor.parameters(), lr=self.config["learning_rate"]
        )
        self.liquidity_optimizer = torch.optim.Adam(
            self.liquidity_analyzer.parameters(), lr=self.config["learning_rate"]
        )
        self.prediction_accuracy = 0.0
        self.successful_predictions = 0
        self.total_predictions = 0

    def _create_spread_predictor(self) -> nn.Module:
        """Создание модели для предсказания спредов."""
        return nn.Sequential(
            nn.Linear(self.config["input_dim"], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _create_liquidity_analyzer(self) -> nn.Module:
        """Создание модели для анализа ликвидности."""
        return nn.Sequential(
            nn.Linear(self.config["input_dim"], 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # support, resistance, neutral
        )

    def predict_spread(self, features: List[float]) -> float:
        """Предсказание спреда."""
        try:
            if len(features) != self.config["input_dim"]:
                logger.warning(
                    f"Expected {self.config['input_dim']} features, got {len(features)}"
                )
                return 0.0
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            with torch.no_grad():
                prediction = self.spread_predictor(features_tensor)
                return float(prediction.item())
        except Exception as e:
            logger.error(f"Error predicting spread: {str(e)}")
            return 0.0

    def predict_liquidity(self, features: List[float]) -> Dict[str, float]:
        """Предсказание ликвидности."""
        try:
            if len(features) != self.config["input_dim"]:
                logger.warning(
                    f"Expected {self.config['input_dim']} features, got {len(features)}"
                )
                return {"support": 0.0, "resistance": 0.0, "neutral": 0.0}
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            with torch.no_grad():
                prediction = self.liquidity_analyzer(features_tensor)
                probabilities = torch.softmax(prediction, dim=1)
                return {
                    "support": float(probabilities[0][0].item()),
                    "resistance": float(probabilities[0][1].item()),
                    "neutral": float(probabilities[0][2].item()),
                }
        except Exception as e:
            logger.error(f"Error predicting liquidity: {str(e)}")
            return {"support": 0.0, "resistance": 0.0, "neutral": 0.0}

    def train_spread_predictor(self, features: List[float], target: float) -> float:
        """Обучение модели предсказания спредов."""
        try:
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            target_tensor = torch.FloatTensor([target])
            self.spread_optimizer.zero_grad()
            prediction = self.spread_predictor(features_tensor)
            loss = nn.MSELoss()(prediction, target_tensor)
            loss.backward()
            self.spread_optimizer.step()
            return float(loss.item())
        except Exception as e:
            logger.error(f"Error training spread predictor: {str(e)}")
            return 0.0

    def train_liquidity_analyzer(
        self, features: List[float], targets: List[float]
    ) -> float:
        """Обучение модели анализа ликвидности."""
        try:
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            targets_tensor = torch.FloatTensor(targets)
            self.liquidity_optimizer.zero_grad()
            prediction = self.liquidity_analyzer(features_tensor)
            loss = nn.MSELoss()(prediction, targets_tensor)
            loss.backward()
            self.liquidity_optimizer.step()
            return float(loss.item())
        except Exception as e:
            logger.error(f"Error training liquidity analyzer: {str(e)}")
            return 0.0

    def extract_features(
        self, market_data: pd.DataFrame, order_book: Dict[str, Any]
    ) -> List[float]:
        """Извлечение признаков для ML-моделей."""
        try:
            features = []
            # Признаки из рыночных данных
            if not market_data.empty:
                # Ценовые признаки
                features.append(float(market_data["close"].pct_change().mean()))
                features.append(float(market_data["close"].pct_change().std()))
                features.append(
                    float(market_data["high"].max() / market_data["low"].min() - 1)
                )
                # Объемные признаки
                features.append(float(market_data["volume"].mean()))
                features.append(float(market_data["volume"].std()))
                features.append(float(market_data["volume"].pct_change().mean()))
                # Волатильность
                features.append(float(market_data["close"].rolling(window=20).std().mean()))
                # Технические индикаторы
                sma_20 = market_data["close"].rolling(window=20).mean()
                sma_last = sma_20.iloc[-1]
                if pd.notna(sma_last):
                    features.append(float(market_data["close"].iloc[-1] / sma_last - 1))
                else:
                    features.append(0.0)
                # RSI
                delta = market_data["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                rsi_last = rsi.iloc[-1]
                if pd.notna(rsi_last):
                    features.append(float(rsi_last))
                else:
                    features.append(50.0)
            # Признаки из ордербука
            if order_book and "bids" in order_book and "asks" in order_book:
                bids = order_book["bids"]
                asks = order_book["asks"]
                if bids and asks:
                    # Спред
                    best_bid = (
                        bids[0]["price"] if isinstance(bids[0], dict) else bids[0]
                    )
                    best_ask = (
                        asks[0]["price"] if isinstance(asks[0], dict) else asks[0]
                    )
                    spread = (best_ask - best_bid) / best_bid if best_bid > 0 else 0.0
                    features.append(spread)
                    # Дисбаланс
                    bid_volume = (
                        sum(order["size"] for order in bids[:5])
                        if isinstance(bids[0], dict)
                        else sum(bids[:5])
                    )
                    ask_volume = (
                        sum(order["size"] for order in asks[:5])
                        if isinstance(asks[0], dict)
                        else sum(asks[:5])
                    )
                    total_volume = bid_volume + ask_volume
                    imbalance = (
                        (bid_volume - ask_volume) / total_volume
                        if total_volume > 0
                        else 0.0
                    )
                    features.append(imbalance)
                    # Глубина рынка
                    features.append(float(len(bids)))
                    features.append(float(len(asks)))
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
            # Дополняем до нужного размера
            while len(features) < self.config["input_dim"]:
                features.append(0.0)
            return features[: self.config["input_dim"]]
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return [0.0] * self.config["input_dim"]
