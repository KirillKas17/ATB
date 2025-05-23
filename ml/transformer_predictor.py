import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import talib
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class PositionalEncoding(nn.Module):
    """
    Класс для добавления позиционного кодирования к входным данным.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return x


class PriceTransformer(nn.Module):
    """
    Transformer-модель для предсказания ценовых рядов (seq-to-one или seq-to-seq).
    """

    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.1,
        max_len: int = 256,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_size)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Создаёт маску для предотвращения утечки будущей информации."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, x, src_mask: Optional[torch.Tensor] = None):
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        if src_mask is not None:
            out = self.transformer_encoder(x, mask=src_mask)
        else:
            out = self.transformer_encoder(x)
        out = self.norm(out)
        # seq-to-one: берём последний выход
        return self.head(out[:, -1, :])


# --- Тренировка, валидация, инференс ---
def train_model(
    model: PriceTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 7,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    log_csv: Optional[str] = None,
) -> Dict:
    """
    Тренировка модели с early stopping и checkpointing.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            mask = model.generate_square_subsequent_mask(xb.size(1)).to(device)
            preds = model(xb, src_mask=mask)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                mask = model.generate_square_subsequent_mask(xb.size(1)).to(device)
                preds = model(xb, src_mask=mask)
                loss = criterion(preds, yb)
                val_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}")
        if log_csv:
            with open(log_csv, "a") as f:
                f.write(f"{epoch+1},{train_loss},{val_loss}\n")
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    return history


def evaluate_model(
    model: PriceTransformer, loader: DataLoader, device: str = "cpu"
) -> float:
    """
    Оценка модели на валидационном наборе.
    """
    model.eval()
    criterion = nn.MSELoss()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            mask = model.generate_square_subsequent_mask(xb.size(1)).to(device)
            preds = model(xb, src_mask=mask)
            loss = criterion(preds, yb)
            losses.append(loss.item())
    return np.mean(losses)


def predict(model: PriceTransformer, x: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    Предсказание для входной последовательности x (batch, seq_len, input_size)
    """
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x, dtype=torch.float32).to(device)
        mask = model.generate_square_subsequent_mask(xb.size(1)).to(device)
        preds = model(xb, src_mask=mask)
        return preds.cpu().numpy()


# --- Пример препроцессинга ---
def preprocess_data(
    df: pd.DataFrame,
    window: int,
    target_col: str = "close",
    feature_cols: Optional[list] = None,
    scaler=None,
):
    """
    Преобразует DataFrame в обучающие окна для Transformer.
    Возвращает X, y, scaler
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]
    X, y = [], []
    for i in range(len(df) - window):
        X.append(df[feature_cols].iloc[i : i + window].values)
        y.append(df[target_col].iloc[i + window])
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    # Нормализация
    if scaler is not None:
        X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y = scaler.fit_transform(y)
    return X, y


# --- Логирование прогнозов ---
def log_predictions(y_true: np.ndarray, y_pred: np.ndarray, out_csv: str):
    """
    Логирует прогнозы и реальные значения в CSV.
    """
    df = pd.DataFrame({"y_true": y_true.flatten(), "y_pred": y_pred.flatten()})
    df.to_csv(out_csv, index=False)


@dataclass
class TransformerConfig:
    """Конфигурация трансформера"""

    d_model: int = 64
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 256
    dropout: float = 0.1
    batch_size: int = 32
    learning_rate: float = 0.001
    max_epochs: int = 100
    early_stopping_patience: int = 10
    min_samples: int = 100
    max_samples: int = 1000
    sequence_length: int = 50
    prediction_horizon: int = 1
    feature_importance_threshold: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TransformerMetrics:
    """Метрики трансформера"""

    accuracy: float
    precision: float
    recall: float
    f1: float
    loss: float
    last_update: datetime
    samples_count: int
    epoch_count: int
    error_count: int
    feature_importance: Dict[str, float]
    hyperparameters: Dict[str, Any]


class TimeSeriesDataset(Dataset):
    """Датасет для временных рядов"""

    def __init__(
        self, data: pd.DataFrame, sequence_length: int, prediction_horizon: int
    ):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1

    def __getitem__(self, idx):
        # Получение последовательности
        sequence = self.data.iloc[idx : idx + self.sequence_length]

        # Получение целевой переменной
        target = self.data.iloc[
            idx + self.sequence_length + self.prediction_horizon - 1
        ]

        return torch.FloatTensor(sequence.values), torch.FloatTensor(target.values)


class TransformerModel(nn.Module):
    """Модель трансформера"""

    def __init__(self, config: TransformerConfig, input_size: int):
        super().__init__()

        self.config = config

        # Эмбеддинг входных данных
        self.embedding = nn.Linear(input_size, config.d_model)

        # Позиционное кодирование
        self.pos_encoder = PositionalEncoding(config.d_model, config.dropout)

        # Трансформер
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=config.num_layers
        )

        # Выходной слой
        self.decoder = nn.Linear(config.d_model, input_size)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # Эмбеддинг
        src = self.embedding(src)

        # Позиционное кодирование
        src = self.pos_encoder(src)

        # Трансформер
        output = self.transformer_encoder(src)

        # Декодирование
        output = self.decoder(output)

        return output


class TransformerPredictor:
    """Предсказание с помощью трансформера"""

    def __init__(self, config: Optional[TransformerConfig] = None):
        """Инициализация предсказателя"""
        self.config = config or TransformerConfig()
        self.transformer_dir = Path("transformer")
        self.transformer_dir.mkdir(parents=True, exist_ok=True)

        # Данные
        self.data_buffer = pd.DataFrame()
        self.metrics_history = []

        # Модели
        self.models = {}
        self.scalers = {}
        self.metrics = {}

        # Кэш
        self._prediction_cache = {}
        self._feature_cache = {}

        # Блокировки
        self._model_lock = Lock()
        self._metrics_lock = Lock()

        # Загрузка состояния
        self._load_state()

    def _load_state(self):
        """Загрузка состояния"""
        try:
            state_file = self.transformer_dir / "state.json"
            if state_file.exists():
                with open(state_file, "r") as f:
                    state = json.load(f)
                    self.metrics_history = state.get("metrics_history", [])
                    self.metrics = state.get("metrics", {})
        except Exception as e:
            logger.error(f"Ошибка загрузки состояния: {e}")

    def _save_state(self):
        """Сохранение состояния"""
        try:
            state_file = self.transformer_dir / "state.json"
            state = {"metrics_history": self.metrics_history, "metrics": self.metrics}
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния: {e}")

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Извлечение признаков"""
        try:
            features = pd.DataFrame()

            # Ценовые признаки
            features["returns"] = df["close"].pct_change()
            features["log_returns"] = np.log1p(features["returns"])
            features["volatility"] = features["returns"].rolling(20).std()

            # Технические индикаторы
            features["rsi"] = talib.RSI(df["close"])
            features["macd"], features["macd_signal"], _ = talib.MACD(df["close"])
            features["bb_upper"], features["bb_middle"], features["bb_lower"] = (
                talib.BBANDS(df["close"])
            )
            features["atr"] = talib.ATR(df["high"], df["low"], df["close"])
            features["adx"] = talib.ADX(df["high"], df["low"], df["close"])

            # Объемные признаки
            features["volume_ma"] = df["volume"].rolling(20).mean()
            features["volume_std"] = df["volume"].rolling(20).std()
            features["volume_ratio"] = df["volume"] / features["volume_ma"]

            # Моментум
            features["momentum"] = talib.MOM(df["close"], timeperiod=10)
            features["roc"] = talib.ROC(df["close"], timeperiod=10)

            # Волатильность
            features["high_low_ratio"] = df["high"] / df["low"]
            features["close_open_ratio"] = df["close"] / df["open"]

            # Тренд
            features["trend"] = talib.ADX(df["high"], df["low"], df["close"])
            features["trend_strength"] = abs(features["trend"])

            # Нормализация
            scaler = StandardScaler()
            features_scaled = pd.DataFrame(
                scaler.fit_transform(features.fillna(0)),
                columns=features.columns,
                index=features.index,
            )

            return features_scaled

        except Exception as e:
            logger.error(f"Ошибка извлечения признаков: {e}")
            return pd.DataFrame()

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Расчет метрик"""
        try:
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, average="weighted")),
                "recall": float(recall_score(y_true, y_pred, average="weighted")),
                "f1": float(f1_score(y_true, y_pred, average="weighted")),
            }
        except Exception as e:
            logger.error(f"Ошибка расчета метрик: {e}")
            return {}

    def _get_feature_importance(self, model_id: str) -> Dict[str, float]:
        """Получение важности признаков"""
        try:
            if model_id not in self.models:
                return {}

            model = self.models[model_id]
            if not hasattr(model, "feature_importances_"):
                return {}

            features = self._extract_features(self.data_buffer)
            importance = dict(zip(features.columns, model.feature_importances_))

            # Нормализация
            total = sum(importance.values())
            if total > 0:
                importance = {k: v / total for k, v in importance.items()}

            # Фильтрация по порогу
            importance = {
                k: v
                for k, v in importance.items()
                if v >= self.config.feature_importance_threshold
            }

            return importance

        except Exception as e:
            logger.error(f"Ошибка получения важности признаков: {e}")
            return {}

    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Оптимизация гиперпараметров"""
        try:

            def objective(trial):
                # Параметры для оптимизации
                params = {
                    "d_model": trial.suggest_int("d_model", 32, 128),
                    "nhead": trial.suggest_int("nhead", 4, 16),
                    "num_layers": trial.suggest_int("num_layers", 2, 8),
                    "dim_feedforward": trial.suggest_int("dim_feedforward", 128, 512),
                    "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                    "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.01),
                }

                # Создание модели
                model = TransformerModel(
                    TransformerConfig(**params), input_size=X.shape[1]
                )

                # Кросс-валидация
                scores = []
                for i in range(5):  # 5-fold cross-validation
                    # Разделение данных
                    split_size = len(X) // 5
                    start_idx = i * split_size
                    end_idx = (i + 1) * split_size

                    X_train = np.concatenate([X[:start_idx], X[end_idx:]])
                    y_train = np.concatenate([y[:start_idx], y[end_idx:]])
                    X_val = X[start_idx:end_idx]
                    y_val = y[start_idx:end_idx]

                    # Обучение и оценка
                    model.train()
                    optimizer = torch.optim.Adam(
                        model.parameters(), lr=params["learning_rate"]
                    )
                    criterion = nn.MSELoss()

                    for epoch in range(10):  # 10 эпох для каждой fold
                        optimizer.zero_grad()
                        output = model(torch.FloatTensor(X_train))
                        loss = criterion(output, torch.FloatTensor(y_train))
                        loss.backward()
                        optimizer.step()

                    # Оценка
                    model.eval()
                    with torch.no_grad():
                        val_output = model(torch.FloatTensor(X_val))
                        val_loss = criterion(val_output, torch.FloatTensor(y_val))
                        scores.append(
                            -val_loss.item()
                        )  # Отрицательный loss для максимизации

                return np.mean(scores)

            # Оптимизация
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=50)

            return study.best_params

        except Exception as e:
            logger.error(f"Ошибка оптимизации гиперпараметров: {e}")
            return {}

    def update(self, df: pd.DataFrame, model_id: str):
        """Обновление модели"""
        try:
            # Добавление данных в буфер
            self.data_buffer = pd.concat([self.data_buffer, df]).tail(
                self.config.max_samples
            )

            if len(self.data_buffer) < self.config.min_samples:
                return

            # Извлечение признаков
            features = self._extract_features(self.data_buffer)

            # Подготовка данных
            X = features.values
            y = (
                self.data_buffer["close"].shift(-1) > self.data_buffer["close"]
            ).values[:-1]
            X = X[
                :-1
            ]  # Убираем последнюю строку, так как для нее нет целевой переменной

            # Нормализация
            if model_id not in self.scalers:
                self.scalers[model_id] = StandardScaler()
            X_scaled = self.scalers[model_id].fit_transform(X)

            # Оптимизация гиперпараметров
            best_params = self._optimize_hyperparameters(X_scaled, y)

            # Создание модели
            model = TransformerModel(
                TransformerConfig(**best_params), input_size=X_scaled.shape[1]
            )

            # Обучение модели
            model.train()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=best_params["learning_rate"]
            )
            criterion = nn.MSELoss()

            # Ранний останов
            best_loss = float("inf")
            patience_counter = 0

            for epoch in range(self.config.max_epochs):
                optimizer.zero_grad()
                output = model(torch.FloatTensor(X_scaled))
                loss = criterion(output, torch.FloatTensor(y))
                loss.backward()
                optimizer.step()

                # Проверка раннего останова
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stopping_patience:
                    break

            self.models[model_id] = model

            # Расчет метрик
            model.eval()
            with torch.no_grad():
                y_pred = model(torch.FloatTensor(X_scaled)).numpy()
                metrics = self._calculate_metrics(y, y_pred)

            # Обновление истории метрик
            self.metrics_history.append(
                {"timestamp": datetime.now(), "model_id": model_id, **metrics}
            )

            # Ограничение истории
            self.metrics_history = self.metrics_history[-self.config.metrics_window :]

            # Получение важности признаков
            feature_importance = self._get_feature_importance(model_id)

            # Обновление метрик
            self.metrics[model_id] = TransformerMetrics(
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
                loss=float(loss.item()),
                last_update=datetime.now(),
                samples_count=len(self.data_buffer),
                epoch_count=epoch + 1,
                error_count=self.metrics.get(model_id, {}).get("error_count", 0),
                feature_importance=feature_importance,
                hyperparameters=best_params,
            ).__dict__

            # Сохранение состояния
            self._save_state()

            logger.info(f"Модель {model_id} обновлена. Метрики: {metrics}")

        except Exception as e:
            logger.error(f"Ошибка обновления модели: {e}")
            if model_id in self.metrics:
                self.metrics[model_id]["error_count"] += 1
            raise

    def predict(self, df: pd.DataFrame, model_id: str) -> Tuple[np.ndarray, float]:
        """Предсказание"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Модель {model_id} не найдена")

            # Извлечение признаков
            features = self._extract_features(df)

            # Нормализация
            X_scaled = self.scalers[model_id].transform(features.values)

            # Предсказание
            model = self.models[model_id]
            model.eval()
            with torch.no_grad():
                predictions = model(torch.FloatTensor(X_scaled)).numpy()

            # Расчет уверенности
            confidence = float(np.mean(np.abs(predictions)))

            return predictions, confidence

        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            raise

    def get_metrics(self, model_id: Optional[str] = None) -> Dict:
        """Получение метрик"""
        if model_id:
            return self.metrics.get(model_id, {})
        return self.metrics

    def get_history(self, model_id: Optional[str] = None) -> List[Dict]:
        """Получение истории"""
        if model_id:
            return [m for m in self.metrics_history if m["model_id"] == model_id]
        return self.metrics_history

    def reset(self):
        """Сброс состояния"""
        self.data_buffer = pd.DataFrame()
        self.metrics_history = []
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self._prediction_cache.clear()
        self._feature_cache.clear()
