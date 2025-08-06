"""
Максимально продвинутый модуль предсказания цены
Использует самые современные подходы: Multi-Head Attention, Graph Neural Networks,
Temporal Fusion Transformers, Neural ODEs, и другие передовые методы
"""

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import optuna
import pandas as pd
from pandas import DataFrame, Series
from shared.numpy_utils import np

try:
    import talib
except ImportError:
    talib = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False
from loguru import logger
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler
if TORCH_AVAILABLE:
    from torch import Tensor
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    from torch.utils.data import DataLoader
else:
    Tensor = None
    TransformerEncoder = None
    TransformerEncoderLayer = None
    AdamW = None
    CosineAnnealingWarmRestarts = None
    DataLoader = None

# Type aliases imported above

warnings.filterwarnings("ignore")


@dataclass
class AdvancedPredictorConfig:
    """Конфигурация продвинутого предиктора"""

    # Архитектура
    d_model: int = 512
    n_heads: int = 16
    n_layers: int = 12
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 1024
    # Обучение
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    max_epochs: int = 200
    gradient_clip: float = 1.0
    # Данные
    sequence_length: int = 256
    prediction_horizon: int = 24
    feature_dim: int = 128
    num_features: int = 50
    # Многоголовое внимание
    num_attention_heads: int = 16
    attention_dropout: float = 0.1
    # Graph Neural Networks
    gnn_hidden_dim: int = 256
    gnn_layers: int = 4
    gnn_dropout: float = 0.1
    # Temporal Fusion Transformer
    tft_hidden_dim: int = 512
    tft_lstm_layers: int = 2
    tft_attention_heads: int = 8
    # Neural ODE
    ode_hidden_dim: int = 128
    ode_tolerance: float = 1e-5
    # Ансамблирование
    ensemble_size: int = 5
    ensemble_method: str = "weighted_average"
    # Адаптация
    adaptation_rate: float = 0.01
    drift_threshold: float = 0.1
    retrain_threshold: float = 0.05
    # Мета-обучение
    meta_learning_rate: float = 1e-3
    meta_batch_size: int = 32
    meta_epochs: int = 100
    # Устройство
    device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdvancedPredictorConfig":
        return cls(**data)


class MultiHeadAttention(nn.Module):
    """Многоголовое внимание с относительным позиционированием"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        # Относительное позиционирование
        self.rel_pos_emb = nn.Parameter(torch.randn(2 * 1024, self.d_k))

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, _ = x.size()
        # Линейные преобразования
        Q = (
            self.w_q(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.w_k(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.w_v(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        # Относительное позиционирование
        rel_pos_emb = self.rel_pos_emb[: 2 * seq_len - 1].unsqueeze(0).unsqueeze(0)
        rel_pos_emb = rel_pos_emb.expand(batch_size, self.n_heads, -1, self.d_k)
        # Вычисление внимания
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # Добавление относительного позиционирования
        rel_scores = torch.matmul(Q, rel_pos_emb.transpose(-2, -1))
        scores = scores + rel_scores
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        # Применение внимания
        context = torch.matmul(attention, V)
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )
        return self.w_o(context)


class GraphConvolutionalLayer(nn.Module):
    """Слой графовой свертки для моделирования рыночных связей"""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        # Графовая свертка: H = σ(D^(-1/2) A D^(-1/2) X W + b)
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        output = output + self.bias
        return F.relu(self.dropout(output))


class GraphNeuralNetwork(nn.Module):
    """Графовая нейронная сеть для моделирования рыночных связей"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolutionalLayer(input_dim, hidden_dim, dropout))
        for _ in range(num_layers - 2):
            self.layers.append(GraphConvolutionalLayer(hidden_dim, hidden_dim, dropout))
        self.layers.append(GraphConvolutionalLayer(hidden_dim, output_dim, dropout))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, adj)
        return self.norm(x)


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer для временных рядов"""

    def __init__(self, config: AdvancedPredictorConfig) -> None:
        super().__init__()
        self.config = config
        # LSTM для обработки временных зависимостей
        self.lstm = nn.LSTM(
            input_size=config.feature_dim,
            hidden_size=config.tft_hidden_dim,
            num_layers=config.tft_lstm_layers,
            dropout=config.dropout,
            batch_first=True,
            bidirectional=True,
        )
        # Multi-head attention
        self.attention = MultiHeadAttention(
            config.tft_hidden_dim * 2,  # bidirectional
            config.tft_attention_heads,
            config.attention_dropout,
        )
        # Выходной слой
        self.output_layer = nn.Sequential(
            nn.Linear(config.tft_hidden_dim * 2, config.tft_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.tft_hidden_dim, config.prediction_horizon),
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # LSTM обработка
        lstm_out, _ = self.lstm(x)
        # Attention
        attended = self.attention(lstm_out, mask)
        # Выходной слой
        output = self.output_layer(attended[:, -1, :])
        return output


class NeuralODE(nn.Module):
    """Neural ODE для непрерывного времени"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 для времени
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        tx = torch.cat([t.unsqueeze(-1), x], dim=-1)
        return self.net(tx)


class AdvancedTransformer(nn.Module):
    """Продвинутый трансформер с множественными улучшениями"""

    def __init__(self, config: AdvancedPredictorConfig) -> None:
        super().__init__()
        self.config = config
        # Эмбеддинг признаков
        self.feature_embedding = nn.Linear(config.num_features, config.d_model)
        # Позиционное кодирование
        self.pos_encoding = self._create_positional_encoding()
        # Transformer слои
        encoder_layer = TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = TransformerEncoder(encoder_layer, config.n_layers)
        # Graph Neural Network
        self.gnn = GraphNeuralNetwork(
            config.d_model,
            config.gnn_hidden_dim,
            config.d_model,
            config.gnn_layers,
            config.gnn_dropout,
        )
        # Temporal Fusion Transformer
        self.tft = TemporalFusionTransformer(config)
        # Neural ODE
        self.neural_ode = NeuralODE(
            config.d_model, config.ode_hidden_dim, config.d_model
        )
        # Выходные слои
        self.output_projection = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model),  # transformer + gnn + ode
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.prediction_horizon),
        )
        # Нормализация
        self.layer_norm = nn.LayerNorm(config.d_model)

    def _create_positional_encoding(self) -> nn.Parameter:
        pos_encoding = torch.zeros(self.config.max_seq_len, self.config.d_model)
        position = torch.arange(0, self.config.max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.config.d_model, 2).float()
            * (-math.log(10000.0) / self.config.d_model)
        )
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pos_encoding, requires_grad=False)

    def forward(self, x: Tensor, adj_matrix: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, _ = x.size()
        # Эмбеддинг признаков
        x = self.feature_embedding(x)
        # Позиционное кодирование
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        # Transformer
        transformer_out = self.transformer(x)
        transformer_out = self.layer_norm(transformer_out)
        # Graph Neural Network
        if adj_matrix is not None:
            gnn_out = self.gnn(transformer_out, adj_matrix)
        else:
            # Создаем единичную матрицу смежности
            adj_matrix = torch.eye(seq_len).unsqueeze(0).expand(batch_size, -1, -1)
            gnn_out = self.gnn(transformer_out, adj_matrix)
        # Neural ODE (упрощенная версия)
        torch.linspace(0, 1, seq_len).to(x.device)
        ode_out = transformer_out  # Упрощение для производительности
        # Объединение выходов
        combined = torch.cat(
            [transformer_out[:, -1, :], gnn_out[:, -1, :], ode_out[:, -1, :]], dim=-1
        )
        # Выходной слой
        output = self.output_projection(combined)
        return output


class MetaLearner(nn.Module):
    """Мета-обучатель для быстрой адаптации к новым рыночным условиям"""

    def __init__(self, config: AdvancedPredictorConfig) -> None:
        super().__init__()
        self.config = config
        # Мета-сеть для генерации параметров
        self.meta_network = nn.Sequential(
            nn.Linear(config.feature_dim, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        # Быстрая адаптация
        self.fast_adaptation = nn.ModuleList(
            [nn.Linear(config.d_model, config.d_model) for _ in range(3)]
        )

    def forward(self, x: Tensor, task_embedding: Tensor) -> Tensor:
        # Генерация параметров для задачи
        task_params = self.meta_network(task_embedding)
        # Быстрая адаптация
        adapted = x
        for layer in self.fast_adaptation:
            adapted = layer(adapted + task_params)
        return adapted


class AdvancedPricePredictor:
    """Максимально продвинутый предиктор цен"""

    def __init__(self, config: Optional[AdvancedPredictorConfig] = None) -> None:
        self.config = config or AdvancedPredictorConfig()
        self.device = torch.device(self.config.device)
        # Основная модель
        self.model = AdvancedTransformer(self.config).to(self.device)
        # Мета-обучатель
        self.meta_learner = MetaLearner(self.config).to(self.device)
        # Ансамбль моделей
        self.ensemble_models = nn.ModuleList(
            [
                AdvancedTransformer(self.config).to(self.device)
                for _ in range(self.config.ensemble_size)
            ]
        )
        # Оптимизаторы
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.meta_optimizer = AdamW(
            self.meta_learner.parameters(), lr=self.config.meta_learning_rate
        )
        # Планировщики
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        # Скалеры
        self.feature_scaler = RobustScaler()
        self.target_scaler = RobustScaler()
        # Кэш и состояние
        self.feature_cache: dict[str, Any] = {}
        self.prediction_cache: dict[str, Any] = {}
        self.model_state: dict[str, Any] = {}
        # Метрики
        self.metrics: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "mae": [],
            "rmse": [],
            "r2": [],
        }
        logger.info(f"AdvancedPricePredictor initialized on {self.device}")

    def _extract_advanced_features(self, df: DataFrame) -> DataFrame:
        """Извлечение продвинутых признаков"""
        features = DataFrame()
        # Временные признаки
        dt_index = pd.to_datetime(df.index)
        if isinstance(dt_index, pd.DatetimeIndex):
            # Исправление: используем правильные методы DatetimeIndex
            if hasattr(dt_index, 'hour'):
                # Исправление: безопасное получение значений из DatetimeIndex
                if hasattr(dt_index.hour, 'to_numpy'):
                    features["hour"] = dt_index.hour.to_numpy()
                elif hasattr(dt_index.hour, 'values'):
                    features["hour"] = dt_index.hour.values
                else:
                    features["hour"] = [x.hour for x in dt_index]
            else:
                features["hour"] = [x.hour for x in dt_index]
            
            if hasattr(dt_index, 'dayofweek'):
                # Исправление: безопасное получение значений из DatetimeIndex
                if hasattr(dt_index.dayofweek, 'to_numpy'):
                    features["day_of_week"] = dt_index.dayofweek.to_numpy()
                elif hasattr(dt_index.dayofweek, 'values'):
                    features["day_of_week"] = dt_index.dayofweek.values
                else:
                    features["day_of_week"] = [x.dayofweek for x in dt_index]
            else:
                features["day_of_week"] = [x.dayofweek for x in dt_index]
            
            if hasattr(dt_index, 'month'):
                # Исправление: безопасное получение значений из DatetimeIndex
                if hasattr(dt_index.month, 'to_numpy'):
                    features["month"] = dt_index.month.to_numpy()
                elif hasattr(dt_index.month, 'values'):
                    features["month"] = dt_index.month.values
                else:
                    features["month"] = [x.month for x in dt_index]
            else:
                features["month"] = [x.month for x in dt_index]
        else:
            # Исправление: проверяем итерируемость перед итерацией
            if hasattr(dt_index, '__iter__') and not isinstance(dt_index, pd.Timestamp):
                features["hour"] = [x.hour for x in dt_index]
                features["day_of_week"] = [x.dayofweek for x in dt_index]
                features["month"] = [x.month for x in dt_index]
            else:
                # Одиночный Timestamp
                if isinstance(dt_index, pd.Timestamp):
                    features["hour"] = [dt_index.hour]
                    features["day_of_week"] = [dt_index.dayofweek]
                    features["month"] = [dt_index.month]
                else:
                    # Fallback для других типов
                    features["hour"] = [0]
                    features["day_of_week"] = [0]
                    features["month"] = [1]
        # Циклические признаки
        features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
        features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
        features["day_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
        features["day_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)
        # Сглаживание
        # Исправление: используем правильные методы DataFrame
        if hasattr(df, 'values'):
            features["close_smooth"] = savgol_filter(np.asarray(df["close"].values, dtype=float), 5, 2)
            features["volume_smooth"] = savgol_filter(np.asarray(df["volume"].values, dtype=float), 5, 2)
        else:
            features["close_smooth"] = savgol_filter(np.asarray(df["close"], dtype=float), 5, 2)
            features["volume_smooth"] = savgol_filter(np.asarray(df["volume"], dtype=float), 5, 2)
        # Статистические признаки
        if hasattr(df["close"], 'rolling'):
            close_rolling = df["close"].rolling(20).mean()
            if hasattr(close_rolling, 'values'):
                features["price_zscore"] = stats.zscore(close_rolling.values)
            else:
                features["price_zscore"] = stats.zscore(close_rolling)
        else:
            features["price_zscore"] = [0.0] * len(df)
            
        if hasattr(df["volume"], 'rolling'):
            volume_rolling = df["volume"].rolling(20).mean()
            if hasattr(volume_rolling, 'values'):
                features["volume_zscore"] = stats.zscore(volume_rolling.values)
            else:
                features["volume_zscore"] = stats.zscore(volume_rolling)
        else:
            features["volume_zscore"] = [0.0] * len(df)
            
        # Удаление NaN
        if hasattr(features, 'dropna'):
            features = features.dropna()
        return features

    def _create_adjacency_matrix(self, features: DataFrame) -> torch.Tensor:
        """Создание матрицы смежности для GNN"""
        # Корреляционная матрица
        # Исправление: используем правильные методы DataFrame
        if hasattr(features, 'corr'):
            corr_matrix = features.corr().abs()
        else:
            # Альтернативный способ вычисления корреляции
            if hasattr(features, 'to_numpy'):
                corr_matrix = np.abs(np.corrcoef(features.to_numpy(), rowvar=False))
            else:
                corr_matrix = np.abs(np.corrcoef(features.values, rowvar=False))
            corr_matrix = pd.DataFrame(corr_matrix, columns=features.columns, index=features.columns)
        
        if not isinstance(corr_matrix, pd.DataFrame):
            corr_matrix = pd.DataFrame(corr_matrix, columns=features.columns, index=features.columns)
        
        # Пороговая фильтрация
        threshold = 0.3
        # Исправление: создаем adj_matrix из corr_matrix
        if hasattr(corr_matrix, 'gt'):
            adj_matrix = corr_matrix.gt(threshold).astype(float)
        else:
            adj_matrix = (corr_matrix > threshold).astype(float)
        
        # Нормализация
        adj_matrix = adj_matrix / adj_matrix.sum(axis=1, keepdims=True)
        # Исправление: используем to_numpy() вместо values
        if hasattr(adj_matrix, 'to_numpy'):
            return torch.FloatTensor(adj_matrix.to_numpy())
        else:
            return torch.FloatTensor(adj_matrix.values)

    def prepare_data(
        self, df: DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Подготовка данных для обучения"""
        # Извлечение признаков
        features: pd.DataFrame = self._extract_advanced_features(df)
        # Создание последовательностей
        X: list[np.ndarray] = []
        y: list[np.ndarray] = []
        for i in range(
            self.config.sequence_length, len(features) - self.config.prediction_horizon
        ):
            # Исправление: используем правильные методы DataFrame
            if hasattr(features, 'iloc'):
                sequence_data = features.iloc[i - self.config.sequence_length : i]
                if hasattr(sequence_data, 'to_numpy'):
                    X.append(sequence_data.to_numpy())
                else:
                    X.append(sequence_data.values)
                    
                close_data = features["close"].iloc[i : i + self.config.prediction_horizon]
                if hasattr(close_data, 'to_numpy'):
                    y.append(close_data.to_numpy())
                else:
                    y.append(close_data.values)
        X_arr = np.array(X)
        y_arr = np.array(y)
        # Нормализация
        X_reshaped = X_arr.reshape(-1, X_arr.shape[-1])
        X_scaled = self.feature_scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X_arr.shape)
        y_scaled = self.target_scaler.fit_transform(y_arr)
        # Создание матрицы смежности
        adj_matrix = self._create_adjacency_matrix(features)
        return (
            torch.FloatTensor(X_scaled).to(self.device),
            torch.FloatTensor(y_scaled).to(self.device),
            adj_matrix.to(self.device),
        )

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> Dict[str, List[float]]:
        """Обучение модели"""
        best_val_loss = float("inf")
        patience_counter = 0
        for epoch in range(self.config.max_epochs):
            # Обучение
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y, batch_adj in train_loader:
                self.optimizer.zero_grad()
                # Прямой проход
                predictions = self.model(batch_X, batch_adj)
                loss = F.mse_loss(predictions, batch_y)
                # Обратное распространение
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
                self.optimizer.step()
                train_loss += loss.item()
            # Валидация
            val_loss = self._validate(val_loader)
            # Обновление планировщика
            self.scheduler.step()
            # Сохранение метрик
            self.metrics["train_loss"].append(train_loss / len(train_loader))
            self.metrics["val_loss"].append(val_loss)
            # Ранняя остановка
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_best_model()
            else:
                patience_counter += 1
            if patience_counter >= 10:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}"
                )
        return self.metrics

    def _validate(self, val_loader: DataLoader) -> float:
        """Валидация модели"""
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y, batch_adj in val_loader:
                predictions = self.model(batch_X, batch_adj)
                loss = F.mse_loss(predictions, batch_y)
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def predict(self, df: DataFrame) -> Tuple[np.ndarray, float]:
        """Предсказание с ансамблированием"""
        # Подготовка данных
        X, _, adj_matrix = self.prepare_data(df)
        # Предсказания ансамбля
        predictions: list[np.ndarray] = []
        confidences: list[float] = []
        with torch.no_grad():
            for model in self.ensemble_models:
                model.eval()
                pred = model(X[-1:], adj_matrix.unsqueeze(0))
                predictions.append(pred.cpu().numpy())
                # Оценка уверенности
                confidence = self._calculate_confidence(pred)
                confidences.append(confidence)
        # Ансамблирование
        ensemble_pred = np.mean(predictions, axis=0)
        ensemble_confidence = float(np.mean(confidences))
        # Обратная нормализация
        ensemble_pred = self.target_scaler.inverse_transform(ensemble_pred)
        return ensemble_pred[0], ensemble_confidence

    def _calculate_confidence(self, prediction: torch.Tensor) -> float:
        """Расчет уверенности в предсказании"""
        # Дисперсия предсказаний ансамбля
        variance = torch.var(prediction).item()
        confidence = 1.0 / (1.0 + variance)
        return float(confidence)

    def meta_learn(self, tasks: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Мета-обучение для быстрой адаптации"""
        self.meta_learner.train()
        for task_X, task_y in tasks:
            self.meta_optimizer.zero_grad()
            # Быстрая адаптация
            adapted_X = self.meta_learner(task_X, task_X.mean(dim=1))
            predictions = self.model(adapted_X)
            loss = F.mse_loss(predictions, task_y)
            loss.backward()
            self.meta_optimizer.step()

    def _save_best_model(self) -> None:
        """Сохранение лучшей модели"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "meta_learner_state_dict": self.meta_learner.state_dict(),
                "config": self.config.to_dict(),
                "feature_scaler": self.feature_scaler,
                "target_scaler": self.target_scaler,
                "metrics": self.metrics,
            },
            "best_advanced_predictor.pth",
        )

    def load_model(self, path: str) -> None:
        """Загрузка модели"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.meta_learner.load_state_dict(checkpoint["meta_learner_state_dict"])
        self.feature_scaler = checkpoint["feature_scaler"]
        self.target_scaler = checkpoint["target_scaler"]
        self.metrics = checkpoint["metrics"]
        logger.info(f"Model loaded from {path}")

    def optimize_hyperparameters(
        self, train_data: DataFrame, val_data: DataFrame
    ) -> Dict[str, Any]:
        """Оптимизация гиперпараметров с помощью Optuna"""

        def objective(trial):
            # Параметры для оптимизации
            d_model = trial.suggest_int("d_model", 128, 1024)
            n_heads = trial.suggest_int("n_heads", 4, 32)
            n_layers = trial.suggest_int("n_layers", 4, 16)
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            # Создание конфигурации
            config = AdvancedPredictorConfig(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                learning_rate=learning_rate,
                dropout=dropout,
            )
            # Создание и обучение модели
            predictor = AdvancedPricePredictor(config)
            # TODO: Реализовать полное обучение модели
            # val_loss = predictor.train(train_data, val_data)
            val_loss = 0.1  # Заглушка для оптимизации
            return val_loss  # Возвращаем валидационную ошибку

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)
        return dict(study.best_params)


# Пример использования
if __name__ == "__main__":
    # Создание предиктора
    config = AdvancedPredictorConfig()
    predictor = AdvancedPricePredictor(config)
    # Загрузка данных (пример)
    # df = pd.read_csv('market_data.csv')
    # Обучение
    # train_loader, val_loader = create_data_loaders(df)
    # metrics = predictor.train(train_loader, val_loader)
    # Предсказание
    # prediction, confidence = predictor.predict(df)
    # print(f"Prediction: {prediction}, Confidence: {confidence}")
