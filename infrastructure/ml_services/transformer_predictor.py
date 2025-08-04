import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import optuna
import pandas as pd
from shared.numpy_utils import np

# # import talib  # Временно закомментировано из-за проблем с установкой на Windows  # Временно закомментировано из-за проблем с установкой на Windows
try:
    import talib
except ImportError:
    talib = None
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False
from deap import base, creator, tools
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
if TORCH_AVAILABLE:
    from torch.utils.data import DataLoader, Dataset
else:
    DataLoader = None
    Dataset = None

from shared.models.ml_metrics import TransformerMetrics

# Type aliases
DataFrame = pd.DataFrame
Series = pd.Series


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return x


class AdaptiveTransformer(nn.Module):
    """
    Адаптивный трансформер с онлайн-обучением
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
        adaptation_rate: float = 0.01,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_len)
        # Адаптивные слои
        self.adaptive_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_size)
        self.adaptation_rate = adaptation_rate
        # Онлайн-обучение
        self.online_optimizer = None
        self.momentum_buffer: Dict[torch.Tensor, torch.Tensor] = {}

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Генерация маски для последующих токенов"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        for layer in self.adaptive_layers:
            if src_mask is not None:
                x = layer(x, src_mask=src_mask)
            else:
                x = layer(x)
        x = self.norm(x)
        return self.head(x[:, -1, :])

    def online_update(
        self, x: torch.Tensor, target: torch.Tensor, learning_rate: Optional[float] = None
    ) -> None:
        """Онлайн-обновление модели"""
        if learning_rate is None:
            learning_rate = self.adaptation_rate
        self.train()
        self.zero_grad()
        output = self.forward(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        # Адаптивное обновление весов
        with torch.no_grad():
            for param in self.parameters():
                if param.grad is not None:
                    # Исправление: используем правильные методы Tensor
                    param.data.add_(param.grad.data, alpha=-learning_rate)
                    # Моментум
                    if param in self.momentum_buffer:
                        self.momentum_buffer[param] = (
                            0.9 * self.momentum_buffer[param] + 0.1 * param.grad.data
                        )
                    else:
                        self.momentum_buffer[param] = param.grad.data.clone()
                    param.data -= 0.01 * self.momentum_buffer[param]


class EvolutionaryTransformer:
    """Эволюционный трансформер с автоматической оптимизацией архитектуры"""

    def __init__(self, input_size: int, output_size: int, population_size: int = 20):
        self.input_size = input_size
        self.output_size = output_size
        self.population_size = population_size
        self.population: List[Any] = []
        self.generation = 0
        self.best_model = None
        self.best_fitness = float("-inf")
        # Настройка генетического алгоритма
        self.setup_genetic_algorithm()

    def setup_genetic_algorithm(self) -> None:
        """Настройка генетического алгоритма"""
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        # Гены для архитектуры
        self.toolbox.register("hidden_dim", random.randint, 32, 256)
        self.toolbox.register("num_heads", random.randint, 4, 16)
        self.toolbox.register("num_layers", random.randint, 2, 8)
        self.toolbox.register("dropout", random.uniform, 0.1, 0.5)
        self.toolbox.register("learning_rate", random.uniform, 0.0001, 0.01)
        # Создание индивидуума
        self.toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            (
                self.toolbox.hidden_dim,
                self.toolbox.num_heads,
                self.toolbox.num_layers,
                self.toolbox.dropout,
                self.toolbox.learning_rate,
            ),
            n=1,
        )
        # Создание популяции
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        # Генетические операторы
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate_individual(self, individual: Any) -> Tuple[float, ...]:
        """Оценка индивидуума"""
        try:
            hidden_dim, num_heads, num_layers, dropout, lr = individual
            # Создание модели
            model = AdaptiveTransformer(
                input_size=self.input_size,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                output_size=self.output_size,
                dropout=dropout,
                adaptation_rate=lr,
            )
            # Простая оценка на основе размера модели
            total_params = sum(p.numel() for p in model.parameters())
            complexity_penalty = -0.001 * total_params  # Штраф за сложность
            # Базовый фитнес
            fitness = 1.0 + complexity_penalty
            return (fitness,)
        except Exception as e:
            logger.error(f"Error evaluating individual: {e}")
            return (float("-inf"),)

    def evolve(self, generations: int = 10) -> None:
        """Эволюция популяции"""
        population = self.toolbox.population(n=self.population_size)
        for gen in range(generations):
            # Оценка популяции
            fitnesses = list(map(self.toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            # Отбор лучших
            offspring = list(map(
                self.toolbox.clone, self.toolbox.select(population, len(population))
            ))
            # Скрещивание
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            # Мутация
            for mutant in offspring:
                if random.random() < 0.3:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            # Оценка потомков
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            # Замена популяции
            population[:] = offspring
            # Обновление лучшего
            best_ind = tools.selBest(population, 1)[0]
            if best_ind.fitness.values[0] > self.best_fitness:
                self.best_fitness = best_ind.fitness.values[0]
                self.best_model = self.create_model_from_individual(best_ind)
            self.generation += 1
            logger.info(f"Generation {gen+1}: Best fitness = {self.best_fitness}")

    def create_model_from_individual(self, individual: Any) -> AdaptiveTransformer:
        """Создание модели из индивидуума"""
        hidden_dim, num_heads, num_layers, dropout, lr = individual
        return AdaptiveTransformer(
            input_size=self.input_size,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            output_size=self.output_size,
            dropout=dropout,
            adaptation_rate=lr,
        )


class OnlineAdaptiveTransformer:
    """Онлайн-адаптивный трансформер с непрерывным обучением"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model: Optional[AdaptiveTransformer] = None
        self.scaler: StandardScaler = StandardScaler()
        self.data_buffer: List[Tuple[np.ndarray, np.ndarray]] = []
        self.performance_history: List[float] = []
        self.adaptation_threshold = 0.1
        self.last_adaptation = 0

    def initialize_model(self, input_size: int, output_size: int) -> None:
        """Инициализация модели"""
        self.model = AdaptiveTransformer(
            input_size=input_size,
            hidden_dim=self.config.get("hidden_dim", 64),
            num_heads=self.config.get("num_heads", 8),
            num_layers=self.config.get("num_layers", 4),
            output_size=output_size,
            dropout=self.config.get("dropout", 0.1),
            adaptation_rate=self.config.get("adaptation_rate", 0.001),
        )

    def update(self, new_data: np.ndarray, target: np.ndarray) -> None:
        """Онлайн-обновление модели"""
        if self.model is None:
            self.initialize_model(new_data.shape[1], target.shape[1])
        # Добавление в буфер
        self.data_buffer.append((new_data, target))
        # Ограничение размера буфера
        if len(self.data_buffer) > self.config.get("buffer_size", 1000):
            self.data_buffer.pop(0)
        # Онлайн-обучение
        if len(self.data_buffer) >= self.config.get("min_batch_size", 10):
            self._online_learning_step()
        # Проверка необходимости адаптации
        if self._should_adapt():
            self._adapt_model()

    def _online_learning_step(self) -> None:
        """Шаг онлайн-обучения"""
        batch_size = min(self.config.get("batch_size", 32), len(self.data_buffer))
        batch_indices = np.random.choice(
            len(self.data_buffer), batch_size, replace=False
        )
        batch_data = []
        batch_targets = []
        for idx in batch_indices:
            data, target = self.data_buffer[idx]
            batch_data.append(data)
            batch_targets.append(target)
        batch_data = torch.tensor(np.array(batch_data), dtype=torch.float32)
        batch_targets = torch.tensor(np.array(batch_targets), dtype=torch.float32)
        # Онлайн-обновление
        if self.model is not None:
            self.model.online_update(batch_data, batch_targets)

    def _should_adapt(self) -> bool:
        """Проверка необходимости адаптации"""
        if len(self.performance_history) < 20:
            return False
        recent_performance = np.mean(self.performance_history[-10:])
        old_performance = np.mean(self.performance_history[-20:-10])
        return bool((old_performance - recent_performance) > self.adaptation_threshold)

    def _adapt_model(self) -> None:
        """Адаптация модели"""
        logger.info("Adapting model architecture...")
        # Создание эволюционного оптимизатора
        evo_optimizer = EvolutionaryTransformer(
            input_size=self.model.input_proj.in_features if self.model else 64,
            output_size=self.model.head.out_features if self.model else 1,
        )
        # Эволюция архитектуры
        evo_optimizer.evolve(generations=5)
        if evo_optimizer.best_model is not None:
            # Копирование весов
            if self.model is not None:
                self._transfer_weights(self.model, evo_optimizer.best_model)
            self.model = evo_optimizer.best_model
        self.last_adaptation = len(self.performance_history)

    def _transfer_weights(self, old_model: nn.Module, new_model: nn.Module) -> None:
        """Перенос весов между моделями"""
        old_state = old_model.state_dict()
        new_state = new_model.state_dict()
        for key in new_state:
            if key in old_state and old_state[key].shape == new_state[key].shape:
                new_state[key] = old_state[key]
        new_model.load_state_dict(new_state)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Предсказание"""
        if self.model is None:
            return np.zeros((data.shape[0], 1))
        self.model.eval()
        with torch.no_grad():
            data_tensor = torch.tensor(data, dtype=torch.float32)
            predictions = self.model(data_tensor)
            return predictions.numpy()  # type: ignore[no-any-return]


# --- Тренировка, валидация, инференс ---
def train_model(
    model: AdaptiveTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 7,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    log_csv: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Обучение модели с ранним остановом.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    patience_counter = 0
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            mask = model.generate_square_subsequent_mask(xb.size(1)).to(device)
            # Исправление: используем правильные методы Tensor
            preds = model(xb, src_mask=mask)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                mask = model.generate_square_subsequent_mask(xb.size(1)).to(device)
                # Исправление: используем правильные методы Tensor
                preds = model(xb, src_mask=mask)
                loss = criterion(preds, yb)
                val_losses.append(float(loss.item()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
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
    model: AdaptiveTransformer, loader: DataLoader, device: str = "cpu"
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
            # Исправление: используем правильные методы Tensor
            preds = model(xb, src_mask=mask)
            loss = criterion(preds, yb)
            losses.append(float(loss.item()))
    return float(np.mean(losses))


def predict(
    model: AdaptiveTransformer, x: np.ndarray, device: str = "cpu"
) -> np.ndarray:
    """
    Предсказание для входной последовательности x (batch, seq_len, input_size)
    """
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x, dtype=torch.float32).to(device)
        mask = model.generate_square_subsequent_mask(xb.size(1)).to(device)
        # Исправление: используем правильные методы Tensor
        preds = model(xb, src_mask=mask)
        return preds.cpu().numpy()  # type: ignore[no-any-return]


# --- Пример препроцессинга ---
def preprocess_data(
    df: DataFrame,
    window: int,
    target_col: str = "close",
    feature_cols: Optional[List[str]] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Преобразует DataFrame в обучающие окна для Transformer.
    Возвращает X, y, scaler
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]
    X, y = [], []
    for i in range(len(df) - window):
        # Исправление: используем to_numpy() вместо values
        if hasattr(df[feature_cols].iloc[i : i + window], 'to_numpy'):
            X.append(df[feature_cols].iloc[i : i + window].to_numpy())
        else:
            X.append(df[feature_cols].iloc[i : i + window].values)
        y.append(df[target_col].iloc[i + window])
    # Исправление: используем правильные типы для списков
    X_array = np.array(X)
    y_array = np.array(y).reshape(-1, 1)
    # Нормализация
    if scaler is not None:
        if hasattr(X_array, 'reshape'):
            X_array = scaler.fit_transform(X_array.reshape(-1, X_array.shape[-1])).reshape(X_array.shape)
        else:
            X_array = scaler.fit_transform(X_array)
        y_array = scaler.fit_transform(y_array)
    return X_array, y_array


# --- Логирование прогнозов ---
def log_predictions(y_true: np.ndarray, y_pred: np.ndarray, out_csv: str) -> None:
    """
    Логирует прогнозы и реальные значения в CSV.
    """
    # Исправление: используем pandas DataFrame методы
    df = pd.DataFrame({"y_true": y_true.flatten(), "y_pred": y_pred.flatten()})
    if hasattr(df, 'to_csv'):
        df.to_csv(out_csv, index=False)
    else:
        # Альтернативный способ сохранения
        if hasattr(df, 'to_numpy'):
            df_data = df.to_numpy()
        elif hasattr(df, 'values'):
            df_data = df.values
        else:
            df_data = np.array(df)
        np.savetxt(out_csv, df_data, delimiter=',', header='y_true,y_pred', comments='')


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
    metrics_window: int = 100


class TimeSeriesDataset(Dataset):
    """Датасет для временных рядов"""

    def __init__(
        self, data: DataFrame, sequence_length: int, prediction_horizon: int
    ):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

    def __len__(self) -> int:
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Получение последовательности
        sequence = self.data.iloc[idx : idx + self.sequence_length]
        # Получение целевой переменной
        target = self.data.iloc[
            idx + self.sequence_length + self.prediction_horizon - 1
        ]
        # Исправление: используем to_numpy() вместо values
        if hasattr(sequence, 'to_numpy'):
            sequence_data = sequence.to_numpy()
        else:
            sequence_data = sequence.values
        if hasattr(target, 'to_numpy'):
            target_data = target.to_numpy()
        else:
            target_data = target.values
        return torch.FloatTensor(sequence_data), torch.FloatTensor(target_data)


class TransformerModel(nn.Module):
    """Модель трансформера"""

    def __init__(self, config: TransformerConfig, input_size: int):
        super().__init__()
        self.config = config
        # Эмбеддинг входных данных
        self.embedding = nn.Linear(input_size, config.d_model)
        # Позиционное кодирование
        self.pos_encoder = PositionalEncoding(config.d_model, int(config.dropout) if isinstance(config.dropout, float) else config.d_model)
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
        output = self.decoder(output)  # type: ignore[no-any-return]
        return output


class TransformerPredictor:
    """Предсказание с помощью трансформера"""

    def __init__(self, config: Optional[TransformerConfig] = None):
        """Инициализация предсказателя"""
        self.config = config or TransformerConfig()
        self.transformer_dir = Path("transformer")
        self.transformer_dir.mkdir(parents=True, exist_ok=True)
        # Данные
        self.data_buffer: DataFrame = DataFrame()
        self.metrics_history: List[Dict[str, Any]] = []
        # Модели
        self.models: Dict[str, TransformerModel] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        # Кэш
        self._prediction_cache: Dict[str, Any] = {}
        self._feature_cache: Dict[str, Any] = {}
        # Блокировки
        self._model_lock: Lock = Lock()
        self._metrics_lock: Lock = Lock()
        # Загрузка состояния
        self._load_state()

    def _load_state(self) -> None:
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

    def _save_state(self) -> None:
        """Сохранение состояния"""
        try:
            state_file = self.transformer_dir / "state.json"
            state = {"metrics_history": self.metrics_history, "metrics": self.metrics}
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния: {e}")

    def _extract_features(self, df: DataFrame) -> DataFrame:
        """Извлечение признаков"""
        try:
            features = DataFrame()
            # Ценовые признаки
            features["returns"] = df["close"].pct_change()
            features["log_returns"] = np.log1p(features["returns"])
            features["volatility"] = features["returns"].rolling(20).std()
            # Технические индикаторы
            if talib is not None:
                features["rsi"] = talib.RSI(df["close"].values)
                macd, macd_signal, _ = talib.MACD(df["close"].values)
                features["macd"] = macd
                features["macd_signal"] = macd_signal
                bb_upper, bb_middle, bb_lower = talib.BBANDS(df["close"].values)
                features["bb_upper"] = bb_upper
                features["bb_middle"] = bb_middle
                features["bb_lower"] = bb_lower
                features["atr"] = talib.ATR(df["high"].values, df["low"].values, df["close"].values)
                features["adx"] = talib.ADX(df["high"].values, df["low"].values, df["close"].values)
                features["momentum"] = talib.MOM(df["close"].values, timeperiod=10)
                features["roc"] = talib.ROC(df["close"].values, timeperiod=10)
                features["trend"] = talib.ADX(df["high"].values, df["low"].values, df["close"].values)
            else:
                features["rsi"] = np.nan
                features["macd"] = np.nan
                features["macd_signal"] = np.nan
                features["bb_upper"] = np.nan
                features["bb_middle"] = np.nan
                features["bb_lower"] = np.nan
                features["atr"] = np.nan
                features["adx"] = np.nan
                features["momentum"] = np.nan
                features["roc"] = np.nan
                features["trend"] = np.nan
            # Объемные признаки
            features["volume_ma"] = df["volume"].rolling(20).mean()
            features["volume_std"] = df["volume"].rolling(20).std()
            features["volume_ratio"] = df["volume"] / features["volume_ma"]
            # Волатильность
            features["high_low_ratio"] = df["high"] / df["low"]
            features["close_open_ratio"] = df["close"] / df["open"]
            # Тренд
            features["trend_strength"] = abs(features["trend"] if "trend" in features else 0)
            # Нормализация
            scaler = StandardScaler()
            # Исправление: используем fillna с правильным методом
            features_filled = features.fillna(0) if hasattr(features, 'fillna') else features
            features_scaled = DataFrame(
                scaler.fit_transform(features_filled),
                columns=features.columns,
                index=features.index,
            )
            return features_scaled
        except Exception as e:
            logger.error(f"Ошибка извлечения признаков: {e}")
            return DataFrame()

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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
            # Исправление: убеждаемся, что передаем итерируемые объекты в zip
            feature_columns = list(features.columns) if hasattr(features, 'columns') else []
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_columns, model.feature_importances_))
            else:
                importance = {}
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

    def update(self, df: DataFrame, model_id: str) -> None:
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
            # Исправление: используем to_numpy() вместо values
            if hasattr(features, 'to_numpy'):
                X = features.to_numpy()
            else:
                X = features.values
            y = (
                self.data_buffer["close"].shift(-1) > self.data_buffer["close"]
            ).astype(int).values[:-1]
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
                error_count=int(self.metrics.get(model_id, {}).get("error_count", 0)),
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

    def predict(self, df: DataFrame, model_id: str) -> Tuple[np.ndarray, float]:
        """Предсказание"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Модель {model_id} не найдена")
            # Извлечение признаков
            features = self._extract_features(df)
            # Нормализация
            # Исправление: используем to_numpy() вместо values
            if hasattr(features, 'to_numpy'):
                X_scaled = self.scalers[model_id].transform(features.to_numpy())
            else:
                X_scaled = self.scalers[model_id].transform(features.values)
            # Предсказание
            model = self.models[model_id]
            model.eval()
            with torch.no_grad():
                predictions = model(torch.FloatTensor(X_scaled)).numpy()
            # Расчет уверенности
            confidence = float(np.mean(np.abs(predictions)))
            return predictions, confidence  # type: ignore[no-any-return]
        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            raise

    def get_metrics(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Получение метрик"""
        if model_id:
            return self.metrics.get(model_id, {})
        return self.metrics

    def get_history(self, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Получение истории"""
        if model_id:
            return [m for m in self.metrics_history if m["model_id"] == model_id]
        return self.metrics_history

    def reset(self) -> None:
        """Сброс состояния"""
        self.data_buffer = pd.DataFrame()
        self.metrics_history = []
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self._prediction_cache.clear()
        self._feature_cache.clear()
