from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class CodeQualityDataset(Dataset):
    """Датасет для обучения моделей качества кода."""

    def __init__(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


class CodeQualityNeuralNetwork(nn.Module):
    """Нейронная сеть для оценки качества кода."""

    def __init__(self, input_size: int, hidden_sizes: Optional[List[int]] = None) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]
        layers: List[nn.Module] = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.BatchNorm1d(hidden_size),
                ]
            )
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)  # type: ignore[no-any-return]
