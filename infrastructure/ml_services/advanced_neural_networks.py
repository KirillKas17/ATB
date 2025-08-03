"""
Продвинутый модуль нейронных сетей для финансового прогнозирования
Включает Transformer, Self-Attention, Graph Neural Networks, Temporal Convolutions
"""

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, LayerNorm, Dropout
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

# Попытка импорта дополнительных библиотек
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, TransformerConv
    from torch_geometric.data import Data, Batch
    GEOMETRIC_AVAILABLE = True
except ImportError:
    GEOMETRIC_AVAILABLE = False
    logger.warning("PyTorch Geometric not available, graph features disabled")


@dataclass
class NeuralNetworkConfig:
    """Конфигурация нейронных сетей"""
    
    # Общие параметры
    input_dim: int = 50
    hidden_dim: int = 512
    output_dim: int = 1
    num_layers: int = 6
    dropout: float = 0.1
    
    # Transformer параметры
    num_heads: int = 8
    ff_dim: int = 2048
    max_seq_length: int = 512
    use_positional_encoding: bool = True
    
    # Temporal Convolutional Network
    tcn_channels: List[int] = field(default_factory=lambda: [256, 256, 256])
    kernel_size: int = 3
    dilation_base: int = 2
    
    # Graph Neural Network
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 3
    gnn_heads: int = 4
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 4000
    
    # Advanced features
    use_residual_connections: bool = True
    use_layer_norm: bool = True
    use_attention_dropout: bool = True
    gradient_clipping: float = 1.0


class PositionalEncoding(nn.Module):
    """Улучшенное позиционное кодирование"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Создаем матрицу позиционных кодирований
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Используем улучшенную формулу для более стабильного обучения
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Регистрируем как буфер (не обучаемый параметр)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """Относительное позиционное кодирование (как в T5)"""
    
    def __init__(self, d_model: int, max_relative_position: int = 128):
        super().__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # Обучаемые embeddings для относительных позиций
        self.relative_position_embeddings = nn.Embedding(
            2 * max_relative_position + 1, d_model
        )
        
    def forward(self, seq_len: int) -> torch.Tensor:
        """Создает матрицу относительных позиций"""
        range_vec = torch.arange(seq_len)
        range_mat = range_vec.unsqueeze(0).expand(seq_len, seq_len)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # Обрезаем до максимального относительного расстояния
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        
        # Сдвигаем, чтобы индексы были положительными
        final_mat = distance_mat_clipped + self.max_relative_position
        
        return self.relative_position_embeddings(final_mat)


class MultiHeadSelfAttention(nn.Module):
    """Улучшенное многоголовочное самовнимание"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, 
                 use_relative_position: bool = True):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Линейные проекции для Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(dropout)
        
        # Относительное позиционное кодирование
        self.use_relative_position = use_relative_position
        if use_relative_position:
            self.relative_position_encoding = RelativePositionalEncoding(self.head_dim)
            
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] or None
            
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = x.size()
        
        # Проекции Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Вычисляем attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Добавляем относительное позиционное кодирование
        if self.use_relative_position:
            relative_position_scores = self._relative_position_attention(Q, seq_len)
            attention_scores = attention_scores + relative_position_scores
        
        # Применяем маску
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Применяем attention к V
        context = torch.matmul(attention_weights, V)
        
        # Объединяем головы
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Финальная проекция
        output = self.w_o(context)
        output = self.dropout(output)
        
        return output, attention_weights
    
    def _relative_position_attention(self, Q: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Вычисляет attention scores с относительным позиционным кодированием"""
        relative_embeddings = self.relative_position_encoding(seq_len).to(Q.device)
        
        # Q: [batch_size, num_heads, seq_len, head_dim]
        # relative_embeddings: [seq_len, seq_len, head_dim]
        
        # Расширяем для batch и heads
        relative_embeddings = relative_embeddings.unsqueeze(0).unsqueeze(0)
        relative_embeddings = relative_embeddings.expand(
            Q.size(0), Q.size(1), -1, -1, -1
        )
        
        # Вычисляем relative attention scores
        relative_scores = torch.matmul(
            Q.unsqueeze(-2), relative_embeddings.transpose(-2, -1)
        ).squeeze(-2)
        
        return relative_scores


class FeedForwardNetwork(nn.Module):
    """Улучшенная feed-forward сеть"""
    
    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1, 
                 activation: str = 'relu'):
        super().__init__()
        
        self.w1 = nn.Linear(d_model, ff_dim)
        self.w2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Выбор функции активации
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            self.activation = nn.ReLU()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(self.activation(self.w1(x))))


class TransformerEncoderLayer(nn.Module):
    """Улучшенный слой Transformer encoder"""
    
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, 
                 dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()
        
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, ff_dim, dropout, activation)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class AdvancedTransformer(nn.Module):
    """Продвинутый Transformer для финансового прогнозирования"""
    
    def __init__(self, config: NeuralNetworkConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Positional encoding
        if config.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(
                config.hidden_dim, config.max_seq_length, config.dropout
            )
        else:
            self.positional_encoding = None
            
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                config.hidden_dim, config.num_heads, config.ff_dim, config.dropout
            ) for _ in range(config.num_layers)
        ])
        
        # Output projection
        self.output_norm = LayerNorm(config.hidden_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )
        
        # Инициализация весов
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Инициализация весов с Xavier/He"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            mask: [batch_size, seq_len, seq_len] or None
            
        Returns:
            output: [batch_size, seq_len, output_dim]
        """
        # Input projection
        x = self.input_projection(x)
        
        # Positional encoding
        if self.positional_encoding is not None:
            x = x.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
            x = self.positional_encoding(x)
            x = x.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Output
        x = self.output_norm(x)
        output = self.output_projection(x)
        
        return output


class TemporalConvolutionalNetwork(nn.Module):
    """Temporal Convolutional Network для временных рядов"""
    
    def __init__(self, config: NeuralNetworkConfig):
        super().__init__()
        self.config = config
        
        layers = []
        input_channels = config.input_dim
        
        for i, hidden_channels in enumerate(config.tcn_channels):
            dilation = config.dilation_base ** i
            padding = (config.kernel_size - 1) * dilation
            
            # Causal convolution
            conv = nn.Conv1d(
                input_channels, hidden_channels, config.kernel_size,
                padding=padding, dilation=dilation
            )
            
            # Residual connection
            residual = None
            if input_channels != hidden_channels:
                residual = nn.Conv1d(input_channels, hidden_channels, 1)
                
            layer = TemporalBlock(
                conv, hidden_channels, config.dropout, residual
            )
            layers.append(layer)
            input_channels = hidden_channels
            
        self.tcn = nn.Sequential(*layers)
        
        # Output projection
        self.output_projection = nn.Linear(
            config.tcn_channels[-1], config.output_dim
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            
        Returns:
            output: [batch_size, seq_len, output_dim]
        """
        # Переставляем для conv1d: [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)
        
        # TCN
        x = self.tcn(x)
        
        # Обратно: [batch_size, seq_len, hidden_dim]
        x = x.transpose(1, 2)
        
        # Output projection
        output = self.output_projection(x)
        
        return output


class TemporalBlock(nn.Module):
    """Блок временной свертки с residual connections"""
    
    def __init__(self, conv: nn.Conv1d, hidden_channels: int, 
                 dropout: float, residual: Optional[nn.Conv1d] = None):
        super().__init__()
        
        self.conv1 = conv
        self.conv2 = nn.Conv1d(
            hidden_channels, hidden_channels, conv.kernel_size[0],
            padding=conv.padding[0], dilation=conv.dilation[0]
        )
        
        self.norm1 = nn.BatchNorm1d(hidden_channels)
        self.norm2 = nn.BatchNorm1d(hidden_channels)
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.residual = residual
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Сохраняем для residual connection
        residual = x
        
        # Первая свертка
        out = self.conv1(x)
        out = self._causal_crop(out, x.size(2))
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Вторая свертка
        out = self.conv2(out)
        out = self._causal_crop(out, x.size(2))
        out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.residual is not None:
            residual = self.residual(residual)
            
        return self.activation(out + residual)
    
    def _causal_crop(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        """Обрезает выход для причинности"""
        if x.size(2) > target_length:
            return x[:, :, :target_length]
        return x


class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network для анализа связей между активами"""
    
    def __init__(self, config: NeuralNetworkConfig):
        super().__init__()
        self.config = config
        
        if not GEOMETRIC_AVAILABLE:
            logger.warning("PyTorch Geometric not available, using dummy GNN")
            self.gnn_layers = nn.ModuleList([
                nn.Linear(config.input_dim, config.gnn_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.gnn_hidden_dim, config.output_dim)
            ])
            self.is_dummy = True
            return
            
        self.is_dummy = False
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        
        # Первый слой
        self.gnn_layers.append(
            GATConv(config.input_dim, config.gnn_hidden_dim, 
                   heads=config.gnn_heads, dropout=config.dropout)
        )
        
        # Скрытые слои
        for _ in range(config.gnn_num_layers - 2):
            self.gnn_layers.append(
                GATConv(config.gnn_hidden_dim * config.gnn_heads, 
                       config.gnn_hidden_dim, heads=config.gnn_heads, 
                       dropout=config.dropout)
            )
            
        # Последний слой
        self.gnn_layers.append(
            GATConv(config.gnn_hidden_dim * config.gnn_heads, 
                   config.output_dim, heads=1, dropout=config.dropout)
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment [num_nodes] or None
            
        Returns:
            output: [num_nodes, output_dim]
        """
        if self.is_dummy:
            # Dummy implementation
            x = self.gnn_layers[0](x)
            x = self.gnn_layers[1](x)
            x = self.gnn_layers[2](x)
            return x
            
        # GNN forward pass
        for i, layer in enumerate(self.gnn_layers[:-1]):
            x = layer(x, edge_index)
            x = F.elu(x)
            x = self.dropout(x)
            
        # Последний слой без активации
        x = self.gnn_layers[-1](x, edge_index)
        
        return x


class MultiModalFusionNetwork(nn.Module):
    """Мультимодальная сеть для объединения разных типов данных"""
    
    def __init__(self, config: NeuralNetworkConfig):
        super().__init__()
        self.config = config
        
        # Подсети для разных модальностей
        self.transformer = AdvancedTransformer(config)
        self.tcn = TemporalConvolutionalNetwork(config)
        self.gnn = GraphNeuralNetwork(config)
        
        # Fusion layers
        fusion_input_dim = config.output_dim * 3  # 3 модальности
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )
        
        # Attention weights for modalities
        self.modality_attention = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_input_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_input_dim // 2, 3),  # 3 модальности
            nn.Softmax(dim=-1)
        )
        
    def forward(self, sequential_data: torch.Tensor, 
                graph_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            sequential_data: [batch_size, seq_len, input_dim]
            graph_data: (node_features, edge_index) or None
            mask: [batch_size, seq_len, seq_len] or None
            
        Returns:
            Dictionary with outputs and attention weights
        """
        # Transformer output
        transformer_out = self.transformer(sequential_data, mask)
        transformer_pooled = transformer_out.mean(dim=1)  # Global average pooling
        
        # TCN output
        tcn_out = self.tcn(sequential_data)
        tcn_pooled = tcn_out.mean(dim=1)  # Global average pooling
        
        # GNN output (если доступны graph data)
        if graph_data is not None:
            node_features, edge_index = graph_data
            gnn_out = self.gnn(node_features, edge_index)
            gnn_pooled = gnn_out.mean(dim=0, keepdim=True)  # Global pooling
            gnn_pooled = gnn_pooled.expand(transformer_pooled.size(0), -1)
        else:
            # Используем нули для GNN если нет graph data
            gnn_pooled = torch.zeros_like(transformer_pooled)
        
        # Объединяем выходы всех модальностей
        fused_features = torch.cat([transformer_pooled, tcn_pooled, gnn_pooled], dim=-1)
        
        # Вычисляем attention weights для модальностей
        modality_weights = self.modality_attention(fused_features)
        
        # Применяем attention к каждой модальности
        weighted_transformer = transformer_pooled * modality_weights[:, 0:1]
        weighted_tcn = tcn_pooled * modality_weights[:, 1:2] 
        weighted_gnn = gnn_pooled * modality_weights[:, 2:3]
        
        weighted_features = torch.cat([weighted_transformer, weighted_tcn, weighted_gnn], dim=-1)
        
        # Final fusion
        output = self.fusion(weighted_features)
        
        return {
            'output': output,
            'transformer_out': transformer_out,
            'tcn_out': tcn_out,
            'modality_weights': modality_weights,
            'fused_features': fused_features
        }


class AdaptiveLearningRateScheduler:
    """Адаптивный планировщик learning rate"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, config: NeuralNetworkConfig):
        self.optimizer = optimizer
        self.config = config
        self.step_count = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.patience = 10
        self.factor = 0.5
        
    def step(self, loss: float):
        """Обновляет learning rate на основе loss"""
        self.step_count += 1
        
        # Warmup phase
        if self.step_count <= self.config.warmup_steps:
            lr = self.config.learning_rate * (self.step_count / self.config.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Adaptive reduction
            if loss < self.best_loss:
                self.best_loss = loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                current_lr = self.optimizer.param_groups[0]['lr']
                new_lr = current_lr * self.factor
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                self.patience_counter = 0
                logger.info(f"Reduced learning rate to {new_lr}")


class EnsembleNeuralNetwork(nn.Module):
    """Ансамбль нейронных сетей для улучшенного прогнозирования"""
    
    def __init__(self, configs: List[NeuralNetworkConfig], ensemble_method: str = 'weighted'):
        super().__init__()
        self.ensemble_method = ensemble_method
        
        # Создаем разные архитектуры
        self.models = nn.ModuleList()
        
        for config in configs:
            # Варьируем архитектуры
            if len(self.models) % 3 == 0:
                model = AdvancedTransformer(config)
            elif len(self.models) % 3 == 1:
                model = TemporalConvolutionalNetwork(config)
            else:
                model = MultiModalFusionNetwork(config)
                
            self.models.append(model)
        
        # Веса для ансамбля
        if ensemble_method == 'weighted':
            self.ensemble_weights = nn.Parameter(torch.ones(len(configs)))
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: Input tensor
            **kwargs: Additional arguments for specific models
            
        Returns:
            Ensemble output
        """
        outputs = []
        
        for i, model in enumerate(self.models):
            if isinstance(model, MultiModalFusionNetwork):
                # Для multimodal network передаем дополнительные аргументы
                result = model(x, **kwargs)
                output = result['output']
            else:
                # Для остальных моделей только основной вход
                output = model(x)
                if output.dim() > 2:
                    output = output.mean(dim=1)  # Pool over sequence dimension
                    
            outputs.append(output)
        
        # Объединяем выходы
        if self.ensemble_method == 'average':
            return torch.stack(outputs).mean(dim=0)
        elif self.ensemble_method == 'weighted':
            weights = F.softmax(self.ensemble_weights, dim=0)
            weighted_outputs = torch.stack([w * out for w, out in zip(weights, outputs)])
            return weighted_outputs.sum(dim=0)
        else:
            # Просто конкатенация
            return torch.cat(outputs, dim=-1)


class MetaLearningNetwork(nn.Module):
    """Мета-обучающаяся сеть для быстрой адаптации к новым рынкам"""
    
    def __init__(self, config: NeuralNetworkConfig):
        super().__init__()
        self.config = config
        
        # Base network
        self.base_network = AdvancedTransformer(config)
        
        # Meta-learning components
        self.meta_network = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim)
        )
        
        # Task embedding
        self.task_embedding = nn.Embedding(100, config.hidden_dim)  # Support 100 different tasks
        
        # Adaptation parameters
        self.adaptation_layers = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.hidden_dim) 
            for _ in range(3)
        ])
        
    def forward(self, x: torch.Tensor, task_id: int = 0, 
                adapt_steps: int = 5) -> torch.Tensor:
        """
        Args:
            x: Input data [batch_size, seq_len, input_dim]
            task_id: ID of the current task/market
            adapt_steps: Number of adaptation steps
            
        Returns:
            Adapted output
        """
        # Base forward pass
        base_output = self.base_network(x)
        
        # Get task embedding
        task_emb = self.task_embedding(torch.tensor(task_id, device=x.device))
        
        # Meta-adaptation
        adapted_output = base_output
        for step in range(adapt_steps):
            # Compute adaptation direction
            meta_signal = self.meta_network(task_emb)
            
            # Apply adaptation layers
            for adaptation_layer in self.adaptation_layers:
                adapted_output = adapted_output + adaptation_layer(meta_signal.unsqueeze(0).unsqueeze(0))
        
        return adapted_output
    
    def fast_adapt(self, support_x: torch.Tensor, support_y: torch.Tensor,
                   query_x: torch.Tensor, learning_rate: float = 0.01) -> torch.Tensor:
        """MAML-style fast adaptation"""
        
        # Compute gradients on support set
        support_pred = self.forward(support_x)
        support_loss = F.mse_loss(support_pred.squeeze(), support_y)
        
        # Compute gradients
        grads = torch.autograd.grad(support_loss, self.parameters(), create_graph=True)
        
        # Update parameters (conceptually)
        adapted_params = []
        for param, grad in zip(self.parameters(), grads):
            adapted_params.append(param - learning_rate * grad)
        
        # Forward pass on query set with adapted parameters
        # Note: This is a simplified version. Full MAML requires more complex parameter updates
        query_pred = self.forward(query_x)
        
        return query_pred


class AdvancedNeuralNetworkFactory:
    """Фабрика для создания продвинутых нейронных сетей"""
    
    @staticmethod
    def create_transformer(input_dim: int, output_dim: int = 1, 
                          **kwargs) -> AdvancedTransformer:
        """Создает продвинутый трансформер"""
        config = NeuralNetworkConfig(input_dim=input_dim, output_dim=output_dim, **kwargs)
        return AdvancedTransformer(config)
    
    @staticmethod
    def create_tcn(input_dim: int, output_dim: int = 1, 
                   **kwargs) -> TemporalConvolutionalNetwork:
        """Создает TCN"""
        config = NeuralNetworkConfig(input_dim=input_dim, output_dim=output_dim, **kwargs)
        return TemporalConvolutionalNetwork(config)
    
    @staticmethod
    def create_gnn(input_dim: int, output_dim: int = 1, 
                   **kwargs) -> GraphNeuralNetwork:
        """Создает GNN"""
        config = NeuralNetworkConfig(input_dim=input_dim, output_dim=output_dim, **kwargs)
        return GraphNeuralNetwork(config)
    
    @staticmethod
    def create_multimodal(input_dim: int, output_dim: int = 1, 
                         **kwargs) -> MultiModalFusionNetwork:
        """Создает мультимодальную сеть"""
        config = NeuralNetworkConfig(input_dim=input_dim, output_dim=output_dim, **kwargs)
        return MultiModalFusionNetwork(config)
    
    @staticmethod
    def create_ensemble(input_dim: int, output_dim: int = 1, 
                       num_models: int = 5, **kwargs) -> EnsembleNeuralNetwork:
        """Создает ансамбль сетей"""
        configs = []
        for i in range(num_models):
            config = NeuralNetworkConfig(input_dim=input_dim, output_dim=output_dim, **kwargs)
            # Варьируем параметры для разнообразия
            config.num_layers = max(3, config.num_layers + random.randint(-2, 2))
            config.hidden_dim = max(64, config.hidden_dim + random.randint(-128, 128))
            config.num_heads = max(2, config.num_heads + random.randint(-2, 2))
            configs.append(config)
        
        return EnsembleNeuralNetwork(configs)
    
    @staticmethod
    def create_meta_learning(input_dim: int, output_dim: int = 1,
                           **kwargs) -> MetaLearningNetwork:
        """Создает мета-обучающуюся сеть"""
        config = NeuralNetworkConfig(input_dim=input_dim, output_dim=output_dim, **kwargs)
        return MetaLearningNetwork(config)


# Экспорт основных классов
__all__ = [
    'NeuralNetworkConfig',
    'AdvancedTransformer',
    'TemporalConvolutionalNetwork', 
    'GraphNeuralNetwork',
    'MultiModalFusionNetwork',
    'EnsembleNeuralNetwork',
    'MetaLearningNetwork',
    'AdvancedNeuralNetworkFactory',
    'AdaptiveLearningRateScheduler'
]