"""
Пример использования Mirror Neuron агента для анализа рыночных паттернов и генерации торговых сигналов.

Демонстрирует:
- Интеграцию с Mirror Neuron агентом
- Анализ рыночных паттернов
- Генерацию торговых сигналов
- Симуляцию торговли
"""

import asyncio
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import uuid4
from datetime import datetime, timedelta

from shared.numpy_utils import np
import pandas as pd
from loguru import logger

from domain.entities.signal import Signal, SignalType, SignalStrength
from domain.types import Symbol
from domain.value_objects.timestamp import Timestamp
from shared.models.example_types import ExampleConfig, ExampleResult, ExampleMode
from examples.base_example import BaseExample, register_example


@register_example("mirror_neuron_signal")
class MirrorNeuronSignalExample(BaseExample):
    """
    Пример использования Mirror Neuron агента для анализа паттернов и генерации сигналов.
    """
    
    def __init__(self, config: Optional[ExampleConfig] = None):
        super().__init__(config)
        self.mirror_agent: Optional[Any] = None  # Mirror Neuron агент
        self.market_data: Optional[pd.DataFrame] = None
        self.signals: List[Signal] = []
        self.trades: List[Dict[str, Any]] = []
        
    async def setup(self) -> None:
        """Инициализация примера."""
        logger.info("Setting up Mirror Neuron Signal example...")
        
        # Генерация тестовых данных
        self.market_data = self.generate_market_data()
        
        # Инициализация Mirror Neuron агента (заглушка)
        self.mirror_agent = self._create_mirror_agent()
        
        # Обучение агента
        await self.train_agent()
        
        logger.info("Mirror Neuron Signal example setup completed")
    
    def _create_mirror_agent(self) -> Any:
        """Создание Mirror Neuron агента (заглушка)."""
        # В реальной системе здесь была бы инициализация настоящего агента
        class MockMirrorAgent:
            async def train(self, data: Dict[str, Any]) -> None:
                logger.info("Mock agent training completed")
            
            async def analyze_patterns(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
                # Симуляция анализа паттернов
                patterns = []
                for i in range(5):
                    patterns.append({
                        'id': f'pattern_{i}',
                        'type': np.random.choice(['bullish', 'bearish', 'reversal']),
                        'confidence': np.random.uniform(0.6, 0.95),
                        'neural_activation': np.random.uniform(0.5, 0.9)
                    })
                return patterns
        
        return MockMirrorAgent()
    
    async def run(self) -> ExampleResult:
        """Выполнение примера."""
        logger.info("Running Mirror Neuron Signal example...")
        start_time = time.time()
        
        try:
            # Анализ рыночных паттернов
            patterns = await self.analyze_market_patterns()
            
            # Генерация торговых сигналов
            signals = await self.generate_trading_signals(patterns)
            self.signals = signals
            
            # Симуляция торговли
            trades = await self.simulate_trading(signals)
            self.trades = trades
            
            # Расчет результатов
            trades_executed = len(trades)
            total_pnl = sum(
                Decimal(str(trade.get("pnl", 0))) if trade.get("pnl", 0) != 0 else Decimal("0") 
                for trade in trades
            ) if trades else Decimal("0")
            
            # Расчет метрик
            performance_metrics = self.calculate_performance_metrics(trades)
            
            duration = time.time() - start_time
            
            return ExampleResult(
                success=True,
                duration_seconds=duration,
                trades_executed=trades_executed,
                total_pnl=total_pnl if isinstance(total_pnl, Decimal) else Decimal("0"),
                max_drawdown=performance_metrics["max_drawdown"],
                sharpe_ratio=performance_metrics["sharpe_ratio"],
                metadata={
                    "signals_generated": len(signals),
                    "trading_signals": len(signals),
                    "pattern_accuracy": self.calculate_pattern_accuracy(),
                    "neural_network_performance": self.get_neural_performance()
                }
            )
            
        except Exception as e:
            logger.error(f"Error in Mirror Neuron Signal example: {e}")
            duration = time.time() - start_time
            
            return ExampleResult(
                success=False,
                duration_seconds=duration,
                trades_executed=0,
                total_pnl=Decimal("0"),
                max_drawdown=Decimal("0"),
                sharpe_ratio=0.0,
                error_message=str(e)
            )
    
    async def cleanup(self) -> None:
        """Очистка ресурсов."""
        if self.mirror_agent:
            await self.mirror_agent.cleanup()
        
        logger.info("Mirror Neuron Signal example cleanup completed")
    
    def validate_prerequisites(self) -> bool:
        """Проверка необходимых условий."""
        return (
            self.config is not None and
            len(self.config.symbols) > 0 and
            self.config.risk_level > 0
        )
    
    def generate_market_data(self) -> pd.DataFrame:
        """Генерация тестовых рыночных данных."""
        np.random.seed(42)
        
        # Параметры данных
        periods = 1000
        start_price = 100.0
        
        # Генерация цен с паттернами
        prices = [start_price]
        volumes = []
        
        for i in range(periods):
            # Базовое случайное блуждание
            base_return = np.random.normal(0, 0.02)
            
            # Добавление паттернов
            if i % 50 == 0:  # Паттерн каждые 50 периодов
                pattern_return = np.random.normal(0.05, 0.01)  # Тренд
                base_return += pattern_return
            
            if i % 100 == 0:  # Волатильность каждые 100 периодов
                volatility_spike = np.random.normal(0, 0.05)
                base_return += volatility_spike
            
            # Обновление цены
            new_price = prices[-1] * (1 + base_return)
            prices.append(new_price)
            
            # Генерация объема
            base_volume = np.random.randint(1000, 10000)
            volume_multiplier = 1 + abs(base_return) * 10  # Больше объема при движении
            volumes.append(int(base_volume * volume_multiplier))
        
        # Создание DataFrame
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=periods),
            periods=periods,
            freq='1min'
        )
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices[:-1],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
            'close': prices[1:],
            'volume': volumes
        })
        
        df['symbol'] = self.config.symbols[0] if self.config.symbols else "BTCUSDT"
        df = df.set_index('timestamp')
        return df
    
    async def train_agent(self) -> None:
        """Обучение Mirror Neuron агента."""
        if not self.mirror_agent or self.market_data is None:
            return
        
        logger.info("Training Mirror Neuron agent...")
        
        # Подготовка данных для обучения
        training_data = self.prepare_training_data()
        
        # Обучение агента
        await self.mirror_agent.train(training_data)
        
        logger.info("Mirror Neuron agent training completed")
    
    def prepare_training_data(self) -> Dict[str, Any]:
        """Подготовка данных для обучения."""
        if self.market_data is None:
            return {}
        
        # Извлечение признаков
        df = self.market_data.copy()
        
        # Технические индикаторы
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['price_ma'] = df['close'].rolling(20).mean()
        
        # Паттерны
        df['trend'] = np.where(df['close'] > df['price_ma'], 1, -1)
        df['volume_spike'] = np.where(df['volume'] > df['volume_ma'] * 1.5, 1, 0)
        
        # Удаление NaN значений
        df = df.dropna()
        
        return {
            'features': df[['returns', 'volatility', 'volume_ma', 'price_ma', 'trend', 'volume_spike']].values,  # type: ignore
            'targets': df['returns'].shift(-1).dropna().values,  # type: ignore
            'timestamps': df.index[:-1].values  # type: ignore
        }
    
    async def analyze_market_patterns(self) -> List[Dict[str, Any]]:
        """Анализ рыночных паттернов."""
        if not self.mirror_agent or self.market_data is None:
            return []
        
        logger.info("Analyzing market patterns...")
        
        # Подготовка текущих данных
        current_data = self.prepare_current_data()
        
        # Анализ паттернов
        patterns = await self.mirror_agent.analyze_patterns(current_data)
        
        logger.info(f"Found {len(patterns)} patterns")
        return patterns
    
    def prepare_current_data(self) -> Dict[str, Any]:
        """Подготовка текущих данных для анализа."""
        if self.market_data is None:
            return {}
        
        # Последние 100 периодов для анализа
        recent_data: pd.DataFrame = self.market_data.iloc[-100:] if len(self.market_data) >= 100 else self.market_data
        
        # Извлечение признаков
        df: pd.DataFrame = recent_data.copy()
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['price_ma'] = df['close'].rolling(20).mean()
        
        # Безопасное извлечение последних значений
        current_price = df['close'].iloc[-1] if len(df) > 0 else 0.0  # type: ignore[index]
        current_volume = df['volume'].iloc[-1] if len(df) > 0 else 0.0  # type: ignore[index]
        
        return {
            'features': df[['returns', 'volatility', 'volume_ma', 'price_ma']].dropna().to_numpy(),
            'timestamps': list(df.index) if hasattr(df.index, '__iter__') else [],  # type: ignore[arg-type]
            'current_price': current_price,
            'current_volume': current_volume
        }
    
    async def generate_trading_signals(self, patterns: List[Dict[str, Any]]) -> List[Signal]:
        """Генерация торговых сигналов на основе паттернов."""
        signals = []
        
        for pattern in patterns:
            if pattern.get('confidence', 0) > 0.7:  # Высокая уверенность
                signal_type = self.determine_signal_type(pattern)
                strength = self.calculate_signal_strength(pattern)
                
                signal = Signal(
                    id=uuid4(),
                    strategy_id=uuid4(),
                    trading_pair=self.config.symbols[0] if self.config.symbols else "BTCUSDT",
                    signal_type=signal_type,
                    strength=strength,
                    timestamp=datetime.now(),
                    metadata={
                        'pattern_id': str(pattern.get('id', '')),
                        'confidence': str(pattern.get('confidence', 0)),
                        'pattern_type': str(pattern.get('type', '')),
                        'neural_activation': str(pattern.get('neural_activation', 0))
                    }
                )
                
                signals.append(signal)
        
        logger.info(f"Generated {len(signals)} trading signals")
        return signals
    
    def determine_signal_type(self, pattern: Dict[str, Any]) -> SignalType:
        """Определение типа сигнала на основе паттерна."""
        pattern_type = pattern.get('type', '')
        confidence = pattern.get('confidence', 0)
        
        if 'bullish' in pattern_type.lower() and confidence > 0.8:
            return SignalType.BUY
        elif 'bearish' in pattern_type.lower() and confidence > 0.8:
            return SignalType.SELL
        elif 'reversal' in pattern_type.lower():
            return SignalType.CLOSE
        else:
            return SignalType.HOLD
    
    def calculate_signal_strength(self, pattern: Dict[str, Any]) -> SignalStrength:
        """Расчет силы сигнала."""
        confidence = pattern.get('confidence', 0)
        neural_activation = pattern.get('neural_activation', 0)
        
        # Комбинированная оценка
        combined_score = (confidence + neural_activation) / 2
        
        if combined_score > 0.9:
            return SignalStrength.VERY_STRONG
        elif combined_score > 0.7:
            return SignalStrength.STRONG
        else:
            return SignalStrength.MEDIUM
    
    async def simulate_trading(self, signals: List[Signal]) -> List[Dict[str, Any]]:
        """Симуляция торговли на основе сигналов."""
        trades = []
        current_position = 0
        entry_price = 0.0
        
        for signal in signals:
            if signal.signal_type == SignalType.BUY and current_position <= 0:
                # Открытие длинной позиции
                entry_price = float(str(signal.metadata.get('current_price', 100.0)))
                current_position = 1
                
            elif signal.signal_type == SignalType.SELL and current_position >= 0:
                # Открытие короткой позиции
                entry_price = float(str(signal.metadata.get('current_price', 100.0)))
                current_position = -1
                
            elif signal.signal_type == SignalType.CLOSE and current_position != 0:
                # Закрытие позиции
                exit_price = float(str(signal.metadata.get('current_price', 100.0)))
                pnl = (exit_price - entry_price) * current_position
                
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': current_position,
                    'pnl': pnl,
                    'signal_id': str(signal.id)
                })
                
                current_position = 0
        
        return trades
    
    def calculate_performance_metrics(self, trades: List[Dict[str, Any]], initial_balance: Decimal = Decimal("10000")) -> Dict[str, Any]:
        """Расчет метрик производительности."""
        if not trades:
            return {"max_drawdown": Decimal("0"), "sharpe_ratio": 0.0}
        
        # Простой расчет метрик
        pnls = [trade.get("pnl", 0) for trade in trades]
        total_pnl = sum(pnls)
        
        # Максимальная просадка (упрощенная)
        max_drawdown = Decimal(str(max(0, -min(pnls))))
        
        # Коэффициент Шарпа (упрощенный)
        if len(pnls) > 1:
            mean_return = np.mean(pnls)
            std_return = np.std(pnls)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        return {
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio
        }
    
    def calculate_pattern_accuracy(self) -> float:
        """Расчет точности паттернов."""
        if not self.mirror_agent:
            return 0.0
        
        # Здесь должна быть логика расчета точности
        # Пока возвращаем фиктивное значение
        return 0.85
    
    def get_neural_performance(self) -> Dict[str, float]:
        """Получение метрик производительности нейронной сети."""
        if not self.mirror_agent:
            return {}
        
        # Здесь должны быть реальные метрики
        return {
            'accuracy': 0.87,
            'precision': 0.82,
            'recall': 0.79,
            'f1_score': 0.80
        }


async def main():
    """Основная функция для запуска примера."""
    config = ExampleConfig(
        symbols=["BTCUSDT"],
        mode=ExampleMode.DEMO,
        risk_level=0.5,
        max_positions=5,
        enable_logging=True
    )
    
    example = MirrorNeuronSignalExample(config)
    result = await example.execute_with_metrics()
    
    example.log_performance_summary(result)
    
    return result


if __name__ == "__main__":
    asyncio.run(main()) 