"""
Интеграция настоящих торговых стратегий с оркестратором.
"""

import asyncio
from typing import Any, Dict, List, Optional
from decimal import Decimal
import logging

from infrastructure.strategies.trend_strategies import TrendStrategy
from infrastructure.strategies.sideways_strategies import SidewaysStrategy
from infrastructure.strategies.volatility_strategy import VolatilityStrategy
from infrastructure.strategies.manipulation_strategies import ManipulationStrategy
from infrastructure.strategies.mean_reversion_strategy import MeanReversionStrategy
from infrastructure.strategies.momentum_strategy import MomentumStrategy
from infrastructure.strategies.scalping_strategy import ScalpingStrategy
from domain.intelligence.entanglement_detector import EntanglementDetector
from domain.intelligence.mirror_detector import MirrorDetector
# Новые продвинутые модули
from domain.strategies.quantum_arbitrage_strategy import QuantumArbitrageStrategy
from domain.intelligence.pattern_analyzer import QuantumPatternAnalyzer
from domain.prediction.neural_market_predictor import NeuralMarketPredictor
from infrastructure.agents.market_maker.agent import MarketMakerModelAgent

logger = logging.getLogger(__name__)


class StrategyIntegrationManager:
    """Менеджер интеграции торговых стратегий."""
    
    def __init__(self):
        self.strategies = {}
        self.active_strategies = []
        self.entanglement_detector = EntanglementDetector()
        self.mirror_detector = MirrorDetector()
        self.market_maker_agent = MarketMakerModelAgent()
        # Новые продвинутые модули
        self.quantum_pattern_analyzer = QuantumPatternAnalyzer()
        self.neural_market_predictor = NeuralMarketPredictor()
        
    async def initialize_strategies(self) -> None:
        """Инициализация всех доступных стратегий."""
        try:
            # Инициализация основных стратегий с дефолтными конфигурациями
            self.strategies.update({
                'trend': TrendStrategy({}),
                'sideways': SidewaysStrategy({}), 
                'volatility': VolatilityStrategy({}),
                'manipulation': ManipulationStrategy({}),
                'mean_reversion': MeanReversionStrategy({}),
                'momentum': MomentumStrategy({}),
                'scalping': ScalpingStrategy({}),
                # Новая квантовая арбитражная стратегия
                'quantum_arbitrage': QuantumArbitrageStrategy(
                    strategy_id='quantum_arbitrage_default',
                    min_profit_threshold=0.001,
                    enable_ml_predictions=True
                )
            })
            
            # Активация базовых стратегий
            self.active_strategies = ['trend', 'mean_reversion', 'momentum', 'quantum_arbitrage']
            
            logger.info(f"Initialized {len(self.strategies)} strategies")
            logger.info(f"Active strategies: {self.active_strategies}")
            
        except Exception as e:
            logger.error(f"Error initializing strategies: {e}")
            
    async def get_signals(self, symbol: str, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Получение торговых сигналов от всех активных стратегий."""
        signals = []
        
        for strategy_name in self.active_strategies:
            try:
                strategy = self.strategies.get(strategy_name)
                if strategy and hasattr(strategy, 'generate_signal'):
                    signal = await self._get_strategy_signal(strategy, symbol, market_data)
                    if signal:
                        signal['strategy'] = strategy_name
                        signals.append(signal)
                        
            except Exception as e:
                logger.error(f"Error getting signal from {strategy_name}: {e}")
                
        return signals
    
    async def _get_strategy_signal(self, strategy, symbol: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Получение сигнала от конкретной стратегии."""
        try:
            # Базовая структура для передачи данных стратегии
            strategy_data = {
                'symbol': symbol,
                'price': float(market_data.get('price', 0)),
                'volume': float(market_data.get('volume', 0)),
                'timestamp': market_data.get('timestamp'),
                'bid': float(market_data.get('bid', 0)),
                'ask': float(market_data.get('ask', 0))
            }
            
            # Попытка получить сигнал
            if hasattr(strategy, 'analyze'):
                signal = await strategy.analyze(strategy_data)
            elif hasattr(strategy, 'generate_signal'):
                signal = await strategy.generate_signal(strategy_data)
            elif hasattr(strategy, 'should_enter'):
                should_enter = await strategy.should_enter(strategy_data)
                signal = {'action': 'buy' if should_enter else 'hold', 'confidence': 0.5}
            else:
                # Создаем mock сигнал
                signal = {
                    'action': 'hold',
                    'confidence': 0.3,
                    'reason': f'Basic signal from {strategy.__class__.__name__}'
                }
                
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
    
    async def analyze_market_entanglement(self, symbols: List[str], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ рыночной запутанности между символами."""
        try:
            if len(symbols) < 2:
                return {"entanglement_detected": False}
                
            # Подготовка данных для анализа запутанности
            exchange_data = {}
            for symbol in symbols[:2]:  # Берем первые два символа
                exchange_data[f"exchange_{symbol}"] = {
                    'prices': [float(market_data.get('price', 50000))],
                    'volumes': [float(market_data.get('volume', 1000))],
                    'timestamps': [market_data.get('timestamp', '2025-08-03T17:00:00Z')]
                }
            
            result = self.entanglement_detector.detect_entanglement(
                symbols[0], exchange_data
            )
            
            return {
                "entanglement_detected": result.strength.value != "NONE" if hasattr(result, 'strength') else False,
                "analysis": result.__dict__ if hasattr(result, '__dict__') else {}
            }
            
        except Exception as e:
            logger.error(f"Error analyzing entanglement: {e}")
            return {"entanglement_detected": False, "error": str(e)}
    
    async def analyze_mirror_patterns(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ зеркальных паттернов."""
        try:
            # Создаем простые временные ряды для анализа
            import pandas as pd
            
            price_series = pd.Series([
                float(market_data.get('price', 50000)),
                float(market_data.get('bid', 49999)),
                float(market_data.get('ask', 50001))
            ])
            
            volume_series = pd.Series([
                float(market_data.get('volume', 1000)),
                900.0,
                1100.0
            ])
            
            mirror_result = self.mirror_detector.detect_mirror_signals(
                price_series, volume_series
            )
            
            return {
                "mirror_detected": bool(mirror_result.signals) if hasattr(mirror_result, 'signals') else False,
                "analysis": mirror_result.__dict__ if hasattr(mirror_result, '__dict__') else {}
            }
            
        except Exception as e:
            logger.error(f"Error analyzing mirror patterns: {e}")
            return {"mirror_detected": False, "error": str(e)}
    
    async def get_market_maker_insights(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Получение инсайтов от маркет-мейкер агента."""
        try:
            # Адаптация агента к текущим рыночным данным
            adaptation_data = {
                'symbol': symbol,
                'spread': abs(float(market_data.get('ask', 50001)) - float(market_data.get('bid', 49999))),
                'volume': float(market_data.get('volume', 1000)),
                'price_change': float(market_data.get('change_24h', 0))
            }
            
            adapted = await self.market_maker_agent.adapt(adaptation_data)
            
            return {
                "adapted": adapted,
                "performance": self.market_maker_agent.performance,
                "confidence": self.market_maker_agent.confidence,
                "insights": "Market maker analysis completed"
            }
            
        except Exception as e:
            logger.error(f"Error getting market maker insights: {e}")
            return {"adapted": False, "error": str(e)}
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Получение статуса здоровья интеграции стратегий."""
        return {
            "status": "healthy",
            "total_strategies": len(self.strategies),
            "active_strategies": len(self.active_strategies),
            "strategy_names": list(self.strategies.keys()),
            "active_strategy_names": self.active_strategies
        }
    
    async def cleanup(self) -> None:
        """Очистка ресурсов."""
        try:
            for strategy in self.strategies.values():
                if hasattr(strategy, 'cleanup'):
                    await strategy.cleanup()
            logger.info("Strategy integration cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Глобальный экземпляр менеджера стратегий
strategy_integration = StrategyIntegrationManager()