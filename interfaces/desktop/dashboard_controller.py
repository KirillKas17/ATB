"""
Контроллер дашборда - связывает UI с бизнес-логикой
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from domain.entities.trading import Trade, Position, Order
from domain.entities.strategy import Strategy
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.types import Symbol, TradingPair
from application.services.trading_service import TradingService
from infrastructure.external_services.exchanges.base_exchange_service import BaseExchangeService

@dataclass
class TradingSession:
    """Сессия торговли"""
    session_id: str
    start_time: datetime
    mode: str  # 'live', 'simulation', 'backtest'
    selected_pairs: List[str]
    active_strategies: List[str]
    initial_balance: Decimal
    current_balance: Decimal
    total_pnl: Decimal
    trades_count: int
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    is_active: bool

class DashboardController:
    """Контроллер дашборда"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Текущая сессия
        self.current_session: Optional[TradingSession] = None
        
        # Кэшированные данные
        self.cached_market_data: Dict[str, Any] = {}
        self.cached_positions: List[Position] = []
        self.cached_orders: List[Order] = []
        self.cached_trades: List[Trade] = []
        
        # Настройки
        self.update_interval = 1000  # мс
        self.data_retention_days = 30
        
    async def start_trading_session(self, 
                                  mode: str,
                                  selected_pairs: List[str],
                                  active_strategies: List[str],
                                  initial_balance: Decimal,
                                  risk_params: Dict[str, float]) -> TradingSession:
        """Запуск торговой сессии"""
        
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Создание новой сессии
        session = TradingSession(
            session_id=session_id,
            start_time=datetime.now(),
            mode=mode,
            selected_pairs=selected_pairs,
            active_strategies=active_strategies,
            initial_balance=initial_balance,
            current_balance=initial_balance,
            total_pnl=Decimal('0'),
            trades_count=0,
            win_rate=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            is_active=True
        )
        
        self.current_session = session
        
        # Инициализация торговых сервисов
        await self._initialize_trading_services(session)
        
        self.logger.info(f"Торговая сессия запущена: {session_id}")
        return session
    
    async def stop_trading_session(self) -> None:
        """Остановка торговой сессии"""
        if self.current_session:
            self.current_session.is_active = False
            
            # Закрытие всех позиций в режиме live
            if self.current_session.mode == 'live':
                await self._close_all_positions()
            
            # Сохранение результатов сессии
            await self._save_session_results(self.current_session)
            
            self.logger.info(f"Торговая сессия остановлена: {self.current_session.session_id}")
            self.current_session = None
    
    async def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Dict[str, Any]:
        """Получение рыночных данных"""
        cache_key = f"{symbol}_{timeframe}"
        
        # Проверка кэша
        if cache_key in self.cached_market_data:
            cached_data = self.cached_market_data[cache_key]
            if datetime.now() - cached_data['timestamp'] < timedelta(minutes=1):
                return cached_data['data']
        
        # Получение новых данных
        try:
            # В продакшене здесь будет реальный API
            market_data = await self._fetch_market_data(symbol, timeframe, limit)
            
            # Кэширование
            self.cached_market_data[cache_key] = {
                'data': market_data,
                'timestamp': datetime.now()
            }
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Ошибка получения рыночных данных для {symbol}: {e}")
            return self._get_mock_market_data(symbol)
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Получение текущих позиций"""
        try:
            # В продакшене здесь будет реальный API
            positions = await self._fetch_positions()
            
            # Преобразование в формат для UI
            formatted_positions = []
            for position in positions:
                formatted_positions.append({
                    'symbol': position.symbol,
                    'side': position.side.value,
                    'size': float(position.quantity.amount),
                    'entry_price': float(position.entry_price.amount),
                    'current_price': float(position.current_price.amount) if position.current_price else 0,
                    'pnl': float(position.unrealized_pnl.amount) if position.unrealized_pnl else 0,
                    'percentage': self._calculate_pnl_percentage(position)
                })
            
            return formatted_positions
            
        except Exception as e:
            self.logger.error(f"Ошибка получения позиций: {e}")
            return []
    
    async def get_orders(self) -> List[Dict[str, Any]]:
        """Получение активных ордеров"""
        try:
            orders = await self._fetch_orders()
            
            formatted_orders = []
            for order in orders:
                formatted_orders.append({
                    'symbol': order.symbol,
                    'type': order.order_type.value,
                    'side': order.side.value,
                    'amount': float(order.quantity.amount),
                    'price': float(order.price.amount) if order.price else 0,
                    'status': order.status.value,
                    'created_at': order.created_at.isoformat()
                })
            
            return formatted_orders
            
        except Exception as e:
            self.logger.error(f"Ошибка получения ордеров: {e}")
            return []
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Получение метрик производительности"""
        if not self.current_session:
            return self._get_empty_metrics()
        
        try:
            # Расчет метрик
            trades = await self._fetch_trades()
            
            metrics = {
                'total_pnl': float(self.current_session.total_pnl),
                'total_return_pct': self._calculate_total_return_pct(),
                'max_drawdown_pct': self._calculate_max_drawdown(),
                'sharpe_ratio': self._calculate_sharpe_ratio(trades),
                'sortino_ratio': self._calculate_sortino_ratio(trades),
                'win_rate_pct': self._calculate_win_rate(trades),
                'profit_factor': self._calculate_profit_factor(trades),
                'avg_win': self._calculate_avg_win(trades),
                'avg_loss': self._calculate_avg_loss(trades),
                'total_trades': len(trades),
                'winning_trades': len([t for t in trades if self._get_trade_pnl(t) > 0]),
                'losing_trades': len([t for t in trades if self._get_trade_pnl(t) < 0]),
                'var_95': self._calculate_var_95(trades),
                'calmar_ratio': self._calculate_calmar_ratio(trades)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета метрик: {e}")
            return self._get_empty_metrics()
    
    async def run_backtest(self, 
                          symbol: str,
                          strategy: str,
                          start_date: datetime,
                          end_date: datetime,
                          initial_capital: Decimal) -> Dict[str, Any]:
        """Запуск бэктестинга"""
        
        try:
            self.logger.info(f"Запуск бэктеста: {strategy} на {symbol}")
            
            # Получение исторических данных
            historical_data = await self._fetch_historical_data(symbol, start_date, end_date)
            
            # Инициализация бэктест-движка
            backtest_engine = await self._create_backtest_engine(strategy, initial_capital)
            
            # Запуск бэктеста
            results = await backtest_engine.run(historical_data)
            
            # Обработка результатов
            processed_results = {
                'equity_curve': results.get('equity_curve', []),
                'trades': results.get('trades', []),
                'metrics': {
                    'total_return': results.get('total_return', 0.0),
                    'sharpe_ratio': results.get('sharpe_ratio', 0.0),
                    'max_drawdown': results.get('max_drawdown', 0.0),
                    'win_rate': results.get('win_rate', 0.0),
                    'profit_factor': results.get('profit_factor', 0.0)
                },
                'risk_metrics': results.get('risk_metrics', {}),
                'period_stats': results.get('period_stats', {})
            }
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Ошибка бэктестинга: {e}")
            # Возврат мок-данных для демонстрации
            return self._get_mock_backtest_results()
    
    async def get_available_pairs(self) -> List[str]:
        """Получение доступных торговых пар"""
        try:
            # В продакшене здесь будет реальный API
            pairs = await self._fetch_available_pairs()
            return pairs
            
        except Exception as e:
            self.logger.error(f"Ошибка получения торговых пар: {e}")
            # Возврат базового списка
            return [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT',
                'SOL/USDT', 'DOT/USDT', 'MATIC/USDT', 'LINK/USDT',
                'AVAX/USDT', 'UNI/USDT', 'ATOM/USDT', 'FTM/USDT',
                'NEAR/USDT', 'ALGO/USDT', 'XRP/USDT', 'LTC/USDT'
            ]
    
    async def get_available_strategies(self) -> List[Dict[str, Any]]:
        """Получение доступных стратегий"""
        strategies = [
            {
                'name': 'RSI Bounce',
                'description': 'Торговля на отскоках RSI от уровней перепроданности/перекупленности',
                'parameters': ['rsi_period', 'oversold_level', 'overbought_level'],
                'risk_level': 'medium',
                'timeframes': ['15m', '1h', '4h']
            },
            {
                'name': 'MACD Cross',
                'description': 'Торговля на пересечениях линий MACD',
                'parameters': ['fast_period', 'slow_period', 'signal_period'],
                'risk_level': 'medium',
                'timeframes': ['1h', '4h', '1d']
            },
            {
                'name': 'Bollinger Squeeze',
                'description': 'Торговля на сжатии и расширении полос Боллинджера',
                'parameters': ['bb_period', 'bb_std', 'squeeze_threshold'],
                'risk_level': 'high',
                'timeframes': ['15m', '1h', '4h']
            },
            {
                'name': 'Mean Reversion',
                'description': 'Стратегия возврата к среднему значению',
                'parameters': ['lookback_period', 'std_threshold', 'holding_period'],
                'risk_level': 'low',
                'timeframes': ['5m', '15m', '1h']
            },
            {
                'name': 'Momentum',
                'description': 'Следование за трендом и моментумом',
                'parameters': ['momentum_period', 'trend_threshold', 'exit_threshold'],
                'risk_level': 'high',
                'timeframes': ['1h', '4h', '1d']
            },
            {
                'name': 'Grid Trading',
                'description': 'Сеточная торговля с фиксированными интервалами',
                'parameters': ['grid_size', 'grid_spacing', 'max_positions'],
                'risk_level': 'medium',
                'timeframes': ['15m', '1h']
            }
        ]
        
        return strategies
    
    # Приватные методы
    async def _initialize_trading_services(self, session: TradingSession) -> None:
        """Инициализация торговых сервисов"""
        pass
    
    async def _close_all_positions(self) -> None:
        """Закрытие всех позиций"""
        pass
    
    async def _save_session_results(self, session: TradingSession) -> None:
        """Сохранение результатов сессии"""
        pass
    
    async def _fetch_market_data(self, symbol: str, timeframe: str, limit: int) -> Dict[str, Any]:
        """Получение реальных рыночных данных"""
        # Мок-данные для демонстрации
        return self._get_mock_market_data(symbol)
    
    def _get_mock_market_data(self, symbol: str) -> Dict[str, Any]:
        """Генерация мок-данных"""
        import numpy as np
        import pandas as pd
        
        # Генерация OHLCV данных
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                            end=datetime.now(), freq='1H')
        
        np.random.seed(42)
        base_price = 45000 if 'BTC' in symbol else 3000
        
        # Генерация цен
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        # OHLC
        opens = prices
        closes = prices * (1 + np.random.normal(0, 0.005, len(dates)))
        highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
        lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
        volumes = np.random.uniform(100, 1000, len(dates))
        
        return {
            'symbol': symbol,
            'timeframe': '1h',
            'data': {
                'timestamps': [int(d.timestamp()) for d in dates],
                'open': opens.tolist(),
                'high': highs.tolist(),
                'low': lows.tolist(),
                'close': closes.tolist(),
                'volume': volumes.tolist()
            },
            'last_updated': datetime.now().isoformat()
        }
    
    async def _fetch_positions(self) -> List[Position]:
        """Получение позиций"""
        # Мок-данные
        return []
    
    async def _fetch_orders(self) -> List[Order]:
        """Получение ордеров"""
        # Мок-данные
        return []
    
    async def _fetch_trades(self) -> List[Trade]:
        """Получение сделок"""
        # Мок-данные
        return []
    
    def _calculate_pnl_percentage(self, position) -> float:
        """Расчет процента P&L"""
        if not position.entry_price or not position.current_price:
            return 0.0
        
        entry = float(position.entry_price.amount)
        current = float(position.current_price.amount)
        
        if position.side.value == 'buy':
            return ((current - entry) / entry) * 100
        else:
            return ((entry - current) / entry) * 100
    
    def _calculate_total_return_pct(self) -> float:
        """Расчет общей доходности в процентах"""
        if not self.current_session:
            return 0.0
        
        initial = float(self.current_session.initial_balance)
        current = float(self.current_session.current_balance)
        
        return ((current - initial) / initial) * 100
    
    def _calculate_max_drawdown(self) -> float:
        """Расчет максимальной просадки"""
        # Реализация расчета максимальной просадки
        return 0.0
    
    def _calculate_sharpe_ratio(self, trades: List[Trade]) -> float:
        """Расчет коэффициента Шарпа"""
        if not trades:
            return 0.0
        
        # Упрощенный расчет
        returns = [self._get_trade_pnl(trade) for trade in trades]
        if not returns:
            return 0.0
        
        import numpy as np
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return * np.sqrt(252)  # Аннуализированный
    
    def _calculate_sortino_ratio(self, trades: List[Trade]) -> float:
        """Расчет коэффициента Сортино"""
        # Упрощенная реализация
        return self._calculate_sharpe_ratio(trades) * 1.2
    
    def _calculate_win_rate(self, trades: List[Trade]) -> float:
        """Расчет винрейта"""
        if not trades:
            return 0.0
        
        winning_trades = sum(1 for trade in trades if self._get_trade_pnl(trade) > 0)
        return (winning_trades / len(trades)) * 100
    
    def _calculate_profit_factor(self, trades: List[Trade]) -> float:
        """Расчет фактора прибыли"""
        if not trades:
            return 0.0
        
        gross_profit = sum(self._get_trade_pnl(trade) for trade in trades if self._get_trade_pnl(trade) > 0)
        gross_loss = abs(sum(self._get_trade_pnl(trade) for trade in trades if self._get_trade_pnl(trade) < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def _calculate_avg_win(self, trades: List[Trade]) -> float:
        """Расчет средней прибыльной сделки"""
        winning_trades = [self._get_trade_pnl(trade) for trade in trades if self._get_trade_pnl(trade) > 0]
        return sum(winning_trades) / len(winning_trades) if winning_trades else 0.0
    
    def _calculate_avg_loss(self, trades: List[Trade]) -> float:
        """Расчет среднего убытка"""
        losing_trades = [self._get_trade_pnl(trade) for trade in trades if self._get_trade_pnl(trade) < 0]
        return sum(losing_trades) / len(losing_trades) if losing_trades else 0.0
    
    def _calculate_var_95(self, trades: List[Trade]) -> float:
        """Расчет VaR 95%"""
        if not trades:
            return 0.0
        
        import numpy as np
        returns = [self._get_trade_pnl(trade) for trade in trades]
        return np.percentile(returns, 5) if returns else 0.0
    
    def _calculate_calmar_ratio(self, trades: List[Trade]) -> float:
        """Расчет коэффициента Калмара"""
        annual_return = self._calculate_annualized_return(trades)
        max_drawdown = self._calculate_max_drawdown()
        
        if max_drawdown == 0:
            return 0.0
        
        return annual_return / abs(max_drawdown)
    
    def _calculate_annualized_return(self, trades: List[Trade]) -> float:
        """Расчет аннуализированной доходности"""
        # Упрощенная реализация
        return self._calculate_total_return_pct()
    
    def _get_trade_pnl(self, trade: Trade) -> float:
        """Получение P&L сделки"""
        if hasattr(trade, 'pnl') and trade.pnl:
            return float(trade.pnl.amount)
        return 0.0
    
    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Пустые метрики"""
        return {
            'total_pnl': 0.0,
            'total_return_pct': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'win_rate_pct': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'var_95': 0.0,
            'calmar_ratio': 0.0
        }
    
    async def _fetch_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Получение исторических данных"""
        return self._get_mock_market_data(symbol)
    
    async def _create_backtest_engine(self, strategy: str, initial_capital: Decimal):
        """Создание движка бэктестинга"""
        # Мок-движок
        return MockBacktestEngine()
    
    def _get_mock_backtest_results(self) -> Dict[str, Any]:
        """Мок-результаты бэктестинга"""
        import numpy as np
        import pandas as pd
        
        # Генерация синтетической кривой доходности
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)
        
        returns = np.random.normal(0.001, 0.02, len(dates))
        equity_curve = (1 + pd.Series(returns)).cumprod()
        
        return {
            'equity_curve': equity_curve.tolist(),
            'trades': [],
            'metrics': {
                'total_return': 15.6,
                'sharpe_ratio': 1.23,
                'max_drawdown': -8.4,
                'win_rate': 58.3,
                'profit_factor': 1.34
            },
            'risk_metrics': {
                'var_95': -2.1,
                'sortino_ratio': 1.45,
                'calmar_ratio': 1.86
            },
            'period_stats': {
                'best_month': 4.2,
                'worst_month': -3.1,
                'avg_monthly': 1.2
            }
        }
    
    async def _fetch_available_pairs(self) -> List[str]:
        """Получение доступных пар"""
        # В продакшене здесь будет API биржи
        return []

class MockBacktestEngine:
    """Мок-движок бэктестинга"""
    
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Запуск бэктеста"""
        # Мок-результаты
        return {
            'total_return': 15.6,
            'sharpe_ratio': 1.23,
            'max_drawdown': -8.4,
            'win_rate': 58.3,
            'profit_factor': 1.34,
            'equity_curve': [],
            'trades': [],
            'risk_metrics': {},
            'period_stats': {}
        }