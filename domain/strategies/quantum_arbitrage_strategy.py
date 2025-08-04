# -*- coding: utf-8 -*-
"""
Quantum-Enhanced Arbitrage Strategy with Advanced Mathematical Models.
Implements sophisticated arbitrage detection using quantum algorithms,
machine learning, and multi-dimensional market analysis.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import concurrent.futures
from uuid import uuid4

from shared.numpy_utils import np
import pandas as pd
from loguru import logger
from scipy import stats, optimize
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from domain.entities.market import MarketData
from domain.entities.signal import Signal, SignalType
from domain.entities.strategy import Strategy, StrategyType
from domain.entities.trading_pair import TradingPair
from domain.strategies.base_strategy import BaseStrategy, StrategyMetrics
from domain.types.strategy_types import StrategyId
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.entities.symbol import Symbol
from domain.intelligence.quantum_pattern_analyzer import (
    QuantumPatternAnalyzer, 
    MultidimensionalPattern,
    PatternDimension
)


class ArbitrageType(Enum):
    """Типы арбитража."""
    SPATIAL = "spatial"  # Пространственный арбитраж между биржами
    TEMPORAL = "temporal"  # Временной арбитраж
    STATISTICAL = "statistical"  # Статистический арбитраж
    TRIANGULAR = "triangular"  # Треугольный арбитраж
    LATENCY = "latency"  # Латентностный арбитраж
    QUANTUM = "quantum"  # Квантовый арбитраж


class RiskLevel(Enum):
    """Уровни риска арбитража."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class ArbitrageOpportunity:
    """Арбитражная возможность."""
    opportunity_id: str
    arbitrage_type: ArbitrageType
    trading_pairs: List[TradingPair]
    expected_profit: Decimal
    risk_level: RiskLevel
    confidence_score: float
    execution_window: timedelta
    market_data: Dict[str, MarketData]
    quantum_patterns: List[MultidimensionalPattern]
    discovery_timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_profit_potential(self) -> float:
        """Расчёт потенциала прибыли с учётом рисков."""
        base_profit = float(self.expected_profit)
        
        # Корректировка на риск
        risk_multipliers = {
            RiskLevel.VERY_LOW: 0.95,
            RiskLevel.LOW: 0.85,
            RiskLevel.MEDIUM: 0.7,
            RiskLevel.HIGH: 0.5,
            RiskLevel.EXTREME: 0.2
        }
        
        risk_adjusted_profit = base_profit * risk_multipliers[self.risk_level]
        
        # Корректировка на доверие
        confidence_adjusted_profit = risk_adjusted_profit * self.confidence_score
        
        # Корректировка на временное окно
        time_factor = min(1.0, 300.0 / self.execution_window.total_seconds())  # 5 минут оптимально
        
        return confidence_adjusted_profit * time_factor
    
    def is_actionable(self, min_profit_threshold: float = 0.001) -> bool:
        """Проверка возможности исполнения арбитража."""
        return (
            self.calculate_profit_potential() > min_profit_threshold and
            self.confidence_score > 0.7 and
            self.execution_window.total_seconds() > 5  # Минимум 5 секунд
        )


@dataclass
class ArbitrageExecutionPlan:
    """План исполнения арбитража."""
    opportunity: ArbitrageOpportunity
    execution_steps: List[Dict[str, Any]]
    estimated_execution_time: timedelta
    capital_requirement: Decimal
    expected_return: Decimal
    risk_metrics: Dict[str, float]
    backup_plans: List[Dict[str, Any]] = field(default_factory=list)
    
    def validate_execution_feasibility(self, available_capital: Decimal) -> bool:
        """Валидация возможности исполнения."""
        return (
            available_capital >= self.capital_requirement and
            len(self.execution_steps) > 0 and
            self.estimated_execution_time <= self.opportunity.execution_window
        )


class QuantumArbitrageStrategy(BaseStrategy):
    """
    Продвинутая квантовая арбитражная стратегия с машинным обучением
    и многомерным анализом рыночных возможностей.
    """
    
    def __init__(
        self,
        strategy_id: str,
        min_profit_threshold: float = 0.001,
        max_risk_level: RiskLevel = RiskLevel.MEDIUM,
        quantum_precision: int = 1000,
        enable_ml_predictions: bool = True,
        max_concurrent_opportunities: int = 5
    ):
        # Создаем правильный StrategyId
        strategy_id_obj = StrategyId(strategy_id) if isinstance(strategy_id, str) else strategy_id
        super().__init__(strategy_id_obj, {"type": "ARBITRAGE"})
        
        self.min_profit_threshold = min_profit_threshold
        self.max_risk_level = max_risk_level
        self.quantum_precision = quantum_precision
        self.enable_ml_predictions = enable_ml_predictions
        self.max_concurrent_opportunities = max_concurrent_opportunities
        
        # Квантовый анализатор паттернов
        self.quantum_analyzer = QuantumPatternAnalyzer(
            quantum_precision=quantum_precision,
            enable_parallel_processing=True
        )
        
        # Модели машинного обучения
        self.price_predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.scaler = RobustScaler()
        
        # Данные и состояние
        self.market_data_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.price_correlations: Dict[Tuple[str, str], float] = {}
        self.active_opportunities: Dict[str, ArbitrageOpportunity] = {}
        self.execution_plans: Dict[str, ArbitrageExecutionPlan] = {}
        
        # Статистика и метрики
        self.strategy_statistics = {
            'total_opportunities_found': 0,
            'successful_arbitrages': 0,
            'failed_arbitrages': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'average_execution_time': 0.0,
            'opportunity_types': defaultdict(int),
            'risk_distribution': defaultdict(int),
            'quantum_patterns_utilized': 0
        }
        
        # Модель обучена?
        self._ml_models_trained = False
        
        logger.info(f"QuantumArbitrageStrategy initialized: {strategy_id}")
    
    async def analyze_market(self, market_data: Dict[str, MarketData]) -> List[Signal]:
        """
        Анализ рынка для поиска арбитражных возможностей.
        
        Args:
            market_data: Словарь рыночных данных по символам
            
        Returns:
            Список торговых сигналов
        """
        start_time = time.time()
        signals = []
        
        try:
            # Обновление исторических данных
            await self._update_market_history(market_data)
            
            # Обучение ML моделей при необходимости
            if not self._ml_models_trained and len(self.market_data_history) > 0:
                await self._train_ml_models()
            
            # Квантовый анализ паттернов
            quantum_patterns = await self._perform_quantum_analysis(market_data)
            
            # Поиск арбитражных возможностей
            opportunities = await self._detect_arbitrage_opportunities(
                market_data, quantum_patterns
            )
            
            # Создание планов исполнения
            execution_plans = await self._create_execution_plans(opportunities)
            
            # Генерация сигналов
            signals = await self._generate_arbitrage_signals(execution_plans)
            
            # Обновление статистики
            processing_time = (time.time() - start_time) * 1000
            await self._update_statistics(opportunities, processing_time)
            
            logger.info(
                f"Quantum arbitrage analysis: {len(opportunities)} opportunities, "
                f"{len(signals)} signals, {processing_time:.2f}ms"
            )
            
        except Exception as e:
            logger.error(f"Error in quantum arbitrage analysis: {e}")
        
        return signals
    
    async def _update_market_history(self, market_data: Dict[str, MarketData]) -> None:
        """Обновление исторических рыночных данных."""
        for symbol, data in market_data.items():
            self.market_data_history[symbol].append({
                'timestamp': data.timestamp,
                'price': float(data.close.amount),
                'volume': float(data.volume.amount) if data.volume else 0.0,
                'high': float(data.high.amount),
                'low': float(data.low.amount),
                'volatility': self._calculate_volatility(data)
            })
    
    def _calculate_volatility(self, market_data: MarketData) -> float:
        """Расчёт волатильности."""
        price_range = float(market_data.high.amount - market_data.low.amount)
        mid_price = float((market_data.high.amount + market_data.low.amount) / 2)
        return price_range / mid_price if mid_price > 0 else 0.0
    
    async def _train_ml_models(self) -> None:
        """Обучение моделей машинного обучения."""
        try:
            # Подготовка данных для обучения
            training_data = []
            target_prices = []
            
            for symbol, history in self.market_data_history.items():
                if len(history) < 50:  # Минимум данных для обучения
                    continue
                
                history_list = list(history)
                for i in range(10, len(history_list) - 1):  # Окно в 10 периодов
                    # Признаки
                    features = []
                    for j in range(i-10, i):
                        data_point = history_list[j]
                        features.extend([
                            data_point['price'],
                            data_point['volume'],
                            data_point['volatility']
                        ])
                    
                    # Целевая переменная (цена следующего периода)
                    target = history_list[i+1]['price']
                    
                    training_data.append(features)
                    target_prices.append(target)
            
            if len(training_data) < 100:  # Минимум для обучения
                return
            
            # Нормализация данных
            X = np.array(training_data)
            y = np.array(target_prices)
            
            X_scaled = self.scaler.fit_transform(X)
            
            # Разделение на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Обучение предиктора цен
            self.price_predictor.fit(X_train, y_train)
            
            # Обучение детектора аномалий
            self.anomaly_detector.fit(X_train)
            
            # Валидация моделей
            y_pred = self.price_predictor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"ML models trained: MSE={mse:.6f}, R²={r2:.3f}")
            self._ml_models_trained = True
            
        except Exception as e:
            logger.error(f"Error training ML models: {e}")
    
    async def _perform_quantum_analysis(
        self, 
        market_data: Dict[str, MarketData]
    ) -> List[MultidimensionalPattern]:
        """Квантовый анализ рыночных данных."""
        patterns = []
        
        try:
            # Подготовка данных для квантового анализа
            for symbol, data in market_data.items():
                history = list(self.market_data_history[symbol])
                if len(history) < 20:
                    continue
                
                # Извлечение временных рядов
                prices = np.array([h['price'] for h in history])
                volumes = np.array([h['volume'] for h in history])
                timestamps = np.array([h['timestamp'].timestamp() for h in history])
                
                # Дополнительные признаки
                additional_features = {
                    'volatility': np.array([h['volatility'] for h in history]),
                    'price_momentum': np.diff(prices, prepend=prices[0]),
                    'volume_momentum': np.diff(volumes, prepend=volumes[0])
                }
                
                # Квантовый анализ
                symbol_patterns = await self.quantum_analyzer.analyze_market_data(
                    prices, volumes, timestamps, additional_features
                )
                
                # Фильтрация паттернов по релевантности для арбитража
                relevant_patterns = [
                    pattern for pattern in symbol_patterns
                    if self._is_arbitrage_relevant_pattern(pattern)
                ]
                
                patterns.extend(relevant_patterns)
                
                # Сохранение паттернов в анализатор
                for pattern in relevant_patterns:
                    await self.quantum_analyzer.save_pattern_to_database(pattern)
        
        except Exception as e:
            logger.error(f"Error in quantum analysis: {e}")
        
        return patterns
    
    def _is_arbitrage_relevant_pattern(self, pattern: MultidimensionalPattern) -> bool:
        """Проверка релевантности паттерна для арбитража."""
        # Паттерны с высокой сложностью и доверием
        if pattern.complexity_score > 0.6 and pattern.confidence > 0.7:
            return True
        
        # Аномальные паттерны
        if pattern.metadata.get('pattern_type') == 'quantum_anomaly':
            return True
        
        # Периодические паттерны с короткими периодами
        if (pattern.metadata.get('pattern_type') == 'quantum_periodicity' and 
            pattern.metadata.get('period', float('inf')) < 300):  # Менее 5 минут
            return True
        
        # Тренды с высокой значимостью
        if (pattern.metadata.get('pattern_type') == 'quantum_trend' and 
            pattern.metadata.get('r_squared', 0) > 0.8):
            return True
        
        return False
    
    async def _detect_arbitrage_opportunities(
        self,
        market_data: Dict[str, MarketData],
        quantum_patterns: List[MultidimensionalPattern]
    ) -> List[ArbitrageOpportunity]:
        """Обнаружение арбитражных возможностей."""
        opportunities = []
        
        # Пространственный арбитраж
        spatial_opportunities = await self._detect_spatial_arbitrage(market_data)
        opportunities.extend(spatial_opportunities)
        
        # Статистический арбитраж на основе квантовых паттернов
        statistical_opportunities = await self._detect_statistical_arbitrage(
            market_data, quantum_patterns
        )
        opportunities.extend(statistical_opportunities)
        
        # Треугольный арбитраж
        triangular_opportunities = await self._detect_triangular_arbitrage(market_data)
        opportunities.extend(triangular_opportunities)
        
        # Временной арбитраж на основе предсказаний ML
        if self._ml_models_trained:
            temporal_opportunities = await self._detect_temporal_arbitrage(market_data)
            opportunities.extend(temporal_opportunities)
        
        # Фильтрация по критериям риска и прибыльности
        filtered_opportunities = [
            opp for opp in opportunities
            if self._validate_opportunity(opp)
        ]
        
        return filtered_opportunities[:self.max_concurrent_opportunities]
    
    async def _detect_spatial_arbitrage(
        self, 
        market_data: Dict[str, MarketData]
    ) -> List[ArbitrageOpportunity]:
        """Обнаружение пространственного арбитража между биржами."""
        opportunities = []
        
        # Группировка по торговым парам
        pair_groups = defaultdict(list)
        for symbol, data in market_data.items():
            # Предполагаем формат symbol: EXCHANGE:PAIR
            if ':' in symbol:
                exchange, pair = symbol.split(':', 1)
                pair_groups[pair].append((exchange, symbol, data))
        
        # Поиск арбитража между биржами
        for pair, exchanges_data in pair_groups.items():
            if len(exchanges_data) < 2:
                continue
            
            # Сортировка по цене
            exchanges_data.sort(key=lambda x: x[2].close.amount)
            
            lowest_exchange, lowest_symbol, lowest_data = exchanges_data[0]
            highest_exchange, highest_symbol, highest_data = exchanges_data[-1]
            
            # Расчёт потенциальной прибыли
            price_diff = highest_data.close.amount - lowest_data.close.amount
            profit_percentage = price_diff / lowest_data.close.amount
            
            if profit_percentage > self.min_profit_threshold:
                # Оценка рисков
                risk_level = self._assess_spatial_arbitrage_risk(
                    lowest_data, highest_data, profit_percentage
                )
                
                # Расчёт доверительной оценки
                confidence = self._calculate_spatial_arbitrage_confidence(
                    exchanges_data, profit_percentage
                )
                
                # Оценка временного окна
                execution_window = self._estimate_spatial_execution_window(
                    lowest_exchange, highest_exchange
                )
                
                opportunity = ArbitrageOpportunity(
                    opportunity_id=f"spatial_{pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    arbitrage_type=ArbitrageType.SPATIAL,
                    trading_pairs=[
                        self._create_trading_pair_from_symbol(lowest_symbol),
                        self._create_trading_pair_from_symbol(highest_symbol)
                    ],
                    expected_profit=price_diff,
                    risk_level=risk_level,
                    confidence_score=confidence,
                    execution_window=execution_window,
                    market_data={
                        'buy_exchange': lowest_data,
                        'sell_exchange': highest_data
                    },
                    quantum_patterns=[],
                    metadata={
                        'arbitrage_type': 'spatial',
                        'buy_exchange': lowest_exchange,
                        'sell_exchange': highest_exchange,
                        'profit_percentage': float(profit_percentage)
                    }
                )
                
                opportunities.append(opportunity)
        
        return opportunities
    
    async def _detect_statistical_arbitrage(
        self,
        market_data: Dict[str, MarketData],
        quantum_patterns: List[MultidimensionalPattern]
    ) -> List[ArbitrageOpportunity]:
        """Обнаружение статистического арбитража."""
        opportunities = []
        
        # Анализ корреляций между активами
        symbols = list(market_data.keys())
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols[i+1:], i+1):
                # Расчёт исторической корреляции
                correlation = await self._calculate_correlation(symbol1, symbol2)
                
                if abs(correlation) > 0.7:  # Высокая корреляция
                    # Поиск отклонений от нормальной корреляции
                    current_spread = self._calculate_current_spread(
                        market_data[symbol1], market_data[symbol2]
                    )
                    
                    historical_spread = await self._calculate_historical_spread(
                        symbol1, symbol2
                    )
                    
                    # Z-score отклонения
                    if len(historical_spread) > 20:
                        spread_mean = np.mean(historical_spread)
                        spread_std = np.std(historical_spread)
                        z_score = (current_spread - spread_mean) / (spread_std + 1e-8)
                        
                        if abs(z_score) > 2.0:  # Значимое отклонение
                            # Поиск релевантных квантовых паттернов
                            relevant_patterns = [
                                p for p in quantum_patterns
                                if symbol1 in str(p.pattern_id) or symbol2 in str(p.pattern_id)
                            ]
                            
                            # Определение направления арбитража
                            if z_score > 2.0:  # Спред слишком широк
                                expected_profit = abs(current_spread - spread_mean) / 2
                            else:  # Спред слишком узок
                                expected_profit = abs(spread_mean - current_spread) / 2
                            
                            risk_level = self._assess_statistical_arbitrage_risk(
                                z_score, correlation, relevant_patterns
                            )
                            
                            confidence = min(abs(z_score) / 3.0, 1.0)  # Нормализация к [0,1]
                            
                            opportunity = ArbitrageOpportunity(
                                opportunity_id=f"statistical_{symbol1}_{symbol2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                arbitrage_type=ArbitrageType.STATISTICAL,
                                trading_pairs=[
                                    self._create_trading_pair_from_symbol(symbol1),
                                    self._create_trading_pair_from_symbol(symbol2)
                                ],
                                expected_profit=Decimal(str(expected_profit)),
                                risk_level=risk_level,
                                confidence_score=confidence,
                                execution_window=timedelta(minutes=30),  # Статистический арбитраж - более долгий
                                market_data={
                                    'asset1': market_data[symbol1],
                                    'asset2': market_data[symbol2]
                                },
                                quantum_patterns=relevant_patterns,
                                metadata={
                                    'arbitrage_type': 'statistical',
                                    'correlation': correlation,
                                    'z_score': z_score,
                                    'current_spread': current_spread,
                                    'historical_spread_mean': spread_mean,
                                    'direction': 'long_spread' if z_score < -2 else 'short_spread'
                                }
                            )
                            
                            opportunities.append(opportunity)
        
        return opportunities
    
    async def _detect_triangular_arbitrage(
        self, 
        market_data: Dict[str, MarketData]
    ) -> List[ArbitrageOpportunity]:
        """Обнаружение треугольного арбитража."""
        opportunities = []
        
        # Извлечение валютных пар
        currency_pairs = {}
        for symbol, data in market_data.items():
            # Предполагаем формат BASEQUOTE
            if len(symbol) >= 6:
                base = symbol[:3]
                quote = symbol[3:6]
                currency_pairs[(base, quote)] = (symbol, data)
        
        # Поиск треугольных циклов
        currencies = set()
        for base, quote in currency_pairs.keys():
            currencies.add(base)
            currencies.add(quote)
        
        for base_currency in currencies:
            for intermediate_currency in currencies:
                if intermediate_currency == base_currency:
                    continue
                    
                for quote_currency in currencies:
                    if quote_currency in [base_currency, intermediate_currency]:
                        continue
                    
                    # Проверяем наличие всех необходимых пар
                    pair1 = (base_currency, intermediate_currency)  # A/B
                    pair2 = (intermediate_currency, quote_currency)  # B/C
                    pair3 = (quote_currency, base_currency)  # C/A
                    
                    # Альтернативные направления
                    pair1_rev = (intermediate_currency, base_currency)
                    pair2_rev = (quote_currency, intermediate_currency)
                    pair3_rev = (base_currency, quote_currency)
                    
                    # Находим доступные пары
                    available_pairs = []
                    rates = []
                    
                    # Пара 1
                    if pair1 in currency_pairs:
                        available_pairs.append(pair1)
                        rates.append(float(currency_pairs[pair1][1].close.amount))
                    elif pair1_rev in currency_pairs:
                        available_pairs.append(pair1_rev)
                        rates.append(1.0 / float(currency_pairs[pair1_rev][1].close.amount))
                    else:
                        continue
                    
                    # Пара 2
                    if pair2 in currency_pairs:
                        available_pairs.append(pair2)
                        rates.append(float(currency_pairs[pair2][1].close.amount))
                    elif pair2_rev in currency_pairs:
                        available_pairs.append(pair2_rev)
                        rates.append(1.0 / float(currency_pairs[pair2_rev][1].close.amount))
                    else:
                        continue
                    
                    # Пара 3
                    if pair3 in currency_pairs:
                        available_pairs.append(pair3)
                        rates.append(float(currency_pairs[pair3][1].close.amount))
                    elif pair3_rev in currency_pairs:
                        available_pairs.append(pair3_rev)
                        rates.append(1.0 / float(currency_pairs[pair3_rev][1].close.amount))
                    else:
                        continue
                    
                    if len(rates) == 3:
                        # Расчёт треугольного арбитража
                        # Конечная сумма = начальная_сумма * rate1 * rate2 * rate3
                        final_amount = rates[0] * rates[1] * rates[2]
                        profit = final_amount - 1.0  # Начинаем с 1 единицы базовой валюты
                        
                        if profit > self.min_profit_threshold:
                            risk_level = self._assess_triangular_arbitrage_risk(
                                rates, profit
                            )
                            
                            confidence = self._calculate_triangular_arbitrage_confidence(
                                available_pairs, rates
                            )
                            
                            opportunity = ArbitrageOpportunity(
                                opportunity_id=f"triangular_{base_currency}_{intermediate_currency}_{quote_currency}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                arbitrage_type=ArbitrageType.TRIANGULAR,
                                trading_pairs=[
                                    self._create_trading_pair_from_currencies(pair[0], pair[1])
                                    for pair in available_pairs
                                ],
                                expected_profit=Decimal(str(profit)),
                                risk_level=risk_level,
                                confidence_score=confidence,
                                execution_window=timedelta(seconds=30),  # Треугольный арбитраж быстрый
                                market_data={
                                    f'pair_{i}': currency_pairs[available_pairs[i]][1]
                                    for i in range(len(available_pairs))
                                },
                                quantum_patterns=[],
                                metadata={
                                    'arbitrage_type': 'triangular',
                                    'currencies': [base_currency, intermediate_currency, quote_currency],
                                    'exchange_rates': rates,
                                    'final_multiplier': final_amount
                                }
                            )
                            
                            opportunities.append(opportunity)
        
        return opportunities
    
    async def _detect_temporal_arbitrage(
        self, 
        market_data: Dict[str, MarketData]
    ) -> List[ArbitrageOpportunity]:
        """Обнаружение временного арбитража с ML предсказаниями."""
        opportunities = []
        
        if not self._ml_models_trained:
            return opportunities
        
        for symbol, data in market_data.items():
            history = list(self.market_data_history[symbol])
            if len(history) < 20:
                continue
            
            try:
                # Подготовка признаков для предсказания
                features = []
                for i in range(max(0, len(history) - 10), len(history)):
                    data_point = history[i]
                    features.extend([
                        data_point['price'],
                        data_point['volume'],
                        data_point['volatility']
                    ])
                
                # Дополнение до нужного размера
                while len(features) < 30:  # 10 периодов * 3 признака
                    features.append(0.0)
                
                features_array = np.array(features).reshape(1, -1)
                features_scaled = self.scaler.transform(features_array)
                
                # Предсказание будущей цены
                predicted_price = self.price_predictor.predict(features_scaled)[0]
                current_price = float(data.close.amount)
                
                # Проверка на аномалию
                anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
                is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
                
                # Расчёт ожидаемой прибыли
                price_change = predicted_price - current_price
                profit_percentage = price_change / current_price
                
                # Проверка условий арбитража
                if abs(profit_percentage) > self.min_profit_threshold and is_anomaly:
                    risk_level = self._assess_temporal_arbitrage_risk(
                        profit_percentage, history
                    )
                    
                    confidence = self._calculate_temporal_arbitrage_confidence(
                        profit_percentage
                    )
                    
                    # Оценка временного окна на основе волатильности
                    recent_volatility = np.mean([h['volatility'] for h in history[-5:]])
                    execution_window = timedelta(
                        minutes=max(5, min(60, 30 / (recent_volatility + 0.01)))
                    )
                    
                    opportunity = ArbitrageOpportunity(
                        opportunity_id=f"temporal_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        arbitrage_type=ArbitrageType.TEMPORAL,
                        trading_pairs=[self._create_trading_pair_from_symbol(symbol)],
                        expected_profit=Decimal(str(abs(price_change))),
                        risk_level=risk_level,
                        confidence_score=confidence,
                        execution_window=execution_window,
                        market_data={'current': data},
                        quantum_patterns=[],
                        metadata={
                            'arbitrage_type': 'temporal',
                            'predicted_price': predicted_price,
                            'current_price': current_price,
                            'direction': 'buy' if price_change > 0 else 'sell',
                            'anomaly_score': anomaly_score,
                            'profit_percentage': float(profit_percentage)
                        }
                    )
                    
                    opportunities.append(opportunity)
                    
            except Exception as e:
                logger.debug(f"Error in temporal arbitrage for {symbol}: {e}")
        
        return opportunities
    
    def _validate_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """Валидация арбитражной возможности."""
        # Проверка минимальной прибыли
        if opportunity.calculate_profit_potential() < self.min_profit_threshold:
            return False
        
        # Проверка уровня риска
        risk_levels = [RiskLevel.VERY_LOW, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.EXTREME]
        if risk_levels.index(opportunity.risk_level) > risk_levels.index(self.max_risk_level):
            return False
        
        # Проверка временного окна
        if opportunity.execution_window.total_seconds() < 5:
            return False
        
        # Проверка доверительной оценки
        if opportunity.confidence_score < 0.5:
            return False
        
        return True
    
    # Helper methods for risk assessment and calculations
    def _assess_spatial_arbitrage_risk(
        self, 
        low_data: MarketData, 
        high_data: MarketData, 
        profit_percentage: float
    ) -> RiskLevel:
        """Оценка риска пространственного арбитража."""
        # Базовый риск на основе прибыли
        if profit_percentage > 0.05:  # > 5%
            return RiskLevel.HIGH
        elif profit_percentage > 0.02:  # > 2%
            return RiskLevel.MEDIUM
        elif profit_percentage > 0.005:  # > 0.5%
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW
    
    def _calculate_spatial_arbitrage_confidence(
        self, 
        exchanges_data: List[Tuple[str, str, MarketData]], 
        profit_percentage: float
    ) -> float:
        """Расчёт доверительной оценки пространственного арбитража."""
        # Базовая уверенность на основе количества бирж
        base_confidence = min(len(exchanges_data) / 5.0, 1.0)
        
        # Корректировка на объём торгов
        volumes = [float(data[2].volume.amount) if data[2].volume else 0.0 for data in exchanges_data]
        volume_factor = min(np.mean(volumes) / 10000.0, 1.0)  # Нормализация к объёму
        
        # Корректировка на размер прибыли
        profit_factor = min(profit_percentage / 0.01, 1.0)  # Нормализация к 1%
        
        return (base_confidence + volume_factor + profit_factor) / 3.0
    
    def _estimate_spatial_execution_window(self, exchange1: str, exchange2: str) -> timedelta:
        """Оценка временного окна для пространственного арбитража."""
        # Базовое время - 2 минуты
        base_time = timedelta(minutes=2)
        
        # Корректировка на тип биржи (можно расширить)
        known_fast_exchanges = ['binance', 'ftx', 'coinbase']
        
        if exchange1.lower() in known_fast_exchanges and exchange2.lower() in known_fast_exchanges:
            return base_time
        else:
            return base_time * 2  # Медленные биржи - больше времени
    
    async def _calculate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Расчёт корреляции между двумя активами."""
        if (symbol1, symbol2) in self.price_correlations:
            return self.price_correlations[(symbol1, symbol2)]
        
        history1 = list(self.market_data_history[symbol1])
        history2 = list(self.market_data_history[symbol2])
        
        if len(history1) < 20 or len(history2) < 20:
            return 0.0
        
        # Синхронизация данных по времени
        min_length = min(len(history1), len(history2))
        prices1 = np.array([h['price'] for h in history1[-min_length:]])
        prices2 = np.array([h['price'] for h in history2[-min_length:]])
        
        correlation = np.corrcoef(prices1, prices2)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # Кеширование
        self.price_correlations[(symbol1, symbol2)] = correlation
        self.price_correlations[(symbol2, symbol1)] = correlation
        
        return correlation
    
    def _calculate_current_spread(self, data1: MarketData, data2: MarketData) -> float:
        """Расчёт текущего спреда между активами."""
        price1 = float(data1.close.amount)
        price2 = float(data2.close.amount)
        return price1 - price2
    
    async def _calculate_historical_spread(self, symbol1: str, symbol2: str) -> List[float]:
        """Расчёт исторического спреда."""
        history1 = list(self.market_data_history[symbol1])
        history2 = list(self.market_data_history[symbol2])
        
        min_length = min(len(history1), len(history2))
        if min_length < 10:
            return []
        
        spreads = []
        for i in range(min_length):
            spread = history1[i]['price'] - history2[i]['price']
            spreads.append(spread)
        
        return spreads
    
    def _assess_statistical_arbitrage_risk(
        self, 
        z_score: float, 
        correlation: float, 
        patterns: List[MultidimensionalPattern]
    ) -> RiskLevel:
        """Оценка риска статистического арбитража."""
        base_risk = RiskLevel.MEDIUM
        
        # Корректировка на Z-score
        if abs(z_score) > 3.0:
            base_risk = RiskLevel.LOW  # Сильное отклонение - низкий риск
        elif abs(z_score) < 2.0:
            base_risk = RiskLevel.HIGH  # Слабое отклонение - высокий риск
        
        # Корректировка на корреляцию
        if abs(correlation) > 0.9:
            return RiskLevel.VERY_LOW  # Очень высокая корреляция
        elif abs(correlation) < 0.5:
            return RiskLevel.EXTREME  # Низкая корреляция
        
        return base_risk
    
    # Additional helper methods...
    def _create_trading_pair_from_symbol(self, symbol: str) -> TradingPair:
        """Создание торговой пары из символа."""
        # Упрощённая реализация
        return TradingPair(
            base_currency=Currency.BTC,  # Заглушка
            quote_currency=Currency.USD,
            symbol=Symbol(symbol)
        )
    
    def _create_trading_pair_from_currencies(self, base: str, quote: str) -> TradingPair:
        """Создание торговой пары из валют."""
        return TradingPair(
            base_currency=Currency(base) if hasattr(Currency, base) else Currency.BTC,
            quote_currency=Currency(quote) if hasattr(Currency, quote) else Currency.USD,
            symbol=Symbol(f"{base}{quote}")
        )
    
    async def _create_execution_plans(
        self, 
        opportunities: List[ArbitrageOpportunity]
    ) -> List[ArbitrageExecutionPlan]:
        """Создание планов исполнения арбитража."""
        plans = []
        
        for opportunity in opportunities:
            # Создание детального плана исполнения
            execution_steps = await self._generate_execution_steps(opportunity)
            
            # Оценка требований к капиталу
            capital_requirement = self._estimate_capital_requirement(opportunity)
            
            # Оценка времени исполнения
            execution_time = self._estimate_execution_time(opportunity)
            
            # Расчёт метрик риска
            risk_metrics = self._calculate_risk_metrics(opportunity)
            
            plan = ArbitrageExecutionPlan(
                opportunity=opportunity,
                execution_steps=execution_steps,
                estimated_execution_time=execution_time,
                capital_requirement=capital_requirement,
                expected_return=opportunity.expected_profit,
                risk_metrics=risk_metrics
            )
            
            plans.append(plan)
        
        return plans
    
    async def _generate_execution_steps(
        self, 
        opportunity: ArbitrageOpportunity
    ) -> List[Dict[str, Any]]:
        """Генерация шагов исполнения арбитража."""
        steps = []
        
        if opportunity.arbitrage_type == ArbitrageType.SPATIAL:
            steps = [
                {
                    'action': 'buy',
                    'exchange': opportunity.metadata['buy_exchange'],
                    'symbol': opportunity.trading_pairs[0].symbol,
                    'price': opportunity.market_data['buy_exchange'].close.amount
                },
                {
                    'action': 'sell',
                    'exchange': opportunity.metadata['sell_exchange'],
                    'symbol': opportunity.trading_pairs[1].symbol,
                    'price': opportunity.market_data['sell_exchange'].close.amount
                }
            ]
        elif opportunity.arbitrage_type == ArbitrageType.TRIANGULAR:
            currencies = opportunity.metadata['currencies']
            rates = opportunity.metadata['exchange_rates']
            
            for i, (rate, pair) in enumerate(zip(rates, opportunity.trading_pairs)):
                steps.append({
                    'action': 'exchange',
                    'step': i + 1,
                    'from_currency': currencies[i],
                    'to_currency': currencies[(i + 1) % len(currencies)],
                    'rate': rate,
                    'symbol': pair.symbol
                })
        
        # Добавляем временные метки и валидацию
        for i, step in enumerate(steps):
            step.update({
                'step_order': i + 1,
                'estimated_duration': timedelta(seconds=10 + i * 5),
                'validation_required': True
            })
        
        return steps
    
    def _estimate_capital_requirement(self, opportunity: ArbitrageOpportunity) -> Decimal:
        """Оценка требований к капиталу."""
        # Базовое требование
        base_amount = Decimal('1000.0')  # $1000 минимум
        
        # Корректировка на тип арбитража
        if opportunity.arbitrage_type == ArbitrageType.TRIANGULAR:
            return base_amount * 2  # Треугольный арбитраж требует больше капитала
        elif opportunity.arbitrage_type == ArbitrageType.STATISTICAL:
            return base_amount * 3  # Статистический арбитраж - ещё больше
        
        return base_amount
    
    def _estimate_execution_time(self, opportunity: ArbitrageOpportunity) -> timedelta:
        """Оценка времени исполнения."""
        base_time = timedelta(seconds=30)
        
        # Корректировка на тип арбитража
        type_multipliers = {
            ArbitrageType.SPATIAL: 1.0,
            ArbitrageType.TRIANGULAR: 0.5,  # Быстрый
            ArbitrageType.STATISTICAL: 2.0,  # Медленный
            ArbitrageType.TEMPORAL: 1.5
        }
        
        multiplier = type_multipliers.get(opportunity.arbitrage_type, 1.0)
        return base_time * multiplier
    
    def _calculate_risk_metrics(self, opportunity: ArbitrageOpportunity) -> Dict[str, float]:
        """Расчёт метрик риска."""
        return {
            'value_at_risk_95': 0.05,  # 5% VaR
            'maximum_drawdown': 0.03,  # 3% максимальная просадка
            'sharpe_ratio': 2.0,  # Хороший Sharpe ratio
            'execution_risk': 0.02,  # 2% риск исполнения
            'market_impact': 0.01  # 1% рыночное воздействие
        }
    
    async def _generate_arbitrage_signals(
        self, 
        execution_plans: List[ArbitrageExecutionPlan]
    ) -> List[Signal]:
        """Генерация торговых сигналов из планов исполнения."""
        signals = []
        
        for plan in execution_plans:
            opportunity = plan.opportunity
            
            # Проверка возможности исполнения
            if not plan.validate_execution_feasibility(Decimal('10000')):  # $10k доступно
                continue
            
            # Генерация сигнала для каждого шага
            for step in plan.execution_steps:
                signal_type = SignalType.BUY if step['action'] == 'buy' else SignalType.SELL
                
                signal = Signal(
                    strategy_id=self.strategy_id.value if hasattr(self.strategy_id, 'value') else self.strategy_id,
                    trading_pair=str(opportunity.trading_pairs[0].symbol),  # Строка
                    signal_type=signal_type,
                    confidence=Decimal(str(opportunity.confidence_score)),
                    price=Money(Decimal(str(step.get('price', 0))), Currency.USD),
                    quantity=Decimal('1.0'),  # Упрощение
                    metadata={
                        'arbitrage_type': opportunity.arbitrage_type.value,
                        'opportunity_id': opportunity.opportunity_id,
                        'execution_step': step,
                        'risk_level': opportunity.risk_level.value,
                        'expected_profit': str(opportunity.expected_profit)
                    }
                )
                
                signals.append(signal)
        
        return signals
    
    async def _update_statistics(
        self, 
        opportunities: List[ArbitrageOpportunity], 
        processing_time: float
    ) -> None:
        """Обновление статистики стратегии."""
        self.strategy_statistics['total_opportunities_found'] += len(opportunities)
        
        for opportunity in opportunities:
            self.strategy_statistics['opportunity_types'][opportunity.arbitrage_type.value] += 1
            self.strategy_statistics['risk_distribution'][opportunity.risk_level.value] += 1
            self.strategy_statistics['quantum_patterns_utilized'] += len(opportunity.quantum_patterns)
        
        # Обновление активных возможностей
        for opportunity in opportunities:
            self.active_opportunities[opportunity.opportunity_id] = opportunity
    
    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Получение статистики стратегии."""
        return {
            **self.strategy_statistics,
            'active_opportunities_count': len(self.active_opportunities),
            'ml_models_trained': self._ml_models_trained,
            'quantum_analyzer_stats': self.quantum_analyzer.get_analysis_statistics()
        }
    
    def _assess_triangular_arbitrage_risk(self, rates: List[float], profit: float) -> RiskLevel:
        """Оценка риска треугольного арбитража."""
        # Расчёт на основе волатильности курсов и размера прибыли
        rate_volatility = np.std(rates) if len(rates) > 1 else 0.0
        
        if profit > 0.01 and rate_volatility < 0.05:  # 1% прибыль, низкая волатильность
            return RiskLevel.LOW
        elif profit > 0.005:  # 0.5% прибыль
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH
    
    def _calculate_triangular_arbitrage_confidence(self, rates: List[float]) -> float:
        """Расчёт уверенности в треугольном арбитраже."""
        if len(rates) < 3:
            return 0.0
        
        # Расчёт на основе стабильности курсов
        rate_stability = 1.0 - min(np.std(rates), 0.5)  # Ограничиваем максимальную нестабильность
        final_amount = rates[0] * rates[1] * rates[2]
        profit_score = min(final_amount - 1.0, 0.1) * 10  # Нормализуем прибыль
        
        return min(rate_stability * profit_score, 1.0)
    
    def _assess_temporal_arbitrage_risk(self, price_difference: float, time_window: int) -> RiskLevel:
        """Оценка риска временного арбитража."""
        # Больший временной интервал = выше риск
        time_risk = min(time_window / 3600, 1.0)  # Нормализуем к часам
        price_risk = min(abs(price_difference) / 0.1, 1.0)  # Нормализуем к 10%
        
        combined_risk = (time_risk + price_risk) / 2
        
        if combined_risk < 0.3:
            return RiskLevel.LOW
        elif combined_risk < 0.6:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH
    
    def _calculate_temporal_arbitrage_confidence(self, price_difference: float) -> float:
        """Расчёт уверенности во временном арбитраже."""
        # Чем больше разность цен, тем выше уверенность
        return min(abs(price_difference) / 0.05, 1.0)  # Нормализуем к 5%