"""
Примеры использования улучшенных типов для замены избыточного использования Any.
"""

from typing import Dict, List, Optional, TypedDict
from datetime import datetime
from decimal import Decimal

from domain.types.enhanced_types import (
    FlexibleValue, JsonValue, ConfigValue, CacheValue,
    EntityId, ResourceId, TaskId, Timestamp, Amount, Percentage,
    MarketDataType, OrderBookDataType, TradeDataType,
    OrderDataType, PositionDataType, BalanceDataType,
    TechnicalIndicatorType, PatternDataType, SignalDataType, PredictionDataType,
    DatabaseConfigType, CacheConfigType, ExchangeConfigType, StrategyConfigType,
    HealthStatusType, ErrorInfoType, PerformanceMetricsType,
    QueryFilterType, QueryOptionsType, BulkOperationResultType,
    AgentContextType, AgentResponseType,
    ValidationRuleType, ValidationContextType,
    CacheEntryType, CacheStatsType,
    EventDataType, EventHandlerType,
    StrategyDataType, StrategyResultType,
    ModelDataType, TrainingDataType, PredictionResultType,
    MetricDataType, AlertDataType, LogEntryType,
    SecurityEventType, AuthenticationDataType,
    OperationResult, ValidationResult, ProcessingResult
)


# ============================================================================
# ТИПИЗИРОВАННЫЕ СЛОВАРИ ДЛЯ СООТВЕТСТВИЯ ПРОТОКОЛАМ
# ============================================================================

class MarketDataDict(TypedDict):
    symbol: str
    price: Amount
    volume: Amount
    timestamp: Timestamp
    bid: Amount
    ask: Amount
    high: Amount
    low: Amount


class OrderBookDataDict(TypedDict):
    symbol: str
    timestamp: Timestamp
    bids: List[Dict[str, Amount]]
    asks: List[Dict[str, Amount]]


class TradeDataDict(TypedDict):
    id: str
    symbol: str
    price: Amount
    quantity: Amount
    side: str
    timestamp: Timestamp


class OrderDataDict(TypedDict):
    id: str
    symbol: str
    side: str
    type: str
    quantity: Amount
    price: Optional[Amount]
    status: str
    timestamp: Timestamp


class PositionDataDict(TypedDict):
    symbol: str
    side: str
    quantity: Amount
    entry_price: Amount
    current_price: Amount
    unrealized_pnl: Amount
    realized_pnl: Amount


class BalanceDataDict(TypedDict):
    currency: str
    available: Amount
    locked: Amount
    total: Amount


class TechnicalIndicatorDict(TypedDict):
    name: str
    value: float
    timestamp: Timestamp
    parameters: Dict[str, FlexibleValue]


class PatternDataDict(TypedDict):
    name: str
    confidence: Percentage
    start_time: Timestamp
    end_time: Timestamp
    parameters: Dict[str, FlexibleValue]


class SignalDataDict(TypedDict):
    type: str
    strength: Percentage
    direction: str
    timestamp: Timestamp
    metadata: Dict[str, FlexibleValue]


class PredictionDataDict(TypedDict):
    model_id: str
    target: str
    value: float
    confidence: Percentage
    timestamp: Timestamp
    features: Dict[str, float]


class DatabaseConfigDict(TypedDict):
    host: str
    port: int
    database: str
    username: str
    password: str
    pool_size: int
    max_overflow: int


class CacheConfigDict(TypedDict):
    type: str
    host: Optional[str]
    port: Optional[int]
    ttl: int
    max_size: int
    eviction_policy: str


class ExchangeConfigDict(TypedDict):
    name: str
    api_key: str
    api_secret: str
    testnet: bool
    rate_limit: int


class StrategyConfigDict(TypedDict):
    name: str
    enabled: bool
    parameters: Dict[str, FlexibleValue]
    risk_limits: Dict[str, float]


class HealthStatusDict(TypedDict):
    status: str
    timestamp: Timestamp
    components: Dict[str, str]
    metrics: Dict[str, float]


class ErrorInfoDict(TypedDict):
    code: str
    message: str
    details: Optional[str]
    timestamp: Timestamp
    context: Dict[str, FlexibleValue]


class PerformanceMetricsDict(TypedDict):
    operation: str
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    timestamp: Timestamp


class BulkOperationResultDict(TypedDict):
    success_count: int
    error_count: int
    errors: List[Dict[str, FlexibleValue]]
    processed_ids: List[str]


class AgentContextDict(TypedDict):
    agent_id: str
    session_id: str
    timestamp: Timestamp
    data: Dict[str, FlexibleValue]
    state: Dict[str, FlexibleValue]


class AgentResponseDict(TypedDict):
    success: bool
    data: Optional[FlexibleValue]
    error: Optional[str]
    metadata: Dict[str, FlexibleValue]


class ValidationRuleDict(TypedDict):
    field: str
    rule_type: str
    parameters: Dict[str, FlexibleValue]
    message: str


class ValidationContextDict(TypedDict):
    entity_type: str
    operation: str
    data: Dict[str, FlexibleValue]
    user_id: Optional[str]


class PredictionResultDict(TypedDict):
    model_id: EntityId
    prediction: float
    confidence: Percentage
    features: Dict[str, float]
    timestamp: Timestamp


# ============================================================================
# ПРИМЕРЫ ЗАМЕНЫ Any НА СПЕЦИФИЧНЫЕ ТИПЫ
# ============================================================================

class MarketDataService:
    """Сервис для работы с рыночными данными с улучшенной типизацией."""

    def __init__(self, config: ConfigValue):
        """Инициализация сервиса."""
        self.config = config
        self._cache: Dict[str, CacheValue] = {}

    async def get_market_data(self, symbol: str) -> MarketDataDict:
        """Получение рыночных данных."""
        # Вместо Dict[str, Any] используем MarketDataDict
        return MarketDataDict(
            symbol=symbol,
            price=Decimal('50000.00'),
            volume=Decimal('100.5'),
            timestamp=datetime.now(),
            bid=Decimal('49999.00'),
            ask=Decimal('50001.00'),
            high=Decimal('51000.00'),
            low=Decimal('49000.00')
        )

    async def get_order_book(self, symbol: str) -> OrderBookDataDict:
        """Получение ордербука."""
        # Вместо Dict[str, Any] используем OrderBookDataDict
        return OrderBookDataDict(
            symbol=symbol,
            timestamp=datetime.now(),
            bids=[
                {'price': Decimal('49999.00'), 'quantity': Decimal('10.0')},
                {'price': Decimal('49998.00'), 'quantity': Decimal('15.0')}
            ],
            asks=[
                {'price': Decimal('50001.00'), 'quantity': Decimal('8.0')},
                {'price': Decimal('50002.00'), 'quantity': Decimal('12.0')}
            ]
        )

    async def process_trade(self, trade_data: TradeDataDict) -> OperationResult:
        """Обработка сделки."""
        try:
            # Вместо Any используем TradeDataDict
            processed_trade = {
                'id': trade_data['id'],
                'symbol': trade_data['symbol'],
                'price': trade_data['price'],
                'quantity': trade_data['quantity'],
                'side': trade_data['side'],
                'timestamp': trade_data['timestamp']
            }
            
            return OperationResult(
                success=True,
                message="Trade processed successfully",
                data=processed_trade
            )
        except Exception as e:
            return OperationResult(
                success=False,
                message="Trade processing failed",
                error=str(e)
            )


class TradingService:
    """Сервис для торговых операций с улучшенной типизацией."""

    async def create_order(self, order_data: OrderDataDict) -> OperationResult:
        """Создание ордера."""
        # Вместо Dict[str, Any] используем OrderDataDict
        try:
            order = {
                'id': order_data['id'],
                'symbol': order_data['symbol'],
                'side': order_data['side'],
                'type': order_data['type'],
                'quantity': order_data['quantity'],
                'price': order_data['price'],
                'status': 'PENDING',
                'timestamp': datetime.now()
            }
            
            return OperationResult(
                success=True,
                message="Order created successfully",
                data=order
            )
        except Exception as e:
            return OperationResult(
                success=False,
                message="Order creation failed",
                error=str(e)
            )

    async def get_position(self, symbol: str) -> PositionDataDict:
        """Получение позиции."""
        # Вместо Dict[str, Any] используем PositionDataDict
        return PositionDataDict(
            symbol=symbol,
            side='LONG',
            quantity=Decimal('1.5'),
            entry_price=Decimal('50000.00'),
            current_price=Decimal('51000.00'),
            unrealized_pnl=Decimal('1500.00'),
            realized_pnl=Decimal('0.00')
        )

    async def get_balance(self, currency: str) -> BalanceDataDict:
        """Получение баланса."""
        # Вместо Dict[str, Any] используем BalanceDataDict
        return BalanceDataDict(
            currency=currency,
            available=Decimal('10000.00'),
            locked=Decimal('500.00'),
            total=Decimal('10500.00')
        )


class AnalyticsService:
    """Сервис для аналитики с улучшенной типизацией."""

    async def calculate_technical_indicator(
        self, 
        data: List[float], 
        indicator_type: str
    ) -> TechnicalIndicatorDict:
        """Расчет технического индикатора."""
        # Вместо Dict[str, Any] используем TechnicalIndicatorDict
        return TechnicalIndicatorDict(
            name=indicator_type,
            value=0.75,
            timestamp=datetime.now(),
            parameters={
                'period': 14,
                'data_length': len(data)
            }
        )

    async def detect_pattern(self, price_data: List[float]) -> PatternDataDict:
        """Обнаружение паттерна."""
        # Вместо Dict[str, Any] используем PatternDataDict
        return PatternDataDict(
            name='DOUBLE_TOP',
            confidence=0.85,
            start_time=datetime.now(),
            end_time=datetime.now(),
            parameters={
                'price_level': 50000.0,
                'volume_threshold': 1000.0
            }
        )

    async def generate_signal(
        self, 
        indicators: List[TechnicalIndicatorDict],
        patterns: List[PatternDataDict]
    ) -> SignalDataDict:
        """Генерация сигнала."""
        # Вместо Dict[str, Any] используем SignalDataDict
        return SignalDataDict(
            type='BUY',
            strength=0.8,
            direction='BULLISH',
            timestamp=datetime.now(),
            metadata={
                'indicators_count': len(indicators),
                'patterns_count': len(patterns)
            }
        )

    async def make_prediction(
        self, 
        model_id: str, 
        features: Dict[str, float]
    ) -> PredictionDataDict:
        """Создание предсказания."""
        # Вместо Dict[str, Any] используем PredictionDataDict
        return PredictionDataDict(
            model_id=model_id,
            target='price',
            value=52000.0,
            confidence=0.75,
            timestamp=datetime.now(),
            features=features
        )


class ConfigurationService:
    """Сервис для работы с конфигурацией с улучшенной типизацией."""

    def get_database_config(self) -> DatabaseConfigDict:
        """Получение конфигурации базы данных."""
        # Вместо Dict[str, Any] используем DatabaseConfigDict
        return DatabaseConfigDict(
            host='localhost',
            port=5432,
            database='trading_db',
            username='user',
            password='password',
            pool_size=10,
            max_overflow=20
        )

    def get_cache_config(self) -> CacheConfigDict:
        """Получение конфигурации кэша."""
        # Вместо Dict[str, Any] используем CacheConfigDict
        return CacheConfigDict(
            type='redis',
            host='localhost',
            port=6379,
            ttl=300,
            max_size=1000,
            eviction_policy='lru'
        )

    def get_exchange_config(self) -> ExchangeConfigDict:
        """Получение конфигурации биржи."""
        # Вместо Dict[str, Any] используем ExchangeConfigDict
        return ExchangeConfigDict(
            name='binance',
            api_key='your_api_key',
            api_secret='your_api_secret',
            testnet=True,
            rate_limit=100
        )

    def get_strategy_config(self) -> StrategyConfigDict:
        """Получение конфигурации стратегии."""
        # Вместо Dict[str, Any] используем StrategyConfigDict
        return StrategyConfigDict(
            name='mean_reversion',
            enabled=True,
            parameters={
                'lookback_period': 20,
                'threshold': 0.02
            },
            risk_limits={
                'max_position_size': 0.1,
                'max_drawdown': 0.05
            }
        )


class MonitoringService:
    """Сервис для мониторинга с улучшенной типизацией."""

    def get_health_status(self) -> HealthStatusDict:
        """Получение статуса здоровья."""
        # Вместо Dict[str, Any] используем HealthStatusDict
        return HealthStatusDict(
            status='HEALTHY',
            timestamp=datetime.now(),
            components={
                'database': 'OK',
                'cache': 'OK',
                'exchange': 'OK'
            },
            metrics={
                'response_time': 0.05,
                'error_rate': 0.001
            }
        )

    def log_error(self, error: Exception, context: str) -> ErrorInfoDict:
        """Логирование ошибки."""
        # Вместо Dict[str, Any] используем ErrorInfoDict
        return ErrorInfoDict(
            code=type(error).__name__,
            message=str(error),
            details=None,
            timestamp=datetime.now(),
            context={
                'context': context,
                'module': self.__class__.__name__
            }
        )

    def record_performance_metrics(
        self, 
        operation: str, 
        duration: float
    ) -> PerformanceMetricsDict:
        """Запись метрик производительности."""
        # Вместо Dict[str, Any] используем PerformanceMetricsDict
        return PerformanceMetricsDict(
            operation=operation,
            duration_ms=duration * 1000,
            memory_mb=128.5,
            cpu_percent=15.2,
            timestamp=datetime.now()
        )


class RepositoryService:
    """Сервис для работы с репозиториями с улучшенной типизацией."""

    async def query_entities(
        self, 
        filters: List[QueryFilterType],
        options: Optional[QueryOptionsType] = None
    ) -> List[Dict[str, FlexibleValue]]:
        """Запрос сущностей."""
        # Вместо List[Dict[str, Any]] используем List[Dict[str, FlexibleValue]]
        return [
            {
                'id': '123',
                'name': 'Test Entity',
                'status': 'active',
                'created_at': datetime.now().isoformat()  # Преобразуем в строку
            }
        ]

    async def bulk_operation(
        self, 
        entities: List[Dict[str, FlexibleValue]]
    ) -> BulkOperationResultDict:
        """Массовая операция."""
        # Вместо Dict[str, Any] используем BulkOperationResultDict
        return BulkOperationResultDict(
            success_count=len(entities),
            error_count=0,
            errors=[],
            processed_ids=[str(i) for i in range(len(entities))]
        )


class AgentService:
    """Сервис для работы с агентами с улучшенной типизацией."""

    async def create_agent_context(
        self, 
        agent_id: str, 
        session_id: str
    ) -> AgentContextDict:
        """Создание контекста агента."""
        # Вместо Dict[str, Any] используем AgentContextDict
        return AgentContextDict(
            agent_id=agent_id,
            session_id=session_id,
            timestamp=datetime.now(),
            data={
                'market_state': 'bullish',
                'risk_level': 'medium'
            },
            state={
                'current_strategy': 'mean_reversion',
                'position_count': 3
            }
        )

    async def process_agent_response(
        self, 
        context: AgentContextDict
    ) -> AgentResponseDict:
        """Обработка ответа агента."""
        # Вместо Dict[str, Any] используем AgentResponseDict
        return AgentResponseDict(
            success=True,
            data={
                'action': 'BUY',
                'symbol': 'BTC/USD',
                'quantity': 0.1
            },
            error=None,
            metadata={
                'confidence': 0.85,
                'reasoning': 'Strong bullish signal detected'
            }
        )


class ValidationService:
    """Сервис для валидации с улучшенной типизацией."""

    def create_validation_rules(self) -> List[ValidationRuleDict]:
        """Создание правил валидации."""
        # Вместо List[Dict[str, Any]] используем List[ValidationRuleDict]
        return [
            ValidationRuleDict(
                field='symbol',
                rule_type='not_empty',
                parameters={},
                message='Symbol cannot be empty'
            ),
            ValidationRuleDict(
                field='price',
                rule_type='positive',
                parameters={},
                message='Price must be positive'
            )
        ]

    def create_validation_context(
        self, 
        entity_type: str, 
        operation: str
    ) -> ValidationContextDict:
        """Создание контекста валидации."""
        # Вместо Dict[str, Any] используем ValidationContextDict
        return ValidationContextDict(
            entity_type=entity_type,
            operation=operation,
            data={},
            user_id='user123'
        )

    async def validate_data(
        self, 
        data: Dict[str, FlexibleValue],
        rules: List[ValidationRuleDict]
    ) -> ValidationResult:
        """Валидация данных."""
        errors: List[str] = []
        warnings: List[str] = []
        
        for rule in rules:
            field = rule['field']
            if field not in data:
                errors.append(rule['message'])
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


# ============================================================================
# ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ В ФУНКЦИЯХ
# ============================================================================

async def process_market_data(
    market_data: MarketDataDict,
    indicators: List[TechnicalIndicatorDict]
) -> ProcessingResult:
    """Обработка рыночных данных."""
    try:
        processed_count = 1
        error_count = 0
        errors: List[str] = []
        
        # Обработка данных
        processed_data = {
            'symbol': market_data['symbol'],
            'price': market_data['price'],
            'indicators_count': len(indicators)
        }
        
        return ProcessingResult(
            success=True,
            processed_count=processed_count,
            error_count=error_count,
            errors=errors,
            data=processed_data
        )
    except Exception as e:
        return ProcessingResult(
            success=False,
            processed_count=0,
            error_count=1,
            errors=[str(e)]
        )


async def handle_trading_event(event: EventDataType) -> None:
    """Обработка торгового события."""
    if isinstance(event, dict):
        event_type = event.get('event_type')
        data = event.get('data')
        
        if event_type == 'ORDER_CREATED' and isinstance(data, dict):
            # Обработка создания ордера
            order_data: OrderDataDict = data
            print(f"Order created: {order_data['symbol']}")
        
        elif event_type == 'TRADE_EXECUTED' and isinstance(data, dict):
            # Обработка исполнения сделки
            trade_data: TradeDataDict = data
            print(f"Trade executed: {trade_data['symbol']} at {trade_data['price']}")


async def analyze_strategy_performance(
    strategy_data: StrategyDataType,
    results: List[StrategyResultType]
) -> Dict[str, float]:
    """Анализ производительности стратегии."""
    # Вместо Dict[str, Any] возвращаем Dict[str, float]
    return {
        'total_trades': len(results),
        'win_rate': 0.65,
        'profit_factor': 1.8,
        'max_drawdown': 0.12,
        'sharpe_ratio': 1.2
    }


async def train_model(
    model_data: ModelDataType,
    training_data: List[TrainingDataType]
) -> PredictionResultDict:
    """Обучение модели."""
    # Вместо Dict[str, Any] используем PredictionResultDict
    model_id = model_data.get('id', 'unknown') if isinstance(model_data, dict) else 'unknown'
    return PredictionResultDict(
        model_id=model_id,
        prediction=0.75,
        confidence=0.85,
        features={
            'feature1': 0.5,
            'feature2': 0.3,
            'feature3': 0.2
        },
        timestamp=datetime.now()
    )


# ============================================================================
# ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ В КЛАССАХ
# ============================================================================

class ImprovedTradingSystem:
    """Улучшенная торговая система с правильной типизацией."""

    def __init__(self, config: ConfigValue):
        self.config = config
        self.market_service = MarketDataService(config)
        self.trading_service = TradingService()
        self.analytics_service = AnalyticsService()
        self.monitoring_service = MonitoringService()

    async def execute_trading_cycle(self) -> OperationResult:
        """Выполнение торгового цикла."""
        try:
            # Получение рыночных данных
            market_data = await self.market_service.get_market_data('BTC/USD')
            
            # Анализ данных
            indicator = await self.analytics_service.calculate_technical_indicator(
                [50000, 50100, 50200], 'SMA'
            )
            
            # Генерация сигнала
            signal = await self.analytics_service.generate_signal([indicator], [])
            
            # Создание ордера при наличии сигнала
            if signal['type'] == 'BUY' and signal['strength'] > 0.7:
                order_data = OrderDataDict(
                    id='order123',
                    symbol='BTC/USD',
                    side='BUY',
                    type='MARKET',
                    quantity=Decimal('0.1'),
                    price=None,
                    status='PENDING',
                    timestamp=datetime.now()
                )
                
                result = await self.trading_service.create_order(order_data)
                return result
            
            return OperationResult(
                success=True,
                message="No trading signal generated",
                data=None
            )
            
        except Exception as e:
            return OperationResult(
                success=False,
                message="Trading cycle failed",
                error=str(e)
            )

    def get_system_health(self) -> HealthStatusDict:
        """Получение здоровья системы."""
        return self.monitoring_service.get_health_status()


# ============================================================================
# ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ В ТЕСТАХ
# ============================================================================

def test_market_data_service():
    """Тест сервиса рыночных данных."""
    service = MarketDataService({})
    
    # Тест с правильными типами
    market_data = MarketDataDict(
        symbol='BTC/USD',
        price=Decimal('50000.00'),
        volume=Decimal('100.5'),
        timestamp=datetime.now(),
        bid=Decimal('49999.00'),
        ask=Decimal('50001.00'),
        high=Decimal('51000.00'),
        low=Decimal('49000.00')
    )
    
    # Проверка типов
    assert isinstance(market_data['symbol'], str)
    assert isinstance(market_data['price'], Decimal)
    assert isinstance(market_data['volume'], Decimal)
    assert isinstance(market_data['timestamp'], datetime)


def test_validation_service():
    """Тест сервиса валидации."""
    service = ValidationService()
    
    # Создание правил валидации
    rules = service.create_validation_rules()
    
    # Проверка типов
    for rule in rules:
        assert isinstance(rule['field'], str)
        assert isinstance(rule['rule_type'], str)
        assert isinstance(rule['parameters'], dict)
        assert isinstance(rule['message'], str)


if __name__ == "__main__":
    # Запуск тестов
    test_market_data_service()
    test_validation_service()
    print("All tests passed!") 