# Отчет по нереализованным функциям в Application слое
## Общая статистика
- Всего найдено проблем: 960
### Распределение по типам:
- TODO/FIXME: 3
- Возврат по умолчанию: 351
- Не реализовано: 47
- Подозрительная реализация: 510
- Упрощенная реализация: 49

## Детальный список проблем:

### 📁 application\__init__.py
Найдено проблем: 4

#### Строка 28: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    "RiskService",
    "MLService",
    # Use Cases
    "DefaultTradingPairManagementUseCase",
    "DefaultOrderManagementUseCase",
    "DefaultPositionManagementUseCase",
    "DefaultRiskManagementUseCase",
```

#### Строка 29: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    "MLService",
    # Use Cases
    "DefaultTradingPairManagementUseCase",
    "DefaultOrderManagementUseCase",
    "DefaultPositionManagementUseCase",
    "DefaultRiskManagementUseCase",
    "TradingOrchestratorUseCase",
```

#### Строка 30: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    # Use Cases
    "DefaultTradingPairManagementUseCase",
    "DefaultOrderManagementUseCase",
    "DefaultPositionManagementUseCase",
    "DefaultRiskManagementUseCase",
    "TradingOrchestratorUseCase",
    # Analysis
```

#### Строка 31: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    "DefaultTradingPairManagementUseCase",
    "DefaultOrderManagementUseCase",
    "DefaultPositionManagementUseCase",
    "DefaultRiskManagementUseCase",
    "TradingOrchestratorUseCase",
    # Analysis
    "EntanglementMonitor",
```

### 📁 application\analysis\entanglement_monitor.py
Найдено проблем: 25

#### Строка 377: get_entanglement_history
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python
                        continue
            return history[-limit:]  # Возвращаем последние записи
        except FileNotFoundError:
            return []

    async def analyze_entanglement(self, symbol1: str, symbol2: str, timeframe: str) -> Dict[str, Any]:
        """Анализ запутанности между двумя символами."""
```

#### Строка 381: analyze_entanglement
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python

    async def analyze_entanglement(self, symbol1: str, symbol2: str, timeframe: str) -> Dict[str, Any]:
        """Анализ запутанности между двумя символами."""
        # Заглушка для совместимости с тестами
        return {
            "entanglement_score": 0.7,
            "correlation": 0.8,
```

#### Строка 391: analyze_correlations
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python

    async def analyze_correlations(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Анализ корреляций между символами."""
        # Заглушка для совместимости с тестами
        return {
            "correlation_matrix": {},
            "strong_correlations": [],
```

#### Строка 399: calculate_correlation
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
            "strong_correlations": [],
            "weak_correlations": [],
            "correlation_clusters": []
        }

    def calculate_correlation(self, prices1: List[Any], prices2: List[Any]) -> float:
        """Расчет корреляции между двумя рядами цен."""
        # Заглушка для совместимости с тестами
        return 0.8

    def calculate_phase_shift(self, prices1: List[Any], prices2: List[Any]) -> float:
```

#### Строка 401: calculate_correlation
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python

    def calculate_correlation(self, prices1: List[Any], prices2: List[Any]) -> float:
        """Расчет корреляции между двумя рядами цен."""
        # Заглушка для совместимости с тестами
        return 0.8

    def calculate_phase_shift(self, prices1: List[Any], prices2: List[Any]) -> float:
```

#### Строка 402: calculate_correlation
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return 0'
**Код:**
```python
    def calculate_correlation(self, prices1: List[Any], prices2: List[Any]) -> float:
        """Расчет корреляции между двумя рядами цен."""
        # Заглушка для совместимости с тестами
        return 0.8

    def calculate_phase_shift(self, prices1: List[Any], prices2: List[Any]) -> float:
        """Расчет фазового сдвига между двумя рядами цен."""
```

#### Строка 404: calculate_phase_shift
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
    def calculate_correlation(self, prices1: List[Any], prices2: List[Any]) -> float:
        """Расчет корреляции между двумя рядами цен."""
        # Заглушка для совместимости с тестами
        return 0.8

    def calculate_phase_shift(self, prices1: List[Any], prices2: List[Any]) -> float:
        """Расчет фазового сдвига между двумя рядами цен."""
        # Заглушка для совместимости с тестами
        return 0.1

    def calculate_entanglement_score(self, correlation: float, phase_shift: float, volatility_ratio: float) -> float:
```

#### Строка 406: calculate_phase_shift
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python

    def calculate_phase_shift(self, prices1: List[Any], prices2: List[Any]) -> float:
        """Расчет фазового сдвига между двумя рядами цен."""
        # Заглушка для совместимости с тестами
        return 0.1

    def calculate_entanglement_score(self, correlation: float, phase_shift: float, volatility_ratio: float) -> float:
```

#### Строка 407: calculate_phase_shift
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return 0'
**Код:**
```python
    def calculate_phase_shift(self, prices1: List[Any], prices2: List[Any]) -> float:
        """Расчет фазового сдвига между двумя рядами цен."""
        # Заглушка для совместимости с тестами
        return 0.1

    def calculate_entanglement_score(self, correlation: float, phase_shift: float, volatility_ratio: float) -> float:
        """Расчет оценки запутанности."""
```

#### Строка 409: calculate_entanglement_score
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
    def calculate_phase_shift(self, prices1: List[Any], prices2: List[Any]) -> float:
        """Расчет фазового сдвига между двумя рядами цен."""
        # Заглушка для совместимости с тестами
        return 0.1

    def calculate_entanglement_score(self, correlation: float, phase_shift: float, volatility_ratio: float) -> float:
        """Расчет оценки запутанности."""
        # Заглушка для совместимости с тестами
        return 0.7

    def detect_correlation_clusters(self, correlation_matrix: Dict[str, Dict[str, float]], threshold: float = 0.6) -> List[List[str]]:
```

#### Строка 411: calculate_entanglement_score
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python

    def calculate_entanglement_score(self, correlation: float, phase_shift: float, volatility_ratio: float) -> float:
        """Расчет оценки запутанности."""
        # Заглушка для совместимости с тестами
        return 0.7

    def detect_correlation_clusters(self, correlation_matrix: Dict[str, Dict[str, float]], threshold: float = 0.6) -> List[List[str]]:
```

#### Строка 412: calculate_entanglement_score
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return 0'
**Код:**
```python
    def calculate_entanglement_score(self, correlation: float, phase_shift: float, volatility_ratio: float) -> float:
        """Расчет оценки запутанности."""
        # Заглушка для совместимости с тестами
        return 0.7

    def detect_correlation_clusters(self, correlation_matrix: Dict[str, Dict[str, float]], threshold: float = 0.6) -> List[List[str]]:
        """Обнаружение кластеров корреляции."""
```

#### Строка 416: detect_correlation_clusters
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python

    def detect_correlation_clusters(self, correlation_matrix: Dict[str, Dict[str, float]], threshold: float = 0.6) -> List[List[str]]:
        """Обнаружение кластеров корреляции."""
        # Заглушка для совместимости с тестами
        return [["BTC/USD", "ETH/USD"]]

    def calculate_volatility_ratio(self, prices1: List[Any], prices2: List[Any]) -> float:
```

#### Строка 421: calculate_volatility_ratio
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python

    def calculate_volatility_ratio(self, prices1: List[Any], prices2: List[Any]) -> float:
        """Расчет отношения волатильности."""
        # Заглушка для совместимости с тестами
        return 1.2

    async def monitor_changes(self, symbol1: str, symbol2: str, timeframe: str, window_size: int) -> Dict[str, Any]:
```

#### Строка 426: monitor_changes
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python

    async def monitor_changes(self, symbol1: str, symbol2: str, timeframe: str, window_size: int) -> Dict[str, Any]:
        """Мониторинг изменений запутанности."""
        # Заглушка для совместимости с тестами
        return {
            "current_entanglement": 0.7,
            "entanglement_trend": "stable",
```

#### Строка 436: detect_breakdown
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python

    def detect_breakdown(self, historical_scores: List[float], threshold: float = 0.5) -> bool:
        """Обнаружение разрыва запутанности."""
        # Заглушка для совместимости с тестами
        if len(historical_scores) < 2:
            return False
        return historical_scores[-1] < threshold
```

#### Строка 438: detect_breakdown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
        """Обнаружение разрыва запутанности."""
        # Заглушка для совместимости с тестами
        if len(historical_scores) < 2:
            return False
        return historical_scores[-1] < threshold

    def calculate_trend(self, historical_scores: List[float]) -> str:
```

#### Строка 443: calculate_trend
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python

    def calculate_trend(self, historical_scores: List[float]) -> str:
        """Расчет тренда запутанности."""
        # Заглушка для совместимости с тестами
        if len(historical_scores) < 2:
            return "stable"
        if historical_scores[-1] > historical_scores[0]:
```

#### Строка 453: validate_data
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
        elif historical_scores[-1] < historical_scores[0]:
            return "decreasing"
        else:
            return "stable"

    def validate_data(self, data: Any) -> bool:
        """Валидация входных данных."""
        # Заглушка для совместимости с тестами
        if data is None:
            return False
        if isinstance(data, list) and len(data) == 0:
```

#### Строка 455: validate_data
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python

    def validate_data(self, data: Any) -> bool:
        """Валидация входных данных."""
        # Заглушка для совместимости с тестами
        if data is None:
            return False
        if isinstance(data, list) and len(data) == 0:
```

#### Строка 457: validate_data
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
        """Валидация входных данных."""
        # Заглушка для совместимости с тестами
        if data is None:
            return False
        if isinstance(data, list) and len(data) == 0:
            return False
        if not isinstance(data, list):
```

#### Строка 459: validate_data
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
        if data is None:
            return False
        if isinstance(data, list) and len(data) == 0:
            return False
        if not isinstance(data, list):
            return False
        return True
```

#### Строка 461: validate_data
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
        if isinstance(data, list) and len(data) == 0:
            return False
        if not isinstance(data, list):
            return False
        return True

    def calculate_confidence_interval(self, prices1: List[Any], prices2: List[Any], confidence_level: float) -> Dict[str, float]:
```

#### Строка 462: validate_data
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            return False
        if not isinstance(data, list):
            return False
        return True

    def calculate_confidence_interval(self, prices1: List[Any], prices2: List[Any], confidence_level: float) -> Dict[str, float]:
        """Расчет доверительного интервала."""
```

#### Строка 466: calculate_confidence_interval
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python

    def calculate_confidence_interval(self, prices1: List[Any], prices2: List[Any], confidence_level: float) -> Dict[str, float]:
        """Расчет доверительного интервала."""
        # Заглушка для совместимости с тестами
        return {
            "lower_bound": 0.6,
            "upper_bound": 0.9
```

### 📁 application\di_container_refactored.py
Найдено проблем: 25

#### Строка 34: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
from application.use_cases.manage_orders import (
    CreateOrderRequest,
    CreateOrderResponse,
    DefaultOrderManagementUseCase,
    OrderManagementUseCase,
)
from application.use_cases.manage_positions import DefaultPositionManagementUseCase
```

#### Строка 37: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    DefaultOrderManagementUseCase,
    OrderManagementUseCase,
)
from application.use_cases.manage_positions import DefaultPositionManagementUseCase
from application.use_cases.manage_risk import DefaultRiskManagementUseCase
from application.use_cases.manage_trading_pairs import (
    DefaultTradingPairManagementUseCase,
```

#### Строка 38: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    OrderManagementUseCase,
)
from application.use_cases.manage_positions import DefaultPositionManagementUseCase
from application.use_cases.manage_risk import DefaultRiskManagementUseCase
from application.use_cases.manage_trading_pairs import (
    DefaultTradingPairManagementUseCase,
)
```

#### Строка 40: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
from application.use_cases.manage_positions import DefaultPositionManagementUseCase
from application.use_cases.manage_risk import DefaultRiskManagementUseCase
from application.use_cases.manage_trading_pairs import (
    DefaultTradingPairManagementUseCase,
)

# Use Cases
```

#### Строка 45: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python

# Use Cases
from application.use_cases.trading_orchestrator.core import (
    DefaultTradingOrchestratorUseCase,
)

# Импорты модуля evolution
```

#### Строка 88: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    StrategyServiceProtocol,
    TradingServiceProtocol,
)
from domain.services.correlation_chain import DefaultCorrelationChain
from domain.services.market_metrics import MarketMetricsService
from domain.services.pattern_discovery import PatternConfig, PatternDiscovery
from domain.services.risk_analysis import (
```

#### Строка 92: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
from domain.services.market_metrics import MarketMetricsService
from domain.services.pattern_discovery import PatternConfig, PatternDiscovery
from domain.services.risk_analysis import (
    DefaultRiskAnalysisService,
    RiskAnalysisService,
)
from domain.services.signal_service import DefaultSignalService, SignalService
```

#### Строка 95: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    DefaultRiskAnalysisService,
    RiskAnalysisService,
)
from domain.services.signal_service import DefaultSignalService, SignalService

# Domain Services
from domain.services.strategy_service import DefaultStrategyService, StrategyService
```

#### Строка 98: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
from domain.services.signal_service import DefaultSignalService, SignalService

# Domain Services
from domain.services.strategy_service import DefaultStrategyService, StrategyService
from domain.services.technical_analysis import (
    DefaultTechnicalAnalysisService,
    ITechnicalAnalysisService,
```

#### Строка 100: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
# Domain Services
from domain.services.strategy_service import DefaultStrategyService, StrategyService
from domain.services.technical_analysis import (
    DefaultTechnicalAnalysisService,
    ITechnicalAnalysisService,
)

```

#### Строка 239: has
**Класс:** RepositoryRegistry
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python


class RepositoryRegistry(Registry):
    pass


class ServiceRegistry(Registry):
```

#### Строка 243: has
**Класс:** RepositoryRegistry
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python


class ServiceRegistry(Registry):
    pass


class AgentRegistry(Registry):
```

#### Строка 247: has
**Класс:** RepositoryRegistry
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python


class AgentRegistry(Registry):
    pass


class UseCaseRegistry(Registry):
```

#### Строка 251: has
**Класс:** RepositoryRegistry
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python


class UseCaseRegistry(Registry):
    pass


class ConfigurationManager:
```

#### Строка 280: get_config
**Класс:** Container
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        LiquidityAnalyzer, config=config.liquidity_analyzer
    )
    ml_predictor: providers.Provider[MLPredictor] = providers.Singleton(MLPredictor, config=config.ml_predictor)
    technical_analysis_service = providers.Singleton(DefaultTechnicalAnalysisService)
    market_metrics_service = providers.Singleton(MarketMetricsService)
    correlation_chain_service = providers.Singleton(DefaultCorrelationChain)
    signal_service = providers.Singleton(DefaultSignalService)
```

#### Строка 282: get_config
**Класс:** Container
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    ml_predictor: providers.Provider[MLPredictor] = providers.Singleton(MLPredictor, config=config.ml_predictor)
    technical_analysis_service = providers.Singleton(DefaultTechnicalAnalysisService)
    market_metrics_service = providers.Singleton(MarketMetricsService)
    correlation_chain_service = providers.Singleton(DefaultCorrelationChain)
    signal_service = providers.Singleton(DefaultSignalService)
    strategy_service = providers.Singleton(DefaultStrategyService)
    pattern_discovery = providers.Singleton(
```

#### Строка 283: get_config
**Класс:** Container
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    technical_analysis_service = providers.Singleton(DefaultTechnicalAnalysisService)
    market_metrics_service = providers.Singleton(MarketMetricsService)
    correlation_chain_service = providers.Singleton(DefaultCorrelationChain)
    signal_service = providers.Singleton(DefaultSignalService)
    strategy_service = providers.Singleton(DefaultStrategyService)
    pattern_discovery = providers.Singleton(
        PatternDiscovery,
```

#### Строка 284: Unknown
**Класс:** Container
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    market_metrics_service = providers.Singleton(MarketMetricsService)
    correlation_chain_service = providers.Singleton(DefaultCorrelationChain)
    signal_service = providers.Singleton(DefaultSignalService)
    strategy_service = providers.Singleton(DefaultStrategyService)
    pattern_discovery = providers.Singleton(
        PatternDiscovery,
        config=providers.Callable(
```

#### Строка 302: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
            similarity_threshold=0.8,
        ),
    )
    risk_analysis_service = providers.Singleton(DefaultRiskAnalysisService)
    # Новые компоненты domain/strategies
    strategy_factory = providers.Singleton(get_strategy_factory)
    strategy_registry = providers.Singleton(get_strategy_registry)
```

#### Строка 398: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    )
    # Use Cases
    manage_orders_use_case = providers.Singleton(
        DefaultOrderManagementUseCase,
        order_validator=order_validator,
        trading_service=trading_service,
        market_service=market_service,
```

#### Строка 404: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        market_service=market_service,
    )
    manage_positions_use_case = providers.Singleton(
        DefaultPositionManagementUseCase,
        portfolio_service=portfolio_service,
        risk_service=risk_service,
        market_service=market_service,
```

#### Строка 410: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        market_service=market_service,
    )
    manage_risk_use_case = providers.Singleton(
        DefaultRiskManagementUseCase,
        risk_service=risk_service,
        market_service=market_service,
        risk_analysis=risk_analysis_service,
```

#### Строка 416: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        risk_analysis=risk_analysis_service,
    )
    manage_trading_pairs_use_case = providers.Singleton(
        DefaultTradingPairManagementUseCase,
        market_service=market_service,
        strategy_service=strategy_service_app,
    )
```

#### Строка 421: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        strategy_service=strategy_service_app,
    )
    trading_orchestrator_use_case = providers.Singleton(
        DefaultTradingOrchestratorUseCase,
        session_service=session_service,
        trading_service=trading_service,
        strategy_factory=strategy_factory,
```

#### Строка 465: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
        Any,
        api_key=config.bybit.api_key,
        api_secret=config.bybit.api_secret,
        testnet=config.bybit.testnet,
    )
    account_manager: providers.Provider[Any] = providers.Singleton(Any, bybit_client=bybit_client)
    # Risk and Technical Analysis Services
```

### 📁 application\entanglement\stream_manager.py
Найдено проблем: 6

#### Строка 35: Unknown
**Класс:** StreamManager
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'debug'
**Код:**
```python
        self.is_running = False
        self.monitored_symbols: set = set()
        self.entanglement_callbacks: List[Callable] = []
        self.debug_mode = False
        self._last_sequence_ids: Dict[str, int] = {}
        self.stats = {
            "total_detections": 0,
```

#### Строка 82: subscribe_symbol
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return success
        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol}: {e}")
            return False

    async def unsubscribe_symbol(self, symbol: str) -> bool:
        """Отписка от символа."""
```

#### Строка 94: unsubscribe_symbol
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return success
        except Exception as e:
            logger.error(f"Failed to unsubscribe from {symbol}: {e}")
            return False

    async def start_monitoring(self):
        """Запуск мониторинга запутанности."""
```

#### Строка 137: _handle_order_book_update
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'debug'
**Код:**
```python
                self._validate_sequence(update.symbol, update.sequence_id)

            # Логирование для отладки
            if self.debug_mode:
                bids_count = len(update.bids) if hasattr(update.bids, '__len__') and not callable(update.bids) else 0
                asks_count = len(update.asks) if hasattr(update.asks, '__len__') and not callable(update.asks) else 0
                logger.debug(
```

#### Строка 140: _handle_order_book_update
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'debug'
**Код:**
```python
            if self.debug_mode:
                bids_count = len(update.bids) if hasattr(update.bids, '__len__') and not callable(update.bids) else 0
                asks_count = len(update.asks) if hasattr(update.asks, '__len__') and not callable(update.asks) else 0
                logger.debug(
                    f"Processing order book update for {update.symbol}: {bids_count} bids, {asks_count} asks"
                )

```

#### Строка 144: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
                    f"Processing order book update for {update.symbol}: {bids_count} bids, {asks_count} asks"
                )

            # Дополнительная обработка может быть добавлена здесь
            # Например, фильтрация, нормализация, агрегация и т.д.

        except Exception as e:
```

### 📁 application\evolution\evolution_orchestrator.py
Найдено проблем: 25

#### Строка 142: _get_historical_data
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'empty'
**Код:**
```python
        for pair in trading_pairs:
            try:
                data = self.market_data_provider(pair, start_date, end_date)
                if isinstance(data, pd.DataFrame) and not data.empty:
                    data["trading_pair"] = pair
                    all_data.append(data)
            except Exception as e:
```

#### Строка 151: _get_historical_data
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'empty'
**Код:**
```python
            raise ValueError("Не удалось получить исторические данные")
        # Объединить данные
        combined_data = pd.concat(all_data, ignore_index=True)
        if not combined_data.empty:
            combined_data = combined_data.sort_values("timestamp").reset_index(drop=True)
        return combined_data

```

#### Строка 161: _evaluate_population
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'debug'
**Код:**
```python
        evaluations = {}
        for i, candidate in enumerate(self.population):
            try:
                self.logger.debug(
                    f"Оценка стратегии {i+1}/{len(self.population)}: {candidate.name}"
                )
                # Оценить стратегию
```

#### Строка 181: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'debug'
**Код:**
```python
                    )
                else:
                    candidate.update_status(EvolutionStatus.REJECTED)
                    self.logger.debug(
                        f"Стратегия {candidate.name} отклонена: {evaluation.approval_reason}"
                    )
            except Exception as e:
```

#### Строка 197: _select_best_candidates
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python
            self.evaluations[c.id] for c in self.population if c.id in self.evaluations
        ]
        if not evaluations_list:
            return []
        # Выбрать топ стратегий
        selected_count = max(1, self.context.population_size // 2)
        selected_candidates = self.strategy_selector.select_top_strategies(
```

#### Строка 280: _should_stop_evolution
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
            "best_fitness": self.stats.get("best_fitness_achieved", 0.0),
            "timestamp": datetime.now(),
        }
        self.evolution_history.append(generation_stats)

    def _should_stop_evolution(self) -> bool:
        """Проверить условия остановки эволюции."""
        # Остановка по количеству одобренных стратегий
        if len(self.approved_strategies) >= 10:
            self.logger.info("Достигнуто достаточное количество одобренных стратегий")
            return True
```

#### Строка 285: _should_stop_evolution
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
        # Остановка по количеству одобренных стратегий
        if len(self.approved_strategies) >= 10:
            self.logger.info("Достигнуто достаточное количество одобренных стратегий")
            return True
        # Остановка по отсутствию улучшений
        if len(self.evolution_history) >= 10:
            recent_generations = self.evolution_history[-10:]
```

#### Строка 292: _should_stop_evolution
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            recent_fitnesses = [g["best_fitness"] for g in recent_generations]
            if max(recent_fitnesses) - min(recent_fitnesses) < 0.01:
                self.logger.info("Нет улучшений в последних 10 поколениях")
                return True
        return False

    async def _finalize_evolution(self) -> None:
```

#### Строка 293: _should_stop_evolution
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            if max(recent_fitnesses) - min(recent_fitnesses) < 0.01:
                self.logger.info("Нет улучшений в последних 10 поколениях")
                return True
        return False

    async def _finalize_evolution(self) -> None:
        """Завершить эволюцию."""
```

#### Строка 419: pause_evolution
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
    async def pause_evolution(self) -> None:
        """Приостановить эволюцию."""
        self.logger.info("Эволюция приостановлена")
        # Здесь можно добавить логику приостановки

    async def resume_evolution(self) -> None:
        """Возобновить эволюцию."""
```

#### Строка 424: resume_evolution
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
    async def resume_evolution(self) -> None:
        """Возобновить эволюцию."""
        self.logger.info("Эволюция возобновлена")
        # Здесь можно добавить логику возобновления

    async def stop_evolution(self) -> None:
        """Остановить эволюцию."""
```

#### Строка 429: stop_evolution
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
    async def stop_evolution(self) -> None:
        """Остановить эволюцию."""
        self.logger.info("Эволюция остановлена")
        # Здесь можно добавить логику остановки

    def get_selection_statistics(self) -> Dict[str, Any]:
        """Получить статистику отбора."""
```

#### Строка 473: save_strategy_to_storage
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
        try:
            await self.strategy_storage.save_strategy_candidate(candidate)
            self.logger.info(f"Стратегия {candidate.name} сохранена в хранилище")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка сохранения стратегии {candidate.name}: {e}")
            return False
```

#### Строка 476: save_strategy_to_storage
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Ошибка сохранения стратегии {candidate.name}: {e}")
            return False

    async def load_strategies_from_storage(self) -> List[StrategyCandidate]:
        """Загрузить стратегии из хранилища."""
```

#### Строка 486: load_strategies_from_storage
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python
            return candidates
        except Exception as e:
            self.logger.error(f"Ошибка загрузки стратегий: {e}")
            return []

    async def cache_strategy_evaluation(
        self, candidate_id: UUID, evaluation: StrategyEvaluationResult
```

#### Строка 494: load_strategies_from_storage
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
        """Кэшировать оценку стратегии."""
        try:
            await self.evolution_cache.set_evaluation(candidate_id, evaluation)
            return True
        except Exception as e:
            self.logger.error(f"Ошибка кэширования оценки: {e}")
            return False
```

#### Строка 497: load_strategies_from_storage
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Ошибка кэширования оценки: {e}")
            return False

    async def get_cached_evaluation(
        self, candidate_id: UUID
```

#### Строка 507: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            return await self.evolution_cache.get_evaluation(candidate_id)
        except Exception as e:
            self.logger.error(f"Ошибка получения кэшированной оценки: {e}")
            return None

    async def create_evolution_backup(self) -> bool:
        """Создать резервную копию эволюции."""
```

#### Строка 524: create_evolution_backup
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            }
            backup_metadata = self.evolution_backup.create_backup("evolution_backup")
            self.logger.info(f"Создан бэкап эволюции: {backup_metadata}")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка создания резервной копии: {e}")
            return False
```

#### Строка 527: create_evolution_backup
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Ошибка создания резервной копии: {e}")
            return False

    async def restore_evolution_from_backup(self, backup_id: str) -> bool:
        """Восстановить эволюцию из резервной копии."""
```

#### Строка 554: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            self.stats = backup_data["stats"]
            self.current_generation = backup_data["current_generation"]
            self.logger.info("Эволюция восстановлена из резервной копии")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка восстановления из резервной копии: {e}")
            return False
```

#### Строка 557: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Ошибка восстановления из резервной копии: {e}")
            return False

    async def run_evolution_migration(self) -> bool:
        """Запустить миграцию эволюции."""
```

#### Строка 564: run_evolution_migration
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
        try:
            await self.evolution_migration.run_migration()
            self.logger.info("Миграция эволюции выполнена")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка миграции эволюции: {e}")
            return False
```

#### Строка 567: run_evolution_migration
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Ошибка миграции эволюции: {e}")
            return False

    async def get_evolution_metrics(self) -> Dict[str, Any]:
        """Получить метрики эволюции."""
```

#### Строка 586: get_evolution_metrics
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
            }
        except Exception as e:
            self.logger.error(f"Ошибка получения метрик эволюции: {e}")
            return {}

```

### 📁 application\filters\orderbook_filter.py
Найдено проблем: 3

#### Строка 163: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'debug'
**Код:**
```python
            result = self.noise_analyzer.analyze_noise(intelligence_order_book)
            # Логируем анализ если включено
            if self.config.log_analysis:
                logger.debug(
                    f"Noise analysis for {order_book.exchange}:{order_book.symbol}: "
                    f"FD={result.fractal_dimension:.3f}, "
                    f"Entropy={result.entropy:.3f}, "
```

#### Строка 442: is_order_book_filtered
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return order_book.meta.get("filtered", False)
        except Exception as e:
            logger.error(f"Error checking if order book is filtered: {e}")
            return False

    def get_noise_analysis_result(
        self, order_book: OrderBookSnapshot
```

#### Строка 452: is_order_book_filtered
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            return order_book.meta.get("noise_analysis")
        except Exception as e:
            logger.error(f"Error getting noise analysis result: {e}")
            return None

```

### 📁 application\market\mm_follow_controller.py
Найдено проблем: 21

#### Строка 152: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
                MarketMakerSymbol(symbol), order_book, trades
            )
            if not pattern:
                return None
            # Ищем похожие исторические паттерны
            features = self.pattern_classifier.extract_features(order_book, trades)
            features_dict: Dict[str, Any] = (
```

#### Строка 164: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            if not similar_patterns:
                # Сохраняем новый паттерн для будущего анализа
                await self.pattern_memory.save_pattern(symbol, pattern)
                return None
            # Выбираем лучший совпадающий паттерн
            best_match_data = similar_patterns[0]
            # Создаем MatchedPattern из данных
```

#### Строка 180: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            )
            # Проверяем критерии для генерации сигнала
            if not self._should_generate_signal(best_match):
                return None
            # Генерируем сигнал следования
            follow_signal = self._generate_follow_signal(symbol, pattern, best_match)
            # Проверяем, что сигнал был успешно создан
```

#### Строка 185: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            follow_signal = self._generate_follow_signal(symbol, pattern, best_match)
            # Проверяем, что сигнал был успешно создан
            if not follow_signal:
                return None
            # Сохраняем активный сигнал
            self.active_signals[symbol] = follow_signal
            self.signal_history.append(follow_signal)
```

#### Строка 193: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            return follow_signal
        except Exception as e:
            print(f"Error processing pattern: {e}")
            return None

    async def record_pattern_result(
        self, symbol: str, pattern: MarketMakerPattern, result: PatternResult
```

#### Строка 212: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return success
        except Exception as e:
            print(f"Error recording pattern result: {e}")
            return False

    async def get_follow_recommendations(self, symbol: str) -> List[FollowSignal]:
        """Получение рекомендаций по следованию"""
```

#### Строка 238: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python
            return recommendations
        except Exception as e:
            print(f"Error getting follow recommendations: {e}")
            return []

    async def update_follow_result(
        self, signal: FollowSignal, result: FollowResult
```

#### Строка 254: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            # Удаляем сигнал из активных
            if signal.symbol in self.active_signals:
                del self.active_signals[signal.symbol]
            return True
        except Exception as e:
            print(f"Error updating follow result: {e}")
            return False
```

#### Строка 257: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            print(f"Error updating follow result: {e}")
            return False

    def _should_generate_signal(self, matched_pattern: MatchedPattern) -> bool:
        """Проверка критериев для генерации сигнала"""
```

#### Строка 267: _should_generate_signal
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
                matched_pattern.pattern_memory.accuracy
                < self.config["min_accuracy_threshold"]
            ):
                return False
            # Проверяем минимальную уверенность
            if (
                matched_pattern.confidence_boost
```

#### Строка 273: _should_generate_signal
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
                matched_pattern.confidence_boost
                < self.config["min_confidence_threshold"]
            ):
                return False
            # Проверяем количество активных сигналов
            if len(self.active_signals) >= self.config["max_active_signals"]:
                return False
```

#### Строка 276: _should_generate_signal
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
                return False
            # Проверяем количество активных сигналов
            if len(self.active_signals) >= self.config["max_active_signals"]:
                return False
            # Проверяем силу сигнала
            if (
                matched_pattern.signal_strength < 0.01
```

#### Строка 281: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            if (
                matched_pattern.signal_strength < 0.01
            ):  # Минимальная ожидаемая доходность
                return False
            return True
        except Exception as e:
            print(f"Error checking signal criteria: {e}")
```

#### Строка 282: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
                matched_pattern.signal_strength < 0.01
            ):  # Минимальная ожидаемая доходность
                return False
            return True
        except Exception as e:
            print(f"Error checking signal criteria: {e}")
            return False
```

#### Строка 285: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            print(f"Error checking signal criteria: {e}")
            return False

    def _generate_follow_signal(
        self, symbol: str, pattern: MarketMakerPattern, matched_pattern: MatchedPattern
```

#### Строка 327: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            return signal
        except Exception as e:
            print(f"Error generating follow signal: {e}")
            return None

    def _determine_direction(self, matched_pattern: MatchedPattern) -> str:
        """Определение направления торговли"""
```

#### Строка 443: _is_signal_valid
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return signal_age <= max_age
        except Exception as e:
            print(f"Error checking signal validity: {e}")
            return False

    def _should_recommend_pattern(self, pattern_memory: Any) -> bool:
        """Проверка необходимости рекомендации паттерна"""
```

#### Строка 452: _should_recommend_pattern
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            if pattern_memory.last_seen:
                time_since_last = datetime.now() - pattern_memory.last_seen
                if time_since_last < timedelta(hours=1):  # Недавно уже был
                    return False
            # Проверяем точность и количество наблюдений
            return (
                pattern_memory.accuracy >= self.config["min_accuracy_threshold"]
```

#### Строка 460: _should_recommend_pattern
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            )
        except Exception as e:
            print(f"Error checking pattern recommendation: {e}")
            return False

    def _create_recommendation_from_history(
        self, symbol: str, pattern_memory: Any
```

#### Строка 503: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            return signal
        except Exception as e:
            print(f"Error creating recommendation from history: {e}")
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики контроллера"""
```

#### Строка 524: get_statistics
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
            }
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}

```

### 📁 application\monitoring\pattern_observer.py
Найдено проблем: 15

#### Строка 301: stop_observation
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
                observation.status = ObserverStatus.CANCELLED

                logger.info(f"Stopped observation for pattern {pattern_id}")
                return True

            return False

```

#### Строка 303: stop_observation
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
                logger.info(f"Stopped observation for pattern {pattern_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error stopping observation {pattern_id}: {e}")
```

#### Строка 307: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error stopping observation {pattern_id}: {e}")
            return False

    def get_observation_state(self, pattern_id: str) -> Optional[ObservationState]:
        """Получение состояния наблюдения."""
```

#### Строка 408: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
                current_price == observation.current_price
                and current_volume == observation.current_volume
            ):
                return False

            # Обновляем текущие значения
            observation.current_price = current_price
```

#### Строка 440: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            # Увеличиваем счетчик периодов
            observation.elapsed_periods += 1

            return True

        except Exception as e:
            logger.error(f"Error updating observation state: {e}")
```

#### Строка 444: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error updating observation state: {e}")
            return False

    def _should_complete_observation(self, observation: ObservationState) -> bool:
        """Проверка необходимости завершения наблюдения."""
```

#### Строка 451: _should_complete_observation
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
        try:
            # Завершаем по времени
            if observation.elapsed_periods >= self.config.observation_periods:
                return True

            # Завершаем по достижению порогов
            current_price_change = (
```

#### Строка 459: _should_complete_observation
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            ) / observation.start_price

            if current_price_change >= self.config.profit_threshold_percent:
                return True

            if current_price_change <= self.config.loss_threshold_percent:
                return True
```

#### Строка 462: _should_complete_observation
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
                return True

            if current_price_change <= self.config.loss_threshold_percent:
                return True

            # Завершаем по волатильности
            if observation.volatilities:
```

#### Строка 468: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            if observation.volatilities:
                current_volatility = observation.volatilities[-1]
                if current_volatility >= self.config.volatility_threshold:
                    return True

            return False

```

#### Строка 470: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
                if current_volatility >= self.config.volatility_threshold:
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking observation completion: {e}")
```

#### Строка 474: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error checking observation completion: {e}")
            return False

    def _create_pattern_outcome(
        self, observation: ObservationState, current_data: Dict[str, Any]
```

#### Строка 549: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error creating pattern outcome: {e}")
            return None

    async def _complete_observation(
        self, pattern_id: str, outcome: PatternOutcome
```

#### Строка 655: _calculate_outcome_confidence
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return 0'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error calculating outcome confidence: {e}")
            return 0.0

    def _analyze_volume_profile(self, volume_changes: List[float]) -> str:
        """Анализ профиля объема."""
```

#### Строка 742: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
        avg_volume_change = 0.0
        
        if completed_count > 0:
            # В реальной системе здесь был бы анализ завершенных наблюдений
            # Пока возвращаем базовую статистику
            avg_duration = 30.0  # Примерное значение
            avg_price_change = 0.02  # 2%
```

### 📁 application\orchestration\trading_orchestrator.py
Найдено проблем: 15

#### Строка 14: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from application.use_cases.manage_orders import DefaultOrderManagementUseCase
from application.use_cases.manage_positions import PositionManagementUseCase
from application.use_cases.manage_risk import RiskManagementUseCase
from application.use_cases.manage_trading_pairs import TradingPairManagementUseCase
```

#### Строка 64: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        sentiment_analyzer: SentimentAnalyzerProtocol,
        portfolio_manager: PortfolioManagerProtocol,
        evolution_manager: EvolutionManagerProtocol,
        order_use_case: DefaultOrderManagementUseCase,
        position_use_case: PositionManagementUseCase,
        risk_use_case: RiskManagementUseCase,
        trading_pair_use_case: TradingPairManagementUseCase,
```

#### Строка 242: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
    ) -> Dict[str, Any]:
        """Выполнение технического анализа."""
        try:
            # В реальной системе здесь был бы вызов технического анализа
            return {
                "rsi": 50.0,
                "macd": 0.0,
```

#### Строка 252: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
            }
        except Exception as e:
            self.logger.error(f"Error performing technical analysis: {e}")
            return {}

    async def _perform_risk_analysis(
        self, symbol: str, market_data: Dict[str, Any]
```

#### Строка 259: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
    ) -> Dict[str, Any]:
        """Выполнение анализа рисков."""
        try:
            # В реальной системе здесь был бы анализ рисков
            return {
                "volatility": 0.15,
                "var_95": 0.02,
```

#### Строка 268: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
            }
        except Exception as e:
            self.logger.error(f"Error performing risk analysis: {e}")
            return {}

    async def _get_sentiment_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Получение анализа настроений."""
```

#### Строка 289: _get_sentiment_analysis
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python

        except Exception as e:
            self.logger.error(f"Error getting sentiment analysis: {e}")
            return None

    async def _analyze_with_evolution_agents(
        self,
```

#### Строка 299: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
    ) -> Dict[str, Any]:
        """Анализ с использованием эволюционных агентов."""
        try:
            # В реальной системе здесь был бы анализ эволюционными агентами
            return {
                "evolution_score": 0.7,
                "adaptation_level": 0.8,
```

#### Строка 307: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
            }
        except Exception as e:
            self.logger.error(f"Error analyzing with evolution agents: {e}")
            return {}

    async def _generate_trading_signals(
        self,
```

#### Строка 321: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
        try:
            signals: List[Signal] = []

            # В реальной системе здесь была бы логика генерации сигналов
            # на основе всех анализов
            # Пока возвращаем пустой список

```

#### Строка 329: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python

        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {e}")
            return []

    async def _validate_signal(self, signal: Signal, strategy: Any) -> bool:
        """Валидация торгового сигнала."""
```

#### Строка 338: _validate_signal
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            risk_valid = await self.risk_use_case.check_order_risk(signal, None)
            if not risk_valid:
                self.logger.warning(f"Signal {signal} failed risk validation")
                return False

            # Дополнительные проверки могут быть добавлены здесь
            return True
```

#### Строка 340: _validate_signal
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
                self.logger.warning(f"Signal {signal} failed risk validation")
                return False

            # Дополнительные проверки могут быть добавлены здесь
            return True

        except Exception as e:
```

#### Строка 341: _validate_signal
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
                return False

            # Дополнительные проверки могут быть добавлены здесь
            return True

        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
```

#### Строка 345: _validate_signal
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python

        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return False

    async def _process_trading_signal(self, signal: Signal) -> None:
        """Обработка торгового сигнала."""
```

### 📁 application\prediction\combined_predictor.py
Найдено проблем: 5

#### Строка 29: Unknown
**Класс:** CombinedPredictionResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    pattern_prediction: Optional[EnhancedPredictionResult] = None

    # Сигналы сессий
    session_signals: Dict[str, SessionInfluenceSignal] = field(default_factory=dict)
    aggregated_session_signal: Optional[SessionInfluenceSignal] = None

    # Комбинированный результат
```

#### Строка 44: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    session_position_multiplier: float = 1.0

    # Метаданные
    prediction_timestamp: Timestamp = field(default_factory=Timestamp.now)
    alignment_score: float = 0.0  # Совпадение направлений паттерна и сессий

    def to_dict(self) -> Dict[str, Any]:
```

#### Строка 178: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error in combined prediction for {symbol}: {e}")
            return None

    def _combine_predictions(
        self,
```

#### Строка 269: Unknown
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'базовая'
**Код:**
```python
            result.final_return_percent = (
                session_score * 2.0
            )  # 2% при максимальном скоре
            result.final_duration_minutes = 30  # Базовая длительность

        # Применяем модификаторы сессий
        if self.config["enable_session_modifiers"] and aggregated_session_signal:
```

#### Строка 408: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            logger.error(
                f"Error getting prediction with session context for {symbol}: {e}"
            )
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики предиктора."""
```

### 📁 application\prediction\pattern_predictor.py
Найдено проблем: 10

#### Строка 133: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        }

        # Настройки по умолчанию
        self.default_config = {
            "confidence_threshold": 0.7,
            "min_similar_cases": 3,
            "max_lookback_days": 30,
```

#### Строка 143: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        }

        # Обновляем конфигурацию
        self.config = {**self.default_config, **self.config}

    def predict_pattern_outcome(
        self,
```

#### Строка 201: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error predicting pattern outcome: {e}")
            return None

    def predict_with_custom_features(
        self,
```

#### Строка 243: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error predicting with custom features: {e}")
            return None

    def _execute_prediction(
        self,
```

#### Строка 264: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
                logger.warning(
                    f"Insufficient similar cases: {len(similar_cases)} < {request.min_similar_cases}"
                )
                return None

            # Анализируем контекст рынка
            market_context = self._analyze_market_context(request, market_context)
```

#### Строка 277: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
                logger.info(
                    f"Prediction confidence {prediction.confidence:.3f} below threshold {request.confidence_threshold}"
                )
                return None

            # Оценка риска
            risk_assessment = {}
```

#### Строка 308: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error executing prediction: {e}")
            return None

    def _extract_features_from_detection(
        self, pattern_detection: PatternDetection
```

#### Строка 614: _calculate_data_quality_score
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return 0'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error calculating data quality score: {e}")
            return 0.0

    def _generate_cache_key(
        self,
```

#### Строка 656: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
                    # Удаляем устаревший кэш
                    del self.prediction_cache[cache_key]

            return None

        except Exception as e:
            logger.error(f"Error getting cached prediction: {e}")
```

#### Строка 660: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error getting cached prediction: {e}")
            return None

    def _cache_prediction(
        self, cache_key: str, result: EnhancedPredictionResult
```

### 📁 application\prediction\reversal_controller.py
Найдено проблем: 17

#### Строка 14: Unknown
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'временно'
**Код:**
```python
from domain.protocols.agent_protocols import AgentContextProtocol
from domain.types.prediction_types import OrderBookData

# from infrastructure.core.analysis.global_prediction_engine import GlobalPredictionEngine  # Временно закомментировано
from shared.logging import get_logger


```

#### Строка 148: _get_market_data
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            return data
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None

    async def _get_order_book(self, symbol: str) -> Optional[Dict]:
        """Получение данных ордербука."""
```

#### Строка 159: _get_order_book
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            return order_book
        except Exception as e:
            self.logger.error(f"Error getting order book for {symbol}: {e}")
            return None

    async def _integrate_signal(self, signal: ReversalSignal) -> None:
        """Интеграция сигнала разворота с системой."""
```

#### Строка 170: _integrate_signal
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'debug'
**Код:**
```python
            if self.config.enable_signal_filtering:
                if not self._should_accept_signal(signal):
                    self.integration_stats["signals_filtered"] = int(self.integration_stats.get("signals_filtered", 0)) + 1
                    self.logger.debug(f"Signal filtered for {signal.symbol}: {signal}")
                    return

            # Анализ согласованности с глобальными прогнозами
```

#### Строка 241: _should_accept_signal
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
        try:
            # Проверка минимальной уверенности
            if signal.confidence < 0.3:
                return False

            # Проверка минимальной силы сигнала
            if signal.signal_strength < 0.4:
```

#### Строка 245: _should_accept_signal
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python

            # Проверка минимальной силы сигнала
            if signal.signal_strength < 0.4:
                return False

            # Проверка времени жизни
            if signal.is_expired:
```

#### Строка 249: _should_accept_signal
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python

            # Проверка времени жизни
            if signal.is_expired:
                return False

            # Проверка на дублирование
            if signal.symbol in self.active_signals:
```

#### Строка 262: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
                        / signal.pivot_price.value
                        < 0.01
                    ):
                        return False

            return True

```

#### Строка 264: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
                    ):
                        return False

            return True

        except Exception as e:
            self.logger.error(f"Error in signal acceptance check: {e}")
```

#### Строка 268: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python

        except Exception as e:
            self.logger.error(f"Error in signal acceptance check: {e}")
            return False

    async def _calculate_agreement_score(self, signal: ReversalSignal) -> float:
        """Вычисление оценки согласованности с глобальными прогнозами."""
```

#### Строка 302: Unknown
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'временно'
**Код:**
```python
            #         else:
            #             agreement_factors.append(0.0)

            #     # Сравнение временного горизонта
            #     global_horizon = global_prediction.get("horizon_hours", 24)
            #     signal_horizon = signal.horizon.total_seconds() / 3600
            #     horizon_diff = abs(global_horizon - signal_horizon) / max(
```

#### Строка 329: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return 0'
**Код:**
```python
            if agreement_factors:
                return sum(agreement_factors)
            else:
                return 0.5  # Нейтральная оценка

        except Exception as e:
            self.logger.error(f"Error calculating agreement score: {e}")
```

#### Строка 333: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return 0'
**Код:**
```python

        except Exception as e:
            self.logger.error(f"Error calculating agreement score: {e}")
            return 0.5

    async def _detect_controversy(self, signal: ReversalSignal) -> List[str]:
        """Обнаружение спорных аспектов сигнала."""
```

#### Строка 379: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python

        except Exception as e:
            self.logger.error(f"Error detecting controversy: {e}")
            return []

    async def _integrate_with_global_prediction(self, signal: ReversalSignal) -> None:
        """Интеграция с глобальным прогнозированием."""
```

#### Строка 384: _integrate_with_global_prediction
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'временно'
**Код:**
```python
    async def _integrate_with_global_prediction(self, signal: ReversalSignal) -> None:
        """Интеграция с глобальным прогнозированием."""
        try:
            # Временно закомментировано из-за отсутствия GlobalPredictionEngine
            # if self.global_predictor:
            #     global_prediction = await self.global_predictor.get_prediction(signal.symbol)
            #     if global_prediction:
```

#### Строка 445: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python
                return all_signals
        except Exception as e:
            self.logger.error(f"Error getting active signals: {e}")
            return []

    async def get_signal_statistics(self) -> Dict[str, Any]:
        """Получение статистики сигналов."""
```

#### Строка 485: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python

        except Exception as e:
            self.logger.error(f"Error getting signal statistics: {e}")
            return {}

    async def stop_monitoring(self) -> None:
        """Остановка мониторинга."""
```

### 📁 application\protocols\service_protocols.py
Найдено проблем: 2

#### Строка 288: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python

    @abstractmethod
    async def evaluate_model(
        self, model_id: str, test_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Оценка модели."""
        ...
```

#### Строка 351: validate_signal
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
        ...

    @abstractmethod
    async def backtest_strategy(
        self, strategy_id: StrategyId, historical_data: List[Dict[str, Any]]
    ) -> StrategyPerformance:
        """Бэктестинг стратегии."""
```

### 📁 application\protocols\use_case_protocols.py
Найдено проблем: 4

#### Строка 318: __init__
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
        self.trading_pairs = trading_pairs


class ExecuteStrategyRequest:
    """Запрос на выполнение стратегии."""

    def __init__(
```

#### Строка 332: __init__
**Класс:** ExecuteStrategyRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
        self.parameters = parameters


class ExecuteStrategyResponse:
    """Ответ на выполнение стратегии."""

    def __init__(self, success: bool, message: str = ""):
```

#### Строка 553: get_trading_pair_by_id
**Класс:** TradingOrchestratorUseCase
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python

    @abstractmethod
    async def execute_strategy(
        self, request: ExecuteStrategyRequest
    ) -> ExecuteStrategyResponse:
        """Выполнение торговой стратегии."""
        ...
```

#### Строка 554: get_trading_pair_by_id
**Класс:** TradingOrchestratorUseCase
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
    @abstractmethod
    async def execute_strategy(
        self, request: ExecuteStrategyRequest
    ) -> ExecuteStrategyResponse:
        """Выполнение торговой стратегии."""
        ...

```

### 📁 application\risk\liquidity_gravity_monitor.py
Найдено проблем: 13

#### Строка 32: Unknown
**Класс:** RiskAssessmentResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    overall_risk: float
    recommendations: List[str]
    timestamp: Timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
```

#### Строка 149: _monitoring_cycle
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python
        """Основной цикл мониторинга."""
        try:
            for symbol in self.monitored_symbols:
                # Получаем данные ордербука (заглушка)
                order_book = await self._get_order_book_snapshot(symbol)

                if order_book:
```

#### Строка 173: Unknown
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python
    async def _get_order_book_snapshot(
        self, symbol: str
    ) -> Optional[OrderBookSnapshot]:
        """Получение снимка ордербука (заглушка)."""
        try:
            # В реальной системе здесь был бы запрос к бирже
            # Пока используем заглушку для тестирования
```

#### Строка 175: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
    ) -> Optional[OrderBookSnapshot]:
        """Получение снимка ордербука (заглушка)."""
        try:
            # В реальной системе здесь был бы запрос к бирже
            # Пока используем заглушку для тестирования
            import random

```

#### Строка 176: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'заглушку'
**Код:**
```python
        """Получение снимка ордербука (заглушка)."""
        try:
            # В реальной системе здесь был бы запрос к бирже
            # Пока используем заглушку для тестирования
            import random

            # Симулируем данные ордербука
```

#### Строка 198: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {e}")
            return None

    def _assess_risk(
        self,
```

#### Строка 395: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
        except Exception as e:
            logger.error(f"Error checking alerts for {symbol}: {e}")

    def get_latest_risk_assessment(self, symbol: str) -> Optional[RiskAssessmentResult]:
        """Получение последней оценки риска для символа."""
        try:
            if symbol in self.risk_history and self.risk_history[symbol]:
```

#### Строка 400: get_latest_risk_assessment
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
        try:
            if symbol in self.risk_history and self.risk_history[symbol]:
                return self.risk_history[symbol][-1]
            return None
        except Exception as e:
            logger.error(f"Error getting latest risk assessment for {symbol}: {e}")
            return None
```

#### Строка 402: get_latest_risk_assessment
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
                return self.risk_history[symbol][-1]
            return None
        except Exception as e:
            logger.error(f"Error getting latest risk assessment for {symbol}: {e}")
            return None

    def get_risk_history(
```

#### Строка 403: get_latest_risk_assessment
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            return None
        except Exception as e:
            logger.error(f"Error getting latest risk assessment for {symbol}: {e}")
            return None

    def get_risk_history(
        self, symbol: str, limit: int = 100
```

#### Строка 412: get_latest_risk_assessment
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python
        try:
            if symbol in self.risk_history:
                return self.risk_history[symbol][-limit:]
            return []
        except Exception as e:
            logger.error(f"Error getting risk history for {symbol}: {e}")
            return []
```

#### Строка 415: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python
            return []
        except Exception as e:
            logger.error(f"Error getting risk history for {symbol}: {e}")
            return []

    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Получение статистики мониторинга."""
```

#### Строка 438: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
            return stats
        except Exception as e:
            logger.error(f"Error getting monitoring statistics: {e}")
            return {}

```

### 📁 application\services\cache_service.py
Найдено проблем: 29

#### Строка 43: Unknown
**Класс:** CacheStrategy
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python

    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl: Optional[int] = None  # Time to live в секундах
```

#### Строка 44: Unknown
**Класс:** CacheStrategy
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl: Optional[int] = None  # Time to live в секундах
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### Строка 47: Unknown
**Класс:** CacheStrategy
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl: Optional[int] = None  # Time to live в секундах
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Проверка истечения срока действия."""
```

#### Строка 52: is_expired
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
    def is_expired(self) -> bool:
        """Проверка истечения срока действия."""
        if self.ttl is None:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl)

    def update_access(self) -> None:
```

#### Строка 70: update_access
**Класс:** CacheStats
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    miss_count: int = 0
    eviction_count: int = 0
    total_size_bytes: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def hit_rate(self) -> float:
```

#### Строка 134: get
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            return entry
        elif entry and entry.is_expired():
            await self.delete(key)
        return None

    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Установка записи."""
```

#### Строка 143: set
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            if len(self.storage) >= self.max_size:
                await self._evict_entries()
            self.storage[key] = entry
            return True
        except Exception as e:
            self.logger.error(f"Failed to set cache entry: {e}")
            return False
```

#### Строка 146: set
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Failed to set cache entry: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Удаление записи."""
```

#### Строка 153: delete
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
        try:
            if key in self.storage:
                del self.storage[key]
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete cache entry: {e}")
```

#### Строка 154: delete
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            if key in self.storage:
                del self.storage[key]
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete cache entry: {e}")
            return False
```

#### Строка 157: delete
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete cache entry: {e}")
            return False

    async def clear(self) -> bool:
        """Очистка кэша."""
```

#### Строка 163: clear
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
        """Очистка кэша."""
        try:
            self.storage.clear()
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return False
```

#### Строка 166: clear
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return False

    async def get_all_keys(self) -> List[str]:
        """Получение всех ключей."""
```

#### Строка 230: stop
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Cache service stopped")

    async def get(self, key: str) -> Optional[Any]:
```

#### Строка 242: get
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
                return entry.value
            else:
                self.stats.miss_count += 1
                return None
        except Exception as e:
            self.logger.error(f"Failed to get from cache: {e}")
            self.stats.miss_count += 1
```

#### Строка 246: get
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
        except Exception as e:
            self.logger.error(f"Failed to get from cache: {e}")
            self.stats.miss_count += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Установка значения в кэш."""
```

#### Строка 259: set
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return success
        except Exception as e:
            self.logger.error(f"Failed to set cache value: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Удаление значения из кэша."""
```

#### Строка 270: delete
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return success
        except Exception as e:
            self.logger.error(f"Failed to delete from cache: {e}")
            return False

    async def clear(self, pattern: str = "*") -> bool:
        """Очистка кэша по паттерну."""
```

#### Строка 292: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
                return deleted_count > 0
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша."""
```

#### Строка 317: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
            }
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return {}

    async def get_multi(self, keys: List[str]) -> Dict[str, Any]:
        """Получение нескольких значений."""
```

#### Строка 338: set_multi
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return success_count == len(data)
        except Exception as e:
            self.logger.error(f"Failed to set multiple cache values: {e}")
            return False

    async def delete_multi(self, keys: List[str]) -> bool:
        """Удаление нескольких значений."""
```

#### Строка 350: delete_multi
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return success_count == len(keys)
        except Exception as e:
            self.logger.error(f"Failed to delete multiple cache values: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Проверка существования ключа."""
```

#### Строка 359: exists
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return entry is not None
        except Exception as e:
            self.logger.error(f"Failed to check cache key existence: {e}")
            return False

    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Увеличение числового значения."""
```

#### Строка 377: increment
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            return new_value
        except Exception as e:
            self.logger.error(f"Failed to increment cache value: {e}")
            return None

    async def decrement(self, key: str, amount: int = 1) -> Optional[int]:
        """Уменьшение числового значения."""
```

#### Строка 423: _cleanup_expired_entries
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'debug'
**Код:**
```python
            if expired_count > 0:
                self.stats.eviction_count += expired_count
                self.stats.total_entries = await self.storage.get_size()
                self.logger.debug(f"Cleaned up {expired_count} expired cache entries")
        except Exception as e:
            self.logger.error(f"Error cleaning up expired cache entries: {e}")

```

#### Строка 430: _matches_pattern
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Проверка соответствия ключа паттерну."""
        if pattern == "*":
            return True
        # Простая реализация паттернов
        if "*" in pattern:
            # Заменяем * на .* для regex
```

#### Строка 431: _matches_pattern
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'простая'
**Код:**
```python
        """Проверка соответствия ключа паттерну."""
        if pattern == "*":
            return True
        # Простая реализация паттернов
        if "*" in pattern:
            # Заменяем * на .* для regex
            import re
```

#### Строка 442: _serialize_json
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python

    def _serialize_json(self, value: Any) -> str:
        """Сериализация в JSON."""
        return json.dumps(value, default=str)

    def _serialize_pickle(self, value: Any) -> bytes:
        """Сериализация через pickle."""
```

#### Строка 460: get_serialized_size
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return 0'
**Код:**
```python
            serialized = self._serialize_json(value)
            return len(serialized.encode("utf-8"))
        except Exception:
            return 0

```

### 📁 application\services\implementations\cache_service_impl.py
Найдено проблем: 37

#### Строка 24: __init__
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        # Кэш хранилище
        self._cache: Dict[str, Dict[str, Any]] = {}
        # Конфигурация
        self.default_ttl = self.config.get("default_ttl", 300)  # 5 минут
        self.max_size = self.config.get("max_size", 10000)
        self.cleanup_interval = self.config.get("cleanup_interval", 60)  # секунды
        self.eviction_policy = self.config.get(
```

#### Строка 53: validate_config
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    async def validate_config(self) -> bool:
        """Валидация конфигурации."""
        required_configs = [
            "default_ttl",
            "max_size",
            "cleanup_interval",
            "eviction_policy",
```

#### Строка 61: validate_config
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
        for config_key in required_configs:
            if config_key not in self.config:
                self.logger.error(f"Missing required config: {config_key}")
                return False
        return True

    async def get(self, key: str) -> Optional[Any]:
```

#### Строка 62: validate_config
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            if config_key not in self.config:
                self.logger.error(f"Missing required config: {config_key}")
                return False
        return True

    async def get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша."""
```

#### Строка 73: _get_impl
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
        try:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None
            entry = self._cache[key]
            # Проверяем истечение срока действия
            if self._is_expired(entry):
```

#### Строка 79: _get_impl
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            if self._is_expired(entry):
                await self.delete(key)
                self._stats["misses"] += 1
                return None
            # Обновляем время последнего доступа
            entry["last_accessed"] = datetime.now()
            entry["access_count"] += 1
```

#### Строка 90: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
        except Exception as e:
            self.logger.error(f"Error getting from cache: {e}")
            self._stats["misses"] += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Установка значения в кэш."""
```

#### Строка 111: _set_impl
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
                "created_at": datetime.now(),
                "last_accessed": datetime.now(),
                "access_count": 0,
                "ttl": ttl or self.default_ttl,
                "size_bytes": (
                    len(serialized_value)
                    if isinstance(serialized_value, bytes)
```

#### Строка 125: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            self._cache[key] = entry
            self._stats["total_size_bytes"] += entry["size_bytes"]
            self._stats["sets"] += 1
            return True
        except Exception as e:
            self.logger.error(f"Error setting cache value: {e}")
            return False
```

#### Строка 128: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Error setting cache value: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Удаление значения из кэша."""
```

#### Строка 143: _delete_impl
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
                self._stats["total_size_bytes"] -= old_size
                del self._cache[key]
                self._stats["deletes"] += 1
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error deleting from cache: {e}")
```

#### Строка 144: _delete_impl
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
                del self._cache[key]
                self._stats["deletes"] += 1
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error deleting from cache: {e}")
            return False
```

#### Строка 147: _delete_impl
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return False
        except Exception as e:
            self.logger.error(f"Error deleting from cache: {e}")
            return False

    async def clear(self, pattern: str = "*") -> bool:
        """Очистка кэша по паттерну."""
```

#### Строка 169: _clear_impl
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
                        keys_to_delete.append(key)
                for key in keys_to_delete:
                    await self.delete(key)
            return True
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return False
```

#### Строка 172: _clear_impl
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша."""
```

#### Строка 197: _get_stats_impl
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
                "total_size_bytes": self._stats["total_size_bytes"],
                "max_size": self.max_size,
                "eviction_policy": self.eviction_policy,
                "default_ttl": self.default_ttl,
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            }
        except Exception as e:
```

#### Строка 202: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
            }
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {}

    async def exists(self, key: str) -> bool:
        """Проверка существования ключа."""
```

#### Строка 212: _exists_impl
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
        """Реализация проверки существования ключа."""
        try:
            if key not in self._cache:
                return False
            entry = self._cache[key]
            # Проверяем истечение срока действия
            if self._is_expired(entry):
```

#### Строка 217: _exists_impl
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            # Проверяем истечение срока действия
            if self._is_expired(entry):
                await self.delete(key)
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error checking key existence: {e}")
```

#### Строка 218: _exists_impl
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            if self._is_expired(entry):
                await self.delete(key)
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error checking key existence: {e}")
            return False
```

#### Строка 221: _exists_impl
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Error checking key existence: {e}")
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        """Установка времени жизни ключа."""
```

#### Строка 232: _expire_impl
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
        try:
            if key in self._cache:
                self._cache[key]["ttl"] = ttl
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error setting key expiration: {e}")
```

#### Строка 233: _expire_impl
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            if key in self._cache:
                self._cache[key]["ttl"] = ttl
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error setting key expiration: {e}")
            return False
```

#### Строка 236: _expire_impl
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return False
        except Exception as e:
            self.logger.error(f"Error setting key expiration: {e}")
            return False

    async def get_multi(self, keys: List[str]) -> Dict[str, Any]:
        """Получение нескольких значений из кэша."""
```

#### Строка 253: _get_multi_impl
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
            return result
        except Exception as e:
            self.logger.error(f"Error getting multiple values from cache: {e}")
            return {}

    async def set_multi(self, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Установка нескольких значений в кэш."""
```

#### Строка 274: set_multi
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return success_count == len(data)
        except Exception as e:
            self.logger.error(f"Error setting multiple values in cache: {e}")
            return False

    async def delete_multi(self, keys: List[str]) -> bool:
        """Удаление нескольких значений из кэша."""
```

#### Строка 293: _delete_multi_impl
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return success_count == len(keys)
        except Exception as e:
            self.logger.error(f"Error deleting multiple values from cache: {e}")
            return False

    def generate_key(self, *args, **kwargs) -> str:
        """Генерация ключа кэша."""
```

#### Строка 316: _is_expired
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
        """Проверка истечения срока действия записи."""
        try:
            if entry["ttl"] is None:
                return False
            expiration_time = entry["created_at"] + timedelta(seconds=entry["ttl"])
            return datetime.now() > expiration_time
        except Exception as e:
```

#### Строка 321: _is_expired
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            return datetime.now() > expiration_time
        except Exception as e:
            self.logger.error(f"Error checking expiration: {e}")
            return True

    def _serialize(self, value: Any) -> tuple[Any, str]:
        """Сериализация значения."""
```

#### Строка 328: _serialize
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        try:
            # Пробуем JSON для простых типов
            try:
                json_value = json.dumps(value, default=str)
                return json_value, "json"
            except (TypeError, ValueError):
                pass
```

#### Строка 331: _serialize
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
                json_value = json.dumps(value, default=str)
                return json_value, "json"
            except (TypeError, ValueError):
                pass
            # Используем pickle для сложных объектов
            try:
                pickle_value = pickle.dumps(value)
```

#### Строка 337: _serialize
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
                pickle_value = pickle.dumps(value)
                return pickle_value, "pickle"
            except (TypeError, ValueError):
                pass
            # Fallback к строке
            return str(value), "string"
        except Exception as e:
```

#### Строка 363: _matches_pattern
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
        """Проверка соответствия ключа паттерну."""
        try:
            if pattern == "*":
                return True
            # Простая проверка по подстроке
            return pattern in key
        except Exception as e:
```

#### Строка 364: _matches_pattern
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'простая'
**Код:**
```python
        try:
            if pattern == "*":
                return True
            # Простая проверка по подстроке
            return pattern in key
        except Exception as e:
            self.logger.error(f"Error matching pattern: {e}")
```

#### Строка 368: _matches_pattern
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return pattern in key
        except Exception as e:
            self.logger.error(f"Error matching pattern: {e}")
            return False

    async def _evict_entries(self) -> None:
        """Вытеснение записей из кэша."""
```

#### Строка 419: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python
            return [key for key, _ in sorted_entries[:evict_count]]
        except Exception as e:
            self.logger.error(f"Error selecting keys for eviction: {e}")
            return []

    async def _cleanup_loop(self) -> None:
        """Цикл очистки кэша."""
```

#### Строка 447: _stats_loop
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'debug'
**Код:**
```python
                # Логируем статистику
                stats = await self.get_stats()
                if stats:
                    self.logger.debug(f"Cache stats: {stats}")
            except Exception as e:
                self.logger.error(f"Error in stats loop: {e}")

```

### 📁 application\services\implementations\market_service_impl.py
Найдено проблем: 17

#### Строка 61: validate_config
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
        for config_key in required_configs:
            if config_key not in self.config:
                self.logger.error(f"Missing required config: {config_key}")
                return False
        return True

    async def get_market_data(self, symbol: Symbol) -> Optional[MarketData]:
```

#### Строка 62: validate_config
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            if config_key not in self.config:
                self.logger.error(f"Missing required config: {config_key}")
                return False
        return True

    async def get_market_data(self, symbol: Symbol) -> Optional[MarketData]:
        """Получение рыночных данных."""
```

#### Строка 86: _get_market_data_impl
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            self._market_data_cache[symbol_str] = market_data
            self._cleanup_cache_if_needed()
            return market_data
        return None

    async def get_historical_data(
        self,
```

#### Строка 152: _get_current_price_impl
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            self._price_cache[symbol_str] = (market_data.close, market_data.timestamp)
            self._cleanup_cache_if_needed()
            return market_data.close
        return None

    async def get_order_book(
        self, symbol: Symbol, depth: int = 10
```

#### Строка 173: Unknown
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python
            cached_orderbook = self._orderbook_cache[cache_key]
            if not self._is_cache_expired(cached_orderbook.get("timestamp")):
                return cached_orderbook
        # Получаем данные из репозитория - заглушка, так как метода нет
        # orderbook = await self.market_repository.get_order_book(symbol, depth)
        # Временно возвращаем None
        return None
```

#### Строка 175: Unknown
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'временно'
**Код:**
```python
                return cached_orderbook
        # Получаем данные из репозитория - заглушка, так как метода нет
        # orderbook = await self.market_repository.get_order_book(symbol, depth)
        # Временно возвращаем None
        return None

    async def get_market_metrics(self, symbol: Symbol) -> Optional[Dict[str, Any]]:
```

#### Строка 176: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
        # Получаем данные из репозитория - заглушка, так как метода нет
        # orderbook = await self.market_repository.get_order_book(symbol, depth)
        # Временно возвращаем None
        return None

    async def get_market_metrics(self, symbol: Symbol) -> Optional[Dict[str, Any]]:
        """Получение рыночных метрик."""
```

#### Строка 191: get_market_metrics
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
        # Получаем рыночные данные
        market_data = await self.get_market_data(symbol)
        if not market_data:
            return None
        # Рассчитываем метрики - заглушка, так как метод ожидает DataFrame
        # metrics = self.market_metrics_service.calculate_trend_metrics(market_data)
        return {
```

#### Строка 192: get_market_metrics
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python
        market_data = await self.get_market_data(symbol)
        if not market_data:
            return None
        # Рассчитываем метрики - заглушка, так как метод ожидает DataFrame
        # metrics = self.market_metrics_service.calculate_trend_metrics(market_data)
        return {
            "symbol": str(symbol),
```

#### Строка 225: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
        if callback not in self._subscribers[symbol_str]:
            self._subscribers[symbol_str].append(callback)
            self.logger.info(f"Added subscriber for {symbol_str}")
            return True
        return False

    async def analyze_market(self, symbol: Symbol) -> ProtocolMarketAnalysis:
```

#### Строка 226: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            self._subscribers[symbol_str].append(callback)
            self.logger.info(f"Added subscriber for {symbol_str}")
            return True
        return False

    async def analyze_market(self, symbol: Symbol) -> ProtocolMarketAnalysis:
        """Анализ рынка."""
```

#### Строка 242: _analyze_market_impl
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python
            raise ValueError(f"No market data available for {symbol}")
        # Получаем технические индикаторы
        technical_indicators = await self.get_technical_indicators(symbol)
        # Анализируем рынок - заглушка
        return ProtocolMarketAnalysis(
            data={
                "symbol": symbol,
```

#### Строка 273: get_technical_indicators
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python
        market_data = await self.get_market_data(symbol)
        if not market_data:
            raise ValueError(f"No market data available for {symbol}")
        # Рассчитываем технические индикаторы - заглушка
        return ProtocolTechnicalIndicators(
            data={
                "symbol": symbol,
```

#### Строка 296: _is_cache_expired
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
    def _is_cache_expired(self, timestamp: Optional[TimestampValue]) -> bool:
        """Проверка истечения срока действия кэша."""
        if not timestamp:
            return True
        value = getattr(timestamp, "value", timestamp)
        cache_age = (datetime.now() - value).total_seconds()
        return cache_age > self.cache_ttl_seconds
```

#### Строка 374: validate_input
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
        self._price_cache.clear()
        self._orderbook_cache.clear()
        self.logger.info("MarketService stopped")

    # Реализация абстрактных методов из BaseService
    def validate_input(self, data: Any) -> bool:
        """Валидация входных данных."""
        if isinstance(data, dict):
            # Валидация для рыночных данных
            required_fields = ["symbol", "timestamp"]
            return all(field in data for field in required_fields)
```

#### Строка 383: validate_input
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
        elif isinstance(data, str):
            # Валидация для символа
            return len(data.strip()) > 0
        return False

    def process(self, data: Any) -> Any:
        """Обработка данных."""
```

#### Строка 405: _process_market_data
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'временно'
**Код:**
```python
                    processed[key] = float(processed[key])
                except (ValueError, TypeError):
                    processed[key] = 0.0
        # Нормализация временной метки
        if "timestamp" in processed:
            try:
                if isinstance(processed["timestamp"], str):
```

### 📁 application\services\implementations\ml_service_impl.py
Найдено проблем: 29

#### Строка 67: validate_config
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
        for config_key in required_configs:
            if config_key not in self.config:
                self.logger.error(f"Missing required config: {config_key}")
                return False
        return True

    async def predict_price(
```

#### Строка 68: validate_config
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            if config_key not in self.config:
                self.logger.error(f"Missing required config: {config_key}")
                return False
        return True

    async def predict_price(
        self, symbol: Symbol, features: Dict[str, Any]
```

#### Строка 92: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
        model = await self._get_model_for_symbol(symbol)
        if not model:
            self.logger.warning(f"No model available for {symbol}")
            return None
        # Подготавливаем признаки
        prepared_features = await self._prepare_features(symbol, features)
        # Делаем предсказание
```

#### Строка 120: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            # Обновляем метрики модели
            await self._update_model_metrics(model.id, prediction_result)
            return ml_prediction
        return None

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Анализ настроений."""
```

#### Строка 141: _analyze_sentiment_impl
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
            if not self._is_sentiment_cache_expired(cached_sentiment):
                return cached_sentiment
        # Анализируем настроения
        # Здесь нет метода analyze_sentiment, оставляем заглушку или реализуем через сторонний сервис
        sentiment_result = None  # TODO: реализовать через внешний сервис или удалить
        if sentiment_result:
            result = {
```

#### Строка 142: _analyze_sentiment_impl
**Тип:** TODO/FIXME
**Описание:** Найдено триггерное слово: 'todo'
**Код:**
```python
                return cached_sentiment
        # Анализируем настроения
        # Здесь нет метода analyze_sentiment, оставляем заглушку или реализуем через сторонний сервис
        sentiment_result = None  # TODO: реализовать через внешний сервис или удалить
        if sentiment_result:
            result = {
                "sentiment_score": float(sentiment_result.sentiment_score),
```

#### Строка 187: detect_patterns
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python
        pattern_model = await self._get_pattern_detection_model()
        if not pattern_model:
            self.logger.warning("No pattern detection model available")
            return []
        # Здесь нет метода detect_patterns, оставляем заглушку или реализуем через сторонний сервис
        patterns: List[Any] = []  # TODO: реализовать через внешний сервис или удалить
        result = []
```

#### Строка 188: detect_patterns
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
        if not pattern_model:
            self.logger.warning("No pattern detection model available")
            return []
        # Здесь нет метода detect_patterns, оставляем заглушку или реализуем через сторонний сервис
        patterns: List[Any] = []  # TODO: реализовать через внешний сервис или удалить
        result = []
        for pattern in patterns:
```

#### Строка 189: detect_patterns
**Тип:** TODO/FIXME
**Описание:** Найдено триггерное слово: 'todo'
**Код:**
```python
            self.logger.warning("No pattern detection model available")
            return []
        # Здесь нет метода detect_patterns, оставляем заглушку или реализуем через сторонний сервис
        patterns: List[Any] = []  # TODO: реализовать через внешний сервис или удалить
        result = []
        for pattern in patterns:
            pattern_detection = PatternDetection(
```

#### Строка 226: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
        risk_model = await self._get_risk_model()
        if not risk_model:
            self.logger.warning("No risk model available")
            return {}
        # Здесь нет метода calculate_risk_metrics, оставляем заглушку или реализуем через сторонний сервис
        risk_metrics = None  # TODO: реализовать через внешний сервис или удалить
        return {}
```

#### Строка 227: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
        if not risk_model:
            self.logger.warning("No risk model available")
            return {}
        # Здесь нет метода calculate_risk_metrics, оставляем заглушку или реализуем через сторонний сервис
        risk_metrics = None  # TODO: реализовать через внешний сервис или удалить
        return {}

```

#### Строка 228: Unknown
**Тип:** TODO/FIXME
**Описание:** Найдено триггерное слово: 'todo'
**Код:**
```python
            self.logger.warning("No risk model available")
            return {}
        # Здесь нет метода calculate_risk_metrics, оставляем заглушку или реализуем через сторонний сервис
        risk_metrics = None  # TODO: реализовать через внешний сервис или удалить
        return {}

    async def train_model(
```

#### Строка 229: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
            return {}
        # Здесь нет метода calculate_risk_metrics, оставляем заглушку или реализуем через сторонний сервис
        risk_metrics = None  # TODO: реализовать через внешний сервис или удалить
        return {}

    async def train_model(
        self, model_id: str, training_data: List[Dict[str, Any]]
```

#### Строка 250: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            model = await self.ml_repository.get_model(entity_id)
            if not model:
                self.logger.error(f"Model {model_id} not found")
                return False
            # Здесь нет метода train_model, используем train_models если возможно
            # await self.ml_predictor.train_models(training_data)  # если training_data — DataFrame
            # После обучения сохраняем модель
```

#### Строка 251: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
            if not model:
                self.logger.error(f"Model {model_id} not found")
                return False
            # Здесь нет метода train_model, используем train_models если возможно
            # await self.ml_predictor.train_models(training_data)  # если training_data — DataFrame
            # После обучения сохраняем модель
            await self.ml_repository.save_model(model)
```

#### Строка 258: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            # Обновляем активную модель
            self._active_models[model_id] = model
            self.logger.info(f"Model {model_id} trained successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error training model {model_id}: {e}")
            return False
```

#### Строка 261: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Error training model {model_id}: {e}")
            return False

    async def evaluate_model(
        self, model_id: str, test_data: List[Dict[str, Any]]
```

#### Строка 264: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
            return False

    async def evaluate_model(
        self, model_id: str, test_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Оценка модели."""
        return await self._execute_with_metrics(
```

#### Строка 268: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
    ) -> Dict[str, Any]:
        """Оценка модели."""
        return await self._execute_with_metrics(
            "evaluate_model", self._evaluate_model_impl, model_id, test_data
        )

    async def _evaluate_model_impl(
```

#### Строка 272: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
        )

    async def _evaluate_model_impl(
        self, model_id: str, test_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Реализация оценки модели."""
        try:
```

#### Строка 282: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
            model = await self.ml_repository.get_model(entity_id)
            if not model:
                return {"error": f"Model {model_id} not found"}
            # Здесь нет метода evaluate_model, оставляем заглушку или реализуем через внешний сервис
            return {}
        except Exception as e:
            self.logger.error(f"Error evaluating model {model_id}: {e}")
```

#### Строка 283: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
            if not model:
                return {"error": f"Model {model_id} not found"}
            # Здесь нет метода evaluate_model, оставляем заглушку или реализуем через внешний сервис
            return {}
        except Exception as e:
            self.logger.error(f"Error evaluating model {model_id}: {e}")
            return {"error": str(e)}
```

#### Строка 343: _is_prediction_cache_expired
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
            return
        # Обновляем accuracy, precision, recall, f1_score, mse, mae если есть
        # model.update_metrics({...})
        await self.ml_repository.save_model(model)

    def _is_prediction_cache_expired(self, prediction: MLPrediction) -> bool:
        """Проверка истечения срока действия кэша предсказаний."""
        # Упрощенно - считаем кэш истекшим через 5 минут
        return True

    def _is_sentiment_cache_expired(self, sentiment_data: Dict[str, Any]) -> bool:
```

#### Строка 346: _is_prediction_cache_expired
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
    def _is_prediction_cache_expired(self, prediction: MLPrediction) -> bool:
        """Проверка истечения срока действия кэша предсказаний."""
        # Упрощенно - считаем кэш истекшим через 5 минут
        return True

    def _is_sentiment_cache_expired(self, sentiment_data: Dict[str, Any]) -> bool:
        """Проверка истечения срока действия кэша настроений."""
```

#### Строка 352: _is_sentiment_cache_expired
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
        """Проверка истечения срока действия кэша настроений."""
        timestamp = sentiment_data.get("timestamp")
        if not timestamp:
            return True
        cache_age = (datetime.now() - timestamp).total_seconds()
        return cache_age > self.sentiment_cache_ttl

```

#### Строка 413: validate_input
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
        self._sentiment_cache.clear()
        self._active_models.clear()
        self._model_metrics.clear()

    # Реализация абстрактных методов из BaseService
    def validate_input(self, data: Any) -> bool:
        """Валидация входных данных."""
        if isinstance(data, dict):
            # Валидация для ML данных
            if "features" in data:
                return isinstance(data["features"], dict) and len(data["features"]) > 0
```

#### Строка 426: validate_input
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
        elif isinstance(data, str):
            # Валидация для текста
            return len(data.strip()) > 0
        return False

    def process(self, data: Any) -> Any:
        """Обработка данных."""
```

#### Строка 452: _process_ml_data
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
                        processed["features"][key] = float(value)
                    except (ValueError, TypeError):
                        # Оставляем как строку
                        pass
        # Нормализация метаданных
        if "metadata" in processed and isinstance(processed["metadata"], dict):
            for key, value in processed["metadata"].items():
```

#### Строка 462: _process_text
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'базовая'
**Код:**
```python

    def _process_text(self, text: str) -> str:
        """Обработка текста."""
        # Базовая очистка текста
        cleaned = text.strip()
        # Удаление лишних пробелов
        import re
```

### 📁 application\services\implementations\notification_service_impl.py
Найдено проблем: 17

#### Строка 34: __init__
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            }
            self._notifications.append(notification)
            self.logger.info(f"Notification sent: {message}")
            return True
        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")
            return False
```

#### Строка 37: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")
            return False

    async def send_alert(self, alert_type: str, data: Dict[str, Any]) -> bool:
        """Отправка алерта."""
```

#### Строка 49: send_alert
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            }
            self._notifications.append(alert)
            self.logger.info(f"Alert sent: {alert_type}")
            return True
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")
            return False
```

#### Строка 52: send_alert
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")
            return False

    async def subscribe_to_alerts(self, user_id: str, alert_types: List[str]) -> bool:
        """Подписка на алерты."""
```

#### Строка 64: subscribe_to_alerts
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            }
            self._notifications.append(subscription)
            self.logger.info(f"User {user_id} subscribed to alerts: {alert_types}")
            return True
        except Exception as e:
            self.logger.error(f"Error subscribing to alerts: {e}")
            return False
```

#### Строка 67: subscribe_to_alerts
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Error subscribing to alerts: {e}")
            return False

    async def send_trade_notification(self, trade: Trade) -> bool:
        """Отправка уведомления о сделке."""
```

#### Строка 83: send_trade_notification
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            }
            self._notifications.append(trade_notification)
            self.logger.info(f"Trade notification sent: {trade.id}")
            return True
        except Exception as e:
            self.logger.error(f"Error sending trade notification: {e}")
            return False
```

#### Строка 86: send_trade_notification
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Error sending trade notification: {e}")
            return False

    async def send_risk_alert(
        self, portfolio_id: PortfolioId, risk_level: str, details: Dict[str, Any]
```

#### Строка 102: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            }
            self._notifications.append(risk_alert)
            self.logger.info(f"Risk alert sent for portfolio {portfolio_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error sending risk alert: {e}")
            return False
```

#### Строка 105: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Error sending risk alert: {e}")
            return False

    async def send_performance_report(
        self, portfolio_id: PortfolioId, metrics: PerformanceMetrics
```

#### Строка 120: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            }
            self._notifications.append(performance_report)
            self.logger.info(f"Performance report sent for portfolio {portfolio_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error sending performance report: {e}")
            return False
```

#### Строка 123: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Error sending performance report: {e}")
            return False

    async def send_bulk_notifications(
        self, notifications: List[Dict[str, Any]]
```

#### Строка 132: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
        try:
            for notification in notifications:
                await self.send_notification(**notification)
            return True
        except Exception as e:
            self.logger.error(f"Error sending bulk notifications: {e}")
            return False
```

#### Строка 135: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Error sending bulk notifications: {e}")
            return False

    def get_notifications(
        self, notification_type: Optional[str] = None, limit: int = 100
```

#### Строка 150: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python
            return notifications[-limit:]
        except Exception as e:
            self.logger.error(f"Error getting notifications: {e}")
            return []

    def clear_notifications(self) -> bool:
        """Очистка уведомлений."""
```

#### Строка 156: clear_notifications
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
        """Очистка уведомлений."""
        try:
            self._notifications.clear()
            return True
        except Exception as e:
            self.logger.error(f"Error clearing notifications: {e}")
            return False
```

#### Строка 159: clear_notifications
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Error clearing notifications: {e}")
            return False

```

### 📁 application\services\implementations\portfolio_service_impl.py
Найдено проблем: 23

#### Строка 82: validate_config
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
        for config_key in required_configs:
            if config_key not in self.config:
                self.logger.error(f"Missing required config: {config_key}")
                return False
        return True

    async def calculate_weights(
```

#### Строка 83: validate_config
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            if config_key not in self.config:
                self.logger.error(f"Missing required config: {config_key}")
                return False
        return True

    async def calculate_weights(
        self, portfolio_id: UUID
```

#### Строка 109: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
            self.logger.error(
                f"Error calculating weights for portfolio {portfolio_id}: {e}"
            )
            return {}

    async def rebalance_portfolio(
        self, portfolio_id: UUID, target_weights: Dict[Symbol, Decimal]
```

#### Строка 135: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            )
            if not rebalance_orders:
                self.logger.info(f"Portfolio {portfolio_id} is already balanced")
                return True
            # Выполняем ордера ребалансировки
            success_count = 0
            for order in rebalance_orders:
```

#### Строка 140: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
            success_count = 0
            for order in rebalance_orders:
                try:
                    # Здесь должна быть логика исполнения ордера
                    # Пока просто логируем
                    self.logger.info(f"Rebalancing order: {order}")
                    success_count += 1
```

#### Строка 153: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            # Очищаем кэш
            self._clear_portfolio_cache(portfolio_id)
            self.logger.info(f"Portfolio {portfolio_id} rebalanced successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error rebalancing portfolio {portfolio_id}: {e}")
            return False
```

#### Строка 156: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Error rebalancing portfolio {portfolio_id}: {e}")
            return False

    async def calculate_pnl(self, portfolio_id: PortfolioId) -> Money:
        """Расчет P&L."""
```

#### Строка 269: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
            self.logger.error(
                f"Error getting portfolio balance for {portfolio_id}: {e}"
            )
            return {}

    async def update_portfolio_balance(
        self, portfolio_id: UUID, changes: Dict[str, Money]
```

#### Строка 304: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
                self.logger.info(
                    f"Portfolio {portfolio_id} balance updated successfully"
                )
                return True
            else:
                self.logger.error(f"Failed to update portfolio {portfolio_id} balance")
                return False
```

#### Строка 307: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
                return True
            else:
                self.logger.error(f"Failed to update portfolio {portfolio_id} balance")
                return False
        except Exception as e:
            self.logger.error(
                f"Error updating portfolio balance for {portfolio_id}: {e}"
```

#### Строка 312: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            self.logger.error(
                f"Error updating portfolio balance for {portfolio_id}: {e}"
            )
            return False

    async def get_portfolio_performance(
        self, portfolio_id: UUID, period: str = "1d"
```

#### Строка 387: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            # Получаем портфель
            portfolio = await self.get_portfolio_by_id(portfolio_id)
            if not portfolio:
                return False, [f"Portfolio {portfolio_id} not found"]
            # Проверяем лимиты риска
            if hasattr(portfolio, 'risk_limits') and portfolio.risk_limits:
                # Проверяем максимальный размер позиции
```

#### Строка 434: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            self.logger.error(
                f"Error validating portfolio constraints for {portfolio_id}: {e}"
            )
            return False, [f"Validation error: {str(e)}"]

    async def get_portfolio_by_id(
        self, portfolio_id: UUID
```

#### Строка 455: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            return portfolio
        except Exception as e:
            self.logger.error(f"Error getting portfolio {portfolio_id}: {e}")
            return None

    async def _calculate_rebalance_orders(
        self,
```

#### Строка 498: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python
            return orders
        except Exception as e:
            self.logger.error(f"Error calculating rebalance orders: {e}")
            return []

    async def _calculate_portfolio_volatility(
        self, portfolio_id: PortfolioId
```

#### Строка 648: _is_portfolio_cache_expired
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
                [p for p in portfolios if p.is_active]
            )
        except Exception as e:
            self.logger.error(f"Error loading portfolio statistics: {e}")

    def _is_portfolio_cache_expired(self, portfolio: Portfolio) -> bool:
        """Проверка истечения срока действия кэша портфеля."""
        # Упрощенно - считаем кэш истекшим через 5 минут
        return True

    def _is_metrics_cache_expired(self, metrics: PortfolioMetrics) -> bool:
```

#### Строка 651: _is_portfolio_cache_expired
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
    def _is_portfolio_cache_expired(self, portfolio: Portfolio) -> bool:
        """Проверка истечения срока действия кэша портфеля."""
        # Упрощенно - считаем кэш истекшим через 5 минут
        return True

    def _is_metrics_cache_expired(self, metrics: PortfolioMetrics) -> bool:
        """Проверка истечения срока действия кэша метрик."""
```

#### Строка 653: _is_metrics_cache_expired
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
    def _is_portfolio_cache_expired(self, portfolio: Portfolio) -> bool:
        """Проверка истечения срока действия кэша портфеля."""
        # Упрощенно - считаем кэш истекшим через 5 минут
        return True

    def _is_metrics_cache_expired(self, metrics: PortfolioMetrics) -> bool:
        """Проверка истечения срока действия кэша метрик."""
        # Упрощенно - считаем кэш истекшим через 1 минуту
        return True

    def _is_balance_cache_expired(self, balance: Dict[str, Money]) -> bool:
```

#### Строка 656: _is_metrics_cache_expired
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
    def _is_metrics_cache_expired(self, metrics: PortfolioMetrics) -> bool:
        """Проверка истечения срока действия кэша метрик."""
        # Упрощенно - считаем кэш истекшим через 1 минуту
        return True

    def _is_balance_cache_expired(self, balance: Dict[str, Money]) -> bool:
        """Проверка истечения срока действия кэша баланса."""
```

#### Строка 658: _is_balance_cache_expired
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
    def _is_metrics_cache_expired(self, metrics: PortfolioMetrics) -> bool:
        """Проверка истечения срока действия кэша метрик."""
        # Упрощенно - считаем кэш истекшим через 1 минуту
        return True

    def _is_balance_cache_expired(self, balance: Dict[str, Money]) -> bool:
        """Проверка истечения срока действия кэша баланса."""
        # Упрощенно - всегда обновляем баланс
        return True

    def _is_performance_cache_expired(self, performance: PerformanceMetrics) -> bool:
```

#### Строка 661: _is_balance_cache_expired
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
    def _is_balance_cache_expired(self, balance: Dict[str, Money]) -> bool:
        """Проверка истечения срока действия кэша баланса."""
        # Упрощенно - всегда обновляем баланс
        return True

    def _is_performance_cache_expired(self, performance: PerformanceMetrics) -> bool:
        """Проверка истечения срока действия кэша производительности."""
```

#### Строка 663: _is_performance_cache_expired
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
    def _is_balance_cache_expired(self, balance: Dict[str, Money]) -> bool:
        """Проверка истечения срока действия кэша баланса."""
        # Упрощенно - всегда обновляем баланс
        return True

    def _is_performance_cache_expired(self, performance: PerformanceMetrics) -> bool:
        """Проверка истечения срока действия кэша производительности."""
        # Упрощенно - считаем кэш истекшим через 1 час
        return True

    def _clear_portfolio_cache(self, portfolio_id: PortfolioId) -> None:
```

#### Строка 666: _is_performance_cache_expired
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
    def _is_performance_cache_expired(self, performance: PerformanceMetrics) -> bool:
        """Проверка истечения срока действия кэша производительности."""
        # Упрощенно - считаем кэш истекшим через 1 час
        return True

    def _clear_portfolio_cache(self, portfolio_id: PortfolioId) -> None:
        """Очистка кэша портфеля."""
```

### 📁 application\services\implementations\risk_service_impl.py
Найдено проблем: 6

#### Строка 46: validate_config
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
        for config_key in required_configs:
            if config_key not in self.config:
                self.logger.error(f"Missing required config: {config_key}")
                return False
        return True

    async def assess_portfolio_risk(self, portfolio_id: PortfolioId) -> RiskMetrics:
```

#### Строка 47: validate_config
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            if config_key not in self.config:
                self.logger.error(f"Missing required config: {config_key}")
                return False
        return True

    async def assess_portfolio_risk(self, portfolio_id: PortfolioId) -> RiskMetrics:
        """Оценка риска портфеля через domain/services/risk_analysis.py."""
```

#### Строка 132: get_risk_alerts
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python
        """Получение алертов риска."""
        risk_metrics = await self.assess_portfolio_risk(portfolio_id)
        # Возвращаем пустой список алертов, так как метод не реализован в RiskAnalysisService
        return []

    def _is_risk_cache_expired(self, risk_metrics: RiskMetrics) -> bool:
        """Проверка истечения срока действия кэша рисков."""
```

#### Строка 134: _is_risk_cache_expired
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
        """Получение алертов риска."""
        risk_metrics = await self.assess_portfolio_risk(portfolio_id)
        # Возвращаем пустой список алертов, так как метод не реализован в RiskAnalysisService
        return []

    def _is_risk_cache_expired(self, risk_metrics: RiskMetrics) -> bool:
        """Проверка истечения срока действия кэша рисков."""
        # Упрощенная проверка - считаем кэш истекшим через 5 минут
        return True

    async def _risk_monitoring_loop(self) -> None:
```

#### Строка 136: _is_risk_cache_expired
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python

    def _is_risk_cache_expired(self, risk_metrics: RiskMetrics) -> bool:
        """Проверка истечения срока действия кэша рисков."""
        # Упрощенная проверка - считаем кэш истекшим через 5 минут
        return True

    async def _risk_monitoring_loop(self) -> None:
```

#### Строка 137: _is_risk_cache_expired
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
    def _is_risk_cache_expired(self, risk_metrics: RiskMetrics) -> bool:
        """Проверка истечения срока действия кэша рисков."""
        # Упрощенная проверка - считаем кэш истекшим через 5 минут
        return True

    async def _risk_monitoring_loop(self) -> None:
        """Цикл мониторинга рисков."""
```

### 📁 application\services\implementations\trading_service_impl.py
Найдено проблем: 30

#### Строка 75: validate_config
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
        for config_key in required_configs:
            if config_key not in self.config:
                self.logger.error(f"Missing required config: {config_key}")
                return False
        return True

    async def place_order(self, order: Order) -> bool:
```

#### Строка 76: validate_config
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            if config_key not in self.config:
                self.logger.error(f"Missing required config: {config_key}")
                return False
        return True

    async def place_order(self, order: Order) -> bool:
        """Размещение ордера."""
```

#### Строка 90: _place_order_impl
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            # Валидируем ордер
            if not await self._validate_order(order):
                self.logger.error(f"Order validation failed for {order.id}")
                return False
            # Проверяем баланс
            if not await self._check_balance(order):
                self.logger.error(f"Insufficient balance for order {order.id}")
```

#### Строка 94: _place_order_impl
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            # Проверяем баланс
            if not await self._check_balance(order):
                self.logger.error(f"Insufficient balance for order {order.id}")
                return False
            # Добавляем ордер в очередь
            await self._order_queue.put(order)
            # Кэшируем ордер
```

#### Строка 105: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            else:
                self._trading_stats["total_orders"] = 1
            self.logger.info(f"Order {order.id} queued for execution")
            return True
        except Exception as e:
            self.logger.error(f"Error placing order {order.id}: {e}")
            current_failed = self._trading_stats.get("failed_orders", 0)
```

#### Строка 113: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
                self._trading_stats["failed_orders"] = int(current_failed) + 1
            else:
                self._trading_stats["failed_orders"] = 1
            return False

    async def cancel_order(self, order_id: OrderId) -> bool:
        """Отмена ордера."""
```

#### Строка 128: _cancel_order_impl
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            order = await self.trading_repository.get_order(order_id)
            if not order:
                self.logger.error(f"Order {order_id} not found")
                return False

            # Проверяем, что ордер можно отменить
            if order.status not in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]:
```

#### Строка 133: _cancel_order_impl
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            # Проверяем, что ордер можно отменить
            if order.status not in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]:
                self.logger.error(f"Cannot cancel order {order_id} with status {order.status}")
                return False

            # Отменяем ордер
            order.status = OrderStatus.CANCELLED
```

#### Строка 151: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
                self._trading_stats["successful_orders"] = 1

            self.logger.info(f"Order {order_id} cancelled successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
```

#### Строка 160: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
                self._trading_stats["failed_orders"] = int(current_failed) + 1
            else:
                self._trading_stats["failed_orders"] = 1
            return False

    async def get_order_status(self, order_id: OrderId) -> Optional[OrderStatus]:
        """Получение статуса ордера."""
```

#### Строка 184: _get_order_status_impl
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
                self._order_cache[str(order_id)] = order
                return order.status

            return None

        except Exception as e:
            self.logger.error(f"Error getting order status for {order_id}: {e}")
```

#### Строка 188: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python

        except Exception as e:
            self.logger.error(f"Error getting order status for {order_id}: {e}")
            return None

    async def get_account_balance(self) -> Dict[str, Money]:
        """Получение баланса аккаунта."""
```

#### Строка 213: _get_account_balance_impl
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python

        except Exception as e:
            self.logger.error(f"Error getting account balance: {e}")
            return {}

    async def get_trade_history(self, symbol: Symbol, limit: int = 100) -> List[Trade]:
        """Получение истории сделок."""
```

#### Строка 237: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python

        except Exception as e:
            self.logger.error(f"Error getting trade history for {symbol}: {e}")
            return []

    async def get_open_orders(self, symbol: Optional[Symbol] = None) -> List[Order]:
        """Получение открытых ордеров."""
```

#### Строка 262: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python
            return orders
        except Exception as e:
            self.logger.error(f"Error getting open orders: {e}")
            return []

    async def _validate_order(self, order: Order) -> bool:
        """Валидация ордера."""
```

#### Строка 269: _validate_order
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
        try:
            # Проверяем обязательные поля
            if not order.symbol or not order.amount or not order.side:
                return False
            # Проверяем размер ордера
            if order.amount.value <= 0:
                return False
```

#### Строка 272: _validate_order
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
                return False
            # Проверяем размер ордера
            if order.amount.value <= 0:
                return False
            # Проверяем цену для лимитных ордеров
            if order.order_type.value == "LIMIT" and not order.price:
                return False
```

#### Строка 275: _validate_order
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
                return False
            # Проверяем цену для лимитных ордеров
            if order.order_type.value == "LIMIT" and not order.price:
                return False
            # Проверяем стоп-цену для стоп-ордеров
            if order.order_type.value == "STOP" and not order.stop_price:
                return False
```

#### Строка 278: _validate_order
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
                return False
            # Проверяем стоп-цену для стоп-ордеров
            if order.order_type.value == "STOP" and not order.stop_price:
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error validating order: {e}")
```

#### Строка 279: _validate_order
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            # Проверяем стоп-цену для стоп-ордеров
            if order.order_type.value == "STOP" and not order.stop_price:
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error validating order: {e}")
            return False
```

#### Строка 282: _validate_order
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Error validating order: {e}")
            return False

    async def _check_balance(self, order: Order) -> bool:
        """Проверка баланса для ордера."""
```

#### Строка 299: _check_balance
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
                    order_cost = float(order.amount.value) * float(order.price.value)
                else:
                    # Для рыночных ордеров используем текущую цену
                    order_cost = float(order.amount.value) * 1.0  # Упрощенная логика
                return float(available_balance.value) >= float(order_cost)
            return False
        except Exception as e:
```

#### Строка 301: _check_balance
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
                    # Для рыночных ордеров используем текущую цену
                    order_cost = float(order.amount.value) * 1.0  # Упрощенная логика
                return float(available_balance.value) >= float(order_cost)
            return False
        except Exception as e:
            self.logger.error(f"Error checking balance: {e}")
            return False
```

#### Строка 304: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return False
        except Exception as e:
            self.logger.error(f"Error checking balance: {e}")
            return False

    async def _order_processor_loop(self) -> None:
        """Цикл обработки ордеров."""
```

#### Строка 391: _is_balance_cache_expired
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
    def _is_trade_cache_expired(self, trade: Trade) -> bool:
        """Проверка истечения срока действия кэша сделок."""
        cache_age = (datetime.now() - trade.timestamp).total_seconds()
        return cache_age > self.trade_cache_ttl

    def _is_balance_cache_expired(self, balance: Dict[str, Money]) -> bool:
        """Проверка истечения кэша баланса."""
        # Простая реализация - в реальной системе нужно отслеживать время кэширования
        return False

    async def get_trading_statistics(self) -> Dict[str, Any]:
```

#### Строка 393: _is_balance_cache_expired
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'простая'
**Код:**
```python

    def _is_balance_cache_expired(self, balance: Dict[str, Money]) -> bool:
        """Проверка истечения кэша баланса."""
        # Простая реализация - в реальной системе нужно отслеживать время кэширования
        return False

    async def get_trading_statistics(self) -> Dict[str, Any]:
```

#### Строка 394: _is_balance_cache_expired
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
    def _is_balance_cache_expired(self, balance: Dict[str, Money]) -> bool:
        """Проверка истечения кэша баланса."""
        # Простая реализация - в реальной системе нужно отслеживать время кэширования
        return False

    async def get_trading_statistics(self) -> Dict[str, Any]:
        """Получение статистики торговли."""
```

#### Строка 426: stop
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
            try:
                await self._order_processor_task
            except asyncio.CancelledError:
                pass
        # Очищаем кэши
        self._order_cache.clear()
        self._trade_cache.clear()
```

#### Строка 432: stop
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'empty'
**Код:**
```python
        self._trade_cache.clear()
        self._balance_cache.clear()
        # Очищаем очередь
        while not self._order_queue.empty():
            try:
                self._order_queue.get_nowait()
            except asyncio.QueueEmpty:
```

#### Строка 435: stop
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'empty'
**Код:**
```python
        while not self._order_queue.empty():
            try:
                self._order_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

```

### 📁 application\services\market_analysis_service.py
Найдено проблем: 9

#### Строка 28: __init__
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=100)
            latest_data = await self.market_repository.get_market_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
```

#### Строка 35: __init__
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
                end_time=end_time,
                limit=100,
            )
            if not latest_data:
                raise DomainError(f"No market data found for {symbol}")
            # Используем domain-сервис для расчёта сводки рынка
            summary = self.domain_market_analysis_service.calculate_market_summary(
```

#### Строка 39: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
                raise DomainError(f"No market data found for {symbol}")
            # Используем domain-сервис для расчёта сводки рынка
            summary = self.domain_market_analysis_service.calculate_market_summary(
                latest_data, symbol, timeframe
            )
            # Преобразуем в ожидаемый формат
            return {
```

#### Строка 74: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
                end_time=end_time,
            )
            if not market_data:
                return {}
            # Собираем данные
            price_volume_data = []
            for data in market_data:
```

#### Строка 85: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
                except Exception:
                    continue
            if not price_volume_data:
                return {}
            # Используем domain-сервис для расчёта профиля объёма
            volume_profile_result = (
                self.domain_market_analysis_service.calculate_volume_profile(
```

#### Строка 118: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
                end_time=end_time,
            )
            if not market_data:
                return {}
            # Извлекаем данные
            prices = []
            volumes = []
```

#### Строка 131: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
                except Exception:
                    continue
            if len(prices) < 20:
                return {}
            # Используем domain-сервис для анализа рыночного режима
            market_regime_result = (
                self.domain_market_analysis_service.calculate_market_regime(
```

#### Строка 164: _extract_numeric_value
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return 0'
**Код:**
```python
            elif isinstance(value_obj, (int, float, Decimal)):
                return float(value_obj)
            else:
                return 0.0
        except (ValueError, TypeError, AttributeError):
            return 0.0

```

#### Строка 166: _extract_numeric_value
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return 0'
**Код:**
```python
            else:
                return 0.0
        except (ValueError, TypeError, AttributeError):
            return 0.0

```

### 📁 application\services\market_data_service.py
Найдено проблем: 8

#### Строка 79: get_volume_profile
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
    async def get_volume_profile(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Получение профиля объема."""
        try:
            # В реальной системе здесь был бы запрос к репозиторию
            return {
                "symbol": symbol,
                "poc_price": "51000",
```

#### Строка 92: get_market_regime_analysis
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
    async def get_market_regime_analysis(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Получение анализа рыночного режима."""
        try:
            # В реальной системе здесь был бы запрос к репозиторию
            return {
                "symbol": symbol,
                "regime": "trending",
```

#### Строка 156: get_real_time_price
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            )
            if market_data:
                return self._extract_numeric_value(market_data[0].close_price)
            return None
        except Exception as e:
            raise ExchangeError(f"Error getting real-time price: {str(e)}")

```

#### Строка 163: get_market_depth
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
    async def get_market_depth(self, symbol: str, depth: int = 10) -> dict:
        """Получение глубины рынка."""
        try:
            # В реальной системе здесь был бы запрос к бирже
            return {
                "symbol": symbol,
                "bids": [],
```

#### Строка 176: subscribe_to_updates
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
    async def subscribe_to_updates(self, symbol: str, callback: Callable) -> bool:
        """Подписка на обновления рыночных данных."""
        try:
            # В реальной системе здесь была бы подписка на WebSocket
            return True
        except Exception as e:
            raise ExchangeError(f"Error subscribing to updates: {str(e)}")
```

#### Строка 177: subscribe_to_updates
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
        """Подписка на обновления рыночных данных."""
        try:
            # В реальной системе здесь была бы подписка на WebSocket
            return True
        except Exception as e:
            raise ExchangeError(f"Error subscribing to updates: {str(e)}")

```

#### Строка 205: _extract_numeric_value
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return 0'
**Код:**
```python
            elif isinstance(value_obj, (int, float, Decimal)):
                return float(value_obj)
            else:
                return 0.0
        except (ValueError, TypeError, AttributeError):
            return 0.0

```

#### Строка 207: _extract_numeric_value
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return 0'
**Код:**
```python
            else:
                return 0.0
        except (ValueError, TypeError, AttributeError):
            return 0.0

```

### 📁 application\services\market_service.py
Найдено проблем: 12

#### Строка 61: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
                limit=168,  # 7 дней * 24 часа
            )
            if not market_data:
                return {}
            # Анализируем данные
            latest_data = market_data[-1]
            oldest_data = market_data[0]
```

#### Строка 63: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
            if not market_data:
                return {}
            # Анализируем данные
            latest_data = market_data[-1]
            oldest_data = market_data[0]
            # Рассчитываем изменения
            price_change = (
```

#### Строка 67: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
            oldest_data = market_data[0]
            # Рассчитываем изменения
            price_change = (
                self._extract_numeric_value(latest_data.close_price)
                - self._extract_numeric_value(oldest_data.close_price)
            )
            price_change_percent = (
```

#### Строка 88: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "current_price": self._extract_numeric_value(latest_data.close_price),
                "price_change_24h": price_change,
                "price_change_percent": price_change_percent,
                "volume_24h": sum(volumes),
```

#### Строка 95: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
                "high_24h": max(high_prices),
                "low_24h": min(low_prices),
                "volatility": (max(high_prices) - min(low_prices))
                / self._extract_numeric_value(latest_data.close_price)
                * 100,
                "trend_direction": "up" if price_change > 0 else "down",
                "support_levels": [min(low_prices)],
```

#### Строка 100: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
                "trend_direction": "up" if price_change > 0 else "down",
                "support_levels": [min(low_prices)],
                "resistance_levels": [max(high_prices)],
                "timestamp": latest_data.timestamp.isoformat(),
            }
        except Exception as e:
            raise ExchangeError(f"Error getting market summary: {str(e)}")
```

#### Строка 123: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
        try:
            if alert_types is None:
                alert_types = []
            # В реальной системе здесь была бы логика анализа алертов
            alerts = []
            # Проверяем волатильность
            market_summary = await self.get_market_summary(symbol)
```

#### Строка 153: get_market_depth
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
    async def get_market_depth(self, symbol: str, depth: int = 10) -> Dict[str, Any]:
        """Получение глубины рынка."""
        try:
            # В реальной системе здесь был бы запрос к бирже
            return {
                "symbol": symbol,
                "bids": [
```

#### Строка 172: get_order_book
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
    async def get_order_book(self, symbol: str) -> Dict[str, Any]:
        """Получение стакана заявок."""
        try:
            # В реальной системе здесь был бы запрос к бирже
            return {
                "symbol": symbol,
                "bids": [
```

#### Строка 191: get_recent_trades
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Получение последних сделок."""
        try:
            # В реальной системе здесь был бы запрос к бирже
            trades = []
            base_price = 100.0
            for i in range(limit):
```

#### Строка 222: _extract_numeric_value
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return 0'
**Код:**
```python
            elif isinstance(value_obj, (int, float, Decimal)):
                return float(value_obj)
            else:
                return 0.0
        except (ValueError, TypeError, AttributeError):
            return 0.0

```

#### Строка 224: _extract_numeric_value
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return 0'
**Код:**
```python
            else:
                return 0.0
        except (ValueError, TypeError, AttributeError):
            return 0.0

```

### 📁 application\services\ml_service.py
Найдено проблем: 9

#### Строка 137: delete_model
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
        try:
            model = await self.get_model(model_id)
            if not model:
                return False
            await self.ml_repository.delete_model(EntityId(UUID(model_id)))
            return True
        except Exception as e:
```

#### Строка 139: delete_model
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            if not model:
                return False
            await self.ml_repository.delete_model(EntityId(UUID(model_id)))
            return True
        except Exception as e:
            raise MLModelError(f"Error deleting model: {str(e)}")

```

#### Строка 233: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
            # Создание модели для предсказания (в реальной системе модель должна быть сохранена)
            if model.model_type == ModelType.RANDOM_FOREST:
                ml_model = RandomForestRegressor(random_state=42)
                # В реальной системе здесь должна быть загрузка обученной модели
                # ml_model = joblib.load(f"models/{model_id}.pkl")
            else:
                ml_model = RandomForestRegressor(random_state=42)
```

#### Строка 256: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
            raise MLModelError(f"Error making prediction: {str(e)}")

    async def evaluate_model(
        self, model_id: str, test_data: pd.DataFrame, test_target: pd.Series
    ) -> Dict[str, float]:
        """Оценка модели на тестовых данных."""
        try:
```

#### Строка 271: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
            from sklearn.metrics import mean_absolute_error, mean_squared_error  # type: ignore

            # Подготовка данных
            X = test_data.to_numpy()
            y = test_target.to_numpy()

            # Создание модели для оценки (в реальной системе модель должна быть загружена)
```

#### Строка 272: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python

            # Подготовка данных
            X = test_data.to_numpy()
            y = test_target.to_numpy()

            # Создание модели для оценки (в реальной системе модель должна быть загружена)
            ml_model = RandomForestRegressor(random_state=42)
```

#### Строка 276: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python

            # Создание модели для оценки (в реальной системе модель должна быть загружена)
            ml_model = RandomForestRegressor(random_state=42)
            # В реальной системе здесь должна быть загрузка обученной модели
            # ml_model = joblib.load(f"models/{model_id}.pkl")

            # Предсказание на тестовых данных
```

#### Строка 353: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
        except Exception as e:
            raise MLModelError(f"Error getting predictions: {str(e)}")

    async def get_latest_prediction(self, model_id: str) -> Optional[Prediction]:
        """Получение последнего предсказания модели."""
        try:
            predictions = await self.get_predictions(model_id, limit=1)
```

#### Строка 359: get_latest_prediction
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
            predictions = await self.get_predictions(model_id, limit=1)
            return predictions[0] if predictions else None
        except Exception as e:
            raise MLModelError(f"Error getting latest prediction: {str(e)}")

```

### 📁 application\services\news_trading_integration.py
Найдено проблем: 4

#### Строка 125: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
            )
        except Exception as e:
            logging.error(f"Ошибка генерации торгового сигнала: {e}")
            return self._create_default_signal()

    def _calculate_combined_sentiment(
        self,
```

#### Строка 161: _determine_signal_direction
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'базовая'
**Код:**
```python
        market_volatility: float,
    ) -> float:
        """Рассчитывает силу сигнала."""
        # Базовая сила на основе абсолютного значения сентимента
        social_sentiment = social_data.sentiment_score if social_data else 0.0
        base_strength = abs(news_data.sentiment_score + social_sentiment) / 2
        # Усиление при высокой волатильности
```

#### Строка 182: Unknown
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'базовая'
**Код:**
```python
        technical_sentiment: float,
    ) -> float:
        """Рассчитывает уверенность в сигнале."""
        # Базовая уверенность на основе количества данных
        news_confidence = min(len(news_data.news_items) / 10.0, 1.0)
        social_confidence = social_data.confidence if social_data else 0.0
        # Средняя уверенность
```

#### Строка 227: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        unique_topics = list(set(topics))
        return unique_topics[:10]

    def _create_default_signal(self) -> TradingSignal:
        """Создает сигнал по умолчанию при ошибках."""
        return TradingSignal(
            direction="hold",
```

### 📁 application\services\notification_service.py
Найдено проблем: 26

#### Строка 40: __init__
**Класс:** Notification
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class Notification:
    """Уведомление."""

    id: UUID = field(default_factory=uuid4)
    user_id: str = ""
    title: str = ""
    message: str = ""
```

#### Строка 47: __init__
**Класс:** Notification
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    level: NotificationLevel = NotificationLevel.INFO
    type: NotificationType = NotificationType.SYSTEM
    priority: NotificationPriority = NotificationPriority.NORMAL
    channels: List[NotificationChannel] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
```

#### Строка 48: __init__
**Класс:** Notification
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    type: NotificationType = NotificationType.SYSTEM
    priority: NotificationPriority = NotificationPriority.NORMAL
    channels: List[NotificationChannel] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    delivered: bool = False
```

#### Строка 49: __init__
**Класс:** Notification
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    priority: NotificationPriority = NotificationPriority.NORMAL
    channels: List[NotificationChannel] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    delivered: bool = False
    read: bool = False
```

#### Строка 59: Unknown
**Класс:** Alert
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class Alert:
    """Алерт."""

    id: UUID = field(default_factory=uuid4)
    alert_type: str = ""
    title: str = ""
    message: str = ""
```

#### Строка 64: Unknown
**Класс:** Alert
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    title: str = ""
    message: str = ""
    severity: NotificationLevel = NotificationLevel.WARNING
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
```

#### Строка 65: Unknown
**Класс:** Alert
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    message: str = ""
    severity: NotificationLevel = NotificationLevel.WARNING
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
```

#### Строка 100: send
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
    async def send(self, notification: Notification) -> bool:
        """Отправка email уведомления."""
        try:
            # Здесь должна быть реальная реализация отправки email
            # Пока что просто логируем
            self.logger.info(
                f"Email notification sent to {notification.user_id}: {notification.title}"
```

#### Строка 105: send
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            self.logger.info(
                f"Email notification sent to {notification.user_id}: {notification.title}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            return False
```

#### Строка 108: send
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            return False

    async def is_available(self) -> bool:
        """Проверка доступности SMTP."""
```

#### Строка 112: is_available
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python

    async def is_available(self) -> bool:
        """Проверка доступности SMTP."""
        return True  # Упрощенная проверка

    def get_name(self) -> str:
        return "email"
```

#### Строка 129: send
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
    async def send(self, notification: Notification) -> bool:
        """Отправка webhook уведомления."""
        try:
            # Здесь должна быть реальная реализация отправки webhook
            # Пока что просто логируем
            self.logger.info(f"Webhook notification sent: {notification.title}")
            return True
```

#### Строка 132: send
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            # Здесь должна быть реальная реализация отправки webhook
            # Пока что просто логируем
            self.logger.info(f"Webhook notification sent: {notification.title}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
            return False
```

#### Строка 135: send
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
            return False

    async def is_available(self) -> bool:
        """Проверка доступности webhook."""
```

#### Строка 139: is_available
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python

    async def is_available(self) -> bool:
        """Проверка доступности webhook."""
        return True  # Упрощенная проверка

    def get_name(self) -> str:
        return "webhook"
```

#### Строка 150: __init__
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python

    def __init__(self):
        self.templates: Dict[str, NotificationTemplate] = {}
        self._load_default_templates()

    def _load_default_templates(self) -> None:
        """Загрузка шаблонов по умолчанию."""
```

#### Строка 152: __init__
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        self.templates: Dict[str, NotificationTemplate] = {}
        self._load_default_templates()

    def _load_default_templates(self) -> None:
        """Загрузка шаблонов по умолчанию."""
        self.templates = {
            "trade_executed": NotificationTemplate(
```

#### Строка 209: format_message
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
        """Форматирование сообщения по шаблону."""
        template = self.get_template(template_name)
        if not template:
            return None
        try:
            return template.message.format(**kwargs)
        except KeyError as e:
```

#### Строка 270: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
                metadata=metadata or {},
            )
            await self.notification_queue.put(notification)
            return True
        except Exception as e:
            self.logger.error(f"Failed to queue notification: {e}")
            return False
```

#### Строка 273: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Failed to queue notification: {e}")
            return False

    async def send_alert(
        self,
```

#### Строка 303: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
                },
            )
            await self.notification_queue.put(notification)
            return True
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
            return False
```

#### Строка 306: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
            return False

    async def send_trade_notification(self, trade: Trade) -> bool:
        """Отправка уведомления о сделке."""
```

#### Строка 328: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            )
        except Exception as e:
            self.logger.error(f"Failed to send trade notification: {e}")
            return False

    async def send_risk_alert(
        self, portfolio_id: str, risk_level: str, details: Dict[str, Any]
```

#### Строка 349: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            )
        except Exception as e:
            self.logger.error(f"Failed to send risk alert: {e}")
            return False

    async def send_order_notification(self, order: Order, event_type: str) -> bool:
        """Отправка уведомления об ордере."""
```

#### Строка 372: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            )
        except Exception as e:
            self.logger.error(f"Failed to send order notification: {e}")
            return False

    async def send_position_notification(
        self, position: Position, event_type: str
```

#### Строка 406: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            )
        except Exception as e:
            self.logger.error(f"Failed to send position notification: {e}")
            return False

    async def _process_notification_queue(self):
        """Обработка очереди уведомлений."""
```

### 📁 application\services\order_creator.py
Найдено проблем: 2

#### Строка 118: _generate_warnings
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
        if order.order_type == OrderType.MARKET and order.side == OrderSide.SELL:
            # Проверка достаточности позиции для продажи
            try:
                # В реальной системе здесь был бы запрос к репозиторию позиций
                # position = await self.position_repository.get_position_by_symbol(
                #     portfolio.id, order.trading_pair
                # )
```

#### Строка 124: _generate_warnings
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
                # )
                # if not position or position.volume.amount < order.quantity:
                #     warnings.append("Insufficient position for sell order")
                # Упрощенная проверка - предполагаем, что позиция есть
                warnings.append(
                    "Position check required - implement position repository integration"
                )
```

### 📁 application\services\order_validator.py
Найдено проблем: 5

#### Строка 38: __init__
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'базовая'
**Код:**
```python
    ) -> Tuple[bool, List[str]]:
        """Валидация ордера."""
        errors = []
        # Базовая валидация
        basic_errors = self._validate_basic_order(order)
        errors.extend(basic_errors)
        # Валидация размера
```

#### Строка 39: __init__
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'basic'
**Код:**
```python
        """Валидация ордера."""
        errors = []
        # Базовая валидация
        basic_errors = self._validate_basic_order(order)
        errors.extend(basic_errors)
        # Валидация размера
        min_size = (
```

#### Строка 40: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'basic'
**Код:**
```python
        errors = []
        # Базовая валидация
        basic_errors = self._validate_basic_order(order)
        errors.extend(basic_errors)
        # Валидация размера
        min_size = (
            min_order_size
```

#### Строка 65: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'basic'
**Код:**
```python
        errors.extend(limit_errors)
        return len(errors) == 0, errors

    def _validate_basic_order(self, order: Order) -> List[str]:
        """Валидация базовых параметров ордера."""
        errors = []
        # Проверка количества
```

#### Строка 190: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
                f"Order size {order_value} exceeds maximum position size {max_position_size}"
            )
        # Проверка дневного лимита ордеров
        # Здесь можно добавить логику проверки количества ордеров за день
        return errors

    def validate_stop_loss(
```

### 📁 application\services\portfolio_service.py
Найдено проблем: 17

#### Строка 61: get_portfolio_by_account
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
    async def get_portfolio_by_account(self, account_id: str) -> Optional[Portfolio]:
        """Получение портфеля по ID аккаунта."""
        try:
            # Упрощенная реализация - нет метода get_portfolio_by_account
            return None
        except Exception as e:
            raise DomainError(f"Error getting portfolio by account: {str(e)}")
```

#### Строка 62: get_portfolio_by_account
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
        """Получение портфеля по ID аккаунта."""
        try:
            # Упрощенная реализация - нет метода get_portfolio_by_account
            return None
        except Exception as e:
            raise DomainError(f"Error getting portfolio by account: {str(e)}")

```

#### Строка 81: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
            if not portfolio:
                raise DomainError(f"Portfolio {portfolio_id} not found")
            
            # Упрощенная реализация - обновляем общий баланс
            # Исправление: total_balance это свойство, которое нельзя изменять напрямую
            # В реальной системе здесь была бы логика обновления через специальные методы
            portfolio.updated_at = Timestamp.now()
```

#### Строка 83: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
            
            # Упрощенная реализация - обновляем общий баланс
            # Исправление: total_balance это свойство, которое нельзя изменять напрямую
            # В реальной системе здесь была бы логика обновления через специальные методы
            portfolio.updated_at = Timestamp.now()
            
            await self.portfolio_repository.save_portfolio(portfolio)
```

#### Строка 100: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
            if not portfolio:
                raise DomainError(f"Portfolio {portfolio_id} not found")
            
            # Упрощенная реализация - нет метода add_position
            portfolio.updated_at = Timestamp.now()
            await self.portfolio_repository.save_portfolio(portfolio)
            return portfolio
```

#### Строка 116: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
            if not portfolio:
                raise DomainError(f"Portfolio {portfolio_id} not found")
            
            # Упрощенная реализация - нет метода update_position
            portfolio.updated_at = Timestamp.now()
            await self.portfolio_repository.save_portfolio(portfolio)
            return portfolio
```

#### Строка 130: remove_position
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
            if not portfolio:
                raise DomainError(f"Portfolio {portfolio_id} not found")
            
            # Упрощенная реализация - нет метода remove_position
            portfolio.updated_at = Timestamp.now()
            await self.portfolio_repository.save_portfolio(portfolio)
            return portfolio
```

#### Строка 144: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
        try:
            portfolio = await self.get_portfolio(portfolio_id)
            if not portfolio:
                return None
            # Упрощенная реализация - нет метода get_position
            return None
        except Exception as e:
```

#### Строка 145: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
            portfolio = await self.get_portfolio(portfolio_id)
            if not portfolio:
                return None
            # Упрощенная реализация - нет метода get_position
            return None
        except Exception as e:
            raise DomainError(f"Error getting position: {str(e)}")
```

#### Строка 146: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            if not portfolio:
                return None
            # Упрощенная реализация - нет метода get_position
            return None
        except Exception as e:
            raise DomainError(f"Error getting position: {str(e)}")

```

#### Строка 157: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
        try:
            portfolio = await self.get_portfolio(portfolio_id)
            if not portfolio:
                return None
            # Упрощенная реализация - возвращаем общий баланс
            return portfolio.total_balance
        except Exception as e:
```

#### Строка 158: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
            portfolio = await self.get_portfolio(portfolio_id)
            if not portfolio:
                return None
            # Упрощенная реализация - возвращаем общий баланс
            return portfolio.total_balance
        except Exception as e:
            raise DomainError(f"Error getting balance: {str(e)}")
```

#### Строка 179: calculate_unrealized_pnl
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
            portfolio = await self.get_portfolio(portfolio_id)
            if not portfolio:
                raise DomainError(f"Portfolio {portfolio_id} not found")
            # Упрощенная реализация - возвращаем нулевой P&L
            return Money(Decimal("0"), Currency.USD)
        except Exception as e:
            raise DomainError(f"Error calculating unrealized P&L: {str(e)}")
```

#### Строка 189: get_open_positions
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
        try:
            portfolio = await self.get_portfolio(portfolio_id)
            if not portfolio:
                return {}
            # Упрощенная реализация - возвращаем пустой словарь
            return {}
        except Exception as e:
```

#### Строка 190: get_open_positions
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
            portfolio = await self.get_portfolio(portfolio_id)
            if not portfolio:
                return {}
            # Упрощенная реализация - возвращаем пустой словарь
            return {}
        except Exception as e:
            raise DomainError(f"Error getting open positions: {str(e)}")
```

#### Строка 191: get_open_positions
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
            if not portfolio:
                return {}
            # Упрощенная реализация - возвращаем пустой словарь
            return {}
        except Exception as e:
            raise DomainError(f"Error getting open positions: {str(e)}")

```

#### Строка 222: calculate_risk_metrics
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенные'
**Код:**
```python
            if not portfolio:
                raise DomainError(f"Portfolio {portfolio_id} not found")
            
            # Упрощенные метрики риска
            return {
                "total_equity": float(portfolio.total_balance.amount),
                "margin_level": float(portfolio.free_margin.amount / portfolio.total_balance.amount) if portfolio.total_balance.amount > 0 else 0.0,
```

### 📁 application\services\risk_assessor.py
Найдено проблем: 6

#### Строка 153: _calculate_total_exposure
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенный'
**Код:**
```python

    def _calculate_total_exposure(self, portfolio: Portfolio) -> Decimal:
        """Расчет общего риска портфеля."""
        # Упрощенный расчет - сумма всех позиций
        total_exposure = Decimal("0")
        # Здесь должен быть перебор позиций портфеля, например: for position in portfolio.positions:
        # total_exposure += position.size.to_decimal() * position.entry_price.to_decimal()
```

#### Строка 155: _calculate_total_exposure
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
        """Расчет общего риска портфеля."""
        # Упрощенный расчет - сумма всех позиций
        total_exposure = Decimal("0")
        # Здесь должен быть перебор позиций портфеля, например: for position in portfolio.positions:
        # total_exposure += position.size.to_decimal() * position.entry_price.to_decimal()
        # Пока позиций нет, возвращаем 0
        return (
```

#### Строка 166: _calculate_max_drawdown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенный'
**Код:**
```python

    def _calculate_max_drawdown(self, portfolio: Portfolio) -> Decimal:
        """Расчет максимальной просадки."""
        # Упрощенный расчет - разница между максимальным и текущим балансом
        # Здесь должен быть перебор истории баланса, пока возвращаем 0
        return Decimal("0")

```

#### Строка 167: _calculate_max_drawdown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
    def _calculate_max_drawdown(self, portfolio: Portfolio) -> Decimal:
        """Расчет максимальной просадки."""
        # Упрощенный расчет - разница между максимальным и текущим балансом
        # Здесь должен быть перебор истории баланса, пока возвращаем 0
        return Decimal("0")

    def _calculate_value_at_risk(self, portfolio: Portfolio) -> Decimal:
```

#### Строка 172: _calculate_value_at_risk
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенный'
**Код:**
```python

    def _calculate_value_at_risk(self, portfolio: Portfolio) -> Decimal:
        """Расчет Value at Risk."""
        # Упрощенный расчет - 5% от общего баланса
        return portfolio.balance.amount * Decimal("0.05")

    def _calculate_sharpe_ratio(self, portfolio: Portfolio) -> Decimal:
```

#### Строка 177: _calculate_sharpe_ratio
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенный'
**Код:**
```python

    def _calculate_sharpe_ratio(self, portfolio: Portfolio) -> Decimal:
        """Расчет коэффициента Шарпа."""
        # Упрощенный расчет - возвращаем базовое значение
        # Исправлено: убираем проблемную логику с пустыми списками
        return Decimal("1.0")  # Базовый коэффициент Шарпа

```

### 📁 application\services\risk_service.py
Найдено проблем: 13

#### Строка 104: Unknown
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python

            return RiskValidationResult(
                is_valid=True,
                reason="All risk checks passed",
                risk_level=float(total_risk),
            )

```

#### Строка 153: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
            # Проверяем корреляцию с другими позициями
            correlation_count = 0
            for position in existing_positions:
                # Упрощенная логика: считаем корреляцию на основе направления
                if hasattr(position, 'side') and hasattr(signal, 'signal_type'):
                    if position.side.value == signal.signal_type.value:
                        correlation_count += 1
```

#### Строка 172: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенный'
**Код:**
```python
    ) -> Decimal:
        """Расчет общего риска портфеля."""
        try:
            # Упрощенный расчет риска портфеля
            total_risk = Decimal("0")

            # Риск от существующих позиций
```

#### Строка 204: _calculate_daily_loss_risk
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенный'
**Код:**
```python
    async def _calculate_daily_loss_risk(self, portfolio: Portfolio) -> Decimal:
        """Расчет риска дневных убытков."""
        try:
            # Упрощенный расчет дневных убытков
            # В реальной системе здесь был бы анализ P&L за день

            daily_pnl = Decimal("0")
```

#### Строка 205: _calculate_daily_loss_risk
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
        """Расчет риска дневных убытков."""
        try:
            # Упрощенный расчет дневных убытков
            # В реальной системе здесь был бы анализ P&L за день

            daily_pnl = Decimal("0")
            existing_positions: List[Position] = self._get_portfolio_positions(portfolio)
```

#### Строка 229: _get_portfolio_positions
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python

    def _get_portfolio_positions(self, portfolio: Portfolio) -> List[Position]:
        """Получение позиций портфеля."""
        # В реальной системе здесь был бы вызов репозитория позиций
        # Пока возвращаем пустой список
        return []

```

#### Строка 231: _get_portfolio_positions
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python
        """Получение позиций портфеля."""
        # В реальной системе здесь был бы вызов репозитория позиций
        # Пока возвращаем пустой список
        return []

    async def calculate_risk_metrics(self) -> Dict[str, float]:
        """Расчет всех риск-метрик."""
```

#### Строка 238: calculate_risk_metrics
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
        try:
            portfolio = await self.portfolio_repository.get_portfolio(PortfolioId(uuid4()))
            if not portfolio:
                return {}

            existing_positions: List[Position] = self._get_portfolio_positions(portfolio)

```

#### Строка 258: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python

        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {}

    async def close_risky_positions(self) -> List[str]:
        """Закрытие рискованных позиций."""
```

#### Строка 265: close_risky_positions
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python
        try:
            portfolio = await self.portfolio_repository.get_portfolio(PortfolioId(uuid4()))
            if not portfolio:
                return []

            closed_positions: List[str] = []
            existing_positions: List[Position] = self._get_portfolio_positions(portfolio)
```

#### Строка 271: close_risky_positions
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
            existing_positions: List[Position] = self._get_portfolio_positions(portfolio)

            for position in existing_positions:
                # Упрощенная логика определения рискованных позиций
                if hasattr(position, 'quantity') and hasattr(position, 'entry_price'):
                    position_risk = (
                        position.quantity.amount
```

#### Строка 291: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python

        except Exception as e:
            self.logger.error(f"Error closing risky positions: {e}")
            return []

    async def _close_position(self, position: Position) -> None:
        """Закрытие позиции."""
```

#### Строка 315: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
                updated_at=Timestamp.now()
            )

            # В реальной системе здесь был бы вызов exchange API
            # await self.exchange_service.place_order(close_order)

            self.logger.info(
```

### 📁 application\services\service_factory.py
Найдено проблем: 57

#### Строка 43: Unknown
**Класс:** ServiceFactory
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
        self, config: Optional[Dict[str, Any]] = None
    ) -> MarketService:
        """Создание сервиса рынка."""
        pass

    @abstractmethod
    def create_ml_service(self, config: Optional[Dict[str, Any]] = None) -> MLService:
```

#### Строка 48: create_ml_service
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
    @abstractmethod
    def create_ml_service(self, config: Optional[Dict[str, Any]] = None) -> MLService:
        """Создание ML сервиса."""
        pass

    @abstractmethod
    def create_trading_service(
```

#### Строка 55: create_ml_service
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
        self, config: Optional[Dict[str, Any]] = None
    ) -> TradingService:
        """Создание торгового сервиса."""
        pass

    @abstractmethod
    def create_strategy_service(
```

#### Строка 62: create_ml_service
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
        self, config: Optional[Dict[str, Any]] = None
    ) -> StrategyService:
        """Создание сервиса стратегий."""
        pass

    @abstractmethod
    def create_portfolio_service(
```

#### Строка 69: Unknown
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
        self, config: Optional[Dict[str, Any]] = None
    ) -> PortfolioService:
        """Создание сервиса портфелей."""
        pass

    @abstractmethod
    def create_risk_service(
```

#### Строка 76: Unknown
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
        self, config: Optional[Dict[str, Any]] = None
    ) -> RiskService:
        """Создание сервиса рисков."""
        pass

    @abstractmethod
    def create_cache_service(
```

#### Строка 83: Unknown
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
        self, config: Optional[Dict[str, Any]] = None
    ) -> CacheService:
        """Создание сервиса кэширования."""
        pass

    @abstractmethod
    def create_notification_service(
```

#### Строка 90: Unknown
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
        self, config: Optional[Dict[str, Any]] = None
    ) -> NotificationService:
        """Создание сервиса уведомлений."""
        pass


class DefaultServiceFactory(ServiceFactory):
```

#### Строка 93: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        pass


class DefaultServiceFactory(ServiceFactory):
    """Реализация фабрики сервисов по умолчанию."""

    def __init__(self, global_config: Optional[Dict[str, Any]] = None):
```

#### Строка 275: _get_risk_repository
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
        """Объединение конфигураций."""
        merged = base_config.copy()
        merged.update(override_config)
        return merged

    def _get_risk_repository(self):
        """Получить репозиторий рисков."""
        # Заглушка для репозитория рисков
        return None

    def _get_technical_analysis_service(self):
```

#### Строка 277: _get_risk_repository
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python

    def _get_risk_repository(self):
        """Получить репозиторий рисков."""
        # Заглушка для репозитория рисков
        return None

    def _get_technical_analysis_service(self):
```

#### Строка 278: _get_risk_repository
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
    def _get_risk_repository(self):
        """Получить репозиторий рисков."""
        # Заглушка для репозитория рисков
        return None

    def _get_technical_analysis_service(self):
        """Получить сервис технического анализа."""
```

#### Строка 280: _get_technical_analysis_service
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
    def _get_risk_repository(self):
        """Получить репозиторий рисков."""
        # Заглушка для репозитория рисков
        return None

    def _get_technical_analysis_service(self):
        """Получить сервис технического анализа."""
        # Заглушка для сервиса технического анализа
        return None

    def _get_market_metrics_service(self):
```

#### Строка 282: _get_technical_analysis_service
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python

    def _get_technical_analysis_service(self):
        """Получить сервис технического анализа."""
        # Заглушка для сервиса технического анализа
        return None

    def _get_market_metrics_service(self):
```

#### Строка 283: _get_technical_analysis_service
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
    def _get_technical_analysis_service(self):
        """Получить сервис технического анализа."""
        # Заглушка для сервиса технического анализа
        return None

    def _get_market_metrics_service(self):
        """Получить сервис рыночных метрик."""
```

#### Строка 285: _get_market_metrics_service
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
    def _get_technical_analysis_service(self):
        """Получить сервис технического анализа."""
        # Заглушка для сервиса технического анализа
        return None

    def _get_market_metrics_service(self):
        """Получить сервис рыночных метрик."""
        # Заглушка для сервиса рыночных метрик
        return None

    def _get_market_repository(self):
```

#### Строка 287: _get_market_metrics_service
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python

    def _get_market_metrics_service(self):
        """Получить сервис рыночных метрик."""
        # Заглушка для сервиса рыночных метрик
        return None

    def _get_market_repository(self):
```

#### Строка 288: _get_market_metrics_service
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
    def _get_market_metrics_service(self):
        """Получить сервис рыночных метрик."""
        # Заглушка для сервиса рыночных метрик
        return None

    def _get_market_repository(self):
        """Получить репозиторий рынка."""
```

#### Строка 290: _get_market_repository
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
    def _get_market_metrics_service(self):
        """Получить сервис рыночных метрик."""
        # Заглушка для сервиса рыночных метрик
        return None

    def _get_market_repository(self):
        """Получить репозиторий рынка."""
        # Заглушка для репозитория рынка
        return None

    def _get_ml_predictor(self):
```

#### Строка 292: _get_market_repository
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python

    def _get_market_repository(self):
        """Получить репозиторий рынка."""
        # Заглушка для репозитория рынка
        return None

    def _get_ml_predictor(self):
```

#### Строка 293: _get_market_repository
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
    def _get_market_repository(self):
        """Получить репозиторий рынка."""
        # Заглушка для репозитория рынка
        return None

    def _get_ml_predictor(self):
        """Получить ML предиктор."""
```

#### Строка 295: _get_ml_predictor
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
    def _get_market_repository(self):
        """Получить репозиторий рынка."""
        # Заглушка для репозитория рынка
        return None

    def _get_ml_predictor(self):
        """Получить ML предиктор."""
        # Заглушка для ML предиктора
        return None

    def _get_ml_repository(self):
```

#### Строка 297: _get_ml_predictor
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python

    def _get_ml_predictor(self):
        """Получить ML предиктор."""
        # Заглушка для ML предиктора
        return None

    def _get_ml_repository(self):
```

#### Строка 298: _get_ml_predictor
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
    def _get_ml_predictor(self):
        """Получить ML предиктор."""
        # Заглушка для ML предиктора
        return None

    def _get_ml_repository(self):
        """Получить ML репозиторий."""
```

#### Строка 300: _get_ml_repository
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
    def _get_ml_predictor(self):
        """Получить ML предиктор."""
        # Заглушка для ML предиктора
        return None

    def _get_ml_repository(self):
        """Получить ML репозиторий."""
        # Заглушка для ML репозитория
        return None

    def _get_signal_service(self):
```

#### Строка 302: _get_ml_repository
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python

    def _get_ml_repository(self):
        """Получить ML репозиторий."""
        # Заглушка для ML репозитория
        return None

    def _get_signal_service(self):
```

#### Строка 303: _get_ml_repository
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
    def _get_ml_repository(self):
        """Получить ML репозиторий."""
        # Заглушка для ML репозитория
        return None

    def _get_signal_service(self):
        """Получить сервис сигналов."""
```

#### Строка 305: _get_signal_service
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
    def _get_ml_repository(self):
        """Получить ML репозиторий."""
        # Заглушка для ML репозитория
        return None

    def _get_signal_service(self):
        """Получить сервис сигналов."""
        # Заглушка для сервиса сигналов
        return None

    def _get_trading_repository(self):
```

#### Строка 307: _get_signal_service
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python

    def _get_signal_service(self):
        """Получить сервис сигналов."""
        # Заглушка для сервиса сигналов
        return None

    def _get_trading_repository(self):
```

#### Строка 308: _get_signal_service
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
    def _get_signal_service(self):
        """Получить сервис сигналов."""
        # Заглушка для сервиса сигналов
        return None

    def _get_trading_repository(self):
        """Получить торговый репозиторий."""
```

#### Строка 310: _get_trading_repository
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
    def _get_signal_service(self):
        """Получить сервис сигналов."""
        # Заглушка для сервиса сигналов
        return None

    def _get_trading_repository(self):
        """Получить торговый репозиторий."""
        # Заглушка для торгового репозитория
        return None

    def _get_portfolio_optimizer(self):
```

#### Строка 312: _get_trading_repository
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python

    def _get_trading_repository(self):
        """Получить торговый репозиторий."""
        # Заглушка для торгового репозитория
        return None

    def _get_portfolio_optimizer(self):
```

#### Строка 313: _get_trading_repository
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
    def _get_trading_repository(self):
        """Получить торговый репозиторий."""
        # Заглушка для торгового репозитория
        return None

    def _get_portfolio_optimizer(self):
        """Получить оптимизатор портфеля."""
```

#### Строка 315: _get_portfolio_optimizer
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
    def _get_trading_repository(self):
        """Получить торговый репозиторий."""
        # Заглушка для торгового репозитория
        return None

    def _get_portfolio_optimizer(self):
        """Получить оптимизатор портфеля."""
        # Заглушка для оптимизатора портфеля
        return None

    def _get_portfolio_repository(self):
```

#### Строка 317: _get_portfolio_optimizer
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python

    def _get_portfolio_optimizer(self):
        """Получить оптимизатор портфеля."""
        # Заглушка для оптимизатора портфеля
        return None

    def _get_portfolio_repository(self):
```

#### Строка 318: _get_portfolio_optimizer
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
    def _get_portfolio_optimizer(self):
        """Получить оптимизатор портфеля."""
        # Заглушка для оптимизатора портфеля
        return None

    def _get_portfolio_repository(self):
        """Получить репозиторий портфеля."""
```

#### Строка 320: _get_portfolio_repository
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
    def _get_portfolio_optimizer(self):
        """Получить оптимизатор портфеля."""
        # Заглушка для оптимизатора портфеля
        return None

    def _get_portfolio_repository(self):
        """Получить репозиторий портфеля."""
        # Заглушка для репозитория портфеля
        return None

    def get_service_instance(self, service_type: str) -> Optional[Any]:
```

#### Строка 322: _get_portfolio_repository
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python

    def _get_portfolio_repository(self):
        """Получить репозиторий портфеля."""
        # Заглушка для репозитория портфеля
        return None

    def get_service_instance(self, service_type: str) -> Optional[Any]:
```

#### Строка 323: _get_portfolio_repository
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
    def _get_portfolio_repository(self):
        """Получить репозиторий портфеля."""
        # Заглушка для репозитория портфеля
        return None

    def get_service_instance(self, service_type: str) -> Optional[Any]:
        """Получить экземпляр сервиса по типу."""
```

#### Строка 367: __init__
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python

    def __init__(self):
        self._factories: Dict[str, Type[ServiceFactory]] = {}
        self._default_factory: Optional[Type[ServiceFactory]] = None

    def register_factory(self, name: str, factory_class: Type[ServiceFactory]) -> None:
        """Регистрация фабрики."""
```

#### Строка 373: register_factory
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        """Регистрация фабрики."""
        self._factories[name] = factory_class

    def set_default_factory(self, factory_class: Type[ServiceFactory]) -> None:
        """Установка фабрики по умолчанию."""
        self._default_factory = factory_class

```

#### Строка 375: set_default_factory
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python

    def set_default_factory(self, factory_class: Type[ServiceFactory]) -> None:
        """Установка фабрики по умолчанию."""
        self._default_factory = factory_class

    def get_factory(self, name: str) -> Optional[Type[ServiceFactory]]:
        """Получение фабрики по имени."""
```

#### Строка 381: get_factory
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        """Получение фабрики по имени."""
        return self._factories.get(name)

    def get_default_factory(self) -> Optional[Type[ServiceFactory]]:
        """Получение фабрики по умолчанию."""
        return self._default_factory

```

#### Строка 383: get_default_factory
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python

    def get_default_factory(self) -> Optional[Type[ServiceFactory]]:
        """Получение фабрики по умолчанию."""
        return self._default_factory

    def create_factory(
        self, name: str, config: Optional[Dict[str, Any]] = None
```

#### Строка 385: create_factory
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python

    def get_default_factory(self) -> Optional[Type[ServiceFactory]]:
        """Получение фабрики по умолчанию."""
        return self._default_factory

    def create_factory(
        self, name: str, config: Optional[Dict[str, Any]] = None
    ) -> Optional[ServiceFactory]:
        """Создание фабрики по имени."""
        factory_class = self.get_factory(name)
        if factory_class:
```

#### Строка 392: get_default_factory
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
        factory_class = self.get_factory(name)
        if factory_class:
            return factory_class()
        return None

    def create_default_factory(
        self, config: Optional[Dict[str, Any]] = None
```

#### Строка 394: get_default_factory
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
            return factory_class()
        return None

    def create_default_factory(
        self, config: Optional[Dict[str, Any]] = None
    ) -> Optional[ServiceFactory]:
        """Создание фабрики по умолчанию."""
```

#### Строка 394: create_default_factory
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
        factory_class = self.get_factory(name)
        if factory_class:
            return factory_class()
        return None

    def create_default_factory(
        self, config: Optional[Dict[str, Any]] = None
    ) -> Optional[ServiceFactory]:
        """Создание фабрики по умолчанию."""
        if self._default_factory:
            return self._default_factory()
```

#### Строка 398: get_default_factory
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        self, config: Optional[Dict[str, Any]] = None
    ) -> Optional[ServiceFactory]:
        """Создание фабрики по умолчанию."""
        if self._default_factory:
            return self._default_factory()
        return None

```

#### Строка 399: get_default_factory
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    ) -> Optional[ServiceFactory]:
        """Создание фабрики по умолчанию."""
        if self._default_factory:
            return self._default_factory()
        return None


```

#### Строка 400: get_default_factory
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
        """Создание фабрики по умолчанию."""
        if self._default_factory:
            return self._default_factory()
        return None


# Глобальный реестр фабрик
```

#### Строка 405: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python

# Глобальный реестр фабрик
_factory_registry = ServiceFactoryRegistry()
_factory_registry.register_factory("default", DefaultServiceFactory)
_factory_registry.set_default_factory(DefaultServiceFactory)


```

#### Строка 406: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
# Глобальный реестр фабрик
_factory_registry = ServiceFactoryRegistry()
_factory_registry.register_factory("default", DefaultServiceFactory)
_factory_registry.set_default_factory(DefaultServiceFactory)


def get_service_factory(
```

#### Строка 410: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python


def get_service_factory(
    name: str = "default", config: Optional[Dict[str, Any]] = None
) -> ServiceFactory:
    """Получение фабрики сервисов."""
    factory = _factory_registry.create_factory(name, config)
```

#### Строка 415: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    """Получение фабрики сервисов."""
    factory = _factory_registry.create_factory(name, config)
    if factory is None:
        factory = _factory_registry.create_default_factory(config)
    if factory is None:
        raise ValueError(f"Could not create service factory: {name}")
    return factory
```

#### Строка 426: register_service_factory
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    _factory_registry.register_factory(name, factory_class)


def set_default_service_factory(factory_class: Type[ServiceFactory]) -> None:
    """Установка фабрики сервисов по умолчанию."""
    _factory_registry.set_default_factory(factory_class)

```

#### Строка 428: set_default_service_factory
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python

def set_default_service_factory(factory_class: Type[ServiceFactory]) -> None:
    """Установка фабрики сервисов по умолчанию."""
    _factory_registry.set_default_factory(factory_class)

```

### 📁 application\services\technical_analysis_service.py
Найдено проблем: 3

#### Строка 55: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
                except Exception:
                    continue
            if not df_data:
                return {}
            df = pd.DataFrame(df_data)
            # Используем domain-сервис для расчёта индикаторов
            analysis_result = self.technical_analysis_service.analyze_market_data(
```

#### Строка 88: _extract_numeric_value
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return 0'
**Код:**
```python
            elif isinstance(value_obj, (int, float, Decimal)):
                return float(value_obj)
            else:
                return 0.0
        except (ValueError, TypeError, AttributeError):
            return 0.0

```

#### Строка 90: _extract_numeric_value
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return 0'
**Код:**
```python
            else:
                return 0.0
        except (ValueError, TypeError, AttributeError):
            return 0.0

```

### 📁 application\services\trading_service.py
Найдено проблем: 35

#### Строка 230: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        """Получение метрик риска."""


class DefaultTradingService(TradingService):
    """Реализация сервиса торговых операций."""

    async def create_order(
```

#### Строка 317: cancel_order
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
        self, portfolio_id: Optional[PortfolioId] = None
    ) -> List[Order]:
        """Получить активные ордера."""
        # Упрощенная реализация
        return []

    async def get_order_history(
```

#### Строка 318: cancel_order
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python
    ) -> List[Order]:
        """Получить активные ордера."""
        # Упрощенная реализация
        return []

    async def get_order_history(
        self,
```

#### Строка 328: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
        end_time: Optional[datetime] = None,
    ) -> List[Order]:
        """Получить историю ордеров."""
        # Упрощенная реализация
        return []

    async def get_trade_history(
```

#### Строка 329: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python
    ) -> List[Order]:
        """Получить историю ордеров."""
        # Упрощенная реализация
        return []

    async def get_trade_history(
        self,
```

#### Строка 339: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
        end_time: Optional[datetime] = None,
    ) -> List[Trade]:
        """Получить историю сделок."""
        # Упрощенная реализация
        return []

    async def start_trading_session(self, portfolio_id: PortfolioId) -> TradingSession:
```

#### Строка 340: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python
    ) -> List[Trade]:
        """Получить историю сделок."""
        # Упрощенная реализация
        return []

    async def start_trading_session(self, portfolio_id: PortfolioId) -> TradingSession:
        """Начать торговую сессию."""
```

#### Строка 349: start_trading_session
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
            start_time=datetime.now(),
        )

        # Упрощенная реализация - нет метода save_session
        return session

    async def end_trading_session(self, session_id: UUID) -> TradingSession:
```

#### Строка 354: end_trading_session
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python

    async def end_trading_session(self, session_id: UUID) -> TradingSession:
        """Завершить торговую сессию."""
        # Упрощенная реализация - нет метода get_session
        session = TradingSession(
            id=OrderId(session_id),  # Исправление: используем правильный тип
            start_time=datetime.now(),
```

#### Строка 391: get_order
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
        """Обновление ордера."""
        order = await self.trading_repository.get_order(order_id)
        if not order:
            return None

        if price is not None:
            order.price = price
```

#### Строка 437: get_trade
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python

    async def get_trade(self, trade_id: TradeId) -> Optional[Trade]:
        """Получение сделки."""
        # Упрощенная реализация
        return None

    async def create_position(
```

#### Строка 438: get_trade
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
    async def get_trade(self, trade_id: TradeId) -> Optional[Trade]:
        """Получение сделки."""
        # Упрощенная реализация
        return None

    async def create_position(
        self,
```

#### Строка 472: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
        if take_profit:
            position.take_profit = take_profit

        # Упрощенная реализация - нет метода save_position
        return position

    async def update_position(
```

#### Строка 483: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Position]:
        """Обновление позиции."""
        # Упрощенная реализация
        return None

    async def close_position(
```

#### Строка 484: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
    ) -> Optional[Position]:
        """Обновление позиции."""
        # Упрощенная реализация
        return None

    async def close_position(
        self,
```

#### Строка 495: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Закрытие позиции."""
        # Упрощенная реализация
        return True

    async def get_position(self, position_id: PositionId) -> Optional[Position]:
```

#### Строка 496: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
    ) -> bool:
        """Закрытие позиции."""
        # Упрощенная реализация
        return True

    async def get_position(self, position_id: PositionId) -> Optional[Position]:
        """Получение позиции."""
```

#### Строка 500: get_position
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python

    async def get_position(self, position_id: PositionId) -> Optional[Position]:
        """Получение позиции."""
        # Упрощенная реализация
        return None

    async def get_active_positions(
```

#### Строка 501: get_position
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
    async def get_position(self, position_id: PositionId) -> Optional[Position]:
        """Получение позиции."""
        # Упрощенная реализация
        return None

    async def get_active_positions(
        self, symbol: Optional[str] = None
```

#### Строка 507: get_position
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
        self, symbol: Optional[str] = None
    ) -> List[Position]:
        """Получение активных позиций."""
        # Упрощенная реализация
        return []

    async def get_position_history(
```

#### Строка 508: get_position
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python
    ) -> List[Position]:
        """Получение активных позиций."""
        # Упрощенная реализация
        return []

    async def get_position_history(
        self,
```

#### Строка 518: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
        limit: Optional[int] = None,
    ) -> List[Position]:
        """Получение истории позиций."""
        # Упрощенная реализация
        return []

    async def get_trading_statistics(
```

#### Строка 519: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python
    ) -> List[Position]:
        """Получение истории позиций."""
        # Упрощенная реализация
        return []

    async def get_trading_statistics(
        self,
```

#### Строка 528: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Получение торговой статистики."""
        # Упрощенная реализация
        trades: List[Any] = []  # Получение сделок
        
        if not trades:
```

#### Строка 556: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        total_pnl = sum(trade.pnl for trade in trades if hasattr(trade, 'pnl'))
        average_trade = total_pnl / total_trades if total_trades > 0 else 0.0
        
        best_trade = max((trade.pnl for trade in trades if hasattr(trade, 'pnl')), default=0.0)
        worst_trade = min((trade.pnl for trade in trades if hasattr(trade, 'pnl')), default=0.0)
        
        # Упрощенные расчеты
```

#### Строка 557: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        average_trade = total_pnl / total_trades if total_trades > 0 else 0.0
        
        best_trade = max((trade.pnl for trade in trades if hasattr(trade, 'pnl')), default=0.0)
        worst_trade = min((trade.pnl for trade in trades if hasattr(trade, 'pnl')), default=0.0)
        
        # Упрощенные расчеты
        profit_factor = 1.0 if total_pnl > 0 else 0.0
```

#### Строка 559: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенные'
**Код:**
```python
        best_trade = max((trade.pnl for trade in trades if hasattr(trade, 'pnl')), default=0.0)
        worst_trade = min((trade.pnl for trade in trades if hasattr(trade, 'pnl')), default=0.0)
        
        # Упрощенные расчеты
        profit_factor = 1.0 if total_pnl > 0 else 0.0
        sharpe_ratio = 1.0 if total_pnl > 0 else 0.0
        max_drawdown = 0.0
```

#### Строка 585: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Money]:
        """Расчет P&L."""
        # Упрощенная реализация
        trades: List[Any] = []  # Получение сделок
        
        total_pnl = sum(trade.pnl for trade in trades if hasattr(trade, 'pnl'))
```

#### Строка 603: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
        end_date: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Получение метрик риска."""
        # Упрощенная реализация
        return {
            "volatility": 0.0,
            "var_95": 0.0,
```

#### Строка 616: _calculate_max_drawdown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return 0'
**Код:**
```python
    def _calculate_max_drawdown(self, cumulative_pnl: List[float]) -> float:
        """Расчет максимальной просадки."""
        if not cumulative_pnl:
            return 0.0
        
        peak = cumulative_pnl[0]
        max_dd = 0.0
```

#### Строка 634: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return 0'
**Код:**
```python
    ) -> float:
        """Расчет коэффициента Шарпа."""
        if not returns:
            return 0.0
        
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
```

#### Строка 641: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return 0'
**Код:**
```python
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return 0.0
        
        return (avg_return - risk_free_rate) / std_dev

```

#### Строка 652: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Получение метрик производительности."""
        # Упрощенная реализация
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
```

#### Строка 666: get_position_analysis
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python

    async def get_position_analysis(self, position_id: PositionId) -> Dict[str, Any]:
        """Получение анализа позиции."""
        # Упрощенная реализация
        return {
            "position_id": str(position_id),
            "current_pnl": 0.0,
```

#### Строка 676: get_portfolio_risk_summary
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python

    async def get_portfolio_risk_summary(self, portfolio_id: PortfolioId) -> Dict[str, Any]:
        """Получение сводки рисков портфеля."""
        # Упрощенная реализация
        return {
            "portfolio_id": str(portfolio_id),
            "total_risk": 0.0,
```

### 📁 application\signal\session_signal_engine.py
Найдено проблем: 20

#### Строка 36: Unknown
**Класс:** SessionInfluenceSignal
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    reversal_probability: float = 0.0
    false_breakout_probability: float = 0.0
    # Метаданные
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
```

#### Строка 67: Unknown
**Класс:** SessionSignalEngine
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'заглушку'
**Код:**
```python
        config: Optional[Dict[str, Any]] = None,
    ):
        self.session_marker = session_marker or SessionMarker()
        # Создаем заглушку для registry, если не передана
        registry: Dict[str, Any] = {}  # Заглушка для registry
        self.session_analyzer = session_analyzer or SessionInfluenceAnalyzer(
            registry=registry, session_marker=self.session_marker  # type: ignore
```

#### Строка 68: Unknown
**Класс:** SessionSignalEngine
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'заглушка'
**Код:**
```python
    ):
        self.session_marker = session_marker or SessionMarker()
        # Создаем заглушку для registry, если не передана
        registry: Dict[str, Any] = {}  # Заглушка для registry
        self.session_analyzer = session_analyzer or SessionInfluenceAnalyzer(
            registry=registry, session_marker=self.session_marker  # type: ignore
        )
```

#### Строка 117: stop
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'debug'
**Код:**
```python
                await self._update_task
            except asyncio.CancelledError:
                # Корректная обработка отмены задачи
                logger.debug("Update task cancelled successfully")
        logger.info("SessionSignalEngine stopped")

    async def generate_signal(
```

#### Строка 141: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'debug'
**Код:**
```python
            # Получаем контекст сессий
            session_context = self.session_marker.get_session_context(timestamp)
            if not session_context.primary_session:
                logger.debug(f"No active session for {symbol} at {timestamp}")
                return None
            # Анализируем влияние сессии
            if market_data:
```

#### Строка 142: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            session_context = self.session_marker.get_session_context(timestamp)
            if not session_context.primary_session:
                logger.debug(f"No active session for {symbol} at {timestamp}")
                return None
            # Анализируем влияние сессии
            if market_data:
                # Преобразуем в DataFrame если нужно
```

#### Строка 155: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'basic'
**Код:**
```python
                )
            else:
                # Используем базовый анализ без рыночных данных
                influence_result = self._generate_basic_influence_result(
                    symbol, session_context, timestamp
                )
            if not influence_result:
```

#### Строка 159: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
                    symbol, session_context, timestamp
                )
            if not influence_result:
                return None
            # Создаем сигнал
            signal = self._create_signal_from_influence_result(influence_result)
            if signal and signal.confidence >= self.config["min_confidence_threshold"]:
```

#### Строка 175: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            return signal
        except Exception as e:
            logger.error(f"Error generating session signal for {symbol}: {e}")
            return None

    async def get_current_signals(self, symbol: str) -> List[SessionInfluenceSignal]:
        """Получение текущих сигналов для символа."""
```

#### Строка 180: get_current_signals
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python
    async def get_current_signals(self, symbol: str) -> List[SessionInfluenceSignal]:
        """Получение текущих сигналов для символа."""
        if symbol not in self.signals:
            return []
        # Фильтруем актуальные сигналы
        current_time = Timestamp.now()
        ttl_minutes = self.config["signal_ttl_minutes"]
```

#### Строка 197: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
        """Получение агрегированного сигнала для символа."""
        current_signals = await self.get_current_signals(symbol)
        if not current_signals:
            return None
        # Агрегируем сигналы по весу уверенности
        total_weight = 0.0
        weighted_score = 0.0
```

#### Строка 208: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            weighted_score += signal.score * weight
            weighted_confidence += signal.confidence * weight
        if total_weight == 0:
            return None
        # Вычисляем средневзвешенные значения
        avg_score = weighted_score / total_weight
        avg_confidence = weighted_confidence / total_weight
```

#### Строка 247: get_session_analysis
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
                    "analysis": None,
                }
            # Получаем последний сигнал
            latest_signal = max(current_signals, key=lambda s: s.timestamp)
            # Получаем сводку влияния
            influence_summary = self.session_analyzer.get_influence_summary(symbol)  # type: ignore
            analysis = {
```

#### Строка 253: get_session_analysis
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
            analysis = {
                "symbol": symbol,
                "has_signals": True,
                "latest_signal": latest_signal.to_dict(),
                "total_signals": len(current_signals),
                "avg_confidence": sum(s.confidence for s in current_signals) / len(current_signals),
                "tendency_distribution": {
```

#### Строка 313: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            return signal
        except Exception as e:
            logger.error(f"Error creating signal from influence result: {e}")
            return None

    def _generate_basic_influence_result(
        self, symbol: str, session_context: MarketSessionContext, timestamp: Timestamp
```

#### Строка 315: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'basic'
**Код:**
```python
            logger.error(f"Error creating signal from influence result: {e}")
            return None

    def _generate_basic_influence_result(
        self, symbol: str, session_context: MarketSessionContext, timestamp: Timestamp
    ) -> Optional[SessionInfluenceResult]:
        """Генерация базового результата влияния без рыночных данных."""
```

#### Строка 330: Unknown
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
                try:
                    session_type = SessionType(session_context.primary_session)
                except ValueError:
                    pass
            
            # Определяем фазу сессии
            session_phase = SessionPhase.MID_SESSION  # Используем существующую фазу
```

#### Строка 367: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'basic'
**Код:**
```python
            )
            return result
        except Exception as e:
            logger.error(f"Error generating basic influence result: {e}")
            return None

    def _store_signal(self, symbol: str, signal: SessionInfluenceSignal) -> None:
```

#### Строка 368: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            return result
        except Exception as e:
            logger.error(f"Error generating basic influence result: {e}")
            return None

    def _store_signal(self, symbol: str, signal: SessionInfluenceSignal) -> None:
        """Сохранение сигнала."""
```

#### Строка 424: _signal_update_loop
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
        """Цикл обновления сигналов."""
        while self._running:
            try:
                # Здесь можно добавить логику периодического обновления сигналов
                await asyncio.sleep(self.config["signal_update_interval_seconds"])
            except asyncio.CancelledError:
                break
```

### 📁 application\strategy_advisor\mirror_map_builder.py
Найдено проблем: 13

#### Строка 29: Unknown
**Класс:** MirrorMap
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    assets: List[str]
    mirror_map: Dict[str, List[str]]
    correlation_matrix: Optional[CorrelationMatrix] = None
    clusters: List[List[str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Timestamp = field(
        default_factory=Timestamp.now
```

#### Строка 30: Unknown
**Класс:** MirrorMap
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    mirror_map: Dict[str, List[str]]
    correlation_matrix: Optional[CorrelationMatrix] = None
    clusters: List[List[str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Timestamp = field(
        default_factory=Timestamp.now
    )
```

#### Строка 32: Unknown
**Класс:** MirrorMap
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    clusters: List[List[str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Timestamp = field(
        default_factory=Timestamp.now
    )

    def get_mirror_assets(self, asset: str) -> List[str]:
```

#### Строка 44: get_correlation
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
    def is_mirror_pair(self, asset1: str, asset2: str) -> bool:
        """Проверка, являются ли активы зеркальной парой."""
        mirror_assets = self.get_mirror_assets(asset1)
        return asset2 in mirror_assets

    def get_correlation(self, asset1: str, asset2: str) -> float:
        """Получение корреляции между активами."""
        if self.correlation_matrix:
            return self.correlation_matrix.get_correlation(asset1, asset2)
        return 0.0

```

#### Строка 48: get_correlation
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return 0'
**Код:**
```python
        """Получение корреляции между активами."""
        if self.correlation_matrix:
            return self.correlation_matrix.get_correlation(asset1, asset2)
        return 0.0

    def get_lag(self, asset1: str, asset2: str) -> int:
        """Получение лага между активами."""
```

#### Строка 50: get_lag
**Тип:** Возврат по умолчанию
**Описание:** Функция возвращает значение по умолчанию
**Код:**
```python
        """Получение корреляции между активами."""
        if self.correlation_matrix:
            return self.correlation_matrix.get_correlation(asset1, asset2)
        return 0.0

    def get_lag(self, asset1: str, asset2: str) -> int:
        """Получение лага между активами."""
        if self.correlation_matrix:
            return self.correlation_matrix.get_lag(asset1, asset2)
        return 0

```

#### Строка 54: get_lag
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return 0'
**Код:**
```python
        """Получение лага между активами."""
        if self.correlation_matrix:
            return self.correlation_matrix.get_lag(asset1, asset2)
        return 0

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
```

#### Строка 151: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'null'
**Код:**
```python
                    )
                    continue
                # Проверяем наличие данных
                if hasattr(series, 'isnull') and series.isnull().all():
                    logger.warning(f"All NaN data for asset {asset}")
                    continue
                valid_data[asset] = series
```

#### Строка 159: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
            return valid_data
        except Exception as e:
            logger.error(f"Error validating price data: {e}")
            return {}

    def _build_correlation_matrix_parallel(
        self, assets: List[str], price_data: Dict[str, pd.Series]
```

#### Строка 256: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            )
        except Exception as e:
            logger.error(f"Error processing asset pair {asset1}-{asset2}: {e}")
            return None

    def _build_mirror_map_from_matrix(
        self, correlation_matrix: CorrelationMatrix
```

#### Строка 286: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
            return mirror_map
        except Exception as e:
            logger.error(f"Error building mirror map from matrix: {e}")
            return {}

    def build_mirror_map(
        self,
```

#### Строка 425: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python
            return result
        except Exception as e:
            logger.error(f"Error getting mirror assets for strategy: {e}")
            return []

    def analyze_mirror_clusters(self, mirror_map: MirrorMap) -> Dict[str, Any]:
        """
```

#### Строка 476: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing mirror clusters: {e}")
            return {}

    def get_mirror_map_statistics(self) -> Dict[str, Any]:
        """Получение статистики построения карты зеркальных зависимостей."""
```

### 📁 application\symbol_selection\analytics.py
Найдено проблем: 1

#### Строка 90: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return 0'
**Код:**
```python
            return min(max(final_score, 0.0), 1.0)
        except Exception as e:
            self.logger.error(f"Error calculating opportunity score: {e}")
            return 0.5

    def get_market_data_for_phase(self, symbol: str) -> pd.DataFrame:
        """Получение рыночных данных для анализа фазы."""
```

### 📁 application\symbol_selection\cache.py
Найдено проблем: 7

#### Строка 31: should_update
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
    def should_update(self) -> bool:
        """Проверка необходимости обновления."""
        if not self._last_update:
            return True

        time_since_update = (datetime.now() - self._last_update).total_seconds()
        return time_since_update >= self.config.update_interval_seconds
```

#### Строка 73: get_cached_profile
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
                if current_time - timestamp < self.config.cache_ttl_seconds:
                    return profile

            return None

        except Exception as e:
            self.logger.error(f"Error getting cached profile for {symbol}: {e}")
```

#### Строка 77: get_cached_profile
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python

        except Exception as e:
            self.logger.error(f"Error getting cached profile for {symbol}: {e}")
            return None

    def calculate_cache_hit_rate(self) -> float:
        """Расчет hit rate кэша."""
```

#### Строка 83: calculate_cache_hit_rate
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return 0'
**Код:**
```python
        """Расчет hit rate кэша."""
        try:
            if not self._cache:
                return 0.0

            current_time = time.time()
            valid_entries = sum(
```

#### Строка 95: calculate_cache_hit_rate
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return 0'
**Код:**
```python
            return valid_entries / len(self._cache) if self._cache else 0.0

        except Exception:
            return 0.0

    def update_performance_metrics(
        self, processing_time: float, symbols_count: int
```

#### Строка 141: get_cache_stats
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python

        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {}

    def get_cached_profiles(self) -> Dict[str, SymbolProfile]:
        """Получение всех кэшированных профилей."""
```

#### Строка 154: get_cached_profiles
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
            return valid_profiles
        except Exception as e:
            self.logger.error(f"Error getting cached profiles: {e}")
            return {}

    def get_hit_rate(self) -> float:
        """Получение hit rate кэша."""
```

### 📁 application\symbol_selection\filters.py
Найдено проблем: 30

#### Строка 21: __init__
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
        self.config = config
        self.logger = logger.bind(name=self.__class__.__name__)

    def passes_basic_filters(self, profile: SymbolProfile) -> bool:
        """Проверка базовых фильтров."""
        try:
            # Проверяем минимальный opportunity score
```

#### Строка 35: passes_basic_filters
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
                )

            if score_value < self.config.min_opportunity_score:
                return False

            # Проверяем минимальный confidence threshold
            if hasattr(profile.opportunity_score, "confidence"):
```

#### Строка 44: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
                confidence_value = 1.0  # Дефолтное значение

            if confidence_value < self.config.min_confidence_threshold:
                return False

            return True

```

#### Строка 46: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            if confidence_value < self.config.min_confidence_threshold:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error in basic filters: {e}")
```

#### Строка 49: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'basic'
**Код:**
```python
            return True

        except Exception as e:
            self.logger.error(f"Error in basic filters: {e}")
            return False

    async def passes_correlation_filter(
```

#### Строка 50: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python

        except Exception as e:
            self.logger.error(f"Error in basic filters: {e}")
            return False

    async def passes_correlation_filter(
        self, profile: SymbolProfile, selected_profiles: List[SymbolProfile]
```

#### Строка 52: Unknown
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
            self.logger.error(f"Error in basic filters: {e}")
            return False

    async def passes_correlation_filter(
        self, profile: SymbolProfile, selected_profiles: List[SymbolProfile]
    ) -> bool:
        """Проверка корреляционного фильтра."""
```

#### Строка 58: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
        """Проверка корреляционного фильтра."""
        try:
            if not selected_profiles:
                return True

            # Простая проверка корреляции (в реальной системе используем CorrelationChain)
            # Пока пропускаем все
```

#### Строка 60: Unknown
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'простая'
**Код:**
```python
            if not selected_profiles:
                return True

            # Простая проверка корреляции (в реальной системе используем CorrelationChain)
            # Пока пропускаем все
            return True

```

#### Строка 62: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python

            # Простая проверка корреляции (в реальной системе используем CorrelationChain)
            # Пока пропускаем все
            return True

        except Exception as e:
            self.logger.error(f"Error in correlation filter: {e}")
```

#### Строка 66: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python

        except Exception as e:
            self.logger.error(f"Error in correlation filter: {e}")
            return True

    async def passes_pattern_memory_filter(self, profile: SymbolProfile) -> bool:
        """Проверка фильтра памяти паттернов."""
```

#### Строка 68: Unknown
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
            self.logger.error(f"Error in correlation filter: {e}")
            return True

    async def passes_pattern_memory_filter(self, profile: SymbolProfile) -> bool:
        """Проверка фильтра памяти паттернов."""
        try:
            # В реальной системе проверяем PatternMemory
```

#### Строка 73: passes_pattern_memory_filter
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
        try:
            # В реальной системе проверяем PatternMemory
            # Пока пропускаем все
            return True

        except Exception as e:
            self.logger.error(f"Error in pattern memory filter: {e}")
```

#### Строка 77: passes_pattern_memory_filter
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python

        except Exception as e:
            self.logger.error(f"Error in pattern memory filter: {e}")
            return True

    async def passes_session_filter(self, profile: SymbolProfile) -> bool:
        """Проверка фильтра сессий."""
```

#### Строка 79: passes_pattern_memory_filter
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
            self.logger.error(f"Error in pattern memory filter: {e}")
            return True

    async def passes_session_filter(self, profile: SymbolProfile) -> bool:
        """Проверка фильтра сессий."""
        try:
            # В реальной системе проверяем SessionEngine
```

#### Строка 84: passes_session_filter
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
        try:
            # В реальной системе проверяем SessionEngine
            # Пока пропускаем все
            return True

        except Exception as e:
            self.logger.error(f"Error in session filter: {e}")
```

#### Строка 88: passes_session_filter
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python

        except Exception as e:
            self.logger.error(f"Error in session filter: {e}")
            return True

    async def passes_liquidity_filter(self, profile: SymbolProfile) -> bool:
        """Проверка фильтра ликвидности."""
```

#### Строка 90: passes_session_filter
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
            self.logger.error(f"Error in session filter: {e}")
            return True

    async def passes_liquidity_filter(self, profile: SymbolProfile) -> bool:
        """Проверка фильтра ликвидности."""
        try:
            # Проверяем score ликвидности из профиля
```

#### Строка 99: passes_liquidity_filter
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python

        except Exception as e:
            self.logger.error(f"Error in liquidity filter: {e}")
            return True

    async def passes_reversal_filter(self, profile: SymbolProfile) -> bool:
        """Проверка фильтра предсказания разворотов."""
```

#### Строка 101: passes_liquidity_filter
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
            self.logger.error(f"Error in liquidity filter: {e}")
            return True

    async def passes_reversal_filter(self, profile: SymbolProfile) -> bool:
        """Проверка фильтра предсказания разворотов."""
        try:
            # В реальной системе проверяем ReversalPredictor
```

#### Строка 106: passes_reversal_filter
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
        try:
            # В реальной системе проверяем ReversalPredictor
            # Пока пропускаем все
            return True

        except Exception as e:
            self.logger.error(f"Error in reversal filter: {e}")
```

#### Строка 110: passes_reversal_filter
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python

        except Exception as e:
            self.logger.error(f"Error in reversal filter: {e}")
            return True

    async def apply_advanced_filters(
        self, profiles: List[SymbolProfile]
```

#### Строка 121: Unknown
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python

            for profile in profiles:
                # Базовые фильтры
                if not self.passes_basic_filters(profile):
                    continue

                # Корреляционный фильтр
```

#### Строка 126: Unknown
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python

                # Корреляционный фильтр
                if self.config.enable_correlation_filtering:
                    if not await self.passes_correlation_filter(
                        profile, filtered_profiles
                    ):
                        continue
```

#### Строка 133: Unknown
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python

                # Фильтр памяти паттернов
                if self.config.enable_pattern_memory_integration:
                    if not await self.passes_pattern_memory_filter(profile):
                        continue

                # Фильтр сессий
```

#### Строка 138: Unknown
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python

                # Фильтр сессий
                if self.config.enable_session_alignment:
                    if not await self.passes_session_filter(profile):
                        continue

                # Фильтр ликвидности
```

#### Строка 143: Unknown
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python

                # Фильтр ликвидности
                if self.config.enable_liquidity_gravity:
                    if not await self.passes_liquidity_filter(profile):
                        continue

                # Фильтр предсказания разворотов
```

#### Строка 148: Unknown
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python

                # Фильтр предсказания разворотов
                if self.config.enable_reversal_prediction:
                    if not await self.passes_reversal_filter(profile):
                        continue

                filtered_profiles.append(profile)
```

#### Строка 162: apply_filters
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'простая'
**Код:**
```python
    def apply_filters(self, profiles: dict) -> dict:
        """Применение фильтров к профилям символов."""
        try:
            # Простая реализация - возвращаем профили как есть
            # В реальной системе здесь должна быть логика фильтрации
            return profiles
        except Exception as e:
```

#### Строка 163: apply_filters
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
        """Применение фильтров к профилям символов."""
        try:
            # Простая реализация - возвращаем профили как есть
            # В реальной системе здесь должна быть логика фильтрации
            return profiles
        except Exception as e:
            self.logger.error(f"Error applying filters: {e}")
```

### 📁 application\symbol_selection\opportunity_selector.py
Найдено проблем: 6

#### Строка 89: stop
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'debug'
**Код:**
```python
                await self._update_task
            except asyncio.CancelledError:
                # Корректная обработка отмены задачи
                self.logger.debug("Update task cancelled successfully")
        self.logger.info("DOASS stopped")

    async def get_symbols_for_analysis(self, limit: int = 10) -> List[str]:
```

#### Строка 114: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python
            return result.selected_symbols
        except Exception as e:
            self.logger.error(f"Error getting symbols for analysis: {e}")
            return []

    async def get_detailed_analysis(self, limit: int = 10) -> SymbolSelectionResult:
        """
```

#### Строка 190: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return []'
**Код:**
```python
            return symbols[: self.config.max_symbols_per_cycle]
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []

    async def _analyze_symbols_parallel(
        self, symbols: List[str]
```

#### Строка 218: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
            return profiles
        except Exception as e:
            self.logger.error(f"Error in parallel analysis: {e}")
            return {}

    async def _analyze_symbols_sequential(
        self, symbols: List[str]
```

#### Строка 235: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
            return profiles
        except Exception as e:
            self.logger.error(f"Error in sequential analysis: {e}")
            return {}

    async def _analyze_single_symbol(self, symbol: str) -> SymbolProfile:
        """Анализ одного символа."""
```

#### Строка 252: _analyze_single_symbol
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
            market_phase = MarketPhase.NO_STRUCTURE
            try:
                # Исправляем вызов classify_market_phase - передаем MarketDataFrame вместо str
                market_data = None  # Здесь должен быть MarketDataFrame
                if market_data:
                    market_phase_result = self.market_phase_classifier.classify_market_phase(market_data)
                    market_phase = market_phase_result.phase if hasattr(market_phase_result, 'phase') else MarketPhase.NO_STRUCTURE
```

### 📁 application\symbol_selection\types.py
Найдено проблем: 17

#### Строка 64: Unknown
**Класс:** SymbolSelectionResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class SymbolSelectionResult:
    """Результат выбора символов с продвинутой аналитикой."""

    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)
    # Основные результаты
    selected_symbols: List[str] = field(default_factory=list)
```

#### Строка 65: Unknown
**Класс:** SymbolSelectionResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    """Результат выбора символов с продвинутой аналитикой."""

    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)
    # Основные результаты
    selected_symbols: List[str] = field(default_factory=list)
    opportunity_scores: Dict[str, float] = field(default_factory=dict)
```

#### Строка 67: Unknown
**Класс:** SymbolSelectionResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)
    # Основные результаты
    selected_symbols: List[str] = field(default_factory=list)
    opportunity_scores: Dict[str, float] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    market_phases: Dict[str, MarketPhase] = field(default_factory=dict)
```

#### Строка 68: Unknown
**Класс:** SymbolSelectionResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    timestamp: datetime = field(default_factory=datetime.now)
    # Основные результаты
    selected_symbols: List[str] = field(default_factory=list)
    opportunity_scores: Dict[str, float] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    market_phases: Dict[str, MarketPhase] = field(default_factory=dict)
    # Продвинутая аналитика
```

#### Строка 69: Unknown
**Класс:** SymbolSelectionResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    # Основные результаты
    selected_symbols: List[str] = field(default_factory=list)
    opportunity_scores: Dict[str, float] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    market_phases: Dict[str, MarketPhase] = field(default_factory=dict)
    # Продвинутая аналитика
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
```

#### Строка 70: Unknown
**Класс:** SymbolSelectionResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    selected_symbols: List[str] = field(default_factory=list)
    opportunity_scores: Dict[str, float] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    market_phases: Dict[str, MarketPhase] = field(default_factory=dict)
    # Продвинутая аналитика
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    entanglement_groups: List[List[str]] = field(default_factory=list)
```

#### Строка 72: Unknown
**Класс:** SymbolSelectionResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    market_phases: Dict[str, MarketPhase] = field(default_factory=dict)
    # Продвинутая аналитика
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    entanglement_groups: List[List[str]] = field(default_factory=list)
    pattern_memory_insights: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    session_alignment_scores: Dict[str, float] = field(default_factory=dict)
```

#### Строка 73: Unknown
**Класс:** SymbolSelectionResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    market_phases: Dict[str, MarketPhase] = field(default_factory=dict)
    # Продвинутая аналитика
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    entanglement_groups: List[List[str]] = field(default_factory=list)
    pattern_memory_insights: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    session_alignment_scores: Dict[str, float] = field(default_factory=dict)
    liquidity_gravity_scores: Dict[str, float] = field(default_factory=dict)
```

#### Строка 74: Unknown
**Класс:** SymbolSelectionResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    # Продвинутая аналитика
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    entanglement_groups: List[List[str]] = field(default_factory=list)
    pattern_memory_insights: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    session_alignment_scores: Dict[str, float] = field(default_factory=dict)
    liquidity_gravity_scores: Dict[str, float] = field(default_factory=dict)
    reversal_probabilities: Dict[str, float] = field(default_factory=dict)
```

#### Строка 75: Unknown
**Класс:** SymbolSelectionResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    entanglement_groups: List[List[str]] = field(default_factory=list)
    pattern_memory_insights: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    session_alignment_scores: Dict[str, float] = field(default_factory=dict)
    liquidity_gravity_scores: Dict[str, float] = field(default_factory=dict)
    reversal_probabilities: Dict[str, float] = field(default_factory=dict)
    # Метаданные
```

#### Строка 76: Unknown
**Класс:** SymbolSelectionResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    entanglement_groups: List[List[str]] = field(default_factory=list)
    pattern_memory_insights: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    session_alignment_scores: Dict[str, float] = field(default_factory=dict)
    liquidity_gravity_scores: Dict[str, float] = field(default_factory=dict)
    reversal_probabilities: Dict[str, float] = field(default_factory=dict)
    # Метаданные
    processing_time_ms: float = 0.0
```

#### Строка 77: Unknown
**Класс:** SymbolSelectionResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    pattern_memory_insights: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    session_alignment_scores: Dict[str, float] = field(default_factory=dict)
    liquidity_gravity_scores: Dict[str, float] = field(default_factory=dict)
    reversal_probabilities: Dict[str, float] = field(default_factory=dict)
    # Метаданные
    processing_time_ms: float = 0.0
    total_symbols_analyzed: int = 0
```

#### Строка 82: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    processing_time_ms: float = 0.0
    total_symbols_analyzed: int = 0
    cache_hit_rate: float = 0.0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    # Детальная информация
    detailed_profiles: Dict[str, SymbolProfile] = field(default_factory=dict)
    rejection_reasons: Dict[str, List[str]] = field(default_factory=dict)
```

#### Строка 84: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    cache_hit_rate: float = 0.0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    # Детальная информация
    detailed_profiles: Dict[str, SymbolProfile] = field(default_factory=dict)
    rejection_reasons: Dict[str, List[str]] = field(default_factory=dict)
    optimization_suggestions: List[str] = field(default_factory=list)
    # Дополнительные поля для совместимости
```

#### Строка 85: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    # Детальная информация
    detailed_profiles: Dict[str, SymbolProfile] = field(default_factory=dict)
    rejection_reasons: Dict[str, List[str]] = field(default_factory=dict)
    optimization_suggestions: List[str] = field(default_factory=list)
    # Дополнительные поля для совместимости
    profiles: List[SymbolProfile] = field(default_factory=list)
```

#### Строка 86: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    # Детальная информация
    detailed_profiles: Dict[str, SymbolProfile] = field(default_factory=dict)
    rejection_reasons: Dict[str, List[str]] = field(default_factory=dict)
    optimization_suggestions: List[str] = field(default_factory=list)
    # Дополнительные поля для совместимости
    profiles: List[SymbolProfile] = field(default_factory=list)
    total_analyzed: int = 0
```

#### Строка 88: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    rejection_reasons: Dict[str, List[str]] = field(default_factory=dict)
    optimization_suggestions: List[str] = field(default_factory=list)
    # Дополнительные поля для совместимости
    profiles: List[SymbolProfile] = field(default_factory=list)
    total_analyzed: int = 0
    total_filtered: int = 0
    total_selected: int = 0
```

### 📁 application\types.py
Найдено проблем: 166

#### Строка 62: Unknown
**Класс:** MarketSummary
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
# Базовые типы для совместимости с тестами
class MarketSummary:
    """Сводка рынка."""
    pass

class PriceLevel:
    """Уровень цены."""
```

#### Строка 66: Unknown
**Класс:** MarketSummary
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python

class PriceLevel:
    """Уровень цены."""
    pass

class VolumeLevel:
    """Уровень объема."""
```

#### Строка 70: Unknown
**Класс:** MarketSummary
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python

class VolumeLevel:
    """Уровень объема."""
    pass

class MoneyAmount:
    """Денежная сумма."""
```

#### Строка 74: Unknown
**Класс:** MarketSummary
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python

class MoneyAmount:
    """Денежная сумма."""
    pass

class Timestamp:
    """Временная метка."""
```

#### Строка 78: Unknown
**Класс:** MarketSummary
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python

class Timestamp:
    """Временная метка."""
    pass

# Новые типы для строгой типизации
ParameterValue = Union[
```

#### Строка 254: Unknown
**Класс:** StrategyExecutionStatus
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class BaseRequest:
    """Базовый класс для всех запросов application слоя."""

    request_id: UUID = field(default_factory=uuid4)
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
```

#### Строка 256: Unknown
**Класс:** StrategyExecutionStatus
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python

    request_id: UUID = field(default_factory=uuid4)
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))

```

#### Строка 258: Unknown
**Класс:** BaseRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass
```

#### Строка 268: Unknown
**Класс:** BaseRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    success: bool = True
    message: str = ""
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
```

#### Строка 270: Unknown
**Класс:** BaseRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))

```

#### Строка 271: Unknown
**Класс:** BaseResponse
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: TimestampValue(datetime.now())
    )
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


```

#### Строка 272: Unknown
**Класс:** BaseResponse
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    )
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass
```

#### Строка 291: Unknown
**Класс:** PaginatedRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python

    total_count: int = 0
    has_more: bool = False
    page_info: Dict[str, Union[int, bool, str]] = field(default_factory=dict)


# ============================================================================
```

#### Строка 321: Unknown
**Класс:** CreateOrderResponse
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    exchange_order_id: Optional[str] = None
    order: Optional[Order] = None
    estimated_cost: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )


```

#### Строка 360: Unknown
**Класс:** GetOrdersRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class GetOrdersResponse(PaginatedResponse):
    """Ответ с ордерами."""

    orders: List[Order] = field(default_factory=list)
    total_value: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
```

#### Строка 362: Unknown
**Класс:** GetOrdersRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python

    orders: List[Order] = field(default_factory=list)
    total_value: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )


```

#### Строка 404: Unknown
**Класс:** UpdatePositionRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    """Ответ на обновление позиции."""

    position: Optional[Position] = None
    changes: Dict[str, Union[str, float, int, bool]] = field(default_factory=dict)
    updated: bool = False


```

#### Строка 426: Unknown
**Класс:** ClosePositionRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    close_price: Optional[Price] = None
    closed: bool = False
    closed_volume: Volume = field(
        default_factory=lambda: Volume(Decimal("0"), Currency.USDT)
    )
    realized_pnl: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
```

#### Строка 429: Unknown
**Класс:** ClosePositionResponse
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: Volume(Decimal("0"), Currency.USDT)
    )
    realized_pnl: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )


```

#### Строка 448: Unknown
**Класс:** GetPositionsRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class GetPositionsResponse(PaginatedResponse):
    """Ответ с позициями."""

    positions: List[Position] = field(default_factory=list)
    total_pnl: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USDT))
    unrealized_pnl: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
```

#### Строка 449: Unknown
**Класс:** GetPositionsRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    """Ответ с позициями."""

    positions: List[Position] = field(default_factory=list)
    total_pnl: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USDT))
    unrealized_pnl: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
```

#### Строка 451: Unknown
**Класс:** GetPositionsRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    positions: List[Position] = field(default_factory=list)
    total_pnl: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USDT))
    unrealized_pnl: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )


```

#### Строка 472: Unknown
**Класс:** PositionMetrics
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    total_pnl: Optional[Money] = None
    is_open: bool = True
    created_at: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    updated_at: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
```

#### Строка 475: Unknown
**Класс:** PositionMetrics
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: TimestampValue(datetime.now())
    )
    updated_at: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    days_held: int = 0
    avg_entry_price: Optional[PriceValue] = None
```

#### Строка 483: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    margin_used: Optional[Money] = None
    leverage: Decimal = Decimal("1")
    liquidation_price: Optional[PriceValue] = None
    risk_metrics: RiskMetricsDict = field(default_factory=dict)


# ============================================================================
```

#### Строка 505: Unknown
**Класс:** RiskAssessmentRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class RiskAssessmentResponse(BaseResponse):
    """Ответ с оценкой риска."""

    portfolio_risk: RiskMetricsDict = field(default_factory=dict)
    position_risks: List[RiskMetricsDict] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    risk_score: Decimal = Decimal("0")
```

#### Строка 506: Unknown
**Класс:** RiskAssessmentRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    """Ответ с оценкой риска."""

    portfolio_risk: RiskMetricsDict = field(default_factory=dict)
    position_risks: List[RiskMetricsDict] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    risk_score: Decimal = Decimal("0")
    is_acceptable: bool = True
```

#### Строка 507: Unknown
**Класс:** RiskAssessmentRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python

    portfolio_risk: RiskMetricsDict = field(default_factory=dict)
    position_risks: List[RiskMetricsDict] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    risk_score: Decimal = Decimal("0")
    is_acceptable: bool = True
    var_95: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USDT))
```

#### Строка 510: Unknown
**Класс:** RiskAssessmentResponse
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    recommendations: List[str] = field(default_factory=list)
    risk_score: Decimal = Decimal("0")
    is_acceptable: bool = True
    var_95: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USDT))
    var_99: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USDT))
    max_drawdown: Decimal = Decimal("0")

```

#### Строка 511: Unknown
**Класс:** RiskAssessmentResponse
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    risk_score: Decimal = Decimal("0")
    is_acceptable: bool = True
    var_95: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USDT))
    var_99: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USDT))
    max_drawdown: Decimal = Decimal("0")


```

#### Строка 532: Unknown
**Класс:** RiskLimitRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class RiskLimitResponse(BaseResponse):
    """Ответ с лимитами риска."""

    current_risk: RiskMetricsDict = field(default_factory=dict)
    risk_limits: RiskMetricsDict = field(default_factory=dict)
    limits_set: bool = False

```

#### Строка 533: Unknown
**Класс:** RiskLimitRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    """Ответ с лимитами риска."""

    current_risk: RiskMetricsDict = field(default_factory=dict)
    risk_limits: RiskMetricsDict = field(default_factory=dict)
    limits_set: bool = False


```

#### Строка 598: Unknown
**Класс:** GetTradingPairsRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class GetTradingPairsResponse(PaginatedResponse):
    """Ответ с торговыми парами."""

    trading_pairs: List[TradingPair] = field(default_factory=list)


@dataclass(kw_only=True)
```

#### Строка 605: Unknown
**Класс:** GetTradingPairsResponse
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class TradingPairMetrics:
    """Метрики торговой пары."""

    volume_24h: VolumeValue = field(default_factory=lambda: VolumeValue(Decimal("0")))
    price_change_24h: PriceValue = field(
        default_factory=lambda: PriceValue(Decimal("0"))
    )
```

#### Строка 607: Unknown
**Класс:** GetTradingPairsResponse
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python

    volume_24h: VolumeValue = field(default_factory=lambda: VolumeValue(Decimal("0")))
    price_change_24h: PriceValue = field(
        default_factory=lambda: PriceValue(Decimal("0"))
    )
    price_change_percent_24h: Percentage = field(
        default_factory=lambda: Percentage(Decimal("0"))
```

#### Строка 610: Unknown
**Класс:** GetTradingPairsResponse
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: PriceValue(Decimal("0"))
    )
    price_change_percent_24h: Percentage = field(
        default_factory=lambda: Percentage(Decimal("0"))
    )
    high_24h: PriceValue = field(default_factory=lambda: PriceValue(Decimal("0")))
    low_24h: PriceValue = field(default_factory=lambda: PriceValue(Decimal("0")))
```

#### Строка 612: Unknown
**Класс:** GetTradingPairsResponse
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    price_change_percent_24h: Percentage = field(
        default_factory=lambda: Percentage(Decimal("0"))
    )
    high_24h: PriceValue = field(default_factory=lambda: PriceValue(Decimal("0")))
    low_24h: PriceValue = field(default_factory=lambda: PriceValue(Decimal("0")))
    last_price: PriceValue = field(default_factory=lambda: PriceValue(Decimal("0")))
    bid: Optional[PriceValue] = None
```

#### Строка 613: Unknown
**Класс:** GetTradingPairsResponse
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: Percentage(Decimal("0"))
    )
    high_24h: PriceValue = field(default_factory=lambda: PriceValue(Decimal("0")))
    low_24h: PriceValue = field(default_factory=lambda: PriceValue(Decimal("0")))
    last_price: PriceValue = field(default_factory=lambda: PriceValue(Decimal("0")))
    bid: Optional[PriceValue] = None
    ask: Optional[PriceValue] = None
```

#### Строка 614: Unknown
**Класс:** GetTradingPairsResponse
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    )
    high_24h: PriceValue = field(default_factory=lambda: PriceValue(Decimal("0")))
    low_24h: PriceValue = field(default_factory=lambda: PriceValue(Decimal("0")))
    last_price: PriceValue = field(default_factory=lambda: PriceValue(Decimal("0")))
    bid: Optional[PriceValue] = None
    ask: Optional[PriceValue] = None
    spread: Optional[PriceValue] = None
```

#### Строка 627: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
# STRATEGY EXECUTION TYPES
# ============================================================================
@dataclass(kw_only=True)
class ExecuteStrategyRequest(BaseRequest):
    """Запрос на выполнение стратегии."""

    strategy_id: StrategyId
```

#### Строка 636: Unknown
**Класс:** ExecuteStrategyRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    amount: Optional[VolumeValue] = None
    risk_level: Optional[RiskLevel] = None
    use_sentiment_analysis: bool = True
    parameters: ParameterDict = field(default_factory=dict)


@dataclass(kw_only=True)
```

#### Строка 640: Unknown
**Класс:** ExecuteStrategyRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python


@dataclass(kw_only=True)
class ExecuteStrategyResponse(BaseResponse):
    """Ответ на выполнение стратегии."""

    orders_created: List[Order] = field(default_factory=list)
```

#### Строка 643: Unknown
**Класс:** ExecuteStrategyRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class ExecuteStrategyResponse(BaseResponse):
    """Ответ на выполнение стратегии."""

    orders_created: List[Order] = field(default_factory=list)
    signals_generated: List[Signal] = field(default_factory=list)
    sentiment_analysis: Optional[Dict[str, Union[str, float, int]]] = None
    execution_time_ms: float = 0.0
```

#### Строка 644: Unknown
**Класс:** ExecuteStrategyRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    """Ответ на выполнение стратегии."""

    orders_created: List[Order] = field(default_factory=list)
    signals_generated: List[Signal] = field(default_factory=list)
    sentiment_analysis: Optional[Dict[str, Union[str, float, int]]] = None
    execution_time_ms: float = 0.0
    executed: bool = False
```

#### Строка 665: Unknown
**Класс:** ProcessSignalRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class ProcessSignalResponse(BaseResponse):
    """Ответ на обработку сигнала."""

    orders_created: List[Order] = field(default_factory=list)
    sentiment_analysis: Optional[Dict[str, Union[str, float, int]]] = None
    risk_assessment: Optional[RiskMetricsDict] = None
    processed: bool = False
```

#### Строка 679: Unknown
**Класс:** ProcessSignalResponse
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    """Запрос на ребалансировку портфеля."""

    portfolio_id: PortfolioId
    target_weights: Dict[Symbol, Decimal] = field(default_factory=dict)
    tolerance: Decimal = Decimal("0.05")
    use_sentiment_analysis: bool = True
    rebalance_strategy: str = "optimal"
```

#### Строка 689: Unknown
**Класс:** PortfolioRebalanceRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class PortfolioRebalanceResponse(BaseResponse):
    """Ответ на ребалансировку портфеля."""

    orders_created: List[Order] = field(default_factory=list)
    current_weights: Dict[Symbol, Decimal] = field(default_factory=dict)
    target_weights: Dict[Symbol, Decimal] = field(default_factory=dict)
    sentiment_analysis: Optional[Dict[str, Union[str, float, int]]] = None
```

#### Строка 690: Unknown
**Класс:** PortfolioRebalanceRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    """Ответ на ребалансировку портфеля."""

    orders_created: List[Order] = field(default_factory=list)
    current_weights: Dict[Symbol, Decimal] = field(default_factory=dict)
    target_weights: Dict[Symbol, Decimal] = field(default_factory=dict)
    sentiment_analysis: Optional[Dict[str, Union[str, float, int]]] = None
    rebalance_cost: Money = field(
```

#### Строка 691: Unknown
**Класс:** PortfolioRebalanceRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python

    orders_created: List[Order] = field(default_factory=list)
    current_weights: Dict[Symbol, Decimal] = field(default_factory=dict)
    target_weights: Dict[Symbol, Decimal] = field(default_factory=dict)
    sentiment_analysis: Optional[Dict[str, Union[str, float, int]]] = None
    rebalance_cost: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
```

#### Строка 694: Unknown
**Класс:** PortfolioRebalanceRequest
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    target_weights: Dict[Symbol, Decimal] = field(default_factory=dict)
    sentiment_analysis: Optional[Dict[str, Union[str, float, int]]] = None
    rebalance_cost: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    rebalanced: bool = False

```

#### Строка 707: Unknown
**Класс:** TradingSession
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    """Сессия торговли."""

    session_id: str = ""
    portfolio_id: PortfolioId = field(default_factory=lambda: PortfolioId(uuid4()))
    start_time: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
```

#### Строка 709: Unknown
**Класс:** TradingSession
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    session_id: str = ""
    portfolio_id: PortfolioId = field(default_factory=lambda: PortfolioId(uuid4()))
    start_time: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    end_time: Optional[TimestampValue] = None
    orders_created: List[Order] = field(default_factory=list)
```

#### Строка 712: Unknown
**Класс:** TradingSession
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: TimestampValue(datetime.now())
    )
    end_time: Optional[TimestampValue] = None
    orders_created: List[Order] = field(default_factory=list)
    trades_executed: List[Trade] = field(default_factory=list)
    pnl: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USDT))
    status: SessionStatus = SessionStatus.ACTIVE
```

#### Строка 713: Unknown
**Класс:** TradingSession
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    )
    end_time: Optional[TimestampValue] = None
    orders_created: List[Order] = field(default_factory=list)
    trades_executed: List[Trade] = field(default_factory=list)
    pnl: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USDT))
    status: SessionStatus = SessionStatus.ACTIVE
    strategy_id: Optional[StrategyId] = None
```

#### Строка 714: Unknown
**Класс:** TradingSession
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    end_time: Optional[TimestampValue] = None
    orders_created: List[Order] = field(default_factory=list)
    trades_executed: List[Trade] = field(default_factory=list)
    pnl: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USDT))
    status: SessionStatus = SessionStatus.ACTIVE
    strategy_id: Optional[StrategyId] = None
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))
```

#### Строка 717: Unknown
**Класс:** TradingSession
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    pnl: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USDT))
    status: SessionStatus = SessionStatus.ACTIVE
    strategy_id: Optional[StrategyId] = None
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
```

#### Строка 729: Unknown
**Класс:** SessionMetrics
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    losing_trades: int = 0
    win_rate: Decimal = Decimal("0")
    avg_trade_pnl: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    max_drawdown: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
```

#### Строка 732: Unknown
**Класс:** SessionMetrics
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    max_drawdown: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    sharpe_ratio: Decimal = Decimal("0")
    total_volume: VolumeValue = field(default_factory=lambda: VolumeValue(Decimal("0")))
```

#### Строка 735: Unknown
**Класс:** SessionMetrics
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    sharpe_ratio: Decimal = Decimal("0")
    total_volume: VolumeValue = field(default_factory=lambda: VolumeValue(Decimal("0")))


# ============================================================================
```

#### Строка 743: Unknown
**Класс:** ServiceConfig
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'базовая'
**Код:**
```python
# ============================================================================
@dataclass(kw_only=True)
class ServiceConfig:
    """Базовая конфигурация сервиса."""

    enabled: bool = True
    timeout_seconds: int = 30
```

#### Строка 794: Unknown
**Класс:** MarketAnalysis
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    sentiment_score: Decimal
    confidence: ConfidenceLevel
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )


```

#### Строка 809: Unknown
**Класс:** TradingSignal
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    price: PriceValue
    timestamp: TimestampValue
    confidence: ConfidenceLevel
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
```

#### Строка 827: Unknown
**Класс:** PortfolioMetrics
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    alpha: Decimal
    correlation_matrix: Dict[Symbol, Dict[Symbol, Decimal]]
    sector_allocation: Dict[str, Money]
    risk_metrics: RiskMetricsDict = field(default_factory=dict)
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
```

#### Строка 829: Unknown
**Класс:** PortfolioMetrics
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    sector_allocation: Dict[str, Money]
    risk_metrics: RiskMetricsDict = field(default_factory=dict)
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )


```

#### Строка 837: Unknown
**Класс:** TechnicalIndicators
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class TechnicalIndicators:
    """Технические индикаторы."""

    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
```

#### Строка 839: Unknown
**Класс:** TechnicalIndicators
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python

    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    sma_20: Optional[Decimal] = None
    sma_50: Optional[Decimal] = None
```

#### Строка 852: Unknown
**Класс:** TechnicalIndicators
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    bollinger_middle: Optional[Decimal] = None
    bollinger_lower: Optional[Decimal] = None
    atr: Optional[Decimal] = None
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
```

#### Строка 859: Unknown
**Класс:** VolumeProfile
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class VolumeProfile:
    """Профиль объема."""

    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
```

#### Строка 861: Unknown
**Класс:** VolumeProfile
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python

    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    poc_price: Optional[PriceValue] = None  # Point of Control
    value_areas: List[Dict[str, Union[float, str]]] = field(default_factory=list)
```

#### Строка 864: Unknown
**Класс:** VolumeProfile
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: TimestampValue(datetime.now())
    )
    poc_price: Optional[PriceValue] = None  # Point of Control
    value_areas: List[Dict[str, Union[float, str]]] = field(default_factory=list)
    volume_nodes: List[Dict[str, Union[float, str]]] = field(default_factory=list)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))

```

#### Строка 865: Unknown
**Класс:** VolumeProfile
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    )
    poc_price: Optional[PriceValue] = None  # Point of Control
    value_areas: List[Dict[str, Union[float, str]]] = field(default_factory=list)
    volume_nodes: List[Dict[str, Union[float, str]]] = field(default_factory=list)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


```

#### Строка 866: Unknown
**Класс:** VolumeProfile
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    poc_price: Optional[PriceValue] = None  # Point of Control
    value_areas: List[Dict[str, Union[float, str]]] = field(default_factory=list)
    volume_nodes: List[Dict[str, Union[float, str]]] = field(default_factory=list)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
```

#### Строка 873: Unknown
**Класс:** VolumeProfile
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class SupportResistanceLevels:
    """Уровни поддержки и сопротивления."""

    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
```

#### Строка 875: Unknown
**Класс:** VolumeProfile
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python

    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    support_levels: List[PriceValue] = field(default_factory=list)
    resistance_levels: List[PriceValue] = field(default_factory=list)
```

#### Строка 877: Unknown
**Класс:** SupportResistanceLevels
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    support_levels: List[PriceValue] = field(default_factory=list)
    resistance_levels: List[PriceValue] = field(default_factory=list)
    strength_scores: Dict[str, Decimal] = field(default_factory=dict)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))
```

#### Строка 878: Unknown
**Класс:** SupportResistanceLevels
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: TimestampValue(datetime.now())
    )
    support_levels: List[PriceValue] = field(default_factory=list)
    resistance_levels: List[PriceValue] = field(default_factory=list)
    strength_scores: Dict[str, Decimal] = field(default_factory=dict)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))

```

#### Строка 879: Unknown
**Класс:** SupportResistanceLevels
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    )
    support_levels: List[PriceValue] = field(default_factory=list)
    resistance_levels: List[PriceValue] = field(default_factory=list)
    strength_scores: Dict[str, Decimal] = field(default_factory=dict)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


```

#### Строка 880: Unknown
**Класс:** SupportResistanceLevels
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    support_levels: List[PriceValue] = field(default_factory=list)
    resistance_levels: List[PriceValue] = field(default_factory=list)
    strength_scores: Dict[str, Decimal] = field(default_factory=dict)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
```

#### Строка 887: Unknown
**Класс:** SupportResistanceLevels
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class MarketRegime:
    """Рыночный режим."""

    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
```

#### Строка 889: Unknown
**Класс:** SupportResistanceLevels
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python

    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    regime_type: str = ""  # "trending", "ranging", "volatile", "quiet"
    confidence: ConfidenceLevel = field(
```

#### Строка 893: Unknown
**Класс:** MarketRegime
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    )
    regime_type: str = ""  # "trending", "ranging", "volatile", "quiet"
    confidence: ConfidenceLevel = field(
        default_factory=lambda: ConfidenceLevel(Decimal("0"))
    )
    volatility: Decimal = Decimal("0")
    trend_strength: Decimal = Decimal("0")
```

#### Строка 897: Unknown
**Класс:** MarketRegime
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    )
    volatility: Decimal = Decimal("0")
    trend_strength: Decimal = Decimal("0")
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
```

#### Строка 904: Unknown
**Класс:** OrderBookSnapshot
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class OrderBookSnapshot:
    """Снимок ордербука."""

    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
```

#### Строка 906: Unknown
**Класс:** OrderBookSnapshot
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python

    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
```

#### Строка 908: Unknown
**Класс:** OrderBookSnapshot
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    spread: Optional[PriceValue] = None
    depth: Optional[Dict[str, Union[float, int]]] = None
```

#### Строка 909: Unknown
**Класс:** OrderBookSnapshot
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: TimestampValue(datetime.now())
    )
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    spread: Optional[PriceValue] = None
    depth: Optional[Dict[str, Union[float, int]]] = None
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))
```

#### Строка 912: Unknown
**Класс:** OrderBookSnapshot
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    asks: List[OrderBookLevel] = field(default_factory=list)
    spread: Optional[PriceValue] = None
    depth: Optional[Dict[str, Union[float, int]]] = None
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


# ============================================================================
```

#### Строка 922: Unknown
**Класс:** PortfolioSummary
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class PortfolioSummary:
    """Сводка портфеля."""

    portfolio_id: PortfolioId = field(default_factory=lambda: PortfolioId(uuid4()))
    total_balance: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
```

#### Строка 924: Unknown
**Класс:** PortfolioSummary
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python

    portfolio_id: PortfolioId = field(default_factory=lambda: PortfolioId(uuid4()))
    total_balance: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    available_balance: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
```

#### Строка 927: Unknown
**Класс:** PortfolioSummary
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    available_balance: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    total_equity: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
```

#### Строка 930: Unknown
**Класс:** PortfolioSummary
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    total_equity: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    unrealized_pnl: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
```

#### Строка 933: Unknown
**Класс:** PortfolioSummary
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    unrealized_pnl: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    realized_pnl: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
```

#### Строка 936: Unknown
**Класс:** PortfolioSummary
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    realized_pnl: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    margin_used: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
```

#### Строка 939: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    margin_used: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    margin_level: Percentage = field(default_factory=lambda: Percentage(Decimal("0")))
    open_positions: int = 0
```

#### Строка 941: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    margin_used: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    margin_level: Percentage = field(default_factory=lambda: Percentage(Decimal("0")))
    open_positions: int = 0
    open_orders: int = 0
    timestamp: TimestampValue = field(
```

#### Строка 945: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    open_positions: int = 0
    open_orders: int = 0
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))

```

#### Строка 947: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
```

#### Строка 954: Unknown
**Класс:** RiskMetrics
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class RiskMetrics:
    """Метрики риска."""

    portfolio_id: PortfolioId = field(default_factory=lambda: PortfolioId(uuid4()))
    var_95: Decimal = Decimal("0")
    var_99: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
```

#### Строка 964: Unknown
**Класс:** RiskMetrics
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    beta: Decimal = Decimal("0")
    correlation: Decimal = Decimal("0")
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))

```

#### Строка 966: Unknown
**Класс:** RiskMetrics
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
```

#### Строка 973: Unknown
**Класс:** PerformanceMetrics
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class PerformanceMetrics:
    """Метрики производительности."""

    portfolio_id: PortfolioId = field(default_factory=lambda: PortfolioId(uuid4()))
    total_return: Percentage = field(default_factory=lambda: Percentage(Decimal("0")))
    annualized_return: Percentage = field(
        default_factory=lambda: Percentage(Decimal("0"))
```

#### Строка 974: Unknown
**Класс:** PerformanceMetrics
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    """Метрики производительности."""

    portfolio_id: PortfolioId = field(default_factory=lambda: PortfolioId(uuid4()))
    total_return: Percentage = field(default_factory=lambda: Percentage(Decimal("0")))
    annualized_return: Percentage = field(
        default_factory=lambda: Percentage(Decimal("0"))
    )
```

#### Строка 976: Unknown
**Класс:** PerformanceMetrics
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    portfolio_id: PortfolioId = field(default_factory=lambda: PortfolioId(uuid4()))
    total_return: Percentage = field(default_factory=lambda: Percentage(Decimal("0")))
    annualized_return: Percentage = field(
        default_factory=lambda: Percentage(Decimal("0"))
    )
    volatility: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
```

#### Строка 985: Unknown
**Класс:** PerformanceMetrics
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    win_rate: Decimal = Decimal("0")
    profit_factor: Decimal = Decimal("0")
    average_win: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    average_loss: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
```

#### Строка 988: Unknown
**Класс:** PerformanceMetrics
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    average_loss: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    largest_win: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
```

#### Строка 991: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    largest_win: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    largest_loss: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
```

#### Строка 994: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    largest_loss: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
```

#### Строка 997: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))

```

#### Строка 999: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
```

#### Строка 1006: Unknown
**Класс:** StrategyPerformance
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class StrategyPerformance:
    """Производительность стратегии."""

    strategy_id: StrategyId = field(default_factory=lambda: StrategyId(uuid4()))
    total_return: Percentage = field(default_factory=lambda: Percentage(Decimal("0")))
    sharpe_ratio: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
```

#### Строка 1007: Unknown
**Класс:** StrategyPerformance
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    """Производительность стратегии."""

    strategy_id: StrategyId = field(default_factory=lambda: StrategyId(uuid4()))
    total_return: Percentage = field(default_factory=lambda: Percentage(Decimal("0")))
    sharpe_ratio: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    win_rate: Decimal = Decimal("0")
```

#### Строка 1016: Unknown
**Класс:** StrategyPerformance
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    winning_trades: int = 0
    losing_trades: int = 0
    average_win: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    average_loss: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
```

#### Строка 1019: Unknown
**Класс:** StrategyPerformance
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    average_loss: Money = field(
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
```

#### Строка 1022: Unknown
**Класс:** StrategyPerformance
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: Money(Decimal("0"), Currency.USDT)
    )
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))

```

#### Строка 1024: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


# ============================================================================
```

#### Строка 1035: Unknown
**Класс:** MLPrediction
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    """ML предсказание."""

    model_id: str = ""
    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    prediction_type: str = ""
    predicted_value: Decimal = Decimal("0")
    confidence: ConfidenceLevel = field(
```

#### Строка 1039: Unknown
**Класс:** MLPrediction
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    prediction_type: str = ""
    predicted_value: Decimal = Decimal("0")
    confidence: ConfidenceLevel = field(
        default_factory=lambda: ConfidenceLevel(Decimal("0"))
    )
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
```

#### Строка 1042: Unknown
**Класс:** MLPrediction
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: ConfidenceLevel(Decimal("0"))
    )
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    features: Dict[str, Union[str, float, int, bool]] = field(default_factory=dict)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))
```

#### Строка 1044: Unknown
**Класс:** MLPrediction
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    features: Dict[str, Union[str, float, int, bool]] = field(default_factory=dict)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


```

#### Строка 1045: Unknown
**Класс:** MLPrediction
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: TimestampValue(datetime.now())
    )
    features: Dict[str, Union[str, float, int, bool]] = field(default_factory=dict)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
```

#### Строка 1057: Unknown
**Класс:** NewsItem
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    content: str = ""
    source: str = ""
    published_at: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    sentiment_score: Decimal = Decimal("0")
    relevance_score: Decimal = Decimal("0")
```

#### Строка 1062: Unknown
**Класс:** NewsItem
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    sentiment_score: Decimal = Decimal("0")
    relevance_score: Decimal = Decimal("0")
    url: Optional[str] = None
    symbols: List[Symbol] = field(default_factory=list)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


```

#### Строка 1063: Unknown
**Класс:** NewsItem
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    relevance_score: Decimal = Decimal("0")
    url: Optional[str] = None
    symbols: List[Symbol] = field(default_factory=list)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
```

#### Строка 1070: Unknown
**Класс:** SocialSentiment
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class SocialSentiment:
    """Социальный сентимент."""

    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    sentiment_score: Decimal = Decimal("0")
    fear_greed_index: Decimal = Decimal("0")
    posts_count: int = 0
```

#### Строка 1077: Unknown
**Класс:** SocialSentiment
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    positive_posts: int = 0
    negative_posts: int = 0
    neutral_posts: int = 0
    trending_topics: List[str] = field(default_factory=list)
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
```

#### Строка 1079: Unknown
**Класс:** SocialSentiment
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    neutral_posts: int = 0
    trending_topics: List[str] = field(default_factory=list)
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))

```

#### Строка 1081: Unknown
**Класс:** SocialSentiment
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
```

#### Строка 1089: Unknown
**Класс:** PatternDetection
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    """Обнаружение паттерна."""

    pattern_id: str = ""
    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    pattern_type: str = ""
    confidence: ConfidenceLevel = field(
        default_factory=lambda: ConfidenceLevel(Decimal("0"))
```

#### Строка 1092: Unknown
**Класс:** PatternDetection
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    pattern_type: str = ""
    confidence: ConfidenceLevel = field(
        default_factory=lambda: ConfidenceLevel(Decimal("0"))
    )
    start_time: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
```

#### Строка 1095: Unknown
**Класс:** PatternDetection
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: ConfidenceLevel(Decimal("0"))
    )
    start_time: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    end_time: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
```

#### Строка 1098: Unknown
**Класс:** PatternDetection
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: TimestampValue(datetime.now())
    )
    end_time: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    price_levels: Dict[str, PriceValue] = field(default_factory=dict)
    volume_profile: Optional[VolumeProfileDict] = None
```

#### Строка 1100: Unknown
**Класс:** PatternDetection
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    end_time: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    price_levels: Dict[str, PriceValue] = field(default_factory=dict)
    volume_profile: Optional[VolumeProfileDict] = None
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))

```

#### Строка 1102: Unknown
**Класс:** PatternDetection
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    )
    price_levels: Dict[str, PriceValue] = field(default_factory=dict)
    volume_profile: Optional[VolumeProfileDict] = None
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
```

#### Строка 1109: Unknown
**Класс:** EntanglementResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class EntanglementResult:
    """Результат анализа запутанности."""

    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    exchange_pair: str = ""
    is_entangled: bool = False
    correlation_score: Decimal = Decimal("0")
```

#### Строка 1115: Unknown
**Класс:** EntanglementResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    correlation_score: Decimal = Decimal("0")
    lag_ms: Decimal = Decimal("0")
    confidence: ConfidenceLevel = field(
        default_factory=lambda: ConfidenceLevel(Decimal("0"))
    )
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
```

#### Строка 1118: Unknown
**Класс:** EntanglementResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: ConfidenceLevel(Decimal("0"))
    )
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))

```

#### Строка 1120: Unknown
**Класс:** EntanglementResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
```

#### Строка 1127: Unknown
**Класс:** MirrorSignal
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class MirrorSignal:
    """Зеркальный сигнал."""

    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    mirror_symbol: Symbol = field(default_factory=lambda: Symbol(""))
    correlation: Decimal = Decimal("0")
    lag: int = 0
```

#### Строка 1128: Unknown
**Класс:** MirrorSignal
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    """Зеркальный сигнал."""

    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    mirror_symbol: Symbol = field(default_factory=lambda: Symbol(""))
    correlation: Decimal = Decimal("0")
    lag: int = 0
    signal_strength: Decimal = Decimal("0")
```

#### Строка 1133: Unknown
**Класс:** MirrorSignal
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    lag: int = 0
    signal_strength: Decimal = Decimal("0")
    confidence: ConfidenceLevel = field(
        default_factory=lambda: ConfidenceLevel(Decimal("0"))
    )
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
```

#### Строка 1136: Unknown
**Класс:** MirrorSignal
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=lambda: ConfidenceLevel(Decimal("0"))
    )
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))

```

#### Строка 1138: Unknown
**Класс:** MirrorSignal
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
```

#### Строка 1146: Unknown
**Класс:** SessionInfluence
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    """Влияние сессии."""

    session_type: str = ""
    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    influence_score: Decimal = Decimal("0")
    volatility_impact: Decimal = Decimal("0")
    volume_impact: Decimal = Decimal("0")
```

#### Строка 1152: Unknown
**Класс:** SessionInfluence
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    volume_impact: Decimal = Decimal("0")
    momentum_impact: Decimal = Decimal("0")
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))

```

#### Строка 1154: Unknown
**Класс:** SessionInfluence
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
```

#### Строка 1161: Unknown
**Класс:** SessionInfluence
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class LiquidityGravityResult:
    """Результат анализа гравитации ликвидности."""

    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    gravity_score: Decimal = Decimal("0")
    liquidity_score: Decimal = Decimal("0")
    volatility_score: Decimal = Decimal("0")
```

#### Строка 1167: Unknown
**Класс:** LiquidityGravityResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    volatility_score: Decimal = Decimal("0")
    anomaly_detected: bool = False
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))

```

#### Строка 1169: Unknown
**Класс:** LiquidityGravityResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
```

#### Строка 1176: Unknown
**Класс:** LiquidityGravityResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
class EvolutionResult:
    """Результат эволюции стратегии."""

    strategy_id: StrategyId = field(default_factory=lambda: StrategyId(uuid4()))
    generation: int = 0
    fitness_score: Decimal = Decimal("0")
    performance_metrics: PerformanceMetricsDict = field(default_factory=dict)
```

#### Строка 1179: Unknown
**Класс:** EvolutionResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    strategy_id: StrategyId = field(default_factory=lambda: StrategyId(uuid4()))
    generation: int = 0
    fitness_score: Decimal = Decimal("0")
    performance_metrics: PerformanceMetricsDict = field(default_factory=dict)
    parameters: ParameterDict = field(default_factory=dict)
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
```

#### Строка 1180: Unknown
**Класс:** EvolutionResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    generation: int = 0
    fitness_score: Decimal = Decimal("0")
    performance_metrics: PerformanceMetricsDict = field(default_factory=dict)
    parameters: ParameterDict = field(default_factory=dict)
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
```

#### Строка 1182: Unknown
**Класс:** EvolutionResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    performance_metrics: PerformanceMetricsDict = field(default_factory=dict)
    parameters: ParameterDict = field(default_factory=dict)
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))

```

#### Строка 1184: Unknown
**Класс:** EvolutionResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
```

#### Строка 1198: Unknown
**Класс:** SymbolSelectionResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    processing_time_ms: float = 0.0
    total_symbols_analyzed: int = 0
    cache_hit_rate: Decimal = Decimal("0")
    performance_metrics: PerformanceMetricsDict = field(default_factory=dict)
    detailed_profiles: Dict[Symbol, Dict[str, Union[str, float, int, bool]]] = field(
        default_factory=dict
    )
```

#### Строка 1200: Unknown
**Класс:** SymbolSelectionResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    cache_hit_rate: Decimal = Decimal("0")
    performance_metrics: PerformanceMetricsDict = field(default_factory=dict)
    detailed_profiles: Dict[Symbol, Dict[str, Union[str, float, int, bool]]] = field(
        default_factory=dict
    )
    rejection_reasons: Dict[Symbol, List[str]] = field(default_factory=dict)
    optimization_suggestions: List[str] = field(default_factory=list)
```

#### Строка 1202: Unknown
**Класс:** SymbolSelectionResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    detailed_profiles: Dict[Symbol, Dict[str, Union[str, float, int, bool]]] = field(
        default_factory=dict
    )
    rejection_reasons: Dict[Symbol, List[str]] = field(default_factory=dict)
    optimization_suggestions: List[str] = field(default_factory=list)


```

#### Строка 1203: Unknown
**Класс:** SymbolSelectionResult
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        default_factory=dict
    )
    rejection_reasons: Dict[Symbol, List[str]] = field(default_factory=dict)
    optimization_suggestions: List[str] = field(default_factory=list)


# ============================================================================
```

#### Строка 1219: Unknown
**Класс:** NotificationTemplate
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    level: NotificationLevel
    type: NotificationType
    priority: NotificationPriority = NotificationPriority.NORMAL
    channels: List[NotificationChannel] = field(default_factory=list)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


```

#### Строка 1220: Unknown
**Класс:** NotificationTemplate
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    type: NotificationType
    priority: NotificationPriority = NotificationPriority.NORMAL
    channels: List[NotificationChannel] = field(default_factory=list)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))


@dataclass(kw_only=True)
```

#### Строка 1228: Unknown
**Класс:** NotificationTemplate
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    """Конфигурация уведомлений."""

    email_enabled: bool = True
    email_config: Dict[str, Union[str, int, bool]] = field(default_factory=dict)
    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=dict)
```

#### Строка 1231: Unknown
**Класс:** NotificationConfig
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    email_config: Dict[str, Union[str, int, bool]] = field(default_factory=dict)
    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=dict)
    sms_enabled: bool = False
    sms_config: Dict[str, Union[str, int, bool]] = field(default_factory=dict)
    push_enabled: bool = False
```

#### Строка 1233: Unknown
**Класс:** NotificationConfig
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=dict)
    sms_enabled: bool = False
    sms_config: Dict[str, Union[str, int, bool]] = field(default_factory=dict)
    push_enabled: bool = False
    push_config: Dict[str, Union[str, int, bool]] = field(default_factory=dict)
    telegram_enabled: bool = False
```

#### Строка 1235: Unknown
**Класс:** NotificationConfig
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    sms_enabled: bool = False
    sms_config: Dict[str, Union[str, int, bool]] = field(default_factory=dict)
    push_enabled: bool = False
    push_config: Dict[str, Union[str, int, bool]] = field(default_factory=dict)
    telegram_enabled: bool = False
    telegram_config: Dict[str, Union[str, int, bool]] = field(default_factory=dict)
    default_channels: List[NotificationChannel] = field(
```

#### Строка 1237: Unknown
**Класс:** NotificationConfig
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    push_enabled: bool = False
    push_config: Dict[str, Union[str, int, bool]] = field(default_factory=dict)
    telegram_enabled: bool = False
    telegram_config: Dict[str, Union[str, int, bool]] = field(default_factory=dict)
    default_channels: List[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.EMAIL]
    )
```

#### Строка 1238: Unknown
**Класс:** NotificationConfig
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    push_config: Dict[str, Union[str, int, bool]] = field(default_factory=dict)
    telegram_enabled: bool = False
    telegram_config: Dict[str, Union[str, int, bool]] = field(default_factory=dict)
    default_channels: List[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.EMAIL]
    )
    retry_attempts: int = 3
```

#### Строка 1239: Unknown
**Класс:** NotificationConfig
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    telegram_enabled: bool = False
    telegram_config: Dict[str, Union[str, int, bool]] = field(default_factory=dict)
    default_channels: List[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.EMAIL]
    )
    retry_attempts: int = 3
    retry_delay: float = 1.0
```

### 📁 application\use_cases\__init__.py
Найдено проблем: 12

#### Строка 8: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
"""

from .manage_positions import (
    DefaultPositionManagementUseCase,
    PositionManagementUseCase,
)
from .manage_trading_pairs import (
```

#### Строка 12: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    PositionManagementUseCase,
)
from .manage_trading_pairs import (
    DefaultTradingPairManagementUseCase,
    TradingPairManagementUseCase,
)
from .trading_orchestrator.core import (
```

#### Строка 16: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    TradingPairManagementUseCase,
)
from .trading_orchestrator.core import (
    DefaultTradingOrchestratorUseCase,
    TradingOrchestratorUseCase,
)
from .trading_orchestrator.dtos import (
```

#### Строка 20: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
    TradingOrchestratorUseCase,
)
from .trading_orchestrator.dtos import (
    ExecuteStrategyRequest,
    ExecuteStrategyResponse,
    PortfolioRebalanceRequest,
    PortfolioRebalanceResponse,
```

#### Строка 21: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
)
from .trading_orchestrator.dtos import (
    ExecuteStrategyRequest,
    ExecuteStrategyResponse,
    PortfolioRebalanceRequest,
    PortfolioRebalanceResponse,
    ProcessSignalRequest,
```

#### Строка 32: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
__all__ = [
    # Order Management
    "OrderManagementUseCase",
    "DefaultOrderManagementUseCase",
    # Position Management
    "PositionManagementUseCase",
    "DefaultPositionManagementUseCase",
```

#### Строка 35: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    "DefaultOrderManagementUseCase",
    # Position Management
    "PositionManagementUseCase",
    "DefaultPositionManagementUseCase",
    # Risk Management
    "RiskManagementUseCase",
    "DefaultRiskManagementUseCase",
```

#### Строка 38: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    "DefaultPositionManagementUseCase",
    # Risk Management
    "RiskManagementUseCase",
    "DefaultRiskManagementUseCase",
    # Trading Pair Management
    "TradingPairManagementUseCase",
    "DefaultTradingPairManagementUseCase",
```

#### Строка 41: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    "DefaultRiskManagementUseCase",
    # Trading Pair Management
    "TradingPairManagementUseCase",
    "DefaultTradingPairManagementUseCase",
    # Trading Orchestrator
    "TradingOrchestratorUseCase",
    "DefaultTradingOrchestratorUseCase",
```

#### Строка 44: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
    "DefaultTradingPairManagementUseCase",
    # Trading Orchestrator
    "TradingOrchestratorUseCase",
    "DefaultTradingOrchestratorUseCase",
    "ExecuteStrategyRequest",
    "ExecuteStrategyResponse",
    "ProcessSignalRequest",
```

#### Строка 45: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
    # Trading Orchestrator
    "TradingOrchestratorUseCase",
    "DefaultTradingOrchestratorUseCase",
    "ExecuteStrategyRequest",
    "ExecuteStrategyResponse",
    "ProcessSignalRequest",
    "ProcessSignalResponse",
```

#### Строка 46: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
    "TradingOrchestratorUseCase",
    "DefaultTradingOrchestratorUseCase",
    "ExecuteStrategyRequest",
    "ExecuteStrategyResponse",
    "ProcessSignalRequest",
    "ProcessSignalResponse",
    "PortfolioRebalanceRequest",
```

### 📁 application\use_cases\manage_orders.py
Найдено проблем: 10

#### Строка 57: Unknown
**Класс:** OrderManagementUseCase
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        self.position_repository = position_repository


class DefaultOrderManagementUseCase(OrderManagementUseCase):
    """Реализация use case для управления ордерами."""

    def __init__(
```

#### Строка 250: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            order = await self.order_repository.get_by_id(order_id)
            if order and order.portfolio_id == portfolio_id:
                return order
            return None
        except Exception as e:
            logger.error(f"Error getting order by ID: {e}")
            return None
```

#### Строка 253: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            return None
        except Exception as e:
            logger.error(f"Error getting order by ID: {e}")
            return None

    async def update_order_status(
        self,
```

#### Строка 273: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
        try:
            order = await self.order_repository.get_by_id(order_id)
            if not order:
                return False

            order.status = status
            order.updated_at = Timestamp.now()  # Исправление: используем Timestamp
```

#### Строка 282: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
                order.filled_quantity = filled_amount

            await self.order_repository.save(order)
            return True

        except Exception as e:
            logger.error(f"Error updating order status: {e}")
```

#### Строка 286: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error updating order status: {e}")
            return False

    async def validate_order(
        self, request: CreateOrderRequest
```

#### Строка 347: execute_signal
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
            True если сигнал исполнен успешно
        """
        try:
            # В реальной системе здесь была бы логика исполнения сигнала
            # Пока возвращаем заглушку
            logger.info(f"Executing signal: {signal}")
            return True
```

#### Строка 348: execute_signal
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'заглушку'
**Код:**
```python
        """
        try:
            # В реальной системе здесь была бы логика исполнения сигнала
            # Пока возвращаем заглушку
            logger.info(f"Executing signal: {signal}")
            return True
        except Exception as e:
```

#### Строка 350: execute_signal
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            # В реальной системе здесь была бы логика исполнения сигнала
            # Пока возвращаем заглушку
            logger.info(f"Executing signal: {signal}")
            return True
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return False
```

#### Строка 353: execute_signal
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return False

```

### 📁 application\use_cases\manage_positions.py
Найдено проблем: 7

#### Строка 298: get_position_metrics
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            position = await self.position_repository.get_by_id(PositionId(position_id))
            
            if not position:
                return None

            return PositionMetrics(
                position_id=position.id,
```

#### Строка 319: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error getting position metrics: {e}")
            return None

    async def close_position_partial(
        self, position_id: str, close_volume: Decimal, close_price: Decimal
```

#### Строка 329: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            position = await self.position_repository.get_by_id(PositionId(position_id))
            
            if not position or not position.is_open:
                return False
            close_vol = Volume(close_volume, Currency.USDT)
            close_prc = Price(close_price, Currency.USDT)
            
```

#### Строка 340: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            # Обновление позиции
            await self.position_repository.update(position)
            
            return True

        except Exception as e:
            logger.error(f"Error partially closing position: {e}")
```

#### Строка 344: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error partially closing position: {e}")
            return False

    async def get_position_statistics(self, portfolio_id: Optional[str] = None) -> Dict:
        """Получение статистики позиций."""
```

#### Строка 397: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error getting position statistics: {e}")
            return {}


class DefaultPositionManagementUseCase(PositionManagementUseCase):
```

#### Строка 400: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
            return {}


class DefaultPositionManagementUseCase(PositionManagementUseCase):
    """Реализация по умолчанию для управления позициями."""

    def __init__(
```

### 📁 application\use_cases\manage_risk.py
Найдено проблем: 15

#### Строка 272: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
            # Расчет общего P&L
            total_pnl = Decimal("0")
            
            # Расчет максимального drawdown (упрощенная версия)
            max_drawdown = min(total_pnl, Decimal("0"))
            
            # Расчет VaR (упрощенная версия)
```

#### Строка 275: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
            # Расчет максимального drawdown (упрощенная версия)
            max_drawdown = min(total_pnl, Decimal("0"))
            
            # Расчет VaR (упрощенная версия)
            var_95 = total_exposure * Decimal("0.05")  # 5% VaR
            
            # Расчет волатильности (упрощенная версия)
```

#### Строка 278: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
            # Расчет VaR (упрощенная версия)
            var_95 = total_exposure * Decimal("0.05")  # 5% VaR
            
            # Расчет волатильности (упрощенная версия)
            volatility = float(abs(total_pnl) / total_exposure) if total_exposure > 0 else 0.0
            
            # Расчет коэффициента Шарпа (упрощенная версия)
```

#### Строка 281: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
            # Расчет волатильности (упрощенная версия)
            volatility = float(abs(total_pnl) / total_exposure) if total_exposure > 0 else 0.0
            
            # Расчет коэффициента Шарпа (упрощенная версия)
            # Исправление: приводим к float перед делением
            sharpe_ratio = float(total_pnl) / float(volatility) if volatility > 0 else 0.0
            
```

#### Строка 285: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
            # Исправление: приводим к float перед делением
            sharpe_ratio = float(total_pnl) / float(volatility) if volatility > 0 else 0.0
            
            # Расчет корреляционной матрицы (упрощенная версия)
            correlation_matrix = {}
            for i, pos1 in enumerate(positions):
                for j, pos2 in enumerate(positions):
```

#### Строка 291: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
                for j, pos2 in enumerate(positions):
                    if i != j:
                        key = f"pos_{i}_pos_{j}"
                        correlation_matrix[key] = 0.5  # Упрощенная корреляция
            
            # Расчет риска концентрации
            if total_exposure > 0:
```

#### Строка 333: _get_market_data
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
        try:
            market_data: Dict[str, float] = {}
            for position in positions:
                # Упрощенная версия - используем текущую цену позиции
                market_data[str(position.trading_pair.symbol)] = float(position.current_price.amount)
            return market_data
        except Exception as e:
```

#### Строка 338: _get_market_data
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python
            return market_data
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {}

    async def calculate_position_risk(self, position: Position) -> PositionRisk:
        """Расчет риска отдельной позиции."""
```

#### Строка 346: calculate_position_risk
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
            # Расчет P&L
            unrealized_pnl = position.unrealized_pnl.amount if position.unrealized_pnl else Decimal("0")
            
            # Расчет риска (упрощенная версия)
            # Исправление: добавляем проверки на существование атрибута
            notional_value = getattr(position, 'notional_value', None)
            if notional_value and notional_value.amount > 0:
```

#### Строка 361: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
            else:
                var_contribution = Decimal("0")
            
            # Корреляционный риск (упрощенная версия)
            correlation_risk = 0.5

            return PositionRisk(
```

#### Строка 534: check_order_risk
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
            True если риск приемлем
        """
        try:
            # В реальной системе здесь была бы логика проверки риска ордера
            # Пока возвращаем заглушку
            logger.info(f"Checking order risk for signal: {signal}")
            return True
```

#### Строка 535: check_order_risk
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'заглушку'
**Код:**
```python
        """
        try:
            # В реальной системе здесь была бы логика проверки риска ордера
            # Пока возвращаем заглушку
            logger.info(f"Checking order risk for signal: {signal}")
            return True
        except Exception as e:
```

#### Строка 537: check_order_risk
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            # В реальной системе здесь была бы логика проверки риска ордера
            # Пока возвращаем заглушку
            logger.info(f"Checking order risk for signal: {signal}")
            return True
        except Exception as e:
            logger.error(f"Error checking order risk: {e}")
            return False
```

#### Строка 540: check_order_risk
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return True
        except Exception as e:
            logger.error(f"Error checking order risk: {e}")
            return False


class DefaultRiskManagementUseCase(RiskManagementUseCase):
```

#### Строка 543: check_order_risk
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
            return False


class DefaultRiskManagementUseCase(RiskManagementUseCase):
    """Реализация по умолчанию для управления рисками."""

    def __init__(
```

### 📁 application\use_cases\manage_trading_pairs.py
Найдено проблем: 29

#### Строка 44: Unknown
**Класс:** TradingPairManagementUseCase
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
        self, request: CreateTradingPairRequest
    ) -> CreateTradingPairResponse:
        """Создание новой торговой пары."""
        pass

    @abstractmethod
    async def get_trading_pairs(
```

#### Строка 51: Unknown
**Класс:** TradingPairManagementUseCase
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
        self, request: GetTradingPairsRequest
    ) -> GetTradingPairsResponse:
        """Получение списка торговых пар."""
        pass

    @abstractmethod
    async def update_trading_pair(
```

#### Строка 58: Unknown
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
        self, request: UpdateTradingPairRequest
    ) -> UpdateTradingPairResponse:
        """Обновление торговой пары."""
        pass

    @abstractmethod
    async def get_trading_pair_by_id(self, trading_pair_id: str) -> Optional[TradingPair]:
```

#### Строка 63: get_trading_pair_by_id
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
    @abstractmethod
    async def get_trading_pair_by_id(self, trading_pair_id: str) -> Optional[TradingPair]:
        """Получение торговой пары по ID."""
        pass

    @abstractmethod
    async def delete_trading_pair(self, trading_pair_id: str) -> bool:
```

#### Строка 68: delete_trading_pair
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
    @abstractmethod
    async def delete_trading_pair(self, trading_pair_id: str) -> bool:
        """Удаление торговой пары."""
        pass

    @abstractmethod
    async def calculate_trading_pair_metrics(
```

#### Строка 75: delete_trading_pair
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
        self, trading_pair: TradingPair
    ) -> TradingPairMetrics:
        """Расчет метрик торговой пары."""
        pass

    @abstractmethod
    async def validate_trading_pair(self, trading_pair: TradingPair) -> bool:
```

#### Строка 80: validate_trading_pair
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
    @abstractmethod
    async def validate_trading_pair(self, trading_pair: TradingPair) -> bool:
        """Валидация торговой пары."""
        pass

    @abstractmethod
    async def get_trading_pair_status(self, trading_pair_id: str) -> str:
```

#### Строка 85: get_trading_pair_status
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
    @abstractmethod
    async def get_trading_pair_status(self, trading_pair_id: str) -> str:
        """Получение статуса торговой пары."""
        pass

    @abstractmethod
    async def activate_trading_pair(self, trading_pair_id: str) -> bool:
```

#### Строка 90: activate_trading_pair
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
    @abstractmethod
    async def activate_trading_pair(self, trading_pair_id: str) -> bool:
        """Активация торговой пары."""
        pass

    @abstractmethod
    async def deactivate_trading_pair(self, trading_pair_id: str) -> bool:
```

#### Строка 95: deactivate_trading_pair
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
    @abstractmethod
    async def deactivate_trading_pair(self, trading_pair_id: str) -> bool:
        """Деактивация торговой пары."""
        pass

    @abstractmethod
    async def calculate_liquidity_metrics(
```

#### Строка 102: deactivate_trading_pair
**Тип:** Не реализовано
**Описание:** Найдено триггерное слово: 'pass'
**Код:**
```python
        self, trading_pair: TradingPair
    ) -> Dict[str, float]:
        """Расчет метрик ликвидности."""
        pass


class DefaultTradingPairManagementUseCase(TradingPairManagementUseCase):
```

#### Строка 105: deactivate_trading_pair
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        pass


class DefaultTradingPairManagementUseCase(TradingPairManagementUseCase):
    """Реализация по умолчанию для управления торговыми парами."""

    def __init__(
```

#### Строка 140: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
                is_active=request.is_active,
                min_order_size=Volume(Decimal(str(request.min_amount)), base_currency) if request.min_amount else None,
                max_order_size=Volume(Decimal(str(request.max_amount)), base_currency) if request.max_amount else None,
                price_precision=PricePrecision(8),  # Default precision
                volume_precision=VolumePrecision(8),  # Default precision
            )

```

#### Строка 141: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
                min_order_size=Volume(Decimal(str(request.min_amount)), base_currency) if request.min_amount else None,
                max_order_size=Volume(Decimal(str(request.max_amount)), base_currency) if request.max_amount else None,
                price_precision=PricePrecision(8),  # Default precision
                volume_precision=VolumePrecision(8),  # Default precision
            )

            # Сохранение торговой пары
```

#### Строка 257: get_trading_pair_by_id
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            return await self.trading_pair_repository.get_by_symbol(trading_pair_id)
        except Exception as e:
            logger.error(f"Error getting trading pair by ID: {e}")
            return None

    async def delete_trading_pair(self, trading_pair_id: str) -> bool:
        """Удаление торговой пары."""
```

#### Строка 265: delete_trading_pair
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            return await self.trading_pair_repository.delete(trading_pair_id)
        except Exception as e:
            logger.error(f"Error deleting trading pair: {e}")
            return False

    async def calculate_trading_pair_metrics(
        self, trading_pair: TradingPair
```

#### Строка 313: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'упрощенная'
**Код:**
```python
            ask = None
            spread = None

            # Расчет волатильности (упрощенная версия)
            volatility = Decimal("0")
            if market_data.high and market_data.low:
                volatility = (market_data.high.value - market_data.low.value) / market_data.low.value
```

#### Строка 348: validate_trading_pair
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
        try:
            # Проверка базовых условий
            if not trading_pair.symbol:
                return False
            if trading_pair.base_currency == trading_pair.quote_currency:
                return False
            if trading_pair.price_precision < 0 or trading_pair.volume_precision < 0:
```

#### Строка 350: validate_trading_pair
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            if not trading_pair.symbol:
                return False
            if trading_pair.base_currency == trading_pair.quote_currency:
                return False
            if trading_pair.price_precision < 0 or trading_pair.volume_precision < 0:
                return False

```

#### Строка 352: validate_trading_pair
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            if trading_pair.base_currency == trading_pair.quote_currency:
                return False
            if trading_pair.price_precision < 0 or trading_pair.volume_precision < 0:
                return False

            # Проверка рыночных данных
            market_data_list = await self.market_repository.get_market_data(trading_pair.symbol, "1d")
```

#### Строка 357: validate_trading_pair
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            # Проверка рыночных данных
            market_data_list = await self.market_repository.get_market_data(trading_pair.symbol, "1d")
            if not market_data_list or len(market_data_list) == 0:
                return False

            return True

```

#### Строка 359: validate_trading_pair
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            if not market_data_list or len(market_data_list) == 0:
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating trading pair: {e}")
```

#### Строка 363: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error validating trading pair: {e}")
            return False

    async def get_trading_pair_status(self, trading_pair_id: str) -> str:
        """Получение статуса торговой пары."""
```

#### Строка 388: activate_trading_pair
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            trading_pair = await self.trading_pair_repository.get_by_symbol(trading_pair_id)
            
            if not trading_pair:
                return False

            trading_pair.activate()
            await self.trading_pair_repository.update(trading_pair)
```

#### Строка 393: activate_trading_pair
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            trading_pair.activate()
            await self.trading_pair_repository.update(trading_pair)
            
            return True

        except Exception as e:
            logger.error(f"Error activating trading pair: {e}")
```

#### Строка 397: activate_trading_pair
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error activating trading pair: {e}")
            return False

    async def deactivate_trading_pair(self, trading_pair_id: str) -> bool:
        """Деактивация торговой пары."""
```

#### Строка 405: deactivate_trading_pair
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
            trading_pair = await self.trading_pair_repository.get_by_symbol(trading_pair_id)
            
            if not trading_pair:
                return False

            trading_pair.deactivate()
            await self.trading_pair_repository.update(trading_pair)
```

#### Строка 410: deactivate_trading_pair
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return true'
**Код:**
```python
            trading_pair.deactivate()
            await self.trading_pair_repository.update(trading_pair)
            
            return True

        except Exception as e:
            logger.error(f"Error deactivating trading pair: {e}")
```

#### Строка 414: deactivate_trading_pair
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error deactivating trading pair: {e}")
            return False

    async def calculate_liquidity_metrics(
        self, trading_pair: TradingPair
```

### 📁 application\use_cases\trading_orchestrator.py
Найдено проблем: 6

#### Строка 7: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python

# Импортируем основной класс из core.py
from .trading_orchestrator.core import (
    DefaultTradingOrchestratorUseCase,
    TradingOrchestratorUseCase,
)

```

#### Строка 13: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python

# Импортируем DTO из нового модуля
from .trading_orchestrator.dtos import (
    ExecuteStrategyRequest,
    ExecuteStrategyResponse,
    PortfolioRebalanceRequest,
    PortfolioRebalanceResponse,
```

#### Строка 14: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
# Импортируем DTO из нового модуля
from .trading_orchestrator.dtos import (
    ExecuteStrategyRequest,
    ExecuteStrategyResponse,
    PortfolioRebalanceRequest,
    PortfolioRebalanceResponse,
    ProcessSignalRequest,
```

#### Строка 25: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
# Экспортируем все публичные интерфейсы
__all__ = [
    "TradingOrchestratorUseCase",
    "DefaultTradingOrchestratorUseCase",
    "ExecuteStrategyRequest",
    "ExecuteStrategyResponse",
    "ProcessSignalRequest",
```

#### Строка 26: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
__all__ = [
    "TradingOrchestratorUseCase",
    "DefaultTradingOrchestratorUseCase",
    "ExecuteStrategyRequest",
    "ExecuteStrategyResponse",
    "ProcessSignalRequest",
    "ProcessSignalResponse",
```

#### Строка 27: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
    "TradingOrchestratorUseCase",
    "DefaultTradingOrchestratorUseCase",
    "ExecuteStrategyRequest",
    "ExecuteStrategyResponse",
    "ProcessSignalRequest",
    "ProcessSignalResponse",
    "PortfolioRebalanceRequest",
```

### 📁 application\use_cases\trading_orchestrator\__init__.py
Найдено проблем: 4

#### Строка 6: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
"""

from application.types import (
    ExecuteStrategyRequest,
    ExecuteStrategyResponse,
    PortfolioRebalanceRequest,
    PortfolioRebalanceResponse,
```

#### Строка 7: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python

from application.types import (
    ExecuteStrategyRequest,
    ExecuteStrategyResponse,
    PortfolioRebalanceRequest,
    PortfolioRebalanceResponse,
    ProcessSignalRequest,
```

#### Строка 16: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
)

__all__ = [
    "ExecuteStrategyRequest",
    "ExecuteStrategyResponse",
    "ProcessSignalRequest",
    "ProcessSignalResponse",
```

#### Строка 17: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python

__all__ = [
    "ExecuteStrategyRequest",
    "ExecuteStrategyResponse",
    "ProcessSignalRequest",
    "ProcessSignalResponse",
    "PortfolioRebalanceRequest",
```

### 📁 application\use_cases\trading_orchestrator\core.py
Найдено проблем: 28

#### Строка 14: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
from loguru import logger

from application.types import (
    ExecuteStrategyRequest,
    ExecuteStrategyResponse,
    PortfolioRebalanceRequest,
    PortfolioRebalanceResponse,
```

#### Строка 15: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python

from application.types import (
    ExecuteStrategyRequest,
    ExecuteStrategyResponse,
    PortfolioRebalanceRequest,
    PortfolioRebalanceResponse,
    ProcessSignalRequest,
```

#### Строка 31: Unknown
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'временно'
**Код:**
```python
from domain.intelligence.mirror_detector import MirrorDetector
from domain.intelligence.noise_analyzer import NoiseAnalyzer

# Временно закомментированные импорты агентов для обхода ошибок
# from infrastructure.agents.agent_whales import WhalesAgent
# from infrastructure.agents.agent_risk import RiskAgent
# from infrastructure.agents.agent_portfolio import PortfolioAgent
```

#### Строка 118: Unknown
**Класс:** TradingOrchestratorUseCase
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python

    @abstractmethod
    async def execute_strategy(
        self, request: ExecuteStrategyRequest
    ) -> ExecuteStrategyResponse:
        """Выполнение торговой стратегии."""

```

#### Строка 119: Unknown
**Класс:** TradingOrchestratorUseCase
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
    @abstractmethod
    async def execute_strategy(
        self, request: ExecuteStrategyRequest
    ) -> ExecuteStrategyResponse:
        """Выполнение торговой стратегии."""

    @abstractmethod
```

#### Строка 161: get_trading_session
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'default'
**Код:**
```python
        """Проверка торговых условий."""


class DefaultTradingOrchestratorUseCase(TradingOrchestratorUseCase):
    """Реализация use case для оркестрации торговли."""

    def __init__(
```

#### Строка 186: Unknown
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'временно'
**Код:**
```python
        evolutionary_transformer: Optional[EvolutionaryTransformer] = None,
        pattern_discovery: Optional[PatternDiscovery] = None,
        meta_learning: Optional[MetaLearning] = None,
        # Временно закомментированные агенты - заменены на Any
        agent_risk: Optional[Any] = None,
        agent_portfolio: Optional[Any] = None,
        agent_meta_controller: Optional[Any] = None,
```

#### Строка 301: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
        }

    async def execute_strategy(
        self, request: ExecuteStrategyRequest
    ) -> ExecuteStrategyResponse:
        """Выполнение торговой стратегии."""
        try:
```

#### Строка 302: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python

    async def execute_strategy(
        self, request: ExecuteStrategyRequest
    ) -> ExecuteStrategyResponse:
        """Выполнение торговой стратегии."""
        try:
            start_time = datetime.now()
```

#### Строка 310: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
            # Получаем стратегию
            strategy = await self._get_strategy(str(request.strategy_id))
            if not strategy:
                return ExecuteStrategyResponse(
                    success=False,
                    errors=["Strategy not found"],
                    execution_time_ms=0,
```

#### Строка 346: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python

            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return ExecuteStrategyResponse(
                success=True,
                orders_created=orders,
                signals_generated=signal_list if signals else [],
```

#### Строка 356: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error executing strategy: {e}")
            return ExecuteStrategyResponse(
                success=False,
                errors=[str(e)],
                execution_time_ms=0,
```

#### Строка 374: _get_strategy
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
            return await self._create_strategy_from_infrastructure(strategy_id)
        except Exception as e:
            logger.error(f"Error getting strategy {strategy_id}: {e}")
            return None

    async def _create_strategy_from_infrastructure(
        self, strategy_id: str
```

#### Строка 387: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'test'
**Код:**
```python
                return AdaptiveStrategyGenerator(  # type: ignore
                    market_regime_agent=None,
                    meta_learner=None,
                    backtest_results={},
                    base_strategies=[],
                    config={}
                )
```

#### Строка 405: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
                return VolatilityStrategy()  # type: ignore
            else:
                logger.warning(f"Unknown strategy type: {strategy_id}")
                return None
        except Exception as e:
            logger.error(f"Error creating strategy {strategy_id}: {e}")
            return None
```

#### Строка 408: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
                return None
        except Exception as e:
            logger.error(f"Error creating strategy {strategy_id}: {e}")
            return None

    async def process_signal(
        self, request: ProcessSignalRequest
```

#### Строка 513: stop_trading_session
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python
        """Остановка торговой сессии."""
        try:
            if not self.session_service:
                return False

            # Останавливаем сессию - исправляем метод
            return await self.session_service.close_session(session_id)
```

#### Строка 520: stop_trading_session
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error stopping trading session: {e}")
            return False

    async def get_trading_session(self, session_id: str) -> Optional[TradingSession]:
        """Получение торговой сессии."""
```

#### Строка 526: get_trading_session
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python
        """Получение торговой сессии."""
        try:
            if not self.session_service:
                return None

            # Получаем сессию - исправляем метод
            return await self.session_service.get_session(session_id)
```

#### Строка 533: get_trading_session
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error getting trading session: {e}")
            return None

    async def calculate_portfolio_weights(
        self, portfolio_id: str
```

#### Строка 543: Unknown
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'временно'
**Код:**
```python
            # Получаем позиции портфеля - исправляем метод
            positions = await self.position_repository.get_open_positions()
            
            # Фильтруем позиции по портфелю (временное решение)
            portfolio_positions = [pos for pos in positions if hasattr(pos, 'portfolio_id') and str(pos.portfolio_id) == portfolio_id]
            
            # Рассчитываем общую стоимость портфеля
```

#### Строка 565: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error calculating portfolio weights: {e}")
            return {}

    async def validate_trading_conditions(
        self, portfolio_id: str, symbol: str
```

#### Строка 586: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
                errors.append("Invalid symbol")

            # Проверяем торговые условия
            # Здесь можно добавить дополнительные проверки

            return len(errors) == 0, errors

```

#### Строка 592: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return false'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error validating trading conditions: {e}")
            return False, [str(e)]

    async def _create_order_from_signal(
        self, signal: Signal, portfolio_id: str
```

#### Строка 626: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error creating order from signal: {e}")
            return None

    async def _create_rebalance_order(
        self, symbol: str, weight_diff: Decimal, portfolio_id: str
```

#### Строка 658: Unknown
**Тип:** Возврат по умолчанию
**Описание:** Найдено триггерное слово: 'return none'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error creating rebalance order: {e}")
            return None

    async def _calculate_portfolio_risk(self, portfolio_id: str) -> Dict[str, float]:
        """Расчет риска портфеля."""
```

#### Строка 666: _calculate_portfolio_risk
**Тип:** Упрощенная реализация
**Описание:** Найдено триггерное слово: 'временно'
**Код:**
```python
            # Получаем позиции портфеля - исправляем метод
            positions = await self.position_repository.get_open_positions()
            
            # Фильтруем позиции по портфелю (временное решение)
            portfolio_positions = [pos for pos in positions if hasattr(pos, 'portfolio_id') and str(pos.portfolio_id) == portfolio_id]
            
            # Рассчитываем базовые метрики риска
```

#### Строка 681: Unknown
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'return {}'
**Код:**
```python

        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return {}

```

### 📁 application\use_cases\trading_orchestrator\update_handlers.py
Найдено проблем: 5

#### Строка 62: update_session_influence_analysis
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
        try:
            for symbol in symbols:
                if self.orchestrator.session_service:
                    # Получаем рыночные данные для анализа (в реальной реализации здесь был бы вызов получения данных)
                    # Пока используем заглушку
                    market_data = None  # В реальной реализации здесь были бы данные
                    if market_data is not None:
```

#### Строка 63: update_session_influence_analysis
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'заглушку'
**Код:**
```python
            for symbol in symbols:
                if self.orchestrator.session_service:
                    # Получаем рыночные данные для анализа (в реальной реализации здесь был бы вызов получения данных)
                    # Пока используем заглушку
                    market_data = None  # В реальной реализации здесь были бы данные
                    if market_data is not None:
                        await self.orchestrator.session_service.analyze_session_influence(
```

#### Строка 64: update_session_influence_analysis
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
                if self.orchestrator.session_service:
                    # Получаем рыночные данные для анализа (в реальной реализации здесь был бы вызов получения данных)
                    # Пока используем заглушку
                    market_data = None  # В реальной реализации здесь были бы данные
                    if market_data is not None:
                        await self.orchestrator.session_service.analyze_session_influence(
                            symbol, market_data
```

#### Строка 85: update_session_marker
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'здесь'
**Код:**
```python
                    context = (
                        self.orchestrator.session_service.get_current_session_context()
                    )
                    # В реальной реализации здесь можно было бы сохранить контекст или использовать его
                    logger.debug(f"Updated session context for {symbol}: {context}")
                elif self.orchestrator.session_marker:  # deprecated
                    await self.orchestrator.session_marker.mark_session(symbol)
```

#### Строка 86: update_session_marker
**Тип:** Подозрительная реализация
**Описание:** Найдено триггерное слово: 'debug'
**Код:**
```python
                        self.orchestrator.session_service.get_current_session_context()
                    )
                    # В реальной реализации здесь можно было бы сохранить контекст или использовать его
                    logger.debug(f"Updated session context for {symbol}: {context}")
                elif self.orchestrator.session_marker:  # deprecated
                    await self.orchestrator.session_marker.mark_session(symbol)
        except Exception as e:
```