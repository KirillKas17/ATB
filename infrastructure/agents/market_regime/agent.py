from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from domain.type_definitions.agent_types import AgentConfig, ProcessingResult, AgentType
from infrastructure.agents.base_agent import AgentStatus, BaseAgent
from shared.logging import setup_logger

from .indicators import DefaultIndicatorCalculator
from .types import IIndicatorCalculator, MarketRegime

logger = setup_logger(__name__)


class MarketRegimeAgent(BaseAgent):
    """
    Агент для определения рыночного режима: ансамбль индикаторов, ML/кластеризация, unit-тестируемость.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Инициализация агента с параметрами конфигурации.
        :param config: словарь с параметрами для детекции режима
        """
        # Преобразуем config в AgentConfig для базового класса
        agent_config: AgentConfig = {
            "name": "MarketRegimeAgent",
            "agent_type": "market_regime",  # [1] правильный AgentType
            "max_position_size": config.get("max_position_size", 0.1) if config else 0.1,
            "max_portfolio_risk": config.get("max_portfolio_risk", 0.1) if config else 0.1,
            "max_risk_per_trade": config.get("max_risk_per_trade", 0.02) if config else 0.02,
            "confidence_threshold": config.get("confidence_threshold", 0.8) if config else 0.8,
            "risk_threshold": config.get("risk_threshold", 0.8) if config else 0.8,
            "performance_threshold": config.get("performance_threshold", 0.05) if config else 0.05,
            "rebalance_interval": config.get("rebalance_interval", 3600) if config else 3600,
            "processing_timeout_ms": config.get("processing_timeout_ms", 5000) if config else 5000,
            "retry_attempts": config.get("retry_attempts", 3) if config else 3,
            "enable_evolution": config.get("enable_evolution", False) if config else False,
            "enable_learning": config.get("enable_learning", False) if config else False,
            "metadata": config.get("metadata", {}) if config else {},
        }
        # Преобразуем AgentConfig в dict для BaseAgent
        agent_config_dict = {
            "retry_attempts": getattr(agent_config, 'retry_attempts', 3),
            "enable_evolution": getattr(agent_config, 'enable_evolution', False),
            "enable_learning": getattr(agent_config, 'enable_learning', False),
            "metadata": getattr(agent_config, 'metadata', {}),
        }
        super().__init__("MarketRegimeAgent", "market_regime", agent_config_dict)

        # Сохраняем конфигурацию агента
        self._agent_config = config or {
            "adx_period": 14,
            "adx_threshold": 25,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "atr_period": 14,
            "volatility_threshold": 2.0,
            "obv_threshold": 0.7,
            "confidence_threshold": 0.8,
        }
        
        self.current_regime: Optional[MarketRegime] = None
        self.regime_confidence: float = 0.0
        self.regime_history: List[Tuple[MarketRegime, float]] = []
        self.calculator: IIndicatorCalculator = DefaultIndicatorCalculator()
        self.symbol: str = ""
        self.current_price: float = 0.0
        # Индикаторы
        self.indicators: Dict[str, Any] = {}
        self._initialize_indicators()

    @property
    def agent_config(self) -> Dict[str, Any]:
        """Конфигурация агента."""
        return self._agent_config

    async def initialize(self) -> bool:
        """Инициализация агента рыночного режима."""
        try:
            # Валидация конфигурации
            if not self.validate_config():
                return False
            self._update_state(AgentStatus.HEALTHY)
            self.update_confidence(0.7)
            logger.info(f"MarketRegimeAgent initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize MarketRegimeAgent: {e}")
            self.record_error(f"Initialization failed: {e}")
            return False

    async def process(self, data: Any) -> ProcessingResult:
        """Обработка данных для определения рыночного режима."""
        start_time = datetime.now()
        try:
            if isinstance(data, dict):
                dataframe = data.get("market_data")
                symbol = data.get("symbol", "")
                if dataframe is None:
                    raise ValueError("Market data is required for regime detection")
                # Определение режима
                regime, confidence = self.detect_regime(dataframe)
                # Обновление символа и цены
                self.symbol = symbol
                if not dataframe.empty:
                    self.current_price = float(dataframe["close"].iloc[-1])
                # Проверка изменения режима
                regime_change = self.detect_regime_change(dataframe)
                result_data = {
                    "regime": regime.name,
                    "confidence": confidence,
                    "regime_change": regime_change,
                    "indicators": self.indicators,
                    "current_price": self.current_price,
                    "symbol": symbol,
                }
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                self.record_success(processing_time)
                return ProcessingResult(
                    success=True,
                    data=result_data,
                    confidence=self.get_confidence(),
                    risk_score=self.get_risk_score(),
                    processing_time_ms=processing_time,
                    timestamp=datetime.now(),  # [2] обязательное поле
                    metadata={"agent_type": "market_regime"},  # [2] обязательное поле
                    errors=[],  # [2] обязательное поле
                    warnings=[]  # [2] обязательное поле
                )
            else:
                raise ValueError("Invalid data format for MarketRegimeAgent")
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.record_error(f"Processing failed: {e}", processing_time)
            return ProcessingResult(
                success=False,
                data={"error": str(e)},
                confidence=0.0,
                risk_score=1.0,
                processing_time_ms=processing_time,
                timestamp=datetime.now(),  # [2] обязательное поле
                metadata={"agent_type": "market_regime"},  # [2] обязательное поле
                errors=[str(e)],  # [2] обязательное поле
                warnings=[]  # [2] обязательное поле
            )

    async def cleanup(self) -> None:
        """Очистка ресурсов агента рыночного режима."""
        try:
            # Очистка истории режимов
            self.regime_history.clear()
            # Сброс текущего режима
            self.current_regime = None
            self.regime_confidence = 0.0
            # Очистка индикаторов
            self._initialize_indicators()
            logger.info("MarketRegimeAgent cleanup completed")
        except Exception as e:
            logger.error(f"Error during MarketRegimeAgent cleanup: {e}")

    def validate_config(self) -> bool:
        """Валидация конфигурации агента рыночного режима."""
        try:
            required_keys = [
                "adx_period",
                "adx_threshold",
                "rsi_period",
                "rsi_overbought",
                "rsi_oversold",
                "atr_period",
                "volatility_threshold",
                "confidence_threshold",
            ]
            for key in required_keys:
                if key not in self.agent_config:
                    logger.error(f"Missing required config key: {key}")
                    return False
                value = self.agent_config[key]
                if not isinstance(value, (int, float)) or value <= 0:
                    logger.error(f"Invalid config value for {key}: {value}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Config validation error: {e}")
            return False

    def get_regime_summary(self) -> Dict[str, Any]:
        """Получение сводки текущего рыночного режима."""
        return {
            "current_regime": (
                self.current_regime.name if self.current_regime else "UNKNOWN"
            ),
            "confidence": self.regime_confidence,
            "symbol": self.symbol,
            "current_price": self.current_price,
            "history_length": len(self.regime_history),
            "indicators": list(self.indicators.keys()),
        }

    def reset_regime(self) -> None:
        """Сброс состояния агента рыночного режима."""
        self.current_regime = None
        self.regime_confidence = 0.0
        self.regime_history.clear()
        self._initialize_indicators()
        logger.info("MarketRegimeAgent state reset")

    def _initialize_indicators(self) -> None:
        """Инициализация индикаторов."""
        self.indicators = {
            "adx": None,
            "rsi": None,
            "atr": None,
            "obv": None,
            "fractals": None,
            "wave_clusters": None,
        }

    def detect_regime(self, dataframe: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """Определение рыночного режима на основе индикаторов."""
        try:
            # Расчет индикаторов
            self._calculate_indicators(dataframe)
            # Расчет скора для каждого режима
            regime_scores = self._calculate_regime_scores()
            # Выбор режима с максимальным скором
            best_regime = max(regime_scores.items(), key=lambda x: x[1])[0]
            confidence = regime_scores[best_regime]
            # Обновление истории
            if self.current_regime != best_regime:
                self.regime_history.append((best_regime, confidence))
            self.current_regime = best_regime
            self.regime_confidence = confidence
            return best_regime, confidence
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return MarketRegime.ANOMALY, 0.0

    def _calculate_indicators(self, dataframe: pd.DataFrame) -> None:
        """Расчет технических индикаторов."""
        try:
            if dataframe.empty:
                logger.warning("Empty dataframe provided for indicator calculation")
                return
            # Использование калькулятора индикаторов
            calculated_indicators = self.calculator.calculate(dataframe)
            # Обновление индикаторов
            for key, value in calculated_indicators.items():
                if key in self.indicators:
                    self.indicators[key] = value
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")

    def _calculate_regime_scores(self) -> Dict[MarketRegime, float]:
        """Расчет скора для каждого рыночного режима."""
        scores: Dict[MarketRegime, float] = {}
        try:
            scores[MarketRegime.TREND] = self._calculate_trend_score()
            scores[MarketRegime.SIDEWAYS] = self._calculate_sideways_score()
            scores[MarketRegime.REVERSAL] = self._calculate_reversal_score()
            scores[MarketRegime.MANIPULATION] = self._calculate_manipulation_score()
            scores[MarketRegime.VOLATILITY] = self._calculate_volatility_score()
            scores[MarketRegime.ANOMALY] = self._calculate_anomaly_score()
        except Exception as e:
            logger.error(f"Error calculating regime scores: {e}")
            # Возвращаем ANOMALY как режим по умолчанию
            scores = {regime: 0.0 for regime in MarketRegime}
            scores[MarketRegime.ANOMALY] = 1.0
        return scores

    def _get_last(self, arr: Any, n: int = 1) -> Optional[Any]:
        """Получение последних n значений из массива."""
        if arr is None or len(arr) == 0:
            return None
        return arr[-n] if n == 1 else arr[-n:]

    def _calculate_trend_score(self) -> float:
        """Расчет скора трендового режима."""
        try:
            adx = self._get_last(self.indicators["adx"])
            rsi = self._get_last(self.indicators["rsi"])
            if adx is None or rsi is None:
                return 0.0
            # Высокий ADX указывает на сильный тренд
            adx_score = min(adx / self.agent_config["adx_threshold"], 1.0)
            # RSI не в зонах перекупленности/перепроданности
            rsi_score = 1.0 - abs(rsi - 50) / 50
            return adx_score * 0.7 + rsi_score * 0.3
        except Exception as e:
            logger.error(f"Error calculating trend score: {e}")
            return 0.0

    def _calculate_sideways_score(self) -> float:
        """Расчет скора бокового режима."""
        try:
            adx = self._get_last(self.indicators["adx"])
            if adx is None:
                return 0.0
            # Низкий ADX указывает на боковое движение
            return max(0.0, 1.0 - adx / self.agent_config["adx_threshold"])
        except Exception as e:
            logger.error(f"Error calculating sideways score: {e}")
            return 0.0

    def _calculate_reversal_score(self) -> float:
        """Расчет скора режима разворота."""
        try:
            rsi = self._get_last(self.indicators["rsi"])
            fractals = self._get_last(self.indicators["fractals"])
            if rsi is None:
                return 0.0
            # RSI в зонах перекупленности/перепроданности
            rsi_extreme = 0.0
            if rsi > self.agent_config["rsi_overbought"] or rsi < self.agent_config["rsi_oversold"]:
                rsi_extreme = 1.0
            # Наличие фракталов указывает на возможный разворот
            fractal_score = 1.0 if fractals else 0.0
            return rsi_extreme * 0.6 + fractal_score * 0.4
        except Exception as e:
            logger.error(f"Error calculating reversal score: {e}")
            return 0.0

    def _calculate_manipulation_score(self) -> float:
        """Расчет скора режима манипуляции."""
        try:
            # Простая эвристика: аномальные паттерны в объеме
            obv = self._get_last(self.indicators["obv"])
            if obv is None:
                return 0.0
            # Высокая волатильность OBV может указывать на манипуляции
            return min(abs(obv) / self.agent_config["obv_threshold"], 1.0)
        except Exception as e:
            logger.error(f"Error calculating manipulation score: {e}")
            return 0.0

    def _calculate_volatility_score(self) -> float:
        """Расчет скора волатильного режима."""
        try:
            atr = self._get_last(self.indicators["atr"])
            if atr is None:
                return 0.0
            # Высокий ATR указывает на высокую волатильность
            return min(atr / self.agent_config["volatility_threshold"], 1.0)
        except Exception as e:
            logger.error(f"Error calculating volatility score: {e}")
            return 0.0

    def _calculate_anomaly_score(self) -> float:
        """Расчет скора аномального режима."""
        try:
            # Аномалия определяется как несоответствие ожидаемым паттернам
            wave_clusters = self._get_last(self.indicators["wave_clusters"])
            if wave_clusters is None:
                return 0.0
            # Необычные волновые кластеры указывают на аномалию
            return min(len(wave_clusters) / 10, 1.0) if wave_clusters else 0.0
        except Exception as e:
            logger.error(f"Error calculating anomaly score: {e}")
            return 0.0

    def get_current_regime_label(self) -> str:
        """Получение текстового представления текущего режима."""
        return self.current_regime.name if self.current_regime else "UNKNOWN"

    def regime_confidence_score(self) -> float:
        """Получение скора уверенности в текущем режиме."""
        return self.regime_confidence

    def get_regime_history(self) -> List[Tuple[MarketRegime, float]]:
        """Получение истории режимов."""
        return self.regime_history.copy()

    async def get_signals(self) -> List[Dict[str, Any]]:
        """Получение торговых сигналов на основе текущего режима."""
        signals: List[Dict[str, Any]] = []
        try:
            if self.current_regime is None:
                return signals
            # Создание сигнала на основе режима
            signal_data = {
                "action": self._get_action_for_regime(),
                "confidence": self.regime_confidence,
                "position_size": self._get_position_size_for_regime(),
                "stop_loss": 0.0,  # Будет рассчитан отдельно
                "take_profit": 0.0,  # Будет рассчитан отдельно
                "source": "market_regime",
                "timestamp": datetime.now(),
                "explanation": f"Signal based on {self.current_regime.name} regime",
            }
            signals.append(signal_data)
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
        return signals

    def _get_action_for_regime(self) -> str:
        """Определение действия на основе режима."""
        if self.current_regime == MarketRegime.TREND:
            return "buy"  # Следование тренду
        elif self.current_regime == MarketRegime.REVERSAL:
            return "sell"  # Ожидание разворота
        elif self.current_regime == MarketRegime.MANIPULATION:
            return "hold"  # Избегание манипуляций
        else:
            return "hold"  # Остальные режимы

    def _get_position_size_for_regime(self) -> float:
        """Определение размера позиции на основе режима."""
        if self.current_regime == MarketRegime.TREND and self.regime_confidence > 0.8:
            return 1.0  # Полная позиция
        elif (
            self.current_regime == MarketRegime.REVERSAL
            and self.regime_confidence > 0.7
        ):
            return 0.5  # Половина позиции
        else:
            return 0.0  # Без позиции

    def detect_regime_change(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Определение изменения рыночного режима."""
        try:
            if len(self.regime_history) < 2:
                return {"changed": False, "from": None, "to": None}
            previous_regime, previous_confidence = self.regime_history[-2]
            current_regime, current_confidence = self.regime_history[-1]
            regime_changed = previous_regime != current_regime
            confidence_change = current_confidence - previous_confidence
            return {
                "changed": regime_changed,
                "from": previous_regime.name if regime_changed else None,
                "to": current_regime.name if regime_changed else None,
                "confidence_change": confidence_change,
                "previous_confidence": previous_confidence,
                "current_confidence": current_confidence,
            }
        except Exception as e:
            logger.error(f"Error detecting regime change: {e}")
            return {"changed": False, "from": None, "to": None}
