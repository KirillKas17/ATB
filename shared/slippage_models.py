"""
Улучшенные модели проскальзывания для реалистичного моделирования торговых издержек.
"""

import random
from decimal import Decimal
from enum import Enum
from typing import Dict, Optional, Tuple
import math


class MarketRegime(Enum):
    """Типы рыночных режимов."""
    NORMAL = "normal"
    VOLATILE = "volatile"
    TRENDING = "trending"
    SIDEWAYS = "sideways"
    HIGH_VOLUME = "high_volume"
    LOW_VOLUME = "low_volume"


class SlippageModel:
    """Базовая модель проскальзывания."""
    
    def __init__(self, base_spread_bps: Decimal = Decimal("2")):
        """
        Инициализация модели.
        
        Args:
            base_spread_bps: Базовый спред в базисных пунктах
        """
        self.base_spread_bps = base_spread_bps
    
    def calculate_slippage(
        self, 
        price: Decimal, 
        volume: Decimal, 
        side: str,
        market_regime: MarketRegime = MarketRegime.NORMAL,
        volatility: Optional[Decimal] = None,
        avg_volume: Optional[Decimal] = None
    ) -> Decimal:
        """
        Расчет проскальзывания.
        
        Args:
            price: Цена ордера
            volume: Объем ордера
            side: Сторона ("buy" или "sell")
            market_regime: Текущий рыночный режим
            volatility: Волатильность (если доступна)
            avg_volume: Средний объем торгов
            
        Returns:
            Проскальзывание в абсолютных единицах
        """
        raise NotImplementedError


class LinearSlippageModel(SlippageModel):
    """Линейная модель проскальзывания."""
    
    def __init__(
        self, 
        base_spread_bps: Decimal = Decimal("2"),
        volume_impact_factor: Decimal = Decimal("0.0001")
    ):
        super().__init__(base_spread_bps)
        self.volume_impact_factor = volume_impact_factor
    
    def calculate_slippage(
        self, 
        price: Decimal, 
        volume: Decimal, 
        side: str,
        market_regime: MarketRegime = MarketRegime.NORMAL,
        volatility: Optional[Decimal] = None,
        avg_volume: Optional[Decimal] = None
    ) -> Decimal:
        """Линейная модель: проскальзывание пропорционально объему."""
        
        # Базовое проскальзывание от спреда
        base_slippage = price * self.base_spread_bps / Decimal("10000") / 2
        
        # Влияние объема
        volume_impact = volume * self.volume_impact_factor * price
        
        # Коррекция на рыночный режим
        regime_multiplier = self._get_regime_multiplier(market_regime)
        
        # Коррекция на волатильность
        volatility_multiplier = Decimal("1")
        if volatility:
            volatility_multiplier = Decimal("1") + volatility * 5  # Волатильность усиливает проскальзывание
        
        total_slippage = (base_slippage + volume_impact) * regime_multiplier * volatility_multiplier
        
        # Проскальзывание всегда против направления сделки
        return total_slippage if side == "buy" else -total_slippage
    
    def _get_regime_multiplier(self, regime: MarketRegime) -> Decimal:
        """Получение множителя для рыночного режима."""
        multipliers = {
            MarketRegime.NORMAL: Decimal("1.0"),
            MarketRegime.VOLATILE: Decimal("2.5"),
            MarketRegime.TRENDING: Decimal("1.2"),
            MarketRegime.SIDEWAYS: Decimal("0.8"),
            MarketRegime.HIGH_VOLUME: Decimal("0.7"),
            MarketRegime.LOW_VOLUME: Decimal("1.8")
        }
        return multipliers.get(regime, Decimal("1.0"))


class SquareRootSlippageModel(SlippageModel):
    """Модель проскальзывания с квадратным корнем (более реалистичная)."""
    
    def __init__(
        self, 
        base_spread_bps: Decimal = Decimal("2"),
        impact_coefficient: Decimal = Decimal("0.1"),
        temporary_impact_decay: Decimal = Decimal("0.5")
    ):
        super().__init__(base_spread_bps)
        self.impact_coefficient = impact_coefficient
        self.temporary_impact_decay = temporary_impact_decay
    
    def calculate_slippage(
        self, 
        price: Decimal, 
        volume: Decimal, 
        side: str,
        market_regime: MarketRegime = MarketRegime.NORMAL,
        volatility: Optional[Decimal] = None,
        avg_volume: Optional[Decimal] = None
    ) -> Decimal:
        """Модель квадратного корня: более реалистичное влияние объема."""
        
        # Базовое проскальзывание
        base_slippage = price * self.base_spread_bps / Decimal("10000") / 2
        
        # Постоянное влияние на рынок (permanent impact)
        if avg_volume and avg_volume > 0:
            volume_ratio = volume / avg_volume
            permanent_impact = self.impact_coefficient * (volume_ratio ** Decimal("0.5")) * price
        else:
            permanent_impact = self.impact_coefficient * (volume ** Decimal("0.5")) * price / 100
        
        # Временное влияние (temporary impact) - восстанавливается со временем
        temporary_impact = permanent_impact * self.temporary_impact_decay
        
        # Коррекция на рыночные условия
        regime_multiplier = self._get_regime_multiplier(market_regime)
        
        # Влияние волатильности
        volatility_factor = Decimal("1")
        if volatility:
            volatility_factor = Decimal("1") + volatility * 3
        
        total_impact = (base_slippage + permanent_impact + temporary_impact) * regime_multiplier * volatility_factor
        
        # Добавляем случайный компонент для реалистичности
        random_factor = Decimal(str(1 + random.gauss(0, 0.1)))
        total_impact *= random_factor
        
        return total_impact if side == "buy" else -total_impact
    
    def _get_regime_multiplier(self, regime: MarketRegime) -> Decimal:
        """Получение множителя для рыночного режима."""
        return {
            MarketRegime.NORMAL: Decimal("1.0"),
            MarketRegime.VOLATILE: Decimal("3.0"),
            MarketRegime.TRENDING: Decimal("1.1"),
            MarketRegime.SIDEWAYS: Decimal("0.9"),
            MarketRegime.HIGH_VOLUME: Decimal("0.6"),
            MarketRegime.LOW_VOLUME: Decimal("2.2")
        }.get(regime, Decimal("1.0"))


class AdaptiveSlippageModel(SlippageModel):
    """Адаптивная модель проскальзывания с машинным обучением."""
    
    def __init__(
        self, 
        base_spread_bps: Decimal = Decimal("2"),
        learning_rate: Decimal = Decimal("0.01")
    ):
        super().__init__(base_spread_bps)
        self.learning_rate = learning_rate
        self.historical_data: Dict[str, list] = {
            "volumes": [],
            "slippages": [],
            "regimes": [],
            "volatilities": []
        }
        self.model_weights = {
            "volume": Decimal("0.5"),
            "volatility": Decimal("0.3"),
            "regime": Decimal("0.2")
        }
    
    def calculate_slippage(
        self, 
        price: Decimal, 
        volume: Decimal, 
        side: str,
        market_regime: MarketRegime = MarketRegime.NORMAL,
        volatility: Optional[Decimal] = None,
        avg_volume: Optional[Decimal] = None
    ) -> Decimal:
        """Адаптивная модель с обучением на исторических данных."""
        
        # Базовая модель
        base_slippage = price * self.base_spread_bps / Decimal("10000") / 2
        
        # Предсказание на основе обученной модели
        predicted_impact = self._predict_impact(volume, volatility, market_regime, avg_volume)
        
        total_slippage = base_slippage + predicted_impact * price
        
        # Сохраняем данные для обучения
        self._store_observation(volume, total_slippage, market_regime, volatility)
        
        return total_slippage if side == "buy" else -total_slippage
    
    def _predict_impact(
        self, 
        volume: Decimal, 
        volatility: Optional[Decimal], 
        regime: MarketRegime,
        avg_volume: Optional[Decimal]
    ) -> Decimal:
        """Предсказание влияния на основе модели."""
        
        # Нормализация объема
        if avg_volume and avg_volume > 0:
            volume_feature = volume / avg_volume
        else:
            volume_feature = volume / Decimal("1000")  # Некий базовый объем
        
        # Волатильность как фактор
        volatility_feature = volatility or Decimal("0.02")
        
        # Режим как численное значение
        regime_feature = Decimal(str(list(MarketRegime).index(regime)))
        
        # Простая линейная модель
        impact = (
            self.model_weights["volume"] * volume_feature +
            self.model_weights["volatility"] * volatility_feature +
            self.model_weights["regime"] * regime_feature / 10
        )
        
        return impact / 100  # Масштабируем результат
    
    def _store_observation(
        self, 
        volume: Decimal, 
        slippage: Decimal, 
        regime: MarketRegime,
        volatility: Optional[Decimal]
    ) -> None:
        """Сохранение наблюдения для обучения."""
        self.historical_data["volumes"].append(float(volume))
        self.historical_data["slippages"].append(float(slippage))
        self.historical_data["regimes"].append(regime)
        self.historical_data["volatilities"].append(float(volatility or Decimal("0")))
        
        # Ограничиваем размер истории
        max_history = 1000
        for key in self.historical_data:
            if len(self.historical_data[key]) > max_history:
                self.historical_data[key] = self.historical_data[key][-max_history:]
    
    def update_model(self) -> None:
        """Обновление весов модели на основе исторических данных."""
        if len(self.historical_data["volumes"]) < 10:
            return
        
        # Простое обновление весов на основе корреляции
        # В реальной реализации здесь был бы более сложный алгоритм обучения
        pass


class SlippageCalculator:
    """Основной калькулятор проскальзывания."""
    
    def __init__(self, model: SlippageModel = None) -> None:
        """
        Инициализация калькулятора.
        
        Args:
            model: Модель проскальзывания (по умолчанию квадратный корень)
        """
        self.model = model or SquareRootSlippageModel()
    
    def calculate_execution_price(
        self,
        intended_price: Decimal,
        volume: Decimal,
        side: str,
        market_conditions: Dict = None
    ) -> Tuple[Decimal, Decimal]:
        """
        Расчет фактической цены исполнения с учетом проскальзывания.
        
        Args:
            intended_price: Желаемая цена
            volume: Объем ордера
            side: Сторона сделки
            market_conditions: Рыночные условия
            
        Returns:
            Tuple[фактическая_цена, проскальзывание]
        """
        conditions = market_conditions or {}
        
        slippage = self.model.calculate_slippage(
            price=intended_price,
            volume=volume,
            side=side,
            market_regime=conditions.get("regime", MarketRegime.NORMAL),
            volatility=conditions.get("volatility"),
            avg_volume=conditions.get("avg_volume")
        )
        
        execution_price = intended_price + slippage
        
        return execution_price, slippage
    
    def estimate_trading_costs(
        self,
        orders: list,
        market_conditions: Dict = None
    ) -> Dict[str, Decimal]:
        """
        Оценка общих торговых издержек для серии ордеров.
        
        Args:
            orders: Список ордеров [{"price": Decimal, "volume": Decimal, "side": str}]
            market_conditions: Рыночные условия
            
        Returns:
            Словарь с торговыми издержками
        """
        total_slippage = Decimal("0")
        total_volume = Decimal("0")
        
        for order in orders:
            execution_price, slippage = self.calculate_execution_price(
                order["price"],
                order["volume"],
                order["side"],
                market_conditions
            )
            
            total_slippage += abs(slippage)
            total_volume += order["volume"]
        
        avg_slippage_bps = (total_slippage / sum(o["price"] for o in orders) * Decimal("10000")) if orders else Decimal("0")
        
        return {
            "total_slippage": total_slippage,
            "total_volume": total_volume,
            "avg_slippage_bps": avg_slippage_bps,
            "slippage_percentage": total_slippage / total_volume if total_volume > 0 else Decimal("0")
        }