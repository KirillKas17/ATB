"""
Use case для управления торговыми парами.
Модуль предоставляет функциональность для создания, обновления, валидации
и управления торговыми парами с расчетом метрик и мониторингом состояния.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, cast

from pydantic import BaseModel, Field, validator

from application.types import (
    CreateTradingPairRequest,
    CreateTradingPairResponse,
    GetTradingPairsRequest,
    GetTradingPairsResponse,
    TradingPairMetrics,
    UpdateTradingPairRequest,
    UpdateTradingPairResponse,
)
from domain.entities.market import MarketData, OrderBook
from domain.entities.trading_pair import TradingPair, PairStatus
from domain.repositories.market_repository import MarketRepository
from domain.repositories.trading_pair_repository import TradingPairRepository
from domain.types import Symbol, PriceValue, VolumeValue, PricePrecision, VolumePrecision
from domain.value_objects.currency import Currency
from domain.value_objects.percentage import Percentage
from domain.value_objects.volume import Volume

logger = logging.getLogger(__name__)


class TradingPairManagementUseCase(ABC):
    """Абстрактный use case для управления торговыми парами."""

    @abstractmethod
    async def create_trading_pair(
        self, request: CreateTradingPairRequest
    ) -> CreateTradingPairResponse:
        """Создание новой торговой пары."""
        pass

    @abstractmethod
    async def get_trading_pairs(
        self, request: GetTradingPairsRequest
    ) -> GetTradingPairsResponse:
        """Получение списка торговых пар."""
        pass

    @abstractmethod
    async def update_trading_pair(
        self, request: UpdateTradingPairRequest
    ) -> UpdateTradingPairResponse:
        """Обновление торговой пары."""
        pass

    @abstractmethod
    async def get_trading_pair_by_id(self, trading_pair_id: str) -> Optional[TradingPair]:
        """Получение торговой пары по ID."""
        pass

    @abstractmethod
    async def delete_trading_pair(self, trading_pair_id: str) -> bool:
        """Удаление торговой пары."""
        pass

    @abstractmethod
    async def calculate_trading_pair_metrics(
        self, trading_pair: TradingPair
    ) -> TradingPairMetrics:
        """Расчет метрик торговой пары."""
        pass

    @abstractmethod
    async def validate_trading_pair(self, trading_pair: TradingPair) -> bool:
        """Валидация торговой пары."""
        pass

    @abstractmethod
    async def get_trading_pair_status(self, trading_pair_id: str) -> str:
        """Получение статуса торговой пары."""
        pass

    @abstractmethod
    async def activate_trading_pair(self, trading_pair_id: str) -> bool:
        """Активация торговой пары."""
        pass

    @abstractmethod
    async def deactivate_trading_pair(self, trading_pair_id: str) -> bool:
        """Деактивация торговой пары."""
        pass

    @abstractmethod
    async def calculate_liquidity_metrics(
        self, trading_pair: TradingPair
    ) -> Dict[str, float]:
        """Расчет метрик ликвидности."""
        pass


class DefaultTradingPairManagementUseCase(TradingPairManagementUseCase):
    """Реализация по умолчанию для управления торговыми парами."""

    def __init__(
        self,
        trading_pair_repository: TradingPairRepository,
        market_repository: MarketRepository,
    ):
        self.trading_pair_repository = trading_pair_repository
        self.market_repository = market_repository

    async def create_trading_pair(
        self, request: CreateTradingPairRequest
    ) -> CreateTradingPairResponse:
        """Создание новой торговой пары."""
        try:
            # Получение валют с проверкой
            base_currency = Currency.from_string(request.base_currency)
            quote_currency = Currency.from_string(request.quote_currency)
            
            if base_currency is None or quote_currency is None:
                return CreateTradingPairResponse(
                    success=False,
                    trading_pair=None,
                    message="Invalid currency format",
                )
            
            # Создание торговой пары
            trading_pair = TradingPair(
                symbol=Symbol(request.base_currency + "/" + request.quote_currency),
                base_currency=base_currency,
                quote_currency=quote_currency,
                is_active=request.is_active,
                min_order_size=Volume(Decimal(str(request.min_amount)), base_currency) if request.min_amount else None,
                max_order_size=Volume(Decimal(str(request.max_amount)), base_currency) if request.max_amount else None,
                price_precision=PricePrecision(8),  # Default precision
                volume_precision=VolumePrecision(8),  # Default precision
            )

            # Сохранение торговой пары
            await self.trading_pair_repository.save(trading_pair)

            return CreateTradingPairResponse(
                success=True,
                trading_pair=trading_pair,
                message="Trading pair created successfully",
            )

        except Exception as e:
            logger.error(f"Error creating trading pair: {e}")
            return CreateTradingPairResponse(
                success=False,
                trading_pair=None,
                message=f"Error creating trading pair: {str(e)}",
            )

    async def get_trading_pairs(
        self, request: GetTradingPairsRequest
    ) -> GetTradingPairsResponse:
        """Получение списка торговых пар."""
        try:
            trading_pairs = []
            
            if hasattr(request, 'is_active') and request.is_active is not None:
                # Получение только активных/неактивных пар
                all_pairs = await self.trading_pair_repository.get_all()
                trading_pairs = [pair for pair in all_pairs if pair.is_active == request.is_active]
            else:
                # Получение всех пар
                trading_pairs = await self.trading_pair_repository.get_all()

            # Фильтрация по бирже
            if hasattr(request, 'exchange') and request.exchange:
                trading_pairs = [pair for pair in trading_pairs if request.exchange.lower() in str(pair.symbol).lower()]

            # Фильтрация по базовой валюте
            if hasattr(request, 'base_currency') and request.base_currency:
                trading_pairs = [pair for pair in trading_pairs if request.base_currency.lower() in pair.base_currency.code.lower()]

            # Фильтрация по котируемой валюте
            if hasattr(request, 'quote_currency') and request.quote_currency:
                trading_pairs = [pair for pair in trading_pairs if request.quote_currency.lower() in pair.quote_currency.code.lower()]

            # Фильтрация по статусу
            if hasattr(request, 'status') and request.status:
                trading_pairs = [pair for pair in trading_pairs if pair.status.value == request.status]

            return GetTradingPairsResponse(
                success=True,
                trading_pairs=trading_pairs,
                message=f"Retrieved {len(trading_pairs)} trading pairs",
            )

        except Exception as e:
            logger.error(f"Error getting trading pairs: {e}")
            return GetTradingPairsResponse(
                success=False,
                trading_pairs=[],
                message=f"Error getting trading pairs: {str(e)}",
            )

    async def update_trading_pair(
        self, request: UpdateTradingPairRequest
    ) -> UpdateTradingPairResponse:
        """Обновление торговой пары."""
        try:
            # Получение существующей пары
            trading_pair = await self.trading_pair_repository.get_by_symbol(request.pair_id)
            
            if not trading_pair:
                return UpdateTradingPairResponse(
                    success=False,
                    trading_pair=None,
                    updated=False,
                    message="Trading pair not found",
                )

            # Обновление полей
            if hasattr(request, 'is_active') and request.is_active is not None:
                trading_pair.is_active = request.is_active
            if hasattr(request, 'min_amount') and request.min_amount and trading_pair.base_currency:
                trading_pair.min_order_size = Volume(Decimal(str(request.min_amount)), cast(Currency, trading_pair.base_currency))
            if hasattr(request, 'max_amount') and request.max_amount and trading_pair.base_currency:
                trading_pair.max_order_size = Volume(Decimal(str(request.max_amount)), cast(Currency, trading_pair.base_currency))

            trading_pair.updated_at = datetime.now()

            # Сохранение обновленной пары
            updated_pair = await self.trading_pair_repository.update(trading_pair)

            return UpdateTradingPairResponse(
                success=True,
                trading_pair=updated_pair,
                updated=True,
                message="Trading pair updated successfully",
            )

        except Exception as e:
            logger.error(f"Error updating trading pair: {e}")
            return UpdateTradingPairResponse(
                success=False,
                trading_pair=None,
                updated=False,
                message=f"Error updating trading pair: {str(e)}",
            )

    async def get_trading_pair_by_id(self, trading_pair_id: str) -> Optional[TradingPair]:
        """Получение торговой пары по ID."""
        try:
            return await self.trading_pair_repository.get_by_symbol(trading_pair_id)
        except Exception as e:
            logger.error(f"Error getting trading pair by ID: {e}")
            return None

    async def delete_trading_pair(self, trading_pair_id: str) -> bool:
        """Удаление торговой пары."""
        try:
            return await self.trading_pair_repository.delete(trading_pair_id)
        except Exception as e:
            logger.error(f"Error deleting trading pair: {e}")
            return False

    async def calculate_trading_pair_metrics(
        self, trading_pair: TradingPair
    ) -> TradingPairMetrics:
        """Расчет метрик торговой пары."""
        try:
            # Получение рыночных данных
            market_data_list = await self.market_repository.get_market_data(trading_pair.symbol, "1d")
            if not market_data_list or len(market_data_list) == 0:
                return TradingPairMetrics(
                    volume_24h=VolumeValue(Decimal("0")),
                    price_change_24h=PriceValue(Decimal("0")),
                    price_change_percent_24h=Percentage(Decimal("0")),
                    high_24h=PriceValue(Decimal("0")),
                    low_24h=PriceValue(Decimal("0")),
                    last_price=PriceValue(Decimal("0")),
                    volatility=Decimal("0"),
                )

            # Берем последние данные
            market_data = market_data_list[-1] if market_data_list else None
            if not market_data:
                return TradingPairMetrics(
                    volume_24h=VolumeValue(Decimal("0")),
                    price_change_24h=PriceValue(Decimal("0")),
                    price_change_percent_24h=Percentage(Decimal("0")),
                    high_24h=PriceValue(Decimal("0")),
                    low_24h=PriceValue(Decimal("0")),
                    last_price=PriceValue(Decimal("0")),
                    volatility=Decimal("0"),
                )

            # Расчет метрик
            price_change_24h = PriceValue(Decimal("0"))
            if market_data.close and market_data.open:
                price_change_24h = PriceValue(market_data.close.value - market_data.open.value)

            volume_24h = VolumeValue(market_data.volume.value) if market_data.volume else VolumeValue(Decimal("0"))
            high_24h = PriceValue(market_data.high.value) if market_data.high else PriceValue(Decimal("0"))
            low_24h = PriceValue(market_data.low.value) if market_data.low else PriceValue(Decimal("0"))
            last_price = PriceValue(market_data.close.value) if market_data.close else PriceValue(Decimal("0"))

            # Расчет спреда bid-ask (пока не реализовано)
            bid = None
            ask = None
            spread = None

            # Расчет волатильности (упрощенная версия)
            volatility = Decimal("0")
            if market_data.high and market_data.low:
                volatility = (market_data.high.value - market_data.low.value) / market_data.low.value

            return TradingPairMetrics(
                volume_24h=volume_24h,
                price_change_24h=price_change_24h,
                price_change_percent_24h=Percentage(Decimal("0")),
                high_24h=high_24h,
                low_24h=low_24h,
                last_price=last_price,
                bid=bid,
                ask=ask,
                spread=spread,
                volatility=volatility,
            )

        except Exception as e:
            logger.error(f"Error calculating trading pair metrics: {e}")
            return TradingPairMetrics(
                volume_24h=VolumeValue(Decimal("0")),
                price_change_24h=PriceValue(Decimal("0")),
                price_change_percent_24h=Percentage(Decimal("0")),
                high_24h=PriceValue(Decimal("0")),
                low_24h=PriceValue(Decimal("0")),
                last_price=PriceValue(Decimal("0")),
                volatility=Decimal("0"),
            )

    async def validate_trading_pair(self, trading_pair: TradingPair) -> bool:
        """Валидация торговой пары."""
        try:
            # Проверка базовых условий
            if not trading_pair.symbol:
                return False
            if trading_pair.base_currency == trading_pair.quote_currency:
                return False
            if trading_pair.price_precision < 0 or trading_pair.volume_precision < 0:
                return False

            # Проверка рыночных данных
            market_data_list = await self.market_repository.get_market_data(trading_pair.symbol, "1d")
            if not market_data_list or len(market_data_list) == 0:
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating trading pair: {e}")
            return False

    async def get_trading_pair_status(self, trading_pair_id: str) -> str:
        """Получение статуса торговой пары."""
        try:
            trading_pair = await self.trading_pair_repository.get_by_symbol(trading_pair_id)
            
            if not trading_pair:
                return "NOT_FOUND"
            
            if trading_pair.is_active:
                return "ACTIVE"
            else:
                return "INACTIVE"

        except Exception as e:
            logger.error(f"Error getting trading pair status: {e}")
            return "ERROR"

    async def activate_trading_pair(self, trading_pair_id: str) -> bool:
        """Активация торговой пары."""
        try:
            trading_pair = await self.trading_pair_repository.get_by_symbol(trading_pair_id)
            
            if not trading_pair:
                return False

            trading_pair.activate()
            await self.trading_pair_repository.update(trading_pair)
            
            return True

        except Exception as e:
            logger.error(f"Error activating trading pair: {e}")
            return False

    async def deactivate_trading_pair(self, trading_pair_id: str) -> bool:
        """Деактивация торговой пары."""
        try:
            trading_pair = await self.trading_pair_repository.get_by_symbol(trading_pair_id)
            
            if not trading_pair:
                return False

            trading_pair.deactivate()
            await self.trading_pair_repository.update(trading_pair)
            
            return True

        except Exception as e:
            logger.error(f"Error deactivating trading pair: {e}")
            return False

    async def calculate_liquidity_metrics(
        self, trading_pair: TradingPair
    ) -> Dict[str, float]:
        """Расчет метрик ликвидности."""
        try:
            market_data_list = await self.market_repository.get_market_data(trading_pair.symbol, "1d")
            if not market_data_list or len(market_data_list) == 0:
                return {
                    "bid_ask_spread": 0.0,
                    "order_book_depth": 0.0,
                    "liquidity_score": 0.0,
                }

            market_data = market_data_list[-1]
            if not market_data:
                return {
                    "bid_ask_spread": 0.0,
                    "order_book_depth": 0.0,
                    "liquidity_score": 0.0,
                }

            # Расчет спреда bid-ask (пока не реализовано)
            bid_ask_spread = 0.0
            # Расчет глубины стакана (пока не реализовано)
            order_book_depth = 0.0
            # Расчет оценки ликвидности
            liquidity_score = 0.0
            if bid_ask_spread > 0:
                liquidity_score = min(1.0 / bid_ask_spread, 1.0)
            if order_book_depth > 0:
                liquidity_score = min(liquidity_score + order_book_depth / 1000000, 1.0)
            return {
                "bid_ask_spread": bid_ask_spread,
                "order_book_depth": order_book_depth,
                "liquidity_score": liquidity_score,
            }
        except Exception as e:
            logger.error(f"Error calculating liquidity metrics: {e}")
            return {
                "bid_ask_spread": 0.0,
                "order_book_depth": 0.0,
                "liquidity_score": 0.0,
            }
