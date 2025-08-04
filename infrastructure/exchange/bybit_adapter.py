import hashlib
import hmac
import time
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, cast
from uuid import UUID

import aiohttp

from domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from domain.entities.portfolio_fixed import Balance, Position
from domain.exceptions import ExchangeError
from domain.type_definitions import OrderId, VolumeValue
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.volume import Volume


class BybitAdapter:
    """Адаптер для работы с Bybit API"""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = (
            "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
        )
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "BybitAdapter":
        """Асинхронный контекстный менеджер - вход"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> None:
        """Асинхронный контекстный менеджер - выход"""
        if self.session:
            await self.session.close()

    def _generate_signature(self, timestamp: str, recv_window: str, params: str) -> str:
        """Генерация подписи для API запросов"""
        param_str = timestamp + self.api_key + recv_window + params
        return hmac.new(
            self.api_secret.encode("utf-8"), param_str.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Dict[str, Any]:
        """Выполнить HTTP запрос к API"""
        if not self.session:
            raise ExchangeError("Session not initialized")

        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}

        if signed:
            timestamp = str(int(time.time() * 1000))
            recv_window = "5000"

            # Подготовить параметры для подписи
            if params:
                param_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
            elif data:
                param_str = "&".join([f"{k}={v}" for k, v in sorted(data.items())])
            else:
                param_str = ""

            signature = self._generate_signature(timestamp, recv_window, param_str)

            headers.update(
                {
                    "X-BAPI-API-KEY": self.api_key,
                    "X-BAPI-SIGN": signature,
                    "X-BAPI-SIGN-TYPE": "2",
                    "X-BAPI-TIMESTAMP": timestamp,
                    "X-BAPI-RECV-WINDOW": recv_window,
                }
            )

        try:
            if method.upper() == "GET":
                async with self.session.get(
                    url, params=params, headers=headers
                ) as response:
                    result = await response.json()
            elif method.upper() == "POST":
                async with self.session.post(
                    url, json=data, headers=headers
                ) as response:
                    result = await response.json()
            else:
                raise ExchangeError(f"Unsupported HTTP method: {method}")

            if response.status != 200:
                raise ExchangeError(f"API request failed: {result}")

            return result

        except aiohttp.ClientError as e:
            raise ExchangeError(f"Network error: {e}")
        except Exception as e:
            raise ExchangeError(f"Request failed: {e}")

    async def get_account_info(self) -> Dict[str, Any]:
        """Получить информацию об аккаунте"""
        endpoint = "/v5/account/wallet-balance"
        params = {"accountType": "UNIFIED"}

        result = await self._make_request("GET", endpoint, params=params, signed=True)
        return cast(Dict[str, Any], result.get("result", {}))

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Получить позиции"""
        endpoint = "/v5/position/list"
        params = {"category": "linear"}

        result = await self._make_request("GET", endpoint, params=params, signed=True)
        return cast(List[Dict[str, Any]], result.get("result", {}).get("list", []))

    async def get_open_orders(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Получить открытые ордера"""
        endpoint = "/v5/order/realtime"
        params = {"category": "linear"}

        if symbol:
            params["symbol"] = symbol

        result = await self._make_request("GET", endpoint, params=params, signed=True)
        return cast(List[Dict[str, Any]], result.get("result", {}).get("list", []))

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        qty: str,
        price: Optional[str] = None,
        stop_loss: Optional[str] = None,
        take_profit: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Разместить ордер"""
        endpoint = "/v5/order/create"
        data = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": qty,
        }

        if price:
            data["price"] = price
        if stop_loss:
            data["stopLoss"] = stop_loss
        if take_profit:
            data["takeProfit"] = take_profit

        result = await self._make_request("POST", endpoint, data=data, signed=True)
        return cast(Dict[str, Any], result.get("result", {}))

    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Отменить ордер"""
        endpoint = "/v5/order/cancel"
        data = {"category": "linear", "symbol": symbol, "orderId": order_id}

        result = await self._make_request("POST", endpoint, data=data, signed=True)
        return cast(Dict[str, Any], result.get("result", {}))

    async def get_market_data(
        self, symbol: str, interval: str = "1", limit: int = 200
    ) -> List[Dict[str, Any]]:
        """Получить рыночные данные"""
        endpoint = "/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }

        result = await self._make_request("GET", endpoint, params=params)
        return cast(List[Dict[str, Any]], result.get("result", {}).get("list", []))

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Получить тикер"""
        endpoint = "/v5/market/tickers"
        params = {"category": "linear", "symbol": symbol}

        result = await self._make_request("GET", endpoint, params=params)
        ticker_list = result.get("result", {}).get("list", [])
        if ticker_list and len(ticker_list) > 0:
            return cast(Dict[str, Any], ticker_list[0])
        return cast(Dict[str, Any], {})

    def convert_order_to_domain(self, bybit_order: Dict[str, Any]) -> Order:
        """Конвертировать ордер Bybit в доменную модель"""
        return Order(
            id=OrderId(UUID(bybit_order.get("orderId", ""))),
            trading_pair=bybit_order.get("symbol", ""),
            side=OrderSide.BUY if bybit_order.get("side") == "Buy" else OrderSide.SELL,
            order_type=OrderType(bybit_order.get("orderType", "").lower()),
            quantity=VolumeValue(Decimal(bybit_order.get("qty", "0"))),
            price=(
                Price(
                    Decimal(bybit_order.get("price", "0")), Currency.USD
                )
                if bybit_order.get("price")
                else None
            ),
            status=OrderStatus(bybit_order.get("orderStatus", "").lower()),
            filled_quantity=VolumeValue(Decimal(bybit_order.get("cumExecQty", "0"))),
            created_at=Timestamp.from_datetime(
                datetime.fromtimestamp(int(bybit_order.get("createdTime", "0")) / 1000)
            ),
            updated_at=Timestamp.from_datetime(
                datetime.fromtimestamp(int(bybit_order.get("updatedTime", "0")) / 1000)
            ),
        )

    def convert_position_to_domain(self, bybit_position: Dict[str, Any]) -> Position:
        """Конвертировать позицию Bybit в доменную модель"""
        return Position(
            id=UUID(bybit_position.get("positionIdx", "")),
            trading_pair=bybit_position.get("symbol", ""),
            side="long" if bybit_position.get("side") == "Buy" else "short",
            quantity=Volume(Decimal(bybit_position.get("size", "0"))),
            average_price=Money(
                Decimal(bybit_position.get("avgPrice", "0")), Currency.USD
            ),
            unrealized_pnl=Money(
                Decimal(bybit_position.get("unrealisedPnl", "0")), Currency.USD
            ),
            margin_used=Money(
                Decimal(bybit_position.get("positionMargin", "0")), Currency.USD
            ),
            leverage=Decimal(bybit_position.get("leverage", "1")),
        )

    def convert_balance_to_domain(self, bybit_balance: Dict[str, Any]) -> Balance:
        """Конвертировать баланс Bybit в доменную модель"""
        return Balance(
            currency=Currency(bybit_balance.get("coin", "USD")),
            available=Money(
                Decimal(bybit_balance.get("availableToWithdraw", "0")), Currency.USD
            ),
            total=Money(Decimal(bybit_balance.get("walletBalance", "0")), Currency.USD),
            locked=Money(Decimal(bybit_balance.get("usedMargin", "0")), Currency.USD),
        )
