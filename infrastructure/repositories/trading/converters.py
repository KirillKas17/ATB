"""
Конвертеры для преобразования между доменными объектами и моделями.
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID, uuid4

from domain.entities.account import Account, Balance
from domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from domain.entities.position import Position
from domain.value_objects.volume import Volume
from domain.entities.trading_pair import TradingPair
from domain.type_definitions import OrderId, PortfolioId, PositionId, PositionSide, Symbol, VolumeValue
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp

from .models import (
    AccountModel,
    LiquidityAnalysisModel,
    OrderModel,
    OrderPatternModel,
    PositionModel,
    TradingMetricsModel,
    TradingPairModel,
)


class TradingEntityConverter:
    """Конвертер для торговых сущностей."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def order_to_model(self, order: Order, account_id: str) -> OrderModel:
        """Преобразование Order в OrderModel."""
        try:
            return OrderModel(
                id=str(order.id),
                trading_pair_id=str(order.trading_pair),
                account_id=account_id,
                side=order.side.value,
                order_type=order.order_type.value,
                quantity=str(order.quantity),
                price=str(order.price.amount) if order.price else None,
                status=order.status.value,
                created_at=str(order.created_at),
                updated_at=str(order.updated_at) if order.updated_at else None,
                filled_quantity=(
                    str(order.filled_quantity)
                    if hasattr(order, "filled_quantity")
                    else "0"
                ),
                remaining_quantity=(
                    str(order.remaining_quantity)
                    if hasattr(order, "remaining_quantity")
                    else "0"
                ),
                average_price=(
                    str(order.average_price)
                    if hasattr(order, "average_price") and order.average_price
                    else None
                ),
                commission=(
                    str(order.commission)
                    if hasattr(order, "commission") and order.commission
                    else None
                ),
                commission_asset=(
                    order.commission_asset
                    if hasattr(order, "commission_asset")
                    else None
                ),
                time_in_force=(
                    order.time_in_force if hasattr(order, "time_in_force") else "GTC"
                ),
                stop_price=(
                    str(order.stop_price)
                    if hasattr(order, "stop_price") and order.stop_price
                    else None
                ),
                iceberg_qty=(
                    str(order.iceberg_qty)
                    if hasattr(order, "iceberg_qty") and order.iceberg_qty
                    else None
                ),
                is_working=order.is_working if hasattr(order, "is_working") else True,
                orig_client_order_id=(
                    order.orig_client_order_id
                    if hasattr(order, "orig_client_order_id")
                    else None
                ),
                update_time=(
                    order.update_time.isoformat()
                    if hasattr(order, "update_time") and order.update_time
                    else None
                ),
                working_time=(
                    order.working_time.isoformat()
                    if hasattr(order, "working_time") and order.working_time
                    else None
                ),
                price_protect=(
                    order.price_protect if hasattr(order, "price_protect") else False
                ),
                self_trade_prevention_mode=(
                    order.self_trade_prevention_mode
                    if hasattr(order, "self_trade_prevention_mode")
                    else "NONE"
                ),
                good_till_date=(
                    order.good_till_date.isoformat()
                    if hasattr(order, "good_till_date") and order.good_till_date
                    else None
                ),
                prevent_match=(
                    order.prevent_match if hasattr(order, "prevent_match") else False
                ),
                prevent_match_expiration_time=(
                    order.prevent_match_expiration_time.isoformat()
                    if hasattr(order, "prevent_match_expiration_time")
                    and order.prevent_match_expiration_time
                    else None
                ),
                used_margin=(
                    str(order.used_margin)
                    if hasattr(order, "used_margin") and order.used_margin
                    else None
                ),
                used_margin_asset=(
                    order.used_margin_asset
                    if hasattr(order, "used_margin_asset")
                    else None
                ),
                is_margin_trade=(
                    order.is_margin_trade
                    if hasattr(order, "is_margin_trade")
                    else False
                ),
                is_isolated=(
                    order.is_isolated if hasattr(order, "is_isolated") else False
                ),
                quote_order_qty=(
                    str(order.quote_order_qty)
                    if hasattr(order, "quote_order_qty") and order.quote_order_qty
                    else None
                ),
                quote_order_qty_asset=(
                    order.quote_order_qty_asset
                    if hasattr(order, "quote_order_qty_asset")
                    else None
                ),
                quote_commission=(
                    str(order.quote_commission)
                    if hasattr(order, "quote_commission") and order.quote_commission
                    else None
                ),
                quote_commission_asset=(
                    order.quote_commission_asset
                    if hasattr(order, "quote_commission_asset")
                    else None
                ),
                quote_precision=(
                    order.quote_precision if hasattr(order, "quote_precision") else None
                ),
                base_precision=(
                    order.base_precision if hasattr(order, "base_precision") else None
                ),
                status_detail=(
                    order.status_detail if hasattr(order, "status_detail") else None
                ),
                strategy_id=(
                    str(order.strategy_id)
                    if hasattr(order, "strategy_id") and order.strategy_id
                    else None
                ),
                strategy_type=(
                    order.strategy_type if hasattr(order, "strategy_type") else None
                ),
                parent_order_id=(
                    str(order.parent_order_id)
                    if hasattr(order, "parent_order_id") and order.parent_order_id
                    else None
                ),
                child_orders=(
                    [str(child_id) for child_id in order.child_orders]
                    if hasattr(order, "child_orders") and order.child_orders
                    else []
                ),
                tags=order.tags if hasattr(order, "tags") else {},
                metadata=order.metadata if hasattr(order, "metadata") else {},
            )
        except Exception as e:
            self.logger.error(f"Error converting order to model: {e}")
            raise

    def model_to_order(
        self, model: OrderModel, trading_pair: TradingPair, account: Account
    ) -> Order:
        """Преобразование OrderModel в Order."""
        try:
            from domain.type_definitions import TradingPair as TradingPairType
            
            return Order(
                id=OrderId(UUID(model.id)) if model.id else OrderId(uuid4()),
                trading_pair=TradingPairType(str(trading_pair)),
                side=OrderSide(model.side),
                order_type=OrderType(model.order_type),
                quantity=VolumeValue(Decimal(model.quantity)),
                price=Price(Decimal(model.price), Currency.USD) if model.price else None,
                status=OrderStatus(model.status),
                created_at=Timestamp.from_iso(model.created_at),
                updated_at=Timestamp.from_iso(model.updated_at) if model.updated_at else Timestamp.now(),
            )
        except Exception as e:
            self.logger.error(f"Error converting model to order: {e}")
            raise

    def position_to_model(self, position: Position, account_id: str) -> PositionModel:
        """Преобразование Position в PositionModel."""
        try:
            return PositionModel(
                id=str(position.id),
                trading_pair_id=str(position.trading_pair),
                account_id=account_id,
                side=position.side.value,
                quantity=str(position.volume.to_decimal()),
                average_price=str(position.entry_price.amount),
                unrealized_pnl=(
                    str(position.unrealized_pnl.amount) if position.unrealized_pnl else None
                ),
                realized_pnl=(
                    str(position.realized_pnl.amount) if position.realized_pnl else None
                ),
                margin_type=(
                    position.margin_type
                    if hasattr(position, "margin_type")
                    else "ISOLATED"
                ),
                isolated_margin=(
                    str(position.isolated_margin)
                    if hasattr(position, "isolated_margin") and position.isolated_margin
                    else None
                ),
                entry_price=(
                    str(position.entry_price)
                    if hasattr(position, "entry_price") and position.entry_price
                    else None
                ),
                mark_price=(
                    str(position.mark_price)
                    if hasattr(position, "mark_price") and position.mark_price
                    else None
                ),
                un_realized_pnl=(
                    str(position.un_realized_pnl)
                    if hasattr(position, "un_realized_pnl") and position.un_realized_pnl
                    else None
                ),
                liquidation_price=(
                    str(position.liquidation_price)
                    if hasattr(position, "liquidation_price")
                    and position.liquidation_price
                    else None
                ),
                leverage=(
                    str(position.leverage) if hasattr(position, "leverage") else "1"
                ),
                margin_ratio=(
                    str(position.margin_ratio)
                    if hasattr(position, "margin_ratio") and position.margin_ratio
                    else None
                ),
                margin_ratio_status=(
                    position.margin_ratio_status
                    if hasattr(position, "margin_ratio_status")
                    else "NORMAL"
                ),
                risk_level=(
                    position.risk_level if hasattr(position, "risk_level") else "LOW"
                ),
                created_at=str(position.created_at),
                updated_at=str(position.updated_at) if position.updated_at else None,
                last_update_time=(
                    position.last_update_time.isoformat()
                    if hasattr(position, "last_update_time")
                    and position.last_update_time
                    else None
                ),
                position_side=(
                    position.position_side
                    if hasattr(position, "position_side")
                    else "BOTH"
                ),
                hedge_mode=(
                    position.hedge_mode if hasattr(position, "hedge_mode") else False
                ),
                open_order_initial_margin=(
                    str(position.open_order_initial_margin)
                    if hasattr(position, "open_order_initial_margin")
                    and position.open_order_initial_margin
                    else None
                ),
                max_notional=(
                    str(position.max_notional)
                    if hasattr(position, "max_notional") and position.max_notional
                    else None
                ),
                bid_notional=(
                    str(position.bid_notional)
                    if hasattr(position, "bid_notional") and position.bid_notional
                    else None
                ),
                ask_notional=(
                    str(position.ask_notional)
                    if hasattr(position, "ask_notional") and position.ask_notional
                    else None
                ),
                strategy_id=(
                    str(position.strategy_id)
                    if hasattr(position, "strategy_id") and position.strategy_id
                    else None
                ),
                strategy_type=(
                    position.strategy_type
                    if hasattr(position, "strategy_type")
                    else None
                ),
                tags=position.tags if hasattr(position, "tags") else {},
                metadata=position.metadata if hasattr(position, "metadata") else {},
            )
        except Exception as e:
            self.logger.error(f"Error converting position to model: {e}")
            raise

    def model_to_position(
        self, model: PositionModel, trading_pair: TradingPair, account: Account, portfolio_id: str
    ) -> Position:
        """Преобразование PositionModel в Position."""
        try:
            from domain.entities.position import PositionSide as DomainPositionSide
            from domain.type_definitions import TradingPair as TradingPairType
            from domain.type_definitions import PortfolioId
            return Position(
                id=PositionId(UUID(model.id)),
                portfolio_id=PortfolioId(UUID(portfolio_id)) if portfolio_id else PortfolioId(uuid4()),
                trading_pair=trading_pair,
                side=DomainPositionSide(model.side),
                volume=Volume(Decimal(model.quantity) if model.quantity else Decimal("0"), Currency.USD),
                entry_price=Price(Decimal(model.average_price) if model.average_price else Decimal("0"), Currency.USD),
                current_price=Price(Decimal(model.average_price) if model.average_price else Decimal("0"), Currency.USD),
                unrealized_pnl=(
                    Money(Decimal(model.unrealized_pnl), Currency.USD)
                    if model.unrealized_pnl
                    else None
                ),
                realized_pnl=(
                    Money(Decimal(model.realized_pnl), Currency.USD)
                    if model.realized_pnl
                    else None
                ),
                created_at=Timestamp.from_iso(model.created_at),
                updated_at=Timestamp.from_iso(model.updated_at) if model.updated_at else Timestamp.now(),
            )
        except Exception as e:
            self.logger.error(f"Error converting model to position: {e}")
            raise

    def trading_pair_to_model(self, trading_pair: TradingPair) -> TradingPairModel:
        """
        Преобразование TradingPair в TradingPairModel.
        """
        try:
            status_value = (
                trading_pair.status.value if hasattr(trading_pair.status, 'value') else str(trading_pair.status)
            )
            return TradingPairModel(
                id=str(trading_pair.symbol),
                symbol=str(trading_pair.symbol),
                base_asset=str(trading_pair.base_currency.code),
                quote_asset=str(trading_pair.quote_currency.code),
                status=status_value,
                base_asset_precision=(
                    trading_pair.base_asset_precision
                    if hasattr(trading_pair, "base_asset_precision")
                    else 8
                ),
                quote_precision=(
                    trading_pair.quote_precision
                    if hasattr(trading_pair, "quote_precision")
                    else 8
                ),
                quote_precision_commission=(
                    trading_pair.quote_precision_commission
                    if hasattr(trading_pair, "quote_precision_commission")
                    else 8
                ),
                order_types=(
                    trading_pair.order_types
                    if hasattr(trading_pair, "order_types")
                    else ["LIMIT", "MARKET"]
                ),
                iceberg_allowed=(
                    trading_pair.iceberg_allowed
                    if hasattr(trading_pair, "iceberg_allowed")
                    else True
                ),
                oco_allowed=(
                    trading_pair.oco_allowed
                    if hasattr(trading_pair, "oco_allowed")
                    else True
                ),
                is_spot_trading_allowed=(
                    trading_pair.is_spot_trading_allowed
                    if hasattr(trading_pair, "is_spot_trading_allowed")
                    else True
                ),
                is_margin_trading_allowed=(
                    trading_pair.is_margin_trading_allowed
                    if hasattr(trading_pair, "is_margin_trading_allowed")
                    else False
                ),
                filters=(
                    trading_pair.filters if hasattr(trading_pair, "filters") else []
                ),
                permissions=(
                    trading_pair.permissions
                    if hasattr(trading_pair, "permissions")
                    else ["SPOT"]
                ),
                default_self_trade_prevention_mode=(
                    trading_pair.default_self_trade_prevention_mode
                    if hasattr(trading_pair, "default_self_trade_prevention_mode")
                    else "NONE"
                ),
                allowed_self_trade_prevention_modes=(
                    trading_pair.allowed_self_trade_prevention_modes
                    if hasattr(trading_pair, "allowed_self_trade_prevention_modes")
                    else ["NONE"]
                ),
                created_at=str(trading_pair.created_at),
                updated_at=str(trading_pair.updated_at) if trading_pair.updated_at else None,
                tags=trading_pair.tags if hasattr(trading_pair, "tags") else {},
                metadata=(
                    trading_pair.metadata if hasattr(trading_pair, "metadata") else {}
                ),
            )
        except Exception as e:
            self.logger.error(f"Error converting trading pair to model: {e}")
            raise

    def model_to_trading_pair(self, model: TradingPairModel) -> TradingPair:
        """
        Преобразование TradingPairModel в TradingPair.
        """
        try:
            base_currency = Currency.from_string(model.base_asset) if model.base_asset else Currency.USD
            quote_currency = Currency.from_string(model.quote_asset) if model.quote_asset else Currency.USD
            
            if base_currency is None or quote_currency is None:
                raise ValueError("Invalid currency data in model")
            
            return TradingPair(
                symbol=Symbol(model.symbol),
                base_currency=base_currency,
                quote_currency=quote_currency,
                created_at=datetime.fromisoformat(model.created_at),
                updated_at=(
                    datetime.fromisoformat(model.updated_at)
                    if model.updated_at
                    else datetime.now()
                ),
            )
        except Exception as e:
            self.logger.error(f"Error converting model to trading pair: {e}")
            raise

    def account_to_model(self, account: Account) -> AccountModel:
        """Преобразование Account в AccountModel."""
        try:
            return AccountModel(
                id=account.id,
                name=account.exchange_name,
                email=account.email if hasattr(account, "email") else None,
                status=account.status if hasattr(account, "status") else "ACTIVE",
                account_type=(
                    account.account_type if hasattr(account, "account_type") else "SPOT"
                ),
                permissions=(
                    account.permissions if hasattr(account, "permissions") else ["SPOT"]
                ),
                maker_commission=(
                    str(account.maker_commission)
                    if hasattr(account, "maker_commission")
                    else "0.001"
                ),
                taker_commission=(
                    str(account.taker_commission)
                    if hasattr(account, "taker_commission")
                    else "0.001"
                ),
                buyer_commission=(
                    str(account.buyer_commission)
                    if hasattr(account, "buyer_commission")
                    else "0.001"
                ),
                seller_commission=(
                    str(account.seller_commission)
                    if hasattr(account, "seller_commission")
                    else "0.001"
                ),
                can_trade=account.can_trade if hasattr(account, "can_trade") else True,
                can_withdraw=(
                    account.can_withdraw if hasattr(account, "can_withdraw") else True
                ),
                can_deposit=(
                    account.can_deposit if hasattr(account, "can_deposit") else True
                ),
                update_time=(
                    account.update_time.isoformat()
                    if hasattr(account, "update_time") and account.update_time
                    else None
                ),
                balances=[
                    {
                        "asset": balance.currency,
                        "free": str(balance.available),
                        "locked": str(balance.locked),
                    }
                    for balance in account.balances
                ],
                created_at=account.created_at.isoformat(),
                updated_at=(
                    account.updated_at.isoformat() if account.updated_at else None
                ),
                tags=account.tags if hasattr(account, "tags") else {},
                metadata=account.metadata if hasattr(account, "metadata") else {},
            )
        except Exception as e:
            self.logger.error(f"Error converting account to model: {e}")
            raise

    def model_to_account(self, model: AccountModel) -> Account:
        """Преобразование AccountModel в Account."""
        try:
            balances = []
            for balance_data in model.balances:
                currency = balance_data["asset"]
                available = Decimal(balance_data["free"])
                locked = Decimal(balance_data.get("locked", "0"))
                balance = Balance(currency, available, locked)
                balances.append(balance)
            return Account(
                account_id=model.id,
                exchange_name=model.name,
                balances=balances,
                created_at=datetime.fromisoformat(model.created_at),
                updated_at=(
                    datetime.fromisoformat(model.updated_at)
                    if model.updated_at
                    else datetime.now()
                ),
            )
        except Exception as e:
            self.logger.error(f"Error converting model to account: {e}")
            raise
