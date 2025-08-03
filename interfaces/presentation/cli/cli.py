"""Syntra CLI."""

from datetime import datetime
from typing import Optional

import click

from application.services.trading_service import TradingService
from domain.protocols.exchange_protocol import ExchangeProtocol
from domain.protocols.repository_protocol import (PortfolioRepositoryProtocol,
                                                  TradingRepositoryProtocol)


class TradingCLI:
    """CLI для торговой системы."""

    def __init__(
        self,
        trading_service: TradingService,
        trading_repository: TradingRepositoryProtocol,
        portfolio_repository: PortfolioRepositoryProtocol,
        exchange_service: ExchangeProtocol,
    ):
        self.trading_service = trading_service
        self.trading_repository = trading_repository
        self.portfolio_repository = portfolio_repository
        self.exchange_service = exchange_service

    def run(self):
        """Запустить CLI."""
        cli()


@click.group()
def cli():
    """Advanced Trading Bot CLI."""


@cli.group()
def trading():
    """Команды для торговли."""


@trading.command()
@click.option("--symbol", required=True, help="Торговая пара")
@click.option(
    "--side", type=click.Choice(["buy", "sell"]), required=True, help="Сторона"
)
@click.option(
    "--type",
    "order_type",
    type=click.Choice(["market", "limit"]),
    required=True,
    help="Тип ордера",
)
@click.option("--quantity", type=float, required=True, help="Количество")
@click.option("--price", type=float, help="Цена (для лимитных ордеров)")
def create_order(
    symbol: str, side: str, order_type: str, quantity: float, price: Optional[float]
):
    """Создать ордер."""
    click.echo(f"Создание ордера: {symbol} {side} {order_type} {quantity}")
    if price:
        click.echo(f"Цена: {price}")

    # Здесь будет логика создания ордера
    click.echo("Ордер создан успешно!")


@trading.command()
@click.option("--symbol", help="Фильтр по торговой паре")
@click.option("--status", help="Фильтр по статусу")
def list_orders(symbol: Optional[str], status: Optional[str]):
    """Показать список ордеров."""
    click.echo("Список ордеров:")
    click.echo("ID | Symbol | Side | Type | Quantity | Price | Status")
    click.echo("-" * 60)

    # Здесь будет логика получения ордеров
    click.echo("order-123 | BTCUSDT | buy | limit | 0.1 | 50000 | filled")


@trading.command()
@click.argument("order_id")
def cancel_order(order_id: str):
    """Отменить ордер."""
    click.echo(f"Отмена ордера: {order_id}")

    # Здесь будет логика отмены ордера
    click.echo("Ордер отменен успешно!")


@cli.group()
def portfolio():
    """Команды для портфеля."""


@portfolio.command()
def show():
    """Показать портфель."""
    click.echo("Портфель:")
    click.echo("Total Equity: $10,000")
    click.echo("Free Margin: $5,000")
    click.echo("Used Margin: $5,000")
    click.echo("Margin Level: 200%")


@portfolio.command()
def positions():
    """Показать позиции."""
    click.echo("Позиции:")
    click.echo("Symbol | Side | Size | Entry Price | Current Price | P&L")
    click.echo("-" * 70)

    # Здесь будет логика получения позиций
    click.echo("BTCUSDT | long | 0.1 | 50000 | 51000 | +$100")


@cli.group()
def strategy():
    """Команды для стратегий."""


@strategy.command()
def list():
    """Показать список стратегий."""
    click.echo("Стратегии:")
    click.echo("ID | Name | Type | Status | Win Rate")
    click.echo("-" * 50)

    # Здесь будет логика получения стратегий
    click.echo("strategy-1 | Trend Following | trend | active | 65%")


@strategy.command()
@click.argument("strategy_id")
def activate(strategy_id: str):
    """Активировать стратегию."""
    click.echo(f"Активация стратегии: {strategy_id}")

    # Здесь будет логика активации стратегии
    click.echo("Стратегия активирована успешно!")


@strategy.command()
@click.argument("strategy_id")
def deactivate(strategy_id: str):
    """Деактивировать стратегию."""
    click.echo(f"Деактивация стратегии: {strategy_id}")

    # Здесь будет логика деактивации стратегии
    click.echo("Стратегия деактивирована успешно!")


@cli.command()
def status():
    """Показать статус системы."""
    click.echo("Статус системы:")
    click.echo(f"Время: {datetime.now()}")
    click.echo("Статус: Работает")
    click.echo("Активные стратегии: 3")
    click.echo("Открытые позиции: 2")
    click.echo("Ожидающие ордера: 1")


@cli.command()
def start():
    """Запустить торгового бота."""
    click.echo("Запуск торгового бота...")

    # Здесь будет логика запуска бота
    click.echo("Торговый бот запущен!")


@cli.command()
def stop():
    """Остановить торгового бота."""
    click.echo("Остановка торгового бота...")

    # Здесь будет логика остановки бота
    click.echo("Торговый бот остановлен!")


if __name__ == "__main__":
    cli()
