import sys
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


def setup_logger(name: str) -> "logger":
    """Настройка логгера"""
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        log_path / "trading_bot.log",
        rotation="1 day",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
    )

    return logger.bind(name=name)


def log_trade(trade_data: Dict[str, Any], logger: Optional["logger"] = None) -> None:
    """Логирование торговой операции"""
    if logger is None:
        logger = setup_logger("trade_logger")

    logger.info(
        "Trade executed: {symbol} | {side} | Price: {price} | Size: {size} | PnL: {pnl}",
        symbol=trade_data.get("symbol"),
        side=trade_data.get("side"),
        price=trade_data.get("price"),
        size=trade_data.get("size"),
        pnl=trade_data.get("pnl", 0),
    )


def log_error(error: Exception, context: Dict[str, Any], logger: Optional["logger"] = None) -> None:
    """Логирование ошибки"""
    if logger is None:
        logger = setup_logger("error_logger")

    logger.error("Error occurred: {error} | Context: {context}", error=str(error), context=context)
