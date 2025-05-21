import logging
from typing import Dict

logger = logging.getLogger(__name__)


def is_valid_order(order: Dict) -> bool:
    """
    Проверка валидности ордера.

    Args:
        order: Ордер для проверки

    Returns:
        bool: True если ордер валиден
    """
    try:
        # Проверка наличия обязательных полей
        required_fields = ["id", "symbol", "type", "side", "price", "amount"]
        if not all(field in order for field in required_fields):
            return False

        # Проверка значений полей
        if order["price"] <= 0 or order["amount"] <= 0:
            return False

        return True

    except Exception as e:
        logger.error(f"Ошибка при проверке валидности ордера: {str(e)}")
        return False


async def clear_invalid_orders(exchange, trading_pairs: Dict) -> None:
    """
    Очистка невалидных ордеров.

    Args:
        exchange: Объект биржи
        trading_pairs: Словарь торговых пар
    """
    try:
        # Получение всех открытых ордеров
        open_orders = await exchange.get_open_orders()

        # Проверка каждого ордера
        for order in open_orders:
            if not is_valid_order(order) or order["symbol"] not in trading_pairs:
                await exchange.cancel_order(order.id)
                logger.info(f"Отменен невалидный ордер: {order.id}")

    except Exception as e:
        logger.error(f"Ошибка при очистке невалидных ордеров: {str(e)}")
        raise
