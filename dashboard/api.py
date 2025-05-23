import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import aiohttp
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from loguru import logger
from pydantic import BaseModel, field_validator
from redis import asyncio as aioredis

from core.correlation_chain import CorrelationChain
from ml.meta_learning import MetaLearning

from .status import (
    broadcast_status,  # и другие необходимые функции
    get_all_pairs_status,
)

# Конфигурация логгера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trading_bot.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Константы
CACHE_EXPIRE_TIME = 60  # секунды
MAX_WEBSOCKET_CONNECTIONS = 100
WEBSOCKET_PING_INTERVAL = 30  # секунды
WEBSOCKET_PING_TIMEOUT = 10  # секунды

# Доступные торговые пары и таймфреймы
SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "ADA/USDT",
    "XRP/USDT",
    "DOT/USDT",
    "DOGE/USDT",
    "AVAX/USDT",
    "MATIC/USDT",
]

TIMEFRAMES = [
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
    "1d",
    "1w",
]


# Модели данных
class TradingRequest(BaseModel):
    symbol: str
    timeframe: str
    strategy: str

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v):
        if v not in SYMBOLS:
            raise ValueError(f"Invalid symbol: {v}")
        return v

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, v):
        if v not in TIMEFRAMES:
            raise ValueError(f"Invalid timeframe: {v}")
        return v


class BacktestRequest(BaseModel):
    symbol: str
    timeframe: str
    strategy: str
    period: str

    @field_validator("period")
    @classmethod
    def validate_period(cls, v):
        valid_periods = ["1d", "1w", "1m", "3m", "6m", "1y"]
        if v not in valid_periods:
            raise ValueError(f"Invalid period: {v}")
        return v


# Глобальные переменные
active_connections: Set[WebSocket] = set()
bot_state = {
    "running": False,
    "training": False,
    "current_symbol": None,
    "current_timeframe": None,
    "last_update": datetime.now(),
    "errors": [],
    "warnings": [],
}

# Инициализация компонентов
meta_learning = MetaLearning()
correlation_chain = CorrelationChain()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация Redis для кэширования
    redis = aioredis.from_url(
        "redis://localhost", encoding="utf8", decode_responses=True
    )
    FastAPICache.init(RedisBackend(redis), prefix="trading_bot_cache")

    # Запуск фоновых задач
    asyncio.create_task(periodic_status_update())
    asyncio.create_task(cleanup_inactive_connections())

    yield

    # Очистка при завершении
    await redis.close()
    for connection in active_connections:
        await connection.close()


app = FastAPI(title="Trading Bot API", lifespan=lifespan)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Эндпоинты API
@app.get("/symbols")
@cache(expire=CACHE_EXPIRE_TIME)
async def get_symbols() -> List[str]:
    """Получение списка доступных торговых пар"""
    return SYMBOLS


@app.get("/status")
@cache(expire=5)  # Короткое время кэширования для статуса
async def get_status():
    """Получение расширенного статуса торгового бота"""
    try:
        status = {
            "mode": "trading",
            "pairs": await get_all_pairs_status(),
            "system_metrics": await get_system_metrics(),
            "last_update": datetime.now().isoformat(),
            "errors": bot_state["errors"][-5:],  # Последние 5 ошибок
            "warnings": bot_state["warnings"][-5:],  # Последние 5 предупреждений
        }
        return status
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/start")
async def start_bot(request: TradingRequest, background_tasks: BackgroundTasks) -> Dict:
    """Запуск торгового бота"""
    try:
        if bot_state["running"]:
            raise HTTPException(status_code=400, detail="Bot is already running")

        # Проверка доступности биржи
        if not await check_exchange_availability():
            raise HTTPException(status_code=503, detail="Exchange is not available")

        bot_state["running"] = True
        bot_state["current_symbol"] = request.symbol
        bot_state["current_timeframe"] = request.timeframe
        bot_state["last_update"] = datetime.now()

        # Запуск фоновых задач
        background_tasks.add_task(initialize_trading, request)
        background_tasks.add_task(broadcast_status)

        # Отправляем начальный статус
        await broadcast_status()

        return {
            "status": "success",
            "message": f"Bot started for {request.symbol} on {request.timeframe}",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error starting bot: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop")
async def stop_bot(request: TradingRequest) -> Dict:
    """Stop trading bot for specified symbol"""
    try:
        if not bot_state["running"]:
            raise HTTPException(status_code=400, detail="Bot is not running")

        bot_state["running"] = False
        bot_state["current_symbol"] = None
        bot_state["current_timeframe"] = None

        # Notify all connected clients
        await broadcast_status()

        return {"status": "success", "message": f"Bot stopped for {request.symbol}"}
    except Exception as e:
        logger.error(f"Error stopping bot: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def start_training(request: TradingRequest) -> Dict:
    """Start training for specified symbol and timeframe"""
    try:
        if bot_state["training"]:
            raise HTTPException(
                status_code=400, detail="Training is already in progress"
            )

        bot_state["training"] = True

        # Notify all connected clients
        await broadcast_status()

        return {
            "status": "success",
            "message": f"Training started for {request.symbol} on {request.timeframe}",
        }
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtest")
async def get_backtest(symbol: str, timeframe: str) -> Dict:
    """Get backtest results for specified symbol and timeframe"""
    try:
        # Получаем данные бэктеста асинхронно
        backtest_data = await get_backtest_data(symbol, timeframe)
        return backtest_data
    except Exception as e:
        logger.error(f"Error getting backtest results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs")
async def get_logs() -> FileResponse:
    """Get trading bot logs"""
    try:
        log_file = Path("trading_bot.log")
        if not log_file.exists():
            raise HTTPException(status_code=404, detail="Log file not found")
        return FileResponse(log_file, filename="trading_logs.txt")
    except Exception as e:
        logger.error(f"Error getting logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket эндпоинты
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if len(active_connections) >= MAX_WEBSOCKET_CONNECTIONS:
        await websocket.close(code=1008, reason="Maximum connections reached")
        return

    await websocket.accept()
    active_connections.add(websocket)

    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(), timeout=WEBSOCKET_PING_INTERVAL
                )
                await handle_websocket_message(websocket, data)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        active_connections.remove(websocket)


# Вспомогательные функции
async def check_exchange_availability() -> bool:
    """Проверка доступности биржи"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.bybit.com/v5/market/time") as response:
                return response.status == 200
    except Exception:
        return False


async def initialize_trading(request: TradingRequest):
    """Инициализация торговли"""
    try:
        meta_learning.init_pair_structure(request.symbol)
        await correlation_chain.initialize(request.symbol)
        await broadcast_message(
            {
                "type": "initialization_complete",
                "symbol": request.symbol,
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        logger.error(f"Error initializing trading: {str(e)}")
        bot_state["errors"].append(
            {
                "type": "initialization_error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }
        )


async def cleanup_inactive_connections():
    """Очистка неактивных WebSocket соединений"""
    while True:
        await asyncio.sleep(WEBSOCKET_PING_INTERVAL)
        for connection in list(active_connections):
            try:
                await connection.send_json({"type": "ping"})
            except Exception:
                active_connections.remove(connection)


async def handle_websocket_message(websocket: WebSocket, message: str):
    """Обработка входящих WebSocket сообщений"""
    try:
        data = json.loads(message)
        if data["type"] == "subscribe":
            await handle_subscription(websocket, data)
        elif data["type"] == "unsubscribe":
            await handle_unsubscription(websocket, data)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON message: {message}")
    except Exception as e:
        logger.error(f"Error handling WebSocket message: {str(e)}")


async def broadcast_message(message: dict):
    """Отправка сообщения всем подключенным клиентам"""
    for connection in list(active_connections):
        try:
            await connection.send_json(message)
        except Exception:
            active_connections.remove(connection)


async def periodic_status_update():
    """Периодическое обновление статуса"""
    while True:
        try:
            await broadcast_status()
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error in periodic status update: {str(e)}")
            await asyncio.sleep(1)


# Добавляем асинхронную функцию для получения данных бэктеста
async def get_backtest_data(symbol: str, timeframe: str) -> Dict:
    """Get backtest data asynchronously"""
    # Mock backtest data (replace with actual data)
    return {
        "equity_curve": [100, 102, 101, 103, 105, 104, 106],
        "performance_metrics": {
            "total_trades": 100,
            "win_rate": 65.2,
            "profit_factor": 1.8,
            "max_drawdown": 15.5,
            "sharpe_ratio": 1.2,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
