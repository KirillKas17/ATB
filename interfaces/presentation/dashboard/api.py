import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Any

import aiohttp
# from core.correlation_chain import CorrelationChain  # Временно отключен
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from loguru import logger
# from ml.meta_learning import MetaLearning  # Временно отключен
from pydantic import BaseModel, field_validator
from redis import asyncio as aioredis

# from .status import broadcast_status  # Временно отключен
# from .status import get_all_pairs_status  # Временно отключен

# Импорт системы аналитики
try:
    from infrastructure.entity_system.core import (force_entity_analysis,
                                                   get_entity_status)

    ENTITY_AVAILABLE = True
except ImportError:
    ENTITY_AVAILABLE = False
    # Создаем заглушки для отсутствующих функций
    async def force_entity_analysis() -> dict[str, Any]:
        """Заглушка для отсутствующей функции force_entity_analysis"""
        return {"status": "unavailable", "message": "Entity system not available"}
    
    def get_entity_status() -> dict[str, Any]:
        """Заглушка для отсутствующей функции get_entity_status"""
        return {"status": "unavailable", "message": "Entity system not available"}

# Импорт эволюционной системы
try:
    from core.efficiency_validator import efficiency_validator
    from core.evolution_manager import evolution_manager

    EVOLUTION_AVAILABLE = True
except ImportError:
    EVOLUTION_AVAILABLE = False

# Конфигурация логгера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trading_bot.log"), logging.StreamHandler()],
)
api_logger: logging.Logger = logging.getLogger(__name__)

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
bot_state: Dict[str, Any] = {
    "running": False,
    "training": False,
    "current_symbol": None,
    "current_timeframe": None,
    "last_update": datetime.now(),
    "errors": [],
    "warnings": [],
}

# Глобальные переменные для хранения состояния
system_status = {
    "is_running": False,
    "start_time": None,
    "uptime": 0,
    "last_update": datetime.now(),
}

# Симуляция данных для демонстрации
mock_data: Dict[str, Any] = {
    "pnl_data": [],
    "positions": [],
    "trades": [],
    "performance_metrics": {
        "total_pnl": 1250.50,
        "daily_pnl": 45.20,
        "win_rate": 0.78,
        "total_trades": 1247,
        "sharpe_ratio": 2.34,
        "max_drawdown": -8.7,
        "sortino_ratio": 3.12,
        "calmar_ratio": 1.89,
    },
}

# Инициализация компонентов
# meta_learning = MetaLearning()  # Временно отключен
# correlation_chain = CorrelationChain()  # Временно отключен


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

    # Инициализация при запуске
    api_logger.info("ATB Dashboard API starting...")
    system_status["is_running"] = True
    system_status["start_time"] = datetime.now()

    # Инициализация симуляционных данных
    await initialize_mock_data()

    yield

    # Очистка при завершении
    await redis.close()
    for connection in active_connections:
        await connection.close()

    # Очистка при остановке
    api_logger.info("ATB Dashboard API shutting down...")
    system_status["is_running"] = False


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
        api_logger.error(f"Error getting status: {str(e)}")
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
        bot_state["current_symbol"] = str(request.symbol)
        bot_state["current_timeframe"] = str(request.timeframe)
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
        api_logger.error(f"Error starting bot: {str(e)}")
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
        api_logger.error(f"Error stopping bot: {str(e)}")
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
        api_logger.error(f"Error starting training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtest")
async def get_backtest(symbol: str, timeframe: str) -> Dict:
    """Get backtest results for specified symbol and timeframe"""
    try:
        # Получаем данные бэктеста асинхронно
        backtest_data = await get_backtest_data(symbol, timeframe)
        return backtest_data
    except Exception as e:
        api_logger.error(f"Error getting backtest results: {str(e)}")
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
        api_logger.error(f"Error getting logs: {str(e)}")
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
        api_logger.error(f"WebSocket error: {str(e)}")
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
        api_logger.error(f"Error initializing trading: {str(e)}")
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
            # Заглушка для отсутствующей функции
            api_logger.info(f"Subscription request: {data}")
        elif data["type"] == "unsubscribe":
            # Заглушка для отсутствующей функции
            api_logger.info(f"Unsubscription request: {data}")
    except json.JSONDecodeError:
        api_logger.error(f"Invalid JSON message: {message}")
    except Exception as e:
        api_logger.error(f"Error handling WebSocket message: {str(e)}")


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
            api_logger.error(f"Error in periodic status update: {str(e)}")
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


@app.get("/")
async def root():
    """Корневой эндпоинт."""
    return {
        "message": "ATB Dashboard API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v1/status")
async def get_status_v1():
    """Получение статуса системы."""
    try:
        if system_status["start_time"]:
            uptime = (datetime.now() - system_status["start_time"]).total_seconds()
            system_status["uptime"] = uptime

        status_data = {
            "system": {
                "is_running": system_status["is_running"],
                "uptime": system_status["uptime"],
                "last_update": system_status["last_update"].isoformat(),
            },
            "components": {},
        }

        # Статус эволюционной системы
        if EVOLUTION_AVAILABLE:
            try:
                evolution_status = evolution_manager.get_system_health()
                status_data["components"]["evolution"] = {
                    "available": True,
                    "status": evolution_status,
                }
            except Exception as e:
                api_logger.error(f"Error getting evolution status: {e}")
                status_data["components"]["evolution"] = {
                    "available": True,
                    "status": {"error": str(e)},
                }
        else:
            status_data["components"]["evolution"] = {"available": False, "status": {}}

        # Статус системы аналитики
        if ENTITY_AVAILABLE:
            try:
                entity_status = get_entity_status()
                status_data["components"]["entity_analytics"] = {
                    "available": True,
                    "status": entity_status,
                }
            except Exception as e:
                api_logger.error(f"Error getting entity status: {e}")
                status_data["components"]["entity_analytics"] = {
                    "available": True,
                    "status": {"error": str(e)},
                }
        else:
            status_data["components"]["entity_analytics"] = {
                "available": False,
                "status": {},
            }

        system_status["last_update"] = datetime.now()

        return status_data

    except Exception as e:
        api_logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/performance")
async def get_performance():
    """Получение метрик производительности."""
    try:
        return {
            "metrics": mock_data["performance_metrics"],
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        api_logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/pnl")
async def get_pnl_data(period: str = "24h"):
    """Получение данных P&L."""
    try:
        # Фильтрация данных по периоду
        now = datetime.now()
        if period == "1h":
            start_time = now - timedelta(hours=1)
        elif period == "24h":
            start_time = now - timedelta(days=1)
        elif period == "7d":
            start_time = now - timedelta(days=7)
        elif period == "30d":
            start_time = now - timedelta(days=30)
        else:
            start_time = now - timedelta(days=1)

        filtered_data = [
            point
            for point in mock_data["pnl_data"]
            if datetime.fromisoformat(point["timestamp"]) >= start_time
        ]

        return {
            "data": filtered_data,
            "period": period,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        api_logger.error(f"Error getting PnL data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/positions")
async def get_positions():
    """Получение активных позиций."""
    try:
        return {
            "positions": mock_data["positions"],
            "count": len(mock_data["positions"]),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        api_logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/trades")
async def get_trades(limit: int = 50):
    """Получение истории сделок."""
    try:
        trades = mock_data["trades"][:limit]
        return {
            "trades": trades,
            "count": len(trades),
            "total_count": len(mock_data["trades"]),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        api_logger.error(f"Error getting trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# API для эволюционной системы
@app.get("/api/v1/evolution/stats")
async def get_evolution_stats():
    """Получение статистики эволюции."""
    try:
        if not EVOLUTION_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="Evolution system not available"
            )

        stats = efficiency_validator.get_validation_stats()
        return {"stats": stats, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        api_logger.error(f"Error getting evolution stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/evolution/health")
async def get_evolution_health():
    """Получение здоровья эволюционной системы."""
    try:
        if not EVOLUTION_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="Evolution system not available"
            )

        health = evolution_manager.get_system_health()
        return {"health": health, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        api_logger.error(f"Error getting evolution health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/evolution/force/{component_name}")
async def force_evolution(component_name: str):
    """Принудительная эволюция компонента."""
    try:
        if not EVOLUTION_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="Evolution system not available"
            )

        # Здесь должна быть логика принудительной эволюции
        api_logger.info(f"Forcing evolution for component: {component_name}")

        return {
            "message": f"Evolution triggered for {component_name}",
            "component": component_name,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        api_logger.error(f"Error forcing evolution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# API для системы аналитики
@app.get("/api/v1/entity/status")
async def get_entity_status_api():
    """Получение статуса системы аналитики."""
    try:
        if not ENTITY_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="Entity analytics system not available"
            )

        status = get_entity_status()
        return {"status": status, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        api_logger.error(f"Error getting entity status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/entity/analysis_history")
async def get_entity_analysis_history(limit: int = 10):
    """Получение истории анализа."""
    try:
        if not ENTITY_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="Entity analytics system not available"
            )

        # Здесь должна быть логика получения истории анализа
        # Пока возвращаем симуляционные данные
        history = [
            {
                "id": f"analysis_{i}",
                "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
                "type": "code_analysis",
                "files_analyzed": 150 + i * 10,
                "issues_found": 5 + i,
                "hypotheses_generated": 3 + i % 2,
                "status": "completed",
            }
            for i in range(limit)
        ]

        return {
            "history": history,
            "count": len(history),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        api_logger.error(f"Error getting entity analysis history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/entity/applied_improvements")
async def get_entity_applied_improvements(limit: int = 10):
    """Получение примененных улучшений."""
    try:
        if not ENTITY_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="Entity analytics system not available"
            )

        # Здесь должна быть логика получения примененных улучшений
        # Пока возвращаем симуляционные данные
        improvements = [
            {
                "id": f"improvement_{i}",
                "timestamp": (datetime.now() - timedelta(days=i)).isoformat(),
                "type": "code_refactoring",
                "target": f"strategies/strategy_{i}.py",
                "performance_improvement": 0.05 + i * 0.02,
                "confidence": 0.8 + i * 0.05,
                "status": "applied",
            }
            for i in range(limit)
        ]

        return {
            "improvements": improvements,
            "count": len(improvements),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        api_logger.error(f"Error getting entity applied improvements: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/entity/force_analysis")
async def force_entity_analysis_api():
    """Принудительный запуск анализа."""
    try:
        if not ENTITY_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="Entity analytics system not available"
            )

        # Запуск анализа в фоновом режиме
        asyncio.create_task(force_entity_analysis())

        return {
            "message": "Entity analysis started",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        api_logger.error(f"Error forcing entity analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/entity/experiments")
async def get_entity_experiments(limit: int = 10):
    """Получение истории экспериментов."""
    try:
        if not ENTITY_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="Entity analytics system not available"
            )

        # Здесь должна быть логика получения экспериментов
        # Пока возвращаем симуляционные данные
        experiments = [
            {
                "id": f"experiment_{i}",
                "timestamp": (datetime.now() - timedelta(hours=i * 2)).isoformat(),
                "hypothesis_id": f"hypothesis_{i}",
                "type": "code_refactoring",
                "target": f"strategies/strategy_{i}.py",
                "status": "completed",
                "performance_improvement": 0.08 + i * 0.01,
                "confidence": 0.85 + i * 0.02,
                "statistical_significance": 0.95 + i * 0.01,
            }
            for i in range(limit)
        ]

        return {
            "experiments": experiments,
            "count": len(experiments),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        api_logger.error(f"Error getting entity experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Общие API эндпоинты
@app.get("/api/v1/metrics")
async def get_all_metrics():
    """Получение всех метрик системы."""
    try:
        metrics = {
            "system": {
                "uptime": system_status["uptime"],
                "is_running": system_status["is_running"],
            },
            "trading": mock_data["performance_metrics"],
            "evolution": {},
            "entity_analytics": {},
        }

        # Добавление метрик эволюции
        if EVOLUTION_AVAILABLE:
            try:
                evolution_stats = efficiency_validator.get_validation_stats()
                metrics["evolution"] = evolution_stats
            except Exception as e:
                api_logger.error(f"Error getting evolution metrics: {e}")
                metrics["evolution"] = {"error": str(e)}

        # Добавление метрик системы аналитики
        if ENTITY_AVAILABLE:
            try:
                entity_status = get_entity_status()
                metrics["entity_analytics"] = entity_status
            except Exception as e:
                api_logger.error(f"Error getting entity metrics: {e}")
                metrics["entity_analytics"] = {"error": str(e)}

        return {"metrics": metrics, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        api_logger.error(f"Error getting all metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Проверка здоровья API."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "evolution": EVOLUTION_AVAILABLE,
            "entity_analytics": ENTITY_AVAILABLE,
        },
    }


# Обработка ошибок
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Глобальный обработчик исключений."""
    api_logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat(),
        },
    )


async def initialize_mock_data():
    """Инициализация симуляционных данных."""
    # Генерация P&L данных
    start_date = datetime.now() - timedelta(days=30)
    current_pnl = 0

    for i in range(720):  # 30 дней * 24 часа
        timestamp = start_date + timedelta(hours=i)
        pnl_change = (i % 24 - 12) * 2.5 + (i % 7 - 3) * 10  # Симуляция колебаний
        current_pnl += pnl_change

        mock_data["pnl_data"].append(
            {
                "timestamp": timestamp.isoformat(),
                "pnl": current_pnl,
                "change": pnl_change,
            }
        )

    # Генерация позиций
    mock_data["positions"] = [
        {
            "symbol": "BTC/USDT",
            "side": "long",
            "size": 0.05,
            "entry_price": 45000,
            "current_price": 45250,
            "pnl": 125.50,
            "pnl_percent": 2.78,
            "timestamp": datetime.now().isoformat(),
        },
        {
            "symbol": "ETH/USDT",
            "side": "short",
            "size": 0.1,
            "entry_price": 3200,
            "current_price": 3180,
            "pnl": 20.00,
            "pnl_percent": 0.63,
            "timestamp": datetime.now().isoformat(),
        },
    ]

    # Генерация сделок
    for i in range(50):
        mock_data["trades"].append(
            {
                "id": f"trade_{i+1}",
                "symbol": "BTC/USDT" if i % 2 == 0 else "ETH/USDT",
                "side": "buy" if i % 3 == 0 else "sell",
                "size": 0.01 + (i % 10) * 0.005,
                "price": 45000 + (i % 100) * 50,
                "pnl": (i % 20 - 10) * 25.5,
                "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
            }
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
