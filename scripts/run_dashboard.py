import asyncio
import json
import signal
import sys
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import psutil
import uvicorn
from dashboard.api import create_api
from dashboard.websocket import create_websocket_app
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger


@dataclass
class DashboardConfig:
    """Конфигурация дашборда"""

    host: str = "0.0.0.0"
    port: int = 5000
    log_level: str = "info"
    reload: bool = True
    workers: int = 1
    timeout: int = 60
    max_connections: int = 1000
    static_dir: str = "dashboard/static"
    api_prefix: str = "/api"
    ws_prefix: str = "/ws"
    browser_url: str = "http://localhost:5000"
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    allowed_methods: List[str] = field(default_factory=lambda: ["*"])
    allowed_headers: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class DashboardMetrics:
    """Метрики дашборда"""

    start_time: datetime
    uptime: float
    memory_usage: float
    cpu_usage: float
    active_connections: int
    total_requests: int
    error_count: int
    last_error: Optional[str] = None


class DashboardManager:
    """Менеджер дашборда"""

    def __init__(self, config: Optional[DashboardConfig] = None):
        """Инициализация менеджера"""
        self.config = config or DashboardConfig()
        self._server: Optional[uvicorn.Server] = None
        self._start_time: Optional[datetime] = None
        self._metrics_lock = asyncio.Lock()
        self.metrics_history: List[DashboardMetrics] = []

        # Метрики для подсчета
        self._active_connections = 0
        self._total_requests = 0
        self._error_count = 0
        self._last_error: Optional[str] = None

        # Настройка разрешенных origins
        if self.config.allowed_origins is None:
            self.config.allowed_origins = ["*"]
        if self.config.allowed_methods is None:
            self.config.allowed_methods = ["*"]
        if self.config.allowed_headers is None:
            self.config.allowed_headers = ["*"]

    def increment_requests(self):
        """Увеличить счетчик запросов"""
        self._total_requests += 1

    def increment_errors(self, error_message: str):
        """Увеличить счетчик ошибок"""
        self._error_count += 1
        self._last_error = error_message

    def update_connections(self, count: int):
        """Обновить количество активных соединений"""
        self._active_connections = count

    def create_app(self) -> FastAPI:
        """Создание приложения дашборда"""
        try:
            # Создание FastAPI приложения
            app = FastAPI(
                title="Trading Dashboard",
                description="Real-time trading dashboard",
                version="1.0.0",
                docs_url="/docs",
                redoc_url="/redoc",
            )

            # Настройка CORS
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.allowed_origins,
                allow_credentials=True,
                allow_methods=self.config.allowed_methods,
                allow_headers=self.config.allowed_headers,
            )

            # Монтирование статических файлов
            static_dir = Path(self.config.static_dir)
            if static_dir.exists():
                app.mount(
                    "/static", StaticFiles(directory=str(static_dir)), name="static"
                )
            else:
                logger.warning(f"Статическая директория {static_dir} не найдена")

            # Подключение API
            api_app = create_api()
            app.mount(self.config.api_prefix, api_app)

            # Подключение WebSocket
            ws_app = create_websocket_app()
            app.mount(self.config.ws_prefix, ws_app)

            # Добавление обработчиков ошибок
            @app.exception_handler(HTTPException)
            async def http_exception_handler(request, exc):
                return JSONResponse(
                    status_code=exc.status_code, content={"detail": exc.detail}
                )

            @app.exception_handler(Exception)
            async def general_exception_handler(request, exc):
                logger.error(f"Необработанная ошибка: {str(exc)}")
                return JSONResponse(
                    status_code=500, content={"detail": "Internal server error"}
                )

            # Добавление middleware для метрик
            @app.middleware("http")
            async def metrics_middleware(request, call_next):
                datetime.now()
                try:
                    response = await call_next(request)
                    return response
                except Exception as e:
                    logger.error(f"Ошибка обработки запроса: {str(e)}")
                    raise
                finally:
                    await self._update_metrics()

            return app

        except Exception as e:
            logger.error(f"Ошибка создания приложения: {str(e)}")
            raise

    async def _update_metrics(self):
        """Обновление метрик"""
        try:
            async with self._metrics_lock:
                if self._start_time is None:
                    return

                current_time = datetime.now()
                uptime = (current_time - self._start_time).total_seconds()

                # Получение системных метрик
                process = psutil.Process()
                memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                cpu_usage = process.cpu_percent()

                # Создание метрик
                metrics = DashboardMetrics(
                    start_time=self._start_time,
                    uptime=uptime,
                    memory_usage=memory_usage,
                    cpu_usage=cpu_usage,
                    active_connections=self._active_connections,
                    total_requests=self._total_requests,
                    error_count=self._error_count,
                    last_error=self._last_error,
                )

                self.metrics_history.append(metrics)

                # Ограничение истории метрик
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]

        except Exception as e:
            logger.error(f"Ошибка обновления метрик: {str(e)}")

    def _handle_shutdown(self, signum, frame):
        """Обработка сигналов завершения"""
        logger.info("Получен сигнал завершения")
        if self._server:
            self._server.should_exit = True

    async def start(self):
        """Запуск дашборда"""
        try:
            # Загрузка переменных окружения
            load_dotenv()

            # Создание приложения
            app = self.create_app()

            # Открытие браузера
            try:
                webbrowser.open(self.config.browser_url)
            except Exception as e:
                logger.error(f"Ошибка открытия браузера: {str(e)}")

            # Регистрация обработчиков сигналов
            signal.signal(signal.SIGINT, self._handle_shutdown)
            signal.signal(signal.SIGTERM, self._handle_shutdown)

            # Запуск сервера
            self._start_time = datetime.now()
            config = uvicorn.Config(
                app,
                host=self.config.host,
                port=self.config.port,
                log_level=self.config.log_level,
                reload=self.config.reload,
                workers=self.config.workers,
                timeout_keep_alive=self.config.timeout,
                limit_concurrency=self.config.max_connections,
            )
            self._server = uvicorn.Server(config)
            await self._server.serve()

        except Exception as e:
            logger.error(f"Ошибка запуска дашборда: {str(e)}")
            raise

    async def stop(self):
        """Остановка дашборда"""
        try:
            if self._server:
                self._server.should_exit = True
                await self._server.shutdown()

            # Сохранение метрик
            metrics_file = Path("logs/dashboard_metrics.json")
            metrics_file.parent.mkdir(parents=True, exist_ok=True)

            with open(metrics_file, "w") as f:
                json.dump(
                    [m.__dict__ for m in self.metrics_history], f, indent=2, default=str
                )

            logger.info("Дашборд остановлен")

        except Exception as e:
            logger.error(f"Ошибка остановки дашборда: {str(e)}")
            raise


async def main():
    """Основная функция"""
    try:
        # Инициализация менеджера
        manager = DashboardManager()

        # Запуск дашборда
        await manager.start()

    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Настройка логирования
    logger.add(
        "logs/dashboard_{time}.log", rotation="1 day", retention="7 days", level="INFO"
    )

    # Запуск асинхронного main
    asyncio.run(main())
