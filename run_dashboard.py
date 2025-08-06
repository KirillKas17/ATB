#!/usr/bin/env python3
"""
Запуск дашборда ATB с проверкой передачи данных.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import websockets

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Создание FastAPI приложения
app = FastAPI(title="ATB Dashboard", version="1.0.0")

# Монтирование статических файлов
dashboard_path = Path("interfaces/presentation/dashboard")
if dashboard_path.exists():
    app.mount("/static", StaticFiles(directory=str(dashboard_path)), name="static")

# Хранилище для WebSocket соединений
active_connections: List[WebSocket] = []

# Моковые данные для тестирования
MOCK_DATA = {
    "system_status": {
        "status": "online",
        "uptime": "02:15:30",
        "cpu_usage": 23.5,
        "memory_usage": 1.2,
        "active_connections": 5
    },
    "trading_data": {
        "total_pnl": 1234.56,
        "daily_pnl": 123.45,
        "active_positions": 3,
        "win_rate": 78.5,
        "total_trades": 1247,
        "profitable_trades": 978,
        "losing_trades": 269
    },
    "positions": [
        {
            "symbol": "BTC/USDT",
            "side": "long",
            "size": 0.1,
            "entry_price": 45000,
            "current_price": 45200,
            "pnl": 200,
            "pnl_percent": 0.44
        },
        {
            "symbol": "ETH/USDT",
            "side": "short",
            "size": 1.5,
            "entry_price": 3200,
            "current_price": 3180,
            "pnl": 30,
            "pnl_percent": 0.94
        }
    ],
    "analytics": {
        "rsi": 45.2,
        "macd": 0.023,
        "bollinger_position": "middle",
        "ai_signals": [
            {"type": "buy", "strength": "strong", "symbol": "BTC", "message": "Сильный сигнал на покупку BTC"},
            {"type": "sell", "strength": "weak", "symbol": "ETH", "message": "Слабый сигнал на продажу ETH"}
        ]
    }
}

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Главная страница дашборда."""
    html_file = dashboard_path / "index.html"
    if html_file.exists():
        with open(html_file, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head><title>ATB Dashboard</title></head>
        <body>
            <h1>ATB Dashboard</h1>
            <p>Файл index.html не найден в папке dashboard</p>
        </body>
        </html>
        """)

@app.get("/api/status")
async def get_system_status():
    """Получение статуса системы."""
    logger.info("Запрос статуса системы")
    return MOCK_DATA["system_status"]

@app.get("/api/trading")
async def get_trading_data():
    """Получение торговых данных."""
    logger.info("Запрос торговых данных")
    return MOCK_DATA["trading_data"]

@app.get("/api/positions")
async def get_positions():
    """Получение активных позиций."""
    logger.info("Запрос позиций")
    return MOCK_DATA["positions"]

@app.get("/api/analytics")
async def get_analytics():
    """Получение аналитических данных."""
    logger.info("Запрос аналитики")
    return MOCK_DATA["analytics"]

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint для real-time данных."""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WebSocket подключен. Всего соединений: {len(active_connections)}")
    
    try:
        while True:
            # Отправка данных каждые 5 секунд
            await asyncio.sleep(5)
            
            # Обновление времени работы
            MOCK_DATA["system_status"]["uptime"] = "02:15:35"
            MOCK_DATA["system_status"]["active_connections"] = len(active_connections)
            
            # Отправка обновленных данных
            await websocket.send_text(json.dumps({
                "type": "data_update",
                "timestamp": datetime.now().isoformat(),
                "data": MOCK_DATA
            }))
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"WebSocket отключен. Осталось соединений: {len(active_connections)}")
    except Exception as e:
        logger.error(f"Ошибка WebSocket: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

@app.get("/api/health")
async def health_check():
    """Проверка здоровья API."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(active_connections),
        "dashboard_files": {
            "index_html": (dashboard_path / "index.html").exists(),
            "dashboard_js": (dashboard_path / "dashboard.js").exists(),
            "style_css": (dashboard_path / "style.css").exists()
        }
    }

async def broadcast_data():
    """Отправка данных всем подключенным клиентам."""
    while True:
        if active_connections:
            # Обновление данных
            MOCK_DATA["system_status"]["uptime"] = "02:15:40"
            MOCK_DATA["system_status"]["active_connections"] = len(active_connections)
            
            message = json.dumps({
                "type": "broadcast",
                "timestamp": datetime.now().isoformat(),
                "data": MOCK_DATA
            })
            
            # Отправка всем подключенным клиентам
            for connection in active_connections[:]:  # Копия списка для безопасного удаления
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Ошибка отправки данных: {e}")
                    if connection in active_connections:
                        active_connections.remove(connection)
        
        await asyncio.sleep(10)  # Обновление каждые 10 секунд

@app.on_event("startup")
async def startup_event():
    """Событие запуска приложения."""
    logger.info("Запуск ATB Dashboard...")
    logger.info(f"Дашборд доступен по адресу: http://localhost:8000")
    logger.info(f"API документация: http://localhost:8000/docs")
    
    # Запуск фоновой задачи для broadcast
    asyncio.create_task(broadcast_data())

def check_dashboard_files():
    """Проверка наличия файлов дашборда."""
    required_files = ["index.html", "dashboard.js", "style.css"]
    missing_files = []
    
    for file_name in required_files:
        file_path = dashboard_path / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        logger.warning(f"Отсутствуют файлы дашборда: {missing_files}")
        return False
    
    logger.info("Все файлы дашборда найдены")
    return True

def main():
    """Главная функция."""
    print("=" * 60)
    print("           ATB Dashboard Launcher")
    print("=" * 60)
    print()
    
    # Проверка файлов дашборда
    if not check_dashboard_files():
        print("Предупреждение: Некоторые файлы дашборда отсутствуют")
        print("Дашборд может работать некорректно")
        print()
    
    # Проверка API
    print("Проверка API endpoints:")
    print("  - GET /api/status - Статус системы")
    print("  - GET /api/trading - Торговые данные")
    print("  - GET /api/positions - Активные позиции")
    print("  - GET /api/analytics - Аналитические данные")
    print("  - GET /api/health - Проверка здоровья")
    print("  - WS /ws - WebSocket для real-time данных")
    print()
    
    # Запуск сервера
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False
        )
    except KeyboardInterrupt:
        print("\nДашборд остановлен")
    except Exception as e:
        logger.error(f"Ошибка запуска сервера: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()