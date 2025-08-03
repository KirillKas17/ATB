"""
Веб-дашборд мониторинга для ATB.
Предоставляет веб-интерфейс для мониторинга производительности,
метрик и состояния системы в реальном времени.
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger

# Импорты для интеграции с системой мониторинга
try:
    from .config_validator import config_validator
    from .metrics_analyzer import MetricsAnalyzer
    from .performance_monitor import performance_monitor
except ImportError:
    # Fallback для случаев, когда модули недоступны
    performance_monitor_fallback: Optional[Any] = None
    MetricsAnalyzer_fallback: Optional[type] = None
    config_validator_fallback: Optional[Any] = None


def get_performance_monitor():
    """Безопасное получение performance_monitor."""
    return globals().get('performance_monitor', performance_monitor_fallback)

def get_metrics_analyzer():
    """Безопасное получение MetricsAnalyzer."""
    return globals().get('MetricsAnalyzer', MetricsAnalyzer_fallback)

def get_config_validator():
    """Безопасное получение config_validator."""
    return globals().get('config_validator', config_validator_fallback)


class DashboardStatus(Enum):
    """Статус дашборда."""

    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    ERROR = "error"


@dataclass
class DashboardMetric:
    """Метрика для дашборда."""

    name: str
    value: float
    unit: str
    trend: str  # "up", "down", "stable"
    status: str  # "normal", "warning", "error", "critical"
    timestamp: datetime
    component: str
    description: Optional[str] = None


@dataclass
class DashboardAlert:
    """Алерт для дашборда."""

    id: str
    level: str
    message: str
    component: str
    timestamp: datetime
    resolved: bool
    acknowledged: bool = False


class WebSocketManager:
    """Менеджер WebSocket соединений."""

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []
        self.connection_data: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """Подключение клиента."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_data[websocket] = {
            "client_id": client_id,
            "connected_at": datetime.now(),
            "subscriptions": set(),
        }
        logger.info(f"WebSocket client {client_id} connected")

    def disconnect(self, websocket: WebSocket):
        """Отключение клиента."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            client_id = self.connection_data.get(websocket, {}).get(
                "client_id", "unknown"
            )
            del self.connection_data[websocket]
            logger.info(f"WebSocket client {client_id} disconnected")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Отправить персональное сообщение."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending message to WebSocket: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: str, exclude: Optional[WebSocket] = None):
        """Отправить сообщение всем клиентам."""
        disconnected = []
        for connection in self.active_connections:
            if connection != exclude:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to WebSocket: {e}")
                    disconnected.append(connection)
        for connection in disconnected:
            self.disconnect(connection)


class MonitoringDashboard:
    """
    Веб-дашборд мониторинга.
    Предоставляет REST API и WebSocket для мониторинга
    производительности и состояния системы.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.app = FastAPI(title="ATB Monitoring Dashboard", version="1.0.0")
        self.websocket_manager = WebSocketManager()
        self.status = DashboardStatus.ONLINE
        self.start_time = datetime.now()
        # Настройка CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        # Инициализация компонентов
        metrics_analyzer_class = get_metrics_analyzer()
        self.metrics_analyzer = metrics_analyzer_class() if metrics_analyzer_class is not None else None
        # Настройка маршрутов
        self._setup_routes()
        # Запуск фоновых задач
        self._start_background_tasks()

    def _setup_routes(self):
        """Настройка маршрутов API."""

        @self.app.get("/", response_class=HTMLResponse)
        async def get_dashboard():
            """Главная страница дашборда."""
            return self._get_dashboard_html()

        @self.app.get("/api/health")
        async def health_check():
            """Проверка здоровья системы."""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "uptime": (datetime.now() - self.start_time).total_seconds(),
                "version": "1.0.0",
            }

        @self.app.get("/api/status")
        async def get_status():
            """Получить статус системы."""
            return {
                "status": self.status.value,
                "start_time": self.start_time.isoformat(),
                "uptime": (datetime.now() - self.start_time).total_seconds(),
                "active_connections": len(self.websocket_manager.active_connections),
            }

        @self.app.get("/api/metrics")
        async def get_metrics():
            """Получить текущие метрики."""
            return await self._get_current_metrics()

        @self.app.get("/api/metrics/{metric_name}")
        async def get_metric(metric_name: str):
            """Получить конкретную метрику."""
            return await self._get_metric_details(metric_name)

        @self.app.get("/api/alerts")
        async def get_alerts():
            """Получить активные алерты."""
            return await self._get_active_alerts()

        @self.app.post("/api/alerts/{alert_id}/acknowledge")
        async def acknowledge_alert(alert_id: str):
            """Подтвердить алерт."""
            return await self._acknowledge_alert(alert_id)

        @self.app.get("/api/performance")
        async def get_performance_summary():
            """Получить сводку производительности."""
            return await self._get_performance_summary()

        @self.app.get("/api/components")
        async def get_components():
            """Получить список компонентов."""
            return await self._get_components_list()

        @self.app.get("/api/components/{component_name}")
        async def get_component_details(component_name: str):
            """Получить детали компонента."""
            return await self._get_component_details(component_name)

        @self.app.get("/api/config/validation")
        async def get_config_validation():
            """Получить результаты валидации конфигурации."""
            return await self._get_config_validation()

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint для real-time обновлений."""
            await self._handle_websocket(websocket)

    def _get_dashboard_html(self) -> str:
        """Генерирует HTML для дашборда."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ATB Monitoring Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    text-align: center;
                }
                .metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 20px;
                }
                .metric-card {
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                .metric-title {
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 10px;
                    color: #333;
                }
                .metric-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #667eea;
                }
                .metric-unit {
                    font-size: 14px;
                    color: #666;
                }
                .status-indicator {
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    margin-right: 8px;
                }
                .status-online { background-color: #4caf50; }
                .status-warning { background-color: #ff9800; }
                .status-error { background-color: #f44336; }
                .status-offline { background-color: #9e9e9e; }
                .chart-container {
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }
                .alerts-container {
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                .alert-item {
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 5px;
                    border-left: 4px solid #f44336;
                }
                .alert-warning { border-left-color: #ff9800; }
                .alert-error { border-left-color: #f44336; }
                .alert-info { border-left-color: #2196f3; }
                .refresh-btn {
                    background: #667eea;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    margin: 10px 0;
                }
                .refresh-btn:hover {
                    background: #5a6fd8;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>ATB Monitoring Dashboard</h1>
                    <p>Real-time system monitoring and performance metrics</p>
                </div>
                
                <button class="refresh-btn" onclick="refreshData()">Refresh Data</button>
                
                <div class="metrics-grid" id="metricsGrid">
                    <!-- Metrics will be populated here -->
                </div>
                
                <div class="chart-container">
                    <h3>Performance Overview</h3>
                    <canvas id="performanceChart" width="400" height="200"></canvas>
                </div>
                
                <div class="alerts-container">
                    <h3>Active Alerts</h3>
                    <div id="alertsContainer">
                        <!-- Alerts will be populated here -->
                    </div>
                </div>
            </div>
            
            <script>
                let performanceChart;
                let ws;
                
                function connectWebSocket() {
                    ws = new WebSocket(`ws://${window.location.host}/ws`);
                    
                    ws.onopen = function() {
                        console.log('WebSocket connected');
                    };
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        if (data.type === 'metrics_update') {
                            updateMetrics(data.metrics);
                        } else if (data.type === 'alert_update') {
                            updateAlerts(data.alerts);
                        } else if (data.type === 'performance_update') {
                            updatePerformanceChart(data.performance);
                        }
                    };
                    
                    ws.onclose = function() {
                        console.log('WebSocket disconnected');
                        setTimeout(connectWebSocket, 5000);
                    };
                }
                
                function updateMetrics(metrics) {
                    const grid = document.getElementById('metricsGrid');
                    grid.innerHTML = '';
                    
                    metrics.forEach(metric => {
                        const card = document.createElement('div');
                        card.className = 'metric-card';
                        
                        const statusClass = `status-${metric.status}`;
                        card.innerHTML = `
                            <div class="metric-title">
                                <span class="status-indicator ${statusClass}"></span>
                                ${metric.name}
                            </div>
                            <div class="metric-value">${metric.value}</div>
                            <div class="metric-unit">${metric.unit}</div>
                            <div>Trend: ${metric.trend}</div>
                        `;
                        
                        grid.appendChild(card);
                    });
                }
                
                function updateAlerts(alerts) {
                    const container = document.getElementById('alertsContainer');
                    container.innerHTML = '';
                    
                    if (alerts.length === 0) {
                        container.innerHTML = '<p>No active alerts</p>';
                        return;
                    }
                    
                    alerts.forEach(alert => {
                        const alertDiv = document.createElement('div');
                        alertDiv.className = `alert-item alert-${alert.level}`;
                        alertDiv.innerHTML = `
                            <strong>${alert.level.toUpperCase()}</strong>: ${alert.message}<br>
                            <small>Component: ${alert.component} | Time: ${new Date(alert.timestamp).toLocaleString()}</small>
                        `;
                        container.appendChild(alertDiv);
                    });
                }
                
                function updatePerformanceChart(performance) {
                    if (!performanceChart) {
                        const ctx = document.getElementById('performanceChart').getContext('2d');
                        performanceChart = new Chart(ctx, {
                            type: 'line',
                            data: {
                                labels: performance.labels || [],
                                datasets: [{
                                    label: 'CPU Usage',
                                    data: performance.cpu || [],
                                    borderColor: '#667eea',
                                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                                    tension: 0.4
                                }, {
                                    label: 'Memory Usage',
                                    data: performance.memory || [],
                                    borderColor: '#764ba2',
                                    backgroundColor: 'rgba(118, 75, 162, 0.1)',
                                    tension: 0.4
                                }]
                            },
                            options: {
                                responsive: true,
                                scales: {
                                    y: {
                                        beginAtZero: true,
                                        max: 100
                                    }
                                }
                            }
                        });
                    } else {
                        performanceChart.data.labels = performance.labels || [];
                        performanceChart.data.datasets[0].data = performance.cpu || [];
                        performanceChart.data.datasets[1].data = performance.memory || [];
                        performanceChart.update();
                    }
                }
                
                async function refreshData() {
                    try {
                        // Fetch metrics
                        const metricsResponse = await fetch('/api/metrics');
                        const metrics = await metricsResponse.json();
                        updateMetrics(metrics);
                        
                        // Fetch alerts
                        const alertsResponse = await fetch('/api/alerts');
                        const alerts = await alertsResponse.json();
                        updateAlerts(alerts);
                        
                        // Fetch performance
                        const performanceResponse = await fetch('/api/performance');
                        const performance = await performanceResponse.json();
                        updatePerformanceChart(performance);
                    } catch (error) {
                        console.error('Error refreshing data:', error);
                    }
                }
                
                // Initialize
                connectWebSocket();
                refreshData();
                
                // Auto-refresh every 30 seconds
                setInterval(refreshData, 30000);
            </script>
        </body>
        </html>
        """

    async def _handle_websocket(self, websocket: WebSocket):
        """Обработка WebSocket соединения."""
        client_id = f"client_{len(self.websocket_manager.active_connections)}"
        await self.websocket_manager.connect(websocket, client_id)
        try:
            while True:
                # Ожидание сообщений от клиента
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Обработка подписок
                if message.get("type") == "subscribe":
                    subscriptions = message.get("subscriptions", [])
                    self.websocket_manager.connection_data[websocket]["subscriptions"] = set(subscriptions)
                    
                # Отправка подтверждения
                await self.websocket_manager.send_personal_message(
                    json.dumps({"type": "subscription_confirmed", "subscriptions": list(subscriptions)}),
                    websocket
                )
        except WebSocketDisconnect:
            self.websocket_manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self.websocket_manager.disconnect(websocket)

    async def _get_current_metrics(self) -> List[Dict[str, Any]]:
        """Получить текущие метрики."""
        performance_monitor = get_performance_monitor()
        if not performance_monitor:
            return []
        
        try:
            summary = performance_monitor.get_metrics_summary()
            metrics = []
            
            for metric_name, metric_data in summary.items():
                if isinstance(metric_data, dict):
                    metrics.append({
                        "name": metric_name,
                        "value": metric_data.get("latest", 0),
                        "unit": self._get_metric_unit(metric_name),
                        "trend": self._get_metric_trend(metric_data),
                        "status": self._get_metric_status(metric_data),
                        "timestamp": datetime.now().isoformat(),
                        "component": metric_data.get("component", "system"),
                        "description": self._get_metric_description(metric_name)
                    })
            
            return metrics
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return []

    def _get_metric_trend(self, metric_data: Dict[str, Any]) -> str:
        """Определить тренд метрики."""
        history = metric_data.get("history", [])
        if len(history) < 2:
            return "stable"
        
        recent = history[-1]
        previous = history[-2]
        
        if recent > previous * 1.05:
            return "up"
        elif recent < previous * 0.95:
            return "down"
        else:
            return "stable"

    def _get_metric_status(self, metric_data: Dict[str, Any]) -> str:
        """Определить статус метрики."""
        latest = metric_data.get("latest", 0)
        threshold = metric_data.get("threshold", 80)
        
        if latest >= threshold:
            return "critical"
        elif latest >= threshold * 0.8:
            return "warning"
        else:
            return "normal"

    def _get_metric_unit(self, metric_name: str) -> str:
        """Получить единицу измерения метрики."""
        units = {
            "cpu_usage": "%",
            "memory_usage": "%",
            "disk_usage": "%",
            "network_io": "MB/s",
            "response_time": "ms",
            "error_rate": "%",
            "throughput": "req/s"
        }
        return units.get(metric_name, "")

    def _get_metric_description(self, metric_name: str) -> str:
        """Получить описание метрики."""
        descriptions = {
            "cpu_usage": "CPU utilization percentage",
            "memory_usage": "Memory utilization percentage",
            "disk_usage": "Disk space utilization percentage",
            "network_io": "Network input/output rate",
            "response_time": "Average response time",
            "error_rate": "Error rate percentage",
            "throughput": "Requests per second"
        }
        return descriptions.get(metric_name, "")

    async def _get_metric_details(self, metric_name: str) -> Dict[str, Any]:
        """Получить детали метрики."""
        performance_monitor = get_performance_monitor()
        if not performance_monitor:
            raise HTTPException(status_code=404, detail="Performance monitor not available")
        
        try:
            summary = performance_monitor.get_metrics_summary()
            metric_data = summary.get(metric_name)
            
            if not metric_data:
                raise HTTPException(status_code=404, detail="Metric not found")
            
            history = metric_data.get("history", [])
            timestamps = [
                (datetime.now() - timedelta(minutes=i)).isoformat()
                for i in range(len(history), 0, -1)
            ]
            
            return {
                "name": metric_name,
                "current": metric_data.get("latest", 0),
                "history": {"values": history, "timestamps": timestamps},
                "statistics": {
                    "min": min(history) if history else 0,
                    "max": max(history) if history else 0,
                    "avg": sum(history) / len(history) if history else 0,
                    "count": len(history),
                },
                "unit": self._get_metric_unit(metric_name),
                "description": self._get_metric_description(metric_name),
            }
        except Exception as e:
            logger.error(f"Error getting metric details: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Получить активные алерты."""
        performance_monitor = get_performance_monitor()
        if not performance_monitor:
            return []
        
        try:
            alerts_summary = performance_monitor.get_alerts_summary(resolved=False)
            active_alerts = [
                alert for alert in performance_monitor.alerts if not alert.resolved
            ]
            return [
                {
                    "id": f"alert_{i}",
                    "level": alert.level.value,
                    "message": alert.message,
                    "component": alert.component,
                    "timestamp": alert.timestamp.isoformat(),
                    "resolved": alert.resolved,
                    "acknowledged": getattr(alert, "acknowledged", False),
                }
                for i, alert in enumerate(active_alerts)
            ]
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []

    async def _acknowledge_alert(self, alert_id: str) -> Dict[str, Any]:
        """Подтвердить алерт."""
        performance_monitor = get_performance_monitor()
        if not performance_monitor:
            raise HTTPException(
                status_code=404, detail="Performance monitor not available"
            )
        
        try:
            # Находим алерт по ID
            alert_index = int(alert_id.split("_")[1])
            active_alerts = [
                alert for alert in performance_monitor.alerts if not alert.resolved
            ]
            if alert_index >= len(active_alerts):
                raise HTTPException(status_code=404, detail="Alert not found")
            
            alert = active_alerts[alert_index]
            # Устанавливаем атрибут acknowledged, если он существует
            if hasattr(alert, 'acknowledged'):
                alert.acknowledged = True
            else:
                # Если атрибута нет, создаем его
                setattr(alert, 'acknowledged', True)
            
            return {"status": "acknowledged", "alert_id": alert_id}
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid alert ID format")
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def _get_performance_summary(self) -> Dict[str, Any]:
        """Получить сводку производительности."""
        performance_monitor = get_performance_monitor()
        if not performance_monitor:
            return {"error": "Performance monitor not available"}
        
        try:
            summary = performance_monitor.get_metrics_summary()
            # Подготавливаем данные для графика
            cpu_data = summary.get("cpu_usage", {})
            memory_data = summary.get("memory_usage", {})
            # Получаем историю за последние 10 точек
            cpu_history = cpu_data.get("history", [])[-10:] if "history" in cpu_data else []
            memory_history = (
                memory_data.get("history", [])[-10:] if "history" in memory_data else []
            )
            return {
                "labels": [f"T{i}" for i in range(len(cpu_history))],
                "cpu": cpu_history,
                "memory": memory_history,
                "summary": {
                    "cpu_avg": cpu_data.get("avg", 0),
                    "memory_avg": memory_data.get("avg", 0),
                    "cpu_current": cpu_data.get("latest", 0),
                    "memory_current": memory_data.get("latest", 0),
                },
            }
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": "Failed to get performance summary"}

    async def _get_components_list(self) -> List[Dict[str, Any]]:
        """Получить список компонентов."""
        performance_monitor = get_performance_monitor()
        if not performance_monitor:
            return []
        
        try:
            summary = performance_monitor.get_metrics_summary()
            components = set()
            for metric_data in summary.values():
                components.add(metric_data.get("component", "unknown"))
            return [
                {
                    "name": component,
                    "status": "online",  # Упрощённая логика
                    "metrics_count": len(
                        [m for m in summary.values() if m.get("component") == component]
                    ),
                }
                for component in components
            ]
        except Exception as e:
            logger.error(f"Error getting components list: {e}")
            return []

    async def _get_component_details(self, component_name: str) -> Dict[str, Any]:
        """Получить детали компонента."""
        performance_monitor = get_performance_monitor()
        if not performance_monitor:
            raise HTTPException(
                status_code=404, detail="Performance monitor not available"
            )
        
        try:
            summary = performance_monitor.get_metrics_summary()
            component_metrics = {
                name: data
                for name, data in summary.items()
                if data.get("component") == component_name
            }
            if not component_metrics:
                raise HTTPException(status_code=404, detail="Component not found")
            return {
                "name": component_name,
                "metrics": component_metrics,
                "status": "online",
                "last_update": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting component details: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def _get_config_validation(self) -> Dict[str, Any]:
        """Получить результаты валидации конфигурации."""
        config_validator = get_config_validator()
        if not config_validator:
            return {"error": "Config validator not available"}
        
        try:
            return {
                "issues": config_validator.get_issues_summary(),
                "has_critical_issues": config_validator.has_critical_issues(),
                "total_issues": len(config_validator.issues),
            }
        except Exception as e:
            logger.error(f"Error getting config validation: {e}")
            return {"error": "Failed to get config validation"}

    def _start_background_tasks(self):
        """Запуск фоновых задач."""

        async def background_updates():
            while True:
                try:
                    # Отправка обновлений через WebSocket
                    await self._send_websocket_updates()
                    await asyncio.sleep(30)  # Обновление каждые 30 секунд
                except Exception as e:
                    logger.error(f"Error in background updates: {e}")
                    await asyncio.sleep(60)  # Увеличенная задержка при ошибке

        # Запускаем асинхронную задачу
        asyncio.create_task(background_updates())

    async def _send_websocket_updates(self):
        """Отправка обновлений через WebSocket."""
        if not self.websocket_manager.active_connections:
            return
        
        try:
            # Обновление метрик
            metrics = await self._get_current_metrics()
            await self.websocket_manager.broadcast(
                json.dumps({"type": "metrics_update", "metrics": metrics})
            )
            # Обновление алертов
            alerts = await self._get_active_alerts()
            await self.websocket_manager.broadcast(
                json.dumps({"type": "alert_update", "alerts": alerts})
            )
            # Обновление производительности
            performance = await self._get_performance_summary()
            await self.websocket_manager.broadcast(
                json.dumps({"type": "performance_update", "performance": performance})
            )
        except Exception as e:
            logger.error(f"Error sending WebSocket updates: {e}")

    def start(self):
        """Запуск дашборда."""
        logger.info(f"Starting monitoring dashboard on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)

    def stop(self):
        """Остановка дашборда."""
        logger.info("Stopping monitoring dashboard")
        self.status = DashboardStatus.OFFLINE


# Глобальный экземпляр дашборда
dashboard: Optional[MonitoringDashboard] = None


def start_dashboard(host: str = "0.0.0.0", port: int = 8080):
    """
    Запуск дашборда мониторинга.
    Args:
        host: Хост для запуска
        port: Порт для запуска
    """
    global dashboard
    dashboard = MonitoringDashboard(host, port)
    dashboard.start()


def stop_dashboard():
    """Остановка дашборда мониторинга."""
    global dashboard
    if dashboard:
        dashboard.stop()


if __name__ == "__main__":
    start_dashboard()
