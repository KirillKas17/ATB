"""
Дашборд мониторинга для Syntra.

Особенности:
- Визуализация метрик производительности в реальном времени
- Отображение алертов и их статуса
- Графики CPU, памяти, сети
- Статистика EventBus и кэша
- Управление алертами
"""

import logging
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import dash
import plotly.express as px
import plotly.graph_objs as go
from dash import Input, Output, callback_context, dcc, html
from plotly.subplots import make_subplots

from domain.services.technical_analysis import TechnicalAnalysisService
from infrastructure.core.data_pipeline import DataPipeline
from infrastructure.core.fibonacci_tools import FibonacciLevels
from infrastructure.core.health_checker import HealthChecker
from infrastructure.core.system_monitor import AlertLevel, SystemMonitor
from infrastructure.messaging.optimized_event_bus import EventBus
from shared.unified_cache import get_cache_manager

logger = logging.getLogger(__name__)


class MonitoringDashboard:
    """Дашборд для мониторинга системы."""
    
    def __init__(self, system_monitor: SystemMonitor, event_bus: EventBus, cache_manager=None):
        self.system_monitor = system_monitor
        self.event_bus = event_bus
        self.cache_manager = cache_manager or get_cache_manager()
        
        # Интервал обновления
        self.update_interval = 5000  # 5 секунд
        
        # Создание Dash приложения
        self.app = dash.Dash(__name__, title="Syntra Monitoring Dashboard")
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self) -> None:
        """Настройка макета дашборда."""
        self.app.layout = html.Div([
            # Заголовок
            html.H1("Syntra - Monitoring Dashboard", 
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
            
            # Статус системы
            html.Div([
                html.H3("System Status", style={'color': '#34495e'}),
                html.Div(id='system-status', style={'marginBottom': 20}),
            ]),
            
            # Алерты
            html.Div([
                html.H3("Alerts", style={'color': '#34495e'}),
                html.Div(id='alerts-container', style={'marginBottom': 20}),
            ]),
            
            # Метрики производительности
            html.Div([
                html.H3("Performance Metrics", style={'color': '#34495e'}),
                
                # Графики системы
                dcc.Graph(id='system-metrics-graph', style={'height': 400}),
                
                # Метрики EventBus
                html.Div([
                    html.H4("EventBus Metrics", style={'color': '#7f8c8d'}),
                    html.Div(id='eventbus-metrics', style={'marginBottom': 20}),
                ]),
                
                # Метрики кэша
                html.Div([
                    html.H4("Cache Metrics", style={'color': '#7f8c8d'}),
                    html.Div(id='cache-metrics', style={'marginBottom': 20}),
                ]),
            ]),
            
            # Интервал обновления
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0
            ),
            
            # Скрытые div для хранения данных
            html.Div(id='metrics-data', style={'display': 'none'}),
        ])
    
    def setup_callbacks(self) -> None:
        """Настройка обратных вызовов."""
        
        @self.app.callback(
            Output('system-status', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_system_status(n):
            """Обновление статуса системы."""
            try:
                status = self.system_monitor.get_status()
                
                # Определение цвета статуса
                status_color = '#27ae60' if status['is_running'] else '#e74c3c'
                status_text = "Running" if status['is_running'] else "Stopped"
                
                return html.Div([
                    html.Div([
                        html.Span("Status: ", style={'fontWeight': 'bold'}),
                        html.Span(status_text, style={'color': status_color}),
                    ], style={'marginBottom': 10}),
                    
                    html.Div([
                        html.Span("Alerts: ", style={'fontWeight': 'bold'}),
                        html.Span(f"Total: {status['alerts']['total']}, "
                                f"Unacknowledged: {status['alerts']['unacknowledged']}, "
                                f"Critical: {status['alerts']['critical']}"),
                    ], style={'marginBottom': 10}),
                    
                    html.Div([
                        html.Span("Integrations: ", style={'fontWeight': 'bold'}),
                        html.Span(", ".join(status['integrations']) if status['integrations'] else "None"),
                    ]),
                ])
            except Exception as e:
                return html.Div(f"Error loading status: {str(e)}", style={'color': '#e74c3c'})
        
        @self.app.callback(
            Output('alerts-container', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_alerts(n):
            """Обновление алертов."""
            try:
                alert_manager = self.system_monitor.alert_manager
                alerts = alert_manager.get_alerts(limit=10)
                
                if not alerts:
                    return html.Div("No alerts", style={'color': '#7f8c8d', 'fontStyle': 'italic'})
                
                alert_cards = []
                for alert in alerts:
                    # Определение цвета алерта
                    color_map = {
                        AlertLevel.INFO: '#3498db',
                        AlertLevel.WARNING: '#f39c12',
                        AlertLevel.ERROR: '#e67e22',
                        AlertLevel.CRITICAL: '#e74c3c'
                    }
                    
                    alert_color = color_map.get(alert.level, '#7f8c8d')
                    
                    card = html.Div([
                        html.Div([
                            html.Span(alert.level.value.upper(), 
                                    style={'color': alert_color, 'fontWeight': 'bold'}),
                            html.Span(f" - {alert.source}", style={'color': '#7f8c8d'}),
                        ], style={'marginBottom': 5}),
                        
                        html.Div(alert.message, style={'marginBottom': 5}),
                        
                        html.Div([
                            html.Small(f"Time: {alert.timestamp.strftime('%H:%M:%S')}"),
                            html.Br(),
                            html.Small(f"Status: {'Acknowledged' if alert.acknowledged else 'New'}"),
                        ], style={'color': '#7f8c8d', 'fontSize': '12px'}),
                    ], style={
                        'border': f'1px solid {alert_color}',
                        'borderRadius': '5px',
                        'padding': '10px',
                        'marginBottom': '10px',
                        'backgroundColor': '#f8f9fa'
                    })
                    
                    alert_cards.append(card)
                
                return html.Div(alert_cards)
                
            except Exception as e:
                return html.Div(f"Error loading alerts: {str(e)}", style={'color': '#e74c3c'})
        
        @self.app.callback(
            Output('system-metrics-graph', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_system_metrics(n):
            """Обновление графиков системных метрик."""
            try:
                performance_monitor = self.system_monitor.performance_monitor
                metrics = performance_monitor.get_metrics(limit=100)
                
                # Создание подграфиков
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('CPU Usage', 'Memory Usage', 'Disk Usage', 'Network I/O'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # CPU Usage
                if 'system.cpu_percent' in metrics:
                    cpu_data = metrics['system.cpu_percent']
                    if cpu_data:
                        timestamps = [m['timestamp'] for m in cpu_data]
                        values = [m['value'] for m in cpu_data]
                        
                        fig.add_trace(
                            go.Scatter(x=timestamps, y=values, name='CPU %', 
                                      line=dict(color='#e74c3c')),
                            row=1, col=1
                        )
                
                # Memory Usage
                if 'system.memory_percent' in metrics:
                    mem_data = metrics['system.memory_percent']
                    if mem_data:
                        timestamps = [m['timestamp'] for m in mem_data]
                        values = [m['value'] for m in mem_data]
                        
                        fig.add_trace(
                            go.Scatter(x=timestamps, y=values, name='Memory %', 
                                      line=dict(color='#3498db')),
                            row=1, col=2
                        )
                
                # Disk Usage
                if 'system.disk_usage' in metrics:
                    disk_data = metrics['system.disk_usage']
                    if disk_data:
                        timestamps = [m['timestamp'] for m in disk_data]
                        values = [m['value'] for m in disk_data]
                        
                        fig.add_trace(
                            go.Scatter(x=timestamps, y=values, name='Disk %', 
                                      line=dict(color='#f39c12')),
                            row=2, col=1
                        )
                
                # Network I/O
                if 'system.network_bytes_sent' in metrics and 'system.network_bytes_recv' in metrics:
                    sent_data = metrics['system.network_bytes_sent']
                    recv_data = metrics['system.network_bytes_recv']
                    
                    if sent_data and recv_data:
                        timestamps = [m['timestamp'] for m in sent_data]
                        sent_values = [m['value'] / (1024*1024) for m in sent_data]  # MB
                        recv_values = [m['value'] / (1024*1024) for m in recv_data]  # MB
                        
                        fig.add_trace(
                            go.Scatter(x=timestamps, y=sent_values, name='Sent (MB)', 
                                      line=dict(color='#27ae60')),
                            row=2, col=2
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=timestamps, y=recv_values, name='Received (MB)', 
                                      line=dict(color='#9b59b6')),
                            row=2, col=2
                        )
                
                # Обновление макета
                fig.update_layout(
                    height=400,
                    showlegend=True,
                    title_text="System Performance Metrics",
                    title_x=0.5
                )
                
                return fig
                
            except Exception as e:
                # Возврат пустого графика при ошибке
                return go.Figure().add_annotation(
                    text=f"Error loading metrics: {str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
        
        @self.app.callback(
            Output('eventbus-metrics', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_eventbus_metrics(n):
            """Обновление метрик EventBus."""
            try:
                metrics = self.event_bus.get_metrics()
                
                hit_rate = metrics.get('handler_cache_hits', 0) / max(metrics.get('handler_cache_hits', 0) + metrics.get('handler_cache_misses', 0), 1) * 100
                return html.Div([
                    html.Div([
                        html.Span("Events Processed: ", style={'fontWeight': 'bold'}),
                        html.Span(f"{metrics.get('events_processed', 0):,}"),
                    ], style={'marginBottom': 5}),
                    
                    html.Div([
                        html.Span("Events Dropped: ", style={'fontWeight': 'bold'}),
                        html.Span(f"{metrics.get('events_dropped', 0):,}"),
                    ], style={'marginBottom': 5}),
                    
                    html.Div([
                        html.Span("Avg Processing Time: ", style={'fontWeight': 'bold'}),
                        html.Span(f"{metrics.get('avg_processing_time', 0):.4f}s"),
                    ], style={'marginBottom': 5}),
                    
                    html.Div([
                        html.Span("Cache Hit Rate: ", style={'fontWeight': 'bold'}),
                        html.Span(f"{hit_rate:.1f}%"),
                    ]),
                ])
                
            except Exception as e:
                return html.Div(f"Error loading EventBus metrics: {str(e)}", style={'color': '#e74c3c'})
        
        @self.app.callback(
            Output('cache-metrics', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_cache_metrics(n):
            """Обновление метрик кэша."""
            try:
                all_metrics = self.cache_manager.get_all_metrics()
                
                if not all_metrics:
                    return html.Div("No cache metrics available", style={'color': '#7f8c8d'})
                
                cache_cards = []
                for cache_name, metrics in all_metrics.items():
                    hit_rate = 0
                    if metrics.get('hits', 0) + metrics.get('misses', 0) > 0:
                        hit_rate = metrics.get('hits', 0) / (metrics.get('hits', 0) + metrics.get('misses', 0)) * 100
                    
                    card = html.Div([
                        html.H5(cache_name, style={'color': '#2c3e50', 'marginBottom': 10}),
                        
                        html.Div([
                            html.Span("Hit Rate: ", style={'fontWeight': 'bold'}),
                            html.Span(f"{hit_rate:.1f}%"),
                        ], style={'marginBottom': 5}),
                        
                        html.Div([
                            html.Span("Size: ", style={'fontWeight': 'bold'}),
                            html.Span(f"{metrics.get('size', 0)}/{metrics.get('max_size', 0)}"),
                        ], style={'marginBottom': 5}),
                        
                        html.Div([
                            html.Span("Memory: ", style={'fontWeight': 'bold'}),
                            html.Span(f"{metrics.get('memory_usage_mb', 0):.2f} MB"),
                        ], style={'marginBottom': 5}),
                        
                        html.Div([
                            html.Span("Compressions: ", style={'fontWeight': 'bold'}),
                            html.Span(f"{metrics.get('compressions', 0)}"),
                        ]),
                    ], style={
                        'border': '1px solid #bdc3c7',
                        'borderRadius': '5px',
                        'padding': '15px',
                        'marginBottom': '15px',
                        'backgroundColor': '#ecf0f1'
                    })
                    
                    cache_cards.append(card)
                
                return html.Div(cache_cards)
                
            except Exception as e:
                return html.Div(f"Error loading cache metrics: {str(e)}", style={'color': '#e74c3c'})
    
    def run(self, host: str = '0.0.0.0', port: int = 8050, debug: bool = False) -> None:
        """Запуск дашборда."""
        self.app.run_server(host=host, port=port, debug=debug)


def create_monitoring_dashboard(system_monitor: SystemMonitor, event_bus: EventBus, 
                               cache_manager: Any) -> MonitoringDashboard:
    """Создание дашборда мониторинга."""
    return MonitoringDashboard(system_monitor, event_bus, cache_manager)


if __name__ == "__main__":
    # Пример использования
    import asyncio
    
    async def main():
        # Создание компонентов (в реальном приложении они должны быть уже инициализированы)
        from infrastructure.core.system_monitor import SystemMonitor
        from infrastructure.messaging.optimized_event_bus import EventBus
        from shared.unified_cache import get_cache_manager
        
        system_monitor = SystemMonitor()
        event_bus = EventBus()
        cache_manager = get_cache_manager()
        
        # Создание дашборда
        dashboard = create_monitoring_dashboard(system_monitor, event_bus, cache_manager)
        
        # Запуск дашборда
        dashboard.run(debug=True)
    
    asyncio.run(main()) 