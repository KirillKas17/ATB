"""
Dashboard для системы ATB с интеграцией Entity Analytics.

Включает мониторинг RL-метрик, CI/CD статуса, истории откатов и управления улучшениями.
"""

import asyncio
from datetime import datetime, timedelta

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, callback_context, dcc, html
from loguru import logger

# Entity компоненты - заглушки
ENTITY_AVAILABLE = False

class ImprovementApplier:
    """Заглушка для отсутствующего класса ImprovementApplier"""
    def __init__(self):
        pass

class EntityAnalytics:
    """Заглушка для отсутствующего класса EntityAnalytics"""
    def __init__(self):
        pass

logger.info("Entity модули инициализированы с заглушками для демонстрации")


class EntityDashboard:
    """Dashboard для Entity Analytics системы."""

    def __init__(self):
        """Инициализация dashboard."""
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True,
        )

        self.entity_analytics = None
        self.improvement_applier = None

        # Инициализация Entity компонентов
        if ENTITY_AVAILABLE:
            self.entity_analytics = EntityAnalytics()
            self.improvement_applier = ImprovementApplier()

        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        """Настройка макета dashboard."""
        self.app.layout = dbc.Container(
            [
                # Заголовок
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H1(
                                    "Entity Analytics Dashboard",
                                    className="text-center mb-4",
                                ),
                                html.Hr(),
                            ]
                        )
                    ]
                ),
                # Навигация
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Nav(
                                    [
                                        dbc.NavItem(
                                            dbc.NavLink(
                                                "Обзор",
                                                href="#overview",
                                                id="nav-overview",
                                            )
                                        ),
                                        dbc.NavItem(
                                            dbc.NavLink(
                                                "RL Метрики",
                                                href="#rl-metrics",
                                                id="nav-rl-metrics",
                                            )
                                        ),
                                        dbc.NavItem(
                                            dbc.NavLink(
                                                "CI/CD Статус",
                                                href="#cicd-status",
                                                id="nav-cicd",
                                            )
                                        ),
                                        dbc.NavItem(
                                            dbc.NavLink(
                                                "История Откатов",
                                                href="#rollback-history",
                                                id="nav-rollback",
                                            )
                                        ),
                                        dbc.NavItem(
                                            dbc.NavLink(
                                                "Управление",
                                                href="#control",
                                                id="nav-control",
                                            )
                                        ),
                                    ],
                                    pills=True,
                                    className="mb-4",
                                )
                            ]
                        )
                    ]
                ),
                # Основной контент
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                # Обзор
                                html.Div(
                                    id="overview-content",
                                    children=[self.create_overview_section()],
                                ),
                                # RL Метрики
                                html.Div(
                                    id="rl-metrics-content",
                                    children=[self.create_rl_metrics_section()],
                                    style={"display": "none"},
                                ),
                                # CI/CD Статус
                                html.Div(
                                    id="cicd-content",
                                    children=[self.create_cicd_section()],
                                    style={"display": "none"},
                                ),
                                # История откатов
                                html.Div(
                                    id="rollback-content",
                                    children=[self.create_rollback_section()],
                                    style={"display": "none"},
                                ),
                                # Управление
                                html.Div(
                                    id="control-content",
                                    children=[self.create_control_section()],
                                    style={"display": "none"},
                                ),
                            ]
                        )
                    ]
                ),
                # Интервалы обновления
                dcc.Interval(
                    id="interval-component",
                    interval=30 * 1000,  # 30 секунд
                    n_intervals=0,
                ),
                # Store для данных
                dcc.Store(id="entity-data-store"),
            ],
            fluid=True,
        )

    def create_overview_section(self):
        """Создание секции обзора."""
        return dbc.Card(
            [
                dbc.CardHeader("Обзор системы Entity Analytics"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                # Статус системы
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        html.H4(
                                                            "Статус системы",
                                                            className="card-title",
                                                        ),
                                                        html.Div(id="system-status"),
                                                        dbc.Button(
                                                            "Запустить",
                                                            id="start-system",
                                                            color="success",
                                                            className="me-2",
                                                        ),
                                                        dbc.Button(
                                                            "Остановить",
                                                            id="stop-system",
                                                            color="danger",
                                                        ),
                                                    ]
                                                )
                                            ]
                                        )
                                    ],
                                    width=6,
                                ),
                                # Основные метрики
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        html.H4(
                                                            "Основные метрики",
                                                            className="card-title",
                                                        ),
                                                        html.Div(id="main-metrics"),
                                                    ]
                                                )
                                            ]
                                        )
                                    ],
                                    width=6,
                                ),
                            ]
                        ),
                        # Графики
                        dbc.Row(
                            [
                                dbc.Col(
                                    [dcc.Graph(id="improvements-timeline")], width=6
                                ),
                                dbc.Col([dcc.Graph(id="performance-chart")], width=6),
                            ],
                            className="mt-4",
                        ),
                    ]
                ),
            ]
        )

    def create_rl_metrics_section(self):
        """Создание секции RL метрик."""
        return dbc.Card(
            [
                dbc.CardHeader("Reinforcement Learning Метрики"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                # Общие метрики RL
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        html.H5(
                                                            "Эффективность RL-стратегий",
                                                            className="card-title",
                                                        ),
                                                        html.Div(
                                                            id="rl-overview-metrics"
                                                        ),
                                                    ]
                                                )
                                            ]
                                        )
                                    ],
                                    width=4,
                                ),
                                # Успешность стратегий
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        html.H5(
                                                            "Успешность стратегий",
                                                            className="card-title",
                                                        ),
                                                        dcc.Graph(
                                                            id="rl-success-chart"
                                                        ),
                                                    ]
                                                )
                                            ]
                                        )
                                    ],
                                    width=4,
                                ),
                                # Производительность
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        html.H5(
                                                            "Производительность",
                                                            className="card-title",
                                                        ),
                                                        dcc.Graph(
                                                            id="rl-performance-chart"
                                                        ),
                                                    ]
                                                )
                                            ]
                                        )
                                    ],
                                    width=4,
                                ),
                            ]
                        ),
                        # Детальная таблица стратегий
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H5("Жизненный цикл стратегий"),
                                        html.Div(id="rl-strategies-table"),
                                    ]
                                )
                            ],
                            className="mt-4",
                        ),
                        # График истории производительности
                        dbc.Row(
                            [dbc.Col([dcc.Graph(id="rl-performance-history")])],
                            className="mt-4",
                        ),
                    ]
                ),
            ]
        )

    def create_cicd_section(self):
        """Создание секции CI/CD статуса."""
        return dbc.Card(
            [
                dbc.CardHeader("CI/CD Статус"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                # Статус CI/CD
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        html.H5(
                                                            "Статус CI/CD",
                                                            className="card-title",
                                                        ),
                                                        html.Div(
                                                            id="cicd-status-overview"
                                                        ),
                                                        dbc.Button(
                                                            "Включить CI/CD",
                                                            id="enable-cicd",
                                                            color="success",
                                                            className="me-2",
                                                        ),
                                                        dbc.Button(
                                                            "Отключить CI/CD",
                                                            id="disable-cicd",
                                                            color="warning",
                                                        ),
                                                    ]
                                                )
                                            ]
                                        )
                                    ],
                                    width=6,
                                ),
                                # Статистика деплоев
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        html.H5(
                                                            "Статистика деплоев",
                                                            className="card-title",
                                                        ),
                                                        html.Div(id="deploy-stats"),
                                                    ]
                                                )
                                            ]
                                        )
                                    ],
                                    width=6,
                                ),
                            ]
                        ),
                        # Конфигурация CI/CD
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H5("Конфигурация CI/CD"),
                                        html.Div(id="cicd-config"),
                                    ]
                                )
                            ],
                            className="mt-4",
                        ),
                        # История деплоев
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H5("История деплоев"),
                                        html.Div(id="deploy-history"),
                                    ]
                                )
                            ],
                            className="mt-4",
                        ),
                    ]
                ),
            ]
        )

    def create_rollback_section(self):
        """Создание секции истории откатов."""
        return dbc.Card(
            [
                dbc.CardHeader("История откатов"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                # Статистика откатов
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        html.H5(
                                                            "Статистика откатов",
                                                            className="card-title",
                                                        ),
                                                        html.Div(id="rollback-stats"),
                                                    ]
                                                )
                                            ]
                                        )
                                    ],
                                    width=4,
                                ),
                                # Причины откатов
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        html.H5(
                                                            "Причины откатов",
                                                            className="card-title",
                                                        ),
                                                        dcc.Graph(
                                                            id="rollback-reasons-chart"
                                                        ),
                                                    ]
                                                )
                                            ]
                                        )
                                    ],
                                    width=4,
                                ),
                                # Временная диаграмма
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        html.H5(
                                                            "Временная диаграмма",
                                                            className="card-title",
                                                        ),
                                                        dcc.Graph(
                                                            id="rollback-timeline"
                                                        ),
                                                    ]
                                                )
                                            ]
                                        )
                                    ],
                                    width=4,
                                ),
                            ]
                        ),
                        # Детальная таблица откатов
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H5("Детальная история откатов"),
                                        html.Div(id="rollback-details-table"),
                                    ]
                                )
                            ],
                            className="mt-4",
                        ),
                    ]
                ),
            ]
        )

    def create_control_section(self):
        """Создание секции управления."""
        return dbc.Card(
            [
                dbc.CardHeader("Управление системой"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                # Принудительный анализ
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        html.H5(
                                                            "Принудительный анализ",
                                                            className="card-title",
                                                        ),
                                                        dbc.Button(
                                                            "Запустить анализ",
                                                            id="force-analysis",
                                                            color="primary",
                                                            className="mb-3",
                                                        ),
                                                        html.Div(id="analysis-status"),
                                                    ]
                                                )
                                            ]
                                        )
                                    ],
                                    width=6,
                                ),
                                # Управление улучшениями
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        html.H5(
                                                            "Управление улучшениями",
                                                            className="card-title",
                                                        ),
                                                        html.Div(
                                                            id="improvements-control"
                                                        ),
                                                    ]
                                                )
                                            ]
                                        )
                                    ],
                                    width=6,
                                ),
                            ]
                        ),
                        # Настройки системы
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H5("Настройки системы"),
                                        html.Div(id="system-settings"),
                                    ]
                                )
                            ],
                            className="mt-4",
                        ),
                    ]
                ),
            ]
        )

    def setup_callbacks(self):
        """Настройка callback функций."""

        # Навигация
        @self.app.callback(
            [
                Output(f"{section}-content", "style")
                for section in ["overview", "rl-metrics", "cicd", "rollback", "control"]
            ],
            [
                Input(f"nav-{section}", "n_clicks")
                for section in ["overview", "rl-metrics", "cicd", "rollback", "control"]
            ],
        )
        def update_navigation(*args):
            ctx = callback_context
            if not ctx.triggered:
                return [{"display": "block" if i == 0 else "none"} for i in range(5)]

            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            section_map = {
                "nav-overview": 0,
                "nav-rl-metrics": 1,
                "nav-cicd": 2,
                "nav-rollback": 3,
                "nav-control": 4,
            }

            active_section = section_map.get(button_id, 0)
            return [
                {"display": "block" if i == active_section else "none"}
                for i in range(5)
            ]

        # Обновление данных
        @self.app.callback(
            Output("entity-data-store", "data"),
            Input("interval-component", "n_intervals"),
        )
        def update_entity_data(n):
            if not ENTITY_AVAILABLE:
                return {}

            try:
                data = {
                    "system_status": (
                        self.entity_analytics.get_status()
                        if self.entity_analytics
                        else {}
                    ),
                    "rl_metrics": (
                        self.improvement_applier.get_rl_effectiveness_metrics()
                        if self.improvement_applier
                        else {}
                    ),
                    "cicd_status": (
                        self.improvement_applier.get_cicd_status()
                        if self.improvement_applier
                        else {}
                    ),
                    "rollback_history": (
                        self.improvement_applier.get_rollback_history()
                        if self.improvement_applier
                        else []
                    ),
                    "timestamp": datetime.now().isoformat(),
                }
                return data
            except Exception as e:
                logger.error(f"Ошибка обновления данных: {e}")
                return {}

        # Обновление статуса системы
        @self.app.callback(
            Output("system-status", "children"), Input("entity-data-store", "data")
        )
        def update_system_status(data):
            if not data:
                return html.P("Данные недоступны")

            status = data.get("system_status", {})
            is_running = status.get("is_running", False)

            return dbc.Alert(
                [
                    html.H6("Статус: " + ("Запущена" if is_running else "Остановлена")),
                    html.P(f"Цикл анализа: {status.get('analysis_cycle', 0)}"),
                    html.P(f"AI уверенность: {status.get('ai_confidence', 0):.2f}"),
                    html.P(f"ML точность: {status.get('ml_accuracy', 0):.2f}"),
                ],
                color="success" if is_running else "danger",
            )

        # Обновление RL метрик
        @self.app.callback(
            Output("rl-overview-metrics", "children"),
            Input("entity-data-store", "data"),
        )
        def update_rl_metrics(data):
            if not data:
                return html.P("Данные недоступны")

            rl_metrics = data.get("rl_metrics", {})

            return html.Div(
                [
                    html.P(f"Всего стратегий: {rl_metrics.get('total_strategies', 0)}"),
                    html.P(f"Успешных: {rl_metrics.get('successful_strategies', 0)}"),
                    html.P(f"Неудачных: {rl_metrics.get('failed_strategies', 0)}"),
                    html.P(
                        f"Средняя производительность: {rl_metrics.get('average_performance', 0):.2f}"
                    ),
                ]
            )

        # График успешности RL стратегий
        @self.app.callback(
            Output("rl-success-chart", "figure"), Input("entity-data-store", "data")
        )
        def update_rl_success_chart(data):
            if not data:
                return go.Figure()

            rl_metrics = data.get("rl_metrics", {})
            successful = rl_metrics.get("successful_strategies", 0)
            failed = rl_metrics.get("failed_strategies", 0)

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=["Успешные", "Неудачные"],
                        values=[successful, failed],
                        hole=0.3,
                    )
                ]
            )
            fig.update_layout(title="Распределение успешности стратегий")
            return fig

        # График производительности RL
        @self.app.callback(
            Output("rl-performance-chart", "figure"), Input("entity-data-store", "data")
        )
        def update_rl_performance_chart(data):
            if not data:
                return go.Figure()

            rl_metrics = data.get("rl_metrics", {})
            performance_history = rl_metrics.get("performance_history", [])

            if not performance_history:
                return go.Figure()

            df = pd.DataFrame(performance_history)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["performance"],
                    mode="lines+markers",
                    name="Производительность",
                )
            )
            fig.update_layout(
                title="История производительности RL-стратегий",
                xaxis_title="Время",
                yaxis_title="Производительность",
            )
            return fig

        # Обновление CI/CD статуса
        @self.app.callback(
            Output("cicd-status-overview", "children"),
            Input("entity-data-store", "data"),
        )
        def update_cicd_status(data):
            if not data:
                return html.P("Данные недоступны")

            cicd_status = data.get("cicd_status", {})
            enabled = cicd_status.get("enabled", False)

            return dbc.Alert(
                [
                    html.H6("CI/CD: " + ("Включен" if enabled else "Отключен")),
                    html.P(f"Всего деплоев: {cicd_status.get('total_deployments', 0)}"),
                    html.P(
                        f"Успешных деплоев: {cicd_status.get('successful_deployments', 0)}"
                    ),
                ],
                color="success" if enabled else "warning",
            )

        # Обновление статистики откатов
        @self.app.callback(
            Output("rollback-stats", "children"), Input("entity-data-store", "data")
        )
        def update_rollback_stats(data):
            if not data:
                return html.P("Данные недоступны")

            rollback_history = data.get("rollback_history", [])

            return html.Div(
                [
                    html.P(f"Всего откатов: {len(rollback_history)}"),
                    html.P(
                        f"За последние 24 часа: {len([r for r in rollback_history if datetime.fromisoformat(r.get('rollback_time', '')).replace(tzinfo=None) > datetime.now() - timedelta(days=1)])}"
                    ),
                ]
            )

        # График причин откатов
        @self.app.callback(
            Output("rollback-reasons-chart", "figure"),
            Input("entity-data-store", "data"),
        )
        def update_rollback_reasons_chart(data):
            if not data:
                return go.Figure()

            rollback_history = data.get("rollback_history", [])

            if not rollback_history:
                return go.Figure()

            reasons = {}
            for rollback in rollback_history:
                reason = rollback.get("reason", "unknown")
                reasons[reason] = reasons.get(reason, 0) + 1

            fig = go.Figure(
                data=[go.Bar(x=list(reasons.keys()), y=list(reasons.values()))]
            )
            fig.update_layout(
                title="Причины откатов", xaxis_title="Причина", yaxis_title="Количество"
            )
            return fig

        # Управление системой
        @self.app.callback(
            Output("analysis-status", "children"),
            Input("force-analysis", "n_clicks"),
            prevent_initial_call=True,
        )
        def force_analysis(n_clicks):
            if not ENTITY_AVAILABLE or not self.entity_analytics:
                return dbc.Alert("Entity Analytics недоступен", color="danger")

            try:
                asyncio.create_task(self.entity_analytics.force_analysis())
                return dbc.Alert("Принудительный анализ запущен", color="success")
            except Exception as e:
                return dbc.Alert(f"Ошибка запуска анализа: {e}", color="danger")

    def run(self, debug=True, port=8050):
        """Запуск dashboard."""
        logger.info(f"Запуск Entity Dashboard на порту {port}")
        self.app.run_server(debug=debug, port=port)


def main():
    """Основная функция запуска dashboard."""
    dashboard = EntityDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
