// Конфигурация
const CONFIG = {
    API_URL: 'http://localhost:8000',
    WS_URL: 'ws://localhost:8000/ws',
    POLLING_INTERVAL: 5000,
    MAX_RECONNECT_ATTEMPTS: 5,
    RECONNECT_DELAY: 5000,
    MAX_LOGS: 1000,
    CHART_UPDATE_INTERVAL: 1000,
    ERROR_NOTIFICATION_DURATION: 5000
};

// Состояние приложения
const state = {
    ws: null,
    symbols: [],
    timeframes: ['1m', '5m', '15m', '1h', '4h', '1d'],
    currentSymbol: null,
    currentTimeframe: null,
    isPolling: false,
    reconnectAttempts: 0,
    charts: {},
    logs: [],
    lastUpdate: null,
    systemStatus: {
        mode: 'stopped',
        errors: [],
        warnings: []
    }
};

// Кэш для оптимизации
const cache = {
    status: null,
    lastUpdate: null,
    pairData: new Map(),
    correlations: new Map()
};

// DOM элементы
const elements = {
    symbolSelect: document.getElementById('symbol'),
    timeframeSelect: document.getElementById('timeframe'),
    resultsTable: document.getElementById('resultsTable').getElementsByTagName('tbody')[0],
    logsContainer: document.getElementById('logs'),
    startBotBtn: document.getElementById('startBot'),
    stopBotBtn: document.getElementById('stopBot'),
    startTrainingBtn: document.getElementById('startTraining'),
    exportLogsBtn: document.getElementById('exportLogs'),
    viewBacktestBtn: document.getElementById('viewBacktest'),
    runBacktestBtn: document.getElementById('runBacktest'),
    backtestStrategySelect: document.getElementById('backtestStrategy'),
    backtestPeriodSelect: document.getElementById('backtestPeriod'),
    backtestResults: document.getElementById('backtestResults'),
    systemStatus: document.getElementById('systemStatus'),
    errorContainer: document.getElementById('errorContainer')
};

// Современный дашборд торгового бота
class ATBDashboard {
    constructor() {
        this.charts = {};
        this.isRunning = false;
        this.startTime = null;
        this.uptimeInterval = null;
        this.dataUpdateInterval = null;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeCharts();
        this.startDataUpdates();
        this.updateSystemStatus();
        this.loadPositions();
    }

    setupEventListeners() {
        // Навигация
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const sectionName = e.currentTarget.dataset.section;
                this.switchSection(sectionName);
            });
        });

        // Кнопки управления
        document.getElementById('startBtn').addEventListener('click', () => this.startBot());
        document.getElementById('stopBtn').addEventListener('click', () => this.stopBot());
        document.getElementById('refreshPositions').addEventListener('click', () => this.loadPositions());

        // Переключатели времени для графиков
        document.querySelectorAll('.time-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.updateChartData(e.target.dataset.period);
            });
        });
    }

    switchSection(sectionName) {
        console.log('Переключение на секцию:', sectionName);
        
        // Скрыть все секции
        document.querySelectorAll('.content-section').forEach(section => {
            section.classList.remove('active');
        });

        // Показать выбранную секцию
        const targetSection = document.getElementById(sectionName + 'Section');
        if (targetSection) {
            targetSection.classList.add('active');
            console.log('Секция активирована:', targetSection.id);
        } else {
            console.error('Секция не найдена:', sectionName + 'Section');
        }

        // Обновить активную навигацию
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        
        const activeNavItem = document.querySelector(`[data-section="${sectionName}"]`);
        if (activeNavItem) {
            activeNavItem.classList.add('active');
            console.log('Навигация обновлена:', activeNavItem.textContent);
        } else {
            console.error('Элемент навигации не найден:', sectionName);
        }
    }

    initializeCharts() {
        // P&L график
        const pnlCtx = document.getElementById('pnlChart').getContext('2d');
        this.charts.pnl = new Chart(pnlCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'P&L',
                    data: [],
                    borderColor: '#34C759',
                    backgroundColor: 'rgba(52, 199, 89, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'hour'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.6)'
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.6)',
                            callback: function(value) {
                                return '$' + value.toFixed(2);
                            }
                        }
                    }
                }
            }
        });

        // График распределения
        const distCtx = document.getElementById('distributionChart').getContext('2d');
        this.charts.distribution = new Chart(distCtx, {
            type: 'doughnut',
            data: {
                labels: ['Прибыль', 'Убыток'],
                datasets: [{
                    data: [78, 22],
                    backgroundColor: ['#34C759', '#FF3B30'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: 'rgba(255, 255, 255, 0.8)',
                            padding: 20
                        }
                    }
                }
            }
        });

        // График активности торгов
        const tradesCtx = document.getElementById('tradesChart').getContext('2d');
        this.charts.trades = new Chart(tradesCtx, {
            type: 'bar',
            data: {
                labels: ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'],
                datasets: [{
                    label: 'Сделки',
                    data: [12, 19, 15, 25, 22, 18, 14],
                    backgroundColor: 'rgba(0, 122, 255, 0.8)',
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.6)'
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.6)'
                        }
                    }
                }
            }
        });

        // Заполнить начальными данными
        this.generateSampleData();
    }

    generateSampleData() {
        // Генерируем данные за последние 24 часа
        const now = new Date();
        const labels = [];
        const data = [];
        let currentPnl = 0;

        for (let i = 23; i >= 0; i--) {
            const time = new Date(now.getTime() - i * 60 * 60 * 1000);
            labels.push(time);
            
            // Симулируем изменение P&L
            const change = (Math.random() - 0.5) * 100;
            currentPnl += change;
            data.push(currentPnl);
        }

        this.charts.pnl.data.labels = labels;
        this.charts.pnl.data.datasets[0].data = data;
        this.charts.pnl.update();
    }

    startBot() {
        if (this.isRunning) {
            this.showNotification('Бот уже запущен', 'warning');
            return;
        }

        this.isRunning = true;
        this.startTime = new Date();
        this.startUptimeCounter();
        
        document.getElementById('systemState').textContent = 'Активен';
        document.getElementById('systemStatus').className = 'status-indicator online';
        
        this.showNotification('Торговый бот запущен', 'success');
        
        // Обновляем кнопки
        document.getElementById('startBtn').style.opacity = '0.5';
        document.getElementById('stopBtn').style.opacity = '1';
    }

    stopBot() {
        if (!this.isRunning) {
            this.showNotification('Бот уже остановлен', 'warning');
            return;
        }

        this.isRunning = false;
        this.stopUptimeCounter();
        
        document.getElementById('systemState').textContent = 'Остановлен';
        document.getElementById('systemStatus').className = 'status-indicator offline';
        
        this.showNotification('Торговый бот остановлен', 'info');
        
        // Обновляем кнопки
        document.getElementById('startBtn').style.opacity = '1';
        document.getElementById('stopBtn').style.opacity = '0.5';
    }

    startUptimeCounter() {
        this.uptimeInterval = setInterval(() => {
            if (this.startTime) {
                const uptime = new Date() - this.startTime;
                const hours = Math.floor(uptime / (1000 * 60 * 60));
                const minutes = Math.floor((uptime % (1000 * 60 * 60)) / (1000 * 60));
                const seconds = Math.floor((uptime % (1000 * 60)) / 1000);
                
                document.getElementById('uptime').textContent = 
                    `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            }
        }, 1000);
    }

    stopUptimeCounter() {
        if (this.uptimeInterval) {
            clearInterval(this.uptimeInterval);
            this.uptimeInterval = null;
        }
    }

    updateSystemStatus() {
        // Симулируем обновление системных метрик
        setInterval(() => {
            if (this.isRunning) {
                // CPU
                const cpu = Math.random() * 50 + 10;
                document.getElementById('cpuUsage').textContent = `${cpu.toFixed(1)}%`;
                
                // Память
                const memory = Math.random() * 2 + 0.5;
                document.getElementById('memoryUsage').textContent = `${memory.toFixed(1)}GB`;
                
                // P&L
                const pnlChange = (Math.random() - 0.5) * 50;
                const currentPnl = parseFloat(document.getElementById('totalPnl').textContent.replace(/[^0-9.-]/g, '')) + pnlChange;
                document.getElementById('totalPnl').textContent = `$${currentPnl.toFixed(2)}`;
                document.getElementById('totalPnl').className = currentPnl >= 0 ? 'metric-value positive' : 'metric-value negative';
                
                // 24ч P&L
                const dailyChange = (Math.random() - 0.5) * 20;
                const currentDaily = parseFloat(document.getElementById('dailyPnl').textContent.replace(/[^0-9.-]/g, '')) + dailyChange;
                document.getElementById('dailyPnl').textContent = `$${currentDaily.toFixed(2)}`;
                document.getElementById('dailyPnl').className = currentDaily >= 0 ? 'metric-value positive' : 'metric-value negative';
            }
        }, 5000);
    }

    loadPositions() {
        // Симулируем загрузку позиций
        const positions = [
            {
                symbol: 'BTCUSDT',
                side: 'LONG',
                size: '0.5',
                entryPrice: '43250.50',
                currentPrice: '43520.30',
                pnl: '+$134.90',
                pnlPercent: '+2.1%'
            },
            {
                symbol: 'ETHUSDT',
                side: 'SHORT',
                size: '2.0',
                entryPrice: '2650.20',
                currentPrice: '2620.80',
                pnl: '+$58.80',
                pnlPercent: '+1.1%'
            },
            {
                symbol: 'ADAUSDT',
                side: 'LONG',
                size: '1000',
                entryPrice: '0.4850',
                currentPrice: '0.4720',
                pnl: '-$130.00',
                pnlPercent: '-2.7%'
            }
        ];

        const container = document.getElementById('positionsContainer');
        container.innerHTML = '';

        positions.forEach(pos => {
            const card = document.createElement('div');
            card.className = 'glass-card';
            card.innerHTML = `
                <div class="position-header">
                    <h4>${pos.symbol}</h4>
                    <span class="position-side ${pos.side.toLowerCase()}">${pos.side}</span>
                </div>
                <div class="position-details">
                    <div class="detail">
                        <span class="label">Размер:</span>
                        <span class="value">${pos.size}</span>
                    </div>
                    <div class="detail">
                        <span class="label">Цена входа:</span>
                        <span class="value">$${pos.entryPrice}</span>
                    </div>
                    <div class="detail">
                        <span class="label">Текущая цена:</span>
                        <span class="value">$${pos.currentPrice}</span>
                    </div>
                    <div class="detail">
                        <span class="label">P&L:</span>
                        <span class="value ${pos.pnl.startsWith('+') ? 'positive' : 'negative'}">${pos.pnl} (${pos.pnlPercent})</span>
                    </div>
                </div>
            `;
            container.appendChild(card);
        });

        document.getElementById('activePositions').textContent = positions.length;
    }

    startDataUpdates() {
        this.dataUpdateInterval = setInterval(() => {
            if (this.isRunning) {
                this.updateChartData();
            }
        }, 10000);
    }

    updateChartData(period = '1h') {
        // Обновляем данные графиков
        const now = new Date();
        const newData = (Math.random() - 0.5) * 50;
        
        // Добавляем новую точку на P&L график
        this.charts.pnl.data.labels.push(now);
        this.charts.pnl.data.datasets[0].data.push(newData);
        
        // Удаляем старые точки (оставляем последние 24)
        if (this.charts.pnl.data.labels.length > 24) {
            this.charts.pnl.data.labels.shift();
            this.charts.pnl.data.datasets[0].data.shift();
        }
        
        this.charts.pnl.update('none');
    }

    showNotification(message, type = 'info') {
        const container = document.getElementById('notificationsContainer');
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        container.appendChild(notification);
        
        // Удаляем уведомление через 5 секунд
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 5000);
    }
}

// Добавляем CSS анимацию для slideOut
const style = document.createElement('style');
style.textContent = `
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    .position-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--spacing-md);
    }
    
    .position-header h4 {
        font-size: 18px;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .position-side {
        padding: var(--spacing-xs) var(--spacing-sm);
        border-radius: var(--border-radius-small);
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .position-side.long {
        background: rgba(52, 199, 89, 0.2);
        color: var(--success-color);
    }
    
    .position-side.short {
        background: rgba(255, 59, 48, 0.2);
        color: var(--danger-color);
    }
    
    .position-details {
        display: flex;
        flex-direction: column;
        gap: var(--spacing-sm);
    }
    
    .detail {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .detail .label {
        color: var(--text-secondary);
        font-size: 14px;
    }
    
    .detail .value {
        font-weight: 600;
        font-size: 14px;
    }
    
    .notification.success {
        border-left: 4px solid var(--success-color);
    }
    
    .notification.warning {
        border-left: 4px solid var(--warning-color);
    }
    
    .notification.error {
        border-left: 4px solid var(--danger-color);
    }
    
    .notification.info {
        border-left: 4px solid var(--primary-color);
    }
`;
document.head.appendChild(style);

// Инициализация дашборда
document.addEventListener('DOMContentLoaded', () => {
    new ATBDashboard();
});
