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

// Инициализация
async function init() {
    try {
        await loadSymbols();
        setupWebSocket();
        setupEventListeners();
        setupCharts();
        startPolling();
        setupErrorHandling();
    } catch (error) {
        handleError('Initialization error', error);
    }
}

// Загрузка символов
async function loadSymbols() {
    try {
        const response = await fetch(`${CONFIG.API_URL}/symbols`);
        if (!response.ok) throw new Error('Failed to load symbols');
        
        state.symbols = await response.json();
        
        elements.symbolSelect.innerHTML = state.symbols.map(symbol => 
            `<option value="${symbol}">${symbol}</option>`
        ).join('');
        
        state.currentSymbol = state.symbols[0];
        state.currentTimeframe = state.timeframes[0];
        
        await updateDashboard();
    } catch (error) {
        handleError('Error loading symbols', error);
    }
}

// Настройка WebSocket
function setupWebSocket() {
    if (state.ws) {
        state.ws.close();
    }

    state.ws = new WebSocket(CONFIG.WS_URL);
    
    state.ws.onopen = () => {
        state.reconnectAttempts = 0;
        addLog('WebSocket connected', 'success');
        subscribeToUpdates();
    };
    
    state.ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        } catch (error) {
            handleError('WebSocket message parsing error', error);
        }
    };
    
    state.ws.onclose = () => {
        addLog('WebSocket disconnected', 'warning');
        handleReconnect();
    };
    
    state.ws.onerror = (error) => {
        handleError('WebSocket error', error);
    };
}

// Обработка переподключения
function handleReconnect() {
    if (state.reconnectAttempts < CONFIG.MAX_RECONNECT_ATTEMPTS) {
        state.reconnectAttempts++;
        setTimeout(setupWebSocket, CONFIG.RECONNECT_DELAY);
    } else {
        handleError('Maximum reconnection attempts reached', new Error('WebSocket connection failed'));
    }
}

// Подписка на обновления
function subscribeToUpdates() {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify({
            type: 'subscribe',
            data: {
                symbol: state.currentSymbol,
                timeframe: state.currentTimeframe
            }
        }));
    }
}

// Обработчики событий
function setupEventListeners() {
    elements.symbolSelect.addEventListener('change', handleSymbolChange);
    elements.timeframeSelect.addEventListener('change', handleTimeframeChange);
    elements.startBotBtn.addEventListener('click', startBot);
    elements.stopBotBtn.addEventListener('click', stopBot);
    elements.startTrainingBtn.addEventListener('click', startTraining);
    elements.exportLogsBtn.addEventListener('click', exportLogs);
    elements.viewBacktestBtn.addEventListener('click', viewBacktest);
    elements.runBacktestBtn.addEventListener('click', runBacktest);
    
    // Добавляем обработчики для новых элементов
    window.addEventListener('resize', debounce(updateChartsLayout, 250));
    document.addEventListener('visibilitychange', handleVisibilityChange);
}

// Обработка изменения видимости страницы
function handleVisibilityChange() {
    if (document.hidden) {
        stopPolling();
    } else {
        startPolling();
        updateDashboard();
    }
}

// API вызовы
async function startBot() {
    try {
        const response = await fetch(`${CONFIG.API_URL}/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol: state.currentSymbol,
                timeframe: state.currentTimeframe
            })
        });
        
        if (!response.ok) throw new Error('Failed to start bot');
        
        const data = await response.json();
        addLog(`Bot started for ${state.currentSymbol} ${state.currentTimeframe}`, 'success');
        updateSystemStatus(data);
    } catch (error) {
        handleError('Error starting bot', error);
    }
}

// Обновление дашборда
async function updateDashboard() {
    if (!state.currentSymbol || !state.currentTimeframe) return;
    
    try {
        await Promise.all([
            fetchStatus(),
            fetchBacktest(),
            updatePairStatuses(),
            updateCorrelations()
        ]);
        
        updateCharts();
        updateSystemMetrics();
    } catch (error) {
        handleError('Error updating dashboard', error);
    }
}

// Обработка ошибок
function handleError(context, error) {
    console.error(`${context}:`, error);
    
    const errorMessage = {
        context,
        message: error.message,
        timestamp: new Date().toISOString()
    };
    
    state.systemStatus.errors.push(errorMessage);
    addLog(`${context}: ${error.message}`, 'error');
    showErrorNotification(errorMessage);
    
    // Очистка старых ошибок
    if (state.systemStatus.errors.length > 10) {
        state.systemStatus.errors.shift();
    }
}

// Показ уведомлений об ошибках
function showErrorNotification(error) {
    const notification = document.createElement('div');
    notification.className = 'error-notification';
    notification.innerHTML = `
        <strong>${error.context}</strong>
        <p>${error.message}</p>
    `;
    
    elements.errorContainer.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, CONFIG.ERROR_NOTIFICATION_DURATION);
}

// Утилиты
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', init);

// WebSocket Message Handler
function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'status_update':
            updateStatus(data.data);
            break;
        case 'log':
            addLog(data.message);
            break;
        case 'error':
            addLog('Error: ' + data.message);
            break;
        case 'backtest':
            updateBacktestResults(data.data);
            break;
    }
}

// Logging
function addLog(message) {
    const logContainer = document.getElementById('log-content');
    const now = new Date();
    const time = now.toLocaleTimeString();
    const entry = document.createElement('div');
    entry.textContent = `[${time}] ${message}`;
    logContainer.appendChild(entry);
    logContainer.scrollTop = logContainer.scrollHeight;
}

// Polling
function startPolling() {
    if (!state.isPolling) {
        state.isPolling = true;
        setInterval(() => {
            if (state.currentSymbol && state.currentTimeframe) {
                fetchStatus();
            }
        }, CONFIG.POLLING_INTERVAL);
    }
}

// Функции для отображения статуса пар
async function updatePairStatuses() {
    try {
        const response = await fetch('/pairs/status');
        const statuses = await response.json();
        
        const statusContainer = document.getElementById('pair-statuses');
        statusContainer.innerHTML = '';
        
        for (const [pair, status] of Object.entries(statuses)) {
            const pairElement = document.createElement('div');
            pairElement.className = 'pair-status';
            
            const progress = status.progress;
            const progressBar = `
                <div class="progress">
                    <div class="progress-bar" role="progressbar" 
                         style="width: ${progress.data_collection * 100}%">
                        Сбор данных: ${Math.round(progress.data_collection * 100)}%
                    </div>
                    <div class="progress-bar" role="progressbar" 
                         style="width: ${progress.model_training * 100}%">
                        Обучение: ${Math.round(progress.model_training * 100)}%
                    </div>
                    <div class="progress-bar" role="progressbar" 
                         style="width: ${progress.backtest_completion * 100}%">
                        Бэктест: ${Math.round(progress.backtest_completion * 100)}%
                    </div>
                    <div class="progress-bar" role="progressbar" 
                         style="width: ${progress.correlation_analysis * 100}%">
                        Корреляции: ${Math.round(progress.correlation_analysis * 100)}%
                    </div>
                </div>
            `;
            
            pairElement.innerHTML = `
                <h3>${pair}</h3>
                <div class="status-info">
                    <p>Статус: ${status.meta_status}</p>
                    <p>Готовность к торговле: ${status.is_trade_ready ? 'Да' : 'Нет'}</p>
                    <p>Последнее обновление: ${new Date(status.last_update).toLocaleString()}</p>
                </div>
                ${progressBar}
            `;
            
            statusContainer.appendChild(pairElement);
        }
    } catch (error) {
        console.error('Error updating pair statuses:', error);
    }
}

// Функции для отображения корреляций
async function updateCorrelations() {
    try {
        const response = await fetch('/correlations');
        const structure = await response.json();
        
        const correlationContainer = document.getElementById('correlations');
        correlationContainer.innerHTML = '';
        
        // Отображение хабов
        const hubsSection = document.createElement('div');
        hubsSection.innerHTML = `
            <h3>Основные пары (хабы)</h3>
            <div class="hubs-list">
                ${structure.hubs.map(hub => `
                    <div class="hub-item">
                        <p>Пара: ${hub.pair}</p>
                        <p>Влияние: ${hub.influence_count} пар</p>
                        <p>Средняя корреляция: ${(hub.average_correlation * 100).toFixed(1)}%</p>
                    </div>
                `).join('')}
            </div>
        `;
        correlationContainer.appendChild(hubsSection);
        
        // Отображение кластеров
        const clustersSection = document.createElement('div');
        clustersSection.innerHTML = `
            <h3>Кластеры коррелирующих пар</h3>
            <div class="clusters-list">
                ${structure.clusters.map(cluster => `
                    <div class="cluster-item">
                        <p>Кластер: ${cluster.join(' → ')}</p>
                    </div>
                `).join('')}
            </div>
        `;
        correlationContainer.appendChild(clustersSection);
        
        // Отображение изолированных пар
        const isolatedSection = document.createElement('div');
        isolatedSection.innerHTML = `
            <h3>Изолированные пары</h3>
            <div class="isolated-list">
                ${structure.isolated.map(pair => `
                    <div class="isolated-item">
                        <p>${pair}</p>
                    </div>
                `).join('')}
            </div>
        `;
        correlationContainer.appendChild(isolatedSection);
    } catch (error) {
        console.error('Error updating correlations:', error);
    }
}

// Обновление данных каждые 5 секунд
setInterval(() => {
    updatePairStatuses();
    updateCorrelations();
}, 5000);

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    updatePairStatuses();
    updateCorrelations();
});

async function updatePairStatusUI() {
    const response = await fetch('/status');
    const data = await response.json();

    const pairs = data.pairs;
    const container = document.getElementById('pair-status-container');
    container.innerHTML = '';

    Object.entries(pairs).forEach(([symbol, info]) => {
        const whale = info.whale_activity;
        const cvd = info.cvd_status;
        const zone = info.zone_match;

        container.innerHTML += `
        <div class="pair-card">
            <h3>${symbol}</h3>
            <div>Статус: <span>${info.status}</span></div>
            <div>Последний сигнал: <b>${info.last_signal.type}</b> (уверенность: ${(info.signal_confidence * 100).toFixed(1)}%)</div>
            <div>Whale: ${whale.active ? `<span class="whale-active">Да</span> (${whale.side}, ${whale.volume}, ${whale.price})` : 'Нет'}</div>
            <div>CVD: <span class="cvd-${cvd.trend}">${cvd.trend}</span> (${cvd.value})</div>
            <div>Зона: <span class="zone-${zone.type}">${zone.type}</span> (${zone.price}, сила: ${zone.strength})</div>
            <div>Обновлено: ${info.last_update}</div>
        </div>
        `;
    });
}

// Пример: добавление лога при старте
addLog('Дашборд запущен.');

updatePairStatusUI();
setInterval(updatePairStatusUI, 5000);
