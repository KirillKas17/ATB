// ATB Dashboard - Main Application Logic

class ATBDashboard {
    constructor() {
        this.config = {
            apiUrl: 'http://localhost:8000',
            wsUrl: 'ws://localhost:8000/ws',
            updateInterval: 5000,
            reconnectAttempts: 5,
            reconnectDelay: 3000
        };
        
        this.state = {
            isConnected: false,
            isBotRunning: false,
            currentSection: 'overview',
            data: {
                system: {},
                trading: {},
                positions: [],
                analytics: {}
            },
            charts: {},
            intervals: {}
        };
        
        this.ws = null;
        this.reconnectCount = 0;
        
        this.init();
    }
    
    async init() {
        console.log('ATB Dashboard initializing...');
        
        // Инициализация компонентов
        this.initUI();
        this.initEventListeners();
        this.initCharts();
        this.loadSettings();
        
        // Подключение к API
        await this.connectToAPI();
        
        // Запуск обновлений
        this.startDataUpdates();
        
        console.log('ATB Dashboard initialized');
    }
    
    initUI() {
        // Обновление версии приложения
        this.updateAppVersion();
        
        // Инициализация навигации
        this.initNavigation();
        
        // Инициализация статуса подключения
        this.updateConnectionStatus('connecting');
    }
    
    async updateAppVersion() {
        try {
            const version = await window.electronAPI.getAppVersion();
            const versionElement = document.querySelector('.version');
            if (versionElement) {
                versionElement.textContent = `v${version}`;
            }
        } catch (error) {
            console.error('Failed to get app version:', error);
        }
    }
    
    initNavigation() {
        const navItems = document.querySelectorAll('.nav-item');
        
        navItems.forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const section = item.dataset.section;
                this.switchSection(section);
            });
        });
    }
    
    switchSection(sectionName) {
        // Обновление активной секции в навигации
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        
        document.querySelector(`[data-section="${sectionName}"]`).classList.add('active');
        
        // Скрытие всех секций
        document.querySelectorAll('.content-section').forEach(section => {
            section.classList.remove('active');
        });
        
        // Показ выбранной секции
        const targetSection = document.getElementById(`${sectionName}Section`);
        if (targetSection) {
            targetSection.classList.add('active');
            this.state.currentSection = sectionName;
            
            // Обновление данных для секции
            this.updateSectionData(sectionName);
        }
    }
    
    initEventListeners() {
        // Кнопки управления ботом
        document.getElementById('startBotBtn')?.addEventListener('click', () => {
            this.startBot();
        });
        
        document.getElementById('stopBotBtn')?.addEventListener('click', () => {
            this.stopBot();
        });
        
        // Кнопки обновления
        document.getElementById('refreshOverview')?.addEventListener('click', () => {
            this.refreshData('overview');
        });
        
        document.getElementById('refreshTrading')?.addEventListener('click', () => {
            this.refreshData('trading');
        });
        
        document.getElementById('refreshAnalytics')?.addEventListener('click', () => {
            this.refreshData('analytics');
        });
        
        document.getElementById('refreshPerformance')?.addEventListener('click', () => {
            this.refreshData('performance');
        });
        
        // Кнопка новой сделки
        document.getElementById('newTradeBtn')?.addEventListener('click', () => {
            this.showNewTradeModal();
        });
        
        // Кнопки экспорта
        document.getElementById('exportDataBtn')?.addEventListener('click', () => {
            this.exportData();
        });
        
        document.getElementById('exportReportBtn')?.addEventListener('click', () => {
            this.exportReport();
        });
        
        // Переключатели времени для графиков
        document.querySelectorAll('.time-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.updateChartData(e.target.dataset.period);
            });
        });
        
        // Настройки
        document.getElementById('apiUrl')?.addEventListener('change', (e) => {
            this.config.apiUrl = e.target.value;
            this.saveSettings();
        });
        
        document.getElementById('wsUrl')?.addEventListener('change', (e) => {
            this.config.wsUrl = e.target.value;
            this.saveSettings();
        });
        
        document.getElementById('updateInterval')?.addEventListener('change', (e) => {
            this.config.updateInterval = parseInt(e.target.value) * 1000;
            this.saveSettings();
            this.restartDataUpdates();
        });
        
        // Модальные окна
        document.getElementById('modalClose')?.addEventListener('click', () => {
            this.hideModal();
        });
        
        document.getElementById('modalOverlay')?.addEventListener('click', (e) => {
            if (e.target.id === 'modalOverlay') {
                this.hideModal();
            }
        });
        
        // Обработка событий от главного процесса
        this.initMainProcessEvents();
    }
    
    initMainProcessEvents() {
        // Экспорт данных
        window.electronAPI.onExportData(() => {
            this.exportData();
        });
        
        // Открытие настроек
        window.electronAPI.onOpenSettings(() => {
            this.switchSection('settings');
        });
        
        // Управление ботом
        window.electronAPI.onStartBot(() => {
            this.startBot();
        });
        
        window.electronAPI.onStopBot(() => {
            this.stopBot();
        });
        
        // Новая сделка
        window.electronAPI.onNewTrade(() => {
            this.showNewTradeModal();
        });
        
        // Обновление данных
        window.electronAPI.onRefreshData(() => {
            this.refreshAllData();
        });
        
        // Экспорт отчета
        window.electronAPI.onExportReport(() => {
            this.exportReport();
        });
        
        // Проверка обновлений
        window.electronAPI.onCheckUpdates(() => {
            this.checkForUpdates();
        });
    }
    
    async connectToAPI() {
        try {
            // Проверка доступности API
            const response = await fetch(`${this.config.apiUrl}/api/health`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const health = await response.json();
            console.log('API health check:', health);
            
            // Подключение WebSocket
            this.connectWebSocket();
            
            // Загрузка начальных данных
            await this.loadInitialData();
            
            this.updateConnectionStatus('connected');
            this.showNotification('Подключение установлено', 'success');
            
        } catch (error) {
            console.error('Failed to connect to API:', error);
            this.updateConnectionStatus('error');
            this.showNotification('Ошибка подключения к API', 'error');
            
            // Повторная попытка подключения
            setTimeout(() => {
                this.connectToAPI();
            }, this.config.reconnectDelay);
        }
    }
    
    connectWebSocket() {
        try {
            this.ws = new WebSocket(this.config.wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.reconnectCount = 0;
                this.updateConnectionStatus('connected');
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus('disconnected');
                
                // Автоматическое переподключение
                if (this.reconnectCount < this.config.reconnectAttempts) {
                    this.reconnectCount++;
                    setTimeout(() => {
                        this.connectWebSocket();
                    }, this.config.reconnectDelay);
                }
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('error');
            };
            
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
        }
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'data_update':
                this.updateData(data.data);
                break;
            case 'bot_status':
                this.updateBotStatus(data.status);
                break;
            case 'notification':
                this.showNotification(data.message, data.level);
                break;
            default:
                console.log('Unknown WebSocket message type:', data.type);
        }
    }
    
    async loadInitialData() {
        try {
            const [system, trading, positions, analytics] = await Promise.all([
                this.fetchData('/api/status'),
                this.fetchData('/api/trading'),
                this.fetchData('/api/positions'),
                this.fetchData('/api/analytics')
            ]);
            
            this.state.data = {
                system: system || {},
                trading: trading || {},
                positions: positions || [],
                analytics: analytics || {}
            };
            
            this.updateUI();
            
        } catch (error) {
            console.error('Failed to load initial data:', error);
        }
    }
    
    async fetchData(endpoint) {
        try {
            const response = await fetch(`${this.config.apiUrl}${endpoint}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`Failed to fetch ${endpoint}:`, error);
            return null;
        }
    }
    
    updateData(newData) {
        this.state.data = { ...this.state.data, ...newData };
        this.updateUI();
    }
    
    updateUI() {
        this.updateSystemStatus();
        this.updateTradingData();
        this.updatePositions();
        this.updateAnalytics();
        this.updateCharts();
    }
    
    updateSystemStatus() {
        const data = this.state.data.system;
        
        // Обновление статуса системы
        document.getElementById('systemStatus')?.textContent = data.status || 'Неизвестно';
        document.getElementById('uptime')?.textContent = data.uptime || '00:00:00';
        document.getElementById('cpuUsage')?.textContent = `${data.cpu_usage || 0}%`;
        document.getElementById('memoryUsage')?.textContent = `${data.memory_usage || 0} MB`;
    }
    
    updateTradingData() {
        const data = this.state.data.trading;
        
        // Обновление торговых данных
        document.getElementById('totalPnl')?.textContent = `$${(data.total_pnl || 0).toFixed(2)}`;
        document.getElementById('dailyPnl')?.textContent = `$${(data.daily_pnl || 0).toFixed(2)}`;
        document.getElementById('winRate')?.textContent = `${(data.win_rate || 0).toFixed(1)}%`;
        document.getElementById('activePositions')?.textContent = data.active_positions || 0;
        document.getElementById('totalTrades')?.textContent = data.total_trades || 0;
        document.getElementById('todayTrades')?.textContent = data.today_trades || 0;
        
        // Обновление цветов для P&L
        const totalPnlElement = document.getElementById('totalPnl');
        const dailyPnlElement = document.getElementById('dailyPnl');
        
        if (totalPnlElement) {
            totalPnlElement.className = `metric-value ${(data.total_pnl || 0) >= 0 ? 'positive' : 'negative'}`;
        }
        
        if (dailyPnlElement) {
            dailyPnlElement.className = `metric-value ${(data.daily_pnl || 0) >= 0 ? 'positive' : 'negative'}`;
        }
    }
    
    updatePositions() {
        const positions = this.state.data.positions || [];
        const container = document.getElementById('positionsList');
        
        if (!container) return;
        
        container.innerHTML = '';
        
        if (positions.length === 0) {
            container.innerHTML = '<div class="empty-state">Нет активных позиций</div>';
            return;
        }
        
        positions.forEach(position => {
            const positionElement = this.createPositionElement(position);
            container.appendChild(positionElement);
        });
    }
    
    createPositionElement(position) {
        const div = document.createElement('div');
        div.className = 'position-item';
        
        const pnlClass = (position.pnl || 0) >= 0 ? 'positive' : 'negative';
        const pnlSign = (position.pnl || 0) >= 0 ? '+' : '';
        
        div.innerHTML = `
            <div class="position-header">
                <span class="position-symbol">${position.symbol}</span>
                <span class="position-side ${position.side}">${position.side.toUpperCase()}</span>
            </div>
            <div class="position-details">
                <div class="position-size">Размер: ${position.size}</div>
                <div class="position-price">Цена: $${position.entry_price}</div>
                <div class="position-pnl ${pnlClass}">P&L: ${pnlSign}$${(position.pnl || 0).toFixed(2)} (${pnlSign}${(position.pnl_percent || 0).toFixed(2)}%)</div>
            </div>
        `;
        
        return div;
    }
    
    updateAnalytics() {
        const data = this.state.data.analytics;
        
        // Обновление индикаторов
        this.updateIndicators(data);
        
        // Обновление сигналов
        this.updateSignals(data);
    }
    
    updateIndicators(data) {
        const container = document.getElementById('indicatorsList');
        if (!container) return;
        
        const indicators = [
            { name: 'RSI', value: data.rsi || 0, unit: '' },
            { name: 'MACD', value: data.macd || 0, unit: '' },
            { name: 'Bollinger', value: data.bollinger_position || 'middle', unit: '' }
        ];
        
        container.innerHTML = '';
        
        indicators.forEach(indicator => {
            const div = document.createElement('div');
            div.className = 'indicator-item';
            div.innerHTML = `
                <span class="indicator-name">${indicator.name}</span>
                <span class="indicator-value">${indicator.value}${indicator.unit}</span>
            `;
            container.appendChild(div);
        });
    }
    
    updateSignals(data) {
        const container = document.getElementById('signalsList');
        if (!container) return;
        
        const signals = data.ai_signals || [];
        
        container.innerHTML = '';
        
        if (signals.length === 0) {
            container.innerHTML = '<div class="empty-state">Нет активных сигналов</div>';
            return;
        }
        
        signals.forEach(signal => {
            const div = document.createElement('div');
            div.className = `signal-item ${signal.type}`;
            div.innerHTML = `
                <span class="signal-icon">${signal.type === 'buy' ? '📈' : '📉'}</span>
                <span class="signal-text">${signal.message}</span>
                <span class="signal-strength">${signal.strength}</span>
            `;
            container.appendChild(div);
        });
    }
    
    updateConnectionStatus(status) {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-text');
        
        if (!statusDot || !statusText) return;
        
        statusDot.className = 'status-dot';
        statusText.textContent = 'Подключение...';
        
        switch (status) {
            case 'connected':
                statusDot.classList.add('connected');
                statusText.textContent = 'Подключено';
                this.state.isConnected = true;
                break;
            case 'connecting':
                statusText.textContent = 'Подключение...';
                this.state.isConnected = false;
                break;
            case 'disconnected':
                statusText.textContent = 'Отключено';
                this.state.isConnected = false;
                break;
            case 'error':
                statusDot.classList.add('error');
                statusText.textContent = 'Ошибка';
                this.state.isConnected = false;
                break;
        }
    }
    
    updateBotStatus(status) {
        this.state.isBotRunning = status === 'running';
        
        const startBtn = document.getElementById('startBotBtn');
        const stopBtn = document.getElementById('stopBotBtn');
        
        if (startBtn) {
            startBtn.disabled = this.state.isBotRunning;
            startBtn.style.opacity = this.state.isBotRunning ? '0.5' : '1';
        }
        
        if (stopBtn) {
            stopBtn.disabled = !this.state.isBotRunning;
            stopBtn.style.opacity = !this.state.isBotRunning ? '0.5' : '1';
        }
        
        // Обновление статуса торговли
        const tradingStatus = document.getElementById('tradingStatus');
        if (tradingStatus) {
            tradingStatus.textContent = this.state.isBotRunning ? 'Активна' : 'Остановлена';
            tradingStatus.className = `status-badge ${this.state.isBotRunning ? 'active' : ''}`;
        }
    }
    
    async startBot() {
        try {
            // Здесь будет вызов API для запуска бота
            this.showNotification('Запуск бота...', 'info');
            
            // Имитация запуска
            setTimeout(() => {
                this.updateBotStatus('running');
                this.showNotification('Бот запущен', 'success');
            }, 1000);
            
        } catch (error) {
            console.error('Failed to start bot:', error);
            this.showNotification('Ошибка запуска бота', 'error');
        }
    }
    
    async stopBot() {
        try {
            // Здесь будет вызов API для остановки бота
            this.showNotification('Остановка бота...', 'info');
            
            // Имитация остановки
            setTimeout(() => {
                this.updateBotStatus('stopped');
                this.showNotification('Бот остановлен', 'success');
            }, 1000);
            
        } catch (error) {
            console.error('Failed to stop bot:', error);
            this.showNotification('Ошибка остановки бота', 'error');
        }
    }
    
    startDataUpdates() {
        // Очистка существующих интервалов
        this.stopDataUpdates();
        
        // Запуск обновлений данных
        this.state.intervals.dataUpdate = setInterval(() => {
            this.refreshAllData();
        }, this.config.updateInterval);
        
        // Запуск обновления времени работы
        this.state.intervals.uptimeUpdate = setInterval(() => {
            this.updateUptime();
        }, 1000);
    }
    
    stopDataUpdates() {
        Object.values(this.state.intervals).forEach(interval => {
            if (interval) {
                clearInterval(interval);
            }
        });
        this.state.intervals = {};
    }
    
    restartDataUpdates() {
        this.stopDataUpdates();
        this.startDataUpdates();
    }
    
    async refreshAllData() {
        if (!this.state.isConnected) return;
        
        try {
            await this.loadInitialData();
        } catch (error) {
            console.error('Failed to refresh data:', error);
        }
    }
    
    async refreshData(section) {
        try {
            switch (section) {
                case 'overview':
                    await this.loadInitialData();
                    break;
                case 'trading':
                    const trading = await this.fetchData('/api/trading');
                    const positions = await this.fetchData('/api/positions');
                    this.state.data.trading = trading || {};
                    this.state.data.positions = positions || [];
                    this.updateTradingData();
                    this.updatePositions();
                    break;
                case 'analytics':
                    const analytics = await this.fetchData('/api/analytics');
                    this.state.data.analytics = analytics || {};
                    this.updateAnalytics();
                    break;
                case 'performance':
                    // Обновление метрик производительности
                    break;
            }
        } catch (error) {
            console.error(`Failed to refresh ${section} data:`, error);
        }
    }
    
    updateUptime() {
        const uptimeElement = document.getElementById('uptime');
        if (uptimeElement) {
            // Здесь будет логика обновления времени работы
            // Пока что просто обновляем каждую секунду
        }
    }
    
    showNotification(message, type = 'info') {
        const container = document.getElementById('notificationsContainer');
        if (!container) return;
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        container.appendChild(notification);
        
        // Автоматическое удаление через 5 секунд
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
    }
    
    showModal(title, content) {
        const modal = document.getElementById('modal');
        const modalTitle = document.getElementById('modalTitle');
        const modalContent = document.getElementById('modalContent');
        const modalOverlay = document.getElementById('modalOverlay');
        
        if (modalTitle) modalTitle.textContent = title;
        if (modalContent) modalContent.innerHTML = content;
        if (modalOverlay) modalOverlay.classList.add('active');
    }
    
    hideModal() {
        const modalOverlay = document.getElementById('modalOverlay');
        if (modalOverlay) modalOverlay.classList.remove('active');
    }
    
    showNewTradeModal() {
        const content = `
            <div class="form-group">
                <label for="tradeSymbol">Символ:</label>
                <input type="text" id="tradeSymbol" placeholder="BTC/USDT">
            </div>
            <div class="form-group">
                <label for="tradeSide">Сторона:</label>
                <select id="tradeSide">
                    <option value="buy">Покупка</option>
                    <option value="sell">Продажа</option>
                </select>
            </div>
            <div class="form-group">
                <label for="tradeAmount">Количество:</label>
                <input type="number" id="tradeAmount" placeholder="0.1">
            </div>
            <div class="form-actions">
                <button class="btn primary" onclick="dashboard.executeTrade()">Выполнить</button>
                <button class="btn secondary" onclick="dashboard.hideModal()">Отмена</button>
            </div>
        `;
        
        this.showModal('Новая сделка', content);
    }
    
    async executeTrade() {
        const symbol = document.getElementById('tradeSymbol')?.value;
        const side = document.getElementById('tradeSide')?.value;
        const amount = document.getElementById('tradeAmount')?.value;
        
        if (!symbol || !amount) {
            this.showNotification('Заполните все поля', 'error');
            return;
        }
        
        try {
            // Здесь будет вызов API для выполнения сделки
            this.showNotification('Сделка выполняется...', 'info');
            
            // Имитация выполнения сделки
            setTimeout(() => {
                this.hideModal();
                this.showNotification('Сделка выполнена', 'success');
                this.refreshData('trading');
            }, 1000);
            
        } catch (error) {
            console.error('Failed to execute trade:', error);
            this.showNotification('Ошибка выполнения сделки', 'error');
        }
    }
    
    async exportData() {
        try {
            const data = {
                timestamp: new Date().toISOString(),
                system: this.state.data.system,
                trading: this.state.data.trading,
                positions: this.state.data.positions,
                analytics: this.state.data.analytics
            };
            
            const result = await window.electronAPI.showSaveDialog({
                title: 'Экспорт данных',
                defaultPath: `atb-data-${new Date().toISOString().split('T')[0]}.json`,
                filters: [
                    { name: 'JSON Files', extensions: ['json'] }
                ]
            });
            
            if (!result.canceled && result.filePath) {
                await window.electronAPI.writeFile(result.filePath, JSON.stringify(data, null, 2));
                this.showNotification('Данные экспортированы', 'success');
            }
            
        } catch (error) {
            console.error('Failed to export data:', error);
            this.showNotification('Ошибка экспорта данных', 'error');
        }
    }
    
    async exportReport() {
        try {
            const report = this.generateReport();
            
            const result = await window.electronAPI.showSaveDialog({
                title: 'Экспорт отчета',
                defaultPath: `atb-report-${new Date().toISOString().split('T')[0]}.html`,
                filters: [
                    { name: 'HTML Files', extensions: ['html'] }
                ]
            });
            
            if (!result.canceled && result.filePath) {
                await window.electronAPI.writeFile(result.filePath, report);
                this.showNotification('Отчет экспортирован', 'success');
            }
            
        } catch (error) {
            console.error('Failed to export report:', error);
            this.showNotification('Ошибка экспорта отчета', 'error');
        }
    }
    
    generateReport() {
        const data = this.state.data;
        const timestamp = new Date().toLocaleString();
        
        return `
            <!DOCTYPE html>
            <html>
            <head>
                <title>ATB Trading Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .header { text-align: center; margin-bottom: 30px; }
                    .section { margin-bottom: 20px; }
                    .metric { display: flex; justify-content: space-between; margin: 5px 0; }
                    .positive { color: green; }
                    .negative { color: red; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>ATB Trading Report</h1>
                    <p>Сгенерирован: ${timestamp}</p>
                </div>
                
                <div class="section">
                    <h2>Статус системы</h2>
                    <div class="metric">Статус: ${data.system.status || 'Неизвестно'}</div>
                    <div class="metric">Время работы: ${data.system.uptime || '00:00:00'}</div>
                    <div class="metric">CPU: ${data.system.cpu_usage || 0}%</div>
                    <div class="metric">Память: ${data.system.memory_usage || 0} MB</div>
                </div>
                
                <div class="section">
                    <h2>Торговые данные</h2>
                    <div class="metric">Общий P&L: <span class="${(data.trading.total_pnl || 0) >= 0 ? 'positive' : 'negative'}">$${(data.trading.total_pnl || 0).toFixed(2)}</span></div>
                    <div class="metric">24ч P&L: <span class="${(data.trading.daily_pnl || 0) >= 0 ? 'positive' : 'negative'}">$${(data.trading.daily_pnl || 0).toFixed(2)}</span></div>
                    <div class="metric">Win Rate: ${(data.trading.win_rate || 0).toFixed(1)}%</div>
                    <div class="metric">Активных позиций: ${data.trading.active_positions || 0}</div>
                </div>
                
                <div class="section">
                    <h2>Активные позиции</h2>
                    ${(data.positions || []).map(pos => `
                        <div class="metric">
                            ${pos.symbol} (${pos.side.toUpperCase()}) - 
                            <span class="${(pos.pnl || 0) >= 0 ? 'positive' : 'negative'}">$${(pos.pnl || 0).toFixed(2)}</span>
                        </div>
                    `).join('')}
                </div>
            </body>
            </html>
        `;
    }
    
    async loadSettings() {
        try {
            const settings = await window.electronAPI.readFile('settings.json');
            if (settings.success) {
                const data = JSON.parse(settings.data);
                this.config = { ...this.config, ...data };
                
                // Обновление полей настроек
                document.getElementById('apiUrl').value = this.config.apiUrl;
                document.getElementById('wsUrl').value = this.config.wsUrl;
                document.getElementById('updateInterval').value = this.config.updateInterval / 1000;
            }
        } catch (error) {
            console.error('Failed to load settings:', error);
        }
    }
    
    async saveSettings() {
        try {
            const settings = {
                apiUrl: this.config.apiUrl,
                wsUrl: this.config.wsUrl,
                updateInterval: this.config.updateInterval
            };
            
            await window.electronAPI.writeFile('settings.json', JSON.stringify(settings, null, 2));
        } catch (error) {
            console.error('Failed to save settings:', error);
        }
    }
    
    async checkForUpdates() {
        this.showNotification('Проверка обновлений...', 'info');
        
        // Здесь будет логика проверки обновлений
        setTimeout(() => {
            this.showNotification('Обновления не найдены', 'info');
        }, 2000);
    }
    
    updateSectionData(section) {
        switch (section) {
            case 'overview':
                this.updateSystemStatus();
                this.updateTradingData();
                break;
            case 'trading':
                this.updateTradingData();
                this.updatePositions();
                break;
            case 'analytics':
                this.updateAnalytics();
                break;
            case 'performance':
                // Обновление метрик производительности
                break;
        }
    }
    
    initCharts() {
        // Инициализация графиков будет в отдельном файле
        console.log('Charts initialization');
    }
    
    updateCharts() {
        // Обновление графиков будет в отдельном файле
        console.log('Charts update');
    }
    
    updateChartData(period) {
        // Обновление данных графиков будет в отдельном файле
        console.log('Chart data update for period:', period);
    }
}

// Инициализация приложения
let dashboard;

document.addEventListener('DOMContentLoaded', () => {
    dashboard = new ATBDashboard();
    
    // Очистка при закрытии
    window.addEventListener('beforeunload', () => {
        if (dashboard) {
            dashboard.stopDataUpdates();
        }
    });
}); 