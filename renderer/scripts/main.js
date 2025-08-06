// ============================================================================
// ATB Trading System - Main JavaScript
// Основная логика для Electron приложения
// ============================================================================

class ATBApp {
    constructor() {
        this.isConnected = false;
        this.currentTab = 'overview';
        this.updateInterval = null;
        this.charts = {};
        
        // Инициализация при загрузке DOM
        document.addEventListener('DOMContentLoaded', () => {
            this.init();
        });
    }

    async init() {
        console.log('🚀 ATB Trading System Enhanced Desktop v3.1 - Initializing...');
        
        try {
            // Инициализация компонентов
            this.initEventListeners();
            this.initTabNavigation();
            this.initCharts();
            
            // Запуск обновлений
            this.startDataUpdates();
            
            // Проверка подключения к Electron API
            this.checkElectronAPI();
            
            console.log('✅ ATB App initialized successfully');
            this.showNotification('Система', 'ATB Trading System Enhanced запущен');
            
        } catch (error) {
            console.error('❌ Error initializing ATB App:', error);
            this.showError('Ошибка инициализации', error.message);
        }
    }

    checkElectronAPI() {
        if (window.electronAPI) {
            console.log('✅ Electron API доступен');
            this.isConnected = true;
            this.updateConnectionStatus(true);
            
            // Подписка на события от главного процесса
            window.electronAPI.onNavigateToTab((event, tabName) => {
                this.switchTab(tabName);
            });
            
            window.electronAPI.onStartTrading((event) => {
                this.startTrading();
            });
            
            window.electronAPI.onStopTrading((event) => {
                this.stopTrading();
            });
            
        } else {
            console.warn('⚠️ Electron API недоступен - работаем в демо режиме');
            this.isConnected = false;
            this.updateConnectionStatus(false);
            
            // Загрузка демо данных
            this.loadDemoData();
        }
    }

    initEventListeners() {
        // Обработчики кнопок заголовка
        const settingsBtn = document.getElementById('settingsBtn');
        const minimizeBtn = document.getElementById('minimizeBtn');
        const closeBtn = document.getElementById('closeBtn');

        if (settingsBtn) {
            settingsBtn.addEventListener('click', () => this.openSettings());
        }

        if (minimizeBtn) {
            minimizeBtn.addEventListener('click', () => this.minimizeWindow());
        }

        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.closeWindow());
        }

        // Обработчики панели инструментов
        const tradingToggle = document.getElementById('tradingToggle');
        const evolutionBtn = document.getElementById('evolutionBtn');
        const envBtn = document.getElementById('envBtn');
        const tradingMode = document.getElementById('tradingMode');

        if (tradingToggle) {
            tradingToggle.addEventListener('click', () => this.toggleTrading());
        }

        if (evolutionBtn) {
            evolutionBtn.addEventListener('click', () => this.openEvolution());
        }

        if (envBtn) {
            envBtn.addEventListener('click', () => this.openEnvSettings());
        }

        if (tradingMode) {
            tradingMode.addEventListener('change', (e) => this.changeTradingMode(e.target.value));
        }

        // Обработчики графиков
        const priceSymbol = document.getElementById('priceSymbol');
        if (priceSymbol) {
            priceSymbol.addEventListener('change', (e) => this.changePriceSymbol(e.target.value));
        }
    }

    initTabNavigation() {
        const tabButtons = document.querySelectorAll('.tab-btn');
        
        tabButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const tabName = button.getAttribute('data-tab');
                this.switchTab(tabName);
            });
        });
    }

    switchTab(tabName) {
        // Обновление кнопок
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        const activeButton = document.querySelector(`[data-tab="${tabName}"]`);
        if (activeButton) {
            activeButton.classList.add('active');
        }

        // Обновление контента
        document.querySelectorAll('.tab-pane').forEach(pane => {
            pane.classList.remove('active');
        });
        
        const activePane = document.getElementById(tabName);
        if (activePane) {
            activePane.classList.add('active');
            
            // Специальные действия для каждой вкладки
            this.onTabSwitch(tabName);
        }

        this.currentTab = tabName;
        console.log(`📋 Переключение на вкладку: ${tabName}`);
    }

    onTabSwitch(tabName) {
        switch (tabName) {
            case 'overview':
                this.updateOverviewData();
                break;
            case 'system':
                this.updateSystemData();
                break;
            case 'evolution':
                this.updateEvolutionData();
                break;
            case 'trading':
                this.updateTradingData();
                break;
            case 'portfolio':
                this.updatePortfolioData();
                break;
            case 'settings':
                this.updateSettingsData();
                break;
        }
    }

    initCharts() {
        this.initPriceChart();
        this.initPnlChart();
        this.initSystemChart();
    }

    initPriceChart() {
        const canvas = document.getElementById('priceChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        
        this.charts.price = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'BTC/USDT',
                    data: [],
                    borderColor: '#e94560',
                    backgroundColor: 'rgba(233, 69, 96, 0.1)',
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
                        labels: {
                            color: '#eee'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#ccc' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        ticks: { color: '#ccc' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    initPnlChart() {
        const canvas = document.getElementById('pnlChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        
        this.charts.pnl = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'P&L',
                    data: [],
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
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
                        labels: {
                            color: '#eee'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#ccc' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        ticks: { color: '#ccc' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    initSystemChart() {
        const canvas = document.getElementById('systemChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        
        this.charts.system = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'CPU %',
                        data: [],
                        borderColor: '#e94560',
                        backgroundColor: 'rgba(233, 69, 96, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4
                    },
                    {
                        label: 'Memory %',
                        data: [],
                        borderColor: '#45b7d1',
                        backgroundColor: 'rgba(69, 183, 209, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#eee'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#ccc' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { 
                            color: '#ccc',
                            callback: function(value) {
                                return value + '%';
                            }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    startDataUpdates() {
        // Обновление каждые 5 секунд
        this.updateInterval = setInterval(() => {
            this.updateData();
        }, 5000);

        // Первое обновление сразу
        this.updateData();
    }

    async updateData() {
        try {
            if (this.isConnected && window.electronAPI) {
                // Получение реальных данных через Electron API
                await this.updateRealData();
            } else {
                // Обновление демо данных
                this.updateDemoData();
            }
            
            this.updateLastUpdateTime();
            
        } catch (error) {
            console.error('❌ Error updating data:', error);
        }
    }

    async updateRealData() {
        try {
            // Системные метрики
            const systemMetrics = await window.electronAPI.getSystemMetrics();
            this.updateSystemMetrics(systemMetrics);

            // Статус эволюции
            const evolutionStatus = await window.electronAPI.getEvolutionStatus();
            this.updateEvolutionStatus(evolutionStatus);

            // ENV конфигурация
            const envConfig = await window.electronAPI.getEnvConfig();
            this.updateEnvStatus(envConfig);

        } catch (error) {
            console.error('❌ Error fetching real data:', error);
            this.fallbackToDemoData();
        }
    }

    loadDemoData() {
        console.log('📊 Загрузка демо данных...');
        
        // Генерация демо данных для графиков
        this.generateDemoChartData();
        
        // Демо системные метрики
        this.updateSystemMetrics({
            cpu: { percent: 25, cores: 8, frequency: 3200 },
            memory: { percent: 45, total: 16000000000, free: 8800000000 },
            disk: { percent: 60, total: 500000000000, free: 200000000000 },
            network: { bytes_sent: 1024000, bytes_recv: 2048000 }
        });
    }

    updateDemoData() {
        // Обновление демо данных с небольшими изменениями
        const currentTime = new Date().toLocaleTimeString();
        
        // Системные метрики
        const cpu = Math.random() * 30 + 20;
        const memory = Math.random() * 20 + 40;
        const disk = 60 + Math.random() * 5;
        
        this.updateSystemMetrics({
            cpu: { percent: Math.round(cpu), cores: 8, frequency: 3200 },
            memory: { percent: Math.round(memory), total: 16000000000, free: 8800000000 },
            disk: { percent: Math.round(disk), total: 500000000000, free: 200000000000 },
            network: { bytes_sent: 1024000 + Math.random() * 100000, bytes_recv: 2048000 + Math.random() * 200000 }
        });

        // Обновление графиков
        this.updateChartData(cpu, memory);
        
        // P&L данные
        const pnl = (Math.random() - 0.5) * 100;
        this.updatePnlData(pnl);
    }

    updateSystemMetrics(metrics) {
        // CPU
        const cpuPercent = document.getElementById('cpuPercent');
        const cpuMini = document.getElementById('cpuMini');
        const cpuCores = document.getElementById('cpuCores');
        const cpuFreq = document.getElementById('cpuFreq');
        
        if (cpuPercent) cpuPercent.textContent = `${metrics.cpu.percent}%`;
        if (cpuMini) cpuMini.textContent = `${metrics.cpu.percent}%`;
        if (cpuCores) cpuCores.textContent = metrics.cpu.cores;
        if (cpuFreq) cpuFreq.textContent = `${metrics.cpu.frequency} MHz`;

        // Memory
        const memoryPercent = document.getElementById('memoryPercent');
        const ramMini = document.getElementById('ramMini');
        const memoryTotal = document.getElementById('memoryTotal');
        const memoryFree = document.getElementById('memoryFree');
        
        if (memoryPercent) memoryPercent.textContent = `${metrics.memory.percent}%`;
        if (ramMini) ramMini.textContent = `${metrics.memory.percent}%`;
        if (memoryTotal) memoryTotal.textContent = this.formatBytes(metrics.memory.total);
        if (memoryFree) memoryFree.textContent = this.formatBytes(metrics.memory.free);

        // Disk
        const diskPercent = document.getElementById('diskPercent');
        const diskTotal = document.getElementById('diskTotal');
        const diskFree = document.getElementById('diskFree');
        
        if (diskPercent) diskPercent.textContent = `${metrics.disk.percent}%`;
        if (diskTotal) diskTotal.textContent = this.formatBytes(metrics.disk.total);
        if (diskFree) diskFree.textContent = this.formatBytes(metrics.disk.free);

        // Network
        const networkSent = document.getElementById('networkSent');
        const networkRecv = document.getElementById('networkRecv');
        
        if (networkSent) networkSent.textContent = this.formatBytes(metrics.network.bytes_sent);
        if (networkRecv) networkRecv.textContent = this.formatBytes(metrics.network.bytes_recv);
    }

    updateChartData(cpu, memory) {
        const currentTime = new Date().toLocaleTimeString();
        
        if (this.charts.system) {
            const chart = this.charts.system;
            
            // Добавление новых данных
            chart.data.labels.push(currentTime);
            chart.data.datasets[0].data.push(cpu);
            chart.data.datasets[1].data.push(memory);
            
            // Ограничение количества точек
            if (chart.data.labels.length > 20) {
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
                chart.data.datasets[1].data.shift();
            }
            
            chart.update('none');
        }
    }

    updatePnlData(pnl) {
        const currentPnl = document.getElementById('currentPnl');
        if (currentPnl) {
            currentPnl.textContent = `$${pnl.toFixed(2)}`;
            currentPnl.className = `pnl-value ${pnl >= 0 ? 'positive' : 'negative'}`;
        }

        // Обновление графика P&L
        if (this.charts.pnl) {
            const chart = this.charts.pnl;
            const currentTime = new Date().toLocaleTimeString();
            
            chart.data.labels.push(currentTime);
            chart.data.datasets[0].data.push(pnl);
            
            if (chart.data.labels.length > 20) {
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
            }
            
            chart.update('none');
        }
    }

    generateDemoChartData() {
        // Генерация данных для графика цен
        if (this.charts.price) {
            const chart = this.charts.price;
            const basePrice = 45000;
            
            for (let i = 0; i < 20; i++) {
                const time = new Date(Date.now() - (20 - i) * 60000).toLocaleTimeString();
                const price = basePrice + (Math.random() - 0.5) * 2000;
                
                chart.data.labels.push(time);
                chart.data.datasets[0].data.push(price);
            }
            
            chart.update();
        }
    }

    // Методы для обновления данных по вкладкам
    updateOverviewData() {
        console.log('📊 Обновление данных обзора');
    }

    updateSystemData() {
        console.log('🖥️ Обновление системных данных');
    }

    updateEvolutionData() {
        console.log('🧬 Обновление данных эволюции');
        
        // Обновление данных через Evolution UI если доступен
        if (window.evolutionUI) {
            window.evolutionUI.updateEvolutionData();
        }
    }

    updateTradingData() {
        console.log('📈 Обновление торговых данных');
    }

    updatePortfolioData() {
        console.log('💼 Обновление данных портфеля');
    }

    updateSettingsData() {
        console.log('⚙️ Обновление настроек');
    }

    // Обработчики событий
    toggleTrading() {
        const button = document.getElementById('tradingToggle');
        const icon = button.querySelector('i');
        const text = button.querySelector('span');
        
        if (button.dataset.active === 'true') {
            // Остановить торговлю
            button.dataset.active = 'false';
            icon.className = 'fas fa-play';
            text.textContent = 'Запустить торговлю';
            button.classList.remove('active');
            
            this.updateSystemStatus(false);
            this.showNotification('Торговля', 'Торговля остановлена');
            
        } else {
            // Запустить торговлю
            button.dataset.active = 'true';
            icon.className = 'fas fa-stop';
            text.textContent = 'Остановить торговлю';
            button.classList.add('active');
            
            this.updateSystemStatus(true);
            this.showNotification('Торговля', 'Торговля запущена');
        }
    }

    updateSystemStatus(isActive) {
        const statusDot = document.querySelector('#systemStatus .status-dot');
        const statusText = document.querySelector('#systemStatus span');
        const systemStatusText = document.getElementById('systemStatusText');
        
        if (isActive) {
            statusDot.className = 'status-dot online';
            statusText.textContent = 'Система активна';
            if (systemStatusText) systemStatusText.textContent = 'Активна';
        } else {
            statusDot.className = 'status-dot offline';
            statusText.textContent = 'Система остановлена';
            if (systemStatusText) systemStatusText.textContent = 'Остановлена';
        }
    }

    updateConnectionStatus(isConnected) {
        const connectionDot = document.querySelector('#connectionStatus .connection-dot');
        const connectionText = document.querySelector('#connectionStatus span');
        const connectionIndicator = document.querySelector('.connection-indicator');
        const connectionTextEl = document.getElementById('connectionText');
        
        if (isConnected) {
            if (connectionDot) connectionDot.className = 'connection-dot online';
            if (connectionText) connectionText.textContent = 'Подключено';
            if (connectionIndicator) connectionIndicator.className = 'connection-indicator online';
            if (connectionTextEl) connectionTextEl.textContent = 'Подключение: Активно';
        } else {
            if (connectionDot) connectionDot.className = 'connection-dot offline';
            if (connectionText) connectionText.textContent = 'Отключено';
            if (connectionIndicator) connectionIndicator.className = 'connection-indicator offline';
            if (connectionTextEl) connectionTextEl.textContent = 'Подключение: Отключено';
        }
    }

    changeTradingMode(mode) {
        const envModeStatus = document.getElementById('envModeStatus');
        const modeText = document.getElementById('modeText');
        
        const modeNames = {
            simulation: '🎮 Симуляция',
            paper: '📝 Бумажная торговля',
            live: '💰 Реальная торговля'
        };
        
        const modeName = modeNames[mode] || '🎮 Симуляция';
        
        if (envModeStatus) envModeStatus.textContent = modeName;
        if (modeText) modeText.textContent = modeName.substring(2);
        
        this.showNotification('Режим торговли', `Переключено на: ${modeName}`);
    }

    changePriceSymbol(symbol) {
        if (this.charts.price) {
            this.charts.price.data.datasets[0].label = symbol;
            this.charts.price.update();
        }
        
        this.showNotification('График', `Переключено на пару: ${symbol}`);
    }

    // Действия с окном
    minimizeWindow() {
        console.log('➖ Сворачивание окна');
        // В реальном Electron это будет обрабатываться main процессом
    }

    closeWindow() {
        console.log('❌ Закрытие окна');
        // В реальном Electron это будет обрабатываться main процессом
    }

    openSettings() {
        this.switchTab('settings');
    }

    openEvolution() {
        this.switchTab('evolution');
    }

    openEnvSettings() {
        this.switchTab('settings');
        // Дополнительная логика для открытия секции ENV
    }

    // Утилиты
    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    updateLastUpdateTime() {
        const lastUpdateTime = document.getElementById('lastUpdateTime');
        if (lastUpdateTime) {
            const now = new Date().toLocaleTimeString();
            lastUpdateTime.textContent = `Последнее обновление: ${now}`;
        }
    }

    showNotification(title, message) {
        console.log(`🔔 ${title}: ${message}`);
        
        // В реальном Electron можно использовать системные уведомления
        if (window.electronAPI) {
            window.electronAPI.showNotification(title, message);
        } else {
            // Веб уведомления как fallback
            if (Notification.permission === 'granted') {
                new Notification(title, { body: message });
            }
        }
    }

    showError(title, message) {
        console.error(`❌ ${title}: ${message}`);
        
        // Показ ошибки в интерфейсе
        this.addLogEntry('error', `${title}: ${message}`);
    }

    addLogEntry(level, message) {
        const logContainer = document.getElementById('logContainer');
        if (!logContainer) return;
        
        const time = new Date().toLocaleTimeString();
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.innerHTML = `
            <span class="log-time">${time}</span>
            <span class="log-level ${level}">${level.toUpperCase()}</span>
            <span class="log-message">${message}</span>
        `;
        
        logContainer.appendChild(entry);
        
        // Ограничение количества записей
        while (logContainer.children.length > 50) {
            logContainer.removeChild(logContainer.firstChild);
        }
        
        // Автоскролл
        logContainer.scrollTop = logContainer.scrollHeight;
    }

    fallbackToDemoData() {
        console.warn('⚠️ Переключение на демо данные из-за ошибки подключения');
        this.isConnected = false;
        this.updateConnectionStatus(false);
        this.loadDemoData();
    }

    destroy() {
        // Очистка ресурсов
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        // Очистка графиков
        Object.values(this.charts).forEach(chart => {
            if (chart) chart.destroy();
        });
        
        // Отписка от событий Electron
        if (window.electronAPI) {
            window.electronAPI.removeAllListeners('navigate-to-tab');
            window.electronAPI.removeAllListeners('start-trading');
            window.electronAPI.removeAllListeners('stop-trading');
        }
        
        console.log('🧹 ATB App destroyed');
    }
}

// Создание экземпляра приложения
window.atbApp = new ATBApp();

// Очистка при выгрузке страницы
window.addEventListener('beforeunload', () => {
    if (window.atbApp) {
        window.atbApp.destroy();
    }
});

console.log('⚡ ATB Trading System Enhanced Desktop v3.1 - Main script loaded');