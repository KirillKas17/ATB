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
        
        // Проверяем подключение к backend серверу
        this.checkBackendConnection();
    }
    
    async checkBackendConnection() {
        try {
            const response = await fetch('http://localhost:3001/api/health');
            if (response.ok) {
                console.log('✅ Backend сервер доступен');
                this.updateConnectionStatus(true);
            } else {
                console.warn('⚠️ Backend сервер недоступен');
                this.updateConnectionStatus(false);
            }
        } catch (error) {
            console.warn('⚠️ Не удалось подключиться к backend серверу:', error);
            this.updateConnectionStatus(false);
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

        // Обработчики P&L графика
        const pnlTimeframe = document.getElementById('pnlTimeframe');
        const refreshPnlChart = document.getElementById('refreshPnlChart');
        
        if (pnlTimeframe) {
            pnlTimeframe.addEventListener('change', (e) => this.updatePnlChartData());
        }
        
        if (refreshPnlChart) {
            refreshPnlChart.addEventListener('click', () => this.updatePnlChartData());
        }

        // Обработчики событий
        const logLevel = document.getElementById('logLevel');
        const clearLogs = document.getElementById('clearLogs');
        const exportLogs = document.getElementById('exportLogs');
        
        if (logLevel) {
            logLevel.addEventListener('change', (e) => this.filterLogs(e.target.value));
        }
        
        if (clearLogs) {
            clearLogs.addEventListener('click', () => this.clearLogs());
        }
        
        if (exportLogs) {
            exportLogs.addEventListener('click', () => this.exportLogs());
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
            case 'pnl-chart':
                this.updatePnlChartData();
                break;
            case 'events':
                this.updateEventsData();
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
        this.initPnlChartFull();
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

    initPnlChartFull() {
        const canvas = document.getElementById('pnlChartFull');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        
        this.charts.pnlFull = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'P&L',
                    data: [],
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    borderWidth: 3,
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
                            color: '#eee',
                            font: {
                                size: 14
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { 
                            color: '#ccc',
                            font: {
                                size: 12
                            }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        ticks: { 
                            color: '#ccc',
                            font: {
                                size: 12
                            }
                        },
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
            const backendUrl = 'http://localhost:3001';
            
            // Системные метрики
            const systemResponse = await fetch(`${backendUrl}/api/system/metrics`);
            if (systemResponse.ok) {
                const systemMetrics = await systemResponse.json();
                this.updateSystemMetrics(systemMetrics);
                this.updateConnectionStatus(true);
            }

            // Торговые данные
            const tradingResponse = await fetch(`${backendUrl}/api/trading/status`);
            if (tradingResponse.ok) {
                const tradingData = await tradingResponse.json();
                this.updateTradingData(tradingData);
            }

            // Данные эволюции
            const evolutionResponse = await fetch(`${backendUrl}/api/evolution/status`);
            if (evolutionResponse.ok) {
                const evolutionData = await evolutionResponse.json();
                this.updateEvolutionData(evolutionData);
            }

            // ML данные
            const mlResponse = await fetch(`${backendUrl}/api/ml/status`);
            if (mlResponse.ok) {
                const mlData = await mlResponse.json();
                this.updateMLData(mlData);
            }

            // Портфель
            const portfolioResponse = await fetch(`${backendUrl}/api/trading/portfolio`);
            if (portfolioResponse.ok) {
                const portfolioData = await portfolioResponse.json();
                this.updatePortfolioData(portfolioData);
            }

            // Рыночные данные
            const marketResponse = await fetch(`${backendUrl}/api/trading/market-data`);
            if (marketResponse.ok) {
                const marketData = await marketResponse.json();
                this.updateMarketData(marketData);
            }

            // Обновление данных обзора если мы на этой вкладке
            if (this.currentTab === 'overview') {
                await this.updateOverviewData();
            }

        } catch (error) {
            console.error('❌ Error fetching real data:', error);
            this.updateConnectionStatus(false);
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
    async updateOverviewData() {
        console.log('📊 Обновление данных обзора');
        
        try {
            // Получаем данные от backend
            const backendUrl = 'http://localhost:3001';
            
            // 1. Системные данные
            const systemResponse = await fetch(`${backendUrl}/api/system/metrics`);
            if (systemResponse.ok) {
                const systemData = await systemResponse.json();
                this.updateSystemBlock(systemData);
            }
            
            // 2. Баланс и P&L данные
            const portfolioResponse = await fetch(`${backendUrl}/api/trading/portfolio`);
            if (portfolioResponse.ok) {
                const portfolioData = await portfolioResponse.json();
                this.updateBalanceBlock(portfolioData);
            }
            
            // 3. Активные стратегии
            const evolutionResponse = await fetch(`${backendUrl}/api/evolution/status`);
            if (evolutionResponse.ok) {
                const evolutionData = await evolutionResponse.json();
                this.updateStrategiesBlock(evolutionData);
            }
            
            // 4. Активные позиции
            const tradingResponse = await fetch(`${backendUrl}/api/trading/status`);
            if (tradingResponse.ok) {
                const tradingData = await tradingResponse.json();
                this.updatePositionsBlock(tradingData);
            }
            
            // 5. Открытые ордера
            if (tradingResponse.ok) {
                this.updateOrdersBlock(tradingData);
            }
            
        } catch (error) {
            console.error('❌ Ошибка обновления данных обзора:', error);
            this.fallbackToDemoData();
        }
    }

    updateSystemBlock(systemData) {
        // Обновляем блок "Система"
        const systemStatusText = document.getElementById('systemStatusText');
        const uptimeText = document.getElementById('uptimeText');
        const modeText = document.getElementById('modeText');
        
        if (systemStatusText) {
            systemStatusText.textContent = 'Активна';
        }
        
        if (uptimeText) {
            // Вычисляем время работы с момента запуска
            const startTime = new Date(Date.now() - 3600000); // 1 час назад для демо
            const uptime = Date.now() - startTime.getTime();
            const hours = Math.floor(uptime / 3600000);
            const minutes = Math.floor((uptime % 3600000) / 60000);
            const seconds = Math.floor((uptime % 60000) / 1000);
            uptimeText.textContent = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
        
        if (modeText) {
            modeText.textContent = 'Симуляция';
        }
    }

    updateBalanceBlock(portfolioData) {
        // Обновляем блок "Баланс и P&L"
        const totalBalance = document.getElementById('totalBalance');
        const currentPnl = document.getElementById('currentPnl');
        const dailyPnl = document.getElementById('dailyPnl');
        const monthlyPnl = document.getElementById('monthlyPnl');
        
        // Проверяем статус подключения к бирже
        if (portfolioData.status === 'disconnected') {
            if (totalBalance) {
                totalBalance.textContent = 'Не подключено';
                totalBalance.style.color = 'var(--text-muted)';
            }
            
            if (currentPnl) {
                currentPnl.textContent = 'Нет данных';
                currentPnl.className = 'pnl-value';
                currentPnl.style.color = 'var(--text-muted)';
            }
            
            if (dailyPnl) {
                dailyPnl.textContent = 'Нет данных';
                dailyPnl.className = 'pnl-value';
                dailyPnl.style.color = 'var(--text-muted)';
            }
            
            if (monthlyPnl) {
                monthlyPnl.textContent = 'Нет данных';
                monthlyPnl.className = 'pnl-value';
                monthlyPnl.style.color = 'var(--text-muted)';
            }
        } else {
            if (totalBalance) {
                totalBalance.textContent = `$${portfolioData.balance?.toFixed(2) || '0.00'}`;
                totalBalance.style.color = 'var(--text-primary)';
            }
            
            if (currentPnl) {
                const pnl = portfolioData.pnl || 0;
                currentPnl.textContent = `$${pnl.toFixed(2)}`;
                currentPnl.className = `pnl-value ${pnl >= 0 ? 'positive' : 'negative'}`;
                currentPnl.style.color = '';
            }
            
            if (dailyPnl) {
                const daily = portfolioData.dailyPnL || 0;
                dailyPnl.textContent = `$${daily.toFixed(2)}`;
                dailyPnl.className = `pnl-value ${daily >= 0 ? 'positive' : 'negative'}`;
                dailyPnl.style.color = '';
            }
            
            if (monthlyPnl) {
                const monthly = portfolioData.monthlyPnL || 0;
                monthlyPnl.textContent = `$${monthly.toFixed(2)}`;
                monthlyPnl.className = `pnl-value ${monthly >= 0 ? 'positive' : 'negative'}`;
                monthlyPnl.style.color = '';
            }
        }
    }

    updateStrategiesBlock(evolutionData) {
        // Обновляем блок "Активные стратегии"
        const strategiesList = document.getElementById('strategiesList');
        if (!strategiesList) return;
        
        strategiesList.innerHTML = '';
        
        if (evolutionData.strategies && evolutionData.strategies.length > 0) {
            // Показываем только активные стратегии
            const activeStrategies = evolutionData.strategies.filter(s => s.status === 'active').slice(0, 4);
            
            if (activeStrategies.length > 0) {
                activeStrategies.forEach(strategy => {
                    const strategyElement = this.createStrategyElement(strategy);
                    strategiesList.appendChild(strategyElement);
                });
            } else {
                // Нет активных стратегий
                const noStrategiesElement = document.createElement('div');
                noStrategiesElement.className = 'strategy-item';
                noStrategiesElement.innerHTML = `
                    <div class="strategy-icon">📋</div>
                    <div class="strategy-info">
                        <div class="strategy-name">Нет активных стратегий</div>
                        <div class="strategy-status paused">Ожидание</div>
                    </div>
                `;
                strategiesList.appendChild(noStrategiesElement);
            }
        } else {
            // Нет стратегий вообще
            const noStrategiesElement = document.createElement('div');
            noStrategiesElement.className = 'strategy-item';
            noStrategiesElement.innerHTML = `
                <div class="strategy-icon">📋</div>
                <div class="strategy-info">
                    <div class="strategy-name">Стратегии не загружены</div>
                    <div class="strategy-status paused">Ожидание</div>
                </div>
            `;
            strategiesList.appendChild(noStrategiesElement);
        }
    }

    updatePositionsBlock(tradingData) {
        // Обновляем блок "Активные позиции"
        const positionsTable = document.getElementById('positionsTable');
        if (!positionsTable) return;
        
        const tbody = positionsTable.querySelector('tbody');
        if (!tbody) return;
        
        tbody.innerHTML = '';
        
        if (tradingData.exchangeConnected && tradingData.positions && tradingData.positions.length > 0) {
            tradingData.positions.forEach(position => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${position.symbol}</td>
                    <td>${position.size?.toFixed(4) || '0.0000'}</td>
                    <td>$${position.entryPrice?.toFixed(2) || '0.00'}</td>
                    <td class="${position.pnl >= 0 ? 'positive' : 'negative'}">$${position.pnl?.toFixed(2) || '0.00'}</td>
                `;
                tbody.appendChild(row);
            });
        } else {
            // Показываем сообщение о том, что нет данных
            const row = document.createElement('tr');
            row.innerHTML = `
                <td colspan="4" style="text-align: center; color: var(--text-muted); padding: 20px;">
                    ${tradingData.exchangeConnected ? 'Нет активных позиций' : 'Биржа не подключена'}
                </td>
            `;
            tbody.appendChild(row);
        }
    }

    updateOrdersBlock(tradingData) {
        // Обновляем блок "Открытые ордера"
        const ordersTable = document.getElementById('ordersTable');
        if (!ordersTable) return;
        
        const tbody = ordersTable.querySelector('tbody');
        if (!tbody) return;
        
        tbody.innerHTML = '';
        
        if (tradingData.exchangeConnected && tradingData.orders && tradingData.orders.length > 0) {
            tradingData.orders.forEach(order => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${order.symbol}</td>
                    <td>${order.type}</td>
                    <td>${order.size?.toFixed(4) || '0.0000'}</td>
                    <td>$${order.price?.toFixed(2) || '0.00'}</td>
                `;
                tbody.appendChild(row);
            });
        } else {
            // Показываем сообщение о том, что нет данных
            const row = document.createElement('tr');
            row.innerHTML = `
                <td colspan="4" style="text-align: center; color: var(--text-muted); padding: 20px;">
                    ${tradingData.exchangeConnected ? 'Нет открытых ордеров' : 'Биржа не подключена'}
                </td>
            `;
            tbody.appendChild(row);
        }
    }

    updatePnlChartData() {
        console.log('💹 Обновление данных P&L графика');
        
        // Обновление полного P&L графика
        if (this.charts.pnlFull) {
            const chart = this.charts.pnlFull;
            const data = this.generatePnlChartData();
            
            chart.data.labels = data.labels;
            chart.data.datasets[0].data = data.values;
            chart.update();
        }
        
        // Обновление статистики P&L
        this.updatePnlStats();
    }

    updateEventsData() {
        console.log('📝 Обновление данных событий');
        
        // Добавление новых событий в лог
        this.addDemoEvents();
        
        // Обновление фильтра событий
        this.initEventsFilter();
    }

    generatePnlChartData() {
        // Генерация демо данных для P&L графика
        const labels = [];
        const values = [];
        const now = new Date();
        
        for (let i = 30; i >= 0; i--) {
            const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
            labels.push(date.toLocaleDateString());
            
            // Генерация реалистичных P&L данных
            const baseValue = 1000;
            const volatility = 200;
            const trend = Math.sin(i / 10) * 100;
            const random = (Math.random() - 0.5) * volatility;
            const value = baseValue + trend + random;
            
            values.push(value);
        }
        
        return { labels, values };
    }

    updatePnlStats() {
        // Обновление статистики P&L
        const totalPnL = document.getElementById('totalPnL');
        const maxDrawdown = document.getElementById('maxDrawdown');
        const winRate = document.getElementById('winRate');
        const sharpeRatio = document.getElementById('sharpeRatio');
        
        if (totalPnL) totalPnL.textContent = '$1,234.56';
        if (maxDrawdown) maxDrawdown.textContent = '12.5%';
        if (winRate) winRate.textContent = '68.2%';
        if (sharpeRatio) sharpeRatio.textContent = '1.85';
    }

    addDemoEvents() {
        const logContainer = document.getElementById('eventsLogContainer');
        if (!logContainer) return;
        
        const events = [
            { time: '14:23:45', level: 'info', message: 'Обновление данных портфеля' },
            { time: '14:23:42', level: 'success', message: 'Ордер BTC/USDT исполнен' },
            { time: '14:23:38', level: 'warning', message: 'Высокая волатильность ETH/USDT' },
            { time: '14:23:35', level: 'info', message: 'Анализ рыночных данных' },
            { time: '14:23:30', level: 'success', message: 'Стратегия "Трендовая" активирована' }
        ];
        
        events.forEach(event => {
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry';
            logEntry.innerHTML = `
                <span class="log-time">${event.time}</span>
                <span class="log-level ${event.level}">${event.level.toUpperCase()}</span>
                <span class="log-message">${event.message}</span>
            `;
            logContainer.appendChild(logEntry);
        });
    }

    initEventsFilter() {
        const logLevel = document.getElementById('logLevel');
        const clearLogs = document.getElementById('clearLogs');
        const exportLogs = document.getElementById('exportLogs');
        
        if (logLevel) {
            logLevel.addEventListener('change', (e) => {
                this.filterLogs(e.target.value);
            });
        }
        
        if (clearLogs) {
            clearLogs.addEventListener('click', () => {
                this.clearLogs();
            });
        }
        
        if (exportLogs) {
            exportLogs.addEventListener('click', () => {
                this.exportLogs();
            });
        }
    }

    filterLogs(level) {
        const logEntries = document.querySelectorAll('#eventsLogContainer .log-entry');
        
        logEntries.forEach(entry => {
            const logLevel = entry.querySelector('.log-level');
            if (level === 'all' || logLevel.classList.contains(level)) {
                entry.style.display = 'flex';
            } else {
                entry.style.display = 'none';
            }
        });
    }

    clearLogs() {
        const logContainer = document.getElementById('eventsLogContainer');
        if (logContainer) {
            logContainer.innerHTML = '';
            this.addLogEntry('info', 'Лог событий очищен');
        }
    }

    exportLogs() {
        const logContainer = document.getElementById('eventsLogContainer');
        if (!logContainer) return;
        
        const logEntries = logContainer.querySelectorAll('.log-entry');
        let logText = 'ATB Trading System - Лог событий\n';
        logText += '=====================================\n\n';
        
        logEntries.forEach(entry => {
            const time = entry.querySelector('.log-time').textContent;
            const level = entry.querySelector('.log-level').textContent;
            const message = entry.querySelector('.log-message').textContent;
            logText += `[${time}] ${level}: ${message}\n`;
        });
        
        // Создание и скачивание файла
        const blob = new Blob([logText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `atb_logs_${new Date().toISOString().split('T')[0]}.txt`;
        a.click();
        URL.revokeObjectURL(url);
        
        this.showNotification('Экспорт', 'Лог событий экспортирован');
    }

    updateSystemData() {
        console.log('🖥️ Обновление системных данных');
    }

    updateEvolutionData(evolutionData) {
        try {
            console.log('🧬 Обновление данных эволюции:', evolutionData);
            
            // Обновление статуса эволюции
            const evolutionStatus = document.getElementById('evolutionStatus');
            const evolutionToggle = document.getElementById('evolutionBtn');
            
            if (evolutionStatus) {
                evolutionStatus.textContent = evolutionData.running ? 'Активна' : 'Остановлена';
                evolutionStatus.className = evolutionData.running ? 'status-badge online' : 'status-badge offline';
            }
            
            if (evolutionToggle) {
                const icon = evolutionToggle.querySelector('i');
                const text = evolutionToggle.querySelector('span');
                if (icon) icon.className = evolutionData.running ? 'fas fa-stop' : 'fas fa-play';
                if (text) text.textContent = evolutionData.running ? 'Остановить эволюцию' : 'Запустить эволюцию';
            }

            // Обновление стратегий
            const strategiesContainer = document.getElementById('evolutionStrategies');
            if (strategiesContainer && evolutionData.strategies) {
                strategiesContainer.innerHTML = '';
                evolutionData.strategies.forEach(strategy => {
                    const strategyElement = this.createStrategyElement(strategy);
                    strategiesContainer.appendChild(strategyElement);
                });
            }

            // Обновление статистики эволюции
            if (evolutionData.statistics) {
                const totalEvolutions = document.getElementById('totalEvolutions');
                const bestPerformance = document.getElementById('bestPerformance');
                const averageGenerations = document.getElementById('averageGenerations');
                
                if (totalEvolutions) totalEvolutions.textContent = evolutionData.statistics.totalEvolutions || 0;
                if (bestPerformance) bestPerformance.textContent = `${(evolutionData.statistics.bestPerformance || 0).toFixed(2)}%`;
                if (averageGenerations) averageGenerations.textContent = evolutionData.statistics.averageGenerations || 0;
            }

            // Обновление через Evolution UI если доступен
            if (window.evolutionUI) {
                window.evolutionUI.updateEvolutionData(evolutionData);
            }

        } catch (error) {
            console.error('Error updating evolution data:', error);
        }
    }

    createStrategyElement(strategy) {
        const element = document.createElement('div');
        element.className = 'strategy-item';
        
        // Определяем иконку в зависимости от типа стратегии
        const getStrategyIcon = (type) => {
            const icons = {
                'trend': '🔄',
                'momentum': '📊',
                'mean_reversion': '📈',
                'arbitrage': '🎯',
                'ml_enhanced': '🤖',
                'scalping': '⚡'
            };
            return icons[type] || '📋';
        };
        
        // Определяем статус для отображения
        const getStatusText = (status) => {
            return status === 'active' ? 'Активна' : 'Пауза';
        };
        
        element.innerHTML = `
            <div class="strategy-icon">${getStrategyIcon(strategy.type)}</div>
            <div class="strategy-info">
                <div class="strategy-name">${strategy.name}</div>
                <div class="strategy-status ${strategy.status}">${getStatusText(strategy.status)}</div>
            </div>
        `;
        return element;
    }

    updateTradingData(tradingData) {
        try {
            console.log('📈 Обновление торговых данных:', tradingData);
            
            // Обновление статуса торговли
            const tradingStatus = document.getElementById('tradingStatus');
            const tradingToggle = document.getElementById('tradingToggle');
            
            if (tradingStatus) {
                tradingStatus.textContent = tradingData.isRunning ? 'Активна' : 'Остановлена';
                tradingStatus.className = tradingData.isRunning ? 'status-badge online' : 'status-badge offline';
            }
            
            if (tradingToggle) {
                const icon = tradingToggle.querySelector('i');
                const text = tradingToggle.querySelector('span');
                if (icon) icon.className = tradingData.isRunning ? 'fas fa-stop' : 'fas fa-play';
                if (text) text.textContent = tradingData.isRunning ? 'Остановить торговлю' : 'Запустить торговлю';
            }

            // Обновление метрик производительности
            if (tradingData.performanceMetrics) {
                const totalReturn = document.getElementById('totalReturn');
                const sharpeRatio = document.getElementById('sharpeRatio');
                const maxDrawdown = document.getElementById('maxDrawdown');
                const winRate = document.getElementById('winRate');
                
                if (totalReturn) totalReturn.textContent = `${tradingData.performanceMetrics.totalReturn.toFixed(2)}%`;
                if (sharpeRatio) sharpeRatio.textContent = tradingData.performanceMetrics.sharpeRatio.toFixed(2);
                if (maxDrawdown) maxDrawdown.textContent = `${tradingData.performanceMetrics.maxDrawdown.toFixed(2)}%`;
                if (winRate) winRate.textContent = `${tradingData.performanceMetrics.winRate.toFixed(1)}%`;
            }

            // Обновление активных позиций
            const activePositions = document.getElementById('activePositions');
            if (activePositions && tradingData.currentPositions) {
                activePositions.textContent = tradingData.currentPositions.size || 0;
            }

            // Обновление истории сделок
            if (tradingData.tradeHistory) {
                this.updateTradeHistory(tradingData.tradeHistory);
            }

        } catch (error) {
            console.error('Error updating trading data:', error);
        }
    }

    updatePortfolioData(portfolioData) {
        try {
            console.log('💼 Обновление данных портфеля:', portfolioData);
            
            // Обновление баланса портфеля
            if (portfolioData.portfolio) {
                const totalBalance = document.getElementById('totalBalance');
                const availableBalance = document.getElementById('availableBalance');
                const unrealizedPnL = document.getElementById('unrealizedPnL');
                const realizedPnL = document.getElementById('realizedPnL');
                const equity = document.getElementById('equity');
                
                if (totalBalance) totalBalance.textContent = `$${portfolioData.portfolio.totalBalance.toFixed(2)}`;
                if (availableBalance) availableBalance.textContent = `$${portfolioData.portfolio.availableBalance.toFixed(2)}`;
                if (unrealizedPnL) {
                    unrealizedPnL.textContent = `$${portfolioData.portfolio.unrealizedPnL.toFixed(2)}`;
                    unrealizedPnL.className = portfolioData.portfolio.unrealizedPnL >= 0 ? 'positive' : 'negative';
                }
                if (realizedPnL) {
                    realizedPnL.textContent = `$${portfolioData.portfolio.realizedPnL.toFixed(2)}`;
                    realizedPnL.className = portfolioData.portfolio.realizedPnL >= 0 ? 'positive' : 'negative';
                }
                if (equity) equity.textContent = `$${portfolioData.portfolio.equity.toFixed(2)}`;
            }

            // Обновление рисковых метрик
            if (portfolioData.riskMetrics) {
                const currentDrawdown = document.getElementById('currentDrawdown');
                const exposurePercent = document.getElementById('exposurePercent');
                const leverageRatio = document.getElementById('leverageRatio');
                
                if (currentDrawdown) currentDrawdown.textContent = `${portfolioData.riskMetrics.currentDrawdown.toFixed(2)}%`;
                if (exposurePercent) exposurePercent.textContent = `${portfolioData.riskMetrics.exposurePercent.toFixed(1)}%`;
                if (leverageRatio) leverageRatio.textContent = portfolioData.riskMetrics.leverageRatio.toFixed(2);
            }

        } catch (error) {
            console.error('Error updating portfolio data:', error);
        }
    }

    updateMarketData(marketData) {
        try {
            console.log('📊 Обновление рыночных данных:', marketData);
            
            // Обновление цен активов
            const priceElements = document.querySelectorAll('[data-symbol]');
            priceElements.forEach(element => {
                const symbol = element.dataset.symbol;
                const price = marketData.get ? marketData.get(symbol) : marketData[symbol];
                if (price) {
                    element.textContent = `$${price.toFixed(2)}`;
                }
            });

            // Обновление графиков цен
            if (this.charts.priceChart) {
                const chartData = Array.from(marketData.entries ? marketData.entries() : Object.entries(marketData)).map(([symbol, price]) => ({
                    symbol,
                    price,
                    timestamp: new Date()
                }));
                this.updatePriceChart(chartData);
            }

        } catch (error) {
            console.error('Error updating market data:', error);
        }
    }

    updateMLData(mlData) {
        try {
            console.log('🤖 Обновление ML данных:', mlData);
            
            // Обновление статуса ML моделей
            const mlStatus = document.getElementById('mlStatus');
            if (mlStatus) {
                mlStatus.textContent = mlData.isTraining ? 'Обучение' : 'Готов';
                mlStatus.className = mlData.isTraining ? 'status-badge warning' : 'status-badge online';
            }

            // Обновление информации о моделях
            const modelsContainer = document.getElementById('mlModels');
            if (modelsContainer && mlData.models) {
                modelsContainer.innerHTML = '';
                mlData.models.forEach(model => {
                    const modelElement = this.createModelElement(model);
                    modelsContainer.appendChild(modelElement);
                });
            }

            // Обновление прогресса обучения
            if (mlData.isTraining) {
                const trainingProgress = document.getElementById('trainingProgress');
                if (trainingProgress) {
                    trainingProgress.style.width = `${mlData.trainingProgress || 0}%`;
                }
            }

        } catch (error) {
            console.error('Error updating ML data:', error);
        }
    }

    updateTradeHistory(tradeHistory) {
        try {
            const historyContainer = document.getElementById('tradeHistory');
            if (!historyContainer) return;

            historyContainer.innerHTML = '';
            
            tradeHistory.slice(-10).forEach(trade => {
                const tradeElement = document.createElement('div');
                tradeElement.className = 'trade-item';
                tradeElement.innerHTML = `
                    <div class="trade-symbol">${trade.symbol}</div>
                    <div class="trade-type ${trade.type}">${trade.type}</div>
                    <div class="trade-price">$${trade.price.toFixed(2)}</div>
                    <div class="trade-pnl ${trade.pnl >= 0 ? 'positive' : 'negative'}">
                        $${trade.pnl.toFixed(2)}
                    </div>
                    <div class="trade-time">${new Date(trade.timestamp).toLocaleTimeString()}</div>
                `;
                historyContainer.appendChild(tradeElement);
            });

        } catch (error) {
            console.error('Error updating trade history:', error);
        }
    }

    createModelElement(model) {
        const element = document.createElement('div');
        element.className = 'model-item';
        element.innerHTML = `
            <div class="model-header">
                <h4>${model.name}</h4>
                <span class="model-status ${model.status}">${model.status}</span>
            </div>
            <div class="model-metrics">
                <div class="metric">
                    <span class="label">Точность:</span>
                    <span class="value">${model.accuracy.toFixed(1)}%</span>
                </div>
                <div class="metric">
                    <span class="label">Версия:</span>
                    <span class="value">${model.version}</span>
                </div>
            </div>
        `;
        return element;
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
        console.log(`🔗 Обновление статуса подключения: ${isConnected ? 'Подключено' : 'Отключено'}`);
        
        // Обновляем все элементы статуса подключения
        const connectionElements = [
            document.querySelector('#connectionStatus .connection-dot'),
            document.querySelector('#connectionStatus span'),
            document.querySelector('.connection-indicator'),
            document.getElementById('connectionText'),
            document.querySelector('.status-dot'),
            document.querySelector('.connection-status span')
        ];
        
        connectionElements.forEach(element => {
            if (element) {
                if (isConnected) {
                    if (element.classList.contains('connection-dot') || element.classList.contains('status-dot')) {
                        element.className = element.className.replace(/offline|online/g, '') + ' online';
                    } else if (element.classList.contains('connection-indicator')) {
                        element.className = 'connection-indicator online';
                    } else if (element.tagName === 'SPAN') {
                        element.textContent = 'Подключено';
                    }
                } else {
                    if (element.classList.contains('connection-dot') || element.classList.contains('status-dot')) {
                        element.className = element.className.replace(/offline|online/g, '') + ' offline';
                    } else if (element.classList.contains('connection-indicator')) {
                        element.className = 'connection-indicator offline';
                    } else if (element.tagName === 'SPAN') {
                        element.textContent = 'Отключено';
                    }
                }
            }
        });
        
        // Обновляем текст статуса
        const statusTextElements = [
            document.getElementById('connectionText'),
            document.querySelector('.connection-status span')
        ];
        
        statusTextElements.forEach(element => {
            if (element) {
                element.textContent = isConnected ? 'Подключение: Активно' : 'Подключение: Отключено';
            }
        });
        
        this.isConnected = isConnected;
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