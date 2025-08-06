// ============================================================================
// ATB Trading System - Trading Manager JavaScript
// Управление торговлей в Electron приложении
// ============================================================================

class TradingUI {
    constructor() {
        this.isRunning = false;
        this.currentMode = 'simulation';
        this.equityChart = null;
        this.selectedStrategy = null;
        
        this.initializeElements();
        this.setupEventListeners();
        this.initializeCharts();
    }

    initializeElements() {
        // Основные контролы
        this.tradingModeSelect = document.getElementById('tradingModeSelect');
        this.startTradingBtn = document.getElementById('startTradingBtn');
        this.stopTradingBtn = document.getElementById('stopTradingBtn');
        this.runBacktestBtn = document.getElementById('runBacktestBtn');
        this.closeAllPositionsBtn = document.getElementById('closeAllPositionsBtn');

        // Элементы портфеля
        this.totalEquityEl = document.getElementById('totalEquity');
        this.unrealizedPnLEl = document.getElementById('unrealizedPnL');
        this.realizedPnLEl = document.getElementById('realizedPnL');
        this.marginLevelEl = document.getElementById('marginLevel');
        this.currentDrawdownEl = document.getElementById('currentDrawdown');
        this.winRateEl = document.getElementById('winRate');

        // Элементы метрик производительности
        this.totalReturnEl = document.getElementById('totalReturn');
        this.sharpeRatioEl = document.getElementById('sharpeRatio');
        this.maxDrawdownEl = document.getElementById('maxDrawdown');
        this.profitFactorEl = document.getElementById('profitFactor');
        this.volatilityEl = document.getElementById('volatility');
        this.calmarRatioEl = document.getElementById('calmarRatio');

        // Риск-метрики
        this.valueAtRiskEl = document.getElementById('valueAtRisk');
        this.exposurePercentEl = document.getElementById('exposurePercent');
        this.leverageRatioEl = document.getElementById('leverageRatio');
        this.riskRewardRatioEl = document.getElementById('riskRewardRatio');

        // Списки и таблицы
        this.tradingStrategiesListEl = document.getElementById('tradingStrategiesList');
        this.marketDataListEl = document.getElementById('marketDataList');
        this.openPositionsTableEl = document.getElementById('openPositionsTable');
        this.recentTradesTableEl = document.getElementById('recentTradesTable');

        // Фильтры
        this.strategiesFilterEl = document.getElementById('strategiesFilter');
        this.equityPeriodEl = document.getElementById('equityPeriod');
    }

    setupEventListeners() {
        // Основные контролы торговли
        if (this.startTradingBtn) {
            this.startTradingBtn.addEventListener('click', () => this.startTrading());
        }

        if (this.stopTradingBtn) {
            this.stopTradingBtn.addEventListener('click', () => this.stopTrading());
        }

        if (this.runBacktestBtn) {
            this.runBacktestBtn.addEventListener('click', () => this.runBacktest());
        }

        if (this.closeAllPositionsBtn) {
            this.closeAllPositionsBtn.addEventListener('click', () => this.closeAllPositions());
        }

        // Смена режима торговли
        if (this.tradingModeSelect) {
            this.tradingModeSelect.addEventListener('change', (e) => {
                this.currentMode = e.target.value;
            });
        }

        // Фильтры
        if (this.strategiesFilterEl) {
            this.strategiesFilterEl.addEventListener('change', () => this.filterStrategies());
        }

        if (this.equityPeriodEl) {
            this.equityPeriodEl.addEventListener('change', () => this.updateEquityChart());
        }
    }

    initializeCharts() {
        // График эквити
        const equityCanvas = document.getElementById('equityChart');
        if (equityCanvas) {
            const ctx = equityCanvas.getContext('2d');
            
            this.equityChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Эквити',
                        data: [],
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0, 255, 136, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }, {
                        label: 'Просадка',
                        data: [],
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        borderWidth: 1,
                        fill: false,
                        yAxisID: 'y1'
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
                            beginAtZero: false,
                            ticks: { 
                                color: '#ccc',
                                callback: function(value) {
                                    return '$' + value.toFixed(0);
                                }
                            },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            ticks: { 
                                color: '#ff6b6b',
                                callback: function(value) {
                                    return value.toFixed(1) + '%';
                                }
                            },
                            grid: {
                                drawOnChartArea: false,
                            },
                        }
                    }
                }
            });
        }
    }

    async updateTradingData() {
        try {
            if (!window.electronAPI) {
                this.loadDemoTradingData();
                return;
            }

            const tradingStatus = await window.electronAPI.getTradingStatus();
            
            if (tradingStatus && tradingStatus.success) {
                this.updatePortfolioMetrics(tradingStatus.portfolio);
                this.updatePerformanceMetrics(tradingStatus.performance);
                this.updateRiskMetrics(tradingStatus.riskMetrics);
                this.updateStrategiesList(tradingStatus.strategies);
                this.updateMarketData(tradingStatus.marketData);
                this.updatePositionsTable(tradingStatus.openPositions);
                this.updateTradesTable(tradingStatus.recentTrades);
                this.updateTradingControls(tradingStatus.running, tradingStatus.mode);
                this.updateEquityChart(tradingStatus.portfolioHistory);
            }

        } catch (error) {
            console.error('❌ Error updating trading data:', error);
            this.loadDemoTradingData();
        }
    }

    loadDemoTradingData() {
        // Демо данные для торговли
        const demoPortfolio = {
            totalBalance: 10000.0,
            equity: 10456.78,
            unrealizedPnL: 234.12,
            realizedPnL: 222.66,
            margin: 1500.0,
            marginLevel: 696.45,
            freeMargin: 8956.78
        };

        const demoPerformance = {
            totalReturn: 4.57,
            sharpeRatio: 1.23,
            maxDrawdown: 8.5,
            winRate: 68.2,
            profitFactor: 1.85,
            volatility: 12.3,
            calmarRatio: 0.54
        };

        const demoRiskMetrics = {
            currentDrawdown: 2.1,
            valueAtRisk: 450.23,
            exposurePercent: 15.0,
            leverageRatio: 1.15,
            riskRewardRatio: 1.8
        };

        const demoStrategies = [
            {
                id: 'trend_master',
                name: 'Trend Master Pro',
                type: 'trend_following',
                status: 'active',
                allocation: 30,
                performance: {
                    totalReturn: 15.8,
                    currentPnL: 1847.32
                }
            },
            {
                id: 'scalper_ai',
                name: 'AI Scalper Elite',
                type: 'scalping',
                status: 'active',
                allocation: 25,
                performance: {
                    totalReturn: 22.4,
                    currentPnL: 2240.15
                }
            }
        ];

        const demoMarketData = [
            { symbol: 'BTCUSDT', price: 43256.78, change24h: 2.34, volume24h: 1234567890 },
            { symbol: 'ETHUSDT', price: 2634.12, change24h: -1.23, volume24h: 987654321 },
            { symbol: 'BNBUSDT', price: 298.45, change24h: 0.87, volume24h: 456789123 }
        ];

        const demoPositions = [
            {
                symbol: 'BTCUSDT',
                side: 'long',
                size: 0.25,
                entryPrice: 42800.0,
                currentPrice: 43256.78,
                unrealizedPnL: 114.20,
                returnPercent: 1.07
            },
            {
                symbol: 'ETHUSDT',
                side: 'short',
                size: 1.5,
                entryPrice: 2650.0,
                currentPrice: 2634.12,
                unrealizedPnL: 23.82,
                returnPercent: 0.60
            }
        ];

        const demoTrades = [
            {
                symbol: 'BTCUSDT',
                side: 'long',
                size: 0.1,
                realizedPnL: 156.78,
                closeTime: new Date(Date.now() - 300000).toISOString()
            },
            {
                symbol: 'ETHUSDT',
                side: 'short',
                size: 0.8,
                realizedPnL: -89.45,
                closeTime: new Date(Date.now() - 600000).toISOString()
            }
        ];

        this.updatePortfolioMetrics(demoPortfolio);
        this.updatePerformanceMetrics(demoPerformance);
        this.updateRiskMetrics(demoRiskMetrics);
        this.updateStrategiesList(demoStrategies);
        this.updateMarketData(demoMarketData);
        this.updatePositionsTable(demoPositions);
        this.updateTradesTable(demoTrades);
        this.updateTradingControls(this.isRunning, this.currentMode);
        this.generateDemoEquityData();
    }

    updatePortfolioMetrics(portfolio) {
        if (this.totalEquityEl) {
            this.totalEquityEl.textContent = `$${portfolio.equity.toLocaleString('en-US', {minimumFractionDigits: 2})}`;
        }
        
        if (this.unrealizedPnLEl) {
            const unrealizedPnL = portfolio.unrealizedPnL || 0;
            this.unrealizedPnLEl.textContent = `$${unrealizedPnL.toFixed(2)}`;
            this.unrealizedPnLEl.className = `portfolio-value ${unrealizedPnL >= 0 ? 'positive' : 'negative'}`;
        }
        
        if (this.realizedPnLEl) {
            const realizedPnL = portfolio.realizedPnL || 0;
            this.realizedPnLEl.textContent = `$${realizedPnL.toFixed(2)}`;
            this.realizedPnLEl.className = `portfolio-value ${realizedPnL >= 0 ? 'positive' : 'negative'}`;
        }
        
        if (this.marginLevelEl) {
            this.marginLevelEl.textContent = `${(portfolio.marginLevel || 0).toFixed(1)}%`;
        }
    }

    updatePerformanceMetrics(performance) {
        if (this.totalReturnEl) {
            const totalReturn = performance.totalReturn || 0;
            this.totalReturnEl.textContent = `${totalReturn.toFixed(2)}%`;
            this.totalReturnEl.className = `metric-value ${totalReturn >= 0 ? 'positive' : 'negative'}`;
        }
        
        if (this.sharpeRatioEl) {
            this.sharpeRatioEl.textContent = (performance.sharpeRatio || 0).toFixed(2);
        }
        
        if (this.maxDrawdownEl) {
            this.maxDrawdownEl.textContent = `${(performance.maxDrawdown || 0).toFixed(1)}%`;
        }
        
        if (this.profitFactorEl) {
            this.profitFactorEl.textContent = (performance.profitFactor || 0).toFixed(2);
        }
        
        if (this.volatilityEl) {
            this.volatilityEl.textContent = `${(performance.volatility || 0).toFixed(1)}%`;
        }
        
        if (this.calmarRatioEl) {
            this.calmarRatioEl.textContent = (performance.calmarRatio || 0).toFixed(2);
        }
        
        if (this.winRateEl) {
            this.winRateEl.textContent = `${(performance.winRate || 0).toFixed(1)}%`;
        }
        
        if (this.currentDrawdownEl) {
            const drawdown = performance.currentDrawdown || 0;
            this.currentDrawdownEl.textContent = `${drawdown.toFixed(1)}%`;
        }
    }

    updateRiskMetrics(riskMetrics) {
        if (this.valueAtRiskEl) {
            this.valueAtRiskEl.textContent = `$${(riskMetrics.valueAtRisk || 0).toFixed(0)}`;
        }
        
        if (this.exposurePercentEl) {
            this.exposurePercentEl.textContent = `${(riskMetrics.exposurePercent || 0).toFixed(1)}%`;
        }
        
        if (this.leverageRatioEl) {
            this.leverageRatioEl.textContent = `${(riskMetrics.leverageRatio || 1).toFixed(2)}x`;
        }
        
        if (this.riskRewardRatioEl) {
            this.riskRewardRatioEl.textContent = (riskMetrics.riskRewardRatio || 0).toFixed(1);
        }
    }

    updateStrategiesList(strategies) {
        if (!this.tradingStrategiesListEl || !strategies) return;

        this.tradingStrategiesListEl.innerHTML = '';

        strategies.forEach(strategy => {
            const strategyEl = this.createStrategyElement(strategy);
            this.tradingStrategiesListEl.appendChild(strategyEl);
        });
    }

    createStrategyElement(strategy) {
        const element = document.createElement('div');
        element.className = 'trading-strategy-item';
        element.dataset.strategyId = strategy.id;

        const statusClass = strategy.status || 'paused';
        const pnlClass = strategy.performance.currentPnL >= 0 ? 'positive' : 'negative';

        element.innerHTML = `
            <div class="strategy-trading-header">
                <div class="strategy-trading-name">${strategy.name}</div>
                <div class="strategy-trading-status ${statusClass}">${strategy.status}</div>
            </div>
            <div class="strategy-trading-metrics">
                <div class="strategy-trading-metric">
                    <span>Аллокация:</span>
                    <span class="value">${strategy.allocation}%</span>
                </div>
                <div class="strategy-trading-metric">
                    <span>Доходность:</span>
                    <span class="value">${strategy.performance.totalReturn.toFixed(1)}%</span>
                </div>
                <div class="strategy-trading-metric">
                    <span>P&L:</span>
                    <span class="value ${pnlClass}">$${strategy.performance.currentPnL.toFixed(2)}</span>
                </div>
            </div>
        `;

        element.addEventListener('click', () => this.selectStrategy(strategy));

        return element;
    }

    updateMarketData(marketData) {
        if (!this.marketDataListEl || !marketData) return;

        this.marketDataListEl.innerHTML = '';

        marketData.forEach(data => {
            const marketEl = this.createMarketDataElement(data);
            this.marketDataListEl.appendChild(marketEl);
        });
    }

    createMarketDataElement(data) {
        const element = document.createElement('div');
        element.className = 'market-data-item';

        const changeClass = data.change24h >= 0 ? 'positive' : 'negative';

        element.innerHTML = `
            <div class="market-data-header">
                <div class="market-data-symbol">${data.symbol}</div>
                <div class="market-data-change ${changeClass}">
                    ${data.change24h >= 0 ? '+' : ''}${data.change24h.toFixed(2)}%
                </div>
            </div>
            <div class="market-data-price">$${data.price.toLocaleString('en-US', {minimumFractionDigits: 2})}</div>
            <div class="market-data-volume">
                Vol: ${(data.volume24h / 1000000).toFixed(1)}M
            </div>
        `;

        return element;
    }

    updatePositionsTable(positions) {
        if (!this.openPositionsTableEl || !positions) return;

        const tbody = this.openPositionsTableEl.querySelector('tbody');
        if (!tbody) return;

        tbody.innerHTML = '';

        if (positions.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" class="no-data">Нет открытых позиций</td></tr>';
            return;
        }

        positions.forEach(position => {
            const row = document.createElement('tr');
            const pnlClass = position.unrealizedPnL >= 0 ? 'positive' : 'negative';

            row.innerHTML = `
                <td>${position.symbol}</td>
                <td>
                    <span class="side-badge ${position.side}">${position.side.toUpperCase()}</span>
                </td>
                <td>${position.size}</td>
                <td class="${pnlClass}">
                    $${position.unrealizedPnL.toFixed(2)}
                    <small>(${position.returnPercent.toFixed(2)}%)</small>
                </td>
            `;

            tbody.appendChild(row);
        });
    }

    updateTradesTable(trades) {
        if (!this.recentTradesTableEl || !trades) return;

        const tbody = this.recentTradesTableEl.querySelector('tbody');
        if (!tbody) return;

        tbody.innerHTML = '';

        if (trades.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" class="no-data">Нет сделок</td></tr>';
            return;
        }

        trades.forEach(trade => {
            const row = document.createElement('tr');
            const pnlClass = trade.realizedPnL >= 0 ? 'positive' : 'negative';
            const time = new Date(trade.closeTime).toLocaleTimeString();

            row.innerHTML = `
                <td>${time}</td>
                <td>${trade.symbol}</td>
                <td>
                    <span class="side-badge ${trade.side}">${trade.side.toUpperCase()}</span>
                </td>
                <td class="${pnlClass}">$${trade.realizedPnL.toFixed(2)}</td>
            `;

            tbody.appendChild(row);
        });
    }

    generateDemoEquityData() {
        if (!this.equityChart) return;

        const labels = [];
        const equityData = [];
        const drawdownData = [];
        
        let currentEquity = 10000;
        let peak = 10000;

        for (let i = 0; i < 100; i++) {
            const time = new Date(Date.now() - (100 - i) * 60 * 60 * 1000);
            labels.push(time.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }));
            
            // Симуляция изменений эквити
            const change = (Math.random() - 0.48) * 100;
            currentEquity += change;
            equityData.push(currentEquity);
            
            // Расчет просадки
            if (currentEquity > peak) peak = currentEquity;
            const drawdown = ((peak - currentEquity) / peak) * 100;
            drawdownData.push(drawdown);
        }

        this.equityChart.data.labels = labels.slice(-50); // Последние 50 точек
        this.equityChart.data.datasets[0].data = equityData.slice(-50);
        this.equityChart.data.datasets[1].data = drawdownData.slice(-50);
        this.equityChart.update();
    }

    updateEquityChart(portfolioHistory) {
        if (!this.equityChart) return;

        if (portfolioHistory && portfolioHistory.length > 0) {
            const labels = portfolioHistory.map(point => {
                const time = new Date(point.timestamp);
                return time.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' });
            });

            const equityData = portfolioHistory.map(point => point.equity);
            const drawdownData = portfolioHistory.map(point => {
                // Рассчитываем просадку для каждой точки
                const maxEquity = Math.max(...equityData.slice(0, equityData.indexOf(point.equity) + 1));
                return ((maxEquity - point.equity) / maxEquity) * 100;
            });

            this.equityChart.data.labels = labels;
            this.equityChart.data.datasets[0].data = equityData;
            this.equityChart.data.datasets[1].data = drawdownData;
            this.equityChart.update();
        } else {
            this.generateDemoEquityData();
        }
    }

    updateTradingControls(isRunning, mode) {
        this.isRunning = isRunning;

        if (this.startTradingBtn) {
            this.startTradingBtn.disabled = isRunning;
            this.startTradingBtn.innerHTML = isRunning ? 
                '<i class="fas fa-spinner fa-spin"></i><span>Запущено</span>' :
                '<i class="fas fa-play"></i><span>Запустить торговлю</span>';
        }

        if (this.stopTradingBtn) {
            this.stopTradingBtn.disabled = !isRunning;
        }

        if (this.tradingModeSelect && mode) {
            this.tradingModeSelect.value = mode;
            this.currentMode = mode;
        }

        // Обновление статуса в header
        const statusEl = document.querySelector('#systemStatus');
        if (statusEl) {
            if (isRunning) {
                statusEl.querySelector('.status-dot').className = 'status-dot online';
                statusEl.querySelector('span').textContent = `Торговля активна (${mode})`;
            } else {
                statusEl.querySelector('.status-dot').className = 'status-dot offline';
                statusEl.querySelector('span').textContent = 'Торговля остановлена';
            }
        }
    }

    async startTrading() {
        try {
            if (window.electronAPI) {
                const result = await window.electronAPI.startTrading(this.currentMode);
                if (result.success) {
                    this.showNotification('Торговля', `Торговля запущена в режиме ${this.currentMode}`);
                    this.isRunning = true;
                    this.updateTradingControls(true, this.currentMode);
                } else {
                    this.showError('Ошибка', result.message || 'Не удалось запустить торговлю');
                }
            } else {
                // Демо режим
                this.isRunning = true;
                this.updateTradingControls(true, this.currentMode);
                this.showNotification('Торговля', `Торговля запущена в режиме ${this.currentMode} (демо режим)`);
            }
        } catch (error) {
            this.showError('Ошибка', error.message);
        }
    }

    async stopTrading() {
        try {
            if (window.electronAPI) {
                const result = await window.electronAPI.stopTrading();
                if (result.success) {
                    this.showNotification('Торговля', 'Торговля остановлена');
                    this.isRunning = false;
                    this.updateTradingControls(false, this.currentMode);
                } else {
                    this.showError('Ошибка', result.message || 'Не удалось остановить торговлю');
                }
            } else {
                // Демо режим
                this.isRunning = false;
                this.updateTradingControls(false, this.currentMode);
                this.showNotification('Торговля', 'Торговля остановлена (демо режим)');
            }
        } catch (error) {
            this.showError('Ошибка', error.message);
        }
    }

    async runBacktest() {
        try {
            const parameters = {
                startDate: '2024-01-01',
                endDate: new Date().toISOString().split('T')[0],
                initialCapital: 10000,
                days: 90
            };

            this.showNotification('Бэктест', 'Запуск бэктеста...');

            if (window.electronAPI) {
                const result = await window.electronAPI.runBacktest(parameters);
                if (result.success) {
                    this.displayBacktestResults(result);
                } else {
                    this.showError('Ошибка', result.message || 'Ошибка бэктеста');
                }
            } else {
                // Демо бэктест
                setTimeout(() => {
                    this.displayDemoBacktestResults();
                }, 3000);
            }
        } catch (error) {
            this.showError('Ошибка', error.message);
        }
    }

    displayBacktestResults(result) {
        const modal = this.createBacktestModal(result);
        document.body.appendChild(modal);
    }

    displayDemoBacktestResults() {
        const demoResult = {
            success: true,
            startDate: '2024-01-01',
            endDate: new Date().toISOString().split('T')[0],
            initialCapital: 10000,
            results: {
                finalCapital: 11567.89,
                totalReturn: 15.68,
                maxDrawdown: 8.5,
                sharpeRatio: 1.23,
                winRate: 67.8,
                totalTrades: 156,
                profitFactor: 1.85,
                calmarRatio: 1.84,
                sortinoRatio: 1.67
            }
        };

        this.displayBacktestResults(demoResult);
    }

    createBacktestModal(result) {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content backtest-modal">
                <div class="modal-header">
                    <h3>📊 Результаты бэктеста</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="backtest-summary">
                        <div class="backtest-metric">
                            <span class="metric-label">Период:</span>
                            <span class="metric-value">${result.startDate} - ${result.endDate}</span>
                        </div>
                        <div class="backtest-metric">
                            <span class="metric-label">Начальный капитал:</span>
                            <span class="metric-value">$${result.initialCapital.toLocaleString()}</span>
                        </div>
                        <div class="backtest-metric">
                            <span class="metric-label">Финальный капитал:</span>
                            <span class="metric-value">$${result.results.finalCapital.toLocaleString()}</span>
                        </div>
                    </div>
                    <div class="backtest-metrics-grid">
                        <div class="backtest-metric-card">
                            <div class="metric-name">Total Return</div>
                            <div class="metric-value ${result.results.totalReturn >= 0 ? 'positive' : 'negative'}">
                                ${result.results.totalReturn.toFixed(2)}%
                            </div>
                        </div>
                        <div class="backtest-metric-card">
                            <div class="metric-name">Sharpe Ratio</div>
                            <div class="metric-value">${result.results.sharpeRatio.toFixed(2)}</div>
                        </div>
                        <div class="backtest-metric-card">
                            <div class="metric-name">Max Drawdown</div>
                            <div class="metric-value">${result.results.maxDrawdown.toFixed(2)}%</div>
                        </div>
                        <div class="backtest-metric-card">
                            <div class="metric-name">Win Rate</div>
                            <div class="metric-value">${result.results.winRate.toFixed(1)}%</div>
                        </div>
                        <div class="backtest-metric-card">
                            <div class="metric-name">Total Trades</div>
                            <div class="metric-value">${result.results.totalTrades}</div>
                        </div>
                        <div class="backtest-metric-card">
                            <div class="metric-name">Profit Factor</div>
                            <div class="metric-value">${result.results.profitFactor.toFixed(2)}</div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        modal.querySelector('.modal-close').addEventListener('click', () => {
            document.body.removeChild(modal);
        });

        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                document.body.removeChild(modal);
            }
        });

        return modal;
    }

    async closeAllPositions() {
        try {
            if (window.electronAPI) {
                const result = await window.electronAPI.closeAllPositions();
                if (result.success) {
                    this.showNotification('Позиции', `Закрыто ${result.closedPositions.length} позиций`);
                    this.updateTradingData(); // Обновить данные
                } else {
                    this.showError('Ошибка', result.message || 'Не удалось закрыть позиции');
                }
            } else {
                // Демо режим
                this.showNotification('Позиции', 'Все позиции закрыты (демо режим)');
                this.updateTradingData();
            }
        } catch (error) {
            this.showError('Ошибка', error.message);
        }
    }

    selectStrategy(strategy) {
        // Снятие выделения с предыдущей стратегии
        const previousSelected = this.tradingStrategiesListEl.querySelector('.trading-strategy-item.selected');
        if (previousSelected) {
            previousSelected.classList.remove('selected');
        }

        // Выделение новой стратегии
        const newSelected = this.tradingStrategiesListEl.querySelector(`[data-strategy-id="${strategy.id}"]`);
        if (newSelected) {
            newSelected.classList.add('selected');
        }

        this.selectedStrategy = strategy;
        console.log('Selected strategy:', strategy.name);
    }

    filterStrategies() {
        const filter = this.strategiesFilterEl?.value || 'all';
        const strategies = this.tradingStrategiesListEl?.querySelectorAll('.trading-strategy-item');

        if (!strategies) return;

        strategies.forEach(strategy => {
            const status = strategy.querySelector('.strategy-trading-status').textContent.trim();
            
            if (filter === 'all') {
                strategy.style.display = 'block';
            } else {
                strategy.style.display = status === filter ? 'block' : 'none';
            }
        });
    }

    showNotification(title, message) {
        console.log(`🔔 ${title}: ${message}`);
        if (window.atbApp) {
            window.atbApp.showNotification(title, message);
        }
    }

    showError(title, message) {
        console.error(`❌ ${title}: ${message}`);
        if (window.atbApp) {
            window.atbApp.showError(title, message);
        }
    }
}

// Инициализация Trading UI
let tradingUI = null;

document.addEventListener('DOMContentLoaded', () => {
    tradingUI = new TradingUI();
    
    // Первоначальная загрузка данных
    setTimeout(() => {
        if (tradingUI) {
            tradingUI.updateTradingData();
        }
    }, 1500);

    // Периодическое обновление данных торговли
    setInterval(() => {
        if (tradingUI) {
            tradingUI.updateTradingData();
        }
    }, 5000); // Каждые 5 секунд
});

// Экспорт для использования в других модулях
if (typeof window !== 'undefined') {
    window.tradingUI = tradingUI;
}

console.log('⚡ Trading UI module loaded');