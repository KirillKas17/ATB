// ============================================================================
// ATB Trading System - Evolution Manager JavaScript
// Управление эволюцией стратегий в Electron приложении
// ============================================================================

class EvolutionUI {
    constructor() {
        this.selectedStrategy = null;
        this.evolutionChart = null;
        this.isEvolutionRunning = false;
        
        this.initializeElements();
        this.setupEventListeners();
        this.initializeChart();
    }

    initializeElements() {
        // Кнопки управления
        this.evolutionStartBtn = document.getElementById('evolutionStartBtn');
        this.evolutionStopBtn = document.getElementById('evolutionStopBtn');
        this.forceEvolutionBtn = document.getElementById('forceEvolutionBtn');
        this.evolveStrategyBtn = document.getElementById('evolveStrategyBtn');
        this.saveEvolutionConfigBtn = document.getElementById('saveEvolutionConfig');

        // Статистика
        this.totalStrategiesEl = document.getElementById('totalStrategies');
        this.activeStrategiesEl = document.getElementById('activeStrategies');
        this.totalEvolutionsEl = document.getElementById('totalEvolutions');
        this.avgPerformanceEl = document.getElementById('avgPerformance');

        // Списки и контейнеры
        this.strategiesListEl = document.getElementById('strategiesEvolutionList');
        this.strategyDetailsEl = document.getElementById('strategyDetails');
        this.evolutionHistoryEl = document.getElementById('evolutionHistory');
        this.selectedStrategyNameEl = document.getElementById('selectedStrategyName');

        // Настройки
        this.evolutionEnabledEl = document.getElementById('evolutionEnabled');
        this.evolutionIntervalEl = document.getElementById('evolutionInterval');
        this.mutationRateEl = document.getElementById('mutationRate');
        this.mutationRateValueEl = document.getElementById('mutationRateValue');
        this.autoEvolutionEl = document.getElementById('autoEvolution');
        this.strategyFilterEl = document.getElementById('strategyFilter');

        // Топ стратегии
        this.bestStrategyEl = document.getElementById('bestStrategy');
        this.worstStrategyEl = document.getElementById('worstStrategy');
    }

    setupEventListeners() {
        // Кнопки управления эволюцией
        if (this.evolutionStartBtn) {
            this.evolutionStartBtn.addEventListener('click', () => this.startEvolution());
        }

        if (this.evolutionStopBtn) {
            this.evolutionStopBtn.addEventListener('click', () => this.stopEvolution());
        }

        if (this.forceEvolutionBtn) {
            this.forceEvolutionBtn.addEventListener('click', () => this.forceEvolution());
        }

        if (this.evolveStrategyBtn) {
            this.evolveStrategyBtn.addEventListener('click', () => this.evolveSelectedStrategy());
        }

        if (this.saveEvolutionConfigBtn) {
            this.saveEvolutionConfigBtn.addEventListener('click', () => this.saveConfig());
        }

        // Настройки
        if (this.mutationRateEl) {
            this.mutationRateEl.addEventListener('input', (e) => {
                const value = Math.round(e.target.value * 100);
                this.mutationRateValueEl.textContent = `${value}%`;
            });
        }

        if (this.strategyFilterEl) {
            this.strategyFilterEl.addEventListener('change', () => this.filterStrategies());
        }
    }

    initializeChart() {
        const canvas = document.getElementById('evolutionChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        
        this.evolutionChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Fitness Score',
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

    async updateEvolutionData() {
        try {
            if (!window.electronAPI) {
                this.loadDemoEvolutionData();
                return;
            }

            const evolutionStatus = await window.electronAPI.getEvolutionStatus();
            
            if (evolutionStatus && evolutionStatus.success) {
                this.updateEvolutionStats(evolutionStatus);
                this.updateStrategiesList(evolutionStatus.strategies);
                this.updateEvolutionHistory(evolutionStatus.history);
                this.updateTopStrategies(evolutionStatus.statistics);
                this.updateEvolutionControls(evolutionStatus.running);
            }

        } catch (error) {
            console.error('❌ Error updating evolution data:', error);
            this.loadDemoEvolutionData();
        }
    }

    loadDemoEvolutionData() {
        // Демо данные для эволюции
        const demoData = {
            success: true,
            running: this.isEvolutionRunning,
            statistics: {
                total_strategies: 4,
                active_strategies: 3,
                total_evolutions: 127,
                average_performance: 1245.67,
                best_strategy: {
                    name: 'Trend Following Strategy',
                    performance: 2340.52,
                    win_rate: 67.8
                },
                worst_strategy: {
                    name: 'Scalping Strategy',
                    performance: -456.23,
                    win_rate: 34.2
                }
            },
            strategies: [
                {
                    id: 'trend_following',
                    name: 'Trend Following Strategy',
                    type: 'trend',
                    status: 'excellent',
                    performance: {
                        win_rate: 67.8,
                        profit_loss: 2340.52,
                        total_trades: 156,
                        sharpe_ratio: 1.23
                    },
                    evolution: {
                        generation: 23,
                        fitness_score: 87.4,
                        evolution_count: 45,
                        last_evolution: new Date(Date.now() - 3600000).toISOString()
                    }
                },
                {
                    id: 'mean_reversion',
                    name: 'Mean Reversion Strategy',
                    type: 'reversion',
                    status: 'good',
                    performance: {
                        win_rate: 58.3,
                        profit_loss: 1234.67,
                        total_trades: 203,
                        sharpe_ratio: 0.89
                    },
                    evolution: {
                        generation: 18,
                        fitness_score: 74.2,
                        evolution_count: 32,
                        last_evolution: new Date(Date.now() - 7200000).toISOString()
                    }
                },
                {
                    id: 'scalping',
                    name: 'Scalping Strategy',
                    type: 'scalping',
                    status: 'poor',
                    performance: {
                        win_rate: 34.2,
                        profit_loss: -456.23,
                        total_trades: 89,
                        sharpe_ratio: -0.34
                    },
                    evolution: {
                        generation: 15,
                        fitness_score: 42.1,
                        evolution_count: 28,
                        last_evolution: new Date(Date.now() - 10800000).toISOString()
                    }
                },
                {
                    id: 'ml_strategy',
                    name: 'Machine Learning Strategy',
                    type: 'ml',
                    status: 'average',
                    performance: {
                        win_rate: 52.1,
                        profit_loss: 567.89,
                        total_trades: 134,
                        sharpe_ratio: 0.45
                    },
                    evolution: {
                        generation: 12,
                        fitness_score: 65.8,
                        evolution_count: 22,
                        last_evolution: new Date(Date.now() - 5400000).toISOString()
                    }
                }
            ],
            history: [
                {
                    timestamp: new Date(Date.now() - 1800000).toISOString(),
                    type: 'strategy_evolved',
                    message: 'Strategy Trend Following evolved. Performance change: +45.23'
                },
                {
                    timestamp: new Date(Date.now() - 3600000).toISOString(),
                    type: 'evolution_started',
                    message: 'Strategy evolution process started'
                }
            ]
        };

        this.updateEvolutionStats(demoData);
        this.updateStrategiesList(demoData.strategies);
        this.updateEvolutionHistory(demoData.history);
        this.updateTopStrategies(demoData.statistics);
        this.updateEvolutionControls(demoData.running);
    }

    updateEvolutionStats(data) {
        if (this.totalStrategiesEl) {
            this.totalStrategiesEl.textContent = data.statistics?.total_strategies || 0;
        }
        
        if (this.activeStrategiesEl) {
            this.activeStrategiesEl.textContent = data.statistics?.active_strategies || 0;
        }
        
        if (this.totalEvolutionsEl) {
            this.totalEvolutionsEl.textContent = data.statistics?.total_evolutions || 0;
        }
        
        if (this.avgPerformanceEl) {
            const avgPerf = data.statistics?.average_performance || 0;
            this.avgPerformanceEl.textContent = `$${avgPerf.toFixed(2)}`;
        }
    }

    updateStrategiesList(strategies) {
        if (!this.strategiesListEl || !strategies) return;

        this.strategiesListEl.innerHTML = '';

        strategies.forEach(strategy => {
            const strategyEl = this.createStrategyElement(strategy);
            this.strategiesListEl.appendChild(strategyEl);
        });
    }

    createStrategyElement(strategy) {
        const element = document.createElement('div');
        element.className = 'strategy-evolution-item';
        element.dataset.strategyId = strategy.id;

        const statusClass = strategy.status || 'average';
        const profitClass = strategy.performance.profit_loss >= 0 ? 'positive' : 'negative';

        element.innerHTML = `
            <div class="strategy-evolution-status ${statusClass}"></div>
            <div class="strategy-evolution-header">
                <div class="strategy-evolution-name">${strategy.name}</div>
                <div class="strategy-evolution-type ${strategy.type}">${strategy.type}</div>
            </div>
            <div class="strategy-evolution-metrics">
                <div class="strategy-evolution-metric">
                    <span>Win Rate:</span>
                    <span class="value">${strategy.performance.win_rate.toFixed(1)}%</span>
                </div>
                <div class="strategy-evolution-metric">
                    <span>P&L:</span>
                    <span class="value ${profitClass}">$${strategy.performance.profit_loss.toFixed(2)}</span>
                </div>
                <div class="strategy-evolution-metric">
                    <span>Generation:</span>
                    <span class="value">${strategy.evolution.generation}</span>
                </div>
                <div class="strategy-evolution-metric">
                    <span>Fitness:</span>
                    <span class="value">${strategy.evolution.fitness_score.toFixed(1)}%</span>
                </div>
            </div>
        `;

        element.addEventListener('click', () => this.selectStrategy(strategy));

        return element;
    }

    selectStrategy(strategy) {
        // Снятие выделения с предыдущей стратегии
        const previousSelected = this.strategiesListEl.querySelector('.strategy-evolution-item.selected');
        if (previousSelected) {
            previousSelected.classList.remove('selected');
        }

        // Выделение новой стратегии
        const newSelected = this.strategiesListEl.querySelector(`[data-strategy-id="${strategy.id}"]`);
        if (newSelected) {
            newSelected.classList.add('selected');
        }

        this.selectedStrategy = strategy;
        this.updateStrategyDetails(strategy);
        this.updateEvolutionChart(strategy);

        // Активация кнопки эволюции
        if (this.evolveStrategyBtn) {
            this.evolveStrategyBtn.disabled = false;
        }

        if (this.selectedStrategyNameEl) {
            this.selectedStrategyNameEl.textContent = `🎯 ${strategy.name}`;
        }
    }

    updateStrategyDetails(strategy) {
        if (!this.strategyDetailsEl || !strategy) return;

        const statusBadgeClass = this.getStatusBadgeClass(strategy.status);
        
        this.strategyDetailsEl.innerHTML = `
            <div class="strategy-detail-content">
                <div class="strategy-detail-header">
                    <div class="strategy-detail-name">${strategy.name}</div>
                    <div class="strategy-detail-badge ${statusBadgeClass}">${strategy.status}</div>
                </div>
                
                <div class="strategy-detail-performance">
                    <div class="performance-metric">
                        <div class="performance-value">${strategy.performance.win_rate.toFixed(1)}%</div>
                        <div class="performance-label">Win Rate</div>
                    </div>
                    <div class="performance-metric">
                        <div class="performance-value">$${strategy.performance.profit_loss.toFixed(2)}</div>
                        <div class="performance-label">P&L</div>
                    </div>
                    <div class="performance-metric">
                        <div class="performance-value">${strategy.performance.total_trades}</div>
                        <div class="performance-label">Trades</div>
                    </div>
                    <div class="performance-metric">
                        <div class="performance-value">${strategy.performance.sharpe_ratio.toFixed(2)}</div>
                        <div class="performance-label">Sharpe</div>
                    </div>
                </div>

                <div class="strategy-parameters">
                    <h4>Параметры стратегии</h4>
                    <div class="parameters-grid" id="parametersGrid">
                        <!-- Параметры будут загружены динамически -->
                    </div>
                </div>

                <div class="evolution-info">
                    <h4>Информация об эволюции</h4>
                    <div class="evolution-metrics">
                        <div class="evolution-metric">
                            <span>Поколение:</span>
                            <span>${strategy.evolution.generation}</span>
                        </div>
                        <div class="evolution-metric">
                            <span>Fitness Score:</span>
                            <span>${strategy.evolution.fitness_score.toFixed(1)}%</span>
                        </div>
                        <div class="evolution-metric">
                            <span>Эволюций:</span>
                            <span>${strategy.evolution.evolution_count}</span>
                        </div>
                        <div class="evolution-metric">
                            <span>Последняя эволюция:</span>
                            <span>${this.formatDate(strategy.evolution.last_evolution)}</span>
                        </div>
                    </div>
                </div>
            </div>
        `;

        this.loadStrategyParameters(strategy);
    }

    async loadStrategyParameters(strategy) {
        try {
            let parameters = {};
            
            if (window.electronAPI) {
                const details = await window.electronAPI.getStrategyDetails(strategy.id);
                if (details && details.success) {
                    parameters = details.strategy.parameters;
                }
            } else {
                // Демо параметры
                parameters = this.getDemoParameters(strategy.type);
            }

            this.displayParameters(parameters);

        } catch (error) {
            console.error('Error loading strategy parameters:', error);
            this.displayParameters(this.getDemoParameters(strategy.type));
        }
    }

    getDemoParameters(type) {
        const demoParams = {
            trend: {
                sma_fast: 20,
                sma_slow: 50,
                rsi_period: 14,
                stop_loss: 2.0,
                take_profit: 4.0
            },
            reversion: {
                bollinger_period: 20,
                bollinger_std: 2.0,
                rsi_period: 14,
                entry_threshold: 0.8,
                exit_threshold: 0.5
            },
            scalping: {
                timeframe: '1m',
                ema_fast: 5,
                ema_slow: 13,
                stop_loss: 0.5,
                take_profit: 1.0
            },
            ml: {
                lookback_period: 50,
                feature_count: 20,
                learning_rate: 0.001,
                epochs: 100,
                batch_size: 32
            }
        };

        return demoParams[type] || {};
    }

    displayParameters(parameters) {
        const parametersGrid = document.getElementById('parametersGrid');
        if (!parametersGrid) return;

        parametersGrid.innerHTML = '';

        for (const [key, value] of Object.entries(parameters)) {
            const paramEl = document.createElement('div');
            paramEl.className = 'parameter-item';
            paramEl.innerHTML = `
                <div class="parameter-label">${key.replace(/_/g, ' ')}</div>
                <div class="parameter-value">${value}</div>
            `;
            parametersGrid.appendChild(paramEl);
        }
    }

    getStatusBadgeClass(status) {
        const classes = {
            excellent: 'badge success',
            good: 'badge info',
            average: 'badge warning',
            poor: 'badge danger'
        };
        return classes[status] || 'badge';
    }

    updateEvolutionChart(strategy) {
        if (!this.evolutionChart || !strategy) return;

        // Генерация демо данных для графика эволюции
        const generations = [];
        const fitnessData = [];
        
        let currentFitness = 30 + Math.random() * 20; // Начальное значение
        
        for (let i = 0; i <= strategy.evolution.generation; i++) {
            generations.push(`Gen ${i}`);
            
            // Симуляция улучшения с некоторыми флуктуациями
            currentFitness += (Math.random() - 0.3) * 5;
            currentFitness = Math.max(0, Math.min(100, currentFitness));
            fitnessData.push(Math.round(currentFitness * 100) / 100);
        }

        this.evolutionChart.data.labels = generations.slice(-20); // Последние 20 поколений
        this.evolutionChart.data.datasets[0].data = fitnessData.slice(-20);
        this.evolutionChart.data.datasets[0].label = `${strategy.name} - Fitness Score`;
        this.evolutionChart.update();
    }

    updateEvolutionHistory(history) {
        if (!this.evolutionHistoryEl || !history) return;

        this.evolutionHistoryEl.innerHTML = '';

        history.slice(-10).reverse().forEach(entry => {
            const historyEl = document.createElement('div');
            historyEl.className = 'history-entry';
            
            const time = new Date(entry.timestamp).toLocaleTimeString();
            
            historyEl.innerHTML = `
                <div class="history-time">${time}</div>
                <div class="history-type ${entry.type}">${entry.type.replace(/_/g, ' ')}</div>
                <div class="history-message">${entry.message}</div>
            `;
            
            this.evolutionHistoryEl.appendChild(historyEl);
        });
    }

    updateTopStrategies(statistics) {
        if (statistics?.best_strategy && this.bestStrategyEl) {
            const best = statistics.best_strategy;
            this.bestStrategyEl.querySelector('.top-strategy-name').textContent = best.name;
            this.bestStrategyEl.querySelector('.top-strategy-performance').textContent = 
                `$${best.performance.toFixed(2)} (${best.win_rate.toFixed(1)}%)`;
        }

        if (statistics?.worst_strategy && this.worstStrategyEl) {
            const worst = statistics.worst_strategy;
            this.worstStrategyEl.querySelector('.top-strategy-name').textContent = worst.name;
            this.worstStrategyEl.querySelector('.top-strategy-performance').textContent = 
                `$${worst.performance.toFixed(2)} (${worst.win_rate.toFixed(1)}%)`;
        }
    }

    updateEvolutionControls(isRunning) {
        this.isEvolutionRunning = isRunning;

        if (this.evolutionStartBtn) {
            this.evolutionStartBtn.disabled = isRunning;
        }

        if (this.evolutionStopBtn) {
            this.evolutionStopBtn.disabled = !isRunning;
        }

        // Обновление статуса в интерфейсе
        const statusEl = document.querySelector('#systemStatus');
        if (statusEl && isRunning) {
            statusEl.querySelector('.status-dot').className = 'status-dot online';
            statusEl.querySelector('span').textContent = 'Эволюция активна';
        }
    }

    async startEvolution() {
        try {
            if (window.electronAPI) {
                const result = await window.electronAPI.startEvolution();
                if (result.success) {
                    this.showNotification('Эволюция', 'Эволюция стратегий запущена');
                    this.isEvolutionRunning = true;
                    this.updateEvolutionControls(true);
                } else {
                    this.showError('Ошибка', result.message || 'Не удалось запустить эволюцию');
                }
            } else {
                // Демо режим
                this.isEvolutionRunning = true;
                this.updateEvolutionControls(true);
                this.showNotification('Эволюция', 'Эволюция стратегий запущена (демо режим)');
            }
        } catch (error) {
            this.showError('Ошибка', error.message);
        }
    }

    async stopEvolution() {
        try {
            if (window.electronAPI) {
                const result = await window.electronAPI.stopEvolution();
                if (result.success) {
                    this.showNotification('Эволюция', 'Эволюция стратегий остановлена');
                    this.isEvolutionRunning = false;
                    this.updateEvolutionControls(false);
                } else {
                    this.showError('Ошибка', result.message || 'Не удалось остановить эволюцию');
                }
            } else {
                // Демо режим
                this.isEvolutionRunning = false;
                this.updateEvolutionControls(false);
                this.showNotification('Эволюция', 'Эволюция стратегий остановлена (демо режим)');
            }
        } catch (error) {
            this.showError('Ошибка', error.message);
        }
    }

    async forceEvolution() {
        try {
            if (window.electronAPI) {
                const result = await window.electronAPI.forceEvolution();
                if (result.success) {
                    this.showNotification('Эволюция', 'Принудительная эволюция выполнена');
                    this.updateEvolutionData(); // Обновление данных
                } else {
                    this.showError('Ошибка', result.message || 'Не удалось выполнить эволюцию');
                }
            } else {
                // Демо режим
                this.showNotification('Эволюция', 'Принудительная эволюция выполнена (демо режим)');
                this.simulateDemoEvolution();
            }
        } catch (error) {
            this.showError('Ошибка', error.message);
        }
    }

    async evolveSelectedStrategy() {
        if (!this.selectedStrategy) {
            this.showError('Ошибка', 'Выберите стратегию для эволюции');
            return;
        }

        try {
            if (window.electronAPI) {
                const result = await window.electronAPI.forceEvolution(this.selectedStrategy.id);
                if (result.success) {
                    this.showNotification('Эволюция', `Стратегия ${this.selectedStrategy.name} эволюционировала`);
                    this.updateEvolutionData();
                } else {
                    this.showError('Ошибка', result.message || 'Не удалось эволюционировать стратегию');
                }
            } else {
                // Демо режим
                this.showNotification('Эволюция', `Стратегия ${this.selectedStrategy.name} эволюционировала (демо режим)`);
                this.simulateDemoEvolution();
            }
        } catch (error) {
            this.showError('Ошибка', error.message);
        }
    }

    simulateDemoEvolution() {
        // Имитация изменений после эволюции
        if (this.selectedStrategy) {
            this.selectedStrategy.evolution.generation++;
            this.selectedStrategy.evolution.evolution_count++;
            this.selectedStrategy.evolution.fitness_score += (Math.random() - 0.5) * 10;
            this.selectedStrategy.evolution.fitness_score = Math.max(0, Math.min(100, this.selectedStrategy.evolution.fitness_score));
            
            this.updateStrategyDetails(this.selectedStrategy);
            this.updateEvolutionChart(this.selectedStrategy);
        }
    }

    async saveConfig() {
        try {
            const config = {
                enabled: this.evolutionEnabledEl?.checked || true,
                interval: (this.evolutionIntervalEl?.value || 1) * 3600000, // Конвертация в миллисекунды
                mutationRate: this.mutationRateEl?.value || 0.1,
                autoEvolution: this.autoEvolutionEl?.checked || false
            };

            if (window.electronAPI) {
                const result = await window.electronAPI.updateEvolutionConfig(config);
                if (result.success) {
                    this.showNotification('Настройки', 'Конфигурация эволюции сохранена');
                } else {
                    this.showError('Ошибка', result.message || 'Не удалось сохранить настройки');
                }
            } else {
                // Демо режим
                this.showNotification('Настройки', 'Конфигурация эволюции сохранена (демо режим)');
            }
        } catch (error) {
            this.showError('Ошибка', error.message);
        }
    }

    filterStrategies() {
        const filter = this.strategyFilterEl?.value || 'all';
        const strategies = this.strategiesListEl?.querySelectorAll('.strategy-evolution-item');

        if (!strategies) return;

        strategies.forEach(strategy => {
            const status = strategy.querySelector('.strategy-evolution-status').className;
            
            if (filter === 'all') {
                strategy.style.display = 'block';
            } else {
                strategy.style.display = status.includes(filter) ? 'block' : 'none';
            }
        });
    }

    formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleString('ru-RU', {
            day: '2-digit',
            month: '2-digit',
            hour: '2-digit',
            minute: '2-digit'
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

// Инициализация Evolution UI
let evolutionUI = null;

document.addEventListener('DOMContentLoaded', () => {
    evolutionUI = new EvolutionUI();
    
    // Первоначальная загрузка данных
    setTimeout(() => {
        if (evolutionUI) {
            evolutionUI.updateEvolutionData();
        }
    }, 1000);
});

// Экспорт для использования в других модулях
if (typeof window !== 'undefined') {
    window.evolutionUI = evolutionUI;
}

console.log('⚡ Evolution UI module loaded');