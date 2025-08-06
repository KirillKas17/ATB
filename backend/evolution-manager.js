const fs = require('fs').promises;
const path = require('path');
const { spawn } = require('child_process');

class EvolutionManager {
    constructor() {
        this.isRunning = false;
        this.strategies = new Map();
        this.evolutionHistory = [];
        this.config = {
            enabled: true,
            interval: 3600000, // 1 час
            autoEvolution: false,
            maxGenerations: 100,
            populationSize: 50,
            mutationRate: 0.1,
            crossoverRate: 0.8
        };
        
        this.initializeStrategies();
    }

    initializeStrategies() {
        // Инициализация базовых стратегий
        const baseStrategies = [
            {
                id: 'trend_following',
                name: 'Trend Following Strategy',
                type: 'trend',
                parameters: {
                    sma_fast: 20,
                    sma_slow: 50,
                    rsi_period: 14,
                    rsi_oversold: 30,
                    rsi_overbought: 70,
                    stop_loss: 2.0,
                    take_profit: 4.0
                },
                performance: {
                    total_trades: 0,
                    winning_trades: 0,
                    losing_trades: 0,
                    win_rate: 0.0,
                    profit_loss: 0.0,
                    max_drawdown: 0.0,
                    sharpe_ratio: 0.0
                },
                evolution: {
                    generation: 0,
                    fitness_score: 0.0,
                    last_evolution: new Date().toISOString(),
                    evolution_count: 0
                }
            },
            {
                id: 'mean_reversion',
                name: 'Mean Reversion Strategy',
                type: 'reversion',
                parameters: {
                    bollinger_period: 20,
                    bollinger_std: 2.0,
                    rsi_period: 14,
                    entry_threshold: 0.8,
                    exit_threshold: 0.5,
                    stop_loss: 1.5,
                    take_profit: 3.0
                },
                performance: {
                    total_trades: 0,
                    winning_trades: 0,
                    losing_trades: 0,
                    win_rate: 0.0,
                    profit_loss: 0.0,
                    max_drawdown: 0.0,
                    sharpe_ratio: 0.0
                },
                evolution: {
                    generation: 0,
                    fitness_score: 0.0,
                    last_evolution: new Date().toISOString(),
                    evolution_count: 0
                }
            },
            {
                id: 'scalping',
                name: 'Scalping Strategy',
                type: 'scalping',
                parameters: {
                    timeframe: '1m',
                    ema_fast: 5,
                    ema_slow: 13,
                    macd_fast: 12,
                    macd_slow: 26,
                    macd_signal: 9,
                    stop_loss: 0.5,
                    take_profit: 1.0
                },
                performance: {
                    total_trades: 0,
                    winning_trades: 0,
                    losing_trades: 0,
                    win_rate: 0.0,
                    profit_loss: 0.0,
                    max_drawdown: 0.0,
                    sharpe_ratio: 0.0
                },
                evolution: {
                    generation: 0,
                    fitness_score: 0.0,
                    last_evolution: new Date().toISOString(),
                    evolution_count: 0
                }
            },
            {
                id: 'ml_strategy',
                name: 'Machine Learning Strategy',
                type: 'ml',
                parameters: {
                    lookback_period: 50,
                    feature_count: 20,
                    model_type: 'lstm',
                    learning_rate: 0.001,
                    epochs: 100,
                    batch_size: 32,
                    dropout_rate: 0.2
                },
                performance: {
                    total_trades: 0,
                    winning_trades: 0,
                    losing_trades: 0,
                    win_rate: 0.0,
                    profit_loss: 0.0,
                    max_drawdown: 0.0,
                    sharpe_ratio: 0.0
                },
                evolution: {
                    generation: 0,
                    fitness_score: 0.0,
                    last_evolution: new Date().toISOString(),
                    evolution_count: 0
                }
            }
        ];

        // Заполнение демо данными
        baseStrategies.forEach(strategy => {
            // Симуляция случайной производительности
            strategy.performance.total_trades = Math.floor(Math.random() * 1000) + 100;
            strategy.performance.winning_trades = Math.floor(strategy.performance.total_trades * (0.4 + Math.random() * 0.3));
            strategy.performance.losing_trades = strategy.performance.total_trades - strategy.performance.winning_trades;
            strategy.performance.win_rate = (strategy.performance.winning_trades / strategy.performance.total_trades) * 100;
            strategy.performance.profit_loss = (Math.random() - 0.3) * 10000;
            strategy.performance.max_drawdown = Math.random() * 15;
            strategy.performance.sharpe_ratio = (Math.random() - 0.2) * 3;
            
            // Симуляция эволюции
            strategy.evolution.generation = Math.floor(Math.random() * 50);
            strategy.evolution.fitness_score = Math.random() * 100;
            strategy.evolution.evolution_count = Math.floor(Math.random() * 100);
            
            this.strategies.set(strategy.id, strategy);
        });
    }

    async getStatus() {
        try {
            const strategiesArray = Array.from(this.strategies.values());
            
            // Подсчет общей статистики
            const totalStrategies = strategiesArray.length;
            const activeStrategies = strategiesArray.filter(s => s.performance.win_rate > 50).length;
            const avgPerformance = strategiesArray.reduce((sum, s) => sum + s.performance.profit_loss, 0) / totalStrategies;
            const totalEvolutions = strategiesArray.reduce((sum, s) => sum + s.evolution.evolution_count, 0);

            return {
                success: true,
                enabled: this.config.enabled,
                running: this.isRunning,
                statistics: {
                    total_strategies: totalStrategies,
                    active_strategies: activeStrategies,
                    average_performance: avgPerformance,
                    total_evolutions: totalEvolutions,
                    best_strategy: this.getBestStrategy(),
                    worst_strategy: this.getWorstStrategy()
                },
                strategies: strategiesArray.map(strategy => ({
                    id: strategy.id,
                    name: strategy.name,
                    type: strategy.type,
                    status: this.getStrategyStatus(strategy),
                    performance: {
                        win_rate: Math.round(strategy.performance.win_rate * 100) / 100,
                        profit_loss: Math.round(strategy.performance.profit_loss * 100) / 100,
                        total_trades: strategy.performance.total_trades,
                        sharpe_ratio: Math.round(strategy.performance.sharpe_ratio * 100) / 100
                    },
                    evolution: {
                        generation: strategy.evolution.generation,
                        fitness_score: Math.round(strategy.evolution.fitness_score * 100) / 100,
                        evolution_count: strategy.evolution.evolution_count,
                        last_evolution: strategy.evolution.last_evolution
                    }
                })),
                config: this.config,
                history: this.evolutionHistory.slice(-10), // Последние 10 записей
                timestamp: new Date().toISOString()
            };

        } catch (error) {
            console.error('❌ Ошибка получения статуса эволюции:', error);
            return {
                success: false,
                error: error.message,
                timestamp: new Date().toISOString()
            };
        }
    }

    getStrategyStatus(strategy) {
        if (strategy.performance.win_rate > 60 && strategy.performance.profit_loss > 0) {
            return 'excellent';
        } else if (strategy.performance.win_rate > 50 && strategy.performance.profit_loss > 0) {
            return 'good';
        } else if (strategy.performance.win_rate > 40) {
            return 'average';
        } else {
            return 'poor';
        }
    }

    getBestStrategy() {
        let best = null;
        let bestScore = -Infinity;

        for (const strategy of this.strategies.values()) {
            const score = strategy.performance.profit_loss * (strategy.performance.win_rate / 100);
            if (score > bestScore) {
                bestScore = score;
                best = strategy;
            }
        }

        return best ? {
            id: best.id,
            name: best.name,
            performance: best.performance.profit_loss,
            win_rate: best.performance.win_rate
        } : null;
    }

    getWorstStrategy() {
        let worst = null;
        let worstScore = Infinity;

        for (const strategy of this.strategies.values()) {
            const score = strategy.performance.profit_loss * (strategy.performance.win_rate / 100);
            if (score < worstScore) {
                worstScore = score;
                worst = strategy;
            }
        }

        return worst ? {
            id: worst.id,
            name: worst.name,
            performance: worst.performance.profit_loss,
            win_rate: worst.performance.win_rate
        } : null;
    }

    async start() {
        try {
            if (this.isRunning) {
                return {
                    success: false,
                    message: 'Evolution is already running',
                    timestamp: new Date().toISOString()
                };
            }

            if (!this.config.enabled) {
                return {
                    success: false,
                    message: 'Evolution is disabled in configuration',
                    timestamp: new Date().toISOString()
                };
            }

            this.isRunning = true;
            console.log('🧬 Starting strategy evolution...');

            // Запуск фонового процесса эволюции
            this.startEvolutionLoop();

            // Добавление записи в историю
            this.addHistoryEntry('evolution_started', 'Strategy evolution process started');

            return {
                success: true,
                message: 'Evolution started successfully',
                timestamp: new Date().toISOString()
            };

        } catch (error) {
            console.error('❌ Ошибка запуска эволюции:', error);
            return {
                success: false,
                error: error.message,
                timestamp: new Date().toISOString()
            };
        }
    }

    async stop() {
        try {
            if (!this.isRunning) {
                return {
                    success: false,
                    message: 'Evolution is not running',
                    timestamp: new Date().toISOString()
                };
            }

            this.isRunning = false;
            console.log('🧬 Stopping strategy evolution...');

            // Остановка процесса эволюции
            if (this.evolutionInterval) {
                clearInterval(this.evolutionInterval);
                this.evolutionInterval = null;
            }

            // Добавление записи в историю
            this.addHistoryEntry('evolution_stopped', 'Strategy evolution process stopped');

            return {
                success: true,
                message: 'Evolution stopped successfully',
                timestamp: new Date().toISOString()
            };

        } catch (error) {
            console.error('❌ Ошибка остановки эволюции:', error);
            return {
                success: false,
                error: error.message,
                timestamp: new Date().toISOString()
            };
        }
    }

    startEvolutionLoop() {
        // Первая эволюция через 10 секунд
        setTimeout(() => {
            this.performEvolution();
        }, 10000);

        // Запуск периодической эволюции
        this.evolutionInterval = setInterval(() => {
            if (this.isRunning) {
                this.performEvolution();
            }
        }, this.config.interval);
    }

    async performEvolution() {
        try {
            console.log('🧬 Performing strategy evolution...');

            // Выбор случайной стратегии для эволюции
            const strategiesArray = Array.from(this.strategies.values());
            const strategyToEvolve = strategiesArray[Math.floor(Math.random() * strategiesArray.length)];

            if (!strategyToEvolve) return;

            // Сохранение предыдущих параметров
            const previousParams = { ...strategyToEvolve.parameters };
            const previousPerformance = strategyToEvolve.performance.profit_loss;

            // Мутация параметров
            this.mutateStrategy(strategyToEvolve);

            // Симуляция тестирования новых параметров
            await this.simulateStrategyTest(strategyToEvolve);

            // Обновление эволюционных данных
            strategyToEvolve.evolution.generation++;
            strategyToEvolve.evolution.evolution_count++;
            strategyToEvolve.evolution.last_evolution = new Date().toISOString();

            // Расчет fitness score
            const fitnessScore = this.calculateFitnessScore(strategyToEvolve);
            strategyToEvolve.evolution.fitness_score = fitnessScore;

            // Добавление записи в историю
            const improvement = strategyToEvolve.performance.profit_loss - previousPerformance;
            this.addHistoryEntry('strategy_evolved', 
                `Strategy ${strategyToEvolve.name} evolved. Performance change: ${improvement.toFixed(2)}`);

            console.log(`✅ Strategy ${strategyToEvolve.name} evolved successfully`);

        } catch (error) {
            console.error('❌ Ошибка выполнения эволюции:', error);
            this.addHistoryEntry('evolution_error', `Evolution error: ${error.message}`);
        }
    }

    mutateStrategy(strategy) {
        // Мутация параметров стратегии
        const mutationRate = this.config.mutationRate;

        for (const [key, value] of Object.entries(strategy.parameters)) {
            if (Math.random() < mutationRate) {
                if (typeof value === 'number') {
                    // Мутация числовых параметров
                    const mutationFactor = 1 + (Math.random() - 0.5) * 0.2; // ±10%
                    strategy.parameters[key] = Math.max(1, value * mutationFactor);
                } else if (typeof value === 'string') {
                    // Мутация строковых параметров (если применимо)
                    // Здесь можно добавить логику для строковых параметров
                }
            }
        }
    }

    async simulateStrategyTest(strategy) {
        // Симуляция тестирования стратегии с новыми параметрами
        return new Promise(resolve => {
            setTimeout(() => {
                // Генерация новых показателей производительности
                const performanceVariation = (Math.random() - 0.5) * 0.3; // ±15%
                
                strategy.performance.total_trades += Math.floor(Math.random() * 10) + 1;
                
                const newWinRate = Math.max(0, Math.min(100, 
                    strategy.performance.win_rate * (1 + performanceVariation)));
                
                strategy.performance.win_rate = newWinRate;
                strategy.performance.winning_trades = Math.floor(
                    strategy.performance.total_trades * (newWinRate / 100));
                strategy.performance.losing_trades = 
                    strategy.performance.total_trades - strategy.performance.winning_trades;

                // Обновление P&L
                const pnlVariation = (Math.random() - 0.4) * 2000; // Небольшой биас к улучшению
                strategy.performance.profit_loss += pnlVariation;

                // Обновление других метрик
                strategy.performance.max_drawdown = Math.max(0, 
                    strategy.performance.max_drawdown + (Math.random() - 0.5) * 2);
                strategy.performance.sharpe_ratio += (Math.random() - 0.5) * 0.1;

                resolve();
            }, 1000); // Симуляция времени тестирования
        });
    }

    calculateFitnessScore(strategy) {
        // Расчет комплексного показателя fitness
        const winRateWeight = 0.3;
        const profitWeight = 0.4;
        const drawdownWeight = 0.2;
        const sharpeWeight = 0.1;

        const normalizedWinRate = strategy.performance.win_rate / 100;
        const normalizedProfit = Math.max(0, strategy.performance.profit_loss / 10000);
        const normalizedDrawdown = Math.max(0, 1 - (strategy.performance.max_drawdown / 50));
        const normalizedSharpe = Math.max(0, (strategy.performance.sharpe_ratio + 2) / 4);

        const fitnessScore = (
            normalizedWinRate * winRateWeight +
            normalizedProfit * profitWeight +
            normalizedDrawdown * drawdownWeight +
            normalizedSharpe * sharpeWeight
        ) * 100;

        return Math.round(fitnessScore * 100) / 100;
    }

    addHistoryEntry(type, message) {
        const entry = {
            timestamp: new Date().toISOString(),
            type: type,
            message: message
        };

        this.evolutionHistory.push(entry);

        // Ограничение размера истории
        if (this.evolutionHistory.length > 100) {
            this.evolutionHistory = this.evolutionHistory.slice(-100);
        }
    }

    async forceEvolution(strategyId) {
        try {
            const strategy = this.strategies.get(strategyId);
            if (!strategy) {
                return {
                    success: false,
                    message: `Strategy ${strategyId} not found`,
                    timestamp: new Date().toISOString()
                };
            }

            // Принудительная эволюция конкретной стратегии
            await this.evolveSpecificStrategy(strategy);

            return {
                success: true,
                message: `Strategy ${strategy.name} evolved successfully`,
                strategy: strategy,
                timestamp: new Date().toISOString()
            };

        } catch (error) {
            console.error('❌ Ошибка принудительной эволюции:', error);
            return {
                success: false,
                error: error.message,
                timestamp: new Date().toISOString()
            };
        }
    }

    async evolveSpecificStrategy(strategy) {
        const previousPerformance = strategy.performance.profit_loss;
        
        // Мутация и тестирование
        this.mutateStrategy(strategy);
        await this.simulateStrategyTest(strategy);
        
        // Обновление данных
        strategy.evolution.generation++;
        strategy.evolution.evolution_count++;
        strategy.evolution.last_evolution = new Date().toISOString();
        strategy.evolution.fitness_score = this.calculateFitnessScore(strategy);
        
        const improvement = strategy.performance.profit_loss - previousPerformance;
        this.addHistoryEntry('forced_evolution', 
            `Forced evolution of ${strategy.name}. Performance change: ${improvement.toFixed(2)}`);
    }

    async updateConfig(newConfig) {
        try {
            // Валидация конфигурации
            if (newConfig.interval && newConfig.interval < 60000) {
                throw new Error('Evolution interval cannot be less than 1 minute');
            }

            if (newConfig.mutationRate && (newConfig.mutationRate < 0 || newConfig.mutationRate > 1)) {
                throw new Error('Mutation rate must be between 0 and 1');
            }

            // Обновление конфигурации
            this.config = { ...this.config, ...newConfig };

            // Перезапуск с новой конфигурацией если работает
            if (this.isRunning) {
                await this.stop();
                await this.start();
            }

            this.addHistoryEntry('config_updated', 'Evolution configuration updated');

            return {
                success: true,
                message: 'Configuration updated successfully',
                config: this.config,
                timestamp: new Date().toISOString()
            };

        } catch (error) {
            console.error('❌ Ошибка обновления конфигурации:', error);
            return {
                success: false,
                error: error.message,
                timestamp: new Date().toISOString()
            };
        }
    }

    async getStrategyDetails(strategyId) {
        try {
            const strategy = this.strategies.get(strategyId);
            if (!strategy) {
                return {
                    success: false,
                    message: `Strategy ${strategyId} not found`,
                    timestamp: new Date().toISOString()
                };
            }

            return {
                success: true,
                strategy: strategy,
                timestamp: new Date().toISOString()
            };

        } catch (error) {
            return {
                success: false,
                error: error.message,
                timestamp: new Date().toISOString()
            };
        }
    }

    // Интеграция с Python системой эволюции
    async integratePythonEvolution() {
        try {
            // Попытка запуска Python скрипта эволюции
            const pythonProcess = spawn('python', [
                path.join(process.cwd(), 'infrastructure/core/evolution_manager.py'),
                '--mode', 'evolution',
                '--strategies', JSON.stringify(Array.from(this.strategies.keys()))
            ]);

            pythonProcess.stdout.on('data', (data) => {
                console.log(`Python Evolution: ${data}`);
            });

            pythonProcess.stderr.on('data', (data) => {
                console.error(`Python Evolution Error: ${data}`);
            });

            return new Promise((resolve) => {
                pythonProcess.on('close', (code) => {
                    resolve({
                        success: code === 0,
                        message: code === 0 ? 'Python evolution completed' : 'Python evolution failed',
                        code: code
                    });
                });
            });

        } catch (error) {
            console.warn('⚠️ Python integration not available:', error.message);
            return {
                success: false,
                message: 'Python integration not available',
                error: error.message
            };
        }
    }
}

module.exports = { EvolutionManager };