const fs = require('fs').promises;
const path = require('path');
const { spawn } = require('child_process');

class TradingManager {
    constructor() {
        this.isRunning = false;
        this.mode = 'simulation'; // simulation, paper, live
        this.currentPositions = new Map();
        this.openOrders = new Map();
        this.tradeHistory = [];
        this.portfolioHistory = [];
        this.performanceMetrics = {
            totalReturn: 0,
            sharpeRatio: 0,
            maxDrawdown: 0,
            winRate: 0,
            profitFactor: 0,
            totalTrades: 0,
            averageReturn: 0,
            volatility: 0,
            calmarRatio: 0,
            sortinoRatio: 0
        };
        
        this.portfolio = {
            totalBalance: 10000.0,
            availableBalance: 10000.0,
            unrealizedPnL: 0.0,
            realizedPnL: 0.0,
            equity: 10000.0,
            margin: 0.0,
            marginLevel: 0.0,
            freeMargin: 10000.0
        };

        this.riskMetrics = {
            currentDrawdown: 0.0,
            maxDrawdownPercent: 0.0,
            valueAtRisk: 0.0,
            exposurePercent: 0.0,
            leverageRatio: 1.0,
            riskRewardRatio: 0.0
        };

        this.marketData = new Map();
        this.strategies = new Map();
        
        this.initializeStrategies();
        this.initializeMarketData();
        this.startDataSimulation();
    }

    initializeStrategies() {
        // Инициализация активных торговых стратегий
        const activeStrategies = [
            {
                id: 'trend_master',
                name: 'Trend Master Pro',
                type: 'trend_following',
                status: 'active',
                allocation: 30,
                instruments: ['BTCUSDT', 'ETHUSDT'],
                performance: {
                    totalReturn: 15.8,
                    sharpeRatio: 1.23,
                    maxDrawdown: 8.5,
                    winRate: 68.2,
                    totalTrades: 156,
                    currentPnL: 1847.32
                },
                parameters: {
                    fast_ma: 20,
                    slow_ma: 50,
                    rsi_period: 14,
                    stop_loss: 2.0,
                    take_profit: 4.0,
                    position_size: 0.02
                }
            },
            {
                id: 'scalper_ai',
                name: 'AI Scalper Elite',
                type: 'scalping',
                status: 'active',
                allocation: 25,
                instruments: ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
                performance: {
                    totalReturn: 22.4,
                    sharpeRatio: 1.87,
                    maxDrawdown: 5.2,
                    winRate: 71.5,
                    totalTrades: 423,
                    currentPnL: 2240.15
                },
                parameters: {
                    timeframe: '1m',
                    ema_fast: 5,
                    ema_slow: 13,
                    rsi_oversold: 25,
                    rsi_overbought: 75,
                    position_size: 0.01
                }
            },
            {
                id: 'mean_reversion',
                name: 'Mean Reversion Quantum',
                type: 'mean_reversion',
                status: 'paused',
                allocation: 20,
                instruments: ['ETHUSDT', 'ADAUSDT'],
                performance: {
                    totalReturn: -3.2,
                    sharpeRatio: -0.45,
                    maxDrawdown: 12.8,
                    winRate: 45.3,
                    totalTrades: 89,
                    currentPnL: -320.45
                },
                parameters: {
                    bb_period: 20,
                    bb_std: 2.0,
                    rsi_period: 14,
                    mean_lookback: 50,
                    position_size: 0.015
                }
            },
            {
                id: 'ml_predictor',
                name: 'ML Price Predictor',
                type: 'machine_learning',
                status: 'training',
                allocation: 25,
                instruments: ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
                performance: {
                    totalReturn: 8.9,
                    sharpeRatio: 0.76,
                    maxDrawdown: 7.1,
                    winRate: 58.7,
                    totalTrades: 67,
                    currentPnL: 890.23
                },
                parameters: {
                    model_type: 'lstm',
                    lookback_days: 30,
                    prediction_horizon: 24,
                    confidence_threshold: 0.7,
                    position_size: 0.025
                }
            }
        ];

        activeStrategies.forEach(strategy => {
            this.strategies.set(strategy.id, strategy);
        });
    }

    initializeMarketData() {
        // Инициализация рыночных данных
        const instruments = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT'];
        
        instruments.forEach(symbol => {
            this.marketData.set(symbol, {
                symbol: symbol,
                price: this.getRandomPrice(symbol),
                change24h: (Math.random() - 0.5) * 10,
                volume24h: Math.random() * 1000000000,
                high24h: 0,
                low24h: 0,
                bid: 0,
                ask: 0,
                spread: 0,
                lastUpdate: new Date().toISOString()
            });
        });

        this.updateMarketPrices();
    }

    getRandomPrice(symbol) {
        const basePrices = {
            'BTCUSDT': 43000,
            'ETHUSDT': 2600,
            'BNBUSDT': 300,
            'ADAUSDT': 0.5,
            'SOLUSDT': 100
        };
        
        const basePrice = basePrices[symbol] || 100;
        return basePrice * (0.9 + Math.random() * 0.2); // ±10% вариация
    }

    startDataSimulation() {
        // Симуляция обновления рыночных данных
        setInterval(() => {
            this.updateMarketPrices();
            this.updatePositions();
            this.updatePortfolio();
            this.calculateMetrics();
        }, 2000); // Каждые 2 секунды

        // Периодическое генерирование сделок
        setInterval(() => {
            if (this.isRunning) {
                this.generateTrade();
            }
        }, 5000); // Каждые 5 секунд
    }

    updateMarketPrices() {
        for (const [symbol, data] of this.marketData.entries()) {
            // Симуляция движения цены
            const changePercent = (Math.random() - 0.5) * 0.02; // ±1% изменение
            data.price *= (1 + changePercent);
            data.change24h += changePercent * 100;
            data.volume24h += Math.random() * 10000000;
            
            // Обновление bid/ask
            data.spread = data.price * 0.001; // 0.1% спред
            data.bid = data.price - data.spread / 2;
            data.ask = data.price + data.spread / 2;
            
            data.lastUpdate = new Date().toISOString();
        }
    }

    updatePositions() {
        // Обновление открытых позиций
        for (const [positionId, position] of this.currentPositions.entries()) {
            const marketData = this.marketData.get(position.symbol);
            if (marketData) {
                const currentPrice = marketData.price;
                const priceDiff = currentPrice - position.entryPrice;
                
                if (position.side === 'long') {
                    position.unrealizedPnL = priceDiff * position.size;
                } else {
                    position.unrealizedPnL = -priceDiff * position.size;
                }
                
                position.currentPrice = currentPrice;
                position.returnPercent = (position.unrealizedPnL / (position.entryPrice * position.size)) * 100;
            }
        }
    }

    updatePortfolio() {
        // Расчет общего P&L
        let totalUnrealizedPnL = 0;
        for (const position of this.currentPositions.values()) {
            totalUnrealizedPnL += position.unrealizedPnL || 0;
        }

        this.portfolio.unrealizedPnL = totalUnrealizedPnL;
        this.portfolio.equity = this.portfolio.totalBalance + this.portfolio.realizedPnL + this.portfolio.unrealizedPnL;
        this.portfolio.freeMargin = this.portfolio.equity - this.portfolio.margin;
        
        if (this.portfolio.margin > 0) {
            this.portfolio.marginLevel = (this.portfolio.equity / this.portfolio.margin) * 100;
        }

        // Добавление в историю портфеля
        this.portfolioHistory.push({
            timestamp: new Date().toISOString(),
            equity: this.portfolio.equity,
            balance: this.portfolio.totalBalance,
            unrealizedPnL: this.portfolio.unrealizedPnL,
            realizedPnL: this.portfolio.realizedPnL
        });

        // Ограничение истории
        if (this.portfolioHistory.length > 1000) {
            this.portfolioHistory = this.portfolioHistory.slice(-1000);
        }
    }

    generateTrade() {
        if (Math.random() < 0.3) { // 30% шанс на новую сделку
            const strategies = Array.from(this.strategies.values()).filter(s => s.status === 'active');
            const symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'];
            
            if (strategies.length > 0 && symbols.length > 0) {
                const strategy = strategies[Math.floor(Math.random() * strategies.length)];
                const symbol = symbols[Math.floor(Math.random() * symbols.length)];
                const marketData = this.marketData.get(symbol);
                
                if (marketData) {
                    const side = Math.random() > 0.5 ? 'long' : 'short';
                    const size = 0.1 + Math.random() * 0.5; // 0.1 - 0.6
                    
                    this.openPosition({
                        strategy: strategy.id,
                        symbol: symbol,
                        side: side,
                        size: size,
                        entryPrice: marketData.price,
                        stopLoss: side === 'long' ? 
                            marketData.price * 0.98 : marketData.price * 1.02,
                        takeProfit: side === 'long' ? 
                            marketData.price * 1.04 : marketData.price * 0.96
                    });
                }
            }
        }

        // Закрытие позиций
        for (const [positionId, position] of this.currentPositions.entries()) {
            if (Math.random() < 0.1) { // 10% шанс закрыть позицию
                this.closePosition(positionId);
            }
        }
    }

    openPosition(params) {
        const positionId = `pos_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const position = {
            id: positionId,
            strategy: params.strategy,
            symbol: params.symbol,
            side: params.side,
            size: params.size,
            entryPrice: params.entryPrice,
            currentPrice: params.entryPrice,
            stopLoss: params.stopLoss,
            takeProfit: params.takeProfit,
            unrealizedPnL: 0,
            returnPercent: 0,
            openTime: new Date().toISOString(),
            status: 'open'
        };

        this.currentPositions.set(positionId, position);
        
        // Обновление маржи
        this.portfolio.margin += params.entryPrice * params.size * 0.1; // 10% маржа
        
        return positionId;
    }

    closePosition(positionId) {
        const position = this.currentPositions.get(positionId);
        if (!position) return null;

        position.closePrice = position.currentPrice;
        position.closeTime = new Date().toISOString();
        position.realizedPnL = position.unrealizedPnL;
        position.status = 'closed';

        // Обновление портфеля
        this.portfolio.realizedPnL += position.realizedPnL;
        this.portfolio.margin -= position.entryPrice * position.size * 0.1;

        // Добавление в историю сделок
        this.tradeHistory.push({...position});

        // Удаление из активных позиций
        this.currentPositions.delete(positionId);

        // Ограничение истории
        if (this.tradeHistory.length > 1000) {
            this.tradeHistory = this.tradeHistory.slice(-1000);
        }

        return position;
    }

    calculateMetrics() {
        if (this.tradeHistory.length === 0) return;

        const trades = this.tradeHistory;
        const winningTrades = trades.filter(t => t.realizedPnL > 0);
        const losingTrades = trades.filter(t => t.realizedPnL < 0);

        // Базовые метрики
        this.performanceMetrics.totalTrades = trades.length;
        this.performanceMetrics.winRate = (winningTrades.length / trades.length) * 100;

        const totalReturn = trades.reduce((sum, t) => sum + t.realizedPnL, 0);
        this.performanceMetrics.totalReturn = (totalReturn / 10000) * 100; // Процент от начального капитала

        // Profit Factor
        const grossProfit = winningTrades.reduce((sum, t) => sum + t.realizedPnL, 0);
        const grossLoss = Math.abs(losingTrades.reduce((sum, t) => sum + t.realizedPnL, 0));
        this.performanceMetrics.profitFactor = grossLoss > 0 ? grossProfit / grossLoss : 0;

        // Средний доход
        this.performanceMetrics.averageReturn = totalReturn / trades.length;

        // Максимальная просадка
        this.calculateDrawdown();

        // Sharpe Ratio
        this.calculateSharpeRatio();

        // Volatility
        this.calculateVolatility();

        // Calmar и Sortino Ratios
        this.calculateAdvancedRatios();

        // Risk метрики
        this.calculateRiskMetrics();
    }

    calculateDrawdown() {
        let peak = 10000; // Начальный капитал
        let maxDrawdown = 0;
        let currentDrawdown = 0;

        for (const portfolioPoint of this.portfolioHistory) {
            if (portfolioPoint.equity > peak) {
                peak = portfolioPoint.equity;
            }

            currentDrawdown = (peak - portfolioPoint.equity) / peak * 100;
            if (currentDrawdown > maxDrawdown) {
                maxDrawdown = currentDrawdown;
            }
        }

        this.performanceMetrics.maxDrawdown = maxDrawdown;
        this.riskMetrics.currentDrawdown = currentDrawdown;
        this.riskMetrics.maxDrawdownPercent = maxDrawdown;
    }

    calculateSharpeRatio() {
        if (this.portfolioHistory.length < 2) return;

        const returns = [];
        for (let i = 1; i < this.portfolioHistory.length; i++) {
            const prevEquity = this.portfolioHistory[i - 1].equity;
            const currentEquity = this.portfolioHistory[i].equity;
            const returnRate = (currentEquity - prevEquity) / prevEquity;
            returns.push(returnRate);
        }

        const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
        const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
        const volatility = Math.sqrt(variance);

        // Предполагаем безрисковую ставку 2% годовых
        const riskFreeRate = 0.02 / 365; // Дневная ставка
        this.performanceMetrics.sharpeRatio = volatility > 0 ? (avgReturn - riskFreeRate) / volatility : 0;
    }

    calculateVolatility() {
        if (this.portfolioHistory.length < 2) return;

        const returns = [];
        for (let i = 1; i < this.portfolioHistory.length; i++) {
            const prevEquity = this.portfolioHistory[i - 1].equity;
            const currentEquity = this.portfolioHistory[i].equity;
            const returnRate = (currentEquity - prevEquity) / prevEquity;
            returns.push(returnRate);
        }

        const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
        const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
        this.performanceMetrics.volatility = Math.sqrt(variance) * Math.sqrt(365) * 100; // Годовая волатильность в %
    }

    calculateAdvancedRatios() {
        if (this.performanceMetrics.maxDrawdown > 0) {
            this.performanceMetrics.calmarRatio = this.performanceMetrics.totalReturn / this.performanceMetrics.maxDrawdown;
        }

        // Sortino Ratio (использует только отрицательную волатильность)
        if (this.portfolioHistory.length >= 2) {
            const negativeReturns = [];
            for (let i = 1; i < this.portfolioHistory.length; i++) {
                const prevEquity = this.portfolioHistory[i - 1].equity;
                const currentEquity = this.portfolioHistory[i].equity;
                const returnRate = (currentEquity - prevEquity) / prevEquity;
                if (returnRate < 0) {
                    negativeReturns.push(returnRate);
                }
            }

            if (negativeReturns.length > 0) {
                const avgNegReturn = negativeReturns.reduce((sum, r) => sum + r, 0) / negativeReturns.length;
                const downwardDeviation = Math.sqrt(
                    negativeReturns.reduce((sum, r) => sum + Math.pow(r - avgNegReturn, 2), 0) / negativeReturns.length
                );
                
                const avgTotalReturn = this.performanceMetrics.totalReturn / 365; // Дневная доходность
                this.performanceMetrics.sortinoRatio = downwardDeviation > 0 ? avgTotalReturn / downwardDeviation : 0;
            }
        }
    }

    calculateRiskMetrics() {
        // Value at Risk (95% уверенность)
        if (this.portfolioHistory.length >= 20) {
            const returns = [];
            for (let i = 1; i < this.portfolioHistory.length; i++) {
                const prevEquity = this.portfolioHistory[i - 1].equity;
                const currentEquity = this.portfolioHistory[i].equity;
                const returnRate = (currentEquity - prevEquity) / prevEquity;
                returns.push(returnRate);
            }

            returns.sort((a, b) => a - b);
            const varIndex = Math.floor(returns.length * 0.05);
            this.riskMetrics.valueAtRisk = Math.abs(returns[varIndex] || 0) * this.portfolio.equity;
        }

        // Exposure
        let totalExposure = 0;
        for (const position of this.currentPositions.values()) {
            totalExposure += position.entryPrice * position.size;
        }
        this.riskMetrics.exposurePercent = (totalExposure / this.portfolio.equity) * 100;

        // Leverage
        this.riskMetrics.leverageRatio = totalExposure / this.portfolio.equity;

        // Risk-Reward Ratio
        const winningTrades = this.tradeHistory.filter(t => t.realizedPnL > 0);
        const losingTrades = this.tradeHistory.filter(t => t.realizedPnL < 0);
        
        if (winningTrades.length > 0 && losingTrades.length > 0) {
            const avgWin = winningTrades.reduce((sum, t) => sum + t.realizedPnL, 0) / winningTrades.length;
            const avgLoss = Math.abs(losingTrades.reduce((sum, t) => sum + t.realizedPnL, 0) / losingTrades.length);
            this.riskMetrics.riskRewardRatio = avgWin / avgLoss;
        }
    }

    async startTrading(mode = 'simulation') {
        try {
            if (this.isRunning) {
                return {
                    success: false,
                    message: 'Trading is already running',
                    timestamp: new Date().toISOString()
                };
            }

            this.mode = mode;
            this.isRunning = true;

            console.log(`🚀 Starting trading in ${mode} mode...`);

            // Активация стратегий
            for (const strategy of this.strategies.values()) {
                if (strategy.status === 'paused') {
                    strategy.status = 'active';
                }
            }

            return {
                success: true,
                message: `Trading started in ${mode} mode`,
                mode: this.mode,
                timestamp: new Date().toISOString()
            };

        } catch (error) {
            console.error('❌ Error starting trading:', error);
            return {
                success: false,
                error: error.message,
                timestamp: new Date().toISOString()
            };
        }
    }

    async stopTrading() {
        try {
            if (!this.isRunning) {
                return {
                    success: false,
                    message: 'Trading is not running',
                    timestamp: new Date().toISOString()
                };
            }

            this.isRunning = false;
            console.log('🛑 Stopping trading...');

            // Приостановка стратегий
            for (const strategy of this.strategies.values()) {
                if (strategy.status === 'active') {
                    strategy.status = 'paused';
                }
            }

            // Закрытие всех позиций в режиме остановки
            const openPositions = Array.from(this.currentPositions.keys());
            for (const positionId of openPositions) {
                this.closePosition(positionId);
            }

            return {
                success: true,
                message: 'Trading stopped successfully',
                closedPositions: openPositions.length,
                timestamp: new Date().toISOString()
            };

        } catch (error) {
            console.error('❌ Error stopping trading:', error);
            return {
                success: false,
                error: error.message,
                timestamp: new Date().toISOString()
            };
        }
    }

    async runBacktest(parameters) {
        try {
            console.log('📊 Running backtest...');

            const backtestResult = {
                success: true,
                parameters: parameters,
                startDate: parameters.startDate || '2024-01-01',
                endDate: parameters.endDate || new Date().toISOString().split('T')[0],
                initialCapital: parameters.initialCapital || 10000,
                results: {
                    finalCapital: 0,
                    totalReturn: 0,
                    maxDrawdown: 0,
                    sharpeRatio: 0,
                    winRate: 0,
                    totalTrades: 0,
                    profitFactor: 0,
                    calmarRatio: 0,
                    sortinoRatio: 0
                },
                trades: [],
                equity_curve: []
            };

            // Симуляция бэктеста
            const days = parameters.days || 90;
            let capital = backtestResult.initialCapital;
            let peak = capital;
            let maxDD = 0;
            const dailyReturns = [];

            for (let day = 0; day < days; day++) {
                // Симуляция дневной доходности
                const dailyReturn = (Math.random() - 0.45) * 0.03; // Небольшой положительный bias
                capital *= (1 + dailyReturn);
                dailyReturns.push(dailyReturn);

                // Обновление максимальной просадки
                if (capital > peak) peak = capital;
                const drawdown = (peak - capital) / peak;
                if (drawdown > maxDD) maxDD = drawdown;

                // Добавление точки в кривую эквити
                backtestResult.equity_curve.push({
                    date: new Date(Date.now() - (days - day) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
                    equity: capital,
                    drawdown: drawdown * 100
                });

                // Генерация случайных сделок
                if (Math.random() < 0.3) { // 30% шанс на сделку
                    const isWin = Math.random() < 0.58; // 58% win rate
                    const pnl = isWin ? 
                        (50 + Math.random() * 200) : 
                        -(30 + Math.random() * 150);
                    
                    backtestResult.trades.push({
                        date: backtestResult.equity_curve[backtestResult.equity_curve.length - 1].date,
                        symbol: ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'][Math.floor(Math.random() * 3)],
                        side: Math.random() > 0.5 ? 'long' : 'short',
                        pnl: pnl,
                        return_percent: (pnl / 1000) * 100
                    });
                }
            }

            // Расчет финальных метрик
            backtestResult.results.finalCapital = capital;
            backtestResult.results.totalReturn = ((capital - backtestResult.initialCapital) / backtestResult.initialCapital) * 100;
            backtestResult.results.maxDrawdown = maxDD * 100;
            backtestResult.results.totalTrades = backtestResult.trades.length;

            const winningTrades = backtestResult.trades.filter(t => t.pnl > 0);
            backtestResult.results.winRate = (winningTrades.length / backtestResult.results.totalTrades) * 100;

            // Sharpe Ratio
            const avgReturn = dailyReturns.reduce((sum, r) => sum + r, 0) / dailyReturns.length;
            const volatility = Math.sqrt(
                dailyReturns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / dailyReturns.length
            );
            backtestResult.results.sharpeRatio = volatility > 0 ? (avgReturn * 252) / (volatility * Math.sqrt(252)) : 0;

            // Profit Factor
            const grossProfit = winningTrades.reduce((sum, t) => sum + t.pnl, 0);
            const grossLoss = Math.abs(backtestResult.trades.filter(t => t.pnl < 0).reduce((sum, t) => sum + t.pnl, 0));
            backtestResult.results.profitFactor = grossLoss > 0 ? grossProfit / grossLoss : 0;

            // Calmar Ratio
            backtestResult.results.calmarRatio = backtestResult.results.maxDrawdown > 0 ? 
                backtestResult.results.totalReturn / backtestResult.results.maxDrawdown : 0;

            // Sortino Ratio
            const negativeReturns = dailyReturns.filter(r => r < 0);
            if (negativeReturns.length > 0) {
                const downwardDeviation = Math.sqrt(
                    negativeReturns.reduce((sum, r) => sum + r * r, 0) / negativeReturns.length
                );
                backtestResult.results.sortinoRatio = (avgReturn * 252) / (downwardDeviation * Math.sqrt(252));
            }

            backtestResult.timestamp = new Date().toISOString();
            console.log('✅ Backtest completed');

            return backtestResult;

        } catch (error) {
            console.error('❌ Error running backtest:', error);
            return {
                success: false,
                error: error.message,
                timestamp: new Date().toISOString()
            };
        }
    }

    async getStatus() {
        try {
            // Получение данных стратегий
            const strategiesArray = Array.from(this.strategies.values());
            const activeStrategies = strategiesArray.filter(s => s.status === 'active').length;

            // Получение открытых позиций
            const openPositions = Array.from(this.currentPositions.values()).map(pos => ({
                id: pos.id,
                strategy: pos.strategy,
                symbol: pos.symbol,
                side: pos.side,
                size: pos.size,
                entryPrice: pos.entryPrice,
                currentPrice: pos.currentPrice,
                unrealizedPnL: pos.unrealizedPnL,
                returnPercent: pos.returnPercent,
                openTime: pos.openTime
            }));

            // Получение последних сделок
            const recentTrades = this.tradeHistory.slice(-10).map(trade => ({
                id: trade.id,
                strategy: trade.strategy,
                symbol: trade.symbol,
                side: trade.side,
                size: trade.size,
                entryPrice: trade.entryPrice,
                closePrice: trade.closePrice,
                realizedPnL: trade.realizedPnL,
                returnPercent: trade.returnPercent,
                openTime: trade.openTime,
                closeTime: trade.closeTime
            }));

            // Получение рыночных данных
            const marketDataArray = Array.from(this.marketData.values());

            return {
                success: true,
                running: this.isRunning,
                mode: this.mode,
                portfolio: this.portfolio,
                performance: this.performanceMetrics,
                riskMetrics: this.riskMetrics,
                strategies: strategiesArray,
                activeStrategiesCount: activeStrategies,
                openPositions: openPositions,
                recentTrades: recentTrades,
                marketData: marketDataArray,
                portfolioHistory: this.portfolioHistory.slice(-100), // Последние 100 точек
                timestamp: new Date().toISOString()
            };

        } catch (error) {
            console.error('❌ Error getting trading status:', error);
            return {
                success: false,
                error: error.message,
                timestamp: new Date().toISOString()
            };
        }
    }

    async getMarketData() {
        return {
            success: true,
            data: Array.from(this.marketData.values()),
            timestamp: new Date().toISOString()
        };
    }

    async getPortfolioHistory(days = 30) {
        const cutoffDate = new Date(Date.now() - days * 24 * 60 * 60 * 1000);
        const filteredHistory = this.portfolioHistory.filter(
            point => new Date(point.timestamp) >= cutoffDate
        );

        return {
            success: true,
            history: filteredHistory,
            timestamp: new Date().toISOString()
        };
    }

    async updateStrategyParameters(strategyId, parameters) {
        try {
            const strategy = this.strategies.get(strategyId);
            if (!strategy) {
                return {
                    success: false,
                    message: `Strategy ${strategyId} not found`,
                    timestamp: new Date().toISOString()
                };
            }

            strategy.parameters = { ...strategy.parameters, ...parameters };

            return {
                success: true,
                message: `Strategy ${strategy.name} parameters updated`,
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

    async closeAllPositions() {
        try {
            const positionIds = Array.from(this.currentPositions.keys());
            const closedPositions = [];

            for (const positionId of positionIds) {
                const closedPosition = this.closePosition(positionId);
                if (closedPosition) {
                    closedPositions.push(closedPosition);
                }
            }

            return {
                success: true,
                message: `Closed ${closedPositions.length} positions`,
                closedPositions: closedPositions,
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
}

module.exports = { TradingManager };