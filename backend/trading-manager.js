const fs = require('fs').promises;
const path = require('path');
const { spawn } = require('child_process');

class TradingManager {
    constructor() {
        this.isRunning = false;
        this.exchangeConnected = false;
        this.portfolio = {
            balance: 0,
            equity: 0,
            margin: 0,
            freeMargin: 0
        };
        this.positions = [];
        this.orders = [];
        this.tradeHistory = [];
        this.marketData = {};
    }

    async getStatus() {
        return {
            running: this.isRunning,
            exchangeConnected: this.exchangeConnected,
            portfolio: this.portfolio,
            positions: this.positions,
            orders: this.orders,
            marketData: this.marketData,
            status: this.exchangeConnected ? 'connected' : 'disconnected',
            message: this.exchangeConnected ? 'Биржа подключена' : 'Биржа не подключена'
        };
    }

    async startTrading() {
        if (this.isRunning) {
            return { success: false, error: 'Trading already running' };
        }

        this.isRunning = true;
        console.log('💰 Trading started');
        
        // Запускаем торговый цикл
        this.tradingLoop();
        
        return { success: true, message: 'Trading started successfully' };
    }

    async stopTrading() {
        if (!this.isRunning) {
            return { success: false, error: 'Trading not running' };
        }

        this.isRunning = false;
        console.log('💰 Trading stopped');
        
        return { success: true, message: 'Trading stopped successfully' };
    }

    async getPortfolio() {
        if (!this.exchangeConnected) {
            return {
                balance: 0,
                equity: 0,
                margin: 0,
                freeMargin: 0,
                pnl: 0,
                dailyPnL: 0,
                monthlyPnL: 0,
                status: 'disconnected',
                message: 'Биржа не подключена'
            };
        }
        
        return {
            balance: this.portfolio.balance,
            equity: this.portfolio.equity,
            margin: this.portfolio.margin,
            freeMargin: this.portfolio.freeMargin,
            pnl: this.calculateTotalPnL(),
            dailyPnL: this.calculateDailyPnL(),
            monthlyPnL: this.calculateMonthlyPnL(),
            status: 'connected'
        };
    }

    async getMarketData() {
        // Симуляция рыночных данных
        const symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT'];
        const marketData = {};
        
        for (let symbol of symbols) {
            const basePrice = this.getBasePrice(symbol);
            const change = (Math.random() - 0.5) * 0.1; // ±5%
            const price = basePrice * (1 + change);
            const volume = Math.random() * 1000000 + 100000;
            
            marketData[symbol] = {
                symbol,
                price: price.toFixed(2),
                change: (change * 100).toFixed(2),
                volume: volume.toFixed(0),
                high: (price * 1.02).toFixed(2),
                low: (price * 0.98).toFixed(2)
            };
        }
        
        this.marketData = marketData;
        return marketData;
    }

    getBasePrice(symbol) {
        const basePrices = {
            'BTCUSDT': 45000,
            'ETHUSDT': 3000,
            'BNBUSDT': 300,
            'ADAUSDT': 0.5,
            'DOTUSDT': 7
        };
        return basePrices[symbol] || 100;
    }

    calculateTotalPnL() {
        return this.positions.reduce((total, pos) => total + pos.pnl, 0);
    }

    calculateDailyPnL() {
        const today = new Date().toDateString();
        const dailyTrades = this.tradeHistory.filter(trade => 
            new Date(trade.timestamp).toDateString() === today
        );
        return dailyTrades.reduce((total, trade) => total + trade.pnl, 0);
    }

    calculateMonthlyPnL() {
        const now = new Date();
        const monthStart = new Date(now.getFullYear(), now.getMonth(), 1);
        const monthlyTrades = this.tradeHistory.filter(trade => 
            new Date(trade.timestamp) >= monthStart
        );
        return monthlyTrades.reduce((total, trade) => total + trade.pnl, 0);
    }

    async tradingLoop() {
        while (this.isRunning) {
            try {
                // Обновляем рыночные данные
                await this.getMarketData();
                
                // Обновляем позиции
                this.updatePositions();
                
                // Генерируем новые сигналы
                this.generateSignals();
                
                // Пауза между циклами
                await this.sleep(5000); // 5 секунд
                
            } catch (error) {
                console.error('Error in trading loop:', error);
                await this.sleep(1000);
            }
        }
    }

    updatePositions() {
        for (let position of this.positions) {
            const marketPrice = parseFloat(this.marketData[position.symbol]?.price || position.entryPrice);
            const priceChange = (marketPrice - position.entryPrice) / position.entryPrice;
            
            if (position.side === 'long') {
                position.pnl = position.size * priceChange;
            } else {
                position.pnl = position.size * (-priceChange);
            }
            
            position.currentPrice = marketPrice;
        }
        
        // Обновляем equity
        this.portfolio.equity = this.portfolio.balance + this.calculateTotalPnL();
    }

    generateSignals() {
        // Симуляция генерации торговых сигналов
        if (Math.random() < 0.1) { // 10% вероятность сигнала
            const symbols = Object.keys(this.marketData);
            const symbol = symbols[Math.floor(Math.random() * symbols.length)];
            const side = Math.random() > 0.5 ? 'long' : 'short';
            const size = Math.random() * 0.1 + 0.01; // 1-11% от баланса
            
            this.openPosition(symbol, side, size);
        }
    }

    openPosition(symbol, side, size) {
        const price = parseFloat(this.marketData[symbol]?.price || this.getBasePrice(symbol));
        const positionValue = this.portfolio.balance * size;
        
        const position = {
            id: `pos_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            symbol,
            side,
            size: positionValue,
            entryPrice: price,
            currentPrice: price,
            pnl: 0,
            timestamp: new Date().toISOString()
        };
        
        this.positions.push(position);
        
        // Создаем запись в истории
        this.tradeHistory.push({
            id: position.id,
            symbol,
            side,
            size: positionValue,
            price,
            pnl: 0,
            timestamp: new Date().toISOString(),
            type: 'open'
        });
        
        console.log(`💰 Opened ${side} position on ${symbol} at ${price}`);
    }

    closePosition(positionId) {
        const positionIndex = this.positions.findIndex(p => p.id === positionId);
        if (positionIndex === -1) return;
        
        const position = this.positions[positionIndex];
        const currentPrice = parseFloat(this.marketData[position.symbol]?.price || position.entryPrice);
        
        // Обновляем баланс
        this.portfolio.balance += position.pnl;
        this.portfolio.equity = this.portfolio.balance;
        
        // Создаем запись в истории
        this.tradeHistory.push({
            id: position.id,
            symbol: position.symbol,
            side: position.side,
            size: position.size,
            price: currentPrice,
            pnl: position.pnl,
            timestamp: new Date().toISOString(),
            type: 'close'
        });
        
        // Удаляем позицию
        this.positions.splice(positionIndex, 1);
        
        console.log(`💰 Closed position on ${position.symbol}, PnL: ${position.pnl.toFixed(2)}`);
    }

    async runBacktest() {
        console.log('📊 Starting backtest...');
        
        // Симуляция бэктеста
        await this.sleep(2000);
        
        const results = {
            totalTrades: Math.floor(Math.random() * 100) + 50,
            winningTrades: Math.floor(Math.random() * 80) + 30,
            losingTrades: Math.floor(Math.random() * 40) + 10,
            totalPnL: (Math.random() - 0.5) * 2000,
            winRate: Math.random() * 0.4 + 0.6, // 60-100%
            sharpeRatio: Math.random() * 2 + 0.5, // 0.5-2.5
            maxDrawdown: Math.random() * 0.2 + 0.05 // 5-25%
        };
        
        console.log('📊 Backtest completed');
        return { success: true, results };
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

module.exports = { TradingManager };