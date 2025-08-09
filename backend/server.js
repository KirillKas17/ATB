const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const path = require('path');

// Простой CORS middleware вместо пакета cors
const corsMiddleware = (req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
    res.header('Access-Control-Allow-Credentials', true);
    
    if (req.method === 'OPTIONS') {
        res.sendStatus(200);
    } else {
        next();
    }
};

class ATBBackendServer {
    constructor() {
        this.app = express();
        this.server = null;
        this.wss = null;
        this.port = process.env.BACKEND_PORT || 3001;
        this.clients = new Set();
        
        this.setupMiddleware();
        this.setupRoutes();
        this.setupWebSocket();
    }

    setupMiddleware() {
        // CORS
        this.app.use(corsMiddleware);

        // JSON парсер
        this.app.use(express.json({ limit: '10mb' }));
        this.app.use(express.urlencoded({ extended: true }));

        // Статические файлы
        this.app.use('/static', express.static(path.join(__dirname, '../renderer')));

        // Логирование запросов
        this.app.use((req, res, next) => {
            console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
            next();
        });
    }

    setupRoutes() {
        // Главная страница API
        this.app.get('/api', (req, res) => {
            res.json({
                name: 'ATB Trading System Backend',
                version: '3.1.0',
                status: 'running',
                timestamp: new Date().toISOString()
            });
        });

        // Здоровье системы
        this.app.get('/api/health', (req, res) => {
            res.json({
                status: 'healthy',
                uptime: process.uptime(),
                memory: process.memoryUsage(),
                version: process.version,
                platform: process.platform
            });
        });

        // Системные метрики - реальные данные от SystemMonitor
        this.app.get('/api/system/metrics', async (req, res) => {
            try {
                const { SystemMonitor } = require('./system-monitor');
                const systemMonitor = new SystemMonitor();
                const metrics = await systemMonitor.getMetrics();
                res.json(metrics);
            } catch (error) {
                console.error('Error getting system metrics:', error);
                res.status(500).json({ error: error.message });
            }
        });

        // Процессы системы
        this.app.get('/api/system/processes', async (req, res) => {
            try {
                const { SystemMonitor } = require('./system-monitor');
                const systemMonitor = new SystemMonitor();
                const processes = await systemMonitor.getProcesses();
                res.json(processes);
            } catch (error) {
                console.error('Error getting processes:', error);
                res.status(500).json({ error: error.message });
            }
        });

        // Температура CPU
        this.app.get('/api/system/temperature', async (req, res) => {
            try {
                const { SystemMonitor } = require('./system-monitor');
                const systemMonitor = new SystemMonitor();
                const temperature = await systemMonitor.getCPUTemperature();
                res.json(temperature);
            } catch (error) {
                console.error('Error getting temperature:', error);
                res.status(500).json({ error: error.message });
            }
        });

        // Эволюция стратегий - реальные данные от EvolutionManager
        this.app.get('/api/evolution/status', async (req, res) => {
            try {
                const { EvolutionManager } = require('./evolution-manager');
                const evolutionManager = new EvolutionManager();
                const status = await evolutionManager.getStatus();
                res.json(status);
            } catch (error) {
                console.error('Error getting evolution status:', error);
                res.status(500).json({ error: error.message });
            }
        });

        this.app.post('/api/evolution/start', async (req, res) => {
            try {
                const { EvolutionManager } = require('./evolution-manager');
                const evolutionManager = new EvolutionManager();
                const result = await evolutionManager.start();
                res.json(result);
            } catch (error) {
                console.error('Error starting evolution:', error);
                res.status(500).json({ error: error.message });
            }
        });

        this.app.post('/api/evolution/stop', async (req, res) => {
            try {
                const { EvolutionManager } = require('./evolution-manager');
                const evolutionManager = new EvolutionManager();
                const result = await evolutionManager.stop();
                res.json(result);
            } catch (error) {
                console.error('Error stopping evolution:', error);
                res.status(500).json({ error: error.message });
            }
        });

        // Торговые данные - реальные данные от TradingManager
        this.app.get('/api/trading/status', async (req, res) => {
            try {
                const { TradingManager } = require('./trading-manager');
                const tradingManager = new TradingManager();
                const status = await tradingManager.getStatus();
                res.json(status);
            } catch (error) {
                console.error('Error getting trading status:', error);
                res.status(500).json({ error: error.message });
            }
        });

        this.app.get('/api/trading/portfolio', async (req, res) => {
            try {
                const { TradingManager } = require('./trading-manager');
                const tradingManager = new TradingManager();
                const portfolio = await tradingManager.getPortfolio();
                res.json(portfolio);
            } catch (error) {
                console.error('Error getting portfolio:', error);
                res.status(500).json({ error: error.message });
            }
        });

        this.app.get('/api/trading/market-data', async (req, res) => {
            try {
                const { TradingManager } = require('./trading-manager');
                const tradingManager = new TradingManager();
                const marketData = await tradingManager.getMarketData();
                res.json(marketData);
            } catch (error) {
                console.error('Error getting market data:', error);
                res.status(500).json({ error: error.message });
            }
        });

        this.app.post('/api/trading/start', async (req, res) => {
            try {
                const { TradingManager } = require('./trading-manager');
                const tradingManager = new TradingManager();
                const result = await tradingManager.startTrading();
                res.json(result);
            } catch (error) {
                console.error('Error starting trading:', error);
                res.status(500).json({ error: error.message });
            }
        });

        this.app.post('/api/trading/stop', async (req, res) => {
            try {
                const { TradingManager } = require('./trading-manager');
                const tradingManager = new TradingManager();
                const result = await tradingManager.stopTrading();
                res.json(result);
            } catch (error) {
                console.error('Error stopping trading:', error);
                res.status(500).json({ error: error.message });
            }
        });

        // ML данные - реальные данные от MLManager
        this.app.get('/api/ml/status', async (req, res) => {
            try {
                const { MLManager } = require('./ml-manager');
                const mlManager = new MLManager();
                const status = await mlManager.getStatus();
                res.json(status);
            } catch (error) {
                console.error('Error getting ML status:', error);
                res.status(500).json({ error: error.message });
            }
        });

        this.app.get('/api/ml/models', async (req, res) => {
            try {
                const { MLManager } = require('./ml-manager');
                const mlManager = new MLManager();
                const models = await mlManager.getModels();
                res.json(models);
            } catch (error) {
                console.error('Error getting ML models:', error);
                res.status(500).json({ error: error.message });
            }
        });

        this.app.post('/api/ml/training/start', async (req, res) => {
            try {
                const { MLManager } = require('./ml-manager');
                const mlManager = new MLManager();
                const result = await mlManager.startTraining();
                res.json(result);
            } catch (error) {
                console.error('Error starting ML training:', error);
                res.status(500).json({ error: error.message });
            }
        });

        this.app.post('/api/ml/training/stop', async (req, res) => {
            try {
                const { MLManager } = require('./ml-manager');
                const mlManager = new MLManager();
                const result = await mlManager.stopTraining();
                res.json(result);
            } catch (error) {
                console.error('Error stopping ML training:', error);
                res.status(500).json({ error: error.message });
            }
        });

        // ENV конфигурация (заглушка - в реальности будет подключен EnvironmentManager)
        this.app.get('/api/env/config', (req, res) => {
            res.json({
                success: true,
                config: {
                    NODE_ENV: 'production',
                    ATB_MODE: 'simulation',
                    MONITORING_ENABLED: 'true',
                    EVOLUTION_ENABLED: 'true'
                }
            });
        });

        this.app.post('/api/env/config', (req, res) => {
            const { config } = req.body;
            
            res.json({
                success: true,
                message: 'Configuration saved',
                timestamp: new Date().toISOString()
            });
        });

        this.app.post('/api/env/reset', (req, res) => {
            res.json({
                success: true,
                message: 'Configuration reset to defaults',
                timestamp: new Date().toISOString()
            });
        });

        // Торговля
        this.app.post('/api/trading/start', (req, res) => {
            res.json({
                success: true,
                message: 'Trading started',
                timestamp: new Date().toISOString()
            });
        });

        this.app.post('/api/trading/stop', (req, res) => {
            res.json({
                success: true,
                message: 'Trading stopped',
                timestamp: new Date().toISOString()
            });
        });

        this.app.get('/api/trading/status', (req, res) => {
            res.json({
                active: Math.random() > 0.5,
                mode: 'simulation',
                balance: 10000 + Math.random() * 1000,
                pnl: (Math.random() - 0.5) * 500,
                positions: Math.floor(Math.random() * 5),
                timestamp: new Date().toISOString()
            });
        });

        // Портфель
        this.app.get('/api/portfolio', (req, res) => {
            res.json({
                total_balance: 10000 + Math.random() * 1000,
                available_balance: 9000 + Math.random() * 500,
                positions: [
                    {
                        symbol: 'BTCUSDT',
                        size: Math.random() * 0.1,
                        entry_price: 45000 + Math.random() * 5000,
                        current_price: 45000 + Math.random() * 5000,
                        pnl: (Math.random() - 0.5) * 200
                    }
                ],
                orders: [],
                timestamp: new Date().toISOString()
            });
        });

        // Логи
        this.app.get('/api/logs', (req, res) => {
            const { limit = 50 } = req.query;
            
            const logs = [];
            for (let i = 0; i < Math.min(limit, 20); i++) {
                logs.push({
                    timestamp: new Date(Date.now() - i * 60000).toISOString(),
                    level: ['info', 'warning', 'error'][Math.floor(Math.random() * 3)],
                    message: `Sample log message ${i + 1}`
                });
            }
            
            res.json({ logs });
        });

        // Обработка ошибок 404
        this.app.use((req, res) => {
            res.status(404).json({
                error: 'Not Found',
                message: `Route ${req.method} ${req.path} not found`,
                timestamp: new Date().toISOString()
            });
        });

        // Обработка ошибок
        this.app.use((error, req, res, next) => {
            console.error('Server error:', error);
            res.status(500).json({
                error: 'Internal Server Error',
                message: error.message,
                timestamp: new Date().toISOString()
            });
        });
    }

    setupWebSocket() {
        // WebSocket будет настроен при запуске сервера
    }

    async start() {
        try {
            // Создание HTTP сервера
            this.server = http.createServer(this.app);

            // Настройка WebSocket
            this.wss = new WebSocket.Server({ server: this.server });

            this.wss.on('connection', (ws, req) => {
                console.log(`📡 WebSocket клиент подключен: ${req.connection.remoteAddress}`);
                this.clients.add(ws);

                // Отправка приветственного сообщения
                ws.send(JSON.stringify({
                    type: 'welcome',
                    message: 'Connected to ATB Trading System Backend',
                    timestamp: new Date().toISOString()
                }));

                // Обработка сообщений от клиента
                ws.on('message', (message) => {
                    try {
                        const data = JSON.parse(message);
                        this.handleWebSocketMessage(ws, data);
                    } catch (error) {
                        console.error('WebSocket message error:', error);
                    }
                });

                // Обработка отключения
                ws.on('close', () => {
                    console.log('📡 WebSocket клиент отключен');
                    this.clients.delete(ws);
                });

                // Обработка ошибок
                ws.on('error', (error) => {
                    console.error('WebSocket error:', error);
                    this.clients.delete(ws);
                });
            });

            // Запуск сервера
            await new Promise((resolve, reject) => {
                this.server.listen(this.port, (error) => {
                    if (error) {
                        reject(error);
                    } else {
                        resolve();
                    }
                });
            });

            console.log(`✅ ATB Backend Server запущен на порту ${this.port}`);
            console.log(`📡 WebSocket сервер готов к подключениям`);

            // Запуск периодических обновлений
            this.startPeriodicUpdates();

            return {
                success: true,
                port: this.port,
                message: 'Server started successfully'
            };

        } catch (error) {
            console.error('❌ Ошибка запуска сервера:', error);
            throw error;
        }
    }

    async stop() {
        try {
            // Закрытие WebSocket соединений
            if (this.wss) {
                this.wss.close();
                this.clients.clear();
            }

            // Остановка HTTP сервера
            if (this.server) {
                await new Promise((resolve) => {
                    this.server.close(resolve);
                });
            }

            // Остановка периодических обновлений
            if (this.updateInterval) {
                clearInterval(this.updateInterval);
            }

            console.log('✅ ATB Backend Server остановлен');

            return {
                success: true,
                message: 'Server stopped successfully'
            };

        } catch (error) {
            console.error('❌ Ошибка остановки сервера:', error);
            throw error;
        }
    }

    handleWebSocketMessage(ws, data) {
        const { type, payload } = data;

        switch (type) {
            case 'ping':
                ws.send(JSON.stringify({
                    type: 'pong',
                    timestamp: new Date().toISOString()
                }));
                break;

            case 'subscribe':
                // Подписка на обновления
                ws.subscriptions = ws.subscriptions || new Set();
                if (payload && payload.channel) {
                    ws.subscriptions.add(payload.channel);
                    console.log(`📡 Клиент подписался на канал: ${payload.channel}`);
                }
                break;

            case 'unsubscribe':
                // Отписка от обновлений
                if (ws.subscriptions && payload && payload.channel) {
                    ws.subscriptions.delete(payload.channel);
                    console.log(`📡 Клиент отписался от канала: ${payload.channel}`);
                }
                break;

            default:
                console.warn(`⚠️ Неизвестный тип WebSocket сообщения: ${type}`);
        }
    }

    startPeriodicUpdates() {
        // Обновления каждые 5 секунд
        this.updateInterval = setInterval(() => {
            this.broadcastUpdate();
        }, 5000);
    }

    broadcastUpdate() {
        if (this.clients.size === 0) return;

        const updates = {
            system: {
                cpu: Math.round(Math.random() * 100),
                memory: Math.round(Math.random() * 100),
                timestamp: new Date().toISOString()
            },
            trading: {
                pnl: (Math.random() - 0.5) * 100,
                timestamp: new Date().toISOString()
            }
        };

        const message = JSON.stringify({
            type: 'update',
            payload: updates,
            timestamp: new Date().toISOString()
        });

        this.clients.forEach(ws => {
            if (ws.readyState === WebSocket.OPEN) {
                try {
                    ws.send(message);
                } catch (error) {
                    console.error('WebSocket send error:', error);
                    this.clients.delete(ws);
                }
            }
        });
    }

    getStats() {
        return {
            server: {
                running: !!this.server,
                port: this.port,
                uptime: process.uptime()
            },
            websocket: {
                connected_clients: this.clients.size,
                running: !!this.wss
            },
            memory: process.memoryUsage(),
            timestamp: new Date().toISOString()
        };
    }
}

// Функция для запуска сервера (используется в main.js)
async function startBackendServer() {
    const server = new ATBBackendServer();
    await server.start();
    return server;
}

module.exports = { ATBBackendServer, startBackendServer };

// Запуск сервера если файл запущен напрямую
if (require.main === module) {
    startBackendServer()
        .then(server => {
            console.log('🚀 ATB Backend Server запущен на порту', server.port);
        })
        .catch(error => {
            console.error('❌ Ошибка запуска сервера:', error);
            process.exit(1);
        });
}