const fs = require('fs').promises;
const path = require('path');
require('dotenv').config();

class EnvironmentManager {
    constructor() {
        this.config = {
            trading: {
                mode: 'simulation', // simulation, paper, live
                maxPositions: 10,
                maxRiskPerTrade: 0.02, // 2%
                defaultLeverage: 1,
                autoClose: true
            },
            evolution: {
                enabled: true,
                populationSize: 50,
                generations: 100,
                mutationRate: 0.1,
                crossoverRate: 0.8
            },
            ml: {
                enabled: true,
                autoTraining: false,
                trainingInterval: 3600000, // 1 час
                modelRetention: 10,
                predictionThreshold: 0.7
            },
            system: {
                logLevel: 'info',
                maxLogSize: 1000000, // 1MB
                backupInterval: 86400000, // 24 часа
                performanceMonitoring: true
            },
            exchanges: {
                binance: {
                    enabled: true,
                    apiKey: '',
                    secretKey: '',
                    testnet: true
                },
                bybit: {
                    enabled: false,
                    apiKey: '',
                    secretKey: '',
                    testnet: true
                }
            }
        };
    }

    async getConfig() {
        return {
            success: true,
            config: this.config
        };
    }

    async saveConfig(newConfig) {
        try {
            // Валидация конфигурации
            const validation = this.validateConfig(newConfig);
            if (!validation.valid) {
                return { success: false, error: validation.error };
            }
            
            // Обновляем конфигурацию
            this.config = { ...this.config, ...newConfig };
            
            console.log('⚙️ Configuration saved successfully');
            
            return { success: true, message: 'Configuration saved successfully' };
        } catch (error) {
            console.error('Error saving configuration:', error);
            return { success: false, error: error.message };
        }
    }

    validateConfig(config) {
        // Валидация торговых настроек
        if (config.trading) {
            if (config.trading.maxRiskPerTrade > 0.1) {
                return { valid: false, error: 'Max risk per trade cannot exceed 10%' };
            }
            if (config.trading.maxPositions > 100) {
                return { valid: false, error: 'Max positions cannot exceed 100' };
            }
        }
        
        // Валидация эволюционных настроек
        if (config.evolution) {
            if (config.evolution.populationSize > 1000) {
                return { valid: false, error: 'Population size cannot exceed 1000' };
            }
            if (config.evolution.mutationRate > 1) {
                return { valid: false, error: 'Mutation rate cannot exceed 100%' };
            }
        }
        
        // Валидация ML настроек
        if (config.ml) {
            if (config.ml.predictionThreshold > 1) {
                return { valid: false, error: 'Prediction threshold cannot exceed 100%' };
            }
        }
        
        return { valid: true };
    }

    async resetToDefaults() {
        try {
            this.config = {
                trading: {
                    mode: 'simulation',
                    maxPositions: 10,
                    maxRiskPerTrade: 0.02,
                    defaultLeverage: 1,
                    autoClose: true
                },
                evolution: {
                    enabled: true,
                    populationSize: 50,
                    generations: 100,
                    mutationRate: 0.1,
                    crossoverRate: 0.8
                },
                ml: {
                    enabled: true,
                    autoTraining: false,
                    trainingInterval: 3600000,
                    modelRetention: 10,
                    predictionThreshold: 0.7
                },
                system: {
                    logLevel: 'info',
                    maxLogSize: 1000000,
                    backupInterval: 86400000,
                    performanceMonitoring: true
                },
                exchanges: {
                    binance: {
                        enabled: true,
                        apiKey: '',
                        secretKey: '',
                        testnet: true
                    },
                    bybit: {
                        enabled: false,
                        apiKey: '',
                        secretKey: '',
                        testnet: true
                    }
                }
            };
            
            console.log('⚙️ Configuration reset to defaults');
            
            return { success: true, message: 'Configuration reset to defaults' };
        } catch (error) {
            console.error('Error resetting configuration:', error);
            return { success: false, error: error.message };
        }
    }

    async getTradingMode() {
        return {
            success: true,
            mode: this.config.trading.mode
        };
    }

    async setTradingMode(mode) {
        if (!['simulation', 'paper', 'live'].includes(mode)) {
            return { success: false, error: 'Invalid trading mode' };
        }
        
        this.config.trading.mode = mode;
        
        console.log(`⚙️ Trading mode changed to: ${mode}`);
        
        return { success: true, message: `Trading mode changed to: ${mode}` };
    }

    async getExchangeConfig(exchange) {
        if (!this.config.exchanges[exchange]) {
            return { success: false, error: 'Exchange not found' };
        }
        
        return {
            success: true,
            config: this.config.exchanges[exchange]
        };
    }

    async updateExchangeConfig(exchange, config) {
        if (!this.config.exchanges[exchange]) {
            return { success: false, error: 'Exchange not found' };
        }
        
        this.config.exchanges[exchange] = { ...this.config.exchanges[exchange], ...config };
        
        console.log(`⚙️ Exchange config updated for: ${exchange}`);
        
        return { success: true, message: 'Exchange configuration updated' };
    }

    async getSystemInfo() {
        return {
            success: true,
            info: {
                platform: process.platform,
                arch: process.arch,
                nodeVersion: process.version,
                memory: process.memoryUsage(),
                uptime: process.uptime(),
                pid: process.pid
            }
        };
    }

    async getLogLevel() {
        return {
            success: true,
            logLevel: this.config.system.logLevel
        };
    }

    async setLogLevel(level) {
        const validLevels = ['error', 'warn', 'info', 'debug'];
        if (!validLevels.includes(level)) {
            return { success: false, error: 'Invalid log level' };
        }
        
        this.config.system.logLevel = level;
        
        console.log(`⚙️ Log level changed to: ${level}`);
        
        return { success: true, message: `Log level changed to: ${level}` };
    }
}

module.exports = { EnvironmentManager };