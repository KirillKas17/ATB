const { contextBridge, ipcRenderer } = require('electron');

// Создание безопасного API для рендер процесса
contextBridge.exposeInMainWorld('electronAPI', {
    // Системные метрики
    getSystemMetrics: () => ipcRenderer.invoke('get-system-metrics'),
    
    // Эволюция стратегий
    getEvolutionStatus: () => ipcRenderer.invoke('get-evolution-status'),
    startEvolution: () => ipcRenderer.invoke('start-evolution'),
    stopEvolution: () => ipcRenderer.invoke('stop-evolution'),
    forceEvolution: (strategyId) => ipcRenderer.invoke('force-evolution', strategyId),
    getStrategyDetails: (strategyId) => ipcRenderer.invoke('get-strategy-details', strategyId),
    updateEvolutionConfig: (config) => ipcRenderer.invoke('update-evolution-config', config),
    
    // Торговля
    getTradingStatus: () => ipcRenderer.invoke('get-trading-status'),
    startTrading: (mode) => ipcRenderer.invoke('start-trading', mode),
    stopTrading: () => ipcRenderer.invoke('stop-trading'),
    runBacktest: (parameters) => ipcRenderer.invoke('run-backtest', parameters),
    getMarketData: () => ipcRenderer.invoke('get-market-data'),
    getPortfolioHistory: (days) => ipcRenderer.invoke('get-portfolio-history', days),
    closeAllPositions: () => ipcRenderer.invoke('close-all-positions'),
    updateStrategyParameters: (strategyId, parameters) => ipcRenderer.invoke('update-strategy-parameters', strategyId, parameters),
    
    // Машинное обучение
    getMLStatus: () => ipcRenderer.invoke('get-ml-status'),
    startMLTraining: (modelId, config) => ipcRenderer.invoke('start-ml-training', modelId, config),
    stopMLTraining: (modelId) => ipcRenderer.invoke('stop-ml-training', modelId),
    runMLEvaluation: (modelId) => ipcRenderer.invoke('run-ml-evaluation', modelId),
    getModelDetails: (modelId) => ipcRenderer.invoke('get-model-details', modelId),
    updateModelHyperparameters: (modelId, hyperparameters) => ipcRenderer.invoke('update-model-hyperparameters', modelId, hyperparameters),
    deployModel: (modelId, environment) => ipcRenderer.invoke('deploy-model', modelId, environment),
    generatePrediction: (modelId, inputData) => ipcRenderer.invoke('generate-prediction', modelId, inputData),
    getTrainingLogs: (jobId) => ipcRenderer.invoke('get-training-logs', jobId),
    
    // .env конфигурация
    getEnvConfig: () => ipcRenderer.invoke('get-env-config'),
    saveEnvConfig: (config) => ipcRenderer.invoke('save-env-config', config),
    resetEnvConfig: () => ipcRenderer.invoke('reset-env-config'),
    
    // Торговля
    startTrading: () => ipcRenderer.invoke('start-trading'),
    stopTrading: () => ipcRenderer.invoke('stop-trading'),
    
    // Диалоги
    showMessageBox: (options) => ipcRenderer.invoke('show-message-box', options),
    showSaveDialog: (options) => ipcRenderer.invoke('show-save-dialog', options),
    showOpenDialog: (options) => ipcRenderer.invoke('show-open-dialog', options),
    
    // Уведомления
    showNotification: (title, body) => ipcRenderer.invoke('show-notification', title, body),
    
    // Логирование
    log: (level, message) => ipcRenderer.send('log', level, message),
    
    // Подписка на события
    onNavigateToTab: (callback) => ipcRenderer.on('navigate-to-tab', callback),
    onStartTrading: (callback) => ipcRenderer.on('start-trading', callback),
    onStopTrading: (callback) => ipcRenderer.on('stop-trading', callback),
    onImportData: (callback) => ipcRenderer.on('import-data', callback),
    onExportData: (callback) => ipcRenderer.on('export-data', callback),
    
    // Отписка от событий
    removeAllListeners: (channel) => ipcRenderer.removeAllListeners(channel)
});

// Дополнительный API для разработки
if (process.env.NODE_ENV === 'development') {
    contextBridge.exposeInMainWorld('devAPI', {
        openDevTools: () => ipcRenderer.send('open-dev-tools'),
        reload: () => ipcRenderer.send('reload'),
        forceReload: () => ipcRenderer.send('force-reload')
    });
}

// Глобальные переменные для рендер процесса
contextBridge.exposeInMainWorld('appInfo', {
    name: 'ATB Trading System Enhanced',
    version: '3.1.0',
    platform: process.platform,
    isElectron: true,
    isDev: process.env.NODE_ENV === 'development'
});

console.log('⚡ ATB Trading System - Preload script loaded');