const { contextBridge, ipcRenderer } = require('electron');

// Безопасный API для renderer процесса
contextBridge.exposeInMainWorld('electronAPI', {
    // Информация о приложении
    getAppVersion: () => ipcRenderer.invoke('get-app-version'),
    getAppPath: () => ipcRenderer.invoke('get-app-path'),
    
    // Диалоги файлов
    showSaveDialog: (options) => ipcRenderer.invoke('show-save-dialog', options),
    showOpenDialog: (options) => ipcRenderer.invoke('show-open-dialog', options),
    
    // Работа с файлами
    readFile: (filePath) => ipcRenderer.invoke('read-file', filePath),
    writeFile: (filePath, data) => ipcRenderer.invoke('write-file', filePath, data),
    
    // События от главного процесса
    onExportData: (callback) => ipcRenderer.on('export-data', callback),
    onOpenSettings: (callback) => ipcRenderer.on('open-settings', callback),
    onStartBot: (callback) => ipcRenderer.on('start-bot', callback),
    onStopBot: (callback) => ipcRenderer.on('stop-bot', callback),
    onNewTrade: (callback) => ipcRenderer.on('new-trade', callback),
    onRefreshData: (callback) => ipcRenderer.on('refresh-data', callback),
    onExportReport: (callback) => ipcRenderer.on('export-report', callback),
    onCheckUpdates: (callback) => ipcRenderer.on('check-updates', callback),
    
    // Удаление слушателей
    removeAllListeners: (channel) => ipcRenderer.removeAllListeners(channel)
});

// Глобальные переменные для приложения
contextBridge.exposeInMainWorld('appConfig', {
    isDev: process.env.NODE_ENV === 'development',
    platform: process.platform,
    version: process.versions.electron
}); 