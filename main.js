const { app, BrowserWindow, Menu, Tray, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const isDev = process.env.NODE_ENV === 'development';

// Импорт бэкенда
let startBackendServer, SystemMonitor, EvolutionManager, EnvironmentManager;

try {
    ({ startBackendServer } = require('./backend/server'));
    ({ SystemMonitor } = require('./backend/system-monitor'));
    ({ EnvironmentManager } = require('./backend/environment-manager'));
    
    // EvolutionManager заглушка пока не создан
    EvolutionManager = class {
        async getStatus() {
            return {
                enabled: true,
                running: false,
                strategies: [],
                timestamp: new Date().toISOString()
            };
        }
        async start() { return { success: true }; }
        async stop() { return { success: true }; }
    };
} catch (error) {
    console.warn('⚠️ Некоторые backend модули недоступны:', error.message);
    
    // Заглушки для отсутствующих модулей
    startBackendServer = async () => ({ success: false, error: 'Backend not available' });
    SystemMonitor = class {
        async getMetrics() { return {}; }
        async getProcesses() { return []; }
    };
    EvolutionManager = class {
        async getStatus() { return { enabled: false }; }
        async start() { return { success: false }; }
        async stop() { return { success: false }; }
    };
    EnvironmentManager = class {
        async getConfig() { return { success: false }; }
        async saveConfig() { return { success: false }; }
        async resetToDefaults() { return { success: false }; }
    };
}

class ATBDesktopApp {
    constructor() {
        this.mainWindow = null;
        this.tray = null;
        this.backendServer = null;
        this.systemMonitor = null;
        this.evolutionManager = null;
        this.environmentManager = null;
    }

    async init() {
        // Инициализация бэкенда
        this.environmentManager = new EnvironmentManager();
        this.systemMonitor = new SystemMonitor();
        this.evolutionManager = new EvolutionManager();
        
        // Запуск бэкенд сервера
        this.backendServer = await startBackendServer();
        
        console.log('🚀 ATB Trading System Enhanced Desktop v3.1 - Starting...');
    }

    createMainWindow() {
        // Создание главного окна
        this.mainWindow = new BrowserWindow({
            width: 1920,
            height: 1200,
            minWidth: 1400,
            minHeight: 900,
            icon: path.join(__dirname, 'assets', 'icon.png'),
            webPreferences: {
                nodeIntegration: false,
                contextIsolation: true,
                enableRemoteModule: false,
                preload: path.join(__dirname, 'preload.js')
            },
            titleBarStyle: 'default',
            frame: true,
            show: false,
            backgroundColor: '#0a0a0a'
        });

        // Загрузка главной страницы
        const startUrl = isDev 
            ? 'http://localhost:3000' 
            : `file://${path.join(__dirname, 'renderer/index.html')}`;
        
        this.mainWindow.loadURL(startUrl);

        // Показать окно когда готово
        this.mainWindow.once('ready-to-show', () => {
            this.mainWindow.show();
            
            if (isDev) {
                this.mainWindow.webContents.openDevTools();
            }
        });

        // Обработка закрытия
        this.mainWindow.on('close', (event) => {
            if (!app.isQuiting) {
                event.preventDefault();
                this.mainWindow.hide();
                this.showTrayNotification('ATB Trading System', 'Приложение свернуто в трей');
            }
        });

        this.mainWindow.on('closed', () => {
            this.mainWindow = null;
        });

        // Обработка внешних ссылок
        this.mainWindow.webContents.setWindowOpenHandler(({ url }) => {
            shell.openExternal(url);
            return { action: 'deny' };
        });
    }

    createTray() {
        // Создание системного трея
        const iconPath = path.join(__dirname, 'assets', 'tray-icon.png');
        this.tray = new Tray(iconPath);

        const contextMenu = Menu.buildFromTemplate([
            {
                label: '⚡ ATB Trading System v3.1',
                enabled: false
            },
            { type: 'separator' },
            {
                label: '📊 Показать дашборд',
                click: () => {
                    if (this.mainWindow) {
                        this.mainWindow.show();
                        this.mainWindow.focus();
                    }
                }
            },
            {
                label: '🖥️ Системные метрики',
                click: () => {
                    this.showSystemMetrics();
                }
            },
            {
                label: '🧬 Эволюция стратегий',
                click: () => {
                    this.showEvolutionStatus();
                }
            },
            {
                label: '🔧 Настройки .env',
                click: () => {
                    this.openEnvSettings();
                }
            },
            { type: 'separator' },
            {
                label: '🚪 Выход',
                click: () => {
                    app.isQuiting = true;
                    app.quit();
                }
            }
        ]);

        this.tray.setContextMenu(contextMenu);
        this.tray.setToolTip('ATB Trading System Enhanced v3.1');

        // Обработка двойного клика по трею
        this.tray.on('double-click', () => {
            if (this.mainWindow) {
                this.mainWindow.show();
                this.mainWindow.focus();
            }
        });
    }

    createMenu() {
        const template = [
            {
                label: '📁 Файл',
                submenu: [
                    {
                        label: '🔧 Настройки окружения',
                        click: () => this.openEnvSettings()
                    },
                    { type: 'separator' },
                    {
                        label: '📥 Импорт данных',
                        click: () => this.importData()
                    },
                    {
                        label: '📤 Экспорт данных',
                        click: () => this.exportData()
                    },
                    { type: 'separator' },
                    {
                        label: '🚪 Выход',
                        accelerator: 'CmdOrCtrl+Q',
                        click: () => {
                            app.isQuiting = true;
                            app.quit();
                        }
                    }
                ]
            },
            {
                label: '🖥️ Система',
                submenu: [
                    {
                        label: '📊 Системные метрики',
                        click: () => this.showSystemMetrics()
                    },
                    {
                        label: '⚙️ Процессы',
                        click: () => this.showProcesses()
                    },
                    { type: 'separator' },
                    {
                        label: '🔄 Перезагрузить',
                        accelerator: 'CmdOrCtrl+R',
                        click: () => this.mainWindow.reload()
                    }
                ]
            },
            {
                label: '🧬 Эволюция',
                submenu: [
                    {
                        label: '📈 Статус эволюции',
                        click: () => this.showEvolutionStatus()
                    },
                    {
                        label: '▶️ Запустить эволюцию',
                        click: () => this.startEvolution()
                    },
                    {
                        label: '⏹️ Остановить эволюцию',
                        click: () => this.stopEvolution()
                    }
                ]
            },
            {
                label: '💼 Торговля',
                submenu: [
                    {
                        label: '▶️ Запустить торговлю',
                        click: () => this.startTrading()
                    },
                    {
                        label: '⏹️ Остановить торговлю',
                        click: () => this.stopTrading()
                    },
                    { type: 'separator' },
                    {
                        label: '📊 Портфель',
                        click: () => this.showPortfolio()
                    }
                ]
            },
            {
                label: '❓ Справка',
                submenu: [
                    {
                        label: '📖 Документация',
                        click: () => shell.openExternal('https://github.com/atb-trading/docs')
                    },
                    {
                        label: '🐛 Сообщить об ошибке',
                        click: () => shell.openExternal('https://github.com/atb-trading/issues')
                    },
                    { type: 'separator' },
                    {
                        label: 'ℹ️ О программе',
                        click: () => this.showAbout()
                    }
                ]
            }
        ];

        // Дополнительное меню для разработки
        if (isDev) {
            template.push({
                label: '🔧 Разработка',
                submenu: [
                    {
                        label: '🔍 Инструменты разработчика',
                        accelerator: 'F12',
                        click: () => this.mainWindow.webContents.openDevTools()
                    },
                    {
                        label: '🔄 Принудительная перезагрузка',
                        accelerator: 'CmdOrCtrl+Shift+R',
                        click: () => this.mainWindow.webContents.reloadIgnoringCache()
                    }
                ]
            });
        }

        const menu = Menu.buildFromTemplate(template);
        Menu.setApplicationMenu(menu);
    }

    setupIPC() {
        // IPC обработчики для связи с рендер процессом

        // Системные метрики
        ipcMain.handle('get-system-metrics', async () => {
            return await this.systemMonitor.getMetrics();
        });

        // Эволюция стратегий
        ipcMain.handle('get-evolution-status', async () => {
            return await this.evolutionManager.getStatus();
        });

        ipcMain.handle('start-evolution', async () => {
            return await this.evolutionManager.start();
        });

        ipcMain.handle('stop-evolution', async () => {
            return await this.evolutionManager.stop();
        });

        // .env управление
        ipcMain.handle('get-env-config', async () => {
            return await this.environmentManager.getConfig();
        });

        ipcMain.handle('save-env-config', async (event, config) => {
            return await this.environmentManager.saveConfig(config);
        });

        ipcMain.handle('reset-env-config', async () => {
            return await this.environmentManager.resetToDefaults();
        });

        // Торговля
        ipcMain.handle('start-trading', async () => {
            return await this.startTradingSystem();
        });

        ipcMain.handle('stop-trading', async () => {
            return await this.stopTradingSystem();
        });

        // Диалоги
        ipcMain.handle('show-message-box', async (event, options) => {
            const result = await dialog.showMessageBox(this.mainWindow, options);
            return result;
        });

        ipcMain.handle('show-save-dialog', async (event, options) => {
            const result = await dialog.showSaveDialog(this.mainWindow, options);
            return result;
        });

        ipcMain.handle('show-open-dialog', async (event, options) => {
            const result = await dialog.showOpenDialog(this.mainWindow, options);
            return result;
        });

        // Уведомления
        ipcMain.handle('show-notification', (event, title, body) => {
            this.showTrayNotification(title, body);
        });

        // Логирование
        ipcMain.on('log', (event, level, message) => {
            console.log(`[${level.toUpperCase()}] ${message}`);
        });
    }

    // Методы для действий меню
    async showSystemMetrics() {
        if (this.mainWindow) {
            this.mainWindow.show();
            this.mainWindow.focus();
            this.mainWindow.webContents.send('navigate-to-tab', 'system');
        }
    }

    async showEvolutionStatus() {
        if (this.mainWindow) {
            this.mainWindow.show();
            this.mainWindow.focus();
            this.mainWindow.webContents.send('navigate-to-tab', 'evolution');
        }
    }

    async openEnvSettings() {
        if (this.mainWindow) {
            this.mainWindow.show();
            this.mainWindow.focus();
            this.mainWindow.webContents.send('navigate-to-tab', 'settings');
        }
    }

    async showPortfolio() {
        if (this.mainWindow) {
            this.mainWindow.show();
            this.mainWindow.focus();
            this.mainWindow.webContents.send('navigate-to-tab', 'portfolio');
        }
    }

    async showProcesses() {
        const processes = await this.systemMonitor.getProcesses();
        dialog.showMessageBox(this.mainWindow, {
            type: 'info',
            title: 'Системные процессы',
            message: 'Топ процессы по CPU',
            detail: processes.map(p => `${p.name}: ${p.cpu}%`).join('\n')
        });
    }

    async startEvolution() {
        try {
            await this.evolutionManager.start();
            this.showTrayNotification('Эволюция', 'Эволюция стратегий запущена');
        } catch (error) {
            dialog.showErrorBox('Ошибка', `Не удалось запустить эволюцию: ${error.message}`);
        }
    }

    async stopEvolution() {
        try {
            await this.evolutionManager.stop();
            this.showTrayNotification('Эволюция', 'Эволюция стратегий остановлена');
        } catch (error) {
            dialog.showErrorBox('Ошибка', `Не удалось остановить эволюцию: ${error.message}`);
        }
    }

    async startTrading() {
        if (this.mainWindow) {
            this.mainWindow.webContents.send('start-trading');
        }
    }

    async stopTrading() {
        if (this.mainWindow) {
            this.mainWindow.webContents.send('stop-trading');
        }
    }

    async importData() {
        const result = await dialog.showOpenDialog(this.mainWindow, {
            title: 'Импорт данных',
            filters: [
                { name: 'JSON файлы', extensions: ['json'] },
                { name: 'CSV файлы', extensions: ['csv'] },
                { name: 'Все файлы', extensions: ['*'] }
            ]
        });

        if (!result.canceled && result.filePaths.length > 0) {
            this.mainWindow.webContents.send('import-data', result.filePaths[0]);
        }
    }

    async exportData() {
        const result = await dialog.showSaveDialog(this.mainWindow, {
            title: 'Экспорт данных',
            defaultPath: `atb-export-${new Date().toISOString().split('T')[0]}.json`,
            filters: [
                { name: 'JSON файлы', extensions: ['json'] },
                { name: 'CSV файлы', extensions: ['csv'] }
            ]
        });

        if (!result.canceled && result.filePath) {
            this.mainWindow.webContents.send('export-data', result.filePath);
        }
    }

    showAbout() {
        dialog.showMessageBox(this.mainWindow, {
            type: 'info',
            title: 'О программе',
            message: '⚡ ATB Trading System Enhanced v3.1',
            detail: '🚀 Современная торговая система на Electron\n\n' +
                   '✨ Возможности:\n' +
                   '• 🤖 Автоматическая торговля с ИИ\n' +
                   '• 📊 Мониторинг системных ресурсов\n' +
                   '• 🧬 Эволюция стратегий в реальном времени\n' +
                   '• 🔧 Управление .env конфигурацией\n' +
                   '• 💼 Управление портфелем\n' +
                   '• 🔮 ML прогнозирование\n\n' +
                   '© 2024 ATB Trading Team'
        });
    }

    showTrayNotification(title, body) {
        if (this.tray) {
            this.tray.displayBalloon({
                title: title,
                content: body,
                icon: path.join(__dirname, 'assets', 'icon.png')
            });
        }
    }

    async startTradingSystem() {
        // Интеграция с существующей торговой системой
        console.log('🚀 Starting trading system...');
        return { success: true, message: 'Trading system started' };
    }

    async stopTradingSystem() {
        // Остановка торговой системы
        console.log('⏹️ Stopping trading system...');
        return { success: true, message: 'Trading system stopped' };
    }
}

// Инициализация приложения
const atbApp = new ATBDesktopApp();

app.whenReady().then(async () => {
    await atbApp.init();
    atbApp.createMainWindow();
    atbApp.createTray();
    atbApp.createMenu();
    atbApp.setupIPC();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            atbApp.createMainWindow();
        }
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('before-quit', () => {
    app.isQuiting = true;
});

// Экспорт для использования в других модулях
module.exports = { ATBDesktopApp };