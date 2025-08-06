const { app, BrowserWindow, Menu, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs = require('fs');

// Глобальная ссылка на окно
let mainWindow;
let isDev = process.argv.includes('--dev');

// Создание главного окна
function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 1200,
        minHeight: 800,
        icon: path.join(__dirname, 'assets', 'icon.ico'),
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            enableRemoteModule: false,
            preload: path.join(__dirname, 'preload.js')
        },
        show: false,
        titleBarStyle: 'default',
        backgroundColor: '#000000'
    });

    // Загрузка главной страницы
    mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));

    // Показать окно когда готово
    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
        
        // Открыть DevTools в режиме разработки
        if (isDev) {
            mainWindow.webContents.openDevTools();
        }
    });

    // Обработка закрытия окна
    mainWindow.on('closed', () => {
        mainWindow = null;
    });

    // Предотвращение навигации на внешние сайты
    mainWindow.webContents.on('will-navigate', (event, navigationUrl) => {
        const parsedUrl = new URL(navigationUrl);
        
        if (parsedUrl.origin !== 'file://') {
            event.preventDefault();
            shell.openExternal(navigationUrl);
        }
    });

    // Обработка новых окон
    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
        shell.openExternal(url);
        return { action: 'deny' };
    });
}

// Создание меню приложения
function createMenu() {
    const template = [
        {
            label: 'Файл',
            submenu: [
                {
                    label: 'Экспорт данных',
                    accelerator: 'CmdOrCtrl+E',
                    click: () => {
                        mainWindow.webContents.send('export-data');
                    }
                },
                {
                    label: 'Настройки',
                    accelerator: 'CmdOrCtrl+,',
                    click: () => {
                        mainWindow.webContents.send('open-settings');
                    }
                },
                { type: 'separator' },
                {
                    label: 'Выход',
                    accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
                    click: () => {
                        app.quit();
                    }
                }
            ]
        },
        {
            label: 'Вид',
            submenu: [
                {
                    label: 'Перезагрузить',
                    accelerator: 'CmdOrCtrl+R',
                    click: () => {
                        mainWindow.reload();
                    }
                },
                {
                    label: 'Принудительная перезагрузка',
                    accelerator: 'CmdOrCtrl+Shift+R',
                    click: () => {
                        mainWindow.webContents.reloadIgnoringCache();
                    }
                },
                { type: 'separator' },
                {
                    label: 'Увеличить',
                    accelerator: 'CmdOrCtrl+Plus',
                    click: () => {
                        mainWindow.webContents.setZoomLevel(mainWindow.webContents.getZoomLevel() + 1);
                    }
                },
                {
                    label: 'Уменьшить',
                    accelerator: 'CmdOrCtrl+-',
                    click: () => {
                        mainWindow.webContents.setZoomLevel(mainWindow.webContents.getZoomLevel() - 1);
                    }
                },
                {
                    label: 'Сбросить масштаб',
                    accelerator: 'CmdOrCtrl+0',
                    click: () => {
                        mainWindow.webContents.setZoomLevel(0);
                    }
                },
                { type: 'separator' },
                {
                    label: 'Полноэкранный режим',
                    accelerator: 'F11',
                    click: () => {
                        mainWindow.setFullScreen(!mainWindow.isFullScreen());
                    }
                }
            ]
        },
        {
            label: 'Торговля',
            submenu: [
                {
                    label: 'Запустить бота',
                    accelerator: 'CmdOrCtrl+Shift+S',
                    click: () => {
                        mainWindow.webContents.send('start-bot');
                    }
                },
                {
                    label: 'Остановить бота',
                    accelerator: 'CmdOrCtrl+Shift+X',
                    click: () => {
                        mainWindow.webContents.send('stop-bot');
                    }
                },
                { type: 'separator' },
                {
                    label: 'Новая сделка',
                    accelerator: 'CmdOrCtrl+N',
                    click: () => {
                        mainWindow.webContents.send('new-trade');
                    }
                }
            ]
        },
        {
            label: 'Аналитика',
            submenu: [
                {
                    label: 'Обновить данные',
                    accelerator: 'F5',
                    click: () => {
                        mainWindow.webContents.send('refresh-data');
                    }
                },
                {
                    label: 'Экспорт отчета',
                    accelerator: 'CmdOrCtrl+Shift+E',
                    click: () => {
                        mainWindow.webContents.send('export-report');
                    }
                }
            ]
        },
        {
            label: 'Помощь',
            submenu: [
                {
                    label: 'О приложении',
                    click: () => {
                        dialog.showMessageBox(mainWindow, {
                            type: 'info',
                            title: 'О ATB Dashboard',
                            message: 'ATB Trading Dashboard',
                            detail: 'Версия 1.0.0\nСовременный дашборд для торгового бота ATB'
                        });
                    }
                },
                {
                    label: 'Документация',
                    click: () => {
                        shell.openExternal('https://github.com/atb-project/docs');
                    }
                },
                { type: 'separator' },
                {
                    label: 'Проверить обновления',
                    click: () => {
                        mainWindow.webContents.send('check-updates');
                    }
                }
            ]
        }
    ];

    // Добавить DevTools в режиме разработки
    if (isDev) {
        template.push({
            label: 'Разработка',
            submenu: [
                {
                    label: 'Открыть DevTools',
                    accelerator: 'F12',
                    click: () => {
                        mainWindow.webContents.toggleDevTools();
                    }
                },
                {
                    label: 'Перезагрузить',
                    accelerator: 'CmdOrCtrl+Shift+R',
                    click: () => {
                        mainWindow.webContents.reloadIgnoringCache();
                    }
                }
            ]
        });
    }

    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);
}

// IPC обработчики
ipcMain.handle('get-app-version', () => {
    return app.getVersion();
});

ipcMain.handle('get-app-path', () => {
    return app.getAppPath();
});

ipcMain.handle('show-save-dialog', async (event, options) => {
    const result = await dialog.showSaveDialog(mainWindow, options);
    return result;
});

ipcMain.handle('show-open-dialog', async (event, options) => {
    const result = await dialog.showOpenDialog(mainWindow, options);
    return result;
});

ipcMain.handle('read-file', async (event, filePath) => {
    try {
        const data = await fs.promises.readFile(filePath, 'utf8');
        return { success: true, data };
    } catch (error) {
        return { success: false, error: error.message };
    }
});

ipcMain.handle('write-file', async (event, filePath, data) => {
    try {
        await fs.promises.writeFile(filePath, data, 'utf8');
        return { success: true };
    } catch (error) {
        return { success: false, error: error.message };
    }
});

// Обработка событий приложения
app.whenReady().then(() => {
    createWindow();
    createMenu();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

// Обработка ошибок
process.on('uncaughtException', (error) => {
    console.error('Необработанная ошибка:', error);
    dialog.showErrorBox('Ошибка приложения', error.message);
});

// Обработка предупреждений безопасности
app.on('web-contents-created', (event, contents) => {
    contents.on('new-window', (event, navigationUrl) => {
        event.preventDefault();
        shell.openExternal(navigationUrl);
    });
}); 