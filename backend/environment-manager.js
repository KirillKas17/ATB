const fs = require('fs').promises;
const path = require('path');
require('dotenv').config();

class EnvironmentManager {
    constructor() {
        this.envPath = path.join(process.cwd(), '.env');
        this.defaultConfig = {
            // Общие настройки
            NODE_ENV: 'production',
            ENVIRONMENT: 'development',
            DEBUG: 'true',
            ATB_MODE: 'simulation',
            
            // Сервер
            BACKEND_PORT: '3001',
            FRONTEND_PORT: '3000',
            
            // База данных
            DB_HOST: 'localhost',
            DB_PORT: '5432',
            DB_NAME: 'atb_trading',
            DB_USER: 'atb_user',
            DB_PASS: '',
            
            // Биржа
            EXCHANGE_API_KEY: '',
            EXCHANGE_API_SECRET: '',
            EXCHANGE_TESTNET: 'true',
            
            // Мониторинг
            MONITORING_ENABLED: 'true',
            MONITORING_INTERVAL: '10000',
            ALERT_EMAIL: '',
            
            // Эволюция
            EVOLUTION_ENABLED: 'true',
            EVOLUTION_INTERVAL: '3600000',
            AUTO_EVOLUTION: 'false',
            
            // Торговля
            DEFAULT_POSITION_SIZE: '1.0',
            DEFAULT_STOP_LOSS: '2.0',
            MAX_DRAWDOWN: '5.0',
            
            // Логирование
            LOG_LEVEL: 'info',
            LOG_FILE: 'logs/atb.log',
            
            // Безопасность
            ENABLE_CORS: 'true',
            JWT_SECRET: 'your-secret-key'
        };
    }

    async getConfig() {
        try {
            const exists = await this.fileExists(this.envPath);
            
            if (!exists) {
                console.log('📄 .env файл не найден, создание с настройками по умолчанию...');
                await this.createDefaultEnvFile();
            }

            const envContent = await fs.readFile(this.envPath, 'utf8');
            const config = this.parseEnvContent(envContent);
            
            return {
                success: true,
                config: config,
                path: this.envPath
            };

        } catch (error) {
            console.error('❌ Ошибка чтения .env файла:', error);
            return {
                success: false,
                error: error.message,
                config: this.defaultConfig
            };
        }
    }

    async saveConfig(config) {
        try {
            const envContent = this.generateEnvContent(config);
            
            // Создание резервной копии
            await this.createBackup();
            
            // Сохранение нового содержимого
            await fs.writeFile(this.envPath, envContent, 'utf8');
            
            console.log('✅ .env файл сохранен успешно');
            
            return {
                success: true,
                message: 'Конфигурация сохранена успешно'
            };

        } catch (error) {
            console.error('❌ Ошибка сохранения .env файла:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    async resetToDefaults() {
        try {
            await this.createBackup();
            await this.createDefaultEnvFile();
            
            console.log('✅ .env файл сброшен к настройкам по умолчанию');
            
            return {
                success: true,
                message: 'Конфигурация сброшена к настройкам по умолчанию',
                config: this.defaultConfig
            };

        } catch (error) {
            console.error('❌ Ошибка сброса .env файла:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    async createDefaultEnvFile() {
        const content = this.generateEnvContent(this.defaultConfig);
        await fs.writeFile(this.envPath, content, 'utf8');
    }

    generateEnvContent(config) {
        let content = '# ATB Trading System Enhanced Configuration\n';
        content += `# Создано: ${new Date().toISOString()}\n\n`;

        // Группировка настроек
        const groups = {
            'Общие настройки': [
                'NODE_ENV', 'ENVIRONMENT', 'DEBUG', 'ATB_MODE'
            ],
            'Сервер': [
                'BACKEND_PORT', 'FRONTEND_PORT'
            ],
            'База данных': [
                'DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASS'
            ],
            'Биржа': [
                'EXCHANGE_API_KEY', 'EXCHANGE_API_SECRET', 'EXCHANGE_TESTNET'
            ],
            'Мониторинг': [
                'MONITORING_ENABLED', 'MONITORING_INTERVAL', 'ALERT_EMAIL'
            ],
            'Эволюция': [
                'EVOLUTION_ENABLED', 'EVOLUTION_INTERVAL', 'AUTO_EVOLUTION'
            ],
            'Торговля': [
                'DEFAULT_POSITION_SIZE', 'DEFAULT_STOP_LOSS', 'MAX_DRAWDOWN'
            ],
            'Логирование': [
                'LOG_LEVEL', 'LOG_FILE'
            ],
            'Безопасность': [
                'ENABLE_CORS', 'JWT_SECRET'
            ]
        };

        for (const [groupName, keys] of Object.entries(groups)) {
            content += `# ${groupName}\n`;
            
            for (const key of keys) {
                const value = config[key] || this.defaultConfig[key] || '';
                const comment = this.getConfigComment(key);
                
                if (comment) {
                    content += `# ${comment}\n`;
                }
                
                content += `${key}=${value}\n`;
            }
            
            content += '\n';
        }

        return content;
    }

    parseEnvContent(content) {
        const config = {};
        const lines = content.split('\n');

        for (const line of lines) {
            const trimmed = line.trim();
            
            // Пропуск комментариев и пустых строк
            if (trimmed.startsWith('#') || trimmed === '') {
                continue;
            }

            // Парсинг переменных
            const equalIndex = trimmed.indexOf('=');
            if (equalIndex !== -1) {
                const key = trimmed.substring(0, equalIndex).trim();
                const value = trimmed.substring(equalIndex + 1).trim();
                
                // Удаление кавычек если есть
                config[key] = value.replace(/^["']|["']$/g, '');
            }
        }

        return config;
    }

    getConfigComment(key) {
        const comments = {
            NODE_ENV: 'Режим Node.js (production, development)',
            ENVIRONMENT: 'Среда разработки',
            DEBUG: 'Включить отладку (true/false)',
            ATB_MODE: 'Режим торговли (simulation, paper, live)',
            
            BACKEND_PORT: 'Порт бэкенд сервера',
            FRONTEND_PORT: 'Порт фронтенд сервера',
            
            DB_HOST: 'Хост базы данных',
            DB_PORT: 'Порт базы данных',
            DB_NAME: 'Имя базы данных',
            DB_USER: 'Пользователь базы данных',
            DB_PASS: 'Пароль базы данных',
            
            EXCHANGE_API_KEY: 'API ключ биржи',
            EXCHANGE_API_SECRET: 'API секрет биржи',
            EXCHANGE_TESTNET: 'Использовать тестовую сеть (true/false)',
            
            MONITORING_ENABLED: 'Включить мониторинг (true/false)',
            MONITORING_INTERVAL: 'Интервал мониторинга в миллисекундах',
            ALERT_EMAIL: 'Email для уведомлений',
            
            EVOLUTION_ENABLED: 'Включить эволюцию стратегий (true/false)',
            EVOLUTION_INTERVAL: 'Интервал эволюции в миллисекундах',
            AUTO_EVOLUTION: 'Автоматическая эволюция (true/false)',
            
            DEFAULT_POSITION_SIZE: 'Размер позиции по умолчанию',
            DEFAULT_STOP_LOSS: 'Стоп-лосс по умолчанию (%)',
            MAX_DRAWDOWN: 'Максимальная просадка (%)',
            
            LOG_LEVEL: 'Уровень логирования (debug, info, warn, error)',
            LOG_FILE: 'Путь к файлу логов',
            
            ENABLE_CORS: 'Включить CORS (true/false)',
            JWT_SECRET: 'Секретный ключ для JWT токенов'
        };

        return comments[key] || null;
    }

    async createBackup() {
        try {
            const exists = await this.fileExists(this.envPath);
            if (!exists) return;

            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const backupPath = path.join(process.cwd(), `.env.backup.${timestamp}`);
            
            const content = await fs.readFile(this.envPath, 'utf8');
            await fs.writeFile(backupPath, content, 'utf8');
            
            console.log(`💾 Создана резервная копия: ${backupPath}`);

        } catch (error) {
            console.warn('⚠️ Не удалось создать резервную копию:', error.message);
        }
    }

    async fileExists(filePath) {
        try {
            await fs.access(filePath);
            return true;
        } catch {
            return false;
        }
    }

    async validateConfig(config) {
        const errors = [];
        const warnings = [];

        // Проверка обязательных полей
        const required = ['NODE_ENV', 'ATB_MODE', 'BACKEND_PORT'];
        for (const field of required) {
            if (!config[field]) {
                errors.push(`Обязательное поле '${field}' не заполнено`);
            }
        }

        // Проверка числовых значений
        const numeric = ['BACKEND_PORT', 'FRONTEND_PORT', 'DB_PORT', 'MONITORING_INTERVAL', 'EVOLUTION_INTERVAL'];
        for (const field of numeric) {
            if (config[field] && isNaN(Number(config[field]))) {
                errors.push(`Поле '${field}' должно быть числом`);
            }
        }

        // Проверка булевых значений
        const boolean = ['DEBUG', 'EXCHANGE_TESTNET', 'MONITORING_ENABLED', 'EVOLUTION_ENABLED', 'AUTO_EVOLUTION', 'ENABLE_CORS'];
        for (const field of boolean) {
            if (config[field] && !['true', 'false'].includes(config[field].toLowerCase())) {
                errors.push(`Поле '${field}' должно быть true или false`);
            }
        }

        // Проверка режимов
        const validModes = ['simulation', 'paper', 'live'];
        if (config.ATB_MODE && !validModes.includes(config.ATB_MODE)) {
            errors.push(`ATB_MODE должен быть одним из: ${validModes.join(', ')}`);
        }

        // Предупреждения
        if (config.ATB_MODE === 'live' && (!config.EXCHANGE_API_KEY || !config.EXCHANGE_API_SECRET)) {
            warnings.push('Для реальной торговли необходимо указать API ключи биржи');
        }

        if (config.JWT_SECRET === 'your-secret-key') {
            warnings.push('Рекомендуется изменить JWT_SECRET на случайную строку');
        }

        return {
            isValid: errors.length === 0,
            errors,
            warnings
        };
    }

    async getConfigInfo() {
        try {
            const config = await this.getConfig();
            const validation = await this.validateConfig(config.config);
            
            const stats = await fs.stat(this.envPath);
            
            return {
                success: true,
                info: {
                    path: this.envPath,
                    size: stats.size,
                    modified: stats.mtime,
                    created: stats.birthtime,
                    variablesCount: Object.keys(config.config).length,
                    validation: validation
                }
            };

        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }

    // Получение текущих переменных окружения процесса
    getCurrentEnv() {
        const filtered = {};
        
        // Фильтрация только ATB-связанных переменных
        for (const [key, value] of Object.entries(process.env)) {
            if (key.startsWith('ATB_') || 
                key.startsWith('DB_') || 
                key.startsWith('EXCHANGE_') ||
                key.startsWith('MONITORING_') ||
                key.startsWith('EVOLUTION_') ||
                ['NODE_ENV', 'ENVIRONMENT', 'DEBUG', 'LOG_LEVEL'].includes(key)) {
                filtered[key] = value;
            }
        }

        return filtered;
    }

    // Применение конфигурации к process.env
    applyToProcess(config) {
        for (const [key, value] of Object.entries(config)) {
            process.env[key] = value;
        }
    }
}

module.exports = { EnvironmentManager };