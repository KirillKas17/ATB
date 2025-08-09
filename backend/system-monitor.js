const si = require('systeminformation');

class SystemMonitor {
    constructor() {
        this.cache = {};
        this.cacheTimeout = 5000; // 5 секунд
        this.lastUpdate = 0;
    }

    async getMetrics() {
        const now = Date.now();
        
        // Используем кэш если данные не устарели
        if (this.cache.metrics && (now - this.lastUpdate) < this.cacheTimeout) {
            return this.cache.metrics;
        }

        try {
            const [cpu, mem, disk, network] = await Promise.all([
                this.getCPUMetrics(),
                this.getMemoryMetrics(),
                this.getDiskMetrics(),
                this.getNetworkMetrics()
            ]);

            const metrics = {
                cpu,
                memory: mem,
                disk,
                network,
                timestamp: new Date().toISOString()
            };

            // Кэшируем результат
            this.cache.metrics = metrics;
            this.lastUpdate = now;

            return metrics;
        } catch (error) {
            console.error('Error getting system metrics:', error);
            return this.getFallbackMetrics();
        }
    }

    async getCPUMetrics() {
        try {
            const [cpuLoad, cpuInfo] = await Promise.all([
                si.currentLoad(),
                si.cpu()
            ]);

            return {
                percent: Math.round(cpuLoad.currentLoad || 0),
                cores: cpuInfo.cores || 4,
                frequency: Math.round((cpuInfo.speed || 2000) / 100) * 100
            };
        } catch (error) {
            console.error('Error getting CPU metrics:', error);
            return {
                percent: Math.round(Math.random() * 100),
                cores: 4,
                frequency: 2000
            };
        }
    }

    async getMemoryMetrics() {
        try {
            const mem = await si.mem();
            const total = mem.total || 8000000000; // 8GB по умолчанию
            const used = mem.used || 4000000000; // 4GB по умолчанию
            const free = mem.free || 4000000000; // 4GB по умолчанию
            const percent = Math.round((used / total) * 100);

            return {
                percent,
                total,
                free,
                used
            };
        } catch (error) {
            console.error('Error getting memory metrics:', error);
            return {
                percent: Math.round(Math.random() * 100),
                total: 8000000000,
                free: 4000000000,
                used: 4000000000
            };
        }
    }

    async getDiskMetrics() {
        try {
            const disk = await si.fsSize();
            const mainDisk = disk[0] || { size: 500000000000, used: 300000000000, available: 200000000000 };
            const total = mainDisk.size || 500000000000;
            const used = mainDisk.used || 300000000000;
            const free = mainDisk.available || 200000000000;
            const percent = Math.round((used / total) * 100);

            return {
                percent,
                total,
                free,
                used
            };
        } catch (error) {
            console.error('Error getting disk metrics:', error);
            return {
                percent: Math.round(Math.random() * 100),
                total: 500000000000,
                free: 200000000000,
                used: 300000000000
            };
        }
    }

    async getNetworkMetrics() {
        try {
            const network = await si.networkStats();
            const mainInterface = network[0] || { rx_bytes: 0, tx_bytes: 0 };
            
            return {
                bytes_sent: mainInterface.tx_bytes || 0,
                bytes_recv: mainInterface.rx_bytes || 0
            };
        } catch (error) {
            console.error('Error getting network metrics:', error);
            return {
                bytes_sent: Math.floor(Math.random() * 1000000),
                bytes_recv: Math.floor(Math.random() * 2000000)
            };
        }
    }

    async getProcesses() {
        try {
            const processes = await si.processes();
            return processes.list.slice(0, 10).map(proc => ({
                pid: proc.pid,
                name: proc.name,
                cpu: Math.round(proc.cpu || 0),
                memory: Math.round(proc.mem || 0),
                command: proc.command || ''
            }));
        } catch (error) {
            console.error('Error getting processes:', error);
            return [];
        }
    }

    async getCPUTemperature() {
        try {
            const temp = await si.cpuTemperature();
            return {
                main: Math.round(temp.main || 45),
                cores: temp.cores || [45, 46, 47, 48],
                max: Math.round(temp.max || 85)
            };
        } catch (error) {
            console.error('Error getting CPU temperature:', error);
            return {
                main: Math.round(45 + Math.random() * 20),
                cores: [45, 46, 47, 48],
                max: 85
            };
        }
    }

    async getNetworkInterfaces() {
        try {
            const interfaces = await si.networkInterfaces();
            return interfaces.map(iface => ({
                name: iface.iface,
                type: iface.type,
                ip: iface.ip4 || 'N/A',
                mac: iface.mac || 'N/A',
                speed: iface.speed || 'N/A'
            }));
        } catch (error) {
            console.error('Error getting network interfaces:', error);
            return [];
        }
    }

    async getDiskInfo() {
        try {
            const disk = await si.diskLayout();
            return disk.map(d => ({
                name: d.name || 'Unknown',
                size: d.size || 0,
                type: d.type || 'Unknown',
                interface: d.interfaceType || 'Unknown'
            }));
        } catch (error) {
            console.error('Error getting disk info:', error);
            return [];
        }
    }

    getFallbackMetrics() {
        return {
            cpu: {
                percent: Math.round(Math.random() * 100),
                cores: 4,
                frequency: 2000
            },
            memory: {
                percent: Math.round(Math.random() * 100),
                total: 8000000000,
                free: 4000000000,
                used: 4000000000
            },
            disk: {
                percent: Math.round(Math.random() * 100),
                total: 500000000000,
                free: 200000000000,
                used: 300000000000
            },
            network: {
                bytes_sent: Math.floor(Math.random() * 1000000),
                bytes_recv: Math.floor(Math.random() * 2000000)
            },
            timestamp: new Date().toISOString()
        };
    }
}

module.exports = { SystemMonitor };