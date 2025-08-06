const si = require('systeminformation');
const os = require('os');

class SystemMonitor {
    constructor() {
        this.startTime = Date.now();
        this.history = {
            cpu: [],
            memory: [],
            disk: [],
            network: []
        };
        this.maxHistorySize = 300; // 5 минут при обновлении каждую секунду
    }

    async getMetrics() {
        try {
            // Получение системных метрик
            const [cpu, memory, disk, network, processes, osInfo] = await Promise.all([
                si.currentLoad(),
                si.mem(),
                si.fsSize(),
                si.networkStats(),
                si.processes(),
                si.osInfo()
            ]);

            // CPU метрики
            const cpuInfo = await si.cpu();
            const cpuMetrics = {
                percent: Math.round(cpu.currentLoad),
                count: os.cpus().length,
                frequency: cpuInfo.speed || 0,
                cores: cpuInfo.cores || 0,
                loadAverage: os.loadavg()
            };

            // Memory метрики
            const memoryMetrics = {
                total: memory.total,
                used: memory.used,
                free: memory.free,
                available: memory.available,
                percent: Math.round((memory.used / memory.total) * 100),
                swap: {
                    total: memory.swaptotal || 0,
                    used: memory.swapused || 0,
                    percent: memory.swaptotal > 0 ? Math.round((memory.swapused / memory.swaptotal) * 100) : 0
                }
            };

            // Disk метрики (основной диск)
            const mainDisk = disk[0] || { size: 0, used: 0, available: 0 };
            const diskMetrics = {
                total: mainDisk.size,
                used: mainDisk.used,
                free: mainDisk.available,
                percent: Math.round((mainDisk.used / mainDisk.size) * 100) || 0
            };

            // Network метрики
            const networkMetrics = {
                bytes_sent: network.reduce((sum, iface) => sum + (iface.tx_bytes || 0), 0),
                bytes_recv: network.reduce((sum, iface) => sum + (iface.rx_bytes || 0), 0),
                packets_sent: network.reduce((sum, iface) => sum + (iface.tx_packets || 0), 0),
                packets_recv: network.reduce((sum, iface) => sum + (iface.rx_packets || 0), 0)
            };

            // System метрики
            const systemMetrics = {
                processes: processes.all || 0,
                uptime: this.formatUptime(Date.now() - this.startTime),
                platform: osInfo.platform || process.platform,
                hostname: os.hostname(),
                arch: os.arch(),
                nodeVersion: process.version
            };

            // Сохранение истории
            this.addToHistory('cpu', cpuMetrics.percent);
            this.addToHistory('memory', memoryMetrics.percent);
            this.addToHistory('disk', diskMetrics.percent);

            return {
                cpu: cpuMetrics,
                memory: memoryMetrics,
                disk: diskMetrics,
                network: networkMetrics,
                system: systemMetrics,
                timestamp: new Date().toISOString(),
                history: this.history
            };

        } catch (error) {
            console.error('Error getting system metrics:', error);
            return this.getEmptyMetrics();
        }
    }

    async getProcesses() {
        try {
            const processes = await si.processes();
            
            // Сортировка по CPU и получение топ 10
            const topProcesses = processes.list
                .sort((a, b) => (b.cpu || 0) - (a.cpu || 0))
                .slice(0, 10)
                .map(proc => ({
                    pid: proc.pid,
                    name: proc.name,
                    cpu: Math.round(proc.cpu || 0),
                    memory: Math.round(proc.mem || 0),
                    command: proc.command
                }));

            return topProcesses;
        } catch (error) {
            console.error('Error getting processes:', error);
            return [];
        }
    }

    async getCPUTemperature() {
        try {
            const temp = await si.cpuTemperature();
            return {
                main: temp.main || 0,
                cores: temp.cores || [],
                max: temp.max || 0
            };
        } catch (error) {
            console.error('Error getting CPU temperature:', error);
            return { main: 0, cores: [], max: 0 };
        }
    }

    async getNetworkInterfaces() {
        try {
            const interfaces = await si.networkInterfaces();
            return interfaces.map(iface => ({
                iface: iface.iface,
                ip4: iface.ip4,
                ip6: iface.ip6,
                mac: iface.mac,
                speed: iface.speed,
                type: iface.type,
                operstate: iface.operstate
            }));
        } catch (error) {
            console.error('Error getting network interfaces:', error);
            return [];
        }
    }

    async getDiskInfo() {
        try {
            const disks = await si.fsSize();
            return disks.map(disk => ({
                fs: disk.fs,
                type: disk.type,
                size: disk.size,
                used: disk.used,
                available: disk.available,
                percent: Math.round((disk.used / disk.size) * 100),
                mount: disk.mount
            }));
        } catch (error) {
            console.error('Error getting disk info:', error);
            return [];
        }
    }

    addToHistory(metric, value) {
        const timestamp = Date.now();
        this.history[metric].push({ timestamp, value });
        
        // Ограничение размера истории
        if (this.history[metric].length > this.maxHistorySize) {
            this.history[metric] = this.history[metric].slice(-this.maxHistorySize);
        }
    }

    formatUptime(milliseconds) {
        const seconds = Math.floor(milliseconds / 1000);
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    getEmptyMetrics() {
        return {
            cpu: {
                percent: 0,
                count: 0,
                frequency: 0,
                cores: 0,
                loadAverage: [0, 0, 0]
            },
            memory: {
                total: 0,
                used: 0,
                free: 0,
                available: 0,
                percent: 0,
                swap: { total: 0, used: 0, percent: 0 }
            },
            disk: {
                total: 0,
                used: 0,
                free: 0,
                percent: 0
            },
            network: {
                bytes_sent: 0,
                bytes_recv: 0,
                packets_sent: 0,
                packets_recv: 0
            },
            system: {
                processes: 0,
                uptime: '00:00:00',
                platform: process.platform,
                hostname: os.hostname(),
                arch: os.arch(),
                nodeVersion: process.version
            },
            timestamp: new Date().toISOString(),
            history: this.history
        };
    }

    getHistory(metric, limit = 60) {
        if (!this.history[metric]) return [];
        return this.history[metric].slice(-limit);
    }

    clearHistory() {
        this.history = {
            cpu: [],
            memory: [],
            disk: [],
            network: []
        };
    }
}

module.exports = { SystemMonitor };