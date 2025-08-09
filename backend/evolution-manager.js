const fs = require('fs').promises;
const path = require('path');
const { spawn } = require('child_process');

class EvolutionManager {
    constructor() {
        this.isRunning = false;
        this.strategies = [];
        this.statistics = {
            totalEvolutions: 0,
            bestPerformance: 0,
            averageGenerations: 0
        };
        this.generation = 0;
        this.population = [];
    }

    async getStatus() {
        return {
            running: this.isRunning,
            strategies: this.strategies,
            statistics: this.statistics,
            generation: this.generation,
            populationSize: this.population.length
        };
    }

    async start() {
        if (this.isRunning) {
            return { success: false, error: 'Evolution already running' };
        }

        this.isRunning = true;
        this.generation = 0;
        this.population = this.generateInitialPopulation();
        
        console.log('🧬 Evolution started');
        
        // Запускаем эволюционный процесс
        this.evolutionLoop();
        
        return { success: true, message: 'Evolution started successfully' };
    }

    async stop() {
        if (!this.isRunning) {
            return { success: false, error: 'Evolution not running' };
        }

        this.isRunning = false;
        console.log('🧬 Evolution stopped');
        
        return { success: true, message: 'Evolution stopped successfully' };
    }

    generateInitialPopulation() {
        const population = [];
        const populationSize = 10;
        
        for (let i = 0; i < populationSize; i++) {
            population.push(this.generateRandomStrategy());
        }
        
        return population;
    }

    generateRandomStrategy() {
        const strategyTypes = ['trend', 'mean_reversion', 'momentum', 'arbitrage', 'ml_enhanced'];
        const timeframes = ['1m', '5m', '15m', '1h', '4h', '1d'];
        const riskLevels = ['low', 'medium', 'high'];
        
        return {
            id: `strategy_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            name: `Strategy ${Math.floor(Math.random() * 1000)}`,
            type: strategyTypes[Math.floor(Math.random() * strategyTypes.length)],
            timeframe: timeframes[Math.floor(Math.random() * timeframes.length)],
            riskLevel: riskLevels[Math.floor(Math.random() * riskLevels.length)],
            performance: Math.random() * 100,
            fitness: Math.random(),
            generation: 0,
            parameters: this.generateRandomParameters(),
            status: Math.random() > 0.3 ? 'active' : 'paused'
        };
    }

    generateRandomParameters() {
        return {
            lookbackPeriod: Math.floor(Math.random() * 50) + 10,
            threshold: Math.random() * 0.1 + 0.01,
            stopLoss: Math.random() * 0.05 + 0.01,
            takeProfit: Math.random() * 0.1 + 0.02,
            maxPositionSize: Math.random() * 0.1 + 0.01
        };
    }

    async evolutionLoop() {
        while (this.isRunning) {
            try {
                // Оценка фитнеса
                await this.evaluateFitness();
                
                // Селекция
                const selected = this.selection();
                
                // Скрещивание
                const offspring = this.crossover(selected);
                
                // Мутация
                this.mutation(offspring);
                
                // Обновление популяции
                this.population = offspring;
                this.generation++;
                
                // Обновление статистики
                this.updateStatistics();
                
                // Обновление стратегий для отображения
                this.updateStrategies();
                
                // Пауза между поколениями
                await this.sleep(5000); // 5 секунд
                
            } catch (error) {
                console.error('Error in evolution loop:', error);
                await this.sleep(1000);
            }
        }
    }

    async evaluateFitness() {
        for (let strategy of this.population) {
            // Симуляция оценки фитнеса
            strategy.fitness = Math.random();
            strategy.performance = strategy.fitness * 100;
        }
    }

    selection() {
        // Турнирная селекция
        const tournamentSize = 3;
        const selected = [];
        
        while (selected.length < this.population.length) {
            const tournament = [];
            for (let i = 0; i < tournamentSize; i++) {
                const randomIndex = Math.floor(Math.random() * this.population.length);
                tournament.push(this.population[randomIndex]);
            }
            
            const winner = tournament.reduce((best, current) => 
                current.fitness > best.fitness ? current : best
            );
            
            selected.push({ ...winner });
        }
        
        return selected;
    }

    crossover(parents) {
        const offspring = [];
        
        for (let i = 0; i < parents.length; i += 2) {
            if (i + 1 < parents.length) {
                const parent1 = parents[i];
                const parent2 = parents[i + 1];
                
                const child1 = this.crossoverStrategies(parent1, parent2);
                const child2 = this.crossoverStrategies(parent2, parent1);
                
                offspring.push(child1, child2);
            } else {
                offspring.push({ ...parents[i] });
            }
        }
        
        return offspring;
    }

    crossoverStrategies(strategy1, strategy2) {
        const child = { ...strategy1 };
        
        // Скрещивание параметров
        const params1 = strategy1.parameters;
        const params2 = strategy2.parameters;
        
        child.parameters = {
            lookbackPeriod: Math.random() > 0.5 ? params1.lookbackPeriod : params2.lookbackPeriod,
            threshold: Math.random() > 0.5 ? params1.threshold : params2.threshold,
            stopLoss: Math.random() > 0.5 ? params1.stopLoss : params2.stopLoss,
            takeProfit: Math.random() > 0.5 ? params1.takeProfit : params2.takeProfit,
            maxPositionSize: Math.random() > 0.5 ? params1.maxPositionSize : params2.maxPositionSize
        };
        
        child.generation = this.generation;
        child.id = `strategy_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        return child;
    }

    mutation(offspring) {
        const mutationRate = 0.1;
        
        for (let strategy of offspring) {
            if (Math.random() < mutationRate) {
                // Мутация параметров
                const param = Object.keys(strategy.parameters)[
                    Math.floor(Math.random() * Object.keys(strategy.parameters).length)
                ];
                
                strategy.parameters[param] *= (0.8 + Math.random() * 0.4); // ±20%
            }
        }
    }

    updateStatistics() {
        const performances = this.population.map(s => s.performance);
        this.statistics.bestPerformance = Math.max(...performances);
        this.statistics.averageGenerations = this.generation;
        this.statistics.totalEvolutions++;
    }

    updateStrategies() {
        // Обновляем список стратегий для отображения
        this.strategies = this.population
            .sort((a, b) => b.fitness - a.fitness)
            .slice(0, 5)
            .map(strategy => ({
                id: strategy.id,
                name: strategy.name,
                type: strategy.type,
                performance: strategy.performance.toFixed(2),
                status: strategy.status,
                generation: strategy.generation
            }));
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

module.exports = { EvolutionManager };