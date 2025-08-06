// ============================================================================
// ATB Trading System - ML Manager JavaScript
// Управление машинным обучением в Electron приложении
// ============================================================================

class MLUI {
    constructor() {
        this.selectedModel = null;
        this.trainingChart = null;
        this.isTraining = false;
        
        this.initializeElements();
        this.setupEventListeners();
        this.initializeCharts();
    }

    initializeElements() {
        // Основные контролы
        this.mlModelFilterEl = document.getElementById('mlModelFilter');
        this.startMLTrainingBtn = document.getElementById('startMLTrainingBtn');
        this.stopMLTrainingBtn = document.getElementById('stopMLTrainingBtn');
        this.evaluateAllModelsBtn = document.getElementById('evaluateAllModelsBtn');

        // Кнопки для выбранной модели
        this.trainSelectedModelBtn = document.getElementById('trainSelectedModelBtn');
        this.evaluateSelectedModelBtn = document.getElementById('evaluateSelectedModelBtn');
        this.deploySelectedModelBtn = document.getElementById('deploySelectedModelBtn');

        // Статистика ML
        this.totalModelsEl = document.getElementById('totalModels');
        this.trainingModelsEl = document.getElementById('trainingModels');
        this.trainedModelsEl = document.getElementById('trainedModels');
        this.avgAccuracyEl = document.getElementById('avgAccuracy');
        this.activeJobsEl = document.getElementById('activeJobs');

        // Списки и детали
        this.mlModelsListEl = document.getElementById('mlModelsList');
        this.selectedModelNameEl = document.getElementById('selectedModelName');
        this.modelDetailsEl = document.getElementById('modelDetails');
        this.trainingJobsListEl = document.getElementById('trainingJobsList');
        this.hyperparametersSectionEl = document.getElementById('hyperparametersSection');
        this.trainingHistoryEl = document.getElementById('trainingHistory');
        this.evaluationResultsEl = document.getElementById('evaluationResults');

        // Фильтры
        this.modelTypeFilterEl = document.getElementById('modelTypeFilter');
        this.trainingMetricEl = document.getElementById('trainingMetric');
    }

    setupEventListeners() {
        // Основные контролы ML
        if (this.startMLTrainingBtn) {
            this.startMLTrainingBtn.addEventListener('click', () => this.startGeneralTraining());
        }

        if (this.stopMLTrainingBtn) {
            this.stopMLTrainingBtn.addEventListener('click', () => this.stopGeneralTraining());
        }

        if (this.evaluateAllModelsBtn) {
            this.evaluateAllModelsBtn.addEventListener('click', () => this.evaluateAllModels());
        }

        // Контролы для выбранной модели
        if (this.trainSelectedModelBtn) {
            this.trainSelectedModelBtn.addEventListener('click', () => this.trainSelectedModel());
        }

        if (this.evaluateSelectedModelBtn) {
            this.evaluateSelectedModelBtn.addEventListener('click', () => this.evaluateSelectedModel());
        }

        if (this.deploySelectedModelBtn) {
            this.deploySelectedModelBtn.addEventListener('click', () => this.deploySelectedModel());
        }

        // Фильтры
        if (this.mlModelFilterEl) {
            this.mlModelFilterEl.addEventListener('change', () => this.filterModels());
        }

        if (this.modelTypeFilterEl) {
            this.modelTypeFilterEl.addEventListener('change', () => this.filterModels());
        }

        if (this.trainingMetricEl) {
            this.trainingMetricEl.addEventListener('change', () => this.updateTrainingChart());
        }
    }

    initializeCharts() {
        // График обучения
        const trainingCanvas = document.getElementById('trainingChart');
        if (trainingCanvas) {
            const ctx = trainingCanvas.getContext('2d');
            
            this.trainingChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Training Loss',
                        data: [],
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4
                    }, {
                        label: 'Validation Loss',
                        data: [],
                        borderColor: '#ffa726',
                        backgroundColor: 'rgba(255, 167, 38, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4
                    }, {
                        label: 'Training Accuracy',
                        data: [],
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0, 255, 136, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y1',
                        tension: 0.4
                    }, {
                        label: 'Validation Accuracy',
                        data: [],
                        borderColor: '#42a5f5',
                        backgroundColor: 'rgba(66, 165, 245, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y1',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: {
                                color: '#eee'
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Эпохи',
                                color: '#ccc'
                            },
                            ticks: { color: '#ccc' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Loss',
                                color: '#ccc'
                            },
                            ticks: { color: '#ccc' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {
                                display: true,
                                text: 'Accuracy (%)',
                                color: '#ccc'
                            },
                            ticks: { 
                                color: '#00ff88',
                                callback: function(value) {
                                    return value.toFixed(1) + '%';
                                }
                            },
                            grid: {
                                drawOnChartArea: false,
                            },
                        }
                    }
                }
            });
        }
    }

    async updateMLData() {
        try {
            if (!window.electronAPI) {
                this.loadDemoMLData();
                return;
            }

            const mlStatus = await window.electronAPI.getMLStatus();
            
            if (mlStatus && mlStatus.success) {
                this.updateMLStats(mlStatus.summary);
                this.updateModelsList(mlStatus.models);
                this.updateActiveJobs(mlStatus.activeJobs);
                this.updateTrainingHistory(mlStatus.recentHistory);
                this.updateEvaluationResults(mlStatus.evaluationResults);
                this.updateMLControls(mlStatus.isTraining);
            }

        } catch (error) {
            console.error('❌ Error updating ML data:', error);
            this.loadDemoMLData();
        }
    }

    loadDemoMLData() {
        // Демо данные для ML
        const demoStats = {
            totalModels: 4,
            trainingModels: 1,
            trainedModels: 2,
            idleModels: 1,
            averageAccuracy: 75.6,
            modelTypes: {
                lstm: 1,
                transformer: 1,
                xgboost: 1,
                gan: 1
            }
        };

        const demoModels = [
            {
                id: 'lstm_predictor',
                name: 'LSTM Price Predictor',
                type: 'lstm',
                status: 'trained',
                version: '2.1.3',
                accuracy: 76.8,
                lastTrained: new Date(Date.now() - 86400000 * 2).toISOString(),
                trainingProgress: 100,
                performance: {
                    training_accuracy: 76.8,
                    validation_accuracy: 74.2,
                    training_loss: 0.0023,
                    validation_loss: 0.0031
                }
            },
            {
                id: 'transformer_sentiment',
                name: 'Transformer Sentiment Analyzer',
                type: 'transformer',
                status: 'training',
                version: '1.2.0',
                accuracy: 68.5,
                lastTrained: new Date().toISOString(),
                trainingProgress: 67,
                performance: {
                    training_accuracy: 68.5,
                    validation_accuracy: 65.8,
                    training_loss: 0.45,
                    validation_loss: 0.52
                }
            },
            {
                id: 'xgboost_classifier',
                name: 'XGBoost Signal Classifier',
                type: 'xgboost',
                status: 'evaluating',
                version: '3.0.1',
                accuracy: 82.3,
                lastTrained: new Date(Date.now() - 3600000).toISOString(),
                trainingProgress: 100,
                performance: {
                    training_accuracy: 82.3,
                    validation_accuracy: 79.8,
                    test_accuracy: 78.9
                }
            },
            {
                id: 'gan_generator',
                name: 'GAN Price Generator',
                type: 'gan',
                status: 'idle',
                version: '1.0.2',
                accuracy: 0,
                lastTrained: new Date(Date.now() - 86400000 * 7).toISOString(),
                trainingProgress: 0,
                performance: {
                    generator_loss: 1.23,
                    discriminator_loss: 0.87
                }
            }
        ];

        const demoJobs = [
            {
                id: 'job_123',
                modelId: 'transformer_sentiment',
                modelName: 'Transformer Sentiment Analyzer',
                status: 'running',
                progress: 67,
                currentEpoch: 67,
                totalEpochs: 100,
                startTime: new Date(Date.now() - 3600000).toISOString()
            }
        ];

        const demoHistory = [
            {
                timestamp: new Date(Date.now() - 600000).toISOString(),
                modelId: 'lstm_predictor',
                type: 'completed',
                message: 'Training completed successfully'
            },
            {
                timestamp: new Date(Date.now() - 300000).toISOString(),
                modelId: 'transformer_sentiment',
                type: 'started',
                message: 'Training started for Transformer Sentiment Analyzer'
            }
        ];

        this.updateMLStats(demoStats);
        this.updateModelsList(demoModels);
        this.updateActiveJobs(demoJobs);
        this.updateTrainingHistory(demoHistory);
        this.updateMLControls(this.isTraining);
        this.generateDemoTrainingChart();
    }

    updateMLStats(stats) {
        if (this.totalModelsEl) {
            this.totalModelsEl.textContent = stats.totalModels || 0;
        }
        
        if (this.trainingModelsEl) {
            this.trainingModelsEl.textContent = stats.trainingModels || 0;
        }
        
        if (this.trainedModelsEl) {
            this.trainedModelsEl.textContent = stats.trainedModels || 0;
        }
        
        if (this.avgAccuracyEl) {
            this.avgAccuracyEl.textContent = `${(stats.averageAccuracy || 0).toFixed(1)}%`;
        }
        
        if (this.activeJobsEl) {
            this.activeJobsEl.textContent = stats.activeJobs || 0;
        }
    }

    updateModelsList(models) {
        if (!this.mlModelsListEl || !models) return;

        this.mlModelsListEl.innerHTML = '';

        models.forEach(model => {
            const modelEl = this.createModelElement(model);
            this.mlModelsListEl.appendChild(modelEl);
        });
    }

    createModelElement(model) {
        const element = document.createElement('div');
        element.className = 'ml-model-item';
        element.dataset.modelId = model.id;
        element.dataset.modelType = model.type;
        element.dataset.modelStatus = model.status;

        const statusClass = model.status || 'idle';
        const typeClass = model.type || 'unknown';

        element.innerHTML = `
            <div class="ml-model-header">
                <div class="ml-model-icon ${typeClass}">
                    ${this.getModelTypeIcon(model.type)}
                </div>
                <div class="ml-model-info">
                    <div class="ml-model-name">${model.name}</div>
                    <div class="ml-model-version">v${model.version}</div>
                </div>
                <div class="ml-model-status ${statusClass}">${this.getStatusText(model.status)}</div>
            </div>
            <div class="ml-model-metrics">
                <div class="ml-model-metric">
                    <span>Точность:</span>
                    <span class="value">${model.accuracy ? model.accuracy.toFixed(1) + '%' : 'N/A'}</span>
                </div>
                <div class="ml-model-metric">
                    <span>Прогресс:</span>
                    <span class="value">${model.trainingProgress || 0}%</span>
                </div>
                <div class="ml-model-metric">
                    <span>Обновлено:</span>
                    <span class="value">${this.formatDate(model.lastTrained)}</span>
                </div>
            </div>
            ${model.status === 'training' ? `
                <div class="ml-model-progress">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${model.trainingProgress}%"></div>
                    </div>
                </div>
            ` : ''}
        `;

        element.addEventListener('click', () => this.selectModel(model));

        return element;
    }

    getModelTypeIcon(type) {
        const icons = {
            'lstm': '🧠',
            'transformer': '🔤',
            'xgboost': '🌲',
            'gan': '🎭'
        };
        return icons[type] || '🤖';
    }

    getStatusText(status) {
        const statusTexts = {
            'training': 'Обучается',
            'trained': 'Обучена',
            'idle': 'Ожидает',
            'evaluating': 'Оценивается'
        };
        return statusTexts[status] || status;
    }

    selectModel(model) {
        // Снятие выделения с предыдущей модели
        const previousSelected = this.mlModelsListEl.querySelector('.ml-model-item.selected');
        if (previousSelected) {
            previousSelected.classList.remove('selected');
        }

        // Выделение новой модели
        const newSelected = this.mlModelsListEl.querySelector(`[data-model-id="${model.id}"]`);
        if (newSelected) {
            newSelected.classList.add('selected');
        }

        this.selectedModel = model;
        this.updateSelectedModelDetails(model);
        this.updateSelectedModelControls(model);
        this.updateHyperparameters(model);
        this.updateTrainingChartForModel(model);

        if (this.selectedModelNameEl) {
            this.selectedModelNameEl.textContent = `🤖 ${model.name}`;
        }
    }

    updateSelectedModelDetails(model) {
        if (!this.modelDetailsEl) return;

        this.modelDetailsEl.innerHTML = `
            <div class="model-detail-content">
                <div class="model-detail-header">
                    <div class="model-detail-title">
                        <h4>${model.name}</h4>
                        <div class="model-detail-badges">
                            <span class="model-detail-badge type-${model.type}">${model.type.toUpperCase()}</span>
                            <span class="model-detail-badge status-${model.status}">${this.getStatusText(model.status)}</span>
                            <span class="model-detail-badge version">v${model.version}</span>
                        </div>
                    </div>
                </div>
                
                <div class="model-detail-performance">
                    <h5>🎯 Производительность</h5>
                    <div class="performance-metrics-detailed">
                        ${model.performance.training_accuracy ? `
                            <div class="performance-metric-detailed">
                                <span class="metric-label">Training Accuracy:</span>
                                <span class="metric-value">${model.performance.training_accuracy.toFixed(1)}%</span>
                            </div>
                        ` : ''}
                        ${model.performance.validation_accuracy ? `
                            <div class="performance-metric-detailed">
                                <span class="metric-label">Validation Accuracy:</span>
                                <span class="metric-value">${model.performance.validation_accuracy.toFixed(1)}%</span>
                            </div>
                        ` : ''}
                        ${model.performance.training_loss ? `
                            <div class="performance-metric-detailed">
                                <span class="metric-label">Training Loss:</span>
                                <span class="metric-value">${model.performance.training_loss.toFixed(4)}</span>
                            </div>
                        ` : ''}
                        ${model.performance.validation_loss ? `
                            <div class="performance-metric-detailed">
                                <span class="metric-label">Validation Loss:</span>
                                <span class="metric-value">${model.performance.validation_loss.toFixed(4)}</span>
                            </div>
                        ` : ''}
                    </div>
                </div>

                <div class="model-detail-info">
                    <h5>ℹ️ Информация</h5>
                    <div class="model-info-grid">
                        <div class="model-info-item">
                            <span class="info-label">Последнее обучение:</span>
                            <span class="info-value">${this.formatDate(model.lastTrained)}</span>
                        </div>
                        <div class="model-info-item">
                            <span class="info-label">Прогресс обучения:</span>
                            <span class="info-value">${model.trainingProgress}%</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    updateSelectedModelControls(model) {
        const canTrain = model.status === 'idle' || model.status === 'trained';
        const canEvaluate = model.status === 'trained';
        const canDeploy = model.status === 'trained';

        if (this.trainSelectedModelBtn) {
            this.trainSelectedModelBtn.disabled = !canTrain;
        }

        if (this.evaluateSelectedModelBtn) {
            this.evaluateSelectedModelBtn.disabled = !canEvaluate;
        }

        if (this.deploySelectedModelBtn) {
            this.deploySelectedModelBtn.disabled = !canDeploy;
        }
    }

    updateHyperparameters(model) {
        if (!this.hyperparametersSectionEl) return;

        const hyperparams = this.getDemoHyperparameters(model.type);
        
        this.hyperparametersSectionEl.innerHTML = `
            <div class="hyperparameters-form">
                ${Object.entries(hyperparams).map(([key, value]) => `
                    <div class="form-group">
                        <label class="form-label">${this.formatParameterName(key)}:</label>
                        <input type="number" 
                               class="form-input hyperparameter-input" 
                               data-param="${key}" 
                               value="${value}" 
                               step="${this.getParameterStep(key)}">
                    </div>
                `).join('')}
                <button class="form-button" id="saveHyperparametersBtn">
                    <i class="fas fa-save"></i>
                    <span>Сохранить</span>
                </button>
            </div>
        `;

        // Добавление обработчика для кнопки сохранения
        const saveBtn = this.hyperparametersSectionEl.querySelector('#saveHyperparametersBtn');
        if (saveBtn) {
            saveBtn.addEventListener('click', () => this.saveHyperparameters());
        }
    }

    getDemoHyperparameters(type) {
        const hyperparams = {
            'lstm': {
                learning_rate: 0.001,
                batch_size: 64,
                epochs: 100,
                sequence_length: 60
            },
            'transformer': {
                learning_rate: 0.00002,
                batch_size: 16,
                epochs: 10,
                warmup_steps: 500
            },
            'xgboost': {
                learning_rate: 0.1,
                max_depth: 8,
                n_estimators: 1000,
                subsample: 0.8
            },
            'gan': {
                learning_rate_g: 0.0002,
                learning_rate_d: 0.0002,
                batch_size: 128,
                epochs: 500
            }
        };
        return hyperparams[type] || {};
    }

    formatParameterName(param) {
        return param.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    getParameterStep(param) {
        if (param.includes('learning_rate')) return 0.00001;
        if (param.includes('rate')) return 0.01;
        return 1;
    }

    updateActiveJobs(jobs) {
        if (!this.trainingJobsListEl || !jobs) return;

        this.trainingJobsListEl.innerHTML = '';

        if (jobs.length === 0) {
            this.trainingJobsListEl.innerHTML = '<div class="no-jobs">Нет активных заданий</div>';
            return;
        }

        jobs.forEach(job => {
            const jobEl = this.createJobElement(job);
            this.trainingJobsListEl.appendChild(jobEl);
        });
    }

    createJobElement(job) {
        const element = document.createElement('div');
        element.className = 'training-job-item';

        const elapsed = Math.floor((Date.now() - new Date(job.startTime).getTime()) / 1000 / 60);

        element.innerHTML = `
            <div class="job-header">
                <div class="job-model-name">${job.modelName}</div>
                <div class="job-status ${job.status}">${job.status}</div>
            </div>
            <div class="job-progress">
                <div class="job-progress-text">
                    Эпоха ${job.currentEpoch}/${job.totalEpochs} (${job.progress}%)
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${job.progress}%"></div>
                </div>
            </div>
            <div class="job-info">
                <span>Время: ${elapsed} мин</span>
                <button class="job-stop-btn" data-job-id="${job.id}">
                    <i class="fas fa-stop"></i>
                </button>
            </div>
        `;

        // Обработчик для остановки задания
        const stopBtn = element.querySelector('.job-stop-btn');
        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stopTrainingJob(job.id));
        }

        return element;
    }

    updateTrainingHistory(history) {
        if (!this.trainingHistoryEl || !history) return;

        this.trainingHistoryEl.innerHTML = '';

        if (history.length === 0) {
            this.trainingHistoryEl.innerHTML = '<div class="no-history">Нет истории</div>';
            return;
        }

        history.forEach(entry => {
            const historyEl = this.createHistoryElement(entry);
            this.trainingHistoryEl.appendChild(historyEl);
        });
    }

    createHistoryElement(entry) {
        const element = document.createElement('div');
        element.className = 'history-entry';

        element.innerHTML = `
            <div class="history-time">${this.formatDate(entry.timestamp)}</div>
            <div class="history-type ${entry.type}">${entry.type}</div>
            <div class="history-message">${entry.message}</div>
        `;

        return element;
    }

    updateEvaluationResults(results) {
        if (!this.evaluationResultsEl || !results) return;

        this.evaluationResultsEl.innerHTML = '';

        if (results.length === 0) {
            this.evaluationResultsEl.innerHTML = '<div class="no-results">Нет результатов</div>';
            return;
        }

        results.forEach(result => {
            const resultEl = this.createResultElement(result);
            this.evaluationResultsEl.appendChild(resultEl);
        });
    }

    createResultElement(result) {
        const element = document.createElement('div');
        element.className = 'evaluation-result-item';

        element.innerHTML = `
            <div class="result-header">
                <div class="result-model-name">${result.modelName}</div>
                <div class="result-time">${this.formatDate(result.timestamp)}</div>
            </div>
            <div class="result-metrics">
                <div class="result-metric">
                    <span>Accuracy:</span>
                    <span class="value">${result.testMetrics?.accuracy?.toFixed(1) || 'N/A'}%</span>
                </div>
                <div class="result-metric">
                    <span>Precision:</span>
                    <span class="value">${result.testMetrics?.precision?.toFixed(3) || 'N/A'}</span>
                </div>
            </div>
        `;

        return element;
    }

    generateDemoTrainingChart() {
        if (!this.trainingChart) return;

        const epochs = Array.from({length: 50}, (_, i) => i + 1);
        const trainingLoss = epochs.map(e => 0.8 - (e * 0.01) + Math.random() * 0.1);
        const validationLoss = epochs.map(e => 0.85 - (e * 0.008) + Math.random() * 0.12);
        const trainingAcc = epochs.map(e => 40 + (e * 0.6) + Math.random() * 3);
        const validationAcc = epochs.map(e => 35 + (e * 0.55) + Math.random() * 4);

        this.trainingChart.data.labels = epochs;
        this.trainingChart.data.datasets[0].data = trainingLoss;
        this.trainingChart.data.datasets[1].data = validationLoss;
        this.trainingChart.data.datasets[2].data = trainingAcc;
        this.trainingChart.data.datasets[3].data = validationAcc;
        this.trainingChart.update();
    }

    updateTrainingChartForModel(model) {
        if (!this.trainingChart || !model) return;

        // Для выбранной модели показываем ее данные обучения
        if (model.status === 'training' || model.status === 'trained') {
            this.generateDemoTrainingChart();
        } else {
            // Очищаем график для моделей без данных обучения
            this.trainingChart.data.labels = [];
            this.trainingChart.data.datasets.forEach(dataset => {
                dataset.data = [];
            });
            this.trainingChart.update();
        }
    }

    updateMLControls(isTraining) {
        this.isTraining = isTraining;

        if (this.startMLTrainingBtn) {
            this.startMLTrainingBtn.disabled = isTraining;
            this.startMLTrainingBtn.innerHTML = isTraining ? 
                '<i class="fas fa-spinner fa-spin"></i><span>Обучается</span>' :
                '<i class="fas fa-brain"></i><span>Запустить обучение</span>';
        }

        if (this.stopMLTrainingBtn) {
            this.stopMLTrainingBtn.disabled = !isTraining;
        }
    }

    async startGeneralTraining() {
        try {
            if (window.electronAPI) {
                // Выбираем случайную модель для обучения
                const idleModels = Array.from(document.querySelectorAll('.ml-model-item[data-model-status="idle"]'));
                if (idleModels.length > 0) {
                    const randomModel = idleModels[Math.floor(Math.random() * idleModels.length)];
                    const modelId = randomModel.dataset.modelId;
                    
                    const result = await window.electronAPI.startMLTraining(modelId);
                    if (result.success) {
                        this.showNotification('ML', `Обучение запущено для модели`);
                        this.updateMLData();
                    } else {
                        this.showError('Ошибка', result.message || 'Не удалось запустить обучение');
                    }
                }
            } else {
                // Демо режим
                this.isTraining = true;
                this.updateMLControls(true);
                this.showNotification('ML', 'Обучение запущено (демо режим)');
            }
        } catch (error) {
            this.showError('Ошибка', error.message);
        }
    }

    async trainSelectedModel() {
        if (!this.selectedModel) return;

        try {
            if (window.electronAPI) {
                const result = await window.electronAPI.startMLTraining(this.selectedModel.id);
                if (result.success) {
                    this.showNotification('ML', `Обучение запущено для ${this.selectedModel.name}`);
                    this.updateMLData();
                } else {
                    this.showError('Ошибка', result.message || 'Не удалось запустить обучение');
                }
            } else {
                // Демо режим
                this.showNotification('ML', `Обучение запущено для ${this.selectedModel.name} (демо режим)`);
            }
        } catch (error) {
            this.showError('Ошибка', error.message);
        }
    }

    async evaluateSelectedModel() {
        if (!this.selectedModel) return;

        try {
            if (window.electronAPI) {
                const result = await window.electronAPI.runMLEvaluation(this.selectedModel.id);
                if (result.success) {
                    this.showNotification('ML', `Оценка завершена для ${this.selectedModel.name}`);
                    this.displayEvaluationResults(result.results);
                } else {
                    this.showError('Ошибка', result.message || 'Не удалось выполнить оценку');
                }
            } else {
                // Демо режим
                this.showNotification('ML', `Оценка запущена для ${this.selectedModel.name} (демо режим)`);
                setTimeout(() => {
                    this.displayDemoEvaluationResults();
                }, 2000);
            }
        } catch (error) {
            this.showError('Ошибка', error.message);
        }
    }

    async deploySelectedModel() {
        if (!this.selectedModel) return;

        try {
            if (window.electronAPI) {
                const result = await window.electronAPI.deployModel(this.selectedModel.id, 'production');
                if (result.success) {
                    this.showNotification('ML', `Модель ${this.selectedModel.name} развернута в production`);
                } else {
                    this.showError('Ошибка', result.message || 'Не удалось развернуть модель');
                }
            } else {
                // Демо режим
                this.showNotification('ML', `Модель ${this.selectedModel.name} развернута (демо режим)`);
            }
        } catch (error) {
            this.showError('Ошибка', error.message);
        }
    }

    async saveHyperparameters() {
        if (!this.selectedModel) return;

        try {
            const hyperparams = {};
            const inputs = this.hyperparametersSectionEl.querySelectorAll('.hyperparameter-input');
            
            inputs.forEach(input => {
                const param = input.dataset.param;
                const value = parseFloat(input.value);
                hyperparams[param] = value;
            });

            if (window.electronAPI) {
                const result = await window.electronAPI.updateModelHyperparameters(this.selectedModel.id, hyperparams);
                if (result.success) {
                    this.showNotification('ML', 'Гиперпараметры сохранены');
                } else {
                    this.showError('Ошибка', result.message || 'Не удалось сохранить гиперпараметры');
                }
            } else {
                // Демо режим
                this.showNotification('ML', 'Гиперпараметры сохранены (демо режим)');
            }
        } catch (error) {
            this.showError('Ошибка', error.message);
        }
    }

    filterModels() {
        const statusFilter = this.mlModelFilterEl?.value || 'all';
        const typeFilter = this.modelTypeFilterEl?.value || 'all';
        const models = this.mlModelsListEl?.querySelectorAll('.ml-model-item');

        if (!models) return;

        models.forEach(model => {
            const modelStatus = model.dataset.modelStatus;
            const modelType = model.dataset.modelType;
            
            const statusMatch = statusFilter === 'all' || modelStatus === statusFilter;
            const typeMatch = typeFilter === 'all' || modelType === typeFilter;
            
            model.style.display = (statusMatch && typeMatch) ? 'block' : 'none';
        });
    }

    formatDate(dateString) {
        if (!dateString) return 'N/A';
        const date = new Date(dateString);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);

        if (diffMins < 60) return `${diffMins} мин назад`;
        if (diffHours < 24) return `${diffHours} ч назад`;
        if (diffDays < 7) return `${diffDays} дн назад`;
        return date.toLocaleDateString('ru-RU');
    }

    showNotification(title, message) {
        console.log(`🔔 ${title}: ${message}`);
        if (window.atbApp) {
            window.atbApp.showNotification(title, message);
        }
    }

    showError(title, message) {
        console.error(`❌ ${title}: ${message}`);
        if (window.atbApp) {
            window.atbApp.showError(title, message);
        }
    }
}

// Инициализация ML UI
let mlUI = null;

document.addEventListener('DOMContentLoaded', () => {
    mlUI = new MLUI();
    
    // Первоначальная загрузка данных
    setTimeout(() => {
        if (mlUI) {
            mlUI.updateMLData();
        }
    }, 2000);

    // Периодическое обновление данных ML
    setInterval(() => {
        if (mlUI) {
            mlUI.updateMLData();
        }
    }, 10000); // Каждые 10 секунд
});

// Экспорт для использования в других модулях
if (typeof window !== 'undefined') {
    window.mlUI = mlUI;
}

console.log('⚡ ML UI module loaded');