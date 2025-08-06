#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive тесты machine learning моделей для торговой системы.
Критически важно для финансовой системы - точность и надежность ML прогнозов.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from unittest.mock import Mock, patch
import joblib
import json

from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.entities.market_data import MarketData
from infrastructure.ml.models.price_prediction_model import PricePredictionModel
from infrastructure.ml.models.risk_assessment_model import RiskAssessmentModel
from infrastructure.ml.models.sentiment_analysis_model import SentimentAnalysisModel
from infrastructure.ml.models.anomaly_detection_model import AnomalyDetectionModel
from infrastructure.ml.models.portfolio_optimization_model import PortfolioOptimizationModel
from infrastructure.ml.feature_engineering.technical_indicators import TechnicalIndicators
from infrastructure.ml.feature_engineering.market_features import MarketFeatureExtractor
from infrastructure.ml.data_preprocessing.data_cleaner import DataCleaner
from infrastructure.ml.data_preprocessing.feature_scaler import FeatureScaler
from infrastructure.ml.model_validation.cross_validator import CrossValidator
from infrastructure.ml.model_validation.backtester import MLBacktester
from infrastructure.ml.model_monitoring.performance_monitor import ModelPerformanceMonitor
from infrastructure.ml.model_monitoring.drift_detector import DataDriftDetector
from domain.exceptions import MLModelError, ValidationError, PredictionError


class ModelType(Enum):
    """Типы ML моделей."""
    PRICE_PREDICTION = "PRICE_PREDICTION"
    RISK_ASSESSMENT = "RISK_ASSESSMENT"
    SENTIMENT_ANALYSIS = "SENTIMENT_ANALYSIS"
    ANOMALY_DETECTION = "ANOMALY_DETECTION"
    PORTFOLIO_OPTIMIZATION = "PORTFOLIO_OPTIMIZATION"
    MARKET_REGIME_DETECTION = "MARKET_REGIME_DETECTION"
    VOLATILITY_FORECASTING = "VOLATILITY_FORECASTING"


class ModelStatus(Enum):
    """Статусы моделей."""
    TRAINING = "TRAINING"
    TRAINED = "TRAINED"
    DEPLOYED = "DEPLOYED"
    RETRAINING = "RETRAINING"
    ERROR = "ERROR"
    DEPRECATED = "DEPRECATED"


@dataclass
class ModelMetrics:
    """Метрики модели."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse: float
    mae: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    hit_rate: Optional[float] = None


@dataclass
class PredictionResult:
    """Результат предсказания."""
    model_type: ModelType
    prediction: Any
    confidence: float
    timestamp: datetime
    features_used: List[str]
    model_version: str


class TestMLTradingModelsComprehensive:
    """Comprehensive тесты ML моделей."""

    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        """Фикстура рыночных данных."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1H')
        np.random.seed(42)
        
        # Генерируем реалистичные данные
        price_base = 45000
        returns = np.random.normal(0, 0.02, len(dates))
        prices = [price_base]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        volumes = np.random.exponential(100, len(dates))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        return data

    @pytest.fixture
    def technical_indicators(self) -> TechnicalIndicators:
        """Фикстура технических индикаторов."""
        return TechnicalIndicators(
            indicators=['sma', 'ema', 'rsi', 'macd', 'bollinger_bands', 'stochastic'],
            periods=[14, 21, 50, 200],
            smoothing_factors=[0.1, 0.2, 0.3]
        )

    @pytest.fixture
    def feature_extractor(self) -> MarketFeatureExtractor:
        """Фикстура извлечения признаков."""
        return MarketFeatureExtractor(
            price_features=True,
            volume_features=True,
            volatility_features=True,
            momentum_features=True,
            time_features=True,
            market_microstructure_features=True
        )

    @pytest.fixture
    def data_cleaner(self) -> DataCleaner:
        """Фикстура очистки данных."""
        return DataCleaner(
            remove_outliers=True,
            outlier_method='iqr',
            outlier_threshold=3.0,
            fill_missing='interpolate',
            remove_duplicates=True
        )

    def test_price_prediction_model_training_and_validation(
        self,
        sample_market_data: pd.DataFrame,
        technical_indicators: TechnicalIndicators,
        feature_extractor: MarketFeatureExtractor
    ) -> None:
        """Тест обучения и валидации модели предсказания цен."""
        
        # Создаем модель
        price_model = PricePredictionModel(
            model_type='lstm',
            sequence_length=24,  # 24 часа
            hidden_layers=[128, 64, 32],
            dropout_rate=0.2,
            learning_rate=0.001
        )
        
        # Извлекаем технические индикаторы
        technical_features = technical_indicators.calculate_all(sample_market_data)
        
        # Извлекаем рыночные признаки
        market_features = feature_extractor.extract_features(sample_market_data)
        
        # Объединяем признаки
        features_df = pd.concat([sample_market_data, technical_features, market_features], axis=1)
        
        # Подготавливаем данные для обучения
        X, y = price_model.prepare_training_data(
            features_df, 
            target_column='close',
            prediction_horizon=1  # Предсказываем на 1 час вперед
        )
        
        # Разделяем на train/validation/test
        train_size = int(0.7 * len(X))
        val_size = int(0.2 * len(X))
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
        
        # Обучаем модель
        training_history = price_model.train(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            early_stopping=True,
            patience=10
        )
        
        # Проверяем обучение
        assert price_model.status == ModelStatus.TRAINED
        assert len(training_history.history['loss']) > 0
        assert training_history.history['loss'][-1] < training_history.history['loss'][0]  # Loss уменьшается
        
        # Делаем предсказания
        predictions = price_model.predict(X_test)
        
        # Вычисляем метрики
        metrics = price_model.evaluate(X_test, y_test)
        
        # Проверяем качество модели
        assert metrics.mse < 1000000  # Разумная MSE для цен
        assert metrics.mae < 1000     # Разумная MAE
        
        # Проверяем форму предсказаний
        assert len(predictions) == len(y_test)
        assert all(pred > 0 for pred in predictions)  # Цены положительные

    def test_risk_assessment_model_portfolio_analysis(
        self,
        sample_market_data: pd.DataFrame
    ) -> None:
        """Тест модели оценки рисков портфеля."""
        
        # Создаем модель оценки рисков
        risk_model = RiskAssessmentModel(
            model_type='ensemble',
            algorithms=['random_forest', 'gradient_boosting', 'neural_network'],
            risk_metrics=['var', 'cvar', 'max_drawdown', 'sharpe_ratio'],
            confidence_levels=[0.95, 0.99]
        )
        
        # Создаем портфельные данные
        portfolio_data = []
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT']
        
        for symbol in symbols:
            # Добавляем небольшую корреляцию между активами
            correlation_factor = 0.3 if symbol != 'BTCUSDT' else 0.0
            
            symbol_data = sample_market_data.copy()
            if correlation_factor > 0:
                btc_returns = sample_market_data['close'].pct_change()
                symbol_returns = btc_returns * correlation_factor + \
                               np.random.normal(0, 0.02, len(btc_returns)) * (1 - correlation_factor)
                symbol_data['close'] = symbol_data['close'].iloc[0] * (1 + symbol_returns).cumprod()
            
            symbol_data['symbol'] = symbol
            portfolio_data.append(symbol_data)
        
        portfolio_df = pd.concat(portfolio_data, ignore_index=True)
        
        # Определяем веса портфеля
        portfolio_weights = {
            'BTCUSDT': 0.4,
            'ETHUSDT': 0.3,
            'ADAUSDT': 0.2,
            'SOLUSDT': 0.1
        }
        
        # Подготавливаем признаки для оценки рисков
        risk_features = risk_model.prepare_risk_features(
            portfolio_df, 
            portfolio_weights,
            lookback_period=30
        )
        
        # Обучаем модель на исторических данных
        training_data = risk_model.prepare_training_data(risk_features)
        risk_model.train(training_data)
        
        # Оцениваем текущие риски
        current_risk_assessment = risk_model.assess_portfolio_risk(
            portfolio_df.tail(100),  # Последние 100 наблюдений
            portfolio_weights
        )
        
        # Проверяем результаты оценки рисков
        assert 'var_95' in current_risk_assessment
        assert 'var_99' in current_risk_assessment
        assert 'cvar_95' in current_risk_assessment
        assert 'max_drawdown' in current_risk_assessment
        assert 'sharpe_ratio' in current_risk_assessment
        
        # VaR должен быть отрицательным (потери)
        assert current_risk_assessment['var_95'] < 0
        assert current_risk_assessment['var_99'] < 0
        
        # CVaR должен быть больше (по модулю) чем VaR
        assert abs(current_risk_assessment['cvar_95']) >= abs(current_risk_assessment['var_95'])
        assert abs(current_risk_assessment['cvar_99']) >= abs(current_risk_assessment['var_99'])
        
        # Максимальная просадка должна быть от 0 до 1
        assert 0 <= current_risk_assessment['max_drawdown'] <= 1

    def test_sentiment_analysis_model_news_processing(self) -> None:
        """Тест модели анализа настроений для новостей."""
        
        # Создаем модель анализа настроений
        sentiment_model = SentimentAnalysisModel(
            model_type='transformer',
            pretrained_model='finbert',
            fine_tuning=True,
            sentiment_classes=['positive', 'negative', 'neutral'],
            confidence_threshold=0.7
        )
        
        # Тестовые новости
        test_news = [
            {
                "headline": "Bitcoin reaches new all-time high amid institutional adoption",
                "content": "Major corporations continue to add Bitcoin to their treasury reserves...",
                "source": "CoinDesk",
                "timestamp": datetime.utcnow(),
                "category": "cryptocurrency"
            },
            {
                "headline": "Crypto market crashes amid regulatory concerns",
                "content": "Major selling pressure hits cryptocurrency markets as regulators...",
                "source": "Bloomberg",
                "timestamp": datetime.utcnow(),
                "category": "regulation"
            },
            {
                "headline": "Ethereum 2.0 upgrade shows promising performance metrics",
                "content": "The latest Ethereum upgrade demonstrates improved scalability...",
                "source": "Ethereum Foundation",
                "timestamp": datetime.utcnow(),
                "category": "technology"
            },
            {
                "headline": "Central bank digital currencies gain momentum globally",
                "content": "Multiple central banks announce progress on CBDC initiatives...",
                "source": "Reuters",
                "timestamp": datetime.utcnow(),
                "category": "cbdc"
            }
        ]
        
        # Подготавливаем тексты для анализа
        texts = []
        for news in test_news:
            combined_text = f"{news['headline']} {news['content']}"
            texts.append(combined_text)
        
        # Анализируем настроения
        sentiment_results = []
        for i, text in enumerate(texts):
            result = sentiment_model.analyze_sentiment(
                text,
                include_confidence=True,
                return_probabilities=True
            )
            
            sentiment_results.append({
                "news_id": i,
                "sentiment": result.sentiment,
                "confidence": result.confidence,
                "probabilities": result.probabilities,
                "category": test_news[i]["category"]
            })
        
        # Проверяем результаты
        assert len(sentiment_results) == 4
        
        # Проверяем что первая новость позитивная (рост цен)
        btc_news = sentiment_results[0]
        assert btc_news["sentiment"] == "positive"
        assert btc_news["confidence"] > 0.5
        
        # Проверяем что вторая новость негативная (краш)
        crash_news = sentiment_results[1]
        assert crash_news["sentiment"] == "negative"
        assert crash_news["confidence"] > 0.5
        
        # Агрегируем настроения по категориям
        category_sentiment = sentiment_model.aggregate_sentiment_by_category(
            sentiment_results,
            time_window_hours=24
        )
        
        assert "cryptocurrency" in category_sentiment
        assert "regulation" in category_sentiment
        
        # Вычисляем sentiment score для торговых решений
        overall_sentiment_score = sentiment_model.calculate_trading_sentiment_score(
            sentiment_results,
            category_weights={
                "cryptocurrency": 0.4,
                "regulation": 0.3,
                "technology": 0.2,
                "cbdc": 0.1
            }
        )
        
        # Sentiment score должен быть между -1 и 1
        assert -1 <= overall_sentiment_score <= 1

    def test_anomaly_detection_model_trading_patterns(
        self,
        sample_market_data: pd.DataFrame
    ) -> None:
        """Тест модели обнаружения аномалий в торговых паттернах."""
        
        # Создаем модель обнаружения аномалий
        anomaly_model = AnomalyDetectionModel(
            algorithms=['isolation_forest', 'one_class_svm', 'autoencoder'],
            ensemble_method='voting',
            anomaly_threshold=0.1,  # 10% аномалий
            features=['price', 'volume', 'volatility', 'spread']
        )
        
        # Подготавливаем признаки
        features_df = sample_market_data.copy()
        
        # Добавляем дополнительные признаки
        features_df['returns'] = features_df['close'].pct_change()
        features_df['volatility'] = features_df['returns'].rolling(window=24).std()
        features_df['volume_sma'] = features_df['volume'].rolling(window=24).mean()
        features_df['price_volume_ratio'] = features_df['close'] / features_df['volume']
        features_df['spread'] = (features_df['high'] - features_df['low']) / features_df['close']
        
        # Удаляем NaN
        features_df = features_df.dropna()
        
        # Выбираем признаки для обучения
        feature_columns = ['returns', 'volatility', 'volume_sma', 'price_volume_ratio', 'spread']
        X = features_df[feature_columns].values
        
        # Обучаем модель
        anomaly_model.fit(X)
        
        # Добавляем искусственные аномалии для тестирования
        X_with_anomalies = X.copy()
        
        # Аномалия 1: Экстремальная волатильность
        anomaly_indices = []
        for i in range(5):
            idx = np.random.randint(100, len(X_with_anomalies) - 100)
            X_with_anomalies[idx, 1] *= 10  # Увеличиваем волатильность в 10 раз
            anomaly_indices.append(idx)
        
        # Аномалия 2: Аномальный объем
        for i in range(5):
            idx = np.random.randint(100, len(X_with_anomalies) - 100)
            X_with_anomalies[idx, 2] *= 20  # Увеличиваем объем в 20 раз
            anomaly_indices.append(idx)
        
        # Обнаруживаем аномалии
        anomaly_scores = anomaly_model.predict_anomaly_scores(X_with_anomalies)
        anomaly_labels = anomaly_model.predict_anomalies(X_with_anomalies)
        
        # Проверяем результаты
        assert len(anomaly_scores) == len(X_with_anomalies)
        assert len(anomaly_labels) == len(X_with_anomalies)
        
        # Проверяем что искусственные аномалии обнаружены
        detected_anomalies = np.where(anomaly_labels == -1)[0]
        detected_count = len(detected_anomalies)
        
        # Должны обнаружить хотя бы половину искусственных аномалий
        overlap = len(set(anomaly_indices) & set(detected_anomalies))
        detection_rate = overlap / len(anomaly_indices)
        assert detection_rate >= 0.5
        
        # Проверяем аномалии в реальном времени
        real_time_features = X_with_anomalies[-10:]  # Последние 10 наблюдений
        real_time_anomalies = anomaly_model.detect_real_time_anomalies(
            real_time_features,
            return_scores=True
        )
        
        assert len(real_time_anomalies) == 10
        
        # Генерируем алерты для критических аномалий
        critical_threshold = 0.9
        critical_anomalies = anomaly_model.generate_anomaly_alerts(
            real_time_anomalies,
            threshold=critical_threshold
        )
        
        # Проверяем структуру алертов
        for alert in critical_anomalies:
            assert 'timestamp' in alert
            assert 'anomaly_score' in alert
            assert 'features_affected' in alert
            assert 'severity' in alert
            assert alert['anomaly_score'] >= critical_threshold

    def test_portfolio_optimization_model_markowitz(
        self,
        sample_market_data: pd.DataFrame
    ) -> None:
        """Тест модели оптимизации портфеля (Markowitz)."""
        
        # Создаем модель оптимизации портфеля
        portfolio_model = PortfolioOptimizationModel(
            optimization_method='markowitz',
            objective='max_sharpe',
            constraints={
                'max_weight': 0.4,
                'min_weight': 0.05,
                'max_concentration': 0.6,
                'sector_limits': {}
            },
            risk_model='sample_covariance',
            expected_returns_model='historical_mean'
        )
        
        # Создаем данные для нескольких активов
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT']
        portfolio_data = {}
        
        for i, symbol in enumerate(symbols):
            # Создаем коррелированные данные
            correlation_with_btc = 0.3 + i * 0.1
            
            prices = sample_market_data['close'].copy()
            if i > 0:  # Не для BTC
                btc_returns = prices.pct_change()
                noise = np.random.normal(0, 0.03, len(prices))
                correlated_returns = btc_returns * correlation_with_btc + noise * (1 - correlation_with_btc)
                prices = prices.iloc[0] * (1 + correlated_returns).cumprod()
            
            portfolio_data[symbol] = prices
        
        returns_df = pd.DataFrame(portfolio_data).pct_change().dropna()
        
        # Обучаем модель на исторических данных
        portfolio_model.fit(returns_df)
        
        # Оптимизируем портфель
        optimal_weights = portfolio_model.optimize_portfolio(
            expected_returns=None,  # Используем исторические средние
            covariance_matrix=None,  # Используем sample covariance
            risk_aversion=1.0
        )
        
        # Проверяем ограничения
        assert len(optimal_weights) == len(symbols)
        
        # Веса должны быть положительными и в пределах ограничений
        for symbol, weight in optimal_weights.items():
            assert 0.05 <= weight <= 0.4  # Ограничения min/max weight
        
        # Сумма весов должна быть равна 1
        total_weight = sum(optimal_weights.values())
        assert abs(total_weight - 1.0) < 0.01
        
        # Рассчитываем метрики портфеля
        portfolio_metrics = portfolio_model.calculate_portfolio_metrics(
            optimal_weights,
            returns_df
        )
        
        assert 'expected_return' in portfolio_metrics
        assert 'volatility' in portfolio_metrics
        assert 'sharpe_ratio' in portfolio_metrics
        assert 'max_drawdown' in portfolio_metrics
        assert 'var_95' in portfolio_metrics
        
        # Sharpe ratio должен быть разумным
        assert portfolio_metrics['sharpe_ratio'] > -2.0
        assert portfolio_metrics['sharpe_ratio'] < 10.0
        
        # Максимальная просадка должна быть от 0 до 1
        assert 0 <= portfolio_metrics['max_drawdown'] <= 1
        
        # Тестируем робастную оптимизацию
        robust_weights = portfolio_model.robust_optimize(
            returns_df,
            uncertainty_sets=['box', 'elliptical'],
            confidence_level=0.95
        )
        
        # Робастные веса также должны удовлетворять ограничениям
        for symbol, weight in robust_weights.items():
            assert 0.05 <= weight <= 0.4
        
        total_robust_weight = sum(robust_weights.values())
        assert abs(total_robust_weight - 1.0) < 0.01

    def test_model_cross_validation_and_backtesting(
        self,
        sample_market_data: pd.DataFrame,
        technical_indicators: TechnicalIndicators
    ) -> None:
        """Тест кросс-валидации и бэктестинга моделей."""
        
        # Создаем валидатор
        cross_validator = CrossValidator(
            cv_method='time_series_split',
            n_splits=5,
            test_size=0.2,
            gap=24  # 24-часовой разрыв между train и test
        )
        
        # Создаем бэктестер
        backtester = MLBacktester(
            initial_capital=100000,
            transaction_costs=0.001,  # 0.1%
            slippage=0.0005,  # 0.05%
            position_sizing='kelly',
            risk_management=True
        )
        
        # Подготавливаем данные
        features_df = technical_indicators.calculate_all(sample_market_data)
        combined_df = pd.concat([sample_market_data, features_df], axis=1).dropna()
        
        # Создаем простую модель предсказания направления
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Определяем цель (направление движения цены)
        combined_df['future_return'] = combined_df['close'].shift(-1) / combined_df['close'] - 1
        combined_df['direction'] = (combined_df['future_return'] > 0).astype(int)
        combined_df = combined_df.dropna()
        
        # Выбираем признаки
        feature_columns = [col for col in combined_df.columns 
                          if col not in ['timestamp', 'future_return', 'direction', 
                                       'open', 'high', 'low', 'close', 'volume']]
        
        X = combined_df[feature_columns]
        y = combined_df['direction']
        
        # Кросс-валидация
        cv_results = cross_validator.cross_validate(
            RandomForestClassifier(n_estimators=100, random_state=42),
            X, y,
            scoring=['accuracy', 'precision', 'recall', 'f1']
        )
        
        # Проверяем результаты CV
        assert 'test_accuracy' in cv_results
        assert 'test_precision' in cv_results
        assert 'test_recall' in cv_results
        assert 'test_f1' in cv_results
        
        # Средняя точность должна быть больше случайной (>0.5)
        mean_accuracy = np.mean(cv_results['test_accuracy'])
        assert mean_accuracy > 0.5
        
        # Обучаем финальную модель
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Разделяем данные для бэктестинга
        split_point = int(0.8 * len(X_scaled))
        
        X_train, X_test = X_scaled[:split_point], X_scaled[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        model.fit(X_train, y_train)
        
        # Генерируем сигналы
        predictions = model.predict_proba(X_test)[:, 1]  # Вероятность роста
        
        # Создаем торговые сигналы
        signals_df = pd.DataFrame({
            'timestamp': combined_df.iloc[split_point:]['timestamp'].values,
            'price': combined_df.iloc[split_point:]['close'].values,
            'signal_strength': predictions,
            'position': np.where(predictions > 0.6, 1,
                                np.where(predictions < 0.4, -1, 0))
        })
        
        # Запускаем бэктест
        backtest_results = backtester.run_backtest(signals_df)
        
        # Проверяем результаты бэктеста
        assert 'total_return' in backtest_results
        assert 'sharpe_ratio' in backtest_results
        assert 'max_drawdown' in backtest_results
        assert 'win_rate' in backtest_results
        assert 'profit_factor' in backtest_results
        
        # Общий доход должен быть не слишком плохим
        assert backtest_results['total_return'] > -0.5  # Не более 50% потерь
        
        # Win rate должен быть разумным
        assert 0.3 <= backtest_results['win_rate'] <= 0.8
        
        # Максимальная просадка должна быть приемлемой
        assert backtest_results['max_drawdown'] < 0.5  # Менее 50%

    def test_model_performance_monitoring_and_drift_detection(
        self,
        sample_market_data: pd.DataFrame
    ) -> None:
        """Тест мониторинга производительности и детекции дрифта."""
        
        # Создаем монитор производительности
        performance_monitor = ModelPerformanceMonitor(
            metrics=['accuracy', 'precision', 'recall', 'auc'],
            monitoring_window=100,
            alert_thresholds={
                'accuracy': 0.6,
                'precision': 0.5,
                'auc': 0.6
            },
            trend_detection=True
        )
        
        # Создаем детектор дрифта
        drift_detector = DataDriftDetector(
            drift_tests=['ks_test', 'chi2_test', 'psi'],
            reference_window=500,
            monitoring_window=100,
            drift_threshold=0.05,
            feature_importance_weighting=True
        )
        
        # Подготавливаем данные
        features_df = sample_market_data.copy()
        features_df['returns'] = features_df['close'].pct_change()
        features_df['volatility'] = features_df['returns'].rolling(window=24).std()
        features_df = features_df.dropna()
        
        # Создаем референсные данные (первые 500 наблюдений)
        reference_data = features_df.iloc[:500]
        
        # Симулируем изменения в данных со временем (дрифт)
        monitoring_data_segments = []
        
        for i in range(5):  # 5 сегментов по 100 наблюдений
            start_idx = 500 + i * 100
            end_idx = start_idx + 100
            
            segment_data = features_df.iloc[start_idx:end_idx].copy()
            
            # Добавляем дрифт в зависимости от сегмента
            if i >= 2:  # Начиная с 3-го сегмента
                # Сдвигаем распределение цен
                drift_factor = 1 + (i - 1) * 0.1
                segment_data['close'] *= drift_factor
                segment_data['returns'] *= drift_factor
                
                # Увеличиваем волатильность
                segment_data['volatility'] *= (1 + (i - 1) * 0.2)
            
            monitoring_data_segments.append(segment_data)
        
        # Инициализируем референсные статистики
        reference_features = ['close', 'volume', 'returns', 'volatility']
        drift_detector.set_reference_data(reference_data[reference_features])
        
        # Мониторим каждый сегмент
        drift_results = []
        performance_results = []
        
        for i, segment_data in enumerate(monitoring_data_segments):
            # Детекция дрифта
            drift_result = drift_detector.detect_drift(
                segment_data[reference_features],
                return_details=True
            )
            
            drift_results.append({
                'segment': i,
                'drift_detected': drift_result.drift_detected,
                'drift_score': drift_result.drift_score,
                'affected_features': drift_result.affected_features,
                'drift_magnitude': drift_result.drift_magnitude
            })
            
            # Симулируем производительность модели (ухудшается при дрифте)
            base_accuracy = 0.75
            if drift_result.drift_detected:
                # Производительность падает при дрифте
                accuracy = base_accuracy - (drift_result.drift_score * 0.3)
                precision = base_accuracy - (drift_result.drift_score * 0.25)
                recall = base_accuracy - (drift_result.drift_score * 0.2)
            else:
                # Небольшие случайные колебания
                accuracy = base_accuracy + np.random.normal(0, 0.05)
                precision = base_accuracy + np.random.normal(0, 0.04)
                recall = base_accuracy + np.random.normal(0, 0.03)
            
            # Мониторинг производительности
            performance_result = performance_monitor.update_metrics({
                'accuracy': max(0, min(1, accuracy)),
                'precision': max(0, min(1, precision)),
                'recall': max(0, min(1, recall)),
                'auc': max(0, min(1, accuracy + 0.1))
            })
            
            performance_results.append({
                'segment': i,
                'metrics': performance_result.current_metrics,
                'alerts': performance_result.alerts,
                'trend': performance_result.trend_analysis
            })
        
        # Проверяем результаты детекции дрифта
        # Первые сегменты не должны показывать дрифт
        assert not drift_results[0]['drift_detected']
        assert not drift_results[1]['drift_detected']
        
        # Последние сегменты должны показывать дрифт
        assert drift_results[3]['drift_detected'] or drift_results[4]['drift_detected']
        
        # Проверяем мониторинг производительности
        # Должны быть алерты о снижении производительности
        total_alerts = sum(len(result['alerts']) for result in performance_results)
        assert total_alerts > 0
        
        # Проверяем что тренд производительности ухудшается
        final_accuracy = performance_results[-1]['metrics']['accuracy']
        initial_accuracy = performance_results[0]['metrics']['accuracy']
        assert final_accuracy < initial_accuracy

    def test_model_ensemble_and_meta_learning(
        self,
        sample_market_data: pd.DataFrame
    ) -> None:
        """Тест ансамблей моделей и мета-обучения."""
        
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        
        # Подготавливаем данные
        features_df = sample_market_data.copy()
        features_df['returns'] = features_df['close'].pct_change()
        features_df['sma_5'] = features_df['close'].rolling(5).mean()
        features_df['sma_20'] = features_df['close'].rolling(20).mean()
        features_df['rsi'] = self._calculate_rsi(features_df['close'], 14)
        features_df = features_df.dropna()
        
        # Определяем цель
        features_df['future_return'] = features_df['close'].shift(-1) / features_df['close'] - 1
        features_df['direction'] = (features_df['future_return'] > 0).astype(int)
        features_df = features_df.dropna()
        
        feature_columns = ['returns', 'sma_5', 'sma_20', 'rsi']
        X = features_df[feature_columns].values
        y = features_df['direction'].values
        
        # Разделяем данные
        split_point = int(0.8 * len(X))
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        # Создаем базовые модели
        base_models = {
            'rf': RandomForestClassifier(n_estimators=50, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=50, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'lr': LogisticRegression(random_state=42),
            'mlp': MLPClassifier(hidden_layer_sizes=(50,), random_state=42, max_iter=500)
        }
        
        # Обучаем базовые модели
        base_predictions = {}
        base_probabilities = {}
        
        for name, model in base_models.items():
            model.fit(X_train, y_train)
            base_predictions[name] = model.predict(X_test)
            base_probabilities[name] = model.predict_proba(X_test)[:, 1]
        
        # Создаем ансамбль методом голосования
        from scipy import stats
        
        voting_predictions = []
        for i in range(len(X_test)):
            votes = [base_predictions[name][i] for name in base_models.keys()]
            majority_vote = stats.mode(votes, keepdims=True)[0][0]
            voting_predictions.append(majority_vote)
        
        # Создаем ансамбль методом усреднения вероятностей
        avg_probabilities = np.mean([base_probabilities[name] for name in base_models.keys()], axis=0)
        avg_predictions = (avg_probabilities > 0.5).astype(int)
        
        # Создаем стекинг ансамбль
        # Используем предсказания базовых моделей как признаки для мета-модели
        meta_features = np.column_stack([base_probabilities[name] for name in base_models.keys()])
        
        # Разделяем на train/val для мета-модели
        meta_split = int(0.8 * len(meta_features))
        meta_X_train, meta_X_val = meta_features[:meta_split], meta_features[meta_split:]
        meta_y_train, meta_y_val = y_test[:meta_split], y_test[meta_split:]
        
        # Обучаем мета-модель
        meta_model = LogisticRegression(random_state=42)
        meta_model.fit(meta_X_train, meta_y_train)
        stacking_predictions = meta_model.predict(meta_X_val)
        
        # Вычисляем метрики для каждого ансамбля
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        # Базовые модели
        base_accuracies = {}
        for name in base_models.keys():
            base_accuracies[name] = accuracy_score(y_test, base_predictions[name])
        
        # Ансамбли
        voting_accuracy = accuracy_score(y_test, voting_predictions)
        avg_accuracy = accuracy_score(y_test, avg_predictions)
        stacking_accuracy = accuracy_score(meta_y_val, stacking_predictions)
        
        # Проверяем результаты
        # Все модели должны показывать разумную точность
        for name, accuracy in base_accuracies.items():
            assert 0.4 <= accuracy <= 0.8, f"Model {name} accuracy {accuracy} is unreasonable"
        
        # Ансамбли обычно должны быть не хуже лучшей базовой модели
        best_base_accuracy = max(base_accuracies.values())
        
        assert voting_accuracy >= best_base_accuracy * 0.9  # Не более 10% хуже
        assert avg_accuracy >= best_base_accuracy * 0.9
        
        # Стекинг часто показывает лучшие результаты
        # но проверяем только разумность
        assert 0.4 <= stacking_accuracy <= 0.9

    def test_model_deployment_and_serving(self) -> None:
        """Тест развертывания и обслуживания моделей."""
        
        from infrastructure.ml.model_serving.model_server import ModelServer
        from infrastructure.ml.model_serving.model_registry import ModelRegistry
        from infrastructure.ml.model_serving.prediction_cache import PredictionCache
        
        # Создаем реестр моделей
        model_registry = ModelRegistry(
            storage_backend='filesystem',
            versioning=True,
            metadata_tracking=True
        )
        
        # Создаем простую модель для тестирования
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_dummy = np.random.random((100, 5))
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)
        
        # Регистрируем модель
        model_metadata = {
            'model_type': 'price_direction_classifier',
            'version': '1.0.0',
            'training_date': datetime.utcnow().isoformat(),
            'features': ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'],
            'performance_metrics': {
                'accuracy': 0.75,
                'precision': 0.73,
                'recall': 0.76
            },
            'framework': 'sklearn',
            'preprocessing': 'standard_scaler'
        }
        
        model_id = model_registry.register_model(
            model=model,
            metadata=model_metadata,
            model_name='price_direction_v1'
        )
        
        # Создаем кэш предсказаний
        prediction_cache = PredictionCache(
            cache_type='redis',
            ttl_seconds=300,  # 5 минут
            max_cache_size=10000
        )
        
        # Создаем сервер моделей
        model_server = ModelServer(
            model_registry=model_registry,
            prediction_cache=prediction_cache,
            max_batch_size=100,
            timeout_seconds=30
        )
        
        # Загружаем модель в сервер
        model_server.load_model(model_id)
        
        # Проверяем статус модели
        model_status = model_server.get_model_status(model_id)
        assert model_status['status'] == 'loaded'
        assert model_status['model_id'] == model_id
        
        # Тестируем одиночное предсказание
        single_input = {
            'features': [0.1, 0.2, 0.3, 0.4, 0.5],
            'model_id': model_id
        }
        
        single_prediction = model_server.predict(single_input)
        
        assert 'prediction' in single_prediction
        assert 'confidence' in single_prediction
        assert 'model_version' in single_prediction
        assert 'timestamp' in single_prediction
        
        # Тестируем батчевое предсказание
        batch_input = {
            'features_batch': [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.2, 0.3, 0.4, 0.5, 0.6],
                [0.3, 0.4, 0.5, 0.6, 0.7]
            ],
            'model_id': model_id
        }
        
        batch_predictions = model_server.predict_batch(batch_input)
        
        assert len(batch_predictions) == 3
        for pred in batch_predictions:
            assert 'prediction' in pred
            assert 'confidence' in pred
        
        # Тестируем кэширование
        # Повторный запрос должен вернуться из кэша быстрее
        import time
        
        start_time = time.time()
        cached_prediction = model_server.predict(single_input)
        cached_time = time.time() - start_time
        
        # Предсказания должны быть идентичными
        assert single_prediction['prediction'] == cached_prediction['prediction']
        
        # Тестируем мониторинг модели в продакшене
        serving_metrics = model_server.get_serving_metrics(model_id)
        
        assert 'total_requests' in serving_metrics
        assert 'average_latency_ms' in serving_metrics
        assert 'cache_hit_rate' in serving_metrics
        assert 'error_rate' in serving_metrics
        
        assert serving_metrics['total_requests'] >= 4  # 1 single + 3 batch + cached
        assert serving_metrics['cache_hit_rate'] > 0  # Кэш должен сработать

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Вспомогательная функция для расчета RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi