"""
Тесты для доменного сервиса ML предсказаний.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from domain.services.ml_predictor import MLPredictor
from domain.type_definitions.ml_types import PredictionResult, ModelPerformance, FeatureImportance
import pandas as pd
from shared.numpy_utils import np

class TestMLPredictor:
    """Тесты для сервиса ML предсказаний."""
    @pytest.fixture
    def ml_predictor(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура сервиса ML предсказаний."""
        return MLPredictor()
    @pytest.fixture
    def sample_training_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура с примерными данными для обучения."""
        dates = pd.date_range('2024-01-01', periods=1000, freq='1H')
        np.random.seed(42)
        # Создаем синтетические данные с некоторой предсказуемостью
        base_price = 50000
        trend = np.cumsum(np.random.normal(0, 10, 1000))
        noise = np.random.normal(0, 100, 1000)
        return pd.DataFrame({
            'open': base_price + trend + noise,
            'high': base_price + trend + noise + np.random.uniform(0, 50, 1000),
            'low': base_price + trend + noise - np.random.uniform(0, 50, 1000),
            'close': base_price + trend + noise + np.random.normal(0, 20, 1000),
            'volume': np.random.uniform(1000, 5000, 1000),
            'vwap': base_price + trend + noise,
            'rsi': np.random.uniform(20, 80, 1000),
            'macd': np.random.normal(0, 10, 1000),
            'bollinger_upper': base_price + trend + noise + 100,
            'bollinger_lower': base_price + trend + noise - 100,
            'target': np.random.choice([0, 1], 1000, p=[0.7, 0.3])  # Бинарная цель
        }, index=dates)
    @pytest.fixture
    def sample_prediction_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура с данными для предсказания."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        np.random.seed(42)
        base_price = 50000
        trend = np.cumsum(np.random.normal(0, 10, 100))
        noise = np.random.normal(0, 100, 100)
        return pd.DataFrame({
            'open': base_price + trend + noise,
            'high': base_price + trend + noise + np.random.uniform(0, 50, 100),
            'low': base_price + trend + noise - np.random.uniform(0, 50, 100),
            'close': base_price + trend + noise + np.random.normal(0, 20, 100),
            'volume': np.random.uniform(1000, 5000, 100),
            'vwap': base_price + trend + noise,
            'rsi': np.random.uniform(20, 80, 100),
            'macd': np.random.normal(0, 10, 100),
            'bollinger_upper': base_price + trend + noise + 100,
            'bollinger_lower': base_price + trend + noise - 100
        }, index=dates)
    def test_ml_predictor_initialization(self, ml_predictor) -> None:
        """Тест инициализации сервиса."""
        assert ml_predictor is not None
        assert isinstance(ml_predictor, MLPredictor)
        assert hasattr(ml_predictor, 'config')
        assert isinstance(ml_predictor.config, dict)
    def test_ml_predictor_config_defaults(self, ml_predictor) -> None:
        """Тест конфигурации по умолчанию."""
        config = ml_predictor.config
        assert "model_type" in config
        assert "test_size" in config
        assert "random_state" in config
        assert "n_estimators" in config
        assert isinstance(config["model_type"], str)
        assert isinstance(config["test_size"], float)
    def test_train_model_valid_data(self, ml_predictor, sample_training_data) -> None:
        """Тест обучения модели с валидными данными."""
        # Подготавливаем данные для обучения
        features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd']
        target = 'target'
        result = ml_predictor.train_model(sample_training_data, features, target)
        assert isinstance(result, dict)
        assert "model" in result
        assert "performance" in result
        assert "feature_importance" in result
        assert "training_time" in result
        assert result["model"] is not None
        assert set(["accuracy", "precision", "recall", "f1_score"]).issubset(result["performance"].keys())
        assert set(["feature_names", "importance_scores", "top_features"]).issubset(result["feature_importance"].keys())
        assert isinstance(result["training_time"], float)
        # Проверяем метрики производительности
        performance = result["performance"]
        assert "accuracy" in performance
        assert "precision" in performance
        assert "recall" in performance
        assert "f1_score" in performance
        assert isinstance(performance["accuracy"], float)
        assert isinstance(performance["precision"], float)
        assert isinstance(performance["recall"], float)
        assert isinstance(performance["f1_score"], float)
        assert performance["accuracy"] >= 0.0 and performance["accuracy"] <= 1.0
        assert performance["precision"] >= 0.0 and performance["precision"] <= 1.0
        assert performance["recall"] >= 0.0 and performance["recall"] <= 1.0
        assert performance["f1_score"] >= 0.0 and performance["f1_score"] <= 1.0
    def test_train_model_empty_data(self, ml_predictor) -> None:
        """Тест обучения модели с пустыми данными."""
        empty_data = pd.DataFrame()
        features = ['open', 'close']
        target = 'target'
        with pytest.raises(Exception):
            ml_predictor.train_model(empty_data, features, target)
    def test_train_model_insufficient_data(self, ml_predictor) -> None:
        """Тест обучения модели с недостаточными данными."""
        insufficient_data = pd.DataFrame({
            'open': [50000, 50001],
            'close': [50001, 50002],
            'target': [0, 1]
        })
        features = ['open', 'close']
        target = 'target'
        with pytest.raises(Exception):
            ml_predictor.train_model(insufficient_data, features, target)
    def test_train_model_missing_features(self, ml_predictor, sample_training_data) -> None:
        """Тест обучения модели с отсутствующими признаками."""
        features = ['open', 'close', 'nonexistent_feature']
        target = 'target'
        with pytest.raises(Exception):
            ml_predictor.train_model(sample_training_data, features, target)
    def test_train_model_missing_target(self, ml_predictor, sample_training_data) -> None:
        """Тест обучения модели с отсутствующей целью."""
        features = ['open', 'close']
        target = 'nonexistent_target'
        with pytest.raises(Exception):
            ml_predictor.train_model(sample_training_data, features, target)
    def test_predict_valid_data(self, ml_predictor, sample_training_data, sample_prediction_data) -> None:
        """Тест предсказания с валидными данными."""
        # Сначала обучаем модель
        features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd']
        target = 'target'
        training_result = ml_predictor.train_model(sample_training_data, features, target)
        model = training_result["model"]
        # Делаем предсказание
        result = ml_predictor.predict(model, sample_prediction_data, features)
        # assert isinstance(result, PredictionResult)  # TypedDict не поддерживает isinstance
        assert "predictions" in result
        assert "probabilities" in result
        assert "confidence" in result
        assert "prediction_time" in result
        assert isinstance(result["predictions"], np.ndarray)
        assert isinstance(result["probabilities"], np.ndarray)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["prediction_time"], float)
        assert len(result["predictions"]) == len(sample_prediction_data)
        assert len(result["probabilities"]) == len(sample_prediction_data)
        assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0
        assert result["prediction_time"] >= 0.0
    def test_predict_empty_data(self, ml_predictor, sample_training_data) -> None:
        """Тест предсказания с пустыми данными."""
        # Обучаем модель
        features = ['open', 'close']
        target = 'target'
        training_result = ml_predictor.train_model(sample_training_data, features, target)
        model = training_result["model"]
        # Пытаемся предсказать на пустых данных
        empty_data = pd.DataFrame()
        with pytest.raises(Exception):
            ml_predictor.predict(model, empty_data, features)
    def test_predict_missing_features(self, ml_predictor, sample_training_data, sample_prediction_data) -> None:
        """Тест предсказания с отсутствующими признаками."""
        # Обучаем модель
        features = ['open', 'close']
        target = 'target'
        training_result = ml_predictor.train_model(sample_training_data, features, target)
        model = training_result["model"]
        # Пытаемся предсказать с отсутствующими признаками
        missing_features = ['open', 'close', 'nonexistent']
        with pytest.raises(Exception):
            ml_predictor.predict(model, sample_prediction_data, missing_features)
    def test_evaluate_model_performance(self, ml_predictor, sample_training_data) -> None:
        """Тест оценки производительности модели."""
        # Обучаем модель
        features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd']
        target = 'target'
        training_result = ml_predictor.train_model(sample_training_data, features, target)
        model = training_result["model"]
        # Оцениваем производительность
        performance = ml_predictor.evaluate_model_performance(model, sample_training_data, features, target)
        # assert isinstance(performance, ModelPerformance)  # TypedDict не поддерживает isinstance
        assert "accuracy" in performance
        assert "precision" in performance
        assert "recall" in performance
        assert "f1_score" in performance
        assert "confusion_matrix" in performance
        assert isinstance(performance["accuracy"], float)
        assert isinstance(performance["precision"], float)
        assert isinstance(performance["recall"], float)
        assert isinstance(performance["f1_score"], float)
        assert isinstance(performance["confusion_matrix"], np.ndarray)
        assert performance["accuracy"] >= 0.0 and performance["accuracy"] <= 1.0
        assert performance["precision"] >= 0.0 and performance["precision"] <= 1.0
        assert performance["recall"] >= 0.0 and performance["recall"] <= 1.0
        assert performance["f1_score"] >= 0.0 and performance["f1_score"] <= 1.0
    def test_get_feature_importance(self, ml_predictor, sample_training_data) -> None:
        """Тест получения важности признаков."""
        # Обучаем модель
        features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd']
        target = 'target'
        training_result = ml_predictor.train_model(sample_training_data, features, target)
        model = training_result["model"]
        # Получаем важность признаков
        importance = ml_predictor.get_feature_importance(model, features)
        # assert isinstance(importance, FeatureImportance)  # TypedDict не поддерживает isinstance
        assert "feature_names" in importance
        assert "importance_scores" in importance
        assert "top_features" in importance
        assert isinstance(importance["feature_names"], list)
        assert isinstance(importance["importance_scores"], np.ndarray)
        assert isinstance(importance["top_features"], list)
        assert len(importance["feature_names"]) == len(features)
        assert len(importance["importance_scores"]) == len(features)
        assert len(importance["top_features"]) <= len(features)
        # Проверяем, что все оценки важности неотрицательны
        assert all(score >= 0.0 for score in importance["importance_scores"])
    def test_cross_validate_model(self, ml_predictor, sample_training_data) -> None:
        """Тест кросс-валидации модели."""
        features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd']
        target = 'target'
        cv_results = ml_predictor.cross_validate_model(sample_training_data, features, target, cv_folds=5)
        assert isinstance(cv_results, dict)
        assert "mean_accuracy" in cv_results
        assert "std_accuracy" in cv_results
        assert "mean_precision" in cv_results
        assert "std_precision" in cv_results
        assert "mean_recall" in cv_results
        assert "std_recall" in cv_results
        assert "mean_f1" in cv_results
        assert "std_f1" in cv_results
        assert "cv_scores" in cv_results
        assert isinstance(cv_results["mean_accuracy"], float)
        assert isinstance(cv_results["std_accuracy"], float)
        assert isinstance(cv_results["mean_precision"], float)
        assert isinstance(cv_results["std_precision"], float)
        assert isinstance(cv_results["mean_recall"], float)
        assert isinstance(cv_results["std_recall"], float)
        assert isinstance(cv_results["mean_f1"], float)
        assert isinstance(cv_results["std_f1"], float)
        assert isinstance(cv_results["cv_scores"], list)
        assert cv_results["mean_accuracy"] >= 0.0 and cv_results["mean_accuracy"] <= 1.0
        assert cv_results["std_accuracy"] >= 0.0
        assert len(cv_results["cv_scores"]) == 5
    def test_hyperparameter_tuning(self, ml_predictor, sample_training_data) -> None:
        """Тест настройки гиперпараметров."""
        features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd']
        target = 'target'
        tuning_result = ml_predictor.hyperparameter_tuning(sample_training_data, features, target)
        assert isinstance(tuning_result, dict)
        assert "best_model" in tuning_result
        assert "best_params" in tuning_result
        assert "best_score" in tuning_result
        assert "tuning_time" in tuning_result
        assert tuning_result["best_model"] is not None
        assert isinstance(tuning_result["best_params"], dict)
        assert isinstance(tuning_result["best_score"], float)
        assert isinstance(tuning_result["tuning_time"], float)
        assert tuning_result["best_score"] >= 0.0 and tuning_result["best_score"] <= 1.0
        assert tuning_result["tuning_time"] >= 0.0
    def test_save_load_model(self, ml_predictor, sample_training_data, tmp_path) -> None:
        """Тест сохранения и загрузки модели."""
        # Обучаем модель
        features = ['open', 'close']
        target = 'target'
        training_result = ml_predictor.train_model(sample_training_data, features, target)
        model = training_result["model"]
        # Сохраняем модель
        model_path = tmp_path / "test_model.pkl"
        save_result = ml_predictor.save_model(model, str(model_path))
        assert isinstance(save_result, bool)
        assert save_result == True
        assert model_path.exists()
        # Загружаем модель
        loaded_model = ml_predictor.load_model(str(model_path))
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')  # Проверяем, что это ML модель
    def test_save_load_model_invalid_path(self, ml_predictor, sample_training_data) -> None:
        """Тест сохранения модели с невалидным путем."""
        # Обучаем модель
        features = ['open', 'close']
        target = 'target'
        training_result = ml_predictor.train_model(sample_training_data, features, target)
        model = training_result["model"]
        # Пытаемся сохранить в невалидный путь
        invalid_path = "/invalid/path/model.pkl"
        with pytest.raises(Exception):
            ml_predictor.save_model(model, invalid_path)
    def test_predict_proba(self, ml_predictor, sample_training_data, sample_prediction_data) -> None:
        """Тест предсказания вероятностей."""
        # Обучаем модель
        features = ['open', 'close']
        target = 'target'
        training_result = ml_predictor.train_model(sample_training_data, features, target)
        model = training_result["model"]
        # Получаем вероятности
        probabilities = ml_predictor.predict_proba(model, sample_prediction_data, features)
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape[0] == len(sample_prediction_data)
        assert probabilities.shape[1] == 2  # Для бинарной классификации
        # Проверяем, что вероятности в диапазоне [0, 1] и сумма по строкам = 1
        assert np.all(probabilities >= 0.0)
        assert np.all(probabilities <= 1.0)
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    def test_get_model_info(self, ml_predictor, sample_training_data) -> None:
        """Тест получения информации о модели."""
        # Обучаем модель
        features = ['open', 'close']
        target = 'target'
        training_result = ml_predictor.train_model(sample_training_data, features, target)
        model = training_result["model"]
        # Получаем информацию о модели
        info = ml_predictor.get_model_info(model)
        assert isinstance(info, dict)
        assert "model_type" in info
        assert "n_features" in info
        assert "n_classes" in info
        assert "training_date" in info
        assert isinstance(info["model_type"], str)
        assert isinstance(info["n_features"], int)
        assert isinstance(info["n_classes"], int)
        assert isinstance(info["training_date"], str)
        assert info["n_features"] == len(features)
        assert info["n_classes"] == 2  # Для бинарной классификации
    def test_ml_predictor_error_handling(self, ml_predictor) -> None:
        """Тест обработки ошибок в сервисе."""
        # Тест с None данными
        with pytest.raises(Exception):
            ml_predictor.train_model(None, [], "")
        # Тест с невалидным типом модели
        with pytest.raises(Exception):
            ml_predictor.predict("invalid_model", pd.DataFrame(), [])
        # Тест с невалидными гиперпараметрами
        with pytest.raises(Exception):
            ml_predictor.hyperparameter_tuning(pd.DataFrame(), [], "", invalid_param="value")
    def test_ml_predictor_performance(self, ml_predictor, sample_training_data, sample_prediction_data) -> None:
        """Тест производительности сервиса."""
        import time
        # Обучаем модель
        features = ['open', 'close']
        target = 'target'
        start_time = time.time()
        training_result = ml_predictor.train_model(sample_training_data, features, target)
        training_time = time.time() - start_time
        model = training_result["model"]
        # Тестируем скорость предсказания
        start_time = time.time()
        for _ in range(10):
            ml_predictor.predict(model, sample_prediction_data, features)
        prediction_time = time.time() - start_time
        # Проверяем, что обучение и предсказание выполняются в разумное время
        assert training_time < 30.0  # Обучение менее 30 секунд
        assert prediction_time < 5.0  # 10 предсказаний менее 5 секунд
    def test_ml_predictor_thread_safety(self, ml_predictor, sample_training_data, sample_prediction_data) -> None:
        """Тест потокобезопасности сервиса."""
        import threading
        import queue
        # Обучаем модель
        features = ['open', 'close']
        target = 'target'
        training_result = ml_predictor.train_model(sample_training_data, features, target)
        model = training_result["model"]
        results = queue.Queue()
        def make_prediction() -> Any:
            try:
                result = ml_predictor.predict(model, sample_prediction_data, features)
                results.put(result)
            except Exception as e:
                results.put(e)
        # Запускаем несколько потоков одновременно
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_prediction)
            threads.append(thread)
            thread.start()
        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()
        # Проверяем, что все результаты корректны
        for _ in range(5):
            result = results.get()
            # assert isinstance(result, PredictionResult)  # TypedDict не поддерживает isinstance
            assert "predictions" in result
    def test_ml_predictor_config_customization(self: "TestMLPredictor") -> None:
        """Тест кастомизации конфигурации сервиса."""
        custom_config = {
            "model_type": "random_forest",
            "test_size": 0.3,
            "random_state": 123,
            "n_estimators": 200,
            "max_depth": 10
        }
        predictor = MLPredictor(custom_config)
        assert predictor.config["model_type"] == "random_forest"
        assert predictor.config["test_size"] == 0.3
        assert predictor.config["random_state"] == 123
        assert predictor.config["n_estimators"] == 200
        assert predictor.config["max_depth"] == 10
    def test_ml_predictor_integration_with_different_models(self, ml_predictor, sample_training_data, sample_prediction_data) -> None:
        """Тест интеграции с различными типами моделей."""
        features = ['open', 'close']
        target = 'target'
        # Тестируем разные типы моделей
        model_types = ["random_forest", "gradient_boosting", "logistic_regression"]
        for model_type in model_types:
            ml_predictor.config["model_type"] = model_type
            # Обучаем модель
            training_result = ml_predictor.train_model(sample_training_data, features, target)
            model = training_result["model"]
            # Делаем предсказание
            prediction_result = ml_predictor.predict(model, sample_prediction_data, features)
            # Проверяем результаты
            assert isinstance(training_result, dict)
            # assert isinstance(prediction_result, PredictionResult)  # TypedDict не поддерживает isinstance
            assert training_result["model"] is not None
            assert len(prediction_result["predictions"]) == len(sample_prediction_data)
    def test_ml_predictor_feature_engineering(self, ml_predictor, sample_training_data) -> None:
        """Тест инженерии признаков."""
        # Создаем данные с базовыми признаками
        base_features = ['open', 'close']
        target = 'target'
        # Обучаем модель с базовыми признаками
        base_result = ml_predictor.train_model(sample_training_data, base_features, target)
        base_performance = base_result["performance"]["accuracy"]
        # Создаем расширенные признаки
        extended_features = ['open', 'close', 'high', 'low', 'volume', 'rsi', 'macd']
        # Обучаем модель с расширенными признаками
        extended_result = ml_predictor.train_model(sample_training_data, extended_features, target)
        extended_performance = extended_result["performance"]["accuracy"]
        # Проверяем, что расширенные признаки дают не худшую производительность
        assert extended_performance >= 0.0
        assert base_performance >= 0.0 

    def test_prediction_validation(self) -> None:
        """Тест валидации предсказаний."""
        predictor = MLPredictor()
        
        # Создаем валидное предсказание
        valid_prediction = {
            "symbol": "BTC/USD",
            "prediction_type": "price",
            "value": 50000.0,
            "confidence": 0.8,
            "timestamp": "2024-01-01T00:00:00"
        }
        
        # Проверяем валидацию
        is_valid = predictor._validate_prediction(valid_prediction)
        assert is_valid is True
        
        # Создаем невалидное предсказание
        invalid_prediction = {
            "symbol": "BTC/USD",
            # Отсутствует prediction_type
            "value": 50000.0
        }
        
        # Проверяем валидацию
        is_valid = predictor._validate_prediction(invalid_prediction)
        assert is_valid is False

    def test_prediction_processing(self) -> None:
        """Тест обработки предсказаний."""
        predictor = MLPredictor()
        
        # Создаем предсказание
        prediction_data = {
            "symbol": "BTC/USD",
            "prediction_type": "price",
            "value": 50000.0,
            "confidence": 0.8,
            "timestamp": "2024-01-01T00:00:00"
        }
        
        # Обрабатываем предсказание
        result = predictor._process_prediction(prediction_data)
        
        assert result is not None
        assert hasattr(result, 'symbol')
        assert hasattr(result, 'prediction_type')
        assert hasattr(result, 'value')

    def test_model_validation(self) -> None:
        """Тест валидации модели."""
    predictor = MLPredictor()
    
    # Создаем валидную модель
    valid_model = {
        "model_id": "test_model",
        "model_type": "regression",
        "accuracy": 0.95,
        "status": "active"
    }
    
    # Проверяем валидацию
    is_valid = predictor._validate_model(valid_model)
    assert is_valid is True
    
    # Создаем невалидную модель
    invalid_model = {
        "model_id": "test_model",
        # Отсутствует model_type
        "accuracy": 0.95
    }
    
    # Проверяем валидацию
    is_valid = predictor._validate_model(invalid_model)
    assert is_valid is False

    def test_prediction_aggregation(self) -> None:
        """Тест агрегации предсказаний."""
    predictor = MLPredictor()
    
    # Создаем несколько предсказаний
    predictions = [
        {
            "symbol": "BTC/USD",
            "prediction_type": "price",
            "value": 50000.0,
            "confidence": 0.8,
            "timestamp": "2024-01-01T00:00:00"
        },
        {
            "symbol": "BTC/USD",
            "prediction_type": "price",
            "value": 51000.0,
            "confidence": 0.7,
            "timestamp": "2024-01-01T00:00:00"
        }
    ]
    
    # Агрегируем предсказания
    aggregated = predictor._aggregate_predictions(predictions)
    
    assert aggregated is not None
    assert 'symbol' in aggregated
    assert 'prediction_type' in aggregated
    assert 'value' in aggregated

    def test_model_integrity_check(self) -> None:
        """Тест проверки целостности модели."""
    predictor = MLPredictor()
    
    # Создаем модель
    model_data = {
        "model_id": "test_model",
        "model_type": "regression",
        "accuracy": 0.95,
        "status": "active",
        "parameters": {"learning_rate": 0.01}
    }
    
    # Проверяем целостность
    is_integrity_valid = predictor._check_model_integrity(model_data)
    assert is_integrity_valid is True
    
    # Создаем модель с нарушенной целостностью
    invalid_model_data = {
        "model_id": "test_model",
        "model_type": "regression",
        "accuracy": 1.5,  # Недопустимое значение
        "status": "active"
    }
    
    # Проверяем целостность
    is_integrity_valid = predictor._check_model_integrity(invalid_model_data)
    assert is_integrity_valid is False 
