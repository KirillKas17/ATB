"""
Unit тесты для MLPipeline.
Тестирует машинное обучение, включая предобработку данных,
обучение моделей, валидацию и предсказания.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any
from infrastructure.ml_services.ml_pipeline import MLPipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

class TestMLPipeline:
    """Тесты для MLPipeline."""
    @pytest.fixture
    def ml_pipeline(self) -> MLPipeline:
        """Фикстура для MLPipeline."""
        return MLPipeline()
    @pytest.fixture
    def sample_training_data(self) -> tuple:
        """Фикстура с тестовыми данными для обучения."""
        np.random.seed(42)
        n_samples = 1000
        # Создание признаков
        X = np.random.randn(n_samples, 10)
        # Создание целевой переменной (линейная комбинация признаков + шум)
        y = np.dot(X, np.random.randn(10)) + np.random.normal(0, 0.1, n_samples)
        return X, y
    @pytest.fixture
    def sample_dataframe_data(self) -> pd.DataFrame:
        """Фикстура с данными в формате DataFrame."""
        np.random.seed(42)
        n_samples = 1000
        data = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
            'feature_4': np.random.randn(n_samples),
            'feature_5': np.random.randn(n_samples),
            'target': np.random.randn(n_samples)
        })
        return data
    def test_initialization(self, ml_pipeline: MLPipeline) -> None:
        """Тест инициализации ML пайплайна."""
        assert ml_pipeline is not None
        assert hasattr(ml_pipeline, 'models')
        assert hasattr(ml_pipeline, 'preprocessors')
        assert hasattr(ml_pipeline, 'evaluators')
        assert hasattr(ml_pipeline, 'feature_importance')
    def test_create_model(self, ml_pipeline: MLPipeline) -> None:
        """Тест создания модели."""
        # Создание модели
        model = ml_pipeline.create_model('random_forest', {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        })
        # Проверки
        assert model is not None
        assert isinstance(model, RandomForestRegressor)
        assert model.n_estimators == 100
        assert model.max_depth == 10
    def test_create_linear_model(self, ml_pipeline: MLPipeline) -> None:
        """Тест создания линейной модели."""
        # Создание линейной модели
        model = ml_pipeline.create_model('linear_regression', {
            'fit_intercept': True,
            'normalize': False
        })
        # Проверки
        assert model is not None
        assert isinstance(model, LinearRegression)
        assert model.fit_intercept is True
    def test_preprocess_data(self, ml_pipeline: MLPipeline, sample_dataframe_data: pd.DataFrame) -> None:
        """Тест предобработки данных."""
        # Разделение на признаки и целевую переменную
        X = sample_dataframe_data.drop(columns=['target'])
        y = sample_dataframe_data['target']
        # Предобработка данных
        X_processed, y_processed = ml_pipeline.preprocess_data(X, y)
        # Проверки
        assert X_processed is not None
        assert y_processed is not None
        assert isinstance(X_processed, np.ndarray)
        assert isinstance(y_processed, np.ndarray)
        assert len(X_processed) == len(y_processed)
        assert X_processed.shape[1] == X.shape[1]
    def test_train_model(self, ml_pipeline: MLPipeline, sample_training_data: tuple) -> None:
        """Тест обучения модели."""
        X, y = sample_training_data
        # Создание и обучение модели
        model = ml_pipeline.create_model('random_forest', {'n_estimators': 50, 'random_state': 42})
        training_result = ml_pipeline.train_model(model, X, y)
        # Проверки
        assert training_result is not None
        assert "model" in training_result
        assert "training_score" in training_result
        assert "training_time" in training_result
        assert "feature_importance" in training_result
        # Проверка типов данных
        assert isinstance(training_result["model"], RandomForestRegressor)
        assert isinstance(training_result["training_score"], float)
        assert isinstance(training_result["training_time"], float)
        assert isinstance(training_result["feature_importance"], dict)
        # Проверка диапазона training_score
        assert 0.0 <= training_result["training_score"] <= 1.0
    def test_predict(self, ml_pipeline: MLPipeline, sample_training_data: tuple) -> None:
        """Тест предсказания."""
        X, y = sample_training_data
        # Обучение модели
        model = ml_pipeline.create_model('random_forest', {'n_estimators': 50, 'random_state': 42})
        ml_pipeline.train_model(model, X, y)
        # Предсказание
        predictions = ml_pipeline.predict(model, X[:100])
        # Проверки
        assert predictions is not None
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 100
        assert not np.isnan(predictions).any()
    def test_evaluate_model(self, ml_pipeline: MLPipeline, sample_training_data: tuple) -> None:
        """Тест оценки модели."""
        X, y = sample_training_data
        # Разделение на обучающую и тестовую выборки
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        # Обучение модели
        model = ml_pipeline.create_model('random_forest', {'n_estimators': 50, 'random_state': 42})
        ml_pipeline.train_model(model, X_train, y_train)
        # Оценка модели
        evaluation_result = ml_pipeline.evaluate_model(model, X_test, y_test)
        # Проверки
        assert evaluation_result is not None
        assert "mse" in evaluation_result
        assert "rmse" in evaluation_result
        assert "mae" in evaluation_result
        assert "r2_score" in evaluation_result
        assert "explained_variance" in evaluation_result
        # Проверка типов данных
        assert isinstance(evaluation_result["mse"], float)
        assert isinstance(evaluation_result["rmse"], float)
        assert isinstance(evaluation_result["mae"], float)
        assert isinstance(evaluation_result["r2_score"], float)
        assert isinstance(evaluation_result["explained_variance"], float)
        # Проверка логики метрик
        assert evaluation_result["mse"] >= 0
        assert evaluation_result["rmse"] >= 0
        assert evaluation_result["mae"] >= 0
        assert -1.0 <= evaluation_result["r2_score"] <= 1.0
    def test_cross_validate(self, ml_pipeline: MLPipeline, sample_training_data: tuple) -> None:
        """Тест кросс-валидации."""
        X, y = sample_training_data
        # Создание модели
        model = ml_pipeline.create_model('random_forest', {'n_estimators': 50, 'random_state': 42})
        # Кросс-валидация
        cv_result = ml_pipeline.cross_validate(model, X, y, cv=5)
        # Проверки
        assert cv_result is not None
        assert "mean_score" in cv_result
        assert "std_score" in cv_result
        assert "scores" in cv_result
        assert "cv_time" in cv_result
        # Проверка типов данных
        assert isinstance(cv_result["mean_score"], float)
        assert isinstance(cv_result["std_score"], float)
        assert isinstance(cv_result["scores"], list)
        assert isinstance(cv_result["cv_time"], float)
        # Проверка логики
        assert len(cv_result["scores"]) == 5
        assert cv_result["std_score"] >= 0
    def test_feature_selection(self, ml_pipeline: MLPipeline, sample_dataframe_data: pd.DataFrame) -> None:
        """Тест выбора признаков."""
        # Разделение на признаки и целевую переменную
        X = sample_dataframe_data.drop(columns=['target'])
        y = sample_dataframe_data['target']
        # Выбор признаков
        feature_selection_result = ml_pipeline.feature_selection(X, y, method='correlation', threshold=0.8)
        # Проверки
        assert feature_selection_result is not None
        assert "selected_features" in feature_selection_result
        assert "feature_scores" in feature_selection_result
        assert "selection_score" in feature_selection_result
        # Проверка типов данных
        assert isinstance(feature_selection_result["selected_features"], list)
        assert isinstance(feature_selection_result["feature_scores"], dict)
        assert isinstance(feature_selection_result["selection_score"], float)
        # Проверка логики
        assert len(feature_selection_result["selected_features"]) <= len(X.columns)
        assert 0.0 <= feature_selection_result["selection_score"] <= 1.0
    def test_hyperparameter_tuning(self, ml_pipeline: MLPipeline, sample_training_data: tuple) -> None:
        """Тест настройки гиперпараметров."""
        X, y = sample_training_data
        # Параметры для настройки
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5]
        }
        # Настройка гиперпараметров
        tuning_result = ml_pipeline.hyperparameter_tuning(
            X, y, 'random_forest', param_grid, cv=3
        )
        # Проверки
        assert tuning_result is not None
        assert "best_params" in tuning_result
        assert "best_score" in tuning_result
        assert "best_model" in tuning_result
        assert "tuning_time" in tuning_result
        # Проверка типов данных
        assert isinstance(tuning_result["best_params"], dict)
        assert isinstance(tuning_result["best_score"], float)
        assert isinstance(tuning_result["best_model"], RandomForestRegressor)
        assert isinstance(tuning_result["tuning_time"], float)
        # Проверка логики
        assert 0.0 <= tuning_result["best_score"] <= 1.0
    def test_ensemble_models(self, ml_pipeline: MLPipeline, sample_training_data: tuple) -> None:
        """Тест ансамблевых моделей."""
        X, y = sample_training_data
        # Создание ансамбля
        ensemble_result = ml_pipeline.ensemble_models(
            X, y,
            models=['random_forest', 'linear_regression'],
            method='voting'
        )
        # Проверки
        assert ensemble_result is not None
        assert "ensemble_model" in ensemble_result
        assert "ensemble_score" in ensemble_result
        assert "individual_scores" in ensemble_result
        # Проверка типов данных
        assert ensemble_result["ensemble_model"] is not None
        assert isinstance(ensemble_result["ensemble_score"], float)
        assert isinstance(ensemble_result["individual_scores"], dict)
        # Проверка диапазона
        assert 0.0 <= ensemble_result["ensemble_score"] <= 1.0
    def test_model_persistence(self, ml_pipeline: MLPipeline, sample_training_data: tuple) -> None:
        """Тест сохранения и загрузки модели."""
        X, y = sample_training_data
        # Обучение модели
        model = ml_pipeline.create_model('random_forest', {'n_estimators': 50, 'random_state': 42})
        ml_pipeline.train_model(model, X, y)
        # Сохранение модели
        save_result = ml_pipeline.save_model(model, 'test_model.pkl')
        # Проверки сохранения
        assert save_result is not None
        assert "save_path" in save_result
        assert "save_time" in save_result
        assert save_result["save_path"] == 'test_model.pkl'
        # Загрузка модели
        loaded_model = ml_pipeline.load_model('test_model.pkl')
        # Проверки загрузки
        assert loaded_model is not None
        assert isinstance(loaded_model, RandomForestRegressor)
        # Проверка, что загруженная модель работает
        predictions_original = ml_pipeline.predict(model, X[:10])
        predictions_loaded = ml_pipeline.predict(loaded_model, X[:10])
        np.testing.assert_array_almost_equal(predictions_original, predictions_loaded)  # type: ignore
    def test_model_comparison(self, ml_pipeline: MLPipeline, sample_training_data: tuple) -> None:
        """Тест сравнения моделей."""
        X, y = sample_training_data
        # Разделение на обучающую и тестовую выборки
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        # Создание моделей для сравнения
        models = {
            'random_forest': ml_pipeline.create_model('random_forest', {'n_estimators': 50, 'random_state': 42}),
            'linear_regression': ml_pipeline.create_model('linear_regression', {})
        }
        # Сравнение моделей
        comparison_result = ml_pipeline.compare_models(
            models, X_train, y_train, X_test, y_test
        )
        # Проверки
        assert comparison_result is not None
        assert "model_scores" in comparison_result
        assert "best_model" in comparison_result
        assert "comparison_metrics" in comparison_result
        # Проверка типов данных
        assert isinstance(comparison_result["model_scores"], dict)
        assert isinstance(comparison_result["best_model"], str)
        assert isinstance(comparison_result["comparison_metrics"], dict)
        # Проверка логики
        assert len(comparison_result["model_scores"]) == 2
        assert comparison_result["best_model"] in models.keys()
    def test_feature_importance_analysis(self, ml_pipeline: MLPipeline, sample_dataframe_data: pd.DataFrame) -> None:
        """Тест анализа важности признаков."""
        # Разделение на признаки и целевую переменную
        X = sample_dataframe_data.drop(columns=['target'])
        y = sample_dataframe_data['target']
        # Обучение модели
        model = ml_pipeline.create_model('random_forest', {'n_estimators': 50, 'random_state': 42})
        ml_pipeline.train_model(model, X, y)
        # Анализ важности признаков
        importance_analysis = ml_pipeline.analyze_feature_importance(model, X.columns)
        # Проверки
        assert importance_analysis is not None
        assert "feature_importance" in importance_analysis
        assert "top_features" in importance_analysis
        assert "importance_plot_data" in importance_analysis
        # Проверка типов данных
        assert isinstance(importance_analysis["feature_importance"], dict)
        assert isinstance(importance_analysis["top_features"], list)
        assert isinstance(importance_analysis["importance_plot_data"], dict)
        # Проверка логики
        assert len(importance_analysis["feature_importance"]) == len(X.columns)
        assert len(importance_analysis["top_features"]) <= len(X.columns)
    def test_model_interpretation(self, ml_pipeline: MLPipeline, sample_training_data: tuple) -> None:
        """Тест интерпретации модели."""
        X, y = sample_training_data
        # Обучение модели
        model = ml_pipeline.create_model('random_forest', {'n_estimators': 50, 'random_state': 42})
        ml_pipeline.train_model(model, X, y)
        # Интерпретация модели
        interpretation = ml_pipeline.interpret_model(model, X[:100])
        # Проверки
        assert interpretation is not None
        assert "model_complexity" in interpretation
        assert "prediction_explanation" in interpretation
        assert "feature_contributions" in interpretation
        # Проверка типов данных
        assert isinstance(interpretation["model_complexity"], dict)
        assert isinstance(interpretation["prediction_explanation"], dict)
        assert isinstance(interpretation["feature_contributions"], dict)
    def test_error_handling(self, ml_pipeline: MLPipeline) -> None:
        """Тест обработки ошибок."""
        # Тест с некорректными данными
        with pytest.raises(ValueError):
            ml_pipeline.create_model('invalid_model', {})
        with pytest.raises(ValueError):
            ml_pipeline.train_model(None, np.array([]), np.array([]))
    def test_edge_cases(self, ml_pipeline: MLPipeline) -> None:
        """Тест граничных случаев."""
        # Тест с очень маленьким набором данных
        X_small = np.random.randn(5, 3)
        y_small = np.random.randn(5)
        model = ml_pipeline.create_model('random_forest', {'n_estimators': 10, 'random_state': 42})
        training_result = ml_pipeline.train_model(model, X_small, y_small)
        assert training_result is not None
        assert training_result["model"] is not None
    def test_cleanup(self, ml_pipeline: MLPipeline) -> None:
        """Тест очистки ресурсов."""
        # Проверка корректной очистки
        ml_pipeline.cleanup()
        # Проверка, что ресурсы освобождены
        assert ml_pipeline.models == {}
        assert ml_pipeline.preprocessors == {}
        assert ml_pipeline.evaluators == {}
        assert ml_pipeline.feature_importance == {} 
