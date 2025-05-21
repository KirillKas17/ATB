    def _evaluate_bo(self, x: np.ndarray) -> float:
        """
        Оценка точки для байесовской оптимизации.
        
        Args:
            x: Точка для оценки
            
        Returns:
            float: Значение целевой функции
        """
        try:
            # Преобразуем точку в параметры стратегии
            params = self._convert_to_params(x)
            
            # Применяем параметры к стратегии
            self.strategy.update_parameters(params)
            
            # Оцениваем производительность
            performance = self._evaluate_performance()
            
            return -performance  # Минимизируем отрицательную производительность
            
        except Exception as e:
            logger.error(f"Error evaluating point: {str(e)}")
            return float('inf')  # Возвращаем бесконечность в случае ошибки 
