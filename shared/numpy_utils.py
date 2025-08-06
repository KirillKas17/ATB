"""
Numpy utilities with fallback implementations
"""

from typing import Any, List, Union, Optional
import builtins

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    
    # Mock numpy functions
    class MockNumpy:
        @staticmethod
        def array(data: Any) -> List[Any]:
            """Convert to list (mock array)"""
            if isinstance(data, list):
                return data
            return list(data)
        
        @staticmethod
        def asarray(data: Any) -> List[Any]:
            """Convert to array (alias for array)"""
            return MockNumpy.array(data)
        
        @staticmethod
        def mean(data: Any) -> float:
            """Calculate mean"""
            arr = list(data) if not isinstance(data, list) else data
            return sum(arr) / len(arr) if arr else 0.0
        
        @staticmethod
        def std(data: Any) -> float:
            """Calculate standard deviation"""
            arr = list(data) if not isinstance(data, list) else data
            if not arr:
                return 0.0
            mean = sum(arr) / len(arr)
            variance = sum((x - mean) ** 2 for x in arr) / len(arr)
            return float(variance ** 0.5)
        
        @staticmethod
        def min(data: Any) -> float:
            """Get minimum value"""
            arr = list(data) if not isinstance(data, list) else data
            return builtins.min(arr) if arr else 0.0
        
        @staticmethod
        def max(data: Any) -> float:
            """Get maximum value"""
            arr = list(data) if not isinstance(data, list) else data
            return builtins.max(arr) if arr else 0.0
        
        @staticmethod
        def percentile(data: Any, percentile: float) -> float:
            """Calculate percentile"""
            arr = list(data) if not isinstance(data, list) else data
            if not arr:
                return 0.0
            sorted_arr = sorted(arr)
            idx = int(len(sorted_arr) * percentile / 100)
            return float(sorted_arr[builtins.min(idx, len(sorted_arr) - 1)])
        
        @staticmethod
        def var(data: Any) -> float:
            """Calculate variance"""
            arr = list(data) if not isinstance(data, list) else data
            if not arr:
                return 0.0
            mean = sum(arr) / len(arr)
            return float(sum((x - mean) ** 2 for x in arr) / len(arr))
        
        @staticmethod
        def cov(data1: Any, data2: Any = None) -> Any:
            """Calculate covariance matrix"""
            if data2 is None:
                # Single argument - covariance matrix
                if not data1:
                    return [[0.0]]
                return [[1.0]]  # Mock covariance matrix
            else:
                # Two arguments - covariance between arrays
                arr1 = list(data1) if not isinstance(data1, list) else data1
                arr2 = list(data2) if not isinstance(data2, list) else data2
                if not arr1 or not arr2 or len(arr1) != len(arr2):
                    return 0.0
                mean1 = sum(arr1) / len(arr1)
                mean2 = sum(arr2) / len(arr2)
                return sum((x - mean1) * (y - mean2) for x, y in zip(arr1, arr2)) / len(arr1)
        
        @staticmethod
        def polyfit(x: Any, y: Any, degree: int) -> List[float]:
            """Simple linear fit (mock)"""
            # Simple linear regression for degree 1
            if degree == 1 and len(x) > 1:
                n = len(x)
                sum_x = sum(x)
                sum_y = sum(y)
                sum_xy = sum(x[i] * y[i] for i in range(n))
                sum_xx = sum(x[i] * x[i] for i in range(n))
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
                intercept = (sum_y - slope * sum_x) / n
                return [slope, intercept]
            return [0.0, 0.0]
        
        @staticmethod
        def less(a: Any, b: Any) -> List[bool]:
            """Element-wise less than comparison"""
            if isinstance(a, (list, tuple)):
                return [x < b for x in a]
            return [a < b]
        
        @staticmethod
        def greater(a: Any, b: Any) -> List[bool]:
            """Element-wise greater than comparison"""
            if isinstance(a, (list, tuple)):
                return [x > b for x in a]
            return [a > b]
        
        @staticmethod
        def sum(data: Any, axis: Any = None) -> float:
            """Sum of array elements"""
            arr = list(data) if not isinstance(data, list) else data
            return float(sum(arr)) if arr else 0.0
        
        @staticmethod
        def arange(start: float, stop: Optional[float] = None, step: float = 1.0) -> List[float]:
            """Create array with evenly spaced values"""
            if stop is None:
                stop, start = start, 0.0
            result = []
            current = start
            while current < stop:
                result.append(current)
                current += step
            return result
        
        @staticmethod
        def abs(data: Any) -> List[float]:
            """Absolute values"""
            arr = list(data) if not isinstance(data, list) else data
            return [builtins.abs(x) for x in arr]
        
        @staticmethod
        def sqrt(data: Any) -> Union[float, List[float]]:
            """Square root"""
            if isinstance(data, (int, float)):
                return float(data ** 0.5)
            arr = list(data) if not isinstance(data, list) else data
            return [float(x ** 0.5) for x in arr]
        
        @staticmethod
        def clip(data: Any, min_val: float, max_val: float) -> Union[float, List[float]]:
            """Clip values between min and max"""
            if isinstance(data, (int, float)):
                return float(max(min_val, min(max_val, data)))
            arr = list(data) if not isinstance(data, list) else data
            return [float(max(min_val, min(max_val, x))) for x in arr]
        
        class maximum:
            @staticmethod
            def accumulate(data: Any) -> List[float]:
                """Cumulative maximum"""
                arr = list(data) if not isinstance(data, list) else data
                if not arr:
                    return []
                result = [arr[0]]
                for i in range(1, len(arr)):
                    result.append(builtins.max(result[-1], arr[i]))
                return result
            
            @staticmethod
            def reduce(data1: Any, data2: Any) -> Any:
                """Element-wise maximum"""
                if isinstance(data1, (list, tuple)) and isinstance(data2, (list, tuple)):
                    return [builtins.max(x, y) for x, y in zip(data1, data2)]
                return builtins.max(data1, data2)
        
        # maximum class is already defined above
        
        class random:
            @staticmethod
            def random(size: Any = None) -> Any:
                """Random numbers"""
                import random as py_random
                if size is None:
                    return py_random.random()
                if isinstance(size, int):
                    return [py_random.random() for _ in range(size)]
                return [py_random.random() for _ in range(10)]  # Default size
            
            @staticmethod
            def randn(*args: Any) -> Any:
                """Random normal"""
                import random as py_random
                if not args:
                    return py_random.gauss(0, 1)
                size = args[0] if args else 1
                return [py_random.gauss(0, 1) for _ in range(size)]
            
            @staticmethod
            def uniform(low: float = 0.0, high: float = 1.0, size: Any = None) -> Any:
                """Random uniform"""
                import random as py_random
                if size is None:
                    return py_random.uniform(low, high)
                if isinstance(size, int):
                    return [py_random.uniform(low, high) for _ in range(size)]
                return [py_random.uniform(low, high) for _ in range(10)]
        
        @staticmethod
        def linspace(start: float, stop: float, num: int = 50) -> List[float]:
            """Create linearly spaced array"""
            if num <= 1:
                return [start]
            step = (stop - start) / (num - 1)
            return [start + i * step for i in range(num)]
    
    np = MockNumpy()

# If numpy is available, use real numpy
if HAS_NUMPY:
    import numpy
    np = numpy

# Export types and constants
ArrayLike = Union[List[float], List[int], Any]

# Mock ndarray type for type hints
if not HAS_NUMPY:
    class ndarray(list):
        """Mock ndarray that behaves like a list"""
        def __init__(self, data: Any = None):
            super().__init__(data if data is not None else [])
        
        def __pow__(self, other: Any) -> 'ndarray':
            if isinstance(other, (int, float)):
                return ndarray([float(x ** other) for x in self])
            return ndarray([float(x ** y) for x, y in zip(self, other)])
        
        def __mul__(self, other: Any) -> 'ndarray':
            if isinstance(other, (int, float)):
                return ndarray([x * other for x in self])
            return ndarray([x * y for x, y in zip(self, other)])
        
        def __rmul__(self, other: Any) -> 'ndarray':
            return self.__mul__(other)
        
        def __truediv__(self, other: Any) -> 'ndarray':
            if isinstance(other, (int, float)):
                return ndarray([x / other for x in self])
            return ndarray([x / y for x, y in zip(self, other)])
        
        def __add__(self, other: Any) -> 'ndarray':
            if isinstance(other, (int, float)):
                return ndarray([x + other for x in self])
            return ndarray([x + y for x, y in zip(self, other)])
        
        def __sub__(self, other: Any) -> 'ndarray':
            if isinstance(other, (int, float)):
                return ndarray([x - other for x in self])
            return ndarray([x - y for x, y in zip(self, other)])
    
    # Create mock np instance
    np = MockNumpy()
else:
    ndarray = np.ndarray  # type: ignore[misc]

# Export np for external use
__all__ = ['np', 'ndarray', 'HAS_NUMPY']