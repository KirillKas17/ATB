"""
Safe import wrapper for handling missing modules gracefully
"""

import sys
from typing import Any, Dict, Tuple, Optional, Union, Iterator


class SafeImportMock:
    """Mock object that can be used in place of missing imports"""
    
    def __init__(self, name: str = "SafeImportMock") -> None:
        self._name = name
    
    def __getattr__(self, name: str) -> "SafeImportMock":
        return SafeImportMock(f"{self._name}.{name}")
    
    def __call__(self, *args: Any, **kwargs: Any) -> "SafeImportMock":
        return SafeImportMock(f"{self._name}()")
    
    def __bool__(self) -> bool:
        return False
    
    def __iter__(self) -> Iterator[Any]:
        return iter([])
    
    def __str__(self) -> str:
        return f"<{self._name}>"
    
    def __repr__(self) -> str:
        return f"SafeImportMock({self._name})"


def safe_import(
    module_name: str, 
    class_name: Optional[str] = None, 
    fallback: Optional[Any] = None
) -> Any:
    """
    Safely import a module or class, returning a mock if import fails
    
    Args:
        module_name: The module to import
        class_name: Optional class name to import from module
        fallback: Custom fallback object
    
    Returns:
        The imported object or a safe mock
    """
    try:
        module = __import__(module_name, fromlist=[class_name] if class_name else [])
        if class_name:
            return getattr(module, class_name)
        return module
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not import {module_name}.{class_name if class_name else ''}: {e}")
        return fallback if fallback is not None else SafeImportMock(f"{module_name}.{class_name or 'module'}")


def safe_imports(imports_dict: Dict[str, Tuple[str, Optional[str]]]) -> Dict[str, Any]:
    """
    Safely import multiple modules/classes
    
    Args:
        imports_dict: Dict of {name: (module, class_name)}
    
    Returns:
        Dict of imported objects or mocks
    """
    results: Dict[str, Any] = {}
    for name, (module_name, class_name) in imports_dict.items():
        results[name] = safe_import(module_name, class_name)
    return results