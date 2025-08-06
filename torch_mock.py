"""
Mock PyTorch module for development without torch installation
"""
from typing import Any, Iterator, Tuple, Callable, Dict, Type


class MockModule:
    """Мок-модуль для PyTorch с полной типизацией"""
    
    def __init__(self) -> None:
        self._attributes: Dict[str, Any] = {}
    
    def __getattr__(self, name: str) -> "MockModule":
        if name not in self._attributes:
            self._attributes[name] = MockModule()
        # Cast to MockModule to satisfy type checker
        attr = self._attributes[name]
        if isinstance(attr, MockModule):
            return attr
        # If it's not a MockModule (e.g., a function), wrap it
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            if not hasattr(self, '_attributes'):
                super().__setattr__('_attributes', {})
            self._attributes[name] = value
    
    def __call__(self, *args: Any, **kwargs: Any) -> "MockModule":
        return MockModule()
    
    def __bool__(self) -> bool:
        return False
    
    def __iter__(self) -> Iterator[Any]:
        return iter([])
    
    def cuda(self) -> "MockModule":
        return MockModule()
    
    def is_available(self) -> bool:
        return False
        
    def __mro_entries__(self, bases: Tuple[type, ...]) -> Tuple[()]:
        # Return empty tuple for proper inheritance
        return ()


# Create mock torch with proper attribute handling
torch = MockModule()

# Explicitly set up torch modules using setattr to avoid method assignment issues
torch_cuda = MockModule()
torch_cuda._attributes['is_available'] = lambda: False
setattr(torch, 'cuda', torch_cuda)

torch_nn = MockModule()
torch_nn._attributes['Module'] = MockModule
torch_nn._attributes['functional'] = MockModule()
setattr(torch, 'nn', torch_nn)

torch_utils = MockModule()
torch_utils_data = MockModule()
torch_utils._attributes['data'] = torch_utils_data
setattr(torch, 'utils', torch_utils)

setattr(torch, 'optim', MockModule())

# Export commonly used classes
Module: Type[MockModule] = MockModule
Tensor: Type[MockModule] = MockModule
DataLoader: Type[MockModule] = MockModule