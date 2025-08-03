"""
Mock PyTorch module for development without torch installation
"""
from typing import Any, Iterator, Tuple, Callable


class MockModule:
    """Мок-модуль для PyTorch с полной типизацией"""
    
    def __getattr__(self, name: str) -> "MockModule":
        return MockModule()
    
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


# Create mock torch with common attributes
torch = MockModule()
torch.cuda = MockModule()  # type: ignore[method-assign]
torch.cuda.is_available = lambda: False  # type: ignore[method-assign]
torch.nn = MockModule()  # type: ignore[attr-defined]
torch.nn.Module = MockModule()
torch.nn.functional = MockModule()
torch.utils = MockModule()  # type: ignore[attr-defined]
torch.utils.data = MockModule()
torch.optim = MockModule()  # type: ignore[attr-defined]

# Export commonly used classes
Module: type[MockModule] = MockModule
Tensor: type[MockModule] = MockModule
DataLoader: type[MockModule] = MockModule