"""
Mock PyTorch module for development without torch installation
"""

class MockModule:
    def __getattr__(self, name):
        return MockModule()
    
    def __call__(self, *args, **kwargs):
        return MockModule()
    
    def __bool__(self):
        return False
    
    def __iter__(self):
        return iter([])
    
    def cuda(self):
        return MockModule()
    
    def is_available(self):
        return False
        
    def __mro_entries__(self, bases):
        # Return empty tuple for proper inheritance
        return ()

# Create mock torch with common attributes
torch = MockModule()
torch.cuda = MockModule()
torch.cuda.is_available = lambda: False
torch.nn = MockModule()
torch.nn.Module = MockModule()
torch.nn.functional = MockModule()
torch.utils = MockModule()
torch.utils.data = MockModule()
torch.optim = MockModule()

# Export commonly used classes
Module = MockModule
Tensor = MockModule
DataLoader = MockModule