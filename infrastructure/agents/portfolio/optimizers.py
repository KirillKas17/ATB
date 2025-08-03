from abc import ABC


class IPortfolioOptimizer(ABC):
    pass


class MeanVarianceOptimizer(IPortfolioOptimizer):
    pass


class RiskParityOptimizer(IPortfolioOptimizer):
    pass
