"""
35=B 4;O >?@545;5=8O @K=>G=>3> @568<0.
-B>B <>4C;L ?@54>AB02;O5B DC=:F8>=0;L=>ABL 4;O >?@545;5=8O B5:CI53> @K=>G=>3> @568<0
=0 >A=>25 B5E=8G5A:8E 8=48:0B>@>2 8 <0H8==>3> >1CG5=8O.
@E8B5:BC@0:
- MarketRegimeAgent: >A=>2=>9 035=B >?@545;5=8O @568<0
- MarketRegime: ?5@5G8A;5=85 2>7<>6=KE @K=>G=KE @568<>2
- IIndicatorCalculator: 8=B5@D59A 4;O @0AG5B0 8=48:0B>@>2
- DefaultIndicatorCalculator: @50;870F8O :0;L:C;OB>@0 8=48:0B>@>2
A?>;L7>20=85:
    agent = MarketRegimeAgent(config)
    await agent.initialize()
    result = await agent.process(data)
    await agent.cleanup()
"""

__all__ = [
    "MarketRegimeAgent",
    "MarketRegime",
    "IIndicatorCalculator",
    "DefaultIndicatorCalculator",
]
