# Ошибки mypy для слоя application

--- 

## application

- application/market/mm_follow_controller.py:145:25: error: Argument 2 to "find_similar_patterns" of "IPatternMemoryRepository" has incompatible type "dict[str, Any] | PatternFeatures"; expected "dict[str, Any]"  [arg-type]
- application/market/mm_follow_controller.py:157:26: error: Unexpected keyword argument "metadata" for "MatchedPattern"  [call-arg]
- application/market/mm_follow_controller.py:158:32: error: Argument "pattern_memory" to "MatchedPattern" has incompatible type "Any | None"; expected "PatternMemory"  [arg-type]
- application/market/mm_follow_controller.py:162:34: error: Argument "expected_outcome" to "MatchedPattern" has incompatible type "Any | None"; expected "PatternResult"  [arg-type]
- application/filters/orderbook_filter.py:390:40: error: Unsupported operand types for / ("None" and "int")  [operator]
- application/filters/orderbook_filter.py:390:40: error: Incompatible types in assignment (expression has type "float", target has type "int | None")  [assignment]
// ... и так далее для всех ошибок application ... 