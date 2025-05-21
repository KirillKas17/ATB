import numpy as np
import pandas as pd
import pytest

from strategies.manipulation_strategies import (
    ManipulationStrategy,
    manipulation_strategy_fake_breakout,
    manipulation_strategy_stop_hunt,
)
from utils.data_loader import load_market_data

# ... existing code ...
