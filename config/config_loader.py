from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class StrategyConfig(BaseModel):
    default_timeframe: str = Field(default="5m")
    max_active_strategies: int = Field(default=3)
    slippage_model: Literal["realistic", "average", "none"] = Field(default="realistic")
    regime_detection_window: int = Field(default=240)


class TradingConfig(BaseModel):
    max_risk_per_trade: float = Field(default=0.01)
    position_sizing: Literal["dynamic", "fixed"] = Field(default="dynamic")
    leverage: str = Field(default="3x")


class ExchangeConfig(BaseModel):
    use_testnet: bool = Field(default=True)
    reconnect_interval: int = Field(default=5)


class SimulationConfig(BaseModel):
    latency_model: Literal["realistic", "average", "none"] = Field(default="realistic")
    slippage_model: Literal["realistic", "average", "none"] = Field(default="average")
    capital: float = Field(default=10000)


class MLConfig(BaseModel):
    model_update_interval: str = Field(default="6h")
    validation_window: str = Field(default="30d")


class DashboardConfig(BaseModel):
    refresh_interval: str = Field(default="5s")
    theme: Literal["dark", "light"] = Field(default="dark")


class Config(BaseModel):
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)


def load_config(config_path: str = "config/config.yaml") -> Config:
    return ""
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Config: Validated configuration object
    """
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            print(
                f"Config file not found at {config_path}, using default configuration"
            )
            return Config()

        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)
            if not config_data:
                print("Empty config file, using default configuration")
                return Config()

        # Create and validate configuration
        config = Config(**config_data)
        return config

    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {str(e)}")
        return Config()


# Create global configuration instance
config = load_config()


def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns:
        Config: Global configuration object
    """
    return config
