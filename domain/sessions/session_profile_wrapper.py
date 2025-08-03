"""
Обёртка для SessionProfile с удобным API.
"""

from typing import Dict, List, Optional, Union
from loguru import logger

from domain.types.session_types import (
    SessionBehavior,
    SessionPhase,
    SessionTimeWindow,
    SessionType,
    SessionProfile as SessionProfileModel,
    LiquidityProfile,
    SessionIntensity,
    MarketRegime
)
from datetime import time


class SessionProfile:
    """Удобная обёртка для SessionProfile с упрощённым API."""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self._profiles: Dict[SessionType, SessionProfileModel] = {}
        self._initialize_default_profiles()
        logger.info(f"SessionProfile создан для символа {symbol}")
    
    def _initialize_default_profiles(self) -> None:
        """Инициализация профилей по умолчанию."""
        try:
            # Азиатская сессия
            asian_behavior = SessionBehavior(
                typical_volatility_spike_minutes=45,
                volume_peak_hours=[2, 4, 6],
                quiet_hours=[1, 5],
                avg_volume_multiplier=0.8,
                avg_volatility_multiplier=0.9,
                typical_direction_bias=0.1,
                common_patterns=["asian_range", "breakout_failure"],
                false_breakout_probability=0.4,
                reversal_probability=0.25,
                overlap_impact={"london": 1.2, "new_york": 0.9},
            )
            
            asian_time_window = SessionTimeWindow(
                start_time=time(0, 0),
                end_time=time(8, 0),
                timezone="UTC",
            )
            
            asian_profile = SessionProfileModel(
                session_type=SessionType.ASIAN,
                time_window=asian_time_window,
                behavior=asian_behavior,
                description="Азиатская торговая сессия (Токио)",
                typical_volume_multiplier=0.8,
                typical_volatility_multiplier=0.9,
                liquidity_profile=LiquidityProfile.TIGHT,
                intensity_profile=SessionIntensity.NORMAL,
                market_regime_tendency=MarketRegime.RANGING
            )
            
            # Лондонская сессия
            london_behavior = SessionBehavior(
                typical_volatility_spike_minutes=30,
                volume_peak_hours=[8, 10, 12],
                quiet_hours=[13, 15],
                avg_volume_multiplier=1.2,
                avg_volatility_multiplier=1.3,
                typical_direction_bias=0.0,
                common_patterns=["london_breakout", "trend_continuation"],
                false_breakout_probability=0.3,
                reversal_probability=0.2,
                overlap_impact={"asian": 1.1, "new_york": 1.4},
            )
            
            london_time_window = SessionTimeWindow(
                start_time=time(8, 0),
                end_time=time(16, 0),
                timezone="UTC",
            )
            
            london_profile = SessionProfileModel(
                session_type=SessionType.LONDON,
                time_window=london_time_window,
                behavior=london_behavior,
                description="Лондонская торговая сессия",
                typical_volume_multiplier=1.2,
                typical_volatility_multiplier=1.3,
                liquidity_profile=LiquidityProfile.ABUNDANT,
                intensity_profile=SessionIntensity.NORMAL,
                market_regime_tendency=MarketRegime.TRENDING_BULL
            )
            
            # Нью-Йоркская сессия
            ny_behavior = SessionBehavior(
                typical_volatility_spike_minutes=35,
                volume_peak_hours=[14, 16, 18],
                quiet_hours=[19, 21],
                avg_volume_multiplier=1.0,
                avg_volatility_multiplier=1.1,
                typical_direction_bias=-0.05,
                common_patterns=["ny_reversal", "afternoon_drift"],
                false_breakout_probability=0.35,
                reversal_probability=0.3,
                overlap_impact={"london": 1.3, "asian": 0.8},
            )
            
            ny_time_window = SessionTimeWindow(
                start_time=time(13, 0),
                end_time=time(22, 0),
                timezone="UTC",
            )
            
            ny_profile = SessionProfileModel(
                session_type=SessionType.NEW_YORK,
                time_window=ny_time_window,
                behavior=ny_behavior,
                description="Нью-Йоркская торговая сессия",
                typical_volume_multiplier=1.0,
                typical_volatility_multiplier=1.1,
                liquidity_profile=LiquidityProfile.EXCESSIVE,
                intensity_profile=SessionIntensity.NORMAL,
                market_regime_tendency=MarketRegime.TRENDING_BULL
            )
            
            # Сиднейская сессия
            sydney_behavior = SessionBehavior(
                typical_volatility_spike_minutes=25,
                volume_peak_hours=[22, 0, 2],
                quiet_hours=[1, 3],
                avg_volume_multiplier=0.6,
                avg_volatility_multiplier=0.7,
                typical_direction_bias=0.05,
                common_patterns=["sydney_gap", "quiet_range"],
                false_breakout_probability=0.5,
                reversal_probability=0.15,
                overlap_impact={"asian": 1.0, "london": 0.7},
            )
            
            sydney_time_window = SessionTimeWindow(
                start_time=time(22, 0),
                end_time=time(6, 0),
                timezone="UTC",
            )
            
            sydney_profile = SessionProfileModel(
                session_type=SessionType.SYDNEY,
                time_window=sydney_time_window,
                behavior=sydney_behavior,
                description="Сиднейская торговая сессия",
                typical_volume_multiplier=0.6,
                typical_volatility_multiplier=0.7,
                liquidity_profile=LiquidityProfile.SCARCE,
                intensity_profile=SessionIntensity.NORMAL,
                market_regime_tendency=MarketRegime.RANGING
            )
            
            # Сохраняем профили
            self._profiles = {
                SessionType.ASIAN: asian_profile,
                SessionType.LONDON: london_profile,
                SessionType.NEW_YORK: ny_profile,
                SessionType.SYDNEY: sydney_profile
            }
            
            logger.info(f"Initialized {len(self._profiles)} default session profiles")
            
        except Exception as e:
            logger.error(f"Error initializing session profiles: {e}")
            self._profiles = {}
    
    def get_profile(self, session_type: SessionType) -> Optional[SessionProfileModel]:
        """Получить профиль сессии."""
        return self._profiles.get(session_type)
    
    def get_all_profiles(self) -> Dict[SessionType, SessionProfileModel]:
        """Получить все профили."""
        return self._profiles.copy()
    
    def get_current_session_profile(self) -> Optional[SessionProfileModel]:
        """Получить профиль текущей сессии (упрощённая версия)."""
        # В реальной реализации здесь была бы логика определения текущей сессии по времени
        return self._profiles.get(SessionType.LONDON)  # Дефолтная сессия
    
    @property
    def profiles_count(self) -> int:
        """Количество загруженных профилей."""
        return len(self._profiles)