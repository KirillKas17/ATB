#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Syntra - Safe Main Entry Point
Minimal version for testing startup
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add current directory to path for safe imports
sys.path.insert(0, str(Path(__file__).parent))

from safe_import_wrapper import safe_import
from application.di_container_safe import get_safe_service_locator

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Safe main entry point for the trading system."""
    logger.info("🚀 Starting ATB Trading System (Safe Mode)")
    
    try:
        # Load environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv()
            logger.info("✅ Environment variables loaded")
        except ImportError:
            logger.warning("⚠️ python-dotenv not available")
        
        # Initialize service locator
        logger.info("🔧 Initializing service locator...")
        service_locator = get_safe_service_locator()
        logger.info("✅ Service locator initialized")
        
        # Test core services
        logger.info("🧪 Testing core services...")
        
        trading_service = service_locator.trading_service()
        logger.info(f"✅ Trading service available: {type(trading_service).__name__}")
        
        risk_service = service_locator.risk_service()
        logger.info(f"✅ Risk service available: {type(risk_service).__name__}")
        
        market_service = service_locator.market_service()
        logger.info(f"✅ Market service available: {type(market_service).__name__}")
        
        logger.info("🎉 ATB Trading System started successfully!")
        logger.info("💡 System is ready for configuration and trading")
        
        # Keep alive for a moment to show success
        await asyncio.sleep(2)
        
        logger.info("🏁 Safe startup completed. System is operational.")
        
    except Exception as e:
        logger.error(f"❌ Failed to start trading system: {e}")
        logger.exception("Full error details:")
        sys.exit(1)


if __name__ == "__main__":
    print("🚀 ATB Trading System - Safe Startup")
    print("=" * 50)
    asyncio.run(main())