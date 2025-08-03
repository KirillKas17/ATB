#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Syntra - Main Entry Point
Domain-Driven Design Architecture with News Sentiment Integration
"""

import asyncio
import logging
import signal
import sys
from typing import Any

# Configuration imports
from shared.models.config import create_default_config

# Core application imports
from application.di_container_refactored import get_service_locator
from application.use_cases.trading_orchestrator.core import DefaultTradingOrchestratorUseCase

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point for the trading system."""
    try:
        # Initialize configuration
        config = create_default_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize service locator and orchestrator
        service_locator = get_service_locator()
        orchestrator = service_locator.get_use_case(DefaultTradingOrchestratorUseCase)
        logger.info("Trading orchestrator initialized")

        def shutdown_handler(signum: int, frame: Any) -> None:
            """Handle shutdown signals gracefully."""
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(orchestrator.stop())

        # Register signal handlers
        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)
        
        # Start the trading system
        logger.info("Starting trading system...")
        await orchestrator.start()
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)
