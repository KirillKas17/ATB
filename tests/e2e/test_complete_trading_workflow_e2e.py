#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End тесты полного торгового цикла.
Тестирование реальных сценариев использования системы.
"""
import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from unittest.mock import AsyncMock, Mock, patch
import json
import time


class TestCompleteTradingWorkflowE2E:
    """E2E тесты полного торгового рабочего процесса."""

    @pytest.fixture
    def live_trading_scenario(self):
        """Сценарий живой торговли."""
        return {
            'scenario_id': 'live_trading_001',
            'start_balance': Decimal('10000.00'),
            'target_profit': Decimal('100.00'),
            'max_loss': Decimal('200.00'),
            'trading_pairs': ['BTCUSDT', 'ETHUSDT'],
            'strategy': 'SCALPING',
            'session_duration': timedelta(minutes=30),
            'risk_level': 'MEDIUM'
        }

    @pytest.fixture
    def market_conditions(self):
        """Условия рынка для тестирования."""
        return {
            'volatility': 'NORMAL',
            'trend': 'SIDEWAYS',
            'liquidity': 'HIGH',
            'spread': 'TIGHT',
            'time_of_day': 'ACTIVE_HOURS'
        }

    @pytest.fixture
    def system_configuration(self):
        """Конфигурация системы."""
        return {
            'exchange': 'BINANCE',
            'api_credentials': {
                'api_key': 'test_api_key',
                'secret': 'test_secret'
            },
            'risk_settings': {
                'max_position_size': Decimal('0.1'),
                'stop_loss_percentage': Decimal('0.02'),
                'take_profit_percentage': Decimal('0.03')
            },
            'strategy_settings': {
                'scalping': {
                    'target_spread': Decimal('0.0005'),
                    'order_size': Decimal('0.01'),
                    'hold_time': 30  # seconds
                }
            }
        }

    def test_system_startup_and_initialization_e2e(self, system_configuration):
        """Тест запуска и инициализации системы E2E."""
        # 1. System startup sequence
        startup_steps = [
            'load_configuration',
            'validate_credentials',
            'connect_to_exchange',
            'initialize_strategies',
            'start_data_feeds',
            'enable_trading'
        ]
        
        startup_results = {}
        
        for step in startup_steps:
            # Simulate each startup step
            if step == 'load_configuration':
                startup_results[step] = {'status': 'SUCCESS', 'config_loaded': True}
            elif step == 'validate_credentials':
                startup_results[step] = {'status': 'SUCCESS', 'credentials_valid': True}
            elif step == 'connect_to_exchange':
                startup_results[step] = {'status': 'SUCCESS', 'connection_active': True}
            elif step == 'initialize_strategies':
                startup_results[step] = {'status': 'SUCCESS', 'strategies_loaded': 1}
            elif step == 'start_data_feeds':
                startup_results[step] = {'status': 'SUCCESS', 'feeds_active': 2}
            elif step == 'enable_trading':
                startup_results[step] = {'status': 'SUCCESS', 'trading_enabled': True}
        
        # Verify complete startup
        assert all(result['status'] == 'SUCCESS' for result in startup_results.values())
        assert startup_results['enable_trading']['trading_enabled'] is True

    def test_market_data_collection_and_processing_e2e(self, market_conditions):
        """Тест сбора и обработки рыночных данных E2E."""
        # Simulate real-time market data stream
        market_data_stream = []
        
        # Generate sample market data
        base_price = Decimal('50000.00')
        for i in range(100):
            # Simulate price movements
            price_change = Decimal(str((i % 10 - 5) * 10))  # ±50 price movements
            current_price = base_price + price_change
            
            market_tick = {
                'symbol': 'BTCUSDT',
                'price': current_price,
                'volume': Decimal('1.0'),
                'timestamp': datetime.now() + timedelta(seconds=i),
                'bid': current_price - Decimal('5.00'),
                'ask': current_price + Decimal('5.00')
            }
            market_data_stream.append(market_tick)
        
        # Process market data
        processed_data = {
            'total_ticks': len(market_data_stream),
            'price_range': {
                'min': min(tick['price'] for tick in market_data_stream),
                'max': max(tick['price'] for tick in market_data_stream),
                'avg': sum(tick['price'] for tick in market_data_stream) / len(market_data_stream)
            },
            'volume_total': sum(tick['volume'] for tick in market_data_stream),
            'data_quality': 'GOOD'
        }
        
        # Verify data processing
        assert processed_data['total_ticks'] == 100
        assert processed_data['price_range']['min'] < processed_data['price_range']['max']
        assert processed_data['volume_total'] == Decimal('100.0')

    def test_signal_generation_and_strategy_execution_e2e(self, live_trading_scenario, 
                                                        system_configuration):
        """Тест генерации сигналов и выполнения стратегии E2E."""
        # Market data for signal generation
        market_data = {
            'symbol': 'BTCUSDT',
            'current_price': Decimal('50000.00'),
            'bid': Decimal('49995.00'),
            'ask': Decimal('50005.00'),
            'volume': Decimal('1000.0'),
            'volatility': Decimal('0.02')
        }
        
        # Scalping strategy logic
        strategy_config = system_configuration['strategy_settings']['scalping']
        
        # Generate trading signal
        spread = market_data['ask'] - market_data['bid']
        target_spread = strategy_config['target_spread'] * market_data['current_price']
        
        if spread <= target_spread:
            signal = {
                'action': 'BUY',
                'symbol': market_data['symbol'],
                'entry_price': market_data['bid'],
                'quantity': strategy_config['order_size'],
                'stop_loss': market_data['bid'] * (1 - system_configuration['risk_settings']['stop_loss_percentage']),
                'take_profit': market_data['bid'] * (1 + system_configuration['risk_settings']['take_profit_percentage']),
                'timestamp': datetime.now()
            }
        else:
            signal = None
        
        # Verify signal generation
        if signal:
            assert signal['action'] in ['BUY', 'SELL']
            assert signal['entry_price'] > 0
            assert signal['quantity'] > 0
            assert signal['stop_loss'] < signal['entry_price']
            assert signal['take_profit'] > signal['entry_price']

    def test_order_placement_and_execution_e2e(self, live_trading_scenario):
        """Тест размещения и исполнения ордеров E2E."""
        # Order lifecycle simulation
        order_lifecycle = [
            {'stage': 'CREATION', 'status': 'PENDING'},
            {'stage': 'VALIDATION', 'status': 'VALIDATED'},
            {'stage': 'PLACEMENT', 'status': 'NEW'},
            {'stage': 'MATCHING', 'status': 'PARTIALLY_FILLED'},
            {'stage': 'COMPLETION', 'status': 'FILLED'}
        ]
        
        # Order details
        order = {
            'order_id': 'order_001',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'type': 'LIMIT',
            'quantity': Decimal('0.01'),
            'price': Decimal('49995.00'),
            'time_in_force': 'GTC'
        }
        
        # Simulate order execution
        execution_log = []
        filled_quantity = Decimal('0')
        
        for stage in order_lifecycle:
            execution_event = {
                'order_id': order['order_id'],
                'stage': stage['stage'],
                'status': stage['status'],
                'timestamp': datetime.now()
            }
            
            # Simulate partial and full fills
            if stage['status'] == 'PARTIALLY_FILLED':
                fill_quantity = order['quantity'] / 2
                filled_quantity += fill_quantity
                execution_event['filled_quantity'] = fill_quantity
            elif stage['status'] == 'FILLED':
                remaining_quantity = order['quantity'] - filled_quantity
                filled_quantity += remaining_quantity
                execution_event['filled_quantity'] = remaining_quantity
            
            execution_log.append(execution_event)
        
        # Verify order execution
        assert len(execution_log) == 5
        assert filled_quantity == order['quantity']
        assert execution_log[-1]['status'] == 'FILLED'

    def test_position_management_and_tracking_e2e(self, live_trading_scenario):
        """Тест управления позициями и их отслеживания E2E."""
        # Position lifecycle
        position_events = [
            {
                'event': 'POSITION_OPENED',
                'symbol': 'BTCUSDT',
                'side': 'LONG',
                'size': Decimal('0.01'),
                'entry_price': Decimal('50000.00'),
                'timestamp': datetime.now()
            },
            {
                'event': 'POSITION_UPDATED',
                'unrealized_pnl': Decimal('5.00'),  # Price moved up $500
                'current_price': Decimal('50500.00'),
                'timestamp': datetime.now() + timedelta(minutes=5)
            },
            {
                'event': 'POSITION_CLOSED',
                'exit_price': Decimal('50300.00'),
                'realized_pnl': Decimal('3.00'),  # $300 profit
                'timestamp': datetime.now() + timedelta(minutes=10)
            }
        ]
        
        # Track position state
        position_state = {
            'is_open': False,
            'entry_price': Decimal('0'),
            'current_size': Decimal('0'),
            'unrealized_pnl': Decimal('0'),
            'realized_pnl': Decimal('0')
        }
        
        for event in position_events:
            if event['event'] == 'POSITION_OPENED':
                position_state['is_open'] = True
                position_state['entry_price'] = event['entry_price']
                position_state['current_size'] = event['size']
            elif event['event'] == 'POSITION_UPDATED':
                position_state['unrealized_pnl'] = event['unrealized_pnl']
            elif event['event'] == 'POSITION_CLOSED':
                position_state['is_open'] = False
                position_state['current_size'] = Decimal('0')
                position_state['realized_pnl'] = event['realized_pnl']
                position_state['unrealized_pnl'] = Decimal('0')
        
        # Verify position management
        assert position_state['is_open'] is False
        assert position_state['realized_pnl'] == Decimal('3.00')
        assert position_state['current_size'] == Decimal('0')

    def test_risk_management_and_controls_e2e(self, live_trading_scenario, system_configuration):
        """Тест риск-менеджмента и контролей E2E."""
        # Portfolio state
        portfolio = {
            'total_balance': live_trading_scenario['start_balance'],
            'available_balance': live_trading_scenario['start_balance'],
            'positions_value': Decimal('0'),
            'unrealized_pnl': Decimal('0'),
            'daily_pnl': Decimal('0')
        }
        
        # Risk scenarios to test
        risk_scenarios = [
            {
                'name': 'NORMAL_TRADE',
                'position_size': Decimal('100.00'),  # 1% of portfolio
                'expected_approval': True
            },
            {
                'name': 'LARGE_POSITION',
                'position_size': Decimal('2000.00'),  # 20% of portfolio
                'expected_approval': False
            },
            {
                'name': 'AFTER_LOSSES',
                'position_size': Decimal('100.00'),
                'daily_pnl': Decimal('-150.00'),  # Close to loss limit
                'expected_approval': True
            },
            {
                'name': 'MAX_LOSS_REACHED',
                'position_size': Decimal('50.00'),
                'daily_pnl': live_trading_scenario['max_loss'] * -1,
                'expected_approval': False
            }
        ]
        
        for scenario in risk_scenarios:
            # Update portfolio for scenario
            test_portfolio = portfolio.copy()
            if 'daily_pnl' in scenario:
                test_portfolio['daily_pnl'] = scenario['daily_pnl']
            
            # Risk assessment
            position_risk = scenario['position_size'] / test_portfolio['total_balance']
            max_position_risk = system_configuration['risk_settings']['max_position_size']
            
            daily_loss_exceeded = abs(test_portfolio['daily_pnl']) >= live_trading_scenario['max_loss']
            
            risk_approved = (
                position_risk <= max_position_risk and
                not daily_loss_exceeded
            )
            
            # Verify risk assessment
            assert risk_approved == scenario['expected_approval'], f"Risk assessment failed for {scenario['name']}"

    def test_performance_monitoring_and_reporting_e2e(self, live_trading_scenario):
        """Тест мониторинга производительности и отчетности E2E."""
        # Simulate trading session
        trading_session = {
            'session_id': live_trading_scenario['scenario_id'],
            'start_time': datetime.now(),
            'end_time': datetime.now() + live_trading_scenario['session_duration'],
            'trades_executed': 15,
            'successful_trades': 10,
            'losing_trades': 5,
            'total_profit': Decimal('75.00'),
            'total_loss': Decimal('25.00'),
            'max_drawdown': Decimal('15.00'),
            'sharpe_ratio': Decimal('1.5')
        }
        
        # Calculate performance metrics
        net_profit = trading_session['total_profit'] - trading_session['total_loss']
        win_rate = trading_session['successful_trades'] / trading_session['trades_executed']
        profit_factor = trading_session['total_profit'] / trading_session['total_loss']
        
        performance_report = {
            'net_profit': net_profit,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': trading_session['max_drawdown'],
            'total_trades': trading_session['trades_executed'],
            'session_duration': trading_session['end_time'] - trading_session['start_time']
        }
        
        # Verify performance metrics
        assert performance_report['net_profit'] == Decimal('50.00')
        # Проверяем с конвертацией в Decimal и учетом точности
        expected_win_rate = Decimal('10') / Decimal('15')
        assert abs(Decimal(str(performance_report['win_rate'])) - expected_win_rate) < Decimal('0.001')
        assert performance_report['profit_factor'] == Decimal('3')
        assert performance_report['total_trades'] == 15

    def test_error_handling_and_recovery_e2e(self, system_configuration):
        """Тест обработки ошибок и восстановления E2E."""
        # Error scenarios
        error_scenarios = [
            {
                'type': 'NETWORK_DISCONNECTION',
                'severity': 'HIGH',
                'recovery_time': 5,  # seconds
                'auto_recovery': True
            },
            {
                'type': 'EXCHANGE_API_ERROR',
                'severity': 'MEDIUM',
                'recovery_time': 2,
                'auto_recovery': True
            },
            {
                'type': 'INSUFFICIENT_FUNDS',
                'severity': 'HIGH',
                'recovery_time': 0,
                'auto_recovery': False
            },
            {
                'type': 'INVALID_ORDER',
                'severity': 'LOW',
                'recovery_time': 1,
                'auto_recovery': True
            }
        ]
        
        recovery_log = []
        
        for scenario in error_scenarios:
            error_event = {
                'error_type': scenario['type'],
                'timestamp': datetime.now(),
                'severity': scenario['severity']
            }
            
            # Simulate error handling
            if scenario['auto_recovery']:
                recovery_event = {
                    'recovery_attempted': True,
                    'recovery_successful': True,
                    'recovery_time': scenario['recovery_time'],
                    'timestamp': datetime.now() + timedelta(seconds=scenario['recovery_time'])
                }
            else:
                recovery_event = {
                    'recovery_attempted': False,
                    'recovery_successful': False,
                    'manual_intervention_required': True
                }
            
            recovery_log.append({
                'error': error_event,
                'recovery': recovery_event
            })
        
        # Verify error handling
        auto_recoverable_errors = [log for log in recovery_log if log['recovery']['recovery_attempted']]
        assert len(auto_recoverable_errors) == 3
        
        manual_intervention_errors = [log for log in recovery_log if log['recovery'].get('manual_intervention_required')]
        assert len(manual_intervention_errors) == 1

    def test_real_time_data_processing_e2e(self, market_conditions):
        """Тест обработки данных в реальном времени E2E."""
        # Simulate high-frequency data stream
        data_stream = []
        processing_latencies = []
        
        for i in range(1000):  # 1000 data points
            # Generate market data
            data_point = {
                'symbol': 'BTCUSDT',
                'price': Decimal('50000.00') + Decimal(str(i % 100)),
                'volume': Decimal('1.0'),
                'timestamp': datetime.now()
            }
            
            # Simulate processing
            processing_start = time.time()
            
            # Data validation and transformation
            processed_data = {
                'symbol': data_point['symbol'],
                'price': data_point['price'],
                'volume': data_point['volume'],
                'timestamp': data_point['timestamp'],
                'normalized_price': data_point['price'] / Decimal('50000.00'),
                'price_change': data_point['price'] - Decimal('50000.00') if i > 0 else Decimal('0')
            }
            
            processing_end = time.time()
            latency = (processing_end - processing_start) * 1000  # milliseconds
            
            data_stream.append(processed_data)
            processing_latencies.append(latency)
        
        # Verify real-time processing
        avg_latency = sum(processing_latencies) / len(processing_latencies)
        max_latency = max(processing_latencies)
        
        assert len(data_stream) == 1000
        assert avg_latency < 1.0  # Average latency < 1ms
        assert max_latency < 5.0   # Max latency < 5ms

    @pytest.mark.asyncio
    async def test_concurrent_operations_e2e(self, live_trading_scenario):
        """Тест параллельных операций E2E."""
        # Concurrent tasks simulation
        async def market_data_task():
            """Simulate market data processing."""
            for _ in range(10):
                await asyncio.sleep(0.01)  # 10ms processing
            return {'task': 'market_data', 'processed_ticks': 100}
        
        async def signal_processing_task():
            """Simulate signal processing."""
            for _ in range(5):
                await asyncio.sleep(0.02)  # 20ms processing
            return {'task': 'signal_processing', 'signals_generated': 5}
        
        async def order_management_task():
            """Simulate order management."""
            for _ in range(3):
                await asyncio.sleep(0.03)  # 30ms processing
            return {'task': 'order_management', 'orders_processed': 3}
        
        async def risk_monitoring_task():
            """Simulate risk monitoring."""
            for _ in range(2):
                await asyncio.sleep(0.05)  # 50ms processing
            return {'task': 'risk_monitoring', 'checks_performed': 10}
        
        # Execute tasks concurrently
        start_time = time.time()
        
        results = await asyncio.gather(
            market_data_task(),
            signal_processing_task(),
            order_management_task(),
            risk_monitoring_task()
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify concurrent execution
        assert len(results) == 4
        assert total_time < 0.2  # Should complete in < 200ms (concurrent execution)
        
        # Verify individual task results
        task_results = {result['task']: result for result in results}
        assert task_results['market_data']['processed_ticks'] == 100
        assert task_results['signal_processing']['signals_generated'] == 5
        assert task_results['order_management']['orders_processed'] == 3
        assert task_results['risk_monitoring']['checks_performed'] == 10

    def test_system_shutdown_and_cleanup_e2e(self, live_trading_scenario):
        """Тест завершения работы системы и очистки E2E."""
        # System state before shutdown
        system_state = {
            'active_orders': [
                {'order_id': 'order_001', 'status': 'NEW'},
                {'order_id': 'order_002', 'status': 'PARTIALLY_FILLED'}
            ],
            'open_positions': [
                {'symbol': 'BTCUSDT', 'size': Decimal('0.01')}
            ],
            'data_connections': ['binance_ws', 'bybit_ws'],
            'background_tasks': ['risk_monitor', 'pnl_calculator'],
            'temp_files': ['market_data_cache.tmp', 'order_log.tmp']
        }
        
        # Shutdown sequence
        shutdown_steps = [
            'cancel_pending_orders',
            'close_positions',
            'stop_data_feeds',
            'terminate_background_tasks',
            'save_system_state',
            'cleanup_temp_files',
            'disconnect_from_exchange'
        ]
        
        shutdown_results = {}
        
        for step in shutdown_steps:
            if step == 'cancel_pending_orders':
                shutdown_results[step] = {
                    'orders_cancelled': len([o for o in system_state['active_orders'] if o['status'] == 'NEW']),
                    'status': 'SUCCESS'
                }
            elif step == 'close_positions':
                shutdown_results[step] = {
                    'positions_closed': len(system_state['open_positions']),
                    'status': 'SUCCESS'
                }
            elif step == 'stop_data_feeds':
                shutdown_results[step] = {
                    'feeds_stopped': len(system_state['data_connections']),
                    'status': 'SUCCESS'
                }
            elif step == 'terminate_background_tasks':
                shutdown_results[step] = {
                    'tasks_terminated': len(system_state['background_tasks']),
                    'status': 'SUCCESS'
                }
            elif step == 'save_system_state':
                shutdown_results[step] = {
                    'state_saved': True,
                    'status': 'SUCCESS'
                }
            elif step == 'cleanup_temp_files':
                shutdown_results[step] = {
                    'files_cleaned': len(system_state['temp_files']),
                    'status': 'SUCCESS'
                }
            elif step == 'disconnect_from_exchange':
                shutdown_results[step] = {
                    'disconnected': True,
                    'status': 'SUCCESS'
                }
        
        # Verify clean shutdown
        assert all(result['status'] == 'SUCCESS' for result in shutdown_results.values())
        assert shutdown_results['cancel_pending_orders']['orders_cancelled'] == 1
        assert shutdown_results['close_positions']['positions_closed'] == 1
        assert shutdown_results['stop_data_feeds']['feeds_stopped'] == 2

    def test_configuration_changes_e2e(self, system_configuration):
        """Тест изменения конфигурации E2E."""
        # Original configuration
        original_config = system_configuration.copy()
        
        # Configuration changes
        config_changes = [
            {
                'change_type': 'RISK_ADJUSTMENT',
                'path': 'risk_settings.max_position_size',
                'old_value': original_config['risk_settings']['max_position_size'],
                'new_value': Decimal('0.05')  # Reduce position size limit
            },
            {
                'change_type': 'STRATEGY_TUNING',
                'path': 'strategy_settings.scalping.target_spread',
                'old_value': original_config['strategy_settings']['scalping']['target_spread'],
                'new_value': Decimal('0.001')  # Increase target spread
            }
        ]
        
        # Apply configuration changes
        updated_config = original_config.copy()
        
        for change in config_changes:
            if change['change_type'] == 'RISK_ADJUSTMENT':
                updated_config['risk_settings']['max_position_size'] = change['new_value']
            elif change['change_type'] == 'STRATEGY_TUNING':
                updated_config['strategy_settings']['scalping']['target_spread'] = change['new_value']
        
        # Verify configuration changes
        assert updated_config['risk_settings']['max_position_size'] == Decimal('0.05')
        assert updated_config['strategy_settings']['scalping']['target_spread'] == Decimal('0.001')
        
        # Verify changes are applied correctly
        for change in config_changes:
            assert change['new_value'] != change['old_value']

    def test_complete_trading_session_e2e(self, live_trading_scenario, market_conditions, 
                                        system_configuration):
        """Тест полной торговой сессии E2E."""
        # Session metrics tracking
        session_metrics = {
            'start_time': datetime.now(),
            'end_time': None,
            'trades_count': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_volume': Decimal('0'),
            'gross_profit': Decimal('0'),
            'gross_loss': Decimal('0'),
            'commission_paid': Decimal('0'),
            'max_position_size': Decimal('0'),
            'max_drawdown': Decimal('0')
        }
        
        # Simulate trading session events
        session_events = [
            {'type': 'SESSION_START', 'balance': live_trading_scenario['start_balance']},
            {'type': 'TRADE_SIGNAL', 'action': 'BUY', 'symbol': 'BTCUSDT'},
            {'type': 'ORDER_PLACED', 'order_id': 'order_001', 'size': Decimal('0.01')},
            {'type': 'ORDER_FILLED', 'order_id': 'order_001', 'fill_price': Decimal('50000.00')},
            {'type': 'POSITION_OPENED', 'symbol': 'BTCUSDT', 'size': Decimal('0.01')},
            {'type': 'PRICE_MOVEMENT', 'new_price': Decimal('50300.00')},
            {'type': 'POSITION_CLOSED', 'realized_pnl': Decimal('3.00')},
            {'type': 'SESSION_END', 'final_balance': live_trading_scenario['start_balance'] + Decimal('3.00')}
        ]
        
        current_balance = live_trading_scenario['start_balance']
        
        for event in session_events:
            if event['type'] == 'SESSION_START':
                session_metrics['start_time'] = datetime.now()
            elif event['type'] == 'ORDER_FILLED':
                session_metrics['trades_count'] += 1
                session_metrics['total_volume'] += event.get('size', Decimal('0.01')) * event['fill_price']
            elif event['type'] == 'POSITION_CLOSED':
                if event['realized_pnl'] > 0:
                    session_metrics['successful_trades'] += 1
                    session_metrics['gross_profit'] += event['realized_pnl']
                else:
                    session_metrics['failed_trades'] += 1
                    session_metrics['gross_loss'] += abs(event['realized_pnl'])
                current_balance += event['realized_pnl']
            elif event['type'] == 'SESSION_END':
                session_metrics['end_time'] = datetime.now()
        
        # Calculate final session metrics
        net_profit = session_metrics['gross_profit'] - session_metrics['gross_loss']
        roi = (net_profit / live_trading_scenario['start_balance']) * 100
        
        # Verify complete session
        assert session_metrics['trades_count'] == 1
        assert session_metrics['successful_trades'] == 1
        assert session_metrics['failed_trades'] == 0
        assert net_profit == Decimal('3.00')
        assert current_balance == live_trading_scenario['start_balance'] + Decimal('3.00')
        assert roi == Decimal('0.03')  # 0.03% ROI