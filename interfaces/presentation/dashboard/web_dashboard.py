"""
Веб-версия современного торгового дашборда для максимального WOW-эффекта на Twitch.
Flask + WebSocket + Apple-style темная тема + анимации.
"""

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import threading
import time
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import asyncio
import numpy as np
from decimal import Decimal

app = Flask(__name__)
app.config['SECRET_KEY'] = 'atb_trading_dashboard_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

class WebTradingDashboard:
    """Веб-версия торгового дашборда"""
    
    def __init__(self):
        self.live_data = {
            'portfolio_value': 50000.0,
            'daily_pnl': 1250.0,
            'total_trades': 127,
            'win_rate': 68.5,
            'active_positions': 8,
            'ai_confidence': 87.3,
            'market_sentiment': 65.2,
            'volatility': 23.4,
            'prices': {
                'BTCUSDT': 42150.0,
                'ETHUSDT': 2580.0,
                'ADAUSDT': 0.485,
                'SOLUSDT': 98.45,
                'AVAXUSDT': 35.67,
                'DOTUSDT': 6.23,
            },
            'price_history': {},
            'orderbook': {'bids': [], 'asks': []},
            'recent_trades': [],
            'ai_signals': [],
            'market_analysis': [],
            'portfolio_allocation': {},
            'risk_metrics': {},
            'performance_stats': {},
        }
        
        # Инициализация истории цен
        for symbol in self.live_data['prices']:
            self.live_data['price_history'][symbol] = []
            
        self.is_running = False
        self.start_live_updates()
        
    def start_live_updates(self):
        """Запуск live-обновлений"""
        self.is_running = True
        self.update_thread = threading.Thread(target=self._live_update_loop, daemon=True)
        self.update_thread.start()
        
    def _live_update_loop(self):
        """Основной цикл обновления live-данных"""
        while self.is_running:
            # Симуляция различных типов данных
            self._simulate_market_data()
            self._simulate_trading_activity()
            self._simulate_ai_analysis()
            self._simulate_portfolio_changes()
            self._simulate_risk_metrics()
            
            # Отправка обновлений через WebSocket
            socketio.emit('live_update', self.get_dashboard_data())
            
            time.sleep(0.5)  # Обновление каждые 500мс
            
    def _simulate_market_data(self):
        """Симуляция рыночных данных с волатильностью"""
        for symbol in self.live_data['prices']:
            current_price = self.live_data['prices'][symbol]
            
            # Реалистичная волатильность
            if symbol == 'BTCUSDT':
                volatility = 0.015  # 1.5%
            elif symbol == 'ETHUSDT':
                volatility = 0.02   # 2%
            else:
                volatility = 0.03   # 3% для альткоинов
                
            change = random.gauss(0, volatility)  # Нормальное распределение
            new_price = current_price * (1 + change)
            
            # Предотвращение отрицательных цен
            if new_price > 0:
                self.live_data['prices'][symbol] = new_price
                
                # Сохранение истории (последние 100 точек)
                if len(self.live_data['price_history'][symbol]) >= 100:
                    self.live_data['price_history'][symbol].pop(0)
                self.live_data['price_history'][symbol].append({
                    'timestamp': datetime.now().isoformat(),
                    'price': new_price
                })
                
    def _simulate_trading_activity(self):
        """Симуляция торговой активности"""
        # Генерация новых сделок
        if random.random() < 0.4:  # 40% шанс новой сделки
            symbol = random.choice(list(self.live_data['prices'].keys()))
            base_price = self.live_data['prices'][symbol]
            
            trade = {
                'id': random.randint(100000, 999999),
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'side': random.choice(['BUY', 'SELL']),
                'price': base_price + random.uniform(-base_price*0.001, base_price*0.001),
                'quantity': random.uniform(0.001, 1.0),
                'value': 0
            }
            trade['value'] = trade['price'] * trade['quantity']
            
            # Ограничение количества сделок
            if len(self.live_data['recent_trades']) >= 50:
                self.live_data['recent_trades'].pop(0)
            self.live_data['recent_trades'].append(trade)
            
        # Обновление ордербука
        self._update_orderbook()
        
    def _update_orderbook(self):
        """Обновление ордербука"""
        symbol = 'BTCUSDT'  # Фокус на BTC
        base_price = self.live_data['prices'][symbol]
        
        # Генерация бидов (покупки)
        bids = []
        for i in range(10):
            price = base_price - (i + 1) * random.uniform(0.5, 2.0)
            quantity = random.uniform(0.1, 5.0)
            bids.append({
                'price': price,
                'quantity': quantity,
                'total': price * quantity
            })
            
        # Генерация асков (продажи)
        asks = []
        for i in range(10):
            price = base_price + (i + 1) * random.uniform(0.5, 2.0)
            quantity = random.uniform(0.1, 5.0)
            asks.append({
                'price': price,
                'quantity': quantity,
                'total': price * quantity
            })
            
        self.live_data['orderbook'] = {
            'symbol': symbol,
            'bids': bids,
            'asks': asks,
            'spread': asks[0]['price'] - bids[0]['price'] if asks and bids else 0
        }
        
    def _simulate_ai_analysis(self):
        """Симуляция AI анализа и сигналов"""
        if random.random() < 0.15:  # 15% шанс нового сигнала
            signal_types = [
                {'type': 'BUY', 'emoji': '🟢', 'color': '#30D158'},
                {'type': 'SELL', 'emoji': '🔴', 'color': '#FF453A'},
                {'type': 'HOLD', 'emoji': '🟡', 'color': '#FF9F0A'},
                {'type': 'ALERT', 'emoji': '⚠️', 'color': '#FF9F0A'},
            ]
            
            signal_messages = [
                "Strong resistance level detected",
                "Bullish divergence in momentum",
                "Volume spike indicates breakout",
                "Support level holding strong",
                "Whale activity detected",
                "Market sentiment shifting",
                "Technical pattern completion",
                "Risk management activated",
                "Fibonacci level reached",
                "Neural network pattern match"
            ]
            
            signal_type = random.choice(signal_types)
            symbol = random.choice(list(self.live_data['prices'].keys()))
            
            signal = {
                'id': random.randint(1000, 9999),
                'timestamp': datetime.now().isoformat(),
                'type': signal_type['type'],
                'emoji': signal_type['emoji'],
                'color': signal_type['color'],
                'symbol': symbol,
                'message': random.choice(signal_messages),
                'confidence': random.uniform(60, 98),
                'price': self.live_data['prices'][symbol],
                'priority': random.choice(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
            }
            
            # Ограничение количества сигналов
            if len(self.live_data['ai_signals']) >= 20:
                self.live_data['ai_signals'].pop(0)
            self.live_data['ai_signals'].append(signal)
            
        # Обновление анализа рынка
        if random.random() < 0.08:  # 8% шанс нового анализа
            analysis_items = [
                "📊 Technical analysis: RSI approaching oversold",
                "🔍 Pattern recognition: Head and shoulders forming",
                "📈 Volume analysis: Above average trading volume",
                "🌊 Support/Resistance: Key level at $42,000",
                "⚡ Momentum: Bullish crossover detected",
                "🎯 Price target: Next resistance at $44,500",
                "🛡️ Risk assessment: Moderate volatility expected",
                "🔮 ML prediction: 73% probability of upward move",
                "🏦 Institutional flow: Net buying pressure",
                "📱 Social sentiment: Positive retail sentiment"
            ]
            
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'message': random.choice(analysis_items),
                'category': random.choice(['TECHNICAL', 'FUNDAMENTAL', 'SENTIMENT', 'RISK']),
                'impact': random.choice(['LOW', 'MEDIUM', 'HIGH'])
            }
            
            # Ограничение количества анализов
            if len(self.live_data['market_analysis']) >= 30:
                self.live_data['market_analysis'].pop(0)
            self.live_data['market_analysis'].append(analysis)
            
    def _simulate_portfolio_changes(self):
        """Симуляция изменений портфеля"""
        # Обновление основных метрик
        pnl_change = random.uniform(-100, 150)
        self.live_data['daily_pnl'] += pnl_change
        self.live_data['portfolio_value'] = 50000 + self.live_data['daily_pnl']
        
        # Иногда меняем количество активных позиций
        if random.random() < 0.1:
            self.live_data['active_positions'] = max(1, 
                self.live_data['active_positions'] + random.choice([-1, 0, 1]))
                
        # Обновление аллокации портфеля
        symbols = list(self.live_data['prices'].keys())
        total_value = self.live_data['portfolio_value']
        
        allocation = {}
        remaining = 100.0
        
        for i, symbol in enumerate(symbols[:-1]):
            if remaining > 0:
                percent = random.uniform(5, min(30, remaining))
                allocation[symbol] = {
                    'percentage': percent,
                    'value': total_value * (percent / 100),
                    'change_24h': random.uniform(-5, 8)
                }
                remaining -= percent
                
        # Последний символ получает остаток
        if symbols:
            last_symbol = symbols[-1]
            allocation[last_symbol] = {
                'percentage': remaining,
                'value': total_value * (remaining / 100),
                'change_24h': random.uniform(-5, 8)
            }
            
        self.live_data['portfolio_allocation'] = allocation
        
    def _simulate_risk_metrics(self):
        """Симуляция метрик риска"""
        # Обновление AI метрик
        self.live_data['ai_confidence'] = max(0, min(100,
            self.live_data['ai_confidence'] + random.uniform(-3, 3)))
        self.live_data['market_sentiment'] = max(0, min(100,
            self.live_data['market_sentiment'] + random.uniform(-4, 4)))
        self.live_data['volatility'] = max(0, min(100,
            self.live_data['volatility'] + random.uniform(-2, 2)))
            
        # Дополнительные метрики риска
        self.live_data['risk_metrics'] = {
            'var_1d': random.uniform(2, 8),  # Value at Risk 1 день
            'sharpe_ratio': random.uniform(0.5, 2.5),
            'max_drawdown': random.uniform(5, 15),
            'correlation_btc': random.uniform(0.3, 0.9),
            'leverage_ratio': random.uniform(1.1, 3.0),
            'liquidity_score': random.uniform(70, 95)
        }
        
        # Статистика производительности
        self.live_data['performance_stats'] = {
            'total_trades': self.live_data['total_trades'] + random.choice([0, 0, 0, 1]),
            'win_rate': max(40, min(80, self.live_data['win_rate'] + random.uniform(-1, 1))),
            'avg_win': random.uniform(150, 400),
            'avg_loss': random.uniform(-80, -200),
            'largest_win': random.uniform(800, 2000),
            'largest_loss': random.uniform(-300, -800),
            'profit_factor': random.uniform(1.2, 2.8)
        }
        
    def get_dashboard_data(self):
        """Получение всех данных для дашборда"""
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio': {
                'value': self.live_data['portfolio_value'],
                'daily_pnl': self.live_data['daily_pnl'],
                'allocation': self.live_data['portfolio_allocation']
            },
            'prices': self.live_data['prices'],
            'price_history': self.live_data['price_history'],
            'orderbook': self.live_data['orderbook'],
            'recent_trades': self.live_data['recent_trades'][-10:],
            'ai_signals': self.live_data['ai_signals'][-8:],
            'market_analysis': self.live_data['market_analysis'][-10:],
            'ai_metrics': {
                'confidence': self.live_data['ai_confidence'],
                'sentiment': self.live_data['market_sentiment'],
                'volatility': self.live_data['volatility']
            },
            'risk_metrics': self.live_data['risk_metrics'],
            'performance': self.live_data['performance_stats'],
            'trading': {
                'active_positions': self.live_data['active_positions'],
                'total_trades': self.live_data['total_trades'],
                'win_rate': self.live_data['win_rate']
            }
        }

# Глобальный экземпляр дашборда
dashboard = WebTradingDashboard()

@app.route('/')
def index():
    """Главная страница дашборда"""
    return render_template('dashboard.html')

@app.route('/api/dashboard-data')
def get_dashboard_data():
    """API для получения данных дашборда"""
    return jsonify(dashboard.get_dashboard_data())

@socketio.on('connect')
def handle_connect():
    """Обработка подключения клиента"""
    print('Client connected')
    emit('initial_data', dashboard.get_dashboard_data())

@socketio.on('disconnect')
def handle_disconnect():
    """Обработка отключения клиента"""
    print('Client disconnected')

if __name__ == '__main__':
    print("🚀 Starting ATB Web Trading Dashboard...")
    print("💫 Apple-style Dark Theme")
    print("📺 Twitch Demo Mode")
    print("🌐 Web Interface Ready")
    print("🔴 Live Data Streaming via WebSocket")
    print("\n🌍 Dashboard URL: http://localhost:5000")
    
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)