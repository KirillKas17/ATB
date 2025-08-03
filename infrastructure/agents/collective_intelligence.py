"""
Продвинутый модуль коллективного интеллекта для торговых агентов
Включает Multi-Agent Systems, Swarm Intelligence, Consensus Mechanisms
"""

import asyncio
import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

# Импорт из существующих модулей
try:
    from infrastructure.ml_services.neuro_evolution import NEATGenome, NEATEvolution
    from infrastructure.ml_services.advanced_neural_networks import AdvancedTransformer, NeuralNetworkConfig
    NEURO_AVAILABLE = True
except ImportError:
    NEURO_AVAILABLE = False
    logger.warning("Neuro modules not available, using simplified agents")


class AgentRole(Enum):
    """Роли агентов в коллективной системе"""
    SCOUT = "scout"              # Разведчик - ищет возможности
    HUNTER = "hunter"            # Охотник - реализует сделки
    GUARDIAN = "guardian"        # Страж - управляет рисками
    COORDINATOR = "coordinator"  # Координатор - управляет группой
    ANALYST = "analyst"          # Аналитик - анализирует данные
    MESSENGER = "messenger"      # Посланник - передает информацию
    LEADER = "leader"           # Лидер - принимает решения


class CommunicationProtocol(Enum):
    """Протоколы коммуникации между агентами"""
    BROADCAST = "broadcast"      # Широковещание
    UNICAST = "unicast"         # Точка-точка
    MULTICAST = "multicast"     # Группа
    GOSSIP = "gossip"           # Сплетни
    CONSENSUS = "consensus"     # Консенсус


@dataclass
class Message:
    """Сообщение между агентами"""
    id: str = field(default_factory=lambda: str(uuid4()))
    sender_id: str = ""
    receiver_id: Optional[str] = None  # None для broadcast
    message_type: str = "info"
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1  # 1-низкий, 5-критический
    ttl: int = 10  # Time to live
    protocol: CommunicationProtocol = CommunicationProtocol.BROADCAST
    
    def is_expired(self) -> bool:
        """Проверяет, истек ли срок жизни сообщения"""
        return (datetime.now() - self.timestamp).seconds > self.ttl * 60


@dataclass
class AgentState:
    """Состояние агента"""
    energy: float = 100.0          # Энергия агента
    health: float = 100.0          # Здоровье агента
    experience: float = 0.0        # Опыт агента
    reputation: float = 50.0       # Репутация среди других агентов
    stress_level: float = 0.0      # Уровень стресса
    learning_rate: float = 0.01    # Скорость обучения
    confidence: float = 0.5        # Уверенность в решениях
    social_connections: int = 0    # Количество социальных связей


@dataclass
class SwarmConfig:
    """Конфигурация роя"""
    population_size: int = 50
    max_velocity: float = 2.0
    inertia_weight: float = 0.7
    cognitive_parameter: float = 1.5
    social_parameter: float = 1.5
    neighborhood_size: int = 5
    communication_range: float = 10.0
    energy_decay_rate: float = 0.01
    mutation_rate: float = 0.05
    crossover_rate: float = 0.3


class BaseAgent(ABC):
    """Базовый класс для всех агентов"""
    
    def __init__(self, agent_id: str, role: AgentRole, position: Optional[np.ndarray] = None):
        self.id = agent_id
        self.role = role
        self.position = position if position is not None else np.random.random(10)  # 10D пространство
        self.velocity = np.zeros_like(self.position)
        
        # Состояние агента
        self.state = AgentState()
        
        # Коммуникация
        self.message_queue: List[Message] = []
        self.neighbors: Set[str] = set()
        self.blacklist: Set[str] = set()
        
        # Обучение и память
        self.memory: Dict[str, Any] = {}
        self.local_best_position = self.position.copy()
        self.local_best_fitness = float('-inf')
        
        # Neural network для принятия решений
        if NEURO_AVAILABLE:
            config = NeuralNetworkConfig(input_dim=20, hidden_dim=64, output_dim=5)
            self.brain = AdvancedTransformer(config)
        else:
            self.brain = None
            
        # Эволюционные компоненты
        self.genome = None
        if NEURO_AVAILABLE:
            self.genome = NEATGenome(20, 5)  # 20 входов, 5 выходов
            
        logger.info(f"Agent {self.id} created with role {self.role.value}")
    
    @abstractmethod
    async def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Воспринимает окружающую среду"""
        pass
    
    @abstractmethod
    async def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Принимает решение на основе восприятия"""
        pass
    
    @abstractmethod
    async def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Выполняет действие"""
        pass
    
    async def communicate(self, message: Message, swarm: 'Swarm'):
        """Отправляет сообщение другим агентам"""
        try:
            if message.protocol == CommunicationProtocol.BROADCAST:
                for agent_id in swarm.agents:
                    if agent_id != self.id:
                        await swarm.deliver_message(message, agent_id)
                        
            elif message.protocol == CommunicationProtocol.UNICAST:
                if message.receiver_id:
                    await swarm.deliver_message(message, message.receiver_id)
                    
            elif message.protocol == CommunicationProtocol.MULTICAST:
                for neighbor_id in self.neighbors:
                    await swarm.deliver_message(message, neighbor_id)
                    
            elif message.protocol == CommunicationProtocol.GOSSIP:
                # Случайная передача сообщения подмножеству агентов
                gossip_targets = random.sample(
                    list(swarm.agents.keys()), 
                    min(3, len(swarm.agents) - 1)
                )
                for target_id in gossip_targets:
                    if target_id != self.id:
                        await swarm.deliver_message(message, target_id)
                        
        except Exception as e:
            logger.error(f"Communication error for agent {self.id}: {e}")
    
    async def receive_message(self, message: Message):
        """Получает сообщение от другого агента"""
        if not message.is_expired() and message.sender_id not in self.blacklist:
            self.message_queue.append(message)
            await self._process_message(message)
    
    async def _process_message(self, message: Message):
        """Обрабатывает полученное сообщение"""
        try:
            if message.message_type == "market_signal":
                await self._handle_market_signal(message)
            elif message.message_type == "collaboration_request":
                await self._handle_collaboration_request(message)
            elif message.message_type == "reputation_update":
                await self._handle_reputation_update(message)
            elif message.message_type == "knowledge_share":
                await self._handle_knowledge_share(message)
                
        except Exception as e:
            logger.error(f"Message processing error for agent {self.id}: {e}")
    
    async def _handle_market_signal(self, message: Message):
        """Обрабатывает рыночный сигнал"""
        signal_data = message.content.get('signal', {})
        confidence = signal_data.get('confidence', 0.0)
        
        # Обновляем локальное понимание рынка
        self.memory['last_market_signal'] = signal_data
        
        # Повышаем репутацию отправителя если сигнал качественный
        if confidence > 0.8:
            self._update_peer_reputation(message.sender_id, 5)
    
    async def _handle_collaboration_request(self, message: Message):
        """Обрабатывает запрос на сотрудничество"""
        request_type = message.content.get('type', '')
        
        if request_type == "joint_analysis":
            # Решаем, принять ли предложение о совместном анализе
            if self.state.energy > 50 and message.sender_id not in self.blacklist:
                response = Message(
                    sender_id=self.id,
                    receiver_id=message.sender_id,
                    message_type="collaboration_response",
                    content={"accepted": True, "capabilities": self._get_capabilities()},
                    protocol=CommunicationProtocol.UNICAST
                )
                # Отправляем ответ через swarm (потребуется передать ссылку)
    
    async def _handle_reputation_update(self, message: Message):
        """Обрабатывает обновление репутации"""
        reputation_change = message.content.get('change', 0)
        self.state.reputation = max(0, min(100, self.state.reputation + reputation_change))
    
    async def _handle_knowledge_share(self, message: Message):
        """Обрабатывает обмен знаниями"""
        knowledge = message.content.get('knowledge', {})
        
        # Интегрируем новые знания в память
        for key, value in knowledge.items():
            if key not in self.memory or self._is_better_knowledge(value, self.memory.get(key)):
                self.memory[key] = value
                
        # Благодарим за полезную информацию
        self._update_peer_reputation(message.sender_id, 2)
    
    def _update_peer_reputation(self, peer_id: str, change: int):
        """Обновляет репутацию другого агента"""
        if 'peer_reputations' not in self.memory:
            self.memory['peer_reputations'] = {}
        
        current_rep = self.memory['peer_reputations'].get(peer_id, 50)
        self.memory['peer_reputations'][peer_id] = max(0, min(100, current_rep + change))
    
    def _is_better_knowledge(self, new_knowledge: Any, old_knowledge: Any) -> bool:
        """Определяет, лучше ли новое знание"""
        if old_knowledge is None:
            return True
            
        # Простая эвристика - предпочитаем более свежую информацию
        if isinstance(new_knowledge, dict) and isinstance(old_knowledge, dict):
            new_time = new_knowledge.get('timestamp', 0)
            old_time = old_knowledge.get('timestamp', 0)
            return new_time > old_time
            
        return False
    
    def _get_capabilities(self) -> Dict[str, Any]:
        """Возвращает способности агента"""
        return {
            "role": self.role.value,
            "experience": self.state.experience,
            "reputation": self.state.reputation,
            "specializations": getattr(self, 'specializations', [])
        }
    
    def update_fitness(self, fitness: float):
        """Обновляет фитнес агента"""
        if fitness > self.local_best_fitness:
            self.local_best_fitness = fitness
            self.local_best_position = self.position.copy()
            self.state.experience += 1
            self.state.confidence = min(1.0, self.state.confidence + 0.01)
    
    async def evolve(self, global_best_position: np.ndarray, config: SwarmConfig):
        """Эволюционирует агента (PSO + GA)"""
        # Particle Swarm Optimization
        await self._update_velocity(global_best_position, config)
        self._update_position(config)
        
        # Genetic Algorithm mutations
        if random.random() < config.mutation_rate:
            self._mutate()
            
        # Energy decay
        self.state.energy = max(0, self.state.energy - config.energy_decay_rate)
        
        # Stress management
        self._manage_stress()
    
    async def _update_velocity(self, global_best_position: np.ndarray, config: SwarmConfig):
        """Обновляет скорость агента (PSO)"""
        r1, r2 = random.random(), random.random()
        
        cognitive_component = (config.cognitive_parameter * r1 * 
                             (self.local_best_position - self.position))
        social_component = (config.social_parameter * r2 * 
                          (global_best_position - self.position))
        
        self.velocity = (config.inertia_weight * self.velocity + 
                        cognitive_component + social_component)
        
        # Ограничиваем скорость
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > config.max_velocity:
            self.velocity = (self.velocity / velocity_magnitude) * config.max_velocity
    
    def _update_position(self, config: SwarmConfig):
        """Обновляет позицию агента"""
        self.position += self.velocity
        
        # Ограничиваем позицию границами пространства
        self.position = np.clip(self.position, -10, 10)
    
    def _mutate(self):
        """Мутирует агента"""
        mutation_strength = 0.1
        mutation_vector = np.random.normal(0, mutation_strength, self.position.shape)
        self.position += mutation_vector
        
        # Мутируем параметры состояния
        if random.random() < 0.1:
            self.state.learning_rate *= random.uniform(0.9, 1.1)
            self.state.learning_rate = max(0.001, min(0.1, self.state.learning_rate))
    
    def _manage_stress(self):
        """Управляет стрессом агента"""
        # Стресс влияет на производительность
        if self.state.stress_level > 0.7:
            self.state.learning_rate *= 0.95
            self.state.confidence *= 0.98
            
        # Восстановление от стресса
        if self.state.energy > 80:
            self.state.stress_level = max(0, self.state.stress_level - 0.05)


class ScoutAgent(BaseAgent):
    """Агент-разведчик, ищет новые возможности"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, AgentRole.SCOUT)
        self.specializations = ["pattern_recognition", "opportunity_detection", "market_scanning"]
        self.scan_radius = 5.0
        self.discovery_threshold = 0.7
    
    async def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Сканирует среду в поисках возможностей"""
        market_data = environment.get('market_data', {})
        price_movements = environment.get('price_movements', [])
        
        perception = {
            'opportunities': [],
            'threats': [],
            'patterns': [],
            'scan_quality': self.state.confidence
        }
        
        # Анализируем ценовые движения
        if price_movements:
            patterns = self._detect_patterns(price_movements)
            perception['patterns'] = patterns
            
            # Ищем возможности
            for pattern in patterns:
                if pattern['strength'] > self.discovery_threshold:
                    opportunity = {
                        'type': 'price_pattern',
                        'pattern': pattern,
                        'confidence': pattern['strength'],
                        'discovered_by': self.id,
                        'timestamp': datetime.now()
                    }
                    perception['opportunities'].append(opportunity)
        
        return perception
    
    async def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Решает, что делать с найденными возможностями"""
        opportunities = perception.get('opportunities', [])
        
        decision = {
            'action': 'continue_scouting',
            'share_findings': False,
            'opportunities_to_share': [],
            'next_scan_area': None
        }
        
        if opportunities:
            # Фильтруем лучшие возможности для публикации
            good_opportunities = [
                opp for opp in opportunities 
                if opp['confidence'] > 0.8
            ]
            
            if good_opportunities:
                decision['share_findings'] = True
                decision['opportunities_to_share'] = good_opportunities
                decision['action'] = 'share_intelligence'
        
        # Определяем следующую область для сканирования
        decision['next_scan_area'] = self._calculate_next_scan_area()
        
        return decision
    
    async def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Выполняет действие разведчика"""
        result = {'success': True, 'messages_sent': 0}
        
        if decision.get('share_findings', False):
            opportunities = decision.get('opportunities_to_share', [])
            
            for opportunity in opportunities:
                message = Message(
                    sender_id=self.id,
                    message_type="market_signal",
                    content={
                        'signal': opportunity,
                        'source': 'scout_discovery',
                        'reliability': self.state.reputation / 100
                    },
                    priority=3,
                    protocol=CommunicationProtocol.BROADCAST
                )
                
                # Сообщение будет отправлено через swarm
                result['messages_sent'] += 1
        
        # Обновляем состояние после работы
        self.state.energy -= 5
        self.state.experience += 0.1
        
        return result
    
    def _detect_patterns(self, price_movements: List[float]) -> List[Dict[str, Any]]:
        """Обнаруживает паттерны в ценовых движениях"""
        patterns = []
        
        if len(price_movements) < 10:
            return patterns
        
        # Простое обнаружение трендов
        recent_moves = price_movements[-10:]
        trend_strength = self._calculate_trend_strength(recent_moves)
        
        if abs(trend_strength) > 0.3:
            pattern = {
                'type': 'trend',
                'direction': 'up' if trend_strength > 0 else 'down',
                'strength': abs(trend_strength),
                'duration': len(recent_moves),
                'confidence': min(abs(trend_strength) * 2, 1.0)
            }
            patterns.append(pattern)
        
        # Обнаружение разворотов
        reversal_signal = self._detect_reversal(price_movements)
        if reversal_signal:
            patterns.append(reversal_signal)
        
        return patterns
    
    def _calculate_trend_strength(self, movements: List[float]) -> float:
        """Вычисляет силу тренда"""
        if len(movements) < 2:
            return 0.0
            
        changes = [movements[i] - movements[i-1] for i in range(1, len(movements))]
        positive_changes = sum(1 for c in changes if c > 0)
        negative_changes = sum(1 for c in changes if c < 0)
        
        if len(changes) == 0:
            return 0.0
            
        trend_ratio = (positive_changes - negative_changes) / len(changes)
        return trend_ratio
    
    def _detect_reversal(self, movements: List[float]) -> Optional[Dict[str, Any]]:
        """Обнаруживает разворотные паттерны"""
        if len(movements) < 5:
            return None
            
        # Простая логика обнаружения разворота
        recent = movements[-5:]
        if len(set(recent)) == 1:  # Все значения одинаковые
            return None
            
        # Ищем паттерн "высокий-низкий-высокий" или наоборот
        if len(recent) >= 3:
            if (recent[0] < recent[1] > recent[2] and 
                recent[1] - recent[0] > 0.01 and 
                recent[1] - recent[2] > 0.01):
                
                return {
                    'type': 'reversal',
                    'pattern': 'peak',
                    'strength': 0.7,
                    'confidence': 0.6
                }
        
        return None
    
    def _calculate_next_scan_area(self) -> np.ndarray:
        """Вычисляет следующую область для сканирования"""
        # Случайное направление с небольшим смещением от текущей позиции
        direction = np.random.normal(0, 1, self.position.shape)
        direction = direction / np.linalg.norm(direction)
        
        next_area = self.position + direction * self.scan_radius
        return np.clip(next_area, -10, 10)


class HunterAgent(BaseAgent):
    """Агент-охотник, реализует торговые возможности"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, AgentRole.HUNTER)
        self.specializations = ["trade_execution", "opportunity_capture", "fast_reaction"]
        self.aggression_level = 0.7
        self.execution_threshold = 0.6
    
    async def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Воспринимает торговые возможности"""
        market_signals = []
        
        # Ищем сигналы от разведчиков в очереди сообщений
        for message in self.message_queue:
            if (message.message_type == "market_signal" and 
                not message.is_expired()):
                signal = message.content.get('signal', {})
                signal['message_id'] = message.id
                signal['sender_reputation'] = self.memory.get('peer_reputations', {}).get(
                    message.sender_id, 50
                )
                market_signals.append(signal)
        
        # Очищаем обработанные сообщения
        self.message_queue = [m for m in self.message_queue if not m.is_expired()]
        
        perception = {
            'available_signals': market_signals,
            'market_conditions': environment.get('market_data', {}),
            'execution_readiness': self.state.energy / 100,
            'risk_appetite': self.aggression_level * (self.state.confidence ** 2)
        }
        
        return perception
    
    async def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Принимает решение о выполнении сделки"""
        signals = perception.get('available_signals', [])
        execution_readiness = perception.get('execution_readiness', 0)
        risk_appetite = perception.get('risk_appetite', 0)
        
        decision = {
            'action': 'wait',
            'selected_signal': None,
            'execution_plan': None,
            'risk_level': 'low'
        }
        
        if execution_readiness < 0.3:
            decision['action'] = 'rest'
            return decision
        
        # Оцениваем сигналы
        scored_signals = []
        for signal in signals:
            score = self._score_signal(signal, risk_appetite)
            if score > self.execution_threshold:
                scored_signals.append((signal, score))
        
        if scored_signals:
            # Выбираем лучший сигнал
            best_signal, best_score = max(scored_signals, key=lambda x: x[1])
            
            decision['action'] = 'execute_trade'
            decision['selected_signal'] = best_signal
            decision['execution_plan'] = self._create_execution_plan(best_signal, best_score)
            decision['risk_level'] = self._assess_risk_level(best_signal, best_score)
        
        return decision
    
    async def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Выполняет торговое действие"""
        result = {'success': False, 'trade_executed': False, 'details': {}}
        
        if decision['action'] == 'execute_trade':
            signal = decision['selected_signal']
            plan = decision['execution_plan']
            
            # Симулируем выполнение сделки
            trade_success = self._simulate_trade_execution(signal, plan)
            
            if trade_success:
                result['success'] = True
                result['trade_executed'] = True
                result['details'] = {
                    'signal_type': signal.get('type', 'unknown'),
                    'confidence': signal.get('confidence', 0),
                    'execution_quality': plan.get('quality', 0)
                }
                
                # Обновляем состояние после успешной сделки
                self.state.energy -= 15
                self.state.experience += 0.5
                self.state.confidence = min(1.0, self.state.confidence + 0.02)
                
                # Отправляем отчет о выполнении
                await self._send_execution_report(signal, plan, True)
            else:
                result['details']['error'] = 'execution_failed'
                self.state.stress_level = min(1.0, self.state.stress_level + 0.1)
                await self._send_execution_report(signal, plan, False)
                
        elif decision['action'] == 'rest':
            # Восстанавливаем энергию
            self.state.energy = min(100, self.state.energy + 10)
            result['success'] = True
            result['details']['action'] = 'resting'
        
        return result
    
    def _score_signal(self, signal: Dict[str, Any], risk_appetite: float) -> float:
        """Оценивает качество сигнала"""
        base_score = signal.get('confidence', 0)
        sender_reputation = signal.get('sender_reputation', 50) / 100
        
        # Учитываем репутацию отправителя
        reputation_bonus = (sender_reputation - 0.5) * 0.2
        
        # Учитываем тип сигнала
        signal_type = signal.get('type', 'unknown')
        type_multiplier = {
            'price_pattern': 1.0,
            'trend': 0.8,
            'reversal': 1.2,
            'arbitrage': 1.5,
            'unknown': 0.5
        }.get(signal_type, 0.5)
        
        # Учитываем склонность к риску
        risk_adjustment = 1.0 + (risk_appetite - 0.5) * 0.3
        
        final_score = (base_score + reputation_bonus) * type_multiplier * risk_adjustment
        return max(0, min(1, final_score))
    
    def _create_execution_plan(self, signal: Dict[str, Any], score: float) -> Dict[str, Any]:
        """Создает план выполнения сделки"""
        return {
            'entry_confidence': score,
            'position_size': min(score, 0.8),  # Размер позиции пропорционален уверенности
            'stop_loss': 0.02,  # 2% стоп-лосс
            'take_profit': 0.04,  # 4% тейк-профит
            'execution_style': 'aggressive' if score > 0.8 else 'conservative',
            'quality': score
        }
    
    def _assess_risk_level(self, signal: Dict[str, Any], score: float) -> str:
        """Оценивает уровень риска"""
        if score > 0.8:
            return 'high'
        elif score > 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _simulate_trade_execution(self, signal: Dict[str, Any], plan: Dict[str, Any]) -> bool:
        """Симулирует выполнение сделки"""
        # Простая симуляция с вероятностью успеха, зависящей от качества сигнала
        success_probability = plan['entry_confidence'] * (self.state.confidence ** 0.5)
        
        # Учитываем состояние агента
        if self.state.stress_level > 0.7:
            success_probability *= 0.8
        if self.state.energy < 30:
            success_probability *= 0.9
        
        return random.random() < success_probability
    
    async def _send_execution_report(self, signal: Dict[str, Any], plan: Dict[str, Any], success: bool):
        """Отправляет отчет о выполнении сделки"""
        report = Message(
            sender_id=self.id,
            message_type="execution_report",
            content={
                'signal_id': signal.get('message_id', ''),
                'success': success,
                'execution_details': plan,
                'agent_state': {
                    'energy': self.state.energy,
                    'confidence': self.state.confidence
                }
            },
            priority=2,
            protocol=CommunicationProtocol.BROADCAST
        )
        # Отчет будет отправлен через swarm


class GuardianAgent(BaseAgent):
    """Агент-страж, управляет рисками"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, AgentRole.GUARDIAN)
        self.specializations = ["risk_assessment", "threat_detection", "protection"]
        self.vigilance_level = 0.8
        self.risk_tolerance = 0.3
    
    async def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Анализирует риски в окружающей среде"""
        market_data = environment.get('market_data', {})
        
        # Анализируем сообщения от других агентов
        execution_reports = []
        signals = []
        
        for message in self.message_queue:
            if message.message_type == "execution_report":
                execution_reports.append(message.content)
            elif message.message_type == "market_signal":
                signals.append(message.content)
        
        perception = {
            'market_volatility': self._calculate_market_volatility(market_data),
            'agent_performance': self._analyze_agent_performance(execution_reports),
            'signal_quality': self._assess_signal_quality(signals),
            'system_stress': self._measure_system_stress(),
            'threat_level': 'low'
        }
        
        # Определяем общий уровень угрозы
        perception['threat_level'] = self._calculate_threat_level(perception)
        
        return perception
    
    async def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Принимает решения по управлению рисками"""
        threat_level = perception.get('threat_level', 'low')
        market_volatility = perception.get('market_volatility', 0)
        system_stress = perception.get('system_stress', 0)
        
        decision = {
            'action': 'monitor',
            'risk_alerts': [],
            'recommendations': [],
            'emergency_mode': False
        }
        
        # Проверяем критические условия
        if threat_level == 'critical' or market_volatility > 0.8:
            decision['action'] = 'emergency_response'
            decision['emergency_mode'] = True
            decision['risk_alerts'].append({
                'level': 'critical',
                'message': 'High market volatility detected',
                'recommended_action': 'reduce_positions'
            })
        
        elif threat_level == 'high' or system_stress > 0.7:
            decision['action'] = 'heightened_vigilance'
            decision['risk_alerts'].append({
                'level': 'warning',
                'message': 'Elevated risk conditions',
                'recommended_action': 'increase_monitoring'
            })
        
        # Генерируем рекомендации
        decision['recommendations'] = self._generate_risk_recommendations(perception)
        
        return decision
    
    async def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Выполняет действия по управлению рисками"""
        result = {'success': True, 'alerts_sent': 0, 'actions_taken': []}
        
        # Отправляем предупреждения о рисках
        risk_alerts = decision.get('risk_alerts', [])
        for alert in risk_alerts:
            alert_message = Message(
                sender_id=self.id,
                message_type="risk_alert",
                content=alert,
                priority=4 if alert['level'] == 'critical' else 3,
                protocol=CommunicationProtocol.BROADCAST
            )
            result['alerts_sent'] += 1
        
        # Выполняем рекомендованные действия
        recommendations = decision.get('recommendations', [])
        for rec in recommendations:
            if rec['type'] == 'adjust_parameters':
                result['actions_taken'].append('parameter_adjustment')
            elif rec['type'] == 'blacklist_agent':
                self._blacklist_underperforming_agent(rec['agent_id'])
                result['actions_taken'].append(f"blacklisted_{rec['agent_id']}")
        
        # Обновляем состояние
        if decision.get('emergency_mode', False):
            self.state.stress_level = min(1.0, self.state.stress_level + 0.2)
            self.state.energy -= 20
        else:
            self.state.energy -= 5
        
        self.state.experience += 0.1
        
        return result
    
    def _calculate_market_volatility(self, market_data: Dict[str, Any]) -> float:
        """Вычисляет волатильность рынка"""
        price_history = market_data.get('price_history', [])
        if len(price_history) < 10:
            return 0.0
        
        # Простой расчет волатильности
        returns = []
        for i in range(1, len(price_history)):
            ret = (price_history[i] - price_history[i-1]) / price_history[i-1]
            returns.append(ret)
        
        if returns:
            volatility = np.std(returns)
            return min(volatility * 10, 1.0)  # Нормализуем к [0,1]
        
        return 0.0
    
    def _analyze_agent_performance(self, execution_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализирует производительность агентов"""
        if not execution_reports:
            return {'success_rate': 0.5, 'agent_scores': {}}
        
        total_executions = len(execution_reports)
        successful_executions = sum(1 for r in execution_reports if r.get('success', False))
        success_rate = successful_executions / total_executions
        
        # Анализируем по агентам
        agent_performance = {}
        for report in execution_reports:
            agent_id = report.get('agent_id', 'unknown')
            if agent_id not in agent_performance:
                agent_performance[agent_id] = {'total': 0, 'success': 0}
            
            agent_performance[agent_id]['total'] += 1
            if report.get('success', False):
                agent_performance[agent_id]['success'] += 1
        
        agent_scores = {}
        for agent_id, stats in agent_performance.items():
            agent_scores[agent_id] = stats['success'] / stats['total']
        
        return {
            'success_rate': success_rate,
            'agent_scores': agent_scores,
            'total_executions': total_executions
        }
    
    def _assess_signal_quality(self, signals: List[Dict[str, Any]]) -> float:
        """Оценивает качество сигналов"""
        if not signals:
            return 0.5
        
        confidences = [s.get('confidence', 0) for s in signals]
        avg_confidence = np.mean(confidences)
        
        return avg_confidence
    
    def _measure_system_stress(self) -> float:
        """Измеряет стресс системы"""
        # Простая метрика стресса на основе различных факторов
        factors = [
            self.state.stress_level,
            1 - (self.state.energy / 100),
            len(self.message_queue) / 50,  # Перегруженность сообщениями
        ]
        
        return min(np.mean(factors), 1.0)
    
    def _calculate_threat_level(self, perception: Dict[str, Any]) -> str:
        """Вычисляет общий уровень угрозы"""
        volatility = perception.get('market_volatility', 0)
        system_stress = perception.get('system_stress', 0)
        performance = perception.get('agent_performance', {})
        success_rate = performance.get('success_rate', 0.5)
        
        # Агрегированная оценка угрозы
        threat_score = (volatility * 0.4 + 
                       system_stress * 0.3 + 
                       (1 - success_rate) * 0.3)
        
        if threat_score > 0.8:
            return 'critical'
        elif threat_score > 0.6:
            return 'high'
        elif threat_score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _generate_risk_recommendations(self, perception: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Генерирует рекомендации по управлению рисками"""
        recommendations = []
        
        # Анализируем производительность агентов
        agent_scores = perception.get('agent_performance', {}).get('agent_scores', {})
        for agent_id, score in agent_scores.items():
            if score < 0.3:  # Низкая производительность
                recommendations.append({
                    'type': 'blacklist_agent',
                    'agent_id': agent_id,
                    'reason': 'poor_performance',
                    'score': score
                })
        
        # Рекомендации по волатильности
        volatility = perception.get('market_volatility', 0)
        if volatility > 0.6:
            recommendations.append({
                'type': 'adjust_parameters',
                'parameter': 'risk_tolerance',
                'adjustment': 'decrease',
                'reason': 'high_volatility'
            })
        
        return recommendations
    
    def _blacklist_underperforming_agent(self, agent_id: str):
        """Добавляет агента в черный список"""
        self.blacklist.add(agent_id)
        logger.warning(f"Guardian {self.id} blacklisted agent {agent_id}")


class Swarm:
    """Рой агентов с коллективным интеллектом"""
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.agents: Dict[str, BaseAgent] = {}
        self.global_best_position = np.random.random(10)
        self.global_best_fitness = float('-inf')
        
        # Коммуникационная инфраструктура
        self.message_router = MessageRouter()
        self.consensus_manager = ConsensusManager()
        self.knowledge_base = CollectiveKnowledgeBase()
        
        # Статистика роя
        self.generation = 0
        self.performance_history = []
        
        logger.info(f"Swarm initialized with config: {config}")
    
    def add_agent(self, agent: BaseAgent):
        """Добавляет агента в рой"""
        self.agents[agent.id] = agent
        agent.neighbors = self._calculate_neighbors(agent)
        logger.info(f"Added agent {agent.id} with role {agent.role.value} to swarm")
    
    def remove_agent(self, agent_id: str):
        """Удаляет агента из роя"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            # Обновляем соседей для оставшихся агентов
            for agent in self.agents.values():
                agent.neighbors.discard(agent_id)
                agent.blacklist.discard(agent_id)
            logger.info(f"Removed agent {agent_id} from swarm")
    
    async def step(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Выполняет один шаг симуляции роя"""
        step_results = {
            'generation': self.generation,
            'agent_results': {},
            'swarm_metrics': {},
            'messages_processed': 0
        }
        
        # Фаза восприятия
        perceptions = {}
        for agent_id, agent in self.agents.items():
            try:
                perception = await agent.perceive(environment)
                perceptions[agent_id] = perception
            except Exception as e:
                logger.error(f"Perception error for agent {agent_id}: {e}")
                perceptions[agent_id] = {}
        
        # Фаза принятия решений
        decisions = {}
        for agent_id, agent in self.agents.items():
            try:
                decision = await agent.decide(perceptions[agent_id])
                decisions[agent_id] = decision
            except Exception as e:
                logger.error(f"Decision error for agent {agent_id}: {e}")
                decisions[agent_id] = {'action': 'wait'}
        
        # Фаза действий
        for agent_id, agent in self.agents.items():
            try:
                result = await agent.act(decisions[agent_id])
                step_results['agent_results'][agent_id] = result
            except Exception as e:
                logger.error(f"Action error for agent {agent_id}: {e}")
                step_results['agent_results'][agent_id] = {'success': False, 'error': str(e)}
        
        # Обработка сообщений
        step_results['messages_processed'] = await self.message_router.process_pending_messages()
        
        # Консенсус и обновление знаний
        await self._update_collective_knowledge()
        consensus_decisions = await self.consensus_manager.reach_consensus(self.agents)
        
        # Эволюция роя
        await self._evolve_swarm()
        
        # Обновление статистики
        swarm_metrics = self._calculate_swarm_metrics()
        step_results['swarm_metrics'] = swarm_metrics
        self.performance_history.append(swarm_metrics)
        
        self.generation += 1
        
        return step_results
    
    async def deliver_message(self, message: Message, receiver_id: str):
        """Доставляет сообщение агенту"""
        await self.message_router.route_message(message, receiver_id, self.agents)
    
    def _calculate_neighbors(self, agent: BaseAgent) -> Set[str]:
        """Вычисляет соседей агента"""
        neighbors = set()
        
        for other_id, other_agent in self.agents.items():
            if other_id != agent.id:
                distance = np.linalg.norm(agent.position - other_agent.position)
                if distance <= self.config.communication_range:
                    neighbors.add(other_id)
        
        # Ограничиваем размер соседства
        if len(neighbors) > self.config.neighborhood_size:
            neighbors = set(random.sample(list(neighbors), self.config.neighborhood_size))
        
        return neighbors
    
    async def _evolve_swarm(self):
        """Эволюционирует рой"""
        # Обновляем global best
        for agent in self.agents.values():
            if agent.local_best_fitness > self.global_best_fitness:
                self.global_best_fitness = agent.local_best_fitness
                self.global_best_position = agent.local_best_position.copy()
        
        # Эволюционируем каждого агента
        for agent in self.agents.values():
            await agent.evolve(self.global_best_position, self.config)
        
        # Обновляем топологию соседства
        for agent in self.agents.values():
            agent.neighbors = self._calculate_neighbors(agent)
    
    async def _update_collective_knowledge(self):
        """Обновляет коллективную базу знаний"""
        # Собираем знания от всех агентов
        collective_insights = {}
        
        for agent in self.agents.values():
            agent_knowledge = agent.memory.get('insights', {})
            for key, value in agent_knowledge.items():
                if key not in collective_insights:
                    collective_insights[key] = []
                collective_insights[key].append(value)
        
        # Агрегируем знания
        for key, values in collective_insights.items():
            aggregated_value = self.knowledge_base.aggregate_knowledge(key, values)
            self.knowledge_base.update_knowledge(key, aggregated_value)
    
    def _calculate_swarm_metrics(self) -> Dict[str, Any]:
        """Вычисляет метрики роя"""
        if not self.agents:
            return {}
        
        # Базовые метрики
        total_energy = sum(agent.state.energy for agent in self.agents.values())
        avg_energy = total_energy / len(self.agents)
        
        total_experience = sum(agent.state.experience for agent in self.agents.values())
        avg_experience = total_experience / len(self.agents)
        
        total_confidence = sum(agent.state.confidence for agent in self.agents.values())
        avg_confidence = total_confidence / len(self.agents)
        
        # Метрики разнообразия
        positions = np.array([agent.position for agent in self.agents.values()])
        diversity = np.mean(np.std(positions, axis=0))
        
        # Метрики коммуникации
        total_messages = sum(len(agent.message_queue) for agent in self.agents.values())
        total_connections = sum(len(agent.neighbors) for agent in self.agents.values())
        
        return {
            'population_size': len(self.agents),
            'avg_energy': avg_energy,
            'avg_experience': avg_experience,
            'avg_confidence': avg_confidence,
            'diversity': diversity,
            'global_best_fitness': self.global_best_fitness,
            'total_messages': total_messages,
            'total_connections': total_connections,
            'generation': self.generation
        }


class MessageRouter:
    """Маршрутизатор сообщений для агентов"""
    
    def __init__(self):
        self.pending_messages: List[Tuple[Message, str]] = []
        self.message_history: List[Message] = []
        self.delivery_stats = {'delivered': 0, 'failed': 0}
    
    async def route_message(self, message: Message, receiver_id: str, agents: Dict[str, BaseAgent]):
        """Маршрутизирует сообщение к получателю"""
        self.pending_messages.append((message, receiver_id))
    
    async def process_pending_messages(self) -> int:
        """Обрабатывает все ожидающие сообщения"""
        processed_count = 0
        
        while self.pending_messages:
            message, receiver_id = self.pending_messages.pop(0)
            
            if not message.is_expired():
                # Здесь должна быть логика доставки к реальным агентам
                processed_count += 1
                self.delivery_stats['delivered'] += 1
                self.message_history.append(message)
            else:
                self.delivery_stats['failed'] += 1
        
        return processed_count


class ConsensusManager:
    """Менеджер консенсуса для коллективного принятия решений"""
    
    def __init__(self):
        self.consensus_history = []
        self.voting_threshold = 0.6
    
    async def reach_consensus(self, agents: Dict[str, BaseAgent]) -> Dict[str, Any]:
        """Достигает консенсуса среди агентов"""
        if not agents:
            return {}
        
        # Простой механизм консенсуса на основе голосования
        proposals = self._collect_proposals(agents)
        votes = await self._conduct_voting(proposals, agents)
        consensus = self._determine_consensus(votes)
        
        self.consensus_history.append({
            'timestamp': datetime.now(),
            'proposals': proposals,
            'votes': votes,
            'consensus': consensus
        })
        
        return consensus
    
    def _collect_proposals(self, agents: Dict[str, BaseAgent]) -> List[Dict[str, Any]]:
        """Собирает предложения от агентов"""
        proposals = []
        
        for agent in agents.values():
            if hasattr(agent, 'make_proposal'):
                proposal = agent.make_proposal()
                if proposal:
                    proposals.append({
                        'agent_id': agent.id,
                        'proposal': proposal,
                        'weight': agent.state.reputation / 100
                    })
        
        return proposals
    
    async def _conduct_voting(self, proposals: List[Dict[str, Any]], 
                            agents: Dict[str, BaseAgent]) -> Dict[str, List[float]]:
        """Проводит голосование по предложениям"""
        votes = {i: [] for i in range(len(proposals))}
        
        for agent in agents.values():
            agent_votes = self._agent_vote(agent, proposals)
            for i, vote in enumerate(agent_votes):
                votes[i].append(vote)
        
        return votes
    
    def _agent_vote(self, agent: BaseAgent, proposals: List[Dict[str, Any]]) -> List[float]:
        """Агент голосует за предложения"""
        votes = []
        
        for proposal in proposals:
            # Простая логика голосования на основе репутации предлагающего
            proposer_reputation = proposal.get('weight', 0.5)
            agent_confidence = agent.state.confidence
            
            # Агенты с высокой уверенностью более критичны
            vote = proposer_reputation * (1.0 - agent_confidence * 0.3)
            votes.append(max(0, min(1, vote)))
        
        return votes
    
    def _determine_consensus(self, votes: Dict[str, List[float]]) -> Dict[str, Any]:
        """Определяет консенсус на основе голосов"""
        consensus = {'accepted_proposals': [], 'avg_support': 0.0}
        
        for proposal_id, proposal_votes in votes.items():
            if proposal_votes:
                avg_support = np.mean(proposal_votes)
                
                if avg_support >= self.voting_threshold:
                    consensus['accepted_proposals'].append({
                        'id': proposal_id,
                        'support': avg_support
                    })
        
        if consensus['accepted_proposals']:
            all_support = [p['support'] for p in consensus['accepted_proposals']]
            consensus['avg_support'] = np.mean(all_support)
        
        return consensus


class CollectiveKnowledgeBase:
    """База коллективных знаний роя"""
    
    def __init__(self):
        self.knowledge: Dict[str, Any] = {}
        self.knowledge_history: List[Dict[str, Any]] = []
        self.update_count = 0
    
    def update_knowledge(self, key: str, value: Any):
        """Обновляет знания"""
        old_value = self.knowledge.get(key)
        self.knowledge[key] = value
        
        self.knowledge_history.append({
            'timestamp': datetime.now(),
            'key': key,
            'old_value': old_value,
            'new_value': value,
            'update_id': self.update_count
        })
        
        self.update_count += 1
    
    def get_knowledge(self, key: str) -> Any:
        """Получает знания"""
        return self.knowledge.get(key)
    
    def aggregate_knowledge(self, key: str, values: List[Any]) -> Any:
        """Агрегирует знания от нескольких источников"""
        if not values:
            return None
        
        # Простая агрегация - если все значения численные, берем среднее
        if all(isinstance(v, (int, float)) for v in values):
            return np.mean(values)
        
        # Для остальных типов - самое частое значение
        from collections import Counter
        return Counter(values).most_common(1)[0][0]


# Фабрика для создания роев
class SwarmFactory:
    """Фабрика для создания роев агентов"""
    
    @staticmethod
    def create_trading_swarm(population_size: int = 20) -> Swarm:
        """Создает торговый рой"""
        config = SwarmConfig(population_size=population_size)
        swarm = Swarm(config)
        
        # Создаем агентов разных ролей
        role_distribution = {
            AgentRole.SCOUT: max(1, population_size // 5),
            AgentRole.HUNTER: max(1, population_size // 3),
            AgentRole.GUARDIAN: max(1, population_size // 10),
            AgentRole.COORDINATOR: max(1, population_size // 15),
            AgentRole.ANALYST: max(1, population_size // 8)
        }
        
        agent_count = 0
        for role, count in role_distribution.items():
            for i in range(count):
                if agent_count >= population_size:
                    break
                    
                agent_id = f"{role.value}_{i}"
                
                if role == AgentRole.SCOUT:
                    agent = ScoutAgent(agent_id)
                elif role == AgentRole.HUNTER:
                    agent = HunterAgent(agent_id)
                elif role == AgentRole.GUARDIAN:
                    agent = GuardianAgent(agent_id)
                else:
                    # Для остальных ролей используем базовый агент
                    agent = ScoutAgent(agent_id)  # Временно
                    agent.role = role
                
                swarm.add_agent(agent)
                agent_count += 1
        
        logger.info(f"Created trading swarm with {len(swarm.agents)} agents")
        return swarm
    
    @staticmethod
    def create_research_swarm(population_size: int = 15) -> Swarm:
        """Создает исследовательский рой"""
        config = SwarmConfig(
            population_size=population_size,
            mutation_rate=0.1,  # Больше мутаций для исследований
            communication_range=15.0  # Больше коммуникаций
        )
        swarm = Swarm(config)
        
        # Больше скаутов для исследований
        for i in range(population_size):
            agent = ScoutAgent(f"researcher_{i}")
            swarm.add_agent(agent)
        
        return swarm


# Экспорт основных классов
__all__ = [
    'AgentRole', 'CommunicationProtocol', 'Message', 'AgentState', 'SwarmConfig',
    'BaseAgent', 'ScoutAgent', 'HunterAgent', 'GuardianAgent',
    'Swarm', 'MessageRouter', 'ConsensusManager', 'CollectiveKnowledgeBase',
    'SwarmFactory'
]