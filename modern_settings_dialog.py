#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modern Settings Dialog for ATB Trading System
Современный диалог настроек с выбором торговых пар
"""

from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import json
from modern_style_system import *

class ModernSettingsDialog(QDialog):
    """Современный диалог настроек"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_pairs = set()
        self.init_ui()
        self.load_crypto_pairs()
    
    def init_ui(self):
        """Инициализация интерфейса"""
        self.setWindowTitle("⚙️ ATB Settings")
        self.setModal(True)
        self.resize(900, 700)
        
        # Основной layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Заголовок
        header = self.create_header()
        layout.addWidget(header)
        
        # Основной контент
        content = self.create_content()
        layout.addWidget(content)
        
        # Кнопки
        buttons = self.create_buttons()
        layout.addWidget(buttons)
    
    def create_header(self):
        """Создание заголовка"""
        header_widget = QWidget()
        header_widget.setFixedHeight(80)
        header_widget.setStyleSheet(f"""
            background: {ModernStyleSystem.COLORS['gradient_primary']};
            border-bottom: 1px solid {ModernStyleSystem.COLORS['surface_secondary']};
        """)
        
        layout = QHBoxLayout(header_widget)
        layout.setContentsMargins(24, 16, 24, 16)
        
        # Иконка и заголовок
        icon_label = QLabel("⚙️")
        icon_label.setStyleSheet("font-size: 32px; margin-right: 16px;")
        layout.addWidget(icon_label)
        
        title_label = QLabel("ATB Trading System Settings")
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: 700;
            color: white;
        """)
        layout.addWidget(title_label)
        
        layout.addStretch()
        return header_widget
    
    def create_content(self):
        """Создание основного контента"""
        content_widget = QWidget()
        layout = QHBoxLayout(content_widget)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(24)
        
        # Левая панель - Основные настройки
        left_panel = self.create_left_panel()
        layout.addWidget(left_panel, 1)
        
        # Правая панель - Выбор торговых пар
        right_panel = self.create_right_panel()
        layout.addWidget(right_panel, 2)
        
        return content_widget
    
    def create_left_panel(self):
        """Создание левой панели с основными настройками"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(20)
        
        # Заголовок
        title = ModernLabel("🔧 General Settings", "subtitle")
        layout.addWidget(title)
        
        # Настройки торговли
        trading_settings = self.create_trading_settings()
        layout.addWidget(trading_settings)
        
        # Настройки риска
        risk_settings = self.create_risk_settings()
        layout.addWidget(risk_settings)
        
        # Настройки подключения
        connection_settings = self.create_connection_settings()
        layout.addWidget(connection_settings)
        
        layout.addStretch()
        return panel
    
    def create_trading_settings(self):
        """Создание настроек торговли"""
        card = GlassCard("📈 Trading Configuration")
        layout = QFormLayout(card)
        layout.setSpacing(16)
        
        # Режим торговли
        self.trading_mode = ModernComboBox()
        self.trading_mode.addItems(["Simulation", "Paper Trading", "Live Trading"])
        layout.addRow("Trading Mode:", self.trading_mode)
        
        # Размер позиции
        self.position_size = ModernSpinBox()
        self.position_size.setRange(1, 100)
        self.position_size.setValue(10)
        self.position_size.setSuffix("%")
        layout.addRow("Position Size:", self.position_size)
        
        # Максимальное количество позиций
        self.max_positions = ModernSpinBox()
        self.max_positions.setRange(1, 20)
        self.max_positions.setValue(5)
        layout.addRow("Max Positions:", self.max_positions)
        
        # Комиссия
        self.commission = ModernSpinBox()
        self.commission.setRange(0, 1000)
        self.commission.setValue(10)
        self.commission.setSuffix(" bps")
        layout.addRow("Commission:", self.commission)
        
        return card
    
    def create_risk_settings(self):
        """Создание настроек риска"""
        card = GlassCard("⚠️ Risk Management")
        layout = QFormLayout(card)
        layout.setSpacing(16)
        
        # Максимальный убыток за день
        self.max_daily_loss = ModernSpinBox()
        self.max_daily_loss.setRange(1, 100)
        self.max_daily_loss.setValue(5)
        self.max_daily_loss.setSuffix("%")
        layout.addRow("Max Daily Loss:", self.max_daily_loss)
        
        # Stop Loss
        self.stop_loss = ModernSpinBox()
        self.stop_loss.setRange(1, 50)
        self.stop_loss.setValue(10)
        self.stop_loss.setSuffix("%")
        layout.addRow("Stop Loss:", self.stop_loss)
        
        # Take Profit
        self.take_profit = ModernSpinBox()
        self.take_profit.setRange(1, 100)
        self.take_profit.setValue(20)
        self.take_profit.setSuffix("%")
        layout.addRow("Take Profit:", self.take_profit)
        
        # Максимальный убыток на сделку
        self.max_trade_loss = ModernSpinBox()
        self.max_trade_loss.setRange(1, 20)
        self.max_trade_loss.setValue(2)
        self.max_trade_loss.setSuffix("%")
        layout.addRow("Max Trade Loss:", self.max_trade_loss)
        
        return card
    
    def create_connection_settings(self):
        """Создание настроек подключения"""
        card = GlassCard("🔗 Connection Settings")
        layout = QFormLayout(card)
        layout.setSpacing(16)
        
        # API Key
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("Enter your API key")
        layout.addRow("API Key:", self.api_key)
        
        # API Secret
        self.api_secret = QLineEdit()
        self.api_secret.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_secret.setPlaceholderText("Enter your API secret")
        layout.addRow("API Secret:", self.api_secret)
        
        # Exchange
        self.exchange = ModernComboBox()
        self.exchange.addItems(["Binance", "Bybit", "OKX", "KuCoin", "Gate.io"])
        layout.addRow("Exchange:", self.exchange)
        
        # Testnet
        self.testnet = ModernCheckBox("Use Testnet")
        self.testnet.setChecked(True)
        layout.addRow("", self.testnet)
        
        return card
    
    def create_right_panel(self):
        """Создание правой панели с выбором торговых пар"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(20)
        
        # Заголовок
        title = ModernLabel("🪙 Trading Pairs Selection", "subtitle")
        layout.addWidget(title)
        
        # Поиск
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Search cryptocurrencies...")
        self.search_input.textChanged.connect(self.filter_pairs)
        search_layout.addWidget(self.search_input)
        
        # Кнопки выбора
        select_buttons_layout = QHBoxLayout()
        
        self.select_all_btn = ModernButton("Select All", style="primary")
        self.select_all_btn.clicked.connect(self.select_all_pairs)
        select_buttons_layout.addWidget(self.select_all_btn)
        
        self.deselect_all_btn = ModernButton("Deselect All", style="warning")
        self.deselect_all_btn.clicked.connect(self.deselect_all_pairs)
        select_buttons_layout.addWidget(self.deselect_all_btn)
        
        self.select_top_10_btn = ModernButton("Top 10", style="info")
        self.select_top_10_btn.clicked.connect(self.select_top_10)
        select_buttons_layout.addWidget(self.select_top_10_btn)
        
        self.select_top_50_btn = ModernButton("Top 50", style="info")
        self.select_top_50_btn.clicked.connect(self.select_top_50)
        select_buttons_layout.addWidget(self.select_top_50_btn)
        
        search_layout.addLayout(select_buttons_layout)
        layout.addLayout(search_layout)
        
        # Список торговых пар
        pairs_container = self.create_pairs_list()
        layout.addWidget(pairs_container)
        
        # Статистика выбора
        stats = self.create_selection_stats()
        layout.addWidget(stats)
        
        return panel
    
    def create_pairs_list(self):
        """Создание списка торговых пар"""
        card = GlassCard("📊 Available Pairs")
        layout = QVBoxLayout(card)
        
        # Создаем scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Контейнер для чекбоксов
        self.pairs_container = QWidget()
        self.pairs_layout = QGridLayout(self.pairs_container)
        self.pairs_layout.setSpacing(8)
        
        scroll_area.setWidget(self.pairs_container)
        layout.addWidget(scroll_area)
        
        return card
    
    def create_selection_stats(self):
        """Создание статистики выбора"""
        card = GlassCard("📈 Selection Statistics")
        layout = QHBoxLayout(card)
        
        # Выбрано пар
        self.selected_count_label = ModernLabel("Selected: 0", "body")
        layout.addWidget(self.selected_count_label)
        
        layout.addStretch()
        
        # Категории
        self.categories_label = ModernLabel("Categories: All", "body")
        layout.addWidget(self.categories_label)
        
        return card
    
    def create_buttons(self):
        """Создание кнопок"""
        button_widget = QWidget()
        button_widget.setFixedHeight(80)
        button_widget.setStyleSheet(f"""
            background: {ModernStyleSystem.COLORS['surface']};
            border-top: 1px solid {ModernStyleSystem.COLORS['surface_secondary']};
        """)
        
        layout = QHBoxLayout(button_widget)
        layout.setContentsMargins(24, 16, 24, 16)
        
        layout.addStretch()
        
        # Кнопка отмены
        cancel_btn = ModernButton("Cancel", style="warning")
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(cancel_btn)
        
        # Кнопка сохранения
        save_btn = ModernButton("Save Settings", style="success")
        save_btn.clicked.connect(self.accept)
        layout.addWidget(save_btn)
        
        return button_widget
    
    def load_crypto_pairs(self):
        """Загрузка списка криптовалют"""
        # Топ 200 криптовалют по капитализации
        self.crypto_pairs = [
            {"symbol": "BTC", "name": "Bitcoin", "category": "Layer 1"},
            {"symbol": "ETH", "name": "Ethereum", "category": "Layer 1"},
            {"symbol": "BNB", "name": "BNB", "category": "Exchange"},
            {"symbol": "SOL", "name": "Solana", "category": "Layer 1"},
            {"symbol": "XRP", "name": "XRP", "category": "Payment"},
            {"symbol": "USDC", "name": "USD Coin", "category": "Stablecoin"},
            {"symbol": "USDT", "name": "Tether", "category": "Stablecoin"},
            {"symbol": "ADA", "name": "Cardano", "category": "Layer 1"},
            {"symbol": "AVAX", "name": "Avalanche", "category": "Layer 1"},
            {"symbol": "DOGE", "name": "Dogecoin", "category": "Meme"},
            {"symbol": "TRX", "name": "TRON", "category": "Layer 1"},
            {"symbol": "DOT", "name": "Polkadot", "category": "Layer 1"},
            {"symbol": "MATIC", "name": "Polygon", "category": "Layer 2"},
            {"symbol": "LINK", "name": "Chainlink", "category": "Oracle"},
            {"symbol": "TON", "name": "Toncoin", "category": "Layer 1"},
            {"symbol": "UNI", "name": "Uniswap", "category": "DeFi"},
            {"symbol": "BCH", "name": "Bitcoin Cash", "category": "Layer 1"},
            {"symbol": "LTC", "name": "Litecoin", "category": "Layer 1"},
            {"symbol": "ATOM", "name": "Cosmos", "category": "Layer 1"},
            {"symbol": "ETC", "name": "Ethereum Classic", "category": "Layer 1"},
            {"symbol": "XLM", "name": "Stellar", "category": "Payment"},
            {"symbol": "FIL", "name": "Filecoin", "category": "Storage"},
            {"symbol": "APT", "name": "Aptos", "category": "Layer 1"},
            {"symbol": "HBAR", "name": "Hedera", "category": "Layer 1"},
            {"symbol": "CRO", "name": "Cronos", "category": "Exchange"},
            {"symbol": "NEAR", "name": "NEAR Protocol", "category": "Layer 1"},
            {"symbol": "VET", "name": "VeChain", "category": "Supply Chain"},
            {"symbol": "OP", "name": "Optimism", "category": "Layer 2"},
            {"symbol": "ARB", "name": "Arbitrum", "category": "Layer 2"},
            {"symbol": "MKR", "name": "Maker", "category": "DeFi"},
            {"symbol": "ALGO", "name": "Algorand", "category": "Layer 1"},
            {"symbol": "ICP", "name": "Internet Computer", "category": "Layer 1"},
            {"symbol": "THETA", "name": "Theta Network", "category": "Video"},
            {"symbol": "IMX", "name": "Immutable", "category": "Gaming"},
            {"symbol": "XTZ", "name": "Tezos", "category": "Layer 1"},
            {"symbol": "AAVE", "name": "Aave", "category": "DeFi"},
            {"symbol": "EOS", "name": "EOS", "category": "Layer 1"},
            {"symbol": "FLOW", "name": "Flow", "category": "Gaming"},
            {"symbol": "SAND", "name": "The Sandbox", "category": "Gaming"},
            {"symbol": "MANA", "name": "Decentraland", "category": "Gaming"},
            {"symbol": "AXS", "name": "Axie Infinity", "category": "Gaming"},
            {"symbol": "GALA", "name": "Gala", "category": "Gaming"},
            {"symbol": "ENJ", "name": "Enjin Coin", "category": "Gaming"},
            {"symbol": "CHZ", "name": "Chiliz", "category": "Sports"},
            {"symbol": "HOT", "name": "Holo", "category": "Cloud"},
            {"symbol": "BAT", "name": "Basic Attention Token", "category": "Advertising"},
            {"symbol": "ZEC", "name": "Zcash", "category": "Privacy"},
            {"symbol": "DASH", "name": "Dash", "category": "Payment"},
            {"symbol": "NEO", "name": "Neo", "category": "Layer 1"},
            {"symbol": "WAVES", "name": "Waves", "category": "Layer 1"},
            {"symbol": "QTUM", "name": "Qtum", "category": "Layer 1"},
            {"symbol": "IOTA", "name": "IOTA", "category": "IoT"},
            {"symbol": "NANO", "name": "Nano", "category": "Payment"},
            {"symbol": "XMR", "name": "Monero", "category": "Privacy"},
            {"symbol": "ZIL", "name": "Zilliqa", "category": "Layer 1"},
            {"symbol": "ONE", "name": "Harmony", "category": "Layer 1"},
            {"symbol": "ICX", "name": "ICON", "category": "Layer 1"},
            {"symbol": "ONT", "name": "Ontology", "category": "Layer 1"},
            {"symbol": "VTHO", "name": "VeThor Token", "category": "Utility"},
            {"symbol": "ANKR", "name": "Ankr", "category": "Infrastructure"},
            {"symbol": "CKB", "name": "Nervos Network", "category": "Layer 1"},
            {"symbol": "COTI", "name": "COTI", "category": "Payment"},
            {"symbol": "SKL", "name": "Skale", "category": "Layer 2"},
            {"symbol": "RLC", "name": "iExec RLC", "category": "Cloud"},
            {"symbol": "STORJ", "name": "Storj", "category": "Storage"},
            {"symbol": "OCEAN", "name": "Ocean Protocol", "category": "Data"},
            {"symbol": "BAND", "name": "Band Protocol", "category": "Oracle"},
            {"symbol": "NEST", "name": "NEST Protocol", "category": "Oracle"},
            {"symbol": "DIA", "name": "DIA", "category": "Oracle"},
            {"symbol": "API3", "name": "API3", "category": "Oracle"},
            {"symbol": "PHA", "name": "Phala Network", "category": "Privacy"},
            {"symbol": "KEEP", "name": "Keep Network", "category": "Privacy"},
            {"symbol": "NU", "name": "NuCypher", "category": "Privacy"},
            {"symbol": "SCRT", "name": "Secret", "category": "Privacy"},
            {"symbol": "ROSE", "name": "Oasis Network", "category": "Privacy"},
            {"symbol": "MINA", "name": "Mina Protocol", "category": "Layer 1"},
            {"symbol": "KDA", "name": "Kadena", "category": "Layer 1"},
            {"symbol": "KAS", "name": "Kaspa", "category": "Layer 1"},
            {"symbol": "RUNE", "name": "THORChain", "category": "DeFi"},
            {"symbol": "1INCH", "name": "1inch", "category": "DeFi"},
            {"symbol": "SUSHI", "name": "SushiSwap", "category": "DeFi"},
            {"symbol": "CRV", "name": "Curve DAO Token", "category": "DeFi"},
            {"symbol": "COMP", "name": "Compound", "category": "DeFi"},
            {"symbol": "YFI", "name": "yearn.finance", "category": "DeFi"},
            {"symbol": "SNX", "name": "Synthetix", "category": "DeFi"},
            {"symbol": "BAL", "name": "Balancer", "category": "DeFi"},
            {"symbol": "KNC", "name": "Kyber Network", "category": "DeFi"},
            {"symbol": "ZRX", "name": "0x Protocol", "category": "DeFi"},
            {"symbol": "BICO", "name": "Biconomy", "category": "Infrastructure"},
            {"symbol": "LDO", "name": "Lido DAO", "category": "DeFi"},
            {"symbol": "RPL", "name": "Rocket Pool", "category": "DeFi"},
            {"symbol": "SWISE", "name": "StakeWise", "category": "DeFi"},
            {"symbol": "FRAX", "name": "Frax", "category": "Stablecoin"},
            {"symbol": "DAI", "name": "Dai", "category": "Stablecoin"},
            {"symbol": "BUSD", "name": "Binance USD", "category": "Stablecoin"},
            {"symbol": "TUSD", "name": "TrueUSD", "category": "Stablecoin"},
            {"symbol": "USDP", "name": "Pax Dollar", "category": "Stablecoin"},
            {"symbol": "GUSD", "name": "Gemini Dollar", "category": "Stablecoin"},
            {"symbol": "HUSD", "name": "HUSD", "category": "Stablecoin"},
            {"symbol": "USDN", "name": "Neutrino USD", "category": "Stablecoin"},
            {"symbol": "USDK", "name": "USDK", "category": "Stablecoin"},
            {"symbol": "USDJ", "name": "USDJ", "category": "Stablecoin"},
            {"symbol": "FEI", "name": "Fei USD", "category": "Stablecoin"},
            {"symbol": "RAI", "name": "Rai Reflex Index", "category": "Stablecoin"},
            {"symbol": "ALUSD", "name": "Alchemix USD", "category": "Stablecoin"},
            {"symbol": "MIM", "name": "Magic Internet Money", "category": "Stablecoin"},
            {"symbol": "LUSD", "name": "Liquity USD", "category": "Stablecoin"},
            {"symbol": "SUSD", "name": "sUSD", "category": "Stablecoin"},
            {"symbol": "USDK", "name": "USDK", "category": "Stablecoin"},
            {"symbol": "USDN", "name": "USDN", "category": "Stablecoin"},
            {"symbol": "USDJ", "name": "USDJ", "category": "Stablecoin"},
            {"symbol": "FEI", "name": "FEI", "category": "Stablecoin"},
            {"symbol": "RAI", "name": "RAI", "category": "Stablecoin"},
            {"symbol": "ALUSD", "name": "ALUSD", "category": "Stablecoin"},
            {"symbol": "MIM", "name": "MIM", "category": "Stablecoin"},
            {"symbol": "LUSD", "name": "LUSD", "category": "Stablecoin"},
            {"symbol": "SUSD", "name": "sUSD", "category": "Stablecoin"},
        ]
        
        self.populate_pairs_list()
    
    def populate_pairs_list(self):
        """Заполнение списка торговых пар"""
        # Очищаем существующие элементы
        for i in reversed(range(self.pairs_layout.count())):
            self.pairs_layout.itemAt(i).widget().setParent(None)
        
        # Добавляем пары
        row = 0
        col = 0
        max_cols = 3
        
        for pair in self.crypto_pairs:
            # Создаем чекбокс
            checkbox = ModernCheckBox(f"{pair['symbol']}/USDT - {pair['name']}")
            checkbox.setProperty("pair_symbol", pair['symbol'])
            checkbox.setProperty("pair_name", pair['name'])
            checkbox.setProperty("pair_category", pair['category'])
            checkbox.toggled.connect(self.on_pair_toggled)
            
            # Добавляем в сетку
            self.pairs_layout.addWidget(checkbox, row, col)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        self.update_selection_stats()
    
    def filter_pairs(self, search_text):
        """Фильтрация пар по поиску"""
        search_text = search_text.lower()
        
        for i in range(self.pairs_layout.count()):
            widget = self.pairs_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox):
                symbol = widget.property("pair_symbol").lower()
                name = widget.property("pair_name").lower()
                
                if search_text in symbol or search_text in name:
                    widget.setVisible(True)
                else:
                    widget.setVisible(False)
    
    def select_all_pairs(self):
        """Выбрать все пары"""
        for i in range(self.pairs_layout.count()):
            widget = self.pairs_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox) and widget.isVisible():
                widget.setChecked(True)
    
    def deselect_all_pairs(self):
        """Снять выбор со всех пар"""
        for i in range(self.pairs_layout.count()):
            widget = self.pairs_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox):
                widget.setChecked(False)
    
    def select_top_10(self):
        """Выбрать топ 10 пар"""
        self.deselect_all_pairs()
        top_10_symbols = ["BTC", "ETH", "BNB", "SOL", "XRP", "USDC", "USDT", "ADA", "AVAX", "DOGE"]
        
        for i in range(self.pairs_layout.count()):
            widget = self.pairs_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox):
                symbol = widget.property("pair_symbol")
                if symbol in top_10_symbols:
                    widget.setChecked(True)
    
    def select_top_50(self):
        """Выбрать топ 50 пар"""
        self.deselect_all_pairs()
        top_50_symbols = [
            "BTC", "ETH", "BNB", "SOL", "XRP", "USDC", "USDT", "ADA", "AVAX", "DOGE",
            "TRX", "DOT", "MATIC", "LINK", "TON", "UNI", "BCH", "LTC", "ATOM", "ETC",
            "XLM", "FIL", "APT", "HBAR", "CRO", "NEAR", "VET", "OP", "ARB", "MKR",
            "ALGO", "ICP", "THETA", "IMX", "XTZ", "AAVE", "EOS", "FLOW", "SAND", "MANA",
            "AXS", "GALA", "ENJ", "CHZ", "HOT", "BAT", "ZEC", "DASH", "NEO", "WAVES"
        ]
        
        for i in range(self.pairs_layout.count()):
            widget = self.pairs_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox):
                symbol = widget.property("pair_symbol")
                if symbol in top_50_symbols:
                    widget.setChecked(True)
    
    def on_pair_toggled(self, checked):
        """Обработка изменения состояния чекбокса"""
        checkbox = self.sender()
        symbol = checkbox.property("pair_symbol")
        
        if checked:
            self.selected_pairs.add(symbol)
        else:
            self.selected_pairs.discard(symbol)
        
        self.update_selection_stats()
    
    def update_selection_stats(self):
        """Обновление статистики выбора"""
        self.selected_count_label.setText(f"Selected: {len(self.selected_pairs)}")
        
        # Подсчитываем категории
        categories = set()
        for i in range(self.pairs_layout.count()):
            widget = self.pairs_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox) and widget.isChecked():
                category = widget.property("pair_category")
                categories.add(category)
        
        if len(categories) == 0:
            self.categories_label.setText("Categories: None")
        elif len(categories) <= 3:
            self.categories_label.setText(f"Categories: {', '.join(categories)}")
        else:
            self.categories_label.setText(f"Categories: {len(categories)} types")
    
    def get_settings(self):
        """Получить настройки"""
        return {
            'trading_mode': self.trading_mode.currentText(),
            'position_size': self.position_size.value(),
            'max_positions': self.max_positions.value(),
            'commission': self.commission.value(),
            'max_daily_loss': self.max_daily_loss.value(),
            'stop_loss': self.stop_loss.value(),
            'take_profit': self.take_profit.value(),
            'max_trade_loss': self.max_trade_loss.value(),
            'api_key': self.api_key.text(),
            'api_secret': self.api_secret.text(),
            'exchange': self.exchange.currentText(),
            'testnet': self.testnet.isChecked(),
            'selected_pairs': list(self.selected_pairs)
        }
    
    def set_settings(self, settings):
        """Установить настройки"""
        if 'trading_mode' in settings:
            index = self.trading_mode.findText(settings['trading_mode'])
            if index >= 0:
                self.trading_mode.setCurrentIndex(index)
        
        if 'position_size' in settings:
            self.position_size.setValue(settings['position_size'])
        
        if 'max_positions' in settings:
            self.max_positions.setValue(settings['max_positions'])
        
        if 'commission' in settings:
            self.commission.setValue(settings['commission'])
        
        if 'max_daily_loss' in settings:
            self.max_daily_loss.setValue(settings['max_daily_loss'])
        
        if 'stop_loss' in settings:
            self.stop_loss.setValue(settings['stop_loss'])
        
        if 'take_profit' in settings:
            self.take_profit.setValue(settings['take_profit'])
        
        if 'max_trade_loss' in settings:
            self.max_trade_loss.setValue(settings['max_trade_loss'])
        
        if 'api_key' in settings:
            self.api_key.setText(settings['api_key'])
        
        if 'api_secret' in settings:
            self.api_secret.setText(settings['api_secret'])
        
        if 'exchange' in settings:
            index = self.exchange.findText(settings['exchange'])
            if index >= 0:
                self.exchange.setCurrentIndex(index)
        
        if 'testnet' in settings:
            self.testnet.setChecked(settings['testnet'])
        
        if 'selected_pairs' in settings:
            self.selected_pairs = set(settings['selected_pairs'])
            # Обновляем чекбоксы
            for i in range(self.pairs_layout.count()):
                widget = self.pairs_layout.itemAt(i).widget()
                if isinstance(widget, QCheckBox):
                    symbol = widget.property("pair_symbol")
                    widget.setChecked(symbol in self.selected_pairs)
            
            self.update_selection_stats() 