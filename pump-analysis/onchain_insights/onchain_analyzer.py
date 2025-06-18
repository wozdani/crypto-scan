"""
On-chain Analyzer Module
Provides descriptive on-chain insights instead of rigid boolean conditions
"""

import os
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OnChainInsight:
    """Represents a single on-chain insight with descriptive text"""
    message: str
    confidence: float  # 0.0 to 1.0
    source: str  # etherscan, bscscan, dex_scanner, etc.
    timestamp: datetime
    category: str  # whale_activity, dex_inflow, bridge_activity, etc.

class OnChainAnalyzer:
    """
    Analyzes on-chain data and generates descriptive insights for GPT interpretation
    """
    
    def __init__(self):
        self.etherscan_api_key = os.environ.get('ETHERSCAN_API_KEY')
        self.bscscan_api_key = os.environ.get('BSCSCAN_API_KEY')
        self.arbiscan_api_key = os.environ.get('ARBISCAN_API_KEY')
        self.polygonscan_api_key = os.environ.get('POLYGONSCAN_API_KEY')
        self.optimismscan_api_key = os.environ.get('OPTIMISMSCAN_API_KEY')
        
        # Load known DEX addresses
        self.known_dex_addresses = self._load_known_dex_addresses()
        
        # Load contract mappings
        self.contract_cache = self._load_contract_cache()
        
    def _load_known_dex_addresses(self) -> Dict[str, List[str]]:
        """Load known DEX addresses from crypto-scan data"""
        try:
            # Try to load from crypto-scan first
            crypto_scan_path = os.path.join('..', 'crypto-scan', 'data', 'known_dex_addresses.py')
            if os.path.exists(crypto_scan_path):
                with open(crypto_scan_path, 'r') as f:
                    content = f.read()
                    # Extract DEX addresses from Python file
                    # This is a simplified extraction - in production would need proper parsing
                    return self._parse_dex_addresses_from_content(content)
        except Exception as e:
            logger.warning(f"Could not load DEX addresses from crypto-scan: {e}")
        
        # Fallback to basic known DEX addresses
        return {
            'ethereum': [
                '0x7a250d5630b4cf539739df2c5dacb4c659f2488d',  # Uniswap V2 Router
                '0xe592427a0aece92de3edee1f18e0157c05861564',  # Uniswap V3 Router
                '0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45',  # Uniswap V3 Router 2
                '0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f',  # SushiSwap Router
            ],
            'bsc': [
                '0x10ed43c718714eb63d5aa57b78b54704e256024e',  # PancakeSwap Router
                '0x05ff2b0db69458a0750badebc4f9e13add608c7f',  # PancakeSwap Router V1
            ],
            'polygon': [
                '0xa5e0829caced8ffdd4de3c43696c57f7d7a678ff',  # QuickSwap Router
                '0x1b02da8cb0d097eb8d57a175b88c7d8b47997506',  # SushiSwap Router
            ],
            'arbitrum': [
                '0x1b02da8cb0d097eb8d57a175b88c7d8b47997506',  # SushiSwap Router
                '0xe592427a0aece92de3edee1f18e0157c05861564',  # Uniswap V3 Router
            ],
            'optimism': [
                '0xe592427a0aece92de3edee1f18e0157c05861564',  # Uniswap V3 Router
                '0x1b02da8cb0d097eb8d57a175b88c7d8b47997506',  # SushiSwap Router
            ]
        }
    
    def _parse_dex_addresses_from_content(self, content: str) -> Dict[str, List[str]]:
        """Parse DEX addresses from Python file content"""
        # Simplified parsing - would need more robust implementation
        addresses = {}
        lines = content.split('\n')
        
        current_chain = None
        for line in lines:
            line = line.strip()
            if 'ethereum' in line.lower() and '=' in line:
                current_chain = 'ethereum'
                addresses[current_chain] = []
            elif 'bsc' in line.lower() and '=' in line:
                current_chain = 'bsc'
                addresses[current_chain] = []
            elif 'polygon' in line.lower() and '=' in line:
                current_chain = 'polygon'
                addresses[current_chain] = []
            elif '0x' in line and current_chain:
                # Extract address from line
                start = line.find('0x')
                if start != -1:
                    end = start + 42  # Standard Ethereum address length
                    address = line[start:end].lower()
                    if len(address) == 42:
                        addresses[current_chain].append(address)
        
        return addresses
    
    def _load_contract_cache(self) -> Dict[str, str]:
        """Load contract mappings from crypto-scan cache"""
        cache_paths = [
            os.path.join('..', 'crypto-scan', 'token_contract_map.json'),
            os.path.join('..', 'crypto-scan', 'cache', 'coingecko_cache.json'),
            'contract_cache.json'  # Local fallback
        ]
        
        for path in cache_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load contract cache from {path}: {e}")
        
        return {}
    
    def analyze_onchain_activity(self, symbol: str, timeframe_hours: int = 1) -> List[OnChainInsight]:
        """
        Analyze on-chain activity for a symbol and return descriptive insights
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe_hours: Hours to look back for analysis
            
        Returns:
            List of descriptive insights for GPT interpretation
        """
        insights = []
        base_symbol = symbol.replace('USDT', '').replace('BUSD', '').replace('USD', '')
        
        # Get contract address for the token
        contract_info = self._get_contract_info(base_symbol)
        if not contract_info:
            return insights
        
        contract_address = contract_info.get('address')
        chain = contract_info.get('chain', 'ethereum')
        
        if not contract_address:
            return insights
        
        # Analyze different types of on-chain activity
        insights.extend(self._analyze_whale_transactions(contract_address, chain, timeframe_hours))
        insights.extend(self._analyze_dex_activity(contract_address, chain, timeframe_hours))
        insights.extend(self._analyze_new_wallets(contract_address, chain, timeframe_hours))
        insights.extend(self._analyze_bridge_activity(contract_address, chain, timeframe_hours))
        insights.extend(self._analyze_approval_transactions(contract_address, chain, timeframe_hours))
        
        # Sort by confidence and timestamp
        insights.sort(key=lambda x: (x.confidence, x.timestamp), reverse=True)
        
        return insights[:10]  # Return top 10 insights
    
    def _get_contract_info(self, symbol: str) -> Optional[Dict[str, str]]:
        """Get contract address and chain for a symbol"""
        # First check local cache
        if symbol.lower() in self.contract_cache:
            contract_data = self.contract_cache[symbol.lower()]
            if isinstance(contract_data, dict):
                return contract_data
            elif isinstance(contract_data, str):
                return {'address': contract_data, 'chain': 'ethereum'}
        
        # Try different symbol variations
        variations = [symbol, symbol.upper(), symbol.lower()]
        for var in variations:
            if var in self.contract_cache:
                contract_data = self.contract_cache[var]
                if isinstance(contract_data, dict):
                    return contract_data
                elif isinstance(contract_data, str):
                    return {'address': contract_data, 'chain': 'ethereum'}
        
        return None
    
    def _analyze_whale_transactions(self, contract_address: str, chain: str, hours: int) -> List[OnChainInsight]:
        """Analyze whale transactions for descriptive insights"""
        insights = []
        
        try:
            transactions = self._get_recent_transactions(contract_address, chain, hours)
            
            large_transactions = []
            total_volume_usd = 0
            
            for tx in transactions:
                value_usd = self._estimate_transaction_value_usd(tx, contract_address)
                if value_usd and value_usd > 10000:  # $10k+ transactions
                    large_transactions.append((tx, value_usd))
                    total_volume_usd += value_usd
            
            if large_transactions:
                # Generate descriptive insights
                max_tx_value = max(tx[1] for tx in large_transactions)
                
                if max_tx_value > 100000:  # $100k+
                    insights.append(OnChainInsight(
                        message=f"Detected whale transfer of over ${max_tx_value:,.0f} in the last {hours} hour(s).",
                        confidence=0.9,
                        source=f"{chain}scan",
                        timestamp=datetime.now(),
                        category="whale_activity"
                    ))
                elif max_tx_value > 50000:  # $50k+
                    insights.append(OnChainInsight(
                        message=f"Large transaction detected: ${max_tx_value:,.0f} movement on {chain}.",
                        confidence=0.7,
                        source=f"{chain}scan",
                        timestamp=datetime.now(),
                        category="whale_activity"
                    ))
                
                if len(large_transactions) >= 3:
                    insights.append(OnChainInsight(
                        message=f"Multiple large transactions ({len(large_transactions)}) detected within {hours} hour(s), total volume: ${total_volume_usd:,.0f}.",
                        confidence=0.8,
                        source=f"{chain}scan",
                        timestamp=datetime.now(),
                        category="whale_sequence"
                    ))
                    
        except Exception as e:
            logger.error(f"Error analyzing whale transactions: {e}")
        
        return insights
    
    def _analyze_dex_activity(self, contract_address: str, chain: str, hours: int) -> List[OnChainInsight]:
        """Analyze DEX inflow/outflow activity"""
        insights = []
        
        try:
            dex_addresses = self.known_dex_addresses.get(chain, [])
            
            transactions = self._get_recent_transactions(contract_address, chain, hours)
            
            dex_inflows = []
            dex_outflows = []
            
            for tx in transactions:
                if tx.get('to', '').lower() in dex_addresses:
                    value_usd = self._estimate_transaction_value_usd(tx, contract_address)
                    if value_usd and value_usd > 1000:  # $1k+ threshold
                        dex_inflows.append(value_usd)
                elif tx.get('from', '').lower() in dex_addresses:
                    value_usd = self._estimate_transaction_value_usd(tx, contract_address)
                    if value_usd and value_usd > 1000:
                        dex_outflows.append(value_usd)
            
            total_inflow = sum(dex_inflows)
            total_outflow = sum(dex_outflows)
            
            if total_inflow > 50000:  # $50k+ inflow
                insights.append(OnChainInsight(
                    message=f"Significant DEX inflow detected: ${total_inflow:,.0f} flowing into {chain} DEXs in {hours} hour(s).",
                    confidence=0.85,
                    source=f"{chain}_dex_scanner",
                    timestamp=datetime.now(),
                    category="dex_inflow"
                ))
            elif total_inflow > 10000:  # $10k+ inflow
                insights.append(OnChainInsight(
                    message=f"Unusual DEX inflow activity: ${total_inflow:,.0f} detected in the last {hours} hour(s).",
                    confidence=0.6,
                    source=f"{chain}_dex_scanner",
                    timestamp=datetime.now(),
                    category="dex_inflow"
                ))
            
            if total_outflow > total_inflow * 2:  # Significant outflow vs inflow
                insights.append(OnChainInsight(
                    message=f"Heavy DEX outflow detected: ${total_outflow:,.0f} leaving DEXs, may indicate selling pressure.",
                    confidence=0.7,
                    source=f"{chain}_dex_scanner",
                    timestamp=datetime.now(),
                    category="dex_outflow"
                ))
                
        except Exception as e:
            logger.error(f"Error analyzing DEX activity: {e}")
        
        return insights
    
    def _analyze_new_wallets(self, contract_address: str, chain: str, hours: int) -> List[OnChainInsight]:
        """Analyze new wallet interactions"""
        insights = []
        
        try:
            transactions = self._get_recent_transactions(contract_address, chain, hours)
            
            unique_addresses = set()
            for tx in transactions:
                unique_addresses.add(tx.get('from', ''))
                unique_addresses.add(tx.get('to', ''))
            
            # Filter out known addresses (exchanges, DEXs, etc.)
            new_wallets = []
            for addr in unique_addresses:
                if addr and not self._is_known_address(addr, chain):
                    new_wallets.append(addr)
            
            if len(new_wallets) > 20:  # Many new addresses
                insights.append(OnChainInsight(
                    message=f"High wallet activity: {len(new_wallets)} unique addresses interacted with the contract in {hours} hour(s).",
                    confidence=0.6,
                    source=f"{chain}scan",
                    timestamp=datetime.now(),
                    category="wallet_activity"
                ))
            elif len(new_wallets) > 10:
                insights.append(OnChainInsight(
                    message=f"Increased wallet interactions: {len(new_wallets)} new addresses detected recently.",
                    confidence=0.4,
                    source=f"{chain}scan",
                    timestamp=datetime.now(),
                    category="wallet_activity"
                ))
                
        except Exception as e:
            logger.error(f"Error analyzing new wallets: {e}")
        
        return insights
    
    def _analyze_bridge_activity(self, contract_address: str, chain: str, hours: int) -> List[OnChainInsight]:
        """Analyze cross-chain bridge activity"""
        insights = []
        
        try:
            # Known bridge addresses (simplified)
            bridge_addresses = {
                'ethereum': ['0xa0b86a33e6411fbf2137894e4d8e5d2e6c0b0e9d'],  # Example bridge
                'bsc': ['0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d'],      # Example bridge
                'polygon': ['0x2791bca1f2de4661ed88a30c99a7a9449aa84174'], # Example bridge
            }
            
            transactions = self._get_recent_transactions(contract_address, chain, hours)
            bridge_txs = []
            
            bridges = bridge_addresses.get(chain, [])
            for tx in transactions:
                if (tx.get('to', '').lower() in bridges or 
                    tx.get('from', '').lower() in bridges):
                    value_usd = self._estimate_transaction_value_usd(tx, contract_address)
                    if value_usd and value_usd > 5000:  # $5k+ bridge transactions
                        bridge_txs.append(value_usd)
            
            if bridge_txs:
                total_bridge_volume = sum(bridge_txs)
                insights.append(OnChainInsight(
                    message=f"Cross-chain bridge activity detected: ${total_bridge_volume:,.0f} in bridge transactions on {chain}.",
                    confidence=0.7,
                    source=f"{chain}_bridge_scanner",
                    timestamp=datetime.now(),
                    category="bridge_activity"
                ))
                
        except Exception as e:
            logger.error(f"Error analyzing bridge activity: {e}")
        
        return insights
    
    def _analyze_approval_transactions(self, contract_address: str, chain: str, hours: int) -> List[OnChainInsight]:
        """Analyze approval transactions that may indicate upcoming large trades"""
        insights = []
        
        try:
            # Look for approval transactions
            transactions = self._get_recent_transactions(contract_address, chain, hours)
            
            approval_count = 0
            large_approvals = 0
            
            for tx in transactions:
                # Check if transaction is an approval (simplified check)
                if tx.get('input', '').startswith('0x095ea7b3'):  # approve() method signature
                    approval_count += 1
                    value_usd = self._estimate_transaction_value_usd(tx, contract_address)
                    if value_usd and value_usd > 10000:  # $10k+ approvals
                        large_approvals += 1
            
            if large_approvals > 0:
                insights.append(OnChainInsight(
                    message=f"Large approval transactions spotted: {large_approvals} high-value approvals may indicate upcoming trades.",
                    confidence=0.6,
                    source=f"{chain}scan",
                    timestamp=datetime.now(),
                    category="approval_activity"
                ))
            elif approval_count > 5:
                insights.append(OnChainInsight(
                    message=f"Increased approval activity: {approval_count} approval transactions detected in {hours} hour(s).",
                    confidence=0.4,
                    source=f"{chain}scan",
                    timestamp=datetime.now(),
                    category="approval_activity"
                ))
                
        except Exception as e:
            logger.error(f"Error analyzing approval transactions: {e}")
        
        return insights
    
    def _get_recent_transactions(self, contract_address: str, chain: str, hours: int) -> List[Dict]:
        """Get recent transactions for a contract"""
        api_key = getattr(self, f"{chain}scan_api_key", None)
        if not api_key:
            return []
        
        base_urls = {
            'ethereum': 'https://api.etherscan.io/api',
            'bsc': 'https://api.bscscan.com/api',
            'polygon': 'https://api.polygonscan.com/api',
            'arbitrum': 'https://api.arbiscan.io/api',
            'optimism': 'https://api-optimistic.etherscan.io/api'
        }
        
        base_url = base_urls.get(chain)
        if not base_url:
            return []
        
        try:
            # Get recent transactions
            params = {
                'module': 'account',
                'action': 'tokentx',
                'contractaddress': contract_address,
                'startblock': 0,
                'endblock': 99999999,
                'sort': 'desc',
                'apikey': api_key
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            data = response.json()
            
            if data.get('status') == '1':
                transactions = data.get('result', [])
                
                # Filter transactions from last N hours
                cutoff_time = datetime.now() - timedelta(hours=hours)
                recent_txs = []
                
                for tx in transactions[:100]:  # Limit to recent 100 transactions
                    try:
                        tx_time = datetime.fromtimestamp(int(tx.get('timeStamp', 0)))
                        if tx_time > cutoff_time:
                            recent_txs.append(tx)
                    except (ValueError, TypeError):
                        continue
                
                return recent_txs
            
        except Exception as e:
            logger.error(f"Error fetching transactions from {chain}: {e}")
        
        return []
    
    def _estimate_transaction_value_usd(self, tx: Dict, contract_address: str) -> Optional[float]:
        """Estimate USD value of a transaction (simplified)"""
        try:
            # This is a simplified estimation
            # In production, would need token price data and proper decimal handling
            value = int(tx.get('value', 0))
            decimals = int(tx.get('tokenDecimal', 18))
            
            # Convert to token amount
            token_amount = value / (10 ** decimals)
            
            # Rough estimation - would need real price data
            # For now, assume average token price of $1-10 for estimation
            estimated_price = 5.0  # Placeholder
            
            return token_amount * estimated_price
            
        except (ValueError, TypeError):
            return None
    
    def _is_known_address(self, address: str, chain: str) -> bool:
        """Check if address is a known exchange, DEX, or service"""
        known_addresses = {
            'ethereum': [
                '0xdac17f958d2ee523a2206206994597c13d831ec7',  # USDT
                '0xa0b86a33e6411fbf2137894e4d8e5d2e6c0b0e9d',  # USDC
                # Add more known addresses
            ],
            'bsc': [
                '0x55d398326f99059ff775485246999027b3197955',  # USDT
                '0xe9e7cea3dedca5984780bafc599bd69add087d56',  # BUSD
            ],
            # Add more chains
        }
        
        chain_addresses = known_addresses.get(chain, [])
        return address.lower() in [addr.lower() for addr in chain_addresses]
    
    def format_insights_for_gpt(self, insights: List[OnChainInsight]) -> List[str]:
        """
        Format insights as descriptive text messages for GPT
        
        Returns:
            List of descriptive strings for GPT interpretation
        """
        if not insights:
            return ["No significant on-chain activity detected in the analyzed timeframe."]
        
        messages = []
        for insight in insights:
            messages.append(insight.message)
        
        return messages
    
    def get_insights_summary(self, insights: List[OnChainInsight]) -> Dict[str, Any]:
        """Get summary statistics of insights"""
        if not insights:
            return {
                'total_insights': 0,
                'categories': {},
                'avg_confidence': 0.0,
                'sources': {}
            }
        
        categories = {}
        sources = {}
        
        for insight in insights:
            categories[insight.category] = categories.get(insight.category, 0) + 1
            sources[insight.source] = sources.get(insight.source, 0) + 1
        
        avg_confidence = sum(insight.confidence for insight in insights) / len(insights)
        
        return {
            'total_insights': len(insights),
            'categories': categories,
            'avg_confidence': avg_confidence,
            'sources': sources,
            'timespan': f"Last analysis covered recent on-chain activity"
        }