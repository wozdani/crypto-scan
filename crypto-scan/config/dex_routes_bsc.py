"""
BSC DEX Routes Configuration
Configuration of known DEX routers and factory addresses for BSC chain
"""

# BSC DEX Router addresses - add your complete set here
DEX_ROUTERS_BSC = {
    # PancakeSwap V2
    "pancakeswap_v2_router": "0x10ed43c718714eb63d5aa57b78b54704e256024e",
    "pancakeswap_v2_factory": "0xca143ce32fe78f1f7019d7d551a6402fc5350c73",
    
    # PancakeSwap V1 (legacy)
    "pancakeswap_v1_router": "0x05ff2b0db69458a0750badebc4f9e13add608c7f",
    "pancakeswap_v1_factory": "0xbcfccbde45ce874adcb698cc183debcf17952812",
    
    # Uniswap V2 (also deployed on BSC)
    "uniswap_v2_router": "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",
    "uniswap_v2_factory": "0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f",
    
    # SushiSwap
    "sushiswap_router": "0x1b02da8cb0d097eb8d57a175b88c7d8b47997506",
    "sushiswap_factory": "0xc35dadb65012ec5796536bd9864ed8773abc74c4",
    
    # Thena (Popular BSC DEX)
    "thena_router": "0xd4ae6eca985340dd434d38f470accce4dc78d109",
    "thena_factory": "0xafd89d21bdb66d00817d4153e055830b1c2b3970",
    
    # MDEX (Multi-chain DEX)
    "mdex_router": "0x7dae51bd3e3376b8c7c4561919ab32297d4ddc82",
    "mdex_factory": "0x3cd1c46068daea5ebb0d3f55f6915b10648062b8",
    
    # BiSwap
    "biswap_router": "0x3a6d8ca21d1cf76f653a67577fa0d27453350dd8",
    "biswap_factory": "0x858e3312ed3a876947ea49d572a7c42de08af7ee",
    
    # ApeSwap
    "apeswap_router": "0xcf0febd3f17cef5b47b0cd257acf6025c5bff3b7",
    "apeswap_factory": "0x0841bd0b734e4f5853f0dd8d7ea041c241fb0da6",
    
    # BakerySwap
    "bakeryswap_router": "0xcde540d7eafe93ac5fe6233bee57e1270d3e330f",
    
    # JulSwap
    "julswap_router": "0xbd67d157502a23309db761c41965600c2ec788b2",
    "julswap_factory": "0x553990f2cba90272390f62c5bdb1681ffc899675",
    
    # KyberSwap (Elastic)
    "kyberswap_elastic_router": "0xc1e7df5dec9827954b6c59b9ec3a6c5c56c47e67",
    
    # 1inch Router (aggregator)
    "1inch_v4_router": "0x1111111254fb6c44bac0bed2854e76f90643097d",
    "1inch_v5_router": "0x1111111254eeb25477b68fb85ed929f73a960582"
}

# Additional configuration
DEX_CONFIG = {
    "minimum_liquidity_usd": 1000,  # Minimum USD value to consider
    "swap_signature": "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822",
    "transfer_signature": "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
}