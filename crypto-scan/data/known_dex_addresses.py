"""
Known DEX addresses for different blockchain networks
Used for detecting DEX inflow transactions
"""

DEX_ADDRESSES = {
    "ethereum": [
        "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Uniswap V2 Router
        "0xe592427a0aece92de3edee1f18e0157c05861564",  # Uniswap V3 SwapRouter
        "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",  # Uniswap V3 SwapRouter02
        "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f",  # SushiSwap Router
        "0x1111111254fb6c44bac0bed2854e76f90643097d",  # 1inch V4 AggregationRouterV4
        "0x881d40237659c251811cec9c364ef91dc08d300c",  # Metamask Swap Router
        "0x6131b5fae19ea4f9d964eac0408e4408b66337b5",  # Kyber Network Router
    ],
    "bsc": [
        "0x10ed43c718714eb63d5aa57b78b54704e256024e",  # PancakeSwap V2 Router
        "0x13f4ea83d0bd40e75c8222255bc855a974568dd4",  # PancakeSwap V3 SwapRouter
        "0x1b81d678ffb9c0263b24a97847620c99d213eb14",  # ApeSwap Router
        "0x05ff2b0db69458a0750badebc4f9e13add608c7f",  # BakerySwap Router
        "0x1111111254fb6c44bac0bed2854e76f90643097d",  # 1inch V4 AggregationRouterV4
    ],
    "polygon": [
        "0xa5e0829caced8ffdd4de3c43696c57f7d7a678ff",  # QuickSwap Router
        "0x1b02da8cb0d097eb8d57a175b88c7d8b47997506",  # SushiSwap Router
        "0xe592427a0aece92de3edee1f18e0157c05861564",  # Uniswap V3 SwapRouter
        "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",  # Uniswap V3 SwapRouter02
        "0x1111111254fb6c44bac0bed2854e76f90643097d",  # 1inch V4 AggregationRouterV4
    ],
    "arbitrum": [
        "0xe592427a0aece92de3edee1f18e0157c05861564",  # Uniswap V3 SwapRouter
        "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",  # Uniswap V3 SwapRouter02
        "0x1b02da8cb0d097eb8d57a175b88c7d8b47997506",  # SushiSwap Router
        "0x1111111254fb6c44bac0bed2854e76f90643097d",  # 1inch V4 AggregationRouterV4
    ],
    "optimism": [
        "0xe592427a0aece92de3edee1f18e0157c05861564",  # Uniswap V3 SwapRouter
        "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",  # Uniswap V3 SwapRouter02
        "0x1b02da8cb0d097eb8d57a175b88c7d8b47997506",  # SushiSwap Router
        "0x1111111254fb6c44bac0bed2854e76f90643097d",  # 1inch V4 AggregationRouterV4
    ]
}