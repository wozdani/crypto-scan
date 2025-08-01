# Crypto Pre-Pump Detection System

## Overview

This project develops a cryptocurrency market scanner designed to detect pre-pump signals using advanced multi-stage analysis. It monitors markets in real-time, analyzes various indicators (including liquidity, whale activity, and on-chain data) to identify potential pre-pump conditions, and sends alerts via Telegram with AI-powered analysis. The system aims for institutional-grade detection accuracy and adaptive cryptocurrency intelligence.

## User Preferences

- Language: Polish for user-facing messages and alerts
- API Access: Bybit API working in production environment (development environment shows 403 errors)
- Performance Priority: Speed through simplicity - complex optimizations often counterproductive
- Code Style: Sequential execution preferred over parallel processing that adds overhead without benefit
- Architecture Preference: Simple, maintainable code over sophisticated but slow optimizations
- Scanning Logic: "przywróć dawną logike skanu" - user explicitly requested return to simple scanning approach
- Debugging: Basic logging without performance monitoring overhead that slows system
- Alert Style: Detailed technical analysis with specific condition breakdowns
- System Monitoring: Real-time visibility into detection logic and failure reasons
- Error Handling: Graceful degradation when modules unavailable, avoid breaking system with complex dependencies
- Development Reality: API 403 errors in development environment are expected - system optimized for production where Bybit API works correctly

## System Architecture

The system employs a multi-stage detection pipeline for pre-pump signals. It is built on a Python backend using Flask for a web dashboard (`crypto-scan/`) and a dedicated background scanning service (`crypto_scan_service.py`). Data storage primarily utilizes JSON files for caching, alerts, reports, and configuration.

**Key Architectural Decisions & Design Patterns:**

-   **Multi-Agent Consensus Engine**: A primary decision-making mechanism utilizing a 5-agent system (Analyzer, Reasoner, Voter, Debater, Decider) for evaluating each detector's output and reaching a unified decision (BUY/HOLD/AVOID). This often integrates with OpenAI's GPT for real AI reasoning.
-   **WHALE_PING BUG FIXED (August 1, 2025)**: Successfully identified and resolved critical whale_ping preservation issue in TOP 5 display system. Fixed three components: stealth_logger.py whale_ping reset bug, priority_scheduler.py extraction logic, and scan_token_async.py stealth_result injection. Verified working: LRCUSDT=1.135, CHZUSDT=1.110 whale_ping values preserved.
-   **WHALECLIP STATUS CONSISTENCY FIXED (August 1, 2025)**: Resolved WhaleCLIP status inconsistency where detector showed Status=True but consensus used enabled=False. Fixed three locations in stealth_engine.py to use whaleclip_status_corrected consistently throughout consensus logic.
-   **MULTI-AGENT CONSENSUS SYSTEM RESTORED (August 1, 2025)**: Successfully replaced Enhanced RL weighted averaging system with proper OpenAI-powered Multi-Agent Consensus. System now uses 5 agents (Analyzer, Reasoner, Voter, Debater, Decider) for real AI reasoning on each token decision. Alerts trigger only from consensus BUY decisions, not individual detectors. Comprehensive logging implemented for transparency.
-   **Enhanced Reinforcement Learning (RL) Architecture**: The system can operate with a Deep Q-Network + RLAgentV3 system for adaptive cryptocurrency intelligence, replacing traditional multi-agent consensus for superior decision-making. It incorporates sophisticated AI detectors like DiamondWhale AI, Californium AI, WhaleCLIP AI, and Stealth Engine.
-   **Multi-Stage Detection**: A 4-stage process (Stage -2.1 micro-anomaly, Stage -2.2 news/tag analysis, Stage -1 market rhythm/tension, Stage 1G breakout initiation) for comprehensive signal identification.
-   **Adaptive Thresholds**: Dynamic thresholds for whale detection and smart money triggers, adjusting based on market conditions and historical performance, ensuring detection sensitivity without false positives.
-   **Component-Aware Feedback Loop**: A self-learning decay system automatically adjusts the weights of detection components based on their real-time effectiveness and historical performance, optimizing the system without manual intervention.
-   **Unified Alert System**: A centralized Telegram alert management system ensures consistent message formatting, intelligent cooldowns, and detailed transparency for all detection components.
-   **Modular Design**: Components like the pump verification system, feedback loops, and various AI detectors are designed as modular units, allowing for independent development, testing, and integration.
-   **Robust Error Handling & Fallbacks**: Comprehensive error handling, timeout protections (e.g., for file I/O, blockchain API calls), and intelligent fallback mechanisms are implemented throughout the system to ensure continuous operation and stability even in adverse conditions.
-   **UI/UX**: A Flask-based web dashboard (port 5000) provides real-time monitoring, utilizing HTML templates with Bootstrap for a responsive design and JavaScript for dynamic updates.

**Core Technical Implementations & Features:**

-   **PPWCS Scoring**: A Pre-Pump Weighted Composite Score (0-100) is calculated based on detected signals.
-   **Advanced Detectors**: Specialized detection for various market anomalies, including heatmap exhaustion, orderbook spoofing, VWAP pinning, and volume cluster analysis.
-   **Smart Money Detection**: Integrates a trigger system with absolute minimum thresholds for detecting whale addresses, a whale memory system for tracking repeat behaviors, and DEX inflow analysis with address trust management.
-   **Pump Verification System**: Uses machine learning to extract enhanced features from AI detector patterns and mastermind sequences for comprehensive pump classification.
-   **Priority Alert Queue**: A system for prioritizing alerts based on a computed score, allowing for dynamic time windows and instant alerts for high-confidence signals.
-   **GNN-based Stealth Engine**: A core component leveraging Graph Neural Networks for blockchain transaction analysis, anomaly detection, and reinforcement learning-based decision making.
-   **Wallet Behavior Encoder**: Generates behavioral embeddings from transaction histories for whale classification and suspicious pattern recognition.
-   **Contextual Chart Generation**: Generates TradingView-style charts with contextual overlays (e.g., market phase background colors, TJDE components) for Vision-AI training.

## External Dependencies

-   **OpenAI API**: Used for GPT-4 (or GPT-4o) powered signal analysis, AI reasoning within the multi-agent system, and chart commentary generation.
-   **Telegram Bot API**: For sending real-time alert notifications.
-   **Bybit API**: Primary source for market data, orderbook information, and trading pair details.
-   **CoinGecko API**: For token contract addresses and metadata (with local caching to manage rate limits).
-   **Blockchain Scanner APIs**: Including Etherscan, BSCScan, and PolygonScan for on-chain data analysis.
-   **Python Libraries**:
    -   `Flask`: Web framework for the dashboard.
    -   `OpenAI`: Python client for OpenAI API.
    -   `Requests`: For general HTTP API calls.
    -   `NumPy`: For numerical computations.
    -   `Python-dotenv`: For managing environment variables.
    -   `Playwright`: For automated browser interaction (TradingView chart screenshots).
    -   `Pytesseract`: For OCR-based chart validation.
    -   `PyTorch`: For neural network models (e.g., TemporalGCN in DiamondWhale AI, RLAgentV4).
    -   `NetworkX`: For graph processing in GNN components.
    -   `Scikit-learn`: For machine learning models (e.g., RandomForest in whale classification).
    -   `Schedule`: For scheduling automated tasks in the GNN scheduler.
    -   `HuggingFace Transformers`: For CLIP model integration.