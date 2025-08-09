# Crypto Pre-Pump Detection System

## Overview

This project is a cryptocurrency market scanner designed to detect pre-pump signals using advanced multi-stage analysis. It monitors markets in real-time, analyzes various indicators (liquidity, whale activity, on-chain data) to identify potential pre-pump conditions, and sends AI-powered analysis alerts via Telegram. The system aims for institutional-grade detection accuracy and adaptive cryptocurrency intelligence. Its business vision is to provide a robust, intelligent tool for navigating volatile crypto markets, offering a significant edge in identifying lucrative opportunities before major price movements.

## Recent Changes (August 2025)

- **Canonical Price System**: Deployed frozen price system across all modules ensuring single price per token per scan round for consistent multi-agent reasoning
- **Chain Router Integration**: Implemented consistent (chain, contract) mapping between whale_ping and dex_inflow modules to eliminate chain mismatch issues like WIF ethereum vs Solana routing
- **Stealth Engine Integration**: Chain router system integrated into stealth_signals.py for dex_inflow and whale_ping consistency with proper chain mismatch detection
- **Multi-Agent Consensus**: Enhanced 5-agent AI consensus system with canonical price integration for consistent decision making across all detector evaluations

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

The system employs a multi-stage detection pipeline for pre-pump signals, built on a Python backend using Flask for a web dashboard and a dedicated background scanning service. Data storage primarily utilizes JSON files for caching, alerts, reports, and configuration.

**Key Architectural Decisions & Design Patterns:**

-   **Multi-Agent Consensus Engine**: A primary decision-making mechanism utilizing a 5-agent system (Analyzer, Reasoner, Voter, Debater, Decider) for evaluating each detector's output and reaching a unified decision (BUY/HOLD/AVOID), often integrating with OpenAI's GPT for real AI reasoning. Alerts are triggered only from consensus BUY decisions.
-   **Multi-Stage Detection**: A 4-stage process (micro-anomaly, news/tag analysis, market rhythm/tension, breakout initiation) for comprehensive signal identification.
-   **Adaptive Thresholds**: Dynamic thresholds for whale detection and smart money triggers, adjusting based on market conditions and historical performance.
-   **Component-Aware Feedback Loop**: A self-learning decay system automatically adjusts the weights of detection components based on their real-time effectiveness and historical performance.
-   **Unified Alert System**: Centralized Telegram alert management ensures consistent message formatting, intelligent cooldowns, and detailed transparency. Alerts are processed in real-time with strict BUY-ONLY filtering, sent immediately upon consensus decision, and require a minimum of 2 active detectors.
-   **Modular Design**: Components like the pump verification system, feedback loops, and various AI detectors are designed as modular units.
-   **Robust Error Handling & Fallbacks**: Comprehensive error handling, timeout protections, and intelligent fallback mechanisms (e.g., OpenAI API retries with exponential backoff and fallback to gpt-3.5-turbo) ensure continuous operation.
-   **UI/UX**: A Flask-based web dashboard (port 5000) provides real-time monitoring, utilizing HTML templates with Bootstrap for a responsive design and JavaScript for dynamic updates.
-   **Core Technical Implementations & Features**:
    -   **PPWCS Scoring**: A Pre-Pump Weighted Composite Score (0-100) is calculated based on detected signals.
    -   **Advanced Detectors**: Specialized detection for market anomalies (heatmap exhaustion, orderbook spoofing, VWAP pinning, volume cluster analysis).
    -   **Smart Money Detection**: Integrates a trigger system with minimum thresholds for whale addresses, a whale memory system, and DEX inflow analysis with address trust management.
    -   **Pump Verification System**: Uses machine learning for comprehensive pump classification, including automatic cleanup of old data.
    -   **Priority Alert Queue**: Prioritizes alerts based on a computed score, allowing dynamic time windows and instant alerts.
    -   **GNN-based Stealth Engine**: Leverages Graph Neural Networks for blockchain transaction analysis, anomaly detection, and reinforcement learning.
    -   **Wallet Behavior Encoder**: Generates behavioral embeddings from transaction histories for whale classification.
    -   **Contextual Chart Generation**: Generates TradingView-style charts with contextual overlays for Vision-AI training.
    -   **Individual Multi-Agent Processing**: The Multi-Agent Consensus System uses individual OpenAI API calls per detector (1 call per detector for all 5 agents) rather than batch processing, providing better reliability and easier debugging for each detector's 5-agent evaluation.
    -   **GPT-5 Integration**: The system utilizes GPT-5 for enhanced AI capabilities across all OpenAI API calls, providing superior crypto analysis and reasoning.
    -   **Canonical Price System**: Single frozen price per token per scan round eliminates dispersed price fallback logic, ensuring consistent pricing across all modules and agents.
    -   **Chain Router System**: Consistent (chain, contract) mapping across whale_ping and dex_inflow modules prevents chain mismatches like WIF ethereum vs Solana routing inconsistencies.

## External Dependencies

-   **OpenAI API**: For GPT-5 powered signal analysis, AI reasoning within the multi-agent system, and chart commentary generation.
-   **Telegram Bot API**: For sending real-time alert notifications.
-   **Bybit API**: Primary source for market data, orderbook information, and trading pair details.
-   **CoinGecko API**: For token contract addresses and metadata.
-   **Blockchain Scanner APIs**: Including Etherscan, BSCScan, and PolygonScan for on-chain data analysis.
-   **Python Libraries**:
    -   `Flask`: Web framework.
    -   `OpenAI`: Python client for OpenAI API.
    -   `Requests`: For HTTP API calls.
    -   `NumPy`: For numerical computations.
    -   `Python-dotenv`: For managing environment variables.
    -   `Playwright`: For automated browser interaction (TradingView chart screenshots).
    -   `Pytesseract`: For OCR-based chart validation.
    -   `PyTorch`: For neural network models.
    -   `NetworkX`: For graph processing.
    -   `Scikit-learn`: For machine learning models.
    -   `Schedule`: For scheduling automated tasks.
    -   `HuggingFace Transformers`: For CLIP model integration.