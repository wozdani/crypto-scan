# Crypto Pre-Pump Detection System

## Overview

This project is a cryptocurrency market scanner designed to detect pre-pump signals using advanced multi-stage analysis. It monitors markets in real-time, analyzes various indicators (liquidity, whale activity, on-chain data) to identify potential pre-pump conditions, and sends AI-powered analysis alerts via Telegram. The system aims for institutional-grade detection accuracy and adaptive cryptocurrency intelligence, providing a robust, intelligent tool for navigating volatile crypto markets and identifying lucrative opportunities before major price movements.

## User Preferences

- Language: Polish for user-facing messages and alerts
- API Access: Bybit API working in production environment (development environment shows 403 errors)
- Blockchain APIs: Etherscan V2 integration ready (add ETHERSCAN_V2_API_KEY to enable), current legacy fallback working
- Performance Priority: Speed through simplicity - complex optimizations often counterproductive
- Code Style: Sequential execution preferred over parallel processing that adds overhead without benefit
- Architecture Preference: Simple, maintainable code over sophisticated but slow optimizations
- Scanning Logic: "przywróć dawną logike skanu" - user explicitly requested return to simple scanning approach
- Debugging: Clean, essential logging without repetitive performance metrics that clutter output
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
-   **Fixed AlertDecision Enum Conversion Issue**: Resolved critical bug where consensus decisions were being converted from proper BUY/HOLD/AVOID strings to AlertDecision enum values (causing system to return AlertDecision.NO_ALERT instead of "HOLD"). Multi-Agent consensus now maintains string format throughout the pipeline, eliminating hardcoded WATCH fallbacks and ensuring proper decision flow.
    -   **GPT-4o Integration**: The system utilizes GPT-4o for reliable AI capabilities across all OpenAI API calls, providing consistent crypto analysis and reasoning.
    -   **Canonical Price System**: Single frozen price per token per scan round eliminates dispersed price fallback logic, ensuring consistent pricing across all modules and agents.
    -   **Chain Router System**: Consistent (chain, contract) mapping across whale_ping and dex_inflow modules prevents chain mismatches.
    -   **Pre-Confirmatory Poke System**: Detects when 2/3 AI agents score just under threshold (0.60-0.68) and forces additional data loading (orderbook, chain inflow) before finalizing HOLD decisions.
    -   **CoinGecko Token Validation**: Automatic filtering of non-existent tokens (e.g., 1000000PEIPEIUSDT) based on CoinGecko cache validation prevents scanning of invalid symbols, improving system efficiency and data quality.
    -   **Enhanced Explore Mode System**: Advanced agent learning system with proper file naming convention (TOKEN_YYYYMMDD_HHMMSS_DETECTORS format before verification, TOKEN_YYYYMMDD_HHMMSS_OUTCOME after 6h verification), automated pump verification scheduler running every 6 hours at 02:00, 08:00, 14:00, 20:00 UTC, comprehensive detector tracking and confidence scoring, and integrated file lifecycle management with 3-day retention policy. **Threshold aligned at 0.7** - explore mode now only activates for high-quality signals above consensus threshold.
-   **Threshold-Aware Learning System**: Intelligent detector weight adaptation based on understanding of 0.7 consensus threshold. System tracks scores above/near/below threshold (≥0.7/0.6-0.69/<0.6), penalizes "near miss" signals that don't trigger consensus, and rewards detectors for exceeding the actionable threshold. Automated weight updates every 3 hours with periodic threshold tracking saves every 2 hours.
-   **Canonical Price System**: Single source of truth pricing with strict priority (ticker_last → orderbook_mid → candle_15m → candle_5m) eliminates price desynchronization issues and ensures all modules use consistent pricing data. Multi-agent consensus fully integrated with canonical pricing, eliminating $0 price displays and "unknown" source references.

## External Dependencies

-   **OpenAI API**: For GPT-4o powered signal analysis, AI reasoning within the multi-agent system, and chart commentary generation.
-   **Telegram Bot API**: For sending real-time alert notifications.
-   **Bybit API**: Primary source for market data, orderbook information, and trading pair details.
-   **CoinGecko API**: For token contract addresses and metadata.
-   **Blockchain Scanner APIs**: Etherscan V2 API with fallback to legacy chain-specific APIs (Etherscan, BSCScan, PolygonScan, ArbScan, OptimismScan) for comprehensive on-chain data analysis.
-   **Python Libraries**:
    -   `Flask`: Web framework.
    -   `OpenAI`: Python client for OpenAI API.
    -   `Requests`: For HTTP API calls.
    -   `NumPy`: For numerical computations.
    -   `Python-dotenv`: For managing environment variables.
    -   `Playwright`: For automated browser interaction.
    -   `Pytesseract`: For OCR-based chart validation.
    -   `PyTorch`: For neural network models.
    -   `NetworkX`: For graph processing.
    -   `Scikit-learn`: For machine learning models.
    -   `Schedule`: For scheduling automated tasks.
    -   `HuggingFace Transformers`: For CLIP model integration.