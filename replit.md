# Crypto Pre-Pump Detection System

## Overview

This is a sophisticated cryptocurrency market scanner that detects pre-pump signals using advanced multi-stage analysis. The system monitors cryptocurrency markets in real-time, analyzes various indicators to identify potential pre-pump conditions, and sends alerts via Telegram with AI-powered analysis.

## Project Structure

### crypto-scan/ - Pre-Pump Scanner & Stage -1 Detection
- **Main Scanner**: Advanced PPWCS v2.8+ scoring with Stage -1 rhythm detection
- **Dashboard**: Flask web interface for real-time monitoring (port 5000)
- **Service**: Background scanning service with multi-stage analysis
- **Features**: Pre-pump detection, Stage -1 market tension detection, liquidity behavior analysis

### pump-analysis/ - Historical Pump Analysis
- **GPT Analysis**: AI-powered analysis of historical pump events
- **Detector Generation**: Automatic Python function generation from pump patterns
- **Testing Framework**: Comprehensive validation system for generated detectors
- **Machine Learning**: Pattern recognition and classification capabilities

## System Architecture

### Backend Architecture
- **Flask Web Application**: Provides a dashboard interface for monitoring system status and viewing alerts
- **Python Service**: Main scanning service (`crypto_scan_service.py`) that continuously monitors markets
- **File-based Data Storage**: Uses JSON files for caching, storing alerts, reports, and configuration data
- **Real-time Processing**: Continuous market scanning with configurable intervals

### Frontend Architecture
- **Flask Templates**: HTML templates with Bootstrap styling for responsive design
- **JavaScript Dashboard**: Real-time data updates with auto-refresh functionality
- **RESTful API**: JSON endpoints for status, alerts, and market data

## Key Components

### Multi-Stage Detection System
1. **Stage -2.1**: Micro-anomaly detection including whale activity, DEX inflows, volume spikes
2. **Stage -2.2**: News/tag analysis for events like listings, partnerships, exploits
3. **Stage -1**: Market rhythm and tension detection without traditional scoring or indicators
4. **Stage 1G**: Breakout initiation detection with squeeze and accumulation patterns

### Core Modules
- **PPWCS Scoring**: Pre-Pump Weighted Composite Score (0-100 points) calculation
- **Alert System**: Multi-level alerting with cooldown mechanisms
- **Take Profit Engine**: Dynamic TP level forecasting based on signal strength
- **Advanced Detectors**: Specialized detection for heatmap exhaustion, orderbook spoofing, VWAP pinning, volume cluster analysis

### Data Sources and APIs
- **Bybit API**: Market data, orderbook, and trading pair information
- **CoinGecko API**: Token contract addresses and metadata (cached to avoid rate limits)
- **Blockchain Scanners**: Etherscan, BSCScan, PolygonScan for on-chain data
- **OpenAI GPT**: AI-powered signal analysis for high-confidence alerts

## Data Flow

1. **Market Scanning**: Continuous monitoring of cryptocurrency symbols from Bybit
2. **Multi-Stage Analysis**: Each symbol goes through 4-stage detection pipeline
3. **PPWCS Calculation**: Composite scoring based on detected signals and patterns
4. **Alert Generation**: Alerts triggered based on score thresholds and signal quality
5. **AI Analysis**: High-confidence signals sent to GPT for additional analysis
6. **Notification**: Telegram alerts sent with formatted messages and AI insights
7. **Data Storage**: Results cached in JSON files with historical tracking

## External Dependencies

### APIs and Services
- **OpenAI API**: GPT-4 for signal analysis (configured via `OPENAI_API_KEY`)
- **Telegram Bot API**: Alert notifications (configured via `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`)
- **Bybit API**: Market data (authenticated with API keys)
- **CoinGecko API**: Token metadata with caching system
- **Blockchain APIs**: Multiple scanner APIs for on-chain analysis

### Python Dependencies
- Flask and Flask-SQLAlchemy for web framework
- OpenAI for AI integration
- Requests for HTTP API calls
- NumPy for numerical computations
- Python-dotenv for environment management

## Deployment Strategy

### Development Environment
- Uses Replit workflows for parallel service execution
- Flask dashboard on port 5000
- Background scanning service runs independently
- File-based storage for rapid development

### Production Considerations
- **Database**: Models prepared for PostgreSQL migration from file storage
- **Caching**: CoinGecko data cached locally to avoid rate limits
- **Error Handling**: Comprehensive error handling and logging
- **Rate Limiting**: Built-in cooldown mechanisms and API throttling

### Configuration
- Environment variables for all API keys and sensitive data
- Configurable alert thresholds and scanning intervals
- Polish language support for user-facing messages

## Recent Changes

### June 21, 2025 - Alert System Integration Fix - PRODUCTION READY
- **Critical Alert Function Fixed**: Naprawiono process_alert() w alert_system.py - KERNELUSDT score 57 teraz wysyła alerty
- **Telegram 400 Error Fixed**: Dodano Markdown escape dla special characters - VELOUSDT Level 3 alert działa poprawnie  
- **Variable Scope Error Fixed**: Usunięto redundant import os w save_score() - BIGTIMEUSDT scoring działa poprawnie
- **Simplified Alert Logic**: Zastąpiono skomplikowaną cache logic prostym wywołaniem send_alert() function
- **JSON Corruption Fixed**: Dodano atomic writes i error handling dla wszystkich JSON operations w scoring.py
- **Dashboard API Fixed**: Naprawiono get_top_performers() obsługujący różne formaty danych JSON

### June 21, 2025 - Pre-Pump Alert Logic Fix - PRODUCTION READY
- **Critical Alert Logic Fixed**: Zmieniono warunek z OR na AND w get_alert_level() - PPWCS 52 z checklist 5 teraz generuje alert
- **Improved Thresholds**: Level 2 dla PPWCS ≥40 OR (PPWCS ≥35 AND checklist ≥35), Level 3 dla PPWCS ≥50 OR combined strength
- **Bug Resolution**: System nie odrzuca już wysokich PPWCS scores z powodu niskich checklist scores
- **Enhanced Logic**: Bardzo słabe PPWCS (<15) nadal blokuje alert niezależnie od checklist

### June 21, 2025 - Trader AI Engine + Advanced Debug System - PRODUCTION READY
- **Intelligent Decision System**: Zastąpiono sztywne reguły trend-mode heurystyczną analizą symulującą myślenie tradera
- **Multi-Layer Analysis**: analyze_market_structure() + analyze_candle_behavior() + interpret_orderbook() + simulate_trader_decision()
- **Adaptive Scoring System**: compute_trader_score() z context-aware weights - impulse/pullback/breakout mają różne priorytety
- **Comprehensive Debug Logging**: logs/trader_debug_log.txt z pełną strukturą JSON per analiza symbol
- **Enhanced Terminal Prints**: [TRADER DEBUG] z każdym etapem, [TRADER SCORE] z breakdown, [REASONS] z decision logic
- **Alert Logging**: logs/alerted_symbols_log.txt - osobne logi dla high-quality setups score ≥0.75
- **Debug Symbol Tool**: debug_symbol.py dla detailed single-symbol analysis z step-by-step breakdown
- **Quality Assessment**: excellent/strong/good/neutral-watch/weak/very_poor z context adjustment info
- **Production Integration**: Pełna integracja z crypto_scan_service.py - audytowalny decision trail

### June 23, 2025 - Complete Auto-Labeling Training Data Collection System - PRODUCTION READY ✅
- **Chart Export Module**: Created vision_ai/chart_exporter.py for automatic export of charts in SYMBOL_YYYYMMDD_HHMM.png format after TJDE scans
- **GPT Chart Labeler**: Implemented vision_ai/gpt_chart_labeler.py using GPT-4o Vision API to generate setup descriptions in format: "breakout-continuation | trending-up | volume-backed"
- **Auto Label Runner**: Built vision_ai/auto_label_runner.py for complete automation - captures top 5 TJDE tokens, exports charts, generates GPT labels, creates training pairs
- **Training Data Pairs**: Automatic creation of chart.png + description.txt pairs in data/vision_ai/train_data/ with enhanced TJDE context metadata
- **Production Integration**: Full integration into crypto_scan_service.py - saves TJDE results and automatically runs training data collection after each scan
- **Intelligent Descriptions**: GPT generates technical setup descriptions combining visual analysis with TJDE context (scores, decisions, market structure)
- **Data Management**: Automatic cleanup of old files, statistics tracking, and session summaries for training data quality monitoring
- **Seamless Workflow**: Complete end-to-end pipeline from token scanning → chart export → GPT labeling → training pair creation for Vision-AI model development
- **Robust Architecture**: Error handling, fallback descriptions, and comprehensive logging ensure reliable training data collection in production environment

## User Preferences

- Language: Polish for user-facing messages and alerts
- API Access: Bybit API working in production environment
- Debugging: Comprehensive logging preferred for troubleshooting
- Alert Style: Detailed technical analysis with specific condition breakdowns
- System Monitoring: Real-time visibility into detection logic and failure reasons