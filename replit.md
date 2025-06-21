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

### June 21, 2025 - Trend Mode 2.0 Complete System - PRODUCTION READY
- **Revolutionary Scoring System**: Zastąpienie binarnej logiki trend_active nowym modelem z 3 kategoriami detektorów
- **Core Detectors (+10 pts)**: uptrend_15m, pullback_flow, calm_before_trend, orderbook_freeze, heatmap_vacuum
- **Helper Detectors (+5 pts)**: vwap_pinning, human_flow, micro_echo, sr_structure_score>7, volume_flow_consistency  
- **Negative Detectors (-10 pts)**: chaotic_flow≥40%, ask_domination>80x, fake_pulse, high_volatility, strong_sell_pressure
- **3-Level Alert System**: Watchlist (≥25pts), Active Entry (≥40pts), Confirmed Trend (≥60pts)
- **Advanced Trailing Logic**: Alert tylko gdy score wzrasta +5 AND przekracza próg poziomu
- **Production Integration**: Pełne zastąpienie legacy trend mode w crypto_scan_service.py

### June 21, 2025 - Trailing Scoring & Alert Engine - PRODUCTION READY
- **Trend Mode Alert Engine**: Kompletny system trailing scoring z historią 5 ostatnich wyników dla każdego symbolu
- **Enhanced Scoring System**: Integracja pullback_flow_pattern (+10), flow_consistency (+5), bullish orderbook (+5) i innych detektorów
- **Alert Triggers**: Score wzrost +10 punktów OR przekroczenie progu ≥70 OR pullback trigger z wysokim score ≥65
- **Production Integration**: Pełna integracja z crypto_scan_service.py przez trend_mode_integration.py helper
- **Real-time Processing**: Automatyczne zbieranie danych z modułów, obliczanie enhanced score i generowanie alertów

### June 21, 2025 - Pullback Flow Pattern Module - PRODUCTION READY
- **Nowy moduł pullback_flow_pattern**: Wykrywa koniec korekty i potencjalne wejście w pozycję z analizą 15M downtrend + 5M reversal signals
- **Integracja z Trend Mode Pipeline**: Dodano jako 10. detektor z maksymalnie 18 punktami w systemie scoringu
- **Zaawansowana analiza orderbook**: Wykrywa malejący ask pressure, rosnący bid pressure, vacuum i freeze conditions
- **System scoring 0-100**: Confidence score z progiem 50 punktów i minimum 3 entry triggers dla aktywacji
- **Pełna kompatybilność**: Wrapper functions dla integracji z istniejącym systemem detektorów PPWCS

### June 21, 2025 - Critical Function Fix & Trend Mode Debug System - PRODUCTION READY
- **Critical function name error fixed**: Resolved undefined `is_valid_symbol` function causing main scanning loop failure by replacing with existing `is_valid_perpetual_symbol`
- **System stability restored**: Fixed crash in symbol validation that was preventing any scanning operations from completing successfully

### June 21, 2025 - Trend Mode Debug & API Connectivity System - PRODUCTION READY
- **Comprehensive debugging system implemented**: Added trend_debug_log.txt with detailed condition analysis showing up_ratio calculations, support level distances, and bid pressure status
- **Enhanced symbol fetching reliability**: Emergency fallback to 24 essential trading pairs when Bybit cache/API unavailable to ensure continuous operation
- **Multi-endpoint API fallback strategy**: Sequential testing of linear, spot, and inverse categories for maximum data availability across different market conditions
- **Lowered detection thresholds for testing**: Temporarily reduced up_ratio to 0.5 (from 0.6) and support margin to 0.004 (from 0.002) for enhanced signal detection sensitivity
- **Complete debugging workflow**: Step-by-step analysis logging including 15M trend validation, 5M entry signal detection, S/R level identification, and orderbook pressure monitoring
- **Production-ready monitoring system**: Full visibility into Trend Mode decision process with comprehensive failure reason tracking and detailed condition breakdowns
- **Enhanced trend_mode_debugger.py tool**: Batch testing capability for multiple symbols with detailed analysis reports and debugging summaries

### June 20, 2025 - Support/Resistance Trend Mode System - PRODUCTION READY
- **Recent 3h trend analysis**: is_strong_recent_trend() analyzes only last 12x15M candles requiring ≥60% green candles for trend confirmation
- **Historical S/R level detection**: find_support_resistance_levels() identifies pivot points from 3-24h history (84 candles) with tolerance-based level clustering
- **Support proximity testing**: is_price_near_support() validates current price testing support levels within 0.2% margin
- **5M correction entry timing**: is_entry_after_correction() detects end of correction with declining ask volume + rising bid volume
- **Perfect condition scoring**: 100 points for 3h trend + near support + entry signal, 70 points for setup without entry
- **Real-time S/R integration**: Complete Bybit API integration for 15M/5M data, current price, and orderbook analysis
- **Complete system replacement**: Full replacement of old trend detection with sophisticated S/R-based multi-timeframe analysis

## User Preferences

- Language: Polish for user-facing messages and alerts
- API Access: Bybit API working in production environment
- Debugging: Comprehensive logging preferred for troubleshooting
- Alert Style: Detailed technical analysis with specific condition breakdowns
- System Monitoring: Real-time visibility into detection logic and failure reasons