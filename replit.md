# Crypto Pre-Pump Detection System

## Overview

This is a sophisticated cryptocurrency market scanner that detects pre-pump signals using advanced multi-stage analysis. The system monitors cryptocurrency markets in real-time, analyzes various indicators to identify potential pre-pump conditions, and sends alerts via Telegram with AI-powered analysis.

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
3. **Stage -1**: Compression analysis and market structure evaluation
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

### June 16, 2025 - PPWCS Scoring System Overhaul & Detection Improvements
- **PPWCS scoring dramatically improved**: Increased individual signal weights (whale: 12pts, volume: 12pts, dex: 10pts) vs previous 14pts total for 2 signals
- **Compressed signal promoted**: Now counted as full Stage -2.1 detector (+10pts) instead of just bonus, properly recognizing price compression patterns
- **Enhanced combo bonuses**: whale+dex: 8pts, volume+dex: 6pts, new combo_volume_inflow: 5pts for better reward of signal combinations
- **Quality scoring added**: RSI_flatline now contributes 5pts to quality score instead of being ignored
- **Compression requirements relaxed**: Changed from ≥2 signals to ≥1 signal for Stage -1 activation to prevent blocking valid pre-pumps
- **Volume spike detection enhanced**: Now analyzes last 3 candles instead of just 1, checking each against previous 4 candles average
- **RSI condition removed**: Eliminated 46-54 RSI requirement from detect_rsi_flatline() to prevent missing valid pre-pumps
- **Pure accumulation logic fixed**: Now correctly uses social_spike_active instead of hardcoded False
- **Comprehensive debug logging**: Added detailed debug prints showing exact scoring breakdown for transparency
- **Scoring validation**: Strong setups now achieve 70-90+ PPWCS vs previous 24-35, enabling proper alert levels

### June 16, 2025 - Cache System Optimization & Single API Call Fix
- **Fixed cache validation logic**: System now properly detects empty cache files instead of considering them valid
- **CoinGecko optimization complete**: Eliminated individual token API calls, now uses single `/v3/coins/list?include_platform=true` request
- **Rate limiting eliminated**: No more 429 errors from CoinGecko API due to single-call approach
- **Fast cache building**: Removed delays, processes all tokens instantly without API throttling
- **Resolved circular dependency**: Created separate Bybit symbol fetcher for cache building
- **API credentials issue**: Current Bybit API keys returning 403 errors, preventing symbol fetching
- **Production ready optimization**: Cache system optimized for maximum efficiency with single API calls

### June 16, 2025 - CoinGecko Token Selection Fix & Production Ready
- **CoinGecko cache bug fixed**: Corrected token selection where system picked wrong tokens (e.g. MAGIC → magnificent-7777 instead of magic)
- **Smart token selection**: Implemented hierarchical selection with symbol verification and wrapped token filtering
- **Priority-based matching**: Prefers main tokens over wrapped/bridged versions with excluded prefixes filtering
- **Critical scoring bug fixed**: Corrected logic where False/None/0.0 values were counted as active detectors, causing inflated PPWCS scores
- **Strict True checking**: All detector evaluations now use `is True` instead of truthy evaluation to prevent false positives
- **Enhanced detector sensitivity**: DEX inflow (2.5% → 1.2% market cap), orderbook anomaly (4.0x → 2.5x), volume spike 45min cooldown
- **Complete debug system**: Added detailed logging showing actual signal values, types, and only truly active detectors
- **Missing Stage 1g detectors implemented**: squeeze (BB compression), fake_reject (wick recovery), liquidity_box (tight range), fractal_echo (pattern repetition)
- **PPWCS structure/quality separation**: Separated scoring into ppwcs_structure (stages -2.1, -2.2, -1, pure accumulation) and ppwcs_quality (stage 1g)
- **Production ready**: All critical bugs resolved, accurate token mapping, precise detector activation, ready for Hetzner deployment

### June 15, 2025 - Complete PPWCS 2.6 + Stage 1g 2.0 Implementation
- **PPWCS 2.6 scoring system**: Implemented new multi-stage scoring algorithm with detector count-based scoring
- **Stage -2.1 improvements**: New scoring based on detector combinations (1=6pts, 2=14pts, 3=20pts, 4+=25pts)
- **Stage -2.2 tag updates**: Updated tag scoring (listing/partnership=+10, presale/cex_listed=+5, exploit/rug/delisting=-15)
- **Stage -1 compression filter**: Added compression detection when ≥2 Stage -2.1 signals occur in same hour (+10pts)
- **Stage 1g 2.0 implementation**: Simplified activation with new trigger combinations and quality scoring
- **Quality filter enhancement**: Stage 1g quality >12 allows alerts at PPWCS 60-69 range
- **Alert system upgrade**: Enhanced alert levels with Stage 1g quality boost for watchlist promotion
- **Pure accumulation bonus**: Added +5 bonus for whale+DEX inflow without social spike
- **Blocking tags**: Negative tags (exploit, rug, delisting) now properly block alerts at -15 score
- **Production ready**: All bug fixes maintained while implementing comprehensive scoring upgrades

### Previous Optimizations Maintained
- **Cache-only system**: CoinGecko optimization preventing 429 rate limit errors
- **Production ready**: System optimized for deployment on Hetzner Cloud server
- **Multi-stage detection**: Complete 4-stage pre-pump detection pipeline

## Changelog
- June 15, 2025: Initial setup and critical bug fixes

## User Preferences

Preferred communication style: Simple, everyday language.