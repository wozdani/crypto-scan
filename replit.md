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

### June 15, 2025 - Complete Project Replacement & Bug Fixes
- **Full project replacement**: Replaced entire codebase with user's latest ZIP archive
- **Critical bug fixes**: Fixed 'str' object has no attribute 'get' errors in detector functions
- **Telegram bot restored**: Fixed TELEGRAM_BOT_TOKEN configuration and tested successfully
- **Missing functions added**: Implemented detect_dex_inflow_anomaly and detect_event_tag functions
- **Type handling improved**: All detector functions now return proper tuple (bool, float) format
- **Environment loading**: Added proper .env loading in crypto_scan_service.py
- **NoneType error fixed**: Resolved 'NoneType' object has no attribute 'get' error in get_market_data()
- **System stability**: All Python errors resolved, service runs without critical exceptions

### Previous Optimizations Maintained
- **Cache-only system**: CoinGecko optimization preventing 429 rate limit errors
- **Production ready**: System optimized for deployment on Hetzner Cloud server
- **Multi-stage detection**: Complete 4-stage pre-pump detection pipeline

## Changelog
- June 15, 2025: Initial setup and critical bug fixes

## User Preferences

Preferred communication style: Simple, everyday language.