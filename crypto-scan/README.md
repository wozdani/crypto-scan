# Crypto Pre-Pump Scanner & Trend Mode

Advanced cryptocurrency market scanner with PPWCS v2.8+ scoring system and PPWCS-T 2.0 trend mode for detecting pre-pump signals and stable growth patterns.

## System Architecture

### Core Components
- **Pre-Pump Detection**: Multi-stage analysis pipeline with 4-stage detection system
- **Trend Mode v1.0**: Professional trend continuation detection with EMA alignment
- **PPWCS-T 2.0**: Trend mode boost logic for stable growth without breakouts (0-20 points)
- **Real-time Dashboard**: Flask web interface with live market monitoring

### Key Features
- **PPWCS v2.8 Scoring**: Advanced weighted composite scoring (0-242+ points)
- **Multi-Stage Pipeline**: Stage -2.1 (micro-anomaly), -2.2 (news/tags), -1 (compression), 1G (breakout)
- **Trend Enhancement**: 4 advanced trend detectors with breakout exclusion filter
- **AI Integration**: GPT-4 analysis for high-confidence signals
- **Multi-chain Support**: Ethereum, BSC, Polygon, Arbitrum, Optimism

## Quick Start

```bash
cd crypto-scan
uv run python app.py        # Start dashboard (port 5000)
python crypto_scan_service.py  # Start scanner service
```

## Configuration

Required environment variables in `.env`:
```
OPENAI_API_KEY=your_openai_key
TELEGRAM_BOT_TOKEN=your_telegram_token  
TELEGRAM_CHAT_ID=your_chat_id
BYBIT_API_KEY=your_bybit_key
BYBIT_SECRET_KEY=your_bybit_secret
```

## API Endpoints

- `GET /` - Dashboard interface
- `GET /api/status` - System status
- `GET /api/recent-alerts` - Recent pre-pump alerts
- `GET /api/trend-alerts` - Trend mode alerts
- `GET /api/market-overview` - Market statistics

## Data Storage

- **Cache**: `cache/` - CoinGecko token cache and temporary data
- **Reports**: `reports/` - Historical scan reports and signal data
- **Data**: `data/` - Persistent storage for alerts and analysis

## Dependencies

See `pyproject.toml` for complete dependency list. Key packages:
- Flask & Flask-SQLAlchemy for web framework
- OpenAI for GPT integration
- Requests for API calls
- NumPy for numerical computations