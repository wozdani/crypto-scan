# Crypto Pre-Pump Detection System

A sophisticated cryptocurrency market scanner that detects pre-pump signals using advanced multi-stage analysis.

## Features

- **Multi-Stage Detection**: 4-stage pre-pump analysis (-2.1, -2.2, -1, 1g)
- **PPWCS Scoring**: Pre-Pump Weighted Composite Score (0-100 points)
- **Real-time Alerts**: Telegram notifications with Polish language support
- **GPT Analysis**: AI-powered signal quality assessment for high-confidence alerts
- **Advanced Detectors**: 
  - Heatmap Exhaustion Detection
  - Orderbook Spoofing Detection
  - VWAP Pinning Analysis
  - Volume Cluster Slope Analysis
- **Take Profit Engine**: Dynamic TP level forecasting
- **Web Dashboard**: Real-time monitoring interface

## Production Structure

### Core Files
- `app.py` - Web dashboard application
- `crypto_scan_service.py` - Main scanning service
- `models.py` - Database models

### Detection Modules
- `stages/` - Multi-stage detection algorithms
- `utils/` - Core utilities and detectors

### Frontend
- `templates/` - HTML templates
- `static/` - CSS/JavaScript assets

## Environment Variables Required

```
DATABASE_URL=postgresql://...
BYBIT_API_KEY=your_bybit_api_key
BYBIT_SECRET_KEY=your_bybit_secret_key
OPENAI_API_KEY=your_openai_api_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

## Deployment

1. Set up PostgreSQL database
2. Configure environment variables
3. Run database migrations
4. Start both services:
   - Web dashboard: `python app.py`
   - Scanner service: `python crypto_scan_service.py`

## Alert System

- **Strong Alerts** (PPWCS ≥ 80): Immediate Telegram + GPT analysis
- **Active Alerts** (PPWCS 70-79): Standard Telegram notifications
- **Watchlist** (PPWCS 60-69): Logged for monitoring

## GPT Integration

High-confidence signals (PPWCS ≥ 80) receive automated GPT analysis with:
- Risk assessment
- Confidence levels
- Price predictions
- Entry recommendations
- Quality scoring (0-100)