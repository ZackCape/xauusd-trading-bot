# 📊 XAU/USD & USD Intraday Quantitative Intelligence Bot

A production-ready automated quantitative intraday trading intelligence engine for **XAU/USD (Spot Gold)** and the **US Dollar (DXY / US10Y)** featuring a full interactive **Discord Bot (`discord.py`)** with Slash Commands, rich color-coded Embeds, and synchronized 15-minute background forecasting.

---

## 🚀 Key Features & Architecture

### 1. Interactive Discord Bot (`discord.py`)
- **Slash Commands (`@bot.tree.command`)**:
  - `/predict` : Runs the full live M15/H1 Gold + DXY/US10Y + News pipeline and replies with a rich color-coded Embed showing the forecast, institutional key levels, detected candlestick patterns, and recommended ATR stop buffer.
  - `/levels` : Displays live XAU/USD price, Previous Day High/Low (PDH/PDL), Asian Session High/Low (01:00-09:00 CAT), Day-anchored VWAP, 20/50 EMAs, 14m RSI, and 15m/1h ATR.
  - `/news` : Displays today's USD High/Medium impact economic calendar releases (with minutes remaining) and breaking headlines from ForexLive and FXStreet.
- **Color-Coded Visual Sentiment**:
  - 🟢 **Green (`0x2ECC71`)**: BULLISH Bias
  - 🔴 **Red (`0xE74C3C`)**: BEARISH Bias
  - 🟡 **Gold (`0xF1C40F`)**: RANGE-BOUND Bias
  - 🟠 **Orange (`0xE67E22`)**: NO-TRADE NEWS RISK / Red-Folder Warning
- **Role Pings**:
  - Configurable `DISCORD_ALERT_ROLE_ID` to notify a trading team or VIP role when a high-conviction setup (`>=8/10` probability) or Red-Folder news risk is triggered.
- **Non-Blocking Architecture**:
  - All heavy network requests (MetaTrader 5, Yahoo Finance, Forex Factory, LLM API calls) are wrapped in `asyncio.to_thread()` to ensure the Discord gateway heartbeat is never blocked.
  - Commands use `interaction.response.defer(thinking=True)` to prevent 3-second interaction timeouts.

---

### 2. Dual-Mode Market Data Engine
- **MetaTrader 5 Terminal Bridge (`DATA_SOURCE = "AUTO"` | `"MT5"`)**:
  - Zero-latency broker ticks and M15/H1/D1 bars directly from your desktop MT5 terminal.
  - Automatically resolves broker symbol naming variations (`XAUUSD`, `GOLD`, `XAUUSD.m`, `XAUUSD.pro`).
- **Yahoo Finance Fallback (`yfinance`)**:
  - Seamlessly falls back to `GC=F` (Gold Futures) or `XAUUSD=X` if MetaTrader 5 is not running or fails to initialize.
  - Live correlation feeds for the **US Dollar Index (`DX-Y.NYB`)** and **US 10-Year Treasury Yields (`^TNX`)**.
- **Institutional Mathematical Indicators & SMC Levels**:
  - 14-period RSI (Wilder's Smoothing), 20 EMA, 50 EMA, Intraday Day-anchored VWAP, 14-period ATR.
  - Previous Day High (PDH) & Previous Day Low (PDL).
  - Asian Session High/Low (01:00 to 09:00 CAT / UTC+2).
  - Smart Money Concepts (SMC) & Price Action Detector: Bullish/Bearish Engulfing, Hammer/Pinbars, Fair Value Gap (FVG) imbalances, and Liquidity Sweeps above/below Asian H/L or PDH/PDL.

---

### 3. Economic Calendar & News Scraper
- **Forex Factory Calendar JSON**:
  - Scrapes `https://nfs.faireconomy.media/ff_calendar_thisweek.json` with an in-memory cache to prevent 429 rate limits.
  - Filters today's **High** and **Medium** impact USD events with exact minutes remaining.
  - **Red-Folder Alert**: Detects CPI, NFP, PPI, FOMC, or Rate Decisions within 30 minutes.
- **Breaking Forex Headlines**:
  - Scrapes and parses latest news from `ForexLive` and `FXStreet` RSS feeds.

---

### 4. Hybrid Prediction Engine (AI LLM + Quant Fallback)
- **AI LLM API Integration**:
  - Supports **Google Gemini (`google-genai`)** and **OpenAI (`openai`)** via `.env`.
  - Analytical rules:
    1. **Inverse Macro Rule**: Rising DXY + US10Y = Bearish Gold. Falling DXY + US10Y = Bullish Gold.
    2. **Liquidity Sweep Rule**: Rejection of Asian High/PDH with high RSI = Bearish reversal. Rejection of Asian Low/PDL with low RSI = Bullish reversal.
    3. **Red-Folder Rule**: High impact event within 30m overrides bias to `"HIGH VOLATILITY WHIPSAW - STAND ASIDE"`.
- **Deterministic Rule-Based Fallback**:
  - Operates automatically when no LLM API key is provided, generating the exact required institutional forecast format.

---

### 5. Automated 15-Minute Background Scheduler & Webhook Fallback
- Synchronizes with every 15-minute candle close (`:00:05`, `:15:05`, `:30:05`, `:45:05`).
- Pushes the forecast Embed to `DISCORD_ALERT_CHANNEL_ID`.
- Forwards to `DISCORD_WEBHOOK_URL` and Telegram if configured.
- Supports `--once` CLI flag for quick diagnostics.

---

## 📦 Installation & Setup

### 1. Install Dependencies
```bash
pip install pandas numpy yfinance requests feedparser schedule python-dotenv discord.py MetaTrader5 google-genai openai
```

### 2. Configure Environment (`.env`)
Edit `.env` with your credentials:
```ini
DATA_SOURCE=AUTO
MT5_SYMBOL=XAUUSD

# Discord Bot Credentials
DISCORD_BOT_TOKEN=your_bot_token_here
DISCORD_ALERT_CHANNEL_ID=123456789012345678
DISCORD_ALERT_ROLE_ID=987654321098765432

# Webhook & Telegram Fallback (Optional)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# AI Keys (Optional)
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash

OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o
```

---

## 🏃 Execution Commands

### Diagnostic CLI Run (Single Pass):
```bash
python xauusd_trading_bot.py --once
```

### Run Interactive Discord Bot (24/7 Slash Commands & 15m Alerts):
```bash
python xauusd_trading_bot.py
```
*(If no `DISCORD_BOT_TOKEN` is found in `.env`, the script automatically falls back to standalone terminal and webhook loop mode).*
