# 📊 CapeTrader XAU/USD & USD Intraday Quantitative Intelligence Bot

A production-grade quantitative intraday trading intelligence engine and 24/7 cloud-hosted Discord bot for **XAU/USD (Spot Gold)** and the **US Dollar (DXY / US10Y)**. Powered by a 3-timeframe institutional hierarchy (H1 Macro Trend, M15 Structure/FVG, M5 Sniper Trigger), an automated 3-strategy scanner, a paper trading auto-journal with win/loss tracking, and a built-in Keep-Alive Flask web server for 24/7 cloud hosting on Render, Replit, or Railway.

---

## 🚀 Key Features & Architecture

### 1. Interactive Discord Bot (`discord.py`)
Featuring 7 institutional slash commands registered under `@bot.tree.command`:

| Slash Command | Parameters | Description |
|---|---|---|
| `/predict` | *None* | Runs the complete 3-timeframe pipeline, projects the next 1–4 candles with AI/Quant bias, attaches a live high-res M5 candlestick chart, and triggers role pings for high-conviction setups (`>=8/10`). |
| `/levels` | *None* | Displays live XAU/USD price, Previous Day High/Low (PDH/PDL), Asian Session High/Low (01:00–09:00 CAT), Day-anchored VWAP, 20/50 EMAs, RSI, and ATR buffers. |
| `/news` | *None* | Scrapes today's USD High/Medium impact economic events (with minutes remaining and Red-Folder warnings) plus breaking headlines from ForexLive and FXStreet. |
| `/chart` | `timeframe` (5m, 15m, 1h) | Generates and uploads a dark-themed candlestick chart overlaid with 20/50 EMAs, VWAP, PDH/PDL, and Asian High/Low levels. |
| `/calc` | `balance`, `risk_percent`, `stop_loss_dollars` (optional) | Calculates exact MT5 contract sizing for Spot Gold (Standard, Mini, and Micro Lots), Max Dollar Risk, 1:2 / 1:3 RRR profit projections, and minimum lot size warnings. If stop-loss is omitted, live 1.5x M5 & M15 ATR distances are auto-calculated. |
| `/strategies` | *None* | Scans live M5, M15, and H1 data across 3 institutional strategies (Asian Judas Swing, Breakout & Retest, EMA/VWAP Trend Scalper) and reports Active Setups, Watch Zones, and Inactive states. |
| `/track` | `action` (open, close, stats), `strategy` (1, 2, 3), `direction`, `stop_loss`, `take_profit`, `trade_id`, `result` | Paper Trading & Auto-Journaling engine. Logs paper trades into an SQLite database (`journal.db`), monitors them continuously, and calculates strategy win rate % and net points. |

---

### 2. Automated 3-Strategy Institutional Engine (`StrategyScanner`)
- **Strategy 1: Asian Session Liquidity Sweep (Judas Swing)**
  - *Trigger:* Price wicks above Asian High / PDH (Bearish Sweep) or below Asian Low / PDL (Bullish Sweep), closes back inside the range on M5/M15, and confirms an opposite rejection candle or FVG fill.
  - *Targets:* Entry at market execution, tight stop beyond the sweep wick, and target at VWAP / opposite side of the Asian range.
- **Strategy 2: London/NY Breakout & Retest**
  - *Trigger:* Strong M15 candle body closes cleanly above Asian High/PDH or below Asian Low/PDL with 20 EMA and VWAP alignment, followed by an M5 pullback to retest the broken level.
  - *Targets:* Retest entry zone, stop-loss 1.5x M5 ATR back inside the broken level, 1:2 RRR target.
- **Strategy 3: EMA 20/50 + VWAP Trend Pullback Scalper**
  - *Trigger:* H1 and M15 in clean trend alignment (Price > VWAP > 20 EMA > 50 EMA for Bullish; inverse for Bearish), M5 RSI pulls back to 40–52 (bullish) or 48–60 (bearish), and price prints a rejection/bounce candle near the 20 EMA.
  - *Targets:* Sniper entry, 1.5x M5 ATR stop-loss, 1:2 and 1:3 RRR take-profit targets.

---

### 3. Paper Trading & Auto-Journaling System (`TradeJournal`)
- **SQLite Database (`journal.db`)**: Thread-safe database storing Trade ID, Strategy ID, Direction (BUY/SELL), Entry Price, Stop-Loss, Take-Profit, Status (OPEN, WON, LOST), Open/Close Timestamps, and Net Gold Points PnL.
- **Automatic Hit/Miss Detection (15m Background Loop)**:
  - During every 15-minute candle close scan, the bot checks all `OPEN` trades against the candle's High, Low, and Close.
  - If Take-Profit is crossed, the trade is automatically marked as **WON** and a celebratory green embed (`🎉 TAKE-PROFIT HIT!`) is dispatched to the alert channel.
  - If Stop-Loss is breached, the trade is automatically marked as **LOST** and an alert red embed (`🛑 STOP-LOSS HIT`) is dispatched.
- **Strategy Win Rate Breakdown (`/track action:stats`)**:
  - Displays overall win rate %, total closed trades, cumulative Gold points PnL, and individual performance cards for Strategy 1, Strategy 2, and Strategy 3.
  - Automatically awards the **Top Performing Strategy** banner to the highest win-rate setup.

---

### 4. 24/7 Cloud Deployment (Render, Replit, Railway) & Keep-Alive Server
The bot features a built-in lightweight **Flask Keep-Alive web server** that runs in a background daemon thread:
- Binds to host `0.0.0.0` and uses the dynamic port provided by `PORT` (default `8080`).
- Root Route (`GET /`): Returns `CapeTrader XAU/USD Intelligence Bot is Alive & Running 24/7 🚀`.
- Health Check (`GET /health`): Returns JSON health status and UTC timestamp.
- **Procfile** included: `web: python xauusd_trading_bot.py`.
- **Render Blueprint** included: `render.yaml`.
- Seamless **yfinance fallback** when deployed on Linux cloud environments where the desktop MetaTrader 5 terminal is unavailable (`DATA_SOURCE = "AUTO"` or `"YFINANCE"`).

#### Free 24/7 Uptime with UptimeRobot:
1. Deploy the service to [Render.com](https://render.com) as a Web Service.
2. Copy your Render web service URL (e.g., `https://xauusd-trading-bot-xxxx.onrender.com`).
3. Create a free account at [UptimeRobot.com](https://uptimerobot.com).
4. Add a new monitor:
   - **Monitor Type:** `HTTP(s)`
   - **Friendly Name:** `CapeTrader Gold Bot`
   - **URL (or IP):** `https://xauusd-trading-bot-xxxx.onrender.com/`
   - **Monitoring Interval:** `5 minutes`
5. UptimeRobot will ping the Keep-Alive server every 5 minutes, preventing Render's free tier container from sleeping.

---

## 📦 Installation & Configuration

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Create a `.env` file in the project root:
```ini
# Market Data Mode: "AUTO" (prefers MT5 on Windows, falls back to yfinance), "MT5", or "YFINANCE"
DATA_SOURCE=AUTO
MT5_SYMBOL=XAUUSD

# Discord Bot Credentials
DISCORD_BOT_TOKEN=your_bot_token_here
DISCORD_ALERT_CHANNEL_ID=your_channel_id_here
DISCORD_ALERT_ROLE_ID=your_role_id_here
DISCORD_ALERT_USER_ID=your_user_id_here

# Keep-Alive Web Server (Dynamic Cloud Port)
PORT=8080

# AI LLM Keys (Optional - falls back to deterministic quant engine if omitted)
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.8-flash

OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o

# Webhook & Telegram Fallback (Optional)
DISCORD_WEBHOOK_URL=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

---

## 🏃 Execution Commands

### Diagnostic CLI Single Scan:
```bash
python xauusd_trading_bot.py --once
```

### Run Bot Locally (Discord Bot + 15m Loop + Keep-Alive Web Server):
```bash
python xauusd_trading_bot.py
```

---

## 📊 Discord Bot Usage Guide

### 🧮 Calculating Risk & MT5 Lot Size
```
/calc balance:500 risk_percent:1.0
```
*Calculates exact lot size using live 1.5x M5 & M15 ATR distances.*

```
/calc balance:1000 risk_percent:2.0 stop_loss_dollars:4.50
```
*Calculates contract size for a custom $4.50 stop distance.*

### 🔍 Scanning Institutional Setups
```
/strategies
```
*Scans Judas Swing, Breakout & Retest, and Trend Pullback.*

### 📝 Paper Trading & Auto-Journaling
```
/track action:Open strategy:1
```
*Opens a new paper trade for Strategy 1 using live price and ATR parameters.*

```
/track action:Stats
```
*Views the comprehensive performance journal, win rates, and open positions.*

```
/track action:Close trade_id:1 result:WIN
```
*Manually closes paper trade #1 with a winning result.*

---

## 🛡️ License & Disclaimer
This software is provided for educational and quantitative research purposes only. Trading Spot Gold (XAU/USD) carries significant financial risk. Always practice sound risk management and never risk more capital than you can afford to lose.
