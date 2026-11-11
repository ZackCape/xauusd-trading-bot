#!/usr/bin/env python3
"""
========================================================================================
XAU/USD (Gold) & USD Intraday Trading Intelligence Bot
========================================================================================
Production-ready automated quantitative intelligence engine analyzing live candlesticks,
institutional liquidity levels (PDH/PDL, Asian Session), macro correlations (DXY, US10Y),
and breaking forex news.

Features:
  - 3-Timeframe Analytical Hierarchy:
      * H1 (1-Hour)   : Main Trend Direction & Dominant Order Flow
      * M15 (15-Minute): Setup & Zone Confirmation (FVGs, Liquidity Sweeps)
      * M5 (5-Minute)  : Sniper Entry Trigger & Tight Scalp Stop-Loss (1.5x M5 ATR)
  - Full Interactive discord.py Bot with Slash Commands:
      * /predict    : Full 3-timeframe forecast with auto-attached live M5 candlestick chart
      * /levels     : Live price, PDH/PDL, Asian High/Low, and M5/M15/H1 technical indicators
      * /news       : High/Medium impact USD calendar countdowns and breaking headlines
      * /chart      : Live high-res dark-themed candlestick chart (5m, 15m, 1h) with key levels
      * /calc       : MT5 Spot Gold Lot Size & Risk Calculator (Standard, Mini, Micro with RRR)
      * /strategies : Automated 3-Strategy Institutional Scanner (Judas Swing, Breakout, Scalper)
  - Automated 3-Strategy Detector Engine (StrategyScanner):
      * Strategy 1: Asian Session Liquidity Sweep & Reversal (Judas Swing)
      * Strategy 2: London/NY Breakout & Retest
      * Strategy 3: EMA 20/50 + VWAP Trend Pullback Scalper
  - Rich color-coded discord.Embed visual reports (Green, Red, Gold, Orange)
  - Synchronized 15-minute background loop (:00, :15, :30, :45) with auto-attached chart
  - Non-blocking async execution (asyncio.to_thread)
  - High-conviction (>=8/10), Red-Folder, and Active Strategy role/user pings
  - Built-in Lightweight Keep-Alive Flask Server for 24/7 Hosting (Replit, Render, Railway, UptimeRobot)
  - CLI diagnostic mode (--once) and Webhook/Telegram fallback
========================================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import io
import json
import logging
import os
import re
import sqlite3
import sys
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# Headless matplotlib configuration before pyplot import
import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

try:
    import mplfinance as mpf
    HAS_MPLFINANCE = True
except ImportError:
    HAS_MPLFINANCE = False
    mpf = None

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Interactive Discord Bot Framework
try:
    import discord
    from discord import app_commands
    from discord.ext import commands, tasks
    HAS_DISCORD = True
except ImportError:
    HAS_DISCORD = False

# Optional Feedparser
try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False

# Optional Schedule
try:
    import schedule
    HAS_SCHEDULE = True
except ImportError:
    HAS_SCHEDULE = False

# Optional MetaTrader 5
try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except (ImportError, Exception):
    HAS_MT5 = False

# Optional Google GenAI SDK
try:
    from google import genai
    from google.genai import types as genai_types
    HAS_GOOGLE_GENAI = True
except (ImportError, Exception):
    HAS_GOOGLE_GENAI = False

# Optional OpenAI SDK
try:
    import openai
    HAS_OPENAI = True
except (ImportError, Exception):
    HAS_OPENAI = False

# Optional pandas_ta
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except (ImportError, Exception):
    HAS_PANDAS_TA = False

# Optional yfinance
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

# Optional Flask (Keep-Alive Web Server for 24/7 Cloud Hosting)
try:
    from flask import Flask
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

# Ensure standard output handles UTF-8 symbols and emojis on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# --------------------------------------------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("XAUUSD_Intelligence")

# --------------------------------------------------------------------------------------
# GLOBAL CONFIGURATION CONSTANTS
# --------------------------------------------------------------------------------------
DATA_SOURCE = os.getenv("DATA_SOURCE", "AUTO").upper()  # "MT5", "YFINANCE", or "AUTO"

# MT5 Settings
MT5_PATH = os.getenv("MT5_PATH", "")
MT5_LOGIN = os.getenv("MT5_LOGIN", "")
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
MT5_SYMBOL = os.getenv("MT5_SYMBOL", "XAUUSD")

# yfinance Symbols
YF_GOLD_TICKERS = ["GC=F", "XAUUSD=X"]
YF_DXY_TICKER = os.getenv("YF_DXY_TICKER", "DX-Y.NYB")
YF_US10Y_TICKER = os.getenv("YF_US10Y_TICKER", "^TNX")

# Discord Bot Settings
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "").strip()
DISCORD_ALERT_CHANNEL_ID = os.getenv("DISCORD_ALERT_CHANNEL_ID", "").strip()
DISCORD_ALERT_ROLE_ID = os.getenv("DISCORD_ALERT_ROLE_ID", "").strip()
DISCORD_ALERT_USER_ID = os.getenv("DISCORD_ALERT_USER_ID", "").strip()

# Paper Trading & Auto-Journaling Settings
JOURNAL_DB_PATH = os.getenv("JOURNAL_DB_PATH", "journal.db").strip()

# Webhooks & Telegram Fallback
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# LLM Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Exact System Prompt enforcing 3-Timeframe Analytical Hierarchy
SYSTEM_PROMPT = """You are an institutional XAU/USD (Spot Gold) intraday quantitative analyst. Analyze the provided JSON payload containing 5m, 15m, and 1h OHLCV metrics for XAU/USD, DXY, and US10Y, key liquidity levels (PDH/PDL, Asian High/Low), and breaking Forex news to project the next 1 to 4 candlesticks.

Follow this strict 3-Timeframe Analytical Hierarchy:
1. H1 (1-Hour): Establishes the macro trend direction and dominant order flow.
2. M15 (15-Minute): Identifies structural setups, Fair Value Gaps (FVGs), and institutional liquidity zones.
3. M5 (5-Minute): Pinpoints sniper entry triggers, immediate candle patterns (engulfing/pinbars), and tight scalping stop-losses (1.5x M5 ATR).

Strict Analytical Rules:
1. Inverse Macro Rule: Rising DXY + Rising US10Y = Bearish pressure on XAU/USD. Falling DXY + Falling US10Y = Bullish pressure on XAU/USD.
2. Liquidity Sweep Rule: If XAU/USD swept Asian High or PDH while RSI > 70 and DXY is bouncing, project a bearish reversal/rejection candle back into the range. Reverse for bullish sweeps of Asian Low/PDL.
3. Red-Folder Rule: If a High-Impact USD event (CPI, NFP, PPI, FOMC) is scheduled within 30 minutes, override directional bias to "HIGH VOLATILITY WHIPSAW - STAND ASIDE".

Respond strictly in this format:
📊 **XAU/USD INTRADAY FORECAST (Next 1-4 Candles)**
• **Bias:** [BULLISH / BEARISH / RANGE-BOUND / NO-TRADE NEWS RISK]
• **Probability Score:** [X/10]
• **USD & Macro Driver:** [1 sentence on DXY/US10Y movement + headline sentiment]
• **Candlestick Projection:** [Specific prediction of next M5/M15 candle behavior, wicks, or FVG fill]
• **Key Zones to Watch:**
  - Support / Bullish Target: $[Price]
  - Resistance / Bearish Target: $[Price]
  - Recommended M5 ATR Stop Buffer: $[M5 ATR * 1.5]
  - Recommended M15 ATR Stop Buffer: $[M15 ATR * 1.5]
"""


# ======================================================================================
# MODULE 1: DUAL-MODE MARKET DATA ENGINE (MT5 + YFINANCE FALLBACK)
# ======================================================================================

class MarketDataEngine:
    """
    Handles live market data ingestion from MetaTrader 5 with seamless fallback
    to Yahoo Finance for M5, M15, H1, and D1 timeframes. Computes technical indicators,
    institutional levels, and price-action candle patterns.
    """

    _last_known_candles: Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]] = None

    def __init__(self, mode: str = "AUTO"):
        self.mode = mode.upper()
        self.mt5_initialized = False
        self.active_gold_symbol = MT5_SYMBOL
        self._init_mt5_if_requested()

    def _init_mt5_if_requested(self) -> bool:
        """Attempt to initialize MetaTrader 5 if mode is MT5 or AUTO."""
        if self.mode not in ["MT5", "AUTO"]:
            logger.info("Market data mode is '%s'. Skipping MT5 initialization.", self.mode)
            return False

        if not HAS_MT5:
            logger.warning("MetaTrader5 package is not available on this system.")
            return False

        try:
            init_kwargs = {}
            if MT5_PATH:
                init_kwargs["path"] = MT5_PATH
            if MT5_LOGIN and MT5_LOGIN.isdigit():
                init_kwargs["login"] = int(MT5_LOGIN)
            if MT5_PASSWORD:
                init_kwargs["password"] = MT5_PASSWORD
            if MT5_SERVER:
                init_kwargs["server"] = MT5_SERVER

            if init_kwargs:
                self.mt5_initialized = bool(mt5.initialize(**init_kwargs))
            else:
                self.mt5_initialized = bool(mt5.initialize())

            if self.mt5_initialized:
                try:
                    term_info = mt5.terminal_info()
                    logger.info("MetaTrader 5 initialized successfully! Terminal: %s", term_info)
                except Exception:
                    logger.info("MetaTrader 5 initialized successfully!")
                self.active_gold_symbol = self._resolve_mt5_symbol(MT5_SYMBOL)
                return True
            else:
                try:
                    err_code, err_desc = mt5.last_error()
                    logger.warning(
                        "MetaTrader 5 initialization failed: (%d) %s. Graceful fallback to Yahoo Finance.",
                        err_code, err_desc
                    )
                except Exception:
                    logger.warning("MetaTrader 5 initialization returned False. Fallback to Yahoo Finance.")
                return False
        except Exception as exc:
            logger.warning("Error during MetaTrader 5 initialization: %s. Fallback to Yahoo Finance.", exc)
            self.mt5_initialized = False
            return False

    def _resolve_mt5_symbol(self, preferred_symbol: str) -> str:
        """Find the matching Gold symbol in broker's Market Watch with fallback safety."""
        try:
            candidates = [preferred_symbol, "XAUUSD", "GOLD", "XAUUSD.m", "XAUUSD.pro", "XAUUSDraw"]
            for sym in candidates:
                try:
                    info = mt5.symbol_info(sym)
                    if info is not None:
                        if not info.visible:
                            mt5.symbol_select(sym, True)
                        logger.info("Selected MT5 symbol: %s", sym)
                        return sym
                except Exception:
                    pass

            all_symbols = mt5.symbols_get()
            if all_symbols:
                for s in all_symbols:
                    if "XAU" in s.name or "GOLD" in s.name.upper():
                        try:
                            mt5.symbol_select(s.name, True)
                            logger.info("Discovered and selected MT5 symbol: %s", s.name)
                            return s.name
                        except Exception:
                            pass
        except Exception as exc:
            logger.warning("Minor error resolving MT5 symbol: %s", exc)

        return preferred_symbol

    def fetch_mt5_ohlcv(self, timeframe_m: int, count: int = 200) -> Optional[pd.DataFrame]:
        """Fetch OHLCV candlestick data directly from MetaTrader 5 with live tick alignment."""
        if not self.mt5_initialized:
            return None

        tf_map = {
            5: mt5.TIMEFRAME_M5,
            15: mt5.TIMEFRAME_M15,
            60: mt5.TIMEFRAME_H1,
            1440: mt5.TIMEFRAME_D1,
        }
        mt5_tf = tf_map.get(timeframe_m, mt5.TIMEFRAME_M15)

        try:
            # Lookback count of at least 150-200 bars captures full session down to latest minute
            rates = mt5.copy_rates_from_pos(self.active_gold_symbol, mt5_tf, 0, count)
            if rates is None or len(rates) == 0:
                logger.warning("MT5 copy_rates_from_pos returned empty data for %s", self.active_gold_symbol)
                return None

            df = pd.DataFrame(rates)
            if df.empty or "time" not in df.columns:
                return None

            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.set_index("time")
            df = df.rename(columns={"tick_volume": "volume"})
            df.columns = [c.lower() for c in df.columns]

            # Synchronize active candle close/high/low with live tick if available
            try:
                tick = mt5.symbol_info_tick(self.active_gold_symbol)
                if tick and tick.bid > 0:
                    live_bid = float(tick.bid)
                    if live_bid > df.iloc[-1]["high"]:
                        df.iloc[-1, df.columns.get_loc("high")] = live_bid
                    if live_bid < df.iloc[-1]["low"]:
                        df.iloc[-1, df.columns.get_loc("low")] = live_bid
                    df.iloc[-1, df.columns.get_loc("close")] = live_bid
            except Exception as tick_err:
                logger.debug("Minor tick retrieval issue: %s", tick_err)

            return df[["open", "high", "low", "close", "volume"]].copy()
        except Exception as exc:
            logger.warning("Failed to fetch MT5 rates for %s: %s", self.active_gold_symbol, exc)
            return None

    def fetch_yfinance_ohlcv(self, ticker: str, interval: str = "15m", period: str = "5d") -> pd.DataFrame:
        """Fetch OHLCV candlestick data via Yahoo Finance with MultiIndex normalization."""
        if not HAS_YFINANCE:
            raise RuntimeError("yfinance is required for market data fallback. Please install yfinance.")

        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            raise ValueError(f"No data returned from yfinance for ticker {ticker}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [str(col).lower() for col in df.columns]

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        required_cols = ["open", "high", "low", "close"]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Missing expected column '{col}' in yfinance data for {ticker}")

        if "volume" not in df.columns:
            df["volume"] = 0.0

        return df[["open", "high", "low", "close", "volume"]].copy()

    def get_gold_candles(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
        """
        Pulls M5, M15, H1, and D1 data for Gold.
        Returns (df_m5, df_m15, df_h1, df_d1, source_name).
        Guarantees zero-crash fallback even during network drops.
        """
        # 1. Try MT5 if requested and active
        if self.mt5_initialized and self.mode in ["MT5", "AUTO"]:
            try:
                df_m5 = self.fetch_mt5_ohlcv(5, count=200)
                df_m15 = self.fetch_mt5_ohlcv(15, count=200)
                df_h1 = self.fetch_mt5_ohlcv(60, count=200)
                df_d1 = self.fetch_mt5_ohlcv(1440, count=60)
                if df_m5 is not None and df_m15 is not None and df_h1 is not None and not df_m15.empty:
                    logger.info("Using live MetaTrader 5 feed for %s", self.active_gold_symbol)
                    candles = (df_m5, df_m15, df_h1, df_d1 if df_d1 is not None else df_h1, f"MT5:{self.active_gold_symbol}")
                    MarketDataEngine._last_known_candles = candles
                    return candles
                else:
                    logger.warning("MT5 returned empty or incomplete candles. Falling back to yfinance.")
            except Exception as mt5_err:
                logger.warning("MT5 data retrieval error: %s. Falling back to yfinance.", mt5_err)

        # 2. Fallback to yfinance
        logger.info("Using Yahoo Finance fallback for Gold data.")
        last_error = None
        for ticker in YF_GOLD_TICKERS:
            try:
                df_m5 = self.fetch_yfinance_ohlcv(ticker, interval="5m", period="5d")
                df_m15 = self.fetch_yfinance_ohlcv(ticker, interval="15m", period="5d")
                df_h1 = self.fetch_yfinance_ohlcv(ticker, interval="1h", period="1mo")
                df_d1 = self.fetch_yfinance_ohlcv(ticker, interval="1d", period="3mo")
                if not df_m5.empty and not df_m15.empty:
                    candles = (df_m5, df_m15, df_h1, df_d1, f"yfinance:{ticker}")
                    MarketDataEngine._last_known_candles = candles
                    return candles
            except Exception as exc:
                last_error = exc
                logger.warning("yfinance fetch failed for %s: %s. Trying alternative...", ticker, exc)

        # 3. Resilient cache fallback to preserve uptime during momentary network blips
        if MarketDataEngine._last_known_candles is not None:
            logger.warning("Both MT5 and yfinance unavailable. Returning last known cached candles to prevent downtime.")
            return MarketDataEngine._last_known_candles

        raise RuntimeError(f"Failed to fetch Gold candlestick data from any source. Error: {last_error}")

    def get_macro_correlation_data(self) -> Dict[str, Any]:
        """
        Pulls live M15 and H1 data for DXY (US Dollar Index) and US10Y (10-Year Treasury Yields).
        Computes price, net change, and prevailing intraday momentum.
        """
        result = {
            "dxy": {"price": 100.0, "change_pct_45m": 0.0, "trend": "NEUTRAL", "status": "AVAILABLE"},
            "us10y": {"yield": 4.5, "change_pct_45m": 0.0, "trend": "NEUTRAL", "status": "AVAILABLE"}
        }

        # DXY Data
        try:
            df_dxy = self.fetch_yfinance_ohlcv(YF_DXY_TICKER, interval="15m", period="5d")
            closes = df_dxy["close"].dropna()
            latest_dxy = float(closes.iloc[-1])
            prev_dxy = float(closes.iloc[-4]) if len(closes) >= 4 else float(closes.iloc[0])
            dxy_chg = ((latest_dxy - prev_dxy) / prev_dxy) * 100.0

            if dxy_chg > 0.08:
                dxy_trend = "RISING / BOUNCING"
            elif dxy_chg < -0.08:
                dxy_trend = "FALLING / SOFTENING"
            else:
                dxy_trend = "CONSOLIDATING"

            result["dxy"] = {
                "price": round(latest_dxy, 3),
                "change_pct_45m": round(dxy_chg, 3),
                "trend": dxy_trend,
                "status": "LIVE"
            }
        except Exception as exc:
            logger.warning("Could not fetch DXY correlation data: %s", exc)
            result["dxy"]["status"] = f"ERROR: {exc}"

        # US10Y Yield Data
        try:
            df_tnx = self.fetch_yfinance_ohlcv(YF_US10Y_TICKER, interval="15m", period="5d")
            closes_tnx = df_tnx["close"].dropna()
            latest_tnx = float(closes_tnx.iloc[-1])
            prev_tnx = float(closes_tnx.iloc[-4]) if len(closes_tnx) >= 4 else float(closes_tnx.iloc[0])
            tnx_chg = ((latest_tnx - prev_tnx) / prev_tnx) * 100.0

            if tnx_chg > 0.20:
                tnx_trend = "RISING"
            elif tnx_chg < -0.20:
                tnx_trend = "FALLING"
            else:
                tnx_trend = "STABLE"

            result["us10y"] = {
                "yield": round(latest_tnx, 3),
                "change_pct_45m": round(tnx_chg, 3),
                "trend": tnx_trend,
                "status": "LIVE"
            }
        except Exception as exc:
            logger.warning("Could not fetch US10Y correlation data: %s", exc)
            result["us10y"]["status"] = f"ERROR: {exc}"

        return result

    @staticmethod
    def calculate_indicators(df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates 14-period RSI, 20 EMA, 50 EMA, VWAP, and 14-period ATR.
        Uses pure pandas vector calculations guaranteeing 100% mathematical accuracy.
        """
        df = df_input.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"].fillna(0.0)

        # 1. RSI (14 periods - Wilder's Smoothing)
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df["rsi_14"] = (100.0 - (100.0 / (1.0 + rs))).bfill().fillna(50.0)

        # 2. 20 EMA and 50 EMA
        df["ema_20"] = close.ewm(span=20, adjust=False).mean()
        df["ema_50"] = close.ewm(span=50, adjust=False).mean()

        # 3. VWAP (Intraday anchored to each trading day)
        tp = (high + low + close) / 3.0
        vol_clean = volume.copy()
        if (vol_clean == 0).all():
            df["vwap"] = df["ema_20"]
        else:
            vol_clean = vol_clean.replace(0, 1)
            pv = tp * vol_clean
            df["vwap"] = (
                pv.groupby(df.index.date).cumsum() / vol_clean.groupby(df.index.date).cumsum()
            ).bfill().fillna(df["ema_20"])

        # 4. ATR (14 periods)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = tr.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean().bfill()

        return df

    @staticmethod
    def calculate_institutional_levels(
        df_intraday: pd.DataFrame, df_daily: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Calculates:
          - Previous Day High (PDH) & Previous Day Low (PDL)
          - Asian Session High & Asian Session Low (01:00 to 09:00 CAT / UTC+2)
        """
        cat_tz = datetime.timezone(datetime.timedelta(hours=2))

        # Convert intraday data to CAT (UTC+2)
        if df_intraday.index.tz is None:
            df_cat = df_intraday.tz_localize("UTC").tz_convert(cat_tz)
        else:
            df_cat = df_intraday.tz_convert(cat_tz)

        # 1. Asian Session High and Low (01:00 to 09:00 CAT)
        now_cat = datetime.datetime.now(cat_tz)
        today_cat = now_cat.date()

        asian_mask = (
            (df_cat.index.date == today_cat)
            & (df_cat.index.time >= datetime.time(1, 0))
            & (df_cat.index.time <= datetime.time(9, 0))
        )
        asian_candles = df_cat[asian_mask]

        if asian_candles.empty or len(asian_candles) < 4:
            unique_dates = sorted(list(set(d for d in df_cat.index.date if d <= today_cat)))
            if len(unique_dates) >= 2:
                prev_date = unique_dates[-2]
                prev_asian_mask = (
                    (df_cat.index.date == prev_date)
                    & (df_cat.index.time >= datetime.time(1, 0))
                    & (df_cat.index.time <= datetime.time(9, 0))
                )
                prev_asian_candles = df_cat[prev_asian_mask]
                if not prev_asian_candles.empty:
                    asian_candles = prev_asian_candles

        if not asian_candles.empty:
            asian_high = float(asian_candles["high"].max())
            asian_low = float(asian_candles["low"].min())
        else:
            recent_8h = df_intraday.tail(32)
            asian_high = float(recent_8h["high"].max())
            asian_low = float(recent_8h["low"].min())

        # 2. Previous Day High (PDH) and Low (PDL)
        if df_daily is not None and len(df_daily) >= 2:
            pdh = float(df_daily["high"].iloc[-2])
            pdl = float(df_daily["low"].iloc[-2])
        else:
            daily_groups = df_cat.groupby(df_cat.index.date)
            dates = sorted(list(daily_groups.groups.keys()))
            if len(dates) >= 2:
                prev_day_df = daily_groups.get_group(dates[-2])
                pdh = float(prev_day_df["high"].max())
                pdl = float(prev_day_df["low"].min())
            else:
                pdh = float(df_intraday["high"].max())
                pdl = float(df_intraday["low"].min())

        return {
            "pdh": round(pdh, 2),
            "pdl": round(pdl, 2),
            "asian_high": round(asian_high, 2),
            "asian_low": round(asian_low, 2),
        }

    @staticmethod
    def detect_candlestick_patterns(
        df: pd.DataFrame,
        pdh: float,
        pdl: float,
        asian_high: float,
        asian_low: float,
        timeframe_label: str = "M15",
    ) -> List[str]:
        """
        Price Action & Candlestick Pattern Detector across the latest 3 candles:
          - Bullish / Bearish Engulfing
          - Pinbar / Hammer / Shooting Star
          - Liquidity Sweep wick above/below Asian High/Low or PDH/PDL
          - 3-candle Fair Value Gap (FVG) imbalance
        """
        if len(df) < 3:
            return []

        c0 = df.iloc[-3]
        c1 = df.iloc[-2]
        c2 = df.iloc[-1]

        patterns: List[str] = []

        # 1. Engulfing Patterns (c1 into c2)
        if c1["close"] < c1["open"] and c2["close"] > c2["open"]:
            if c2["open"] <= c1["close"] * 1.0005 and c2["close"] >= c1["open"] * 0.9995:
                patterns.append(f"{timeframe_label} Bullish Engulfing")
        elif c1["close"] > c1["open"] and c2["close"] < c2["open"]:
            if c2["open"] >= c1["close"] * 0.9995 and c2["close"] <= c1["open"] * 1.0005:
                patterns.append(f"{timeframe_label} Bearish Engulfing")

        # 2. Pinbar / Hammer / Shooting Star (check c2 and c1)
        for label, candle in [("Latest", c2), ("Prior", c1)]:
            rng = float(candle["high"] - candle["low"])
            if rng > 0:
                body = float(abs(candle["close"] - candle["open"]))
                upper_wick = float(candle["high"] - max(candle["open"], candle["close"]))
                lower_wick = float(min(candle["open"], candle["close"]) - candle["low"])

                if lower_wick >= 2.0 * body and body <= 0.35 * rng and upper_wick <= 0.25 * rng:
                    patterns.append(f"{timeframe_label} Bullish Pinbar/Hammer ({label})")
                elif upper_wick >= 2.0 * body and body <= 0.35 * rng and lower_wick <= 0.25 * rng:
                    patterns.append(f"{timeframe_label} Bearish Pinbar/Shooting Star ({label})")

        # 3. Liquidity Sweeps on c2 (latest) or c1
        for label, candle in [("Latest", c2), ("Prior", c1)]:
            rng = float(candle["high"] - candle["low"])
            if rng > 0:
                upper_wick = float(candle["high"] - max(candle["open"], candle["close"]))
                lower_wick = float(min(candle["open"], candle["close"]) - candle["low"])

                # Sweep of Asian High
                if candle["high"] > asian_high and candle["close"] < asian_high and upper_wick >= 0.25 * rng:
                    patterns.append(f"{timeframe_label} Bearish Liquidity Sweep of Asian High (${asian_high:.2f}) [{label}]")

                # Sweep of PDH
                if candle["high"] > pdh and candle["close"] < pdh and upper_wick >= 0.25 * rng:
                    patterns.append(f"{timeframe_label} Bearish Liquidity Sweep of PDH (${pdh:.2f}) [{label}]")

                # Sweep of Asian Low
                if candle["low"] < asian_low and candle["close"] > asian_low and lower_wick >= 0.25 * rng:
                    patterns.append(f"{timeframe_label} Bullish Liquidity Sweep of Asian Low (${asian_low:.2f}) [{label}]")

                # Sweep of PDL
                if candle["low"] < pdl and candle["close"] > pdl and lower_wick >= 0.25 * rng:
                    patterns.append(f"{timeframe_label} Bullish Liquidity Sweep of PDL (${pdl:.2f}) [{label}]")

        # 4. 3-candle Fair Value Gap (FVG) Imbalance
        if c0["high"] < c2["low"]:
            gap_size = float(c2["low"] - c0["high"])
            patterns.append(
                f"{timeframe_label} Bullish FVG Imbalance [${c0['high']:.2f} - ${c2['low']:.2f}] (Gap: ${gap_size:.2f})"
            )
        elif c0["low"] > c2["high"]:
            gap_size = float(c0["low"] - c2["high"])
            patterns.append(
                f"{timeframe_label} Bearish FVG Imbalance [${c2['high']:.2f} - ${c0['low']:.2f}] (Gap: ${gap_size:.2f})"
            )

        return patterns


# ======================================================================================
# MODULE 1.5: AUTOMATED 3-STRATEGY DETECTOR ENGINE (StrategyScanner)
# ======================================================================================

class StrategyScanner:
    """
    Automated Institutional Strategy Detector for XAU/USD:
      1. Asian Session Liquidity Sweep & Reversal (Judas Swing)
      2. London/NY Breakout & Retest
      3. EMA 20/50 + VWAP Trend Pullback Scalper
    """

    @classmethod
    def scan_judas_swing(
        cls,
        df_m5: pd.DataFrame,
        df_m15: pd.DataFrame,
        levels: Dict[str, float],
        current_price: float,
        atr_m5: float,
        patterns_m5: List[str],
        patterns_m15: List[str],
    ) -> Dict[str, Any]:
        """
        Strategy 1: Asian Session Liquidity Sweep & Reversal (Judas Swing)
        - Price wicked above Asian High / PDH (Bearish Sweep) or below Asian Low / PDL (Bullish Sweep)
        - Closed back inside the range on M5/M15
        - Formed an opposite candle or FVG
        """
        asian_high = levels["asian_high"]
        asian_low = levels["asian_low"]
        pdh = levels["pdh"]
        pdl = levels["pdl"]

        recent_m5 = df_m5.tail(6)
        recent_high = float(recent_m5["high"].max())
        recent_low = float(recent_m5["low"].min())
        latest_c = df_m5.iloc[-1]
        prior_c = df_m5.iloc[-2]

        all_patterns = patterns_m5 + patterns_m15

        # Check Bearish Sweep (Asian High or PDH)
        swept_high_level = None
        if recent_high > asian_high:
            swept_high_level = ("Asian High", asian_high)
        elif recent_high > pdh:
            swept_high_level = ("PDH", pdh)

        # Check Bullish Sweep (Asian Low or PDL)
        swept_low_level = None
        if recent_low < asian_low:
            swept_low_level = ("Asian Low", asian_low)
        elif recent_low < pdl:
            swept_low_level = ("PDL", pdl)

        # Bearish Judas Swing Evaluation
        if swept_high_level:
            lvl_name, lvl_val = swept_high_level
            closed_back_below = current_price < lvl_val
            has_bearish_trigger = (
                latest_c["close"] < latest_c["open"]
                or prior_c["close"] < prior_c["open"]
                or any("Bearish" in p for p in all_patterns)
            )

            if closed_back_below and has_bearish_trigger:
                sl = round(recent_high + max(1.0, 0.5 * atr_m5), 2)
                sl_dist = round(sl - current_price, 2)
                vwap = float(df_m5["vwap"].iloc[-1])
                tp1 = round(vwap if vwap < current_price else current_price - 1.5 * sl_dist, 2)
                tp2 = round(asian_low if asian_low < current_price else current_price - 2.5 * sl_dist, 2)
                rrr = round(abs(current_price - tp1) / max(0.5, sl_dist), 1)

                return {
                    "name": "Asian Session Liquidity Sweep (Judas Swing)",
                    "status": "✅ ACTIVE SETUP READY",
                    "direction": "BEARISH",
                    "entry_label": "Market Execution",
                    "entry": current_price,
                    "sl": sl,
                    "sl_distance": sl_dist,
                    "tp1": tp1,
                    "tp2": tp2,
                    "rrr": f"1:{rrr}",
                    "swept_level": f"{lvl_name} (${lvl_val:.2f})",
                    "trigger": f"Price swept {lvl_name} (${lvl_val:.2f}) up to ${recent_high:.2f}, rejected back inside, and confirmed bearish momentum.",
                    "explanation": f"Bearish Judas Swing active. Swept {lvl_name} (${lvl_val:.2f}), Entry: ${current_price:.2f}, SL: ${sl:.2f}, TP1: ${tp1:.2f}.",
                }
            elif not closed_back_below:
                return {
                    "name": "Asian Session Liquidity Sweep (Judas Swing)",
                    "status": "⏳ FORMING / WATCHING",
                    "direction": "BEARISH",
                    "swept_level": f"{lvl_name} (${lvl_val:.2f})",
                    "explanation": f"Price probed above {lvl_name} (${lvl_val:.2f}) reaching ${recent_high:.2f}. Waiting for M5 candle close back below ${lvl_val:.2f} and bearish rejection candle.",
                }

        # Bullish Judas Swing Evaluation
        if swept_low_level:
            lvl_name, lvl_val = swept_low_level
            closed_back_above = current_price > lvl_val
            has_bullish_trigger = (
                latest_c["close"] > latest_c["open"]
                or prior_c["close"] > prior_c["open"]
                or any("Bullish" in p for p in all_patterns)
            )

            if closed_back_above and has_bullish_trigger:
                sl = round(recent_low - max(1.0, 0.5 * atr_m5), 2)
                sl_dist = round(current_price - sl, 2)
                vwap = float(df_m5["vwap"].iloc[-1])
                tp1 = round(vwap if vwap > current_price else current_price + 1.5 * sl_dist, 2)
                tp2 = round(asian_high if asian_high > current_price else current_price + 2.5 * sl_dist, 2)
                rrr = round(abs(tp1 - current_price) / max(0.5, sl_dist), 1)

                return {
                    "name": "Asian Session Liquidity Sweep (Judas Swing)",
                    "status": "✅ ACTIVE SETUP READY",
                    "direction": "BULLISH",
                    "entry_label": "Market Execution",
                    "entry": current_price,
                    "sl": sl,
                    "sl_distance": sl_dist,
                    "tp1": tp1,
                    "tp2": tp2,
                    "rrr": f"1:{rrr}",
                    "swept_level": f"{lvl_name} (${lvl_val:.2f})",
                    "trigger": f"Price swept {lvl_name} (${lvl_val:.2f}) down to ${recent_low:.2f}, reclaimed range, and formed bullish rejection.",
                    "explanation": f"Bullish Judas Swing active. Swept {lvl_name} (${lvl_val:.2f}), Entry: ${current_price:.2f}, SL: ${sl:.2f}, TP1: ${tp1:.2f}.",
                }
            elif not closed_back_above:
                return {
                    "name": "Asian Session Liquidity Sweep (Judas Swing)",
                    "status": "⏳ FORMING / WATCHING",
                    "direction": "BULLISH",
                    "swept_level": f"{lvl_name} (${lvl_val:.2f})",
                    "explanation": f"Price probed below {lvl_name} (${lvl_val:.2f}) down to ${recent_low:.2f}. Waiting for M5 candle close back above ${lvl_val:.2f} and bullish confirmation.",
                }

        # Check if approaching boundaries (within 1.5x ATR or $3.00)
        dist_high = asian_high - current_price
        dist_low = current_price - asian_low
        threshold = max(3.0, atr_m5 * 1.5)

        if 0 < dist_high <= threshold:
            return {
                "name": "Asian Session Liquidity Sweep (Judas Swing)",
                "status": "⏳ FORMING / WATCHING",
                "direction": "BEARISH BIAS",
                "swept_level": f"Asian High (${asian_high:.2f})",
                "explanation": f"Price is within ${dist_high:.2f} of Asian High (${asian_high:.2f}). Watching for a Judas sweep and rejection wick.",
            }
        elif 0 < dist_low <= threshold:
            return {
                "name": "Asian Session Liquidity Sweep (Judas Swing)",
                "status": "⏳ FORMING / WATCHING",
                "direction": "BULLISH BIAS",
                "swept_level": f"Asian Low (${asian_low:.2f})",
                "explanation": f"Price is within ${dist_low:.2f} of Asian Low (${asian_low:.2f}). Watching for a Judas sweep and rejection wick.",
            }

        return {
            "name": "Asian Session Liquidity Sweep (Judas Swing)",
            "status": "❌ INACTIVE",
            "direction": "NEUTRAL",
            "explanation": f"Price trading inside Asian range (${asian_low:.2f} - ${asian_high:.2f}); no recent liquidity sweep.",
        }

    @classmethod
    def scan_breakout_retest(
        cls,
        df_m5: pd.DataFrame,
        df_m15: pd.DataFrame,
        levels: Dict[str, float],
        current_price: float,
        atr_m5: float,
        atr_m15: float,
    ) -> Dict[str, Any]:
        """
        Strategy 2: London/NY Breakout & Retest
        - Strong M15 candle closed cleanly above PDH/Asian High or below PDL/Asian Low
        - Price aligned with 20 EMA and VWAP
        - M5 pulling back to retest the broken level
        """
        asian_high = levels["asian_high"]
        asian_low = levels["asian_low"]
        pdh = levels["pdh"]
        pdl = levels["pdl"]

        recent_m15 = df_m15.tail(6)
        vwap_m15 = float(df_m15["vwap"].iloc[-1])
        ema20_m15 = float(df_m15["ema_20"].iloc[-1])

        # Check for clean Bullish Breakout on M15
        bullish_break_level = None
        for i in range(len(recent_m15)):
            c = recent_m15.iloc[i]
            body = abs(c["close"] - c["open"])
            rng = c["high"] - c["low"]
            if body >= 0.4 * rng and c["close"] > c["open"]:
                if c["close"] > asian_high + 0.5:
                    bullish_break_level = ("Asian High", asian_high)
                elif c["close"] > pdh + 0.5:
                    bullish_break_level = ("PDH", pdh)

        # Check for clean Bearish Breakout on M15
        bearish_break_level = None
        for i in range(len(recent_m15)):
            c = recent_m15.iloc[i]
            body = abs(c["close"] - c["open"])
            rng = c["high"] - c["low"]
            if body >= 0.4 * rng and c["close"] < c["open"]:
                if c["close"] < asian_low - 0.5:
                    bearish_break_level = ("Asian Low", asian_low)
                elif c["close"] < pdl - 0.5:
                    bearish_break_level = ("PDL", pdl)

        # Bullish Breakout & Retest
        if bullish_break_level and current_price >= vwap_m15 and current_price >= ema20_m15 * 0.998:
            lvl_name, lvl_val = bullish_break_level
            recent_m5_lows = df_m5.tail(4)
            min_low_m5 = float(recent_m5_lows["low"].min())
            distance_to_level = abs(min_low_m5 - lvl_val)

            is_retesting = distance_to_level <= max(2.5, 1.2 * atr_m5) and current_price >= lvl_val - 0.5 * atr_m5
            latest_m5 = df_m5.iloc[-1]
            prior_m5 = df_m5.iloc[-2]
            has_bounce = (
                latest_m5["close"] >= latest_m5["open"]
                or prior_m5["close"] >= prior_m5["open"]
                or (latest_m5["close"] - latest_m5["low"]) >= 0.4 * (latest_m5["high"] - latest_m5["low"])
            )

            if is_retesting and has_bounce:
                sl = round(lvl_val - 1.5 * atr_m5, 2)
                sl_dist = round(current_price - sl, 2)
                tp1 = round(current_price + 2.0 * sl_dist, 2)
                tp2 = round(current_price + 3.0 * sl_dist, 2)
                return {
                    "name": "London/NY Breakout & Retest",
                    "status": "✅ ACTIVE SETUP READY",
                    "direction": "BULLISH",
                    "entry_label": "Retest Entry Zone",
                    "entry": current_price,
                    "entry_zone": f"${lvl_val:.2f} - ${lvl_val + round(0.5 * atr_m5, 2):.2f}",
                    "sl": sl,
                    "sl_distance": sl_dist,
                    "tp1": tp1,
                    "tp2": tp2,
                    "rrr": "1:2.0",
                    "trigger": f"M15 broke above {lvl_name} (${lvl_val:.2f}); M5 is retesting and holding with VWAP alignment.",
                    "explanation": f"Bullish Breakout & Retest active. Retest zone: ${lvl_val:.2f}-${lvl_val + round(0.5 * atr_m5, 2):.2f}, SL: ${sl:.2f}, TP: ${tp1:.2f}.",
                }
            else:
                return {
                    "name": "London/NY Breakout & Retest",
                    "status": "⏳ FORMING / WATCHING",
                    "direction": "BULLISH",
                    "entry_zone": f"${lvl_val:.2f} - ${lvl_val + round(0.5 * atr_m5, 2):.2f}",
                    "explanation": f"Strong M15 breakout above {lvl_name} (${lvl_val:.2f}). Waiting for M5 pullback to retest the ${lvl_val:.2f} support zone.",
                }

        # Bearish Breakout & Retest
        if bearish_break_level and current_price <= vwap_m15 and current_price <= ema20_m15 * 1.002:
            lvl_name, lvl_val = bearish_break_level
            recent_m5_highs = df_m5.tail(4)
            max_high_m5 = float(recent_m5_highs["high"].max())
            distance_to_level = abs(max_high_m5 - lvl_val)

            is_retesting = distance_to_level <= max(2.5, 1.2 * atr_m5) and current_price <= lvl_val + 0.5 * atr_m5
            latest_m5 = df_m5.iloc[-1]
            prior_m5 = df_m5.iloc[-2]
            has_rejection = (
                latest_m5["close"] <= latest_m5["open"]
                or prior_m5["close"] <= prior_m5["open"]
                or (latest_m5["high"] - latest_m5["close"]) >= 0.4 * (latest_m5["high"] - latest_m5["low"])
            )

            if is_retesting and has_rejection:
                sl = round(lvl_val + 1.5 * atr_m5, 2)
                sl_dist = round(sl - current_price, 2)
                tp1 = round(current_price - 2.0 * sl_dist, 2)
                tp2 = round(current_price - 3.0 * sl_dist, 2)
                return {
                    "name": "London/NY Breakout & Retest",
                    "status": "✅ ACTIVE SETUP READY",
                    "direction": "BEARISH",
                    "entry_label": "Retest Entry Zone",
                    "entry": current_price,
                    "entry_zone": f"${lvl_val - round(0.5 * atr_m5, 2):.2f} - ${lvl_val:.2f}",
                    "sl": sl,
                    "sl_distance": sl_dist,
                    "tp1": tp1,
                    "tp2": tp2,
                    "rrr": "1:2.0",
                    "trigger": f"M15 broke below {lvl_name} (${lvl_val:.2f}); M5 is retesting and rejecting under VWAP.",
                    "explanation": f"Bearish Breakout & Retest active. Retest zone: ${lvl_val - round(0.5 * atr_m5, 2):.2f}-${lvl_val:.2f}, SL: ${sl:.2f}, TP: ${tp1:.2f}.",
                }
            else:
                return {
                    "name": "London/NY Breakout & Retest",
                    "status": "⏳ FORMING / WATCHING",
                    "direction": "BEARISH",
                    "entry_zone": f"${lvl_val - round(0.5 * atr_m5, 2):.2f} - ${lvl_val:.2f}",
                    "explanation": f"Strong M15 breakout below {lvl_name} (${lvl_val:.2f}). Waiting for M5 pullback to retest the ${lvl_val:.2f} resistance zone.",
                }

        return {
            "name": "London/NY Breakout & Retest",
            "status": "❌ INACTIVE",
            "direction": "NEUTRAL",
            "explanation": "No sustained M15 candle breakout beyond Asian session or PDH/PDL levels.",
        }

    @classmethod
    def scan_trend_pullback(
        cls,
        df_m5: pd.DataFrame,
        df_m15: pd.DataFrame,
        df_h1: pd.DataFrame,
        current_price: float,
        atr_m5: float,
        m5_metrics: Dict[str, Any],
        m15_metrics: Dict[str, Any],
        h1_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Strategy 3: EMA 20/50 + VWAP Trend Pullback Scalper
        - H1 & M15 in clear trend alignment
        - M5 RSI pulls back to 40-52 (uptrend) or 48-60 (downtrend)
        - Price touches/nears 20 EMA / VWAP and forms a trigger bounce/rejection candle
        """
        h1_ema20 = h1_metrics.get("ema_20", current_price)
        h1_ema50 = h1_metrics.get("ema_50", current_price)

        m15_ema20 = m15_metrics.get("ema_20", current_price)
        m15_ema50 = m15_metrics.get("ema_50", current_price)
        m15_vwap = m15_metrics.get("vwap", current_price)

        m5_ema20 = m5_metrics.get("ema_20", current_price)
        m5_vwap = m5_metrics.get("vwap", current_price)
        m5_rsi = m5_metrics.get("rsi_14", 50.0)

        # Bullish Trend Alignment
        h1_bullish = current_price > h1_ema20 and h1_ema20 > h1_ema50
        m15_bullish = current_price > m15_vwap and m15_ema20 > m15_ema50

        # Bearish Trend Alignment
        h1_bearish = current_price < h1_ema20 and h1_ema20 < h1_ema50
        m15_bearish = current_price < m15_vwap and m15_ema20 < m15_ema50

        latest_m5 = df_m5.iloc[-1]
        prior_m5 = df_m5.iloc[-2]
        recent_m5_lows = float(df_m5["low"].tail(4).min())
        recent_m5_highs = float(df_m5["high"].tail(4).max())

        if h1_bullish and m15_bullish:
            # Check M5 Pullback
            rsi_in_pullback = 38.0 <= m5_rsi <= 53.0 or any(38.0 <= r <= 53.0 for r in df_m5["rsi_14"].tail(4))
            touched_ema_or_vwap = abs(recent_m5_lows - m5_ema20) <= max(1.5, 0.6 * atr_m5) or abs(recent_m5_lows - m5_vwap) <= max(1.5, 0.6 * atr_m5)
            has_bounce = (
                latest_m5["close"] > latest_m5["open"]
                or prior_m5["close"] > prior_m5["open"]
                or (latest_m5["close"] - latest_m5["low"]) >= 0.4 * (latest_m5["high"] - latest_m5["low"])
            )

            if rsi_in_pullback and touched_ema_or_vwap and has_bounce:
                sl = round(current_price - 1.5 * atr_m5, 2)
                sl_dist = round(current_price - sl, 2)
                tp1 = round(current_price + 2.0 * sl_dist, 2)
                tp2 = round(current_price + 3.0 * sl_dist, 2)
                return {
                    "name": "EMA 20/50 + VWAP Trend Pullback Scalper",
                    "status": "✅ ACTIVE SETUP READY",
                    "direction": "BULLISH",
                    "entry_label": "Scalp Entry",
                    "entry": current_price,
                    "sl": sl,
                    "sl_distance": sl_dist,
                    "tp1": tp1,
                    "tp2": tp2,
                    "rrr": "1:2.0",
                    "trigger": f"H1 & M15 bullish trend confirmed. M5 pulled back to 20 EMA (${m5_ema20:.2f}), RSI cooled to {m5_rsi:.1f}, and printed a bounce candle.",
                    "explanation": f"Bullish Trend Pullback Scalp active. Entry: ${current_price:.2f}, SL: ${sl:.2f}, TP (1:2): ${tp1:.2f}.",
                }
            else:
                missing = []
                if m5_rsi > 55.0:
                    missing.append(f"M5 RSI is high ({m5_rsi:.1f}); waiting for cooling to 40-52")
                if not touched_ema_or_vwap:
                    missing.append(f"waiting for price to dip near 20 EMA (${m5_ema20:.2f})")
                if not has_bounce:
                    missing.append("waiting for bullish bounce trigger candle")
                explanation = "Bullish trend confirmed on H1 & M15: " + ("; ".join(missing) if missing else "monitoring 20 EMA interaction.")
                return {
                    "name": "EMA 20/50 + VWAP Trend Pullback Scalper",
                    "status": "⏳ FORMING / WATCHING",
                    "direction": "BULLISH",
                    "explanation": explanation,
                }

        if h1_bearish and m15_bearish:
            rsi_in_pullback = 47.0 <= m5_rsi <= 62.0 or any(47.0 <= r <= 62.0 for r in df_m5["rsi_14"].tail(4))
            touched_ema_or_vwap = abs(recent_m5_highs - m5_ema20) <= max(1.5, 0.6 * atr_m5) or abs(recent_m5_highs - m5_vwap) <= max(1.5, 0.6 * atr_m5)
            has_rejection = (
                latest_m5["close"] < latest_m5["open"]
                or prior_m5["close"] < prior_m5["open"]
                or (latest_m5["high"] - latest_m5["close"]) >= 0.4 * (latest_m5["high"] - latest_m5["low"])
            )

            if rsi_in_pullback and touched_ema_or_vwap and has_rejection:
                sl = round(current_price + 1.5 * atr_m5, 2)
                sl_dist = round(sl - current_price, 2)
                tp1 = round(current_price - 2.0 * sl_dist, 2)
                tp2 = round(current_price - 3.0 * sl_dist, 2)
                return {
                    "name": "EMA 20/50 + VWAP Trend Pullback Scalper",
                    "status": "✅ ACTIVE SETUP READY",
                    "direction": "BEARISH",
                    "entry_label": "Scalp Entry",
                    "entry": current_price,
                    "sl": sl,
                    "sl_distance": sl_dist,
                    "tp1": tp1,
                    "tp2": tp2,
                    "rrr": "1:2.0",
                    "trigger": f"H1 & M15 bearish trend confirmed. M5 pulled back to 20 EMA (${m5_ema20:.2f}), RSI normalized to {m5_rsi:.1f}, and printed a rejection candle.",
                    "explanation": f"Bearish Trend Pullback Scalp active. Entry: ${current_price:.2f}, SL: ${sl:.2f}, TP (1:2): ${tp1:.2f}.",
                }
            else:
                missing = []
                if m5_rsi < 45.0:
                    missing.append(f"M5 RSI is oversold ({m5_rsi:.1f}); waiting for retrace to 48-60")
                if not touched_ema_or_vwap:
                    missing.append(f"waiting for price to rise near 20 EMA (${m5_ema20:.2f})")
                if not has_rejection:
                    missing.append("waiting for bearish rejection trigger candle")
                explanation = "Bearish trend confirmed on H1 & M15: " + ("; ".join(missing) if missing else "monitoring 20 EMA interaction.")
                return {
                    "name": "EMA 20/50 + VWAP Trend Pullback Scalper",
                    "status": "⏳ FORMING / WATCHING",
                    "direction": "BEARISH",
                    "explanation": explanation,
                }

        return {
            "name": "EMA 20/50 + VWAP Trend Pullback Scalper",
            "status": "❌ INACTIVE",
            "direction": "NEUTRAL",
            "explanation": "No aligned directional trend across H1 (20/50 EMA) and M15 (VWAP); market is currently consolidating.",
        }

    @classmethod
    def scan_all(
        cls,
        df_m5: pd.DataFrame,
        df_m15: pd.DataFrame,
        df_h1: pd.DataFrame,
        levels: Dict[str, float],
        m5_metrics: Dict[str, Any],
        m15_metrics: Dict[str, Any],
        h1_metrics: Dict[str, Any],
        patterns_m5: List[str],
        patterns_m15: List[str],
        current_price: float,
    ) -> Dict[str, Dict[str, Any]]:
        """Scans and evaluates all 3 institutional trading strategies."""
        atr_m5 = m5_metrics.get("atr_14", 3.0)
        atr_m15 = m15_metrics.get("atr_14", 5.5)

        s1 = cls.scan_judas_swing(
            df_m5=df_m5,
            df_m15=df_m15,
            levels=levels,
            current_price=current_price,
            atr_m5=atr_m5,
            patterns_m5=patterns_m5,
            patterns_m15=patterns_m15,
        )
        s2 = cls.scan_breakout_retest(
            df_m5=df_m5,
            df_m15=df_m15,
            levels=levels,
            current_price=current_price,
            atr_m5=atr_m5,
            atr_m15=atr_m15,
        )
        s3 = cls.scan_trend_pullback(
            df_m5=df_m5,
            df_m15=df_m15,
            df_h1=df_h1,
            current_price=current_price,
            atr_m5=atr_m5,
            m5_metrics=m5_metrics,
            m15_metrics=m15_metrics,
            h1_metrics=h1_metrics,
        )

        return {
            "judas_swing": s1,
            "breakout_retest": s2,
            "trend_pullback": s3,
        }


# ======================================================================================
# MODULE 1.6: PAPER TRADING & AUTO-JOURNALING DATABASE (TradeJournal)
# ======================================================================================

class TradeJournal:
    """
    Thread-safe SQLite Paper Trading & Auto-Journaling System for tracking
    the 3 institutional XAU/USD setups:
      1. Asian Session Liquidity Sweep (Judas Swing)
      2. London/NY Breakout & Retest
      3. EMA 20/50 + VWAP Trend Pullback Scalper

    Stores paper trades in SQLite (journal.db), calculates win rates and stats,
    and automatically detects Take-Profit and Stop-Loss triggers during the 15-minute
    background loop.
    """
    DB_PATH = JOURNAL_DB_PATH

    STRATEGY_NAMES = {
        1: "Asian Session Liquidity Sweep (Judas Swing)",
        2: "London/NY Breakout & Retest",
        3: "EMA 20/50 + VWAP Trend Pullback Scalper",
    }

    @classmethod
    def _init_db(cls):
        """Initializes the SQLite database table and indices if they do not exist."""
        try:
            with sqlite3.connect(cls.DB_PATH, timeout=10.0) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS journal_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_id INTEGER NOT NULL,
                        strategy_name TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        stop_loss REAL NOT NULL,
                        take_profit REAL NOT NULL,
                        status TEXT NOT NULL DEFAULT 'OPEN',
                        opened_at TEXT NOT NULL,
                        closed_at TEXT,
                        close_price REAL,
                        pnl_dollars REAL,
                        notes TEXT
                    )
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_journal_status ON journal_trades(status)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_journal_strategy ON journal_trades(strategy_id)
                """)
                conn.commit()
        except Exception as exc:
            logger.error("[TRADE JOURNAL] Failed to initialize database: %s", exc)

    @classmethod
    def open_trade(
        cls,
        strategy_id: int,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        strategy_name: Optional[str] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        """Logs a new paper trade with status 'OPEN'."""
        cls._init_db()
        name = strategy_name or cls.STRATEGY_NAMES.get(strategy_id, f"Strategy {strategy_id}")
        direction_clean = direction.upper().strip()
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        with sqlite3.connect(cls.DB_PATH, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO journal_trades (
                    strategy_id, strategy_name, direction, entry_price,
                    stop_loss, take_profit, status, opened_at, notes
                ) VALUES (?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)
                """,
                (strategy_id, name, direction_clean, float(entry_price), float(stop_loss), float(take_profit), now_iso, notes)
            )
            trade_id = cursor.lastrowid
            conn.commit()

            cursor.execute("SELECT * FROM journal_trades WHERE id = ?", (trade_id,))
            row = cursor.fetchone()
            return dict(row)

    @classmethod
    def close_trade(
        cls,
        trade_id: int,
        result: str,
        close_price: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Closes an open paper trade as WON or LOST and calculates Gold point/dollar PnL."""
        cls._init_db()
        res_upper = result.upper().strip()
        final_status = "WON" if ("WIN" in res_upper or "WON" in res_upper) else "LOST"
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        with sqlite3.connect(cls.DB_PATH, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM journal_trades WHERE id = ?", (trade_id,))
            row = cursor.fetchone()
            if not row:
                return None

            trade = dict(row)
            entry = float(trade["entry_price"])
            sl = float(trade["stop_loss"])
            tp = float(trade["take_profit"])
            direction = trade["direction"].upper()

            if close_price is None:
                final_close_price = tp if final_status == "WON" else sl
            else:
                final_close_price = float(close_price)

            if direction == "BUY":
                pnl = round(final_close_price - entry, 2)
            else:
                pnl = round(entry - final_close_price, 2)

            cursor.execute(
                """
                UPDATE journal_trades
                SET status = ?, closed_at = ?, close_price = ?, pnl_dollars = ?, notes = COALESCE(?, notes)
                WHERE id = ?
                """,
                (final_status, now_iso, final_close_price, pnl, notes, trade_id)
            )
            conn.commit()

            cursor.execute("SELECT * FROM journal_trades WHERE id = ?", (trade_id,))
            updated_row = cursor.fetchone()
            return dict(updated_row)

    @classmethod
    def get_trade(cls, trade_id: int) -> Optional[Dict[str, Any]]:
        """Fetches a specific trade by ID."""
        cls._init_db()
        with sqlite3.connect(cls.DB_PATH, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM journal_trades WHERE id = ?", (trade_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    @classmethod
    def get_open_trades(cls) -> List[Dict[str, Any]]:
        """Returns all currently OPEN paper trades."""
        cls._init_db()
        with sqlite3.connect(cls.DB_PATH, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM journal_trades WHERE status = 'OPEN' ORDER BY id ASC")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """Computes comprehensive win rate and performance metrics across the 3 strategies."""
        cls._init_db()
        with sqlite3.connect(cls.DB_PATH, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM journal_trades")
            all_trades = [dict(r) for r in cursor.fetchall()]

        total_trades = len(all_trades)
        open_trades = [t for t in all_trades if t["status"] == "OPEN"]
        closed_trades = [t for t in all_trades if t["status"] in ("WON", "LOST")]
        won_trades = [t for t in closed_trades if t["status"] == "WON"]
        lost_trades = [t for t in closed_trades if t["status"] == "LOST"]

        closed_count = len(closed_trades)
        won_count = len(won_trades)
        lost_count = len(lost_trades)
        overall_win_rate = (won_count / closed_count * 100.0) if closed_count > 0 else 0.0
        total_pnl = sum((t["pnl_dollars"] or 0.0) for t in closed_trades)

        breakdown = {}
        for s_id in (1, 2, 3):
            s_name = cls.STRATEGY_NAMES[s_id]
            s_trades = [t for t in closed_trades if t["strategy_id"] == s_id]
            s_open = [t for t in open_trades if t["strategy_id"] == s_id]
            s_won = len([t for t in s_trades if t["status"] == "WON"])
            s_lost = len([t for t in s_trades if t["status"] == "LOST"])
            s_closed = len(s_trades)
            s_wr = (s_won / s_closed * 100.0) if s_closed > 0 else 0.0
            s_pnl = sum((t["pnl_dollars"] or 0.0) for t in s_trades)
            breakdown[s_id] = {
                "strategy_id": s_id,
                "name": s_name,
                "closed": s_closed,
                "won": s_won,
                "lost": s_lost,
                "open": len(s_open),
                "win_rate": round(s_wr, 1),
                "pnl": round(s_pnl, 2),
            }

        eligible_strategies = [b for b in breakdown.values() if b["closed"] > 0]
        if eligible_strategies:
            top = max(eligible_strategies, key=lambda x: (x["win_rate"], x["won"]))
            top_strategy_text = f"🏆 Strategy {top['strategy_id']}: {top['name']} ({top['win_rate']:.1f}% Win Rate with {top['won']}W / {top['lost']}L)"
        else:
            top_strategy_text = "No closed trades recorded yet (monitoring active setups)."

        return {
            "total_trades": total_trades,
            "open_trades": len(open_trades),
            "closed_trades": closed_count,
            "won_trades": won_count,
            "lost_trades": lost_count,
            "win_rate": round(overall_win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "breakdown": breakdown,
            "top_strategy": top_strategy_text,
        }

    @classmethod
    def check_open_trades_against_price(
        cls,
        current_price: float,
        high_price: Optional[float] = None,
        low_price: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Scans all OPEN paper trades against current candle price range.
        If Take-Profit or Stop-Loss was breached, automatically marks trade as WON or LOST.
        Returns list of newly resolved trades for Discord alerting.
        """
        open_trades = cls.get_open_trades()
        if not open_trades:
            return []

        h = float(high_price) if high_price is not None else float(current_price)
        l = float(low_price) if low_price is not None else float(current_price)
        c = float(current_price)

        resolved = []
        for t in open_trades:
            t_id = t["id"]
            direction = t["direction"].upper()
            sl = float(t["stop_loss"])
            tp = float(t["take_profit"])

            trade_resolved = False
            result_status = None
            resolved_close_price = None
            resolution_reason = None

            if direction == "BUY":
                if h >= tp or c >= tp:
                    trade_resolved = True
                    result_status = "WON"
                    resolved_close_price = tp
                    resolution_reason = f"Take-Profit target (${tp:.2f}) reached by price high (${h:.2f})"
                elif l <= sl or c <= sl:
                    trade_resolved = True
                    result_status = "LOST"
                    resolved_close_price = sl
                    resolution_reason = f"Stop-Loss level (${sl:.2f}) breached by price low (${l:.2f})"

            elif direction == "SELL":
                if l <= tp or c <= tp:
                    trade_resolved = True
                    result_status = "WON"
                    resolved_close_price = tp
                    resolution_reason = f"Take-Profit target (${tp:.2f}) reached by price low (${l:.2f})"
                elif h >= sl or c >= sl:
                    trade_resolved = True
                    result_status = "LOST"
                    resolved_close_price = sl
                    resolution_reason = f"Stop-Loss level (${sl:.2f}) breached by price high (${h:.2f})"

            if trade_resolved and result_status:
                updated = cls.close_trade(
                    trade_id=t_id,
                    result=result_status,
                    close_price=resolved_close_price,
                    notes=resolution_reason,
                )
                if updated:
                    updated["resolution_reason"] = resolution_reason
                    resolved.append(updated)

        return resolved


# ======================================================================================
# CHARTING ENGINE (MPLFINANCE VISUALIZATION)
# ======================================================================================

def generate_candlestick_chart(
    df: pd.DataFrame,
    timeframe_label: str,
    pdh: float,
    pdl: float,
    asian_high: float,
    asian_low: float,
    candles_count: int = 150,
    current_price: Optional[float] = None,
) -> io.BytesIO:
    """
    Renders a high-resolution dark-themed candlestick chart using mplfinance.
    Overlays:
      - Live Current Price Line (Cyan dotted line with price callout badge)
      - 20 EMA (Cyan) & 50 EMA (Purple) - calculated across full data context
      - Intraday VWAP (Yellow dashed) - anchored from session start
      - Yesterday's Ceiling / PDH (Red solid) & Floor / PDL (Green solid)
      - Morning Ceiling / Asian High (Orange dashed) & Floor / Asian Low (Teal dashed)
    Returns an in-memory BytesIO buffer containing the PNG image.
    Never raises an uncaught exception (uses resilient dark-card fallback on error).
    """
    try:
        df_full = df.copy()
        df_full.columns = [c.lower() for c in df_full.columns]

        close_full = df_full["close"]
        high_full = df_full["high"]
        low_full = df_full["low"]
        vol_full = df_full["volume"].replace(0, 1)

        # Calculate or reuse converged indicators on the full dataset before slicing
        if "ema_20" in df_full.columns:
            ema20_full = df_full["ema_20"]
        else:
            ema20_full = close_full.ewm(span=20, adjust=False).mean()

        if "ema_50" in df_full.columns:
            ema50_full = df_full["ema_50"]
        else:
            ema50_full = close_full.ewm(span=50, adjust=False).mean()

        if "vwap" in df_full.columns:
            vwap_full = df_full["vwap"]
        else:
            tp = (high_full + low_full + close_full) / 3.0
            pv = tp * vol_full
            vwap_full = pv.groupby(df_full.index.date).cumsum() / vol_full.groupby(df_full.index.date).cumsum()
            vwap_full = vwap_full.bfill().fillna(ema20_full)

        if current_price is None or current_price <= 0:
            current_price = float(close_full.iloc[-1])

        # Slicing lookback window (default 150 bars to capture the full afternoon session)
        lookback = min(len(df_full), max(candles_count, 30))
        df_chart = df_full.tail(lookback).copy()

        # Align current active forming candle with live tick price
        df_chart.iloc[-1, df_chart.columns.get_loc("close")] = current_price
        if current_price > df_chart.iloc[-1]["high"]:
            df_chart.iloc[-1, df_chart.columns.get_loc("high")] = current_price
        if current_price < df_chart.iloc[-1]["low"]:
            df_chart.iloc[-1, df_chart.columns.get_loc("low")] = current_price

        ema20 = ema20_full.loc[df_chart.index].copy()
        ema50 = ema50_full.loc[df_chart.index].copy()
        vwap = vwap_full.loc[df_chart.index].copy()

        # Strip timezone to prevent mplfinance index offset or dropped candle artifacts
        if df_chart.index.tz is not None:
            df_chart.index = df_chart.index.tz_localize(None)
            ema20.index = df_chart.index
            ema50.index = df_chart.index
            vwap.index = df_chart.index

        df_chart = df_chart.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )

        close = df_chart["Close"]

        if mpf is not None:
            ap = [
                mpf.make_addplot(ema20, color="cyan", width=1.5, panel=0),
                mpf.make_addplot(ema50, color="#9B59B6", width=1.5, panel=0),
                mpf.make_addplot(vwap, color="#F1C40F", width=1.5, linestyle="dashed", panel=0),
            ]

            fig, axlist = mpf.plot(
                df_chart,
                type="candle",
                style="nightclouds",
                addplot=ap,
                hlines=dict(
                    hlines=[pdh, pdl, asian_high, asian_low],
                    colors=["#E74C3C", "#2ECC71", "#E67E22", "#1ABC9C"],
                    linestyle=["-", "-", "--", "--"],
                    linewidths=[1.5, 1.5, 1.2, 1.2],
                ),
                volume=True,
                returnfig=True,
                figsize=(11, 6.5),
                title=f"XAU/USD Gold ({timeframe_label.upper()}) - Live Institutional Price Action",
            )

            ax = axlist[0]
            # Overlay live spot price line and right-aligned price badge
            ax.axhline(current_price, color="#00FFFF", linestyle=":", linewidth=1.4, alpha=0.9)
            ax.text(
                1.005,
                current_price,
                f" ${current_price:.2f} (Live)",
                transform=ax.get_yaxis_transform(),
                color="#00FFFF",
                fontsize=8.5,
                fontweight="bold",
                va="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#10141d", edgecolor="#00FFFF", lw=1.2),
            )
        else:
            # Graceful pure matplotlib fallback
            fig, ax = plt.subplots(figsize=(11, 6.5), facecolor="#181a1f")
            ax.set_facecolor("#181a1f")
            ax.plot(range(len(close)), close, color="white", lw=1.5, label="Close")
            ax.plot(range(len(ema20)), ema20, color="cyan", lw=1.5)
            ax.plot(range(len(ema50)), ema50, color="#9B59B6", lw=1.5)
            ax.plot(range(len(vwap)), vwap, color="#F1C40F", ls="--", lw=1.5)
            ax.axhline(current_price, color="#00FFFF", linestyle=":", lw=1.4)
            ax.axhline(pdh, color="#E74C3C", lw=1.5)
            ax.axhline(pdl, color="#2ECC71", lw=1.5)
            ax.axhline(asian_high, color="#E67E22", ls="--", lw=1.2)
            ax.axhline(asian_low, color="#1ABC9C", ls="--", lw=1.2)
            ax.set_title(f"XAU/USD Gold ({timeframe_label.upper()}) - Live Institutional Price Action", color="white")
            ax.tick_params(colors="white")

        # Custom clear legend on primary panel
        ax.legend(
            [
                mlines.Line2D([0], [0], color="#00FFFF", lw=1.4, ls="dotted"),
                mlines.Line2D([0], [0], color="cyan", lw=1.5),
                mlines.Line2D([0], [0], color="#9B59B6", lw=1.5),
                mlines.Line2D([0], [0], color="#F1C40F", lw=1.5, ls="dashed"),
                mlines.Line2D([0], [0], color="#E74C3C", lw=1.5),
                mlines.Line2D([0], [0], color="#2ECC71", lw=1.5),
                mlines.Line2D([0], [0], color="#E67E22", lw=1.2, ls="dashed"),
                mlines.Line2D([0], [0], color="#1ABC9C", lw=1.2, ls="dashed"),
            ],
            [
                f"Live Spot Price (${current_price:.2f})",
                "20 EMA",
                "50 EMA",
                "VWAP",
                f"Yesterday Ceiling / PDH (${pdh:.2f})",
                f"Yesterday Floor / PDL (${pdl:.2f})",
                f"Morning Ceiling / Asian High (${asian_high:.2f})",
                f"Morning Floor / Asian Low (${asian_low:.2f})",
            ],
            loc="upper left",
            fontsize=8,
            framealpha=0.85,
            facecolor="#181a1f",
            edgecolor="#333",
            labelcolor="white",
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as exc:
        logger.error("Candlestick chart generation encountered an error: %s", exc, exc_info=True)
        try:
            fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="#181a1f")
            ax.set_facecolor("#181a1f")
            safe_price = current_price if (current_price and current_price > 0) else 4000.0
            banner_text = (
                f"📈 XAU/USD ({timeframe_label.upper()}) Technical Levels Snapshot\n\n"
                f"Live Spot Price: ${safe_price:.2f}\n"
                f"PDH: ${pdh:.2f} | PDL: ${pdl:.2f}\n"
                f"Asian Session: ${asian_low:.2f} – ${asian_high:.2f}\n\n"
                f"(Graphical plot recovering — numeric levels & AI engine intact)"
            )
            ax.text(
                0.5, 0.5, banner_text,
                color="#00FFFF", fontsize=10.5, ha="center", va="center",
                linespacing=1.6, fontweight="bold"
            )
            ax.axis("off")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            plt.close(fig)
            buf.seek(0)
            return buf
        except Exception as fallback_err:
            logger.error("Failed even to build chart fallback image: %s", fallback_err)
            return io.BytesIO()


# ======================================================================================
# MODULE 2: FOREX CALENDAR & BREAKING NEWS SCRAPER
# ======================================================================================

class ForexIntelligenceScraper:
    """
    Scrapes high/medium impact economic calendar releases from Forex Factory,
    and pulls breaking real-time forex headlines from ForexLive and FXStreet.
    """

    CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    NEWS_FEEDS = [
        "https://www.forexlive.com/feed/news",
        "https://www.fxstreet.com/rss/news",
    ]
    _calendar_cache: Dict[str, Any] = {}
    _calendar_cache_time: float = 0.0

    @classmethod
    def fetch_calendar_events(cls) -> Dict[str, Any]:
        """
        Fetches the weekly economic calendar JSON (cached for 10 minutes to prevent rate limits).
        Filters for today's USD events marked 'High' or 'Medium' impact,
        and computes exact minutes remaining until the next release.
        Identifies red-folder risk (CPI, NFP, PPI, FOMC within 30 minutes).
        """
        current_time = time.time()
        if cls._calendar_cache and (current_time - cls._calendar_cache_time < 600):
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            cached_data = cls._calendar_cache.copy()
            events = []
            min_pos = float("inf")
            next_ev = None
            red_alert = False
            red_details = ""
            for ev in cached_data.get("events_raw", []):
                try:
                    event_dt = datetime.datetime.fromisoformat(ev["date"])
                    event_dt = event_dt.astimezone(datetime.timezone.utc) if event_dt.tzinfo else event_dt.replace(tzinfo=datetime.timezone.utc)
                    diff_m = (event_dt - now_utc).total_seconds() / 60.0
                    rec = {
                        "title": ev.get("title", ""),
                        "impact": ev.get("impact", ""),
                        "date_utc": event_dt.strftime("%Y-%m-%d %H:%M UTC"),
                        "minutes_remaining": round(diff_m, 1),
                        "forecast": ev.get("forecast", ""),
                        "previous": ev.get("previous", ""),
                    }
                    events.append(rec)
                    if 0 <= diff_m < min_pos:
                        min_pos = diff_m
                        next_ev = rec
                    if ev.get("impact") == "High" or any(k in ev.get("title", "").upper() for k in ["CPI", "NFP", "PPI", "FOMC", "RATE"]):
                        if -5 <= diff_m <= 30:
                            red_alert = True
                            red_details = f"{rec['title']} is scheduled in {diff_m:.0f} mins" if diff_m >= 0 else f"{rec['title']} was released {abs(diff_m):.0f} mins ago"
                except Exception:
                    continue
            events.sort(key=lambda x: x["minutes_remaining"])
            return {
                "events_today": events,
                "next_event": next_ev,
                "red_folder_alert": red_alert,
                "red_folder_details": red_details,
            }

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json"
        }

        usd_events = []
        next_event = None
        min_positive_diff = float("inf")
        red_folder_alert = False
        red_folder_details = ""

        try:
            req = urllib.request.Request(cls.CALENDAR_URL, headers=headers)
            with urllib.request.urlopen(req, timeout=12) as response:
                raw_data = response.read().decode("utf-8")
                calendar_items = json.loads(raw_data)

            cls._calendar_cache = {
                "events_raw": [
                    item for item in calendar_items
                    if item.get("country") == "USD" and item.get("impact") in ["High", "Medium"]
                ]
            }
            cls._calendar_cache_time = time.time()

            now_utc = datetime.datetime.now(datetime.timezone.utc)

            for item in calendar_items:
                country = item.get("country", "")
                impact = item.get("impact", "")
                date_str = item.get("date", "")
                title = item.get("title", "")

                if country != "USD" or impact not in ["High", "Medium"]:
                    continue

                try:
                    event_dt = datetime.datetime.fromisoformat(date_str)
                    if event_dt.tzinfo is None:
                        event_dt = event_dt.replace(tzinfo=datetime.timezone.utc)
                    else:
                        event_dt = event_dt.astimezone(datetime.timezone.utc)
                except Exception:
                    continue

                diff_minutes = (event_dt - now_utc).total_seconds() / 60.0

                if abs(diff_minutes) <= 1440:
                    event_record = {
                        "title": title,
                        "impact": impact,
                        "date_utc": event_dt.strftime("%Y-%m-%d %H:%M UTC"),
                        "minutes_remaining": round(diff_minutes, 1),
                        "forecast": item.get("forecast", ""),
                        "previous": item.get("previous", ""),
                    }
                    usd_events.append(event_record)

                    if 0 <= diff_minutes < min_positive_diff:
                        min_positive_diff = diff_minutes
                        next_event = event_record

                    is_major_driver = any(keyword in title.upper() for keyword in ["CPI", "NFP", "PPI", "FOMC", "RATE", "PAYROLLS", "FED"])
                    if impact == "High" or is_major_driver:
                        if -5 <= diff_minutes <= 30:
                            red_folder_alert = True
                            red_folder_details = (
                                f"{title} ({impact} Impact) is scheduled in {diff_minutes:.0f} mins"
                                if diff_minutes >= 0
                                else f"{title} ({impact} Impact) was released {abs(diff_minutes):.0f} mins ago"
                            )

            usd_events.sort(key=lambda x: x["minutes_remaining"])

        except Exception as exc:
            logger.warning("Failed to fetch Forex Factory calendar: %s", exc)
            if cls._calendar_cache.get("events_raw"):
                logger.info("Serving economic calendar from backup in-memory cache.")
                now_utc = datetime.datetime.now(datetime.timezone.utc)
                for item in cls._calendar_cache["events_raw"]:
                    try:
                        event_dt = datetime.datetime.fromisoformat(item["date"])
                        event_dt = event_dt.astimezone(datetime.timezone.utc) if event_dt.tzinfo else event_dt.replace(tzinfo=datetime.timezone.utc)
                        diff_m = (event_dt - now_utc).total_seconds() / 60.0
                        if abs(diff_m) <= 1440:
                            usd_events.append({
                                "title": item.get("title", ""),
                                "impact": item.get("impact", ""),
                                "date_utc": event_dt.strftime("%Y-%m-%d %H:%M UTC"),
                                "minutes_remaining": round(diff_m, 1),
                                "forecast": item.get("forecast", ""),
                                "previous": item.get("previous", ""),
                            })
                    except Exception:
                        continue
                usd_events.sort(key=lambda x: x["minutes_remaining"])
                if usd_events:
                    upcoming = [e for e in usd_events if e["minutes_remaining"] >= 0]
                    if upcoming:
                        next_event = upcoming[0]

        return {
            "events_today": usd_events,
            "next_event": next_event,
            "red_folder_alert": red_folder_alert,
            "red_folder_details": red_folder_details,
        }

    @classmethod
    def fetch_breaking_news(cls, limit: int = 10) -> List[Dict[str, str]]:
        """
        Scrapes and parses the latest breaking forex headlines from ForexLive and FXStreet RSS feeds.
        Returns top deduplicated news items.
        """
        headlines: List[Dict[str, str]] = []
        seen_titles = set()

        for url in cls.NEWS_FEEDS:
            try:
                if HAS_FEEDPARSER:
                    fp = feedparser.parse(url)
                    for entry in fp.entries[:limit]:
                        title = entry.get("title", "").strip()
                        clean_title = " ".join(title.split())
                        if clean_title and clean_title.lower() not in seen_titles:
                            seen_titles.add(clean_title.lower())
                            headlines.append({
                                "title": clean_title,
                                "source": "ForexLive" if "forexlive" in url else "FXStreet",
                                "published": entry.get("published", "")
                            })
                else:
                    req = urllib.request.Request(
                        url,
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                    )
                    with urllib.request.urlopen(req, timeout=8) as resp:
                        tree = ET.fromstring(resp.read())
                        items = tree.findall(".//item")
                        for item in items[:limit]:
                            title_elem = item.find("title")
                            pub_elem = item.find("pubDate")
                            if title_elem is not None and title_elem.text:
                                clean_title = " ".join(title_elem.text.strip().split())
                                if clean_title.lower() not in seen_titles:
                                    seen_titles.add(clean_title.lower())
                                    headlines.append({
                                        "title": clean_title,
                                        "source": "ForexLive" if "forexlive" in url else "FXStreet",
                                        "published": pub_elem.text if pub_elem is not None else ""
                                    })
            except Exception as exc:
                logger.warning("Error fetching RSS feed %s: %s", url, exc)

        return headlines[:limit]


# ======================================================================================
# MODULE 3: HYBRID PREDICTION ENGINE (AI LLM + 3-TIMEFRAME QUANT FALLBACK)
# ======================================================================================

class HybridPredictionEngine:
    """
    Packages 3-timeframe technical metrics (M5, M15, H1), institutional levels,
    macro correlations, and news into a structured JSON payload, evaluates against
    Gemini/OpenAI LLM, or triggers the deterministic quantitative scoring fallback.
    """

    @staticmethod
    def build_summary_payload(
        gold_symbol: str,
        current_price: float,
        m5_metrics: Dict[str, Any],
        m15_metrics: Dict[str, Any],
        h1_metrics: Dict[str, Any],
        levels: Dict[str, float],
        m5_patterns: List[str],
        m15_patterns: List[str],
        h1_patterns: List[str],
        macro: Dict[str, Any],
        calendar: Dict[str, Any],
        news: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Constructs an institutional JSON summary for the AI LLM and quant engine."""
        return {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "asset": "XAU/USD (Spot Gold)",
            "data_source": gold_symbol,
            "current_price": round(current_price, 2),
            "analytical_hierarchy": {
                "h1_macro_trend": "Establishes broad trend and dominant order flow direction",
                "m15_setup_structure": "Validates market structure, FVGs, and institutional liquidity pools",
                "m5_sniper_entry": "Triggers immediate scalping executions and defines tight stop-losses",
            },
            "timeframe_m5_sniper": m5_metrics,
            "timeframe_m15_setup": m15_metrics,
            "timeframe_h1_trend": h1_metrics,
            "institutional_liquidity_levels": {
                "previous_day_high_pdh": levels.get("pdh"),
                "previous_day_low_pdl": levels.get("pdl"),
                "asian_session_high": levels.get("asian_high"),
                "asian_session_low": levels.get("asian_low"),
            },
            "detected_candlestick_patterns": {
                "m5_sniper": m5_patterns,
                "m15_setup": m15_patterns,
                "h1_trend": h1_patterns,
            },
            "macro_correlations": macro,
            "economic_calendar": {
                "red_folder_alert": calendar.get("red_folder_alert", False),
                "red_folder_details": calendar.get("red_folder_details", ""),
                "next_usd_release": calendar.get("next_event"),
                "today_usd_events_count": len(calendar.get("events_today", [])),
            },
            "top_forex_news_headlines": [item["title"] for item in news[:7]],
        }

    @classmethod
    def call_llm(cls, payload: Dict[str, Any]) -> Optional[str]:
        """
        Sends the market summary JSON to Google Gemini or OpenAI LLM API.
        Enforces the exact required system prompt and markdown formatting.
        """
        payload_str = json.dumps(payload, indent=2)

        # 1. Try Google Gemini API
        if GEMINI_API_KEY and HAS_GOOGLE_GENAI:
            try:
                logger.info("Querying Google Gemini API (%s)...", GEMINI_MODEL)
                client = genai.Client(api_key=GEMINI_API_KEY, http_options={"timeout": 15000})
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=f"Analyze this 3-timeframe market JSON payload:\n\n{payload_str}",
                    config=genai_types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.2,
                    ),
                )
                if response and response.text:
                    logger.info("Received successful prediction from Gemini.")
                    return response.text.strip()
            except Exception as exc:
                logger.warning("Gemini API call failed: %s", exc)

        # 2. Try OpenAI API
        if OPENAI_API_KEY and HAS_OPENAI:
            try:
                logger.info("Querying OpenAI API (%s)...", OPENAI_MODEL)
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                completion = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": payload_str},
                    ],
                    temperature=0.2,
                )
                content = completion.choices[0].message.content
                if content:
                    logger.info("Received successful prediction from OpenAI.")
                    return content.strip()
            except Exception as exc:
                logger.warning("OpenAI API call failed: %s", exc)

        return None

    @classmethod
    def deterministic_rule_based_forecast(cls, payload: Dict[str, Any]) -> str:
        """
        Local quantitative scoring fallback using 3-Timeframe Hierarchy:
          * H1: Dominant Trend Direction
          * M15: Setup & Institutional Liquidity Confirmation
          * M5: Sniper Entry Trigger & Scalping Stop-Loss
        """
        price = payload["current_price"]
        m5 = payload.get("timeframe_m5_sniper", {})
        m15 = payload.get("timeframe_m15_setup", {})
        h1 = payload.get("timeframe_h1_trend", {})
        levels = payload["institutional_liquidity_levels"]
        macro = payload["macro_correlations"]
        calendar = payload["economic_calendar"]

        patterns_m5 = payload["detected_candlestick_patterns"].get("m5_sniper", [])
        patterns_m15 = payload["detected_candlestick_patterns"].get("m15_setup", [])
        patterns_h1 = payload["detected_candlestick_patterns"].get("h1_trend", [])
        all_patterns = patterns_m5 + patterns_m15 + patterns_h1

        atr_5m = m5.get("atr_14", 3.0)
        atr_15m = m15.get("atr_14", 5.5)
        atr_5m_buffer = round(atr_5m * 1.5, 2)
        atr_15m_buffer = round(atr_15m * 1.5, 2)

        rsi_5m = m5.get("rsi_14", 50.0)
        rsi_15m = m15.get("rsi_14", 50.0)
        vwap_m15 = m15.get("vwap", price)
        ema20_h1 = h1.get("ema_20", price)
        ema50_h1 = h1.get("ema_50", price)

        asian_high = levels.get("asian_session_high", price + 10)
        asian_low = levels.get("asian_session_low", price - 10)
        pdh = levels.get("previous_day_high_pdh", price + 15)
        pdl = levels.get("previous_day_low_pdl", price - 15)

        dxy_data = macro.get("dxy", {})
        dxy_trend = dxy_data.get("trend", "NEUTRAL")
        dxy_price = dxy_data.get("price", 100.0)

        us10y_data = macro.get("us10y", {})
        us10y_trend = us10y_data.get("trend", "NEUTRAL")
        us10y_yield = us10y_data.get("yield", 4.5)

        # ----------------------------------------------------------------------
        # RULE 1: Red-Folder News Rule
        # ----------------------------------------------------------------------
        if calendar.get("red_folder_alert", False):
            details = calendar.get("red_folder_details", "High-Impact USD Release Imminent")
            return (
                "📊 **XAU/USD INTRADAY FORECAST (Next 1-4 Candles)**\n"
                "• **Bias:** NO-TRADE NEWS RISK\n"
                "• **Probability Score:** 9/10\n"
                f"• **USD & Macro Driver:** Upcoming High-Impact USD release ({details}); DXY at {dxy_price} and US10Y at {us10y_yield}% with extreme volatility projected.\n"
                "• **Candlestick Projection:** HIGH VOLATILITY WHIPSAW - STAND ASIDE. Expect violent two-sided liquidity wicks hunting stops beyond Asian High and Low before true directional expansion.\n"
                "• **Key Zones to Watch:**\n"
                f"  - Support / Bullish Target: ${min(asian_low, pdl):.2f}\n"
                f"  - Resistance / Bearish Target: ${max(asian_high, pdh):.2f}\n"
                f"  - Recommended M5 ATR Stop Buffer: ${atr_5m_buffer:.2f}\n"
                f"  - Recommended M15 ATR Stop Buffer: ${atr_15m_buffer:.2f}"
            )

        # ----------------------------------------------------------------------
        # RULE 2: Liquidity Sweep Rule
        # ----------------------------------------------------------------------
        bearish_sweep = any("Bearish Liquidity Sweep" in p for p in all_patterns)
        bullish_sweep = any("Bullish Liquidity Sweep" in p for p in all_patterns)

        if bearish_sweep and (rsi_15m > 65 or "BOUNCING" in dxy_trend or "RISING" in dxy_trend):
            support_target = round(max(vwap_m15, asian_low, price - atr_15m * 2), 2)
            resistance_target = round(price + atr_15m, 2)
            return (
                "📊 **XAU/USD INTRADAY FORECAST (Next 1-4 Candles)**\n"
                "• **Bias:** BEARISH\n"
                "• **Probability Score:** 8/10\n"
                f"• **USD & Macro Driver:** DXY is firming near {dxy_price} while US10Y holds {us10y_yield}%, capping bullion after an institutional liquidity sweep above key highs.\n"
                "• **Candlestick Projection:** Projecting bearish rejection candle closing back below Asian High/PDH, targeting internal range liquidity and VWAP.\n"
                "• **Key Zones to Watch:**\n"
                f"  - Support / Bullish Target: ${support_target:.2f}\n"
                f"  - Resistance / Bearish Target: ${resistance_target:.2f}\n"
                f"  - Recommended M5 ATR Stop Buffer: ${atr_5m_buffer:.2f}\n"
                f"  - Recommended M15 ATR Stop Buffer: ${atr_15m_buffer:.2f}"
            )

        if bullish_sweep and (rsi_15m < 35 or "FALLING" in dxy_trend or "SOFTENING" in dxy_trend):
            support_target = round(price - atr_15m, 2)
            resistance_target = round(min(vwap_m15, asian_high, price + atr_15m * 2), 2)
            return (
                "📊 **XAU/USD INTRADAY FORECAST (Next 1-4 Candles)**\n"
                "• **Bias:** BULLISH\n"
                "• **Probability Score:** 8/10\n"
                f"• **USD & Macro Driver:** DXY softens at {dxy_price} and US10Y eases to {us10y_yield}%, triggering aggressive safe-haven bullion absorption off range lows.\n"
                "• **Candlestick Projection:** Projecting bullish expansion candle away from lower liquidity sweep, seeking to fill upper FVG and test EMA 20.\n"
                "• **Key Zones to Watch:**\n"
                f"  - Support / Bullish Target: ${support_target:.2f}\n"
                f"  - Resistance / Bearish Target: ${resistance_target:.2f}\n"
                f"  - Recommended M5 ATR Stop Buffer: ${atr_5m_buffer:.2f}\n"
                f"  - Recommended M15 ATR Stop Buffer: ${atr_15m_buffer:.2f}"
            )

        # ----------------------------------------------------------------------
        # RULE 3: 3-Timeframe Hierarchy Confluence
        # ----------------------------------------------------------------------
        score = 0.0

        # Macro scoring
        if "FALLING" in dxy_trend or "SOFTENING" in dxy_trend:
            score += 1.5
        elif "RISING" in dxy_trend or "BOUNCING" in dxy_trend:
            score -= 1.5

        if "FALLING" in us10y_trend:
            score += 1.5
        elif "RISING" in us10y_trend:
            score -= 1.5

        # H1 Trend Direction
        if price > ema20_h1 and ema20_h1 > ema50_h1:
            score += 2.0
        elif price < ema20_h1 and ema20_h1 < ema50_h1:
            score -= 2.0

        # M15 Setup Structure
        if price > vwap_m15:
            score += 1.0
        else:
            score -= 1.0

        # M5 Sniper Trigger
        if rsi_5m > 55 and any("Bullish" in p for p in patterns_m5):
            score += 1.5
        elif rsi_5m < 45 and any("Bearish" in p for p in patterns_m5):
            score -= 1.5

        # Classify final bias
        if score >= 2.5:
            bias = "BULLISH"
            prob = min(9, int(6 + abs(score) * 0.5))
            support_target = round(max(asian_low, price - atr_15m * 1.5), 2)
            resistance_target = round(min(asian_high, pdh, price + atr_15m * 2.0), 2)
            if resistance_target <= price:
                resistance_target = round(price + atr_15m * 2.0, 2)
            candlestick_proj = (
                "Expect bullish continuation with shallow lower wicks honoring the M5 20 EMA and expanding toward upper liquidity pools."
            )
            macro_driver = (
                f"Macro tailwinds from subdued DXY ({dxy_price}) and Treasury yields ({us10y_yield}%) reinforce bullion demand."
            )
        elif score <= -2.5:
            bias = "BEARISH"
            prob = min(9, int(6 + abs(score) * 0.5))
            support_target = round(max(asian_low, pdl, price - atr_15m * 2.0), 2)
            if support_target >= price:
                support_target = round(price - atr_15m * 2.0, 2)
            resistance_target = round(min(asian_high, price + atr_15m * 1.5), 2)
            candlestick_proj = (
                "Expect bearish displacement with upper rejection wicks below VWAP, targeting the next fair value discount zone."
            )
            macro_driver = (
                f"Firmer DXY ({dxy_price}) and yields ({us10y_yield}%) exerting downward structural pressure on spot gold."
            )
        else:
            bias = "RANGE-BOUND"
            prob = 6
            support_target = round(asian_low, 2)
            resistance_target = round(asian_high, 2)
            candlestick_proj = (
                "Anticipate rotational, alternating candles oscillating between Asian High and Asian Low without directional breakout."
            )
            macro_driver = (
                f"Conflicting macro signals between DXY ({dxy_price}) and US10Y ({us10y_yield}%) keeping gold boxed in range."
            )

        return (
            "📊 **XAU/USD INTRADAY FORECAST (Next 1-4 Candles)**\n"
            f"• **Bias:** {bias}\n"
            f"• **Probability Score:** {prob}/10\n"
            f"• **USD & Macro Driver:** {macro_driver}\n"
            f"• **Candlestick Projection:** {candlestick_proj}\n"
            "• **Key Zones to Watch:**\n"
            f"  - Support / Bullish Target: ${support_target:.2f}\n"
            f"  - Resistance / Bearish Target: ${resistance_target:.2f}\n"
            f"  - Recommended M5 ATR Stop Buffer: ${atr_5m_buffer:.2f}\n"
            f"  - Recommended M15 ATR Stop Buffer: ${atr_15m_buffer:.2f}"
        )


# ======================================================================================
# MODULE 4: DISCORD BOT EMBEDS, DISPATCH & SYNCHRONOUS ANALYSIS PIPELINE
# ======================================================================================

class DispatchEngine:
    """
    Dispatches formatted intelligence reports to Discord Webhooks and Telegram Bots.
    """

    @staticmethod
    def send_to_discord(message: str) -> bool:
        """Sends message to Discord Webhook."""
        if not DISCORD_WEBHOOK_URL:
            return False

        try:
            payload = {"content": message}
            resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
            if resp.status_code in [200, 204]:
                logger.info("Successfully pushed forecast to Discord Webhook.")
                return True
            else:
                logger.warning("Discord Webhook responded with status %d: %s", resp.status_code, resp.text)
                return False
        except Exception as exc:
            logger.warning("Failed to dispatch to Discord Webhook: %s", exc)
            return False

    @staticmethod
    def send_to_telegram(message: str) -> bool:
        """Sends message to Telegram Bot chat."""
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return False

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        try:
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "Markdown",
            }
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info("Successfully pushed forecast to Telegram chat %s.", TELEGRAM_CHAT_ID)
                return True

            logger.warning("Telegram Markdown rejected (%s). Retrying as plain text...", resp.text)
            payload.pop("parse_mode", None)
            retry_resp = requests.post(url, json=payload, timeout=10)
            if retry_resp.status_code == 200:
                logger.info("Sent plain-text fallback to Telegram successfully.")
                return True
            else:
                logger.warning("Telegram dispatch failed: %s", retry_resp.text)
                return False
        except Exception as exc:
            logger.warning("Failed to dispatch to Telegram: %s", exc)
            return False

    @classmethod
    def broadcast(cls, forecast_text: str):
        """Broadcasts report across configured webhook & telegram channels."""
        discord_sent = cls.send_to_discord(forecast_text)
        telegram_sent = cls.send_to_telegram(forecast_text)

        if not discord_sent and not telegram_sent:
            logger.info("No active webhooks configured. Forecast outputting to stdout only.")


# --------------------------------------------------------------------------------------
# DISCORD EMBED BUILDERS
# --------------------------------------------------------------------------------------

def get_bias_color(bias: str) -> discord.Color:
    """
    Maps trading bias to the exact Discord Embed color requirement:
      - Green  = BULLISH
      - Red    = BEARISH
      - Gold   = RANGE-BOUND
      - Orange = NO-TRADE NEWS RISK
    """
    b = bias.upper()
    if "BULLISH" in b:
        return discord.Color.green()
    elif "BEARISH" in b:
        return discord.Color.red()
    elif "NO-TRADE" in b or "RISK" in b or "STAND ASIDE" in b:
        return discord.Color.orange()
    else:
        return discord.Color.gold()


def parse_forecast_components(forecast_text: str) -> Dict[str, Any]:
    """Parses raw forecast text into individual metric fields."""
    bias_match = re.search(r"• \*\*Bias:\*\*\s*([^\n]+)", forecast_text)
    prob_match = re.search(r"• \*\*Probability Score:\*\*\s*([0-9.]+)", forecast_text)
    driver_match = re.search(r"• \*\*USD & Macro Driver:\*\*\s*([^\n]+)", forecast_text)
    proj_match = re.search(r"• \*\*Candlestick Projection:\*\*\s*([^\n]+)", forecast_text)
    supp_match = re.search(r"Support / Bullish Target:\s*[\$]?([0-9.,]+)", forecast_text)
    res_match = re.search(r"Resistance / Bearish Target:\s*[\$]?([0-9.,]+)", forecast_text)
    atr5_match = re.search(r"Recommended M5 ATR Stop Buffer:\s*[\$]?([0-9.,]+)", forecast_text)
    atr15_match = re.search(r"Recommended M15 ATR Stop Buffer:\s*[\$]?([0-9.,]+)", forecast_text)
    legacy_atr_match = re.search(r"Recommended ATR Stop Buffer:\s*[\$]?([0-9.,]+)", forecast_text)

    bias = bias_match.group(1).strip() if bias_match else "RANGE-BOUND"
    prob_str = prob_match.group(1).strip() if prob_match else "6"
    try:
        prob = float(prob_str)
    except ValueError:
        prob = 6.0

    driver = driver_match.group(1).strip() if driver_match else "Macro conditions neutral."
    proj = proj_match.group(1).strip() if proj_match else "Candlestick consolidation expected."
    support = supp_match.group(1).strip() if supp_match else "N/A"
    resistance = res_match.group(1).strip() if res_match else "N/A"
    atr_m5 = atr5_match.group(1).strip() if atr5_match else "N/A"
    atr_m15 = atr15_match.group(1).strip() if atr15_match else (legacy_atr_match.group(1).strip() if legacy_atr_match else "N/A")

    return {
        "bias": bias,
        "probability": prob,
        "driver": driver,
        "projection": proj,
        "support": support,
        "resistance": resistance,
        "atr_m5": atr_m5,
        "atr_m15": atr_m15,
    }


def build_forecast_embed(analysis: Dict[str, Any]) -> discord.Embed:
    """Builds a rich color-coded Embed for the forecast report with 3-timeframe indicators & active strategies."""
    parsed = analysis["parsed"]
    payload = analysis["payload"]
    levels = payload["institutional_liquidity_levels"]
    macro = payload["macro_correlations"]
    calendar = payload["economic_calendar"]
    current_price = payload["current_price"]
    source_name = payload["data_source"]

    m5 = payload.get("timeframe_m5_sniper", {})
    m15 = payload.get("timeframe_m15_setup", {})
    h1 = payload.get("timeframe_h1_trend", {})

    color = get_bias_color(parsed["bias"])

    embed = discord.Embed(
        title="📊 XAU/USD INTRADAY FORECAST (Next 1-4 Candles)",
        description=(
            f"**• Bias:** `{parsed['bias']}`\n"
            f"**• Probability Score:** `{parsed['probability']}/10`\n"
            f"**• USD & Macro Driver:** {parsed['driver']}\n"
            f"**• Candlestick Projection:** {parsed['projection']}"
        ),
        color=color,
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )

    embed.add_field(
        name="🎯 Key Targets & Dynamic ATR Stop Zones",
        value=(
            f"**Support / Bullish Target:** `${parsed['support']}`\n"
            f"**Resistance / Bearish Target:** `${parsed['resistance']}`\n"
            f"**M5 Tight Scalp Stop (1.5x ATR):** `${parsed['atr_m5']}`\n"
            f"**M15 Structural Stop (1.5x ATR):** `${parsed['atr_m15']}`"
        ),
        inline=False,
    )

    # Institutional Strategy Setups Summary field
    strategies = analysis.get("strategies", {})
    if strategies:
        strat_lines = []
        for key, s in strategies.items():
            status_badge = s.get("status", "❌ INACTIVE")
            strat_name = s.get("name", key)
            direction = f" ({s['direction']})" if s.get("direction") and s["direction"] != "NEUTRAL" else ""

            if "ACTIVE" in status_badge:
                strat_lines.append(
                    f"• **{strat_name}:** {status_badge}{direction}\n"
                    f"  └ **{s.get('entry_label', 'Entry')}:** `${s.get('entry', 0):.2f}` | **SL:** `${s.get('sl', 0):.2f}` | **TP:** `${s.get('tp1', 0):.2f}`"
                )
            elif "FORMING" in status_badge:
                strat_lines.append(f"• **{strat_name}:** {status_badge}{direction}\n  └ _{s.get('explanation', 'Forming')}_")
            else:
                strat_lines.append(f"• **{strat_name}:** {status_badge}\n  └ _{s.get('explanation', 'Inactive')}_")

        embed.add_field(
            name="🧠 Active Institutional Strategy Signals",
            value="\n".join(strat_lines),
            inline=False,
        )

    embed.add_field(
        name="📐 3-Timeframe Plain-English Hierarchy",
        value=(
            f"**H1 (Trend):** 20 EMA `${h1.get('ema_20', 0):.2f}` | 50 EMA `${h1.get('ema_50', 0):.2f}` | RSI `{h1.get('rsi_14', 50):.1f}`\n"
            f"**M15 (Setup):** VWAP `${m15.get('vwap', 0):.2f}` | 20 EMA `${m15.get('ema_20', 0):.2f}` | RSI `{m15.get('rsi_14', 50):.1f}`\n"
            f"**M5 (Trigger):** VWAP `${m5.get('vwap', 0):.2f}` | 20 EMA `${m5.get('ema_20', 0):.2f}` | RSI `{m5.get('rsi_14', 50):.1f}`"
        ),
        inline=False,
    )

    embed.add_field(
        name="🏛️ Institutional Liquidity Levels",
        value=(
            f"**Morning Ceiling (Asian High):** `${levels.get('asian_session_high', 0):.2f}`\n"
            f"**Morning Floor (Asian Low):** `${levels.get('asian_session_low', 0):.2f}`\n"
            f"**Yesterday Ceiling (PDH):** `${levels.get('previous_day_high_pdh', 0):.2f}`\n"
            f"**Yesterday Floor (PDL):** `${levels.get('previous_day_low_pdl', 0):.2f}`"
        ),
        inline=True,
    )

    m5_pats = payload["detected_candlestick_patterns"].get("m5_sniper", [])
    m15_pats = payload["detected_candlestick_patterns"].get("m15_setup", [])
    h1_pats = payload["detected_candlestick_patterns"].get("h1_trend", [])

    embed.add_field(
        name="🕯️ Price Action Patterns",
        value=(
            f"**M5 Sniper:** {', '.join(m5_pats) if m5_pats else 'None'}\n"
            f"**M15 Setup:** {', '.join(m15_pats) if m15_pats else 'None'}\n"
            f"**H1 Trend:** {', '.join(h1_pats) if h1_pats else 'None'}"
        ),
        inline=True,
    )

    dxy = macro.get("dxy", {})
    us10y = macro.get("us10y", {})
    next_ev = calendar.get("next_usd_release")
    next_ev_text = (
        f"{next_ev['title']} ({next_ev['impact']}) in {next_ev['minutes_remaining']:.0f}m"
        if next_ev else "None today"
    )

    embed.add_field(
        name="🌐 Macro & News Backdrop",
        value=(
            f"**DXY Index:** `{dxy.get('price', 100.0)}` ({dxy.get('trend', 'NEUTRAL')})\n"
            f"**US 10Y Yield:** `{us10y.get('yield', 4.5)}%` ({us10y.get('trend', 'NEUTRAL')})\n"
            f"**Next USD Event:** {next_ev_text}"
        ),
        inline=False,
    )

    embed.set_footer(
        text=f"Live Price: ${current_price:.2f} | Feed: {source_name} | Antigravity Quant Bot"
    )
    return embed


def build_error_embed(title: str, description: str, hint: str = "") -> discord.Embed:
    """Builds a standardized warning/advisory embed for gracefully handled exceptions."""
    embed = discord.Embed(
        title=f"⚠️ {title}",
        description=description,
        color=discord.Color.orange(),
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )
    if hint:
        embed.add_field(name="💡 Advisory / Watchdog Notice", value=hint, inline=False)
    embed.set_footer(text="CapeTrader Quantitative Engine • Fault-Tolerant Watchdog")
    return embed


def build_levels_embed(levels_data: Dict[str, Any]) -> discord.Embed:
    """Builds a rich Embed showing live technical indicators across M5, M15, and H1, plus a strategy preview snippet."""
    levels = levels_data["levels"]
    m5 = levels_data["m5"]
    m15 = levels_data["m15"]
    h1 = levels_data["h1"]
    current_price = levels_data["current_price"]
    source_name = levels_data["source_name"]
    macro = levels_data["macro"]
    strategies = levels_data.get("strategies", {})

    active_count = sum(1 for s in strategies.values() if "ACTIVE" in s.get("status", ""))
    embed_color = discord.Color.green() if active_count > 0 else discord.Color.blue()

    embed = discord.Embed(
        title="🏛️ XAU/USD Institutional Levels & 3-Timeframe Metrics",
        description=f"**Current Price:** `${current_price:.2f}` (Feed: `{source_name}`)",
        color=embed_color,
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )

    # 1. Institutional Strategy Status Preview Snippet
    if strategies:
        strat_lines = []
        has_active = False
        s_order = [
            ("⚡ 1. Judas Swing", strategies.get("judas_swing", {})),
            ("🏰 2. Breakout & Retest", strategies.get("breakout_retest", {})),
            ("🌊 3. Trend Scalper", strategies.get("trend_pullback", {})),
        ]
        for name_label, s_data in s_order:
            st = s_data.get("status", "❌ INACTIVE")
            dir_str = f" ({s_data['direction']})" if s_data.get("direction") and s_data["direction"] != "NEUTRAL" else ""
            if "ACTIVE" in st:
                has_active = True
                strat_lines.append(
                    f"• **{name_label}:** {st}{dir_str} — **Entry:** `${s_data.get('entry', 0):.2f}` | **SL:** `${s_data.get('sl', 0):.2f}` | **TP:** `${s_data.get('tp1', 0):.2f}`"
                )
            elif "FORMING" in st:
                strat_lines.append(f"• **{name_label}:** {st}{dir_str} — _{s_data.get('explanation', 'Monitoring setup')}_")
            else:
                strat_lines.append(f"• **{name_label}:** {st}")

        if has_active:
            strat_lines.append("👉 *Active institutional setup detected! Run `/strategies` or `/predict` to execute.*")

        embed.add_field(
            name="🧠 Institutional Strategy Status Preview",
            value="\n".join(strat_lines),
            inline=False,
        )

    embed.add_field(
        name="📊 Institutional Liquidity Zones",
        value=(
            f"**Yesterday's Ceiling (PDH):** `${levels['pdh']:.2f}`\n"
            f"**Yesterday's Floor (PDL):** `${levels['pdl']:.2f}`\n"
            f"**Morning Ceiling (Asian High):** `${levels['asian_high']:.2f}` (01:00-09:00 CAT)\n"
            f"**Morning Floor (Asian Low):** `${levels['asian_low']:.2f}` (01:00-09:00 CAT)"
        ),
        inline=False,
    )

    embed.add_field(
        name="⚡ M5 Sniper Indicators (5-Minute)",
        value=(
            f"**RSI (14):** `{m5['rsi_14']}`\n"
            f"**VWAP:** `${m5['vwap']:.2f}`\n"
            f"**20 EMA:** `${m5['ema_20']:.2f}`\n"
            f"**50 EMA:** `${m5['ema_50']:.2f}`\n"
            f"**ATR (14):** `${m5['atr_14']:.2f}` (Tight Scalp: `${m5['atr_14'] * 1.5:.2f}`)"
        ),
        inline=True,
    )

    embed.add_field(
        name="📈 M15 Setup Indicators (15-Minute)",
        value=(
            f"**RSI (14):** `{m15['rsi_14']}`\n"
            f"**VWAP:** `${m15['vwap']:.2f}`\n"
            f"**20 EMA:** `${m15['ema_20']:.2f}`\n"
            f"**50 EMA:** `${m15['ema_50']:.2f}`\n"
            f"**ATR (14):** `${m15['atr_14']:.2f}` (Swing: `${m15['atr_14'] * 1.5:.2f}`)"
        ),
        inline=True,
    )

    embed.add_field(
        name="⏱️ H1 Trend Indicators (1-Hour)",
        value=(
            f"**RSI (14):** `{h1['rsi_14']}`\n"
            f"**VWAP:** `${h1['vwap']:.2f}`\n"
            f"**20 EMA:** `${h1['ema_20']:.2f}`\n"
            f"**50 EMA:** `${h1['ema_50']:.2f}`\n"
            f"**ATR (14):** `${h1['atr_14']:.2f}`"
        ),
        inline=True,
    )

    dxy = macro.get("dxy", {})
    us10y = macro.get("us10y", {})
    embed.add_field(
        name="🌐 Macro Correlations",
        value=(
            f"**US Dollar Index (DXY):** `{dxy.get('price', 100.0)}` ({dxy.get('trend', 'NEUTRAL')})\n"
            f"**US 10Y Yield (^TNX):** `{us10y.get('yield', 4.5)}%` ({us10y.get('trend', 'NEUTRAL')})"
        ),
        inline=False,
    )

    embed.set_footer(text="CapeTrader Quantitative Trading Bot • Institutional Metrics")
    return embed


def build_news_embed(calendar: Dict[str, Any], news: List[Dict[str, str]]) -> discord.Embed:
    """Builds a rich Embed displaying the economic calendar and breaking news."""
    is_red = calendar.get("red_folder_alert", False)
    color = discord.Color.orange() if is_red else discord.Color.teal()

    embed = discord.Embed(
        title="📰 USD Forex Calendar & Breaking Market News",
        color=color,
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )

    if is_red:
        embed.description = f"⚠️ **HIGH IMPACT RED-FOLDER RISK:** {calendar.get('red_folder_details')}"

    events_today = calendar.get("events_today", [])
    if events_today:
        lines = []
        for ev in events_today[:6]:
            diff = ev["minutes_remaining"]
            time_str = f"in {diff:.0f}m" if diff >= 0 else f"{abs(diff):.0f}m ago"
            lines.append(f"• **[{ev['impact']}] {ev['title']}** — {time_str} (`{ev['date_utc']}`)")
        embed.add_field(
            name="📅 Today's USD Economic Calendar (Forex Factory)",
            value="\n".join(lines),
            inline=False,
        )
    else:
        embed.add_field(
            name="📅 Today's USD Economic Calendar",
            value="No High or Medium impact USD releases scheduled for today.",
            inline=False,
        )

    if news:
        news_lines = []
        for item in news[:7]:
            news_lines.append(f"• `[{item['source']}]` {item['title']}")
        embed.add_field(
            name="⚡ Breaking Forex Headlines (ForexLive & FXStreet)",
            value="\n".join(news_lines),
            inline=False,
        )
    else:
        embed.add_field(
            name="⚡ Breaking Forex Headlines",
            value="No recent breaking headlines fetched.",
            inline=False,
        )

    embed.set_footer(text="Forex Intelligence Engine • Real-time Feeds")
    return embed


def build_strategies_embed(
    strategies: Dict[str, Dict[str, Any]],
    current_price: float,
    levels: Dict[str, float],
    m5_metrics: Optional[Dict[str, Any]] = None,
    m15_metrics: Optional[Dict[str, Any]] = None,
    source_name: str = "MT5/yfinance",
) -> discord.Embed:
    """Builds a dedicated rich Discord Embed displaying the 3 institutional strategies with live technical metrics."""
    active_count = sum(1 for s in strategies.values() if "ACTIVE" in s.get("status", ""))
    forming_count = sum(1 for s in strategies.values() if "FORMING" in s.get("status", ""))

    if active_count > 0:
        color = discord.Color.green()
        for s in strategies.values():
            if "ACTIVE" in s.get("status", "") and s.get("direction") == "BEARISH":
                color = discord.Color.red()
                break
    elif forming_count > 0:
        color = discord.Color.gold()
    else:
        color = discord.Color.blue()

    desc_lines = [
        f"**Live Gold Price:** `${current_price:.2f}` (Feed: `{source_name}`)"
    ]
    if m5_metrics and m15_metrics:
        desc_lines.append(
            f"**M5 Sniper Setup:** VWAP `${m5_metrics.get('vwap', 0):.2f}` | 20 EMA `${m5_metrics.get('ema_20', 0):.2f}` | RSI `{m5_metrics.get('rsi_14', 50):.1f}` | ATR `${m5_metrics.get('atr_14', 0):.2f}`\n"
            f"**M15 Structure:** VWAP `${m15_metrics.get('vwap', 0):.2f}` | 20 EMA `${m15_metrics.get('ema_20', 0):.2f}` | RSI `{m15_metrics.get('rsi_14', 50):.1f}`"
        )
    desc_lines.append(
        f"**Asian Session Range:** `${levels.get('asian_low', 0):.2f}` – `${levels.get('asian_high', 0):.2f}` | **Yesterday:** `${levels.get('pdl', 0):.2f}` – `${levels.get('pdh', 0):.2f}`"
    )

    embed = discord.Embed(
        title="🏛️ XAU/USD Institutional Strategy Scanner",
        description="\n".join(desc_lines),
        color=color,
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )

    # Strategy 1: Asian Session Liquidity Sweep (Judas Swing)
    s1 = strategies.get("judas_swing", {})
    s1_status = s1.get("status", "❌ INACTIVE")
    s1_dir = f" ({s1.get('direction')})" if s1.get("direction") and s1.get("direction") != "NEUTRAL" else ""
    if "ACTIVE" in s1_status:
        s1_val = (
            f"**Status:** {s1_status}{s1_dir}\n"
            f"**Trigger:** {s1.get('trigger')}\n"
            f"• **Execution:** `Market Order` at `${s1.get('entry', 0):.2f}`\n"
            f"• **Stop-Loss:** `${s1.get('sl', 0):.2f}` (Risk: `${s1.get('sl_distance', 0):.2f}`)\n"
            f"• **Take-Profit 1 (VWAP):** `${s1.get('tp1', 0):.2f}` | **TP2 (Range End):** `${s1.get('tp2', 0):.2f}`\n"
            f"• **Risk-to-Reward:** `{s1.get('rrr', '1:2.0')}`"
        )
    elif "FORMING" in s1_status:
        s1_val = f"**Status:** {s1_status}{s1_dir}\n**Details:** {s1.get('explanation')}"
    else:
        s1_val = f"**Status:** {s1_status}\n**Details:** {s1.get('explanation', 'No sweep observed.')}"
    embed.add_field(name="🎯 1. Asian Liquidity Sweep (Judas Swing)", value=s1_val, inline=False)

    # Strategy 2: London/NY Breakout & Retest
    s2 = strategies.get("breakout_retest", {})
    s2_status = s2.get("status", "❌ INACTIVE")
    s2_dir = f" ({s2.get('direction')})" if s2.get("direction") and s2.get("direction") != "NEUTRAL" else ""
    if "ACTIVE" in s2_status:
        s2_val = (
            f"**Status:** {s2_status}{s2_dir}\n"
            f"**Trigger:** {s2.get('trigger')}\n"
            f"• **Retest Zone:** `{s2.get('entry_zone', 'N/A')}`\n"
            f"• **Stop-Loss:** `${s2.get('sl', 0):.2f}` (1.5x M5 ATR inside broken level)\n"
            f"• **Take-Profit (1:2 RRR):** `${s2.get('tp1', 0):.2f}` | **TP2 (1:3 RRR):** `${s2.get('tp2', 0):.2f}`"
        )
    elif "FORMING" in s2_status:
        s2_val = f"**Status:** {s2_status}{s2_dir}\n**Details:** {s2.get('explanation')}"
    else:
        s2_val = f"**Status:** {s2_status}\n**Details:** {s2.get('explanation', 'Inside range.')}"
    embed.add_field(name="🚀 2. London/NY Breakout & Retest", value=s2_val, inline=False)

    # Strategy 3: EMA 20/50 + VWAP Trend Pullback Scalper
    s3 = strategies.get("trend_pullback", {})
    s3_status = s3.get("status", "❌ INACTIVE")
    s3_dir = f" ({s3.get('direction')})" if s3.get("direction") and s3.get("direction") != "NEUTRAL" else ""
    if "ACTIVE" in s3_status:
        s3_val = (
            f"**Status:** {s3_status}{s3_dir}\n"
            f"**Trigger:** {s3.get('trigger')}\n"
            f"• **Entry Price:** `${s3.get('entry', 0):.2f}`\n"
            f"• **Stop-Loss (1.5x ATR):** `${s3.get('sl', 0):.2f}`\n"
            f"• **Take-Profit 1 (1:2 RRR):** `${s3.get('tp1', 0):.2f}` | **TP2 (1:3 RRR):** `${s3.get('tp2', 0):.2f}`"
        )
    elif "FORMING" in s3_status:
        s3_val = f"**Status:** {s3_status}{s3_dir}\n**Details:** {s3.get('explanation')}"
    else:
        s3_val = f"**Status:** {s3_status}\n**Details:** {s3.get('explanation', 'Consolidating / no trend.')}"
    embed.add_field(name="🌊 3. EMA 20/50 + VWAP Trend Pullback Scalper", value=s3_val, inline=False)

    embed.set_footer(text="Antigravity Quantitative Trading Bot • Institutional Strategy Engine")
    return embed


def build_calc_embed(
    balance: float,
    risk_percent: float,
    stop_loss_dollars: Optional[float] = None,
    atr_m5: Optional[float] = None,
    atr_m15: Optional[float] = None,
    current_price: Optional[float] = None,
) -> discord.Embed:
    """
    Builds a beginner-friendly Discord Embed for MT5 Spot Gold Lot Size & Risk Calculations.
    Spot Gold Math:
      - 1.00 Standard Lot = 100 Troy Oz ($1.00 move = $100.00 USD)
      - 0.10 Mini Lot     = 10 Troy Oz  ($1.00 move = $10.00 USD)
      - 0.01 Micro Lot    = 1 Troy Oz   ($1.00 move = $1.00 USD)
    """
    risk_amount_usd = balance * (risk_percent / 100.0)

    embed = discord.Embed(
        title="🧮 MT5 XAU/USD Lot Size & Risk Calculator",
        description=(
            f"**Account Balance:** `${balance:,.2f} USD`\n"
            f"**Desired Risk:** `{risk_percent:.1f}%` = **`${risk_amount_usd:.2f} USD` Max Risk**"
            + (f"\n**Live Gold Price:** `${current_price:,.2f}`" if current_price else "")
        ),
        color=discord.Color.teal(),
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )

    if stop_loss_dollars is not None:
        # Custom stop-loss distance
        sl_dist = float(stop_loss_dollars)
        exact_lot = risk_amount_usd / (sl_dist * 100.0)
        rec_lot = max(0.01, round(exact_lot, 2))
        actual_dollar_risk = rec_lot * sl_dist * 100.0
        actual_risk_pct = (actual_dollar_risk / balance) * 100.0

        tp1_dist = 2.0 * sl_dist
        tp1_profit = rec_lot * tp1_dist * 100.0
        tp2_dist = 3.0 * sl_dist
        tp2_profit = rec_lot * tp2_dist * 100.0

        if rec_lot >= 1.0:
            lot_type = f"({rec_lot:.2f} Standard Lots)"
        elif rec_lot >= 0.10:
            lot_type = f"({rec_lot * 10:.1f} Mini Lots)"
        else:
            lot_type = f"({int(round(rec_lot * 100))} Micro Lots)"

        embed.add_field(
            name=f"📐 Custom Stop-Loss Distance: ${sl_dist:.2f}",
            value=(
                f"• **Recommended MT5 Lot Size:** `{rec_lot:.2f} Lots` {lot_type}\n"
                f"• **Exact Mathematical Sizing:** `{exact_lot:.4f} Lots`\n"
                f"• **Actual Dollar Risk:** `${actual_dollar_risk:.2f}` ({actual_risk_pct:.2f}% of balance)"
            ),
            inline=False,
        )

        embed.add_field(
            name="🎯 Take-Profit Projections (RRR)",
            value=(
                f"• **1:2 Target (+${tp1_dist:.2f}):** Profit = **+${tp1_profit:.2f} USD** (+{(tp1_profit / balance) * 100:.1f}%)\n"
                f"• **1:3 Target (+${tp2_dist:.2f}):** Profit = **+${tp2_profit:.2f} USD** (+{(tp2_profit / balance) * 100:.1f}%)"
            ),
            inline=False,
        )

        if exact_lot < 0.009:
            embed.add_field(
                name="⚠️ MINIMUM LOT SIZE RISK WARNING",
                value=(
                    f"Your desired risk is **${risk_amount_usd:.2f} ({risk_percent:.1f}%)**, but the broker minimum order size is **0.01 Lots** (1 Micro Lot).\n"
                    f"At a **${sl_dist:.2f}** stop-loss, trading 0.01 lots will risk **${actual_dollar_risk:.2f} ({actual_risk_pct:.2f}% of your account)**."
                ),
                inline=False,
            )
    else:
        # Automated Live ATR Stop-Loss mode (M5 Scalp and M15 Swing)
        sl_m5 = round((atr_m5 or 3.0) * 1.5, 2)
        sl_m15 = round((atr_m15 or 5.5) * 1.5, 2)

        # Setup A: M5 Sniper Scalp
        exact_lot_m5 = risk_amount_usd / (sl_m5 * 100.0)
        rec_lot_m5 = max(0.01, round(exact_lot_m5, 2))
        actual_risk_m5 = rec_lot_m5 * sl_m5 * 100.0
        actual_pct_m5 = (actual_risk_m5 / balance) * 100.0
        tp1_m5 = rec_lot_m5 * (2.0 * sl_m5) * 100.0
        tp2_m5 = rec_lot_m5 * (3.0 * sl_m5) * 100.0

        embed.add_field(
            name=f"⚡ Setup A: M5 Sniper Scalp Stop (${sl_m5:.2f})",
            value=(
                f"• **Stop Distance (1.5x M5 ATR):** `${sl_m5:.2f}`\n"
                f"• **Recommended MT5 Lot Size:** `{rec_lot_m5:.2f} Lots` (Exact: `{exact_lot_m5:.4f}`)\n"
                f"• **Actual Dollar Risk:** `${actual_risk_m5:.2f}` ({actual_pct_m5:.2f}% of account)\n"
                f"• **1:2 TP (+${sl_m5 * 2:.2f}):** `+${tp1_m5:.2f} USD` | **1:3 TP (+${sl_m5 * 3:.2f}):** `+${tp2_m5:.2f} USD`"
            ),
            inline=False,
        )

        # Setup B: M15 Structural Swing
        exact_lot_m15 = risk_amount_usd / (sl_m15 * 100.0)
        rec_lot_m15 = max(0.01, round(exact_lot_m15, 2))
        actual_risk_m15 = rec_lot_m15 * sl_m15 * 100.0
        actual_pct_m15 = (actual_risk_m15 / balance) * 100.0
        tp1_m15 = rec_lot_m15 * (2.0 * sl_m15) * 100.0
        tp2_m15 = rec_lot_m15 * (3.0 * sl_m15) * 100.0

        embed.add_field(
            name=f"📈 Setup B: M15 Structural Swing Stop (${sl_m15:.2f})",
            value=(
                f"• **Stop Distance (1.5x M15 ATR):** `${sl_m15:.2f}`\n"
                f"• **Recommended MT5 Lot Size:** `{rec_lot_m15:.2f} Lots` (Exact: `{exact_lot_m15:.4f}`)\n"
                f"• **Actual Dollar Risk:** `${actual_risk_m15:.2f}` ({actual_pct_m15:.2f}% of account)\n"
                f"• **1:2 TP (+${sl_m15 * 2:.2f}):** `+${tp1_m15:.2f} USD` | **1:3 TP (+${sl_m15 * 3:.2f}):** `+${tp2_m15:.2f} USD`"
            ),
            inline=False,
        )

        if exact_lot_m5 < 0.009 or exact_lot_m15 < 0.009:
            embed.add_field(
                name="⚠️ MINIMUM LOT SIZE RISK WARNING",
                value=(
                    f"Your desired risk is **${risk_amount_usd:.2f} ({risk_percent:.1f}%)**.\n"
                    f"Because minimum broker order size is **0.01 Lots** ($1/dollar move), small account balances may risk more than your target percentage."
                ),
                inline=False,
            )

    embed.add_field(
        name="💡 Spot Gold (XAU/USD) Contract Guide",
        value=(
            "• **1.00 Standard Lot** = 100 Troy Oz ($1.00 Gold move = $100.00 USD)\n"
            "• **0.10 Mini Lot** = 10 Troy Oz ($1.00 Gold move = $10.00 USD)\n"
            "• **0.01 Micro Lot** = 1 Troy Oz ($1.00 Gold move = $1.00 USD / 1¢ move = 1¢)"
        ),
        inline=False,
    )

    embed.set_footer(text="Antigravity Risk Engine • MT5 Spot Gold Calculator")
    return embed


def build_trade_open_embed(trade: Dict[str, Any]) -> discord.Embed:
    """Builds a rich Discord Embed when a new paper trade is opened and tracked."""
    entry = float(trade["entry_price"])
    sl = float(trade["stop_loss"])
    tp = float(trade["take_profit"])
    direction = trade["direction"].upper()
    sl_dist = abs(entry - sl)
    tp_dist = abs(tp - entry)
    rrr = round(tp_dist / max(0.1, sl_dist), 2)

    color = discord.Color.green() if direction == "BUY" else discord.Color.red()
    dir_badge = "🟢 BUY (Long)" if direction == "BUY" else "🔴 SELL (Short)"

    embed = discord.Embed(
        title=f"📝 Paper Trade Logged • Trade #{trade['id']}",
        description=(
            f"**Strategy #{trade['strategy_id']}:** `{trade['strategy_name']}`\n"
            f"**Direction:** `{dir_badge}`\n"
            f"**Entry Execution:** `${entry:.2f}`\n"
            f"**Auto-Monitoring:** Active via 15-Minute Candle Close Loop"
        ),
        color=color,
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )
    embed.add_field(
        name="🛡️ Risk & Target Parameters",
        value=(
            f"• **Stop-Loss:** `${sl:.2f}` (Risk: `${sl_dist:.2f}`)\n"
            f"• **Take-Profit:** `${tp:.2f}` (Target: `${tp_dist:.2f}`)\n"
            f"• **Risk-to-Reward Ratio (RRR):** `1:{rrr:.1f}`"
        ),
        inline=False,
    )
    embed.set_footer(text="CapeTrader Paper Trading & Auto-Journaling System • Status: OPEN")
    return embed


def build_trade_close_embed(trade: Dict[str, Any]) -> discord.Embed:
    """Builds a Discord Embed when a paper trade is manually closed."""
    status = trade.get("status", "CLOSED").upper()
    pnl = trade.get("pnl_dollars", 0.0) or 0.0
    is_won = status == "WON" or pnl >= 0
    color = discord.Color.green() if is_won else discord.Color.red()
    result_icon = "🏆 WIN" if is_won else "🛑 LOSS"
    pnl_sign = "+" if pnl >= 0 else ""

    embed = discord.Embed(
        title=f"📁 Paper Trade #{trade['id']} Closed • {result_icon}",
        description=(
            f"**Strategy #{trade['strategy_id']}:** `{trade['strategy_name']}`\n"
            f"**Direction:** `{trade['direction']}`\n"
            f"**Entry Price:** `${float(trade['entry_price']):.2f}`\n"
            f"**Exit Price:** `${float(trade['close_price']):.2f}`\n"
            f"**Net PnL:** **`{pnl_sign}${pnl:.2f} Gold Points ($)`**"
        ),
        color=color,
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )
    embed.set_footer(text="CapeTrader Paper Trading & Auto-Journaling System • Status: CLOSED")
    return embed


def build_trade_resolved_embed(trade: Dict[str, Any]) -> discord.Embed:
    """Builds a Discord Embed when a paper trade is automatically hit (WON/LOST) by the 15m loop."""
    status = trade.get("status", "CLOSED").upper()
    is_won = status == "WON"
    color = discord.Color.gold() if is_won else discord.Color.red()
    title = f"🎉 TAKE-PROFIT HIT! • Paper Trade #{trade['id']}" if is_won else f"🛑 STOP-LOSS HIT • Paper Trade #{trade['id']}"
    pnl = trade.get("pnl_dollars", 0.0) or 0.0
    pnl_sign = "+" if pnl >= 0 else ""
    reason = trade.get("resolution_reason", "Price reached target level")

    embed = discord.Embed(
        title=title,
        description=(
            f"**Strategy #{trade['strategy_id']}:** `{trade['strategy_name']}`\n"
            f"**Direction:** `{trade['direction']}`\n"
            f"**Entry Price:** `${float(trade['entry_price']):.2f}`\n"
            f"**Exit Price:** `${float(trade['close_price']):.2f}`\n"
            f"**Result:** `{'🏆 WON (+TP Hit)' if is_won else '❌ LOST (-SL Hit)'}`\n"
            f"**Points / PnL:** **`{pnl_sign}${pnl:.2f} Gold Points ($)`**\n"
            f"**Trigger:** `{reason}`"
        ),
        color=color,
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )
    embed.set_footer(text="CapeTrader Automated Hit/Miss Detector (15m Candle Loop)")
    return embed


def build_journal_stats_embed(stats: Dict[str, Any], open_trades: List[Dict[str, Any]]) -> discord.Embed:
    """Builds a comprehensive Discord Embed showing strategy win rates and journal statistics."""
    closed = stats["closed_trades"]
    won = stats["won_trades"]
    lost = stats["lost_trades"]
    win_rate = stats["win_rate"]
    total_pnl = stats["total_pnl"]
    pnl_sign = "+" if total_pnl >= 0 else ""

    embed = discord.Embed(
        title="📊 Institutional Strategy Performance Journal",
        description=(
            f"**Overall Win Rate:** **`{win_rate:.1f}%`** ({won} Wins / {lost} Losses)\n"
            f"**Total Tracked Trades:** `{stats['total_trades']}` (`{closed} Closed` • `{stats['open_trades']} Open`)\n"
            f"**Cumulative Closed PnL:** **`{pnl_sign}${total_pnl:.2f} Gold Points ($)`**\n\n"
            f"**{stats['top_strategy']}**"
        ),
        color=discord.Color.gold(),
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )

    b = stats["breakdown"]
    s1 = b[1]
    s2 = b[2]
    s3 = b[3]

    embed.add_field(
        name="⚡ 1. Asian Judas Swing",
        value=(
            f"• **Win Rate:** `{s1['win_rate']:.1f}%`\n"
            f"• **Record:** `{s1['won']}W / {s1['lost']}L` ({s1['closed']} closed)\n"
            f"• **Active Open:** `{s1['open']}`\n"
            f"• **Net PnL:** `{'+' if s1['pnl'] >= 0 else ''}${s1['pnl']:.2f} pts`"
        ),
        inline=True,
    )
    embed.add_field(
        name="🏰 2. Breakout & Retest",
        value=(
            f"• **Win Rate:** `{s2['win_rate']:.1f}%`\n"
            f"• **Record:** `{s2['won']}W / {s2['lost']}L` ({s2['closed']} closed)\n"
            f"• **Active Open:** `{s2['open']}`\n"
            f"• **Net PnL:** `{'+' if s2['pnl'] >= 0 else ''}${s2['pnl']:.2f} pts`"
        ),
        inline=True,
    )
    embed.add_field(
        name="🌊 3. Trend Scalper",
        value=(
            f"• **Win Rate:** `{s3['win_rate']:.1f}%`\n"
            f"• **Record:** `{s3['won']}W / {s3['lost']}L` ({s3['closed']} closed)\n"
            f"• **Active Open:** `{s3['open']}`\n"
            f"• **Net PnL:** `{'+' if s3['pnl'] >= 0 else ''}${s3['pnl']:.2f} pts`"
        ),
        inline=True,
    )

    if open_trades:
        open_desc = []
        for t in open_trades[:5]:
            open_desc.append(
                f"• **#{t['id']}**: {t['direction']} Strat {t['strategy_id']} @ `${t['entry_price']:.2f}` "
                f"(SL: `${t['stop_loss']:.2f}`, TP: `${t['take_profit']:.2f}`)"
            )
        if len(open_trades) > 5:
            open_desc.append(f"...and {len(open_trades) - 5} more")
        embed.add_field(
            name=f"🟢 Active Open Paper Trades ({len(open_trades)})",
            value="\n".join(open_desc),
            inline=False,
        )

    embed.set_footer(text="Antigravity Quantitative Trading Bot • Paper Trading & Auto-Journal")
    return embed


# --------------------------------------------------------------------------------------
# SYNCHRONOUS MARKET DATA PIPELINES (UNIFIED MULTI-TIMEFRAME ENGINE)
# --------------------------------------------------------------------------------------

def fetch_unified_market_intelligence() -> Dict[str, Any]:
    """
    Unified multi-timeframe intelligence engine:
      1. Ingests live tick & candlestick data for M5, M15, H1, D1 and Macro (DXY/US10Y).
      2. Computes VWAP, 20 EMA, 50 EMA, RSI (14), and ATR (14) across all timeframes.
      3. Identifies institutional liquidity levels (PDH, PDL, Asian High, Asian Low).
      4. Evaluates candlestick price action patterns on M5, M15, and H1.
      5. Runs the Automated 3-Strategy Institutional Scanner (Judas Swing, Breakout & Retest, EMA Scalper).
    Returns a unified, cross-functional payload shared by /predict, /levels, /strategies, /calc, and /track.
    """
    engine = MarketDataEngine(mode=DATA_SOURCE)
    df_m5, df_m15, df_h1, df_d1, source_name = engine.get_gold_candles()
    macro_data = engine.get_macro_correlation_data()

    # Calculate indicators on all timeframes
    df_m5 = engine.calculate_indicators(df_m5)
    df_m15 = engine.calculate_indicators(df_m15)
    df_h1 = engine.calculate_indicators(df_h1)
    levels = engine.calculate_institutional_levels(df_m15, df_d1)

    current_price = float(df_m5["close"].iloc[-1])
    current_high = float(df_m5["high"].iloc[-1])
    current_low = float(df_m5["low"].iloc[-1])

    # Candlestick Pattern Detection (M5, M15, H1)
    patterns_m5 = engine.detect_candlestick_patterns(
        df_m5, levels["pdh"], levels["pdl"], levels["asian_high"], levels["asian_low"], "M5"
    )
    patterns_m15 = engine.detect_candlestick_patterns(
        df_m15, levels["pdh"], levels["pdl"], levels["asian_high"], levels["asian_low"], "M15"
    )
    patterns_h1 = engine.detect_candlestick_patterns(
        df_h1, levels["pdh"], levels["pdl"], levels["asian_high"], levels["asian_low"], "H1"
    )

    m5_latest = {
        "close": round(float(df_m5["close"].iloc[-1]), 2),
        "open": round(float(df_m5["open"].iloc[-1]), 2),
        "high": round(float(df_m5["high"].iloc[-1]), 2),
        "low": round(float(df_m5["low"].iloc[-1]), 2),
        "rsi_14": round(float(df_m5["rsi_14"].iloc[-1]), 1),
        "ema_20": round(float(df_m5["ema_20"].iloc[-1]), 2),
        "ema_50": round(float(df_m5["ema_50"].iloc[-1]), 2),
        "vwap": round(float(df_m5["vwap"].iloc[-1]), 2),
        "atr_14": round(float(df_m5["atr_14"].iloc[-1]), 2),
    }

    m15_latest = {
        "close": round(float(df_m15["close"].iloc[-1]), 2),
        "open": round(float(df_m15["open"].iloc[-1]), 2),
        "high": round(float(df_m15["high"].iloc[-1]), 2),
        "low": round(float(df_m15["low"].iloc[-1]), 2),
        "rsi_14": round(float(df_m15["rsi_14"].iloc[-1]), 1),
        "ema_20": round(float(df_m15["ema_20"].iloc[-1]), 2),
        "ema_50": round(float(df_m15["ema_50"].iloc[-1]), 2),
        "vwap": round(float(df_m15["vwap"].iloc[-1]), 2),
        "atr_14": round(float(df_m15["atr_14"].iloc[-1]), 2),
    }

    h1_latest = {
        "close": round(float(df_h1["close"].iloc[-1]), 2),
        "open": round(float(df_h1["open"].iloc[-1]), 2),
        "high": round(float(df_h1["high"].iloc[-1]), 2),
        "low": round(float(df_h1["low"].iloc[-1]), 2),
        "rsi_14": round(float(df_h1["rsi_14"].iloc[-1]), 1),
        "ema_20": round(float(df_h1["ema_20"].iloc[-1]), 2),
        "ema_50": round(float(df_h1["ema_50"].iloc[-1]), 2),
        "vwap": round(float(df_h1["vwap"].iloc[-1]), 2),
        "atr_14": round(float(df_h1["atr_14"].iloc[-1]), 2),
    }

    # Evaluate all 3 Institutional Strategies simultaneously
    strategies = StrategyScanner.scan_all(
        df_m5=df_m5,
        df_m15=df_m15,
        df_h1=df_h1,
        levels=levels,
        m5_metrics=m5_latest,
        m15_metrics=m15_latest,
        h1_metrics=h1_latest,
        patterns_m5=patterns_m5,
        patterns_m15=patterns_m15,
        current_price=current_price,
    )

    active_count = sum(1 for s in strategies.values() if "ACTIVE" in s.get("status", ""))
    forming_count = sum(1 for s in strategies.values() if "FORMING" in s.get("status", ""))

    return {
        "levels": levels,
        "m5": m5_latest,
        "m15": m15_latest,
        "h1": h1_latest,
        "current_price": current_price,
        "current_high": current_high,
        "current_low": current_low,
        "source_name": source_name,
        "macro": macro_data,
        "df_m5": df_m5,
        "df_m15": df_m15,
        "df_h1": df_h1,
        "patterns_m5": patterns_m5,
        "patterns_m15": patterns_m15,
        "patterns_h1": patterns_h1,
        "strategies": strategies,
        "active_strategies_count": active_count,
        "forming_strategies_count": forming_count,
        "engine": engine,
    }


def run_levels_pipeline() -> Dict[str, Any]:
    """Compatibility wrapper returning the full unified market intelligence payload."""
    return fetch_unified_market_intelligence()


def run_analysis_pipeline() -> Dict[str, Any]:
    """
    Executes a complete 3-timeframe market intelligence cycle:
      1. Evaluates unified market data, technical indicators, institutional levels, and strategies.
      2. Scrapes economic calendar and news headlines.
      3. Generates prediction (AI LLM or deterministic quant fallback).
      4. Renders high-resolution live M5 candlestick chart.
      5. Constructs rich Discord Embed and returns complete analysis dict.
    """
    logger.info("=" * 70)
    logger.info("STARTING XAU/USD 3-TIMEFRAME QUANT INTELLIGENCE SCAN (M5 / M15 / H1)")
    logger.info("=" * 70)

    # 1. Unified Market Intelligence Context
    unified = fetch_unified_market_intelligence()
    current_price = unified["current_price"]
    current_high = unified["current_high"]
    current_low = unified["current_low"]
    source_name = unified["source_name"]
    levels = unified["levels"]
    m5_latest = unified["m5"]
    m15_latest = unified["m15"]
    h1_latest = unified["h1"]
    strategies = unified["strategies"]
    patterns_m5 = unified["patterns_m5"]
    patterns_m15 = unified["patterns_m15"]
    patterns_h1 = unified["patterns_h1"]
    macro_data = unified["macro"]
    df_m5 = unified["df_m5"]
    df_m15 = unified["df_m15"]
    df_h1 = unified["df_h1"]

    logger.info("Current Price: $%.2f | Data Source: %s", current_price, source_name)
    logger.info(
        "Institutional Levels -> PDH: $%.2f | PDL: $%.2f | Asian High: $%.2f | Asian Low: $%.2f",
        levels["pdh"], levels["pdl"], levels["asian_high"], levels["asian_low"]
    )
    logger.info("M5 Patterns Detected: %s", patterns_m5 if patterns_m5 else "None")
    logger.info("M15 Patterns Detected: %s", patterns_m15 if patterns_m15 else "None")
    logger.info("Strategy Scan: Judas Swing [%s] | Breakout [%s] | Trend Scalper [%s]",
                strategies["judas_swing"]["status"],
                strategies["breakout_retest"]["status"],
                strategies["trend_pullback"]["status"])

    # 2. News & Calendar Ingestion
    calendar_data = ForexIntelligenceScraper.fetch_calendar_events()
    news_items = ForexIntelligenceScraper.fetch_breaking_news(limit=10)

    # 3. Build Summary Payload
    payload = HybridPredictionEngine.build_summary_payload(
        gold_symbol=source_name,
        current_price=current_price,
        m5_metrics=m5_latest,
        m15_metrics=m15_latest,
        h1_metrics=h1_latest,
        levels=levels,
        m5_patterns=patterns_m5,
        m15_patterns=patterns_m15,
        h1_patterns=patterns_h1,
        macro=macro_data,
        calendar=calendar_data,
        news=news_items,
    )

    # 4. Generate Forecast (AI LLM with Rule-Based Fallback)
    forecast_text = None
    if GEMINI_API_KEY or OPENAI_API_KEY:
        forecast_text = HybridPredictionEngine.call_llm(payload)

    if not forecast_text:
        logger.info("Using institutional deterministic rule-based quantitative scoring engine.")
        forecast_text = HybridPredictionEngine.deterministic_rule_based_forecast(payload)

    parsed = parse_forecast_components(forecast_text)

    # 5. Render Live M5 Candlestick Chart PNG (150 candles lookback with live price axis)
    chart_bytes = b""
    try:
        chart_buf = generate_candlestick_chart(
            df_m5,
            "5m",
            levels["pdh"],
            levels["pdl"],
            levels["asian_high"],
            levels["asian_low"],
            candles_count=150,
            current_price=current_price,
        )
        if chart_buf:
            chart_bytes = chart_buf.getvalue()
    except Exception as chart_err:
        logger.error("Candlestick chart generation error: %s", chart_err, exc_info=True)
        chart_bytes = b""

    # Package Result
    result = {
        "forecast_text": forecast_text,
        "parsed": parsed,
        "payload": payload,
        "levels": levels,
        "current_price": current_price,
        "current_high": current_high,
        "current_low": current_low,
        "source_name": source_name,
        "probability": parsed["probability"],
        "bias": parsed["bias"],
        "m5": m5_latest,
        "m15": m15_latest,
        "h1": h1_latest,
        "strategies": strategies,
        "calendar": calendar_data,
        "news": news_items,
        "chart_bytes": chart_bytes,
        "df_m5": df_m5,
        "df_m15": df_m15,
        "df_h1": df_h1,
        "patterns_m5": patterns_m5,
        "patterns_m15": patterns_m15,
        "patterns_h1": patterns_h1,
    }

    if HAS_DISCORD:
        try:
            result["embed"] = build_forecast_embed(result)
        except Exception as embed_err:
            logger.error("Failed to build rich forecast embed: %s", embed_err, exc_info=True)

    # Output to Console
    print("\n" + "=" * 60)
    print(forecast_text)
    print("=" * 60 + "\n")

    return result


# ======================================================================================
# DISCORD BOT CLIENT & SLASH COMMANDS
# ======================================================================================

if HAS_DISCORD:
    intents = discord.Intents.default()
    bot = commands.Bot(command_prefix="!", intents=intents)

    def get_alert_mention() -> str:
        """Returns Discord mention string for configured role and/or user."""
        mentions = []
        if DISCORD_ALERT_ROLE_ID:
            mentions.append(f"<@&{DISCORD_ALERT_ROLE_ID}>")
        if DISCORD_ALERT_USER_ID:
            mentions.append(f"<@{DISCORD_ALERT_USER_ID}>")
        return " ".join(mentions)

    @bot.event
    async def on_ready():
        logger.info("==================================================")
        logger.info("Discord Bot connected as: %s (ID: %s)", bot.user, bot.user.id)
        logger.info("==================================================")

        try:
            # Sync to all active guilds immediately for instantaneous slash command propagation
            for g in bot.guilds:
                try:
                    bot.tree.copy_global_to(guild=g)
                    await bot.tree.sync(guild=g)
                    logger.info("Instantly synced %d slash command(s) to guild '%s' (%s).", len(bot.tree.get_commands()), g.name, g.id)
                except Exception as g_err:
                    logger.warning("Could not sync commands to guild %s: %s", g.id, g_err)

            synced = await bot.tree.sync()
            logger.info("Successfully synced %d Slash Command(s) globally.", len(synced))
        except Exception as exc:
            logger.error("Failed to sync application slash commands: %s", exc)

        # Dispatch startup forecast asynchronously in the background so on_ready completes instantly
        async def _dispatch_startup_broadcast():
            if not DISCORD_ALERT_CHANNEL_ID:
                return
            try:
                ch_id = int(DISCORD_ALERT_CHANNEL_ID)
                channel = bot.get_channel(ch_id) or await bot.fetch_channel(ch_id)
                if channel:
                    logger.info("[DISCORD BOT] Sending immediate startup forecast to channel %s...", ch_id)
                    startup_data = await asyncio.to_thread(run_analysis_pipeline)
                    startup_embed = startup_data.get("embed") or build_forecast_embed(startup_data)
                    pings = get_alert_mention()
                    welcome_msg = (
                        f"{pings + ' ' if pings else ''}🟢 **XAU/USD Quantitative Intelligence Bot is now Online!**\n"
                        f"3-Timeframe Architecture Active: **H1** (Trend), **M15** (Setup), **M5** (Sniper Trigger).\n"
                        f"Slash Commands: `/predict` | `/levels` | `/news` | `/chart` | `/calc` | `/strategies` | `/track`"
                    )
                    chart_bytes = startup_data.get("chart_bytes", b"")
                    chart_file = None
                    if chart_bytes and len(chart_bytes) > 0:
                        try:
                            chart_file = discord.File(io.BytesIO(chart_bytes), filename="gold_chart.png")
                            startup_embed.set_image(url="attachment://gold_chart.png")
                        except Exception as file_err:
                            logger.warning("[DISCORD BOT] Could not attach startup chart file: %s", file_err)

                    if chart_file:
                        await channel.send(content=welcome_msg, embed=startup_embed, file=chart_file)
                    else:
                        await channel.send(content=welcome_msg, embed=startup_embed)
                    logger.info("[DISCORD BOT] Successfully dispatched startup broadcast to channel %s.", ch_id)
            except Exception as exc:
                logger.warning("[DISCORD BOT] Could not send startup broadcast to channel %s: %s", DISCORD_ALERT_CHANNEL_ID, exc)

        asyncio.create_task(_dispatch_startup_broadcast())

        # Start background 15-minute scheduled forecast loop
        if not scheduled_forecast_loop.is_running():
            scheduled_forecast_loop.start()

    # ----------------------------------------------------------------------------------
    # GLOBAL TREE ERROR HANDLER (Prevents unhandled interaction drops/timeouts)
    # ----------------------------------------------------------------------------------
    @bot.tree.error
    async def on_tree_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
        cmd_name = interaction.command.name if interaction.command else "unknown"
        logger.error("Slash command error on /%s: %s", cmd_name, error, exc_info=True)
        try:
            if not interaction.response.is_done():
                await interaction.response.defer(ephemeral=True)
            err_embed = build_error_embed(
                title=f"/{cmd_name} Command Advisory",
                description=f"Command encountered an execution issue:\n`{error}`",
                hint="Please try again or run `/levels` to verify live feed status."
            )
            await interaction.followup.send(embed=err_embed, ephemeral=True)
        except Exception as handler_err:
            logger.error("Failed to deliver app command error followup: %s", handler_err)

    # ----------------------------------------------------------------------------------
    # SLASH COMMAND: /predict
    # ----------------------------------------------------------------------------------
    @bot.tree.command(
        name="predict",
        description="Run live XAU/USD 3-timeframe forecast (H1 Trend, M15 Setup, M5 Sniper Trigger) with live chart"
    )
    async def predict_command(interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            analysis = await asyncio.to_thread(run_analysis_pipeline)
            embed = analysis.get("embed") or build_forecast_embed(analysis)

            content = None
            prob = analysis.get("probability", 5.0)
            bias = analysis.get("bias", "")
            is_red = analysis["payload"]["economic_calendar"].get("red_folder_alert", False)
            has_active_strat = any("ACTIVE" in s.get("status", "") for s in analysis.get("strategies", {}).values())
            pings = get_alert_mention()

            if pings and (prob >= 8.0 or is_red or has_active_strat or "NO-TRADE" in bias.upper()):
                content = f"{pings} 🚨 **High-Conviction / Institutional Strategy Alert!**"

            # Attach rendered M5 candlestick chart to embed if available
            chart_bytes = analysis.get("chart_bytes", b"")
            chart_file = None
            if chart_bytes and len(chart_bytes) > 0:
                try:
                    chart_file = discord.File(io.BytesIO(chart_bytes), filename="gold_chart.png")
                    embed.set_image(url="attachment://gold_chart.png")
                except Exception as file_err:
                    logger.warning("Could not attach chart file to /predict embed: %s", file_err)

            if chart_file:
                await interaction.followup.send(content=content, embed=embed, file=chart_file)
            else:
                await interaction.followup.send(content=content, embed=embed)
        except Exception as exc:
            logger.error("Error executing /predict slash command: %s", exc, exc_info=True)
            err_embed = build_error_embed(
                title="Forecast Generation Advisory",
                description=f"Failed to generate intraday forecast:\n`{exc}`",
                hint="Verify broker tick connection or market status. Watchdog remains active."
            )
            await interaction.followup.send(embed=err_embed, ephemeral=True)

    # ----------------------------------------------------------------------------------
    # SLASH COMMAND: /levels
    # ----------------------------------------------------------------------------------
    @bot.tree.command(
        name="levels",
        description="Display live XAU/USD price, institutional levels, strategy preview snippet & indicators"
    )
    async def levels_command(interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            levels_data = await asyncio.to_thread(fetch_unified_market_intelligence)
            embed = build_levels_embed(levels_data)
            await interaction.followup.send(embed=embed)
        except Exception as exc:
            logger.error("Error executing /levels slash command: %s", exc, exc_info=True)
            err_embed = build_error_embed(
                title="Institutional Levels Advisory",
                description=f"Failed to retrieve institutional levels:\n`{exc}`",
                hint="Market feed or broker is reconnecting. Will retry automatically."
            )
            await interaction.followup.send(embed=err_embed, ephemeral=True)

    # ----------------------------------------------------------------------------------
    # SLASH COMMAND: /news
    # ----------------------------------------------------------------------------------
    @bot.tree.command(
        name="news",
        description="Display today's USD High/Medium impact economic events and breaking headlines"
    )
    async def news_command(interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            calendar_data = await asyncio.to_thread(ForexIntelligenceScraper.fetch_calendar_events)
            news_items = await asyncio.to_thread(ForexIntelligenceScraper.fetch_breaking_news, 10)
            embed = build_news_embed(calendar_data, news_items)
            await interaction.followup.send(embed=embed)
        except Exception as exc:
            logger.error("Error executing /news slash command: %s", exc, exc_info=True)
            err_embed = build_error_embed(
                title="News & Calendar Advisory",
                description=f"Failed to retrieve economic calendar:\n`{exc}`",
                hint="Forex Factory / RSS feeds may be experiencing temporary rate limits."
            )
            await interaction.followup.send(embed=err_embed, ephemeral=True)

    # ----------------------------------------------------------------------------------
    # SLASH COMMAND: /chart
    # ----------------------------------------------------------------------------------
    @bot.tree.command(
        name="chart",
        description="Generate and post a live technical candlestick chart with institutional levels and EMAs"
    )
    @app_commands.describe(timeframe="Candlestick timeframe (default: 5m)")
    @app_commands.choices(
        timeframe=[
            app_commands.Choice(name="5-Minute (M5 Sniper Scalp)", value="5m"),
            app_commands.Choice(name="15-Minute (M15 Setup & Structure)", value="15m"),
            app_commands.Choice(name="1-Hour (H1 Macro Trend)", value="1h"),
        ]
    )
    async def chart_command(
        interaction: discord.Interaction,
        timeframe: Optional[app_commands.Choice[str]] = None,
    ):
        await interaction.response.defer()
        try:
            tf_choice = timeframe.value if timeframe else "5m"
            levels_data = await asyncio.to_thread(fetch_unified_market_intelligence)
            levels = levels_data["levels"]
            current_price = levels_data["current_price"]

            if tf_choice == "1h":
                chart_df = levels_data["df_h1"]
                tf_title = "1-Hour (H1)"
            elif tf_choice == "15m":
                chart_df = levels_data["df_m15"]
                tf_title = "15-Minute (M15)"
            else:
                chart_df = levels_data["df_m5"]
                tf_title = "5-Minute (M5)"

            embed = discord.Embed(
                title=f"📈 XAU/USD Technical Candlestick Chart ({tf_title})",
                description=(
                    f"**Current Price:** `${current_price:.2f}`\n"
                    f"**Ceiling / PDH:** `${levels['pdh']:.2f}` | **Floor / PDL:** `${levels['pdl']:.2f}`\n"
                    f"**Asian Session Range:** `${levels['asian_low']:.2f}` – `${levels['asian_high']:.2f}`"
                ),
                color=discord.Color.gold(),
                timestamp=datetime.datetime.now(datetime.timezone.utc),
            )

            chart_file = None
            try:
                chart_buf = await asyncio.to_thread(
                    generate_candlestick_chart,
                    chart_df,
                    tf_title,
                    levels["pdh"],
                    levels["pdl"],
                    levels["asian_high"],
                    levels["asian_low"],
                    150,
                    current_price,
                )
                if chart_buf and chart_buf.getbuffer().nbytes > 0:
                    chart_file = discord.File(chart_buf, filename="gold_chart.png")
                    embed.set_image(url="attachment://gold_chart.png")
            except Exception as chart_render_err:
                logger.error("Chart rendering error in /chart command: %s", chart_render_err, exc_info=True)
                embed.add_field(
                    name="⚠️ Chart Visualization Advisory",
                    value=f"Candlestick image rendering encountered an issue: `{chart_render_err}`\nTechnical levels and live price above remain completely valid.",
                    inline=False,
                )

            if chart_file:
                await interaction.followup.send(embed=embed, file=chart_file)
            else:
                await interaction.followup.send(embed=embed)
        except Exception as exc:
            logger.error("Error executing /chart slash command: %s", exc, exc_info=True)
            err_embed = build_error_embed(
                title="Chart Rendering Advisory",
                description=f"Could not render technical chart:\n`{exc}`",
                hint="Verify candlestick feed availability."
            )
            await interaction.followup.send(embed=err_embed, ephemeral=True)

    # ----------------------------------------------------------------------------------
    # SLASH COMMAND: /calc (MT5 Spot Gold Lot Size & Risk Calculator)
    # ----------------------------------------------------------------------------------
    @bot.tree.command(
        name="calc",
        description="MT5 Spot Gold Lot Size & Risk Calculator (Standard, Mini, and Micro Lots)"
    )
    @app_commands.describe(
        balance="Account balance in USD (e.g. 500, 1000, 5000)",
        risk_percent="Percentage of account to risk (e.g. 1.0 for 1%, 2.0 for 2%)",
        stop_loss_dollars="Stop loss distance in Gold dollars (e.g. 4.50). Leave empty to use live ATR"
    )
    async def calc_command(
        interaction: discord.Interaction,
        balance: float,
        risk_percent: float = 1.0,
        stop_loss_dollars: Optional[float] = None,
    ):
        await interaction.response.defer()
        try:
            if balance <= 0 or risk_percent <= 0:
                await interaction.followup.send("❌ **Error:** Balance and risk percentage must be greater than zero.", ephemeral=True)
                return

            if stop_loss_dollars is not None and stop_loss_dollars <= 0:
                await interaction.followup.send("❌ **Error:** Stop-loss distance must be greater than $0.00.", ephemeral=True)
                return

            atr_m5 = None
            atr_m15 = None
            current_price = None

            if stop_loss_dollars is None:
                levels_data = await asyncio.to_thread(fetch_unified_market_intelligence)
                atr_m5 = levels_data["m5"]["atr_14"]
                atr_m15 = levels_data["m15"]["atr_14"]
                current_price = levels_data["current_price"]

            embed = build_calc_embed(
                balance=balance,
                risk_percent=risk_percent,
                stop_loss_dollars=stop_loss_dollars,
                atr_m5=atr_m5,
                atr_m15=atr_m15,
                current_price=current_price,
            )
            await interaction.followup.send(embed=embed)
        except Exception as exc:
            logger.error("Error executing /calc slash command: %s", exc, exc_info=True)
            err_embed = build_error_embed(
                title="Calculator Advisory",
                description=f"Could not calculate lot size from live ATR:\n`{exc}`",
                hint="You can supply a custom stop_loss_dollars parameter or wait for live tick updates."
            )
            await interaction.followup.send(embed=err_embed, ephemeral=True)

    # ----------------------------------------------------------------------------------
    # SLASH COMMAND: /strategies (3-Strategy Institutional Scanner)
    # ----------------------------------------------------------------------------------
    @bot.tree.command(
        name="strategies",
        description="Scan live XAU/USD setups with live price, VWAP & RSI: Judas Swing, Breakout, EMA Scalper"
    )
    async def strategies_command(interaction: discord.Interaction):
        await interaction.response.defer()
        try:
            intel = await asyncio.to_thread(fetch_unified_market_intelligence)
            embed = build_strategies_embed(
                strategies=intel["strategies"],
                current_price=intel["current_price"],
                levels=intel["levels"],
                m5_metrics=intel["m5"],
                m15_metrics=intel["m15"],
                source_name=intel["source_name"],
            )
            await interaction.followup.send(embed=embed)
        except Exception as exc:
            logger.error("Error executing /strategies slash command: %s", exc, exc_info=True)
            err_embed = build_error_embed(
                title="Strategy Scanner Advisory",
                description=f"Encountered an issue running the strategy detector:\n`{exc}`",
                hint="Verify broker tick connection or yfinance availability."
            )
            await interaction.followup.send(embed=err_embed, ephemeral=True)

    # ----------------------------------------------------------------------------------
    # SLASH COMMAND: /track (Institutional Paper Trading & Auto-Journaling)
    # ----------------------------------------------------------------------------------
    @bot.tree.command(
        name="track",
        description="Paper trade & auto-journal the 3 institutional strategies (track win/loss rates & stats)"
    )
    @app_commands.describe(
        action="Action: 'open' new paper trade, 'close' existing trade, or 'stats' journal overview",
        strategy="Institutional Strategy: 1 (Judas Sweep), 2 (Breakout & Retest), 3 (Trend Scalper)",
        direction="Trade direction: BUY (Long) or SELL (Short)",
        stop_loss="Custom Stop-Loss Gold price (leave blank for auto 1.5x ATR stop)",
        take_profit="Custom Take-Profit Gold price (leave blank for auto 1:2 RRR target)",
        trade_id="Trade ID to close (used when action is 'close')",
        result="Close result: WIN or LOSS (used when action is 'close')",
    )
    @app_commands.choices(
        action=[
            app_commands.Choice(name="Open New Paper Trade", value="open"),
            app_commands.Choice(name="Close Existing Trade", value="close"),
            app_commands.Choice(name="View Strategy Win Rate & Journal Stats", value="stats"),
        ],
        strategy=[
            app_commands.Choice(name="1. Asian Session Sweep (Judas Swing)", value="1"),
            app_commands.Choice(name="2. London/NY Breakout & Retest", value="2"),
            app_commands.Choice(name="3. EMA 20/50 + VWAP Trend Pullback", value="3"),
        ],
        direction=[
            app_commands.Choice(name="BUY (Long)", value="BUY"),
            app_commands.Choice(name="SELL (Short)", value="SELL"),
        ],
        result=[
            app_commands.Choice(name="WIN (Take-Profit Reached)", value="WIN"),
            app_commands.Choice(name="LOSS (Stop-Loss Reached)", value="LOSS"),
        ],
    )
    async def track_command(
        interaction: discord.Interaction,
        action: app_commands.Choice[str],
        strategy: Optional[app_commands.Choice[str]] = None,
        direction: Optional[app_commands.Choice[str]] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trade_id: Optional[int] = None,
        result: Optional[app_commands.Choice[str]] = None,
    ):
        await interaction.response.defer()
        try:
            action_val = action.value.lower()

            # ACTION: stats
            if action_val == "stats":
                stats = await asyncio.to_thread(TradeJournal.get_stats)
                open_trades = await asyncio.to_thread(TradeJournal.get_open_trades)
                embed = build_journal_stats_embed(stats, open_trades)
                await interaction.followup.send(embed=embed)
                return

            # ACTION: close
            if action_val == "close":
                if trade_id is None:
                    open_trades = await asyncio.to_thread(TradeJournal.get_open_trades)
                    if not open_trades:
                        await interaction.followup.send(
                            "ℹ️ **No active paper trades found.** Use `/track action:Open` to log a new paper trade.",
                            ephemeral=True,
                        )
                        return
                    lines = ["**Active Open Trades:**"]
                    for t in open_trades[:10]:
                        lines.append(
                            f"• **Trade #{t['id']}:** {t['direction']} Strat {t['strategy_id']} @ `${t['entry_price']:.2f}` "
                            f"(SL: `${t['stop_loss']:.2f}`, TP: `${t['take_profit']:.2f}`)"
                        )
                    lines.append("\n👉 Run `/track action:Close trade_id:<ID> result:<WIN/LOSS>` to close one.")
                    await interaction.followup.send("\n".join(lines), ephemeral=True)
                    return

                trade = await asyncio.to_thread(TradeJournal.get_trade, trade_id)
                if not trade:
                    await interaction.followup.send(f"❌ **Trade #{trade_id} not found in journal.**", ephemeral=True)
                    return
                if trade["status"] != "OPEN":
                    await interaction.followup.send(
                        f"ℹ️ **Trade #{trade_id} is already closed as {trade['status']}.**", ephemeral=True
                    )
                    return

                levels_data = await asyncio.to_thread(run_levels_pipeline)
                current_price = levels_data["current_price"]

                res_str = result.value if result else None
                if not res_str:
                    if trade["direction"] == "BUY":
                        res_str = "WIN" if current_price >= trade["entry_price"] else "LOSS"
                    else:
                        res_str = "WIN" if current_price <= trade["entry_price"] else "LOSS"

                closed_trade = await asyncio.to_thread(
                    TradeJournal.close_trade,
                    trade_id=trade_id,
                    result=res_str,
                    close_price=current_price,
                    notes=f"Manually closed via Discord command at ${current_price:.2f}",
                )
                embed = build_trade_close_embed(closed_trade)
                await interaction.followup.send(embed=embed)
                return

            # ACTION: open
            if action_val == "open":
                strat_id = int(strategy.value) if strategy else 1
                intel = await asyncio.to_thread(fetch_unified_market_intelligence)
                current_price = intel["current_price"]
                atr_m5 = intel["m5"]["atr_14"]
                levels = intel["levels"]
                strat_keys = {1: "judas_swing", 2: "breakout_retest", 3: "trend_pullback"}
                strat_info = intel["strategies"].get(strat_keys.get(strat_id, "judas_swing"), {})

                # Determine direction
                dir_val = direction.value if direction else None
                if not dir_val:
                    if "BULLISH" in strat_info.get("direction", ""):
                        dir_val = "BUY"
                    elif "BEARISH" in strat_info.get("direction", ""):
                        dir_val = "SELL"
                    else:
                        dir_val = "BUY"

                # Determine Stop-Loss
                sl_val = stop_loss
                if sl_val is None:
                    if strat_info.get("sl"):
                        sl_val = float(strat_info["sl"])
                    elif dir_val == "BUY":
                        sl_val = round(current_price - max(1.5, 1.5 * atr_m5), 2)
                    else:
                        sl_val = round(current_price + max(1.5, 1.5 * atr_m5), 2)

                # Determine Take-Profit
                tp_val = take_profit
                if tp_val is None:
                    if strat_info.get("tp1"):
                        tp_val = float(strat_info["tp1"])
                    else:
                        risk_dist = abs(current_price - sl_val)
                        if dir_val == "BUY":
                            tp_val = round(current_price + 2.0 * risk_dist, 2)
                        else:
                            tp_val = round(current_price - 2.0 * risk_dist, 2)

                # Parameter validation
                if dir_val == "BUY" and sl_val >= current_price:
                    await interaction.followup.send(
                        f"❌ **Invalid Stop-Loss:** For BUY trades, Stop-Loss (${sl_val:.2f}) must be BELOW entry price (${current_price:.2f}).",
                        ephemeral=True,
                    )
                    return
                if dir_val == "SELL" and sl_val <= current_price:
                    await interaction.followup.send(
                        f"❌ **Invalid Stop-Loss:** For SELL trades, Stop-Loss (${sl_val:.2f}) must be ABOVE entry price (${current_price:.2f}).",
                        ephemeral=True,
                    )
                    return

                new_trade = await asyncio.to_thread(
                    TradeJournal.open_trade,
                    strategy_id=strat_id,
                    direction=dir_val,
                    entry_price=current_price,
                    stop_loss=sl_val,
                    take_profit=tp_val,
                    strategy_name=TradeJournal.STRATEGY_NAMES.get(strat_id, f"Strategy {strat_id}"),
                    notes="Opened via /track slash command",
                )
                embed = build_trade_open_embed(new_trade)
                await interaction.followup.send(embed=embed)
                return

        except Exception as exc:
            logger.error("Error executing /track slash command: %s", exc, exc_info=True)
            err_embed = build_error_embed(
                title="Paper Trade Journal Advisory",
                description=f"Encountered an issue executing /track:\n`{exc}`",
                hint="Verify parameters and database connectivity."
            )
            await interaction.followup.send(embed=err_embed, ephemeral=True)

    # ----------------------------------------------------------------------------------
    # BACKGROUND TASK: 15-MINUTE CANDLE CLOSE SYNCHRONIZED LOOP
    # ----------------------------------------------------------------------------------
    @tasks.loop(minutes=15)
    async def scheduled_forecast_loop():
        logger.info("[DISCORD BOT] Executing scheduled 15-minute intraday forecast scan...")
        try:
            analysis = await asyncio.to_thread(run_analysis_pipeline)
            current_price = analysis["current_price"]
            embed = analysis.get("embed") or build_forecast_embed(analysis)
            prob = analysis.get("probability", 5.0)
            bias = analysis.get("bias", "")
            is_red = analysis["payload"]["economic_calendar"].get("red_folder_alert", False)
            has_active_strat = any("ACTIVE" in s.get("status", "") for s in analysis.get("strategies", {}).values())

            # 0. Check paper trading journal open positions against live candle range
            try:
                high_p = analysis.get("current_high", current_price)
                low_p = analysis.get("current_low", current_price)
                resolved_trades = await asyncio.to_thread(
                    TradeJournal.check_open_trades_against_price,
                    current_price=current_price,
                    high_price=high_p,
                    low_price=low_p,
                )
                if resolved_trades and DISCORD_ALERT_CHANNEL_ID:
                    ch_id = int(DISCORD_ALERT_CHANNEL_ID)
                    channel = bot.get_channel(ch_id) or await bot.fetch_channel(ch_id)
                    if channel:
                        for r_trade in resolved_trades:
                            res_embed = build_trade_resolved_embed(r_trade)
                            mention = get_alert_mention()
                            content = f"{mention} 🎯 **Paper Trade Hit Detected!**" if mention else None
                            await channel.send(content=content, embed=res_embed)
                            logger.info("[TRADE JOURNAL] Auto-resolved trade #%d as %s", r_trade["id"], r_trade["status"])
            except Exception as j_exc:
                logger.warning("[TRADE JOURNAL] Resolution check encountered an issue: %s", j_exc)

            # Attach live M5 chart to scheduled embed if available
            chart_file = None
            chart_bytes = analysis.get("chart_bytes", b"")
            if chart_bytes and len(chart_bytes) > 0:
                try:
                    chart_file = discord.File(io.BytesIO(chart_bytes), filename="gold_chart.png")
                    embed.set_image(url="attachment://gold_chart.png")
                except Exception as f_err:
                    logger.warning("[DISCORD BOT] Could not attach chart file to scheduled embed: %s", f_err)

            # 1. Post to Discord Alert Channel
            if DISCORD_ALERT_CHANNEL_ID:
                try:
                    ch_id = int(DISCORD_ALERT_CHANNEL_ID)
                    channel = bot.get_channel(ch_id) or await bot.fetch_channel(ch_id)
                    if channel:
                        content = None
                        pings = get_alert_mention()
                        if pings and (prob >= 8.0 or is_red or has_active_strat or "NO-TRADE" in bias.upper()):
                            content = f"{pings} 🚨 **High-Conviction / Institutional Strategy Alert!**"
                        if chart_file:
                            await channel.send(content=content, embed=embed, file=chart_file)
                        else:
                            await channel.send(content=content, embed=embed)
                        logger.info("[DISCORD BOT] Successfully pushed forecast Embed to Channel %s", ch_id)
                    else:
                        logger.warning("[DISCORD BOT] Target channel %s not found.", ch_id)
                except Exception as exc:
                    logger.warning("[DISCORD BOT] Failed to dispatch to alert channel: %s", exc)

            # 2. Forward to Webhook fallback if configured
            if DISCORD_WEBHOOK_URL:
                await asyncio.to_thread(DispatchEngine.send_to_discord, analysis["forecast_text"])

            # 3. Forward to Telegram if configured
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                await asyncio.to_thread(DispatchEngine.send_to_telegram, analysis["forecast_text"])

        except Exception as exc:
            logger.error("[DISCORD BOT] Error in scheduled forecast loop: %s", exc, exc_info=True)
            if DISCORD_ALERT_CHANNEL_ID:
                try:
                    ch_id = int(DISCORD_ALERT_CHANNEL_ID)
                    channel = bot.get_channel(ch_id) or await bot.fetch_channel(ch_id)
                    if channel:
                        err_embed = build_error_embed(
                            title="Scheduled Watchdog Advisory",
                            description=f"Temporary issue updating 15m candle close forecast:\n`{exc}`",
                            hint="Market data feed or broker may be experiencing latency or weekend market closure. The bot watchdog remains active and will automatically retry at the next 15-minute candle boundary.",
                        )
                        await channel.send(embed=err_embed)
                except Exception as send_err:
                    logger.warning("[DISCORD BOT] Failed to dispatch error embed to alert channel: %s", send_err)

    @scheduled_forecast_loop.before_loop
    async def before_scheduled_loop():
        """Precision boundary sleep to align with the next :00, :15, :30, :45 close + 5s buffer."""
        try:
            await bot.wait_until_ready()
            now = datetime.datetime.now()
            minutes_past = now.minute % 15
            seconds_past = minutes_past * 60 + now.second + (now.microsecond / 1_000_000.0)
            seconds_to_wait = (15 * 60) - seconds_past + 5.0
            next_run = now + datetime.timedelta(seconds=seconds_to_wait)

            logger.info(
                "[DISCORD BOT] Synchronizing 15m task loop with next candle close at %s (waiting %.1fs)...",
                next_run.strftime("%H:%M:%S"), seconds_to_wait
            )
            await asyncio.sleep(seconds_to_wait)
        except Exception as align_err:
            logger.warning("[DISCORD BOT] Task loop initial boundary alignment issue: %s", align_err)


# ======================================================================================
# STANDALONE SCHEDULER (WEBHOOK & CLI FALLBACK)
# ======================================================================================

def sleep_until_next_15m_boundary():
    """Calculates exact seconds until the next 15-minute candle close plus a 5-second buffer."""
    now = datetime.datetime.now()
    minutes_past = now.minute % 15
    seconds_past = minutes_past * 60 + now.second + (now.microsecond / 1_000_000.0)
    seconds_to_wait = (15 * 60) - seconds_past + 5.0

    next_run = now + datetime.timedelta(seconds=seconds_to_wait)
    logger.info(
        "Scheduled next scan at %s (waiting %.1f seconds)...",
        next_run.strftime("%H:%M:%S"), seconds_to_wait
    )
    time.sleep(seconds_to_wait)


def run_standalone_loop():
    """Executes the continuous 15-minute scheduling loop when no Discord bot token is provided."""
    logger.info("[STANDALONE] Entering continuous 15-minute candle synchronization scheduler.")
    while True:
        try:
            sleep_until_next_15m_boundary()
            analysis = run_analysis_pipeline()
            DispatchEngine.broadcast(analysis["forecast_text"])
        except KeyboardInterrupt:
            logger.info("Shutdown signal received. Exiting gracefully.")
            break
        except Exception as exc:
            logger.error("Cycle error: %s. Rescheduling next run...", exc, exc_info=True)
            time.sleep(15)


# ======================================================================================
# KEEP-ALIVE WEB SERVER (24/7 CLOUD HOSTING ON RENDER / REPLIT / RAILWAY)
# ======================================================================================

if HAS_FLASK:
    keep_alive_app = Flask("XAUUSD_KeepAlive")
    # Suppress verbose werkzeug access logging to keep terminal/Discord logs clean
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    @keep_alive_app.route("/")
    def keep_alive_index():
        """Root route pinged by UptimeRobot, Render, or Replit to prevent sleep."""
        return "CapeTrader XAU/USD Intelligence Bot is Alive & Running 24/7 🚀", 200

    @keep_alive_app.route("/health")
    def keep_alive_health():
        """JSON healthcheck route."""
        return {
            "status": "online",
            "service": "XAU/USD Trading Intelligence Bot",
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }, 200

    def run_keep_alive_server():
        port = int(os.getenv("PORT", "8080"))
        logger.info("[KEEP-ALIVE] Starting lightweight Flask server on 0.0.0.0:%d", port)
        try:
            keep_alive_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
        except Exception as exc:
            logger.warning("[KEEP-ALIVE] Flask server encountered an error: %s", exc)

    def start_keep_alive():
        """Launches the Flask keep-alive web server in a daemon background thread."""
        server_thread = threading.Thread(
            target=run_keep_alive_server, daemon=True, name="KeepAliveServer"
        )
        server_thread.start()
        logger.info("[KEEP-ALIVE] Background keep-alive thread initialized successfully.")
else:
    def start_keep_alive():
        logger.warning("[KEEP-ALIVE] Flask is not installed. Skipping keep-alive web server.")


# ======================================================================================
# MAIN ENTRYPOINT
# ======================================================================================

def main():
    parser = argparse.ArgumentParser(description="XAU/USD & USD Intraday Trading Intelligence Bot")
    parser.add_argument("--once", action="store_true", help="Run a single market analysis scan and exit.")
    args = parser.parse_args()

    print(r"""
  __   __    _   _   _   _ ____  ____  
  \ \ / /   / \ | | | | | / ___||  _ \ 
   \ V /   / _ \| | | | | \___ \| | | |
    | |   / ___ \ |_| |_| |___) | |_| |
    |_|  /_/   \_\___/\___/|____/|____/ 
  Intraday Quantitative Gold Intelligence Engine (M5/M15/H1 + Visual Charting)
    """)

    # 1. Flag --once: Single CLI run & exit
    if args.once:
        logger.info("Flag --once was specified. Running single diagnostic market scan...")
        analysis = run_analysis_pipeline()
        DispatchEngine.broadcast(analysis["forecast_text"])
        stats = TradeJournal.get_stats()
        logger.info("[TRADE JOURNAL] Current Journal Stats: %d total trades (%d won, %d lost, %d open) | Win Rate: %.1f%%",
                    stats["total_trades"], stats["won_trades"], stats["lost_trades"], stats["open_trades"], stats["win_rate"])
        logger.info("Diagnostic scan complete. Exiting cleanly.")
        sys.exit(0)

    # Immediately initialize the Keep-Alive Web Server so Render/cloud platforms bind to PORT without delay!
    start_keep_alive()

    # 2. Interactive Discord Bot Mode (if DISCORD_BOT_TOKEN is set)
    if DISCORD_BOT_TOKEN and HAS_DISCORD:
        logger.info("Starting interactive Discord Bot with Slash Commands (/predict, /levels, /news, /chart, /calc, /strategies, /track)...")
        try:
            bot.run(DISCORD_BOT_TOKEN)
        except Exception as exc:
            logger.error("Failed to run Discord Bot: %s", exc, exc_info=True)
            sys.exit(1)
    else:
        # 3. Webhook / Standalone Scheduler Mode
        logger.info("=" * 70)
        logger.info("[MODE] No DISCORD_BOT_TOKEN detected in .env.")
        logger.info("Running in Standalone Webhook / Telegram / Terminal loop mode.")
        logger.info("To enable interactive slash commands, add DISCORD_BOT_TOKEN to your .env file.")
        logger.info("=" * 70)

        # Run startup scan
        logger.info("Executing immediate startup market scan...")
        startup_analysis = run_analysis_pipeline()
        DispatchEngine.broadcast(startup_analysis["forecast_text"])

        # Enter standalone 15m loop
        run_standalone_loop()


if __name__ == "__main__":
    main()
