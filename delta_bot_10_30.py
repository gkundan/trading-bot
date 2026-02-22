"""
===========================================
  10-30 STRATEGY BOT — Delta Exchange India
  Deployment: Railway.app
===========================================
"""

import time
import logging
import json
import os
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import requests

# ─────────────────────────────────────────
#  CONFIGURATION (reads from environment variables)
# ─────────────────────────────────────────
API_KEY    = os.environ.get("DELTA_API_KEY", "")
API_SECRET = os.environ.get("DELTA_API_SECRET", "")
BASE_URL   = "https://api.india.delta.exchange"

SYMBOL       = "BTCUSD"
CAPITAL      = float(os.environ.get("CAPITAL", "50.0"))
RISK_PERCENT = 0.01
JOURNAL_FILE = "trade_journal.json"

ST_PERIOD  = 10
ST_MULT    = 3.0
EMA_PERIOD = 50
TF_TREND   = 30   # 30-min chart for trend
TF_ENTRY   = 5    # 5-min chart for entry (closest to 10m available on Delta)

# ─────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler()   # Railway shows logs in dashboard
    ]
)
log = logging.getLogger("10-30-Bot")


# ─────────────────────────────────────────
#  DELTA EXCHANGE REST CLIENT
# ─────────────────────────────────────────
class DeltaClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def get_candles(self, symbol: str, resolution: int, limit: int = 200) -> pd.DataFrame:
        end_time   = int(time.time())
        start_time = end_time - (resolution * 60 * limit)
        resolution_map = {1: "1m", 5: "5m", 10: "5m", 15: "15m", 30: "30m", 60: "1h"}
        res_str = resolution_map.get(resolution, "30m")
        url = f"{BASE_URL}/v2/history/candles"
        params = {
            "resolution": res_str,
            "symbol":     symbol,
            "start":      start_time,
            "end":        end_time,
        }
        try:
            r = self.session.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            if data.get("success") and data.get("result"):
                raw = data["result"]
                if not raw:
                    return pd.DataFrame()
                # Delta Exchange returns list of dicts with keys: time, open, high, low, close, volume
                if isinstance(raw[0], dict):
                    df = pd.DataFrame(raw)
                    # Rename columns if they use different keys
                    col_map = {"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
                    df = df.rename(columns=col_map)
                    # Make sure we have the right columns
                    if "time" not in df.columns and df.columns[0] != "time":
                        df.columns = ["time", "open", "high", "low", "close", "volume"]
                else:
                    # List of lists
                    df = pd.DataFrame(raw, columns=["time", "open", "high", "low", "close", "volume"])
                df = df.sort_values("time").reset_index(drop=True)
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col])
                log.info(f"Candles fetched: {len(df)} rows | Last close: {df['close'].iloc[-1]:.2f}")
                return df
        except Exception as e:
            log.error(f"Failed to fetch candles: {e}")
        return pd.DataFrame()

    def get_ticker(self, symbol: str) -> float:
        url = f"{BASE_URL}/v2/tickers"
        try:
            r = self.session.get(url, timeout=10)
            r.raise_for_status()
            tickers = r.json().get("result", [])
            for t in tickers:
                if t.get("symbol") == symbol:
                    return float(t.get("mark_price", 0))
        except Exception as e:
            log.error(f"Failed to fetch ticker: {e}")
        return 0.0


# ─────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────
def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    hl2 = (df["high"] + df["low"]) / 2
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = [0.0] * len(df)
    direction  = [1]  * len(df)

    for i in range(1, len(df)):
        if upper_band.iloc[i] < upper_band.iloc[i-1] or df["close"].iloc[i-1] > upper_band.iloc[i-1]:
            final_upper = upper_band.iloc[i]
        else:
            final_upper = upper_band.iloc[i-1]

        if lower_band.iloc[i] > lower_band.iloc[i-1] or df["close"].iloc[i-1] < lower_band.iloc[i-1]:
            final_lower = lower_band.iloc[i]
        else:
            final_lower = lower_band.iloc[i-1]

        if supertrend[i-1] == final_upper:
            if df["close"].iloc[i] <= final_upper:
                supertrend[i] = final_upper
                direction[i]  = -1
            else:
                supertrend[i] = final_lower
                direction[i]  = 1
        else:
            if df["close"].iloc[i] >= final_lower:
                supertrend[i] = final_lower
                direction[i]  = 1
            else:
                supertrend[i] = final_upper
                direction[i]  = -1

    df = df.copy()
    df["supertrend"]   = supertrend
    df["st_direction"] = direction
    return df


# ─────────────────────────────────────────
#  TRADE JOURNAL
# ─────────────────────────────────────────
class Journal:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.trades: list = []
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                self.trades = json.load(f)

    def _save(self):
        with open(self.filepath, "w") as f:
            json.dump(self.trades, f, indent=2)

    def log_trade(self, trade: dict):
        self.trades.append(trade)
        self._save()
        log.info(f"📓 Trade logged: {trade}")

    def summary(self):
        if not self.trades:
            log.info("No trades in journal yet.")
            return
        wins   = [t for t in self.trades if t.get("outcome", 0) > 0]
        losses = [t for t in self.trades if t.get("outcome", 0) < 0]
        log.info(f"📊 Total: {len(self.trades)} | Wins: {len(wins)} | Losses: {len(losses)}")


# ─────────────────────────────────────────
#  PAPER TRADING ENGINE
# ─────────────────────────────────────────
class PaperTrader:
    def __init__(self, capital: float):
        self.capital  = capital
        self.position = None

    def open_trade(self, direction, entry, sl, tp, size_btc):
        if self.position:
            log.warning("⚠️  Already in a trade. Skipping.")
            return False
        self.position = {
            "direction": direction,
            "entry":     entry,
            "sl":        sl,
            "tp":        tp,
            "size_btc":  size_btc,
            "open_time": datetime.now(timezone.utc).isoformat()
        }
        log.info(f"📈 [PAPER] {direction.upper()} | Entry:{entry:.2f} SL:{sl:.2f} TP:{tp:.2f} Size:{size_btc:.5f} BTC")
        return True

    def check_exit(self, current_price: float, journal: Journal) -> bool:
        if not self.position:
            return False
        p = self.position
        closed, outcome = False, 0.0

        if p["direction"] == "long":
            if current_price <= p["sl"]:
                outcome = -((p["entry"] - p["sl"]) * p["size_btc"])
                log.info(f"🔴 LONG stopped out at {current_price:.2f} | P&L: ${outcome:.4f}")
                closed = True
            elif current_price >= p["tp"]:
                outcome = (p["tp"] - p["entry"]) * p["size_btc"]
                log.info(f"🟢 LONG hit TP at {current_price:.2f} | P&L: ${outcome:.4f}")
                closed = True
        else:
            if current_price >= p["sl"]:
                outcome = -((p["sl"] - p["entry"]) * p["size_btc"])
                log.info(f"🔴 SHORT stopped out at {current_price:.2f} | P&L: ${outcome:.4f}")
                closed = True
            elif current_price <= p["tp"]:
                outcome = (p["entry"] - p["tp"]) * p["size_btc"]
                log.info(f"🟢 SHORT hit TP at {current_price:.2f} | P&L: ${outcome:.4f}")
                closed = True

        if closed:
            self.capital += outcome
            journal.log_trade({**p, "exit_price": current_price, "outcome": round(outcome, 4),
                                "close_time": datetime.now(timezone.utc).isoformat()})
            self.position = None
        return closed


# ─────────────────────────────────────────
#  STRATEGY
# ─────────────────────────────────────────
class Strategy1030:
    def __init__(self):
        self.client           = DeltaClient()
        self.journal          = Journal(JOURNAL_FILE)
        self.paper            = PaperTrader(CAPITAL)
        self.trades_today     = 0
        self.losses_today     = 0
        self.last_trade_date  = None

    def _reset_daily(self):
        today = datetime.now(timezone.utc).date()
        if self.last_trade_date != today:
            self.trades_today    = 0
            self.losses_today    = 0
            self.last_trade_date = today
            log.info(f"📅 New day started: {today}")

    def get_30min_trend(self) -> str:
        df = self.client.get_candles(SYMBOL, TF_TREND, limit=150)
        if df.empty or len(df) < EMA_PERIOD + ST_PERIOD:
            return "none"
        df["ema50"] = calc_ema(df["close"], EMA_PERIOD)
        df = calc_supertrend(df, ST_PERIOD, ST_MULT)
        last = df.iloc[-1]
        bullish = last["close"] > last["ema50"] and last["st_direction"] == 1
        bearish = last["close"] < last["ema50"] and last["st_direction"] == -1
        if bullish:
            log.info(f"30m → BULLISH | Price={last['close']:.2f} EMA={last['ema50']:.2f} ST=GREEN")
            return "bullish"
        elif bearish:
            log.info(f"30m → BEARISH | Price={last['close']:.2f} EMA={last['ema50']:.2f} ST=RED")
            return "bearish"
        log.info("30m → NO CLEAR TREND")
        return "none"

    def check_10min_entry(self, trend: str):
        df = self.client.get_candles(SYMBOL, TF_ENTRY, limit=100)
        if df.empty or len(df) < ST_PERIOD + 5:
            return None
        df = calc_supertrend(df, ST_PERIOD, ST_MULT)
        last = df.iloc[-1]
        price, st, st_dir = last["close"], last["supertrend"], last["st_direction"]

        if trend == "bullish" and st_dir == 1:
            touched = last["low"] <= st * 1.005  # 0.5% tolerance
            if touched:
                swing_low = df.tail(10)["low"].min()
                sl = swing_low - 10
                sl_dist = price - sl
                if sl_dist <= 0: return None
                tp   = price + 2 * sl_dist
                size = round((CAPITAL * RISK_PERCENT) / sl_dist, 5)
                log.info(f"🎯 LONG setup | Entry={price:.2f} SL={sl:.2f} TP={tp:.2f} Size={size} BTC")
                return {"direction": "long", "entry": price, "sl": sl, "tp": tp, "size_btc": size}

        elif trend == "bearish" and st_dir == -1:
            touched = last["high"] >= st * 0.995  # 0.5% tolerance
            if touched:
                swing_high = df.tail(10)["high"].max()
                sl = swing_high + 10
                sl_dist = sl - price
                if sl_dist <= 0: return None
                tp   = price - 2 * sl_dist
                size = round((CAPITAL * RISK_PERCENT) / sl_dist, 5)
                log.info(f"🎯 SHORT setup | Entry={price:.2f} SL={sl:.2f} TP={tp:.2f} Size={size} BTC")
                return {"direction": "short", "entry": price, "sl": sl, "tp": tp, "size_btc": size}
        return None

    def run_once(self):
        self._reset_daily()
        if self.trades_today >= 1:
            log.info("⛔ Max 1 trade/day reached.")
            return
        if self.losses_today >= 2:
            log.info("⛔ 2 losses today. Stopping for the day.")
            return

        if self.paper.position:
            price = self.client.get_ticker(SYMBOL)
            if price > 0:
                closed = self.paper.check_exit(price, self.journal)
                if closed:
                    last = self.journal.trades[-1]
                    if last.get("outcome", 0) < 0:
                        self.losses_today += 1
            return

        trend = self.get_30min_trend()
        if trend == "none":
            return

        setup = self.check_10min_entry(trend)
        if not setup:
            log.info("No 10m entry signal.")
            return

        opened = self.paper.open_trade(**setup)
        if opened:
            self.trades_today += 1
            log.info(f"✅ Trade entered. Capital: ${self.paper.capital:.2f}")

    def run(self, interval: int = 60):
        log.info("=" * 45)
        log.info("  10-30 Bot | Paper Mode | Railway.app")
        log.info(f"  Symbol: {SYMBOL} | Capital: ${CAPITAL}")
        log.info("=" * 45)
        self.journal.summary()
        while True:
            try:
                log.info(f"── Tick {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC ──")
                self.run_once()
            except KeyboardInterrupt:
                log.info("Bot stopped.")
                break
            except Exception as e:
                log.error(f"Error: {e}")
            time.sleep(interval)


if __name__ == "__main__":
    bot = Strategy1030()
    bot.run(interval=60)
