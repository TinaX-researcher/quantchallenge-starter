#!/usr/bin/env python3
"""
Backtest Runner for QuantChallenge 2025
---------------------------------------
- Input: a full-game JSON file (list of events as described by the docs)
- Simulates a simple market (bid/ask around a "public" win-prob model)
- Drives orderbook + game events into your Strategy
- Fills IOC-at-touch and market orders immediately
- Computes end-of-game payoff and PnL

Usage:
    python backtest_runner.py /path/to/game.json

Outputs:
    - Summary PnL and capital
    - Trade log CSV (./trades.csv)
"""
import sys, json, math, csv, os
from typing import Optional, List, Dict, Any
from collections import deque
from enum import Enum

# ---------------- Types ----------------
class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    TEAM_A = 0  # Home team contract pays $100 on home win

# ---------------- Strategy ----------------
class Strategy:
    def __init__(self) -> None:
        self.reset_state()
        self._clr = None  # optional ML model

    def reset_state(self) -> None:
        self.time_left: Optional[float] = None
        self.home_score: int = 0
        self.away_score: int = 0
        self.possession: str = "unknown"
        self.best_bid: Optional[float] = None
        self.best_bid_qty: float = 0.0
        self.best_ask: Optional[float] = None
        self.best_ask_qty: float = 0.0
        self.mid: Optional[float] = None
        self.ticker = Ticker.TEAM_A
        self.start_capital: float = 100_000.0
        self.capital: float = 100_000.0
        self.position: int = 0
        self.vwap: Optional[float] = None
        self.base_edge_bps: float = 40.0
        self.late_game_threshold_s: float = 180
        self.kelly_frac: float = 0.25
        self.max_pos: int = 1500
        self.max_notional: float = 80_000.0
        self.stop_out_drawdown: float = 0.10
        self.p_ema_alpha: float = 0.30
        self.momo_window: int = 6
        self.vol_window: int = 20
        self.liq_ref_depth: float = 2_000.0
        self.p_hat_ema: Optional[float] = None
        self.p_hat_hist: deque = deque(maxlen=self.momo_window)
        self.mid_hist: deque = deque(maxlen=self.vol_window)
        self._last_desired: int = 0

    # ---- callbacks ----
    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        if ticker != self.ticker: return
        if side == Side.BUY:
            self.best_bid, self.best_bid_qty = price, quantity
        else:
            self.best_ask, self.best_ask_qty = price, quantity
        if self.best_bid is not None and self.best_ask is not None:
            self.mid = 0.5 * (self.best_bid + self.best_ask)
            self.mid_hist.append(self.mid)
        else:
            self.mid = None

    def on_account_update(self, ticker: Ticker, side: Side, price: float, quantity: float, capital_remaining: float) -> None:
        if ticker != self.ticker: return
        self.capital = capital_remaining
        signed = int(quantity if side == Side.BUY else -quantity)
        new_pos = self.position + signed
        if self.position == 0:
            self.vwap = price
        elif (self.position > 0 and signed > 0) or (self.position < 0 and signed < 0):
            self.vwap = (abs(self.position)*(self.vwap or price) + abs(signed)*price) / (abs(self.position)+abs(signed))
        else:
            if (self.position > 0 and new_pos < 0) or (self.position < 0 and new_pos > 0):
                self.vwap = price
            elif new_pos == 0:
                self.vwap = None
        self.position = new_pos
        if self.capital < self.start_capital * (1 - self.stop_out_drawdown):
            self._flatten()

    def on_game_event_update(self, event_type: str, home_away: str, home_score: int, away_score: int,
                             player_name: Optional[str], substituted_player_name: Optional[str], shot_type: Optional[str],
                             assist_player: Optional[str], rebound_type: Optional[str], coordinate_x: Optional[float],
                             coordinate_y: Optional[float], time_seconds: Optional[float]) -> None:
        self.time_left = time_seconds
        self.home_score = home_score
        self.away_score = away_score
        if event_type in ("SCORE","MISSED","TURNOVER","STEAL","FOUL","REBOUND","BLOCK"):
            if home_away in ("home","away"):
                self.possession = home_away

        if event_type == "END_GAME":
            self._flatten()
            self.reset_state()
            return

        if self.mid is None: return
        p_hat = self._estimate_home_win_prob()
        self._update_p_ema_and_hist(p_hat)
        p_mkt = self.mid / 100.0
        edge = p_hat - p_mkt

        p_momo = self._p_momentum_signal()
        vol = self._mid_volatility()
        liq_score = self._liquidity_score()

        min_edge = (self.base_edge_bps / 10_000.0)
        if (self.time_left or 9e9) < self.late_game_threshold_s: min_edge *= 0.5
        min_edge *= (1.0 + 5.0 * vol)

        if abs(edge) < min_edge or (edge * p_momo) <= 0: return

        comp = (abs(edge) / max(min_edge,1e-9)) * max(0.05, abs(p_momo)) * max(0.2, liq_score) / (1.0 + 10.0 * vol)
        comp = max(0.0, min(3.0, comp))

        kelly = max(min(2.0 * p_hat - 1.0, 0.9), -0.9)
        target_frac = 0.25 * kelly * (0.33 + 0.67 * (comp / 3.0))

        self._trade_toward_target(edge, target_frac, liq_score)

    # ---- internals ----
    def _estimate_home_win_prob(self) -> float:
        # Simple heuristic
        t = max(self.time_left or 0.0, 0.0)
        margin = self.home_score - self.away_score
        T = 2880.0 if (self.time_left and self.time_left > 2400) else 2400.0
        T = max(T, t)
        tl = t / T
        a, b = 0.10, 0.30
        x = (a * margin) + (b * margin) * (1.0 - tl)
        if self.possession == "home": x += 0.06 * (1.0 - tl)
        elif self.possession == "away": x -= 0.06 * (1.0 - tl)
        p = 1.0 / (1.0 + math.exp(-x))
        if t < 60: p = 0.05 + 0.90 * p
        return min(max(p, 0.02), 0.98)

    def _update_p_ema_and_hist(self, p_hat: float) -> None:
        if self.p_hat_ema is None: self.p_hat_ema = p_hat
        else: self.p_hat_ema = self.p_hat_ema + 0.30 * (p_hat - self.p_hat_ema)
        self.p_hat_hist.append(self.p_hat_ema)

    def _p_momentum_signal(self) -> float:
        if len(self.p_hat_hist) < 3: return 0.0
        y = list(self.p_hat_hist)
        n = len(y)
        x_mean = (n - 1) / 2.0
        y_mean = sum(y) / n
        num = sum((i - x_mean) * (y[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n)) + 1e-9
        slope = num / den
        return max(-1.0, min(1.0, slope / 0.02))

    def _mid_volatility(self) -> float:
        if len(self.mid_hist) < 5: return 0.0
        rets = []
        for i in range(1, len(self.mid_hist)):
            p0, p1 = self.mid_hist[i - 1], self.mid_hist[i]
            if p0 and p1:
                rets.append((p1 - p0) / max(p0, 1e-6))
        if not rets: return 0.0
        m = sum(rets) / len(rets)
        var = sum((r - m) ** 2 for r in rets) / max(1, len(rets) - 1)
        return math.sqrt(var)

    def _liquidity_score(self) -> float:
        depth = 0.0
        if self.best_bid_qty: depth += self.best_bid_qty
        if self.best_ask_qty: depth += self.best_ask_qty
        return max(0.0, min(1.0, depth / 2000.0))

    def _trade_toward_target(self, edge: float, target_frac: float, liq_score: float) -> None:
        if self.mid is None or self.mid <= 0: return
        liq_scale = 0.5 + 0.5 * liq_score
        max_contracts_by_notional = int((80_000.0 * liq_scale) / self.mid)
        desired = int(target_frac * (self.capital / self.mid))
        desired = max(-max_contracts_by_notional, min(max_contracts_by_notional, desired))
        desired = max(desired, 0) if edge > 0 else min(desired, 0)
        desired = max(-1500, min(1500, desired))
        if abs(desired - self._last_desired) <= 2: return
        self._last_desired = desired
        delta = desired - self.position
        if delta == 0: return
        if delta > 0:
            place_limit_order(Side.BUY, self.ticker, delta, self.best_ask or self.mid, ioc=True)
        else:
            place_limit_order(Side.SELL, self.ticker, -delta, self.best_bid or self.mid, ioc=True)

    def _flatten(self) -> None:
        if self.position == 0: return
        if self.position > 0:
            place_market_order(Side.SELL, self.ticker, self.position)
        else:
            place_market_order(Side.BUY, self.ticker, -self.position)
        self.position = 0
        self.vwap = None
        self._last_desired = 0

# ---------------- Market Simulator ----------------
class MarketSim:
    def __init__(self, spread_cents: float = 0.6, depth_each: float = 1500.0) -> None:
        self.spread = spread_cents
        self.depth_each = depth_each
        self.best_bid = None
        self.best_ask = None

    @staticmethod
    def public_winprob(home_score: int, away_score: int, time_left: float, possession: str) -> float:
        t = max(time_left or 0.0, 0.0)
        margin = home_score - away_score
        T = 2880.0 if (time_left and time_left > 2400) else 2400.0
        T = max(T, t)
        tl = t / T
        a, b = 0.09, 0.26
        x = (a * margin) + (b * margin) * (1.0 - tl)
        if possession == "home": x += 0.05 * (1.0 - tl)
        elif possession == "away": x -= 0.05 * (1.0 - tl)
        p = 1.0 / (1.0 + math.exp(-x))
        if t < 60: p = 0.10 + 0.80 * p
        return min(max(p, 0.03), 0.97)

    def update_quotes(self, strat: Strategy):
        p_pub = self.public_winprob(strat.home_score, strat.away_score, strat.time_left or 0.0, strat.possession)
        mid = 100.0 * p_pub
        self.best_bid = max(0.01, mid - self.spread/2)
        self.best_ask = min(99.99, mid + self.spread/2)
        strat.on_orderbook_update(Ticker.TEAM_A, Side.BUY, self.depth_each, self.best_bid)
        strat.on_orderbook_update(Ticker.TEAM_A, Side.SELL, self.depth_each, self.best_ask)

# ---------------- Glue: order handlers ----------------
TRADE_LOG: List[Dict[str, Any]] = []
_strat: Optional[Strategy] = None
_mkt: Optional[MarketSim] = None

def _record_fill(side: Side, qty: int, price: float, capital_before: float, capital_after: float, tsec: float) -> None:
    TRADE_LOG.append({
        "time_seconds": tsec,
        "side": side.name,
        "qty": qty,
        "price": round(price, 4),
        "capital_before": round(capital_before, 2),
        "capital_after": round(capital_after, 2)
    })

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    global _strat, _mkt
    qty = int(quantity)
    if _strat is None or _mkt is None: return 0
    if side == Side.BUY:
        touch = _mkt.best_ask
        if touch is None: return 0
        if price >= touch:
            fill_px = touch
            capital_before = _strat.capital
            notional = qty * fill_px
            _strat.on_account_update(ticker, side, fill_px, qty, capital_remaining=_strat.capital - notional)
            _record_fill(side, qty, fill_px, capital_before, _strat.capital, _strat.time_left or 0.0)
            return 1
        else: return 0
    else:
        touch = _mkt.best_bid
        if touch is None: return 0
        if price <= touch:
            fill_px = touch
            capital_before = _strat.capital
            notional = qty * fill_px
            _strat.on_account_update(ticker, side, fill_px, qty, capital_remaining=_strat.capital + notional)
            _record_fill(side, qty, fill_px, capital_before, _strat.capital, _strat.time_left or 0.0)
            return 1
        else: return 0

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    global _strat, _mkt
    qty = int(quantity)
    if _strat is None or _mkt is None: return
    if side == Side.BUY:
        touch = _mkt.best_ask or (_strat.mid or 50.0)
        fill_px = touch
        capital_before = _strat.capital
        notional = qty * fill_px
        _strat.on_account_update(ticker, side, fill_px, qty, capital_remaining=_strat.capital - notional)
        _record_fill(side, qty, fill_px, capital_before, _strat.capital, _strat.time_left or 0.0)
    else:
        touch = _mkt.best_bid or (_strat.mid or 50.0)
        fill_px = touch
        capital_before = _strat.capital
        notional = qty * fill_px
        _strat.on_account_update(ticker, side, fill_px, qty, capital_remaining=_strat.capital + notional)
        _record_fill(side, qty, fill_px, capital_before, _strat.capital, _strat.time_left or 0.0)

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    return True

# ---------------- Runner ----------------
def run(file_path: str) -> None:
    global _strat, _mkt, TRADE_LOG
    with open(file_path, "r") as f:
        data = json.load(f)

    events = data if isinstance(data, list) else data.get("events", [])
    if not events:
        print("No events found in JSON.")
        return

    _strat = Strategy()
    _mkt = MarketSim(spread_cents=0.6, depth_each=1500.0)
    TRADE_LOG = []

    for ev in events:
        _mkt.update_quotes(_strat)
        _strat.on_game_event_update(
            event_type=ev.get("event_type"),
            home_away=ev.get("home_away"),
            home_score=int(ev.get("home_score", 0)),
            away_score=int(ev.get("away_score", 0)),
            player_name=ev.get("player_name"),
            substituted_player_name=ev.get("substituted_player_name"),
            shot_type=ev.get("shot_type"),
            assist_player=ev.get("assist_player"),
            rebound_type=ev.get("rebound_type"),
            coordinate_x=ev.get("coordinate_x"),
            coordinate_y=ev.get("coordinate_y"),
            time_seconds=float(ev.get("time_seconds") if ev.get("time_seconds") is not None else 0.0)
        )
        if ev.get("event_type") == "END_GAME": break

    home_won = (_strat.home_score > _strat.away_score)
    payoff = 100
if __name__ == "__main__":
    import sys, json
    print(">> starting template.py")  # sanity

    s = Strategy()
    if len(sys.argv) < 2:
        print("Usage: python template.py /path/to/game.json"); raise SystemExit(1)

    game_path = sys.argv[1]
    print(f">> loading {game_path}")
    with open(game_path, "r") as f:
        data = json.load(f)

    events = data if isinstance(data, list) else data.get("events", [])
    print(f">> events loaded: {len(events)}")

    for i, ev in enumerate(events):
        s.on_game_event_update(
            event_type=ev.get("event_type"),
            home_away=ev.get("home_away"),
            home_score=int(ev.get("home_score", 0)),
            away_score=int(ev.get("away_score", 0)),
            player_name=ev.get("player_name"),
            substituted_player_name=ev.get("substituted_player_name"),
            shot_type=ev.get("shot_type"),
            assist_player=ev.get("assist_player"),
            rebound_type=ev.get("rebound_type"),
            coordinate_x=ev.get("coordinate_x"),
            coordinate_y=ev.get("coordinate_y"),
            time_seconds=float(ev.get("time_seconds") or 0.0),
        )
        if ev.get("event_type") == "END_GAME":
            break

    print("✅ Finished replaying game.")
