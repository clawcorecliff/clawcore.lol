# clawcore.lol

A real-time trading dashboard for a multi-pod Kalshi prediction market trading system. Tracks live and paper positions, P&L, pod health, and system metrics across 10+ concurrent trading pods.

---

## What It Does

- **Live vs Paper split** — side-by-side view of real money positions and shadow paper trades
- **Per-pod detail pages** — win rate, P&L curve, recent trades, live log tail, config snapshot
- **System metrics** — memory pressure, CPU, network throughput, top processes, pod daemon health
- **API-first** — every view is backed by a JSON endpoint; easy to extend or integrate

---

## Stack

| Layer | Tech |
|-------|------|
| Server | Python 3.9, Flask |
| Data | SQLite (paper + live trade DBs) |
| Charts | Chart.js |
| System stats | psutil |
| Styling | Vanilla CSS, dark theme |

---

## Routes

| Route | Description |
|-------|-------------|
| `/` | Overview — all pods, combined P&L, win rates |
| `/pod/<name>` | Per-pod detail — trades, chart, logs, config |
| `/system` | System metrics — memory, CPU, network, processes |
| `/api/overview` | JSON — all pod summaries |
| `/api/trades` | JSON — full trade ledger |
| `/api/pod/<name>` | JSON — pod detail |
| `/api/pod/<name>/logs` | JSON — recent log lines |
| `/api/charts/pnl` | JSON — P&L time series for charting |
| `/api/system` | JSON — live system metrics |

---

## Running Locally

```bash
# Clone and set up
git clone https://github.com/clawcorecliff/clawcore.lol.git
cd clawcore.lol

# Create virtualenv
python3 -m venv .venv
source .venv/bin/activate
pip install flask psutil pyyaml

# Point at your trade DBs (edit these paths in app.py or set env vars)
export PAPER_DB=~/trading-bot/data/paper_trades.db
export LIVE_DB=~/trading-bot/data/live_trades.db
export POD_DIR=~/trading-bot/pods
export LOG_DIR=~/trading-bot/logs

# Run
flask --app app run --port 7799
```

Open [http://localhost:7799](http://localhost:7799).

---

## Trading Bot Context

This dashboard is the front-end for a proprietary prediction market trading system built on [Kalshi](https://kalshi.com). The backend runs 10+ independent pods (BTC, ETH, SOL, XRP, WTI, EUR/USD, USD/JPY, DOGE, INX, and more) each with:

- GBM probability model with EWMA volatility
- Student-t fat tail correction
- Order flow imbalance (OFI) signal
- Longshot bias fade
- Half-Kelly position sizing
- 30% model-market divergence cap
- 25% daily drawdown circuit breaker

Live pods trade real money. Paper pods shadow-trade the same signals without submitting orders, providing a continuous benchmark.

---

## Structure

```
clawcore.lol/
├── app.py              # Flask app, all routes and API logic
├── templates/
│   ├── index.html      # Overview page
│   ├── pod.html        # Per-pod detail
│   └── system_stats.html  # System metrics page
├── static/
│   └── ...             # CSS, JS assets
└── README.md
```

---

## Deployment

Designed to run as a persistent local process on the trading machine, proxied behind Cloudflare or Nginx for remote access.

```bash
# Run headless
nohup flask --app app run --host 0.0.0.0 --port 7799 >> logs/dashboard.log 2>&1 &
```

---

## License

Private. Not open source.
