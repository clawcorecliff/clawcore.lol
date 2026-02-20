from __future__ import annotations

from flask import Flask, render_template, jsonify, request
import sqlite3
import subprocess
import os
import json
import re
import yaml
import glob
from pathlib import Path
from datetime import datetime, date, timedelta
import time

app = Flask(__name__)

PAPER_DB = Path("~/trading-bot/data/paper_trades.db").expanduser()
LIVE_DB = Path("~/trading-bot/data/live_trades.db").expanduser()
LOG_DIR = Path("~/trading-bot/logs").expanduser()
POD_DIR = Path("~/trading-bot/pods").expanduser()

PODS = ["btc", "eth", "sol", "xrp", "wti", "usdjpy", "eurusd", "weather"]

ASSET_DISPLAY = {
    "btc": "BTC", "eth": "ETH", "sol": "SOL", "xrp": "XRP",
    "wti": "WTI", "usdjpy": "USD/JPY", "eurusd": "EUR/USD", "weather": "Weather",
}


# ── DB helpers ──

def _open_db(path: Path):
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        return None


def query(db_path: Path, sql: str, args: tuple = ()) -> list[dict]:
    conn = _open_db(db_path)
    if not conn:
        return []
    try:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]
    except Exception:
        return []
    finally:
        conn.close()


def query_paper(sql, args=()):
    return query(PAPER_DB, sql, args)


def query_live(sql, args=()):
    return query(LIVE_DB, sql, args)


# ── Pod helpers ──

def is_pod_running(pod_name: str) -> bool:
    pid_file = Path(f"/tmp/tradingbot-{pod_name}.pid")
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            return True
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    # Fallback: pgrep
    try:
        r = subprocess.run(["pgrep", "-f", f"pods/{pod_name}/daemon.py"],
                           capture_output=True, text=True, timeout=3)
        return r.returncode == 0
    except Exception:
        return False


def get_pod_config(pod_name: str) -> dict:
    p = POD_DIR / pod_name / "config.yaml"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def get_pod_mode(config: dict) -> str:
    """Return 'live' or 'paper' based on config."""
    trading = config.get("trading", {})
    if trading.get("mode") == "live":
        return "live"
    return "paper"


def get_pod_log_path(pod_name: str):
    log_dir = LOG_DIR / pod_name
    if not log_dir.exists():
        return None
    today = date.today().strftime("%Y-%m-%d")
    today_log = log_dir / f"daemon-{today}.log"
    if today_log.exists():
        return today_log
    logs = sorted(log_dir.glob("daemon-*.log"))
    return logs[-1] if logs else None


def get_pod_last_log_lines(pod_name: str, n: int = 50) -> list[str]:
    log_path = get_pod_log_path(pod_name)
    if not log_path:
        return []
    try:
        r = subprocess.run(["tail", f"-{n}", str(log_path)], capture_output=True, text=True, timeout=3)
        return r.stdout.splitlines()
    except Exception:
        return []


def parse_log_for_status(pod_name: str) -> dict:
    lines = get_pod_last_log_lines(pod_name, 200)
    status = {"last_price": None, "last_sigma": None, "last_scan_ago": None, "last_scan_ts": None, "error_count": 0}
    if not lines:
        return status
    for line in reversed(lines):
        m = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if m:
            try:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                status["last_scan_ts"] = ts.isoformat()
                secs = max(0, int((datetime.now() - ts).total_seconds()))
                if secs < 60:
                    status["last_scan_ago"] = f"{secs}s ago"
                elif secs < 3600:
                    status["last_scan_ago"] = f"{secs // 60}m ago"
                else:
                    status["last_scan_ago"] = f"{secs // 3600}h {(secs % 3600) // 60}m ago"
            except Exception:
                pass
            break
    for line in reversed(lines):
        if "σ=" in line:
            m = re.search(r"σ=([\d.]+)", line)
            if m:
                status["last_sigma"] = float(m.group(1))
                break
    # Price from recent logs
    for line in reversed(lines):
        m = re.search(r"\$([0-9,]+\.?\d*)", line)
        if m:
            try:
                p = float(m.group(1).replace(",", ""))
                if p > 0.001:
                    status["last_price"] = p
                    break
            except Exception:
                pass
    status["error_count"] = sum(1 for l in lines if "[ERROR]" in l)
    return status


def _compute_stats(rows):
    total = len(rows)
    wins = sum(1 for r in rows if r.get("outcome") == "win")
    losses = sum(1 for r in rows if r.get("outcome") == "loss")
    pending = sum(1 for r in rows if r.get("outcome") == "pending")
    settled = wins + losses
    win_rate = (wins / settled * 100) if settled > 0 else None
    total_pnl = sum((r.get("pnl_dollars") or 0) for r in rows if r.get("pnl_dollars") is not None)
    avg_edge = None
    edges = [(r.get("edge") or 0) for r in rows if r.get("edge") is not None]
    if edges:
        avg_edge = sum(edges) / len(edges) * 100

    today = date.today().isoformat()
    today_rows = [r for r in rows if r.get("created_date") == today]
    today_pnl = sum((r.get("pnl_dollars") or 0) for r in today_rows if r.get("pnl_dollars") is not None)

    return {
        "total": total, "wins": wins, "losses": losses, "pending": pending,
        "settled": settled, "win_rate": win_rate, "total_pnl": total_pnl,
        "avg_edge_pct": avg_edge,
        "today_total": len(today_rows), "today_pnl": today_pnl,
    }


# ── Routes ──

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/pod/<name>")
def pod_detail_page(name: str):
    name = name.lower()
    if name not in PODS:
        return "Pod not found", 404
    return render_template("pod.html", pod_name=name)


# ── API: Overview ──

@app.route("/api/overview")
def api_overview():
    """Main overview data with live/paper separation."""
    # Live data
    live_rows = query_live("SELECT * FROM live_trades")
    live_stats = _compute_stats(live_rows)
    live_open = [dict(r) for r in query_live(
        "SELECT * FROM live_trades WHERE outcome='pending' ORDER BY entered_at DESC")]
    for r in live_open:
        if r.get("entered_at"):
            r["entered_at_fmt"] = datetime.utcfromtimestamp(r["entered_at"]).strftime("%m/%d %H:%M")

    # Latest bankroll
    bankroll_row = query_live("SELECT bankroll FROM live_bankroll_log ORDER BY id DESC LIMIT 1")
    live_bankroll = bankroll_row[0]["bankroll"] if bankroll_row else None

    # Live pods
    live_pods = []
    paper_pods = []
    for pod_name in PODS:
        config = get_pod_config(pod_name)
        mode = get_pod_mode(config)
        running = is_pod_running(pod_name)
        log_status = parse_log_for_status(pod_name)
        asset = ASSET_DISPLAY.get(pod_name, pod_name.upper())

        # Get stats from appropriate DB
        if mode == "live":
            rows = query_live("SELECT outcome, pnl_dollars, created_date, edge FROM live_trades WHERE asset=?", (asset,))
        else:
            rows = query_paper("SELECT outcome, pnl_dollars, created_date, edge FROM paper_trades WHERE asset=?", (asset,))
        stats = _compute_stats(rows)

        # Bankroll from config
        pt = config.get("paper_trading", {})
        bankroll = pt.get("bankroll")

        pod_info = {
            "name": pod_name, "asset": asset, "mode": mode, "running": running,
            "price": log_status["last_price"], "sigma": log_status["last_sigma"],
            "last_scan_ago": log_status["last_scan_ago"], "error_count": log_status["error_count"],
            "bankroll": bankroll, **stats,
        }

        if mode == "live":
            live_pods.append(pod_info)
        else:
            paper_pods.append(pod_info)

    # Aggregate paper stats
    all_paper = query_paper("SELECT outcome, pnl_dollars, created_date, edge FROM paper_trades")
    paper_stats = _compute_stats(all_paper)

    return jsonify({
        "live": {
            "bankroll": live_bankroll,
            "stats": live_stats,
            "open_positions": live_open,
            "pods": live_pods,
        },
        "paper": {
            "stats": paper_stats,
            "pods": paper_pods,
        },
        "generated_at": datetime.now().isoformat(),
    })


# ── API: Trades (unified with source param) ──

@app.route("/api/trades")
def api_trades():
    source = request.args.get("source", "paper")  # paper or live
    pod = request.args.get("pod", "")
    outcome = request.args.get("outcome", "")
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")
    limit = min(int(request.args.get("limit", 500)), 2000)
    sort = request.args.get("sort", "entered_at")
    order = request.args.get("order", "desc")

    allowed_sorts = {"entered_at", "asset", "outcome", "pnl_dollars", "edge", "created_date", "cost_basis"}
    if sort not in allowed_sorts:
        sort = "entered_at"
    if order not in ("asc", "desc"):
        order = "desc"

    where, args = [], []
    if pod:
        asset = ASSET_DISPLAY.get(pod.lower(), pod.upper())
        where.append("asset = ?")
        args.append(asset)
    if outcome:
        where.append("outcome = ?")
        args.append(outcome)
    if date_from:
        where.append("created_date >= ?")
        args.append(date_from)
    if date_to:
        where.append("created_date <= ?")
        args.append(date_to)

    w = ("WHERE " + " AND ".join(where)) if where else ""
    table = "live_trades" if source == "live" else "paper_trades"
    db = LIVE_DB if source == "live" else PAPER_DB

    sql = f"""SELECT id, entered_at, settled_at, asset, ticker, series, contract_type,
               side, entry_price_cents, contracts, cost_basis,
               our_fair_value, edge, confidence, kelly_fraction,
               sigma_realized, sigma_implied, underlying_price,
               strike_floor, strike_cap, tau_seconds,
               settlement_result, outcome, pnl_dollars, notes, created_date
        FROM {table} {w} ORDER BY {sort} {order} LIMIT ?"""
    args.append(limit)
    rows = query(db, sql, tuple(args))

    for r in rows:
        if r.get("entered_at"):
            r["entered_at_fmt"] = datetime.utcfromtimestamp(r["entered_at"]).strftime("%Y-%m-%d %H:%M:%S")
        if r.get("settled_at"):
            r["settled_at_fmt"] = datetime.utcfromtimestamp(r["settled_at"]).strftime("%Y-%m-%d %H:%M:%S")
        r["edge_pct"] = round((r.get("edge") or 0) * 100, 1)

    return jsonify({"trades": rows, "count": len(rows), "source": source})


# ── API: Pod detail ──

@app.route("/api/pod/<name>")
def api_pod_detail(name: str):
    name = name.lower()
    if name not in PODS:
        return jsonify({"error": "unknown pod"}), 404

    asset = ASSET_DISPLAY.get(name, name.upper())
    config = get_pod_config(name)
    mode = get_pod_mode(config)
    running = is_pod_running(name)
    log_status = parse_log_for_status(name)

    # Stats from both DBs
    paper_rows = query_paper("SELECT outcome, pnl_dollars, created_date, edge FROM paper_trades WHERE asset=?", (asset,))
    live_rows = query_live("SELECT outcome, pnl_dollars, created_date, edge FROM live_trades WHERE asset=?", (asset,))
    paper_stats = _compute_stats(paper_rows)
    live_stats = _compute_stats(live_rows)

    # Recent trades from appropriate DB
    source = request.args.get("source", mode)
    if source == "live":
        recent = query_live(
            """SELECT id, entered_at, ticker, side, entry_price_cents, contracts,
                      cost_basis, our_fair_value, edge, sigma_realized, outcome,
                      pnl_dollars, created_date, underlying_price
               FROM live_trades WHERE asset=? ORDER BY entered_at DESC LIMIT 30""", (asset,))
    else:
        recent = query_paper(
            """SELECT id, entered_at, ticker, side, entry_price_cents, contracts,
                      cost_basis, our_fair_value, edge, sigma_realized, outcome,
                      pnl_dollars, created_date, underlying_price
               FROM paper_trades WHERE asset=? ORDER BY entered_at DESC LIMIT 30""", (asset,))

    for r in recent:
        if r.get("entered_at"):
            r["entered_at_fmt"] = datetime.utcfromtimestamp(r["entered_at"]).strftime("%Y-%m-%d %H:%M:%S")
        r["edge_pct"] = round((r.get("edge") or 0) * 100, 1)

    # P&L over time for chart
    pnl_paper = query_paper(
        "SELECT entered_at, pnl_dollars FROM paper_trades WHERE asset=? AND outcome!='pending' ORDER BY entered_at", (asset,))
    pnl_live = query_live(
        "SELECT entered_at, pnl_dollars FROM live_trades WHERE asset=? AND outcome!='pending' ORDER BY entered_at", (asset,))

    def cumulative_pnl(rows):
        pts, total = [], 0.0
        for r in rows:
            total += (r["pnl_dollars"] or 0)
            pts.append({"x": r["entered_at"], "y": round(total, 4),
                        "label": datetime.utcfromtimestamp(r["entered_at"]).strftime("%m/%d %H:%M") if r["entered_at"] else ""})
        return pts

    # Bankroll log for live pod
    bankroll_history = []
    if mode == "live":
        bankroll_history = query_live(
            "SELECT timestamp, bankroll, event, amount FROM live_bankroll_log WHERE asset=? ORDER BY timestamp", (asset,))

    return jsonify({
        "name": name, "asset": asset, "mode": mode, "running": running,
        "config": config, "log_status": log_status,
        "paper_stats": paper_stats, "live_stats": live_stats,
        "recent_trades": recent, "source": source,
        "pnl_chart": {"paper": cumulative_pnl(pnl_paper), "live": cumulative_pnl(pnl_live)},
        "bankroll_history": bankroll_history,
    })


@app.route("/api/pod/<name>/logs")
def api_pod_logs(name: str):
    name = name.lower()
    if name not in PODS:
        return jsonify({"error": "unknown pod"}), 404
    n = min(int(request.args.get("n", 50)), 500)
    lines = get_pod_last_log_lines(name, n)
    return jsonify({"lines": lines, "count": len(lines)})


# ── API: Charts ──

@app.route("/api/charts/pnl")
def api_charts_pnl():
    source = request.args.get("source", "paper")
    db = LIVE_DB if source == "live" else PAPER_DB
    table = "live_trades" if source == "live" else "paper_trades"
    rows = query(db, f"SELECT asset, entered_at, pnl_dollars FROM {table} WHERE outcome!='pending' ORDER BY entered_at")

    by_asset = {}
    agg_total = 0.0
    agg_points = []
    running = {}

    for r in rows:
        asset = r["asset"]
        pnl = r["pnl_dollars"] or 0
        ts = r["entered_at"]
        if asset not in by_asset:
            by_asset[asset] = []
            running[asset] = 0.0
        running[asset] += pnl
        agg_total += pnl
        by_asset[asset].append({"x": ts, "y": round(running[asset], 4),
                                "label": datetime.utcfromtimestamp(ts).strftime("%m/%d %H:%M") if ts else ""})
        agg_points.append({"x": ts, "y": round(agg_total, 4),
                           "label": datetime.utcfromtimestamp(ts).strftime("%m/%d %H:%M") if ts else ""})

    return jsonify({"by_asset": by_asset, "aggregate": agg_points, "source": source})


# ── System Stats ──

@app.route("/system")
def system_page():
    return render_template("system_stats.html")

@app.route("/api/system")
def api_system():
    import psutil, time

    # Memory
    vm = psutil.virtual_memory()
    pressure_pct = round((vm.active + vm.wired) / vm.total * 100, 1) if hasattr(vm, "wired") else round(vm.percent, 1)
    wired_mb = round(getattr(vm, "wired", 0) / 1024**2)
    active_mb = round(vm.active / 1024**2)
    compressed_mb = round(getattr(vm, "compressed", 0) / 1024**2)
    cached_mb = round(getattr(vm, "cached", 0) / 1024**2)
    free_mb = round(vm.available / 1024**2)
    total_mb = round(vm.total / 1024**2)

    # CPU
    cpu_pct = psutil.cpu_percent(interval=0.5)
    per_core = psutil.cpu_percent(interval=None, percpu=True)

    # Network (delta over 1s)
    n1 = psutil.net_io_counters()
    time.sleep(1)
    n2 = psutil.net_io_counters()
    recv_kbs = round((n2.bytes_recv - n1.bytes_recv) / 1024, 1)
    sent_kbs = round((n2.bytes_sent - n1.bytes_sent) / 1024, 1)

    # Swap
    sw = psutil.swap_memory()
    swap_used_mb = round(sw.used / 1024**2)
    swap_total_mb = round(sw.total / 1024**2)

    # Top processes by memory
    procs = []
    for p in sorted(psutil.process_iter(["pid", "name", "memory_info"]), 
                    key=lambda x: x.info["memory_info"].rss if x.info["memory_info"] else 0,
                    reverse=True)[:5]:
        try:
            procs.append({
                "name": p.info["name"],
                "pid": p.info["pid"],
                "memory_mb": round(p.info["memory_info"].rss / 1024**2, 1)
            })
        except Exception:
            pass

    # Running pods
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    pod_lines = [l for l in result.stdout.split("\n") if "daemon.py" in l and "grep" not in l]
    pod_names = []
    for l in pod_lines:
        import re as _re
        m = _re.search(r"pods/([^/]+)/daemon\.py", l)
        if m:
            pod_names.append(m.group(1))

    return jsonify({
        "memory": {
            "pressure_pct": pressure_pct,
            "wired_mb": wired_mb,
            "active_mb": active_mb,
            "compressed_mb": compressed_mb,
            "cached_mb": cached_mb,
            "free_mb": free_mb,
            "total_mb": total_mb,
        },
        "cpu": {"usage_pct": cpu_pct, "per_core": per_core},
        "network": {"recv_kbs": recv_kbs, "sent_kbs": sent_kbs},
        "swap": {"used_mb": swap_used_mb, "total_mb": swap_total_mb},
        "processes": procs,
        "pods": {"count": len(pod_names), "names": sorted(pod_names)},
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7799))
    print(f"🚀 Trading Dashboard on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
