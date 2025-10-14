# app.py
"""
Crypto Portfolio Investment Manager
Streamlit app that:
- Fetches crypto price data using yfinance
- Stores data into a local SQLite DB
- Computes basic metrics (min, max, mean, std)
- Forms portfolios (equal-weight, risk-parity or custom)
- Computes portfolio returns and risk
- Runs parallelized risk checks (volatility, Sharpe, max drawdown, Sortino)
- Sends email alerts when rules fail (Gmail app password via env var)
- Runs stress tests (bull, bear, volatile)
- Trains a simple linear regression on returns and reports MSE & R2

How to run:
1. Install requirements:
   pip install yfinance pandas numpy scikit-learn streamlit matplotlib
2. (Optional) Set environment variables for email alerts:
   EMAIL_SENDER=ankemvamsikrishna@gmail.com
   EMAIL_PASSWORD=<your_app_password>
3. Run:
   streamlit run app.py

Note: Do NOT hardcode your email password in the file. Use environment variables or Streamlit secrets.
"""

import os
import sqlite3
import json
import ast
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import streamlit as st

# ---------------------------
# Configuration
# ---------------------------
DB_FILE = "crypto_portfolio.db"
RISK_THRESHOLDS = {
    "volatility": 1.0,     # annualized volatility threshold (100% by default)
    "sharpe": 0.0,         # require Sharpe >= 0
    "max_drawdown": -0.2,  # do not accept drawdown worse than -20%
    "sortino": 0.0
}

# ---------------------------
# Database helpers
# ---------------------------

def init_db(db_file: str = DB_FILE):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    # price data with primary key to avoid duplicates
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS price_data (
            ticker TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adj_close REAL,
            volume REAL,
            PRIMARY KEY (ticker, date)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            ticker TEXT,
            start_date TEXT,
            end_date TEXT,
            min REAL,
            max REAL,
            mean REAL,
            std REAL,
            PRIMARY KEY (ticker, start_date, end_date)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio_results (
            name TEXT PRIMARY KEY,
            weights TEXT,
            start_date TEXT,
            end_date TEXT,
            portfolio_return REAL,
            portfolio_risk REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS risk_checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            check_name TEXT,
            value REAL,
            threshold REAL,
            passed INTEGER,
            checked_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            model_type TEXT,
            mse REAL,
            r2 REAL,
            checked_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def store_price_df(df: pd.DataFrame, db_file: str = DB_FILE):
    """Insert or replace rows into price_data table. Expects columns: ticker,date,open,high,low,close,adj_close,volume"""
    if df is None or df.empty:
        return 0
    rows = []
    for _, r in df.iterrows():
        rows.append((
            r.get("ticker"),
            r.get("date"),
            float(r.get("open", np.nan)) if not pd.isna(r.get("open", np.nan)) else None,
            float(r.get("high", np.nan)) if not pd.isna(r.get("high", np.nan)) else None,
            float(r.get("low", np.nan)) if not pd.isna(r.get("low", np.nan)) else None,
            float(r.get("close", np.nan)) if not pd.isna(r.get("close", np.nan)) else None,
            float(r.get("adj_close", np.nan)) if not pd.isna(r.get("adj_close", np.nan)) else None,
            float(r.get("volume", np.nan)) if not pd.isna(r.get("volume", np.nan)) else None,
        ))
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT OR REPLACE INTO price_data (ticker, date, open, high, low, close, adj_close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows
    )
    conn.commit()
    conn.close()
    return len(rows)


def store_metrics_in_db(ticker: str, start_date: str, end_date: str, metrics: dict, db_file: str = DB_FILE):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO metrics (ticker, start_date, end_date, min, max, mean, std)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (ticker, start_date, end_date, metrics.get("min"), metrics.get("max"), metrics.get("mean"), metrics.get("std"))
    )
    conn.commit()
    conn.close()


def store_portfolio_results(name: str, weights: dict, start_date: str, end_date: str, port_return: float, port_risk: float, db_file: str = DB_FILE):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO portfolio_results (name, weights, start_date, end_date, portfolio_return, portfolio_risk)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (name, json.dumps(weights), start_date, end_date, float(port_return), float(port_risk))
    )
    conn.commit()
    conn.close()


def store_risk_checks_bulk(ticker: str, results: list, db_file: str = DB_FILE):
    """results: list of tuples (check_name, value, threshold, passed)"""
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    rows = []
    for name, val, thr, passed in results:
        rows.append((ticker, name, None if val is None else float(val), thr, int(passed), now))
    cur.executemany(
        """
        INSERT INTO risk_checks (ticker, check_name, value, threshold, passed, checked_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows
    )
    conn.commit()
    conn.close()


def store_prediction_result(name: str, model_type: str, mse: float, r2: float, db_file: str = DB_FILE):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute(
        """
        INSERT INTO predictions (name, model_type, mse, r2, checked_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (name, model_type, float(mse), float(r2), now)
    )
    conn.commit()
    conn.close()


# ---------------------------
# Data fetching & metrics
# ---------------------------

def fetch_ticker_data(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Fetch data using yfinance and return standardized DataFrame.
    start/end should be ISO date strings.
    """
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
    except Exception as e:
        raise
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    # Normalize column names
    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume"
    })
    # Keep only required columns and add ticker
    df = df[["date", "open", "high", "low", "close", "adj_close", "volume"]].copy()
    df["ticker"] = ticker
    # Convert date to YYYY-MM-DD string
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]]


def compute_basic_metrics_from_df(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"min": None, "max": None, "mean": None, "std": None}
    arr = pd.to_numeric(df["adj_close"], errors="coerce").dropna().values
    if len(arr) == 0:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1))
    }


def load_price_df_from_db(ticker: str, start_date: str = None, end_date: str = None, db_file: str = DB_FILE) -> pd.DataFrame:
    conn = sqlite3.connect(db_file)
    q = "SELECT date, adj_close FROM price_data WHERE ticker = ?"
    params = [ticker]
    if start_date:
        q += " AND date >= ?"
        params.append(start_date)
    if end_date:
        q += " AND date <= ?"
        params.append(end_date)
    q += " ORDER BY date ASC"
    df = pd.read_sql_query(q, conn, params=params)
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])  # ensure datetime
    return df


# ---------------------------
# Portfolio & performance
# ---------------------------

def validate_weights(weights: dict) -> bool:
    try:
        total = sum(float(v) for v in weights.values())
    except Exception:
        return False
    return abs(total - 1.0) < 1e-6


def compute_portfolio_returns(tickers: list, weights: dict, start_date: str, end_date: str) -> dict:
    dfs = []
    for t in tickers:
        df = load_price_df_from_db(t, start_date, end_date)
        if df is None or df.empty:
            continue
        df = df.rename(columns={"adj_close": t})
        df = df.set_index("date")
        dfs.append(df[[t]])
    if not dfs:
        return {}
    merged = pd.concat(dfs, axis=1).dropna().sort_index()
    if merged.empty:
        return {}
    daily_ret = merged.pct_change().dropna()
    # Order weights according to merged columns
    ordered_tickers = list(merged.columns)
    w = np.array([float(weights.get(t, 0.0)) for t in ordered_tickers])
    portfolio_daily = daily_ret.values.dot(w)
    portfolio_series = pd.Series(portfolio_daily.ravel(), index=daily_ret.index)
    portfolio_cum = (1 + portfolio_series).cumprod()
    # annualized return and risk (approx)
    port_return = float(portfolio_series.mean() * 252)
    port_risk = float(portfolio_series.std(ddof=1) * np.sqrt(252))
    return {
        "daily_returns": portfolio_series,
        "cum_returns": portfolio_cum,
        "annual_return": port_return,
        "annual_risk": port_risk,
        "daily_ret_table": daily_ret
    }


# ---------------------------
# Risk checks
# ---------------------------

def compute_sharpe(daily_returns: pd.Series, risk_free_rate: float = 0.0):
    ann_mean = daily_returns.mean() * 252
    ann_std = daily_returns.std(ddof=1) * np.sqrt(252)
    if ann_std == 0 or np.isnan(ann_std):
        return np.nan
    return (ann_mean - risk_free_rate) / ann_std


def compute_max_drawdown(cum_returns: pd.Series) -> float:
    roll_max = cum_returns.cummax()
    drawdown = (cum_returns / roll_max) - 1.0
    max_dd = drawdown.min()
    return float(max_dd)


def compute_sortino(daily_returns: pd.Series, target: float = 0.0):
    downside = daily_returns.copy()
    downside[downside > target] = 0
    # downside deviation annualized
    dd = np.sqrt((downside ** 2).mean() * 252)
    ann_mean = daily_returns.mean() * 252
    if dd == 0 or np.isnan(dd):
        return np.nan
    return (ann_mean - target) / dd


def volatility_risk(daily_returns: pd.Series) -> float:
    return float(daily_returns.std(ddof=1) * np.sqrt(252))


def run_risk_checks_for_ticker(ticker: str, start_date: str, end_date: str) -> list:
    df = load_price_df_from_db(ticker, start_date, end_date)
    if df is None or df.empty:
        return []
    series = df.set_index("date")["adj_close"].pct_change().dropna()
    if series.empty:
        return []
    cum = (1 + series).cumprod()
    results = []
    vol = volatility_risk(series)
    passed_vol = 1 if vol <= RISK_THRESHOLDS["volatility"] else 0
    results.append(("volatility", vol, RISK_THRESHOLDS["volatility"], passed_vol))
    sharpe = compute_sharpe(series)
    passed_sh = 1 if (not np.isnan(sharpe) and sharpe >= RISK_THRESHOLDS["sharpe"]) else 0
    results.append(("sharpe", None if np.isnan(sharpe) else float(sharpe), RISK_THRESHOLDS["sharpe"], passed_sh))
    mdd = compute_max_drawdown(cum)
    passed_mdd = 1 if mdd >= RISK_THRESHOLDS["max_drawdown"] else 0
    results.append(("max_drawdown", mdd, RISK_THRESHOLDS["max_drawdown"], passed_mdd))
    sortino = compute_sortino(series)
    passed_sort = 1 if (not np.isnan(sortino) and sortino >= RISK_THRESHOLDS["sortino"]) else 0
    results.append(("sortino", None if np.isnan(sortino) else float(sortino), RISK_THRESHOLDS["sortino"], passed_sort))
    # store in DB
    store_risk_checks_bulk(ticker, results)
    return results


# ---------------------------
# Rule setter & stress tests
# ---------------------------

def equal_weight_rule(tickers: list) -> dict:
    n = len(tickers)
    if n == 0:
        return {}
    return {t: 1.0 / n for t in tickers}


def risk_parity_rule(tickers: list, start_date: str, end_date: str) -> dict:
    vols = {}
    for t in tickers:
        df = load_price_df_from_db(t, start_date, end_date)
        if df is None or df.empty:
            vols[t] = np.nan
            continue
        series = df.set_index("date")["adj_close"].pct_change().dropna()
        vols[t] = float(series.std(ddof=1) * np.sqrt(252)) if not series.empty else np.nan
    # invert vols (lower vol -> higher weight)
    inv = {}
    for t in tickers:
        v = vols.get(t)
        if v is None or np.isnan(v) or v == 0:
            inv[t] = 0.0
        else:
            inv[t] = 1.0 / v
    s = sum(inv.values())
    if s == 0:
        return equal_weight_rule(tickers)
    return {t: inv[t] / s for t in tickers}


def stress_test(portfolio_daily: pd.Series, scenario: str):
    if portfolio_daily is None or portfolio_daily.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    if scenario == "bull":
        mod = portfolio_daily * 1.5
    elif scenario == "bear":
        mod = -abs(portfolio_daily)
    elif scenario == "volatile":
        rng = np.random.default_rng(42)
        factor = rng.normal(1.0, 1.0, size=len(portfolio_daily))
        mod = portfolio_daily * factor
    else:
        mod = portfolio_daily
    cum = (1 + mod).cumprod()
    return mod, cum


# ---------------------------
# Predictions (simple linear regression)
# ---------------------------

def simple_linear_predict(series: pd.Series, n_predict: int = 10):
    if series is None or series.empty or len(series.dropna()) < 5:
        return None
    df = series.dropna().reset_index()
    df.columns = ["date", "value"]
    df["x"] = pd.to_datetime(df["date"]).map(lambda d: d.toordinal())
    X = df[["x"]].values
    y = df["value"].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred_train = model.predict(X)
    mse = mean_squared_error(y, y_pred_train)
    r2 = r2_score(y, y_pred_train)
    last_x = int(X[-1][0])
    future_x = np.array([last_x + i for i in range(1, n_predict + 1)]).reshape(-1, 1)
    preds = model.predict(future_x)
    return {"model": model, "preds": preds, "future_x": future_x, "mse": float(mse), "r2": float(r2)}


# ---------------------------
# Email helper
# ---------------------------

def send_email_alert(subject: str, body: str, to_email: str, sender_email: str = None) -> bool:
    import smtplib
    from email.mime.text import MIMEText

    # Determine sender
    sender = sender_email or os.getenv("EMAIL_SENDER") or st.secrets.get("EMAIL_SENDER") if hasattr(st, "secrets") else None
    passwd = os.getenv("EMAIL_PASSWORD") or (st.secrets.get("EMAIL_PASSWORD") if hasattr(st, "secrets") else None)
    if not sender or not passwd:
        st.warning("Email not sent: EMAIL_SENDER or EMAIL_PASSWORD not configured. Set environment variables or Streamlit secrets.")
        return False
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email
    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(sender, passwd)
        server.sendmail(sender, [to_email], msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Crypto Portfolio Investment Manager", layout="wide")
st.title("🚀 Crypto Portfolio Investment Manager")
st.markdown("A Streamlit app for fetching crypto data, computing portfolio metrics, running parallel risk checks, stress tests, and simple predictions.")

init_db()

with st.sidebar:
    st.header("Inputs & Email (for alerts)")
    tickers_text = st.text_input("Tickers (comma separated, Yahoo format)", value="BTC-USD,ETH-USD,SOL-USD")
    start_date = st.date_input("Start date", value=(datetime.now() - timedelta(days=365)).date())
    end_date = st.date_input("End date", value=datetime.now().date())
    email_recipient = st.text_input("Alert recipient email", value=os.getenv("EMAIL_RECIPIENT", "226m1a4203@bvcr.edu.in"))
    sender_default = os.getenv("EMAIL_SENDER", "ankemvamsikrishna@gmail.com")
    sender_input = st.text_input("(Optional) Sender email (will use env var EMAIL_SENDER if empty)", value=sender_default)
    st.markdown("**Email password should be provided as environment variable `EMAIL_PASSWORD` or via Streamlit secrets.**")
    st.markdown("If you do not want alerts, leave password unset.")
    st.markdown("---")
    st.markdown("***Quick actions***")
    fetch_btn = st.button("1) Fetch data & compute metrics (Parallel)")

# ---------------------------
# Fetch & store data (Milestone 1)
# ---------------------------
if fetch_btn:
    tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]
    st.info(f"Fetching {len(tickers)} tickers from yfinance: {tickers}")
    start_s = start_date.isoformat()
    # make end inclusive
    end_s = (end_date + timedelta(days=1)).isoformat()
    progress = st.progress(0)
    completed = 0
    total = len(tickers)
    failures = []
    with ThreadPoolExecutor(max_workers=min(8, total)) as ex:
        futures = {ex.submit(fetch_ticker_data, t, start_s, end_s): t for t in tickers}
        for f in as_completed(futures):
            t = futures[f]
            try:
                df_t = f.result()
                if df_t is None or df_t.empty:
                    st.warning(f"No data for {t}")
                    failures.append(t)
                else:
                    nrows = store_price_df(df_t)
                    metrics = compute_basic_metrics_from_df(df_t)
                    store_metrics_in_db(t, start_s, end_s, metrics)
                    st.success(f"Stored {t}: {nrows} rows. metrics: {metrics}")
            except Exception as e:
                st.error(f"Failed for {t}: {e}")
                failures.append(t)
            completed += 1
            progress.progress(int(completed / total * 100))
    if failures:
        st.warning(f"Completed with failures for: {failures}")
    else:
        st.success("Fetch & store complete for all tickers.")

st.markdown("---")

# ---------------------------
# Portfolio formation (Milestone 2)
# ---------------------------

st.header("Portfolio formation & returns (Milestone 2)")
col1, col2 = st.columns([1, 1])
with col1:
    rule_choice = st.selectbox("Weight rule", options=["Equal Weight", "Risk Parity", "Custom weights (JSON/Python dict)"])
    custom_weights_text = st.text_area("Custom weights (JSON or Python dict). Example: {\"BTC-USD\":0.5, \"ETH-USD\":0.5}", height=80)
    compute_port_btn = st.button("Compute portfolio returns & risk")
with col2:
    st.write("Portfolio period")
    pstart = st.date_input("Portfolio Start", value=start_date)
    pend = st.date_input("Portfolio End", value=end_date)

if compute_port_btn:
    tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]
    if rule_choice == "Custom weights (JSON/Python dict)" and custom_weights_text.strip():
        try:
            # safe parse
            weights = ast.literal_eval(custom_weights_text.strip())
            if not isinstance(weights, dict):
                raise ValueError("Weights must be a dict-like structure.")
        except Exception as e:
            st.error(f"Invalid custom weights: {e}")
            st.stop()
    else:
        if rule_choice == "Equal Weight":
            weights = equal_weight_rule(tickers)
        else:
            # risk parity
            weights = risk_parity_rule(tickers, pstart.isoformat(), (pend + timedelta(days=1)).isoformat())
    if not validate_weights(weights):
        st.error("Weights do not sum to 1.0. Please adjust them.")
    else:
        st.write("Using weights:", weights)
        res = compute_portfolio_returns(tickers, weights, pstart.isoformat(), (pend + timedelta(days=1)).isoformat())
        if not res:
            st.error("Not enough data to compute portfolio for the selected period. Ensure data exists and dates are correct.")
        else:
            store_portfolio_results("portfolio_1", weights, pstart.isoformat(), (pend + timedelta(days=1)).isoformat(), res["annual_return"], res["annual_risk"])
            st.subheader("Portfolio annual return & risk (approx)")
            st.metric("Annual Return (approx)", f"{res['annual_return']:.4f} ({res['annual_return']*100:.2f}%)")
            st.metric("Annual Volatility (approx)", f"{res['annual_risk']:.4f} ({res['annual_risk']*100:.2f}%)")
            st.subheader("Cumulative returns")
            st.line_chart(res["cum_returns"])

st.markdown("---")

# ---------------------------
# Risk checks (Milestone 3)
# ---------------------------

st.header("Risk checks (parallel) & alerts (Milestone 3)")
if st.button("Run risk checks for all tickers (Parallel)"):
    tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]
    failed_alerts = []
    total = len(tickers)
    progress = st.progress(0)
    done = 0
    with ThreadPoolExecutor(max_workers=min(8, total) if total>0 else 1) as ex:
        futures = {ex.submit(run_risk_checks_for_ticker, t, start_date.isoformat(), (end_date + timedelta(days=1)).isoformat()): t for t in tickers}
        for f in as_completed(futures):
            t = futures[f]
            try:
                results = f.result()
                st.write(f"Results for {t}:")
                for name, val, thr, passed in results:
                    st.write(f"- {name}: {val} (threshold {thr}) -> {'PASS' if passed else 'FAIL'}")
                    if not passed:
                        failed_alerts.append((t, name, val, thr))
            except Exception as e:
                st.error(f"Error checking {t}: {e}")
            done += 1
            progress.progress(int(done/total*100) if total>0 else 100)
    if failed_alerts:
        st.warning(f"{len(failed_alerts)} checks failed. Preparing email alert...")
        body = "Risk checks failed:\n\n" + "\n".join([f"{t} - {name}: {val} (thr {thr})" for t, name, val, thr in failed_alerts])
        sender = sender_input.strip() or None
        sent = send_email_alert("Crypto Portfolio Risk Alert", body, email_recipient, sender_email=sender)
        if sent:
            st.success("Email alert sent.")
    else:
        st.success("All checks passed.")

st.markdown("---")

# ---------------------------
# Stress tests & Predictions (Milestone 4)
# ---------------------------

st.header("Stress tests & Linear Regression predictions (Milestone 4)")
colA, colB = st.columns(2)
with colA:
    if st.button("Run stress tests (using last stored portfolio)"):
        # load last portfolio
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("SELECT weights, start_date, end_date FROM portfolio_results WHERE name='portfolio_1' ORDER BY rowid DESC LIMIT 1")
        row = cur.fetchone()
        conn.close()
        if not row:
            st.error("No portfolio found. Compute portfolio first.")
        else:
            weights = json.loads(row[0])
            sdate = row[1]
            edate = row[2]
            tickers = list(weights.keys())
            port = compute_portfolio_returns(tickers, weights, sdate, edate)
            if not port:
                st.error("Not enough data to compute portfolio returns for stress tests.")
            else:
                daily = port["daily_returns"]
                for scenario in ["bull", "bear", "volatile"]:
                    mod, cum = stress_test(daily, scenario)
                    st.subheader(f"Scenario: {scenario}")
                    st.line_chart(cum)
                    if not cum.empty:
                        st.write(f"Final simulated cumulative return: {cum.iloc[-1]:.4f}")

with colB:
    if st.button("Run linear regression predictions for each ticker & portfolio"):
        tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]
        start_s = start_date.isoformat()
        end_s = (end_date + timedelta(days=1)).isoformat()
        for t in tickers:
            df = load_price_df_from_db(t, start_s, end_s)
            if df is None or df.empty:
                st.write(f"{t}: No data")
                continue
            daily = df.set_index("date")["adj_close"].pct_change().dropna()
            out = simple_linear_predict(daily, n_predict=10)
            if out is None:
                st.write(f"{t}: Not enough data for prediction")
            else:
                st.write(f"{t}: MSE={out['mse']:.8f}, R2={out['r2']:.8f}")
                store_prediction_result(t, "linear_returns", out['mse'], out['r2'])
        # portfolio
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("SELECT weights, start_date, end_date FROM portfolio_results WHERE name='portfolio_1' ORDER BY rowid DESC LIMIT 1")
        row = cur.fetchone()
        conn.close()
        if row:
            weights = json.loads(row[0])
            port = compute_portfolio_returns(list(weights.keys()), weights, row[1], row[2])
            if port:
                outp = simple_linear_predict(port['daily_returns'], n_predict=10)
                if outp:
                    st.write(f"Portfolio: MSE={outp['mse']:.8f}, R2={outp['r2']:.8f}")
                    store_prediction_result('portfolio_1', 'linear_returns', outp['mse'], outp['r2'])
                else:
                    st.write("Portfolio: Not enough data for prediction")
        else:
            st.write("No stored portfolio for portfolio prediction")

st.markdown("---")

st.header("Database & Downloads")
if st.button("Show last metrics table"):
    conn = sqlite3.connect(DB_FILE)
    dfm = pd.read_sql_query("SELECT * FROM metrics ORDER BY rowid DESC LIMIT 100", conn)
    conn.close()
    st.dataframe(dfm)

if st.button("Show last risk checks"):
    conn = sqlite3.connect(DB_FILE)
    dfr = pd.read_sql_query("SELECT * FROM risk_checks ORDER BY id DESC LIMIT 200", conn)
    conn.close()
    st.dataframe(dfr)

if st.button("Show last predictions"):
    conn = sqlite3.connect(DB_FILE)
    dfp = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC LIMIT 100", conn)
    conn.close()
    st.dataframe(dfp)

# Download DB file
with open(DB_FILE, "rb") as f:
    db_bytes = f.read()
st.download_button("Download sqlite DB file", db_bytes, file_name=DB_FILE, mime="application/octet-stream")

st.markdown("---")

st.header("Instructions & Notes")
st.markdown(
    """
1. Email alerts: create a Gmail app password and set environment variables:
   - EMAIL_SENDER (optional, defaults to ankemvamsikrishna@gmail.com)
   - EMAIL_PASSWORD (required to send alerts)
   Example (Linux/Mac):
   ```bash
   export EMAIL_SENDER='ankemvamsikrishna@gmail.com'
   export EMAIL_PASSWORD='your_app_password_here'
   ```
   On Windows (PowerShell):
   ```powershell
   setx EMAIL_SENDER "ankemvamsikrishna@gmail.com"
   setx EMAIL_PASSWORD "your_app_password_here"
   ```

2. Run locally:
   ```bash
   streamlit run app.py
   ```

3. The linear regression model is intentionally simple (date ordinal -> returns). For better performance try multivariate features (lagged returns, volumes, technical indicators).

4. Tweak thresholds in the code constant RISK_THRESHOLDS as per your mentor's grading rubric.
"""
)

st.success("App ready. Fetch data first (1) then build portfolio (2), run risk checks (3), and finally stress tests/predictions (4). Good luck, Vamsi! 🚀")
