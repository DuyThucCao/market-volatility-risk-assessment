"""
Quantitative Risk Assessment of Market Volatility Under Macroeconomic Impacts
==============================================================================
Author  : Thuc Cao  |  duythuccao@outlook.com
GitHub  : https://github.com/DuyThucCao

Pipeline
--------
1. Download adjusted close prices via yfinance (2022–2024)
2. Load and align FOMC / BEA / BLS macro event dates to trading calendar
3. Compute log-returns; run stationarity (ADF, KPSS) diagnostics
4. Compute descriptive statistics, correlation matrix, ACF/PACF summary
5. Run normality tests (Shapiro-Wilk, Jarque-Bera, Anderson-Darling) for all assets
6. Fit ARIMA(1,1,0) + GARCH(1,1)-t per asset and for the equal-weight portfolio
7. Compute dynamic 95% VaR; backtest with Kupiec Proportion-of-Failures (POF) test

Usage
-----
    python market_data.py
"""

import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import kurtosis, skew, shapiro, jarque_bera, anderson, chi2
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TICKERS     = ["SPY", "XLK", "XLF", "XLV", "^GSPC", "^VIX", "^IXIC", "^DJI", "^RUA"]
START_DATE  = "2022-01-01"
END_DATE    = "2024-12-31"

BEA_BLS_CSV = "Dataset/bea_bls_2022_2024.csv"
FOMC_CSV    = "Dataset/fomc_2022_2024_rates.csv"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _lower_cols(df):
    """Normalize all column names to lowercase stripped strings."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _pick_first(df, candidates):
    """Return the first column name from candidates that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_date_series(df, colname):
    """Parse a column to timezone-naive normalized dates."""
    return pd.to_datetime(df[colname], errors="coerce").dt.tz_localize(None).dt.normalize()


def map_to_next_trading_day(dates_like, trading_index):
    """
    Map arbitrary dates (weekends / holidays) to the next available trading day.
    Uses binary search on the sorted trading index for efficiency.
    """
    if dates_like is None or len(dates_like) == 0:
        return pd.Index([])
    dates  = pd.to_datetime(pd.Series(dates_like)).dropna().unique()
    ti     = pd.DatetimeIndex(trading_index)
    mapped = []
    for d in dates:
        pos = ti.searchsorted(d)         # first index position where ti >= d
        if pos < len(ti):
            mapped.append(ti[pos])
    return pd.DatetimeIndex(pd.unique(mapped))


# ─────────────────────────────────────────────────────────────────────────────
# 1. MARKET DATA
# ─────────────────────────────────────────────────────────────────────────────

print("─" * 60)
print("1. Downloading market data from Yahoo Finance …")
raw       = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True)
if raw.empty:
    raise ValueError("No data returned from yfinance. Check tickers or internet connection.")

prices    = raw["Close"].dropna(how="any")
log_prices = np.log(prices)
log_returns = log_prices.diff().dropna()

print(f"   {len(prices)} trading days  |  {prices.shape[1]} assets loaded")


# ─────────────────────────────────────────────────────────────────────────────
# 2. MACRO EVENT CALENDAR
# ─────────────────────────────────────────────────────────────────────────────

print("\n2. Building macro-event calendar …")

calendar = pd.DataFrame(index=prices.index)
calendar["Date"] = calendar.index.normalize()

# --- BEA / BLS (CPI, PCE, GDP) ---
bea_bls_df  = _lower_cols(pd.read_csv(BEA_BLS_CSV, parse_dates=False))
date_col    = _pick_first(bea_bls_df, ["date", "year", "release_date"])
if date_col is None:
    raise ValueError(f"{BEA_BLS_CSV} must contain a date-like column (date, year, or release_date).")

bea_bls_df["Date"] = _to_date_series(bea_bls_df, date_col)
event_col = _pick_first(bea_bls_df, ["event", "description", "note", "title", "indicator"])
if event_col is None:
    bea_bls_df["_event"] = ""
    event_col = "_event"

bls_mask      = bea_bls_df[event_col].str.contains(r"\bCPI\b|consumer price", case=False, na=False)
bea_mask      = bea_bls_df[event_col].str.contains(r"\bPCE\b|personal consumption|GDP", case=False, na=False)
bls_dates_td  = map_to_next_trading_day(bea_bls_df.loc[bls_mask, "Date"].dropna().unique(), calendar.index)
bea_dates_td  = map_to_next_trading_day(bea_bls_df.loc[bea_mask, "Date"].dropna().unique(), calendar.index)

# --- FOMC ---
fomc_df     = _lower_cols(pd.read_csv(FOMC_CSV, parse_dates=False))
date_col_f  = _pick_first(fomc_df, ["date", "meeting_date"])
if date_col_f is None:
    raise ValueError(f"{FOMC_CSV} must contain a 'date' or 'meeting_date' column.")
fomc_df["Date"] = _to_date_series(fomc_df, date_col_f)

rate_col = _pick_first(fomc_df, ["rate_change", "delta_bps", "change_bps", "rate_delta"])
if rate_col:
    rc             = pd.to_numeric(fomc_df[rate_col], errors="coerce").fillna(0)
    fomc_dates_raw = fomc_df.loc[rc != 0, "Date"].dropna().unique()
else:
    fomc_dates_raw = fomc_df["Date"].dropna().unique()

fomc_dates_td = map_to_next_trading_day(fomc_dates_raw, calendar.index)

# Build dummy event columns on the trading calendar
calendar["FOMC"]      = calendar.index.isin(fomc_dates_td).astype(int)
calendar["BLS"]       = calendar.index.isin(bls_dates_td).astype(int)
calendar["BEA"]       = calendar.index.isin(bea_dates_td).astype(int)
calendar["Any_Event"] = calendar[["FOMC", "BLS", "BEA"]].max(axis=1)

Xy = log_returns.join(calendar[["FOMC", "BLS", "BEA", "Any_Event"]].loc[log_returns.index])

print(f"   FOMC events mapped : {calendar['FOMC'].sum()}")
print(f"   BLS (CPI) events   : {calendar['BLS'].sum()}")
print(f"   BEA events         : {calendar['BEA'].sum()}")
print("\n   First 10 days with a macro event:")
print(Xy.loc[calendar["Any_Event"] == 1].head(10))


# ─────────────────────────────────────────────────────────────────────────────
# 3. DESCRIPTIVE STATISTICS & CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n3. Descriptive Statistics …")
desc_stats = pd.DataFrame({
    "Min"      : log_returns.min(),
    "Max"      : log_returns.max(),
    "Median"   : log_returns.median(),
    "Mean"     : log_returns.mean(),
    "Std Dev"  : log_returns.std(),
    "Kurtosis" : log_returns.apply(lambda x: kurtosis(x, fisher=True)),  # excess kurtosis
    "Skewness" : log_returns.apply(skew),
})
print(desc_stats.round(6))

# Correlation matrix
corr_matrix  = log_returns.corr()
benchmarks   = ["^GSPC", "^DJI", "^RUA"]
assets       = [c for c in log_returns.columns if c not in benchmarks]
print("\n   Correlation of assets with market benchmarks:")
print(corr_matrix.loc[assets, benchmarks].round(4))

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm_r", center=0, fmt=".2f")
plt.title("Correlation Matrix of Asset Log Returns (2022–2024)")
plt.tight_layout()
plt.savefig("Plots:Graphs/Correlation Matrix.png", dpi=150)
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 4. STATIONARITY TESTS  (ADF + KPSS)
# ─────────────────────────────────────────────────────────────────────────────

print("\n4. Stationarity Tests (ADF & KPSS) …")

def adf_kpss_report(series, name=""):
    """Print ADF and KPSS p-values for a return series."""
    s = series.dropna()
    _, adf_p, *_ = adfuller(s, autolag="AIC")
    _, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
    adf_result  = "Stationary ✓" if adf_p  < 0.05 else "Non-Stationary ✗"
    kpss_result = "Stationary ✓" if kpss_p > 0.05 else "Non-Stationary ✗"
    print(f"   {name:<8}  ADF: {adf_result} (p={adf_p:.4f})  |  KPSS: {kpss_result} (p={kpss_p:.4f})")

for col in log_returns.columns:
    adf_kpss_report(log_returns[col], name=col)


# ─────────────────────────────────────────────────────────────────────────────
# 5. ACF / PACF SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n5. ACF / PACF Summary …")
acf_pacf_rows = []
for asset in log_returns.columns:
    s = log_returns[asset].dropna()
    acf_pacf_rows.append({
        "Asset"    : asset,
        "ACF(1)"   : round(acf(s,  nlags=1)[1], 5),
        "PACF(1)"  : round(pacf(s, nlags=1)[1], 5),
        "Skewness" : round(desc_stats.loc[asset, "Skewness"], 4),
        "Kurtosis" : round(desc_stats.loc[asset, "Kurtosis"], 4),
    })
print(pd.DataFrame(acf_pacf_rows).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 6. NORMALITY TESTS  (Shapiro-Wilk, Jarque-Bera, Anderson-Darling)
# ─────────────────────────────────────────────────────────────────────────────

print("\n6. Normality Tests …")
normality_rows = []
for col in log_returns.columns:
    x           = log_returns[col].dropna()
    sw_stat, sw_p = shapiro(x)
    jb_stat, jb_p = jarque_bera(x)
    ad_stat     = anderson(x, dist="norm").statistic
    normality_rows.append({
        "Asset"      : col,
        "Shapiro_p"  : round(sw_p, 6),
        "JB_p"       : round(jb_p, 6),
        "AD_stat"    : round(ad_stat, 3),
        "Conclusion" : "Not Normal" if (sw_p < 0.05 or jb_p < 0.05 or ad_stat > 0.787) else "Normal",
    })
print(pd.DataFrame(normality_rows).to_string(index=False))

# Normality plots: Histogram + Q-Q per asset
for col in log_returns.columns:
    x = log_returns[col].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{col} — Normality Check", fontsize=13, fontweight="bold")

    # Histogram with Normal PDF overlay
    sns.histplot(x, bins=50, kde=True, ax=axes[0], color="steelblue")
    xmin, xmax = axes[0].get_xlim()
    x_vals = np.linspace(xmin, xmax, 200)
    pdf    = stats.norm.pdf(x_vals, x.mean(), x.std())
    axes[0].plot(x_vals, pdf * len(x) * (xmax - xmin) / 50,
                 "r--", label="Normal PDF")
    axes[0].set_title("Histogram + Normal PDF")
    axes[0].legend()

    # Q-Q Plot
    stats.probplot(x, dist="norm", plot=axes[1])
    axes[1].set_title("Q–Q Plot")

    plt.tight_layout()
    safe = col.replace("^", "")
    plt.savefig(f"Plots:Graphs/Normality Check {safe}.png", dpi=150)
    plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 7. ARIMA(1,1,0) + GARCH(1,1)-t  PER ASSET
# ─────────────────────────────────────────────────────────────────────────────

print("\n7. Fitting ARIMA(1,1,0) + GARCH(1,1)-t per asset …")
arima_results = {}
garch_results = {}

for col in log_returns.columns:
    print(f"\n   ── {col} ──")

    # ARIMA(1,1,0) on log-prices
    arima_res = ARIMA(log_prices[col].dropna(), order=(1, 1, 0)).fit()
    arima_results[col] = arima_res
    print(arima_res.summary())

    # GARCH(1,1) with Student-t on log-returns (scaled ×100 for numerical stability)
    garch_res = arch_model(
        log_returns[col].dropna() * 100,
        vol="GARCH", p=1, q=1, mean="Constant", dist="t"
    ).fit(disp="off")
    garch_results[col] = garch_res
    print(garch_res.summary())


# ─────────────────────────────────────────────────────────────────────────────
# 8. EQUAL-WEIGHT PORTFOLIO ARIMA + GARCH
# ─────────────────────────────────────────────────────────────────────────────

print("\n8. Equal-weight portfolio ARIMA + GARCH …")

weights           = pd.Series(1 / len(TICKERS), index=log_returns.columns)
portfolio_log_ret = log_returns.dot(weights)

# Portfolio log-price (rebased to 100)
portfolio_log_price = portfolio_log_ret.cumsum() + np.log(100)

arima_port = ARIMA(portfolio_log_price, order=(1, 1, 0)).fit()
print(arima_port.summary())

ret_port   = portfolio_log_ret.dropna() * 100
garch_port = arch_model(
    ret_port, vol="GARCH", p=1, q=1, mean="Constant", dist="t"
).fit(disp="off")
print(garch_port.summary())


# ─────────────────────────────────────────────────────────────────────────────
# 9. DYNAMIC 95% VaR + KUPIEC BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

print("\n9. Computing 95% VaR and running Kupiec POF test …")

alpha      = 0.05
nu         = garch_port.params["nu"]
t_quantile = stats.t.ppf(alpha, df=nu)

# In-sample conditional volatility series
cond_vol       = garch_port.conditional_volatility           # % scale
var_pct_series = garch_port.params["mu"] + cond_vol * t_quantile  # % scale

# Attach to the Xy DataFrame on matching dates
Xy = Xy.copy()
Xy["VaR_95"] = var_pct_series.reindex(Xy.index)

# Violations: actual return < VaR threshold
violations = ret_port[ret_port < Xy["VaR_95"].reindex(ret_port.index)]
print(f"   VaR breaches: {len(violations)} / {len(ret_port)}  "
      f"({len(violations)/len(ret_port)*100:.2f}% observed vs 5.00% expected)")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(ret_port,        color="steelblue", alpha=0.5, linewidth=0.8, label="Portfolio Returns (%)")
plt.plot(Xy["VaR_95"],   color="red",       linestyle="--", linewidth=1.5, label="VaR 95% (GARCH)")
plt.scatter(violations.index, violations.values, color="darkred", s=15, zorder=5, label="VaR Breach")
plt.axhline(0, color="black", linewidth=0.5)
plt.title("Portfolio Returns vs. GARCH Value-at-Risk (95%)  |  2022–2024", fontsize=13, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("Plots:Graphs/VaR 95.png", dpi=150)
plt.show()
plt.close()

# Kupiec Proportion of Failures (POF) Test
def kupiec_test(ret, var, alpha=0.05):
    """
    Kupiec (1995) LR test for VaR model accuracy.

    H₀: Observed breach rate = expected breach rate (alpha).
    Test statistic is chi-square distributed with 1 degree of freedom.
    Fail to reject (p > 0.05) → model is statistically valid.
    """
    n     = len(ret)
    x     = int((ret < var).sum())   # observed breaches
    p     = alpha                    # expected breach rate
    p_hat = x / n                    # observed breach rate

    if x == 0:
        return 0.0, 1.0

    lr = -2 * (
        (n - x) * np.log(1 - p)     + x * np.log(p)
      - (n - x) * np.log(1 - p_hat) - x * np.log(p_hat)
    )
    p_value = 1 - chi2.cdf(lr, df=1)
    return lr, p_value


common_idx = ret_port.index.intersection(Xy["VaR_95"].dropna().index)
lr_stat, p_val = kupiec_test(
    ret_port.loc[common_idx],
    Xy["VaR_95"].loc[common_idx],
    alpha=0.05,
)

print("\n   ═══ KUPIEC POF TEST RESULTS ═══")
print(f"   LR Statistic : {lr_stat:.4f}")
print(f"   P-value      : {p_val:.4f}")
if p_val < 0.05:
    print("   Conclusion   : Reject H₀ — VaR model is mis-specified.")
else:
    print("   Conclusion   : Fail to Reject H₀ — VaR model is statistically valid ✓")

print("\n─" * 60)
print("Pipeline complete.")
