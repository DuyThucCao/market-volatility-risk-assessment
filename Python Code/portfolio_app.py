"""
Quantitative Risk & Macro Analysis Dashboard
=============================================
Author  : Thuc Cao  |  duythuccao@outlook.com
GitHub  : https://github.com/DuyThucCao

Interactive Streamlit dashboard for quantitative risk assessment.
Loads real market data and macro event CSVs; fits ARIMA/GARCH models
on demand; computes dynamic VaR with Kupiec backtesting.

Usage
-----
    pip install streamlit yfinance statsmodels arch scipy pandas matplotlib seaborn
    streamlit run portfolio_app.py

The app expects these files relative to its working directory:
    Dataset/bea_bls_2022_2024.csv
    Dataset/fomc_2022_2024_rates.csv
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import yfinance as yf
from scipy.stats import kurtosis, skew, shapiro, jarque_bera, t, chi2
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Quant Risk Portfolio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main-header { font-size: 2.2rem; color: #1E3A8A; font-weight: 800; }
  .sub-header  { font-size: 1.4rem; color: #1E3A8A; font-weight: 600; }
  .highlight   { background-color: #F3F4F6; padding: 10px; border-radius: 5px;
                 border-left: 5px solid #3B82F6; }
  [data-testid="stMetric"] { background-color: #ffffff; padding: 15px;
                              border-radius: 10px;
                              box-shadow: 2px 2px 10px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📈 Quantitative Risk & Macro Analysis Dashboard</div>',
            unsafe_allow_html=True)
st.markdown("ARIMA/GARCH volatility modeling · Dynamic 95% VaR · Kupiec backtesting · FOMC/CPI event mapping")
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _lower_cols(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _pick_first(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_date_series(df, colname):
    return pd.to_datetime(df[colname], errors="coerce").dt.tz_localize(None).dt.normalize()


def map_to_next_trading_day(dates_like, trading_index):
    """Shift non-trading dates forward to the next available trading day."""
    if dates_like is None or len(dates_like) == 0:
        return pd.Index([])
    dates  = pd.to_datetime(pd.Series(dates_like)).dropna().unique()
    ti     = pd.DatetimeIndex(trading_index)
    mapped = [ti[pos] for d in dates if (pos := ti.searchsorted(d)) < len(ti)]
    return pd.DatetimeIndex(pd.unique(mapped))


def kupiec_test(total, violated, confidence):
    """
    Kupiec (1995) Proportion of Failures test.

    H₀: Observed breach rate = (1 - confidence).
    Returns (LR statistic, p-value).  p > 0.05 → model is valid.
    """
    p_target = 1 - confidence
    p_actual = violated / total if total > 0 else 0
    if violated == 0:
        return 0.0, 1.0
    lr = -2 * (
        (total - violated) * np.log(1 - p_target) + violated * np.log(p_target)
      - (total - violated) * np.log(1 - p_actual) - violated * np.log(p_actual)
    )
    return lr, float(1 - chi2.cdf(lr, df=1))


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (cached for performance)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Downloading market data …")
def fetch_market_data(tickers, start, end):
    """Download adjusted close prices from Yahoo Finance."""
    data = yf.download(list(tickers), start=str(start), end=str(end),
                       auto_adjust=True, progress=False)
    if data.empty:
        return None
    prices = (data["Close"] if not isinstance(data.columns, pd.MultiIndex)
              else data.xs("Close", level="Price", axis=1))
    return prices.dropna(how="any")


@st.cache_data(show_spinner="Loading macro events …")
def load_macro_events():
    """
    Load FOMC and BEA/BLS event dates from the Dataset/ CSVs.
    Falls back to embedded sample data if the files are not found.
    """
    try:
        bea_bls = _lower_cols(pd.read_csv("Dataset/bea_bls_2022_2024.csv"))
        fomc    = _lower_cols(pd.read_csv("Dataset/fomc_2022_2024_rates.csv"))
        return bea_bls, fomc, "csv"
    except FileNotFoundError:
        # Embedded fallback — covers the 2022-2024 FOMC cycle
        fomc_csv = """date,rate_change
2022-01-26,0
2022-03-16,25
2022-05-04,50
2022-06-15,75
2022-07-27,75
2022-09-21,75
2022-11-02,75
2022-12-14,50
2023-02-01,25
2023-03-22,25
2023-05-03,25
2023-06-14,0
2023-07-26,25
2023-09-20,0
2023-11-01,0
2023-12-13,0
2024-01-31,0
2024-03-20,0
2024-09-18,-50
2024-11-07,-25
2024-12-18,-25"""
        bea_bls_csv = """date,indicator,value,source
2022-01-12,CPI All Items,7.00%,BLS
2022-02-10,CPI All Items,7.50%,BLS
2022-03-10,CPI All Items,7.90%,BLS
2022-04-12,CPI All Items,8.50%,BLS
2022-05-11,CPI All Items,8.30%,BLS
2022-06-10,CPI All Items,8.60%,BLS
2022-07-13,CPI All Items,9.10%,BLS
2022-08-10,CPI All Items,8.50%,BLS
2022-09-13,CPI All Items,8.30%,BLS
2022-10-13,CPI All Items,8.20%,BLS
2022-11-10,CPI All Items,7.70%,BLS
2022-12-13,CPI All Items,7.10%,BLS
2023-01-12,CPI All Items,6.50%,BLS
2023-02-14,CPI All Items,6.40%,BLS
2023-03-14,CPI All Items,6.00%,BLS
2023-04-12,CPI All Items,5.00%,BLS
2023-05-10,CPI All Items,4.90%,BLS
2023-06-13,CPI All Items,4.00%,BLS
2023-07-12,CPI All Items,3.20%,BLS
2023-08-10,CPI All Items,3.20%,BLS
2023-09-13,CPI All Items,3.70%,BLS
2023-10-12,CPI All Items,3.70%,BLS
2023-11-14,CPI All Items,3.20%,BLS
2023-12-12,CPI All Items,3.40%,BLS
2024-01-11,CPI All Items,3.40%,BLS
2024-02-13,CPI All Items,3.10%,BLS
2024-03-12,CPI All Items,3.20%,BLS"""
        return (pd.read_csv(io.StringIO(bea_bls_csv)),
                pd.read_csv(io.StringIO(fomc_csv)),
                "fallback")


@st.cache_data(show_spinner="Building event calendar …")
def process_events(_prices_index, bea_bls_df, fomc_df):
    """Map macro events to trading days and build a binary event flag table."""
    calendar      = pd.DataFrame(index=_prices_index)
    calendar["Date"] = calendar.index.normalize()
    trading_index = calendar.index

    bea_bls = _lower_cols(bea_bls_df.copy())
    date_col = _pick_first(bea_bls, ["date", "year", "release_date"])
    if date_col:
        bea_bls["Date"] = _to_date_series(bea_bls, date_col)
    event_col = _pick_first(bea_bls, ["event", "indicator", "description", "note", "title"])
    if event_col is None:
        bea_bls["_event"] = ""
        event_col = "_event"

    bls_mask     = bea_bls[event_col].str.contains(r"\bCPI\b|consumer price", case=False, na=False)
    bea_mask     = bea_bls[event_col].str.contains(r"\bPCE\b|personal consumption|GDP", case=False, na=False)
    bls_dates_td = map_to_next_trading_day(bea_bls.loc[bls_mask, "Date"].dropna().unique(), trading_index)
    bea_dates_td = map_to_next_trading_day(bea_bls.loc[bea_mask, "Date"].dropna().unique(), trading_index)

    fomc = _lower_cols(fomc_df.copy())
    date_col_f = _pick_first(fomc, ["date", "meeting_date"])
    if date_col_f:
        fomc["Date"] = _to_date_series(fomc, date_col_f)
    rate_col = _pick_first(fomc, ["rate_change", "delta_bps", "change_bps", "rate_delta"])
    fomc_raw = (fomc.loc[pd.to_numeric(fomc[rate_col], errors="coerce").fillna(0) != 0, "Date"].dropna().unique()
                if rate_col else fomc["Date"].dropna().unique())
    fomc_dates_td = map_to_next_trading_day(fomc_raw, trading_index)

    calendar["FOMC"]      = calendar.index.isin(fomc_dates_td).astype(int)
    calendar["BLS"]       = calendar.index.isin(bls_dates_td).astype(int)
    calendar["BEA"]       = calendar.index.isin(bea_dates_td).astype(int)
    calendar["Any_Event"] = calendar[["FOMC", "BLS", "BEA"]].max(axis=1)
    return calendar[["FOMC", "BLS", "BEA", "Any_Event"]]


@st.cache_data
def calculate_returns(prices):
    log_prices  = np.log(prices)
    log_returns = log_prices.diff().dropna()
    return log_prices, log_returns


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TICKERS = ["SPY", "XLK", "XLF", "XLV", "^GSPC", "^VIX", "^IXIC", "^RUA"]

st.sidebar.header("⚙️ Configuration")
tickers       = st.sidebar.multiselect("Assets & Benchmarks", DEFAULT_TICKERS, default=DEFAULT_TICKERS)
start_date    = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date      = st.sidebar.date_input("End Date",   pd.to_datetime("2024-12-31"))
st.sidebar.markdown("---")
st.sidebar.subheader("Model Parameters")
var_confidence = st.sidebar.slider("VaR Confidence Level (%)", 90, 99, 95) / 100.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — DATA LOAD
# ─────────────────────────────────────────────────────────────────────────────

prices = log_prices = log_returns = Xy = None
data_loaded = False

try:
    with st.spinner("Loading data …"):
        if not tickers:
            st.warning("Select at least one ticker from the sidebar.")
        else:
            prices = fetch_market_data(tuple(tickers), start_date, end_date)
            if prices is None or prices.empty:
                st.error("No market data returned. Check tickers or internet connection.")
            else:
                bea_bls_df, fomc_df, source = load_macro_events()
                if source == "fallback":
                    st.info("ℹ️  Dataset/ CSVs not found — using embedded macro event data.")
                event_flags               = process_events(prices.index, bea_bls_df, fomc_df)
                log_prices, log_returns   = calculate_returns(prices)
                Xy                        = log_returns.join(event_flags.loc[log_returns.index])
                data_loaded               = True
except Exception as e:
    st.error(f"Data loading error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

if data_loaded:
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data & Events",
        "📈 Statistical Analysis",
        "🤖 ARIMA / GARCH Modeling",
        "🛡️ Portfolio VaR",
    ])

    # ── TAB 1: Data & Event Mapping ─────────────────────────────────────────
    with tab1:
        st.markdown('<div class="sub-header">Market Data & Macro Event Mapping</div>',
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Adjusted Close Prices")
            st.dataframe(prices.tail(10))
        with col2:
            st.subheader("Event Flags (days with macro releases)")
            event_days = event_flags[event_flags["Any_Event"] == 1]
            st.dataframe(event_days.head(15))
        st.write(f"**{len(prices)} trading days**  |  "
                 f"**{len(event_days)} days with macro events**  "
                 f"({event_flags['FOMC'].sum()} FOMC · "
                 f"{event_flags['BLS'].sum()} BLS/CPI · "
                 f"{event_flags['BEA'].sum()} BEA)")

    # ── TAB 2: Statistical Analysis ─────────────────────────────────────────
    with tab2:
        st.markdown('<div class="sub-header">Exploratory Data Analysis</div>',
                    unsafe_allow_html=True)

        st.subheader("Descriptive Statistics (Log Returns)")
        desc = pd.DataFrame({
            "Mean"          : log_returns.mean(),
            "Std Dev"       : log_returns.std(),
            "Skewness"      : log_returns.apply(lambda x: skew(x)),
            "Excess Kurtosis": log_returns.apply(lambda x: kurtosis(x, fisher=True)),
        }).T
        st.dataframe(desc.style.format("{:.5f}").background_gradient(cmap="coolwarm", axis=1))

        st.subheader("Correlation Matrix")
        fig_c, ax_c = plt.subplots(figsize=(10, 6))
        sns.heatmap(log_returns.corr(), annot=True, cmap="coolwarm_r",
                    center=0, fmt=".2f", ax=ax_c)
        st.pyplot(fig_c)
        plt.close(fig_c)

        st.subheader("Normality Tests (Shapiro-Wilk & Jarque-Bera)")
        norm_rows = []
        for col in log_returns.columns:
            x = log_returns[col].dropna()
            sw_p = shapiro(x)[1]
            jb_p = jarque_bera(x)[1]
            norm_rows.append({
                "Asset"          : col,
                "Shapiro-Wilk p" : round(sw_p, 5),
                "Jarque-Bera p"  : round(jb_p, 5),
                "Conclusion"     : "Not Normal ✗" if (sw_p < 0.05 or jb_p < 0.05) else "Normal ✓",
            })
        st.dataframe(pd.DataFrame(norm_rows).set_index("Asset"))
        st.info("**Interpretation:** Rejection of normality (p < 0.05) justifies using a "
                "Student-t distribution in GARCH to capture fat tails.")

    # ── TAB 3: ARIMA / GARCH Modeling ───────────────────────────────────────
    with tab3:
        st.markdown('<div class="sub-header">ARIMA(1,1,0) + GARCH(1,1) Modeling</div>',
                    unsafe_allow_html=True)
        selected = st.selectbox("Select asset to model:", log_returns.columns)

        if st.button(f"▶ Fit models for {selected}"):
            ret_s   = log_returns[selected].dropna()
            price_s = log_prices[selected].dropna()

            col_a, col_g = st.columns(2)

            with col_a:
                st.subheader(f"ARIMA(1,1,0) — {selected}")
                with st.spinner("Fitting ARIMA …"):
                    try:
                        arima_res = ARIMA(price_s, order=(1, 1, 0)).fit()
                        st.text(arima_res.summary().as_text())
                    except Exception as e:
                        st.error(f"ARIMA failed: {e}")

            with col_g:
                st.subheader(f"GARCH(1,1)-t — {selected}")
                with st.spinner("Fitting GARCH …"):
                    try:
                        garch_res = arch_model(
                            ret_s * 100, vol="GARCH", p=1, q=1,
                            mean="Constant", dist="t"
                        ).fit(disp="off")
                        st.text(garch_res.summary().as_text())

                        fig_v, ax_v = plt.subplots(figsize=(10, 3))
                        ax_v.plot(garch_res.conditional_volatility,
                                  color="crimson", linewidth=0.8,
                                  label="Conditional Volatility (%)")
                        ax_v.set_title(f"{selected} — GARCH Conditional Volatility")
                        ax_v.legend()
                        st.pyplot(fig_v)
                        plt.close(fig_v)
                    except Exception as e:
                        st.error(f"GARCH failed: {e}")

    # ── TAB 4: Portfolio VaR ────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="sub-header">Portfolio Value-at-Risk (VaR) & Backtesting</div>',
                    unsafe_allow_html=True)

        benchmarks       = {"^GSPC", "^VIX", "^IXIC", "^DJI", "^RUA"}
        portfolio_assets = [c for c in log_returns.columns if c not in benchmarks]

        if not portfolio_assets:
            st.error("No investable assets selected. Add ETFs like SPY, XLK, XLF, XLV.")
        else:
            st.write(f"**Equal-weight portfolio:** {', '.join(portfolio_assets)}")
            w             = np.full(len(portfolio_assets), 1 / len(portfolio_assets))
            port_log_ret  = log_returns[portfolio_assets].dot(w)

            @st.cache_resource
            def fit_portfolio_garch(ret_key):
                am = arch_model(port_log_ret * 100, vol="GARCH", p=1, q=1,
                                mean="Constant", dist="t")
                return am.fit(disp="off")

            with st.spinner("Fitting Portfolio GARCH …"):
                try:
                    garch_res   = fit_portfolio_garch(tuple(portfolio_assets))
                    nu_p        = garch_res.params["nu"]
                    fitted_mu   = garch_res.params["mu"] / 100
                    fitted_std  = garch_res.conditional_volatility / 100
                    t_q         = t.ppf(1 - var_confidence, df=nu_p)
                    var_series  = fitted_mu + fitted_std * t_q

                    port_df     = pd.DataFrame({
                        "Returns": port_log_ret,
                        "VaR"    : var_series,
                    })
                    violations  = port_df[port_df["Returns"] < port_df["VaR"]]

                    # Chart
                    st.subheader(f"Returns vs. {var_confidence*100:.0f}% GARCH VaR")
                    fig_v, ax_v = plt.subplots(figsize=(12, 5))
                    ax_v.plot(port_df.index, port_df["Returns"] * 100,
                              color="steelblue", alpha=0.5, linewidth=0.8,
                              label="Portfolio Return (%)")
                    ax_v.plot(port_df.index, port_df["VaR"] * 100,
                              color="red", linestyle="--", linewidth=1.5,
                              label=f"VaR {var_confidence*100:.0f}% (GARCH)")
                    ax_v.scatter(violations.index, violations["Returns"] * 100,
                                 color="darkred", s=20, zorder=5, label="Breach")
                    ax_v.axhline(0, color="black", linewidth=0.4)
                    ax_v.set_ylabel("Daily Return (%)")
                    ax_v.legend()
                    st.pyplot(fig_v)
                    plt.close(fig_v)

                    # Kupiec test
                    st.subheader("Backtesting — Kupiec POF Test")
                    n_obs   = len(port_df.dropna())
                    n_viol  = len(violations)
                    exp_viol = n_obs * (1 - var_confidence)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Observations",     f"{n_obs}")
                    m2.metric("Breaches",         f"{n_viol}",
                               delta=f"{n_viol - exp_viol:+.1f} vs expected")
                    m3.metric("Breach Rate",      f"{n_viol/n_obs*100:.2f}%",
                               delta=f"Expected: {(1-var_confidence)*100:.1f}%")

                    lr_stat, p_val = kupiec_test(n_obs, n_viol, var_confidence)
                    st.metric("LR Statistic", f"{lr_stat:.4f}")
                    if p_val >= 0.05:
                        st.success(f"✅ Test Passed  (p = {p_val:.4f})  — VaR model is statistically valid.")
                    else:
                        st.error(f"❌ Test Failed  (p = {p_val:.4f})  — VaR model is mis-specified.")

                    with st.expander("ℹ️ About the Kupiec Test"):
                        st.markdown("""
**Kupiec Proportion of Failures (POF) Test** — Kupiec (1995)

Tests whether the observed breach rate is statistically consistent
with the model's target confidence level.

- **H₀:** Observed breach rate = (1 − confidence)
- **Test statistic:** Likelihood ratio, χ²(1) distributed
- **p > 0.05** → Fail to reject H₀ → Model is valid ✅
- **p < 0.05** → Reject H₀ → Model over- or under-estimates risk ❌
                        """)

                except Exception as e:
                    st.error(f"Portfolio risk error: {e}")

else:
    st.info("⏳ Waiting for data — select tickers and check your connection.")
