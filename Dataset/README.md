# Dataset

| File | Source | Contents |
|------|--------|----------|
| `market_data.xlsx` | Yahoo Finance (`yfinance`) | Adjusted close prices for SPY, XLK, XLF, XLV, GSPC, DJI, RUA, IXIC, VIX — daily, 2022–2024 |
| `bea_bls_2022_2024.csv` | FRED (BEA / BLS) | CPI, GDP, PCE, and other macroeconomic indicator release dates and values, 2022–2024 |
| `fomc_2022_2024_rates.csv` | Federal Reserve | FOMC meeting dates and federal funds rate target range decisions, 2022–2024 |

All macro event dates are aligned to the nearest subsequent trading day in the analysis pipeline (`market_data.py`), since FOMC and BLS releases often fall on non-trading days.
