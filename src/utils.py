import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from scipy.stats import skew, kurtosis

warnings.filterwarnings('ignore')

SECTOR_MAPPING = {
    "NIFTY 50": "^NSEI", "NIFTY BANK": "^NSEBANK", "NIFTY IT": "^CNXIT",
    "NIFTY AUTO": "^CNXAUTO", "NIFTY METAL": "^CNXMETAL", "NIFTY FMCG": "^CNXFMCG",
    "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "GOLD": "GC=F", "SILVER": "SI=F",
    "BITCOIN": "BTC-USD", "RELIANCE": "RELIANCE.NS", "HDFC BANK": "HDFCBANK.NS"
}

def generate_insights(metrics, benchmark):
    """Generates professional managerial insights (Clean Text)."""
    best_asset = metrics['Sharpe Ratio'].idxmax()
    worst_drawdown = metrics['Max Drawdown'].min()
    highest_beta = metrics['Beta'].idxmax()

    insight_text = (
        f"EXECUTIVE SUMMARY: Analysis of {len(metrics)} assets against {benchmark} reveals divergent risk profiles. "
        f"{best_asset} currently offers the superior risk-adjusted return (Sharpe: {metrics.loc[best_asset, 'Sharpe Ratio']:.2f}), "
        f"indicating efficient capital deployment. Conversely, beware of tail risks in assets like {metrics['Max Drawdown'].idxmin()}, "
        f"which shows a historical drawdown of {worst_drawdown:.1%}.\n\n"
        f"MANAGERIAL IMPLICATION: For risk-averse portfolios, consider underweighting {highest_beta} (Beta: {metrics.loc[highest_beta, 'Beta']:.2f}) "
        f"due to its high market sensitivity. Reallocation towards {best_asset} is recommended for optimizing the efficient frontier."
    )
    return insight_text

def resample_data(df, freq='D'):
    if freq == 'D': return df
    return df.resample(freq).last().dropna()

def generate_simulation(tickers, start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    T = len(dates); N = len(tickers)
    mu = 0.0006; sigma = 0.015
    np.random.seed(42)
    returns = np.random.normal(loc=mu, scale=sigma, size=(T, N))
    price_paths = 100 * (1 + returns).cumprod(axis=0)
    return pd.DataFrame(price_paths, index=dates, columns=tickers)

def get_prices(tickers, start, end):
    try:
        data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, timeout=5)
        if isinstance(data.columns, pd.MultiIndex): data = data['Close']
        if data.empty or data.isna().all().all(): raise ValueError("Empty data")
        return data.ffill().bfill(), "live"
    except:
        return generate_simulation(tickers, start, end), "synthetic"

def compute_metrics(prices, market_ticker="^NSEI", rf_rate=0.0, var_conf=0.95):
    rets = prices.pct_change()
    mkt_col = market_ticker if market_ticker in prices.columns else prices.columns[0]
    metrics = []
    for col in prices.columns:
        r = rets[col].dropna()
        if r.empty: continue
        ann_ret = (1 + r.mean())**252 - 1
        ann_vol = r.std() * (252**0.5)
        sharpe = ((ann_ret - rf_rate) / ann_vol) if ann_vol != 0 else 0
        downside = r[r < 0]
        sortino = ((ann_ret - rf_rate) / (downside.std() * (252**0.5))) if len(downside) > 0 else 0
        common = pd.concat([r, rets[mkt_col]], axis=1).dropna()
        beta = common.iloc[:,0].cov(common.iloc[:,1]) / common.iloc[:,1].var() if len(common) > 30 and common.iloc[:,1].var() != 0 else 1.0
        cum = (1+r).cumprod()
        dd = ((cum / cum.expanding().max()) - 1).min()
        var = r.quantile(1 - var_conf)
        metrics.append([col, ann_ret, ann_vol, sharpe, sortino, beta, dd, var, skew(r), kurtosis(r)])
    cols = ['Ticker','Ann Return','Ann Volatility','Sharpe Ratio','Sortino Ratio','Beta','Max Drawdown', f'VaR ({var_conf:.0%})', 'Skew', 'Kurtosis']
    return pd.DataFrame(metrics, columns=cols).set_index('Ticker')

def simulate_portfolio(prices, num_portfolios=3000):
    rets = prices.pct_change().dropna()
    mean_rets = rets.mean() * 252
    cov_matrix = rets.cov() * 252
    num_assets = len(prices.columns)
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(num_assets); weights /= np.sum(weights)
        results[0,i] = np.sum(mean_rets * weights)
        results[1,i] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results[2,i] = results[0,i] / results[1,i]
    return pd.DataFrame(results.T, columns=['Return','Volatility','Sharpe'])

def get_ticker_choices(): return SECTOR_MAPPING.copy()
