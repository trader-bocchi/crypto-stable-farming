"""
Upbit USDT/KRW 시세 및 USD/KRW 환율 데이터 수집 모듈
"""

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import time

DATA_DIR = Path(__file__).parent.parent / "data"


def _parse_date(d) -> datetime:
    if isinstance(d, datetime):
        return d
    if isinstance(d, str):
        return datetime.strptime(d[:10], "%Y-%m-%d")
    if isinstance(d, pd.Timestamp):
        return d.to_pydatetime()
    return d


def fetch_upbit_daily_candles(
    market: str = "KRW-USDT",
    start_date: str = "2025-06-01",
    end_date: str = "2026-03-31",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Upbit 공개 API에서 일봉 OHLCV 데이터를 수집한다.

    Returns:
        DataFrame: index=date(KST), columns=[open, high, low, close, volume]
    """
    DATA_DIR.mkdir(exist_ok=True)
    cache_file = DATA_DIR / f"upbit_{market.replace('-', '_')}_daily.csv"

    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)

    # 캐시 확인
    if use_cache and cache_file.exists():
        df_cache = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df_cache.index = pd.to_datetime(df_cache.index)
        cached_start = df_cache.index.min().to_pydatetime().replace(tzinfo=None)
        cached_end = df_cache.index.max().to_pydatetime().replace(tzinfo=None)
        if cached_start <= start_dt and cached_end >= end_dt:
            mask = (df_cache.index >= pd.Timestamp(start_dt)) & (df_cache.index <= pd.Timestamp(end_dt))
            return df_cache[mask].copy()

    print(f"  Upbit API에서 {market} 일봉 데이터 수집 중...")
    url = "https://api.upbit.com/v1/candles/days"
    headers = {"Accept": "application/json"}

    all_candles: list = []
    current_to = end_dt + timedelta(days=1)

    while True:
        params = {
            "market": market,
            "count": 200,
            "to": current_to.strftime("%Y-%m-%dT00:00:00Z"),
        }
        for attempt in range(3):
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            if resp.status_code == 200:
                break
            if attempt < 2:
                time.sleep(1)
        else:
            print(f"  경고: Upbit API 오류 {resp.status_code}")
            break

        candles = resp.json()
        if not candles:
            break

        all_candles.extend(candles)

        # 가장 오래된 캔들 날짜
        oldest_str = candles[-1]["candle_date_time_kst"]  # KST 기준
        oldest_dt = datetime.strptime(oldest_str[:10], "%Y-%m-%d")

        if oldest_dt <= start_dt:
            break

        current_to = oldest_dt
        time.sleep(0.12)

    if not all_candles:
        raise RuntimeError(f"Upbit에서 {market} 데이터를 가져오지 못했습니다.")

    df = pd.DataFrame(all_candles)
    df["date"] = pd.to_datetime(df["candle_date_time_kst"].str[:10])
    df = df.rename(columns={
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_volume": "volume",
    })
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # 전체 캐시 저장
    df.to_csv(cache_file)

    mask = (df.index >= pd.Timestamp(start_dt)) & (df.index <= pd.Timestamp(end_dt))
    return df[mask].copy()


def fetch_usdkrw_rate(
    start_date: str = "2025-06-01",
    end_date: str = "2026-03-31",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Yahoo Finance에서 USD/KRW 환율 일별 데이터를 수집한다.

    Returns:
        DataFrame: index=date, columns=[open, high, low, close]
    """
    DATA_DIR.mkdir(exist_ok=True)
    cache_file = DATA_DIR / "usdkrw_daily.csv"

    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)

    if use_cache and cache_file.exists():
        df_cache = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df_cache.index = pd.to_datetime(df_cache.index)
        cached_start = df_cache.index.min().to_pydatetime().replace(tzinfo=None)
        cached_end = df_cache.index.max().to_pydatetime().replace(tzinfo=None)
        if cached_start <= start_dt and cached_end >= end_dt:
            mask = (df_cache.index >= pd.Timestamp(start_dt)) & (df_cache.index <= pd.Timestamp(end_dt))
            return df_cache[mask].copy()

    print("  Yahoo Finance에서 USD/KRW 환율 데이터 수집 중...")
    ticker = yf.Ticker("USDKRW=X")
    df = ticker.history(
        start=(start_dt - timedelta(days=5)).strftime("%Y-%m-%d"),
        end=(end_dt + timedelta(days=2)).strftime("%Y-%m-%d"),
        auto_adjust=True,
    )
    if df.empty:
        raise RuntimeError("Yahoo Finance에서 USD/KRW 데이터를 가져오지 못했습니다.")

    df = df[["Open", "High", "Low", "Close"]].rename(columns=str.lower)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    df = df.sort_index()

    df.to_csv(cache_file)

    mask = (df.index >= pd.Timestamp(start_dt)) & (df.index <= pd.Timestamp(end_dt))
    return df[mask].copy()


def merge_usdt_usdkrw(
    usdt_df: pd.DataFrame,
    usdkrw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    USDT/KRW 시세와 USD/KRW 환율을 날짜 기준으로 병합한다.
    주말 등 환율 데이터 누락일은 forward-fill로 보간한다.
    """
    df = usdt_df.copy()
    usdkrw_renamed = usdkrw_df[["close"]].rename(columns={"close": "usdkrw"})
    df = df.join(usdkrw_renamed, how="left")
    df["usdkrw"] = df["usdkrw"].ffill().bfill()

    # 김치프리미엄 계산 (USDT 시세 / USD/KRW 환율 - 1)
    df["kimchi_premium"] = (df["close"] - df["usdkrw"]) / df["usdkrw"] * 100
    return df
