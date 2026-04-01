"""
백테스팅 엔진 및 결과 시각화
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.data_fetcher import fetch_upbit_daily_candles, fetch_usdkrw_rate, merge_usdt_usdkrw
from src.grid_strategy import GridConfig, GridStrategy, Trade

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(__file__).parent.parent / "results"


class Backtester:
    def __init__(
        self,
        config: GridConfig,
        start_date: str = "2025-06-01",
        end_date: str = "2026-03-31",
    ):
        self.config = config
        self.start_date = start_date
        self.end_date = end_date

        self.usdt_df: Optional[pd.DataFrame] = None
        self.usdkrw_df: Optional[pd.DataFrame] = None
        self.merged_df: Optional[pd.DataFrame] = None
        self.strategy: Optional[GridStrategy] = None
        self._portfolio_series: Optional[pd.Series] = None

    # ──────────────────────────────────────────────────── 데이터 로드 ──

    def load_data(self) -> None:
        self.usdt_df = fetch_upbit_daily_candles(
            market="KRW-USDT",
            start_date=self.start_date,
            end_date=self.end_date,
        )
        print(f"  USDT/KRW 일봉 {len(self.usdt_df)}개 로드 완료")

        try:
            self.usdkrw_df = fetch_usdkrw_rate(
                start_date=self.start_date,
                end_date=self.end_date,
            )
            print(f"  USD/KRW 환율 {len(self.usdkrw_df)}개 로드 완료")
            self.merged_df = merge_usdt_usdkrw(self.usdt_df, self.usdkrw_df)
        except Exception as e:
            print(f"  경고: USD/KRW 환율 수집 실패 ({e}) → 김치프리미엄 계산 불가")
            self.usdkrw_df = None
            self.merged_df = self.usdt_df.copy()
            self.merged_df["usdkrw"] = np.nan
            self.merged_df["kimchi_premium"] = np.nan

    # ──────────────────────────────────────────────────── 백테스트 실행 ──

    def run(self) -> dict:
        if self.usdt_df is None:
            self.load_data()

        print(f"\n그리드 백테스트 실행 중...")
        print(f"  기간: {self.start_date} ~ {self.end_date}")
        print(f"  초기투자금: {self.config.initial_investment_krw:,.0f} KRW")
        print(f"  그리드 수: {self.config.grid_count}개 (슬롯 {self.config.n_slots}개)")
        print(f"  그리드 범위: ±{self.config.grid_range_pct / 2 * 100:.1f}%")
        print(f"  재설정 임계: USD/KRW ±{self.config.recenter_threshold_pct * 100:.1f}%")
        print(f"  수수료: {self.config.fee_rate * 100:.3f}%")
        print(f"  김치프리미엄 매수 상한: {self.config.kimchi_buy_max_pct:.1f}%")
        print(f"  김치프리미엄 매도 하한: {self.config.kimchi_sell_min_pct:.1f}%")

        self.strategy = GridStrategy(self.config)
        portfolio_values = []

        for ts, row in self.merged_df.iterrows():
            usdkrw = (
                float(row["usdkrw"])
                if "usdkrw" in row and not np.isnan(row["usdkrw"])
                else None
            )
            kimchi_premium = (
                float(row["kimchi_premium"])
                if "kimchi_premium" in row and not np.isnan(row["kimchi_premium"])
                else None
            )
            self.strategy.process_candle(
                timestamp=ts,
                open_price=float(row["open"]),
                high_price=float(row["high"]),
                low_price=float(row["low"]),
                close_price=float(row["close"]),
                usdkrw_rate=usdkrw,
                kimchi_premium=kimchi_premium,
            )
            portfolio_values.append(self.strategy.portfolio_value(float(row["close"])))

        self._portfolio_series = pd.Series(
            portfolio_values, index=self.merged_df.index, name="portfolio"
        )
        return self._build_results()

    # ──────────────────────────────────────────────────── 결과 집계 ──

    def _build_results(self) -> dict:
        final_price = float(self.merged_df["close"].iloc[-1])
        stats = self.strategy.summary(final_price)

        # 일별 수익률 및 샤프 지수
        daily_ret = self._portfolio_series.pct_change().dropna()
        sharpe = (
            (daily_ret.mean() / daily_ret.std()) * np.sqrt(365)
            if daily_ret.std() > 0
            else 0.0
        )

        # 최대 낙폭
        roll_max = self._portfolio_series.cummax()
        drawdown = (self._portfolio_series - roll_max) / roll_max
        max_dd = float(drawdown.min())

        # 김치프리미엄 통계
        kp = self.merged_df["kimchi_premium"].dropna()
        kimchi_stats: dict = {}
        if not kp.empty:
            kimchi_stats = {
                "kimchi_premium_avg": float(kp.mean()),
                "kimchi_premium_std": float(kp.std()),
                "kimchi_premium_min": float(kp.min()),
                "kimchi_premium_max": float(kp.max()),
            }

        return {
            **stats,
            "sharpe_ratio":     sharpe,
            "max_drawdown_pct": max_dd * 100,
            **kimchi_stats,
        }

    # ──────────────────────────────────────────────────── 콘솔 출력 ──

    def print_summary(self, results: dict) -> None:
        W = 62
        line = "-" * W

        def fmt_row(label: str, value: str) -> str:
            return f"  {label:<34}{value:>22}"

        print(f"\n{'=' * W}")
        print(f"  USDT/KRW 김치프리미엄 그리드 백테스트 결과")
        print(f"  기간: {self.start_date} ~ {self.end_date}")
        print(f"  설정: 그리드 {self.config.grid_count}개 / 범위 ±{self.config.grid_range_pct/2*100:.1f}%")
        print(line)
        print(fmt_row("초기 투자금 (KRW)",         f"{results['initial_investment_krw']:>20,.0f}"))
        print(fmt_row("최종 포트폴리오 가치 (KRW)", f"{results['final_value_krw']:>20,.0f}"))
        print(fmt_row("총 수익률",                  f"{results['total_return_pct']:>19.2f} %"))
        print(fmt_row("실현 손익 (KRW)",            f"{results['realized_pnl_krw']:>20,.0f}"))
        print(fmt_row("총 수수료 (KRW)",            f"{results['total_fees_krw']:>20,.0f}"))
        print(line)
        print(fmt_row("샤프 지수 (연환산)",         f"{results['sharpe_ratio']:>19.4f}"))
        print(fmt_row("최대 낙폭 (MDD)",            f"{results['max_drawdown_pct']:>19.2f} %"))
        print(line)
        print(fmt_row("총 거래 횟수",               f"{results['n_total_trades']:>22d}"))
        print(fmt_row("  매수 체결",                f"{results['n_buy_trades']:>22d}"))
        print(fmt_row("  매도 체결",                f"{results['n_sell_trades']:>22d}"))
        print(fmt_row("그리드 재설정 횟수",         f"{results.get('n_recenter', 0):>22d}"))
        print(line)
        print(fmt_row("최종 KRW 잔고",              f"{results['final_krw_balance']:>20,.0f}"))
        print(fmt_row("최종 USDT 잔고",             f"{results['final_usdt_balance']:>19.4f}"))
        print(fmt_row("USDT 평가액 (KRW)",          f"{results['final_usdt_value_krw']:>20,.0f}"))

        if "kimchi_premium_avg" in results:
            print(line)
            print("  [김치프리미엄 분석]")
            print(fmt_row("  평균",     f"{results['kimchi_premium_avg']:>19.3f} %"))
            print(fmt_row("  표준편차", f"{results['kimchi_premium_std']:>19.3f} %"))
            print(fmt_row("  최솟값",   f"{results['kimchi_premium_min']:>19.3f} %"))
            print(fmt_row("  최댓값",   f"{results['kimchi_premium_max']:>19.3f} %"))

        print(f"{'=' * W}\n")

    # ──────────────────────────────────────────────────── 시각화 ──

    def plot_results(self, results: dict) -> None:
        OUTPUT_DIR.mkdir(exist_ok=True)
        self._plot_main(results)
        self._plot_trades(results)

    def _plot_main(self, results: dict) -> None:
        fig, axes = plt.subplots(4, 1, figsize=(14, 20))
        fig.suptitle(
            f"USDT/KRW 김치프리미엄 그리드 백테스트\n"
            f"{self.start_date} ~ {self.end_date}  |  "
            f"그리드 {self.config.grid_count}개  |  ±{self.config.grid_range_pct/2*100:.1f}%  |  "
            f"재설정 {results.get('n_recenter', 0)}회",
            fontsize=13, fontweight="bold",
        )

        # ── 1. 포트폴리오 가치 & USDT 시세 ───────────────────────────────
        ax1 = axes[0]
        ax1r = ax1.twinx()

        init = self.config.initial_investment_krw
        ax1.plot(
            self._portfolio_series.index,
            self._portfolio_series / 1e6,
            color="#1565C0", lw=2, label="포트폴리오 가치 (M KRW)",
        )
        ax1.axhline(init / 1e6, color="gray", ls="--", alpha=0.7, label="초기 투자금")
        ax1r.plot(
            self.merged_df.index, self.merged_df["close"],
            color="#E65100", lw=1.5, alpha=0.75, label="USDT/KRW 종가",
        )
        if "usdkrw" in self.merged_df.columns:
            ax1r.plot(
                self.merged_df.index, self.merged_df["usdkrw"],
                color="#6A1B9A", lw=1.2, ls="--", alpha=0.8, label="USD/KRW 환율 (공정가치)",
            )

        ax1.set_ylabel("포트폴리오 가치 (백만 KRW)", color="#1565C0")
        ax1r.set_ylabel("USDT/KRW", color="#E65100")
        ax1.set_title("포트폴리오 가치 vs USDT/KRW 시세 vs USD/KRW 환율")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax1r.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # ── 2. 김치프리미엄 ────────────────────────────────────────────────
        ax2 = axes[1]
        kp = self.merged_df["kimchi_premium"].dropna()
        if not kp.empty:
            ax2.plot(kp.index, kp, color="#6A1B9A", lw=1.5, label="김치프리미엄")
            ax2.axhline(0, color="black", ls="-", alpha=0.4)
            ax2.axhline(kp.mean(), color="red", ls="--", alpha=0.7, label=f"평균 {kp.mean():.2f}%")
            # 매수/매도 임계값 표시
            ax2.axhline(
                self.config.kimchi_buy_max_pct,
                color="#E65100", ls=":", lw=1.5,
                label=f"매수 상한 {self.config.kimchi_buy_max_pct:.1f}%",
            )
            ax2.axhline(
                self.config.kimchi_sell_min_pct,
                color="#1565C0", ls=":", lw=1.5,
                label=f"매도 하한 {self.config.kimchi_sell_min_pct:.1f}%",
            )
            ax2.fill_between(kp.index, kp, 0, where=(kp >= 0),
                             color="#4CAF50", alpha=0.3, label="프리미엄(양)")
            ax2.fill_between(kp.index, kp, 0, where=(kp < 0),
                             color="#F44336", alpha=0.3, label="디스카운트(음)")
            ax2.set_ylabel("김치프리미엄 (%)")
            ax2.legend(fontsize=8, ncol=3)
        else:
            ax2.text(0.5, 0.5, "USD/KRW 데이터 없음",
                     transform=ax2.transAxes, ha="center", va="center", color="gray")
        ax2.set_title("USDT 김치프리미엄  (= USDT/KRW ÷ USD/KRW − 1)  및 거래 임계값")
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # ── 3. USD/KRW 환율 추이 & 그리드 재설정 시점 ─────────────────────
        ax3 = axes[2]
        if "usdkrw" in self.merged_df.columns:
            ax3.plot(
                self.merged_df.index, self.merged_df["usdkrw"],
                color="#6A1B9A", lw=1.5, label="USD/KRW 환율",
            )
            # 재설정 발생 시점 표시 (recenter_sell 거래)
            recenter_trades = [
                t for t in self.strategy.trades if t.order_type == "recenter_sell"
            ]
            if recenter_trades:
                r_ts = [t.timestamp for t in recenter_trades]
                r_prices = [
                    float(self.merged_df.loc[t.timestamp, "usdkrw"])
                    if t.timestamp in self.merged_df.index
                    else t.price
                    for t in recenter_trades
                ]
                ax3.scatter(r_ts, r_prices, color="#E65100", marker="x", s=80,
                            zorder=5, label=f"그리드 재설정 ({len(recenter_trades)}회)")
            ax3.set_ylabel("USD/KRW 환율")
            ax3.legend(fontsize=9)
        ax3.set_title("USD/KRW 환율 추이 (그리드 중심 기준)")
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # ── 4. 낙폭 ───────────────────────────────────────────────────────
        ax4 = axes[3]
        roll_max = self._portfolio_series.cummax()
        drawdown = (self._portfolio_series - roll_max) / roll_max * 100
        ax4.fill_between(drawdown.index, drawdown, 0, color="#F44336", alpha=0.6, label="낙폭")
        ax4.set_ylabel("낙폭 (%)")
        ax4.set_title("포트폴리오 낙폭 (Drawdown)")
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        plt.tight_layout()
        out = OUTPUT_DIR / "backtest_results.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"메인 차트 저장: {out}")
        plt.close()

    def _plot_trades(self, results: dict) -> None:
        trades = [t for t in self.strategy.trades if t.order_type in ("buy", "sell")]
        if not trades:
            return

        df_t = pd.DataFrame([{
            "ts":   t.timestamp,
            "type": t.order_type,
            "price": t.price,
            "qty":  t.quantity,
            "pnl":  t.realized_pnl_krw,
        } for t in trades])

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle("거래 내역 분석", fontsize=13, fontweight="bold")

        # ── 1. 체결 위치 ──────────────────────────────────────────────────
        ax1 = axes[0]
        ax1.plot(self.merged_df.index, self.merged_df["close"],
                 color="gray", lw=1, alpha=0.6, label="USDT/KRW")
        if "usdkrw" in self.merged_df.columns:
            ax1.plot(self.merged_df.index, self.merged_df["usdkrw"],
                     color="#6A1B9A", lw=1, ls="--", alpha=0.6, label="USD/KRW")
        for gp in results["grid_prices"]:
            ax1.axhline(gp, color="steelblue", ls=":", lw=0.7, alpha=0.35)

        buys  = df_t[df_t["type"] == "buy"]
        sells = df_t[df_t["type"] == "sell"]
        ax1.scatter(buys["ts"],  buys["price"],  color="#4CAF50", marker="^",
                    s=25, alpha=0.8, zorder=5, label=f"매수 ({len(buys)}회)")
        ax1.scatter(sells["ts"], sells["price"], color="#F44336", marker="v",
                    s=25, alpha=0.8, zorder=5, label=f"매도 ({len(sells)}회)")
        ax1.set_title("체결 위치 및 그리드 레벨")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # ── 2. 누적 실현손익 ──────────────────────────────────────────────
        ax2 = axes[1]
        sell_df = df_t[df_t["type"] == "sell"].copy().sort_values("ts")
        if not sell_df.empty:
            colors = ["#4CAF50" if p >= 0 else "#F44336" for p in sell_df["pnl"]]
            ax2.bar(sell_df["ts"], sell_df["pnl"], color=colors, alpha=0.6,
                    width=pd.Timedelta(hours=18))
            ax2.axhline(0, color="black", lw=0.8)
            ax2r = ax2.twinx()
            ax2r.plot(sell_df["ts"], sell_df["pnl"].cumsum(),
                      color="#1565C0", lw=2, label="누적 실현손익")
            ax2r.set_ylabel("누적 실현손익 (KRW)", color="#1565C0")
            ax2r.legend(loc="upper left", fontsize=9)
            ax2r.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
            )
        ax2.set_ylabel("매도 당 실현손익 (KRW)")
        ax2.set_title("매도 실현손익 및 누적 손익")
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        plt.tight_layout()
        out = OUTPUT_DIR / "trade_analysis.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"거래 분석 차트 저장: {out}")
        plt.close()
