"""
USDT/KRW 그리드 매매 백테스트 실행 스크립트

사용법:
    python main.py                   # 기본 설정으로 실행
    python main.py --sweep           # 파라미터 스윕 실행
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from src.backtester import Backtester
from src.grid_strategy import GridConfig


# ─── 기본 설정 ────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = dict(
    initial_investment_krw=10_000_000,  # 1천만 원
    grid_count=10,
    grid_range_pct=0.06,               # ±3%
    fee_rate=0.0005,                   # Upbit 0.05%
    use_dynamic_center=True,
)

START_DATE = "2025-06-01"
END_DATE   = "2026-03-31"


# ─── 단일 백테스트 ────────────────────────────────────────────────────────────

def run_single(config_kwargs: dict | None = None, plot: bool = True) -> dict:
    cfg_kwargs = {**DEFAULT_CONFIG, **(config_kwargs or {})}
    config = GridConfig(**cfg_kwargs)

    bt = Backtester(config, start_date=START_DATE, end_date=END_DATE)
    bt.load_data()
    results = bt.run()
    bt.print_summary(results)

    if plot:
        bt.plot_results(results)

    return results


# ─── 파라미터 스윕 ───────────────────────────────────────────────────────────

def run_sweep() -> None:
    import pandas as pd

    grid_counts  = [5, 10, 20]
    range_pcts   = [0.04, 0.06, 0.10]

    rows = []
    total = len(grid_counts) * len(range_pcts)
    idx = 0

    # 데이터 공유를 위해 기준 백테스터로 한 번만 로드
    base_bt = Backtester(GridConfig(**DEFAULT_CONFIG), start_date=START_DATE, end_date=END_DATE)
    base_bt.load_data()

    print(f"\n{'='*62}")
    print(f"  파라미터 스윕: {total}가지 조합")
    print(f"{'='*62}")

    for gc in grid_counts:
        for rp in range_pcts:
            idx += 1
            cfg = GridConfig(
                initial_investment_krw=DEFAULT_CONFIG["initial_investment_krw"],
                grid_count=gc,
                grid_range_pct=rp,
                fee_rate=DEFAULT_CONFIG["fee_rate"],
                use_dynamic_center=DEFAULT_CONFIG["use_dynamic_center"],
            )
            bt = Backtester(cfg, start_date=START_DATE, end_date=END_DATE)
            # 이미 로드된 데이터 재사용
            bt.usdt_df   = base_bt.usdt_df
            bt.usdkrw_df = base_bt.usdkrw_df
            bt.merged_df = base_bt.merged_df

            print(f"[{idx}/{total}] 그리드 {gc:>2}개 / 범위 ±{rp/2*100:.1f}%", end=" ... ")
            results = bt.run()
            print(f"수익률 {results['total_return_pct']:+.2f}%  샤프 {results['sharpe_ratio']:.3f}")

            rows.append({
                "grid_count":      gc,
                "range_pct":       rp,
                "total_return_pct": results["total_return_pct"],
                "sharpe_ratio":    results["sharpe_ratio"],
                "max_drawdown_pct": results["max_drawdown_pct"],
                "n_trades":        results["n_total_trades"],
                "realized_pnl_krw": results["realized_pnl_krw"],
            })

    df = pd.DataFrame(rows).sort_values("total_return_pct", ascending=False)
    print(f"\n{'='*62}")
    print("  스윕 결과 (수익률 내림차순)")
    print(f"{'='*62}")
    print(df.to_string(index=False))

    out = Path("results") / "sweep_results.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n스윕 결과 저장: {out}")

    # 최적 파라미터로 차트 생성
    best = df.iloc[0]
    print(f"\n최적 파라미터: 그리드 {int(best.grid_count)}개 / 범위 ±{best.range_pct/2*100:.1f}%")
    print("최적 파라미터로 차트를 생성합니다...")
    run_single(
        config_kwargs=dict(
            grid_count=int(best.grid_count),
            grid_range_pct=best.range_pct,
        ),
        plot=True,
    )


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="USDT/KRW 그리드 매매 백테스트")
    parser.add_argument("--sweep", action="store_true", help="파라미터 스윕 실행")
    parser.add_argument("--grid-count",  type=int,   default=None, help="그리드 수 (기본: 10)")
    parser.add_argument("--range-pct",   type=float, default=None, help="전체 범위 0~1 (기본: 0.06)")
    parser.add_argument("--investment",  type=float, default=None, help="초기 투자금 KRW (기본: 10,000,000)")
    parser.add_argument("--no-plot",     action="store_true",      help="차트 저장 건너뜀")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.sweep:
        run_sweep()
    else:
        overrides: dict = {}
        if args.grid_count is not None:
            overrides["grid_count"] = args.grid_count
        if args.range_pct is not None:
            overrides["grid_range_pct"] = args.range_pct
        if args.investment is not None:
            overrides["initial_investment_krw"] = args.investment

        run_single(config_kwargs=overrides or None, plot=not args.no_plot)
