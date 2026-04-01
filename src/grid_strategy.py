"""
USDT/KRW 그리드 매매 전략 - 김치프리미엄 신호 + 동적 그리드 재설정

핵심 원리:
  USDT의 '공정가치'는 USD/KRW 환율이다.
  - USDT/KRW > USD/KRW (김치프리미엄 양수): USDT 고평가 → 매도 기회
  - USDT/KRW < USD/KRW (김치프리미엄 음수): USDT 저평가 → 매수 기회
  → 그리드를 USD/KRW 환율 중심으로 설정하면 자연스럽게 저평가 시 매수,
    고평가 시 매도가 이루어지며 프리미엄 수렴 시 차익 실현

전략 구조:
  - 그리드 중심: USD/KRW 환율 (공정가치 기준)
  - n_levels개의 가격 레벨 → (n_levels - 1)개의 독립 슬롯
  - 슬롯 i: [grid_prices[i], grid_prices[i+1]] 구간 담당
      · 'empty'  → grid_prices[i]   에서 USDT 매수 대기
      · 'filled' → grid_prices[i+1] 에서 USDT 매도 대기
  - 슬롯 간 완전 독립 → 중복 주문 / 레벨 드리프트 구조적으로 불가
  - USD/KRW가 recenter_threshold 이상 변동 시 그리드 자동 재설정
  - 김치프리미엄 임계값: 극단 구간에서 해당 방향 거래 일시 중단
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import pandas as pd


# ─── 데이터 클래스 ────────────────────────────────────────────────────────────

@dataclass
class GridConfig:
    """그리드 전략 설정"""
    initial_investment_krw: float           # 초기 투자금 (KRW)
    grid_count: int = 10                    # 레벨 수  (슬롯 수 = grid_count - 1)
    grid_range_pct: float = 0.06            # 전체 범위 (0.06 → 중심가 ±3%)
    fee_rate: float = 0.0005                # 거래 수수료 (Upbit: 0.05%)
    recenter_threshold_pct: float = 0.03   # USD/KRW 이 비율 이상 변동 시 그리드 재설정
    kimchi_buy_max_pct: float = 2.0         # 김치프리미엄 ≤ 이 값일 때만 매수 허용
    kimchi_sell_min_pct: float = -0.5       # 김치프리미엄 ≥ 이 값일 때만 매도 허용

    @property
    def n_slots(self) -> int:
        return self.grid_count - 1


@dataclass
class Trade:
    timestamp: pd.Timestamp
    order_type: str        # 'init_buy' | 'buy' | 'sell' | 'recenter_sell'
    price: float
    quantity: float        # USDT 수량
    fee_krw: float
    realized_pnl_krw: float = 0.0


# ─── 핵심 전략 클래스 ─────────────────────────────────────────────────────────

class GridStrategy:
    """
    슬롯 기반 그리드 전략.

    각 슬롯 i는 독립적으로 [grid_prices[i], grid_prices[i+1]] 구간을 관리한다.
    슬롯 간 간섭이 없으므로 기존 레벨 인덱스 방식의 중복 주문·드리프트 문제가
    구조적으로 발생하지 않는다.
    """

    def __init__(self, config: GridConfig):
        self.config = config
        self.krw_balance: float = config.initial_investment_krw
        self.usdt_balance: float = 0.0
        self.trades: List[Trade] = []
        self.grid_prices: List[float] = []
        self.center_price: float = 0.0
        self.initialized: bool = False
        self.recenter_count: int = 0

        # 슬롯 상태
        self._slot_states: Dict[int, Literal["empty", "filled"]] = {}
        self._slot_buy_price: Dict[int, float] = {}   # 취득단가 (P&L 계산용)
        self._slot_qty: Dict[int, float] = {}          # USDT 수량
        self._investment_per_slot: float = 0.0
        self._usdkrw_center: float = 0.0              # 마지막 재설정 시점의 USD/KRW

    # ──────────────────────────────────────────────────────────── 초기화 ──

    def initialize(self, current_price: float, usdkrw_rate: Optional[float] = None) -> None:
        """그리드 레벨 초기화 및 슬롯 상태 배치"""
        cfg = self.config
        self._usdkrw_center = usdkrw_rate if usdkrw_rate else current_price
        self.center_price = self._usdkrw_center

        n = cfg.grid_count
        r = cfg.grid_range_pct
        step = r / (n - 1) if n > 1 else 0
        self.grid_prices = [
            round(self.center_price * (1 - r / 2 + step * i), 4)
            for i in range(n)
        ]

        # 가용 KRW를 슬롯 수로 균등 배분
        n_slots = cfg.n_slots
        self._investment_per_slot = self.krw_balance / n_slots

        self._slot_states.clear()
        self._slot_buy_price.clear()
        self._slot_qty.clear()

        for i in range(n_slots):
            lower = self.grid_prices[i]
            upper = self.grid_prices[i + 1]

            if current_price >= upper:
                # 현재가가 이미 슬롯 상단을 넘은 상태
                # → 현재가에 초기 매수 후 상단에서 매도 대기
                qty = self._investment_per_slot / current_price
                cost = qty * current_price * (1 + cfg.fee_rate)
                if cost <= self.krw_balance:
                    self.krw_balance -= cost
                    self.usdt_balance += qty
                    self._slot_states[i] = "filled"
                    self._slot_buy_price[i] = current_price
                    self._slot_qty[i] = qty
                    self.trades.append(Trade(
                        timestamp=pd.Timestamp("1970-01-01"),
                        order_type="init_buy",
                        price=current_price,
                        quantity=qty,
                        fee_krw=qty * current_price * cfg.fee_rate,
                    ))
                else:
                    # KRW 부족 시 빈 슬롯으로 처리
                    self._slot_states[i] = "empty"
                    self._slot_qty[i] = self._investment_per_slot / lower
            else:
                # current_price ≤ lower  또는  구간 내부
                # → 하단에서 매수 대기
                self._slot_states[i] = "empty"
                self._slot_qty[i] = self._investment_per_slot / lower

        self.initialized = True

    # ──────────────────────────────────────────────────── 그리드 재설정 ──

    def _recenter(
        self,
        timestamp: pd.Timestamp,
        current_price: float,
        usdkrw_rate: float,
    ) -> None:
        """
        USD/KRW 환율이 임계값 이상 변동 시 호출.
        보유 USDT를 현재가에 전량 청산하고, 새로운 USD/KRW 기준으로 그리드 재초기화.
        """
        cfg = self.config
        if self.usdt_balance > 0:
            revenue = self.usdt_balance * current_price * (1 - cfg.fee_rate)
            fee = self.usdt_balance * current_price * cfg.fee_rate
            self.trades.append(Trade(
                timestamp=timestamp,
                order_type="recenter_sell",
                price=current_price,
                quantity=self.usdt_balance,
                fee_krw=fee,
            ))
            self.krw_balance += revenue
            self.usdt_balance = 0.0

        self.recenter_count += 1
        self.initialized = False
        self.initialize(current_price, usdkrw_rate)

    # ──────────────────────────────────────────────────────── 캔들 처리 ──

    def process_candle(
        self,
        timestamp: pd.Timestamp,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        usdkrw_rate: Optional[float] = None,
        kimchi_premium: Optional[float] = None,
    ) -> None:
        """단일 일봉 캔들에 대한 그리드 체결 시뮬레이션"""
        if not self.initialized:
            self.initialize(open_price, usdkrw_rate)
            return

        # ── USD/KRW 변동 감지 → 그리드 재설정 ───────────────────────────
        if usdkrw_rate and self._usdkrw_center:
            rate_change = abs(usdkrw_rate - self._usdkrw_center) / self._usdkrw_center
            if rate_change > self.config.recenter_threshold_pct:
                self._recenter(timestamp, open_price, usdkrw_rate)
                return  # 재설정 당일은 그리드 거래 없음

        # ── 캔들 방향에 따라 체결 순서 결정 ─────────────────────────────
        # 양봉: 저가(매수 확인) → 고가(매도 확인)
        # 음봉: 고가(매도 확인) → 저가(매수 확인)
        bullish = close_price >= open_price
        if bullish:
            self._process_direction(timestamp, low_price,  "buy",  kimchi_premium)
            self._process_direction(timestamp, high_price, "sell", kimchi_premium)
        else:
            self._process_direction(timestamp, high_price, "sell", kimchi_premium)
            self._process_direction(timestamp, low_price,  "buy",  kimchi_premium)

    def _process_direction(
        self,
        timestamp: pd.Timestamp,
        trigger_price: float,
        side: str,
        kimchi_premium: Optional[float],
    ) -> None:
        cfg = self.config

        # ── 김치프리미엄 필터 ────────────────────────────────────────────
        # 프리미엄이 임계값을 초과하면 해당 방향 거래 중단
        #   - 매수: 프리미엄이 너무 높을 때 (USDT 고평가 상태에서 추가 매수 위험)
        #   - 매도: 프리미엄이 너무 낮을 때 (USDT 저평가 상태에서 매도 시 기회 손실)
        if side == "buy" and kimchi_premium is not None:
            if kimchi_premium > cfg.kimchi_buy_max_pct:
                return
        if side == "sell" and kimchi_premium is not None:
            if kimchi_premium < cfg.kimchi_sell_min_pct:
                return

        for i in range(cfg.n_slots):
            if side == "buy" and self._slot_states.get(i) == "empty":
                if trigger_price <= self.grid_prices[i]:
                    self._execute_buy(timestamp, i)
            elif side == "sell" and self._slot_states.get(i) == "filled":
                if trigger_price >= self.grid_prices[i + 1]:
                    self._execute_sell(timestamp, i)

    def _execute_buy(self, timestamp: pd.Timestamp, slot: int) -> None:
        cfg = self.config
        buy_price = self.grid_prices[slot]
        qty = self._slot_qty[slot]
        cost = buy_price * qty * (1 + cfg.fee_rate)

        if self.krw_balance < cost:
            return

        self.krw_balance -= cost
        self.usdt_balance += qty
        self._slot_states[slot] = "filled"
        self._slot_buy_price[slot] = buy_price

        self.trades.append(Trade(
            timestamp=timestamp,
            order_type="buy",
            price=buy_price,
            quantity=qty,
            fee_krw=buy_price * qty * cfg.fee_rate,
        ))

    def _execute_sell(self, timestamp: pd.Timestamp, slot: int) -> None:
        cfg = self.config
        sell_price = self.grid_prices[slot + 1]
        qty = self._slot_qty[slot]

        if self.usdt_balance < qty:
            return

        revenue = sell_price * qty * (1 - cfg.fee_rate)
        self.usdt_balance -= qty
        self.krw_balance += revenue

        buy_price = self._slot_buy_price[slot]
        fee_total = (buy_price + sell_price) * qty * cfg.fee_rate
        pnl = (sell_price - buy_price) * qty - fee_total

        self.trades.append(Trade(
            timestamp=timestamp,
            order_type="sell",
            price=sell_price,
            quantity=qty,
            fee_krw=sell_price * qty * cfg.fee_rate,
            realized_pnl_krw=pnl,
        ))

        self._slot_states[slot] = "empty"
        # 다음 매수를 위한 수량 재산정 (슬롯 하단 가격 기준)
        self._slot_qty[slot] = self._investment_per_slot / self.grid_prices[slot]

    # ──────────────────────────────────────────────────────────── 통계 ──

    def portfolio_value(self, current_price: float) -> float:
        return self.krw_balance + self.usdt_balance * current_price

    def summary(self, final_price: float) -> dict:
        buy_trades  = [t for t in self.trades if t.order_type == "buy"]
        sell_trades = [t for t in self.trades if t.order_type == "sell"]
        total_pnl   = sum(t.realized_pnl_krw for t in sell_trades)
        total_fees  = sum(
            t.fee_krw for t in self.trades
            if t.order_type in ("buy", "sell", "init_buy", "recenter_sell")
        )
        final_value = self.portfolio_value(final_price)
        ret = (final_value - self.config.initial_investment_krw) / self.config.initial_investment_krw

        return {
            "initial_investment_krw": self.config.initial_investment_krw,
            "final_value_krw":        final_value,
            "total_return_pct":       ret * 100,
            "realized_pnl_krw":       total_pnl,
            "total_fees_krw":         total_fees,
            "n_buy_trades":           len(buy_trades),
            "n_sell_trades":          len(sell_trades),
            "n_total_trades":         len(buy_trades) + len(sell_trades),
            "n_recenter":             self.recenter_count,
            "final_krw_balance":      self.krw_balance,
            "final_usdt_balance":     self.usdt_balance,
            "final_usdt_value_krw":   self.usdt_balance * final_price,
            "grid_prices":            self.grid_prices,
            "center_price":           self.center_price,
        }
