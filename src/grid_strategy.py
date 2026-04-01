"""
USDT/KRW 그리드 매매 전략 핵심 로직

전략 개요:
  - 중심가격(USD/KRW 환율 또는 초기 USDT 시세) 기준으로 ±(range/2)% 범위에
    grid_count개의 균등 간격 그리드 레벨을 설정한다.
  - 초기화 시 현재가 하방 레벨 → KRW 보유(매수 대기),
             현재가 상방 레벨 → USDT 보유(매도 대기) 형태로 분산 배치한다.
  - 매수 체결 → 한 레벨 위에 매도 주문 생성
  - 매도 체결 → 한 레벨 아래에 매수 주문 생성
  - 수수료는 각 체결 시 반영한다 (Upbit 기준 0.05%).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd


# ---------------------------------------------------------------------------
# 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class GridConfig:
    """그리드 전략 설정"""
    initial_investment_krw: float       # 초기 투자금 (KRW)
    grid_count: int = 10                # 그리드 레벨 수
    grid_range_pct: float = 0.06        # 전체 범위 (예: 0.06 → 중심가 ±3%)
    fee_rate: float = 0.0005            # 거래 수수료 (Upbit: 0.05%)
    use_dynamic_center: bool = True     # True: USD/KRW 기준, False: 초기 시세 고정

    @property
    def grid_step_pct(self) -> float:
        return self.grid_range_pct / (self.grid_count - 1) if self.grid_count > 1 else 0


@dataclass
class Order:
    level: int
    price: float
    quantity: float        # USDT 수량
    order_type: str        # 'buy' | 'sell'


@dataclass
class Trade:
    timestamp: pd.Timestamp
    order_type: str        # 'buy' | 'sell'
    price: float
    quantity: float        # USDT 수량
    fee_krw: float
    realized_pnl_krw: float = 0.0   # 매도 시에만 의미 있음


# ---------------------------------------------------------------------------
# 핵심 전략 클래스
# ---------------------------------------------------------------------------

class GridStrategy:
    def __init__(self, config: GridConfig):
        self.config = config
        self.krw_balance: float = config.initial_investment_krw
        self.usdt_balance: float = 0.0
        self.active_orders: Dict[int, Order] = {}
        self.trades: List[Trade] = []
        self.grid_prices: List[float] = []
        self.center_price: float = 0.0
        self.initialized: bool = False
        # 각 레벨별 취득단가 (P&L 계산용)
        self._buy_price_per_level: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # 초기화
    # ------------------------------------------------------------------

    def initialize(self, current_price: float, usdkrw_rate: Optional[float] = None) -> None:
        """그리드 레벨 초기화 및 초기 주문 배치"""
        cfg = self.config
        self.center_price = usdkrw_rate if (cfg.use_dynamic_center and usdkrw_rate) else current_price

        n = cfg.grid_count
        r = cfg.grid_range_pct
        step = r / (n - 1) if n > 1 else 0

        self.grid_prices = [
            round(self.center_price * (1 - r / 2 + step * i), 4)
            for i in range(n)
        ]

        investment_per_level = cfg.initial_investment_krw / n

        # 레벨 분류
        above = [i for i, p in enumerate(self.grid_prices) if p > current_price]
        below = [i for i, p in enumerate(self.grid_prices) if p < current_price]

        # ── 상방 레벨: USDT를 미리 매수해두고 매도 주문 세팅 ──────────────
        if above:
            usdt_per_above_level = investment_per_level / current_price
            total_usdt_to_buy = usdt_per_above_level * len(above)
            cost = total_usdt_to_buy * current_price * (1 + cfg.fee_rate)
            if cost <= self.krw_balance:
                self.krw_balance -= cost
                self.usdt_balance += total_usdt_to_buy
                # 초기 매수를 거래 기록에 남김 (P&L 계산 기준)
                self.trades.append(Trade(
                    timestamp=pd.Timestamp("1970-01-01"),  # 초기화 표시
                    order_type="init_buy",
                    price=current_price,
                    quantity=total_usdt_to_buy,
                    fee_krw=cost - total_usdt_to_buy * current_price,
                ))
                for i in above:
                    self.active_orders[i] = Order(
                        level=i,
                        price=self.grid_prices[i],
                        quantity=usdt_per_above_level,
                        order_type="sell",
                    )
                    self._buy_price_per_level[i] = current_price

        # ── 하방 레벨: KRW 유보 후 매수 주문 세팅 ──────────────────────────
        for i in below:
            qty = investment_per_level / self.grid_prices[i]
            self.active_orders[i] = Order(
                level=i,
                price=self.grid_prices[i],
                quantity=qty,
                order_type="buy",
            )

        self.initialized = True

    # ------------------------------------------------------------------
    # 캔들 처리
    # ------------------------------------------------------------------

    def process_candle(
        self,
        timestamp: pd.Timestamp,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        usdkrw_rate: Optional[float] = None,
    ) -> None:
        """단일 캔들에 대한 주문 체결 시뮬레이션"""
        if not self.initialized:
            self.initialize(open_price, usdkrw_rate)

        # 캔들 방향에 따라 저가/고가 탐색 순서 결정
        # 양봉: 저가 → 고가 (매수 먼저, 이후 매도)
        # 음봉: 고가 → 저가 (매도 먼저, 이후 매수)
        bullish = close_price >= open_price
        if bullish:
            self._check_and_execute(timestamp, low_price, "buy")
            self._check_and_execute(timestamp, high_price, "sell")
        else:
            self._check_and_execute(timestamp, high_price, "sell")
            self._check_and_execute(timestamp, low_price, "buy")

    def _check_and_execute(self, timestamp: pd.Timestamp, price: float, side: str) -> None:
        """해당 가격에서 트리거되는 주문을 체결한다"""
        triggered = []
        for level, order in self.active_orders.items():
            if order.order_type != side:
                continue
            if side == "buy" and price <= order.price:
                triggered.append(level)
            elif side == "sell" and price >= order.price:
                triggered.append(level)

        for level in triggered:
            self._execute(timestamp, self.active_orders[level])

    def _execute(self, timestamp: pd.Timestamp, order: Order) -> None:
        """주문 체결 처리"""
        cfg = self.config

        if order.order_type == "buy":
            cost = order.price * order.quantity * (1 + cfg.fee_rate)
            if self.krw_balance < cost:
                return
            self.krw_balance -= cost
            self.usdt_balance += order.quantity
            self._buy_price_per_level[order.level] = order.price
            self.trades.append(Trade(
                timestamp=timestamp,
                order_type="buy",
                price=order.price,
                quantity=order.quantity,
                fee_krw=order.price * order.quantity * cfg.fee_rate,
            ))
            # 체결 후 → 한 레벨 위에 매도 주문
            next_level = order.level + 1
            if next_level < len(self.grid_prices):
                sell_qty = order.quantity
                self.active_orders[order.level] = Order(
                    level=order.level,
                    price=self.grid_prices[next_level],
                    quantity=sell_qty,
                    order_type="sell",
                )
                self._buy_price_per_level[order.level] = order.price
            else:
                del self.active_orders[order.level]

        elif order.order_type == "sell":
            if self.usdt_balance < order.quantity:
                return
            revenue = order.price * order.quantity * (1 - cfg.fee_rate)
            self.usdt_balance -= order.quantity
            self.krw_balance += revenue

            buy_price = self._buy_price_per_level.get(order.level, order.price)
            fee_total = order.price * order.quantity * cfg.fee_rate + buy_price * order.quantity * cfg.fee_rate
            pnl = (order.price - buy_price) * order.quantity - fee_total

            self.trades.append(Trade(
                timestamp=timestamp,
                order_type="sell",
                price=order.price,
                quantity=order.quantity,
                fee_krw=order.price * order.quantity * cfg.fee_rate,
                realized_pnl_krw=pnl,
            ))
            # 체결 후 → 한 레벨 아래에 매수 주문
            prev_level = order.level - 1
            if prev_level >= 0:
                buy_price_new = self.grid_prices[prev_level]
                buy_qty = (cfg.initial_investment_krw / cfg.grid_count) / buy_price_new
                self.active_orders[order.level] = Order(
                    level=order.level,
                    price=buy_price_new,
                    quantity=buy_qty,
                    order_type="buy",
                )
            else:
                del self.active_orders[order.level]

    # ------------------------------------------------------------------
    # 통계
    # ------------------------------------------------------------------

    def portfolio_value(self, current_price: float) -> float:
        return self.krw_balance + self.usdt_balance * current_price

    def summary(self, final_price: float) -> dict:
        buy_trades = [t for t in self.trades if t.order_type == "buy"]
        sell_trades = [t for t in self.trades if t.order_type == "sell"]
        total_pnl = sum(t.realized_pnl_krw for t in sell_trades)
        total_fees = sum(t.fee_krw for t in self.trades if t.order_type in ("buy", "sell"))
        final_value = self.portfolio_value(final_price)
        ret = (final_value - self.config.initial_investment_krw) / self.config.initial_investment_krw

        return {
            "initial_investment_krw": self.config.initial_investment_krw,
            "final_value_krw": final_value,
            "total_return_pct": ret * 100,
            "realized_pnl_krw": total_pnl,
            "total_fees_krw": total_fees,
            "n_buy_trades": len(buy_trades),
            "n_sell_trades": len(sell_trades),
            "n_total_trades": len(buy_trades) + len(sell_trades),
            "final_krw_balance": self.krw_balance,
            "final_usdt_balance": self.usdt_balance,
            "final_usdt_value_krw": self.usdt_balance * final_price,
            "grid_prices": self.grid_prices,
            "center_price": self.center_price,
        }
