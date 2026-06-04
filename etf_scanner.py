#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  多周期共振 ETF 右侧交易自动筛选器（纯技术指标版）
================================================================================

依赖安装：
    pip install akshare pandas tqdm

运行方式：
    python etf_scanner.py                    # 正常运行
    python etf_scanner.py --force            # 强制运行
    python etf_scanner.py --backtest 20230101 20240101  # 回测

修改筛选条件：编辑 scanner_engine.py 顶部 CONFIG 字典
================================================================================
"""

import os
import sys
import time
import logging
import argparse
import warnings
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, total=None, desc="", **kwargs):
            self.iterable = iterable
            self.total = total or (len(iterable) if iterable is not None else 0)
            self.desc = desc
            self.n = 0
        def __iter__(self):
            for item in self.iterable:
                self.n += 1
                if self.n % 50 == 0 or self.n == self.total:
                    print(f"\r  {self.desc}: {self.n}/{self.total}", end="", flush=True)
                yield item
            print()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            self.n += n

from scanner_engine import (
    CONFIG, fetch_etf_list, fetch_etf_kline, pre_filter_etf_list,
    check_weekly_macd_golden_cross, check_weekly_volume,
    check_weekly_rsi, check_weekly_boll, check_monthly_ma,
    run_scan_engine, run_backtest_engine, cleanup_cache,
)

warnings.filterwarnings("ignore")

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO, format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("etf_scanner.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)


def run_scan(force: bool = False):
    """运行完整筛选流程"""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("多周期共振 ETF 右侧交易筛选器（纯技术面）")
    logger.info(f"运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    now = datetime.now()
    is_friday_after = now.weekday() == 4 and now.hour >= 15
    if not is_friday_after and not force:
        logger.warning("今日非周五收盘后（15:30后），数据可能非最新")
        resp = input("是否继续强制运行？(y/N): ").strip().lower()
        if resp != "y":
            logger.info("已退出。可使用 --force 参数强制运行")
            return

    cleanup_cache()

    def progress(step, cur, total):
        if cur % 50 == 0 or cur == total:
            print(f"\r  {step}: {cur}/{total}", end="", flush=True)

    result = run_scan_engine(progress_callback=progress)
    print()

    passed = result["passed_etfs"]
    stats = result["stats"]

    logger.info("\n" + "=" * 60)
    logger.info("筛选统计：")
    logger.info(f"  总参与筛选：{stats['total']}")
    logger.info(f"  MACD金叉通过：{stats['macd_pass']}")
    logger.info(f"  放量确认通过：{stats['volume_pass']}")
    logger.info(f"  RSI多头通过：{stats['rsi_pass']}")
    logger.info(f"  BOLL中轨通过：{stats['boll_pass']}")
    logger.info(f"  月K趋势通过：{stats['monthly_pass']}")
    logger.info(f"  最终入选：{stats['final_pass']}")
    logger.info("=" * 60)

    if not passed:
        logger.info("\n本次筛选无ETF满足所有条件")
    else:
        result_df = pd.DataFrame(passed).sort_values("ret_1m", ascending=False)
        print("\n" + "=" * 100)
        print("  筛选结果 - 符合条件的ETF")
        print("=" * 100)
        display_df = result_df[["code", "name", "sector", "latest_price",
                                 "ret_1m", "ret_3m", "trigger_reasons"]].copy()
        display_df.columns = ["代码", "名称", "板块", "最新价", "近1月涨幅%", "近3月涨幅%", "触发条件"]
        print(display_df.to_string(index=False))
        print("=" * 100)

        today = datetime.now().strftime("%Y%m%d")
        csv_path = os.path.join(CONFIG["RESULTS_DIR"], f"筛选结果_{today}.csv")
        result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"\n结果已保存：{csv_path}")

    elapsed = time.time() - start_time
    logger.info(f"\n运行耗时：{elapsed:.1f}秒")


def main():
    parser = argparse.ArgumentParser(description="多周期共振ETF右侧交易筛选器")
    parser.add_argument("--force", action="store_true", help="强制运行完整筛选")
    parser.add_argument("--backtest", nargs=2, metavar=("START", "END"),
                        help="运行回测：--backtest 20230101 20240101")
    args = parser.parse_args()

    if args.backtest:
        start_time = time.time()
        logger.info(f"回测模式：{args.backtest[0]} ~ {args.backtest[1]}")

        def progress(step, cur, total):
            if cur % 20 == 0 or cur == total:
                print(f"\r  {step}: {cur}/{total}", end="", flush=True)

        result = run_backtest_engine(args.backtest[0], args.backtest[1],
                                      progress_callback=progress)
        print()

        trades_df = result["trades"]
        if trades_df is None or len(trades_df) == 0:
            logger.info("回测期间无信号触发")
            return

        total = result["total_signals"]
        print("\n" + "=" * 60)
        print(f"  回测报告：{args.backtest[0]} ~ {args.backtest[1]}")
        print("=" * 60)
        print(f"  总信号次数：{total}")

        for label in ["1周", "2周", "4周"]:
            col = f"ret_{label}"
            if col in trades_df.columns:
                valid = trades_df[col].dropna()
                if len(valid) > 0:
                    win_rate = (valid > 0).sum() / len(valid) * 100
                    print(f"  持有{label}胜率：{win_rate:.1f}%（{len(valid)}次）")
                    print(f"  持有{label}平均收益：{valid.mean():.2f}%")

        if "max_drawdown_4w" in trades_df.columns:
            valid_dd = trades_df["max_drawdown_4w"].dropna()
            if len(valid_dd) > 0:
                print(f"  4周内平均最大回撤：{valid_dd.mean():.2f}%")
        print("=" * 60)

        csv_path = os.path.join(CONFIG["BACKTEST_DIR"],
                                f"回测报告_{datetime.now().strftime('%Y%m%d')}_{args.backtest[0]}_{args.backtest[1]}.csv")
        trades_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"回测结果已保存：{csv_path}")
        logger.info(f"运行耗时：{time.time() - start_time:.1f}秒")
    else:
        run_scan(force=args.force)


if __name__ == "__main__":
    main()
