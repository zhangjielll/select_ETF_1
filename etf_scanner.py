#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  大A改良版 CAN SLIM + 多周期共振 ETF 右侧交易自动筛选 Agent（纯本地运行版）
================================================================================

依赖安装：
    pip install akshare pandas tqdm

TA-Lib（可选，本脚本默认使用 pandas 手工实现指标，与同花顺参数对齐）：
    Windows: 从 https://github.com/cgohlke/talib-build/releases 下载 whl 安装
    Linux:   sudo apt-get install ta-lib && pip install TA-Lib

运行方式：
    python etf_scanner.py                    # 正常运行（非周五收盘后会提示）
    python etf_scanner.py --force            # 强制运行完整筛选
    python etf_scanner.py --backtest 20230101 20240101  # 运行回测

修改筛选条件：编辑顶部 CONFIG 字典
查看历史结果：results/ 目录下的 CSV 文件
启用基本面占位：将 CONFIG 中 C_ENABLED / A_ENABLED / I_ENABLED 设为 True 并补充数据源
================================================================================
"""

import os
import sys
import time
import json
import pickle
import logging
import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # tqdm 降级：简单进度提示
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

try:
    import akshare as ak
except ImportError:
    print("错误：请先安装 akshare：pip install akshare")
    sys.exit(1)

warnings.filterwarnings("ignore")

# ==================================================================================
# CONFIG - 所有可配置参数集中在此
# ==================================================================================
CONFIG = {
    # ---- 指标参数（与同花顺对齐）----
    "MACD_FAST": 12,            # MACD 快线周期
    "MACD_SLOW": 26,            # MACD 慢线周期
    "MACD_SIGNAL": 9,           # MACD 信号线周期
    "RSI_PERIOD": 14,           # RSI 周期
    "BOLL_PERIOD": 20,          # 布林带周期
    "BOLL_STD": 2,              # 布林带标准差倍数
    "MA_SHORT": 20,             # 短期均线（月K）
    "MA_LONG": 60,              # 长期均线（月K）
    "VOLUME_AVG_WEEKS": 5,      # 成交量均值比较周数
    "VOLUME_RATIO": 1.2,        # 放量倍数阈值

    # ---- 前置过滤阈值 ----
    "MIN_ETF_AGE_DAYS": 365,    # 最小成立天数
    "MIN_DAILY_AMOUNT": 1000,   # 最小日均成交额（万元）
    "MIN_SCALE": 5,             # 最小规模（亿元）
    "TRADE_DAYS_FOR_AVG": 20,   # 日均成交额计算天数

    # ---- CAN SLIM 开关与阈值 ----
    "C_ENABLED": False,         # 当季盈利（占位，默认关闭）
    "A_ENABLED": False,         # 年度盈利（占位，默认关闭）
    "N_ENABLED": True,          # 新高/新催化剂
    "S_ENABLED": True,          # 供需关系
    "L_ENABLED": True,          # 领涨领跌
    "I_ENABLED": False,         # 机构认同（占位，默认关闭）
    "M_ENABLED": True,          # 市场方向

    "N_NEW_HIGH_MONTHS": 3,     # N条件：近N个月新高
    "N_SECTOR_RANK_PCT": 0.3,   # N条件：板块涨幅排名前30%
    "S_SCALE_MIN": 20,          # S条件：最小规模（亿元）
    "S_SCALE_MAX": 200,         # S条件：最大规模（亿元）
    "M_CSI300_MA_WEEKS": 20,    # M条件：沪深300均线周数
    "M_CHINEXT_RSI_THRESHOLD": 40,  # M条件：创业板RSI阈值

    # ---- 技术面筛选开关 ----
    "MACD_CROSS_ENABLED": True,
    "VOLUME_FILTER_ENABLED": True,
    "RSI_FILTER_ENABLED": True,
    "BOLL_FILTER_ENABLED": True,
    "MONTHLY_MA_ENABLED": True,
    "MONTHLY_MA60_ENABLED": False,  # 60月均线（可选）
    "MONTHLY_3M_RETURN_ENABLED": True,

    # ---- 金叉额外过滤 ----
    "MACD_ABOVE_ZERO_FILTER": True,     # 是否过滤零轴下方金叉
    "MACD_NEAR_ZERO_RATIO": 0.02,       # 零轴附近判定比例

    # ---- 缓存设置 ----
    "CACHE_DIR": "cache",
    "CACHE_TTL_HOURS": 24,      # 缓存有效期
    "CACHE_CLEANUP_DAYS": 7,    # 清理N天前的缓存

    # ---- 输出设置 ----
    "RESULTS_DIR": "results",
    "BACKTEST_DIR": "backtest",
    "REQUEST_DELAY": 0.3,       # 请求间隔（秒）
    "MAX_RETRY": 2,             # 网络重试次数

    # ---- 板块映射（ETF名称关键词 -> 板块分类）----
    "SECTOR_KEYWORDS": {
        "半导体": "半导体", "芯片": "半导体", "集成电路": "半导体",
        "医药": "医药", "医疗": "医药", "创新药": "医药", "生物医药": "医药",
        "新能源": "新能源", "光伏": "新能源", "锂电": "新能源", "风电": "新能源",
        "消费": "消费", "食品": "消费", "白酒": "消费", "家电": "消费",
        "银行": "金融", "证券": "金融", "保险": "金融", "金融": "金融",
        "军工": "军工", "国防": "军工",
        "科技": "科技", "信息技术": "科技", "人工智能": "科技", "AI": "科技",
        "地产": "房地产", "房地产": "房地产",
        "汽车": "汽车", "新能源车": "汽车", "智能驾驶": "汽车",
        "有色": "周期", "钢铁": "周期", "煤炭": "周期", "化工": "周期",
        "沪深300": "宽基", "中证500": "宽基", "中证1000": "宽基",
        "创业板": "宽基", "科创": "宽基", "上证50": "宽基", "中证A500": "宽基",
        "恒生": "跨境", "纳斯达克": "跨境", "标普": "跨境",
    },

    # ---- 保留的ETF类型（剔除债券、商品、货币、QDII、跨境）----
    "KEEP_KEYWORDS": ["股票", "行业", "宽基", "主题", "策略", "指数"],
    "EXCLUDE_KEYWORDS": ["债券", "商品", "货币", "QDII", "跨境", "黄金", "原油", "白银"],

    # ---- 指数代码 ----
    "CSI300_CODE": "sh000300",      # 沪深300
    "CHINEXT_CODE": "sz399006",     # 创业板指
}

# ==================================================================================
# 日志配置
# ==================================================================================
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("etf_scanner.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

# ==================================================================================
# 目录初始化
# ==================================================================================
for d in [CONFIG["CACHE_DIR"], CONFIG["RESULTS_DIR"], CONFIG["BACKTEST_DIR"]]:
    os.makedirs(d, exist_ok=True)


# ==================================================================================
# 模块一：缓存管理
# ==================================================================================
def _cache_key(name: str) -> str:
    """生成缓存文件路径"""
    return os.path.join(CONFIG["CACHE_DIR"], f"{name}.pkl")


def save_cache(name: str, data):
    """保存数据到缓存"""
    path = _cache_key(name)
    with open(path, "wb") as f:
        pickle.dump({"timestamp": time.time(), "data": data}, f)


def load_cache(name: str, ttl_hours: float = None):
    """加载缓存数据，过期返回 None"""
    if ttl_hours is None:
        ttl_hours = CONFIG["CACHE_TTL_HOURS"]
    path = _cache_key(name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            cache = pickle.load(f)
        age_hours = (time.time() - cache["timestamp"]) / 3600
        if age_hours > ttl_hours:
            return None
        return cache["data"]
    except Exception:
        return None


def cleanup_cache():
    """清理过期缓存文件"""
    cache_dir = CONFIG["CACHE_DIR"]
    cutoff = time.time() - CONFIG["CACHE_CLEANUP_DAYS"] * 86400
    count = 0
    for f in os.listdir(cache_dir):
        fp = os.path.join(cache_dir, f)
        if os.path.isfile(fp) and os.path.getmtime(fp) < cutoff:
            os.remove(fp)
            count += 1
    if count > 0:
        logger.info(f"已清理 {count} 个过期缓存文件")


# ==================================================================================
# 模块二：数据抓取
# ==================================================================================
def fetch_etf_list() -> pd.DataFrame:
    """获取全市场ETF实时行情列表"""
    cache_name = f"etf_list_{datetime.now().strftime('%Y%m%d')}"
    cached = load_cache(cache_name, ttl_hours=CONFIG["CACHE_TTL_HOURS"])
    if cached is not None:
        logger.info("使用缓存的ETF列表")
        return cached

    logger.info("正在获取全市场ETF列表...")
    for retry in range(CONFIG["MAX_RETRY"] + 1):
        try:
            df = ak.fund_etf_spot_em()
            if df is not None and len(df) > 0:
                save_cache(cache_name, df)
                logger.info(f"获取到 {len(df)} 只ETF")
                return df
        except Exception as e:
            logger.warning(f"获取ETF列表失败(第{retry+1}次): {e}")
            if retry < CONFIG["MAX_RETRY"]:
                time.sleep(1)
    logger.error("无法获取ETF列表，程序退出")
    sys.exit(1)


def fetch_etf_kline(symbol: str, period: str = "周",
                    start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    获取单只ETF的K线数据
    symbol: ETF代码（6位数字）
    period: "周" 或 "月"
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y%m%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    cache_name = f"kline_{symbol}_{period}_{end_date}"
    cached = load_cache(cache_name, ttl_hours=CONFIG["CACHE_TTL_HOURS"])
    if cached is not None:
        return cached

    for retry in range(CONFIG["MAX_RETRY"] + 1):
        try:
            df = ak.fund_etf_hist_em(
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
                adjust="hfq"  # 后复权
            )
            if df is not None and len(df) > 0:
                # 统一列名
                df.columns = [c.strip() for c in df.columns]
                col_map = {}
                for c in df.columns:
                    cl = c.lower()
                    if "日期" in c or "date" in cl:
                        col_map[c] = "date"
                    elif "开盘" in c or "open" in cl:
                        col_map[c] = "open"
                    elif "最高" in c or "high" in cl:
                        col_map[c] = "high"
                    elif "最低" in c or "low" in cl:
                        col_map[c] = "low"
                    elif "收盘" in c or "close" in cl:
                        col_map[c] = "close"
                    elif "成交量" in c or "volume" in cl:
                        col_map[c] = "volume"
                    elif "成交额" in c or "amount" in cl:
                        col_map[c] = "amount"
                    elif "涨跌幅" in c or "change" in cl:
                        col_map[c] = "pct_change"
                df = df.rename(columns=col_map)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values("date").reset_index(drop=True)
                for col in ["open", "high", "low", "close", "volume", "amount"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                save_cache(cache_name, df)
                return df
        except Exception as e:
            logger.warning(f"获取 {symbol} {period}K线失败(第{retry+1}次): {e}")
            if retry < CONFIG["MAX_RETRY"]:
                time.sleep(1)
    return pd.DataFrame()


def fetch_index_kline(symbol: str, period: str = "周") -> pd.DataFrame:
    """获取指数K线数据（沪深300/创业板指）"""
    cache_name = f"index_{symbol}_{period}_{datetime.now().strftime('%Y%m%d')}"
    cached = load_cache(cache_name, ttl_hours=CONFIG["CACHE_TTL_HOURS"])
    if cached is not None:
        return cached

    start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y%m%d")
    end_date = datetime.now().strftime("%Y%m%d")

    for retry in range(CONFIG["MAX_RETRY"] + 1):
        try:
            # akshare 获取指数K线
            code = symbol.replace("sh", "").replace("sz", "")
            df = ak.index_zh_a_hist(
                symbol=code,
                period=period,
                start_date=start_date,
                end_date=end_date,
            )
            if df is not None and len(df) > 0:
                df.columns = [c.strip() for c in df.columns]
                col_map = {}
                for c in df.columns:
                    cl = c.lower()
                    if "日期" in c or "date" in cl:
                        col_map[c] = "date"
                    elif "开盘" in c or "open" in cl:
                        col_map[c] = "open"
                    elif "最高" in c or "high" in cl:
                        col_map[c] = "high"
                    elif "最低" in c or "low" in cl:
                        col_map[c] = "low"
                    elif "收盘" in c or "close" in cl:
                        col_map[c] = "close"
                    elif "成交量" in c or "volume" in cl:
                        col_map[c] = "volume"
                    elif "成交额" in c or "amount" in cl:
                        col_map[c] = "amount"
                    elif "涨跌幅" in c or "change" in cl:
                        col_map[c] = "pct_change"
                df = df.rename(columns=col_map)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values("date").reset_index(drop=True)
                for col in ["open", "high", "low", "close", "volume", "amount"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                save_cache(cache_name, df)
                return df
        except Exception as e:
            logger.warning(f"获取指数 {symbol} {period}K线失败(第{retry+1}次): {e}")
            if retry < CONFIG["MAX_RETRY"]:
                time.sleep(1)
    return pd.DataFrame()


# ==================================================================================
# 模块三：技术指标计算（pandas 实现，与同花顺参数对齐）
# ==================================================================================
def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """计算EMA（与同花顺一致：使用 pandas ewm）"""
    return series.ewm(span=period, adjust=False).mean()


def calc_macd(close: pd.Series, fast: int = None, slow: int = None,
              signal: int = None) -> tuple:
    """
    计算 MACD：DIF, DEA, MACD柱状
    公式与同花顺一致：
      DIF = EMA(close, fast) - EMA(close, slow)
      DEA = EMA(DIF, signal)
      MACD = 2 * (DIF - DEA)
    """
    if fast is None:
        fast = CONFIG["MACD_FAST"]
    if slow is None:
        slow = CONFIG["MACD_SLOW"]
    if signal is None:
        signal = CONFIG["MACD_SIGNAL"]

    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    dif = ema_fast - ema_slow
    dea = calc_ema(dif, signal)
    macd_bar = 2 * (dif - dea)
    return dif, dea, macd_bar


def calc_rsi(close: pd.Series, period: int = None) -> pd.Series:
    """
    计算 RSI（Wilder 平滑法，与同花顺一致）
    """
    if period is None:
        period = CONFIG["RSI_PERIOD"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    # Wilder 平滑 = EMA with alpha=1/period
    avg_gain = gain.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi


def calc_boll(close: pd.Series, period: int = None,
              std_mult: float = None) -> tuple:
    """
    计算布林带：中轨(SMA), 上轨, 下轨
    """
    if period is None:
        period = CONFIG["BOLL_PERIOD"]
    if std_mult is None:
        std_mult = CONFIG["BOLL_STD"]

    mid = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return mid, upper, lower


def calc_ma(close: pd.Series, period: int) -> pd.Series:
    """计算简单移动平均"""
    return close.rolling(window=period).mean()


# ==================================================================================
# 模块四：前置过滤
# ==================================================================================
def pre_filter_etf_list(df: pd.DataFrame) -> pd.DataFrame:
    """
    前置过滤ETF列表：
    1. 剔除成立时间<1年
    2. 剔除近20日日均成交额<1000万
    3. 剔除最新规模<5亿
    4. 保留股票型/行业/宽基，剔除债券/商品/货币/QDII/跨境
    """
    logger.info("=" * 60)
    logger.info("开始前置过滤...")
    initial_count = len(df)

    # 统一列名（akshare fund_etf_spot_em 返回的列名可能有变动）
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if "代码" in c or "code" in cl:
            col_map[c] = "code"
        elif "名称" in c or "name" in cl:
            col_map[c] = "name"
        elif "最新" in c and ("价" in c or "price" in cl):
            col_map[c] = "latest_price"
        elif "成交额" in c or "amount" in cl:
            col_map[c] = "amount"
        elif "规模" in c or "size" in cl or "资产" in c:
            col_map[c] = "scale"
        elif "涨跌幅" in c:
            col_map[c] = "pct_change"
        elif "份额" in c:
            col_map[c] = "shares"
    df = df.rename(columns=col_map)

    # 确保code列为字符串
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.strip()
    else:
        # 尝试用第一列作为代码
        logger.warning("未找到'代码'列，尝试使用第一列")
        df = df.rename(columns={df.columns[0]: "code"})
        df["code"] = df["code"].astype(str).str.strip()

    if "name" not in df.columns:
        df["name"] = ""

    # ---- 类型过滤 ----
    def is_target_type(name):
        name = str(name)
        for kw in CONFIG["EXCLUDE_KEYWORDS"]:
            if kw in name:
                return False
        for kw in CONFIG["KEEP_KEYWORDS"]:
            if kw in name:
                return True
        # 如果名称中没有明确关键词，默认保留（可能是股票型ETF）
        return True

    mask_type = df["name"].apply(is_target_type)
    removed = initial_count - mask_type.sum()
    logger.info(f"  类型过滤：移除 {removed} 只（债券/商品/货币/QDII/跨境等）")
    df = df[mask_type].copy()

    # ---- 成交额过滤（使用实时行情中的成交额）----
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
        # 实时行情中的成交额单位可能是元，转换为万元比较
        # fund_etf_spot_em 返回的成交额通常是元
        median_amount = df["amount"].median()
        if median_amount > 1e8:  # 如果中位数>1亿，说明单位是元
            df["amount_wan"] = df["amount"] / 10000
        else:
            df["amount_wan"] = df["amount"]
        mask_amount = df["amount_wan"] >= CONFIG["MIN_DAILY_AMOUNT"]
        removed = len(df) - mask_amount.sum()
        logger.info(f"  成交额过滤：移除 {removed} 只（日均成交额<{CONFIG['MIN_DAILY_AMOUNT']}万）")
        df = df[mask_amount].copy()

    # ---- 规模过滤 ----
    if "scale" in df.columns:
        df["scale"] = pd.to_numeric(df["scale"], errors="coerce").fillna(0)
        # 规模单位可能是元或亿
        median_scale = df["scale"].median()
        if median_scale > 1e8:  # 单位是元
            df["scale_yi"] = df["scale"] / 1e8
        elif median_scale > 1e4:  # 单位是万元
            df["scale_yi"] = df["scale"] / 1e4
        else:  # 单位已经是亿
            df["scale_yi"] = df["scale"]
        mask_scale = df["scale_yi"] >= CONFIG["MIN_SCALE"]
        removed = len(df) - mask_scale.sum()
        logger.info(f"  规模过滤：移除 {removed} 只（规模<{CONFIG['MIN_SCALE']}亿）")
        df = df[mask_scale].copy()
    else:
        df["scale_yi"] = np.nan
        logger.info("  规模过滤：跳过（未获取到规模数据）")

    # ---- 板块归类 ----
    df["sector"] = df["name"].apply(classify_sector)

    logger.info(f"  前置过滤完成：{initial_count} -> {len(df)} 只ETF")
    logger.info("=" * 60)
    return df


def classify_sector(name: str) -> str:
    """根据ETF名称关键词归类板块"""
    name = str(name)
    for kw, sector in CONFIG["SECTOR_KEYWORDS"].items():
        if kw in name:
            return sector
    return "其他"


# ==================================================================================
# 模块五：技术面筛选
# ==================================================================================
def check_weekly_macd_golden_cross(df_weekly: pd.DataFrame) -> dict:
    """
    检查周K线MACD金叉
    返回: {"pass": bool, "detail": str, "data": dict}
    """
    if len(df_weekly) < CONFIG["MACD_SLOW"] + CONFIG["MACD_SIGNAL"]:
        return {"pass": False, "detail": "周K数据不足", "data": {}}

    close = df_weekly["close"]
    dif, dea, macd_bar = calc_macd(close)

    # 最新一周：DIF > DEA（金叉状态）
    # 前一周：DIF <= DEA（刚穿越）
    if len(dif) < 2:
        return {"pass": False, "detail": "数据不足", "data": {}}

    curr_dif = dif.iloc[-1]
    curr_dea = dea.iloc[-1]
    prev_dif = dif.iloc[-2]
    prev_dea = dea.iloc[-2]
    curr_macd = macd_bar.iloc[-1]
    prev_macd = macd_bar.iloc[-2]

    # 金叉判定：本周DIF>DEA 且 上周DIF<=DEA
    is_golden_cross = (curr_dif > curr_dea) and (prev_dif <= prev_dea)
    # 或者：MACD柱状由负转正
    macd_turn_positive = (curr_macd > 0) and (prev_macd <= 0)

    if not (is_golden_cross or macd_turn_positive):
        return {"pass": False, "detail": "无MACD金叉", "data": {
            "dif": curr_dif, "dea": curr_dea, "macd": curr_macd
        }}

    # 额外过滤：剔除水下弱势假金叉
    if CONFIG["MACD_ABOVE_ZERO_FILTER"]:
        # 条件1：零轴上方金叉（DIF>0 且 DEA>0）
        above_zero = (curr_dif > 0) and (curr_dea > 0)
        # 条件2：零轴附近（DIF > -0.02 * 收盘价）
        curr_close = close.iloc[-1]
        near_zero = curr_dif > (-CONFIG["MACD_NEAR_ZERO_RATIO"] * curr_close)
        if not (above_zero or near_zero):
            return {"pass": False, "detail": "水下弱势金叉，已过滤", "data": {
                "dif": curr_dif, "dea": curr_dea
            }}

    return {"pass": True, "detail": "MACD金叉确认", "data": {
        "dif": round(curr_dif, 4), "dea": round(curr_dea, 4), "macd": round(curr_macd, 4)
    }}


def check_weekly_volume(df_weekly: pd.DataFrame) -> dict:
    """检查周成交量是否放量"""
    if "volume" not in df_weekly.columns or len(df_weekly) < CONFIG["VOLUME_AVG_WEEKS"] + 1:
        return {"pass": True, "detail": "成交量数据不足，跳过", "data": {}}

    vol = df_weekly["volume"]
    curr_vol = vol.iloc[-1]
    avg_vol = vol.iloc[-(CONFIG["VOLUME_AVG_WEEKS"]+1):-1].mean()

    if avg_vol <= 0:
        return {"pass": True, "detail": "均量为0，跳过", "data": {}}

    ratio = curr_vol / avg_vol
    passed = ratio >= CONFIG["VOLUME_RATIO"]
    return {
        "pass": passed,
        "detail": f"量比={ratio:.2f}({'放量' if passed else '缩量'})",
        "data": {"volume_ratio": round(ratio, 2)}
    }


def check_weekly_rsi(df_weekly: pd.DataFrame) -> dict:
    """检查周RSI是否>50"""
    if len(df_weekly) < CONFIG["RSI_PERIOD"] + 1:
        return {"pass": True, "detail": "RSI数据不足，跳过", "data": {}}

    rsi = calc_rsi(df_weekly["close"])
    curr_rsi = rsi.iloc[-1]
    passed = curr_rsi > 50
    return {
        "pass": passed,
        "detail": f"RSI={curr_rsi:.1f}({'多头' if passed else '空头'})",
        "data": {"rsi": round(curr_rsi, 1)}
    }


def check_weekly_boll(df_weekly: pd.DataFrame) -> dict:
    """检查周K是否站稳布林中轨"""
    if len(df_weekly) < CONFIG["BOLL_PERIOD"] + 1:
        return {"pass": True, "detail": "BOLL数据不足，跳过", "data": {}}

    close = df_weekly["close"]
    mid, upper, lower = calc_boll(close)
    curr_close = close.iloc[-1]
    curr_mid = mid.iloc[-1]

    if pd.isna(curr_mid):
        return {"pass": True, "detail": "BOLL计算异常，跳过", "data": {}}

    passed = curr_close >= curr_mid
    return {
        "pass": passed,
        "detail": f"收盘={curr_close:.3f} vs 中轨={curr_mid:.3f}({'站上' if passed else '跌破'})",
        "data": {"close": curr_close, "boll_mid": round(curr_mid, 3)}
    }


def check_monthly_ma(df_monthly: pd.DataFrame) -> dict:
    """检查月K均线趋势"""
    results = {}
    close = df_monthly["close"]

    # MA20
    if len(close) >= CONFIG["MA_SHORT"]:
        ma20 = calc_ma(close, CONFIG["MA_SHORT"])
        curr_ma20 = ma20.iloc[-1]
        curr_close = close.iloc[-1]
        above_ma20 = curr_close > curr_ma20 if not pd.isna(curr_ma20) else True
        results["ma20"] = {
            "pass": above_ma20,
            "detail": f"月MA20={'%.2f' % curr_ma20}({'站上' if above_ma20 else '跌破'})",
        }
    else:
        results["ma20"] = {"pass": True, "detail": "月K不足20根，跳过MA20"}

    # MA60（可选）
    if CONFIG["MONTHLY_MA60_ENABLED"]:
        if len(close) >= CONFIG["MA_LONG"]:
            ma60 = calc_ma(close, CONFIG["MA_LONG"])
            curr_ma60 = ma60.iloc[-1]
            curr_close = close.iloc[-1]
            above_ma60 = curr_close > curr_ma60 if not pd.isna(curr_ma60) else True
            results["ma60"] = {
                "pass": above_ma60,
                "detail": f"月MA60={'%.2f' % curr_ma60}({'站上' if above_ma60 else '跌破'})",
            }
        else:
            results["ma60"] = {"pass": True, "detail": "月K不足60根，跳过MA60"}
    else:
        results["ma60"] = {"pass": True, "detail": "MA60筛选已关闭"}

    # 近3个月涨幅
    if CONFIG["MONTHLY_3M_RETURN_ENABLED"] and len(close) >= 3:
        ret_3m = (close.iloc[-1] / close.iloc[-3] - 1) * 100
        passed = ret_3m > 0
        results["ret_3m"] = {
            "pass": passed,
            "detail": f"近3月涨幅={ret_3m:.2f}%({'正' if passed else '负'})",
            "data": {"ret_3m": round(ret_3m, 2)}
        }
    else:
        results["ret_3m"] = {"pass": True, "detail": "3月涨幅数据不足或已关闭"}

    all_pass = all(r["pass"] for r in results.values())
    detail = " | ".join(r["detail"] for r in results.values())
    return {"pass": all_pass, "detail": detail, "data": results}


# ==================================================================================
# 模块六：CAN SLIM 筛选（大A改良版）
# ==================================================================================

# ---- C: 当季盈利（占位）----
# TODO: 待用户补充ETF跟踪指数的季度扣非净利润数据后实现
# 规则：最新季度扣非净利润同比增长率 > 20%，且连续2个季度正增长
def check_canslim_c(etf_info: dict, df_weekly: pd.DataFrame) -> dict:
    """CAN SLIM - C(Current Earnings)：当前盈利（占位）"""
    if not CONFIG["C_ENABLED"]:
        return {"pass": True, "detail": "[占位] C条件已跳过（未启用）", "data": {}}
    # TODO: 实现逻辑
    # 1. 获取ETF跟踪指数代码
    # 2. 查询该指数成分股的最新季度扣非净利润
    # 3. 计算同比增长率
    # 4. 判断是否>20%且连续2季度正增长
    return {"pass": True, "detail": "[占位] C条件待数据源补充", "data": {}}


# ---- A: 年度盈利（占位）----
# TODO: 待用户补充ETF跟踪指数的年度净利润数据后实现
# 规则：过去3年年度净利润复合增长率 > 15%
def check_canslim_a(etf_info: dict, df_weekly: pd.DataFrame) -> dict:
    """CAN SLIM - A(Annual Earnings)：年度盈利（占位）"""
    if not CONFIG["A_ENABLED"]:
        return {"pass": True, "detail": "[占位] A条件已跳过（未启用）", "data": {}}
    # TODO: 实现逻辑
    # 1. 获取ETF跟踪指数过去3年年度净利润
    # 2. 计算复合增长率 CAGR = (end/start)^(1/years) - 1
    # 3. 判断是否>15%
    return {"pass": True, "detail": "[占位] A条件待数据源补充", "data": {}}


# ---- N: 新高/新催化剂（可实现）----
def check_canslim_n(etf_info: dict, df_weekly: pd.DataFrame,
                    sector_returns: dict) -> dict:
    """CAN SLIM - N(New High)：创近N月新高 + 板块涨幅排名"""
    if not CONFIG["N_ENABLED"]:
        return {"pass": True, "detail": "N条件已关闭", "data": {}}

    if len(df_weekly) < 12:
        return {"pass": False, "detail": "N条件：周K数据不足", "data": {}}

    close = df_weekly["close"]
    n_months = CONFIG["N_NEW_HIGH_MONTHS"]
    n_weeks = n_months * 4  # 近似

    if len(close) < n_weeks:
        return {"pass": False, "detail": "N条件：数据不足", "data": {}}

    # 创近N月新高
    recent_high = close.iloc[-n_weeks:].max()
    curr_close = close.iloc[-1]
    is_new_high = curr_close >= recent_high * 0.98  # 允许2%误差

    # 板块涨幅排名
    sector = etf_info.get("sector", "其他")
    sector_pass = True
    sector_detail = ""
    if sector_returns and sector in sector_returns:
        sorted_sectors = sorted(sector_returns.items(), key=lambda x: x[1], reverse=True)
        rank = next((i+1 for i, (s, _) in enumerate(sorted_sectors) if s == sector), len(sorted_sectors))
        total = len(sorted_sectors)
        rank_pct = rank / total
        sector_pass = rank_pct <= CONFIG["N_SECTOR_RANK_PCT"]
        sector_detail = f"板块排名={rank}/{total}({rank_pct:.0%})"
    else:
        sector_detail = "板块数据未知"

    passed = is_new_high and sector_pass
    detail = f"新高={'是' if is_new_high else '否'} | {sector_detail}"
    return {"pass": passed, "detail": detail, "data": {
        "is_new_high": is_new_high, "sector_rank_pct": rank_pct if sector_returns and sector in sector_returns else None
    }}


# ---- S: 供需关系（可实现）----
def check_canslim_s(etf_info: dict, df_weekly: pd.DataFrame) -> dict:
    """CAN SLIM - S(Supply and Demand)：规模适中 + 份额增长"""
    if not CONFIG["S_ENABLED"]:
        return {"pass": True, "detail": "S条件已关闭", "data": {}}

    scale = etf_info.get("scale_yi", np.nan)
    details = []
    scale_pass = True
    share_growth_pass = True

    # 规模检查：20-200亿
    if not pd.isna(scale):
        in_range = CONFIG["S_SCALE_MIN"] <= scale <= CONFIG["S_SCALE_MAX"]
        scale_pass = in_range
        details.append(f"规模={scale:.1f}亿({'适中' if in_range else '不在区间'})")
    else:
        details.append("规模数据未知")

    # 份额增长（从实时行情中获取的份额变化）
    shares = etf_info.get("shares", np.nan)
    if not pd.isna(shares):
        # 如果有份额增长率数据
        share_growth_pass = shares > 0
        details.append(f"份额变化={'正' if share_growth_pass else '负'}")
    else:
        details.append("份额数据未知，跳过")
        share_growth_pass = True  # 数据不足不淘汰

    passed = scale_pass and share_growth_pass
    return {"pass": passed, "detail": " | ".join(details), "data": {}}


# ---- L: 领涨领跌（可实现）----
def check_canslim_l(etf_info: dict, df_weekly: pd.DataFrame,
                    sector_3m_returns: dict, csi300_3m_ret: float) -> dict:
    """CAN SLIM - L(Leader)：板块领涨 + ETF领涨"""
    if not CONFIG["L_ENABLED"]:
        return {"pass": True, "detail": "L条件已关闭", "data": {}}

    if len(df_weekly) < 12:
        return {"pass": False, "detail": "L条件：数据不足", "data": {}}

    close = df_weekly["close"]
    # ETF近3月涨幅
    etf_3m_ret = (close.iloc[-1] / close.iloc[-12] - 1) * 100 if len(close) >= 12 else 0

    # 板块 vs 沪深300
    sector = etf_info.get("sector", "其他")
    sector_pass = True
    etf_vs_sector_pass = True

    if sector_3m_returns and sector in sector_3m_returns:
        sector_ret = sector_3m_returns[sector]
        sector_pass = sector_ret > csi300_3m_ret
        sector_detail = f"板块3月={sector_ret:.1f}% vs 沪深300={csi300_3m_ret:.1f}%"
    else:
        sector_detail = "板块涨幅数据未知"

    # ETF vs 板块平均
    if sector_3m_returns and sector in sector_3m_returns:
        sector_avg = sector_3m_returns[sector]
        etf_vs_sector_pass = etf_3m_ret > sector_avg

    passed = sector_pass and etf_vs_sector_pass
    detail = f"{sector_detail} | ETF3月={etf_3m_ret:.1f}%"
    return {"pass": passed, "detail": detail, "data": {"etf_3m_ret": round(etf_3m_ret, 2)}}


# ---- I: 机构认同（占位）----
# TODO: 待用户补充ETF机构持仓数据后实现
# 规则：机构持仓比例 > 30%，且近1季度机构持仓比例上升
def check_canslim_i(etf_info: dict, df_weekly: pd.DataFrame) -> dict:
    """CAN SLIM - I(Institutional Sponsorship)：机构认同（占位）"""
    if not CONFIG["I_ENABLED"]:
        return {"pass": True, "detail": "[占位] I条件已跳过（未启用）", "data": {}}
    # TODO: 实现逻辑
    # 1. 获取ETF机构持仓比例
    # 2. 判断是否>30%
    # 3. 获取近1季度变化，判断是否上升
    return {"pass": True, "detail": "[占位] I条件待数据源补充", "data": {}}


# ---- M: 市场方向（可实现）----
def check_canslim_m(csi300_weekly: pd.DataFrame,
                    chinext_weekly: pd.DataFrame) -> dict:
    """CAN SLIM - M(Market Direction)：大盘方向判断"""
    if not CONFIG["M_ENABLED"]:
        return {"pass": True, "detail": "M条件已关闭", "data": {}}

    results = {}

    # 沪深300站上20周均线
    if len(csi300_weekly) >= CONFIG["M_CSI300_MA_WEEKS"]:
        close = csi300_weekly["close"]
        ma20 = calc_ma(close, CONFIG["M_CSI300_MA_WEEKS"])
        curr_close = close.iloc[-1]
        curr_ma20 = ma20.iloc[-1]
        csi_pass = curr_close > curr_ma20 if not pd.isna(curr_ma20) else True
        results["csi300"] = {
            "pass": csi_pass,
            "detail": f"沪深300={'%.0f' % curr_close} vs MA{CONFIG['M_CSI300_MA_WEEKS']}={'%.0f' % curr_ma20}({'多头' if csi_pass else '空头'})"
        }
    else:
        results["csi300"] = {"pass": True, "detail": "沪深300数据不足"}

    # 创业板指RSI > 40
    if len(chinext_weekly) >= CONFIG["RSI_PERIOD"] + 1:
        rsi = calc_rsi(chinext_weekly["close"])
        curr_rsi = rsi.iloc[-1]
        cn_pass = curr_rsi > CONFIG["M_CHINEXT_RSI_THRESHOLD"]
        results["chinext"] = {
            "pass": cn_pass,
            "detail": f"创业板RSI={curr_rsi:.1f}({'正常' if cn_pass else '低迷'})"
        }
    else:
        results["chinext"] = {"pass": True, "detail": "创业板数据不足"}

    all_pass = all(r["pass"] for r in results.values())
    detail = " | ".join(r["detail"] for r in results.values())
    return {"pass": all_pass, "detail": detail, "data": results}


# ==================================================================================
# 模块七：板块涨幅计算
# ==================================================================================
def calc_sector_returns(etf_list_df: pd.DataFrame, kline_cache: dict,
                        weeks: int = 4) -> dict:
    """计算各板块近N周平均涨幅"""
    sector_returns = {}
    sector_counts = {}

    for _, row in etf_list_df.iterrows():
        code = str(row.get("code", ""))
        sector = row.get("sector", "其他")
        if code in kline_cache and len(kline_cache[code]) >= weeks:
            close = kline_cache[code]["close"]
            ret = (close.iloc[-1] / close.iloc[-weeks] - 1) * 100
            if sector not in sector_returns:
                sector_returns[sector] = 0
                sector_counts[sector] = 0
            sector_returns[sector] += ret
            sector_counts[sector] += 1

    # 取平均
    for sector in sector_returns:
        if sector_counts[sector] > 0:
            sector_returns[sector] /= sector_counts[sector]

    return sector_returns


def calc_sector_3m_returns(etf_list_df: pd.DataFrame, kline_cache: dict) -> dict:
    """计算各板块近3个月（约12周）平均涨幅"""
    return calc_sector_returns(etf_list_df, kline_cache, weeks=12)


# ==================================================================================
# 模块八：主筛选流程
# ==================================================================================
def run_scan(force: bool = False):
    """运行完整筛选流程"""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("大A改良版 CAN SLIM + 多周期共振 ETF 右侧交易筛选器")
    logger.info(f"运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 检查是否为周五收盘后
    now = datetime.now()
    is_friday_after = now.weekday() == 4 and now.hour >= 15
    if not is_friday_after and not force:
        logger.warning("今日非周五收盘后（15:30后），数据可能非最新")
        logger.warning("如需强制运行，请使用：python etf_scanner.py --force")
        logger.warning("本次仍将更新缓存数据，但不执行最终筛选")
        resp = input("是否继续强制运行？(y/N): ").strip().lower()
        if resp != "y":
            logger.info("已退出。可使用 --force 参数强制运行")
            return

    # 清理过期缓存
    cleanup_cache()

    # ---- 步骤1：获取ETF列表 ----
    logger.info("\n[步骤1] 获取全市场ETF列表")
    etf_list = fetch_etf_list()
    logger.info(f"全市场ETF总数：{len(etf_list)}")

    # ---- 步骤2：前置过滤 ----
    logger.info("\n[步骤2] 前置过滤")
    etf_filtered = pre_filter_etf_list(etf_list)
    if len(etf_filtered) == 0:
        logger.error("前置过滤后无ETF符合条件，退出")
        return

    # ---- 步骤3：获取K线数据 ----
    logger.info(f"\n[步骤3] 获取K线数据（{len(etf_filtered)}只ETF）")
    weekly_cache = {}   # code -> weekly DataFrame
    monthly_cache = {}  # code -> monthly DataFrame
    failed_codes = []

    codes = etf_filtered["code"].tolist()
    for code in tqdm(codes, desc="获取K线"):
        # 周K
        wk = fetch_etf_kline(code, period="周")
        if wk is not None and len(wk) > 0:
            weekly_cache[code] = wk
        else:
            failed_codes.append(code)
            continue
        time.sleep(CONFIG["REQUEST_DELAY"])

        # 月K
        mk = fetch_etf_kline(code, period="月")
        if mk is not None and len(mk) > 0:
            monthly_cache[code] = mk
        time.sleep(CONFIG["REQUEST_DELAY"])

    logger.info(f"K线获取完成：成功={len(weekly_cache)}，失败={len(failed_codes)}")
    if failed_codes:
        logger.info(f"失败代码示例：{failed_codes[:5]}")

    # ---- 步骤4：获取指数数据 ----
    logger.info("\n[步骤4] 获取大盘指数数据")
    csi300_weekly = fetch_index_kline(CONFIG["CSI300_CODE"], period="周")
    chinext_weekly = fetch_index_kline(CONFIG["CHINEXT_CODE"], period="周")
    time.sleep(CONFIG["REQUEST_DELAY"])

    # 计算沪深300近3月涨幅
    csi300_3m_ret = 0
    if len(csi300_weekly) >= 12:
        c = csi300_weekly["close"]
        csi300_3m_ret = (c.iloc[-1] / c.iloc[-12] - 1) * 100
    logger.info(f"沪深300近3月涨幅：{csi300_3m_ret:.2f}%")

    # ---- 步骤5：计算板块涨幅 ----
    logger.info("\n[步骤5] 计算板块涨幅")
    sector_1m_returns = calc_sector_returns(etf_filtered, weekly_cache, weeks=4)
    sector_3m_returns = calc_sector_3m_returns(etf_filtered, weekly_cache)
    logger.info(f"板块数量：{len(sector_1m_returns)}")

    # ---- 步骤6：大盘方向判断（M条件）----
    logger.info("\n[步骤6] 大盘方向判断")
    m_result = check_canslim_m(csi300_weekly, chinext_weekly)
    logger.info(f"M条件结果：{m_result['detail']}")
    if not m_result["pass"]:
        logger.warning("⚠ 大盘方向偏空，建议谨慎！但继续执行筛选...")

    # ---- 步骤7：逐只ETF筛选 ----
    logger.info(f"\n[步骤7] 逐只ETF筛选（{len(weekly_cache)}只）")
    passed_etfs = []
    stats = {
        "total": len(weekly_cache),
        "macd_pass": 0, "volume_pass": 0, "rsi_pass": 0, "boll_pass": 0,
        "monthly_pass": 0, "canslim_pass": 0, "final_pass": 0,
    }

    for code in tqdm(list(weekly_cache.keys()), desc="筛选中"):
        try:
            wk = weekly_cache[code]
            mk = monthly_cache.get(code, pd.DataFrame())
            row = etf_filtered[etf_filtered["code"] == code]
            if len(row) == 0:
                continue
            etf_info = row.iloc[0].to_dict()

            reasons = []
            indicators = {}

            # --- 技术面筛选 ---
            # MACD金叉
            if CONFIG["MACD_CROSS_ENABLED"]:
                macd_r = check_weekly_macd_golden_cross(wk)
                indicators["macd"] = macd_r["data"]
                if not macd_r["pass"]:
                    continue
                stats["macd_pass"] += 1
                reasons.append("MACD金叉")

            # 放量确认
            if CONFIG["VOLUME_FILTER_ENABLED"]:
                vol_r = check_weekly_volume(wk)
                indicators["volume"] = vol_r["data"]
                if not vol_r["pass"]:
                    continue
                stats["volume_pass"] += 1
                reasons.append(vol_r["detail"])

            # RSI多头
            if CONFIG["RSI_FILTER_ENABLED"]:
                rsi_r = check_weekly_rsi(wk)
                indicators["rsi"] = rsi_r["data"]
                if not rsi_r["pass"]:
                    continue
                stats["rsi_pass"] += 1
                reasons.append(rsi_r["detail"])

            # BOLL站上中轨
            if CONFIG["BOLL_FILTER_ENABLED"]:
                boll_r = check_weekly_boll(wk)
                indicators["boll"] = boll_r["data"]
                if not boll_r["pass"]:
                    continue
                stats["boll_pass"] += 1
                reasons.append("站上BOLL中轨")

            # 月K趋势
            if CONFIG["MONTHLY_MA_ENABLED"] and len(mk) > 0:
                month_r = check_monthly_ma(mk)
                if not month_r["pass"]:
                    continue
                stats["monthly_pass"] += 1
                reasons.append("月K趋势向上")

            # --- CAN SLIM 筛选 ---
            # C（占位）
            c_r = check_canslim_c(etf_info, wk)
            if not c_r["pass"]:
                continue
            if "[占位]" not in c_r["detail"]:
                reasons.append("C:盈利增长")

            # A（占位）
            a_r = check_canslim_a(etf_info, wk)
            if not a_r["pass"]:
                continue
            if "[占位]" not in a_r["detail"]:
                reasons.append("A:年度增长")

            # N（可实现）
            n_r = check_canslim_n(etf_info, wk, sector_1m_returns)
            if not n_r["pass"]:
                continue
            reasons.append("N:新高/板块强")

            # S（可实现）
            s_r = check_canslim_s(etf_info, wk)
            if not s_r["pass"]:
                continue
            reasons.append("S:供需良好")

            # L（可实现）
            l_r = check_canslim_l(etf_info, wk, sector_3m_returns, csi300_3m_ret)
            if not l_r["pass"]:
                continue
            reasons.append("L:领涨")

            # I（占位）
            i_r = check_canslim_i(etf_info, wk)
            if not i_r["pass"]:
                continue
            if "[占位]" not in i_r["detail"]:
                reasons.append("I:机构增持")

            # M（全局，已在步骤6判断，此处记录）
            if not m_result["pass"]:
                reasons.append("⚠M:大盘偏空")

            stats["canslim_pass"] += 1
            stats["final_pass"] += 1

            # 计算涨幅
            close = wk["close"]
            ret_1m = (close.iloc[-1] / close.iloc[-4] - 1) * 100 if len(close) >= 4 else 0
            ret_3m = (close.iloc[-1] / close.iloc[-12] - 1) * 100 if len(close) >= 12 else 0

            passed_etfs.append({
                "code": code,
                "name": etf_info.get("name", ""),
                "sector": etf_info.get("sector", ""),
                "latest_price": close.iloc[-1],
                "ret_1m": round(ret_1m, 2),
                "ret_3m": round(ret_3m, 2),
                "scale_yi": etf_info.get("scale_yi", np.nan),
                "trigger_reasons": " | ".join(reasons),
                "macd_dif": indicators.get("macd", {}).get("dif", np.nan),
                "macd_dea": indicators.get("macd", {}).get("dea", np.nan),
                "rsi": indicators.get("rsi", {}).get("rsi", np.nan),
                "volume_ratio": indicators.get("volume", {}).get("volume_ratio", np.nan),
            })

        except Exception as e:
            logger.warning(f"处理ETF {code} 时出错: {e}")
            continue

    # ---- 步骤8：输出结果 ----
    logger.info("\n" + "=" * 60)
    logger.info("筛选统计：")
    logger.info(f"  总参与筛选：{stats['total']}")
    if CONFIG["MACD_CROSS_ENABLED"]:
        logger.info(f"  MACD金叉通过：{stats['macd_pass']}")
    if CONFIG["VOLUME_FILTER_ENABLED"]:
        logger.info(f"  放量确认通过：{stats['volume_pass']}")
    if CONFIG["RSI_FILTER_ENABLED"]:
        logger.info(f"  RSI多头通过：{stats['rsi_pass']}")
    if CONFIG["BOLL_FILTER_ENABLED"]:
        logger.info(f"  BOLL中轨通过：{stats['boll_pass']}")
    if CONFIG["MONTHLY_MA_ENABLED"]:
        logger.info(f"  月K趋势通过：{stats['monthly_pass']}")
    logger.info(f"  CAN SLIM通过：{stats['canslim_pass']}")
    logger.info(f"  最终入选：{stats['final_pass']}")
    logger.info("=" * 60)

    output_results(passed_etfs, m_result)

    elapsed = time.time() - start_time
    logger.info(f"\n运行耗时：{elapsed:.1f}秒")
    logger.info("筛选完成！")


# ==================================================================================
# 模块九：结果输出
# ==================================================================================
def output_results(passed_etfs: list, m_result: dict):
    """输出筛选结果"""
    today = datetime.now().strftime("%Y%m%d")

    # 控制台输出
    if not passed_etfs:
        logger.info("\n本次筛选无ETF满足所有条件")
    else:
        result_df = pd.DataFrame(passed_etfs)
        result_df = result_df.sort_values("ret_1m", ascending=False)

        print("\n" + "=" * 100)
        print("  筛选结果 - 符合条件的ETF")
        print("=" * 100)
        display_cols = ["code", "name", "sector", "latest_price",
                        "ret_1m", "ret_3m", "trigger_reasons"]
        display_df = result_df[display_cols].copy()
        display_df.columns = ["代码", "名称", "板块", "最新价",
                              "近1月涨幅%", "近3月涨幅%", "触发条件"]
        print(display_df.to_string(index=False))
        print("=" * 100)

    # 保存CSV
    if passed_etfs:
        result_df = pd.DataFrame(passed_etfs)
        csv_path = os.path.join(CONFIG["RESULTS_DIR"], f"筛选结果_{today}.csv")
        result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"\n结果已保存：{csv_path}")

    # 历史对比
    compare_with_history(passed_etfs, today)

    # 大盘趋势报告
    print("\n" + "=" * 60)
    print("  大盘趋势报告")
    print("=" * 60)
    print(f"  {m_result['detail']}")
    if m_result["pass"]:
        print("  结论：大盘趋势偏多，适合右侧交易")
    else:
        print("  结论：大盘趋势偏空，建议控制仓位")
    print("=" * 60)


def compare_with_history(passed_etfs: list, today: str):
    """与上次结果对比，标注新增/移除"""
    results_dir = CONFIG["RESULTS_DIR"]
    files = sorted([f for f in os.listdir(results_dir) if f.startswith("筛选结果_") and f.endswith(".csv")])
    # 找到上一次结果（排除今天）
    prev_file = None
    for f in reversed(files):
        if today not in f:
            prev_file = os.path.join(results_dir, f)
            break

    if prev_file is None or not os.path.exists(prev_file):
        logger.info("无历史结果可供对比")
        return

    try:
        prev_df = pd.read_csv(prev_file, dtype={"code": str})
        prev_codes = set(prev_df["code"].astype(str))
        curr_codes = set(e["code"] for e in passed_etfs)

        new_codes = curr_codes - prev_codes
        removed_codes = prev_codes - curr_codes

        if new_codes:
            new_names = [e["name"] for e in passed_etfs if e["code"] in new_codes]
            logger.info(f"\n📌 新增标的（{len(new_codes)}只）：{', '.join(new_names)}")
        if removed_codes:
            removed_names = prev_df[prev_df["code"].astype(str).isin(removed_codes)]["name"].tolist()
            logger.info(f"\n📌 移除标的（{len(removed_codes)}只）：{', '.join(removed_names)}")
        if not new_codes and not removed_codes:
            logger.info("\n与上次结果一致，无变化")
    except Exception as e:
        logger.warning(f"历史对比失败：{e}")


# ==================================================================================
# 模块十：回测
# ==================================================================================
def run_backtest(start_date: str, end_date: str):
    """
    运行技术面回测
    注意：回测仅基于周/月K线技术面信号，不包含基本面筛选
    start_date / end_date: 格式 "YYYYMMDD"
    """
    logger.info("=" * 60)
    logger.info(f"回测模式：{start_date} ~ {end_date}")
    logger.info("注意：回测仅基于技术面信号（MACD金叉+放量+趋势），不含基本面筛选")
    logger.info("=" * 60)

    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")

    # 获取ETF列表并前置过滤
    etf_list = fetch_etf_list()
    etf_filtered = pre_filter_etf_list(etf_list)

    # 获取全量历史K线（回测用，不使用当天缓存）
    logger.info("获取历史K线数据（回测模式，全量拉取）...")
    all_weekly = {}
    codes = etf_filtered["code"].tolist()

    for code in tqdm(codes, desc="回测数据"):
        try:
            wk = fetch_etf_kline(code, period="周",
                                 start_date=(start_dt - timedelta(days=365)).strftime("%Y%m%d"),
                                 end_date=end_date)
            if wk is not None and len(wk) > 0:
                all_weekly[code] = wk
            time.sleep(CONFIG["REQUEST_DELAY"])
        except Exception:
            continue

    logger.info(f"获取到 {len(all_weekly)} 只ETF的历史周K数据")

    # 获取沪深300用于M条件
    csi300_weekly = fetch_index_kline(CONFIG["CSI300_CODE"], period="周")

    # 按周遍历，模拟信号触发
    logger.info("开始回测计算...")
    trades = []  # 记录所有交易

    # 生成回测周的日期序列
    test_dates = pd.date_range(start=start_dt, end=end_dt, freq="W-FRI")

    for test_date in test_dates:
        # 截取到该日期的数据
        for code, wk_full in all_weekly.items():
            try:
                wk = wk_full[wk_full["date"] <= test_date].copy()
                if len(wk) < 30:
                    continue

                # 检查MACD金叉
                macd_r = check_weekly_macd_golden_cross(wk)
                if not macd_r["pass"]:
                    continue

                # 检查放量
                vol_r = check_weekly_volume(wk)
                if not vol_r["pass"]:
                    continue

                # 检查RSI
                rsi_r = check_weekly_rsi(wk)
                if not rsi_r["pass"]:
                    continue

                # 信号触发，记录买入
                buy_price = wk["close"].iloc[-1]
                buy_date = test_date

                # 计算持有期收益
                future_data = wk_full[wk_full["date"] > test_date]
                if len(future_data) == 0:
                    continue

                hold_results = {}
                for weeks, label in [(1, "1周"), (2, "2周"), (4, "4周")]:
                    if len(future_data) >= weeks:
                        sell_price = future_data["close"].iloc[weeks - 1]
                        ret = (sell_price / buy_price - 1) * 100
                        hold_results[f"ret_{label}"] = round(ret, 2)
                    else:
                        hold_results[f"ret_{label}"] = np.nan

                # 最大回撤（持有4周内）
                if len(future_data) >= 4:
                    period_prices = future_data["close"].iloc[:4]
                    max_dd = ((period_prices.cummax() - period_prices) / period_prices.cummax()).max() * 100
                else:
                    max_dd = np.nan

                trades.append({
                    "code": code,
                    "signal_date": test_date.strftime("%Y-%m-%d"),
                    "buy_price": buy_price,
                    **hold_results,
                    "max_drawdown_4w": round(max_dd, 2) if not pd.isna(max_dd) else np.nan,
                })

            except Exception:
                continue

    # 汇总回测结果
    if not trades:
        logger.info("回测期间无信号触发")
        return

    trades_df = pd.DataFrame(trades)
    total_signals = len(trades_df)

    print("\n" + "=" * 60)
    print(f"  回测报告：{start_date} ~ {end_date}")
    print("=" * 60)
    print(f"  总信号次数：{total_signals}")

    for label in ["1周", "2周", "4周"]:
        col = f"ret_{label}"
        if col in trades_df.columns:
            valid = trades_df[col].dropna()
            if len(valid) > 0:
                win_rate = (valid > 0).sum() / len(valid) * 100
                avg_ret = valid.mean()
                print(f"  持有{label}胜率：{win_rate:.1f}%（{len(valid)}次）")
                print(f"  持有{label}平均收益：{avg_ret:.2f}%")

    if "max_drawdown_4w" in trades_df.columns:
        valid_dd = trades_df["max_drawdown_4w"].dropna()
        if len(valid_dd) > 0:
            print(f"  4周内平均最大回撤：{valid_dd.mean():.2f}%")
            print(f"  4周内最大回撤：{valid_dd.max():.2f}%")

    print("=" * 60)

    # 保存回测结果
    csv_path = os.path.join(CONFIG["BACKTEST_DIR"],
                            f"回测报告_{datetime.now().strftime('%Y%m%d')}_{start_date}_{end_date}.csv")
    trades_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"回测结果已保存：{csv_path}")


# ==================================================================================
# 模块十一：主入口
# ==================================================================================
def main():
    parser = argparse.ArgumentParser(
        description="大A改良版CAN SLIM + 多周期共振ETF右侧交易筛选器"
    )
    parser.add_argument("--force", action="store_true",
                        help="强制运行完整筛选（忽略非周五提示）")
    parser.add_argument("--backtest", nargs=2, metavar=("START", "END"),
                        help="运行回测模式，格式：--backtest 20230101 20240101")
    args = parser.parse_args()

    if args.backtest:
        run_backtest(args.backtest[0], args.backtest[1])
    else:
        run_scan(force=args.force)


if __name__ == "__main__":
    main()
