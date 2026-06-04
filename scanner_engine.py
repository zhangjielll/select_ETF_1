#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF Scanner Engine - 核心逻辑模块
纯技术指标筛选（MACD/RSI/BOLL/均线/成交量）
"""

import os
import sys
import time
import pickle
import logging
import warnings
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

try:
    import akshare as ak
except ImportError:
    ak = None

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ==================================================================================
# CONFIG - 所有可配置参数集中在此
# ==================================================================================
CONFIG = {
    # ---- 指标参数（与同花顺对齐）----
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    "RSI_PERIOD": 14,
    "BOLL_PERIOD": 20,
    "BOLL_STD": 2,
    "MA_SHORT": 20,
    "MA_LONG": 60,
    "VOLUME_AVG_WEEKS": 5,
    "VOLUME_RATIO": 1.2,

    # ---- 前置过滤阈值 ----
    "MIN_ETF_AGE_DAYS": 365,
    "MIN_DAILY_AMOUNT": 1000,
    "MIN_SCALE": 5,
    "TRADE_DAYS_FOR_AVG": 20,

    # ---- 技术面筛选开关 ----
    "MACD_CROSS_ENABLED": True,
    "VOLUME_FILTER_ENABLED": True,
    "RSI_FILTER_ENABLED": True,
    "BOLL_FILTER_ENABLED": True,
    "MONTHLY_MA_ENABLED": True,
    "MONTHLY_MA60_ENABLED": False,
    "MONTHLY_3M_RETURN_ENABLED": True,

    # ---- 金叉额外过滤 ----
    "MACD_ABOVE_ZERO_FILTER": True,
    "MACD_NEAR_ZERO_RATIO": 0.02,

    # ---- 缓存设置 ----
    "CACHE_DIR": "cache",
    "CACHE_TTL_HOURS": 24,
    "CACHE_CLEANUP_DAYS": 7,

    # ---- 输出设置 ----
    "RESULTS_DIR": "results",
    "BACKTEST_DIR": "backtest",
    "REQUEST_DELAY": 0.3,
    "MAX_RETRY": 2,

    # ---- 板块映射 ----
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

    "KEEP_KEYWORDS": ["股票", "行业", "宽基", "主题", "策略", "指数"],
    "EXCLUDE_KEYWORDS": ["债券", "商品", "货币", "QDII", "跨境", "黄金", "原油", "白银"],
}

# 确保目录存在
for d in [CONFIG["CACHE_DIR"], CONFIG["RESULTS_DIR"], CONFIG["BACKTEST_DIR"]]:
    os.makedirs(d, exist_ok=True)


# ==================================================================================
# 缓存管理
# ==================================================================================
def _cache_key(name: str) -> str:
    return os.path.join(CONFIG["CACHE_DIR"], f"{name}.pkl")


def save_cache(name: str, data):
    path = _cache_key(name)
    with open(path, "wb") as f:
        pickle.dump({"timestamp": time.time(), "data": data}, f)


def load_cache(name: str, ttl_hours: float = None):
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
    cache_dir = CONFIG["CACHE_DIR"]
    if not os.path.exists(cache_dir):
        return 0
    cutoff = time.time() - CONFIG["CACHE_CLEANUP_DAYS"] * 86400
    count = 0
    for f in os.listdir(cache_dir):
        fp = os.path.join(cache_dir, f)
        if os.path.isfile(fp) and os.path.getmtime(fp) < cutoff:
            os.remove(fp)
            count += 1
    return count


# ==================================================================================
# 数据抓取
# ==================================================================================
def _normalize_kline_df(df: pd.DataFrame) -> pd.DataFrame:
    """统一K线DataFrame列名和类型"""
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "日期" in c or "date" in cl:
            col_map[c] = "date"
        elif "开盘" in c or ("open" in cl and "prev" not in cl):
            col_map[c] = "open"
        elif "最高" in c or "high" in cl:
            col_map[c] = "high"
        elif "最低" in c or "low" in cl:
            col_map[c] = "low"
        elif ("收盘" in c or "close" in cl) and "prev" not in cl:
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
    return df


def fetch_etf_list() -> pd.DataFrame:
    if ak is None:
        raise ImportError("akshare 未安装")
    cache_name = f"etf_list_{datetime.now().strftime('%Y%m%d')}"
    cached = load_cache(cache_name, ttl_hours=CONFIG["CACHE_TTL_HOURS"])
    if cached is not None:
        return cached

    for retry in range(CONFIG["MAX_RETRY"] + 1):
        try:
            df = ak.fund_etf_spot_em()
            if df is not None and len(df) > 0:
                save_cache(cache_name, df)
                return df
        except Exception as e:
            if retry < CONFIG["MAX_RETRY"]:
                time.sleep(1)
    raise RuntimeError("无法获取ETF列表")


def _resample_kline(df_daily: pd.DataFrame, period: str) -> pd.DataFrame:
    """将日K数据重采样为周K或月K"""
    df = df_daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    if "amount" in df.columns:
        agg["amount"] = "sum"

    cols = [c for c in agg.keys() if c in df.columns]
    df = df[cols]

    if period in ("周", "weekly"):
        resampled = df.resample("W-FRI").agg(agg)
    elif period in ("月", "monthly"):
        resampled = df.resample("ME").agg(agg)
    else:
        return df_daily

    resampled = resampled.dropna(subset=["close"]).reset_index()
    return resampled


def _symbol_to_sina(symbol: str) -> str:
    """将ETF代码转换为新浪格式：5开头=sh，1开头=sz"""
    symbol = str(symbol).strip()
    if symbol.startswith("5"):
        return f"sh{symbol}"
    else:
        return f"sz{symbol}"


def fetch_etf_kline(symbol: str, period: str = "周",
                    start_date: str = None, end_date: str = None) -> pd.DataFrame:
    if ak is None:
        raise ImportError("akshare 未安装")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y%m%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    # akshare fund_etf_hist_em 需要英文 period 参数
    period_map = {"日": "daily", "周": "weekly", "月": "monthly",
                  "daily": "daily", "weekly": "weekly", "monthly": "monthly"}
    ak_period = period_map.get(period, "weekly")

    cache_name = f"kline_{symbol}_{ak_period}_{end_date}"
    cached = load_cache(cache_name, ttl_hours=CONFIG["CACHE_TTL_HOURS"])
    if cached is not None:
        return cached

    # 方法1：东方财富接口 fund_etf_hist_em
    last_error = None
    for retry in range(CONFIG["MAX_RETRY"] + 1):
        try:
            df = ak.fund_etf_hist_em(
                symbol=symbol, period=ak_period,
                start_date=start_date, end_date=end_date, adjust="hfq"
            )
            if df is not None and len(df) > 0:
                df = _normalize_kline_df(df)
                save_cache(cache_name, df)
                return df
        except Exception as e:
            last_error = e
            logger.warning(f"获取 {symbol} {ak_period}K线失败(第{retry+1}次): {e}")
            if retry < CONFIG["MAX_RETRY"]:
                time.sleep(2)

    # 方法2：新浪接口 fund_etf_hist_sina（仅返回日K，需重采样）
    logger.info(f"东方财富接口不可用，尝试新浪接口获取 {symbol} 日K数据...")
    try:
        sina_symbol = _symbol_to_sina(symbol)
        df_sina = ak.fund_etf_hist_sina(symbol=sina_symbol)
        if df_sina is not None and len(df_sina) > 0:
            df_sina = _normalize_kline_df(df_sina)
            # 按日期范围过滤
            df_sina = df_sina[df_sina["date"] >= pd.to_datetime(start_date)]
            df_sina = df_sina[df_sina["date"] <= pd.to_datetime(end_date)]
            # 重采样为目标周期
            if ak_period != "daily":
                df_sina = _resample_kline(df_sina, period)
            save_cache(cache_name, df_sina)
            return df_sina
    except Exception as e2:
        logger.warning(f"新浪接口获取 {symbol} 也失败: {e2}")

    if last_error:
        raise last_error
    return pd.DataFrame()


# ==================================================================================
# 技术指标计算（pandas 实现，与同花顺参数对齐）
# ==================================================================================
def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_macd(close: pd.Series, fast: int = None, slow: int = None,
              signal: int = None) -> tuple:
    if fast is None: fast = CONFIG["MACD_FAST"]
    if slow is None: slow = CONFIG["MACD_SLOW"]
    if signal is None: signal = CONFIG["MACD_SIGNAL"]
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    dif = ema_fast - ema_slow
    dea = calc_ema(dif, signal)
    macd_bar = 2 * (dif - dea)
    return dif, dea, macd_bar


def calc_rsi(close: pd.Series, period: int = None) -> pd.Series:
    if period is None: period = CONFIG["RSI_PERIOD"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi


def calc_boll(close: pd.Series, period: int = None,
              std_mult: float = None) -> tuple:
    if period is None: period = CONFIG["BOLL_PERIOD"]
    if std_mult is None: std_mult = CONFIG["BOLL_STD"]
    mid = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return mid, upper, lower


def calc_ma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(window=period).mean()


# ==================================================================================
# 前置过滤
# ==================================================================================
def classify_sector(name: str) -> str:
    name = str(name)
    for kw, sector in CONFIG["SECTOR_KEYWORDS"].items():
        if kw in name:
            return sector
    return "其他"


def pre_filter_etf_list(df: pd.DataFrame) -> pd.DataFrame:
    initial_count = len(df)

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

    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.strip()
    else:
        df = df.rename(columns={df.columns[0]: "code"})
        df["code"] = df["code"].astype(str).str.strip()

    if "name" not in df.columns:
        df["name"] = ""

    def is_target_type(name):
        name = str(name)
        for kw in CONFIG["EXCLUDE_KEYWORDS"]:
            if kw in name:
                return False
        for kw in CONFIG["KEEP_KEYWORDS"]:
            if kw in name:
                return True
        return True

    mask_type = df["name"].apply(is_target_type)
    df = df[mask_type].copy()

    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
        median_amount = df["amount"].median()
        if median_amount > 1e8:
            df["amount_wan"] = df["amount"] / 10000
        else:
            df["amount_wan"] = df["amount"]
        mask_amount = df["amount_wan"] >= CONFIG["MIN_DAILY_AMOUNT"]
        df = df[mask_amount].copy()

    if "scale" in df.columns:
        df["scale"] = pd.to_numeric(df["scale"], errors="coerce").fillna(0)
        median_scale = df["scale"].median()
        if median_scale > 1e8:
            df["scale_yi"] = df["scale"] / 1e8
        elif median_scale > 1e4:
            df["scale_yi"] = df["scale"] / 1e4
        else:
            df["scale_yi"] = df["scale"]
        mask_scale = df["scale_yi"] >= CONFIG["MIN_SCALE"]
        df = df[mask_scale].copy()
    else:
        df["scale_yi"] = np.nan

    df["sector"] = df["name"].apply(classify_sector)
    return df


def get_sector_list(etf_filtered: pd.DataFrame) -> list:
    """获取板块列表（按ETF数量降序）"""
    if etf_filtered is None or len(etf_filtered) == 0:
        return []
    counts = etf_filtered["sector"].value_counts()
    return counts.index.tolist()


# ==================================================================================
# 技术面筛选
# ==================================================================================
def check_weekly_macd_golden_cross(df_weekly: pd.DataFrame) -> dict:
    if len(df_weekly) < CONFIG["MACD_SLOW"] + CONFIG["MACD_SIGNAL"]:
        return {"pass": False, "detail": "周K数据不足", "data": {}}

    close = df_weekly["close"]
    dif, dea, macd_bar = calc_macd(close)

    if len(dif) < 2:
        return {"pass": False, "detail": "数据不足", "data": {}}

    curr_dif, prev_dif = dif.iloc[-1], dif.iloc[-2]
    curr_dea, prev_dea = dea.iloc[-1], dea.iloc[-2]
    curr_macd, prev_macd = macd_bar.iloc[-1], macd_bar.iloc[-2]

    is_golden_cross = (curr_dif > curr_dea) and (prev_dif <= prev_dea)
    macd_turn_positive = (curr_macd > 0) and (prev_macd <= 0)

    if not (is_golden_cross or macd_turn_positive):
        return {"pass": False, "detail": "无MACD金叉",
                "data": {"dif": curr_dif, "dea": curr_dea, "macd": curr_macd}}

    if CONFIG["MACD_ABOVE_ZERO_FILTER"]:
        above_zero = (curr_dif > 0) and (curr_dea > 0)
        curr_close = close.iloc[-1]
        near_zero = curr_dif > (-CONFIG["MACD_NEAR_ZERO_RATIO"] * curr_close)
        if not (above_zero or near_zero):
            return {"pass": False, "detail": "水下弱势金叉，已过滤",
                    "data": {"dif": curr_dif, "dea": curr_dea}}

    return {"pass": True, "detail": "MACD金叉确认",
            "data": {"dif": round(curr_dif, 4), "dea": round(curr_dea, 4),
                     "macd": round(curr_macd, 4)}}


def check_weekly_volume(df_weekly: pd.DataFrame) -> dict:
    if "volume" not in df_weekly.columns or len(df_weekly) < CONFIG["VOLUME_AVG_WEEKS"] + 1:
        return {"pass": True, "detail": "成交量数据不足，跳过", "data": {}}
    vol = df_weekly["volume"]
    curr_vol = vol.iloc[-1]
    avg_vol = vol.iloc[-(CONFIG["VOLUME_AVG_WEEKS"]+1):-1].mean()
    if avg_vol <= 0:
        return {"pass": True, "detail": "均量为0，跳过", "data": {}}
    ratio = curr_vol / avg_vol
    passed = ratio >= CONFIG["VOLUME_RATIO"]
    return {"pass": passed,
            "detail": f"量比={ratio:.2f}({'放量' if passed else '缩量'})",
            "data": {"volume_ratio": round(ratio, 2)}}


def check_weekly_rsi(df_weekly: pd.DataFrame) -> dict:
    if len(df_weekly) < CONFIG["RSI_PERIOD"] + 1:
        return {"pass": True, "detail": "RSI数据不足，跳过", "data": {}}
    rsi = calc_rsi(df_weekly["close"])
    curr_rsi = rsi.iloc[-1]
    passed = curr_rsi > 50
    return {"pass": passed,
            "detail": f"RSI={curr_rsi:.1f}({'多头' if passed else '空头'})",
            "data": {"rsi": round(curr_rsi, 1)}}


def check_weekly_boll(df_weekly: pd.DataFrame) -> dict:
    if len(df_weekly) < CONFIG["BOLL_PERIOD"] + 1:
        return {"pass": True, "detail": "BOLL数据不足，跳过", "data": {}}
    close = df_weekly["close"]
    mid, upper, lower = calc_boll(close)
    curr_close = close.iloc[-1]
    curr_mid = mid.iloc[-1]
    if pd.isna(curr_mid):
        return {"pass": True, "detail": "BOLL计算异常，跳过", "data": {}}
    passed = curr_close >= curr_mid
    return {"pass": passed,
            "detail": f"收盘={curr_close:.3f} vs 中轨={curr_mid:.3f}({'站上' if passed else '跌破'})",
            "data": {"close": curr_close, "boll_mid": round(curr_mid, 3)}}


def check_monthly_ma(df_monthly: pd.DataFrame) -> dict:
    results = {}
    close = df_monthly["close"]

    if len(close) >= CONFIG["MA_SHORT"]:
        ma20 = calc_ma(close, CONFIG["MA_SHORT"])
        curr_ma20 = ma20.iloc[-1]
        curr_close = close.iloc[-1]
        above_ma20 = curr_close > curr_ma20 if not pd.isna(curr_ma20) else True
        results["ma20"] = {"pass": above_ma20,
                           "detail": f"月MA20={'%.2f' % curr_ma20}({'站上' if above_ma20 else '跌破'})"}
    else:
        results["ma20"] = {"pass": True, "detail": "月K不足20根，跳过MA20"}

    if CONFIG["MONTHLY_MA60_ENABLED"]:
        if len(close) >= CONFIG["MA_LONG"]:
            ma60 = calc_ma(close, CONFIG["MA_LONG"])
            curr_ma60 = ma60.iloc[-1]
            curr_close = close.iloc[-1]
            above_ma60 = curr_close > curr_ma60 if not pd.isna(curr_ma60) else True
            results["ma60"] = {"pass": above_ma60,
                               "detail": f"月MA60={'%.2f' % curr_ma60}({'站上' if above_ma60 else '跌破'})"}
        else:
            results["ma60"] = {"pass": True, "detail": "月K不足60根，跳过MA60"}
    else:
        results["ma60"] = {"pass": True, "detail": "MA60筛选已关闭"}

    if CONFIG["MONTHLY_3M_RETURN_ENABLED"] and len(close) >= 3:
        ret_3m = (close.iloc[-1] / close.iloc[-3] - 1) * 100
        passed = ret_3m > 0
        results["ret_3m"] = {"pass": passed,
                             "detail": f"近3月涨幅={ret_3m:.2f}%({'正' if passed else '负'})",
                             "data": {"ret_3m": round(ret_3m, 2)}}
    else:
        results["ret_3m"] = {"pass": True, "detail": "3月涨幅数据不足或已关闭"}

    all_pass = all(r["pass"] for r in results.values())
    detail = " | ".join(r["detail"] for r in results.values())
    return {"pass": all_pass, "detail": detail, "data": results}


# ==================================================================================
# 主筛选流程（纯技术面）
# ==================================================================================
def run_scan_engine(config_override: dict = None, sector: str = None,
                    progress_callback=None):
    """
    执行纯技术面筛选流程，返回结果字典
    sector: 指定板块筛选（如"半导体"、"医药"），None表示全部板块
    """
    cfg = CONFIG.copy()
    if config_override:
        cfg.update(config_override)

    results = {
        "passed_etfs": [],
        "stats": {},
        "etf_filtered": pd.DataFrame(),
        "sector_stats": {},
        "weekly_cache": {},
        "monthly_cache": {},
        "log": [],
    }

    def log(msg):
        results["log"].append(msg)
        logger.info(msg)

    def progress(step, cur, total):
        if progress_callback:
            progress_callback(step, cur, total)

    # 步骤1：获取ETF列表
    progress("获取ETF列表", 0, 1)
    etf_list = fetch_etf_list()
    log(f"全市场ETF总数：{len(etf_list)}")

    # 步骤2：前置过滤
    progress("前置过滤", 0, 1)
    etf_filtered = pre_filter_etf_list(etf_list)
    log(f"前置过滤后：{len(etf_filtered)} 只ETF")
    results["etf_filtered"] = etf_filtered

    if len(etf_filtered) == 0:
        log("前置过滤后无ETF符合条件")
        return results

    # 板块统计
    sector_counts = etf_filtered["sector"].value_counts().to_dict()
    results["sector_stats"] = sector_counts
    log(f"板块分布：{sector_counts}")

    # 步骤3：板块过滤
    if sector and sector != "全部":
        etf_filtered = etf_filtered[etf_filtered["sector"] == sector].copy()
        log(f"板块筛选「{sector}」后：{len(etf_filtered)} 只ETF")
        if len(etf_filtered) == 0:
            log(f"板块「{sector}」内无ETF符合前置条件")
            return results

    # 步骤3：获取K线数据
    codes = etf_filtered["code"].tolist()
    weekly_cache = {}
    monthly_cache = {}
    failed = 0
    for i, code in enumerate(codes):
        progress("获取K线", i + 1, len(codes))
        try:
            wk = fetch_etf_kline(code, period="周")
            if wk is not None and len(wk) > 0:
                weekly_cache[code] = wk
            else:
                failed += 1
                continue
            time.sleep(cfg["REQUEST_DELAY"])
            mk = fetch_etf_kline(code, period="月")
            if mk is not None and len(mk) > 0:
                monthly_cache[code] = mk
            time.sleep(cfg["REQUEST_DELAY"])
        except Exception:
            failed += 1
            continue

    log(f"K线获取完成：成功={len(weekly_cache)}，失败={failed}")
    results["weekly_cache"] = weekly_cache
    results["monthly_cache"] = monthly_cache

    # 步骤4：逐只筛选（纯技术面）
    passed_etfs = []
    stats = {"total": len(weekly_cache), "macd_pass": 0, "volume_pass": 0,
             "rsi_pass": 0, "boll_pass": 0, "monthly_pass": 0, "final_pass": 0}

    all_codes = list(weekly_cache.keys())
    for i, code in enumerate(all_codes):
        progress("筛选ETF", i + 1, len(all_codes))
        try:
            wk = weekly_cache[code]
            mk = monthly_cache.get(code, pd.DataFrame())
            row = etf_filtered[etf_filtered["code"] == code]
            if len(row) == 0:
                continue
            etf_info = row.iloc[0].to_dict()
            reasons = []
            indicators = {}

            # MACD金叉
            if cfg["MACD_CROSS_ENABLED"]:
                r = check_weekly_macd_golden_cross(wk)
                indicators["macd"] = r["data"]
                if not r["pass"]: continue
                stats["macd_pass"] += 1
                reasons.append("MACD金叉")

            # 放量
            if cfg["VOLUME_FILTER_ENABLED"]:
                r = check_weekly_volume(wk)
                indicators["volume"] = r["data"]
                if not r["pass"]: continue
                stats["volume_pass"] += 1
                reasons.append(r["detail"])

            # RSI
            if cfg["RSI_FILTER_ENABLED"]:
                r = check_weekly_rsi(wk)
                indicators["rsi"] = r["data"]
                if not r["pass"]: continue
                stats["rsi_pass"] += 1
                reasons.append(r["detail"])

            # BOLL
            if cfg["BOLL_FILTER_ENABLED"]:
                r = check_weekly_boll(wk)
                indicators["boll"] = r["data"]
                if not r["pass"]: continue
                stats["boll_pass"] += 1
                reasons.append("站上BOLL中轨")

            # 月K趋势
            if cfg["MONTHLY_MA_ENABLED"] and len(mk) > 0:
                r = check_monthly_ma(mk)
                if not r["pass"]: continue
                stats["monthly_pass"] += 1
                reasons.append("月K趋势向上")

            stats["final_pass"] += 1

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
        except Exception:
            continue

    results["passed_etfs"] = passed_etfs
    results["stats"] = stats
    log(f"筛选完成：最终入选 {stats['final_pass']} 只")
    return results


# ==================================================================================
# 回测引擎
# ==================================================================================
def run_backtest_engine(start_date: str, end_date: str,
                        config_override: dict = None, progress_callback=None) -> dict:
    """回测引擎，基于纯技术面信号"""
    cfg = CONFIG.copy()
    if config_override:
        cfg.update(config_override)

    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")

    trades = []
    log_msgs = []

    def progress(step, cur, total):
        if progress_callback:
            progress_callback(step, cur, total)

    etf_list = fetch_etf_list()
    etf_filtered = pre_filter_etf_list(etf_list)
    log_msgs.append(f"前置过滤后：{len(etf_filtered)} 只ETF")

    codes = etf_filtered["code"].tolist()
    all_weekly = {}
    for i, code in enumerate(codes):
        progress("获取回测数据", i + 1, len(codes))
        try:
            wk = fetch_etf_kline(code, period="周",
                                 start_date=(start_dt - timedelta(days=365)).strftime("%Y%m%d"),
                                 end_date=end_date)
            if wk is not None and len(wk) > 0:
                all_weekly[code] = wk
            time.sleep(cfg["REQUEST_DELAY"])
        except Exception:
            continue

    log_msgs.append(f"获取到 {len(all_weekly)} 只ETF的历史周K数据")

    test_dates = pd.date_range(start=start_dt, end=end_dt, freq="W-FRI")
    for di, test_date in enumerate(test_dates):
        progress("回测计算", di + 1, len(test_dates))
        for code, wk_full in all_weekly.items():
            try:
                wk = wk_full[wk_full["date"] <= test_date].copy()
                if len(wk) < 30:
                    continue

                macd_r = check_weekly_macd_golden_cross(wk)
                if not macd_r["pass"]: continue
                vol_r = check_weekly_volume(wk)
                if not vol_r["pass"]: continue
                rsi_r = check_weekly_rsi(wk)
                if not rsi_r["pass"]: continue

                buy_price = wk["close"].iloc[-1]
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

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return {"trades": trades_df, "log": log_msgs, "total_signals": len(trades)}
