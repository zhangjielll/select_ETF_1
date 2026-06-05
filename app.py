#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit ETF Scanner - 纯技术指标ETF筛选器可视化界面
运行：streamlit run app.py
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from scanner_engine import (
    CONFIG, fetch_etf_list, fetch_etf_kline,
    pre_filter_etf_list, classify_sector, get_sector_list,
    calc_macd, calc_rsi, calc_boll, calc_ma, calc_ema,
    check_weekly_macd_golden_cross, check_weekly_volume,
    check_weekly_rsi, check_weekly_boll, check_monthly_ma,
    run_scan_engine, run_backtest_engine,
    cleanup_cache, load_cache, save_cache,
)

# ==================================================================================
# 页面配置
# ==================================================================================
st.set_page_config(
    page_title="ETF右侧交易筛选器",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stMetric {background-color: #f0f2f6; padding: 15px; border-radius: 10px;}
    .block-container {padding-top: 1rem;}
    div[data-testid="stSidebar"] {background-color: #f8f9fa;}
</style>
""", unsafe_allow_html=True)


# ==================================================================================
# 侧边栏：参数配置
# ==================================================================================
def render_sidebar():
    st.sidebar.title("  参数配置")
    st.sidebar.markdown("---")

    # ---- 指标参数 ----
    st.sidebar.subheader("  技术指标参数")
    macd_fast = st.sidebar.number_input("MACD快线", 1, 50, CONFIG["MACD_FAST"])
    macd_slow = st.sidebar.number_input("MACD慢线", 1, 100, CONFIG["MACD_SLOW"])
    macd_signal = st.sidebar.number_input("MACD信号线", 1, 50, CONFIG["MACD_SIGNAL"])
    rsi_period = st.sidebar.number_input("RSI周期", 1, 50, CONFIG["RSI_PERIOD"])
    boll_period = st.sidebar.number_input("BOLL周期", 1, 100, CONFIG["BOLL_PERIOD"])
    boll_std = st.sidebar.slider("BOLL标准差倍数", 0.5, 4.0, float(CONFIG["BOLL_STD"]), 0.1)

    st.sidebar.markdown("---")

    # ---- 前置过滤 ----
    st.sidebar.subheader("  前置过滤")
    min_amount = st.sidebar.number_input("最小日均成交额(万元)", 0, 10000, CONFIG["MIN_DAILY_AMOUNT"], step=100)
    min_scale = st.sidebar.number_input("最小规模(亿元)", 0, 100, CONFIG["MIN_SCALE"])
    volume_ratio = st.sidebar.slider("放量倍数阈值", 1.0, 3.0, float(CONFIG["VOLUME_RATIO"]), 0.1)

    st.sidebar.markdown("---")

    # ---- 板块选择 ----
    st.sidebar.subheader("  板块筛选")
    sector_options = ["全部"]
    try:
        etf_list = fetch_etf_list()
        etf_filtered = pre_filter_etf_list(etf_list)
        sector_options += get_sector_list(etf_filtered)
        sector_counts = etf_filtered["sector"].value_counts().to_dict()
        st.sidebar.caption(f"前置过滤后共 {len(etf_filtered)} 只ETF")
    except Exception:
        sector_counts = {}

    selected_sector = st.sidebar.selectbox("选择板块", sector_options)
    if selected_sector != "全部" and selected_sector in sector_counts:
        st.sidebar.caption(f"该板块共 {sector_counts[selected_sector]} 只ETF")

    st.sidebar.markdown("---")

    # ---- 技术面开关 ----
    st.sidebar.subheader("  技术面筛选开关")
    macd_cross = st.sidebar.checkbox("MACD金叉", CONFIG["MACD_CROSS_ENABLED"])
    volume_filter = st.sidebar.checkbox("放量确认", CONFIG["VOLUME_FILTER_ENABLED"])
    rsi_filter = st.sidebar.checkbox("RSI多头", CONFIG["RSI_FILTER_ENABLED"])
    boll_filter = st.sidebar.checkbox("BOLL中轨", CONFIG["BOLL_FILTER_ENABLED"])
    monthly_ma = st.sidebar.checkbox("月K趋势", CONFIG["MONTHLY_MA_ENABLED"])
    macd_above_zero = st.sidebar.checkbox("过滤水下金叉", CONFIG["MACD_ABOVE_ZERO_FILTER"])

    config_override = {
        "MACD_FAST": macd_fast, "MACD_SLOW": macd_slow, "MACD_SIGNAL": macd_signal,
        "RSI_PERIOD": rsi_period, "BOLL_PERIOD": boll_period, "BOLL_STD": boll_std,
        "MIN_DAILY_AMOUNT": min_amount, "MIN_SCALE": min_scale, "VOLUME_RATIO": volume_ratio,
        "MACD_CROSS_ENABLED": macd_cross, "VOLUME_FILTER_ENABLED": volume_filter,
        "RSI_FILTER_ENABLED": rsi_filter, "BOLL_FILTER_ENABLED": boll_filter,
        "MONTHLY_MA_ENABLED": monthly_ma, "MACD_ABOVE_ZERO_FILTER": macd_above_zero,
    }
    return config_override, selected_sector


# ==================================================================================
# 图表工具函数
# ==================================================================================
def plot_kline_with_indicators(df: pd.DataFrame, title: str = "K线图",
                                show_macd: bool = True, show_rsi: bool = True,
                                show_boll: bool = True, show_volume: bool = True) -> go.Figure:
    if df is None or len(df) == 0:
        return go.Figure()

    close = df["close"]
    dif, dea, macd_bar = calc_macd(close)
    rsi = calc_rsi(close)
    boll_mid, boll_upper, boll_lower = calc_boll(close)

    subplot_count = 1
    row_heights = [0.5]
    if show_volume and "volume" in df.columns:
        subplot_count += 1
        row_heights.append(0.1)
    if show_macd:
        subplot_count += 1
        row_heights.append(0.15)
    if show_rsi:
        subplot_count += 1
        row_heights.append(0.1)

    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    specs = [[{"secondary_y": False}]]
    if show_volume and "volume" in df.columns:
        specs.append([{"secondary_y": False}])
    if show_macd:
        specs.append([{"secondary_y": False}])
    if show_rsi:
        specs.append([{"secondary_y": False}])

    fig = make_subplots(
        rows=subplot_count, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=row_heights, specs=specs,
    )

    colors_up = '#ef5350'
    colors_down = '#26a69a'

    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color=colors_up, decreasing_line_color=colors_down,
        increasing_fillcolor=colors_up, decreasing_fillcolor=colors_down,
        name="K线",
    ), row=1, col=1)

    if show_boll:
        fig.add_trace(go.Scatter(x=df["date"], y=boll_mid, name="BOLL中轨",
                                  line=dict(color="orange", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=boll_upper, name="BOLL上轨",
                                  line=dict(color="gray", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=boll_lower, name="BOLL下轨",
                                  line=dict(color="gray", width=1, dash="dot"),
                                  fill="tonexty", fillcolor="rgba(128,128,128,0.1)"), row=1, col=1)

    if len(close) >= 20:
        ma20 = calc_ma(close, 20)
        fig.add_trace(go.Scatter(x=df["date"], y=ma20, name="MA20",
                                  line=dict(color="blue", width=1)), row=1, col=1)
    if len(close) >= 60:
        ma60 = calc_ma(close, 60)
        fig.add_trace(go.Scatter(x=df["date"], y=ma60, name="MA60",
                                  line=dict(color="purple", width=1)), row=1, col=1)

    current_row = 1

    if show_volume and "volume" in df.columns:
        current_row += 1
        vol_colors = [colors_up if c >= o else colors_down
                      for c, o in zip(df["close"], df["open"])]
        fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="成交量",
                              marker_color=vol_colors, showlegend=False),
                      row=current_row, col=1)

    if show_macd:
        current_row += 1
        macd_colors = [colors_up if v >= 0 else colors_down for v in macd_bar]
        fig.add_trace(go.Bar(x=df["date"], y=macd_bar, name="MACD柱",
                              marker_color=macd_colors, showlegend=False),
                      row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=dif, name="DIF",
                                  line=dict(color="white", width=1.5)),
                      row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=dea, name="DEA",
                                  line=dict(color="yellow", width=1.5)),
                      row=current_row, col=1)

    if show_rsi:
        current_row += 1
        fig.add_trace(go.Scatter(x=df["date"], y=rsi, name="RSI",
                                  line=dict(color="cyan", width=1.5)),
                      row=current_row, col=1)
        fig.add_hline(y=50, line_dash="dash", line_color="gray",
                      annotation_text="RSI=50", row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=current_row, col=1)

    fig.update_layout(
        title=title, height=600 + (subplot_count - 1) * 100,
        xaxis_rangeslider_visible=False, template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=30),
    )
    return fig


def plot_sector_pie(sector_counts: dict) -> go.Figure:
    fig = go.Figure(data=[go.Pie(
        labels=list(sector_counts.keys()),
        values=list(sector_counts.values()),
        hole=0.4, textinfo="label+percent",
    )])
    fig.update_layout(title="入选ETF板块分布", template="plotly_dark",
                      height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def plot_return_bar(df: pd.DataFrame) -> go.Figure:
    if df is None or len(df) == 0:
        return go.Figure()
    df_sorted = df.sort_values("ret_1m", ascending=True)
    colors = ['#ef5350' if v >= 0 else '#26a69a' for v in df_sorted["ret_1m"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_sorted["name"], x=df_sorted["ret_1m"],
        orientation="h", name="近1月涨幅%",
        marker_color=colors, text=df_sorted["ret_1m"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
    ))
    fig.update_layout(title="入选ETF近1月涨幅排名", template="plotly_dark",
                      height=max(300, len(df_sorted) * 35),
                      xaxis_title="涨幅%", margin=dict(l=150, r=20, t=40, b=30))
    return fig


def plot_backtest_results(trades_df: pd.DataFrame) -> go.Figure:
    if trades_df is None or len(trades_df) == 0:
        return go.Figure()
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("持有1周收益分布", "持有2周收益分布",
                                        "持有4周收益分布", "收益曲线"))
    for i, (col, title) in enumerate([("ret_1周", "1周"), ("ret_2周", "2周"), ("ret_4周", "4周")]):
        if col in trades_df.columns:
            valid = trades_df[col].dropna()
            if len(valid) > 0:
                row = 1 if i < 2 else 2
                c = 1 if i % 2 == 0 else 2
                fig.add_trace(go.Histogram(x=valid, name=title, nbinsx=30,
                                           marker_color='#1f77b4'), row=row, col=c)
    if "signal_date" in trades_df.columns and "ret_1周" in trades_df.columns:
        daily = trades_df.groupby("signal_date")["ret_1周"].mean().sort_index()
        cumulative = daily.cumsum()
        fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative.values,
                                  name="累计收益%", line=dict(color="#ff7f0e", width=2)),
                      row=2, col=2)
    fig.update_layout(template="plotly_dark", height=600, showlegend=False,
                      margin=dict(l=50, r=20, t=50, b=30))
    return fig


# ==================================================================================
# Tab 1: 筛选结果
# ==================================================================================
def tab_results(config_override, sector=None):
    st.header("  筛选结果")

    col1, col2 = st.columns([1, 3])
    with col1:
        run_btn = st.button("  开始筛选", type="primary", use_container_width=True)
    with col2:
        if sector and sector != "全部":
            st.caption(f"当前筛选板块：**{sector}** | MACD + 放量 + RSI + BOLL + 月K趋势")
        else:
            st.caption("当前筛选板块：**全部** | MACD + 放量 + RSI + BOLL + 月K趋势")

    if run_btn:
        progress_bar = st.progress(0, text="准备开始筛选...")

        def progress_cb(step, cur, total):
            pct = cur / max(total, 1)
            progress_bar.progress(pct, text=f"{step}: {cur}/{total}")

        with st.spinner("正在执行筛选..."):
            result = run_scan_engine(config_override, sector=sector,
                                     progress_callback=progress_cb)

        progress_bar.progress(1.0, text="筛选完成！")
        st.session_state["scan_result"] = result

    if "scan_result" in st.session_state:
        result = st.session_state["scan_result"]
        passed = result["passed_etfs"]
        stats = result["stats"]

        st.subheader("  筛选统计")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("参与筛选", stats.get("total", 0))
        c2.metric("MACD通过", stats.get("macd_pass", 0))
        c3.metric("RSI通过", stats.get("rsi_pass", 0))
        c4.metric("BOLL通过", stats.get("boll_pass", 0))
        c5.metric("最终入选", stats.get("final_pass", 0))

        st.markdown("---")

        if not passed:
            st.warning("本次筛选无ETF满足所有条件")
            return

        result_df = pd.DataFrame(passed)
        result_df = result_df.sort_values("ret_1m", ascending=False)

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.subheader("  入选ETF列表")
            display_df = result_df[["code", "name", "sector", "latest_price",
                                     "ret_1m", "ret_3m", "trigger_reasons"]].copy()
            display_df.columns = ["代码", "名称", "板块", "最新价", "近1月%", "近3月%", "触发条件"]
            st.dataframe(display_df, use_container_width=True, height=400)

        with col_right:
            sector_counts = result_df["sector"].value_counts().to_dict()
            if sector_counts:
                st.plotly_chart(plot_sector_pie(sector_counts), use_container_width=True)

        st.subheader("  涨幅排名")
        st.plotly_chart(plot_return_bar(result_df), use_container_width=True)

        today = datetime.now().strftime("%Y%m%d")
        csv_path = os.path.join(CONFIG["RESULTS_DIR"], f"筛选结果_{today}.csv")
        result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        st.info(f"结果已保存至: {csv_path}")

        with st.expander("  运行日志"):
            for msg in result.get("log", []):
                st.text(msg)


# ==================================================================================
# Tab 2: 个股详情（含点击K线分析功能）
# ==================================================================================
def _format_analysis_text(df: pd.DataFrame, idx: int, period: str) -> str:
    """根据K线索引生成技术指标分析报告"""
    close = df["close"]
    dif, dea, macd_bar = calc_macd(close)
    rsi = calc_rsi(close)
    boll_mid, boll_upper, boll_lower = calc_boll(close)
    ma20 = calc_ma(close, 20) if len(close) >= 20 else None
    ma60 = calc_ma(close, 60) if len(close) >= 60 else None

    row = df.iloc[idx]
    date_str = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
    price = row["close"]
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    vol = row.get("volume", None)
    chg = (c / o - 1) * 100 if o > 0 else 0

    # 前一K线数据
    prev_close = df["close"].iloc[idx - 1] if idx > 0 else None
    prev_dif = dif.iloc[idx - 1] if idx > 0 and not pd.isna(dif.iloc[idx - 1]) else None
    prev_dea = dea.iloc[idx - 1] if idx > 0 and not pd.isna(dea.iloc[idx - 1]) else None

    curr_dif = dif.iloc[idx] if not pd.isna(dif.iloc[idx]) else None
    curr_dea = dea.iloc[idx] if not pd.isna(dea.iloc[idx]) else None
    curr_macd = macd_bar.iloc[idx] if not pd.isna(macd_bar.iloc[idx]) else None
    curr_rsi = rsi.iloc[idx] if not pd.isna(rsi.iloc[idx]) else None
    curr_boll_mid = boll_mid.iloc[idx] if not pd.isna(boll_mid.iloc[idx]) else None
    curr_boll_upper = boll_upper.iloc[idx] if not pd.isna(boll_upper.iloc[idx]) else None
    curr_boll_lower = boll_lower.iloc[idx] if not pd.isna(boll_lower.iloc[idx]) else None

    lines = []
    lines.append(f"##   {date_str} ({period}K) 技术指标分析")
    lines.append("")

    # K线形态
    lines.append("###   K线形态")
    body = abs(c - o)
    upper_shadow = h - max(c, o)
    lower_shadow = min(c, o) - l
    amplitude = (h - l) / o * 100 if o > 0 else 0

    if c > o:
        lines.append(f"- **阳线** | 开盘={o:.3f} 收盘={c:.3f} 最高={h:.3f} 最低={l:.3f}")
    else:
        lines.append(f"- **阴线** | 开盘={o:.3f} 收盘={c:.3f} 最高={h:.3f} 最低={l:.3f}")
    lines.append(f"- 涨跌幅: **{chg:+.2f}%** | 振幅: {amplitude:.2f}%")

    if prev_close:
        pct = (c / prev_close - 1) * 100
        lines.append(f"- 相对前一K线: {pct:+.2f}%")

    # K线形态判断
    if body > 0 and lower_shadow > 2 * body and upper_shadow < body * 0.3:
        lines.append("-   **锤子线**（下影线长，可能见底反转）")
    elif body > 0 and upper_shadow > 2 * body and lower_shadow < body * 0.3:
        lines.append("-   **射击之星**（上影线长，可能见顶反转）")
    elif body < (h - l) * 0.1:
        lines.append("-   **十字星**（多空胶着，可能变盘）")
    if vol is not None and idx > 0:
        prev_vol = df["volume"].iloc[idx - 1] if "volume" in df.columns else None
        if prev_vol and prev_vol > 0:
            vol_ratio = vol / prev_vol
            if vol_ratio > 2:
                lines.append(f"-   **明显放量**（量比={vol_ratio:.2f}）")
            elif vol_ratio > 1.5:
                lines.append(f"-   **温和放量**（量比={vol_ratio:.2f}）")
            elif vol_ratio < 0.5:
                lines.append(f"-   **明显缩量**（量比={vol_ratio:.2f}）")
    lines.append("")

    # MACD
    lines.append("###   MACD指标")
    if curr_dif is not None and curr_dea is not None and curr_macd is not None:
        lines.append(f"- DIF: **{curr_dif:.4f}** | DEA: **{curr_dea:.4f}** | MACD柱: **{curr_macd:.4f}**")
        if curr_dif > curr_dea:
            lines.append("- DIF > DEA → **多头排列**")
        else:
            lines.append("- DIF < DEA → **空头排列**")
        if curr_macd > 0:
            lines.append("- MACD红柱 → 多方动能")
        else:
            lines.append("- MACD绿柱 → 空方动能")

        # 金叉/死叉判断
        if prev_dif is not None and prev_dea is not None:
            if curr_dif > curr_dea and prev_dif <= prev_dea:
                lines.append("-   **MACD金叉！**（DIF上穿DEA，看多信号）")
            elif curr_dif < curr_dea and prev_dif >= prev_dea:
                lines.append("-   **MACD死叉！**（DIF下穿DEA，看空信号）")

        # 零轴位置
        if curr_dif > 0 and curr_dea > 0:
            lines.append("- DIF/DEA均在零轴上方 → **强势区域**")
        elif curr_dif < 0 and curr_dea < 0:
            lines.append("- DIF/DEA均在零轴下方 → **弱势区域**")
    else:
        lines.append("- 数据不足，无法计算")
    lines.append("")

    # RSI
    lines.append("###   RSI指标")
    if curr_rsi is not None:
        lines.append(f"- RSI(14): **{curr_rsi:.1f}**")
        if curr_rsi > 80:
            lines.append("-   **超买区间**（>80），注意回调风险")
        elif curr_rsi > 70:
            lines.append("-   **偏强区间**（>70），接近超买")
        elif curr_rsi > 50:
            lines.append("-   **多头区间**（50-70），趋势偏多")
        elif curr_rsi > 30:
            lines.append("-   **空头区间**（30-50），趋势偏空")
        elif curr_rsi > 20:
            lines.append("-   **偏弱区间**（<30），接近超卖")
        else:
            lines.append("-   **超卖区间**（<20），可能存在反弹机会")
    else:
        lines.append("- 数据不足，无法计算")
    lines.append("")

    # BOLL
    lines.append("###   布林带")
    if curr_boll_mid is not None and curr_boll_upper is not None and curr_boll_lower is not None:
        lines.append(f"- 上轨: **{curr_boll_upper:.3f}** | 中轨: **{curr_boll_mid:.3f}** | 下轨: **{curr_boll_lower:.3f}**")
        boll_width = (curr_boll_upper - curr_boll_lower) / curr_boll_mid * 100
        lines.append(f"- 带宽: {boll_width:.2f}%")
        if price >= curr_boll_upper:
            lines.append("-   **触及上轨**，可能面临压力")
        elif price > curr_boll_mid:
            lines.append("- ✅ **中轨上方运行**，趋势偏多")
        elif price > curr_boll_lower:
            lines.append("-   **中轨下方运行**，趋势偏空")
        else:
            lines.append("-   **触及下轨**，可能存在支撑")
        if boll_width < 5:
            lines.append("-   **布林带收窄**，可能即将出现大幅波动")
    else:
        lines.append("- 数据不足，无法计算")
    lines.append("")

    # 均线
    lines.append("###   均线系统")
    ma_checks = []
    if ma20 is not None:
        val = ma20.iloc[idx] if not pd.isna(ma20.iloc[idx]) else None
        if val is not None:
            above = price > val
            lines.append(f"- MA20: **{val:.3f}** ({'站上' if above else '跌破'})")
            ma_checks.append(above)
    if ma60 is not None:
        val = ma60.iloc[idx] if not pd.isna(ma60.iloc[idx]) else None
        if val is not None:
            above = price > val
            lines.append(f"- MA60: **{val:.3f}** ({'站上' if above else '跌破'})")
            ma_checks.append(above)
    if ma20 is not None and ma60 is not None:
        v20 = ma20.iloc[idx] if not pd.isna(ma20.iloc[idx]) else None
        v60 = ma60.iloc[idx] if not pd.isna(ma60.iloc[idx]) else None
        if v20 is not None and v60 is not None:
            if v20 > v60:
                lines.append("- MA20 > MA60 → **均线多头排列**")
            else:
                lines.append("- MA20 < MA60 → **均线空头排列**")
    if not ma_checks:
        lines.append("- 均线数据不足")
    lines.append("")

    # 综合判断
    lines.append("###   综合判断")
    bull_signals = 0
    bear_signals = 0
    if curr_dif is not None and curr_dea is not None:
        if curr_dif > curr_dea: bull_signals += 1
        else: bear_signals += 1
    if curr_rsi is not None:
        if curr_rsi > 50: bull_signals += 1
        else: bear_signals += 1
    if curr_boll_mid is not None:
        if price > curr_boll_mid: bull_signals += 1
        else: bear_signals += 1
    if ma_checks:
        if all(ma_checks): bull_signals += 1
        elif not any(ma_checks): bear_signals += 1

    total = bull_signals + bear_signals
    if total > 0:
        bull_pct = bull_signals / total * 100
        if bull_pct >= 80:
            lines.append(f"-   **强烈看多**（多头信号 {bull_signals}/{total}）")
        elif bull_pct >= 60:
            lines.append(f"-   **偏多**（多头信号 {bull_signals}/{total}）")
        elif bull_pct >= 40:
            lines.append(f"-   **中性**（多头 {bull_signals} / 空头 {bear_signals}）")
        elif bull_pct >= 20:
            lines.append(f"-   **偏空**（空头信号 {bear_signals}/{total}）")
        else:
            lines.append(f"-   **强烈看空**（空头信号 {bear_signals}/{total}）")
    lines.append("")
    lines.append("> ⚠️ 以上分析仅基于技术指标，不构成投资建议。请结合基本面、市场情绪等综合判断。")

    return "\n".join(lines)


def tab_detail():
    st.header("  个股详情")

    col1, col2 = st.columns([1, 2])
    with col1:
        code = st.text_input("ETF代码", "159915", max_chars=6)
    with col2:
        period = st.selectbox("K线周期", ["周", "月"], index=0)

    if st.button("加载K线", type="primary"):
        with st.spinner(f"正在获取 {code} 的K线数据..."):
            try:
                df = fetch_etf_kline(code, period=period)
                if df is not None and len(df) > 0:
                    st.session_state[f"kline_{code}_{period}"] = df
                    st.success(f"获取到 {len(df)} 根K线")
                else:
                    st.error("返回数据为空，请检查ETF代码是否正确")
            except Exception as e:
                st.error(f"获取失败: {type(e).__name__}: {e}")

    cache_key = f"kline_{code}_{period}"
    if cache_key in st.session_state:
        df = st.session_state[cache_key]

        st.subheader("图表配置")
        c1, c2, c3, c4 = st.columns(4)
        show_macd = c1.checkbox("MACD", True)
        show_rsi = c2.checkbox("RSI", True)
        show_boll = c3.checkbox("BOLL", True)
        show_volume = c4.checkbox("成交量", True)

        # 选中的K线索引（通过滑块或点击选择）
        select_key = f"selected_idx_{code}_{period}"
        if select_key not in st.session_state:
            st.session_state[select_key] = len(df) - 1

        fig = plot_kline_with_indicators(df, title=f"{code} {period}K线",
                                          show_macd=show_macd, show_rsi=show_rsi,
                                          show_boll=show_boll, show_volume=show_volume)

        # 使用 on_select 捕获点击事件
        event = st.plotly_chart(fig, use_container_width=True, on_select="rerun",
                                selection_mode=["points"])

        # 处理点击事件：更新选中的K线索引
        if event and hasattr(event, "selection") and event.selection:
            points = event.selection.get("points", [])
            if points:
                point = points[0]
                # 从点击的日期反推K线索引
                clicked_x = point.get("x", None)
                if clicked_x is not None:
                    clicked_date = pd.to_datetime(clicked_x)
                    match = df[df["date"] == clicked_date]
                    if len(match) > 0:
                        st.session_state[select_key] = match.index[0]

        # K线选择器（滑块 + 日期显示）
        st.markdown("---")
        st.subheader("  选择K线进行分析")
        st.caption("点击上方K线图中的蜡烛图可选中该K线，也可用下方滑块选择")

        idx = st.slider(
            "选择K线索引",
            min_value=0, max_value=len(df) - 1,
            value=st.session_state[select_key],
            key=f"slider_{code}_{period}",
            format="%d",
        )
        st.session_state[select_key] = idx

        selected_date = pd.Timestamp(df.iloc[idx]["date"]).strftime("%Y-%m-%d")
        selected_close = df.iloc[idx]["close"]
        st.info(f"已选中: **{selected_date}** | 收盘价: **{selected_close:.3f}**")

        # 分析按钮
        if st.button("  分析该K线技术指标", type="primary", use_container_width=True):
            analysis = _format_analysis_text(df, idx, period)
            st.session_state[f"analysis_{code}_{period}"] = analysis

        # 显示分析结果
        analysis_key = f"analysis_{code}_{period}"
        if analysis_key in st.session_state:
            st.markdown("---")
            st.markdown(st.session_state[analysis_key])

        # 最新指标值
        st.markdown("---")
        st.subheader("  最新指标值")
        close = df["close"]
        dif, dea, macd_bar = calc_macd(close)
        rsi = calc_rsi(close)
        boll_mid, boll_upper, boll_lower = calc_boll(close)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("最新价", f"{close.iloc[-1]:.3f}")
        c2.metric("DIF", f"{dif.iloc[-1]:.4f}")
        c3.metric("DEA", f"{dea.iloc[-1]:.4f}")
        c4.metric("MACD", f"{macd_bar.iloc[-1]:.4f}")
        c5.metric("RSI", f"{rsi.iloc[-1]:.1f}")
        c6.metric("BOLL中轨", f"{boll_mid.iloc[-1]:.3f}")

        st.subheader("  技术面检查")
        wk_check = check_weekly_macd_golden_cross(df)
        vol_check = check_weekly_volume(df)
        rsi_check = check_weekly_rsi(df)
        boll_check = check_weekly_boll(df)

        c1, c2, c3, c4 = st.columns(4)
        c1.success("MACD金叉") if wk_check["pass"] else c1.warning("无MACD金叉")
        c2.success(vol_check["detail"]) if vol_check["pass"] else c2.warning(vol_check["detail"])
        c3.success(rsi_check["detail"]) if rsi_check["pass"] else c3.warning(rsi_check["detail"])
        c4.success("站上BOLL中轨") if boll_check["pass"] else c4.warning("跌破BOLL中轨")

        with st.expander("  原始K线数据"):
            st.dataframe(df.tail(50), use_container_width=True)


# ==================================================================================
# Tab 3: 回测报告
# ==================================================================================
def tab_backtest(config_override):
    st.header("  回测报告")
    st.caption("回测基于纯技术面信号（MACD金叉+放量+RSI），不含基本面筛选")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        start_date = st.date_input("开始日期", datetime(2024, 1, 1))
    with c2:
        end_date = st.date_input("结束日期", datetime.now())
    with c3:
        st.write("")
        st.write("")
        run_bt = st.button("  运行回测", type="primary", use_container_width=True)

    if run_bt:
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        progress_bar = st.progress(0, text="准备回测...")

        def progress_cb(step, cur, total):
            pct = cur / max(total, 1)
            progress_bar.progress(pct, text=f"{step}: {cur}/{total}")

        with st.spinner("正在执行回测..."):
            bt_result = run_backtest_engine(start_str, end_str,
                                             config_override=config_override,
                                             progress_callback=progress_cb)

        progress_bar.progress(1.0, text="回测完成！")
        st.session_state["backtest_result"] = bt_result

    if "backtest_result" in st.session_state:
        bt = st.session_state["backtest_result"]
        trades_df = bt["trades"]
        total = bt["total_signals"]

        if trades_df is None or len(trades_df) == 0:
            st.warning("回测期间无信号触发")
            return

        st.subheader("  回测统计")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总信号次数", total)

        for i, (col, label) in enumerate([("ret_1周", "1周"), ("ret_2周", "2周"), ("ret_4周", "4周")]):
            if col in trades_df.columns:
                valid = trades_df[col].dropna()
                if len(valid) > 0:
                    win_rate = (valid > 0).sum() / len(valid) * 100
                    avg_ret = valid.mean()
                    [c2, c3, c4][i].metric(f"持有{label}胜率", f"{win_rate:.1f}%",
                                            delta=f"均收益 {avg_ret:.2f}%")

        if "max_drawdown_4w" in trades_df.columns:
            valid_dd = trades_df["max_drawdown_4w"].dropna()
            if len(valid_dd) > 0:
                st.metric("4周内平均最大回撤", f"{valid_dd.mean():.2f}%")

        st.markdown("---")

        st.subheader("  收益分布")
        fig = plot_backtest_results(trades_df)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("  交易明细"):
            st.dataframe(trades_df, use_container_width=True, height=400)

        csv_path = os.path.join(
            CONFIG["BACKTEST_DIR"],
            f"回测报告_{datetime.now().strftime('%Y%m%d')}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        )
        trades_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        st.info(f"回测结果已保存至: {csv_path}")


# ==================================================================================
# 主入口
# ==================================================================================
def main():
    st.title("  多周期共振 ETF 右侧交易筛选器")
    st.caption("纯技术指标筛选 | MACD + RSI + BOLL + 均线 + 成交量")

    config_override, selected_sector = render_sidebar()

    tab1, tab2, tab3 = st.tabs(["  筛选结果", "  个股详情", "  回测报告"])

    with tab1:
        tab_results(config_override, sector=selected_sector)
    with tab2:
        tab_detail()
    with tab3:
        tab_backtest(config_override)

    st.markdown("---")
    st.caption("ETF技术面筛选器 v2.0 | 数据源: akshare | 指标参数与同花顺对齐")


if __name__ == "__main__":
    main()
