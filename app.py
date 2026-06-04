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
    pre_filter_etf_list, classify_sector,
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
    return config_override


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
def tab_results(config_override):
    st.header("  筛选结果")

    col1, col2 = st.columns([1, 3])
    with col1:
        run_btn = st.button("  开始筛选", type="primary", use_container_width=True)
    with col2:
        st.caption("点击运行纯技术面筛选流程（MACD + 放量 + RSI + BOLL + 月K趋势）")

    if run_btn:
        progress_bar = st.progress(0, text="准备开始筛选...")

        def progress_cb(step, cur, total):
            pct = cur / max(total, 1)
            progress_bar.progress(pct, text=f"{step}: {cur}/{total}")

        with st.spinner("正在执行筛选..."):
            result = run_scan_engine(config_override, progress_callback=progress_cb)

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
# Tab 2: 个股详情
# ==================================================================================
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

        fig = plot_kline_with_indicators(df, title=f"{code} {period}K线",
                                          show_macd=show_macd, show_rsi=show_rsi,
                                          show_boll=show_boll, show_volume=show_volume)
        st.plotly_chart(fig, use_container_width=True)

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

    config_override = render_sidebar()

    tab1, tab2, tab3 = st.tabs(["  筛选结果", "  个股详情", "  回测报告"])

    with tab1:
        tab_results(config_override)
    with tab2:
        tab_detail()
    with tab3:
        tab_backtest(config_override)

    st.markdown("---")
    st.caption("ETF技术面筛选器 v2.0 | 数据源: akshare | 指标参数与同花顺对齐")


if __name__ == "__main__":
    main()
