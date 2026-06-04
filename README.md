<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/A--Shares-ETF-FF4444?style=flat" />
  <img src="https://img.shields.io/badge/CAN--SLIM-Adapted-00C853?style=flat" />
  <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat" />
</p>

<div align="center">

# select_ETF_1

**大A改良版 CAN SLIM + 多周期共振 ETF 右侧交易自动筛选系统**

</div>

<div align="center">

  [Features](#features) | [Architecture](#architecture) | [Quick Start](#quick-start) | [CLI Usage](#cli-usage) | [Streamlit UI](#streamlit-ui) | [Backtest](#backtest) | [Configuration](#configuration) | [Contributing](#contributing)

</div>

---

## Features

- **CAN SLIM 本土化** — 原版 CAN SLIM 选股法针对 A 股 ETF 深度改造，N/S/L/M 条件全量运行，C/A/I 预留接口
- **多周期共振** — 周 K MACD 金叉（右侧启动）+ 月 K 均线趋势确认 + 大盘方向判断
- **技术指标与同花顺对齐** — MACD(12/26/9)、RSI(14)、BOLL(20,2) 全部使用 pandas 实现，参数 100% 对齐
- **纯本地运行** — 零外部服务依赖，数据缓存 24 小时，10 秒内出结果
- **双模式运行** — CLI 命令行脚本 + Streamlit 可视化界面
- **回测引擎** — 历史信号回测，输出胜率、平均收益、最大回撤

## Architecture

```
select_ETF_1/
├── app.py                  # Streamlit 可视化界面（4 个标签页）
├── scanner_engine.py       # 核心筛选引擎（指标计算 + CAN SLIM + 回测）
├── etf_scanner.py          # CLI 命令行脚本（独立可运行）
├── requirements.txt        # 依赖清单
├── cache/                  # K 线数据缓存（24h TTL，自动清理 7 天前数据）
├── results/                # 筛选结果 CSV
└── backtest/               # 回测报告 CSV
```

### 筛选流程

```
全市场 ETF 列表
    │
    ▼
前置过滤（成立时间 / 成交额 / 规模 / 类型）
    │
    ▼
获取周 K + 月 K 数据（缓存 24h）
    │
    ▼
技术面筛选 ──┬── MACD 金叉（零轴过滤）
             ├── 放量确认（量比 ≥ 1.2）
             ├── RSI 多头（RSI > 50）
             ├── BOLL 站上中轨
             └── 月 K 均线趋势向上
    │
    ▼
CAN SLIM 筛选 ──┬── N: 创近 3 月新高 + 板块涨幅前 30%
                ├── S: 规模 20-200 亿 + 份额净增长
                ├── L: 板块跑赢沪深 300 + ETF 跑赢板块
                └── M: 沪深 300 站上 20 周均线 + 创业板 RSI > 40
    │
    ▼
输出结果（控制台 + CSV + 历史对比）
```

## Quick Start

### 安装依赖

```bash
git clone https://github.com/zhangjielll/select_ETF_1.git
cd select_ETF_1

pip install -r requirements.txt
```

> TA-Lib 可选：本项目默认使用 pandas 实现所有指标，无需安装 TA-Lib 即可运行。

### 一键启动 Streamlit

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501` 即可使用可视化界面。

## CLI Usage

```bash
# 正常运行（非周五收盘后会提示确认）
python etf_scanner.py

# 强制运行完整筛选
python etf_scanner.py --force

# 运行回测
python etf_scanner.py --backtest 20240101 20250101
```

输出示例：

```
====================================================================================================
  筛选结果 - 符合条件的ETF
====================================================================================================
  代码       名称     板块    最新价   近1月涨幅%  近3月涨幅%          触发条件
510300  沪深300ETF   宽基    4.123       3.21       8.56   MACD金叉 | RSI=58.3 | N:新高/板块强
159915  创业板ETF    宽基    2.456       5.12      12.34   MACD金叉 | 量比=1.52 | S:供需良好
====================================================================================================
```

## Streamlit UI

Streamlit 界面包含 4 个标签页：

###  筛选结果
- 一键运行完整筛选流程
- 实时进度条 + 统计指标面板
- 入选 ETF 表格 + 板块分布饼图 + 涨幅排名柱状图

###  个股详情
- 输入任意 ETF 代码查看周 K / 月 K 线
- 可切换 MACD / RSI / BOLL / 成交量副图
- 实时显示最新指标值 + 技术面检查结果

###  大盘趋势
- 沪深 300 / 创业板指 K 线图
- M 条件实时判断（多头 / 空头）
- 板块涨幅排名

###  回测报告
- 选择回测日期范围
- 胜率 / 平均收益 / 最大回撤统计
- 收益分布直方图 + 累计收益曲线

## Backtest

回测基于纯技术面信号（MACD 金叉 + 放量 + RSI），不含基本面筛选：

```python
from scanner_engine import run_backtest_engine

result = run_backtest_engine("20240101", "20250101")
trades_df = result["trades"]

# 统计
win_rate = (trades_df["ret_1周"] > 0).sum() / len(trades_df) * 100
print(f"1 周持有胜率: {win_rate:.1f}%")
```

| 指标 | 说明 |
|------|------|
| `ret_1周` | 持有 1 周收益率 |
| `ret_2周` | 持有 2 周收益率 |
| `ret_4周` | 持有 4 周收益率 |
| `max_drawdown_4w` | 持有 4 周内最大回撤 |

## Configuration

所有可调参数集中在 `scanner_engine.py` 的 `CONFIG` 字典中，Streamlit 界面也可在侧边栏实时调节：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MACD_FAST / SLOW / SIGNAL` | 12 / 26 / 9 | MACD 参数（与同花顺对齐） |
| `RSI_PERIOD` | 14 | RSI 周期 |
| `BOLL_PERIOD / STD` | 20 / 2 | 布林带参数 |
| `MIN_DAILY_AMOUNT` | 1000 万 | 最小日均成交额 |
| `MIN_SCALE` | 5 亿 | 最小 ETF 规模 |
| `VOLUME_RATIO` | 1.2 | 放量倍数阈值 |
| `N_NEW_HIGH_MONTHS` | 3 | N 条件：近 N 月新高 |
| `S_SCALE_MIN / MAX` | 20 / 200 亿 | S 条件：规模区间 |
| `M_CSI300_MA_WEEKS` | 20 | M 条件：沪深 300 均线周数 |

### 启用占位条件

CAN SLIM 中 C/A/I 条件为占位状态，需补充数据源后启用：

```python
CONFIG["C_ENABLED"] = True   # 需提供 ETF 跟踪指数的季度扣非净利润
CONFIG["A_ENABLED"] = True   # 需提供过去 3 年年度净利润复合增长率
CONFIG["I_ENABLED"] = True   # 需提供 ETF 机构持仓比例数据
```

## Data Source

| 数据 | 来源 | 说明 |
|------|------|------|
| ETF 列表 | `akshare.fund_etf_spot_em()` | 全市场场内 ETF 实时行情 |
| ETF K 线 | `akshare.fund_etf_hist_em()` | 周 K / 月 K 后复权数据 |
| 指数 K 线 | `akshare.index_zh_a_hist()` | 沪深 300 / 创业板指 |

数据自动缓存到 `cache/` 目录，有效期 24 小时。

## Contributing

欢迎提交 Issue 和 Pull Request：

- 新增 CAN SLIM 条件的数据源接入
- 优化指标计算性能
- 增加新的技术指标
- 改进 Streamlit 可视化

## License

MIT License

## Disclaimer

本项目仅供学习和研究使用，不构成任何投资建议。股市有风险，投资需谨慎。
