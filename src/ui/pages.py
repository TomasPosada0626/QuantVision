from __future__ import annotations

from datetime import datetime
from typing import Callable

import pandas as pd
import streamlit as st

from analytics.event_tracker import AnalyticsEvent, EventTracker
from analytics.experimentation import ExperimentationService
from integrations.webhooks import WebhookNotifier, WebhookPayload
from services.alerts_service import AlertRule, AlertsService
from services.anomaly_lab_service import run_anomaly_methods
from services.auth_service import AuthService
from services.backtesting_service import BacktestingService
from services.ml_predictor_service import MLPredictorService
from services.performance_service import get_async_job_result, paginate_dataframe, submit_async_job
from services.portfolio_service import PortfolioService, PositionInput
from services.reports_service import ReportsService
from services.risk_analytics_service import summarize_risk
from services.strategy_governance_service import StrategyGovernanceService, StrategyProposal
from services.watchlist_service import WatchlistInput, WatchlistService
from ui.charts import (
    build_anomaly_chart,
    build_candlestick_chart,
    build_comparison_chart,
    build_price_chart,
    build_terminal_multiview,
    build_volume_chart,
)


def render_dashboard_page(market_data: dict[str, pd.DataFrame], focus_ticker: str) -> None:
    st.subheader("Professional Market Dashboard")
    if focus_ticker not in market_data:
        st.info("Load market data from the sidebar to render dashboard metrics.")
        return

    df = market_data[focus_ticker].dropna(subset=["Close"]).copy()
    if df.empty:
        st.warning("No close price data available for this ticker.")
        return

    current_price = float(df["Close"].iloc[-1])
    previous_price = float(df["Close"].iloc[-2]) if len(df) > 1 else current_price
    daily_change_pct = ((current_price / previous_price) - 1) * 100 if previous_price else 0.0
    market_cap_proxy = current_price * float(df["Volume"].tail(20).mean())
    pe_proxy = max(0.0, current_price / max(0.5, float(df["Close"].tail(252).mean()) * 0.08))
    dividend_yield_proxy = max(
        0.0,
        min(8.0, float(df["Close"].pct_change().rolling(252).mean().iloc[-1] * 100)),
    )
    beta_proxy = 1.0 + float(df["Return"].rolling(63).std().fillna(0).iloc[-1]) * 5
    high_52 = float(df["Close"].tail(252).max())
    low_52 = float(df["Close"].tail(252).min())

    row1 = st.columns(4)
    row1[0].metric("Price", f"${current_price:,.2f}", f"{daily_change_pct:.2f}%")
    row1[1].metric("Volume", f"{int(df['Volume'].iloc[-1]):,}")
    row1[2].metric("Market Cap (proxy)", f"${market_cap_proxy:,.0f}")
    row1[3].metric("P/E Ratio (proxy)", f"{pe_proxy:.2f}")

    row2 = st.columns(4)
    row2[0].metric("Dividend Yield (proxy)", f"{dividend_yield_proxy:.2f}%")
    row2[1].metric("Beta (proxy)", f"{beta_proxy:.2f}")
    row2[2].metric("52W High", f"${high_52:,.2f}")
    row2[3].metric("52W Low", f"${low_52:,.2f}")

    st.plotly_chart(build_candlestick_chart(df, focus_ticker), width="stretch")
    st.plotly_chart(build_volume_chart(df, focus_ticker), width="stretch")
    st.plotly_chart(build_terminal_multiview(df, focus_ticker), width="stretch")


def render_anomalies_page(
    market_data: dict[str, pd.DataFrame],
    selected_methods: list[str],
    params: dict[str, float | int],
) -> None:
    st.subheader("Machine Learning Anomaly Detection Lab")
    if not market_data:
        st.info("Load market data from the sidebar to run anomaly detection.")
        return

    tabs = st.tabs(list(market_data.keys()))
    for idx, (ticker, raw_df) in enumerate(market_data.items()):
        with tabs[idx]:
            df = raw_df.copy()
            modeled, points, benchmark = run_anomaly_methods(
                df=df,
                selected_methods=selected_methods,
                zscore_threshold=float(params["zscore_threshold"]),
                iforest_contamination=float(params["iforest_contamination"]),
                dbscan_eps=float(params["dbscan_eps"]),
                dbscan_min_samples=int(params["dbscan_min_samples"]),
                rolling_window=int(params["rolling_window"]),
                quantile_low=float(params["quantile_low"]),
                quantile_high=float(params["quantile_high"]),
                lof_neighbors=int(params["lof_neighbors"]),
                ocsvm_nu=float(params["ocsvm_nu"]),
            )
            fig_price, y_data = build_price_chart(modeled, ticker)
            st.plotly_chart(fig_price, width="stretch")

            fig_anomaly = build_anomaly_chart(modeled, points, y_data)
            st.plotly_chart(fig_anomaly, width="stretch")

            st.markdown("### Benchmark")
            st.dataframe(benchmark, width="stretch")
            st.download_button(
                f"Export {ticker} anomalies CSV",
                modeled.to_csv().encode("utf-8"),
                f"{ticker}_quantvision_anomalies.csv",
            )


def render_comparison_page(market_data: dict[str, pd.DataFrame]) -> None:
    """Render comparative analytics across multiple tickers.

    Args:
        market_data: Mapping of ticker symbols to modeled data frames.
    """
    st.subheader("Asset Comparison")
    if len(market_data) < 2:
        st.info("Select at least two assets to enable comparative analytics.")
        return

    compare_df = pd.DataFrame({ticker: df["Close"] for ticker, df in market_data.items()}).dropna()
    returns = compare_df.pct_change().dropna()
    cumulative = (1 + returns).cumprod() - 1
    drawdowns = compare_df / compare_df.cummax() - 1

    summary_rows: list[dict[str, float | str]] = []
    for ticker in compare_df.columns:
        ticker_returns = returns[ticker]
        summary_rows.append(
            {
                "Ticker": ticker,
                "Return %": float(cumulative[ticker].iloc[-1] * 100),
                "Volatility %": float(ticker_returns.std(ddof=0) * (252**0.5) * 100),
                "Sharpe": float(
                    (ticker_returns.mean() / ticker_returns.std(ddof=0) * (252**0.5))
                    if ticker_returns.std(ddof=0) > 0
                    else 0
                ),
                "Max Drawdown %": float(drawdowns[ticker].min() * 100),
            }
        )

    st.dataframe(pd.DataFrame(summary_rows), width="stretch")
    st.plotly_chart(
        build_comparison_chart(compare_df, "Multi-Asset Price Comparison"),
        width="stretch",
    )
    st.plotly_chart(
        build_comparison_chart(cumulative, "Cumulative Return Comparison"),
        width="stretch",
    )
    st.markdown("### Correlation Matrix")
    st.dataframe(returns.corr(), width="stretch")


def render_backtesting_page(
    market_data: dict[str, pd.DataFrame],
    backtesting_service: BacktestingService,
) -> None:
    """Render strategy backtesting workspace.

    Args:
        market_data: Mapping of ticker symbols to modeled data frames.
        backtesting_service: Service used to execute strategy simulation.
    """
    st.subheader("Backtesting Engine")
    if not market_data:
        st.info("Load market data first.")
        return

    ticker = st.selectbox("Ticker", options=list(market_data.keys()), key="backtest_ticker")
    df = market_data[ticker].copy()
    if df.empty:
        return

    df["buy_signal"] = (df["RSI_14"] < 30) | (df["Anomaly_rolling_quantile_base"])
    df["sell_signal"] = (df["RSI_14"] > 70) | (
        (df["MACD"].shift(1) >= df["MACD_Signal"].shift(1)) & (df["MACD"] < df["MACD_Signal"])
    )
    results = backtesting_service.run_simple_strategy(df, "buy_signal", "sell_signal")

    cols = st.columns(5)
    cols[0].metric("Return", f"{results['Return %']:.2f}%")
    cols[1].metric("Trades", f"{int(results['Trades'])}")
    cols[2].metric("Win Rate", f"{results['Win Rate %']:.2f}%")
    cols[3].metric("Buy & Hold", f"{results['Buy & Hold %']:.2f}%")
    cols[4].metric("Max Drawdown", f"{results['Max Drawdown %']:.2f}%")


def render_risk_page(market_data: dict[str, pd.DataFrame], focus_ticker: str) -> None:
    """Render risk analytics page.

    Args:
        market_data: Mapping of ticker symbols to modeled data frames.
        focus_ticker: Selected ticker for risk summary.
    """
    st.subheader("Risk Analytics")
    if focus_ticker not in market_data:
        st.info("Load market data first.")
        return

    asset_df = market_data[focus_ticker]
    benchmark_options = [ticker for ticker in market_data.keys() if ticker != focus_ticker]
    benchmark_name = (
        st.selectbox("Benchmark", options=benchmark_options, index=0) if benchmark_options else ""
    )
    benchmark_returns = (
        market_data[benchmark_name]["Return"]
        if benchmark_name and benchmark_name in market_data
        else None
    )
    risk = summarize_risk(asset_df["Return"], benchmark_returns=benchmark_returns)
    st.dataframe(pd.DataFrame([risk]), width="stretch")


def render_portfolio_page(
    market_data: dict[str, pd.DataFrame],
    username: str,
    portfolio_service: PortfolioService,
    csrf_token_value: str,
    is_valid_csrf: Callable[[str], bool],
) -> None:
    """Render portfolio management page.

    Args:
        market_data: Mapping of ticker symbols to market frames.
        username: Current authenticated username.
        portfolio_service: Portfolio domain service.
        csrf_token_value: Session CSRF token for the form.
        is_valid_csrf: Callback to validate user token input.
    """
    st.subheader("Portfolio Tracker")
    with st.form("portfolio_add_form"):
        security_token = st.text_input("Security Token", value=csrf_token_value, type="password")
        cols = st.columns(4)
        ticker = cols[0].text_input("Ticker", value="AAPL").upper().strip()
        quantity = cols[1].number_input("Quantity", min_value=0.0, value=10.0, step=1.0)
        buy_price = cols[2].number_input("Buy Price", min_value=0.0, value=100.0, step=0.5)
        buy_date = cols[3].date_input("Buy Date", value=datetime.today()).isoformat()
        submitted = st.form_submit_button("Add Position")
        if submitted and not is_valid_csrf(security_token):
            st.error("Security token mismatch. Reload and try again.")
        elif submitted and ticker and quantity > 0 and buy_price > 0:
            portfolio_service.add_position(
                PositionInput(
                    username=username,
                    ticker=ticker,
                    quantity=float(quantity),
                    buy_price=float(buy_price),
                    buy_date=buy_date,
                )
            )
            st.success("Position saved.")

    positions = portfolio_service.list_positions(username)
    st.dataframe(positions, width="stretch")

    latest_prices: dict[str, float] = {}
    for ticker in positions.get("ticker", pd.Series(dtype=str)).unique().tolist():
        if ticker in market_data and not market_data[ticker].empty:
            latest_prices[ticker] = float(market_data[ticker]["Close"].iloc[-1])
    metrics = portfolio_service.compute_portfolio_metrics(username, latest_prices)

    cols = st.columns(4)
    cols[0].metric("Invested Capital", f"${metrics['Invested Capital']:,.2f}")
    cols[1].metric("Current Value", f"${metrics['Current Value']:,.2f}")
    cols[2].metric("PnL", f"${metrics['PnL']:,.2f}")
    cols[3].metric("ROI", f"{metrics['ROI %']:.2f}%")


def render_watchlists_page(
    username: str,
    watchlist_service: WatchlistService,
    csrf_token_value: str,
    is_valid_csrf: Callable[[str], bool],
) -> None:
    """Render watchlist management page."""
    st.subheader("Watchlists")
    with st.form("watchlist_create_form"):
        security_token = st.text_input("Security Token", value=csrf_token_value, type="password")
        name = st.text_input("Watchlist name", value="Technology")
        create = st.form_submit_button("Create / Get")
        if create and not is_valid_csrf(security_token):
            st.error("Security token mismatch. Reload and try again.")
        elif create and name.strip():
            watchlist_id = watchlist_service.create_watchlist(
                WatchlistInput(username=username, name=name)
            )
            st.session_state["active_watchlist_id"] = watchlist_id
            st.success(f"Watchlist ready: {name}")

    watchlists = paginate_dataframe(
        watchlist_service.list_watchlists(username),
        limit=50,
        sort_by="id",
        descending=True,
    )
    st.dataframe(watchlists, width="stretch")
    if watchlists.empty:
        return

    selected_id = int(
        st.selectbox(
            "Active watchlist",
            options=watchlists["id"].tolist(),
            index=0,
            key="watchlist_id",
        )
    )
    items = watchlist_service.list_items(selected_id)
    st.write("Tickers", ", ".join(items["ticker"].tolist()) if not items.empty else "(empty)")

    col_a, col_b = st.columns(2)
    with col_a:
        ticker_to_add = st.text_input("Ticker to add", value="MSFT").upper().strip()
        if st.button("Add ticker") and ticker_to_add:
            watchlist_service.add_ticker(selected_id, ticker_to_add)
            st.rerun()
    with col_b:
        ticker_to_remove = st.text_input("Ticker to remove", value="").upper().strip()
        if st.button("Remove ticker") and ticker_to_remove:
            watchlist_service.remove_ticker(selected_id, ticker_to_remove)
            st.rerun()


def _evaluate_alert_conditions(
    df: pd.DataFrame,
    rule_type: str,
    threshold: float | None,
) -> tuple[bool, str]:
    """Evaluate whether an alert rule is triggered for the latest observation."""
    if len(df) < 3:
        return False, "insufficient data"

    current = df.iloc[-1]
    previous = df.iloc[-2]

    if rule_type == "anomaly_detected":
        triggered = bool(current.get("Anomaly_rolling_quantile_base", False))
        return triggered, "Rolling quantile anomaly detected"
    if rule_type == "rsi_gt_70":
        triggered = float(current.get("RSI_14", 0)) > 70
        return triggered, f"RSI at {current.get('RSI_14', 0):.2f}"
    if rule_type == "rsi_lt_30":
        triggered = float(current.get("RSI_14", 100)) < 30
        return triggered, f"RSI at {current.get('RSI_14', 0):.2f}"
    if rule_type == "macd_crossover":
        triggered = float(previous.get("MACD", 0)) <= float(
            previous.get("MACD_Signal", 0)
        ) and float(current.get("MACD", 0)) > float(current.get("MACD_Signal", 0))
        return triggered, "MACD bullish crossover"
    if rule_type == "ema_crossover":
        triggered = float(previous.get("EMA_20", 0)) <= float(previous.get("SMA_20", 0)) and float(
            current.get("EMA_20", 0)
        ) > float(current.get("SMA_20", 0))
        return triggered, "EMA crossed above SMA"
    if rule_type == "price_change_pct":
        target = float(threshold if threshold is not None else 5)
        change = ((float(current["Close"]) / float(previous["Close"])) - 1) * 100
        triggered = abs(change) >= target
        return triggered, f"Price changed {change:.2f}%"
    if rule_type == "new_high":
        triggered = float(current["Close"]) >= float(df["Close"].tail(252).max())
        return triggered, "New 52-week high"
    if rule_type == "new_low":
        triggered = float(current["Close"]) <= float(df["Close"].tail(252).min())
        return triggered, "New 52-week low"
    return False, "rule not supported"


def render_alerts_page(
    market_data: dict[str, pd.DataFrame],
    username: str,
    alerts_service: AlertsService,
    csrf_token_value: str,
    is_valid_csrf: Callable[[str], bool],
) -> None:
    """Render smart alerts page."""
    st.subheader("Smart Alerts")
    rule_types = [
        "anomaly_detected",
        "rsi_gt_70",
        "rsi_lt_30",
        "macd_crossover",
        "ema_crossover",
        "price_change_pct",
        "new_high",
        "new_low",
    ]
    with st.form("alert_rule_form"):
        security_token = st.text_input("Security Token", value=csrf_token_value, type="password")
        cols = st.columns(3)
        ticker = cols[0].text_input("Ticker", value="AAPL").upper().strip()
        rule_type = cols[1].selectbox("Rule Type", options=rule_types)
        threshold = cols[2].number_input("Threshold (optional)", value=5.0)
        create_rule = st.form_submit_button("Create Rule")
        if create_rule and not is_valid_csrf(security_token):
            st.error("Security token mismatch. Reload and try again.")
        elif create_rule and ticker:
            threshold_value = threshold if rule_type == "price_change_pct" else None
            alerts_service.create_rule(
                AlertRule(
                    username=username,
                    ticker=ticker,
                    alert_type=rule_type,
                    threshold=threshold_value,
                    active=True,
                )
            )
            st.success("Rule created.")

    rules = alerts_service.list_rules(username)
    st.dataframe(rules, width="stretch")

    if st.button("Evaluate Active Rules"):
        for _, row in rules[rules["active"] == 1].iterrows():
            ticker = str(row["ticker"]).upper()
            if ticker not in market_data:
                continue
            triggered, message = _evaluate_alert_conditions(
                market_data[ticker],
                str(row["alert_type"]),
                row.get("threshold"),
            )
            if triggered:
                alerts_service.emit_alert(username, ticker, str(row["alert_type"]), message)
        st.success("Rule evaluation completed.")

    st.markdown("### Alert History")
    history = paginate_dataframe(
        alerts_service.list_history(username),
        limit=50,
        sort_by="triggered_at",
        descending=True,
    )
    st.dataframe(history, width="stretch")


def render_ai_lab_page(
    market_data: dict[str, pd.DataFrame],
    ml_predictor_service: MLPredictorService,
    event_tracker: EventTracker,
    username: str,
) -> None:
    """Render AI lab with prediction and drift analysis."""
    st.subheader("AI Lab")
    if not market_data:
        st.info("Load market data first.")
        return

    ticker = st.selectbox("AI ticker", options=list(market_data.keys()), key="ai_ticker")
    horizon = st.slider("Prediction horizon (days)", min_value=1, max_value=30, value=5)
    frame = market_data[ticker]

    prediction = ml_predictor_service.predict_next_close(frame["Close"], horizon=horizon)
    drift = ml_predictor_service.detect_factor_drift(frame["Return"].fillna(0.0))

    cols = st.columns(3)
    cols[0].metric("Current Close", f"${prediction['current_close']:.2f}")
    cols[1].metric("Predicted Close", f"${prediction['predicted_close']:.2f}")
    cols[2].metric("Expected Change", f"{prediction['expected_change_pct']:.2f}%")

    st.markdown("### Drift Monitoring")
    drift_cols = st.columns(4)
    drift_cols[0].metric("Recent Mean", f"{float(drift['recent_mean']):.6f}")
    drift_cols[1].metric("Baseline Mean", f"{float(drift['baseline_mean']):.6f}")
    drift_cols[2].metric("Baseline Std", f"{float(drift['baseline_std']):.6f}")
    drift_cols[3].metric("Z-Score", f"{float(drift['z_score']):.3f}")

    if bool(drift["drift_detected"]):
        st.warning("Drift detected: recent distribution differs from baseline.")
    else:
        st.success("No significant drift detected.")

    event_tracker.track(
        AnalyticsEvent(
            username=username,
            feature="ai_lab",
            event_name="run_ml_prediction",
            metadata=f"ticker={ticker};horizon={horizon}",
        )
    )


def render_governance_page(
    governance_service: StrategyGovernanceService,
    webhook_notifier: WebhookNotifier,
    username: str,
    csrf_token_value: str,
    is_valid_csrf: Callable[[str], bool],
) -> None:
    """Render governance approval and webhook page."""
    st.subheader("Strategy Governance")
    with st.form("governance_form"):
        security_token = st.text_input("Security Token", value=csrf_token_value, type="password")
        strategy_name = st.text_input("Strategy name", value="Momentum Plus")
        rationale = st.text_area(
            "Rationale", value="Add anomaly filters to reduce false positives."
        )
        submitted = st.form_submit_button("Submit proposal")
        if submitted and not is_valid_csrf(security_token):
            st.error("Security token mismatch. Reload and try again.")
        elif submitted:
            proposal_id = governance_service.submit_proposal(
                StrategyProposal(
                    strategy_name=strategy_name,
                    created_by=username or "unknown",
                    rationale=rationale,
                )
            )
            st.success(f"Proposal submitted: {proposal_id}")

    proposals = governance_service.list_proposals(limit=200)
    st.dataframe(proposals, width="stretch")
    if proposals.empty:
        return

    proposal_id = int(st.selectbox("Proposal ID", options=proposals["id"].tolist(), key="gov_id"))
    col_a, col_b = st.columns(2)
    if col_a.button("Approve"):
        governance_service.approve_proposal(proposal_id, approved_by=username or "admin")
        st.rerun()
    if col_b.button("Reject"):
        governance_service.reject_proposal(proposal_id, approved_by=username or "admin")
        st.rerun()

    st.markdown("### Webhook Dispatch")
    webhook_url = st.text_input("Webhook URL", value="https://example.com/hook")
    webhook_message = st.text_input("Webhook message", value="Strategy governance update")
    if st.button("Send Webhook Notification"):
        try:
            result = webhook_notifier.send(
                webhook_url,
                WebhookPayload(
                    event="governance_update",
                    message=webhook_message,
                    source="quantvision_ui",
                ),
            )
            st.success(f"Webhook dispatched (status={result['status']})")
        except Exception as exc:
            st.error(f"Webhook failed: {exc}")


def render_analytics_dashboard_page(
    event_tracker: EventTracker,
    experimentation_service: ExperimentationService,
    username: str,
) -> None:
    """Render analytics dashboard and A/B experimentation controls."""
    st.subheader("Usage Analytics")
    events = event_tracker.list_events(limit=500)
    st.dataframe(events, width="stretch")

    top = event_tracker.top_features(limit=10)
    st.markdown("### Top Features")
    st.dataframe(top, width="stretch")

    funnel = event_tracker.funnel()
    st.markdown("### Funnel")
    funnel_steps = [
        "login_success",
        "load_market_data",
        "run_anomaly_methods",
        "export_report",
    ]
    funnel_rows: list[dict[str, str | int | float]] = []
    previous_count: int | None = None
    for step in funnel_steps:
        count = int(funnel.get(step, 0))
        if previous_count is None:
            conversion_from_previous = 100.0 if count > 0 else 0.0
        elif previous_count == 0:
            conversion_from_previous = 0.0
        else:
            conversion_from_previous = (count / previous_count) * 100

        funnel_rows.append(
            {
                "Step": step,
                "Events": count,
                "Conversion From Previous %": float(conversion_from_previous),
            }
        )
        previous_count = count

    funnel_df = pd.DataFrame(funnel_rows)
    st.dataframe(funnel_df, width="stretch")
    st.bar_chart(funnel_df.set_index("Step")["Events"], width="stretch")

    st.markdown("### A/B Experimentation")
    with st.form("ab_create_form"):
        exp_name = st.text_input("Experiment name", value="dashboard_cta_v1")
        exp_feature = st.text_input("Feature", value="dashboard")
        exp_variants = st.text_input("Variants (comma separated)", value="control,treatment")
        exp_hypothesis = st.text_area("Hypothesis", value="Treatment increases click-through rate")
        create_exp = st.form_submit_button("Create / Update Experiment")
        if create_exp:
            variants = [item.strip() for item in exp_variants.split(",") if item.strip()]
            experimentation_service.create_experiment(
                name=exp_name,
                feature=exp_feature,
                variants=variants,
                hypothesis=exp_hypothesis,
            )
            st.success(f"Experiment ready: {exp_name}")

    experiments = experimentation_service.list_experiments()
    st.dataframe(experiments, width="stretch")
    if experiments.empty:
        return

    selected_experiment = st.selectbox("Experiment", options=experiments["name"].tolist())
    username_input = st.text_input("Username for assignment/conversion", value=username or "guest")
    col_a, col_b = st.columns(2)
    if col_a.button("Assign Variant"):
        variant = experimentation_service.assign_variant(selected_experiment, username_input)
        event_tracker.track(
            AnalyticsEvent(
                username=username_input,
                feature="experimentation",
                event_name="ab_exposure",
                metadata=f"experiment={selected_experiment};variant={variant}",
            )
        )
        st.info(f"Assigned variant: {variant}")
    if col_b.button("Mark Conversion"):
        experimentation_service.track_conversion(selected_experiment, username_input)
        event_tracker.track(
            AnalyticsEvent(
                username=username_input,
                feature="experimentation",
                event_name="ab_conversion",
                metadata=f"experiment={selected_experiment}",
            )
        )
        st.success("Conversion marked")

    st.markdown("### Experiment Summary")
    st.dataframe(experimentation_service.summary(selected_experiment), width="stretch")


def render_reports_page(
    market_data: dict[str, pd.DataFrame],
    username: str,
    reports_service: ReportsService,
) -> None:
    """Render report generation and export center."""
    st.subheader("Reports Center")
    if not market_data:
        st.info("Load market data first to generate reports.")
        return

    ticker = st.selectbox("Report ticker", options=list(market_data.keys()), key="report_ticker")
    df = market_data[ticker]
    if df.empty:
        st.warning("Selected ticker has no data.")
        return

    risk_summary = summarize_risk(df["Return"])
    kpis = {
        "Current Price": float(df["Close"].iloc[-1]),
        "Average Volume": float(df["Volume"].tail(20).mean()),
        "Return 1Y %": (
            float(((df["Close"].iloc[-1] / df["Close"].tail(252).iloc[0]) - 1) * 100)
            if len(df) >= 252
            else float(((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1) * 100)
        ),
        "Volatility": float(risk_summary["Volatility"]),
        "Sharpe": float(risk_summary["Sharpe Ratio"]),
    }
    benchmark = pd.DataFrame([risk_summary])

    pdf_bytes = reports_service.build_executive_report(
        title=f"QuantVision Executive Report | {ticker} | {username}",
        kpis=kpis,
        benchmark=benchmark,
    )
    csv_bytes = reports_service.to_csv_bytes(df.tail(500))
    png_bytes = reports_service.to_png_bytes(build_candlestick_chart(df, ticker))

    st.dataframe(pd.DataFrame([kpis]), width="stretch")
    if st.button("Generate PDF Asynchronously"):
        job_id = f"report_{username}_{ticker}_{int(datetime.now().timestamp())}"
        submit_async_job(
            job_id,
            reports_service.build_executive_report,
            f"QuantVision Executive Report | {ticker} | {username}",
            kpis,
            benchmark,
        )
        st.session_state["report_job_id"] = job_id
        st.info(f"Report job queued: {job_id}")

    active_job_id = st.session_state.get("report_job_id", "")
    if active_job_id:
        status, payload = get_async_job_result(active_job_id)
        st.caption(f"Async report job status: {status}")
        if status == "completed" and isinstance(payload, (bytes, bytearray)):
            st.download_button(
                "Download Async Executive Report (PDF)",
                data=payload,
                file_name=f"{ticker}_executive_report_async.pdf",
                mime="application/pdf",
            )
    st.download_button(
        "Download Executive Report (PDF)",
        data=pdf_bytes,
        file_name=f"{ticker}_executive_report.pdf",
        mime="application/pdf",
    )
    st.download_button(
        "Download Technical Dataset (CSV)",
        data=csv_bytes,
        file_name=f"{ticker}_technical_report.csv",
        mime="text/csv",
    )
    if png_bytes:
        st.download_button(
            "Download Chart Snapshot (PNG)",
            data=png_bytes,
            file_name=f"{ticker}_chart_snapshot.png",
            mime="image/png",
        )


def render_admin_page(auth_service: AuthService) -> None:
    """Render administration workspace for role management."""
    st.subheader("Administration")
    users = auth_service.list_users()
    if not users:
        st.info("No users found in local auth database.")
        return

    users_df = pd.DataFrame(users, columns=["username", "email", "role"])
    st.dataframe(users_df, width="stretch")

    selected_username = st.selectbox("User", options=users_df["username"].tolist())
    selected_role = st.selectbox("Role", options=["ADMIN", "ANALYST", "GUEST"])
    if st.button("Update Role"):
        updated = auth_service.set_user_role(selected_username, selected_role)
        if updated:
            st.success(f"Role updated: {selected_username} -> {selected_role}")
            if st.session_state.get("username") == selected_username:
                st.session_state["role"] = selected_role
        else:
            st.error("Role update failed.")
