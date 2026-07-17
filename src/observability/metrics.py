from __future__ import annotations

from typing import Callable, TypeVar, cast

from prometheus_client import REGISTRY, Counter, Gauge, Histogram

_MetricT = TypeVar("_MetricT")


def _get_or_create_metric(name: str, factory: Callable[[], _MetricT]) -> _MetricT:
    """Return an existing metric when already registered in the process registry."""
    try:
        return factory()
    except ValueError as exc:
        if "Duplicated timeseries" not in str(exc):
            raise
        existing = getattr(REGISTRY, "_names_to_collectors", {}).get(name)
        if existing is None:
            raise
        return cast(_MetricT, existing)

anomalies_detected_total = _get_or_create_metric(
    "quantvision_anomalies_detected_total",
    lambda: Counter(
        "quantvision_anomalies_detected_total",
        "Total anomalies detected",
        ["method", "ticker"],
    ),
)

logins_total = _get_or_create_metric(
    "quantvision_logins_total",
    lambda: Counter(
        "quantvision_logins_total",
        "Total login attempts",
        ["success"],
    ),
)

anomaly_detection_duration = _get_or_create_metric(
    "quantvision_anomaly_detection_duration_seconds",
    lambda: Histogram(
        "quantvision_anomaly_detection_duration_seconds",
        "Time to detect anomalies",
        ["method"],
    ),
)

active_sessions = _get_or_create_metric(
    "quantvision_active_sessions",
    lambda: Gauge(
        "quantvision_active_sessions",
        "Number of active sessions",
    ),
)

trades_executed_total = _get_or_create_metric(
    "quantvision_trades_executed_total",
    lambda: Counter(
        "quantvision_trades_executed_total",
        "Total simulated trades executed",
    ),
)

scheduler_failures_total = _get_or_create_metric(
    "quantvision_scheduler_failures_total",
    lambda: Counter(
        "quantvision_scheduler_failures_total",
        "Total scheduler failures",
    ),
)


def record_anomalies_detected(method: str, count: int, ticker: str = "unknown") -> None:
    anomalies_detected_total.labels(method=method, ticker=ticker).inc(max(0, int(count)))


def record_method_runtime(method: str, duration_seconds: float) -> None:
    anomaly_detection_duration.labels(method=method).observe(max(0.0, float(duration_seconds)))


def record_trades_executed(trade_count: int) -> None:
    trades_executed_total.inc(max(0, int(trade_count)))


def record_scheduler_failure(failure_count: int) -> None:
    scheduler_failures_total.inc(max(0, int(failure_count)))


def record_login_attempt(success: bool) -> None:
    logins_total.labels(success=str(bool(success)).lower()).inc()


def set_active_sessions(count: int) -> None:
    active_sessions.set(max(0, int(count)))
