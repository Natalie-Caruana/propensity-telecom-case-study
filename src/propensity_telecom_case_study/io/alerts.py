"""Alerting — desktop notifications for local dev, extensible to Slack/PagerDuty."""

from loguru import logger


def notify(title: str, message: str) -> None:
    """Send a desktop notification (local dev).

    Falls back to a log warning if plyer is unavailable or the platform
    does not support notifications.

    Args:
        title: Notification title.
        message: Notification body text.
    """
    try:
        from plyer import notification  # type: ignore[import-untyped]

        notification.notify(title=title, message=message, timeout=8)
    except Exception:
        logger.warning(f"[ALERT] {title}: {message}")


def alert_on_metric_threshold(
    metric_name: str,
    value: float,
    threshold: float,
    direction: str = "below",
) -> None:
    """Fire an alert if a metric crosses a threshold.

    Args:
        metric_name: Human-readable metric name (e.g. 'roc_auc').
        value: Observed metric value.
        threshold: Threshold to compare against.
        direction: 'below' fires when value < threshold; 'above' when value > threshold.
    """
    triggered = (direction == "below" and value < threshold) or (
        direction == "above" and value > threshold
    )
    if triggered:
        msg = f"{metric_name}={value:.3f} crossed threshold ({direction} {threshold})"
        logger.warning(f"Metric alert: {msg}")
        notify(title="Propensity Model Alert", message=msg)
    else:
        logger.info(
            f"{metric_name}={value:.3f} — within threshold ({direction} {threshold})"
        )
