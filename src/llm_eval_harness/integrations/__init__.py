"""Outbound integrations — webhook notifier, pytest plugin."""

from llm_eval_harness.integrations.notifier import WebhookNotifier, notify_slack

__all__ = ["WebhookNotifier", "notify_slack"]