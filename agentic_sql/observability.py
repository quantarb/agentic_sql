import json
import os
from typing import Any, Dict


def _flatten(prefix: str, value: Any, out: Dict[str, str]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            _flatten(key, v, out)
        return
    if isinstance(value, list):
        out[prefix] = json.dumps(value, ensure_ascii=False)
        return
    out[prefix] = "" if value is None else str(value)


class PhoenixObservability:
    def __init__(
        self,
        *,
        enabled: bool,
        project_name: str,
        endpoint: str = "",
    ) -> None:
        self.enabled = bool(enabled)
        self._tracer = None
        self._provider = None
        self._setup_error = ""

        if not self.enabled:
            return

        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            resolved_endpoint = endpoint or os.environ.get(
                "PHOENIX_COLLECTOR_ENDPOINT",
                "http://localhost:6006/v1/traces",
            )
            resource = Resource.create(
                {
                    "service.name": "agentic_sql",
                    "phoenix.project_name": project_name or "agentic-sql",
                }
            )
            provider = TracerProvider(resource=resource)
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=resolved_endpoint)))
            trace.set_tracer_provider(provider)
            self._provider = provider
            self._tracer = trace.get_tracer("agentic_sql.observability")
        except Exception as e:
            self.enabled = False
            self._setup_error = str(e)

    @property
    def setup_error(self) -> str:
        return self._setup_error

    def log_event(self, *, run_id: int, step_idx: int, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.enabled or self._tracer is None:
            return
        attrs: Dict[str, str] = {
            "run_id": str(run_id),
            "step_idx": str(step_idx),
            "event_type": str(event_type),
        }
        _flatten("payload", payload, attrs)
        with self._tracer.start_as_current_span(f"agentic_sql.{event_type}") as span:
            for k, v in attrs.items():
                span.set_attribute(k, v)

    def shutdown(self) -> None:
        if self._provider is None:
            return
        try:
            self._provider.shutdown()
        except Exception:
            pass

