"""Cross-cutting tracing. Every layer wraps its work in `time_step(layer, name)`
so a single /chat call produces a per-layer latency/token breakdown that the
client renders. Not a layer itself - it's shared plumbing."""

import contextvars
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

logger = logging.getLogger("ai_student_assistant")


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers = [handler]
    logger.setLevel(level)
    logger.propagate = False


@dataclass
class StepRecord:
    layer: str  # intelligence | inference | knowledge | tools
    name: str
    latency_ms: float  # inclusive: wall time of the whole block
    self_ms: float  # exclusive: latency_ms minus time spent in nested steps
    depth: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class CallTrace:
    steps: list[StepRecord] = field(default_factory=list)

    def add(self, record: StepRecord) -> None:
        self.steps.append(record)

    @property
    def total_tokens(self) -> int:
        return sum(s.input_tokens + s.output_tokens for s in self.steps)

    def per_layer_ms(self) -> dict[str, float]:
        """Exclusive time per layer. Steps nest (tools.search_notes wraps
        inference.embed_query), so summing inclusive times would double-count."""
        totals: dict[str, float] = {}
        for s in self.steps:
            totals[s.layer] = round(totals.get(s.layer, 0.0) + s.self_ms, 2)
        return totals

    def as_dicts(self) -> list[dict]:
        return [asdict(s) for s in self.steps]


_current_trace: contextvars.ContextVar["CallTrace | None"] = contextvars.ContextVar(
    "current_trace", default=None
)
# Stack of child-time accumulators for the steps currently open, so a step
# can subtract its children's time and report exclusive time.
_open_steps: contextvars.ContextVar[tuple[list[float], ...]] = contextvars.ContextVar(
    "open_steps", default=()
)


def start_trace() -> CallTrace:
    trace = CallTrace()
    _current_trace.set(trace)
    return trace


def get_trace() -> "CallTrace | None":
    return _current_trace.get()


@contextmanager
def time_step(layer: str, name: str, **meta):
    """Time a block and, if a trace is active, record it. The caller may fill
    token usage into the yielded dict."""
    start = time.perf_counter()
    usage = {"input_tokens": 0, "output_tokens": 0}
    parents = _open_steps.get()
    children_ms = [0.0]
    token = _open_steps.set((*parents, children_ms))
    try:
        yield usage
    finally:
        _open_steps.reset(token)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        if parents:
            parents[-1][0] += latency_ms
        trace = get_trace()
        if trace is not None:
            trace.add(
                StepRecord(
                    layer=layer,
                    name=name,
                    latency_ms=latency_ms,
                    self_ms=round(max(latency_ms - children_ms[0], 0.0), 2),
                    depth=len(parents),
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"],
                    meta=meta,
                )
            )


def log_event(**fields) -> None:
    logger.info(json.dumps(fields, default=str))
