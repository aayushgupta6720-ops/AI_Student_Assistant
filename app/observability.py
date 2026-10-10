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
    self_ms: float  # exclusive: latency_ms minus the time nested steps covered
    depth: int = 0
    start_ms: float = 0.0  # since the trace started
    end_ms: float = 0.0
    # Where the step itself was working, not a nested step: (start_ms, end_ms) pairs.
    own: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    meta: dict = field(default_factory=dict)


def _union(intervals) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _minus(span: tuple[float, float], holes) -> list[tuple[float, float]]:
    """The parts of `span` that none of `holes` cover."""
    out, cursor = [], span[0]
    for start, end in _union(holes):
        if start > cursor:
            out.append((cursor, min(start, span[1])))
        cursor = max(cursor, end)
    if cursor < span[1]:
        out.append((cursor, span[1]))
    return [(a, b) for a, b in out if b > a]


@dataclass
class CallTrace:
    steps: list[StepRecord] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)

    def add(self, record: StepRecord) -> None:
        self.steps.append(record)

    @property
    def total_tokens(self) -> int:
        return sum(s.input_tokens + s.output_tokens for s in self.steps)

    def per_layer_ms(self) -> dict[str, float]:
        """How long each layer was working: the time its steps covered outside
        their nested steps, with overlapping time counted once. Steps nest
        (tools.search_notes wraps inference.embed_query), and tools run in
        parallel: adding up their own times showed six 0.5 s tools as 3 s of
        tools in a 0.5 s turn."""
        by_layer: dict[str, list] = {}
        for s in self.steps:
            by_layer.setdefault(s.layer, []).extend(s.own)
        return {layer: round(sum(b - a for a, b in _union(spans)), 2) for layer, spans in by_layer.items()}

    def as_dicts(self) -> list[dict]:
        return [asdict(s) for s in self.steps]


_current_trace: contextvars.ContextVar["CallTrace | None"] = contextvars.ContextVar(
    "current_trace", default=None
)
# For each step currently open, the spans of the steps nested in it, so a
# step can report the time it covered itself (parallel children overlap).
_open_steps: contextvars.ContextVar[tuple[list[tuple[float, float]], ...]] = contextvars.ContextVar(
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
    children: list[tuple[float, float]] = []  # the spans of steps nested in this one
    token = _open_steps.set((*parents, children))
    try:
        yield usage
    finally:
        _open_steps.reset(token)
        end = time.perf_counter()
        latency_ms = round((end - start) * 1000, 2)
        if parents:
            parents[-1].append((start, end))
        trace = get_trace()
        if trace is not None:
            own = _minus((start, end), children)
            to_ms = lambda t: round((t - trace.started) * 1000, 3)  # noqa: E731
            trace.add(
                StepRecord(
                    layer=layer,
                    name=name,
                    latency_ms=latency_ms,
                    self_ms=round(sum(b - a for a, b in own) * 1000, 2),
                    depth=len(parents),
                    start_ms=to_ms(start),
                    end_ms=to_ms(end),
                    own=[(to_ms(a), to_ms(b)) for a, b in own],
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"],
                    meta=meta,
                )
            )


def log_event(**fields) -> None:
    logger.info(json.dumps(fields, default=str))
