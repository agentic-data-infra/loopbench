"""Span tracing.

Every unit of work runs inside a span tagged with one category. The breakdown
attributes *self time* (span duration minus child spans) to the span's category,
so nested spans never double count. Time inside the root span that no child
covers is charged to ``overhead``.
"""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

CATEGORIES = ("agent", "dp.prep", "train", "dp.examine", "infer.examine", "io", "overhead")


@dataclass
class Span:
    span_id: str
    parent_id: str | None
    name: str
    category: str
    start_ns: int
    end_ns: int = 0
    attrs: dict = field(default_factory=dict)
    child_ns: int = 0

    @property
    def dur_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def self_ns(self) -> int:
        return self.dur_ns - self.child_ns


class Tracer:
    def __init__(self, run_id: str | None = None, enabled: bool = True):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.enabled = enabled
        self.spans: list[Span] = []
        self._local = threading.local()

    def _stack(self) -> list[Span]:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    @contextmanager
    def span(self, name: str, category: str, **attrs):
        if category not in CATEGORIES:
            raise ValueError(f"unknown trace category {category!r}")
        if not self.enabled:
            yield None
            return
        stack = self._stack()
        parent = stack[-1] if stack else None
        sp = Span(uuid.uuid4().hex[:16], parent.span_id if parent else None, name, category,
                  time.perf_counter_ns(), attrs=dict(attrs))
        stack.append(sp)
        try:
            yield sp
        finally:
            sp.end_ns = time.perf_counter_ns()
            stack.pop()
            if parent is not None:
                parent.child_ns += sp.dur_ns
            self.spans.append(sp)

    def breakdown(self) -> dict[str, float]:
        """Seconds of self time per category."""
        out = {c: 0.0 for c in CATEGORIES}
        for sp in self.spans:
            out[sp.category] += sp.self_ns / 1e9
        return out

    def total_s(self) -> float:
        return sum(sp.dur_ns for sp in self.spans if sp.parent_id is None) / 1e9

    def rows(self) -> list[dict]:
        return [dict(run_id=self.run_id, span_id=s.span_id, parent_id=s.parent_id, name=s.name,
                     category=s.category, start_ns=s.start_ns, dur_ns=s.dur_ns, self_ns=s.self_ns,
                     attrs=repr(s.attrs)) for s in self.spans]

    def format_breakdown(self) -> str:
        bd = self.breakdown()
        total = sum(bd.values()) or 1.0
        lines = [f"{'category':<15}{'seconds':>10}{'share':>8}"]
        for c in CATEGORIES:
            lines.append(f"{c:<15}{bd[c]:>10.2f}{100 * bd[c] / total:>7.1f}%")
        lines.append(f"{'total':<15}{total:>10.2f}")
        return "\n".join(lines)

    def format_tree(self, min_s: float = 0.0) -> str:
        children: dict[str | None, list[Span]] = {}
        for s in self.spans:
            children.setdefault(s.parent_id, []).append(s)
        lines: list[str] = []

        def walk(pid, depth):
            for s in sorted(children.get(pid, []), key=lambda s: s.start_ns):
                if s.dur_ns / 1e9 >= min_s:
                    lines.append(f"{'  ' * depth}{s.name:<{48 - 2 * depth}} {s.category:<14}"
                                 f"{s.dur_ns / 1e9:>9.2f}s")
                walk(s.span_id, depth + 1)

        walk(None, 0)
        return "\n".join(lines)


_NULL = Tracer(enabled=False)


def null_tracer() -> Tracer:
    return _NULL
