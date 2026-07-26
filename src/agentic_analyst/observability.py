"""Tracing: the one module that knows Langfuse exists.

Everything else asks `get_tracer()` for a client and never imports Langfuse
itself, for the same reason `settings.py` is the only module that reads the
environment — changing the tracing backend should be a change *here*, not in
eight call sites.

The tracer is a **null object** when unconfigured, never `None`. A Langfuse
client built with `tracing_enabled=False` accepts every span call and silently
drops it, so no caller has to guard with `if tracing_enabled:`. That matters
more than it looks: a `Langfuse | None` return type puts a branch at every call
site, and the one place someone forgets it is an `AttributeError` in the middle
of a paid run. A no-op shaped like the real thing cannot be forgotten at one
call site and remembered at the others.

Two layers get traced, and they need different mechanisms:

* **The graph** — `CallbackHandler` rides along on `.invoke()` and turns each
  LangGraph node into a span. LangGraph is a LangChain runnable, so this is free.
* **The model calls** — instrumented by hand in `llm.py`. The callback handler
  cannot see them: they go to `google-genai` directly and never touch a
  LangChain abstraction. Without that manual half you get a shapely trace with
  no tokens, no prompts and no cost in it.
"""

import logging
from functools import lru_cache

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from agentic_analyst.settings import SETTINGS

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_tracer() -> Langfuse:
    """Build the Langfuse client on first use, then reuse it.

    Lazy for the same reason `llm._get_client` is: constructing at import time
    would mean `import agentic_analyst.observability` does network setup during
    test collection, on a machine that may have no keys at all.

    Keys are passed explicitly rather than left to Langfuse's own environment
    lookup. The SDK would happily read `LANGFUSE_*` itself, but then two
    modules would be reading the environment and `settings.py` would no longer
    be the single source of truth for configuration.
    """
    if not SETTINGS.tracing_enabled:
        # Langfuse logs a WARNING about the absent key. Running without one is
        # a supported mode here, not a mistake, so don't alarm the user about
        # a decision they made on purpose.
        logging.getLogger("langfuse").setLevel(logging.ERROR)
        log.info("Langfuse keys absent — running untraced")
        return Langfuse(tracing_enabled=False)

    log.info("tracing to %s", SETTINGS.langfuse_base_url)
    return Langfuse(
        public_key=SETTINGS.langfuse_public_key,
        secret_key=SETTINGS.langfuse_secret_key,
        base_url=SETTINGS.langfuse_base_url,
    )


def callback_handler() -> CallbackHandler:
    """The LangChain-side adapter that turns graph nodes into spans.

    Takes no credentials: it resolves the process-wide Langfuse singleton at
    construction time.

    The `get_tracer()` call below is load-bearing, and deleting it as redundant
    is a silent failure. `pydantic-settings` reads `.env` without exporting to
    `os.environ`, so a `CallbackHandler` built before the client exists finds no
    key, disables itself, logs one warning, and drops every span for the rest of
    the run. Constructing the client first is what puts the credentials where
    the handler can find them.
    """
    get_tracer()
    return CallbackHandler()


def flush() -> None:
    """Push buffered spans before the process exits.

    Langfuse batches spans and ships them from a background thread. A CLI that
    finishes promptly can therefore complete a whole run and send *nothing* —
    the traces die in a buffer that never got drained, with no error anywhere.
    For a short-lived process this is not an optimisation, it is the difference
    between having traces and not.

    Deliberately a plain function the entry point calls, not an `atexit` hook:
    the run decides when it is over, and a failed run should flush what it
    managed to record just as much as a successful one.
    """
    get_tracer().flush()
