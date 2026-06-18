"""Langfuse wiring with graceful degradation.

If Langfuse credentials are present, `observe` is the real `@observe()` decorator
and traces stream to the cloud/self-host. If not, `observe` becomes a no-op so the
pipeline still runs (e.g. in CI that has no Langfuse keys). This keeps the same
code path for `python -m app.cli` whether or not you have a Langfuse account.
"""
import os
import functools

LANGFUSE_ENABLED = bool(
    os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
)

if LANGFUSE_ENABLED:
    try:
        from langfuse import observe, get_client  # langfuse>=3.x

        _client = get_client()

        def update_current_trace(**kwargs):
            try:
                _client.update_current_trace(**kwargs)
            except Exception:
                pass

        def flush():
            try:
                _client.flush()
            except Exception:
                pass

    except Exception as exc:  # pragma: no cover - import/version guard
        print(f"[observability] Langfuse import failed ({exc}); tracing disabled.")
        LANGFUSE_ENABLED = False

if not LANGFUSE_ENABLED:

    def observe(*dargs, **dkwargs):
        """No-op replacement for langfuse.observe when tracing is disabled."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        # Support both @observe and @observe(name=...)
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return decorator(dargs[0])
        return decorator

    def update_current_trace(**kwargs):
        pass

    def flush():
        pass
