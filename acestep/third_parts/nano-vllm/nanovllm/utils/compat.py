"""Compatibility utilities for optional dependencies.

Provides graceful fallbacks when torch.compile's backend (Triton) is
unavailable — e.g. on Windows or on GPU architectures where Triton
has not yet added support (Blackwell / SM 120 as of early 2026).
"""

from loguru import logger


def maybe_compile(fn=None, **compile_kwargs):
    """Apply ``torch.compile`` only when its backend (Triton) is available.

    Drop-in replacement for the ``@torch.compile`` decorator.  When Triton
    is importable the function is compiled as usual; otherwise the original
    function is returned unmodified so inference still works (just without
    the kernel-fusion speed-up).

    Usage::

        @maybe_compile
        def forward(self, x):
            ...

        # or with keyword arguments:
        @maybe_compile(dynamic=True)
        def forward(self, x):
            ...
    """
    def decorator(func):
        try:
            import triton  # noqa: F401
            import torch
            return torch.compile(func, **compile_kwargs)
        except ImportError:
            logger.info(
                "Triton not available — skipping torch.compile for %s "
                "(inference will use native PyTorch kernels)",
                func.__qualname__,
            )
            return func

    # Support both @maybe_compile and @maybe_compile(...) syntax
    if fn is not None:
        return decorator(fn)
    return decorator
