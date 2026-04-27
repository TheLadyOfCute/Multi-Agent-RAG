"""Workflow assembly and execution package.

The heavy LangGraph workflow is imported lazily by callers that actually run it.
This keeps lightweight imports such as ``src.workflows.factory`` available even
when optional workflow dependencies are not installed in a local learning setup.
"""

__all__: list[str] = []
