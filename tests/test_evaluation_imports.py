from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_importing_simple_evaluator_does_not_eagerly_import_ragas_module() -> None:
    sys.modules.pop("src.evaluation", None)
    sys.modules.pop("src.evaluation.ragas_evaluator", None)
    sys.path.insert(0, str(ROOT))

    try:
        module = importlib.import_module("src.evaluation")
    finally:
        if sys.path and sys.path[0] == str(ROOT):
            sys.path.pop(0)

    assert module.SimpleEvaluator.__name__ == "SimpleEvaluator"
    assert "src.evaluation.ragas_evaluator" not in sys.modules
