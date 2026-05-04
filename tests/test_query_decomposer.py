import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.query_decomposer import QueryDecomposer


class RecordingLLM:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.response


def test_decomposition_prompt_requires_natural_sub_questions():
    llm = RecordingLLM(
        '["What are the two phases of AGCD called?", '
        '"What happens during the first phase of AGCD?", '
        '"What happens during the second phase of AGCD?"]'
    )
    decomposer = QueryDecomposer(llm=llm)

    sub_queries = decomposer._decompose_multihop(
        "What are the two phases of AGCD, and what happens in each phase?"
    )

    assert sub_queries == [
        "What are the two phases of AGCD called?",
        "What happens during the first phase of AGCD?",
        "What happens during the second phase of AGCD?",
    ]
    prompt = llm.prompts[0]
    assert "complete, natural-language question" in prompt
    assert "not a keyword fragment" in prompt
    assert "Preserve the original core entity names" in prompt
    assert '"AGCD two phases names"' in prompt
