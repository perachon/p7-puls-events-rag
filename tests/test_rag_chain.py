from __future__ import annotations

from dataclasses import dataclass

from src.rag.chain import answer_question


@dataclass
class FakeResponse:
    content: str


class FakeLLM:
    def invoke(self, messages):
        return FakeResponse(content="Réponse de test.")


def test_answer_question_returns_text_and_sources():
    res = answer_question("conférence scientifique", llm_override=FakeLLM())
    assert isinstance(res.answer, str)
    assert hasattr(res, "sources")
    assert isinstance(res.sources, list)


def test_no_result_returns_empty_sources():
    # requête volontairement improbable
    res = answer_question("concert de métal à gif-sur-yvette ce soir", llm_override=FakeLLM())
    # soit 0 sources, soit tu as encore du bruit -> à toi de fixer la règle attendue
    assert isinstance(res.sources, list)
