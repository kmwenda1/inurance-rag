from rag import generator


def _sample_chunks():
    return [{"source": "policy.pdf", "page_num": 2, "text": "Coverage text", "score": 0.8}]


def test_generate_answer_empty_chunks():
    result = generator.generate_answer("What is covered?", [])
    assert "could not find" in result["answer"].lower()
    assert result["sources"] == []
    assert result["error"] is None


def test_generate_answer_missing_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(generator, "_groq_client", None)

    result = generator.generate_answer("What is covered?", _sample_chunks())

    assert result["error"] is not None
    assert "api key" in result["answer"].lower()


def test_generate_answer_provider_failure_is_sanitized(monkeypatch):
    monkeypatch.setattr(generator, "get_groq_client", lambda: (_ for _ in ()).throw(RuntimeError("authentication failed with secret-token")))

    result = generator.generate_answer("What is covered?", _sample_chunks())

    assert result["error"] is not None
    assert "secret-token" not in result["answer"]


def test_generate_answer_success(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            class Message:
                content = "Answer from context."

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeClient:
        class chat:
            completions = FakeCompletions()

    monkeypatch.setattr(generator, "get_groq_client", lambda: FakeClient())

    result = generator.generate_answer("What is covered?", _sample_chunks())

    assert result["answer"] == "Answer from context."
    assert result["error"] is None
