from rag import retriever


def test_retrieve_with_confidence_none(monkeypatch):
    monkeypatch.setattr(retriever, "retrieve", lambda q, top_k=8: [])

    result = retriever.retrieve_with_confidence("anything")

    assert result["confidence"] == "NONE"
    assert result["chunks"] == []


def test_retrieve_with_confidence_thresholds(monkeypatch):
    monkeypatch.setattr(retriever, "retrieve", lambda q, top_k=8: [{"score": 0.80}])
    assert retriever.retrieve_with_confidence("q")["confidence"] == "HIGH"

    monkeypatch.setattr(retriever, "retrieve", lambda q, top_k=8: [{"score": 0.60}])
    assert retriever.retrieve_with_confidence("q")["confidence"] == "MEDIUM"

    monkeypatch.setattr(retriever, "retrieve", lambda q, top_k=8: [{"score": 0.30}])
    result = retriever.retrieve_with_confidence("q")
    assert result["confidence"] == "LOW"
    assert result["warning"]


def test_retrieve_formats_chroma_results(monkeypatch):
    class FakeCollection:
        def count(self):
            return 2

        def query(self, query_embeddings, n_results, include):
            return {
                "documents": [["chunk B", "chunk A"]],
                "metadatas": [[{"page_num": 2, "source": "b.pdf"}, {"page_num": 1, "source": "a.pdf"}]],
                "distances": [[0.4, 0.1]],
            }

    monkeypatch.setattr(retriever, "get_collection", lambda: FakeCollection())
    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[0.1, 0.2]])

    chunks = retriever.retrieve("question", top_k=5)

    assert len(chunks) == 2
    assert chunks[0]["score"] >= chunks[1]["score"]
    assert chunks[0]["source"] == "a.pdf"
