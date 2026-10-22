import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from fastembed import TextEmbedding

_EMBEDDING_MODEL = None
_CHROMA_CLIENT = None

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
COLLECTION_NAME = "insurance_documents"

def get_embedding_model():
    """Lazily load the embedding model to avoid startup hangs or crashes."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        print("Loading embedding model (fastembed)...")
        _EMBEDDING_MODEL = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        print("Embedding model ready.")
    return _EMBEDDING_MODEL

def get_chroma_client():
    """Lazily initialize ChromaDB client and ensure directory exists."""
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        os.makedirs(DB_PATH, exist_ok=True)
        _CHROMA_CLIENT = chromadb.PersistentClient(path=DB_PATH)
    return _CHROMA_CLIENT

def get_collection():
    return get_chroma_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Insurance document chunks with embeddings"}
    )

def embed_texts(texts: list[str]) -> list[list[float]]:
    # fastembed returns a generator of numpy arrays
    model = get_embedding_model()
    embeddings_gen = model.embed(texts)
    return [emb.tolist() for emb in embeddings_gen]

def store_chunks(chunks: list[dict]) -> int:
    collection = get_collection()
    ids        = [chunk["chunk_id"] for chunk in chunks]
    texts      = [chunk["text"]     for chunk in chunks]
    metadatas  = [
        {
            "page_num": chunk["page_num"],
            "source":   chunk["source"],
        }
        for chunk in chunks
    ]
    print(f"  Embedding {len(texts)} chunks...")
    embeddings = embed_texts(texts)
    batch_size = 100
    stored = 0
    for i in range(0, len(chunks), batch_size):
        batch_end = i + batch_size
        collection.add(
            ids=ids[i:batch_end],
            documents=texts[i:batch_end],
            embeddings=embeddings[i:batch_end],
            metadatas=metadatas[i:batch_end]
        )
        stored += len(ids[i:batch_end])
    return stored

def chunk_already_exists(chunk_id: str) -> bool:
    collection = get_collection()
    result = collection.get(ids=[chunk_id])
    return len(result["ids"]) > 0

def get_collection_stats() -> dict:
    collection = get_collection()
    count = collection.count()
    return {
        "total_chunks": count,
        "collection_name": COLLECTION_NAME
    }
