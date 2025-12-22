from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


INDEX_DIR = Path("data/index/faiss_events")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def test_faiss_index_dir_exists():
    assert INDEX_DIR.exists(), f"Missing FAISS index dir: {INDEX_DIR}"


def test_faiss_loads():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
    assert vs is not None


def test_faiss_has_vectors():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
    assert vs.index.ntotal > 0


def test_similarity_search_returns_k_docs():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)

    k = 5
    docs = vs.similarity_search("conférence", k=k)

    assert isinstance(docs, list)
    assert len(docs) == k
    assert all(d.page_content for d in docs)


def test_metadata_contains_uid_and_date():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)

    docs = vs.similarity_search("conférence", k=3)
    for d in docs:
        assert "uid" in d.metadata
        assert "first_begin_dt" in d.metadata
