def test_import_faiss():
    import faiss  # noqa: F401


def test_import_langchain_faiss_vectorstore():
    # LangChain récent: FAISS est dans langchain_community
    from langchain_community.vectorstores import FAISS  # noqa: F401


def test_import_huggingface_embeddings():
    # Embeddings HuggingFace via package dédié
    from langchain_huggingface import HuggingFaceEmbeddings  # noqa: F401


def test_import_mistral_client():
    """
    Le brief mentionne `from mistral import MistralClient`,
    mais le SDK officiel actuel est `mistralai`.
    On accepte les deux pour robustesse.
    """
    try:
        from mistralai import Mistral  # noqa: F401
    except Exception:
        # fallback si votre environnement utilise un ancien package/alias
        from mistral import MistralClient  # type: ignore # noqa: F401
