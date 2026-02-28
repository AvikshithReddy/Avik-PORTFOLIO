"""LlamaIndex ingestion pipelines targeting Qdrant collections."""

import json
from pathlib import Path
from typing import List, Dict, Any

import requests
from qdrant_client import QdrantClient
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.readers.github import GithubRepositoryReader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import SimpleDirectoryReader

from app.config import settings
from app.utils.logging import app_logger


def _build_embedding() -> OpenAIEmbedding:
    return OpenAIEmbedding(model=settings.EMBEDDING_MODEL)


def _build_llm() -> OpenAI:
    return OpenAI(model=settings.CHAT_MODEL, temperature=settings.CHAT_TEMPERATURE)


def _build_qdrant_store(collection: str) -> QdrantVectorStore:
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    return QdrantVectorStore(client=client, collection_name=collection, enable_hybrid=settings.QDRANT_HYBRID)


def _list_github_repos() -> List[str]:
    if not settings.GITHUB_USERNAME:
        return []

    headers = {}
    if settings.GITHUB_TOKEN:
        headers["Authorization"] = f"token {settings.GITHUB_TOKEN}"
        url = "https://api.github.com/user/repos"
        params = {
            "per_page": settings.GITHUB_MAX_REPOS,
            "sort": "updated",
            "visibility": "all",
            "affiliation": "owner",
        }
    else:
        url = f"https://api.github.com/users/{settings.GITHUB_USERNAME}/repos"
        params = {
            "per_page": settings.GITHUB_MAX_REPOS,
            "sort": "updated",
        }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        app_logger.warning(f"Failed to list GitHub repos: {exc}")
        return []

    repos = []
    for repo in data:
        owner = repo.get("owner", {}).get("login", "")
        if owner.lower() != settings.GITHUB_USERNAME.lower():
            continue
        name = repo.get("name")
        if name:
            repos.append(name)

    return repos[: settings.GITHUB_MAX_REPOS]


def _annotate_docs(docs: List[Document], source_name: str | None, source_path: Path | None = None) -> None:
    for doc in docs:
        meta = doc.metadata or {}
        if source_name:
            meta.setdefault("source_name", source_name)
        if source_path:
            meta.setdefault("file_path", str(source_path))
        doc.metadata = meta


def _annotate_github_docs(docs: List[Document]) -> None:
    for doc in docs:
        meta = doc.metadata or {}
        repo = meta.get("repo") or meta.get("repository")
        if repo:
            meta.setdefault("source_name", f"GitHub/{repo}")
        else:
            meta.setdefault("source_name", "GitHub")
        doc.metadata = meta


def _load_portfolio_json(path: Path) -> List[Document]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        payload = {}

    content = payload.get("content") or payload.get("text")
    if not content:
        content = path.read_text(encoding="utf-8", errors="ignore")
    title = payload.get("title") or path.stem
    metadata = {
        "source_name": title,
        "title": title,
        "file_path": str(path),
        "generated_at": payload.get("generated_at"),
    }
    return [Document(text=content, metadata=metadata)]


def ingest_resume(force_rebuild: bool = False) -> Dict[str, Any]:
    docs = []
    resume_path = Path(settings.RESUME_PDF_PATH)
    if resume_path.exists():
        docs.extend(SimpleDirectoryReader(input_files=[str(resume_path)]).load_data())

    resume_dir = Path("./data/resume")
    if resume_dir.exists():
        has_files = any(p.is_file() for p in resume_dir.rglob("*"))
        if has_files:
            docs.extend(SimpleDirectoryReader(input_dir=str(resume_dir), recursive=True).load_data())

    _annotate_docs(docs, source_name="Resume", source_path=resume_path if resume_path.exists() else None)
    splitter = SentenceSplitter(
        chunk_size=settings.RAG_CHUNK_SIZE, chunk_overlap=settings.RAG_CHUNK_OVERLAP
    )
    return _index_docs(
        docs,
        splitter,
        settings.QDRANT_RESUME_COLLECTION,
        source_type="resume",
        force_rebuild=force_rebuild,
    )


def ingest_portfolio_site(force_rebuild: bool = False) -> Dict[str, Any]:
    docs = []
    portfolio_json = Path(settings.PORTFOLIO_JSON_PATH)
    if portfolio_json.exists():
        docs.extend(_load_portfolio_json(portfolio_json))
    elif settings.PORTFOLIO_URL:
        urls = [settings.PORTFOLIO_URL]
        docs.extend(SimpleWebPageReader(html_to_text=True).load_data(urls))
        _annotate_docs(docs, source_name="Portfolio Site")

    splitter = SentenceSplitter(
        chunk_size=settings.RAG_CHUNK_SIZE, chunk_overlap=settings.RAG_CHUNK_OVERLAP
    )
    return _index_docs(
        docs,
        splitter,
        settings.QDRANT_PORTFOLIO_COLLECTION,
        source_type="portfolio",
        force_rebuild=force_rebuild,
    )


def ingest_linkedin(force_rebuild: bool = False) -> Dict[str, Any]:
    docs = []
    linkedin_pdf = Path(settings.LINKEDIN_PDF_PATH)
    if linkedin_pdf.exists():
        docs.extend(SimpleDirectoryReader(input_files=[str(linkedin_pdf)]).load_data())
    linkedin_dir = Path(settings.LINKEDIN_DATA_DIR)
    if linkedin_dir.exists():
        docs.extend(SimpleDirectoryReader(input_dir=str(linkedin_dir), recursive=True).load_data())

    _annotate_docs(docs, source_name="LinkedIn", source_path=linkedin_pdf if linkedin_pdf.exists() else None)
    splitter = SentenceSplitter(
        chunk_size=settings.RAG_CHUNK_SIZE, chunk_overlap=settings.RAG_CHUNK_OVERLAP
    )
    return _index_docs(
        docs,
        splitter,
        settings.QDRANT_LINKEDIN_COLLECTION,
        source_type="linkedin",
        force_rebuild=force_rebuild,
    )


def ingest_github(force_rebuild: bool = False) -> Dict[str, Any]:
    repos = []
    if settings.GITHUB_REPOS:
        repos = [r.strip() for r in settings.GITHUB_REPOS.split(",") if r.strip()]
    if not repos:
        repos = _list_github_repos()

    if not repos:
        app_logger.warning("No GitHub repos configured or discovered for ingest.")
        return {"status": "skipped", "documents_processed": 0, "chunks_created": 0}
    reader = GithubRepositoryReader(
        github_token=settings.GITHUB_TOKEN,
        owner=settings.GITHUB_USERNAME,
        repo=repos[0] if len(repos) == 1 else None,
        verbose=False,
        concurrent_requests=4,
    )

    documents: List[Document] = []
    if repos:
        for repo in repos:
            try:
                documents.extend(
                    reader.load_data(
                        branch=settings.GITHUB_BRANCH,
                        owner=settings.GITHUB_USERNAME,
                        repo=repo,
                    )
                )
            except Exception:
                # Fallback for repos that still use "master"
                documents.extend(
                    reader.load_data(
                        branch="master",
                        owner=settings.GITHUB_USERNAME,
                        repo=repo,
                    )
                )
    else:
        try:
            documents.extend(reader.load_data(branch=settings.GITHUB_BRANCH))
        except Exception:
            documents.extend(reader.load_data(branch="master"))

    _annotate_github_docs(documents)
    splitter = CodeSplitter(
        chunk_lines=settings.RAG_CODE_CHUNK_LINES, chunk_overlap=settings.RAG_CODE_CHUNK_OVERLAP
    )
    return _index_docs(
        documents,
        splitter,
        settings.QDRANT_GITHUB_COLLECTION,
        source_type="github",
        force_rebuild=force_rebuild,
    )


def _index_docs(
    docs: List[Document], splitter, collection: str, source_type: str, force_rebuild: bool = False
) -> Dict[str, Any]:
    if not docs:
        return {"status": "skipped", "documents_processed": 0, "chunks_created": 0}

    nodes = splitter.get_nodes_from_documents(docs)
    for node in nodes:
        meta = node.metadata or {}
        meta["source_type"] = source_type
        meta.setdefault("source_name", meta.get("title") or meta.get("file_name") or meta.get("repo") or source_type)
        node.metadata = meta

    embedding = _build_embedding()
    if force_rebuild:
        client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
        try:
            client.delete_collection(collection_name=collection)
        except Exception:
            pass

    vector_store = _build_qdrant_store(collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embedding)

    return {
        "status": "success",
        "documents_processed": len(docs),
        "chunks_created": len(nodes),
        "collection": collection,
    }


def ingest_all(selected: List[str], force_rebuild: bool = False) -> Dict[str, Any]:
    results = []
    total_chunks = 0
    total_docs = 0

    if "resume" in selected:
        r = ingest_resume(force_rebuild=force_rebuild)
        results.append(r)
        total_chunks += r.get("chunks_created", 0)
        total_docs += r.get("documents_processed", 0)

    if "portfolio" in selected:
        r = ingest_portfolio_site(force_rebuild=force_rebuild)
        results.append(r)
        total_chunks += r.get("chunks_created", 0)
        total_docs += r.get("documents_processed", 0)

    if "github" in selected:
        r = ingest_github(force_rebuild=force_rebuild)
        results.append(r)
        total_chunks += r.get("chunks_created", 0)
        total_docs += r.get("documents_processed", 0)

    if "linkedin" in selected:
        r = ingest_linkedin(force_rebuild=force_rebuild)
        results.append(r)
        total_chunks += r.get("chunks_created", 0)
        total_docs += r.get("documents_processed", 0)

    return {
        "status": "success",
        "documents_processed": total_docs,
        "chunks_created": total_chunks,
        "results": results,
    }
