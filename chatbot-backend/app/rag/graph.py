"""LangGraph-based orchestration for portfolio chatbot.

Nodes:
- router: classify message into resume/github/portfolio/general
- retrieve: query LlamaIndex per namespace
- rerank: Cohere rerank top_k
- synthesize: generate grounded answer with inline source tags
- fallback/self-correct: if low support, refine or answer safely
"""

from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from llama_index.core import QueryBundle
from app.utils.logging import app_logger
from app.config import settings
from qdrant_client.http.exceptions import UnexpectedResponse


class ChatState(TypedDict, total=False):
    # keys: message, route, retrieved, reranked, answer, sources, confidence, history
    message: str
    route: str
    retrieved: List[Any]
    reranked: List[Any]
    answer: str
    sources: List[Any]
    confidence: float
    history: List[Any]


def build_graph(router_fn, retrievers: Dict[str, Any], reranker, synthesizer_fn):
    graph = StateGraph(ChatState)

    graph.add_node("router", router_fn)
    graph.add_node("retrieve", lambda state: _retrieve(state, retrievers))
    graph.add_node("rerank", lambda state: _rerank(state, reranker))
    graph.add_node("synthesize", synthesizer_fn)

    graph.add_edge("router", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "synthesize")
    graph.add_edge("synthesize", END)

    # Define starting node for the graph
    graph.set_entry_point("router")

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def _retrieve(state: ChatState, retrievers: Dict[str, Any]):
    route = state.get("route", "general")
    query = state["message"]
    bundle = QueryBundle(query)

    def dedupe(nodes: List):
        unique = {}
        for node in nodes:
            node_id = getattr(node, "node_id", None) or getattr(node, "id_", None)
            if not node_id:
                node_id = node.get_text()[:200]
            score = float(getattr(node, "score", 0.0) or 0.0)
            if node_id not in unique or score > unique[node_id][0]:
                unique[node_id] = (score, node)
        return [item[1] for item in unique.values()]

    if route == "general":
        results = []
        for key, retriever in retrievers.items():
            if key == "general":
                continue
            try:
                results.extend(retriever.retrieve(bundle))
            except UnexpectedResponse as exc:
                # Qdrant returns 404 if collection doesn't exist yet
                if getattr(exc, "status_code", None) == 404:
                    app_logger.warning(f"Collection missing for route={key}; run /api/ingest to create it.")
                    continue
                raise
        results = dedupe(results)
        results.sort(key=lambda n: float(getattr(n, "score", 0.0) or 0.0), reverse=True)
        state["retrieved"] = results
        app_logger.info(f"Retrieved {len(results)} docs for route=general (merged)")
        return state

    retriever = retrievers.get(route)
    if not retriever:
        state["retrieved"] = []
        return state

    try:
        results = retriever.retrieve(bundle)
    except UnexpectedResponse as exc:
        if getattr(exc, "status_code", None) == 404:
            app_logger.warning(f"Collection missing for route={route}; run /api/ingest to create it.")
            state["retrieved"] = []
            return state
        raise
    state["retrieved"] = results
    app_logger.info(f"Retrieved {len(results)} docs for route={route}")
    return state


def _rerank(state: ChatState, reranker):
    docs = state.get("retrieved", [])
    if not docs:
        state["reranked"] = []
        return state

    if getattr(reranker, "client", None) is None:
        sorted_docs = sorted(docs, key=lambda n: float(getattr(n, "score", 0.0) or 0.0), reverse=True)
        state["reranked"] = sorted_docs[: settings.RAG_RERANK_TOP_N]
        return state

    texts = [d.get_text() for d in docs]
    indices = reranker.rerank(state["message"], texts, top_n=min(settings.RAG_RERANK_TOP_N, len(texts)))
    state["reranked"] = [docs[i] for i in indices]
    return state
