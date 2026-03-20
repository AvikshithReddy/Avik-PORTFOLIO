import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

TESTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TESTS_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import chat, sessions  # noqa: E402
import app.main as main_module  # noqa: E402
from app.rag.build_docs import DocumentBuilder  # noqa: E402
from app.rag.index import RAGIndex  # noqa: E402
from app.schemas import ChatRequest  # noqa: E402
from app.utils.text import chunk_text  # noqa: E402


class FakeOpenAIClient:
    def __init__(self):
        self.rewrite_calls = []

    def create_embedding(self, text):
        normalized = text.lower()
        if "financial chatbot" in normalized:
            return [1.0, 0.0]
        return [0.0, 1.0]

    def create_embeddings_batch(self, texts):
        embeddings = []
        for text in texts:
            embeddings.append(self.create_embedding(text))
        return embeddings

    def rewrite_query(self, prompt, model=None, max_tokens=120):
        self.rewrite_calls.append(prompt)
        return "financial chatbot project"

    def chat_completion(self, messages, temperature=0.2, max_tokens=1000, model=None):
        return "Grounded answer"


class TextUtilsTests(unittest.TestCase):
    def test_short_chunks_include_text_metadata(self):
        chunks = chunk_text(
            "Short portfolio summary",
            chunk_size=1200,
            metadata={"source_type": "portfolio", "source_name": "Summary"}
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["metadata"]["text"], "Short portfolio summary")
        self.assertEqual(chunks[0]["metadata"]["char_start"], 0)


class RAGIndexTests(unittest.TestCase):
    def test_search_returns_results_without_logging_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index = RAGIndex(tmpdir)
            index.save_index(
                np.array([[1.0, 0.0], [0.0, 1.0]]),
                [
                    {"source_name": "A", "text": "financial chatbot"},
                    {"source_name": "B", "text": "other project"}
                ]
            )

            results = index.search([1.0, 0.0], top_k=1)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0][0]["source_name"], "A")


class DocumentBuilderTests(unittest.TestCase):
    def test_build_index_filters_empty_chunks_in_lockstep(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index = RAGIndex(tmpdir)
            builder = DocumentBuilder(FakeOpenAIClient(), index)
            builder._process_portfolio = lambda: [
                {
                    "text": "financial chatbot project",
                    "metadata": {
                        "source_type": "portfolio",
                        "source_name": "Project: Financial Chatbot",
                        "locator": "./data/portfolio_data.json",
                        "text": "financial chatbot project"
                    }
                },
                {
                    "text": "   ",
                    "metadata": {
                        "source_type": "portfolio",
                        "source_name": "Empty Chunk",
                        "locator": "./data/portfolio_data.json"
                    }
                }
            ]

            result = builder.build_index(sources=["portfolio"], force_rebuild=True)

            self.assertEqual(result["chunks_created"], 1)
            self.assertTrue(index.loaded)
            self.assertEqual(len(index.metadata), 1)


class ChatWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.original_openai_client = main_module.openai_client
        self.original_rag_index = main_module.rag_index
        self.original_doc_builder = main_module.doc_builder
        self.original_sessions = dict(sessions)
        sessions.clear()

    def tearDown(self):
        main_module.openai_client = self.original_openai_client
        main_module.rag_index = self.original_rag_index
        main_module.doc_builder = self.original_doc_builder
        sessions.clear()
        sessions.update(self.original_sessions)

    def test_chat_rewrites_follow_up_query_and_returns_grounded_answer(self):
        fake_client = FakeOpenAIClient()

        with tempfile.TemporaryDirectory() as tmpdir:
            index = RAGIndex(tmpdir)
            index.save_index(
                np.array([[1.0, 0.0], [0.0, 1.0]]),
                [
                    {
                        "source_type": "portfolio",
                        "source_name": "Project: Financial Chatbot",
                        "locator": "./data/portfolio_data.json",
                        "text": "Built a finance-focused LLM assistant."
                    },
                    {
                        "source_type": "portfolio",
                        "source_name": "Project: Other",
                        "locator": "./data/portfolio_data.json",
                        "text": "Built another project."
                    }
                ]
            )

            main_module.openai_client = fake_client
            main_module.rag_index = index
            main_module.doc_builder = object()

            sessions["session-1"] = [
                {"role": "user", "content": "Tell me about the finance assistant"},
                {"role": "assistant", "content": "It is a portfolio project."}
            ]

            response = asyncio.run(
                chat(
                    ChatRequest(
                        query="Tell me more about that project",
                        session_id="session-1",
                        include_sources=True
                    )
                )
            )

            self.assertEqual(response.answer_mode, "grounded")
            self.assertEqual(response.response, "Grounded answer")
            self.assertEqual(response.sources[0].source_name, "Project: Financial Chatbot")
            self.assertEqual(len(fake_client.rewrite_calls), 1)


if __name__ == "__main__":
    unittest.main()
