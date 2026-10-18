"""
RAG (Retrieval-Augmented Generation) engine.
Builds a FAISS-backed vector index over the built-in nutrition knowledge base
and provides semantic retrieval without requiring an external server.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

from nutrition_agent.data.food_knowledge_base import (
    NUTRITION_DOCUMENTS,
    FOOD_NUTRIENT_DATABASE,
)


# ── Lightweight TF-IDF style retriever (no external vector DB required) ────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_index(docs: list[dict]) -> tuple[list[dict], dict[str, list[int]]]:
    """Build an inverted index from documents."""
    inverted: dict[str, list[int]] = {}
    for idx, doc in enumerate(docs):
        tokens = set(_tokenize(doc["title"] + " " + doc["content"]))
        for token in tokens:
            inverted.setdefault(token, []).append(idx)
    return docs, inverted


_DOCS, _INDEX = _build_index(NUTRITION_DOCUMENTS)


def _score_docs(query: str) -> list[tuple[float, dict]]:
    """BM25-style scoring: TF * IDF."""
    q_tokens = _tokenize(query)
    N = len(_DOCS)
    scores: dict[int, float] = {}
    for token in q_tokens:
        posting = _INDEX.get(token, [])
        if not posting:
            continue
        idf = math.log((N - len(posting) + 0.5) / (len(posting) + 0.5) + 1)
        for idx in posting:
            scores[idx] = scores.get(idx, 0.0) + idf
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, _DOCS[idx]) for idx, score in ranked]


class NutritionRAG:
    """
    Retrieval-Augmented Generation helper.
    Retrieves relevant nutrition documents for a given query and formats them
    as a context block for the LLM prompt.
    """

    def retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        """Return top_k most relevant documents for the query."""
        results = _score_docs(query)
        return [doc for _, doc in results[:top_k]]

    def build_context(self, query: str, top_k: int = 4) -> str:
        """Return a formatted context string for inclusion in a prompt."""
        docs = self.retrieve(query, top_k=top_k)
        if not docs:
            return "No specific nutritional data found."
        parts = []
        for doc in docs:
            parts.append(f"[{doc['title']}]\n{doc['content']}")
        return "\n\n".join(parts)

    def get_food_nutrients(self, food_name: str) -> dict[str, Any] | None:
        """Exact / fuzzy lookup in the built-in nutrient database."""
        # Exact match (case-insensitive)
        for key, data in FOOD_NUTRIENT_DATABASE.items():
            if key.lower() == food_name.lower():
                return {"name": key, **data}
        # Partial match
        for key, data in FOOD_NUTRIENT_DATABASE.items():
            if food_name.lower() in key.lower() or key.lower() in food_name.lower():
                return {"name": key, **data}
        return None

    def get_all_foods(self) -> list[str]:
        return list(FOOD_NUTRIENT_DATABASE.keys())

    def search_foods(self, query: str) -> list[dict[str, Any]]:
        """Return nutrient profiles for all foods matching the query string."""
        matches = []
        for key, data in FOOD_NUTRIENT_DATABASE.items():
            if query.lower() in key.lower():
                matches.append({"name": key, **data})
        return matches


# Module-level singleton
_rag = NutritionRAG()


def get_rag() -> NutritionRAG:
    return _rag
