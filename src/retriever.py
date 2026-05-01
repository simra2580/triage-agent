"""
src/retriever.py  —  TF-IDF Corpus Retrieval

Builds a TF-IDF index over all loaded corpus chunks.
Supports optional company-domain filtering.
Returns top-K ranked chunks with relevance scores.
"""

from __future__ import annotations

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Company → domain aliases used in corpus metadata
COMPANY_ALIASES: dict[str, list[str]] = {
    "HackerRank": ["hackerrank", "hr", "coding", "assessment"],
    "Claude":     ["claude", "anthropic", "ai"],
    "Visa":       ["visa", "payment", "card", "financial"],
    "None":       [],
}


class Retriever:
    """
    TF-IDF retriever over a flat list of corpus chunks.

    Each chunk dict must have:
        {
            "text":    str,
            "source":  str,   # e.g. "hackerrank", "claude", "visa"
            "section": str,   # optional heading
        }
    """

    def __init__(self, corpus: list[dict]):
        if not corpus:
            raise ValueError("Corpus is empty — run corpus scraper or add text files to data/corpus/")

        self.corpus = corpus
        self.texts  = [c["text"] for c in corpus]

        # Fit TF-IDF on all chunks
        self.vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            ngram_range=(1, 2),        # unigrams + bigrams
            max_df=0.90,               # ignore near-universal terms
            min_df=1,
            sublinear_tf=True,         # apply log normalization
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    # ──────────────────────────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        company_filter: str | None = None,
        top_k: int = 4,
    ) -> list[dict]:
        """
        Returns top_k most relevant corpus chunks for the query.

        If company_filter is given, results from that company's domain
        get a 1.5× relevance boost (soft filter, doesn't exclude others).
        """
        query_vec = self.vectorizer.transform([self._clean(query)])
        scores    = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Apply company boost
        if company_filter:
            aliases = COMPANY_ALIASES.get(company_filter, [])
            for i, chunk in enumerate(self.corpus):
                src = chunk.get("source", "").lower()
                if any(alias in src for alias in aliases):
                    scores[i] *= 1.5   # boost same-domain chunks

        # Rank
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    **self.corpus[idx],
                    "score": float(scores[idx]),
                })

        return results

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _clean(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()