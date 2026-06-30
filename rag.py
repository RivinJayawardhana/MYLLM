"""
Lightweight retrieval for context-based Q&A (RAG), pure NumPy + regex.

No extra dependencies beyond numpy (already required by the project).
For higher-quality retrieval you can later swap TfidfRetriever for a
sentence-transformers embedding model — the .retrieve() interface stays the same.

Usage:
    from rag import TfidfRetriever
    r = TfidfRetriever.from_text(open("the-verdict.txt").read())
    for chunk, score in r.retrieve("What did the narrator think?", top_k=3):
        print(score, chunk[:80])
"""

import re
import math
from collections import Counter

import numpy as np

_WORD_RE = re.compile(r"\b\w+\b")


def tokenize(text):
    return _WORD_RE.findall(text.lower())


def chunk_text(text, chunk_words=120, overlap=40):
    """Split text into overlapping word windows.

    Paragraph boundaries are respected first; long paragraphs are then
    sliced into windows so each chunk stays roughly `chunk_words` long.
    """
    chunks = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    for para in paragraphs:
        words = para.split()
        if len(words) <= chunk_words:
            chunks.append(para)
            continue
        step = max(1, chunk_words - overlap)
        for start in range(0, len(words), step):
            window = words[start:start + chunk_words]
            if window:
                chunks.append(" ".join(window))
            if start + chunk_words >= len(words):
                break
    return chunks


class TfidfRetriever:
    def __init__(self, chunks):
        if not chunks:
            raise ValueError("No chunks to index.")
        self.chunks = chunks
        self._build()

    @classmethod
    def from_text(cls, text, chunk_words=120, overlap=40):
        return cls(chunk_text(text, chunk_words, overlap))

    @classmethod
    def from_texts(cls, texts, chunk_words=120, overlap=40):
        chunks = []
        for t in texts:
            chunks.extend(chunk_text(t, chunk_words, overlap))
        return cls(chunks)

    def _build(self):
        tokenized = [tokenize(c) for c in self.chunks]
        vocab = sorted({tok for doc in tokenized for tok in doc})
        self.vocab = {term: i for i, term in enumerate(vocab)}

        n_docs = len(tokenized)
        df = np.zeros(len(vocab))
        for doc in tokenized:
            for term in set(doc):
                df[self.vocab[term]] += 1
        # smoothed idf
        self.idf = np.log((1 + n_docs) / (1 + df)) + 1.0

        self.matrix = np.zeros((n_docs, len(vocab)), dtype=np.float32)
        for row, doc in enumerate(tokenized):
            self.matrix[row] = self._vectorize(doc)

    def _vectorize(self, tokens):
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        if not tokens:
            return vec
        counts = Counter(tokens)
        max_count = max(counts.values())
        for term, c in counts.items():
            idx = self.vocab.get(term)
            if idx is not None:
                tf = c / max_count
                vec[idx] = tf * self.idf[idx]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def retrieve(self, query, top_k=3):
        """Return [(chunk, score), ...] for the top_k most similar chunks."""
        q = self._vectorize(tokenize(query))
        scores = self.matrix @ q  # cosine sim (rows are unit-normalized)
        k = min(top_k, len(self.chunks))
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def context_for(self, query, top_k=3, max_chars=1200):
        """Concatenated top chunks, ready to drop into a prompt's ### Input."""
        parts, total = [], 0
        for chunk, score in self.retrieve(query, top_k=top_k):
            if score <= 0:
                continue
            if total + len(chunk) > max_chars and parts:
                break
            parts.append(chunk)
            total += len(chunk)
        return "\n\n".join(parts)
