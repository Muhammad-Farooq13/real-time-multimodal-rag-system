from __future__ import annotations

from app.services.vector_store import SearchResult


class ResponseGenerator:
    def generate(self, query_text: str, contexts: list[SearchResult], fast_mode: bool) -> str:
        if not contexts:
            return "I do not have enough grounded context to answer confidently."

        snippets = [ctx.text.strip().replace("\n", " ")[:220] for ctx in contexts]

        if fast_mode:
            top = snippets[0]
            return (
                f"Based on retrieved context: {top}. "
                "This answer is grounded in the indexed documents and should be treated as context-bound."
            )

        merged = " ".join(snippets[:3])
        return (
            f"Grounded summary: {merged} "
            "Use citations for verification and request deeper analysis for high-risk decisions."
        )
