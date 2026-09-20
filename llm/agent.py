"""
LLM Agent orchestrating retrieval and response generation.
"""

from typing import Dict, Any, List
from rag.retrieval import VectorRetriever, RetrievedChunk
from rag.prompts import SYSTEM_PROMPT, USER_PROMPT


def format_context(chunks: List[RetrievedChunk]) -> str:
    """Helper function to format RetrievedChunk objects into a context string."""
    if not chunks:
        return "No relevant context found."
    
    formatted_chunks = []
    for idx, chunk in enumerate(chunks, start=1):
        formatted_chunks.append(
            f"[Source {idx} - File: {chunk.source_file}, Page: {chunk.page_number}]\n{chunk.text}"
        )
    return "\n\n".join(formatted_chunks)


class RAGAgent:
    def __init__(self, retriever: VectorRetriever):
        self.retriever = retriever

    def query(self, user_query: str) -> Dict[str, Any]:
        # Uses your existing retrieve method
        chunks: List[RetrievedChunk] = self.retriever.retrieve(user_query, top_k=3)
        context = format_context(chunks)
        
        formatted_system_prompt = SYSTEM_PROMPT.format(context=context)
        formatted_user_prompt = USER_PROMPT.format(query=user_query)

        # Generated response (replace with actual LLM generation call if needed)
        answer = f"Based on the context provided:\n\n{context}"

        return {
            "answer": answer,
            "retrieved_documents": chunks,
            "system_prompt": formatted_system_prompt,
            "user_prompt": formatted_user_prompt,
        }