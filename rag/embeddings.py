"""Embedding generation service using local Sentence Transformers."""

from typing import Sequence
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging_config import logger


class EmbeddingError(Exception):
    """Exception raised when embedding generation fails."""


class EmbeddingService:
    """Local embedding service wrapping SentenceTransformer."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model to avoid startup delays when not needed."""
        if self._model is None:
            logger.info(f"Loading SentenceTransformer embedding model: {self.model_name}")
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.error(f"Failed to load embedding model {self.model_name}: {e}")
                raise EmbeddingError(f"Could not load embedding model '{self.model_name}': {e}") from e
        return self._model

    @property
    def dimension(self) -> int:
        """Get the embedding vector dimension."""
        if hasattr(self.model, "get_embedding_dimension"):
            return int(self.model.get_embedding_dimension())
        return int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Generate L2-normalized embeddings for a sequence of document texts.

        Args:
            texts: List or sequence of text strings.

        Returns:
            2D numpy array of shape (N, dimension) with dtype float32.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        try:
            embeddings = self.model.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Error generating document embeddings: {e}")
            raise EmbeddingError(f"Failed to generate document embeddings: {e}") from e

    def embed_query(self, query: str) -> np.ndarray:
        """Generate an L2-normalized 1D embedding for a search query.

        Args:
            query: Query text string.

        Returns:
            1D numpy array of shape (dimension,) with dtype float32.
        """
        if not query or not query.strip():
            raise EmbeddingError("Query text cannot be empty.")

        try:
            embedding = self.model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise EmbeddingError(f"Failed to generate query embedding: {e}") from e
