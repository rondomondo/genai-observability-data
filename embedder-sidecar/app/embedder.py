"""Sentence-transformers model wrapper.

Loaded once at startup and shared across requests. encode() releases the GIL
so concurrent requests are handled by uvicorn's thread pool without blocking.
"""

import logging

from app.config import Settings

log = logging.getLogger(__name__)


class LocalEmbedder:
    def __init__(self, settings: Settings) -> None:
        from sentence_transformers import SentenceTransformer

        log.info("loading model %s", settings.embed_model)
        self._model = SentenceTransformer(settings.embed_model)
        self._model_name = settings.embed_model
        log.info("model loaded")

    def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: texts to embed.
            is_query: when True and the model supports a query prompt, apply it.

        Returns:
            A list of float vectors, one per input text.
        """
        prompt_name = "query" if is_query and "query" in (self._model.prompts or {}) else None
        kwargs: dict = {"prompt_name": prompt_name} if prompt_name else {}
        vectors = self._model.encode(texts, **kwargs)
        return [v.tolist() for v in vectors]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension() or 0
