"""Stable exception taxonomy for callers of :mod:`rag`."""


class RagError(Exception):
    """Base class for expected RAG failures."""


class ConfigurationError(RagError, ValueError):
    pass


class SourceError(RagError):
    pass


class SourceTooLargeError(SourceError):
    pass


class UnsupportedSourceError(SourceError):
    pass


class ParserError(RagError):
    pass


class EncryptedDocumentError(ParserError):
    pass


class EmptyDocumentError(ParserError):
    pass


class OCRError(ParserError):
    pass


class ChunkingError(RagError):
    pass


class ModelError(RagError):
    pass


class ModelCapabilityError(ModelError):
    pass


class EmbeddingError(ModelError):
    pass


class EmbeddingIdentityError(EmbeddingError):
    pass


class StorageError(RagError):
    pass


class StorageConsistencyError(StorageError):
    pass


class LifecycleError(RagError):
    pass


class ConcurrentUpdateError(LifecycleError):
    pass


class RetrievalError(RagError):
    pass


class RerankerError(RetrievalError):
    pass


class RerankerTimeoutError(RerankerError, TimeoutError):
    pass


__all__ = [name for name in tuple(globals()) if name.endswith("Error")]
