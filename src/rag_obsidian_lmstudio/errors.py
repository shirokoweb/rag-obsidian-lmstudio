"""Exception hierarchy for the library.

Library code raises these — never SystemExit — so embedding callers
(e.g. the MCP server) can always recover with ``except RagError``.
"""


class RagError(Exception):
    """Base class for all errors raised by this package."""


class ConfigError(RagError):
    """Invalid or missing configuration."""


class LMStudioError(RagError):
    """LM Studio server unreachable or returned an unusable response."""


class EmbeddingError(LMStudioError):
    """Embedding request failed."""


class ChatError(LMStudioError):
    """Chat completion request failed."""


class IndexingError(RagError):
    """The document index could not be built."""
