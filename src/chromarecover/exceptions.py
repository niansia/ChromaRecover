"""Public error contract."""


class ChromaRecoverError(Exception):
    """Base error for invalid inputs and unsupported operations."""


class InvalidInputError(ChromaRecoverError):
    """The input cannot be decoded safely."""


class UnsupportedFormatError(ChromaRecoverError):
    """The image format is not supported."""

