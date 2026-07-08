"""Project-specific exception types for AtypEmu."""


class AtypEmuError(Exception):
    """Base exception for all AtypEmu failures."""


class ParseError(AtypEmuError):
    """Raised when input data cannot be parsed safely."""


class ValidationError(AtypEmuError):
    """Raised when an object fails a structural or logical validation."""


class ReweightingError(AtypEmuError):
    """Raised when a reweighting optimization cannot complete."""
