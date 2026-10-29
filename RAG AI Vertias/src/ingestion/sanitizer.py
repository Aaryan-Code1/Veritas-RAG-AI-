import re


class PIISanitizer:
    """Detects and redacts sensitive PII and secrets prior to vector indexing."""

    # Regex patterns for common sensitive enterprise data
    SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")
    API_KEY_PATTERN = re.compile(r"\b(?:sk-[a-zA-Z0-9]{32,}|AKIA[0-9A-Z]{16})\b")
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")

    @classmethod
    def sanitize(cls, text: str, redact_emails: bool = False) -> str:
        """Applies sanitization patterns and replaces secrets with standard tokens."""
        if not text:
            return ""

        sanitized = cls.SSN_PATTERN.sub("[REDACTED_SSN]", text)
        sanitized = cls.CREDIT_CARD_PATTERN.sub("[REDACTED_CARD]", sanitized)
        sanitized = cls.API_KEY_PATTERN.sub("[REDACTED_API_KEY]", sanitized)

        if redact_emails:
            sanitized = cls.EMAIL_PATTERN.sub("[REDACTED_EMAIL]", sanitized)

        return sanitized
