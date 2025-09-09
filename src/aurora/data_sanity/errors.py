class DataSanityError(Exception):
    """Exception raised for data sanity violations."""

def estring(code: str, detail: str = "") -> str:
    """Create standardized error string."""
    return f"{code}: {detail}" if detail else code
