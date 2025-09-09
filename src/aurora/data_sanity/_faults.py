import os


def disabled(code: str) -> bool:
    """Return True if the given rule code is disabled via DS_DISABLE_RULE env.

    Accepts comma-separated list, e.g., "DUP_TS,NON_MONO_INDEX".
    Test-only hook; production ignores unless env is set.
    """
    env = os.getenv("DS_DISABLE_RULE", "")
    if not env:
        return False
    targets = {tok.strip() for tok in env.split(",") if tok.strip()}
    return code in targets


