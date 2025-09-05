import hashlib

def hash_text(text: str) -> str:
    """Generate a SHA-256 hash for the given text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()