def sanitize_input(text: str) -> str:
    """Removes leading/trailing whitespace and converts to lowercase."""
    return text.strip().lower()

def validate_words(words: list, vocab: set) -> tuple:
    """
    Checks a list of words against the model vocabulary.
    Returns a tuple: (list_of_valid_words, list_of_invalid_words)
    """
    valid = [w for w in words if w in vocab]
    invalid = [w for w in words if w not in vocab]
    return valid, invalid
