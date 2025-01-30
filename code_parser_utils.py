"""
Some utilities for code parsing
"""


def remove_triple_backtics(input_text: str) -> str:
    """
    Remove triple backtics from a string
    """
    _text = input_text.replace("```python", "")
    _text = _text.replace("```", "")
    return _text
