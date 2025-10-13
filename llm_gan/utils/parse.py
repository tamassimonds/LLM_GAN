"""Helpers for extracting LLM outputs wrapped in simple XML-like tags."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

_TAG_PATTERN = re.compile(r"<(?P<tag>[A-Za-z0-9_:-]+)>(?P<content>.*?)</(?P=tag)>", re.DOTALL)
_BOXED_PATTERN = re.compile(r"\\boxed\{([^}]+)\}", re.DOTALL)

ParsedTags = Dict[str, Union[str, List[str]]]


def parse_tags(
    text: str,
    tags: Optional[Union[str, Sequence[str]]] = None,
    *,
    strict: bool = False,
) -> Union[ParsedTags, str, Tuple[Optional[Union[str, List[str]]], ...], None]:
    """Extract content wrapped in custom tags from ``text``.

    Parameters
    ----------
    text:
        Input string containing zero or more segments such as
        ``<answer>value</answer>``.
    tags:
        When ``None`` (default) every tag found is returned as a dictionary.  If
        a single tag name (``str``) or a sequence of tags is provided, only
        those tags are returned.  Missing tags yield ``None`` unless ``strict``
        is true.
    strict:
        When true, raise ``ValueError`` if any requested tag is missing.

    Returns
    -------
    dict | str | tuple | None
        ``dict`` of {tag: value} when ``tags`` is ``None``.  When a specific tag
        (string) is requested, the value for that tag (or ``None``) is returned.
        For a sequence of tags, a tuple of values is produced in the same order.

    Notes
    -----
    - Multiple occurrences of the same tag are returned as a list.
    - Leading/trailing whitespace inside tags is stripped.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    matches: Dict[str, List[str]] = {}
    for match in _TAG_PATTERN.finditer(text):
        tag = match.group("tag")
        content = match.group("content").strip()
        matches.setdefault(tag, []).append(content)

    if tags is None:
        return {tag: values[0] if len(values) == 1 else values for tag, values in matches.items()}

    tag_list: List[str]
    if isinstance(tags, str):
        tag_list = [tags]
    elif isinstance(tags, Sequence):
        tag_list = list(tags)
    else:
        raise TypeError("tags must be None, a string, or a sequence of strings")

    extracted: List[Optional[Union[str, List[str]]]] = []
    for tag in tag_list:
        tag_values = matches.get(tag)
        if not tag_values:
            if strict:
                raise ValueError(f"Required tag '{tag}' not found in text: {text}")
            extracted.append(None)
            continue
        extracted.append(tag_values[0] if len(tag_values) == 1 else tag_values)

    if isinstance(tags, str):
        return extracted[0]

    return tuple(extracted)


def parse_boxed(text: str) -> Optional[str]:
    """Extract content from \\boxed{} format.
    
    Parameters
    ----------
    text:
        Input string that may contain \\boxed{content}.
        
    Returns
    -------
    str | None:
        The content inside the last \\boxed{} found, or None if none found.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    
    matches = _BOXED_PATTERN.findall(text)
    if matches:
        return matches[-1].strip()  # Return the last match
    return None
