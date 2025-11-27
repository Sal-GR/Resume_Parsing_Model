import random
import re
from collections import Counter

# ---------------------------
# Zero-width character attack
# ---------------------------

ZERO_WIDTH = "\u200b"  # zero-width space

def insert_zero_width(text: str, p: float = 0.15) -> str:
    """
    Insert zero-width characters after some alphabetic characters
    with probability p. Keeps text visually almost identical.
    """
    if not isinstance(text, str):
        return ""
    out_chars = []
    for ch in text:
        out_chars.append(ch)
        if ch.isalpha() and random.random() < p:
            out_chars.append(ZERO_WIDTH)
    return "".join(out_chars)


# ---------------------------
# Homoglyph substitution
# ---------------------------

HOMOGLYPH_MAP = {
    # lowercase latin -> cyrillic
    "a": "а",
    "e": "е", 
    "o": "о", 
    "p": "р", 
    "c": "с", 
    "x": "х",  
    "y": "у",  
    # uppercase latin -> greek
    "A": "Α", 
    "B": "Β", 
    "E": "Ε",  
    "H": "Η",
    "K": "Κ",
    "M": "Μ", 
    "O": "Ο",
    "P": "Ρ",
    "T": "Τ",
    "X": "Χ",
    "Y": "Υ",
}

def homoglyph_substitute(text: str, p: float = 0.2) -> str:
    """
    Replace some ASCII letters with visually similar Unicode homoglyphs.
    """
    if not isinstance(text, str):
        return ""
    out_chars = []
    for ch in text:
        if ch in HOMOGLYPH_MAP and random.random() < p:
            out_chars.append(HOMOGLYPH_MAP[ch])
        else:
            out_chars.append(ch)
    return "".join(out_chars)


# ---------------------------
# Keyword stuffing attack
# ---------------------------

def _get_candidate_keywords(text: str, top_k: int = 8):
    """
    Very simple keyword extractor:
    - split on whitespace
    - keep tokens of length >= 4
    - ignore numbers and punctuation-only tokens
    - return top_k most common lowercase tokens
    """
    if not isinstance(text, str):
        return []

    tokens = re.findall(r"[A-Za-z][A-Za-z\-]+", text)  # crude but ok here
    tokens = [t.lower() for t in tokens if len(t) >= 4]
    if not tokens:
        return []
    counts = Counter(tokens)
    common = [w for w, _ in counts.most_common(top_k)]
    return common

def keyword_stuffing(text: str, label: str = "", repeat: int = 5) -> str:
    """
    Add a block of repeated keywords at the end of the resume.
    Simulates keyword stuffing for ATS gaming.
    """
    if not isinstance(text, str):
        return ""

    keywords = _get_candidate_keywords(text)
    if label:
        keywords.append(str(label).lower())

    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            uniq.append(k)

    if not uniq:
        return text

    stuffed = []
    for k in uniq[:10]:  # cap number of unique keywords
        stuffed.extend([k] * repeat)

    stuffing_block = "\n\nKEYWORDS: " + " ".join(stuffed) + "\n"
    return text + stuffing_block


# ---------------------------
# Combined / multi-attack
# ---------------------------

def combo_attack(text: str, label: str = "") -> str:
    """
    Apply multiple attacks in sequence:
    1) homoglyph substitution
    2) zero-width insertion
    3) keyword stuffing
    """
    t = homoglyph_substitute(text, p=0.2)
    t = insert_zero_width(t, p=0.1)
    t = keyword_stuffing(t, label=label, repeat=3)
    return t
