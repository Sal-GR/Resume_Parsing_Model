import re
import os
import fitz  # PyMuPDF
import docx
import spacy
from bs4 import BeautifulSoup

nlp = spacy.load("en_core_web_sm")

# -------------------------------------
# TEXT EXTRACTION
# -------------------------------------

def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    else:
        # assume plaintext / .txt
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def extract_text_from_pdf(path: str) -> str:
    text = ""
    doc = fitz.open(path)
    for page in doc:
        text += page.get_text()
    return text


def extract_text_from_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])


# -------------------------------------
# FIELD EXTRACTION
# -------------------------------------

def extract_email(text: str):
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return match.group(0) if match else None


def extract_phone(text: str):
    match = re.search(
        r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}", 
        text
    )
    return match.group(0) if match else None


def extract_name(text: str):
    """
    Uses spaCy's NER to detect PERSON entity near the beginning of the document.
    """
    doc = nlp(text[:400])  # look only at header
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()
    return None


def extract_skills(text: str):
    """
    Extract skills by:
    - scanning the text for common skill sections
    - OR extracting nouns using spaCy
    """
    skills = set()

    # Look for explicit "Skills" section
    skill_sections = re.findall(
        r"(Skills?|Technical Skills|Skills Summary):?\s*(.+?)(\n\n|\Z)", 
        text, 
        flags=re.IGNORECASE | re.DOTALL
    )
    for _, block, _ in skill_sections:
        tokens = re.findall(r"[A-Za-z][A-Za-z+/.#&\- ]{1,40}", block)
        for t in tokens:
            if len(t.strip()) > 2:
                skills.add(t.strip())

    # Also extract nouns & proper nouns (simple heuristic)
    doc = nlp(text)
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and len(token.text) > 2:
            skills.add(token.text)

    # Limit list
    return list(sorted(skills))[:50]


def extract_sections(text: str):
    """
    Break text into logical resume sections (Experience, Education, Summary, etc.)
    """
    text_clean = text.replace("\r", "")
    lines = text_clean.split("\n")

    sections = {}
    current = "Other"
    sections[current] = []

    for line in lines:
        header = line.strip().lower()
        if any(h in header for h in ["experience", "work history", "employment"]):
            current = "Experience"
            sections[current] = []
        elif any(h in header for h in ["education", "academic"]):
            current = "Education"
            sections[current] = []
        elif "summary" in header:
            current = "Summary"
            sections[current] = []
        elif "skills" in header:
            current = "Skills"
            sections[current] = []

        sections[current].append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items()}


# -------------------------------------
# MAIN PARSING FUNCTION
# -------------------------------------

def parse_resume(path: str) -> dict:
    text = extract_text(path)

    sections = extract_sections(text)

    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text),
        "sections": sections,
        "raw_text": text
    }


# -------------------------------------
# TEST
# -------------------------------------

if __name__ == "__main__":
    test_file = "test_resume.txt"
    data = parse_resume(test_file)
    print(data)
