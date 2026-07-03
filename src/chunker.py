import json
import re
from pathlib import Path

DATA_PATH = Path("data/silmarillion_clean.txt")
OUTPUT_PATH = Path("data/chunks.json")

CHUNK_SIZE_WORDS = 400  
OVERLAP_WORDS = 50     


def clean_heading(line: str) -> str:
    """
    Some headings in the source text have spaced-out letters
    (e.g. 'VA L A Q U E N T A' instead of 'VALAQUENTA') - an
    OCR/formatting artifact. Detect this by space density (almost
    every character is separated) and collapse those into one word.
    """
    stripped = line.strip()
    compact = re.sub(r"\s+", "", stripped)
    if compact.isupper() and len(compact) > 3:
        num_spaces = stripped.count(" ")
        if num_spaces >= len(compact) - 2:
            return compact
    return stripped


def is_heading(line: str) -> bool:
    """
    A line is treated as a heading if it's a 'Chapter N ...' line,
    or if it's short and ALL CAPS once whitespace is stripped out
    (covers 'AINULINDALË', 'VA L A Q U E N T A', 'QUENTA', etc.)
    """
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r"^Chapter\s+\d+", stripped):
        return True
    compact = re.sub(r"\s+", "", stripped)
    if 2 <= len(compact) <= 60 and compact.isupper() and not any(c.isdigit() for c in compact):
        return True
    return False


def parse_sections(text: str) -> list[dict]:
    lines = [l.strip() for l in text.split("\n")]

    sections = []
    current_title_parts: list[str] = []
    current_body_paragraphs: list[str] = []

    def flush_section():
        if current_body_paragraphs:
            title = " ".join(current_title_parts).strip() or "Untitled"
            body = " ".join(current_body_paragraphs).strip()
            if body:
                sections.append({"title": title, "body": body})

    for line in lines:
        if not line:
            continue
        if is_heading(line):
            if current_body_paragraphs:
                flush_section()
                current_body_paragraphs = []
                current_title_parts = []
            current_title_parts.append(clean_heading(line))
        else:
            current_body_paragraphs.append(line)

    flush_section() 
    return sections


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ö])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_section(title: str, body: str, section_index: int) -> list[dict]:
    sentences = split_into_sentences(body)
    chunks = []
    chunk_num = 0

    current_sentences: list[str] = []
    current_word_count = 0

    def flush_chunk():
        nonlocal chunk_num
        if not current_sentences:
            return
        chunk_text = " ".join(current_sentences)
        chunks.append({
            "id": f"s{section_index}_c{chunk_num}",
            "section": title,
            "text": chunk_text,
            "word_count": len(chunk_text.split()),
        })
        chunk_num += 1

    for sentence in sentences:
        current_sentences.append(sentence)
        current_word_count += len(sentence.split())

        if current_word_count >= CHUNK_SIZE_WORDS:
            flush_chunk()
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current_sentences):
                w = len(s.split())
                if overlap_count + w > OVERLAP_WORDS:
                    break
                overlap_sentences.insert(0, s)
                overlap_count += w

            current_sentences = overlap_sentences
            current_word_count = overlap_count

    flush_chunk()  
    return chunks


def build_chunks() -> list[dict]:
    raw_text = DATA_PATH.read_text(encoding="utf-8")
    sections = parse_sections(raw_text)

    all_chunks = []
    for i, section in enumerate(sections):
        section_chunks = chunk_section(section["title"], section["body"], i)
        all_chunks.extend(section_chunks)

    return all_chunks


if __name__ == "__main__":
    chunks = build_chunks()

    print(f"Parsed into chunks. Total chunks: {len(chunks)}")
    print(f"\nFirst chunk:")
    print(f"  Section: {chunks[0]['section']}")
    print(f"  Words: {chunks[0]['word_count']}")
    print(f"  Text preview: {chunks[0]['text'][:200]}...")

    print(f"\nA later chunk (index 20), to sanity-check chapter grouping:")
    print(f"  Section: {chunks[20]['section']}")
    print(f"  Words: {chunks[20]['word_count']}")
    print(f"  Text preview: {chunks[20]['text'][:200]}...")

    OUTPUT_PATH.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(chunks)} chunks to {OUTPUT_PATH}")