"""PDF text extraction and lightweight text cleanup/chunking helpers."""

import re

from pypdf import PdfReader


def read_pdf(path, return_pages=False):
    """Read a PDF, optionally returning the extracted text for each page."""
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages)
    return (text, pages) if return_pages else text


def clean_text(text):
    """Normalize whitespace and remove page-number-only lines."""
    cleaned_lines = []
    page_number_line = re.compile(
        r"^(?:page\s+)?(?:[-\u2013\u2014]\s*)?\d+(?:\s*[-\u2013\u2014])?$",
        re.IGNORECASE,
    )

    for line in str(text).splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if page_number_line.fullmatch(line):
            continue
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        cleaned_lines.append(line)

    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()
    return "\n".join(cleaned_lines)


def split_into_chunks(text, chunk_size=800, overlap=150, page_texts=None):
    """Split text into chunks, optionally recording each chunk's page range."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []

    # Keep closing quotes/brackets with the sentence-ending punctuation.
    sentence_ends = [
        match.end()
        for match in re.finditer(
            r"(?<=[.!?])[\"'\u2019\u201d)}\]]*(?=\s|$)", text
        )
    ]

    page_spans = []
    if page_texts is not None:
        page_cursor = 0
        for page_number, page_text in enumerate(page_texts, start=1):
            normalized_page = re.sub(r"\s+", " ", clean_text(page_text)).strip()
            if not normalized_page:
                continue
            if page_spans:
                page_cursor += 1  # The normalized page separator is one space.
            page_spans.append((page_cursor, page_cursor + len(normalized_page), page_number))
            page_cursor += len(normalized_page)

    chunks = []
    start = 0
    while start < len(text):
        target = min(start + chunk_size, len(text))
        end = target

        if target < len(text):
            earlier = [point for point in sentence_ends if start < point <= target]
            # Avoid making a much-too-short chunk just to hit an early period.
            minimum_end = start + int(chunk_size * 0.6)
            earlier = [point for point in earlier if point >= minimum_end]
            if earlier:
                end = earlier[-1]
            else:
                later = [
                    point
                    for point in sentence_ends
                    if target < point <= start + int(chunk_size * 1.3)
                ]
                if later:
                    end = later[0]

        if end <= start:
            end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if page_texts is None:
            chunks.append(chunk_text)
        else:
            touched_pages = [
                page_number
                for page_start, page_end, page_number in page_spans
                if page_start < end and page_end > start
            ]
            chunks.append(
                {
                    "text": chunk_text,
                    "page_start": min(touched_pages) if touched_pages else None,
                    "page_end": max(touched_pages) if touched_pages else None,
                }
            )
        if end >= len(text):
            break

        next_start = max(start + 1, end - overlap)
        # Start the overlap at a sentence boundary when one is nearby.
        boundary = next((point for point in sentence_ends if next_start <= point < end), None)
        start = boundary if boundary is not None else next_start

    return chunks


if __name__ == "__main__":
    sample = (
        "StudyPilot turns course material into manageable notes. "
        "It removes page numbers and extra whitespace. "
        "Then it splits the cleaned text into overlapping chunks. "
    ) * 40
    cleaned_sample = clean_text(sample)
    sample_chunks = split_into_chunks(cleaned_sample)
    print(f"Chunks made: {len(sample_chunks)}")
    first_chunk = sample_chunks[0] if sample_chunks else ""
    second_chunk = sample_chunks[1] if len(sample_chunks) > 1 else ""
    max_shared = min(150, len(first_chunk), len(second_chunk))
    has_overlap = any(
        first_chunk[-size:] == second_chunk[:size]
        for size in range(20, max_shared + 1)
    )
    print(f"First chunk length: {len(first_chunk)}")
    print(f"Second chunk starts with text from the end of the first: {has_overlap}")
