import pytest

from modules.loader import clean_text, split_into_chunks


def test_clean_text_normalizes_whitespace_and_removes_page_numbers():
    text = "  Intro   text  \nPage 2\n\n  More\ttext  \n- 3 -\n"

    assert clean_text(text) == "Intro text\n\nMore text"


def test_split_into_chunks_respects_size_for_unpunctuated_text():
    chunks = split_into_chunks("abcdefghijklmnopqrstuvwxyz", chunk_size=10, overlap=0)

    assert chunks == ["abcdefghij", "klmnopqrst", "uvwxyz"]
    assert all(len(chunk) <= 10 for chunk in chunks)


def test_split_into_chunks_preserves_requested_overlap():
    chunks = split_into_chunks("abcdefghijklmnopqrstuvwxyz", chunk_size=10, overlap=3)

    assert len(chunks) > 1
    assert chunks[0][-3:] == chunks[1][:3]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (10, -1), (10, 10)],
)
def test_split_into_chunks_rejects_invalid_sizes(chunk_size, overlap):
    with pytest.raises(ValueError):
        split_into_chunks("some text", chunk_size=chunk_size, overlap=overlap)
