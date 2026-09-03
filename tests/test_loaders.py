import pytest
from app.rag.loaders import DocumentLoader, SUPPORTED_EXTENSIONS


def test_discover_finds_txt_and_md(sample_raw_dir):
    loader = DocumentLoader(sample_raw_dir)
    files = loader.discover()
    assert len(files) == 2
    extensions = {f.suffix for f in files}
    assert extensions <= SUPPORTED_EXTENSIONS


def test_load_file_txt(sample_raw_dir):
    loader = DocumentLoader(sample_raw_dir)
    doc = loader.load_file(sample_raw_dir / "doc1.txt")
    assert doc.document_id
    assert doc.source == "doc1.txt"
    assert "Wheat rust" in doc.text
    assert doc.metadata["filename"] == "doc1.txt"
    assert doc.metadata["extension"] == ".txt"


def test_load_file_md(sample_raw_dir):
    loader = DocumentLoader(sample_raw_dir)
    doc = loader.load_file(sample_raw_dir / "doc2.md")
    assert doc.source == "doc2.md"
    assert "Cotton" in doc.text


def test_deterministic_document_ids(sample_raw_dir):
    loader = DocumentLoader(sample_raw_dir)
    doc1_a = loader.load_file(sample_raw_dir / "doc1.txt")
    doc1_b = loader.load_file(sample_raw_dir / "doc1.txt")
    assert doc1_a.document_id == doc1_b.document_id


def test_different_files_different_ids(sample_raw_dir):
    loader = DocumentLoader(sample_raw_dir)
    doc1 = loader.load_file(sample_raw_dir / "doc1.txt")
    doc2 = loader.load_file(sample_raw_dir / "doc2.md")
    assert doc1.document_id != doc2.document_id


def test_load_all(sample_raw_dir):
    loader = DocumentLoader(sample_raw_dir)
    docs = loader.load_all()
    assert len(docs) == 2


def test_unsupported_extension_raises(tmp_dir):
    loader = DocumentLoader(tmp_dir)
    bad_file = tmp_dir / "data.csv"
    bad_file.write_text("a,b,c", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        loader.load_file(bad_file)


def test_empty_file_raises(sample_raw_dir):
    loader = DocumentLoader(sample_raw_dir)
    empty_file = sample_raw_dir / "empty.txt"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        loader.load_file(empty_file)


def test_empty_directory(empty_raw_dir):
    loader = DocumentLoader(empty_raw_dir)
    assert loader.discover() == []
    assert loader.load_all() == []


def test_nonexistent_directory(tmp_dir):
    loader = DocumentLoader(tmp_dir / "does_not_exist")
    assert loader.discover() == []
    assert loader.load_all() == []


def test_load_all_skips_bad_files(sample_raw_dir):
    (sample_raw_dir / "empty.txt").write_text("", encoding="utf-8")
    loader = DocumentLoader(sample_raw_dir)
    docs = loader.load_all()
    assert len(docs) == 2
