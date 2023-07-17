import pytest

from lloom.dataset.csvfile_dataset import CSVfileDataset
from lloom.dataset.textfile_dataset import TextfileDataset
from lloom.store.document import Corpus

TEST_DB_DIR = "./test_db"


def test_load_textfile_dataset(tmp_path):
    text_file = tmp_path / "example.txt"
    text_file.write_text("This is the content of the text file.")

    dataset = TextfileDataset(
        source=str(text_file),
        format="txt",
        tokens_per_document=10,
        token_overlap=5,
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    corpus = Corpus(
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    ids = dataset.load()
    corpus.get(ids=ids)

    assert len(corpus.documents) == 2
    assert corpus.documents[0].document == "This is th"
    assert corpus.documents[1].document == "is the con"


def test_load_csvfile_dataset(tmp_path):
    csv_file = tmp_path / "example.csv"
    csv_file.write_text("text\nDocument 1\nDocument 2\nDocument 3")

    dataset = CSVfileDataset(
        source=str(csv_file),
        format="csv",
        tokens_per_document=10,
        token_overlap=5,
        text_field="text",
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    corpus = Corpus(
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    ids = dataset.load()
    corpus.get(ids=ids)

    assert len(corpus.documents) == 3
    assert corpus.documents[0].document == "Document 1"
    assert corpus.documents[1].document == "Document 2"
    assert corpus.documents[2].document == "Document 3"


def test_load_csvfile_dataset_invalid_field(tmp_path):
    csv_file = tmp_path / "example.csv"
    csv_file.write_text("column1,column2\nValue 1,Value 2")

    with pytest.raises(
        ValueError, match=r"Text field 'text' not found in the CSV file."
    ):
        dataset = CSVfileDataset(
            source=str(csv_file),
            format="csv",
            tokens_per_document=10,
            token_overlap=5,
            text_field="text",
            collection_name="test_collection",
            persistent_dir=TEST_DB_DIR,
        )
        dataset.load()


def test_load_csvfile_dataset_no_files(tmp_path):
    with pytest.raises(
        ValueError, match=r"No files found matching the provided glob pattern."
    ):
        dataset = CSVfileDataset(
            source=str(tmp_path / "*.csv"),
            format="csv",
            tokens_per_document=10,
            token_overlap=5,
            text_field="text",
            collection_name="test_collection",
            persistent_dir=TEST_DB_DIR,
        )
        dataset.load()
