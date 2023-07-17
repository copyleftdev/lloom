import json

import pandas as pd
import pytest

from lloom.store.document import Corpus, Document

TEST_DB_DIR = "./test_db"


def test_Document_initialization():
    document = Document(
        document="Test Document",
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    assert document.document == "Test Document"
    document_from_chroma = Document(
        collection_name="test_collection", id=document.id, persistent_dir=TEST_DB_DIR
    )
    assert document_from_chroma.document == "Test Document"


def test_Document_update():
    document = Document(
        document="Test Document",
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    document_from_chroma = Document(
        collection_name="test_collection", id=document.id, persistent_dir=TEST_DB_DIR
    )
    assert document_from_chroma.document == "Test Document"
    document.update(document="Updated Document")
    document_from_chroma = Document(
        collection_name="test_collection", id=document.id, persistent_dir=TEST_DB_DIR
    )
    assert document_from_chroma.document == "Updated Document"
    assert document.document == "Updated Document"


def test_Corpus_initialization():
    corpus = Corpus(
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    assert len(corpus) == 0


def test_Corpus_append():
    document = Document(
        document="Test Document",
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    corpus = Corpus(
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    corpus.append(document)
    assert len(corpus) == 1
    assert corpus[0] == document


def test_Corpus_iteration():
    document1 = Document(
        document="Test Document 1",
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    document2 = Document(
        document="Test Document 2",
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    corpus = Corpus(
        collection_name="test_collection",
        documents=[document1, document2],
        persistent_dir=TEST_DB_DIR,
    )

    for i, document in enumerate(corpus):
        assert document == corpus[i]


def test_Corpus_str():
    document1 = Document(
        document="Test Document 1",
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    document2 = Document(
        document="Test Document 2",
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    corpus = Corpus(
        collection_name="test_collection",
        documents=[document1, document2],
        persistent_dir=TEST_DB_DIR,
    )

    assert str(corpus) == "\n\n".join(
        [document.document for document in corpus.documents]
    )


def test_Corpus_merge_documents():
    document1 = Document(
        document="Test Document 1",
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    document2 = Document(
        document="Test Document 2",
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    corpus = Corpus(
        collection_name="test_collection",
        documents=[document1, document2],
        persistent_dir=TEST_DB_DIR,
    )

    assert corpus.merge_documents() == "\n\n".join(
        [document.document for document in corpus.documents]
    )


def test_Corpus_to_dict():
    document1 = Document(
        document="Test Document 1",
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    document2 = Document(
        document="Test Document 2",
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    corpus = Corpus(
        collection_name="test_collection",
        documents=[document1, document2],
        persistent_dir=TEST_DB_DIR,
    )

    documents_dict = corpus.to_dict()
    assert isinstance(documents_dict, list)
    assert len(documents_dict) == len(corpus)
    for document, document_dict in zip(corpus, documents_dict):
        document_dict_from_obj = {
            k: v for k, v in document.__dict__.items() if k != "chroma_client"
        }
        assert document_dict_from_obj == document_dict


def test_Corpus_to_json():
    document1 = Document(
        document="Test Document 1",
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    document2 = Document(
        document="Test Document 2",
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    corpus = Corpus(
        collection_name="test_collection",
        documents=[document1, document2],
        persistent_dir=TEST_DB_DIR,
    )

    documents_json = corpus.to_json()
    assert isinstance(documents_json, str)
    assert documents_json == json.dumps(corpus.to_dict())


def test_Corpus_to_pandas():
    document1 = Document(
        document="Test Document 1",
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    document2 = Document(
        document="Test Document 2",
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    corpus = Corpus(
        collection_name="test_collection",
        documents=[document1, document2],
        persistent_dir=TEST_DB_DIR,
    )

    df = corpus.to_pandas()
    assert isinstance(df, pd.DataFrame)
    expected_columns = list(
        {k for k in document1.__dict__.keys() if k != "chroma_client"}
    )
    assert set(df.columns).issubset(set(expected_columns)) and set(
        expected_columns
    ).issubset(set(df.columns))
    assert len(df) == len(corpus)


@pytest.fixture
def example_document():
    return Document(
        document="example document",
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )


def test_corpus_init(example_document):
    corpus = Corpus(
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    assert len(corpus) == 0

    corpus = Corpus(
        collection_name="test_collection",
        documents=[example_document],
        persistent_dir=TEST_DB_DIR,
    )
    assert len(corpus) == 1


def test_corpus_get():
    example_document = Document(
        document="another example document",
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    corpus = Corpus(
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    document_id = example_document.id

    corpus.get(ids=[document_id])
    assert len(corpus) == 1
    assert corpus[0].id == document_id
    assert corpus[0].document == example_document.document


def test_corpus_query():
    corpus = Corpus(
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    example_document = Document(
        document="yet another example document",
        collection_name="test_collection",
        persistent_dir=TEST_DB_DIR,
    )
    corpus.append(example_document)
    document_id = example_document.id

    corpus.query(query_texts=example_document.document, n_results=1)
    assert len(corpus) == 1
    assert corpus[0].id == document_id
    assert corpus[0].document == example_document.document
