"""
Data processing utilities for relevance judgment evaluation.
Handles loading and processing of queries, documents, and relevance judgments.
"""

import os
import jsonlines
import pandas as pd
from typing import Dict, Tuple
from tqdm import tqdm
from pathlib import Path

def load_data_files(docs_path: str, queries_path: str, test_qrel_path: str) -> Tuple[Dict[str, str], Dict[str, str], pd.DataFrame]:
    """
    Load all required data files for relevance evaluation.
    
    Args:
        docs_path: Path to documents JSONL file
        queries_path: Path to queries TSV file
        test_qrel_path: Path to relevance judgments file
    
    Returns:
        Tuple containing:
        - Document mapping (docid -> document text)
        - Query mapping (qid -> query text)
        - Relevance judgments DataFrame
    """
    docid_to_doc = get_all_docid_to_doc(docs_path)
    qid_to_query = get_all_query_id_to_query(queries_path)
    test_qrel = pd.read_csv(
        test_qrel_path, 
        sep=" ", 
        header=None, 
        names=['qid', 'Q0', 'docid', 'rel_score']
    )
    return docid_to_doc, qid_to_query, test_qrel

def get_all_docid_to_doc(docs_path: str) -> Dict[str, str]:
    """
    Load documents from JSONL file into a dictionary.
    
    Args:
        docs_path: Path to documents JSONL file
    
    Returns:
        Dictionary mapping document IDs to document text
    """
    docid_to_doc = {}
    with jsonlines.open(docs_path, 'r') as document_file:
        for obj in document_file:
            docid_to_doc[obj['docid']] = obj['doc']
    return docid_to_doc

def get_all_query_id_to_query(query_path: str) -> Dict[str, str]:
    """
    Load queries from TSV file into a dictionary.
    
    Args:
        query_path: Path to queries TSV file
    
    Returns:
        Dictionary mapping query IDs to query text
    """
    query_data = pd.read_csv(
        query_path, 
        sep="\t", 
        header=None, 
        names=['qid', 'qtext']
    )
    return dict(zip(query_data.qid, query_data.qtext))

def process_documents_in_chunks(docid_to_doc: Dict[str, str], chunk_size: int):
    """
    Process documents in smaller chunks to manage memory.
    
    Args:
        docid_to_doc: Dictionary of documents
        chunk_size: Number of documents per chunk
    
    Yields:
        Dictionary containing a chunk of documents
    """
    doc_keys = list(docid_to_doc.keys())
    num_docs = len(doc_keys)
    
    for start_idx in tqdm(range(0, num_docs, chunk_size), 
                         desc="Processing documents", 
                         unit="chunk"):
        chunk_keys = doc_keys[start_idx:start_idx + chunk_size]
        chunk_docs = {k: docid_to_doc[k] for k in chunk_keys}
        yield chunk_docs

def clean_files(*file_paths: str) -> None:
    """
    Remove specified files if they exist.
    
    Args:
        file_paths: Variable number of file paths to clean
    """
    for file_path in file_paths:
        if file_path and Path(file_path).exists():
            Path(file_path).unlink()