# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 23:05:27 2025

@author: dayli
"""

import os
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  

from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

from config import Config

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)



def load_pdfs_as_documents():
    """Task 1: Load PDF files and return Document objects."""

    data_folder = Path(Config.DATA_FOLDER)

    if not data_folder.exists():
        raise FileNotFoundError(f"Data folder not found: {data_folder.resolve()}")

    all_docs = []

    # (a) Scan /data for PDF files
    pdf_files = sorted(data_folder.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in /data.")

    for pdf_path in pdf_files:
        print(f"Loading PDF: {pdf_path.name}")

        # (b) Use PyPDFLoader to extract text from each page
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()  # list[Document]

        # (c) Each page already has page_content + metadata
        all_docs.extend(pages)

    print(f"Total pages loaded: {len(all_docs)}")
    return all_docs

#TASK 2

def split_documents(documents):
    """
    Task 2: Split documents into smaller chunks using RecursiveCharacterTextSplitter.

    Uses:
      - chunk_size = 1000
      - chunk_overlap = 100
    Prints:
      - total number of chunks
      - first 500 characters of the first chunk
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    if chunks:
        first_chunk = chunks[0]
        print("\nPreview of first chunk (first 500 characters):")
        print(first_chunk.page_content[:500])

    return chunks

#TASK 3
def build_vector_store(chunks):
    """
    Task 3: Build the Vector Store (ChromaDB) from document chunks.

    - Uses SentenceTransformerEmbeddings with model 'all-MiniLM-L6-v2'
    - Creates a Chroma vector store with:
        collection_name = 'rag_<studentid>'
        persist_directory = './chroma_db'
    - Prints a confirmation message when ready.
    """

    # a) Download / load the embeddings model
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # b) Build the Chroma vector store from the chunks
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=Config.COLLECTION_NAME,
        persist_directory=Config.CHROMA_DIR,
    )

    # Persist to disk so it can be reused in future runs
    vector_store.persist()

    print("Vector store ready for retrieval.")

    return vector_store


if __name__ == "__main__":
    # Task 1: load PDF pages
    docs = load_pdfs_as_documents()

    # Task 2: split into chunks
    chunks = split_documents(docs)

    # Task 3: build vector store
    vector_store = build_vector_store(chunks)

