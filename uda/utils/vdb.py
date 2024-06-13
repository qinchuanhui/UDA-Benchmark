# functions for vector db and gpt
import PyPDF2
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import logging
import time
from ragatouille import RAGPretrainedModel


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def log_duration(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logging.info(
            f"Function '{func.__name__}' took {duration:.4f} seconds to execute."
        )
        return result

    return wrapper


# @log_duration
def extract_text_from_pdf(pdf_path):
    # Open the PDF file in binary mode
    with open(pdf_path, "rb") as file:
        # Create a PDF file reader object
        reader = PyPDF2.PdfReader(file, strict=False)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    # print("pdf_words:", len(text.split()))
    return text


def split_text(pdf_text, chunk_size=3000, chunk_overlap=300):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # chunk size in characters not in words
        chunk_overlap=chunk_overlap,  # no overlap
    )
    text_chunks = text_splitter.split_text(pdf_text)
    # print("chunk_num:", len(text_chunks))
    # print("per_chunk_len:", len(text_chunks[0].split()))
    return text_chunks


def colbert_retrieve(index_path, query, top_k=30):
    RAG = RAGPretrainedModel.from_index(index_path)
    results = RAG.search(query, k=top_k)
    top_res = results[:top_k]
    contexts = [res["content"] for res in top_res]
    return contexts


def colbert_index(chunks, name="my_index"):
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0", verbose=False)
    index_path = RAG.index(index_name=name, collection=chunks, split_documents=False)
    return index_path


@log_duration
def store_chunks(chunks, db_name, model_name="all-mpnet-base-v2"):
    # model_name = "all-MiniLM-L6-v2"
    chroma_client = chromadb.Client()
    if model_name == "openai":
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key="abcdefg",
            model_name="text-embedding-3-large",
            api_base="https://abcdefg.openai.azure.com/",
            api_type="azure",
            api_version="2024-02-01",
        )
    else:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name, device="cuda"
        )
    collection = chroma_client.create_collection(
        db_name, embedding_function=ef, metadata={"hnsw:space": "cosine"}
    )

    id_list = [str(i) for i in range(len(chunks))]
    if len(chunks) > 400 and model_name == "openai":
        for i in range(0, len(chunks), 100):
            collection.add(documents=chunks[i : i + 100], ids=id_list[i : i + 100])
            time.sleep(10)
    else:
        collection.add(documents=chunks, ids=id_list)
    return collection


def store_pdf(pdf_file_path, db_collection_name, model="all-mpnet-base-v2"):
    pdf_text = extract_text_from_pdf(pdf_file_path)
    chunks = split_text(pdf_text)
    collection = store_chunks(chunks, db_collection_name, model)
    return collection


def prepare_collection(pdf_file_path, collection_name, model):
    if model == "bm25":
        chunks_or_collection = split_text(extract_text_from_pdf(pdf_file_path))
    elif model == "colbert":
        chunks_or_collection = colbert_index(pdf_file_path, collection_name)
    else:
        chunks_or_collection = store_pdf(pdf_file_path, collection_name, model)
    return chunks_or_collection


def reset_vdb(db_name):
    chroma_client = chromadb.Client()
    chroma_client.delete_collection(db_name)
