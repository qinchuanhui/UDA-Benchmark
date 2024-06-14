import PyPDF2
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import logging
import time
from ragatouille import RAGPretrainedModel
import string
import json
import torch


from rank_bm25 import BM25Okapi


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
        device_info = "cuda" if torch.cuda.is_available() else "cpu"
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name, device=device_info
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


def get_contexts(chunks_or_collection, question, model, top_k=30):
    if model == "bm25":
        contexts = BM25_retrieval(chunks_or_collection, question, top_k)
    elif model == "colbert":
        contexts = colbert_retrieve(chunks_or_collection, question, top_k)
    else:
        query_res = chunks_or_collection.query(query_texts=[question], n_results=top_k)
        contexts = query_res["documents"][0]
    return contexts


def reset_vdb(db_name):
    chroma_client = chromadb.Client()
    chroma_client.delete_collection(db_name)


def normalize_answer(s):
    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def BM25_retrieval(chunks, query, top_n):
    word_level_corpus = []
    for chunk in chunks:
        normalized_word = normalize_answer(chunk)
        word_level_corpus.append(normalized_word.split(" "))
    bm25 = BM25Okapi(word_level_corpus)
    word_level_query = normalize_answer(query).split(" ")
    scores = bm25.get_scores(word_level_query)
    # get the top 40 index
    top_index = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
        :top_n
    ]
    res_chunks = [chunks[i] for i in top_index]
    return res_chunks


def word_lcs(gold, pred):  # A is the gold standard, B is the predicted output
    A = normalize_answer(gold).split()
    B = normalize_answer(pred).split()
    dp = [[0] * (len(B) + 1) for _ in range(len(A) + 1)]
    # Fill the matrix in a bottom-up manner
    for i in range(1, len(A) + 1):
        for j in range(1, len(B) + 1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    # backtrack and find the word-level LCS
    lcs_words = []
    i, j = len(A), len(B)
    while i > 0 and j > 0:
        if A[i - 1] == B[j - 1]:
            lcs_words.insert(0, A[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    if len(A) == 0:
        precision = 0.5
    else:
        precision = len(lcs_words) / len(A)
    return precision, lcs_words


def char_lcs(gold, pred):  # A is the gold standard, B is the predicted output
    A = normalize_answer(gold)
    B = normalize_answer(pred)
    dp = [[0] * (len(B) + 1) for _ in range(len(A) + 1)]
    # Fill the matrix in a bottom-up manner
    for i in range(1, len(A) + 1):
        for j in range(1, len(B) + 1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    # backtrack and find the word-level LCS
    lcs_words = []
    i, j = len(A), len(B)
    while i > 0 and j > 0:
        if A[i - 1] == B[j - 1]:
            lcs_words.insert(0, A[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    if len(A) == 0:
        precision = 0.5
    else:
        precision = len(lcs_words) / len(A)
    return precision, lcs_words
