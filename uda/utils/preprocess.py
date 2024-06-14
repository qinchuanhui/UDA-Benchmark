import os
import json
import random
from ast import literal_eval
import pandas as pd


meta_data = {
    "fin": {
        "example_pdf_dir": "dataset/src_doc_files_example/fin_docs",
        "pdf_dir": "dataset/src_doc_files/fin_docs",
        "bench_json_file": "dataset/extended_qa_info_bench/bench_fin_qa.json",
    },
    "tat": {
        "example_pdf_dir": "dataset/src_doc_files_example/tat_docs",
        "pdf_dir": "dataset/src_doc_files/tat_docs",
        "bench_json_file": "dataset/extended_qa_info_bench/bench_tat_qa.json",
    },
    "paper_tab": {
        "example_pdf_dir": "dataset/src_doc_files_example/paper_docs",
        "pdf_dir": "dataset/src_doc_files/paper_docs",
        "bench_json_file": "dataset/extended_qa_info_bench/bench_paper_tab_qa.json",
    },
    "paper_text": {
        "example_pdf_dir": "dataset/src_doc_files_example/paper_docs",
        "pdf_dir": "dataset/src_doc_files/paper_docs",
        "bench_json_file": "dataset/extended_qa_info_bench/bench_paper_text_qa.json",
    },
    "feta": {
        "example_pdf_dir": "dataset/src_doc_files_example/wiki_docs/pdfs",
        "pdf_dir": "dataset/src_doc_files/wiki_docs/pdfs",
        "bench_json_file": "dataset/extended_qa_info_bench/bench_feta_qa.json",
    },
    "nq": {
        "example_pdf_dir": "dataset/src_doc_files_example/wiki_docs/pdfs",
        "pdf_dir": "dataset/src_doc_files/wiki_docs/pdfs",
        "bench_json_file": "dataset/extended_qa_info_bench/bench_nq_qa.json",
    },
}


def get_example_pdf_path(dataset_name, doc_name):
    pdf_dir = meta_data[dataset_name]["example_pdf_dir"]
    pdf_file_path = os.path.join(pdf_dir, doc_name + ".pdf")
    if not os.path.exists(pdf_file_path):
        print(f"PDF file not found at: {pdf_file_path}")
        return None
    return pdf_file_path


def get_pdf_path(dataset_name, doc_name):
    pdf_dir = meta_data[dataset_name]["pdf_dir"]
    pdf_file_path = os.path.join(pdf_dir, doc_name + ".pdf")
    if not os.path.exists(pdf_file_path):
        print(f"PDF file not found at: {pdf_file_path}")
        return None
    return pdf_file_path


def _process_answer_enrtry(dataset_name, entry):
    if dataset_name == "fin":
        answers = {"str_answer": entry["answer_1"], "exe_answer": entry["answer_2"]}
    elif dataset_name == "tat":
        raw_answer = entry["answer"]
        # in hf parquet, the answer is already a sequence
        answer = (
            literal_eval(raw_answer)
            if isinstance(raw_answer, str)
            else list(raw_answer)
        )
        answers = {
            "answer": answer,
            "answer_type": entry["answer_type"],
            "scale": entry["answer_scale"],
        }
    elif dataset_name in ["paper_tab", "paper_text"]:
        answers = [entry["answer_1"], entry["answer_2"], entry["answer_3"]]
    elif dataset_name == "feta":
        answers = entry["answer"]
    elif dataset_name == "nq":
        answers = {
            "short_answer": entry["short_answer"],
            "long_answer": entry["long_answer"],
        }
    return answers


def qa_df_to_dict(dataset_name, df):
    qas_dict = {}
    for item in df.iterrows():
        doc_name = str(item[1]["doc_name"])
        answers = _process_answer_enrtry(dataset_name, item[1])
        qa_dict = {
            "question": item[1]["question"],
            "answers": answers,
            "q_uid": str(item[1]["q_uid"]),
        }
        if doc_name not in qas_dict:
            qas_dict[doc_name] = []
        qas_dict[doc_name].append(qa_dict)
    return qas_dict


# do sampling and check if the qa is already done
def process_qa_list(dataset, qa_list, output_file_name):
    # if dataset == "fin":
    #     qa_list = qa_list[:6]
    # elif dataset == "fin_parse":
    #     qa_list = qa_list[:5]
    # elif dataset == "nq":
    #     qa_list = qa_list[:3]
    # elif dataset == "tat" and len(qa_list) > 3:
    #     # cause a tat-doc has a lot of questions
    #     qa_list = random.sample(qa_list, 3)
    # # use cache
    if os.path.exists(output_file_name):
        with open(output_file_name, "r") as f:
            lines = f.readlines()
            lines = [json.loads(line) for line in lines]
            done_list = [line["q_uid"] for line in lines]
        qa_list = [qa_item for qa_item in qa_list if qa_item["q_uid"] not in done_list]
    return qa_list
