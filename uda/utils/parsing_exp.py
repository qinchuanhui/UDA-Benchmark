import os
import PyPDF2
import json
import sys
from openai import AzureOpenAI
from unstructured.partition.pdf import partition_pdf
from uda.utils.retrieve import word_lcs
from uda.utils.retrieve import extract_text_from_pdf, split_text, log_duration
from uda.utils import llm
from uda.utils import access_config
import numpy as np
import re

PARSING_TMP_DIR = "./experiment/parsing/parsing_tmp"


def fin_well_parsed(q_item):
    pre_text = "\n".join(
        [text for text in q_item["context"]["pre_text"] if text != "."]
    )
    post_text = "\n".join(
        [text for text in q_item["context"]["post_text"] if text != "."]
    )
    # turn table into markdown string
    table_context = " |\n| ".join(
        [" | ".join(row) for row in q_item["context"]["table_ori"]]
    )
    context = pre_text + "\n" + "Table:\n" + table_context + "\n" + post_text
    return context


def fin_raw_extract(q_item, pdf_file_path):
    q_uid = q_item["q_uid"]
    # just use the target_page not the whole pdf
    target_page = int(q_uid.split("page_")[-1].split(".pdf")[0]) - 1
    with open(pdf_file_path, "rb") as file:
        # read the pdf file of target page
        reader = PyPDF2.PdfReader(file)
        pdf_page = reader.pages[target_page]
        context = pdf_page.extract_text()
        return context


def fin_cv(q_item, pdf_file_path, tmp_file_path):
    q_uid = q_item["q_uid"]
    # just use the target_page not the whole pdf
    target_page = int(q_uid.split("page_")[-1].split(".pdf")[0]) - 1
    reader = PyPDF2.PdfReader(open(pdf_file_path, "rb"))
    writer = PyPDF2.PdfWriter()
    writer.add_page(reader.pages[target_page])
    with open(tmp_file_path, "wb") as f:
        writer.write(f)
    # get the whole page content, including table in html format
    elements = partition_pdf(
        filename=tmp_file_path,
        strategy="hi_res",
        extract_images_in_pdf=False,
        url=None,
        infer_table_structure=True,
    )
    context = ""
    for element in elements:
        if "Table" in str(type(element)):
            text = element.metadata.text_as_html
        else:
            text = element.text
        context += text + "\n"
    return context


def fetch_related_paper_page(pdf_file_path, evidence, tmp_file_path):
    with open(pdf_file_path, "rb") as file:
        # Create a PDF file reader object
        reader = PyPDF2.PdfReader(file, strict=False)
        page_scores = []
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text()
            score, _ = word_lcs(evidence, text)
            page_scores.append(score)
        target_page_idx = int(np.argmax(page_scores))
        writer = PyPDF2.PdfWriter()
        writer.add_page(reader.pages[target_page_idx])
        with open(tmp_file_path, "wb") as f:
            writer.write(f)
        return target_page_idx


def paper_raw_extract_table(pdf_file_path, query):
    text = extract_text_from_pdf(pdf_path=pdf_file_path)
    split_char_window = 1500
    chunks = split_text(text, split_char_window, 300)
    chunk_scores = []
    for chunk in chunks:
        score, _ = word_lcs(query, chunk)
        chunk_scores.append(score)
    target_chunk_idx = int(np.argmax(chunk_scores))
    table_context = chunks[target_chunk_idx]
    return table_context


def paper_cv_table(tmp_file_path, query):
    try:
        elements = partition_pdf(
            filename=tmp_file_path,
            strategy="hi_res",
            extract_images_in_pdf=False,
            url=None,
            infer_table_structure=True,
        )
        chunks = []
        for idx, element in enumerate(elements):
            if "Table" in str(type(element)):
                table_text = element.metadata.text_as_html
                context_chunk = (
                    elements[max(idx - 1, 0)].text
                    + table_text
                    + elements[min(idx + 1, len(elements) - 1)].text
                )
                chunks.append(context_chunk)
        if len(chunks) == 0:
            print("No table detected, cv fall to raw_extract")
            return None
    except Exception as e:
        print(f"====Error in cv chunk extraction: {e},  cv fall to raw_extract=====")
        return None
    # there may be multiple tables in one page, we need to find the most related one
    chunk_scores = []
    for chunk in chunks:
        score, _ = word_lcs(query, chunk)
        chunk_scores.append(score)
    target_chunk_idx = int(np.argmax(chunk_scores))
    table_context = chunks[target_chunk_idx]
    return table_context


def attach_llm_to_cv(context):
    def html_to_markdown(table_text):
        messages = [
            {
                "role": "system",
                "content": "Please transform the following table of HTML format into Markdown format, just output the markdown content in ``` block, without any explanation",
            },
            {"role": "user", "content": table_text},
        ]
        response = llm.call_gpt(messages)
        # get the content in the code block
        table_md = response.split("```")[1]
        return table_md

    # find the html table in the text and transform to markdown
    table_pattern = re.compile(r"<table.*?>.*?</table>", re.DOTALL)
    tables = table_pattern.findall(context)
    for table in tables:
        table_md = html_to_markdown(table)
        context = context.replace(table, table_md)
    return context


def get_omni_table(pdf_file_path, output_path):
    def get_img_binary(pdf_file_path, output_path):
        from pdf2image import convert_from_path
        import base64

        image = convert_from_path(pdf_file_path, dpi=300, first_page=1, last_page=1)[0]
        image.save(output_path, "PNG")
        with open(output_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    base64_image = get_img_binary(pdf_file_path, output_path)
    messages = [
        {
            "role": "system",
            "content": "Please extract all the tables from this image. Just output the table content and table title without any other information. The table should be the markdown format within the ``` block.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            ],
        },
    ]

    client = AzureOpenAI(
        api_key=access_config.GPT_OMNI_API_KEY,
        api_version="2024-04-01-preview",
        azure_endpoint=access_config.GPT_OMNI_ENDPOINT,
    )

    response = client.chat.completions.create(
        model=access_config.GPT_OMNI_MODEL,
        messages=messages,
        temperature=0.1,
    )
    res = response.choices[0].message.content
    return res


def get_fin_omni(q_item, pdf_file_path, tmp_file_path):
    os.makedirs(PARSING_TMP_DIR, exist_ok=True)
    output_img_path = os.path.join(PARSING_TMP_DIR, "fin_image.png")

    q_uid = q_item["q_uid"]
    # just use the target_page not the whole pdf
    target_page = int(q_uid.split("page_")[-1].split(".pdf")[0]) - 1
    reader = PyPDF2.PdfReader(open(pdf_file_path, "rb"))
    writer = PyPDF2.PdfWriter()
    writer.add_page(reader.pages[target_page])
    with open(tmp_file_path, "wb") as f:
        writer.write(f)

    table_context = get_omni_table(tmp_file_path, output_img_path)
    pre_text = "\n".join(
        [text for text in q_item["context"]["pre_text"] if text != "."]
    )
    post_text = "\n".join(
        [text for text in q_item["context"]["post_text"] if text != "."]
    )
    context = pre_text + "\n" + "Table:\n" + table_context + "\n" + post_text
    return context


def get_fin_context(q_item, strategy, pdf_file_path):
    os.makedirs(PARSING_TMP_DIR, exist_ok=True)
    tmp_pdf_file_path = os.path.join(PARSING_TMP_DIR, "fin_tmp.pdf")
    if strategy == "well_parsed":
        return fin_well_parsed(q_item)
    elif strategy == "raw_extract":
        return fin_raw_extract(q_item, pdf_file_path)
    elif strategy in ["cv", "cv_llm"]:
        cv_context = fin_cv(q_item, pdf_file_path, tmp_pdf_file_path)
        if strategy == "cv":
            return cv_context
        elif strategy == "cv_llm":
            return attach_llm_to_cv(cv_context)
    elif strategy == "omni":
        return get_fin_omni(q_item, pdf_file_path, tmp_pdf_file_path)

    else:
        print(f"Error: Invalid strategy {strategy}")
        return None


def get_paper_omni_table(tmp_file_path, table_name):
    os.makedirs(PARSING_TMP_DIR, exist_ok=True)
    output_img_path = os.path.join(PARSING_TMP_DIR, "paper_image.png")
    table_context = get_omni_table(tmp_file_path, output_img_path)
    # print("====OMNI table context:=====", table_context)
    pattern = r"```(.*?)```"
    table_contexts = re.findall(pattern, table_context, re.DOTALL)
    table_contexts = [table.strip() for table in table_contexts]
    # get the most related table
    table_scores = []
    for table in table_contexts:
        score, _ = word_lcs(table_name, table)
        table_scores.append(score)
    target_table_idx = int(np.argmax(table_scores))
    table_context = table_contexts[target_table_idx]
    return table_context


def get_paper_context(q_item, strategy, pdf_file_path):
    os.makedirs(PARSING_TMP_DIR, exist_ok=True)
    tmp_file_path = os.path.join(PARSING_TMP_DIR, "paper_tmp.pdf")
    table_names = []
    contexts = []
    text_buffer = []
    table_cache = {}
    for ev_item in q_item["evidence"]:
        contain_table = False
        for ev in ev_item["highlighted_evidence"]:
            if ev.startswith("FLOAT SELECTED: Table"):
                table_name = ev.split("FLOAT SELECTED: ")[-1]
                table_names.append(table_name)
                contain_table = True
            else:
                text_buffer.append(ev)
        # if this answer use no tables, the text_buffer shold not be added to the final context
        if contain_table:
            contexts.extend(text_buffer)

    for table_name in table_names:
        if table_name in table_cache.keys():
            # use the cache
            table_context = table_cache[table_name]
        else:
            page_idx = fetch_related_paper_page(
                pdf_file_path, table_name, tmp_file_path
            )
            if strategy == "raw_extract":
                table_context = paper_raw_extract_table(pdf_file_path, table_name)
            elif strategy == "cv":
                table_context = paper_cv_table(tmp_file_path, table_name)
            elif strategy == "cv_llm":
                table_cv = paper_cv_table(tmp_file_path, table_name)
                table_context = attach_llm_to_cv(table_cv)
            elif strategy == "omni":
                table_context = get_paper_omni_table(tmp_file_path, table_name) + "\n"
            table_cache[table_name] = table_context

        contexts.append(table_context)
        final_context = "\n".join(contexts)
        # print("====Final context:=====\n", final_context)
    return final_context
