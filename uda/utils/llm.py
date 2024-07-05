from openai import AzureOpenAI
import subprocess
import os
import time
from uda.utils import access_config

PROMPT_DICT = {
    "fin": {
        "system": "You are a financial analyzer, given a section of a company's annual report, please answer the question according to the report context. Let's do this step by step. The final answer output should be in the format of 'The answer is: <answer>', and the <answer> must be simple and short (e.g. just an accurate numerical value or phrases). ",
        "user_1": "### Context: ...\n ## Question: What is the average price of the products?\n ### Response:",
        "assistant_1": "There are 8 products with a total price value of 1000, so the average value is 125.00 .\n The answer is: 125.00",
    },
    "tat": {
        "system": "You are a financial analyzer, given a section of a company's annual report, please answer the question according to the report context. Let's do this step by step. The final answer output should be in the format of 'The answer is: <answer>', and the <answer> must be simple and short (e.g. just an accurate numerical value or phrases). ",
        "user_1": "### Context: ...\n ## Question: What is the average price of the products?\n ### Response:",
        "assistant_1": "There are 8 products with a total price value of 1000, so the average value is 125.00 .\n The answer is: 125.00",
    },
    "paper": {
        "system": "You are a scientific researcher, given a section of an academic paper, please answer the question according to the context of the paper. The final answer output should be in the format of 'The answer is: <answer>', and the <answer> should be concise with no explanation.",
        "user_1": "### Context: ...\n ## Question: Which Indian languages do they experiment with\n ### Response:",
        "assistant_1": "The answer is: Hindi, English, Kannada, Telugu, Assamese, Bengali and Malayalam",
    },
    "feta": {
        "system": "Given a section of a document, plese answer the question according to the context. The final answer output should be in the format of 'The answer is: <answer>', and the <answer> should be a natural sentence.",
        "user_1": "### Context: ...\n ## Question: When and in what play did Platt appear at the Music Box Theatre?\n ### Response:",
        "assistant_1": "The answer is: In 2016 and 2017, Platt played in Dear Evan Hansen on Broadway at the Music Box Theatre.",
    },
    "nq": {
        "system": "Given a section of a document, plese answer the question according to the context. The final answer output should be in the format of 'The answer is: <answer>', and the <answer> should be a paragraph from the context or a summarized short phrase.",
        "user_1": "### Context: ...\n ## Question: When will tour de france teams be announced?\n ### Response:",
        "assistant_1": "The answer is: 6 January 2018",
    },
    "finance_code": {
        "system": "Given a section of a company's annual report and corresponding question, please generate the python codes to calculate the answer. You should firstly extract and list the relevant information from the context, and then write the arithmetical python codes in the following block format: ```python\n<python codes>\n```. If the answer does not require any calculation, you should directly write the answer in the format of 'The answer is: <answer>'",
        "user_1": "### Context: ...\n ## Question: What is the average price of the products?\n ### Response:",
        "assistant_1": "The price of product 1,2,3,4 is 700, and the price of product 5,6,7,8 is 900. The python code is ```python\nproducts = [700, 700, 700, 700, 900, 900, 900, 900],\n total_price = sum(products)\n average_price = total_price / len(products)\n print(average_price)\n```",
        "user_2": "### Context: ...\n ## Question: What is the net income in 2009?\n ### Response:",
        "assistant_2": "The answer is: 1000 million",
    },
    "finance_base": {
        "system": "You are a financial analyzer, given a section of a company's annual report, please answer the question according to the report context. The final answer output should be in the format of 'The answer is: <answer>', and the <answer> must be simple and short (e.g. just an accurate numerical value or phrases). ",
        "user_1": "### Context: ...\n ## Question: What is the average price of the products?\n ### Response:",
        "assistant_1": "The answer is: 125.00",
    },
}


def make_prompt(question, context, task_name, llm_type="gpt4"):
    if task_name in ["paper_tab", "paper_text"]:
        task_name = "paper"
    if task_name not in PROMPT_DICT.keys():
        raise ValueError(f"Invalid task name: {task_name}")
    prompt_template = PROMPT_DICT[task_name]
    user_content = f" ### Context: {context}\n ### Question: {question}\n ### Response:"
    if "mixtral" in llm_type or "Mixtral" in llm_type or "mistral" in llm_type:
        messages = [
            {
                "role": "user",
                "content": prompt_template["system"] + "\n" + prompt_template["user_1"],
            },
            {"role": "assistant", "content": prompt_template["assistant_1"]},
            {"role": "user", "content": user_content},
        ]
    else:
        messages = [
            {"role": "system", "content": prompt_template["system"]},
            {"role": "user", "content": prompt_template["user_1"]},
            {"role": "assistant", "content": prompt_template["assistant_1"]},
            {"role": "user", "content": user_content},
        ]
    return messages


def make_prompt_code(question, context, task_name, llm_type="gpt4"):
    if task_name not in ["fin"]:
        raise ValueError(f"Invalid task name: {task_name}")
    if "mixtral" in llm_type or "Mixtral" in llm_type or "mistral" in llm_type:
        raise ValueError(f"llm_gen_experiment is not supported in Mistral")
    prompt_template = PROMPT_DICT["finance_code"]
    user_content = f" ### Context: {context}\n ### Question: {question}\n ### Response:"
    messages = [
        {"role": "system", "content": prompt_template["system"]},
        {"role": "user", "content": prompt_template["user_1"]},
        {"role": "assistant", "content": prompt_template["assistant_1"]},
        {"role": "user", "content": prompt_template["user_2"]},
        {"role": "assistant", "content": prompt_template["assistant_2"]},
        {"role": "user", "content": user_content},
    ]
    return messages


def make_prompt_basic(question, context, task_name, llm_type="gpt4"):
    if task_name not in ["fin"]:
        raise ValueError(f"Invalid task name: {task_name}")
    if "mixtral" in llm_type or "Mixtral" in llm_type or "mistral" in llm_type:
        raise ValueError(f"llm_gen_experiment is not supported in Mistral")
    prompt_template = PROMPT_DICT["finance_base"]
    user_content = f" ### Context: {context}\n ### Question: {question}\n ### Response:"
    messages = [
        {"role": "system", "content": prompt_template["system"]},
        {"role": "user", "content": prompt_template["user_1"]},
        {"role": "assistant", "content": prompt_template["assistant_1"]},
        {"role": "user", "content": user_content},
    ]
    return messages


def call_gpt(messages):
    client = AzureOpenAI(
        api_key=access_config.GPT_API_KEY,
        api_version="2024-04-01-preview",
        azure_endpoint=access_config.GPT_ENDPOINT,
    )

    try:
        response = client.chat.completions.create(
            model=access_config.GPT_MODEL,
            messages=messages,
            temperature=0.1,
        )
        res = response.choices[0].message.content
        return res
    except Exception as e:
        print(e)


if __name__ == "__main__":
    pass
