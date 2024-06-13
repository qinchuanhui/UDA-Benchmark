import json
import importlib
import sys

import re
import json
import io
import contextlib


def execute_and_capture_output(code):
    # Create an in-memory file-like object
    output_capture = io.StringIO()
    # Redirect standard output to the in-memory file-like object
    with contextlib.redirect_stdout(output_capture):
        exec(code)
    captured_output = output_capture.getvalue()
    output_capture.close()
    return captured_output


def code_gen_process(DATASET, data):
    tot_python_code = 0
    error_python_code = 0
    for item in data:
        response = item["response"]
        python_code = re.findall(r"```python\s*(.*?)\s*```", response, re.DOTALL)
        if len(python_code) == 0:
            continue
        python_code = python_code[0]
        python_code = python_code.replace("\\n", "\n")
        python_code = python_code.replace('\\"', '"')
        tot_python_code += 1
        try:
            output = execute_and_capture_output(python_code)
            item["response"] = f"The answer is: {output}"
        except Exception as e:
            print(f"Error in code execution: {e}")
            # print(f"Python code:\n {python_code}")
            error_python_code += 1
            item["response"] = "The answer is: 0"
    print(f"Total python code: {tot_python_code}")
    print(f"Error python code: {error_python_code}")
    return data


def call_f1_eval(dataset, answers, preds):
    eval_module = importlib.import_module(f"eval.utils.{dataset}_eval")
    res = eval_module.evaluate(answers, preds)
    print(res)


def call_finance_eval(dataset, data):
    if dataset == "tat":
        from eval.utils.tat_eval import TaTQAEmAndF1 as EvalClass

        em_and_f1 = EvalClass()
        for res in data:
            em_and_f1(res)
        global_em, global_f1, _, _ = em_and_f1.get_overall_metric()
        print("Exact-match accuracy {0:.2f}".format(global_em * 100))
        print("F1 score {0:.2f}".format(global_f1 * 100))
    elif dataset == "fin":
        from eval.utils.fin_eval import FinQAEm as EvalClass

        em = EvalClass()
        for res in data:
            em(res)
        global_em = em.get_overall_metric()
        print("Exact-match accuracy {0:.2f}".format(global_em * 100))


CODE_GEN = True


def eval_main(dataset_name, file_name):

    if dataset_name in ["paper_tab", "paper_text"]:
        DATASET = "paper"
    else:
        DATASET = dataset_name

    with open(file_name, "r") as f:
        data = f.readlines()
        data = [json.loads(x) for x in data][:]

    print("Total data: ", len(data))
    if DATASET in ["tat", "fin"]:
        if CODE_GEN:
            data = code_gen_process(DATASET, data)
        call_finance_eval(DATASET, data)
    elif DATASET in ["paper", "feta", "nq"]:
        answers = {}
        preds = {}
        for item in data:
            gts = item["answers"]
            if len(gts) == 0:
                continue
            q_uid = item["q_uid"]
            if item["response"] is None:
                continue
            pred = item["response"].split("The answer is: ")[-1]
            answers[q_uid] = gts
            preds[q_uid] = {"answer": pred}
        call_f1_eval(DATASET, answers, preds)


if __name__ == "__main__":

    pass
