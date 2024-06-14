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
    from uda.eval.utils.paper_eval import paper_evaluate
    from uda.eval.utils.feta_eval import feta_evaluate
    from uda.eval.utils.nq_eval import nq_evaluate

    if dataset == "paper":
        res = paper_evaluate(answers, preds)
    elif dataset == "feta":
        res = feta_evaluate(answers, preds)
    elif dataset == "nq":
        res = nq_evaluate(answers, preds)

    print(res)


def call_finance_eval(dataset, data):
    if dataset == "tat":
        from uda.eval.utils.tat_eval import TaTQAEmAndF1 as EvalClass

        em_and_f1 = EvalClass()
        for res in data:
            em_and_f1(res)
        global_em, global_f1, _, _ = em_and_f1.get_overall_metric()
        print("Numerical F1 score: {0:.2f}".format(global_f1 * 100))
    elif dataset == "fin":
        from uda.eval.utils.fin_eval import FinQAEm as EvalClass

        em = EvalClass()
        for res in data:
            em(res)
        global_em = em.get_overall_metric()
        print("Exact-match accuracy: {0:.2f}".format(global_em * 100))


def eval_main(dataset_name, data_list, CODE_GEN=False):

    if dataset_name in ["paper_tab", "paper_text"]:
        DATASET = "paper"
    else:
        DATASET = dataset_name

    # print("Total data: ", len(data_list))
    if DATASET in ["tat", "fin"]:
        if CODE_GEN:
            data_list = code_gen_process(DATASET, data_list)
        call_finance_eval(DATASET, data_list)
    elif DATASET in ["paper", "feta", "nq"]:
        answers = {}
        preds = {}
        for item in data_list:
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


def eval_from_file(dataset_name, file_path, CODE_GEN=False):
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
        eval_main(dataset_name, data, CODE_GEN)


if __name__ == "__main__":

    pass
