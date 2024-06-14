"""
Official script for evaluating models built for the Qasper dataset. The script
outputs Answer F1 and Evidence F1 reported in the paper.
"""

from collections import Counter
import argparse
import string
import re
import json

from uda.eval.utils.basic_utils import *


# def evaluate(gold, predicted):
#     max_answer_f1s = []
#     max_answer_f1s_by_type = {
#         "extractive": [],
#         "abstractive": [],
#         "boolean": [],
#         "none": [],
#     }
#     num_missing_predictions = 0
#     for question_id in gold.keys():
#         if question_id not in predicted:
#             num_missing_predictions += 1
#             max_answer_f1s.append(0.0)
#             continue
#         answer_f1s_and_types = []
#         # may have multiple ground truth answers
#         pred = predicted[question_id]["answer"]
#         for reference in gold[question_id]:
#             if reference["type"] == "boolean":
#                 if pred.lower().startswith("yes"):
#                     pred = "Yes"
#                 elif pred.lower().startswith("no"):
#                     pred = "No"
#             answer_f1s_and_types.append(
#                 (
#                     token_f1_score(pred, reference["answer"]),
#                     reference["type"],
#                 )
#             )
#         max_answer_f1, answer_type = sorted(
#             answer_f1s_and_types, key=lambda x: x[0], reverse=True
#         )[0]
#         max_answer_f1s.append(max_answer_f1)
#         max_answer_f1s_by_type[answer_type].append(max_answer_f1)

#     mean = lambda x: sum(x) / len(x) if x else 0.0
#     return {
#         "Answer F1": mean(max_answer_f1s),
#         "Answer F1 by type": {
#             key: mean(value) for key, value in max_answer_f1s_by_type.items()
#         },
#         "Missing predictions": num_missing_predictions,
#     }


def paper_evaluate(gold, predicted):
    max_answer_f1s = []
    num_missing_predictions = 0
    for question_id, references in gold.items():
        if question_id not in predicted:
            num_missing_predictions += 1
            max_answer_f1s.append(0.0)
            continue
        pred = predicted[question_id]["answer"]
        for reference in gold[question_id]:
            if reference[0].lower() in ["yes", "no"]:
                if pred.lower().startswith("yes"):
                    pred = "Yes"
                elif pred.lower().startswith("no"):
                    pred = "No"
        answer_f1s_and_types = [
            token_f1_score(pred, references[0]),
            token_f1_score(pred, references[1]),
            token_f1_score(pred, references[2]),
        ]
        max_answer_f1 = sorted(answer_f1s_and_types, reverse=True)[0]
        max_answer_f1s.append(max_answer_f1)

    mean = lambda x: sum(x) / len(x) if x else 0.0
    return {
        "Answer F1": mean(max_answer_f1s),
        "Missing predictions": num_missing_predictions,
    }


if __name__ == "__main__":
    pass
