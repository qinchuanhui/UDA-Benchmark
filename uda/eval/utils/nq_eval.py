"""
Official script for evaluating models built for the Qasper dataset. The script
outputs Answer F1 and Evidence F1 reported in the paper.
"""

from uda.eval.utils.basic_utils import *


def nq_evaluate(gold, predicted):
    max_answer_f1s = []
    num_missing_predictions = 0
    for question_id, references in gold.items():
        if question_id not in predicted:
            num_missing_predictions += 1
            max_answer_f1s.append(0.0)
            continue
        pred = predicted[question_id]["answer"]
        answer_f1s_and_types = [
            token_f1_score(pred, references["long_answer"]),
            token_f1_score(pred, references["short_answer"]),
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
