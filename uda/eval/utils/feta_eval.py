from uda.eval.utils.basic_utils import *


def feta_evaluate(gold, predicted):
    max_answer_f1s = []
    num_missing_predictions = 0
    for question_id in gold.keys():
        if question_id not in predicted:
            num_missing_predictions += 1
            max_answer_f1s.append(0.0)
            continue
        # in feta there is only one reference  gold answer
        answer_f1s_and_types = [
            token_f1_score(predicted[question_id]["answer"], gold[question_id])
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
