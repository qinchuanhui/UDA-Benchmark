from typing import Set, Tuple, Union
from uda.eval.utils.finance_utils import *
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import re

IGNORE_SCALE = True


def _answer_to_bags(
    answer: Union[str, List[str], Tuple[str, ...]]
) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = normalize_finance_answer(raw_span, IGNORE_SCALE)
        for token in normalized_span.split():
            if is_number(token):
                normalized_span = normalized_span.replace(
                    token, str(to_number(token, IGNORE_SCALE))
                )
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def extract_gold_answers(response):
    try:
        gold_answers = [
            str(response["answers"]["exe_ans"]),
            response["answers"]["str_ans"],
        ]
    except Exception as e:
        gold_answers = [
            str(response["answers"]["exe_answer"]),
            response["answers"]["str_answer"],
        ]
    return gold_answers


def get_answer(answers: list, scale: str = ""):
    sorted_ans = sorted(answers)
    ans_temp = []
    for ans in sorted_ans:
        ans_str = str(ans)
        if is_number(ans_str):
            ans_num = to_number(ans_str, IGNORE_SCALE)
            if ans_num is None:
                if LOG_DETAIL:
                    print(f"Warning: No number found in {ans}, use the num of 0.")
                ans_num = 0
            else:
                if (
                    "%" in ans_str
                ):  #  has been handled the answer itself is a percentage
                    ans_num = ans_num
                elif type(ans_num) == int:
                    ans_ans_num = ans_num * scale_to_num(scale)
                else:
                    ans_num = round(ans_num, 4) * scale_to_num(scale)
        else:
            ans_num = normalize_finance_answer(ans_str, ignore_scale=IGNORE_SCALE)
        ans_temp.append(ans_num)
    return ans_temp


def get_metrics(
    predicted: Union[str, List[str], Tuple[str, ...]],
    gold: Union[str, List[str], Tuple[str, ...]],
) -> Tuple[float, float]:

    def numbers_match(num1, num2, threshold=0.01):
        # give some threshold for float comparison due to the rounding error
        return abs(num1 - num2) / abs((num1 + num2) / 2 + 1e-9) < threshold

    if type(gold) == float or type(gold) == int:
        if type(predicted) == str:
            try:
                predicted = float(predicted)
            except Exception as e:
                pattern = r"[-+]?\d*\.?\d+"
                matches = re.findall(pattern, predicted)
                if matches:
                    last_match = matches[-1]
                    predicted = float(last_match)
                else:
                    if LOG_DETAIL:
                        print(
                            f"Warning: predicted answer '{predicted}' is not number: {e} "
                        )
                    return 0
        exact_match = 1.0 if numbers_match(predicted, gold) else 0.0

    else:  # yes or no is in string format
        predicted_bags = _answer_to_bags(str(predicted))
        gold_bags = _answer_to_bags(str(gold))

        if set(predicted_bags[0]) == set(gold_bags[0]) and len(
            predicted_bags[0]
        ) == len(gold_bags[0]):
            exact_match = 1.0
        else:
            exact_match = 0.0

    if LOG_DETAIL:
        print("EM:", exact_match, "pred: ", predicted, "gt: ", gold)

    return exact_match


def metric_max_over_ground_truths(metric_fn, predictions, ground_truths):
    scores_for_ground_truths = []
    for pred in predictions:
        for ground_truth in ground_truths:
            score = metric_fn(pred, ground_truth)
            scores_for_ground_truths.append(score)
    if len(scores_for_ground_truths) == 0:
        return 0
    return max(scores_for_ground_truths)


class FinQAEm(object):
    """Based on TATQAEmAndF1"""

    def __init__(self) -> None:
        self._total_em = 0.0
        self._count = 0
        self._details = []

    def __call__(self, response: dict):  # type: ignore

        try:
            prediction = response["response"].split("The answer is:")[-1]
        except Exception as e:
            print("Error in response split: ", response, e)
            prediction = response["response"]
            if prediction is None:
                prediction = ""

        CHARS_TO_REMOVE = "\"'.\n,"
        prediction = prediction.rstrip(CHARS_TO_REMOVE)

        if not prediction:
            exact_match = 0
            f1_score = 0
        else:
            gold_answers = extract_gold_answers(response)
            gold_scale = ""
            pred_scale = ""  # no need to handle scale in prediction
            if not gold_answers:
                exact_match = 0
                f1_score = 0
            else:
                ground_truth_num = get_answer(gold_answers, gold_scale)
                prediction = (
                    prediction if isinstance(prediction, list) else [prediction]
                )
                prediction_num = get_answer(prediction, pred_scale)
                # if LOG_DETAIL:
                #     print(
                #         "gt: ",
                #         gold_answers,
                #         " pred: ",
                #         prediction,
                #         " gt_num: ",
                #         ground_truth_num,
                #         " pred_num: ",
                #         prediction_num,
                #     )
                exact_match = metric_max_over_ground_truths(
                    get_metrics, prediction_num, ground_truth_num
                )
        self._total_em += exact_match
        self._count += 1
        it = {"gt": gold_answers, "pred": prediction, "em": exact_match}
        if LOG_DETAIL:
            print("Final EM: ", exact_match)
        self._details.append(it)

    def get_overall_metric(
        self, reset: bool = False
    ) -> Tuple[float, float, float, float]:
        """
        Returns
        -------
        Average exact match
        """
        exact_match = self._total_em / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return exact_match

    def get_raw(self):
        return self._details

    def reset(self):
        self._total_em = 0.0
        self._count = 0
        self._details = []

    def __str__(self):
        return f"em={self._total_em}, count={self._count})"
