from uda.utils.retrieve import *


def exact_containing(A, B):  # A is the gold standard, B is the predicted output
    A = normalize_answer(A)
    B = normalize_answer(B)
    if A in B:
        return 1
    return 0


def table_to_text(table):
    table_text = ""
    for row in table:
        for cell in row:
            table_text += cell + " "
        table_text += "\n"
    return table_text


# @log_duration
def get_fin_match_score(gt_file_name, my_key, my_qa_id, prediction):
    with open(gt_file_name, "r") as f:
        data = json.load(f)
    qa_found = False
    for qa in data[my_key]:
        qa_id = qa["q_uid"]
        if qa_id != my_qa_id:
            continue
        score_list = []
        evidence = qa["evidence"]
        table_caculated = None
        # Each qa has multiple evidence-items, and we calculate the average precision
        # Since there's only one complete table each query, the table_precision is all the same
        for evi_k in evidence.keys():
            if "text" in evi_k:
                evi_text = evidence[evi_k]
                tmp_precision, _ = word_lcs(evi_text, prediction)
            elif "table" in evi_k:
                if table_caculated is not None:
                    tmp_precision = table_caculated
                else:
                    table_text_1 = table_to_text(qa["context"]["table"])
                    table_text_2 = table_to_text(qa["context"]["table_ori"])
                    res_1, _ = word_lcs(table_text_1, prediction)
                    res_2, _ = word_lcs(table_text_2, prediction)
                    table_caculated = max(res_1, res_2)
                    tmp_precision = table_caculated
            else:
                print(f"{my_qa_id} evidence type not found")
                continue
            score_list.append(tmp_precision)
        avg_score = sum(score_list) / len(score_list)
        # print(f"{my_qa_id} average precision:", avg_score)
        qa_found = True  # if the qa_id is found, then break
        break
    if not qa_found:
        print(f"qa_id {my_qa_id} not found")
    return avg_score


def get_paper_match_score(gt_file_name, my_key, my_qa_id, prediction):
    with open(gt_file_name, "r") as f:
        data = json.load(f)
    qa_found = False
    for qa in data[my_key]:
        qa_id = qa["q_uid"]
        if qa_id != my_qa_id:
            continue
        score_list = []
        # Each qa has multiple evidence, and we calculate the average precision
        for ev in qa["evidence"]:
            tmp_score_list = []
            high_evs = ev["highlighted_evidence"]
            if len(high_evs) == 0:
                continue
            for high_ev in high_evs:
                if high_ev.startswith("FLOAT SELECTED: Table"):
                    table_title = high_ev.split("FLOAT SELECTED: ")[-1]
                    # tmp_precision = exact_containing(table_title, prediction)
                    tmp_precision, _ = word_lcs(table_title, prediction)
                else:
                    tmp_precision, _ = word_lcs(high_ev, prediction)
                tmp_score_list.append(tmp_precision)
            tmp_avg_score = sum(tmp_score_list) / len(tmp_score_list)
            score_list.append(tmp_avg_score)
        if len(score_list) == 0:
            # print(f"No human-anotated evidence for file {my_key} and qa_id {my_qa_id}")
            avg_score = None
            # the division by 0 error should be  processed in the calling function
        else:
            avg_score = sum(score_list) / len(score_list)
        # print(f"{my_qa_id} average precision:", avg_score)
        qa_found = True
        break
    if not qa_found:
        print(f"qa_id {my_qa_id} not found")
        avg_score = None
    return avg_score


def get_feta_match_score(gt_file_name, my_key, my_qa_id, prediction):
    with open(gt_file_name, "r") as f:
        data = json.load(f)
    qa_found = False
    for qa in data[my_key]:
        q_uid = qa["q_uid"]
        if q_uid != my_qa_id:
            continue
        table = qa["evidence"]["table_array"]
        table_text = table_to_text(table)
        # print("====Table text====:\n", table_text)
        # print("====Prediction====:\n", prediction)
        score, _ = char_lcs(table_text, prediction)
        qa_found = True
        break
    if not qa_found:
        print(f"qa_id {my_qa_id} not found")
    return score


def get_nq_match_score(gt_file_name, my_key, my_qa_id, prediction):
    with open(gt_file_name, "r") as f:
        data = json.load(f)
    qa_found = False
    for qa in data[my_key]:
        qa_id = qa["q_uid"]
        if qa_id != my_qa_id:
            continue
        evidence_text = qa["answers"]["long_answer"]
        score, _ = word_lcs(evidence_text, prediction)
        qa_found = True
        break
    if not qa_found:
        print(f"qa_id {my_qa_id} not found")
    return score


def log_score(contexts, doc_name, q_uid, dataset_name, res_file_name, ret_gt_file_name):
    if dataset_name == "fin":
        score_function = get_fin_match_score
    elif dataset_name == "paper_tab":
        score_function = get_paper_match_score
    elif dataset_name == "paper_text":
        score_function = get_paper_match_score
    elif dataset_name == "feta":
        score_function = get_feta_match_score
    elif dataset_name == "nq":
        score_function = get_nq_match_score
    else:
        print("dataset_name not found")
        return

    res = {"doc_name": doc_name, "q_uid": q_uid}
    log_point = [1, 5, 10, 20, 30]
    for num in log_point:
        context_join = "\n".join(contexts[:num])
        num_score = score_function(ret_gt_file_name, doc_name, q_uid, context_join)
        res[f"Top-{num}"] = num_score
        if num_score is None:
            print(f"No human-anotated evidence for file {doc_name} and qa_id {q_uid}")
            return
    print("Retrieval-Match-Scores", res)
    with open(res_file_name, "a") as f:
        f.write(json.dumps(res) + "\n")
