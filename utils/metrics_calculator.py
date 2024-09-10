import os
import re
import json
import argparse
import numpy as np


def judge_in(file_path):
    em_acc = []
    with open(file_path, 'r', encoding='utf-8') as fr:
        results = json.load(fr)

        for result in results:
            model_output = result['model_output']
            truth_answer = result['truth_answer']

            info = model_output.split('\n')
            answer_seg = ''
            for seg in info:
                if 'Answer:' in seg:
                    answer_seg = seg
                    break

            has_ture_answer = False
            for ans in truth_answer:
                if ans.lower() in answer_seg.lower():
                    em_acc.append(1)
                    has_ture_answer = True
                    break
            if not has_ture_answer:
                em_acc.append(0)

    acc = sum(em_acc) / len(em_acc)
    print('acc: {}'.format(acc))


def extract_answer(response):
    # pattern = re.compile(r'[Tt]he answer.*?is(.*?)[#.]')
    pattern = re.compile((r"\[(.*?)\]"))
    response = response.replace('\n', '').replace('\"', '')
    predictions = re.findall(pattern, response)

    return predictions


def extract_answer_mc(response):
    # pattern = re.compile(r'[Tt]he answer.*?is(.*?)[#.]')
    pattern = re.compile((r"\((.*?)\)"))
    response = response.replace('\n', '').replace('\"', '')
    predictions = re.findall(pattern, response)

    return predictions


def extract_choice(response):
    pattern = r'\[(\w)\]'
    matches = re.findall(pattern, response)

    answer = ""

    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            answer = solution
            break

    return answer


def multi_agent_calculate(file_path, dataset_name):
    
    eval_file_path = file_path.replace("/predictions/", "/evaluations/")[:-5] + "_eval.json"

    eval_results = {}
    if os.path.exists(eval_file_path):
        with open(eval_file_path, "r", encoding="utf-8") as fr_eval:
            for result in json.load(fr_eval):
                eval_results[result["question"]] = result

    em_acc = []
    early_stop_acc = []
    avg_round, all_round = 0.0, 0.0
    round_keys = ["Round_6", "Round_5", "Round_4", "Round_3", "Round_2", "Round_1", "Round_0"]
    with open(file_path, 'r', encoding='utf-8') as fr:
        results = json.load(fr)
        not_early_stop = 0
        for idx, result in enumerate(results):
            question = result["question"]
            early_stop_flag = True
            for key in round_keys:
                if key in result.keys():
                    all_round += int(key[-1]) + 1
                    if "[Yes]" not in result[key]["Judge"]:
                        not_early_stop += 1
                        early_stop_flag = False
                    break
            if early_stop_flag:
                if len(eval_results) > 0:
                    eval_result = eval_results[question]["eval_gpt4"]
                    if "[True]" in eval_result:
                        early_stop_acc.append(1.0)
                    else:
                        early_stop_acc.append(0.0)


            if dataset_name == 'mmlu':
                truth_answer = result['answer'][1:2].lower()
            else:
                truth_answer = [x.lower().strip("'") for x in result['answer']]

            response = result["Summary"]
            if dataset_name == 'mmlu':
                agent_answers = extract_choice(response)
            else:
                agent_answers = extract_answer(response)

            agent_answers = [x.lower() for x in agent_answers]
            flag = False

            if len(agent_answers) > 0 and agent_answers[0] in truth_answer:
                flag = True
            if flag:
                em_acc.append(1.0)
            else:
                em_acc.append(0.0)
    if len(early_stop_acc) == 0:
        print('avg round: {}, acc: {} {}, not early stop: {} from {}'.format(all_round / len(results), np.mean(em_acc), np.std(em_acc) / (len(em_acc) ** 0.5), not_early_stop,  file_path))
    else:
        print(sum(early_stop_acc), len(early_stop_acc))
        print(
            'avg round: {}, acc: {} {}, not early stop: {}, early stop acc: {} from {}'.format(all_round / len(results),
                                                                                               np.mean(em_acc),
                                                                                               np.std(em_acc) / (
                                                                                                           len(em_acc) ** 0.5),
                                                                                               not_early_stop,
                                                                                               sum(early_stop_acc) / len(
                                                                                                   early_stop_acc),
                                                                                               file_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument("--result_sa_path", type=str, default="../results/predictions/select_100_single_agent")
    parser.add_argument("--result_ma_path", type=str, default="../results/predictions/select_100_multi_agent")
    args = parser.parse_args()

    print("================ fever 500 =================")
    fever_path = "../results/predictions/fever"
    file_paths = os.listdir(fever_path)
    for file_path in file_paths:
        multi_agent_calculate(os.path.join(fever_path, file_path), file_path.split('_')[0])

    print("================ feverous 500 =================")
    feverous_path = "../results/predictions/feverous"
    file_paths = os.listdir(feverous_path)
    for file_path in file_paths:
        multi_agent_calculate(os.path.join(feverous_path, file_path), file_path.split('_')[0])

    
