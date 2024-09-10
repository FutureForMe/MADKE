import os
import argparse
import warnings
import numpy as np
from tqdm import tqdm

from models.gpt4_agent import GPT4
from utils.data_utils import read_json_data, write_json_data, read_jsonl_data
from prompts.eval_gpt4_prompt import eval_system_prompt, eval_prompt
from utils.metrics_calculator import extract_answer
from utils.evaluation_utils import extract_answer, get_em_f1_scores

warnings.filterwarnings("ignore")


def calculate_gpt4_score(result_data):
    """
    Use GPT4 to evaluate the reasoning result
    """

    eval_gpt4 = GPT4(args, "Eval")

    eval_process = []
    acc = []
    for result in tqdm(result_data, total=len(result_data)):
        question = result["question"]
        reference_answers = ", ".join(result["answer"])
        evaluation_answer = result["llm_answers"]

        if len(evaluation_answer) > 0:
            evaluation_answer = evaluation_answer[-1]
        else:
            evaluation_answer = "No Answer"

        eval_user_prompt = eval_prompt.format(question=question,
                                              reference_answers=reference_answers,
                                              evaluation_answer=evaluation_answer)
        eval_instance = [{"role": "system", "content": eval_system_prompt},
                         {"role": "user", "content": eval_user_prompt}]

        response = eval_gpt4(eval_instance)
        gpt4_answer = extract_answer(response)

        print(response)
        if "True" in gpt4_answer:
            acc.append(1.0)
        else:
            acc.append(0.0)

        if "Summary" not in result:
            result["Summary"] = ""

        eval_process.append({"question": question,
                             "truth_answer": result["answer"],
                             "llm_answer": evaluation_answer,
                             "llm_summary": result["Summary"],
                             "eval_gpt4": response})

    acc_message = 'acc: {} {} from {}'.format(np.mean(acc), np.std(acc) / (len(acc) ** 0.5), args.prediction_path)
    gpt4_score = {"gpt4_score": acc_message}
    print(acc_message)

    return eval_process, gpt4_score


def calculate_em_f1(result_data):
    """
    
    """
    predictions, references = {}, {}
    for result_id, result in enumerate(result_data):
        if isinstance(result["llm_answers"], list):
            if len(result["llm_answers"]) > 0:
                predictions[result_id] = result["llm_answers"][-1]
            else:
                predictions[result_id] = ""
        else:
            predictions[result_id] = result["llm_answers"]
        if isinstance(result["answer"], list):
            references[result_id] = result["answer"]
        else:
            references[result_id] = [result["answer"]]
    
    em_f1_scores = get_em_f1_scores(predictions, references)
    print(em_f1_scores)

    return em_f1_scores


def calculate_acc_c(args, predict_data, eval_data):
    early_stop_questions = []
    avg_stop_round = 0
    for data in predict_data:
        stop_round = data["stop_round"]
        avg_stop_round += (stop_round + 1)
        if stop_round < args.round_num - 1:
            early_stop_questions.append(data["question"])
            continue

        if "[Yes]" in data["Round_" + str(stop_round)]:
            early_stop_questions.append(data["question"])

    all_consistent_num = len(early_stop_questions)
    correct_consistent_num = 0
    if args.dataset_name == "fever":
        for data in predict_data:
            if isinstance(data["llm_answers"], list):
                if len(data["llm_answers"]) > 0:
                    predict_answer = data["llm_answers"][-1]
                else:
                    predict_answer = ""
            else:
                predict_answer = data["llm_answers"]

            truth_answer = data["answer"][-1]
            if truth_answer.lower() == predict_answer.lower() and data["question"] in early_stop_questions:
                correct_consistent_num += 1
    else:
        for data in eval_data[:-1]:
            if "[True]" in data["eval_gpt4"] and data["question"] in early_stop_questions:
                correct_consistent_num += 1

    acc_c = correct_consistent_num / all_consistent_num

    print("Acc_c: {}, correct consistent num: {}, all consistent num: {}, avg stop round: {}".format(acc_c,
                                                                                                     correct_consistent_num / len(predict_data),
                                                                                                     all_consistent_num / len(predict_data),
                                                                                                     avg_stop_round / len(predict_data)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation metrics by GPT4")
    parser.add_argument("--eval_metrics", type=str, default="acc_c", choices=["gpt4_score", "acc_c", "em_f1", "all"])
    parser.add_argument("--dataset_name", type=str, default="triviaqa")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--round_num", type=int, default=3)
    parser.add_argument("--prediction_path", type=str)
    parser.add_argument("--evaluation_path", type=str)
    args = parser.parse_args()

    if args.prediction_path.endswith(".json"):
        predict_data = read_json_data(args.prediction_path)
    else:
        predict_data = read_jsonl_data(args.prediction_path)

    if args.eval_metrics == "gpt4_score":
        eval_process, eval_result = calculate_gpt4_score(predict_data)
        eval_process.append(eval_result)
        write_json_data(args.evaluation_path, eval_process)
    elif args.eval_metrics == "acc_c":
        if "fever" in args.prediction_path:
            eval_data = ""
            args.dataset_name = "fever"
        else:
            eval_data = read_json_data(args.evaluation_path)
        calculate_acc_c(args, predict_data, eval_data)
    elif args.eval_metrics == "em_f1":
        eval_result = calculate_em_f1(predict_data)
    else:
        em_f1_eval_result = calculate_em_f1(predict_data)
        eval_process, gpt4_eval_result = calculate_gpt4_score(predict_data)
        gpt4_eval_result.update(em_f1_eval_result)
        eval_process.append(gpt4_eval_result)
        write_json_data(args.evaluation_path, eval_process)

