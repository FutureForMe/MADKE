import os
import json
import numpy as np
import tiktoken
from data_utils import read_json_data


def num_tokens_from_string(input_str, encoding_name="cl100k_base"):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(input_str))
    return num_tokens


def calculate_baseline_tokens(baseline_path):
    baseline_results = read_json_data(baseline_path)
    all_input_tokens_num, all_output_tokens_num = [], []
    for data in baseline_results:
        instruction = data["Instruction"]
        if isinstance(data["Summary"], list):
            summaries = data["Summary"]
        else:
            summaries = [data["Summary"]]
        input_token_num = num_tokens_from_string(instruction)
        if "sc_3" in baseline_path:
            input_token_num = input_token_num * 3
        all_input_tokens_num.append(input_token_num)

        output_token_num = 0
        for summary in summaries:
            output_token_num += num_tokens_from_string(summary)
        all_output_tokens_num.append(output_token_num)

    print("Avg input tokens: {}, Avg output tokens: {}, from {}".format(np.mean(all_input_tokens_num),
                                                                        np.mean(all_output_tokens_num),
                                                                        baseline_path))


def calculate_mit_debate(debate_path):
    debate_results = read_json_data(debate_path)
    all_input_tokens_num, all_output_tokens_num = [], []
    for data in debate_results:
        input_tokens_num, output_tokens_num = 0, 0
        agent_contexts = data["agent_contexts"]
        for agent_context in agent_contexts:
            for message in agent_context:
                if message["role"] == "user":
                    input_tokens_num += num_tokens_from_string(message["content"])
                else:
                    output_tokens_num += num_tokens_from_string(message["content"])
        all_input_tokens_num.append(input_tokens_num)
        all_output_tokens_num.append(output_tokens_num)
    print("Avg input tokens: {}, Avg output tokens: {}, from {}".format(np.mean(all_input_tokens_num),
                                                                        np.mean(all_output_tokens_num),
                                                                        debate_path))


def calculate_debate_tokens(debate_path):
    debate_results = read_json_data(debate_path)
    all_output_tokens_num = []
    for data in debate_results:
        output_tokens_num = 0
        for key in data:
            if "Round" in key:
                for round_key in data[key]:
                    if "User" in round_key:
                        continue
                    output_tokens_num += num_tokens_from_string(data[key][round_key])
            if "Summary" in key:
                output_tokens_num += num_tokens_from_string(data[key])
        all_output_tokens_num.append(output_tokens_num)
    print(" Avg output tokens: {}, from {}".format(np.mean(all_output_tokens_num), debate_path))


def get_filelist(dir, Filelist):
    newDir = dir
    if os.path.isfile(dir):
        Filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir=os.path.join(dir, s)
            get_filelist(newDir, Filelist)
    return Filelist


if __name__ == "__main__":
    result_path = "../results/predictions/triviaqa/"
    all_paths = get_filelist(result_path, [])
    for file_path in all_paths:
        if "reflect" in file_path:
            continue
        if "qwen15_72b" in file_path:
            if "mit_debate" in file_path:
                calculate_mit_debate(file_path)
            elif "baseline" in file_path:
                calculate_baseline_tokens(file_path)
            else:
                calculate_debate_tokens(file_path)
