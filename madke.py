from tqdm import tqdm
import warnings
from importlib import import_module

from models.chatgpt_agent import ChatGPT
from models.gpt4_agent import GPT4
from parser import get_args
import prompts.triviaqa_prompt as tqa_prompt
import prompts.hotpotqa_prompt as hotpot_prompt
import prompts.fever_prompt as fever_prompt
import prompts.nq_prompt as nq_prompt
import prompts.wikimultihopqa_prompt as wikimultihopqa_prompt
import prompts.medmc_prompt as medmc_prompt

from utils.data_utils import read_json_data, write_json_data, print_args
from utils.metrics_calculator import extract_answer, extract_answer_mc
from utils.prompt_utils import self_choose_evidence, summary_final_answer, judge_consistency
warnings.filterwarnings("ignore")


def select_prompt(args):
    if args.dataset_name == "triviaqa":
        return tqa_prompt
    elif args.dataset_name == "hotpotqa":
        return hotpot_prompt
    elif args.dataset_name == "nq":
        return nq_prompt
    elif args.dataset_name == "fever" or args.dataset_name == "feverous":
        return fever_prompt
    elif args.dataset_name == "wikimultihopqa":
        return wikimultihopqa_prompt
    elif args.dataset_name == "mmlu_med":
        return medmc_prompt
    else:
        print("Prompt type not exist!")


def select_extract_answer(args):
    if args.dataset_name in ["mmlu_med"]:
        return extract_answer_mc
    else:
        return extract_answer


if __name__ == "__main__":
    # Parameters
    args = get_args()
    print_args(args)

    prompt = select_prompt(args)

    test_dataset = read_json_data(args.test_data_path)[59:]

    # Define agent
    Debaters = [ChatGPT(args, "Agent_" + str(i)) for i in range(args.agent_num)]
    Judge = ChatGPT(args, "Judge")
    Summarizer = ChatGPT(args, "Summarizer")

    all_dialog_process = []
    for data in tqdm(test_dataset, total=len(test_dataset)):
        question = data["question"]
        answer = data["answer"]
        wiki_retrieval, google_retrieval = data["wiki_retrieval"], data["google_retrieval"]

        if args.evidence_type == "all_evidence":
            evidence = wiki_retrieval[:args.wiki_num] + google_retrieval[:args.google_num]
        elif args.evidence_type == "only_wiki_5":
            evidence = wiki_retrieval[:args.wiki_num]
        elif args.evidence_type == "only_google_5":
            evidence = google_retrieval[:args.google_num]
        else:
            evidence = []

        all_dialog_process.append({"question": question, "answer": answer, "llm_answers": []})

        agents_dialog_history = {}
        for agent in Debaters:
            agents_dialog_history[agent.agent_name] = []

        # Debate
        for round_num in range(args.max_round):
            all_agent_answer = ""
            all_dialog_process[-1]["stop_round"] = round_num
            all_dialog_process[-1]["Round_{}".format(round_num)] = {}
            for num in range(args.agent_num):
                if round_num == 0:
                    # self choose evidence by agent
                    if args.evidence_type == "no_evidence":
                        chose_evi, evi_res = "", ""
                    else:
                        chose_evi, evi_res = self_choose_evidence(Debaters[num], question, "", evidence, prompt, args.use_self_choose, round_num)
                    # Round 1
                    first_prompt = prompt.debater_first_round_prompt.format(question=question, evidences=chose_evi)
                    debater_instance = [{"role": "system", "content": prompt.debater_system_prompt},
                                        {"role": "user", "content": first_prompt}]

                    agents_dialog_history[Debaters[num].agent_name].extend(debater_instance.copy())
                    response = Debaters[num](debater_instance)
                    agents_dialog_history[Debaters[num].agent_name].append(response)
                else:
                    # self choose evidence by agent
                    if args.evidence_type == "no_evidence":
                        chose_evi, evi_res = "", ""
                    else:
                        chose_evi, evi_res = self_choose_evidence(Debaters[num], question, agents_dialog_history[Debaters[num].agent_name][-1], evidence, prompt, args.use_self_choose, round_num)
                    # Round 2 and beyond
                    agent_historical_answer = "Here is your historical answer: " + agents_dialog_history[Debaters[num].agent_name][-1]
                    other_agent_historical_answer = "Answers from other Agents:\n"
                    for agent_name in agents_dialog_history:
                        if agent_name == Debaters[num].agent_name:
                            continue
                        other_agent_historical_answer += "({}) {}".format(agent_name, agents_dialog_history[agent_name][-1])

                    second_beyond_prompt = prompt.debater_other_round_prompt.format(question=question,
                                                                                    evidences=chose_evi,
                                                                                    answer_from_other_agents=other_agent_historical_answer,
                                                                                    your_historical_answer=agent_historical_answer)

                    debater_instance = [{"role": "system", "content": prompt.debater_system_prompt},
                                        {"role": "user", "content": second_beyond_prompt}]

                    agents_dialog_history[Debaters[num].agent_name].extend(debater_instance.copy())
                    response = Debaters[num](debater_instance)
                    agents_dialog_history[Debaters[num].agent_name].append(response)

                all_dialog_process[-1]["Round_{}".format(round_num)]["User_{}".format(num)] = debater_instance[1]["content"]
                all_dialog_process[-1]["Round_{}".format(round_num)]["Choose_Evi_{}".format(num)] = evi_res
                all_dialog_process[-1]["Round_{}".format(round_num)]["Agent_{}".format(num)] = response
                all_agent_answer += "[Agent {}] {}\n".format(num, response)

            # Judge
            judge_result, judge_response = judge_consistency(Judge, question, all_agent_answer, prompt)
            all_dialog_process[len(all_dialog_process) - 1]["Round_{}".format(round_num)]["Judge"] = judge_response

            if judge_result or round_num == args.max_round - 1:
                if round_num < args.max_round - 1:
                    print("Early stop !")

                # Summary
                summary_result = summary_final_answer(Summarizer, all_agent_answer, question, prompt)
                all_dialog_process[-1]["Summary"] = summary_result
                all_dialog_process[-1]["llm_answers"] = extract_answer(summary_result)
                break

        print(all_dialog_process[-1])

        write_json_data(args.test_output_path, all_dialog_process)



