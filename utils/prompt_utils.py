import re


def get_assistant_message(completion):
    """get message of agent"""

    if completion == "Error":
        return {"role": "assistant", "content": "Error"}

    if "choices" not in completion:
        return {"role": "assistant", "content": completion["error"]["message"]}

    if "content" not in completion["choices"][0]["message"]:
        completion["choices"][0]["message"]["content"] = ""

    return completion["choices"][0]["message"]


def self_choose_evidence(agent, question, history_response, evi_pool, prompt, use_self_choose, round_num):

    all_evi_pool = "Evidence: "
    for i, evi in enumerate(evi_pool):
        evi = evi["contents"].replace("\n", " - ")
        all_evi_pool += "[{}] {}\n".format(i, evi)

    if not use_self_choose:
        evi_res = "Self choose evidence not be used, return {} evidence of all evidence pool.".format(len(evi_pool))
        return all_evi_pool, evi_res

    if round_num == 0:
        evi_prompt = prompt.choose_evidence_prompt.format(evidences=all_evi_pool, question=question)
    else:
        evi_prompt = prompt.other_round_choose_evidence_prompt.format(evidences=all_evi_pool,
                                                                      your_historical_answer=history_response,
                                                                      question=question)

    instance = [{"role": "system", "content": prompt.choose_evidence_system_prompt},
                {"role": "user", "content": evi_prompt}]

    evi_res = agent(instance)

    pattern = re.compile((r"\[(.*?)\]"))
    choose_ids = re.findall(pattern, evi_res)
    choose_ids.reverse()

    # evi_num = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]
    evi_num = [str(i) for i in range(len(evi_pool))]
    choose_evi = ""
    if "No Found" in choose_ids:
        return choose_evi, evi_res
    else:
        choose_evi += "Evidence: "
        # choose_evi += evi_res
        valid_evidence_num = []
        for evi_id in choose_ids:
            if evi_id not in evi_num or evi_id in valid_evidence_num:
                continue
            valid_evidence_num.append(evi_id)
            if len(valid_evidence_num) > 3:
                break
            choose_evi += "({}) {}\n".format(evi_id, evi_pool[int(evi_id)]["contents"].replace("\n", " - "))
        return choose_evi, evi_res


def judge_consistency(judge_agent, question, all_agent_answer, prompt):

    judge_instance = [{"role": "system", "content": prompt.judge_system_prompt},
                      {"role": "user", "content": prompt.judge_prompt.format(question=question,
                                                                             all_answers_from_agents=all_agent_answer)}]
    judge_result = judge_agent(judge_instance)

    pattern = re.compile((r"\[(.*?)\]"))
    result = re.findall(pattern, judge_result)

    if "Yes" in result:
        return True, judge_result
    else:
        return False, judge_result


def summary_final_answer(summary_agent, question, all_agent_answer, prompt):

    summary_instance = [{"role": "system", "content": prompt.summary_system_prompt},
                        {"role": "user", "content": prompt.summary_prompt.format(all_answers_from_agents=all_agent_answer,
                                                                                 question=question)}]

    summary_result = summary_agent(summary_instance)

    return summary_result



