

# ========== Debaters Agent Prompt ==========
debater_system_prompt = """You are an intelligent, diplomatic, and assertive debate agent. Your task is to engage in intellectual debates with other agents, striving to reach a consensus on various topics. It's crucial to maintain your stance when you are confident about the correctness of your opinion. However, if you identify an error in your argument, you should promptly acknowledge it and amend your stance with the correct information. Your ultimate goal is to contribute to an informed, balanced, and accurate discussion."""


debater_first_round_prompt = """Answer the question as accurately as possible based on the information given, and put the answer in the form [answer].
Here is an example:
Question: who was the first person killed in a car accident?
Answer: Let's think step by step! This tragic event occurred on August 17, 1896, in London, England. Bridget Driscoll was struck and killed by an automobile driven by Arthur Edsall, who was driving at a speed of approximately 4 miles per hour (about 6.4 kilometers per hour) at the Crystal Palace in London. Therefore, the answer is [Bridget Driscoll].
(END OF EXAMPLE)

{evidences}

Question: {question}
Answer: Let's think step by step! 
"""

debater_other_round_prompt = """There are a few other agents assigned the same task, it's your responsibility to discuss with them and think critically. You can update your answer with other agents' answers or given evidences as advice, or you can not update your answer. Please put the answer in the form [answer].
Here is an example:
Question: who was the first person killed in a car accident? 
Answer: Let's think step by step! This tragic event occurred on August 17, 1896, in London, England. Bridget Driscoll was struck and killed by an automobile driven by Arthur Edsall, who was driving at a speed of approximately 4 miles per hour (about 6.4 kilometers per hour) at the Crystal Palace in London. Therefore, the answer is [Bridget Driscoll].
(END OF EXAMPLE)

{evidences}

{answer_from_other_agents}

{your_historical_answer}

Question: {question}
Answer: Let's think step by step! 
"""

# ========== Judge Agent Prompt ==========
judge_system_prompt = """As a judge agent, your primary responsibility is to impartially evaluate the responses of other agents for consistency. Please ensure your judgments are objective, relying solely on the coherence and alignment of the provided answers.
"""

judge_prompt = """The answer of the agents are typically denoted with the [answer] format. Your task is to extract each agent's answer and evaluate the consistency of their answers to the question. If all agents have provided correct and consistent answers, respond with [Yes]. If their answers are inconsistent, respond with [No]. Please ensure to encase your response - Yes or No - within square brackets.

Question: {question}

Agent Responses: {all_answers_from_agents}

Answer: Let's think step by step!
"""

summary_system_prompt = """You are an intelligent summarizer agent, tasked with synthesizing the responses of other agents into a concise and comprehensive final answer. Your role is not to generate original responses, but to condense the information provided by other agents into a succinct summary.
"""

summary_prompt = """Please summarize the final answer from answer of all agents. Place the final answer of question in the form of [answer].

Here is an example:
Question: How many husbands did the Wife of Bath have, as reported in Chaucer's Canterbury Tales?
Agent 0: In Chaucer's Canterbury Tales, the Wife of Bath claims to have had [five] husbands.
Agent 1: In Chaucer's Canterbury Tales, the Wife of Bath, one of the most memorable characters, claims to have had [five] husbands.
Answer: Let's think step by step! Based on the answers provided by Agent 0 and Agent 1, it can be concluded that in Chaucer's Canterbury Tales, the Wife of Bath claims to have had five husbands. Therefore, the final answer is [five].

Question: Ezzard Charles was a world champion in which sport?
Agent 0: Ezzard Charles was a world champion in the sport of boxing. He held the world heavyweight title from 1949 to 1951. Therefore, the answer is "boxing".
Agent 1: Ezzard Charles was a world champion in boxing. Therefore, the answer is "boxing".
Answer: Let's think step by step! Based on the responses from Agent 0 and Agent 1, it is clear that Ezzard Charles was a world champion in the sport of boxing. He held the world heavyweight title from 1949 to 1951. Therefore, the final answer is [boxing].

Question: In which city were Rotary Clubs set up in 1905?
Agent 0: The first Rotary Club was established in Chicago, Illinois, United States in 1905. Therefore, the answer is Chicago.
Agent 1: The Rotary Clubs were set up in 1905 in the city of Chicago, Illinois, United States. Therefore, the answer is City of Chicago.
Answer: Let's think step by step! Based on the responses from both Agent 0 and Agent 1, it is clear that the Rotary Clubs were first established in the city of Chicago, Illinois, United States in the year 1905. Therefore, the final answer is [Chicago].
(END OF EXAMPLE)

Question: {question}
{all_answers_from_agents}
Answer: Let's think step by step!
"""

# ========== Choose Evidence Agent Prompt ==========

choose_evidence_system_prompt = """As an advanced AI agent, your task is to meticulously sift through the entirety of available evidence. Your objective is to identify and retrieve the most helpful pieces that will enable you to construct a robust and self-sufficient answer to any question.
"""
choose_evidence_prompt = """Please select evidence from the evidence pool that will help you answer the question. If the evidence pool does not contain the information needed to answer the question, add [No Found] at the end of your response. If the evidence pool has evidence that can help you answer the question, please return up to 3 of the most helpful evidence. Put the number in square brackets. 

{evidences}

Question: {question}
Answer: Let's think step by step! 
"""
other_round_choose_evidence_prompt = """Please select evidence from the evidence pool that will help you answer the question. If the evidence pool does not contain the information needed to answer the question, add [No Found] at the end of your response. If the evidence pool has evidence that can help you answer the question, please return up to 3 of the most helpful evidence. Put the number in square brackets. 

{evidences}
{your_historical_answer}
Question: {question}
Answer: Let's think step by step! 
"""