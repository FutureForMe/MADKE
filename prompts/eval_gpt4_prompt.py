
eval_system_prompt = """As a judge agent, your primary responsibility is to impartially evaluate the evaluation answer for correct. Please ensure your judgments are objective, relying solely on the coherence and alignment of the provided answers.
"""

eval_prompt = """You need to judge the correctness of the evaluation answer based on the reference answers to the question.  Return [True] if the answer to be evaluated is the correct answer to the question, otherwise return [False].
Here are some examples:
Question: The VS-300 was a type of what?
Reference answers: Helicopters, Civilian helicopter, Pescara (helicopter), Cargo helicopter, Copter, Helecopter, List of deadliest helicopter crashes, Helichopper, Helocopter, Cargo Helicopter, Anatomy of a helicopter
Evaluation answer: helicopter
Answer: Let's think step by step! The evaluation answer, "helicopter" is indeed a correct answer to the question, "The VS-300 was a type of what?" It matches with the reference answers provided, which include "Helicopters," "Civilian helicopter," "Cargo helicopter," and other variants of the word "helicopter." Therefore, the answer is [True].

Question: Who played a character based on Bob Fosse in a 1979 Oscar winning film?
Reference answers: Roy Scheider, Roy R. Scheider
Evaluation answer: Cliff Gorman
Answer: Let's think step by step! The evaluation answer is incorrect. The character based on Bob Fosse in the 1979 Oscar-winning film "All That Jazz" was played by Roy Scheider, not Cliff Gorman. Therefore, the answer is [False].
(END OF EXAMPLES)

Question: {question}
Reference answer: {reference_answers}
Evaluation answer: {evaluation_answer}
Judge answer: Let's think step by step! """