import time
import torch.nn as nn
from openai import OpenAI


class Qwen_Agent(nn.Module):
    def __init__(self, args, name):
        super(Qwen_Agent, self).__init__()
        self.temperature = args.temperature
        self.agent_name = name
        self.openai_api_key = args.api_key
        self.openai_api_base = args.base_url
        self.model = args.model
        self.client = OpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_api_base,
        )

    def generate_answer(self, answer_context):
        try:
            start_time = time.time()
            chat_response = self.client.chat.completions.create(
                model=self.model,
                messages=answer_context,
                temperature=self.temperature
            )
            end_time = time.time()
            print(end_time - start_time)
        except Exception as ex:
            print("Find error: {}\nSleep 10s...".format(ex))
            time.sleep(10)
            return self.generate_answer(answer_context)

        return chat_response

    def get_assistant_message(self, completion):
        response = completion.choices[0].message.content
        return response

    def forward(self, answer_context):
        completion = self.generate_answer(answer_context)
        response = self.get_assistant_message(completion)
        return response

