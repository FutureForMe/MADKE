import time
import requests
import torch.nn as nn


class GPT4(nn.Module):
    def __init__(self, args, name):
        super(GPT4, self).__init__()
        self.temperature = args.temperature
        self.agent_name = name
        self.url = "https://{endpoint}.openai.azure.com/openai/deployments/{deployment_id}/chat/completions?api-version=2023-07-01-preview".format(endpoint=args.endpoint, deployment_id = args.deployment_id)
        self.header = {"Content-Type": "application/json", "api-key": args.api_key}

    def generate_answer(self, answer_context):
        try:
            message_json = {"messages": answer_context, "temperature": self.temperature}
            start_time = time.time()
            chat_response = requests.post(url=self.url, headers=self.header, json=message_json, verify=False, timeout=120)
            end_time = time.time()
            time.sleep(5)
            print(end_time - start_time)
        except Exception as ex:
            print("Find error: {}\nSleep 20s...".format(ex))
            time.sleep(20)
            return self.generate_answer(answer_context)

        chat_response = chat_response.json()
        if "error" in chat_response and chat_response["error"]["code"] == '429':
            print("Error Code 429, sleep 10s ...")
            time.sleep(10)
            return self.generate_answer(answer_context)

        return chat_response

    def get_assistant_message(self, completion):
        """get message of agent"""
        if completion == "Error":
            return {"role": "assistant", "content": "Error"}

        if "choices" not in completion:
            return {"role": "assistant", "content": completion["error"]["message"]}

        if "content" not in completion["choices"][0]["message"]:
            completion["choices"][0]["message"]["content"] = ""

        return completion["choices"][0]["message"]

    def forward(self, answer_context):
        completion = self.generate_answer(answer_context)
        response = self.get_assistant_message(completion)
        return response["content"]

