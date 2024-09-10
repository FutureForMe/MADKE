import time
import requests
import torch.nn as nn


class ChatGPT(nn.Module):
    def __init__(self, args, name):
        super(ChatGPT, self).__init__()
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
            print(end_time - start_time)
        except Exception as ex:
            print("Find error: {}\nSleep 20s...".format(ex))
            time.sleep(20)
            return self.generate_answer(answer_context)

        try:
            chat_response = chat_response.json()
        except Exception as ex:
            print("Find Error {}".format(ex))
            return "Error"

        if "error" in chat_response and chat_response["error"]["code"] == '429':
            print("Error Code 429, sleep 10s ...")
            print(chat_response)
            time.sleep(10)
            return self.generate_answer(answer_context)

        return chat_response

    def get_assistant_message(self, completion):
        """get message of agent"""
        if completion == "Error":
            return "Error"

        if "choices" not in completion:
            return completion["error"]["message"]

        if "content" not in completion["choices"][0]["message"]:
            completion["choices"][0]["message"]["content"] = ""

        if not isinstance(completion["choices"][0]["message"]["content"], str):
            completion["choices"][0]["message"]["content"] = "Error"

        return completion["choices"][0]["message"]["content"]

    def forward(self, answer_context):
        completion = self.generate_answer(answer_context)
        response = self.get_assistant_message(completion)
        return response

