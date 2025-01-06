# _Learning to Break:_ Knowledge-Enhanced Reasoning in Multi-Agent Debate System

- 🔥 **Update**: This paper has been accepted to Neurocomputing!!  [[Paper](https://www.sciencedirect.com/science/article/pii/S0925231224018344)]

## Running experiments
**Preparation**
```shell
tqdm
openai
vllm
torch
```


**Run MADKE**
- We used the GPT3.5 and GPT4 apis provided by Microsoft Azure to complete the experiment. You need to specify an endpoint, deployment id, and api key. 
- We used VLLM to deploy Qwen1.5-32B-Chat and Qwen1.5-72B-Chat, and then called the API. Run the corresponding model with the following code.

```shell
python -m vllm.entrypoints.openai.api_server --model /home/llm_models/Qwen1.5-32B-Chat  --served-model-name qwen1.5-32b-chat
```

```shell
python madke.py --test_data_path ./data/triviaqa/triviaqa_test_retrieval_all_evidence.json \
--test_output_path ./results/predictions/triviaqa/triviaqa_test_500_google_chatgpt_a2r3.json \
--endpoint YOUR_ENDPOINT \
--deployment_id YOUR_DEPLOYMENT_ID \
--api_key YOUR_API_KEY 

```

**Run Evaluation**
```shell
python evaluation.py \
--endpoint YOUR_ENDPOINT \
--deployment_id YOUR_DEPLOYMENT_ID \
--api_key YOUR_API_KEY 
```

## Citation
If you find this work helpful, we kindly request that citations refer to the arXiv version:
```bibtex
@article{WANG2025129063,
title = {Learning to break: Knowledge-enhanced reasoning in multi-agent debate system},
journal = {Neurocomputing},
volume = {618},
pages = {129063},
year = {2025},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2024.129063},
url = {https://www.sciencedirect.com/science/article/pii/S0925231224018344},
author = {Haotian Wang and Xiyuan Du and Weijiang Yu and Qianglong Chen and Kun Zhu and Zheng Chu and Lian Yan and Yi Guan}
}
 ```
