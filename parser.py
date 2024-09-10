import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Multi Agent Debate')
    parser.add_argument("--dataset_name", type=str, default="triviaqa",
                        choices=["hotpotqa", "fever", "feverous", "nq", "triviaqa", "strategyqa", "wikimultihopqa", "mmlu_med", "medmcqa"])
    parser.add_argument("--test_data_path", type=str,
                        default="./data/processed/triviaqa/triviaqa_test_retrieval_all_evidence.json")
    parser.add_argument("--test_output_path", type=str,
                        default="./results/predictions/triviaqa/triviaqa_test_500_google_chatgpt_a2r3.json")
    
    parser.add_argument("--endpoint", type=str, required=True)
    parser.add_argument("--deployment_id", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    
    parser.add_argument("--agent_num", type=int, default=2)
    parser.add_argument("--max_round", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--use_self_choose", action="store_false")
    parser.add_argument("--wiki_num", type=int, default=10)
    parser.add_argument("--google_num", type=int, default=5)
    parser.add_argument("--textbooks_num", type=int, default=5)
    parser.add_argument("--evidence_type", type=str, default="all_evidence",
                        choices=["all_evidence", "only_wiki_5", "only_google_5", "only_textbooks_5", "no_evidence"])

    args = parser.parse_args()

    return args
