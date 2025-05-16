
import argparse
import json
import os
import time
import re
import ast

from tqdm import tqdm
import matplotlib.pyplot as plt
from openai import OpenAI
import httpx

from evaluation.evaluate_easy import eval_easy_gen
from evaluation.evaluate_normal import eval_normal_gen
from evaluation.evaluate_hard import eval_hard_gen




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./data/question.jsonl")
    parser.add_argument("--model_id", type=str, default="qwq-32b")
    parser.add_argument("--eval_rounds", type=int, default=5)

    args = parser.parse_args()

    openai_url = os.getenv("OPENAI_BASE_URL", '')
    api_key = os.getenv("OPENAI_API_KEY", '')
    client = OpenAI(base_url=openai_url, api_key = api_key, http_client=httpx.Client(verify=False), timeout=3600)


    eval_result_path = f"eval_results/{args.model_id}"
    if not os.path.exists(eval_result_path):
        os.makedirs(eval_result_path)

    print("Evaluating easy generation...")
    easy_gen_acc = eval_easy_gen(args.model_id, client, args.eval_rounds)
    print("Evaluate easy generation done!" + '\n')

    save_path = f"{args.model_id}/easy_rec/accuracy.jsonl"
    if not os.path.exists(save_path):
        print("Easy recognition results missing!")
        easy_rec_acc = 0
    else:
        with open(save_path, "r") as f:
            record = json.loads(f.readline().strip())
        easy_rec_acc = record["accuracy"]

    print("Evaluating normal generation...")
    normal_gen_acc = eval_normal_gen(args.model_id, client, args.eval_rounds)
    print("Evaluate normal generation done!" + '\n')

    save_path = f"{args.model_id}/normal_rec/accuracy.jsonl"
    if not os.path.exists(save_path):
        print("Normal recognition results missing!")
        normal_rec_acc = 0
    else:
        with open(save_path, "r") as f:
            record = json.loads(f.readline().strip())
        normal_rec_acc = record["accuracy"]

    print("SEvaluating hard generation...")
    hard_gen_acc = eval_hard_gen(args.model_id, client, args.eval_rounds)
    print("Evaluate hard generation done!" + '\n')

    with open(f"eval_results/{args.model_id}/total_gpt4.1_accuracy.jsonl", "w", encoding='utf-8') as f:
        record = {"level": "easy", "task": "generation", "accuracy": easy_gen_acc}
        f.write(json.dumps(record) + '\n')
        record = {"level": "easy", "task": "recognition", "accuracy": easy_rec_acc}
        f.write(json.dumps(record) + '\n')
        record = {"level": "normal", "task": "generation", "accuracy": normal_gen_acc}
        f.write(json.dumps(record) + '\n')
        record = {"level": "normal", "task": "recogntion", "accuracy": normal_rec_acc}
        f.write(json.dumps(record) + '\n')
        record = {"level": "hard", "task": "generation", "accuracy": hard_gen_acc}
        f.write(json.dumps(record) + '\n')

        average_accuracy = (easy_gen_acc + easy_rec_acc + normal_gen_acc + normal_rec_acc + hard_gen_acc) / 5
        record = {"total_average_accuracy": average_accuracy}
        f.write('\n' + json.dumps(record))

    print("All done!")
