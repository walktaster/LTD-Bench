
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
import base64


def eval_easy_gen(model_id, client, times=5):
    result_path = model_id + "/easy_gen/results.jsonl"
    if not os.path.exists(result_path):
        print("Easy generation results missing!")
        return 0
    
    results = []
    with open(result_path, 'r') as data_file:
        for line in data_file:
            data = json.loads(line.strip())
            results.append(data)

    with open(f"eval_results/{model_id}/easy_gen_results.jsonl", "w", encoding='utf-8') as f:
        pass
    with open(f"eval_results/{model_id}/easy_gen_accuracy.jsonl", "w", encoding='utf-8') as f:
        pass

    average_accuracy = 0
    for time in range(times):
        print("Round: ", time)
        correct = 0
        wrong = 0
        fail = 0
        for data in tqdm(results):
            obj = data["ground_truth"]
            question = data["question"]
            mat = data["answer"]

            if mat == "Failed":
                wrong += 1
                record = {"question_id": data["question_id"], "eval_time": time, "eval_result": "Flase" , "ground_truth": data["ground_truth"], "answer": mat, "question": question}
                with open(f"eval_results/{model_id}/easy_gen_results.jsonl", "a", encoding='utf-8') as f:
                    f.write(json.dumps(record) + "\n")
                continue

            if not obj.endswith("lash"):
                obj = '\'' + obj + '\''

            prompt = f'''The following is a dot matrix, where 1 represents fill and 0 represents blank. You need to determine whether the character drawn by the dot matrix is {obj}. If so, output [[Yes]], otherwise output [[No]].
            Here is the dot matrix:
            {mat}'''

            try:
                response = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.0,
                )

                text = response.choices[0].message.content
                answer = text.split("[[")[-1].split("]]")[0].strip()
                if answer == 'Yes':
                    correct += 1
                    record = {"question_id": data["question_id"], "eval_round": time, "eval_result": "True" , "ground_truth": data["ground_truth"], "answer": mat, "question": question}
                elif answer == 'No':
                    wrong += 1
                    record = {"question_id": data["question_id"], "eval_round": time, "eval_result": "Flase" , "ground_truth": data["ground_truth"], "answer": mat, "question": question}
                else:
                    fail += 1
                    record = {"question_id": data["question_id"], "eval_round": time, "eval_result": "eval error" , "ground_truth": data["ground_truth"], "answer": mat, "question": question}
            except:
                fail += 1
                record = {"question_id": data["question_id"], "eval_round": time, "eval_result": "eval error" , "ground_truth": data["ground_truth"], "answer": mat, "question": question}

            with open(f"eval_results/{model_id}/easy_gen_results.jsonl", "a", encoding='utf-8') as f:
                f.write(json.dumps(record) + "\n")
    
        accuracy = correct / len(results)
        average_accuracy += accuracy
        with open(f"eval_results/{model_id}/easy_gen_accuracy.jsonl", "a") as f:
            f.write(json.dumps({"accuracy": accuracy, "eval_round": time, "correct": correct, "wrong": wrong, "eval_error": fail}) + "\n")

    average_accuracy /= times
    with open(f"eval_results/{model_id}/easy_gen_accuracy.jsonl", "a") as f:
        f.write(json.dumps({"Average_accuracy": average_accuracy}) + "\n")
    return average_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./data/query_easy_EN.jsonl")
    parser.add_argument("--model_id", type=str, default="qwq-32b")

    parser.add_argument("--model_url", type=str, default="https://openai.wokaai.cn/v1/")
    parser.add_argument("--api_key", type=str, default="sk-CjbxGrKmsxyseMGUJ4RqiHZK8Z1MhDtuQzHxQRg1YYndQ07D")

    args = parser.parse_args()
    args.savedir = "./eval_score/gpt-4.1"
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    client = OpenAI(base_url=args.model_url, api_key = args.api_key, http_client=httpx.Client(verify=False), timeout=3600)
