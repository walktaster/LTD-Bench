
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


def normal_rec(questions, model_id, client):
    
    save_path = model_id + "/normal_rec"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with open(f"{save_path}/results.jsonl", 'w', encoding='utf-8') as f:
        pass

    correct = 0
    total = 0
    for data in tqdm(questions):
        total += 1
        obj = data["object"]
        question = data["question"]
        prompt = data["prompt"]

        messages = [
            {"role": "user", "content": prompt},
        ]

        response = client.chat.completions.create(
            model=model_id,  
            messages=messages,
            max_tokens=8192,
            temperature=0.0,
            stream=True)

        text = ""
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    text += chunk.choices[0].delta.content

        try:
            pattern = re.compile(r'<<(.*?)>>', re.DOTALL)
            ans = pattern.findall(text)[-1]
            ans = ans.split('<<')[-1].strip()
        except:
            ans = "Failed"

        if ans == obj:
            correct += 1
        with open(f"{save_path}/results.jsonl", "a", encoding='utf-8') as f:
            record = {"question_id": data["question_id"], "ground_truth": obj, "answer": ans, "question":question}
            f.write(json.dumps(record) + "\n")
    
    accuracy = correct/total
    with open(f"{save_path}/accuracy.jsonl", "w", encoding='utf-8') as f:
        record = {"level": "normal", "task": "recognition", "accuracy": accuracy}
        f.write(json.dumps(record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./data/question.jsonl")
    parser.add_argument("--model_id", type=str, default="deepseek-r1")

    args = parser.parse_args()

    base_url = os.getenv("API_BASE_URL")
    api_key = os.getenv("API_KEY")
    client = OpenAI(base_url=args.model_url, api_key = args.api_key, http_client=httpx.Client(verify=False), timeout=3600)

    normal_rec_questions = []
    with open(args.data_path, 'r') as data_file:
        for line in data_file:
            data = json.loads(line.strip())
            if data["level"] == "normal" and data["task"] == "recognition":
                    normal_rec_questions.append(data)
            else:
                continue

    normal_rec(normalrec_questions, args.model_id, client)
