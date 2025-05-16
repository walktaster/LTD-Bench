
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


def hard_gen(questions, model_id, client):
    
    save_path = model_id + "/hard_gen"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with open(f"{save_path}/results.jsonl", 'w', encoding='utf-8') as f:
        pass

    for data in tqdm(questions):
        obj = data["object"]
        question = data["question"]
        prompt = data["prompt"]

        obj_path = f"{save_path}/{obj}"
        if not os.path.exists(obj_path):
            os.makedirs(obj_path)

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
            pattern = re.compile(r'<Code>(.*?)</Code>', re.DOTALL)
            code = pattern.findall(text)[-1]
            code = code.split('<Code>')[-1]
            code = code.replace('plt.show()','')
            code = code.replace('turtle.done()','')
            code = code.replace('test.eps', f'{obj_path}/{obj}.eps')
            code = code.replace('test.ep', f'{obj_path}/{obj}.ep')
            code = code.replace('test.jpg', f'{obj_path}/{obj}.jpg')
            with open(f"{obj_path}/plot.py", "w") as f:
                f.write(code)
            with open(f"{obj_path}/plot.sh", "w") as f:
                f.write(f"export DISPLAY=:1\npython {objdir}/plot.py")
            os.system(f"sh {obj_path}/plot.sh")
            ans_path = f"{obj_path}/{obj}.jpg"
        except:
            ans_path = "Failed"

        try:
            pattern = re.compile(r'<Thought>(.*?)</Thought>', re.DOTALL)
            thought = pattern.findall(text)[-1]
            thought = thought.split('<Thought>')[-1]
            with open(f"{obj_path}/thought.txt", "w") as f:
                f.write(thought)
        except:
            pass
        
        if not os.path.exists(f"{obj_path}/{obj}.jpg"):
            ans_path = "Failed"

        with open(f"{save_path}/results.jsonl", "a", encoding='utf-8') as f:
            record = {"question_id": data["question_id"], "ground_truth": obj, "answer": ans_path, "question":question}
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./data/question.jsonl")
    parser.add_argument("--model_id", type=str, default="deepseek-r1")

    args = parser.parse_args()

    base_url = os.getenv("API_BASE_URL")
    api_key = os.getenv("API_KEY")
    client = OpenAI(base_url=args.model_url, api_key = args.api_key, http_client=httpx.Client(verify=False), timeout=3600)

    hard_gen_questions = []
    with open(args.data_path, 'r') as data_file:
        for line in data_file:
            data = json.loads(line.strip())
            if data["level"] == "hard" and data["task"] == "generation":
                    hard_gen_questions.append(data)
            else:
                continue

    hard_gen(hard_gen_questions, args.model_id, client)
