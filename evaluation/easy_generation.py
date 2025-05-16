
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

from prompt import easy_plot_code

def easy_gen(questions, model_id, client):
    
    save_path = model_id + "/easy_gen"
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
            pattern = re.compile(r'<Mat>(.*?)</Mat>', re.DOTALL)
            mat = pattern.findall(text)[-1]
            mat = mat.split('<Mat>')[-1]

            plot_code = easy_plot_code.replace('mat = []', mat)
            plot_code = plot_code.replace('test.jpg', f'{obj_path}/{obj}.jpg')
            with open(f"{obj_path}/plot.py", "w") as f:
                f.write(plot_code)
            os.system(f"python {obj_path}/plot.py")
        except:
            mat = "Failed"

        with open(f"{obj_path}/mat.py", "w") as f:
            f.write(mat)

        with open(f"{save_path}/results.jsonl", "a", encoding='utf-8') as f:
            record = {"question_id": data["question_id"], "ground_truth": obj, "answer": mat, "question":question}
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./data/question.jsonl")
    parser.add_argument("--model_id", type=str, default="deepseek-r1")

    args = parser.parse_args()

    base_url = os.getenv("API_BASE_URL")
    api_key = os.getenv("API_KEY")
    client = OpenAI(base_url=args.model_url, api_key = args.api_key, http_client=httpx.Client(verify=False), timeout=3600)

    easy_gen_questions = []
    with open(args.data_path, 'r') as data_file:
        for line in data_file:
            data = json.loads(line.strip())
            if data["level"] == "easy" and data["task"] == "generation":
                    easy_gen_questions.append(data)
            else:
                continue

    easy_gen(easy_gen_questions, args.model_id, client)
