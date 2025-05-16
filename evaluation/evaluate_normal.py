
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

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def eval_normal_gen(model_id, client, times=5):
    result_path = model_id + "/normal_gen/results.jsonl"
    if not os.path.exists(result_path):
        print("Normal generation results missing!")
        return 0
    
    results = []
    with open(result_path, 'r') as data_file:
        for line in data_file:
            data = json.loads(line.strip())
            results.append(data)

    with open(f"eval_results/{model_id}/normal_gen_results.jsonl", "w", encoding='utf-8') as f:
        pass
    with open(f"eval_results/{model_id}/normal_gen_accuracy.jsonl", "w", encoding='utf-8') as f:
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
            img_path = data["answer"]

            if img_path == "Failed":
                wrong += 1
                record = {"question_id": data["question_id"], "eval_round": time, "eval_result": "Flase" , "ground_truth": data["ground_truth"], "answer": img_path, "question": question}
                with open(f"eval_results/{model_id}/normal_gen_results.jsonl", "a", encoding='utf-8') as f:
                    f.write(json.dumps(record) + "\n")
                continue

            prompt = f"Please judge whether the character drawn in the given picture is {obj}. If so, output [[Yes]], otherwise output [[No]]."

            try:
                image = encode_image(img_path)
                response = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpg;base64,{image}"},
                                },
                            ],
                        }
                    ],
                    temperature=0.0,
                )

                text = response.choices[0].message.content
                answer = text.split("[[")[-1].split("]]")[0].strip()
                if answer == 'Yes':
                    correct += 1
                    record = {"question_id": data["question_id"], "eval_round": time, "eval_result": "True" , "ground_truth": data["ground_truth"], "answer": img_path, "question": question}
                elif answer == 'No':
                    wrong += 1
                    record = {"question_id": data["question_id"], "eval_round": time, "eval_result": "Flase" , "ground_truth": data["ground_truth"], "answer": img_path, "question": question}
                else:
                    fail += 1
                    record = {"question_id": data["question_id"], "eval_round": time, "eval_result": "eval error" , "ground_truth": data["ground_truth"], "answer": img_path, "question": question}
            except:
                fail += 1
                record = {"question_id": data["question_id"], "eval_round": time, "eval_result": "eval error" , "ground_truth": data["ground_truth"], "answer": img_path, "question": question}

            with open(f"eval_results/{model_id}/normal_gen_results.jsonl", "a", encoding='utf-8') as f:
                f.write(json.dumps(record) + "\n")
    
        accuracy = correct / len(results)
        average_accuracy += accuracy
        with open(f"eval_results/{model_id}/normal_gen_accuracy.jsonl", "a") as f:
            f.write(json.dumps({"accuracy": accuracy, "eval_round": time, "correct": correct, "wrong": wrong, "eval_error": fail}) + "\n")

    average_accuracy /= times
    with open(f"eval_results/{model_id}/normal_gen_accuracy.jsonl", "a") as f:
        f.write(json.dumps({"Average_accuracy": average_accuracy}) + "\n")
    return average_accuracy

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./data/question.jsonl")
    parser.add_argument("--model_id", type=str, default="deepseek-r1-distill-llama-70b")

    parser.add_argument("--eval_rounds", type=int, default=5)

    openai_url = os.getenv("OPENAI_BASE_URL", '')
    api_key = os.getenv("OPENAI_API_KEY", '')
    client = OpenAI(base_url=openai_url, api_key = api_key, http_client=httpx.Client(verify=False), timeout=3600)


    eval_result_path = f"eval_results/{args.model_id}"
    if not os.path.exists(eval_result_path):
        os.makedirs(eval_result_path)

    eval_normal_gen(args.model_id, client, args.eval_rounds)
